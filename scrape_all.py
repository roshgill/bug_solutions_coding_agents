#!/usr/bin/env python3
"""
Large-scale GitHub issue scraper with resumable progress tracking.

Scrapes closed issues with comments from repositories, generates embeddings,
and inserts into PostgreSQL. Tracks progress per repo to enable resumption
on interruption. Ideal for repos with 5,000+ closed issues.

Usage:
    python scrape_all.py

Configure REPOS at the top of the file to specify repositories to scrape.
Progress is tracked in progress.json for resumable ingestion.
"""

import json
import os
import sys
import time
import re
import requests
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration: list of (owner, repo) tuples to scrape
REPOS = [
("facebook", "pyrefly"),
("millionco", "react-doctor"),
("langchain-ai", "langchain"),
("huggingface", "transformers"),
("vercel", "next.js"),
("supabase", "supabase-py"),
("pydantic", "pydantic"),
("tiangolo", "sqlmodel"),
("encode", "httpx"),
("celery", "celery"),
("django", "django"),
("redis", "redis-py")
]

# Environment setup
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").split("?")[0]

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

# Initialize clients
github_headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
openai_client = OpenAI(api_key=OPENAI_API_KEY)

BASE_URL = "https://api.github.com"
PROGRESS_FILE = "progress.json"


def log(message, level="INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def strip_html_tags(text):
    """Remove HTML tags from text."""
    if not text:
        return text
    return re.sub(r"<[^>]+>", "", text)


def load_progress():
    """Load progress from progress.json, create if missing."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save progress to progress.json."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def check_rate_limit_smart(response_headers):
    """
    Check X-RateLimit-Remaining header and sleep if necessary.
    Uses response headers instead of extra API call.
    Returns True if slept (caller should reconnect DB), False otherwise.
    """
    remaining = int(response_headers.get("x-ratelimit-remaining", "100"))
    reset_time = int(response_headers.get("x-ratelimit-reset", "0"))

    if remaining < 100:
        sleep_time = reset_time - time.time() + 1
        if sleep_time > 0:
            log(f"Rate limit approaching ({remaining} left). Sleeping {sleep_time:.0f}s...", "WARN")
            time.sleep(sleep_time)
            return True
    return False


def fetch_issues_page(owner, repo, page):
    """Fetch single page of closed issues. Returns (issues_data, remaining) or raises on 422."""
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    params = {
        "state": "closed",
        "per_page": 100,
        "page": page,
        "sort": "created",
        "direction": "asc",
    }

    response = requests.get(url, headers=github_headers, params=params)

    if response.status_code == 422:
        log(f"  GitHub pagination limit reached at page {page} (422). Stopping.", "WARN")
        return None, "?"

    response.raise_for_status()
    check_rate_limit_smart(response.headers)
    remaining = response.headers.get("x-ratelimit-remaining", "?")

    return response.json(), remaining


def fetch_comments(owner, repo, issue_number):
    """Fetch all comments for a specific issue."""
    comments = []
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    page = 1

    while True:
        params = {"per_page": 100, "page": page}
        response = requests.get(url, headers=github_headers, params=params)
        response.raise_for_status()

        # Check rate limit after request
        check_rate_limit_smart(response.headers)

        data = response.json()
        if not data:
            break

        for comment in data:
            comments.append({
                "author": comment["user"]["login"],
                "body": comment["body"],
                "created_at": comment["created_at"],
                "resolution_comment": False,
            })

        page += 1

    return comments


def find_resolution_comment(comments, closed_at):
    """Find the last comment at or before the closed timestamp."""
    if not comments:
        return None, None

    closed_timestamp = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
    resolution_comment = None
    resolution_index = None

    for i, comment in enumerate(comments):
        comment_timestamp = datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00"))
        if comment_timestamp <= closed_timestamp:
            resolution_comment = comment
            resolution_index = i

    return resolution_index, resolution_comment


def structure_issue(owner, repo, issue, comments):
    """Structure an issue into the desired format."""
    # Find the closed timestamp
    closed_timestamp = issue["closed_at"]

    # Mark resolution comment and get resolution text
    resolution_index, resolution_comment = find_resolution_comment(comments, closed_timestamp)
    if resolution_index is not None:
        comments[resolution_index]["resolution_comment"] = True

    resolution_text = resolution_comment["body"] if resolution_comment else ""

    # Strip HTML tags
    title_clean = strip_html_tags(issue["title"])
    body_clean = strip_html_tags(issue.get("body") or "")
    resolution_text_clean = strip_html_tags(resolution_text)

    # Build embed_text
    embed_parts = [f"TITLE: {title_clean}"]
    if body_clean:
        embed_parts.append(f"PROBLEM: {body_clean}")
    if resolution_text_clean:
        embed_parts.append(f"RESOLUTION: {resolution_text_clean}")

    other_comments = [
        strip_html_tags(c["body"])
        for i, c in enumerate(comments)
        if i != resolution_index and c["body"]
    ]
    if other_comments:
        embed_parts.append(f"COMMENTS: {' '.join(other_comments)}")

    embed_text = "\n".join(embed_parts)

    return {
        "id": f"{repo}#{issue['number']}",
        "repo": repo,
        "issue_number": issue["number"],
        "url": issue["html_url"],
        "title": issue["title"],
        "body": issue.get("body") or "",
        "labels": [label["name"] for label in issue.get("labels", [])],
        "created_at": issue["created_at"],
        "closed_at": issue["closed_at"],
        "comments": comments,
        "resolution_text": resolution_text,
        "embed_text": embed_text,
    }


def generate_embedding(text: str) -> list:
    """Generate embedding using OpenAI's text-embedding-3-small model."""
    truncated_text = text[:8000]
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=truncated_text,
    )
    return response.data[0].embedding


def insert_issue(issue, library_id):
    """Insert a single issue and its embedding into the database."""
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()

    try:
        embedding = generate_embedding(issue["embed_text"])

        cur.execute("""
            INSERT INTO issues
            (issue_number, url, title, body, labels, created_at, closed_at, resolution_text, comments, library_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (issue_number, library_id) DO NOTHING
            RETURNING numeric_id
        """, (
            issue["issue_number"],
            issue["url"],
            issue["title"],
            issue["body"],
            ",".join(issue["labels"]),
            issue["created_at"],
            issue["closed_at"],
            issue["resolution_text"],
            json.dumps(issue["comments"]),
            library_id,
        ))

        result = cur.fetchone()
        if not result:
            log(f"    CONFLICT: {issue['id']} already exists (issue_number={issue['issue_number']}, library_id={library_id})", "DEBUG")
            cur.close()
            conn.close()
            return False

        numeric_id = result[0]
        log(f"    INSERTED: {issue['id']} → numeric_id={numeric_id}", "DEBUG")

        vec = "[" + ",".join(str(x) for x in embedding) + "]"
        cur.execute("""
            INSERT INTO embeddings (issue_id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT DO NOTHING
        """, (numeric_id, vec))

        conn.commit()
        cur.close()
        conn.close()
        return True

    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise Exception(f"Failed to insert {issue['id']}: {e}")


def process_issue(owner, repo, issue, library_id):
    """
    Process a single issue: fetch comments, structure, embed, and insert.
    Returns True if processed, False if skipped.
    """
    try:
        comments = fetch_comments(owner, repo, issue["number"])
        structured = structure_issue(owner, repo, issue, comments)
        inserted = insert_issue(structured, library_id)

        if inserted:
            log(f"    PROCESSED: {repo}#{issue['number']} ({len(comments)} comments)", "DEBUG")

        return inserted

    except Exception as e:
        log(f"  WARNING: Failed to process {repo}#{issue['number']}: {e}", "WARN")
        return False


def get_library_id(conn, owner, repo):
    """Get library_id for a repo name. Creates if doesn't exist with organization."""
    cur = conn.cursor()

    # Check if library exists
    cur.execute("SELECT id FROM libraries WHERE name = %s", (repo,))
    existing = cur.fetchone()

    if existing:
        lib_id = existing[0]
        cur.close()
        return lib_id

    # Create if missing
    log(f"Creating library entry for '{repo}' ({owner})...", "INFO")
    embedding = generate_embedding(repo)

    cur.execute("""
        INSERT INTO libraries (name, embedding, organization)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET id = EXCLUDED.id
        RETURNING id
    """, (repo, embedding, owner))

    lib_id = cur.fetchone()[0]
    conn.commit()
    cur.close()

    return lib_id


def process_repo(owner, repo, progress):
    """
    Process a single repository: scrape pages of issues, embed and insert.
    Updates progress.json after each page.
    """
    repo_key = f"{owner}/{repo}"

    # Skip if already done
    if repo_key in progress and progress[repo_key].get("status") == "done":
        log(f"Repo {repo_key} already done, skipping", "INFO")
        return progress[repo_key]

    # Initialize or resume progress
    if repo_key not in progress:
        progress[repo_key] = {
            "status": "in_progress",
            "last_page": 1,
            "checked_count": 0,
            "processed_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

    start_page = progress[repo_key]["last_page"]
    checked_count = progress[repo_key]["checked_count"]
    processed_count = progress[repo_key]["processed_count"]

    log(f"\nProcessing {repo_key} (resuming from page {start_page})...", "INFO")

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    library_id = get_library_id(conn, owner, repo)
    conn.close()

    page = start_page
    while True:
        try:
            issues_data, remaining = fetch_issues_page(owner, repo, page)

            if issues_data is None:
                break

            if not issues_data:
                break

            page_checked = 0
            page_processed = 0

            for item in issues_data:
                if "pull_request" in item:
                    continue

                page_checked += 1
                checked_count += 1

                if process_issue(owner, repo, item, library_id):
                    page_processed += 1
                    processed_count += 1

            log(f"  Page {page}: checked {page_checked}, processed {page_processed}, remaining requests: {remaining}", "INFO")

            progress[repo_key]["last_page"] = page + 1
            progress[repo_key]["checked_count"] = checked_count
            progress[repo_key]["processed_count"] = processed_count
            progress[repo_key]["timestamp"] = datetime.now().isoformat()
            save_progress(progress)

            page += 1

        except Exception as e:
            log(f"Error on page {page}: {e}", "ERROR")
            progress[repo_key]["timestamp"] = datetime.now().isoformat()
            save_progress(progress)
            raise

    # Mark repo as done
    progress[repo_key]["status"] = "done"
    progress[repo_key]["timestamp"] = datetime.now().isoformat()
    save_progress(progress)

    log(f"✓ Completed {repo_key}: {processed_count} issues with comments processed", "INFO")

    return progress[repo_key]


def main():
    """Main orchestration function."""
    log("Starting large-scale GitHub issue scraper", "INFO")
    log(f"Repos to process: {len(REPOS)}", "INFO")

    progress = load_progress()

    results = []
    for owner, repo in REPOS:
        try:
            result = process_repo(owner, repo, progress)
            results.append(result)
        except Exception as e:
            log(f"✗ Failed {owner}/{repo}: {e}", "ERROR")
            results.append({
                "status": "failed",
                "error": str(e),
            })

    # Final summary
    log(f"\n{'='*70}", "INFO")
    log("SCRAPE SUMMARY", "INFO")
    log(f"{'='*70}", "INFO")

    completed = sum(1 for r in results if r.get("status") == "done")
    failed = sum(1 for r in results if r.get("status") == "failed")
    total_checked = sum(r.get("checked_count", 0) for r in results)
    total_processed = sum(r.get("processed_count", 0) for r in results)

    log(f"Repos processed: {len(results)}", "INFO")
    log(f"  ✓ Completed: {completed}", "INFO")
    log(f"  ✗ Failed: {failed}", "INFO")
    log(f"Total issues checked: {total_checked}", "INFO")
    log(f"Total issues processed (with comments): {total_processed}", "INFO")

    if failed > 0:
        log("\nFailed repos:", "WARN")
        for i, r in enumerate(results):
            if r.get("status") == "failed":
                log(f"  - {REPOS[i]}: {r.get('error')}", "WARN")

    log(f"{'='*70}", "INFO")
    log("Scrape complete", "INFO")


if __name__ == "__main__":
    main()
