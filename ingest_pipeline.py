#!/usr/bin/env python3
"""
Unified GitHub Issue Ingestion Pipeline

Combines ingestion, embedding generation, and database insertion in one workflow.
Configurable via JSON file with support for multiple libraries.

Usage:
    python ingest_pipeline.py libraries_config.json
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

# Configuration
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


def log(message, level="INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def strip_html_tags(text):
    """Remove HTML tags from text."""
    if not text:
        return text
    return re.sub(r"<[^>]+>", "", text)


def check_rate_limit():
    """Check remaining rate limit and sleep if necessary."""
    url = f"{BASE_URL}/rate_limit"
    response = requests.get(url, headers=github_headers)
    data = response.json()
    remaining = data["rate"]["remaining"]
    reset_time = data["rate"]["reset"]

    if remaining < 10:
        sleep_time = reset_time - time.time() + 1
        if sleep_time > 0:
            log(f"Rate limit approaching ({remaining} left). Sleeping {sleep_time:.0f}s...", "WARN")
            time.sleep(sleep_time)


def fetch_comments(owner, repo, issue_number):
    """Fetch all comments for a specific issue."""
    comments = []
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    page = 1

    while True:
        check_rate_limit()

        params = {"per_page": 100, "page": page}
        response = requests.get(url, headers=github_headers, params=params)
        response.raise_for_status()

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


def structure_issue(owner, repo, issue):
    """Structure an issue into the desired format."""
    comments = fetch_comments(owner, repo, issue["number"])

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


def ingest_issues(owner, repo):
    """Fetch and structure all closed issues from a repository."""
    issues = []
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    page = 1

    log(f"Fetching issues from {owner}/{repo}...", "INFO")

    while True:
        check_rate_limit()

        params = {
            "state": "closed",
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "asc",
        }

        response = requests.get(url, headers=github_headers, params=params)
        response.raise_for_status()

        data = response.json()
        if not data:
            break

        for item in data:
            if "pull_request" in item:
                continue

            try:
                structured = structure_issue(owner, repo, item)
                issues.append(structured)
                log(f"  Fetched {repo}#{item['number']}", "DEBUG")
            except Exception as e:
                log(f"  WARNING: Failed to structure {repo}#{item['number']}: {e}", "WARN")
                continue

        page += 1

    log(f"Fetched {len(issues)} issues from {owner}/{repo}", "INFO")
    return issues


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI's text-embedding-3-small model."""
    truncated_text = text[:8000]
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=truncated_text,
    )
    return response.data[0].embedding


def setup_library(org, library_name):
    """Create library entry in database with embedding."""
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()

    try:
        # Check if library already exists
        cur.execute("SELECT id FROM libraries WHERE name = %s", (library_name,))
        existing = cur.fetchone()
        if existing:
            lib_id = existing[0]
            log(f"Library '{library_name}' already exists (id={lib_id})", "INFO")
            cur.close()
            conn.close()
            return lib_id

        # Generate embedding for library name
        log(f"Generating embedding for '{library_name}'...", "INFO")
        embedding = generate_embedding(library_name)

        # Insert into libraries table
        log(f"Inserting library '{library_name}' into database...", "INFO")
        cur.execute("""
            INSERT INTO libraries (name, embedding, organization)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (library_name, embedding, org))

        lib_id = cur.fetchone()[0]
        conn.commit()

        log(f"Library '{library_name}' created with id={lib_id}", "INFO")
        cur.close()
        conn.close()
        return lib_id

    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise Exception(f"Failed to setup library '{library_name}': {e}")


def embed_and_insert(issues, library_id):
    """Generate embeddings and insert issues into database."""
    if not issues:
        log("No issues to process", "INFO")
        return

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()

    try:
        log(f"Generating embeddings and inserting {len(issues)} issues...", "INFO")

        inserted_count = 0
        for i, issue in enumerate(issues, 1):
            try:
                # Generate embedding
                embedding = generate_embedding(issue["embed_text"])

                # Insert into issues table
                cur.execute("""
                    INSERT INTO issues
                    (issue_number, url, title, body, labels, created_at, closed_at, resolution_text, comments, library_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
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
                    log(f"  Issue {issue['id']} already exists, skipping", "DEBUG")
                    continue

                numeric_id = result[0]

                # Insert embedding
                cur.execute("""
                    INSERT INTO embeddings (issue_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (numeric_id, embedding))

                inserted_count += 1
                if i % 10 == 0:
                    log(f"  Processed {i}/{len(issues)} issues", "INFO")

            except Exception as e:
                log(f"  WARNING: Failed to embed/insert {issue['id']}: {e}", "WARN")
                continue

        conn.commit()
        log(f"Successfully inserted {inserted_count}/{len(issues)} issues", "INFO")
        cur.close()
        conn.close()

        return inserted_count

    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise Exception(f"Failed to embed/insert issues: {e}")


def process_library(org, library_name):
    """Process a single library through the full pipeline."""
    log(f"\n{'='*70}", "INFO")
    log(f"Processing {org}/{library_name}", "INFO")
    log(f"{'='*70}", "INFO")

    try:
        # Step 1: Setup library
        lib_id = setup_library(org, library_name)

        # Step 2: Ingest issues
        issues = ingest_issues(org, library_name)

        if not issues:
            log(f"No issues found for {org}/{library_name}", "WARN")
            return {"org": org, "library": library_name, "status": "success", "issues": 0}

        # Step 3: Embed and insert
        inserted_count = embed_and_insert(issues, lib_id)

        log(f"✓ Completed {org}/{library_name}: {inserted_count} issues processed", "INFO")
        return {
            "org": org,
            "library": library_name,
            "status": "success",
            "issues": inserted_count,
        }

    except Exception as e:
        log(f"✗ Failed {org}/{library_name}: {e}", "ERROR")
        return {
            "org": org,
            "library": library_name,
            "status": "failed",
            "error": str(e),
        }


def load_config(path):
    """Load libraries configuration from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    if not isinstance(config, list):
        raise ValueError("Config must be a JSON array of library objects")

    for item in config:
        if "organization" not in item or "library" not in item:
            raise ValueError("Each library object must have 'organization' and 'library' fields")

    return config


def main():
    """Main orchestration function."""
    if len(sys.argv) < 2:
        print("Usage: python ingest_pipeline.py <config.json>")
        print("\nConfig file format:")
        print(json.dumps([{"organization": "openai", "library": "tiktoken"}], indent=2))
        sys.exit(1)

    config_path = sys.argv[1]

    log("Starting unified ingestion pipeline", "INFO")
    log(f"Loading configuration from {config_path}...", "INFO")

    try:
        config = load_config(config_path)
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        sys.exit(1)

    log(f"Found {len(config)} libraries to process", "INFO")

    results = []
    for lib_config in config:
        result = process_library(lib_config["organization"], lib_config["library"])
        results.append(result)

    # Final summary
    log(f"\n{'='*70}", "INFO")
    log("PIPELINE SUMMARY", "INFO")
    log(f"{'='*70}", "INFO")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_issues = sum(r.get("issues", 0) for r in results if r["status"] == "success")

    log(f"Libraries processed: {len(results)}", "INFO")
    log(f"  ✓ Successful: {successful}", "INFO")
    log(f"  ✗ Failed: {failed}", "INFO")
    log(f"Total issues ingested: {total_issues}", "INFO")

    if failed > 0:
        log("\nFailed libraries:", "WARN")
        for r in results:
            if r["status"] == "failed":
                log(f"  - {r['org']}/{r['library']}: {r['error']}", "WARN")

    log(f"{'='*70}", "INFO")
    log("Pipeline complete", "INFO")


if __name__ == "__main__":
    main()
