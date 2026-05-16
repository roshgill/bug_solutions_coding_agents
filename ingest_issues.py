import requests
import json
import time
import os
import re
import sys
import argparse
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env file")

# Default repos (used if no command-line args)
DEFAULT_REPOS = [
    ("openai", "tiktoken"),
]

BASE_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

OUTPUT_FILE = "issues_corpus.json"


def strip_html_tags(text):
    """Remove HTML tags from text."""
    if not text:
        return text
    return re.sub(r"<[^>]+>", "", text)


def check_rate_limit():
    """Check remaining rate limit and sleep if necessary."""
    url = f"{BASE_URL}/rate_limit"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    remaining = data["rate"]["remaining"]
    reset_time = data["rate"]["reset"]

    if remaining < 10:
        sleep_time = reset_time - time.time() + 1
        if sleep_time > 0:
            print(f"Rate limit approaching. Sleeping for {sleep_time:.0f} seconds...")
            time.sleep(sleep_time)


def fetch_issues(owner, repo):
    """Fetch all closed issues from a repository."""
    issues = []
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    page = 1

    while True:
        check_rate_limit()

        params = {
            "state": "closed",
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "asc",
        }

        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        for item in data:
            # Skip pull requests
            if "pull_request" in item:
                continue

            issues.append(item)
            print(f"Fetched {repo}#{item['number']}")

        page += 1

    return issues


def fetch_comments(owner, repo, issue_number):
    """Fetch all comments for a specific issue."""
    comments = []
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    page = 1

    while True:
        check_rate_limit()

        params = {
            "per_page": 100,
            "page": page,
        }

        print(f"    [DEBUG] Fetching comments for {repo}#{issue_number} (page {page})", flush=True)
        response = requests.get(url, headers=HEADERS, params=params)
        remaining = response.headers.get("x-ratelimit-remaining", "?")
        reset_time = response.headers.get("x-ratelimit-reset", "?")
        print(f"    [DEBUG] Status: {response.status_code}, Remaining: {remaining}, Reset: {reset_time}", flush=True)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        for comment in data:
            comments.append(
                {
                    "author": comment["user"]["login"],
                    "body": comment["body"],
                    "created_at": comment["created_at"],
                    "resolution_comment": False,
                }
            )

        page += 1

    return comments


def fetch_events(owner, repo, issue_number):
    """Fetch all events for a specific issue."""
    events = []
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/events"
    page = 1

    while True:
        check_rate_limit()

        params = {
            "per_page": 100,
            "page": page,
        }

        print(f"    [DEBUG] Fetching events for {repo}#{issue_number} (page {page})", flush=True)
        response = requests.get(url, headers=HEADERS, params=params)
        remaining = response.headers.get("x-ratelimit-remaining", "?")
        reset_time = response.headers.get("x-ratelimit-reset", "?")
        print(f"    [DEBUG] Status: {response.status_code}, Remaining: {remaining}, Reset: {reset_time}", flush=True)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        events.extend(data)
        page += 1

    return events


def find_resolution_comment(comments, closed_at):
    """Find the last comment at or before the closed timestamp."""
    if not comments:
        return None, None

    closed_timestamp = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))

    resolution_comment = None
    resolution_index = None

    for i, comment in enumerate(comments):
        comment_timestamp = datetime.fromisoformat(
            comment["created_at"].replace("Z", "+00:00")
        )
        if comment_timestamp <= closed_timestamp:
            resolution_comment = comment
            resolution_index = i

    return resolution_index, resolution_comment


def structure_issue(owner, repo, issue):
    """Structure an issue into the desired format."""
    comments = fetch_comments(owner, repo, issue["number"])
    # events = fetch_events(owner, repo, issue["number"])

    # Find the closed event timestamp
    closed_timestamp = issue["closed_at"]

    # Mark resolution comment and get resolution text
    resolution_index, resolution_comment = find_resolution_comment(
        comments, closed_timestamp
    )
    if resolution_index is not None:
        comments[resolution_index]["resolution_comment"] = True

    resolution_text = resolution_comment["body"] if resolution_comment else ""

    # Strip HTML tags for embed_text only
    title_clean = strip_html_tags(issue["title"])
    body_clean = strip_html_tags(issue.get("body") or "")
    resolution_text_clean = strip_html_tags(resolution_text)

    # Build embed_text with structured format (using cleaned text)
    embed_parts = [f"TITLE: {title_clean}"]
    if body_clean:
        embed_parts.append(f"PROBLEM: {body_clean}")
    if resolution_text_clean:
        embed_parts.append(f"RESOLUTION: {resolution_text_clean}")

    # Add other comments (excluding the resolution comment)
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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Fetch closed issues from GitHub repositories"
    )
    parser.add_argument(
        "owner",
        nargs="?",
        help="GitHub organization/owner (e.g., openai)",
    )
    parser.add_argument(
        "repo",
        nargs="?",
        help="GitHub repository name (e.g., tiktoken)",
    )
    args = parser.parse_args()

    # Determine repos to fetch
    if args.owner and args.repo:
        repos = [(args.owner, args.repo)]
        print(f"Fetching from command-line args: {args.owner}/{args.repo}")
    else:
        repos = DEFAULT_REPOS
        print(f"Fetching from default repos")

    all_issues = []

    for owner, repo in repos:
        print(f"\nFetching issues from {owner}/{repo}...")
        issues = fetch_issues(owner, repo)

        for i, issue in enumerate(issues, 1):
            try:
                structured = structure_issue(owner, repo, issue)
                all_issues.append(structured)
            except Exception as e:
                print(f"  WARNING: Failed to structure issue #{issue['number']}: {e}")
                continue

    # Write to output file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_issues, f, indent=2)

    print(f"\n✓ Wrote {len(all_issues)} issues to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
