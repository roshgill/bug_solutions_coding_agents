import json
import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

DATABASE_URL = os.getenv("DATABASE_URL").split("?")[0]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_FILE = "issues_corpus.json"


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI's text-embedding-3-small model."""
    # Truncate to 8000 chars to avoid token limit (rough estimate: ~4 chars per token)
    truncated_text = text[:8000]
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=truncated_text,
    )
    return response.data[0].embedding


print("Loading issues from corpus...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} not found. Run ingest_issues.py first.")

with open(INPUT_FILE, "r") as f:
    all_issues = json.load(f)

print(f"Loaded {len(all_issues)} issues\n")

conn = psycopg2.connect(DATABASE_URL, sslmode="require")
cur = conn.cursor()

# Get library name from first issue (all issues in corpus are from same repo)
if not all_issues:
    raise ValueError("No issues found in corpus")

repo_name = all_issues[0].get("repo")
if not repo_name:
    raise ValueError("Issue missing 'repo' field")

# Look up library_id by repo name
cur.execute("SELECT id FROM libraries WHERE name = %s", (repo_name,))
lib_row = cur.fetchone()
if not lib_row:
    print(f"WARNING: Library '{repo_name}' not found. Make sure to run setup_libraries.py first.")
    raise ValueError(f"Library '{repo_name}' not found in libraries table")

library_id = lib_row[0]
print(f"Using library_id={library_id} for repo '{repo_name}'\n")

print("Generating embeddings and inserting into database...")
for i, issue in enumerate(all_issues, 1):
    print(f"  Processing {i}/{len(all_issues)}: {issue['id']}")

    # Generate embedding
    embedding = generate_embedding(issue["embed_text"])

    # Insert into issues table
    cur.execute("""
        INSERT INTO issues
        (numeric_id, issue_number, url, title, body, labels, created_at, closed_at, resolution_text, comments, library_id)
        VALUES (DEFAULT, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (numeric_id) DO NOTHING
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
        print(f"    Skipped (already exists)")
        continue

    numeric_id = result[0]

    # Insert embedding
    cur.execute("""
        INSERT INTO embeddings (issue_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (issue_id) DO NOTHING
    """, (numeric_id, embedding))

conn.commit()

# Verify
cur.execute("SELECT COUNT(*) FROM issues")
issue_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM embeddings")
embedding_count = cur.fetchone()[0]

print(f"\n✓ Insertion complete!")
print(f"  Total issues: {issue_count}")
print(f"  Total embeddings: {embedding_count}")

cur.close()
conn.close()
