import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_FILE = "issues_corpus.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "github_issues"


def generate_embedding(text):
    """Generate embedding for text using OpenAI's text-embedding-3-small model."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def main():
    # Load issues from JSON
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run ingest_issues.py first.")

    with open(INPUT_FILE, "r") as f:
        all_issues = json.load(f)

    print(f"Loaded {len(all_issues)} issues from {INPUT_FILE}")

    # Initialize Chroma with persistent storage
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Generate embeddings and store in Chroma
    print(f"\nGenerating embeddings and storing in Chroma...")
    for i, issue in enumerate(all_issues, 1):
        print(f"Processing {i}/{len(all_issues)}: {issue['id']}")

        # Generate embedding for embed_text
        embedding = generate_embedding(issue["embed_text"])

        # Add to Chroma with metadata
        collection.add(
            ids=[issue["id"]],
            embeddings=[embedding],
            documents=[issue["embed_text"]],
            metadatas=[
                {
                    "repo": issue["repo"],
                    "issue_number": str(issue["issue_number"]),
                    "title": issue["title"],
                    "url": issue["url"],
                    "created_at": issue["created_at"],
                    "closed_at": issue["closed_at"],
                    "labels": ",".join(issue["labels"]),
                }
            ],
        )

    print(f"\n✓ Stored {len(all_issues)} issues in Chroma at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
