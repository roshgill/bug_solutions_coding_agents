import json
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "github_issues"
INPUT_FILE = "issues_corpus.json"


def generate_embedding(text):
    """Generate embedding for text using OpenAI's text-embedding-3-small model."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def search_issues(query, n_results=5):
    """Search for issues semantically similar to the query."""
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"Chroma database not found at {CHROMA_DB_PATH}. Run embed_and_store.py first."
        )

    # Generate embedding for query
    query_embedding = generate_embedding(query)

    # Query Chroma
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results


def main():
    # Check if corpus exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run ingest_issues.py first.")
        return

    if not os.path.exists(CHROMA_DB_PATH):
        print(f"❌ Chroma database not found. Run embed_and_store.py first.")
        return

    # Load corpus to get total count
    with open(INPUT_FILE, "r") as f:
        corpus = json.load(f)

    print(f"📊 Loaded {len(corpus)} issues from corpus\n")

    # Test queries
    test_queries = [
        "app crashes on startup",
        "memory leak performance",
        "bluetooth connectivity issues",
        "null pointer exception",
        "authentication login failure",
    ]

    print("🔍 Testing semantic search...\n")

    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 60)

        results = search_issues(query, n_results=3)

        if not results["ids"] or not results["ids"][0]:
            print("  ❌ No results found\n")
            continue

        for i, (doc_id, metadata, distance) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
        ):
            print(f"  {i + 1}. {metadata['title']} ({metadata['repo']}#{metadata['issue_number']})")
            print(f"     Distance: {distance:.4f}")
            print(f"     URL: {metadata['url']}")
            print()


if __name__ == "__main__":
    main()
