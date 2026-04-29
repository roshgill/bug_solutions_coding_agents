import os
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import chromadb

from pydantic import Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

CHROMA_DB_PATH = "/Users/roshgill/Desktop/meta-wearables-scraper/chroma_db"
COLLECTION_NAME = "github_issues"
CORPUS_PATH = "/Users/roshgill/Desktop/meta-wearables-scraper/issues_corpus.json"

# Initialize Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

mcp = FastMCP("BugSolutionsMCP", log_level="ERROR")


def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI's text-embedding-3-small model."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


@mcp.tool(
    name="search_bug_solution",
    description="Search for bug solutions from database"
)
def search_bugs(
    query: str = Field(description="Query for database search")
):
    """Search for similar bug solutions and resolutions.

    Args:
        query: Error message, stack trace, or bug description to search for

    Returns:
        JSON string containing top 3 most similar resolved issues with full thread and metadata
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)

        # Search Chroma for top 3 similar issues
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return json.dumps({"error": "No similar issues found in the database."})

        # Load corpus once
        corpus = {}
        if os.path.exists(CORPUS_PATH):
            with open(CORPUS_PATH, "r") as f:
                corpus_list = json.load(f)
                corpus = {issue["id"]: issue for issue in corpus_list}

        # Format results with full thread data
        output = []
        for i, (doc_id, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            result_entry = {
                "rank": i + 1,
                "id": doc_id,
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "repo": metadata.get("repo", ""),
                "issue_number": metadata.get("issue_number", ""),
                "labels": metadata.get("labels", "").split(",") if metadata.get("labels") else [],
                "created_at": metadata.get("created_at", ""),
                "closed_at": metadata.get("closed_at", ""),
                "distance": round(distance, 4),
                "body": "",
                "resolution_text": "",
                "comments": [],
            }

            # Load full issue data from corpus
            if doc_id in corpus:
                issue = corpus[doc_id]
                result_entry["body"] = issue.get("body", "")
                result_entry["resolution_text"] = issue.get("resolution_text", "")
                result_entry["comments"] = issue.get("comments", [])

            output.append(result_entry)

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error searching issues: {str(e)}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
