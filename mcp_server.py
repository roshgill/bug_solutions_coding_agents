import os
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
from pydantic import Field
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # For local development
    DATABASE_URL = "postgresql://localhost/bug_solutions"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Clean connection string for psycopg2
CLEAN_DB_URL = DATABASE_URL.split("?")[0]

port = int(os.getenv("PORT", "8000"))
mcp = FastMCP("BugSolutionsMCP", host="0.0.0.0", port=port, log_level="ERROR")


def get_db_connection():
    """Get a database connection."""
    return psycopg2.connect(CLEAN_DB_URL, sslmode="require" if "neon" in DATABASE_URL else "disable")


def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI's text-embedding-3-small model."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def vector_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        # Connect to database
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get all embeddings and compute similarity
        cur.execute("SELECT id, embedding FROM embeddings")
        results = cur.fetchall()
        
        if not results:
            return json.dumps({"error": "No issues found in database"})
        
        # Calculate similarities
        similarities = []
        for issue_id, embedding in results:
            similarity = vector_similarity(query_embedding, embedding)
            similarities.append((issue_id, similarity))
        
        # Sort by similarity (descending) and get top 3
        top_3 = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        
        # Fetch issue details
        output = []
        for rank, (issue_id, similarity) in enumerate(top_3, 1):
            cur.execute("""
                SELECT id, repo, issue_number, url, title, body, labels, 
                       created_at, closed_at, resolution_text, comments
                FROM issues WHERE id = %s
            """, (issue_id,))
            
            row = cur.fetchone()
            if row:
                result_entry = {
                    "rank": rank,
                    "id": row[0],
                    "repo": row[1],
                    "issue_number": row[2],
                    "url": row[3],
                    "title": row[4],
                    "body": row[5],
                    "labels": row[6].split(",") if row[6] else [],
                    "created_at": row[7],
                    "closed_at": row[8],
                    "resolution_text": row[9],
                    "comments": json.loads(row[10]) if row[10] else [],
                    "distance": round(1 - similarity, 4),  # Convert similarity to distance
                }
                output.append(result_entry)
        
        cur.close()
        conn.close()
        
        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error searching issues: {str(e)}"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
