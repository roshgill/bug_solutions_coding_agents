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
    name="find_libraries",
    description="Returns top 10 indexed libraries semantically matching your keyword. Call this first when unsure if a library is indexed. Pass the library name, framework, or SDK you're working with."
)
def find_libraries(
    keyword: str = Field(description="Library name, framework, or SDK to search for (e.g., 'wearables', 'next.js', 'stripe')")
):
    """Find indexed libraries by semantic keyword matching.

    Args:
        keyword: Library or framework name to search for

    Returns:
        JSON array of top 10 matching libraries with similarity scores
    """
    try:
        keyword_embedding = generate_embedding(keyword)
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT id, name, embedding FROM libraries")
        results = cur.fetchall()

        if not results:
            return json.dumps([])

        # Calculate similarities
        similarities = []
        for lib_id, lib_name, embedding in results:
            similarity = vector_similarity(keyword_embedding, embedding)
            similarities.append((lib_name, similarity))

        # Sort by similarity and get top 10
        top_10 = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

        output = [
            {"name": lib_name, "distance": round(1 - sim, 4)}
            for lib_name, sim in top_10
        ]

        cur.close()
        conn.close()

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error finding libraries: {str(e)}"})


@mcp.tool(
    name="search_bugs",
    description="Search indexed GitHub issues for bug solutions matching an error message or symptom. Optionally filter by library name from find_libraries. Returns top 3 full issue threads with all comments. Use the exact error string or a short symptom description as the query. After getting results, check comments for linked issue numbers and call fetch_thread to follow them, or use web search for any external URLs."
)
def search_bugs(
    query: str = Field(description="Error message, stack trace, or bug description to search for"),
    library: str = ""
):
    """Search for bug solutions with optional library filtering.

    Args:
        query: Error message, stack trace, or bug description
        library: Optional library name to scope search (defaults to global search)

    Returns:
        JSON array of top 3 matching issues with full details; includes fallback flag if filtered search was empty
    """
    try:
        query_embedding = generate_embedding(query)
        conn = get_db_connection()
        cur = conn.cursor()

        # Try library-filtered search if library provided
        library_fallback = False
        if library:
            cur.execute("""
                SELECT e.issue_id, e.embedding FROM embeddings e
                JOIN issues i ON e.issue_id = i.numeric_id
                JOIN libraries l ON i.library_id = l.id
                WHERE l.name = %s
            """, (library,))
            results = cur.fetchall()

            # Fall back to global search if filtered search has no results
            if not results:
                library_fallback = True
                cur.execute("SELECT issue_id, embedding FROM embeddings")
                results = cur.fetchall()
        else:
            cur.execute("SELECT issue_id, embedding FROM embeddings")
            results = cur.fetchall()

        if not results:
            return json.dumps({"error": "No issues found in database"})

        # Calculate similarities
        similarities = []
        for issue_id, embedding in results:
            similarity = vector_similarity(query_embedding, embedding)
            similarities.append((issue_id, similarity))

        # Sort and get top 3
        top_3 = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

        # Fetch issue details with library name
        output = []
        for rank, (issue_id, similarity) in enumerate(top_3, 1):
            cur.execute("""
                SELECT i.issue_number, i.url, i.title, i.body, i.labels,
                       i.created_at, i.closed_at, i.resolution_text, i.comments, i.numeric_id,
                       l.name as library_name
                FROM issues i
                JOIN libraries l ON i.library_id = l.id
                WHERE i.numeric_id = %s
            """, (issue_id,))

            row = cur.fetchone()
            if row:
                result_entry = {
                    "rank": rank,
                    "id": row[9],
                    "library": row[10],
                    "issue_number": row[0],
                    "url": row[1],
                    "title": row[2],
                    "body": row[3],
                    "labels": row[4].split(",") if row[4] else [],
                    "created_at": row[5],
                    "closed_at": row[6],
                    "resolution_text": row[7],
                    "comments": row[8] if isinstance(row[8], list) else (json.loads(row[8]) if row[8] else []),
                    "distance": round(1 - similarity, 4),
                }
                output.append(result_entry)

        # Add fallback indicator if applicable
        if library_fallback:
            output = {"library_fallback": True, "results": output}
        else:
            output = {"results": output}

        cur.close()
        conn.close()

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error searching issues: {str(e)}"})


@mcp.tool(
    name="fetch_thread",
    description="Returns the complete GitHub issue thread for a specific issue by library name and issue number. Call this to follow a linked issue number found within a search_bugs result. For links pointing to external sites, use web search."
)
def fetch_thread(
    library: str = Field(description="Library name where the issue is indexed (e.g., 'meta-wearables-dat-ios')"),
    issue_number: int = Field(description="GitHub issue number (not the internal numeric_id)")
):
    """Fetch the complete thread for a specific issue.

    Args:
        library: Library name
        issue_number: GitHub issue number

    Returns:
        JSON object with complete issue thread including all comments
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT i.numeric_id, i.issue_number, i.url, i.title, i.body, i.labels,
                   i.created_at, i.closed_at, i.resolution_text, i.comments, l.name
            FROM issues i
            JOIN libraries l ON i.library_id = l.id
            WHERE l.name = %s AND i.issue_number = %s
        """, (library, issue_number))

        row = cur.fetchone()

        if not row:
            cur.close()
            conn.close()
            return json.dumps({"error": f"Issue #{issue_number} not found for library '{library}'"})

        result = {
            "id": row[0],
            "issue_number": row[1],
            "url": row[2],
            "title": row[3],
            "body": row[4],
            "labels": row[5].split(",") if row[5] else [],
            "created_at": row[6],
            "closed_at": row[7],
            "resolution_text": row[8],
            "comments": row[9] if isinstance(row[9], list) else (json.loads(row[9]) if row[9] else []),
            "library": row[10],
        }

        cur.close()
        conn.close()

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error fetching thread: {str(e)}"})


@mcp.tool(
    name="request_bug_solutions",
    description="Request bug solutions to be indexed for a GitHub organization/library. Use this when you encounter an error from a library not yet in the database and want solutions for it indexed."
)
def request_bug_solutions(
    organization: str = Field(description="GitHub organization name (e.g., 'openai', 'fastapi')"),
    library: str = Field(description="GitHub repository name (e.g., 'tiktoken', 'fastapi')")
):
    """Request bug solutions for an organization/library.

    Args:
        organization: GitHub organization name
        library: GitHub repository name

    Returns:
        JSON object confirming the request with timestamp and total request count
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Insert new request
        cur.execute("""
            INSERT INTO requests (organization, library)
            VALUES (%s, %s)
            RETURNING id, organization, library, requested_at
        """, (organization, library))

        row = cur.fetchone()

        # Count how many times this org/library has been requested
        cur.execute("""
            SELECT COUNT(*) FROM requests
            WHERE organization = %s AND library = %s
        """, (organization, library))

        request_count = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        result = {
            "id": row[0],
            "organization": row[1],
            "library": row[2],
            "requested_at": str(row[3]),
            "total_requests": request_count,
            "message": f"Request #{request_count} for {organization}/{library}"
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error requesting bug solutions: {str(e)}"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
