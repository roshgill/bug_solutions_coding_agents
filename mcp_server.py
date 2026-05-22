import os
import json
from typing import List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import psycopg2
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://localhost/bug_solutions"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

CLEAN_DB_URL = DATABASE_URL.split("?")[0]

port = int(os.getenv("PORT", "8000"))
mcp = FastMCP("BugSolutionsMCP", host="0.0.0.0", port=port, log_level="ERROR")


def get_db_connection():
    return psycopg2.connect(CLEAN_DB_URL, sslmode="require" if "neon" in DATABASE_URL else "disable")


def log_hit(tool_name, query=None, library=None, results_returned=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO logs (tool_name, query, library, results_returned) VALUES (%s, %s, %s, %s)",
            (tool_name, query, library, results_returned)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def generate_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def vec(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


class RankingResult(BaseModel):
    top_3_indices: List[int]  # 1-based indices into the candidates list, in order of relevance


def rerank(query: str, candidates: list) -> list:
    numbered = "\n".join([
        f"{i+1}. [{c['library']}] #{c['issue_number']} (distance: {c['distance']:.4f})\n   Title: {c['title']}\n   Body: {c['body']}"
        for i, c in enumerate(candidates)
    ])
    completion = openai_client.chat.completions.parse(
        model="gpt-5.4-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a relevance judge for a bug solutions search engine. Given a developer query and a list of GitHub issues, return the 1-based indices of the 3 most relevant issues in order of relevance."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nCandidates:\n{numbered}"
            }
        ],
        response_format=RankingResult,
    )
    picks = completion.choices[0].message.parsed.top_3_indices
    return [candidates[i - 1]["id"] for i in picks if 0 < i <= len(candidates)]


@mcp.tool(
    name="search_bugs",
    description="""Search indexed GitHub issues for bug solutions matching an error or symptom.

Each result includes:
- title: Issue title
- body: Full problem description
- resolution_text: The comment that resolved the issue
- comments: Full thread with all replies
- library: Repository name
- organization: GitHub organization that owns the repo
- distance: Relevance score (lower is closer)

Steps:
1. Call this tool first with the exact error message or symptom as the query
2. If comments contain linked issue numbers, call fetch_thread to follow them
3. If results are not relevant to your error, call request_bug_solutions to get the library indexed
4. For external URLs in comments, use web search

IMPORTANT: Do not call this tool more than 3 times per question. Use the best result you have."""
)
def search_bugs(
    query: str = Field(description="Exact error message, stack trace snippet, or short symptom description"),
    libraryName: str = Field(default="", description="Repository name to scope the search (e.g. 'langchain', 'fastapi'). Leave empty for global search.")
):
    try:
        embedding = vec(generate_embedding(query))
        conn = get_db_connection()
        cur = conn.cursor()

        library_fallback = False
        if libraryName:
            cur.execute("""
                SELECT i.id, i.issue_number, i.title, i.body, l.name,
                       1 - (i.embedding <=> %s::vector) AS similarity
                FROM issues i
                JOIN libraries l ON i.library_id = l.id
                WHERE l.name = %s
                ORDER BY i.embedding <=> %s::vector
                LIMIT 10
            """, (embedding, libraryName, embedding))
            rows = cur.fetchall()

            if not rows:
                library_fallback = True
                cur.execute("""
                    SELECT i.id, i.issue_number, i.title, i.body, l.name,
                           1 - (i.embedding <=> %s::vector) AS similarity
                    FROM issues i
                    JOIN libraries l ON i.library_id = l.id
                    ORDER BY i.embedding <=> %s::vector
                    LIMIT 10
                """, (embedding, embedding))
                rows = cur.fetchall()
        else:
            cur.execute("""
                SELECT i.id, i.issue_number, i.title, i.body, l.name,
                       1 - (i.embedding <=> %s::vector) AS similarity
                FROM issues i
                JOIN libraries l ON i.library_id = l.id
                ORDER BY i.embedding <=> %s::vector
                LIMIT 10
            """, (embedding, embedding))
            rows = cur.fetchall()

        if not rows:
            return json.dumps({"error": "No issues found in database"})

        candidates = [
            {
                "id": r[0],
                "issue_number": r[1],
                "title": r[2],
                "body": (r[3] or "")[:300],
                "library": r[4],
                "distance": round(1 - r[5], 4),
            }
            for r in rows
        ]

        top_3_ids = rerank(query, candidates)

        output = []
        for rank, issue_id in enumerate(top_3_ids, 1):
            cur.execute("""
                SELECT i.issue_number, i.title, i.body, i.labels,
                       i.created_at, i.closed_at, i.resolution_text, i.comments, i.id,
                       l.name, l.organization,
                       1 - (i.embedding <=> %s::vector) AS similarity
                FROM issues i
                JOIN libraries l ON i.library_id = l.id
                WHERE i.id = %s
            """, (embedding, issue_id))
            row = cur.fetchone()
            if row:
                output.append({
                    "rank": rank,
                    "id": row[8],
                    "library": row[9],
                    "organization": row[10],
                    "issue_number": row[0],
                    "title": row[1],
                    "body": row[2],
                    "labels": row[3].split(",") if row[3] else [],
                    "created_at": row[4],
                    "closed_at": row[5],
                    "resolution_text": row[6],
                    "comments": row[7] if isinstance(row[7], list) else (json.loads(row[7]) if row[7] else []),
                    "distance": round(1 - row[11], 4),
                })

        result = {"library_fallback": True, "results": output} if library_fallback else {"results": output}

        cur.close()
        conn.close()

        log_hit("search_bugs", query=query, library=libraryName or None, results_returned=json.dumps([
            {"issue_number": r["issue_number"], "library": r["library"], "organization": r["organization"], "title": r["title"], "distance": r["distance"]}
            for r in output
        ]))
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error searching issues: {str(e)}"})


@mcp.tool(
    name="fetch_thread",
    description="""Returns the complete GitHub issue thread for a specific issue.

Use this when a search_bugs result references a linked issue number in its comments that may contain the actual fix.
For links pointing to external sites, use web search instead.

Each result includes the full thread: title, body, all comments, resolution text, library, and organization.

IMPORTANT: Do not call this tool more than once per linked issue."""
)
def fetch_thread(
    libraryName: str = Field(description="Library name where the issue is indexed (e.g. 'langchain', 'fastapi')"),
    issueNumber: int = Field(description="GitHub issue number found in a search_bugs comment (e.g. 1234)")
):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT i.id, i.issue_number, i.title, i.body, i.labels,
                   i.created_at, i.closed_at, i.resolution_text, i.comments, l.name, l.organization
            FROM issues i
            JOIN libraries l ON i.library_id = l.id
            WHERE l.name = %s AND i.issue_number = %s
        """, (libraryName, issueNumber))

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return json.dumps({"error": f"Issue #{issueNumber} not found for library '{libraryName}'"})

        result = {
            "id": row[0],
            "issue_number": row[1],
            "title": row[2],
            "body": row[3],
            "labels": row[4].split(",") if row[4] else [],
            "created_at": row[5],
            "closed_at": row[6],
            "resolution_text": row[7],
            "comments": row[8] if isinstance(row[8], list) else (json.loads(row[8]) if row[8] else []),
            "library": row[9],
            "organization": row[10],
        }

        log_hit("fetch_thread", library=libraryName, results_returned=json.dumps({"issue_number": result["issue_number"], "title": result["title"]}))
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error fetching thread: {str(e)}"})


@mcp.tool(
    name="request_bug_solutions",
    description="""Request a GitHub library to be indexed for bug solutions.

Use this when search_bugs returns no useful results and the library is not yet in the database.
Submits a request that is tracked by popularity — frequently requested libraries are prioritized for indexing."""
)
def request_bug_solutions(
    organizationName: str = Field(description="GitHub organization that owns the repo (e.g. 'openai', 'tiangolo')"),
    libraryName: str = Field(description="GitHub repository name (e.g. 'tiktoken', 'fastapi')")
):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO requests (organization, library)
            VALUES (%s, %s)
            RETURNING id, organization, library, requested_at
        """, (organizationName, libraryName))
        row = cur.fetchone()

        cur.execute("""
            SELECT COUNT(*) FROM requests WHERE organization = %s AND library = %s
        """, (organizationName, libraryName))
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
            "message": f"Request #{request_count} for {organizationName}/{libraryName}"
        }

        log_hit("request_bug_solutions", library=libraryName, results_returned=json.dumps({"organization": organizationName, "total_requests": request_count}))
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error requesting bug solutions: {str(e)}"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
