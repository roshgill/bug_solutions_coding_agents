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

# Libraries: (name, organization)
LIBRARIES = [
    ("python-sdk", "modelcontextprotocol")
]


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI's text-embedding-3-small model."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


conn = psycopg2.connect(DATABASE_URL, sslmode="require")
cur = conn.cursor()

print("Creating libraries table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS libraries (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE,
        embedding FLOAT8[],
        organization TEXT
    )
""")

print(f"Embedding and inserting {len(LIBRARIES)} libraries...")
for library_name, organization in LIBRARIES:
    embedding = generate_embedding(library_name)
    cur.execute("""
        INSERT INTO libraries (name, embedding, organization)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) DO NOTHING
    """, (library_name, embedding, organization))
    print(f"  ✓ {library_name} ({organization})")

conn.commit()

# Verify
cur.execute("SELECT id, name, organization FROM libraries ORDER BY id")
result = cur.fetchall()
print(f"\n✓ Libraries table populated with {len(result)} entries:")
for (lib_id, name, org) in result:
    print(f"  {lib_id}. {name} ({org})")

cur.close()
conn.close()
