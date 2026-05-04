import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL").split("?")[0]

print("Connecting to Neon...")
conn = psycopg2.connect(DATABASE_URL, sslmode="require")
cur = conn.cursor()

try:
    print("Creating tables...")
    
    # Create issues table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id VARCHAR(255) PRIMARY KEY,
            repo VARCHAR(255),
            issue_number INT,
            url VARCHAR(500),
            title TEXT,
            body TEXT,
            labels TEXT,
            created_at VARCHAR(50),
            closed_at VARCHAR(50),
            resolution_text TEXT,
            comments JSONB
        )
    """)
    
    # Create embeddings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id VARCHAR(255) PRIMARY KEY,
            embedding FLOAT8[],
            FOREIGN KEY (id) REFERENCES issues(id)
        )
    """)
    
    print("✓ Tables created\n")
    
    # Load exported data
    print("Loading data from chroma_export.json...")
    with open("chroma_export.json", "r") as f:
        data = json.load(f)
    
    print(f"Inserting {len(data['ids'])} records...\n")
    
    # Load corpus for additional data
    with open("issues_corpus.json", "r") as f:
        corpus = {issue["id"]: issue for issue in json.load(f)}
    
    # Insert data in batches
    for i, (issue_id, embedding, document) in enumerate(
        zip(data["ids"], data["embeddings"], data["documents"])
    ):
        # Get full issue data from corpus
        issue_data = corpus.get(issue_id, {})
        
        # Insert into issues table
        cur.execute("""
            INSERT INTO issues 
            (id, repo, issue_number, url, title, body, labels, created_at, closed_at, resolution_text, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            issue_id,
            issue_data.get("repo", ""),
            issue_data.get("issue_number", 0),
            issue_data.get("url", ""),
            issue_data.get("title", ""),
            issue_data.get("body", ""),
            ",".join(issue_data.get("labels", [])),
            issue_data.get("created_at", ""),
            issue_data.get("closed_at", ""),
            issue_data.get("resolution_text", ""),
            json.dumps(issue_data.get("comments", []))
        ))
        
        # Insert embedding
        cur.execute("""
            INSERT INTO embeddings (id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (issue_id, embedding))
        
        if (i + 1) % 10 == 0:
            print(f"  Inserted {i + 1}/{len(data['ids'])} records")
    
    conn.commit()
    print(f"\n✓ Successfully inserted {len(data['ids'])} records to Neon!")
    
    # Verify
    cur.execute("SELECT COUNT(*) FROM issues")
    issue_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM embeddings")
    embedding_count = cur.fetchone()[0]
    
    print(f"  - Issues: {issue_count}")
    print(f"  - Embeddings: {embedding_count}")

except Exception as e:
    conn.rollback()
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    cur.close()
    conn.close()
