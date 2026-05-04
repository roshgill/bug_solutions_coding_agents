import json
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

NEON_CONNECTION_STRING = os.getenv("DATABASE_URL")
if not NEON_CONNECTION_STRING:
    raise ValueError("DATABASE_URL not set in .env")

print("Connecting to Neon PostgreSQL...")

# Connect to Neon - use chromadb with PostgreSQL backend
try:
    neon_client = chromadb.HttpClient(
        host="ep-square-poetry-am8qgphc-pooler.c-5.us-east-1.aws.neon.tech",
        port=5432,
    )
except:
    # Fallback: try direct connection string
    neon_client = chromadb.PersistentClient(path=NEON_CONNECTION_STRING)

print("Loading exported data...")

with open("chroma_export.json", "r") as f:
    data = json.load(f)

print(f"Importing {len(data['ids'])} items to Neon...")

# Get or create collection in Neon
collection = neon_client.get_or_create_collection(name="github_issues")

# Import data in batches (Chroma has size limits)
batch_size = 100
for i in range(0, len(data['ids']), batch_size):
    batch_ids = data['ids'][i:i+batch_size]
    batch_embeddings = data['embeddings'][i:i+batch_size]
    batch_documents = data['documents'][i:i+batch_size]
    batch_metadatas = data['metadatas'][i:i+batch_size]
    
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )
    print(f"  Imported batch {i//batch_size + 1}/{(len(data['ids']) + batch_size - 1)//batch_size}")

print(f"✓ Successfully imported {len(data['ids'])} items to Neon!")
