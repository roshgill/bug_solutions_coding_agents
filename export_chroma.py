import json
import chromadb
import numpy as np

print("Exporting local Chroma data...")

# Load local Chroma
local_client = chromadb.PersistentClient(path="./chroma_db")
collection = local_client.get_collection(name="github_issues")

# Get all data
data = collection.get(include=["embeddings", "documents", "metadatas"])

print(f"Exporting {len(data['ids'])} items...")

# Convert numpy arrays to lists for JSON serialization
export_data = {
    "ids": data["ids"],
    "embeddings": [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in data["embeddings"]],
    "documents": data["documents"],
    "metadatas": data["metadatas"]
}

# Save to JSON for backup and import
with open("chroma_export.json", "w") as f:
    json.dump(export_data, f, indent=2)

print(f"✓ Exported {len(data['ids'])} items to chroma_export.json")
print(f"  - IDs: {len(data['ids'])}")
print(f"  - Embeddings: {len(data['embeddings'])}")
print(f"  - Documents: {len(data['documents'])}")
print(f"  - Metadatas: {len(data['metadatas'])}")
