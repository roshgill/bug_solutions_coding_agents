import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env")
    exit(1)

print(f"Original URL: {DATABASE_URL[:80]}...")

# Clean up the URL - remove extra parameters
clean_url = DATABASE_URL.split("?")[0]  # Remove query params
print(f"Clean URL: {clean_url}\n")

try:
    conn = psycopg2.connect(clean_url, sslmode="require")
    print("✓ Connection established\n")
    
    cur = conn.cursor()
    
    # List all tables
    print("=== Tables in database ===")
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = cur.fetchall()
    
    if not tables:
        print("No tables found!")
    else:
        for (table_name,) in tables:
            print(f"  - {table_name}")
            
            # Count rows in each table
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                print(f"    └─ {count} rows")
            except Exception as e:
                print(f"    └─ Error: {e}")
    
    print("\n=== Checking for Chroma tables ===")
    chroma_tables = ["collections", "embeddings", "documents"]
    for table in chroma_tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table}: {count} rows")
        except:
            print(f"  {table}: NOT FOUND")
    
    cur.close()
    conn.close()

except Exception as e:
    print(f"✗ Connection failed: {e}")
