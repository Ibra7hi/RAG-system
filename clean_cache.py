import sys
import psycopg2

def clean_semantic_cache():
    print("Connecting to PostgreSQL on port 6024...")
    try:
        conn = psycopg2.connect(
            dbname="rag_db",
            user="myuser",
            password="mypassword",
            host="localhost",
            port=6024
        )
        conn.autocommit = True
        cursor = conn.cursor()

        print("Clearing 'semantic_cache' collection from pgvector...")
        
        # 1. Delete all embedded queries/responses associated with the cache
        cursor.execute("""
            DELETE FROM langchain_pg_embedding 
            WHERE collection_id IN (
                SELECT uuid FROM langchain_pg_collection WHERE name = 'semantic_cache'
            );
        """)
        
        # 2. Delete the cache collection metadata
        cursor.execute("DELETE FROM langchain_pg_collection WHERE name = 'semantic_cache';")
        
        print("\n✅ Success: Semantic cache cleared! Your main document index is safe.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error while clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clean_semantic_cache()
