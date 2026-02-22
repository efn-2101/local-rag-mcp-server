import argparse
import sys
from rag_engine import RagEngine

def main():
    parser = argparse.ArgumentParser(description="Synchronize documents into the ChromaDB index.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the index (ignores mtime).")
    args = parser.parse_args()

    print("Initializing RAG Engine...", file=sys.stderr)
    engine = RagEngine()

    print("Starting document synchronization...", file=sys.stderr)
    results = engine.sync_documents(force=args.force)
    
    if results.get("status") == "error":
        print(f"Sync failed: {results.get('message')}", file=sys.stderr)
        sys.exit(1)
        
    print("Sync completed successfully.", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    main()
