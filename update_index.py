import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

import argparse
import sys
import json
from rag_engine import RagEngine

def main():
    parser = argparse.ArgumentParser(description="Synchronize documents into the ChromaDB index.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the index (ignores mtime).")
    args = parser.parse_args()

    print("Initializing RAG Engine...", file=sys.stderr)
    engine = RagEngine(init_bm25=False)

    def on_progress(phase, current, total, filename):
        msg = {
            "type": "progress",
            "phase": phase,
            "current": current,
            "total": total,
            "filename": filename
        }
        print(f"SYNC_UPDATE_JSON:{json.dumps(msg)}", flush=True)

    print("Starting document synchronization...", file=sys.stderr)
    results = engine.sync_documents(force=args.force, progress_callback=on_progress)
    
    # 親プロセスへ最終結果を送信
    print(f"SYNC_UPDATE_JSON:{json.dumps({'type': 'result', 'data': results})}", flush=True)

    if results.get("status") == "error":
        print(f"Sync failed: {results.get('message')}", file=sys.stderr)
        sys.exit(1)
        
    print("Sync completed successfully.", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    main()
