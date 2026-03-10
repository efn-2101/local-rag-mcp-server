"""Remove backslash-path data from ChromaDB directly."""
import os
from rag_engine import RagEngine

engine = RagEngine()

# Get all paths in DB
items = engine.collection.get(include=["metadatas"])
all_paths = set()
if items and items["metadatas"]:
    for meta in items["metadatas"]:
        if meta and "path" in meta:
            all_paths.add(meta["path"])

print("Current paths in DB:", sorted(all_paths))

# Delete any path containing backslash (old format)
deleted_total = 0
for p in list(all_paths):
    if "\\" in p:
        print(f"Deleting backslash path: {repr(p)}")
        results = engine.collection.get(where={"path": p}, include=[])
        if results and results["ids"]:
            engine.collection.delete(ids=results["ids"])
            deleted_total += len(results["ids"])
            print(f"  -> Deleted {len(results['ids'])} chunks")

print(f"\nTotal deleted: {deleted_total} chunks")

# Verify
items2 = engine.collection.get(include=["metadatas"])
remaining = set()
if items2 and items2["metadatas"]:
    for meta in items2["metadatas"]:
        if meta and "path" in meta:
            remaining.add(meta["path"])
print("Remaining paths:", sorted(remaining))
