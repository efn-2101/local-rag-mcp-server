import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rag_engine import RagEngine, _normalize_path

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, engine: RagEngine):
        self.engine = engine

    def _to_relative_path(self, abs_path: str):
        """絶対パスを docs_dir からの相対パスに変換"""
        from typing import Optional
        try:
            p = Path(abs_path).resolve()
            docs_dir = self.engine.docs_dir.resolve()
            if not str(p).startswith(str(docs_dir)):
                return None
            return p.relative_to(docs_dir)
        except Exception:
            return None

    def on_modified(self, event):
        if not event.is_directory:
            try:
                rel_path = self._to_relative_path(event.src_path)
                if rel_path is None:
                    return
                normalized_path = Path(_normalize_path(str(rel_path)))
                print(f"File modified: {normalized_path}", file=sys.stderr)
                if self.engine.has_document_changed(normalized_path):
                    self.engine.add_document(normalized_path)
                else:
                    print(f"Skipped re-indexing (content unchanged): {normalized_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error handling modification for {event.src_path}: {e}", file=sys.stderr)

    def on_created(self, event):
        if not event.is_directory:
            try:
                rel_path = self._to_relative_path(event.src_path)
                if rel_path is None:
                    return
                normalized_path = Path(_normalize_path(str(rel_path)))
                print(f"File created: {normalized_path}", file=sys.stderr)
                self.engine.add_document(normalized_path)
            except Exception as e:
                print(f"Error handling creation for {event.src_path}: {e}", file=sys.stderr)

    def on_deleted(self, event):
        if not event.is_directory:
            try:
                rel_path = self._to_relative_path(event.src_path)
                if rel_path is None:
                    return
                normalized_path = Path(_normalize_path(str(rel_path)))
                print(f"File deleted: {normalized_path}", file=sys.stderr)
                # delete_documentは相対パスが必要
                self.engine.delete_document(normalized_path)
            except Exception as e:
                print(f"Error handling deletion for {event.src_path}: {e}", file=sys.stderr)

    def on_moved(self, event):
        if not event.is_directory:
            try:
                src_rel = self._to_relative_path(event.src_path)
                dest_rel = self._to_relative_path(event.dest_path)
                if src_rel is None or dest_rel is None:
                    return
                src_normalized = Path(_normalize_path(str(src_rel)))
                dest_normalized = Path(_normalize_path(str(dest_rel)))
                print(f"File moved from {src_normalized} to {dest_normalized}", file=sys.stderr)
                self.engine.delete_document(src_normalized)
                self.engine.add_document(dest_normalized)
            except Exception as e:
                print(f"Error handling move from {event.src_path} to {event.dest_path}: {e}", file=sys.stderr)

def start_watcher(engine: RagEngine):
    handler = DocumentHandler(engine)
    observer = Observer()
    try:
        observer.schedule(handler, str(engine.docs_dir), recursive=True)
        observer.start()
        print(f"Started watching {engine.docs_dir}", file=sys.stderr)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping watcher...", file=sys.stderr)
        finally:
            observer.stop()
            observer.join()
    except Exception as e:
        print(f"Failed to start watcher: {e}", file=sys.stderr)

if __name__ == "__main__":
    engine = RagEngine()
    # 初期起動時に既存ファイルをスキャン
    print("Initial scan...", file=sys.stderr)
    for file_path in engine.docs_dir.rglob("*"):
        if file_path.is_file():
            engine.add_document(file_path)
    
    start_watcher(engine)