import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rag_engine import RagEngine

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, engine: RagEngine):
        self.engine = engine

    def on_modified(self, event):
        if not event.is_directory:
            print(f"File modified: {event.src_path}", file=sys.stderr)
            self.engine.add_document(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            print(f"File created: {event.src_path}", file=sys.stderr)
            self.engine.add_document(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            print(f"File deleted: {event.src_path}", file=sys.stderr)
            # delete_documentは相対パスが必要
            self.engine.delete_document(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            print(f"File moved from {event.src_path} to {event.dest_path}", file=sys.stderr)
            self.engine.delete_document(Path(event.src_path))
            self.engine.add_document(Path(event.dest_path))

def start_watcher(engine: RagEngine):
    handler = DocumentHandler(engine)
    observer = Observer()
    observer.schedule(handler, str(engine.docs_dir), recursive=True)
    observer.start()
    print(f"Started watching {engine.docs_dir}", file=sys.stderr)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    engine = RagEngine()
    # 初期起動時に既存ファイルをスキャン
    print("Initial scan...", file=sys.stderr)
    for file_path in engine.docs_dir.rglob("*"):
        if file_path.is_file():
            engine.add_document(file_path)
    
    start_watcher(engine)