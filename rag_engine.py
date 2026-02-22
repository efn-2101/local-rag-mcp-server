import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
# 以下の設定でRustバックエンドでのSQLite不具合やアクセス違反を回避（もし動作するなら）
# os.environ["CHROMA_RUST_BINDINGS"] = "FALSE"
import json
import base64
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import ollama
import chromadb
from chromadb.utils import embedding_functions



class RagEngine:
    def __init__(self, config_path: str = "config.json"):
        # Resolve config_path relative to this script's directory
        base_dir = Path(__file__).parent.absolute()
        config_full_path = base_dir / config_path
        
        try:
            with open(config_full_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Fallback for when config might be passed as an absolute path or existing logic
             with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # Resolve docs_dir and db_dir relative to the config file location
        self.docs_dir = (base_dir / self.config["docs_dir"]).resolve()
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = (base_dir / self.config["db_dir"]).resolve()
        
        self.embedding_model = self.config["embedding_model"]
        self.ocr_model = self.config["ocr_model"]
        
        # Ollama client
        self.ollama_client = ollama.Client(host=self.config["ollama_base_url"])
        
        try:
            self.client = chromadb.PersistentClient(path=str(db_path))
            self.collection = self.client.get_or_create_collection(
                name=self.config["collection_name"]
            )
        except Exception as e:
            print(f"ChromaDB initialization failed: {e}. Recreating database...", file=sys.stderr)
            import shutil
            
            # Delete the corrupted directory
            if db_path.exists():
                try:
                    shutil.rmtree(db_path)
                except Exception as rmtree_err:
                    print(f"Failed to delete corrupted database directory: {rmtree_err}", file=sys.stderr)
                    # Create a backup instead if delete fails
                    import time
                    backup_path = str(db_path) + f"_corrupted_{int(time.time())}"
                    try:
                        os.rename(str(db_path), backup_path)
                        print(f"Moved corrupted database to {backup_path}", file=sys.stderr)
                    except Exception as rename_err:
                         print(f"Could not rename database either: {rename_err}", file=sys.stderr)
            
            # Re-initialize
            db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(db_path))
            self.collection = self.client.get_or_create_collection(
                name=self.config["collection_name"]
            )

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error in get_embedding (len={len(text)}): {e}", file=sys.stderr)
            raise e

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """テキストをチャンクに分割する"""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Adjust cut to nearest newline or space if possible
            if end < text_len:
                # Try to find a newline within the last 20% of the chunk
                lookback = int(chunk_size * 0.2)
                last_newline = chunk.rfind('\n')
                if last_newline > len(chunk) - lookback:
                    end = start + last_newline + 1
                    chunk = text[start:end]
                else:
                    last_space = chunk.rfind(' ')
                    if last_space > len(chunk) - lookback:
                        end = start + last_space + 1
                        chunk = text[start:end]

            chunks.append(chunk)
            start = end - overlap # Move back by overlap
            if start < 0: start = 0 # Safety
            
            # Avoid infinite loop if overlap >= chunk size (should not happen with defaults)
            if end <= start:
                start = end
                
        return chunks

    def add_document(self, file_path: Path):
        """ファイルをインデックスに追加する (チャンク分割あり)"""
        if not file_path.is_file():
            return
            
        # Only process markdown/text files for now
        if file_path.suffix.lower() not in [".md", ".txt"]:
            return

        category = file_path.parent.name if file_path.parent != self.docs_dir else "default"
        # WindowsバックスラッシュのchromaDB whereフィルター不具合を回避するため、常にフォワードスラッシュに正規化
        rel_path = str(file_path.relative_to(self.docs_dir)).replace("\\", "/")
        rel_parts = file_path.relative_to(self.docs_dir).parts
        root_folder = rel_parts[0] if len(rel_parts) > 0 else "default"
        
        # Get file modified time
        mtime = os.path.getmtime(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return
        
        if not content.strip():
            return

        # Delete existing chunks for this file first
        self.delete_document(file_path)

        chunks = self.chunk_text(content)
        print(f"Indexing {rel_path} ({len(content)} chars) -> {len(chunks)} chunks", file=sys.stderr)
        
        for i, chunk in enumerate(chunks):
            # IDはフォワードスラッシュ正規化されたrel_pathを使用
            chunk_id = f"{rel_path}#{i}"
            
            try:
                embedding = self.get_embedding(chunk)
                
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{"root_folder": root_folder, "category": category, "path": rel_path, "chunk_index": i, "overlap": self.config.get("chunk_overlap", 100), "mtime": mtime}],
                    documents=[chunk]
                )
            except Exception as e:
                print(f"CRITICAL ERROR indexing chunk {i} of {rel_path} (len={len(chunk)}): {e}", file=sys.stderr)
                # If 500 error, print response if possible
                if hasattr(e, "response"):
                     print(f"Response: {e.response.text if hasattr(e.response, 'text') else e.response}", file=sys.stderr)

    def delete_document(self, file_path: Path):
        """ファイルをインデックスから削除する"""
        # フォワードスラッシュに正規化（chromaDB whereフィルターがWindowsバックスラッシュで失敗するため）
        rel_path_fwd = str(file_path.relative_to(self.docs_dir)).replace("\\", "/")
        rel_path_bak = rel_path_fwd.replace("/", "\\")  # 旧データ（バックスラッシュ保存）も削除できるようフォールバック

        for rel_path in [rel_path_fwd, rel_path_bak]:
            try:
                # where句でのdeleteがRustバックエンドでクラッシュするため、getしてからid指定で削除する
                results = self.collection.get(where={"path": rel_path}, include=[])
                if results and results["ids"]:
                    self.collection.delete(ids=results["ids"])
                    print(f"Deleted from index: {rel_path} ({len(results['ids'])} chunks)", file=sys.stderr)
            except Exception as e:
                print(f"Error deleting {rel_path}: {e}", file=sys.stderr)


    def search(self, query: str, root_folder: Optional[str] = None, category: Optional[str] = None, n_results: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.get_embedding(query)
        
        # ChromaDB v1.5.x は $and 複合フィルターが非常に制限的（日本語含む値でクラッシュ）
        # なので root_folder のみ DB側でフィルタし、category はPython側でポストフィルタリングする
        where = None
        if root_folder:
            where = {"root_folder": root_folder}
        
        # categoryフィルタがある場合、多めに取得してPythonで絞り込む
        fetch_n = n_results * 5 if category else n_results
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_n,
                where=where
            )
        except Exception as e:
            print(f"Error in search (root={root_folder}, cat={category}): {e}", file=sys.stderr)
            return []
        
        output = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                
                # Pythonレベルでcategoryフィルタリング
                if category:
                    cat_list = category if isinstance(category, list) else [category]
                    if meta.get("category") not in cat_list:
                        continue
                
                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": meta,
                    "distance": results["distances"][0][i]
                })
                if len(output) >= n_results:
                    break
        return output

    def get_roots(self) -> List[str]:
        # ChromaDBからユニークなカテゴリを取得するのは少しトリッキーなので、
        # フォルダ構造から直接取得する
        roots = [d.name for d in self.docs_dir.iterdir() if d.is_dir()]
        return sorted(roots)

    def get_categories(self, allowed_roots: Optional[set] = None) -> List[str]:
        """
        インデックスされているドキュメントのメタデータからユニークなカテゴリを取得する。
        allowed_rootsが指定されている場合は、そのルートフォルダに属するカテゴリのみを返す。
        """
        try:
            result = self.collection.get(include=["metadatas"])
            if not result or not result["metadatas"]:
                return []
            
            categories = set()
            for meta in result["metadatas"]:
                if not meta:
                    continue
                root = meta.get("root_folder")
                cat = meta.get("category")
                
                # ACLの制限（allowed_roots）に合致する場合のみカテゴリを追加
                if cat and cat != "default":
                    if allowed_roots is None or root in allowed_roots:
                        categories.add(cat)
            
            return sorted(list(categories))
        except Exception as e:
            print(f"Error getting categories: {e}", file=sys.stderr)
            return []

    def list_documents(self) -> List[str]:
        """インデックスされている全ドキュメントのパスを取得する"""
        try:
            # メタデータのみ取得してpathを抽出
            result = self.collection.get(include=["metadatas"])
            if not result or not result["metadatas"]:
                return []
            
            paths = set()
            for meta in result["metadatas"]:
                if meta and "path" in meta:
                    paths.add(meta["path"])
            
            return sorted(list(paths))
        except Exception as e:
            print(f"Error listing documents: {e}", file=sys.stderr)
            return []

    def get_document_text(self, doc_path: str) -> Optional[str]:
        """指定されたパス（メタデータpath）のドキュメントのテキストを取得する (全チャンク結合)"""
        try:
            # メタデータで検索
            result = self.collection.get(where={"path": doc_path}, include=["documents", "metadatas"])
            
            if result and result["documents"] and len(result["documents"]) > 0:
                # チャンクをindex順に並べ替え
                chunks_with_index = []
                for i in range(len(result["documents"])):
                    meta = result["metadatas"][i]
                    index = meta.get("chunk_index", 0) if meta else 0
                    overlap = meta.get("overlap", 100) if meta else 100
                    chunks_with_index.append((index, result["documents"][i], overlap))
                
                chunks_with_index.sort(key=lambda x: x[0])
                
                # 結合 (最後のチャンク以外はoverlap分を除去)
                full_text = ""
                for i, (index, content, overlap) in enumerate(chunks_with_index):
                    # 簡易実装: contentの末尾 overlap 文字を除去して結合
                    # ただし、最後のチャンクだけはそのまま結合
                    if i < len(chunks_with_index) - 1:
                        # overlapがcontentの長さ以上なら空にする(あり得ないはずだが)
                        if len(content) > overlap:
                            full_text += content[:-overlap]
                        else:
                            # small content case
                            full_text += content 
                    else:
                        full_text += content
                
                return full_text
            return None
        except Exception as e:
            print(f"Error getting document text: {e}", file=sys.stderr)
            return None

    def sync_documents(self, force: bool = False, allowed_roots=None, progress_callback=None) -> Dict[str, Any]:
        """
        排他制御を利用して、ドキュメントの差分更新を安全に行う。
        Args:
            force: Trueの場合はmtimeを無視して全更新する
            allowed_roots: ACLで許可されたルートディレクトリ名のset。Noneの場合は全ディレクトリを対象とする
            progress_callback: 進捗通知用コールバック (phase, current, total, filename)
        """
        import time
        from filelock import FileLock, Timeout
        from file_converter import FileConverter
        
        lock_file = (Path(__file__).parent.absolute() / "index.lock")
        lock = FileLock(str(lock_file), timeout=5)
        
        results = {
            "status": "success",
            "converted": 0,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
            "message": ""
        }
        
        try:
            with lock:
                print("Acquired index lock. Starting sync...", file=sys.stderr)

                # --- Step 1: PDF/画像等を converted_docs/ に変換 ---
                base_dir = Path(__file__).parent.absolute()
                source_docs_dir_conf = self.config.get("source_docs_dir", "")
                if source_docs_dir_conf:
                    source_dir = Path(source_docs_dir_conf).resolve()
                else:
                    source_dir = None
                
                if source_dir and source_dir.exists():
                    print(f"Converting files from {source_dir}...", file=sys.stderr)
                    converter = FileConverter()
                    valid_md_files = set()
                    
                    for file_path in source_dir.rglob("*"):
                        if not file_path.is_file():
                            continue
                        if file_path.name.startswith("~$"):
                            continue
                        
                        # ACLフィルタ: 許可カテゴリ内のファイルのみ変換する
                        rel = file_path.relative_to(source_dir)
                        root_name = rel.parts[0] if len(rel.parts) > 0 else ""
                        if allowed_roots is not None and root_name not in allowed_roots:
                            continue
                        
                        out_path = self.docs_dir / rel.with_suffix(".md")
                        valid_md_files.add(out_path.resolve())
                        
                        # progress_callbackを FileConverter に渡す
                        def make_pdf_callback(fname):
                            def cb(current, total, _name):
                                if progress_callback:
                                    progress_callback("converting", current, total, fname)
                            return cb
                        
                        try:
                            converted = converter.convert_file(
                                file_path, out_path,
                                progress_callback=make_pdf_callback(file_path.name)
                            )
                            if converted:
                                results["converted"] = results.get("converted", 0) + 1
                        except Exception as e:
                            print(f"Conversion failed for {file_path.name}: {e}", file=sys.stderr)
                            results["errors"] = results.get("errors", 0) + 1
                    
                    # 許可カテゴリ内の孤児MDファイルを削除
                    for md_file in self.docs_dir.rglob("*.md"):
                        if not md_file.is_file():
                            continue
                        # ACL: 許可カテゴリ外はスキップ
                        md_root = md_file.relative_to(self.docs_dir).parts
                        md_root_name = md_root[0] if len(md_root) > 0 else ""
                        if md_file.resolve() not in valid_md_files:
                            print(f"Removing orphaned file: {md_file}", file=sys.stderr)
                            try:
                                md_file.unlink()
                                parent = md_file.parent
                                while parent != self.docs_dir:
                                    if not any(parent.iterdir()):
                                        parent.rmdir()
                                        parent = parent.parent
                                    else:
                                        break
                            except Exception as e:
                                print(f"Error removing {md_file}: {e}", file=sys.stderr)
                else:
                    print("source_docs_dir not configured or not found, skipping file conversion.", file=sys.stderr)

                # --- Step 2: converted_docs/ のMDを ChromaDB に同期 ---
                try:
                    current_items = self.collection.get(include=["metadatas"])
                except Exception as e:
                    print(f"Error retrieving collection data: {e}", file=sys.stderr)
                    current_items = None
                
                # Dictionary of (path: max_mtime_in_db)
                db_mtimes = {}
                # Set of all indexed paths to track deletions
                indexed_paths = set()
                
                if current_items and current_items["metadatas"]:
                    for meta in current_items["metadatas"]:
                        if meta and "path" in meta:
                            p = meta["path"].replace("\\", "/")
                            indexed_paths.add(p)
                            chunk_mtime = meta.get("mtime", 0.0)
                            # Get the max mtime among all chunks for this path safely
                            try:
                                current_max = db_mtimes.get(p, 0.0)
                                if chunk_mtime > current_max:
                                    db_mtimes[p] = chunk_mtime
                            except (TypeError, ValueError):
                                db_mtimes[p] = 0.0

                current_files_on_disk = set()
                
                # Check all files in docs_dir
                for file_path in self.docs_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                        
                    if file_path.suffix.lower() not in [".md", ".txt"]:
                        continue
                        
                    rel_path = str(file_path.relative_to(self.docs_dir)).replace("\\", "/")
                    current_files_on_disk.add(rel_path)
                    
                    try:
                        file_mtime = os.path.getmtime(file_path)
                    except Exception as e:
                        print(f"Error getting mtime for {rel_path}: {e}", file=sys.stderr)
                        results["errors"] += 1
                        continue
                        
                    is_new = rel_path not in indexed_paths
                    # If force is true, we always update. Otherwise we check mtime.
                    # Add a small epsilon (1.0) to avoid float precision issues
                    needs_update = force or (file_mtime > db_mtimes.get(rel_path, 0.0) + 1.0)
                    
                    if is_new or needs_update:
                        try:
                            if progress_callback:
                                progress_callback("indexing", 0, 0, rel_path)
                            self.add_document(file_path)
                            if is_new:
                                results["added"] = results.get("added", 0) + 1
                            else:
                                results["updated"] = results.get("updated", 0) + 1
                        except Exception as e:
                            print(f"Failed to index {rel_path}: {e}", file=sys.stderr)
                            results["errors"] = results.get("errors", 0) + 1
                    else:
                        results["skipped"] = results.get("skipped", 0) + 1

                # Find deleted files (in db but not on disk)
                deleted_files = indexed_paths - current_files_on_disk
                for deleted_path in deleted_files:
                    try:
                        abs_path = self.docs_dir / deleted_path
                        self.delete_document(abs_path)
                        results["deleted"] = results.get("deleted", 0) + 1
                    except Exception as e:
                        print(f"Failed to delete {deleted_path} from index: {e}", file=sys.stderr)
                        results["errors"] = results.get("errors", 0) + 1

                results["message"] = f"Sync complete. Converted: {results.get('converted', 0)}, Added: {results.get('added', 0)}, Updated: {results.get('updated', 0)}, Deleted: {results.get('deleted', 0)}, Skipped: {results.get('skipped', 0)} (Errors: {results.get('errors', 0)})"
                print(results["message"], file=sys.stderr)
                return results

        except Timeout:
            results["status"] = "error"
            results["message"] = "Could not acquire lock. Another sync process is likely running."
            print(results["message"], file=sys.stderr)
            return results
        except Exception as e:
            results["status"] = "error"
            results["message"] = f"An unexpected error occurred during sync: {e}"
            print(results["message"], file=sys.stderr)
            return results