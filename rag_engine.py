import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# 以下の設定で Rust バックエンドでの SQLite 不具合やアクセス違反を回避（もし動作するなら）
# os.environ["CHROMA_RUST_BINDINGS"] = "FALSE"
import json
import sys
import glob
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import ollama
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

# コンテキスト予算管理モジュール
from context_budget import (
    ContextBudget,
    count_tokens,
    parse_markdown_sections,
    extract_section_by_heading,
    compress_document_structure,
    fit_search_results_to_budget,
    truncate_to_tokens,
)


def _normalize_path(path_str: str) -> str:
    """
    パス文字列を正規化する。
    - バックスラッシュ → フォワードスラッシュ
    - 全角空白(U+3000)、タブ(U+0009)、ノーブレークスペース(U+00A0) 等の空白類文字 → 半角空白(U+0020)
    - 連続する半角空白を1つに圧縮
    """
    # バックスラッシュ → フォワードスラッシュ
    normalized = path_str.replace("\\", "/")
    # 各種空白類文字を半角空白に統一
    # U+3000 全角空白, U+00A0 ノーブレークスペース, U+0009 タブ, U+2000-U+200A 一般空白類
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"[\u2000-\u200a]", " ", normalized)
    # 連続する半角空白を1つに圧縮
    normalized = re.sub(r" {2,}", " ", normalized)
    # パス先頭・末尾の空白を除去
    normalized = normalized.strip(" ")
    return normalized


def _compute_file_hash(file_path: Path) -> str:
    """ファイル内容のSHA-256ハッシュを計算する"""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}", file=sys.stderr)
        return ""


def _compute_text_hash(text: str) -> str:
    """テキスト内容のSHA-256ハッシュを計算する"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RagEngine:
    def __init__(self, config_path: str = "config.json"):
        # Resolve config_path relative to this script's directory
        base_dir = Path(__file__).parent.absolute()
        
        # Try multiple locations for config.json
        config_locations = [
            base_dir / config_path,
            Path(config_path),
            base_dir / "config.json"
        ]
        
        self.config = None
        for loc in config_locations:
            if loc.exists():
                try:
                    with open(loc, "r", encoding="utf-8") as f:
                        self.config = json.load(f)
                    print(f"Config loaded from {loc}", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"Error loading config from {loc}: {e}", file=sys.stderr)

        if self.config is None:
            raise FileNotFoundError(f"Could not find or load config file from any of: {config_locations}")

        # Validate required config keys
        required_keys = ["docs_dir", "db_dir", "embedding_model", "ollama_base_url", "collection_name"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        # Resolve docs_dir and db_dir relative to the config file location
        self.docs_dir = (base_dir / self.config["docs_dir"]).resolve()
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = (base_dir / self.config["db_dir"]).resolve()
        
        self.embedding_model = self.config["embedding_model"]
        self.ocr_model = self.config["ocr_model"]
        
        # Ollama client
        self.ollama_client = ollama.Client(host=self.config["ollama_base_url"])
        
        # Additional text extensions from config
        self.extra_text_extensions = self.config.get("extra_text_extensions", [])
        self.allowed_extensions = [".md", ".txt"] + self.extra_text_extensions
        
        # Search settings from config
        search_settings = self.config.get("search_settings", {})
        self.hybrid_search_enabled = search_settings.get("hybrid_search_enabled", False)
        self.reranking_enabled = search_settings.get("reranking_enabled", False)
        self.flashrank_model = search_settings.get("flashrank_model", "ms-marco-MultiBERT-L-12")
        self.flashrank_cache_dir = search_settings.get("flashrank_cache_dir", "./models/flashrank_cache")
        self.flashrank_offline_mode = search_settings.get("flashrank_offline_mode", False)
        self.retrieval_limit_per_method = search_settings.get("retrieval_limit_per_method", 20)
        self.rerank_top_k = search_settings.get("rerank_top_k", 5)
        
        # Context management settings (for large document handling)
        ctx_mgmt = self.config.get("context_management", {})
        self.detail_level = ctx_mgmt.get("detail_level", "auto")  # "auto" | "summary" | "full"
        self.max_context_tokens = ctx_mgmt.get("max_context_tokens", 128000)
        self.max_document_tokens = ctx_mgmt.get("max_document_tokens", 12000)
        self.max_search_result_tokens = ctx_mgmt.get("max_search_result_tokens", 8000)
        self.summary_model = ctx_mgmt.get("summary_model", None)  # None = use default ollama model
        self.summary_ollama_url = ctx_mgmt.get("ollama_base_url", None)  # None = use default
        
        # Initialize context budget calculator
        self.context_budget = ContextBudget(
            max_context_tokens=self.max_context_tokens,
        )
        
        # Optional: separate Ollama client for summarization
        self.summary_ollama_client = None
        if self.summary_ollama_url and self.summary_ollama_url != self.config["ollama_base_url"]:
            try:
                self.summary_ollama_client = ollama.Client(host=self.summary_ollama_url)
            except Exception as e:
                print(f"Warning: Failed to initialize summary Ollama client: {e}", file=sys.stderr)
        
        # BM25 index attributes (will be initialized in sync_documents)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_texts: List[str] = []
        
        # FlashRank ranker (will be initialized safely)
        self.ranker: Optional[Ranker] = None
        
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
        
        # Initialize FlashRank ranker safely
        self._init_flashrank_ranker(base_dir)
        
        # Source file hash tracking for avoiding redundant OCR
        self.source_hashes_path = db_path.parent / ".source_hashes.json"
        self.source_hashes: Dict[str, str] = {}
        self._load_source_hashes()
        
        # Initialize BM25 index from existing documents
        self._init_bm25_index()

    def _load_source_hashes(self):
        """元ファイルのハッシュ状態を読み込む"""
        if self.source_hashes_path.exists():
            try:
                with open(self.source_hashes_path, "r", encoding="utf-8") as f:
                    self.source_hashes = json.load(f)
            except Exception as e:
                print(f"Error loading source hashes: {e}", file=sys.stderr)
                self.source_hashes = {}

    def _save_source_hashes(self):
        """元ファイルのハッシュ状態を保存する"""
        try:
            with open(self.source_hashes_path, "w", encoding="utf-8") as f:
                json.dump(self.source_hashes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving source hashes: {e}", file=sys.stderr)

    def _init_flashrank_ranker(self, base_dir: Path):
        """FlashRank Ranker を安全に初期化（オフライン水際防御）"""
        if not self.reranking_enabled:
            return
        
        try:
            if self.flashrank_offline_mode:
                # オフラインモード：モデルファイルの存在を確認
                model_dir = base_dir / self.flashrank_cache_dir / self.flashrank_model
                onnx_files = list(model_dir.glob("*.onnx"))
                
                if not model_dir.exists() or not onnx_files:
                    raise RuntimeError(
                        f"FlashRank モデルファイルが見つかりません: {model_dir} (*.onnx)"
                    )
                
                # 事前チェックをパスした場合、cache_dir を明示して初期化
                self.ranker = Ranker(
                    model_name=self.flashrank_model,
                    cache_dir=str(base_dir / self.flashrank_cache_dir)
                )
            else:
                # オンラインモード：cache_dir を明示して初期化
                self.ranker = Ranker(
                    model_name=self.flashrank_model,
                    cache_dir=str(base_dir / self.flashrank_cache_dir)
                )
        except Exception as e:
            print(f"FlashRank initialization failed: {e}", file=sys.stderr)
            self.ranker = None

    def _safe_collection_get(self, include=None, where=None):
        """
        ChromaDB Rust バックエンドのバグと SQLite バインド変数制限を回避する
        安全な collection.get() ラッパー。
        
        - 空コレクション時は即座に空結果を返す
        - limit/offset で分割取得し、大規模コレクションでも安全
        """
        # BUG-017 fix: Normalize include parameter
        include = include or []
        BATCH_SIZE = 5000  # SQLite 32,767 制限の余裕を持たせる
        
        # 空コレクション回避: count() は軽量
        try:
            total = self.collection.count()
        except Exception as e:
            print(f"Error in collection.count(): {e}", file=sys.stderr)
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        
        if total == 0:
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        
        # where 付きの場合、対象件数が少ない可能性があるが、安全のため分割取得
        all_ids = []
        all_documents = [] if include and "documents" in include else None
        all_metadatas = [] if include and "metadatas" in include else None
        all_embeddings = [] if include and "embeddings" in include else None
        
        offset = 0
        while True:
            try:
                batch = self.collection.get(
                    where=where,
                    include=include,
                    limit=BATCH_SIZE,
                    offset=offset
                )
            except Exception as e:
                print(f"Error in collection.get() batch offset={offset}: {e}", file=sys.stderr)
                raise
            
            if not batch or not batch.get("ids"):
                break
                
            all_ids.extend(batch["ids"])
            if all_documents is not None:
                all_documents.extend(batch.get("documents", []))
            if all_metadatas is not None:
                all_metadatas.extend(batch.get("metadatas", []))
            if all_embeddings is not None:
                all_embeddings.extend(batch.get("embeddings", []))
            
            if len(batch["ids"]) < BATCH_SIZE:
                break
            offset += BATCH_SIZE
        
        result = {"ids": all_ids}
        if all_documents is not None:
            result["documents"] = all_documents
        if all_metadatas is not None:
            result["metadatas"] = all_metadatas
        if all_embeddings is not None:
            result["embeddings"] = all_embeddings
        
        return result

    def _init_bm25_index(self):
        """既存のドキュメントから BM25 インデックスを初期化"""
        try:
            result = self._safe_collection_get(include=["documents"])
            if result and result["documents"]:
                self.bm25_texts = result["documents"]
                if self.bm25_texts:
                    # Tokenize documents for BM25
                    tokenized_docs = [self._tokenize_text(text) for text in self.bm25_texts]
                    self.bm25_index = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"Error initializing BM25 index: {e}", file=sys.stderr)
            self.bm25_index = None
            self.bm25_texts = []

    def _tokenize_text(self, text: str) -> List[str]:
        """テキストをトークン化（日本語・英語対応）"""
        # 簡易トークン化：空白で分割 + 日本語は文字単位
        import re
        # 英数・記号を空白で分割
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text)
        return tokens

    def _build_bm25_corpus(self):
        """ChromaDB のドキュメントから BM25 コーパスを再構築"""
        try:
            result = self._safe_collection_get(include=["documents"])
            if result and result["documents"]:
                self.bm25_texts = result["documents"]
                if self.bm25_texts:
                    tokenized_docs = [self._tokenize_text(text) for text in self.bm25_texts]
                    self.bm25_index = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"Error building BM25 corpus: {e}", file=sys.stderr)
            self.bm25_index = None
            self.bm25_texts = []

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
        
        # BUG-008 fix: Prevent extreme performance degradation
        if overlap >= chunk_size:
            print(f"Warning: overlap ({overlap}) >= chunk_size ({chunk_size}). Setting overlap to chunk_size // 2.", file=sys.stderr)
            overlap = chunk_size // 2
            
        print(f"[DEBUG] chunk_text: text_len={len(text)}, chunk_size={chunk_size}, overlap={overlap}", file=sys.stderr)
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Adjust cut to nearest newline or space if possible
            if end < text_len:
                chunk = text[start:end]
                # Try to find a newline within the last 20% of the chunk
                lookback = int(chunk_size * 0.2)
                last_newline = chunk.rfind('\n')
                if last_newline > len(chunk) - lookback:
                    end = start + last_newline + 1
                else:
                    last_space = chunk.rfind(' ')
                    if last_space > len(chunk) - lookback:
                        end = start + last_space + 1

            chunk = text[start:end]
            chunks.append(chunk)
            
            prev_start = start
            # Move next start
            start = end - overlap
            
            # Ensure progress and prevent infinite loop
            if start <= prev_start:
                start = end
            
            if start >= text_len:
                break
                
        print(f"[DEBUG] chunk_text: created {len(chunks)} chunks", file=sys.stderr)
        return chunks

    def add_document(self, file_path: Path):
        """ファイルをインデックスに追加する (チャンク分割あり)"""
        if not file_path.is_file():
            return
            
        # Only process markdown/text files for now
        if file_path.suffix.lower() not in self.allowed_extensions:
            return

        category = file_path.parent.name if file_path.parent != self.docs_dir else "default"
        # パスを正規化（バックスラッシュ→スラッシュ、全角空白→半角空白、連続空白圧縮）
        rel_path = _normalize_path(str(file_path.relative_to(self.docs_dir)))
        rel_parts = file_path.relative_to(self.docs_dir).parts
        root_folder = _normalize_path(rel_parts[0]) if len(rel_parts) > 0 else "default"
        
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

        # Compute content hash for change detection
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Delete existing chunks for this file first
        self.delete_document(file_path)

        chunks = self.chunk_text(content)
        print(f"Indexing {rel_path} ({len(content)} chars) -> {len(chunks)} chunks", file=sys.stderr)
        
        for i, chunk in enumerate(chunks):
            # ID はフォワードスラッシュ正規化された rel_path を使用
            chunk_id = f"{rel_path}#{i}"
            
            try:
                embedding = self.get_embedding(chunk)
                
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{"root_folder": root_folder, "category": category, "path": rel_path, "chunk_index": i, "overlap": self.config.get("chunk_overlap", 100), "mtime": mtime, "content_hash": content_hash}],
                    documents=[chunk]
                )
            except Exception as e:
                print(f"CRITICAL ERROR indexing chunk {i} of {rel_path} (len={len(chunk)}): {e}", file=sys.stderr)
                # If 500 error, print response if possible
                if hasattr(e, "response"):
                     print(f"Response: {e.response.text if hasattr(e.response, 'text') else e.response}", file=sys.stderr)
        
    def has_document_changed(self, file_path: Path) -> bool:
        """
        ファイル内容がDBに保存されているものと異なるかをチェック。
        ハッシュベースで判定し、ハッシュがない場合はmtimeフォールバック。
        """
        if not file_path.is_file():
            return True
        
        rel_path = _normalize_path(str(file_path.relative_to(self.docs_dir)))
        
        try:
            result = self._safe_collection_get(where={"path": rel_path}, include=["metadatas"])
            if result and result["metadatas"]:
                db_hash = result["metadatas"][0].get("content_hash", "")
                if db_hash:
                    # BUG-002 fix: Use text hash instead of binary file hash
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        current_hash = _compute_text_hash(content)
                        return current_hash != db_hash
                    except Exception as e:
                        print(f"Error reading file for hash comparison {rel_path}: {e}", file=sys.stderr)
                        return True
                # Fallback to mtime for backward compatibility
                db_mtime = result["metadatas"][0].get("mtime", 0.0)
                file_mtime = os.path.getmtime(file_path)
                return file_mtime > db_mtime + 1.0
        except Exception as e:
            print(f"Error checking document change for {rel_path}: {e}", file=sys.stderr)
        
        return True

    def delete_document(self, file_path: Path):
        """ファイルをインデックスから削除する"""
        # パスを正規化（バックスラッシュ→スラッシュ、全角空白→半角空白、連続空白圧縮）
        rel_path = _normalize_path(str(file_path.relative_to(self.docs_dir)))

        try:
            # where 句での delete が Rust バックエンドでクラッシュするため、get してから id 指定で削除する
            results = self.collection.get(where={"path": rel_path}, include=[])
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                print(f"Deleted from index: {rel_path} ({len(results['ids'])} chunks)", file=sys.stderr)
        except Exception as e:
            print(f"Error deleting {rel_path}: {e}", file=sys.stderr)
        
    def _rrf_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) を用いて 2 つの検索結果を統合
        数式: RRF_Score = 1 / (k + rank_in_dense) + 1 / (k + rank_in_sparse)
        """
        print(f"[DEBUG] _rrf_fusion: dense_results_count={len(dense_results)}, sparse_results_count={len(sparse_results)}, k={k}", file=sys.stderr)
        
        # Create ranking maps (id -> rank)
        dense_ranks = {item["id"]: idx + 1 for idx, item in enumerate(dense_results)}
        sparse_ranks = {item["id"]: idx + 1 for idx, item in enumerate(sparse_results)}
        
        # Build id -> item lookup dicts for correct retrieval (BUG-001 fix)
        dense_by_id = {item["id"]: item for item in dense_results}
        sparse_by_id = {item["id"]: item for item in sparse_results}
        
        # Collect all unique IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        print(f"[DEBUG] _rrf_fusion: all_ids_count={len(all_ids)}, dense_ranks_count={len(dense_ranks)}, sparse_ranks_count={len(sparse_ranks)}", file=sys.stderr)
        
        # Calculate RRF scores
        rrf_scores = []
        for doc_id in all_ids:
            dense_rank = dense_ranks.get(doc_id, float('inf'))
            sparse_rank = sparse_ranks.get(doc_id, float('inf'))
            
            # RRF formula
            rrf_score = 0.0
            if dense_rank != float('inf'):
                rrf_score += 1.0 / (k + dense_rank)
            if sparse_rank != float('inf'):
                rrf_score += 1.0 / (k + sparse_rank)
            
            # Get the document content from either result using id lookup (BUG-001 fix)
            doc_content = None
            doc_metadata = None
            if doc_id in dense_by_id:
                doc_content = dense_by_id[doc_id]["content"]
                doc_metadata = dense_by_id[doc_id]["metadata"]
            elif doc_id in sparse_by_id:
                doc_content = sparse_by_id[doc_id]["content"]
                doc_metadata = sparse_by_id[doc_id]["metadata"]
            
            rrf_scores.append({
                "id": doc_id,
                "content": doc_content,
                "metadata": doc_metadata,
                "rrf_score": rrf_score
            })
        
        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)
        print(f"[DEBUG] _rrf_fusion: returning {len(rrf_scores)} fused results", file=sys.stderr)
        
        return rrf_scores

    def _bm25_search(self, query: str, n_results: int, root_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """BM25 キーワード検索を実行"""
        if not self.bm25_index or not self.bm25_texts:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize_text(query)
        
        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get all IDs to map back
        all_data = self._safe_collection_get(include=["documents", "metadatas"])
        all_ids = all_data["ids"]
        all_docs = all_data["documents"]
        all_metas = all_data["metadatas"]

        # Create results with scores
        all_results = []
        for idx in range(len(bm25_scores)):
            if idx < len(all_ids):
                all_results.append({
                    "id": all_ids[idx],
                    "content": all_docs[idx],
                    "metadata": all_metas[idx],
                    "bm25_score": bm25_scores[idx]
                })
        
        # Filter by root_folder if specified
        if root_folder:
            all_results = [r for r in all_results if r["metadata"].get("root_folder") == root_folder]
            
        # Sort by score and take top n_results
        all_results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return all_results[:n_results]

    def search(self, query: str, root_folder: Optional[str] = None, category: Optional[Union[str, List[str]]] = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索パイプライン（4 ステップ）:
        Step 1: Retrieval（初期検索）
        Step 2: Fusion（RRF による統合）
        Step 3: Re-ranking（FlashRank による再評価）
        Step 4: 結果の返却
        """
        if not query or not query.strip():
            return []

        query_embedding = None
        try:
            query_embedding = self.get_embedding(query)
        except Exception as e:
            print(f"Warning: Failed to get embedding for query: {e}", file=sys.stderr)
        
        # ChromaDB v1.5.x は $and 複合フィルターが非常に制限的（日本語含む値でクラッシュ）
        # なので root_folder のみ DB 側でフィルタし、category は Python 側でポストフィルタリングする
        where = None
        if root_folder:
            where = {"root_folder": root_folder}
        
        # Step 1: Retrieval（初期検索）
        dense_results = []
        sparse_results = []
        
        # ChromaDB (Dense): ベクトル検索
        fetch_n = self.retrieval_limit_per_method if (self.hybrid_search_enabled or category) else n_results
        
        if query_embedding:
            try:
                dense_response = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_n,
                    where=where
                )
                if dense_response["ids"] and len(dense_response["ids"]) > 0:
                    for i in range(len(dense_response["ids"][0])):
                        meta = dense_response["metadatas"][0][i]
                        dense_results.append({
                            "id": dense_response["ids"][0][i],
                            "content": dense_response["documents"][0][i],
                            "metadata": meta,
                            "distance": dense_response["distances"][0][i]
                        })
            except Exception as e:
                print(f"Error in dense search: {e}", file=sys.stderr)
        
        # BM25 (Sparse): キーワード検索
        # ハイブリッド検索が有効な場合、またはベクトル検索が失敗した場合に実行
        if (self.hybrid_search_enabled or not query_embedding) and self.bm25_index:
            sparse_results = self._bm25_search(query, self.retrieval_limit_per_method, root_folder=root_folder)
        
        # Step 2: Fusion（RRF による統合）
        if self.hybrid_search_enabled and dense_results and sparse_results:
            fused_results = self._rrf_fusion(dense_results, sparse_results)
        else:
            # ハイブリッド検索が無効または一方のみの結果
            fused_results = dense_results if dense_results else sparse_results
        
        # Step 3: Re-ranking（FlashRank による再評価）
        if self.reranking_enabled and self.ranker and fused_results:
            # FlashRank が要求する passages フォーマットに変換
            passages = [{"id": item["id"], "text": item["content"]} for item in fused_results]
            
            try:
                request = RerankRequest(query=query, passages=passages)
                rerank_results = self.ranker.rerank(request)
                
                # Rerank results contain 'score' field, sort by score descending
                reranked_items = []
                for result in rerank_results:
                    # Find the original item by id
                    for item in fused_results:
                        if item["id"] == result["id"]:
                            reranked_items.append({
                                **item,
                                "rerank_score": result.get("score", 0.0)
                            })
                            break
                
                # Sort by rerank_score descending
                reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)
                fused_results = reranked_items
            except Exception as e:
                print(f"Error in reranking: {e}", file=sys.stderr)
        
        # Step 4: 結果の返却
        # category フィルタを適用
        print(f"[DEBUG] search: category={category}, type={type(category)}, fused_results_count={len(fused_results)}", file=sys.stderr)
        output = []
        for item in fused_results:
            if category:
                cat_list = category if isinstance(category, list) else [category]
                print(f"[DEBUG] search: checking item category={item['metadata'].get('category')} against cat_list={cat_list}", file=sys.stderr)
                if item["metadata"].get("category") not in cat_list:
                    continue
            
            output.append(item)
            if len(output) >= n_results:
                break
        
        print(f"[DEBUG] search: returning {len(output)} results", file=sys.stderr)
        return output
    
    def search_with_budget(
        self,
        query: str,
        root_folder: Optional[str] = None,
        category: Optional[Union[str, List[str]]] = None,
        n_results: int = 5,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        検索結果をトークン予算内に収めて返却する。
        
        Args:
            query: 検索クエリ
            root_folder: ルートフォルダフィルタ
            category: カテゴリフィルタ
            n_results: 結果件数
            max_tokens: 検索結果の最大トークン数（Noneの場合はconfigの値を使用）
        
        Returns:
            {
                "text": フォーマット済みテキスト,
                "truncated": 切り詰められたか,
                "omitted_ids": 省略された結果IDリスト,
                "result_count": 表示された結果件数,
                "total_results": 全結果件数,
            }
        """
        # 通常の検索を実行
        results = self.search(query, root_folder, category, n_results)
        
        if not results:
            return {
                "text": "No relevant documents found.",
                "truncated": False,
                "omitted_ids": [],
                "result_count": 0,
                "total_results": 0,
            }
        
        effective_max_tokens = max_tokens or self.max_search_result_tokens
        
        # 検索結果を予算内に収める
        formatted_text, truncated, omitted_ids = fit_search_results_to_budget(
            results,
            budget_tokens=effective_max_tokens,
            format_template="--- Result (Root: {root}, Category: {category}, Path: {path}) ---\n{content}\n",
        )
        
        return {
            "text": formatted_text,
            "truncated": truncated,
            "omitted_ids": omitted_ids,
            "result_count": len(results) - len(omitted_ids),
            "total_results": len(results),
        }

    def get_roots(self) -> List[str]:
        # ChromaDB からユニークなカテゴリを取得するのは少しトリッキーなので、
        # フォルダ構造から直接取得する
        roots = [d.name for d in self.docs_dir.iterdir() if d.is_dir()]
        return sorted(roots)

    def get_categories(self, allowed_roots: Optional[set] = None) -> List[str]:
        """
        インデックスされているドキュメントのメタデータからユニークなカテゴリを取得する。
        allowed_roots が指定されている場合は、そのルートフォルダに属するカテゴリのみを返す。
        """
        try:
            result = self._safe_collection_get(include=["metadatas"])
            if not result or not result["metadatas"]:
                return []
            
            categories = set()
            for meta in result["metadatas"]:
                if not meta:
                    continue
                root = meta.get("root_folder")
                cat = meta.get("category")
                
                # ACL の制限（allowed_roots）に合致する場合のみカテゴリを追加
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
            # メタデータのみ取得して path を抽出
            result = self._safe_collection_get(include=["metadatas"])
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

    def get_document_text(
        self,
        doc_path: str,
        section: Optional[str] = None,
        max_tokens: Optional[int] = None,
        detail_level: Optional[str] = None,
    ) -> Optional[str]:
        """
        指定されたパス（メタデータ path）のドキュメントのテキストを取得する。
        
        Args:
            doc_path: ドキュメントのパス
            section: 特定のセクション名（見出し）を指定して抽出
            max_tokens: 返却する最大トークン数（Noneの場合はconfigの値を使用）
            detail_level: "auto" | "summary" | "full"（Noneの場合はconfigの値を使用）
        
        Returns:
            ドキュメントのテキスト（制限・圧縮適用済み）
        """
        # パスを正規化（検索時に全角空白等が混在しても一致するように）
        normalized_path = _normalize_path(doc_path)
        print(f"[DEBUG] get_document_text: doc_path={doc_path}, normalized={normalized_path}, section={section}", file=sys.stderr)
        
        try:
            # メタデータで検索
            result = self._safe_collection_get(where={"path": normalized_path}, include=["documents", "metadatas"])
            
            if not result or not result["documents"] or len(result["documents"]) == 0:
                print(f"[DEBUG] get_document_text: no chunks found for {doc_path}", file=sys.stderr)
                return None
            
            print(f"[DEBUG] get_document_text: found {len(result['documents'])} chunks", file=sys.stderr)
            
            # チャンクを index 順に並べ替え
            chunks_with_index = []
            for i in range(len(result["documents"])):
                meta = result["metadatas"][i]
                index = meta.get("chunk_index", 0) if meta else 0
                overlap = meta.get("overlap", 100) if meta else 100
                chunks_with_index.append((index, result["documents"][i], overlap))
            
            chunks_with_index.sort(key=lambda x: x[0])
            
            # 結合 (チャンク間の重複部分を考慮して結合)
            full_text = ""
            for i, (index, content, overlap) in enumerate(chunks_with_index):
                if i == 0:
                    full_text = content
                else:
                    max_overlap = min(len(full_text), len(content), overlap + 50)
                    actual_overlap = 0
                    for j in range(max_overlap, 0, -1):
                        if full_text.endswith(content[:j]):
                            actual_overlap = j
                            break
                    full_text += content[actual_overlap:]
            
            print(f"[DEBUG] get_document_text: total length={len(full_text)}, tokens={count_tokens(full_text)}", file=sys.stderr)
            
            # セクション指定がある場合は抽出
            if section:
                extracted = extract_section_by_heading(full_text, section)
                if extracted:
                    print(f"[DEBUG] get_document_text: extracted section '{section}', tokens={count_tokens(extracted)}", file=sys.stderr)
                    return extracted
                else:
                    # セクションが見つからない場合は、利用可能なセクション一覧を返す
                    sections = parse_markdown_sections(full_text)
                    available = [s.heading for s in sections[:20]]  # 最大20件
                    return (
                        f"[エラー: セクション '{section}' が見つかりませんでした。]\n\n"
                        f"[利用可能なセクション:]\n" +
                        "\n".join(f"- {h}" for h in available) +
                        "\n\nget_document_content(path=\"" + doc_path + "\", section=\"セクション名\") で詳細を取得できます。"
                    )
            
            # トークン制限の適用
            effective_max_tokens = max_tokens or self.max_document_tokens
            effective_detail_level = detail_level or self.detail_level
            
            total_tokens = count_tokens(full_text)
            
            # fullモードで予算内に収まる場合はそのまま返す
            if effective_detail_level == "full" and total_tokens <= effective_max_tokens:
                return full_text
            
            # autoモードで予算内に収まる場合もそのまま返す
            if effective_detail_level == "auto" and total_tokens <= effective_max_tokens:
                return full_text
            
            # 圧縮が必要な場合
            compressed_text, omitted_sections, was_compressed = compress_document_structure(
                full_text,
                budget_tokens=effective_max_tokens,
                detail_level=effective_detail_level,
            )
            
            if was_compressed:
                print(f"[DEBUG] get_document_text: compressed from {total_tokens} to {count_tokens(compressed_text)} tokens", file=sys.stderr)
            
            return compressed_text
            
        except Exception as e:
            print(f"Error getting document text: {e}", file=sys.stderr)
            return None
    
    def get_document_sections(self, doc_path: str) -> List[Dict[str, Any]]:
        """
        ドキュメントのセクション一覧を取得する。
        
        Returns:
            [{"heading": str, "level": int, "tokens": int}, ...]
        """
        normalized_path = _normalize_path(doc_path)
        try:
            result = self._safe_collection_get(where={"path": normalized_path}, include=["documents"])
            if not result or not result["documents"]:
                return []
            
            # チャンクを結合
            chunks_with_index = []
            for i in range(len(result["documents"])):
                chunks_with_index.append((i, result["documents"][i]))
            chunks_with_index.sort(key=lambda x: x[0])
            full_text = "".join(c[1] for c in chunks_with_index)
            
            sections = parse_markdown_sections(full_text)
            return [
                {
                    "heading": s.heading,
                    "level": s.level,
                    "tokens": s.get_token_count(),
                }
                for s in sections
            ]
        except Exception as e:
            print(f"Error getting document sections: {e}", file=sys.stderr)
            return []

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
                        
                        rel_key = _normalize_path(str(rel))
                        current_hash = _compute_file_hash(file_path)
                        
                        needs_conversion = True
                        if current_hash:
                            stored_hash = self.source_hashes.get(rel_key)
                            if stored_hash:
                                if stored_hash == current_hash and out_path.exists():
                                    needs_conversion = False
                                    print(f"Skipped conversion (content unchanged): {rel}", file=sys.stderr)
                            elif out_path.exists():
                                # Backward compatibility: no stored hash but MD exists
                                needs_conversion = False
                                self.source_hashes[rel_key] = current_hash
                                print(f"Stored hash for existing MD (skipped conversion): {rel}", file=sys.stderr)
                        
                        if not needs_conversion:
                            continue
                        
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
                            if converted is True:
                                results["converted"] = results["converted"] + 1
                                if current_hash:
                                    self.source_hashes[rel_key] = current_hash
                            elif converted is None:
                                # BUG-004 fix: OCR failure should be counted as error
                                results["errors"] = results["errors"] + 1
                                print(f"Conversion failed (OCR error): {rel}", file=sys.stderr)
                        except Exception as e:
                            print(f"Conversion failed for {file_path.name}: {e}", file=sys.stderr)
                            results["errors"] = results["errors"] + 1
                    
                    # source_hashes から存在しないファイルのエントリを削除
                    source_files_on_disk = set()
                    for fp in source_dir.rglob("*"):
                        if fp.is_file() and not fp.name.startswith("~$"):
                            rel = fp.relative_to(source_dir)
                            source_files_on_disk.add(_normalize_path(str(rel)))
                    
                    orphaned_hashes = [k for k in self.source_hashes if k not in source_files_on_disk]
                    for k in orphaned_hashes:
                        del self.source_hashes[k]
                    if orphaned_hashes:
                        print(f"Removed {len(orphaned_hashes)} orphaned hash entries", file=sys.stderr)
                    
                    # 許可カテゴリ内の孤児MDファイルを削除
                    for md_file in self.docs_dir.rglob("*.md"):
                        if not md_file.is_file():
                            continue
                        # ACL: 許可カテゴリ外はスキップ
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
                    current_items = self._safe_collection_get(include=["metadatas"])
                except Exception as e:
                    print(f"Error retrieving collection data: {e}", file=sys.stderr)
                    current_items = None
                
                # Dictionary of (path: max_mtime_in_db)
                db_mtimes = {}
                # Dictionary of (path: content_hash_in_db)
                db_hashes = {}
                # Set of all indexed paths to track deletions
                indexed_paths = set()
                
                if current_items and current_items["metadatas"]:
                    for meta in current_items["metadatas"]:
                        if meta and "path" in meta:
                            p = _normalize_path(meta["path"])
                            indexed_paths.add(p)
                            chunk_mtime = meta.get("mtime", 0.0)
                            # Get the max mtime among all chunks for this path safely
                            try:
                                current_max = db_mtimes.get(p, 0.0)
                                if chunk_mtime > current_max:
                                    db_mtimes[p] = chunk_mtime
                            except (TypeError, ValueError):
                                db_mtimes[p] = 0.0
                            # Get the content_hash (all chunks for a file share the same hash)
                            h = meta.get("content_hash", "")
                            if h:
                                db_hashes[p] = h

                current_files_on_disk = set()
                
                # Check all files in docs_dir
                for file_path in self.docs_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                        
                    if file_path.suffix.lower() not in self.allowed_extensions:
                        continue
                        
                    rel_path = _normalize_path(str(file_path.relative_to(self.docs_dir)))
                    current_files_on_disk.add(rel_path)
                    
                    try:
                        file_mtime = os.path.getmtime(file_path)
                    except Exception as e:
                        print(f"Error getting mtime for {rel_path}: {e}", file=sys.stderr)
                        results["errors"] += 1
                        continue
                    
                    is_new = rel_path not in indexed_paths
                    
                    if force:
                        needs_update = True
                    elif is_new:
                        needs_update = True
                    elif rel_path in db_hashes:
                        # Hash-based change detection (preferred)
                        current_hash = _compute_file_hash(file_path)
                        needs_update = current_hash != db_hashes[rel_path]
                    else:
                        # Fallback: mtime-based change detection for backward compatibility
                        # (existing documents that don't have content_hash yet)
                        needs_update = file_mtime > db_mtimes.get(rel_path, 0.0) + 1.0
                    
                    if is_new or needs_update:
                        try:
                            if progress_callback:
                                progress_callback("indexing", 0, 0, rel_path)
                            self.add_document(file_path)
                            if is_new:
                                results["added"] = results["added"] + 1
                            else:
                                results["updated"] = results["updated"] + 1
                        except Exception as e:
                            print(f"Failed to index {rel_path}: {e}", file=sys.stderr)
                            results["errors"] = results["errors"] + 1
                    else:
                        results["skipped"] = results["skipped"] + 1

                # Find deleted files (in db but not on disk)
                deleted_files = indexed_paths - current_files_on_disk
                for deleted_path in deleted_files:
                    try:
                        abs_path = self.docs_dir / deleted_path
                        self.delete_document(abs_path)
                        results["deleted"] = results["deleted"] + 1
                    except Exception as e:
                        print(f"Failed to delete {deleted_path} from index: {e}", file=sys.stderr)
                        results["errors"] = results["errors"] + 1

                # 元ファイルハッシュ状態を保存
                self._save_source_hashes()
                
                # BM25 インデックスを同期後に一括更新
                self._build_bm25_corpus()

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
