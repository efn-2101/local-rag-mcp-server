import asyncio
import threading
import argparse
import sys
import os
import json
from urllib.parse import parse_qs, urlparse
from contextvars import ContextVar
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import mcp.types as types
from rag_engine import RagEngine
from pathlib import Path
from typing import Optional, Union, Any

print("Initializing RAG Engine...", file=sys.stderr)
engine = RagEngine()
server = Server("local-rag-server")

# ---------------------------------------------------------------------------
# acl.json をメモリにロード
# ---------------------------------------------------------------------------
def _load_acl() -> dict:
    acl_path = Path(__file__).parent / "acl.json"
    if not acl_path.exists():
        return {}
    try:
        with open(acl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # _comment のみ除外。_default は特別エントリとして保持する
        return {k: v for k, v in data.items() if k != "_comment"}
    except Exception as e:
        print(f"Error loading acl.json: {e}", file=sys.stderr)
        return {}

_ACL: dict = _load_acl()

def _resolve_allowed_roots(api_key_header: str) -> Optional[set]:
    """
    api_key_header はカンマ区切りで複数キーを指定可能。
    複数キーの allowed_roots を和集合(UNION)でマージする。
    - キーなし → acl.json の _default エントリを使用（なければ空setで全拒否）
    - 有効キーあり → 全ての許可ルートの和集合
    - 全て未知キー → 空のset（全アクセス拒否）
    """
    if not api_key_header:
        # _default エントリがあればその許可ルート、なければ全拒否
        default_entry = _ACL.get("_default")
        if default_entry:
            roots = set(default_entry.get("allowed_roots", []))
            print(f"Session ACL: no key, using _default allowed={roots}", file=sys.stderr)
            return roots
        print("Session ACL: no key and no _default entry. Denying all.", file=sys.stderr)
        return set()

    keys = [k.strip() for k in api_key_header.split(",") if k.strip()]
    merged: set = set()
    has_valid_key = False

    for key in keys:
        entry = _ACL.get(key)
        if entry is None:
            print(f"Warning: Unknown api_key '{key[:8]}...'. Skipping.", file=sys.stderr)
            continue
        has_valid_key = True
        roots = set(entry.get("allowed_roots", []))
        print(f"Session ACL: key={key[:8]}..., name={entry.get('name','?')}, allowed={roots}", file=sys.stderr)
        merged |= roots

    if not has_valid_key:
        return set()  # 全キーが未知 → 全アクセス拒否

    return merged

# ---------------------------------------------------------------------------
# ContextVar: センチネル値 _UNSET で「未設定」と「None（制限なし）」を区別
# ---------------------------------------------------------------------------
_UNSET = object()  # センチネル値

_ctx_allowed: ContextVar[Any] = ContextVar("allowed_roots", default=_UNSET)
_ctx_default: ContextVar[list] = ContextVar("default_roots", default=[])
_ctx_categories: ContextVar[list] = ContextVar("default_categories", default=[])

def _get_allowed() -> Optional[set]:
    val = _ctx_allowed.get()
    if val is _UNSET:
        # stdioモード: 環境変数から取得
        api_key = os.environ.get("MCP_API_KEY", "").strip()
        return _resolve_allowed_roots(api_key)
    return val  # set（許可ルート）または None（制限なし）

def _get_default() -> list:
    val = _ctx_default.get()
    if not val:
        env_roots = os.environ.get("DEFAULT_ROOTS", "").strip()
        return [c.strip() for c in env_roots.split(",") if c.strip()]
    return val

def _get_default_categories() -> list:
    val = _ctx_categories.get()
    if not val:
        env_cats = os.environ.get("DEFAULT_CATEGORIES", "").strip()
        return [c.strip() for c in env_cats.split(",") if c.strip()]
    return val

def get_effective_roots(requested: Optional[str] = None) -> Optional[list]:
    allowed = _get_allowed()
    default = _get_default()

    base: Optional[list] = [requested] if requested else (default if default else None)

    if allowed is None:
        return base  # 制限なし

    if base:
        filtered = [c for c in base if c in allowed]
        return filtered  # 空リスト = アクセス拒否
    else:
        return list(allowed) if allowed else []

def get_effective_categories(requested: Optional[str] = None) -> Optional[list]:
    default = _get_default_categories()
    if requested:
        return [requested]
    return default if default else None

# ---------------------------------------------------------------------------
# SSEセッションレジストリ（session_id → auth情報）
# handle_sse内のインターセプトで登録し、デバッグ用に使う
# ---------------------------------------------------------------------------
_session_registry: dict = {}

# ---------------------------------------------------------------------------
# ① 同期状態管理
# ---------------------------------------------------------------------------
sync_state: dict = {"status": "idle", "progress": "", "last_result": None}
_sync_lock = threading.Lock()

def _run_sync_background(force: bool, allowed_roots: Optional[set]):
    def on_progress(phase, current, total, filename):
        if phase == "converting":
            pct = int(current / total * 100) if total > 0 else 0
            msg = f"{filename}: {current}/{total}ページ ({pct}%) [OCR変換中]"
        elif phase == "indexing":
            msg = f"{filename} [DB登録中...]"
        else:
            msg = f"{filename}: {current}/{total}"
        with _sync_lock:
            sync_state["progress"] = msg

    with _sync_lock:
        sync_state.update({"status": "running", "progress": "開始しました...", "last_result": None})

    try:
        result = engine.sync_documents(
            force=force,
            allowed_roots=allowed_roots,
            progress_callback=on_progress,
        )
        with _sync_lock:
            sync_state["status"] = "done" if result["status"] == "success" else "error"
            sync_state["progress"] = result["message"]
            sync_state["last_result"] = result
    except Exception as e:
        with _sync_lock:
            sync_state.update({"status": "error", "progress": str(e),
                                "last_result": {"status": "error", "message": str(e)}})


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_documents",
            description=(
                "Search for information in the local document base (RAG). "
                "Respects root folder access control (ACL) and X-Roots header setting."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "root": {"type": "string", "description": "Optional root folder to filter by. (Checks ACL)"},
                    "category": {"type": "string", "description": "Optional category (subfolder) to filter by within the roots."},
                    "n_results": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_roots",
            description="List available document root folders (filtered by access control).",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_categories",
            description="List all available categories (subfolders) within the allowed root folders.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_document_content",
            description="Get the fully indexed content of a specific document by its path.",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path of the document"}},
                "required": ["path"],
            },
        ),
        types.Tool(
            name="list_documents",
            description="List all indexed documents.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="update_index",
            description=(
                "Start synchronizing the RAG document index in the background. "
                "Returns immediately. Use get_sync_status to check progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {"type": "boolean", "description": "Force full rebuild.", "default": False}
                },
            },
        ),
        types.Tool(
            name="get_sync_status",
            description="Get the current status and progress of the background index synchronization.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Optional[dict]
) -> list[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
    if arguments is None:
        arguments = {}

    if name == "search_documents":
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        effective_roots = get_effective_roots(arguments.get("root"))
        effective_categories = get_effective_categories(arguments.get("category"))

        if effective_roots is not None and len(effective_roots) == 0:
            return [types.TextContent(type="text", text="Access denied: you do not have permission to access the requested category.")]

        # If effective_categories is empty list [], we might want to treat it as None to avoid chromadb error
        if effective_categories is not None and len(effective_categories) == 0:
             effective_categories = None

        if effective_roots and len(effective_roots) == 1:
            results = engine.search(query, root_folder=effective_roots[0], category=effective_categories, n_results=n_results)
        elif effective_roots:
            merged = []
            # 複数ルートの場合は各ルートごとに検索してマージ
            for root in effective_roots:
                merged.extend(engine.search(query, root_folder=root, category=effective_categories, n_results=n_results))
            merged.sort(key=lambda x: x.get("distance", 1.0))
            results = merged[:n_results]
        else:
            results = engine.search(query, root_folder=None, category=effective_categories, n_results=n_results)

        if not results:
            return [types.TextContent(type="text", text="No relevant documents found.")]
        formatted = [
            f"--- Result (Root: {r['metadata']['root_folder']}, Category: {r['metadata']['category']}, Path: {r['id']}) ---\n{r['content']}\n"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(formatted))]

    elif name == "list_roots":
        roots = engine.get_roots()
        allowed = _get_allowed()
        if allowed is not None:
            roots = [c for c in roots if c in allowed]
        return [types.TextContent(type="text", text=f"Available root folders: {', '.join(roots)}")]

    elif name == "list_categories":
        allowed = _get_allowed()
        categories = engine.get_categories(allowed_roots=allowed)
        return [types.TextContent(type="text", text=f"Available categories: {', '.join(categories)}")]

    elif name == "get_document_content":
        doc_path = arguments.get("path", "")
        # パスの最初のディレクトリをroot_folderとしてACLチェック
        allowed = _get_allowed()
        if allowed is not None:
            import pathlib
            parts = pathlib.PurePosixPath(doc_path.replace("\\", "/")).parts
            doc_root = parts[0] if len(parts) > 0 else ""
            if doc_root not in allowed:
                return [types.TextContent(type="text", text="Access denied: you do not have permission to access this document.")]
        content = engine.get_document_text(doc_path)
        return [types.TextContent(type="text", text=content if content else "Document not found in index.")]

    elif name == "list_documents":
        docs = engine.list_documents()
        allowed = _get_allowed()
        if allowed is not None:
            # パスの最初のディレクトリ部分が許可ルートのもののみ
            import pathlib
            docs = [
                d for d in docs
                if pathlib.PurePosixPath(d.replace("\\", "/")).parts[0:1]
                and pathlib.PurePosixPath(d.replace("\\", "/")).parts[0] in allowed
            ]
        return [types.TextContent(type="text", text="Indexed Documents:\n" + "\n".join(docs))]

    elif name == "update_index":
        force_flag = arguments.get("force", False)
        with _sync_lock:
            cur_status = sync_state["status"]
            if cur_status == "running":
                return [types.TextContent(type="text", text=(
                    f"同期が既に実行中です。get_sync_status で確認してください。\n進捗: {sync_state['progress']}"
                ))]
            # done/error 状態の場合は idle にリセットして再同期を許可
            if cur_status in ("done", "error"):
                sync_state["status"] = "idle"
                sync_state["progress"] = "開始準備中..."
                sync_state["last_result"] = None

        allowed = _get_allowed()
        t = threading.Thread(target=_run_sync_background, args=(force_flag, allowed), daemon=True)
        t.start()
        return [types.TextContent(type="text", text=(
            "インデックスの同期をバックグラウンドで開始しました。\n"
            "get_sync_status で進捗を確認できます。"
        ))]

    elif name == "get_sync_status":
        with _sync_lock:
            state = dict(sync_state)
        label = {"idle": "待機中", "running": "同期中", "done": "完了", "error": "エラー"}.get(state["status"], state["status"])
        text = f"ステータス: {label}\n進捗: {state['progress']}"
        if state["last_result"]:
            text += f"\n最終結果: {state['last_result'].get('message', '')}"
        if state["status"] == "done":
            text += "\n\n⇒ 前回の同期プロセスは完了しています。新たに追加されたファイルをインデックスに反映させる場合は、再度 update_index を実行してください。"
        elif state["status"] == "error":
            text += "\n\n⇒ エラーが発生しました。update_index で再実行できます。"
        return [types.TextContent(type="text", text=text)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    init_options = InitializationOptions(
        server_name="local-rag-server",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )

    if args.transport == "stdio":
        from mcp.server.stdio import stdio_server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    else:
        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            from urllib.parse import unquote

            def _decode_header_value(raw: str) -> str:
                """
                HTTPヘッダーの非ASCII値をできる限り正しくデコードする。
                CLINEはElectron/fetch経由でUTF-8バイト列をそのままヘッダーに送信することがある。
                StarlettはHTTP/1.1仕様に従いヘッダーをLatin-1で解釈するため、
                UTF-8文字がLatin-1として化けてしまう（いわゆるMojibake）。
                ここでは以下の順で復元を試みる：
                  1. Latin-1→UTF-8 再デコード（Mojibake修正）
                  2. パーセントデコード（%XX形式のURLエンコード）
                  3. そのまま返す
                """
                if not raw:
                    return raw
                # 試1: Mojibake修正 (Latin-1バイト列をUTF-8として再解釈)
                try:
                    fixed = raw.encode("latin-1").decode("utf-8")
                    if fixed != raw:  # 変化があれば Mojibake だったと判断
                        return fixed
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
                # 試2: パーセントデコード
                decoded = unquote(raw)
                return decoded

            # 認証情報: HTTPヘッダー優先、クエリパラメータにフォールバック
            api_key = (
                request.headers.get("x-api-key", "").strip()
                or request.query_params.get("api_key", "").strip()
            )
            cats_param = _decode_header_value(
                request.headers.get("x-roots", "").strip()
                or request.query_params.get("roots", "").strip()
            )
            subcats_param = _decode_header_value(
                request.headers.get("x-categories", "").strip()
                or request.query_params.get("categories", "").strip()
            )
            allowed = _resolve_allowed_roots(api_key)
            default_roots = [c.strip() for c in cats_param.split(",") if c.strip()]
            default_categories = [c.strip() for c in subcats_param.split(",") if c.strip()]

            print(f"SSE connection: api_key={api_key[:8] if api_key else '(none)'}..., allowed={allowed}, default_roots={default_roots}, default_categories={default_categories}", file=sys.stderr)

            # ContextVarをここでセット → server.run()と内部タスクに継承される
            _ctx_allowed.set(allowed)
            _ctx_default.set(default_roots)
            _ctx_categories.set(default_categories)

            # SSEのsendをインターセプトして session_id を session_registry に登録（デバッグ用）
            original_send = request._send

            async def intercepting_send(message):
                if message.get("type") == "http.response.body":
                    body = message.get("body", b"")
                    if isinstance(body, bytes):
                        body = body.decode("utf-8", errors="ignore")
                    for line in body.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("data:") and "session_id=" in stripped:
                            data_part = stripped[len("data:"):].strip()
                            try:
                                parsed_url = urlparse(data_part)
                                qs = parse_qs(parsed_url.query)
                                sid = qs.get("session_id", [None])[0]
                                if sid:
                                    _session_registry[sid] = {"allowed": allowed, "default_roots": default_roots, "default_categories": default_categories}
                                    print(f"Registered session {sid[:8]}... allowed={allowed}", file=sys.stderr)
                            except Exception as e:
                                print(f"Session parse error: {e}", file=sys.stderr)
                await original_send(message)

            async with sse.connect_sse(request.scope, request.receive, intercepting_send) as (read_stream, write_stream):
                await server.run(read_stream, write_stream, init_options)
            return Response()

        app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust in production
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"], # Since we have custom headers like x-categories, just allow all
        )

        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()

if __name__ == "__main__":
    asyncio.run(main())
