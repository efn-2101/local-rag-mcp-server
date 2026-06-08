# ============================================
# Local RAG MCP Server - Docker Image
# Multi-stage build for optimized image size
# ============================================

# ============================================
# Stage 1: Builder - 依存関係のインストール
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# ビルド依存関係（一部パッケージで必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 仮想環境を作成
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime - 最終イメージ
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# 実行時に必要なシステムライブラリのみ
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PyMuPDF用
    libmupdf-dev \
    # その他
    curl \
    && rm -rf /var/lib/apt/lists/*

# 仮想環境をコピー
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# アプリケーションコード
COPY server.py .
COPY rag_engine.py .
COPY file_converter.py .
COPY ocr_engine.py .
COPY file_watcher.py .
COPY update_index.py .
COPY stop.py .
COPY _cleanup_db.py .

# 設定ファイル（テンプレート）
COPY config.json.example config.json.example

# ディレクトリ作成
RUN mkdir -p /app/documents /app/converted_docs /app/chroma_db /app/models/flashrank_cache

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV CHROMA_TELEMETRY=FALSE
ENV ANONYMIZED_TELEMETRY=FALSE

# ポート公開
EXPOSE 8000

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/sse || exit 1

# エントリーポイント
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
