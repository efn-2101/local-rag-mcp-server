#!/bin/bash
# docker-entrypoint.sh
# Local RAG MCP Server Docker Entrypoint Script
set -e

# config.json が存在しない場合はテンプレートからコピー
if [ ! -f /app/config.json ]; then
    cp /app/config.json.example /app/config.json
    echo "Created config.json from template"
fi

# Ollama接続確認（環境変数で指定可能）
OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
echo "Connecting to Ollama at: $OLLAMA_URL"

# Ollama接続待機（オプション）
if [ "${WAIT_FOR_OLLAMA:-true}" = "true" ]; then
    echo "Waiting for Ollama to be ready..."
    max_retries=30
    retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
            echo "Ollama is ready!"
            break
        fi
        retry_count=$((retry_count + 1))
        echo "Waiting for Ollama... ($retry_count/$max_retries)"
        sleep 2
    done
    
    if [ $retry_count -eq $max_retries ]; then
        echo "Warning: Ollama is not responding, but continuing startup..."
    fi
fi

# サーバー起動
echo "Starting Local RAG MCP Server..."
exec python server.py "$@"