# export-docker-image.ps1
# Windows PowerShell用スクリプト
# Dockerイメージをビルドし、オフライン環境用にエクスポートする
#
# 注意: OllamaはDocker Composeに含まれず、外部サービスとして稼働します
# Ollamaモデルは別途ダウンロードして転送する必要があります

param(
    [string]$OutputDir = ".\offline-package",
    [string]$ImageName = "local-rag-mcp-server:latest"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Docker Image Export Script ===" -ForegroundColor Cyan
Write-Host "Output Directory: $OutputDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Ollama is not included in this package." -ForegroundColor Yellow
Write-Host "      Ollama must be installed and running separately on the target system." -ForegroundColor Yellow
Write-Host ""

# 出力ディレクトリ作成
Write-Host "Creating output directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\scripts" | Out-Null

# 1. イメージビルド
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build Docker image" -ForegroundColor Red
    exit 1
}

# 2. イメージ保存
Write-Host "Exporting MCP server image..." -ForegroundColor Yellow
$McpImage = Join-Path $OutputDir "local-rag-mcp-server.tar"

docker save -o $McpImage $ImageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to save MCP server image" -ForegroundColor Red
    exit 1
}

# 3. 設定ファイルとスクリプトをコピー
Write-Host "Copying configuration files..." -ForegroundColor Yellow
Copy-Item "docker-compose.yml" $OutputDir -ErrorAction SilentlyContinue
Copy-Item "config.json.example" $OutputDir -ErrorAction SilentlyContinue
Copy-Item "Dockerfile" $OutputDir -ErrorAction SilentlyContinue
Copy-Item "docker-entrypoint.sh" $OutputDir -ErrorAction SilentlyContinue

# 4. インストールスクリプト作成
Write-Host "Creating installation scripts..." -ForegroundColor Yellow

# Linux用インストールスクリプト
$InstallScript = @'
#!/bin/bash
# install-linux.sh
# Linux用インストールスクリプト（Ubuntu/Debian系、RHEL/CentOS系対応）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Local RAG MCP Server - Linux Installation ==="
echo ""
echo "Note: Ollama must be installed and running separately."
echo "      Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
echo ""

# Dockerインストール確認
install_docker_debian() {
    echo "Installing Docker (Debian/Ubuntu)..."
    
    # 依存パッケージ
    sudo apt-get update
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Docker公式GPGキー
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # リポジトリ追加
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Dockerインストール
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # ユーザーをdockerグループに追加
    sudo usermod -aG docker $USER
    
    echo "Docker installed successfully!"
    echo "Please log out and back in, then run this script again."
    exit 0
}

install_docker_rhel() {
    echo "Installing Docker (RHEL/CentOS/Rocky)..."
    
    # リポジトリ追加
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    
    # Dockerインストール
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # ユーザーをdockerグループに追加
    sudo usermod -aG docker $USER
    
    echo "Docker installed successfully!"
    echo "Please log out and back in, then run this script again."
    exit 0
}

if ! command -v docker &> /dev/null; then
    # ディストリビューション検出
    if [ -f /etc/debian_version ]; then
        install_docker_debian
    elif [ -f /etc/redhat-release ]; then
        install_docker_rhel
    else
        echo "Unsupported distribution. Please install Docker manually."
        exit 1
    fi
fi

# Dockerサービス開始
sudo systemctl start docker
sudo systemctl enable docker

# イメージロード
echo "Loading Docker images..."
docker load -i local-rag-mcp-server.tar

# 設定ファイル
if [ ! -f config.json ]; then
    cp config.json.example config.json
    echo "Created config.json from template."
fi

# ドキュメントディレクトリ作成
mkdir -p documents

# ボリュームディレクトリ作成
mkdir -p data/converted_docs data/chroma_db data/models

echo ""
echo "=== Installation Complete ==="
echo ""
echo "IMPORTANT: Ollama must be installed and running separately."
echo ""
echo "To install Ollama:"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo ""
echo "To download models:"
echo "  ollama pull nomic-embed-text-v2-moe"
echo "  ollama pull glm-ocr"
echo ""
echo "Next steps:"
echo "1. Ensure Ollama is running: ollama serve &"
echo "2. Place your documents in the 'documents' directory"
echo "3. Edit config.json:"
echo "   - Set source_docs_dir to your documents path"
echo "   - Set ollama_base_url (default: http://localhost:11434)"
echo "4. Start services:"
echo "   docker-compose up -d"
echo "5. Test the server:"
echo "   curl http://localhost:8000/sse"
echo ""
'@

$InstallScriptPath = Join-Path $OutputDir "install-linux.sh"
$InstallScript | Out-File -FilePath $InstallScriptPath -Encoding UTF8 -NoNewline

# 改行コードをLFに変換（Git Bash等で実行するため）
if (Get-Command dos2unix -ErrorAction SilentlyContinue) {
    dos2unix $InstallScriptPath
}

# 5. README作成
Write-Host "Creating README..." -ForegroundColor Yellow
$ReadmeContent = @'
# Local RAG MCP Server - Offline Package

このパッケージには、オフライン環境でLocal RAG MCP Serverを実行するために必要なファイルが含まれています。

## 重要: Ollamaについて

**Ollamaはこのパッケージに含まれていません。** Ollamaは独立したサービスとして別途インストール・稼働させる必要があります。

### Ollamaのインストール

```bash
# Linux用インストールスクリプト
curl -fsSL https://ollama.com/install.sh | sh

# または、バイナリをダウンロード
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
sudo cp ollama /usr/local/bin/
sudo chmod +x /usr/local/bin/ollama
```

### 必要なモデル

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull glm-ocr
```

## 含まれるファイル

- `local-rag-mcp-server.tar` - MCPサーバーDockerイメージ
- `docker-compose.yml` - Docker Compose設定
- `config.json.example` - 設定ファイルテンプレート
- `install-linux.sh` - インストールスクリプト

## インストール手順

### 1. Linux端末での準備

```bash
# USBメモリをマウント
sudo mkdir -p /mnt/usb
sudo mount /dev/sdX1 /mnt/usb  # sdX1は適切なデバイスに変更

# パッケージをコピー
cp -r /mnt/usb/offline-package ~/
cd ~/offline-package
```

### 2. インストール実行

```bash
chmod +x install-linux.sh
./install-linux.sh
```

### 3. Ollamaのインストールと起動

```bash
# Ollamaをインストール
curl -fsSL https://ollama.com/install.sh | sh

# Ollamaを起動
ollama serve &

# モデルをダウンロード
ollama pull nomic-embed-text-v2-moe
ollama pull glm-ocr
```

### 4. 設定ファイル編集

```bash
nano config.json
# source_docs_dir, docs_dir, ollama_base_url を設定
```

### 5. サービス起動

```bash
docker-compose up -d
```

### 6. 動作確認

```bash
curl http://localhost:8000/sse
```

## トラブルシューティング

### Dockerがインストールされていない場合

`install-linux.sh` スクリプトが自動的にDockerをインストールします。
インストール後、ログアウトしてから再度ログインしてください。

### Ollama接続エラー

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# 起動していない場合
ollama serve &
```

### ポートが使用中の場合

`docker-compose.yml` の `ports` セクションを編集して、ポート番号を変更してください。

## サポート

問題が発生した場合は、プロジェクトのIssueを確認してください。
'@

$ReadmePath = Join-Path $OutputDir "README.md"
$ReadmeContent | Out-File -FilePath $ReadmePath -Encoding UTF8

# サイズ表示
$TotalSize = (Get-ChildItem $OutputDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ""
Write-Host "Export complete!" -ForegroundColor Green
Write-Host "Output directory: $OutputDir"
Write-Host "Total size: $([math]::Round($TotalSize, 2)) GB"
Write-Host ""
Write-Host "Files created:"
Get-ChildItem $OutputDir -Recurse -File | ForEach-Object {
    $relativePath = $_.FullName.Substring((Resolve-Path $OutputDir).Path.Length + 1)
    Write-Host "  - $relativePath ($([math]::Round($_.Length / 1MB, 2)) MB)"
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Copy the '$OutputDir' folder to USB media"
Write-Host "2. On Linux, run: ./install-linux.sh"
Write-Host "3. Install and start Ollama separately"
Write-Host "4. Start services: docker-compose up -d"
Write-Host ""
Write-Host "IMPORTANT: Ollama is NOT included in this package." -ForegroundColor Yellow
Write-Host "           Install Ollama separately: curl -fsSL https://ollama.com/install.sh | sh" -ForegroundColor Yellow