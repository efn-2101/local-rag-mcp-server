#!/bin/bash
# import-docker-image.sh
# Linux用Dockerイメージインポートスクリプト
# オフライン環境でのLocal RAG MCP Serverインストール用
#
# 注意: OllamaはDocker Composeに含まれず、外部サービスとして稼働します
# Ollamaモデルは別途インストール・ダウンロードする必要があります

set -e

# ============================================
# 設定
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="local-rag-mcp-server:latest"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================
# ヘルパー関数
# ============================================
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# Dockerインストール関数（Debian/Ubuntu系）
# ============================================
install_docker_debian() {
    log_info "Installing Docker (Debian/Ubuntu)..."
    
    # 依存パッケージ
    sudo apt-get update
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        software-properties-common
    
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
    
    log_success "Docker installed successfully!"
    log_warning "Please log out and back in, then run this script again."
    exit 0
}

# ============================================
# Dockerインストール関数（RHEL/CentOS/Rocky系）
# ============================================
install_docker_rhel() {
    log_info "Installing Docker (RHEL/CentOS/Rocky)..."
    
    # リポジトリ追加
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    
    # Dockerインストール
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # ユーザーをdockerグループに追加
    sudo usermod -aG docker $USER
    
    log_success "Docker installed successfully!"
    log_warning "Please log out and back in, then run this script again."
    exit 0
}

# ============================================
# メイン処理
# ============================================
main() {
    echo -e "${CYAN}=== Local RAG MCP Server - Linux Installation ===${NC}"
    echo ""
    echo -e "${YELLOW}Note: Ollama is not included in this package.${NC}"
    echo -e "${YELLOW}      Ollama must be installed and running separately.${NC}"
    echo ""
    
    # パッケージディレクトリに移動
    if [ -d "$PACKAGE_DIR" ]; then
        cd "$PACKAGE_DIR"
        log_info "Working directory: $(pwd)"
    else
        log_error "Package directory not found: $PACKAGE_DIR"
        exit 1
    fi
    
    # Dockerインストール確認
    if ! command -v docker &> /dev/null; then
        # ディストリビューション検出
        if [ -f /etc/debian_version ]; then
            install_docker_debian
        elif [ -f /etc/redhat-release ]; then
            install_docker_rhel
        else
            log_error "Unsupported distribution. Please install Docker manually."
            exit 1
        fi
    fi
    
    # Dockerサービス開始
    log_info "Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # ============================================
    # イメージロード
    # ============================================
    log_info "Loading Docker images..."
    
    # MCPサーバーイメージ
    if [ -f "local-rag-mcp-server.tar" ]; then
        docker load -i local-rag-mcp-server.tar
        log_success "Loaded: local-rag-mcp-server.tar"
    else
        log_error "Image file not found: local-rag-mcp-server.tar"
        exit 1
    fi
    
    # ============================================
    # 設定ファイル
    # ============================================
    log_info "Setting up configuration files..."
    
    if [ ! -f config.json ]; then
        if [ -f config.json.example ]; then
            cp config.json.example config.json
            log_success "Created config.json from template."
        else
            log_warning "config.json.example not found. Please create config.json manually."
        fi
    else
        log_info "config.json already exists. Skipping."
    fi
    
    # ============================================
    # ディレクトリ作成
    # ============================================
    log_info "Creating directories..."
    
    mkdir -p documents
    mkdir -p data/converted_docs
    mkdir -p data/chroma_db
    mkdir -p data/models
    
    log_success "Directories created."
    
    # ============================================
    # docker-compose.yml確認
    # ============================================
    if [ ! -f docker-compose.yml ]; then
        log_error "docker-compose.yml not found!"
        exit 1
    fi
    
    # ============================================
    # 完了メッセージ
    # ============================================
    echo ""
    log_success "=== Installation Complete ==="
    echo ""
    echo -e "${YELLOW}IMPORTANT: Ollama must be installed and running separately.${NC}"
    echo ""
    echo "To install Ollama:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "To download models:"
    echo "  ollama pull nomic-embed-text-v2-moe"
    echo "  ollama pull glm-ocr"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Ensure Ollama is running:"
    echo "   ollama serve &"
    echo ""
    echo "2. Place your documents in the 'documents' directory:"
    echo "   cp -r /path/to/your/documents ./documents/"
    echo ""
    echo "3. Edit config.json:"
    echo "   nano config.json"
    echo "   - Set source_docs_dir to your documents path"
    echo "   - Set ollama_base_url (default: http://localhost:11434)"
    echo ""
    echo "4. Start services:"
    echo "   docker-compose up -d"
    echo ""
    echo "5. Check service status:"
    echo "   docker-compose ps"
    echo ""
    echo "6. Test the server:"
    echo "   curl http://localhost:8000/sse"
    echo ""
    echo "For more information, see README.md"
}

# ============================================
# 実行
# ============================================
main "$@"