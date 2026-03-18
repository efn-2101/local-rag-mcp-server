#!/bin/bash
# =============================================================================
# build-standalone-linux.sh
# =============================================================================
# Linux環境でNuitkaを使用してスタンドアロンバイナリをビルドするスクリプト
# 
# 使用方法:
#   ./scripts/build-standalone-linux.sh [--clean] [--onefile] [--output-dir DIR]
#
# 必要な依存関係:
#   - Python 3.10+
#   - gcc, g++ (Cコンパイラ)
#   - Nuitka
# =============================================================================

set -e

# =============================================================================
# 設定
# =============================================================================

PROJECT_NAME="local-rag-mcp-server"
PROJECT_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# デフォルト値
CLEAN_BUILD=false
ONEFILE=true
OUTPUT_DIR=""

# =============================================================================
# カラー出力
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# ヘルパー関数
# =============================================================================

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

log_section() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# =============================================================================
# 引数解析
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --no-onefile)
                ONEFILE=false
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --clean        Clean build directories before building"
                echo "  --no-onefile    Create directory distribution instead of single file"
                echo "  --output-dir    Specify output directory"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# 依存関係チェック
# =============================================================================

check_dependencies() {
    log_section "Checking Dependencies"
    
    # Python チェック
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
    
    # Cコンパイラ チェック
    if ! command -v gcc &> /dev/null; then
        log_error "GCC is not installed. Please install: sudo apt install build-essential"
        exit 1
    fi
    
    GCC_VERSION=$(gcc --version | head -n1)
    log_info "GCC version: $GCC_VERSION"
    
    # Nuitka チェック
    if ! python3 -c "import nuitka" 2>/dev/null; then
        log_warning "Nuitka is not installed. Installing..."
        pip3 install --user nuitka
    fi
    
    NUITKA_VERSION=$(python3 -m nuitka --version 2>&1 | head -n1)
    log_info "Nuitka version: $NUITKA_VERSION"
    
    log_success "All dependencies satisfied"
}

# =============================================================================
# ビルドディレクトリのクリーンアップ
# =============================================================================

clean_build_dirs() {
    if [ "$CLEAN_BUILD" = true ]; then
        log_section "Cleaning Build Directories"
        
        log_info "Removing build directory..."
        rm -rf "$BUILD_DIR"
        
        log_info "Removing dist directory..."
        rm -rf "$DIST_DIR"
        
        log_info "Removing Nuitka build directories..."
        rm -rf "$PROJECT_ROOT/$PROJECT_NAME.build"
        rm -rf "$PROJECT_ROOT/$PROJECT_NAME.dist"
        rm -rf "$PROJECT_ROOT/$PROJECT_NAME.onefile-build"
        
        log_info "Removing __pycache__ directories..."
        find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        
        log_success "Clean complete"
    fi
}

# =============================================================================
# Python依存関係のインストール
# =============================================================================

install_python_deps() {
    log_section "Installing Python Dependencies"
    
    cd "$PROJECT_ROOT"
    
    # requirements.txt から依存関係をインストール
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip3 install --user -r requirements.txt
    fi
    
    # ビルド用依存関係をインストール
    log_info "Installing build dependencies..."
    pip3 install --user nuitka ordered-set zstandard
    
    log_success "Python dependencies installed"
}

# =============================================================================
# Nuitkaビルド実行
# =============================================================================

run_nuitka_build() {
    log_section "Running Nuitka Build"
    
    cd "$PROJECT_ROOT"
    
    # 出力ディレクトリの設定
    if [ -n "$OUTPUT_DIR" ]; then
        OUTPUT_PATH="$OUTPUT_DIR"
        mkdir -p "$OUTPUT_PATH"
    else
        OUTPUT_PATH="$DIST_DIR"
        mkdir -p "$OUTPUT_PATH"
    fi
    
    # Nuitkaコマンドの構築
    NUITKA_CMD=(
        python3 -m nuitka
        --standalone
    )
    
    # onefileオプション
    if [ "$ONEFILE" = true ]; then
        NUITKA_CMD+=(--onefile)
    fi
    
    # 出力設定
    NUITKA_CMD+=(--output-dir="$OUTPUT_PATH")
    NUITKA_CMD+=(--output-filename="$PROJECT_NAME")
    
    # Pythonフラグ
    NUITKA_CMD+=(--python-flag=no_site)
    
    # 自動ダウンロード許可
    NUITKA_CMD+=(--assume-yes-for-downloads)
    
    # LTO有効化
    NUITKA_CMD+=(--lto=yes)
    
    # パッケージのインクルード
    # Core dependencies
    NUITKA_CMD+=(--include-package=mcp)
    NUITKA_CMD+=(--include-package=starlette)
    NUITKA_CMD+=(--include-package=uvicorn)
    NUITKA_CMD+=(--include-package=ollama)
    NUITKA_CMD+=(--include-package=chromadb)
    
    # Search engines
    NUITKA_CMD+=(--include-package=rank_bm25)
    NUITKA_CMD+=(--include-package=flashrank)
    
    # Document processing
    NUITKA_CMD+=(--include-package=fitz)
    NUITKA_CMD+=(--include-package=PIL)
    NUITKA_CMD+=(--include-package=openpyxl)
    NUITKA_CMD+=(--include-package=docx)
    NUITKA_CMD+=(--include-package=pptx)
    
    # Utilities
    NUITKA_CMD+=(--include-package=watchdog)
    NUITKA_CMD+=(--include-package=psutil)
    NUITKA_CMD+=(--include-package=urllib3)
    NUITKA_CMD+=(--include-package=charset_normalizer)
    
    # ONNX Runtime (for flashrank)
    NUITKA_CMD+=(--include-package=onnxruntime)
    
    # Additional dependencies
    NUITKA_CMD+=(--include-package=httpx)
    NUITKA_CMD+=(--include-package=httpcore)
    NUITKA_CMD+=(--include-package=pydantic)
    NUITKA_CMD+=(--include-package=pydantic_core)
    
    # データファイルのインクルード
    if [ -f "config.json.example" ]; then
        NUITKA_CMD+=(--include-data-file="config.json.example=config.json.example")
    fi
    
    if [ -f "acl.json.example" ]; then
        NUITKA_CMD+=(--include-data-file="acl.json.example=acl.json.example")
    fi
    
    # メインスクリプト
    NUITKA_CMD+=(server.py)
    
    log_info "Running: ${NUITKA_CMD[*]}"
    
    # 環境変数の設定
    export PYTHONPATH="$PROJECT_ROOT"
    
    # ビルド実行
    "${NUITKA_CMD[@]}"
    
    BUILD_RESULT=$?
    
    if [ $BUILD_RESULT -ne 0 ]; then
        log_error "Nuitka build failed with exit code: $BUILD_RESULT"
        exit $BUILD_RESULT
    fi
    
    log_success "Nuitka build completed"
}

# =============================================================================
# 配布パッケージの作成
# =============================================================================

create_distribution_package() {
    log_section "Creating Distribution Package"
    
    if [ -n "$OUTPUT_DIR" ]; then
        PACKAGE_DIR="$OUTPUT_DIR"
    else
        PACKAGE_DIR="$DIST_DIR"
    fi
    
    cd "$PROJECT_ROOT"
    
    # 追加ファイルをコピー
    log_info "Copying additional files..."
    
    # 設定ファイルテンプレート
    if [ -f "config.json.example" ]; then
        cp "config.json.example" "$PACKAGE_DIR/"
        log_info "  - config.json.example"
    fi
    
    if [ -f "acl.json.example" ]; then
        cp "acl.json.example" "$PACKAGE_DIR/"
        log_info "  - acl.json.example"
    fi
    
    # README
    if [ -f "README.md" ]; then
        cp "README.md" "$PACKAGE_DIR/"
        log_info "  - README.md"
    fi
    
    # LICENSE
    if [ -f "LICENSE" ]; then
        cp "LICENSE" "$PACKAGE_DIR/"
        log_info "  - LICENSE"
    fi
    
    # 実行権限の付与
    if [ -f "$PACKAGE_DIR/$PROJECT_NAME" ]; then
        chmod +x "$PACKAGE_DIR/$PROJECT_NAME"
        log_info "Made $PROJECT_NAME executable"
    fi
    
    # インストールスクリプトの作成
    create_install_script "$PACKAGE_DIR"
    
    # README for standalone package
    create_standalone_readme "$PACKAGE_DIR"
    
    log_success "Distribution package created in: $PACKAGE_DIR"
}

# =============================================================================
# インストールスクリプトの作成
# =============================================================================

create_install_script() {
    local PACKAGE_DIR="$1"
    local INSTALL_SCRIPT="$PACKAGE_DIR/install.sh"
    
    log_info "Creating install script..."
    
    cat > "$INSTALL_SCRIPT" << 'INSTALL_EOF'
#!/bin/bash
# =============================================================================
# Local RAG MCP Server - Installation Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Local RAG MCP Server Installation"
echo "=========================================="
echo ""

# 設定ファイルの作成
if [ ! -f config.json ]; then
    echo "[INFO] Creating config.json from template..."
    cp config.json.example config.json
    echo "[INFO] Please edit config.json with your settings."
fi

# 実行権限の付与
chmod +x local-rag-mcp-server 2>/dev/null || true

# Ollamaの確認
echo ""
echo "[INFO] Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "[INFO] Ollama found: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    echo "[WARNING] Ollama is not installed."
    echo "[INFO] Please install Ollama from: https://ollama.com/download"
fi

echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit config.json with your document paths"
echo "2. Ensure Ollama is running:"
echo "   ollama serve &"
echo "3. Pull required models:"
echo "   ollama pull nomic-embed-text-v2-moe"
echo "   ollama pull glm-ocr"
echo "4. Start the server:"
echo "   ./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000"
echo ""
INSTALL_EOF

    chmod +x "$INSTALL_SCRIPT"
    log_info "Created install.sh"
}

# =============================================================================
# スタンドアロンREADMEの作成
# =============================================================================

create_standalone_readme() {
    local PACKAGE_DIR="$1"
    local README_FILE="$PACKAGE_DIR/README-standalone.txt"
    
    log_info "Creating standalone README..."
    
    cat > "$README_FILE" << 'README_EOF'
================================================================================
  Local RAG MCP Server - Standalone Package
================================================================================

Version: 1.0.0

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

- Operating System: Linux (x86_64/amd64)
- Ollama server running on localhost:11434 (or configured URL)
- Required Ollama models:
  - nomic-embed-text-v2-moe (for embeddings)
  - glm-ocr (for OCR, optional)

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------

1. Copy config.json.example to config.json:
   cp config.json.example config.json

2. Edit config.json with your settings:
   - Set "source_docs_dir" to your documents directory
   - Set "ollama_base_url" to your Ollama server URL
     (default: http://localhost:11434)

3. Ensure Ollama is running with required models:
   ollama serve &
   ollama pull nomic-embed-text-v2-moe
   ollama pull glm-ocr

--------------------------------------------------------------------------------
RUNNING
--------------------------------------------------------------------------------

Make the file executable (if not already):
   chmod +x local-rag-mcp-server

Start the server:
   ./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000

For help:
   ./local-rag-mcp-server --help

--------------------------------------------------------------------------------
CONFIGURATION
--------------------------------------------------------------------------------

See config.json.example for all available configuration options:

- source_docs_dir: Directory containing source documents
- docs_dir: Directory for converted markdown files
- ollama_base_url: URL of Ollama server
- embedding_model: Model for embeddings (default: nomic-embed-text-v2-moe)
- ocr_model: Model for OCR (default: glm-ocr)
- chunk_size: Text chunk size for indexing
- chunk_overlap: Overlap between chunks

--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

"Ollama connection failed"
- Ensure Ollama is running: ollama serve
- Check the URL in config.json
- Verify network connectivity

"Model not found"
- Pull the required model: ollama pull <model-name>

"Permission denied"
- Make the file executable: chmod +x local-rag-mcp-server

"Port already in use"
- Use a different port: --port 8001
- Or stop the conflicting service

--------------------------------------------------------------------------------
SUPPORT
--------------------------------------------------------------------------------

For issues and updates, visit the project repository.

================================================================================
README_EOF

    log_info "Created README-standalone.txt"
}

# =============================================================================
# パッケージ情報の表示
# =============================================================================

show_package_info() {
    log_section "Build Complete"
    
    if [ -n "$OUTPUT_DIR" ]; then
        PACKAGE_DIR="$OUTPUT_DIR"
    else
        PACKAGE_DIR="$DIST_DIR"
    fi
    
    log_info "Package location: $PACKAGE_DIR"
    log_info "Executable: $PACKAGE_DIR/$PROJECT_NAME"
    
    # ファイルサイズの表示
    if [ -f "$PACKAGE_DIR/$PROJECT_NAME" ]; then
        SIZE=$(du -h "$PACKAGE_DIR/$PROJECT_NAME" | cut -f1)
        log_info "Executable size: $SIZE"
    fi
    
    # ディレクトリの内容を表示
    log_info "Package contents:"
    ls -la "$PACKAGE_DIR"
    
    echo ""
    log_success "Build completed successfully!"
    echo ""
    echo "To run the server:"
    echo "  cd $PACKAGE_DIR"
    echo "  ./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000"
    echo ""
}

# =============================================================================
# メイン処理
# =============================================================================

main() {
    log_section "Local RAG MCP Server - Standalone Build"
    
    echo "Project: $PROJECT_NAME v$PROJECT_VERSION"
    echo "Platform: Linux"
    echo "Build type: $( [ "$ONEFILE" = true ] && echo "onefile" || echo "directory" )"
    echo ""
    
    # 引数解析
    parse_args "$@"
    
    # 依存関係チェック
    check_dependencies
    
    # クリーンビルド
    clean_build_dirs
    
    # Python依存関係のインストール
    install_python_deps
    
    # Nuitkaビルド
    run_nuitka_build
    
    # 配布パッケージの作成
    create_distribution_package
    
    # 情報表示
    show_package_info
}

# スクリプト実行
main "$@"