# =============================================================================
# build-standalone-windows.ps1
# =============================================================================
# Windows環境でNuitkaを使用してスタンドアロンバイナリをビルドするスクリプト
# 
# 使用方法:
#   .\scripts\build-standalone-windows.ps1 [-Clean] [-Onefile] [-OutputDir DIR] [-Platform linux|windows]
#
# 必要な依存関係:
#   - Python 3.10+
#   - Visual Studio Build Tools (C++ コンパイラ)
#   - Nuitka
# =============================================================================

param(
    [switch]$Clean,
    [switch]$Onefile = $true,
    [string]$OutputDir = "",
    [ValidateSet("linux", "windows")]
    [string]$Platform = "windows",
    [switch]$Help
)

# =============================================================================
# 設定
# =============================================================================

$PROJECT_NAME = "local-rag-mcp-server"
$PROJECT_VERSION = "1.0.0"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"
$DIST_DIR = Join-Path $PROJECT_ROOT "dist"

# =============================================================================
# カラー出力関数
# =============================================================================

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Blue
    Write-Host "  $Title" -ForegroundColor Blue
    Write-Host "============================================" -ForegroundColor Blue
    Write-Host ""
}

# =============================================================================
# ヘルプ表示
# =============================================================================

if ($Help) {
    Write-Host "Usage: .\scripts\build-standalone-windows.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Clean        Clean build directories before building"
    Write-Host "  -Onefile      Create single executable file (default: true)"
    Write-Host "  -OutputDir    Specify output directory"
    Write-Host "  -Platform     Target platform: 'linux' or 'windows' (default: windows)"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "Note: Cross-compilation to Linux from Windows requires WSL2."
    Write-Host "      For Linux builds, run the script in WSL2 environment."
    exit 0
}

# =============================================================================
# 依存関係チェック
# =============================================================================

function Test-Dependencies {
    Write-Section "Checking Dependencies"
    
    # Python チェック
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    
    if (-not $pythonCmd) {
        Write-Error "Python is not installed or not in PATH"
        exit 1
    }
    
    $pythonVersion = & python --version 2>&1
    Write-Info "Python version: $pythonVersion"
    
    # Cコンパイラ チェック (Windows)
    if ($Platform -eq "windows") {
        $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vsWhere) {
            $vsPath = & $vsWhere -latest -property installationPath 2>&1
            Write-Info "Visual Studio: $vsPath"
        } else {
            Write-Warning "Visual Studio Build Tools not found via vswhere"
            Write-Warning "Nuitka will attempt to use MinGW if available"
        }
    }
    
    # Nuitka チェック
    $nuitkaInstalled = & python -c "import nuitka" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Nuitka is not installed. Installing..."
        & pip install nuitka ordered-set zstandard
    }
    
    $nuitkaVersion = & python -m nuitka --version 2>&1 | Select-Object -First 1
    Write-Info "Nuitka version: $nuitkaVersion"
    
    Write-Success "All dependencies satisfied"
}

# =============================================================================
# ビルドディレクトリのクリーンアップ
# =============================================================================

function Clear-BuildDirs {
    if ($Clean) {
        Write-Section "Cleaning Build Directories"
        
        if (Test-Path $BUILD_DIR) {
            Write-Info "Removing build directory..."
            Remove-Item -Recurse -Force $BUILD_DIR
        }
        
        if (Test-Path $DIST_DIR) {
            Write-Info "Removing dist directory..."
            Remove-Item -Recurse -Force $DIST_DIR
        }
        
        # Nuitka build directories
        $nuitkaBuildDir = Join-Path $PROJECT_ROOT "$PROJECT_NAME.build"
        $nuitkaDistDir = Join-Path $PROJECT_ROOT "$PROJECT_NAME.dist"
        $nuitkaOnefileDir = Join-Path $PROJECT_ROOT "$PROJECT_NAME.onefile-build"
        
        if (Test-Path $nuitkaBuildDir) {
            Remove-Item -Recurse -Force $nuitkaBuildDir
        }
        if (Test-Path $nuitkaDistDir) {
            Remove-Item -Recurse -Force $nuitkaDistDir
        }
        if (Test-Path $nuitkaOnefileDir) {
            Remove-Item -Recurse -Force $nuitkaOnefileDir
        }
        
        # __pycache__ directories
        Get-ChildItem -Path $PROJECT_ROOT -Directory -Recurse -Filter "__pycache__" | 
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        
        Write-Success "Clean complete"
    }
}

# =============================================================================
# Python依存関係のインストール
# =============================================================================

function Install-PythonDeps {
    Write-Section "Installing Python Dependencies"
    
    Push-Location $PROJECT_ROOT
    
    # requirements.txt から依存関係をインストール
    $requirementsPath = Join-Path $PROJECT_ROOT "requirements.txt"
    if (Test-Path $requirementsPath) {
        Write-Info "Installing from requirements.txt..."
        & pip install -r $requirementsPath
    }
    
    # ビルド用依存関係をインストール
    Write-Info "Installing build dependencies..."
    & pip install nuitka ordered-set zstandard
    
    Pop-Location
    
    Write-Success "Python dependencies installed"
}

# =============================================================================
# Nuitkaビルド実行
# =============================================================================

function Invoke-NuitkaBuild {
    Write-Section "Running Nuitka Build"
    
    Push-Location $PROJECT_ROOT
    
    # 出力ディレクトリの設定
    if ($OutputDir) {
        $outputPath = $OutputDir
        if (-not (Test-Path $outputPath)) {
            New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
        }
    } else {
        $outputPath = $DIST_DIR
        if (-not (Test-Path $outputPath)) {
            New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
        }
    }
    
    # Nuitka引数の構築
    $nuitkaArgs = @(
        "-m", "nuitka",
        "--standalone"
    )
    
    # onefileオプション
    if ($Onefile) {
        $nuitkaArgs += "--onefile"
    }
    
    # 出力設定
    $nuitkaArgs += "--output-dir=$outputPath"
    $nuitkaArgs += "--output-filename=$PROJECT_NAME"
    
    # Pythonフラグ
    $nuitkaArgs += "--python-flag=no_site"
    
    # 自動ダウンロード許可
    $nuitkaArgs += "--assume-yes-for-downloads"
    
    # LTO有効化
    $nuitkaArgs += "--lto=yes"
    
    # Windows固有の設定
    if ($Platform -eq "windows") {
        $nuitkaArgs += "--windows-console-mode=force"
    }
    
    # パッケージのインクルード
    $packages = @(
        # Core dependencies
        "mcp",
        "starlette",
        "uvicorn",
        "ollama",
        "chromadb",
        # Search engines
        "rank_bm25",
        "flashrank",
        # Document processing
        "fitz",
        "PIL",
        "openpyxl",
        "docx",
        "pptx",
        # Utilities
        "watchdog",
        "psutil",
        "urllib3",
        "charset_normalizer",
        # ONNX Runtime
        "onnxruntime",
        # Additional dependencies
        "httpx",
        "httpcore",
        "pydantic",
        "pydantic_core"
    )
    
    foreach ($package in $packages) {
        $nuitkaArgs += "--include-package=$package"
    }
    
    # データファイルのインクルード
    $configExample = Join-Path $PROJECT_ROOT "config.json.example"
    if (Test-Path $configExample) {
        $nuitkaArgs += "--include-data-file=config.json.example=config.json.example"
    }
    
    $aclExample = Join-Path $PROJECT_ROOT "acl.json.example"
    if (Test-Path $aclExample) {
        $nuitkaArgs += "--include-data-file=acl.json.example=acl.json.example"
    }
    
    # メインスクリプト
    $nuitkaArgs += "server.py"
    
    Write-Info "Running: python $($nuitkaArgs -join ' ')"
    
    # 環境変数の設定
    $env:PYTHONPATH = $PROJECT_ROOT
    
    # ビルド実行
    $process = Start-Process -FilePath "python" -ArgumentList $nuitkaArgs -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-Error "Nuitka build failed with exit code: $($process.ExitCode)"
        Pop-Location
        exit $process.ExitCode
    }
    
    Pop-Location
    
    Write-Success "Nuitka build completed"
}

# =============================================================================
# 配布パッケージの作成
# =============================================================================

function New-DistributionPackage {
    Write-Section "Creating Distribution Package"
    
    if ($OutputDir) {
        $packageDir = $OutputDir
    } else {
        $packageDir = $DIST_DIR
    }
    
    Push-Location $PROJECT_ROOT
    
    # 追加ファイルをコピー
    Write-Info "Copying additional files..."
    
    $filesToCopy = @(
        "config.json.example",
        "acl.json.example",
        "README.md",
        "LICENSE"
    )
    
    foreach ($filename in $filesToCopy) {
        $source = Join-Path $PROJECT_ROOT $filename
        if (Test-Path $source) {
            $dest = Join-Path $packageDir $filename
            Copy-Item $source $dest -Force
            Write-Info "  - $filename"
        }
    }
    
    # Windows用実行ファイル名の調整
    if ($Platform -eq "windows") {
        $exePath = Join-Path $packageDir "$PROJECT_NAME.exe"
        $binPath = Join-Path $packageDir $PROJECT_NAME
        
        if ((Test-Path $binPath) -and -not (Test-Path $exePath)) {
            Rename-Item $binPath "$PROJECT_NAME.exe"
            Write-Info "Renamed executable to $PROJECT_NAME.exe"
        }
    }
    
    # インストールスクリプトの作成
    New-InstallScript $packageDir
    
    # README for standalone package
    New-StandaloneReadme $packageDir
    
    Pop-Location
    
    Write-Success "Distribution package created in: $packageDir"
}

# =============================================================================
# インストールスクリプトの作成
# =============================================================================

function New-InstallScript {
    param([string]$PackageDir)
    
    if ($Platform -eq "windows") {
        $installScript = Join-Path $PackageDir "install.ps1"
        
        Write-Info "Creating Windows install script..."
        
        $scriptContent = @'
# =============================================================================
# Local RAG MCP Server - Windows Installation Script
# =============================================================================

param(
    [string]$InstallDir = "$env:LOCALAPPDATA\local-rag-mcp-server"
)

Write-Host "=========================================="
Write-Host "  Local RAG MCP Server Installation"
Write-Host "=========================================="
Write-Host ""

# Create installation directory
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Write-Host "[INFO] Created directory: $InstallDir"
}

# Copy files
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Copy-Item "$scriptDir\local-rag-mcp-server.exe" $InstallDir -Force -ErrorAction SilentlyContinue
Copy-Item "$scriptDir\config.json.example" $InstallDir -Force -ErrorAction SilentlyContinue
Copy-Item "$scriptDir\acl.json.example" $InstallDir -Force -ErrorAction SilentlyContinue
Copy-Item "$scriptDir\README-standalone.txt" $InstallDir -Force -ErrorAction SilentlyContinue

# Create config.json if not exists
$configPath = Join-Path $InstallDir "config.json"
if (-not (Test-Path $configPath)) {
    $configExample = Join-Path $InstallDir "config.json.example"
    if (Test-Path $configExample) {
        Copy-Item $configExample $configPath
        Write-Host "[INFO] Created config.json from template"
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Installation Complete"
Write-Host "=========================================="
Write-Host ""
Write-Host "Installed to: $InstallDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Edit config.json with your document paths"
Write-Host "2. Ensure Ollama is running:"
Write-Host "   ollama serve"
Write-Host "3. Pull required models:"
Write-Host "   ollama pull nomic-embed-text-v2-moe"
Write-Host "   ollama pull glm-ocr"
Write-Host "4. Start the server:"
Write-Host "   $InstallDir\local-rag-mcp-server.exe --transport sse --host 0.0.0.0 --port 8000"
Write-Host ""
'@
        
        Set-Content -Path $installScript -Value $scriptContent -Encoding UTF8
        Write-Info "Created install.ps1"
    } else {
        $installScript = Join-Path $PackageDir "install.sh"
        
        Write-Info "Creating Linux install script..."
        
        $scriptContent = @'
#!/bin/bash
# =============================================================================
# Local RAG MCP Server - Linux Installation Script
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
'@
        
        Set-Content -Path $installScript -Value $scriptContent -Encoding UTF8
        # Unix改行コードに変換
        $content = [System.IO.File]::ReadAllText($installScript)
        $content = $content -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($installScript, $content)
        Write-Info "Created install.sh"
    }
}

# =============================================================================
# スタンドアロンREADMEの作成
# =============================================================================

function New-StandaloneReadme {
    param([string]$PackageDir)
    
    $readmeFile = Join-Path $PackageDir "README-standalone.txt"
    
    Write-Info "Creating standalone README..."
    
    $platformName = if ($Platform -eq "windows") { "Windows" } else { "Linux" }
    $executableName = if ($Platform -eq "windows") { "local-rag-mcp-server.exe" } else { "local-rag-mcp-server" }
    
    $content = @"
================================================================================
  Local RAG MCP Server - Standalone Package
================================================================================

Version: $PROJECT_VERSION
Platform: $platformName

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

- Operating System: $platformName (x86_64/amd64)
- Ollama server running on localhost:11434 (or configured URL)
- Required Ollama models:
  - nomic-embed-text-v2-moe (for embeddings)
  - glm-ocr (for OCR, optional)

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------

1. Copy config.json.example to config.json:
   copy config.json.example config.json  (Windows)
   cp config.json.example config.json    (Linux)

2. Edit config.json with your settings:
   - Set "source_docs_dir" to your documents directory
   - Set "ollama_base_url" to your Ollama server URL
     (default: http://localhost:11434)

3. Ensure Ollama is running with required models:
   ollama serve
   ollama pull nomic-embed-text-v2-moe
   ollama pull glm-ocr

--------------------------------------------------------------------------------
RUNNING
--------------------------------------------------------------------------------

Make the file executable (Linux only):
   chmod +x $executableName

Start the server:
   ./$executableName --transport sse --host 0.0.0.0 --port 8000

For help:
   ./$executableName --help

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

"Permission denied" (Linux)
- Make the file executable: chmod +x $executableName

"Port already in use"
- Use a different port: --port 8001
- Or stop the conflicting service

--------------------------------------------------------------------------------
SUPPORT
--------------------------------------------------------------------------------

For issues and updates, visit the project repository.

================================================================================
"@
    
    Set-Content -Path $readmeFile -Value $content -Encoding UTF8
    Write-Info "Created README-standalone.txt"
}

# =============================================================================
# パッケージ情報の表示
# =============================================================================

function Show-PackageInfo {
    Write-Section "Build Complete"
    
    if ($OutputDir) {
        $packageDir = $OutputDir
    } else {
        $packageDir = $DIST_DIR
    }
    
    Write-Info "Package location: $packageDir"
    
    $executableName = if ($Platform -eq "windows") { "$PROJECT_NAME.exe" } else { $PROJECT_NAME }
    $executablePath = Join-Path $packageDir $executableName
    
    if (Test-Path $executablePath) {
        $size = (Get-Item $executablePath).Length / 1MB
        Write-Info "Executable size: $([math]::Round($size, 2)) MB"
    }
    
    Write-Info "Package contents:"
    Get-ChildItem $packageDir | ForEach-Object {
        Write-Host "  $($_.Name)"
    }
    
    Write-Host ""
    Write-Success "Build completed successfully!"
    Write-Host ""
    Write-Host "To run the server:"
    Write-Host "  cd $packageDir"
    Write-Host "  .\$executableName --transport sse --host 0.0.0.0 --port 8000"
    Write-Host ""
}

# =============================================================================
# WSL2ビルド関数
# =============================================================================

function Invoke-WSL2Build {
    Write-Section "Building for Linux via WSL2"
    
    Write-Info "Checking WSL2 availability..."
    
    $wslCheck = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wslCheck) {
        Write-Error "WSL2 is not installed. Please install WSL2 first."
        Write-Info "Run: wsl --install"
        exit 1
    }
    
    # WSL2でLinuxビルドスクリプトを実行
    $linuxScript = Join-Path $PROJECT_ROOT "scripts/build-standalone-linux.sh"
    
    if (-not (Test-Path $linuxScript)) {
        Write-Error "Linux build script not found: $linuxScript"
        exit 1
    }
    
    Write-Info "Running Linux build script in WSL2..."
    
    # WSL2内のパスに変換
    $wslProjectRoot = $PROJECT_ROOT -replace "\\", "/" -replace ":", ""
    $wslProjectRoot = "/mnt/$($wslProjectRoot.Substring(0,1).ToLower())$($wslProjectRoot.Substring(1))"
    
    $wslCommand = "cd '$wslProjectRoot' && chmod +x scripts/build-standalone-linux.sh && ./scripts/build-standalone-linux.sh"
    
    if ($Clean) {
        $wslCommand += " --clean"
    }
    
    & wsl bash -c $wslCommand
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "WSL2 build failed with exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    
    Write-Success "WSL2 build completed"
}

# =============================================================================
# メイン処理
# =============================================================================

Write-Section "Local RAG MCP Server - Standalone Build"

Write-Host "Project: $PROJECT_NAME v$PROJECT_VERSION"
Write-Host "Platform: $Platform"
Write-Host "Build type: $(if ($Onefile) { 'onefile' } else { 'directory' })"
Write-Host ""

# Linuxビルドの場合はWSL2を使用
if ($Platform -eq "linux") {
    Invoke-WSL2Build
} else {
    # Windowsビルド
    Test-Dependencies
    Clear-BuildDirs
    Install-PythonDeps
    Invoke-NuitkaBuild
    New-DistributionPackage
}

Show-PackageInfo