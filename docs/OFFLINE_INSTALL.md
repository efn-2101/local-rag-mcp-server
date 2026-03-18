# Local RAG MCP Server - オフライン環境インストール手順書

## 目次

1. [概要](#1-概要)
2. [Docker方式の手順](#2-docker方式の手順)
3. [自己完結型パッケージ方式の手順](#3-自己完結型パッケージ方式の手順)
4. [トラブルシューティング](#4-トラブルシューティング)
5. [付録](#5-付録)

---

## 1. 概要

### 1.1 目的

本文書は、ネットワーク接続がない（オフライン）Linux環境への `Local RAG MCP Server` インストール手順を説明します。オンライン環境（Windows）で必要なファイルを準備し、USBメモリ等を介してオフライン環境へ転送・インストールする方法を提供します。

### 1.2 対象環境

| 項目 | オンライン環境 | オフライン環境 |
|------|---------------|---------------|
| OS | Windows 10/11 | Linux（Ubuntu/Debian系、RHEL/CentOS系等） |
| ネットワーク | インターネット接続あり | インターネット接続なし |
| 用途 | ファイル準備・ビルド | 本番実行環境 |

### 1.3 インストール方式の比較

| 項目 | Docker方式 | 自己完結型パッケージ方式 |
|------|-----------|------------------------|
| **セットアップ容易性** | 中程度（Docker知識必要） | 高い（実行ファイルのみ） |
| **環境隔離性** | 完全隔離 | システム依存 |
| **パッケージサイズ** | ~500MB | 100-250MB |
| **Ollama統合** | 外部サービスとして接続 | 外部サービスとして接続 |
| **更新容易性** | イメージ差し替え | 再ビルド必要 |
| **推奨ユースケース** | 本番運用・検証 | リソース制約環境 |

**注意:** Docker方式ではOllamaをDocker Composeに含まず、外部サービスとして独立して稼働させます。これにより、Ollamaのリソース管理や更新を柔軟に行えます。

### 1.4 前提条件

#### Windows側（オンライン環境）

- Docker Desktop がインストールされていること
- PowerShell 5.1以上 または PowerShell Core 7.x
- 十分なディスク容量（Docker方式: 約1GB、自己完結型: 約1GB）
- USBメモリ（16GB以上推奨、exFAT形式推奨）

#### Linux側（オフライン環境）

- Linux（Ubuntu 20.04+/Debian系、RHEL/CentOS/Rocky系等）
- 4GB以上のメモリ（8GB推奨）
- 10GB以上のディスク容量
- Docker方式の場合: Docker CE がインストール可能であること
- **Ollamaがインストール済みであること**（埋め込みモデル・OCRモデル用）
  - インストール方法: `curl -fsSL https://ollama.com/install.sh | sh`
  - 必要なモデル: `nomic-embed-text-v2-moe`（埋め込み用）、`glm-ocr`（OCR用）

**注意:** コマンド例は主にUbuntu/Debian系をベースにしています。RHEL/CentOS系の場合は、パッケージマネージャを `apt` から `yum` または `dnf` に読み替えてください。

---

## 2. Docker方式の手順

### 2.1 アーキテクチャ概要

**重要:** Docker方式ではOllamaをDocker Composeに含めず、外部サービスとして独立して稼働させます。これにより、Ollamaのリソース管理や更新を柔軟に行えます。

```
┌─────────────────────────────────────────────────────────────┐
│                    オフライン環境                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────┐    ┌─────────────────────────┐  │
│  │  Docker Compose        │    │  Ollama (外部サービス)   │  │
│  │  ┌─────────────────┐   │    │                         │  │
│  │  │  MCP Server     │   │    │  - nomic-embed-text-v2  │  │
│  │  │  Container      │───┼────│  - glm-ocr              │  │
│  │  │                 │   │    │                         │  │
│  │  │  - server.py    │   │    │  Port: 11434            │  │
│  │  │  - rag_engine.py│   │    │  (独立プロセス)          │  │
│  │  │  - file_converter│  │    └─────────────────────────┘  │
│  │  │  - ocr_engine.py│   │                                  │
│  │  └─────────────────┘   │                                  │
│  │                        │                                  │
│  │  Volume Mounts:        │                                  │
│  │  - documents/          │                                  │
│  │  - chroma_db/          │                                  │
│  │  - models/             │                                  │
│  └────────────────────────┘                                  │
│                                                             │
│  接続: MCP Server → Ollama (http://host.docker.internal:11434)  │
│         または OllamaサーバーのIPアドレス                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Windows側（オンライン環境）の手順

#### 2.2.1 Docker Desktopのインストール確認

```powershell
# Dockerのバージョン確認
docker --version
docker-compose --version

# 期待される出力例:
# Docker version 24.0.x, build xxxxx
# Docker Compose version v2.x.x
```

Dockerがインストールされていない場合:
1. https://www.docker.com/products/docker-desktop からダウンロード
2. インストール後、Docker Desktopを起動
3. WSL2バックエンドを有効にする（Windowsの場合）

#### 2.2.2 プロジェクトの取得

```powershell
# プロジェクトをクローン
git clone <repository-url>
cd local-rag-mcp-server

# または、ZIPファイルをダウンロードして展開
```

#### 2.2.3 Dockerイメージのビルドとエクスポート

**方法A: PowerShellスクリプトを使用（推奨）**

```powershell
# 基本的なエクスポート
.\scripts\export-docker-image.ps1 -OutputDir .\offline-package
```

**方法B: 手動実行**

```powershell
# 1. 出力ディレクトリ作成
New-Item -ItemType Directory -Force -Path .\offline-package

# 2. Dockerイメージビルド
docker-compose build

# 3. イメージ保存
docker save -o .\offline-package\local-rag-mcp-server.tar local-rag-mcp-server:latest

# 4. 設定ファイルコピー
Copy-Item docker-compose.yml .\offline-package\
Copy-Item config.json.example .\offline-package\
Copy-Item Dockerfile .\offline-package\
Copy-Item docker-entrypoint.sh .\offline-package\
```

#### 2.2.4 USBメモリへの転送

```powershell
# USBメモリのパスを確認
Get-Volume | Where-Object { $_.DriveType -eq 'Removable' }

# USBメモリにコピー（例: Eドライブ）
Copy-Item -Recurse -Force .\offline-package E:\

# または、エクスプローラーでドラッグ＆ドロップ
explorer .\offline-package
```

### 2.3 Linux側（オフライン環境）の手順

**注意:** 以下のコマンド例はUbuntu/Debian系をベースにしています。RHEL/CentOS/Rocky系の場合は、パッケージマネージャを適切に読み替えてください（例: `apt` → `yum`/`dnf`）。

#### 2.3.1 USBメモリのマウント

```bash
# USBメモリのデバイスを確認
lsblk
# または
sudo fdisk -l

# マウントポイント作成
sudo mkdir -p /mnt/usb

# マウント（デバイス名は環境に合わせて変更）
sudo mount /dev/sdX1 /mnt/usb  # sdX1 を適切なデバイスに変更

# 確認
ls /mnt/usb
```

#### 2.3.2 パッケージのコピー

```bash
# ホームディレクトリにコピー
cp -r /mnt/usb/offline-package ~/
cd ~/offline-package

# ファイル確認
ls -la
# 期待されるファイル:
# - local-rag-mcp-server.tar
# - docker-compose.yml
# - config.json.example
# - install-linux.sh
```

#### 2.3.3 Dockerのインストール（オフライン）

**方法A: インストールスクリプトを使用**

```bash
# インストールスクリプトに実行権限を付与
chmod +x install-linux.sh

# 実行
./install-linux.sh

# Dockerグループに追加された場合、ログアウト/ログインが必要
```

**方法B: 手動インストール（オフライン用パッケージが必要）**

オフライン環境の場合、事前にDockerのパッケージをダウンロードして転送する必要があります:

```bash
# Ubuntu/Debian系の場合
# オンライン環境でパッケージをダウンロード
apt-get download docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 転送後、オフライン環境でインストール
sudo dpkg -i containerd.io_*.deb
sudo dpkg -i docker-ce-cli_*.deb
sudo dpkg -i docker-ce_*.deb
sudo dpkg -i docker-compose-plugin_*.deb

# 依存関係の問題がある場合
sudo apt-get install -f

# RHEL/CentOS/Rocky系の場合
# オンライン環境でパッケージをダウンロード
yum download docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 転送後、オフライン環境でインストール
sudo yum localinstall containerd.io-*.rpm
sudo yum localinstall docker-ce-cli-*.rpm
sudo yum localinstall docker-ce-*.rpm
sudo yum localinstall docker-compose-plugin-*.rpm
```

#### 2.3.4 Ollamaのインストールと設定

**重要:** OllamaはDocker Composeに含まれず、独立したサービスとして稼働します。

```bash
# Ollamaのインストール（オンライン環境で事前にインストール推奨）
curl -fsSL https://ollama.com/install.sh | sh

# または、オフライン環境用にOllamaバイナリを転送
# オンライン環境でダウンロード:
# curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
# 転送後、オフライン環境でインストール:
sudo cp ollama /usr/local/bin/
sudo chmod +x /usr/local/bin/ollama

# Ollamaサービスの起動
ollama serve &

# 必要なモデルのダウンロード（オンライン環境で事前ダウンロード推奨）
ollama pull nomic-embed-text-v2-moe
ollama pull glm-ocr

# モデル一覧確認
ollama list
```

**Ollamaをsystemdサービスとして登録（推奨）:**

```bash
# サービスファイル作成
sudo tee /etc/systemd/system/ollama.service << EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# サービス有効化
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# ステータス確認
sudo systemctl status ollama
```

#### 2.3.5 Dockerイメージのインポート

```bash
# インポートスクリプトを使用
chmod +x scripts/import-docker-image.sh
./scripts/import-docker-image.sh

# または手動でインポート
docker load -i local-rag-mcp-server.tar
```

#### 2.3.6 設定ファイルの編集

```bash
# 設定ファイルを作成
cp config.json.example config.json

# エディタで編集
nano config.json
```

**config.json の設定例:**

```json
{
  "source_docs_dir": "/home/user/documents",
  "docs_dir": "/home/user/converted_docs",
  "db_dir": "/home/user/chroma_db",
  "ollama_base_url": "http://localhost:11434",
  "embedding_model": "nomic-embed-text-v2-moe",
  "ocr_model": "glm-ocr",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**docker-compose.yml の環境変数設定:**

MCPサーバーからOllamaへの接続URLを環境変数で設定できます:

```bash
# 環境変数でOllamaのURLを指定
export OLLAMA_BASE_URL=http://localhost:11434

# または、docker-compose.yml で直接指定
# Docker Desktop (Windows/Mac) の場合:
# OLLAMA_BASE_URL=http://host.docker.internal:11434
# Linux の場合:
# OLLAMA_BASE_URL=http://172.17.0.1:11434
```

**acl.json の設定（グループアクセス制御）:**

複数のユーザーグループに対して異なるドキュメントルートへのアクセス権限を設定する場合、`acl.json` を作成します。

```bash
# 設定ファイルを作成
cp acl.json.example acl.json

# エディタで編集
nano acl.json
```

**acl.json の設定例:**

```json
{
  "_default": {
    "name": "Public",
    "allowed_roots": ["Public"]
  },
  "your-api-key-here": {
    "name": "Group A",
    "allowed_roots": ["Group A", "Public"]
  }
}
```

**設定項目の説明:**

| 項目 | 説明 |
|------|------|
| `_default` | デフォルト設定（認証なしのアクセス用） |
| APIキー文字列 | グループを識別するためのAPIキーをキーとして指定 |
| `name` | グループの表示名 |
| `allowed_roots` | アクセス可能なルート（ドキュメントルート）のリスト |

**注意:** `acl.json` はオプションです。グループアクセス制御が必要ない場合は作成する必要はありません。

#### 2.3.7 サービスの起動

```bash
# ドキュメントディレクトリ作成
mkdir -p documents data/converted_docs data/chroma_db data/models

# Docker Composeで起動
docker-compose up -d

# ステータス確認
docker-compose ps

# ログ確認
docker-compose logs -f
```

#### 2.3.8 サーバーの停止

```bash
# Docker Composeでサービスを停止
docker-compose stop

# または、サービスを完全に停止・削除
docker-compose down

# コンテナ内からstop.pyを使用（通常は docker-compose stop を使用）
docker-compose exec mcp-server python stop.py
```

**注意:** Docker環境では `docker-compose stop` または `docker-compose down` の使用を推奨します。`stop.py` はコンテナ内でサーバープロセスを停止するためのツールですが、Docker環境ではコンテナ自体を停止することが一般的です。

#### 2.3.9 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/sse

# サーバーログ確認
docker-compose logs mcp-server

# コンテナ内での確認
docker exec -it local-rag-mcp-server bash

# Ollama接続確認
curl http://localhost:11434/api/tags
```

---

## 3. 自己完結型パッケージ方式の手順

### 3.1 概要

> **詳細なビルド手順・オプション**: スタンドアロンビルドの詳細な技術情報については、[packaging/standalone/README.md](../packaging/standalone/README.md) を参照してください。

自己完結型パッケージ方式は、Python実行環境を含むスタンドアロンバイナリを作成し、Dockerなしで実行できる方式です。Nuitkaコンパイラを使用してPythonコードをネイティブバイナリに変換します。

**注意:** Ollamaは別途インストール・稼働させる必要があります。MCPサーバーは外部のOllamaサービスに接続します。

### 3.2 Windows側（オンライン環境）の手順

#### 3.2.1 ビルド環境の準備

**WSL2 Linux環境を使用（推奨）:**

```powershell
# WSL2 Linuxを起動
wsl -d Ubuntu-22.04

# または、Linuxターミナルを開く
```

```bash
# システムパッケージの更新
# Ubuntu/Debian系:
sudo apt update && sudo apt upgrade -y

# RHEL/CentOS/Rocky系:
# sudo dnf update -y

# ビルド依存関係のインストール
# Ubuntu/Debian系:
sudo apt install -y python3 python3-pip python3-venv gcc g++ make

# RHEL/CentOS/Rocky系:
# sudo dnf install -y python3 python3-pip gcc gcc-c++ make

# Python仮想環境作成
python3 -m venv venv
source venv/bin/activate

# プロジェクトの依存関係をインストール
pip install -r requirements.txt
pip install -r packaging/standalone/requirements-build.txt
```

**Windowsネイティブ環境を使用:**

```powershell
# Python 3.10以上をインストール（未インストールの場合）
# https://www.python.org/downloads/ からダウンロード

# Visual Studio Build Toolsをインストール
# https://visualstudio.microsoft.com/downloads/

# 仮想環境作成
python -m venv venv
.\venv\Scripts\Activate.ps1

# 依存関係をインストール
pip install -r requirements.txt
pip install -r packaging\standalone\requirements-build.txt
```

#### 3.2.2 スタンドアロンバイナリのビルド

**Linux向けビルド（WSL2使用）:**

```bash
# プロジェクトルートに移動
cd /path/to/local-rag-mcp-server

# ビルドスクリプトを実行
./scripts/build-standalone-linux.sh --clean

# または、Pythonスクリプトを使用
python3 build-standalone.py --platform linux --clean

# 生成物の確認
ls -la dist/
```

**Windows向けビルド（Windowsネイティブ）:**

```powershell
# プロジェクトルートに移動
cd C:\path\to\local-rag-mcp-server

# ビルドスクリプトを実行
.\scripts\build-standalone-windows.ps1 -Clean

# または、Pythonスクリプトを使用
python build-standalone.py --platform windows --clean

# 生成物の確認
dir dist\
```

#### 3.2.3 Ollamaのダウンロード

```bash
# Linux用Ollamaバイナリをダウンロード
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama

# または、公式インストールスクリプトを使用（オンライン環境）
curl -fsSL https://ollama.com/install.sh | sh
```

#### 3.2.4 Ollamaモデルのダウンロード

```bash
# Ollamaサーバーを起動（バックグラウンド）
./ollama serve &

# モデルをダウンロード
./ollama pull nomic-embed-text-v2-moe
./ollama pull glm-ocr

# モデルの確認
./ollama list

# モデルファイルの場所を確認
ls ~/.ollama/models/
```

#### 3.2.5 パッケージング

```bash
# パッケージディレクトリ作成
mkdir -p offline-package

# バイナリをコピー
cp dist/local-rag-mcp-server offline-package/

# 設定ファイルテンプレートをコピー
cp config.json.example offline-package/
cp acl.json.example offline-package/ 2>/dev/null || true

# Ollamaバイナリをコピー
cp ollama offline-package/

# Ollamaモデルをコピー
cp -r ~/.ollama/models offline-package/ollama_models

# READMEを作成
cat > offline-package/README.txt << 'EOF'
Local RAG MCP Server - Standalone Package
==========================================

Installation Steps:
1. Copy this directory to target Ubuntu machine
2. Run: ./install.sh
3. Edit config.json
4. Start Ollama: ./ollama serve &
5. Start server: ./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000

Requirements:
- Ubuntu 20.04+ (glibc 2.31+)
- 4GB+ RAM
- Ollama models (included in ollama_models/)
EOF

# インストールスクリプトを作成
cat > offline-package/install.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Local RAG MCP Server Installation ==="

# バイナリに実行権限を付与
chmod +x local-rag-mcp-server
chmod +x ollama

# Ollamaをシステムにインストール（オプション）
read -p "Install Ollama system-wide? (y/n): " install_ollama
if [ "$install_ollama" = "y" ]; then
    sudo cp ollama /usr/local/bin/
    echo "Ollama installed to /usr/local/bin/ollama"
fi

# 設定ファイル作成
if [ ! -f config.json ]; then
    cp config.json.example config.json
    echo "Created config.json from template"
fi

# モデル配置
if [ -d "ollama_models" ]; then
    mkdir -p ~/.ollama/models
    cp -r ollama_models/* ~/.ollama/models/ 2>/dev/null || true
    echo "Models copied to ~/.ollama/models/"
fi

# ディレクトリ作成
mkdir -p documents converted_docs chroma_db

echo ""
echo "=== Installation Complete ==="
echo "Next steps:"
echo "1. Edit config.json"
echo "2. Start Ollama: ./ollama serve &"
echo "3. Start server: ./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000"
EOF

chmod +x offline-package/install.sh

# アーカイブ作成
tar -czvf local-rag-mcp-server-offline.tar.gz offline-package/

# サイズ確認
du -sh offline-package/
du -sh local-rag-mcp-server-offline.tar.gz
```

#### 3.2.6 USBメモリへの転送

```bash
# USBメモリをマウント（Linux/WSL2）
sudo mkdir -p /mnt/usb
sudo mount /dev/sdX1 /mnt/usb

# コピー
cp local-rag-mcp-server-offline.tar.gz /mnt/usb/

# アンマウント
sudo umount /mnt/usb
```

### 3.3 Linux側（オフライン環境）の手順

**注意:** 以下のコマンド例はUbuntu/Debian系をベースにしています。RHEL/CentOS/Rocky系の場合は、パッケージマネージャを適切に読み替えてください（例: `apt` → `yum`/`dnf`）。

#### 3.3.1 パッケージの展開

```bash
# USBメモリをマウント
sudo mkdir -p /mnt/usb
sudo mount /dev/sdX1 /mnt/usb

# パッケージをコピー
cp /mnt/usb/local-rag-mcp-server-offline.tar.gz ~/
cd ~/

# 展開
tar -xzvf local-rag-mcp-server-offline.tar.gz
cd offline-package
```

#### 3.3.2 インストール実行

```bash
# インストールスクリプトを実行
chmod +x install.sh
./install.sh

# または手動インストール
chmod +x local-rag-mcp-server
```

#### 3.3.3 Ollamaのインストールと設定

**重要:** OllamaはMCPサーバーとは別にインストール・稼働させる必要があります。

```bash
# Ollamaがインストール済みか確認
which ollama

# インストールされていない場合、パッケージに含まれるバイナリを使用
# または、システム全体にインストール
sudo cp ollama /usr/local/bin/
sudo chmod +x /usr/local/bin/ollama

# Ollamaサーバーを起動
ollama serve &

# モデルが正しく配置されているか確認
ls ~/.ollama/models/

# モデル一覧確認
ollama list
```

**Ollamaをsystemdサービスとして登録（推奨）:**

```bash
# サービスファイル作成
sudo tee /etc/systemd/system/ollama.service << EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# サービス有効化
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# ステータス確認
sudo systemctl status ollama
```

#### 3.3.4 設定ファイルの編集

```bash
# 設定ファイルを作成
cp config.json.example config.json

# エディタで編集
nano config.json
```

**設定例:**

```json
{
  "source_docs_dir": "/home/user/documents",
  "docs_dir": "/home/user/converted_docs",
  "db_dir": "/home/user/chroma_db",
  "ollama_base_url": "http://localhost:11434",
  "embedding_model": "nomic-embed-text-v2-moe",
  "ocr_model": "glm-ocr",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**acl.json の設定（グループアクセス制御）:**

複数のユーザーグループに対して異なるドキュメントルートへのアクセス権限を設定する場合、`acl.json` を作成します。

```bash
# 設定ファイルを作成
cp acl.json.example acl.json

# エディタで編集
nano acl.json
```

**acl.json の設定例:**

```json
{
  "_default": {
    "name": "Public",
    "allowed_roots": ["Public"]
  },
  "your-api-key-here": {
    "name": "Group A",
    "allowed_roots": ["Group A", "Public"]
  }
}
```

**設定項目の説明:**

| 項目 | 説明 |
|------|------|
| `_default` | デフォルト設定（認証なしのアクセス用） |
| APIキー文字列 | グループを識別するためのAPIキーをキーとして指定 |
| `name` | グループの表示名 |
| `allowed_roots` | アクセス可能なルート（ドキュメントルート）のリスト |

**注意:** `acl.json` はオプションです。グループアクセス制御が必要ない場合は作成する必要はありません。

#### 3.3.5 サーバーの起動

```bash
# Ollamaサーバーが起動していることを確認
curl http://localhost:11434/api/tags

# MCPサーバーを起動
./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000

# または、フォアグラウンドで実行（デバッグ用）
./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000 --log-level debug
```

#### 3.3.6 サーバーの停止

```bash
# フォアグラウンドで実行している場合
Ctrl+C

# バックグラウンドで実行している場合
# プロセスIDを特定して停止
ps aux | grep local-rag-mcp-server
kill <PID>

# または、stop.pyスクリプトを使用
# 注意: stop.pyはPython環境で実行する必要があります
python stop.py
```

**systemdサービスとして登録している場合:**

```bash
# サービスを停止
sudo systemctl stop local-rag-mcp

# サービスを無効化（自動起動を停止）
sudo systemctl disable local-rag-mcp
```

#### 3.3.7 systemdサービスとして登録（推奨）

サーバーを常時稼働させるため、systemdサービスとして登録することを推奨します。

**注意:** systemd以外のinitシステム（OpenRC、SysV init等）を使用している場合は、適切なサービス設定に読み替えてください。

```bash
# サービスファイル作成
sudo tee /etc/systemd/system/local-rag-mcp.service << EOF
[Unit]
Description=Local RAG MCP Server
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/local-rag-mcp-server
ExecStart=/home/$USER/local-rag-mcp-server/local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# サービス有効化
sudo systemctl daemon-reload
sudo systemctl enable local-rag-mcp
sudo systemctl start local-rag-mcp

# ステータス確認
sudo systemctl status local-rag-mcp
```

**注意:** `Restart=always`により、サーバーが異常終了した場合も自動的に再起動されます。

#### 3.3.7 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/sse

# ポート確認
ss -tlnp | grep 8000

# プロセス確認
ps aux | grep local-rag-mcp-server

# Ollama接続確認
curl http://localhost:11434/api/tags
```

---

## 4. トラブルシューティング

### 4.1 Docker方式の問題

#### Dockerがインストールできない

**症状:** `install-linux.sh` でDockerインストールに失敗する

**原因:** オフライン環境でパッケージリポジトリにアクセスできない

**解決策:**

```bash
# オンライン環境でDockerパッケージをダウンロード
# Ubuntu/Debian系の場合:
mkdir docker-packages
cd docker-packages

# 依存パッケージも含めてダウンロード
apt-get download docker-ce docker-ce-cli containerd.io docker-compose-plugin

# または、依存関係を含めてダウンロード
apt-get install --print-uris docker-ce docker-ce-cli containerd.io docker-compose-plugin | \
  awk '{print $2}' | xargs -I {} apt-get download {}

# USBメモリにコピーして、オフライン環境でインストール
sudo dpkg -i *.deb
sudo apt-get install -f  # 依存関係を解決

# RHEL/CentOS/Rocky系の場合:
# yum download docker-ce docker-ce-cli containerd.io docker-compose-plugin
# sudo yum localinstall *.rpm
```

#### Dockerイメージのロードに失敗

**症状:** `docker load -i local-rag-mcp-server.tar` でエラー

**原因:** イメージファイルが破損している、または不完全

**解決策:**

```bash
# ファイルの整合性確認
sha256sum local-rag-mcp-server.tar

# ファイルサイズ確認
ls -lh local-rag-mcp-server.tar

# 再度エクスポート（オンライン環境で）
docker save -o local-rag-mcp-server.tar local-rag-mcp-server:latest
```

#### コンテナが起動しない

**症状:** `docker-compose up -d` 後、コンテナが終了する

**原因:** 設定エラー、ポート競合、ボリュームマウントの問題

**解決策:**

```bash
# ログ確認
docker-compose logs mcp-server

# 詳細ログ
docker-compose logs --tail=100 mcp-server

# コンテナの状態確認
docker-compose ps

# インタラクティブモードでデバッグ
docker run -it --rm local-rag-mcp-server:latest /bin/bash
```

#### Ollama接続エラー（Docker方式）

**症状:** `Connection refused: http://host.docker.internal:11434` または `Ollama not reachable`

**原因:** Ollamaサーバーが起動していない、またはDockerコンテナからアクセスできない

**解決策:**

```bash
# Ollamaサーバーが起動しているか確認
curl http://localhost:11434/api/tags

# Ollamaが起動していない場合
ollama serve &

# または、systemdサービスとして起動
sudo systemctl start ollama

# Docker Desktop (Windows/Mac) の場合
# docker-compose.yml で host.docker.internal が使用可能か確認
# Linux の場合
# docker-compose.yml で extra_hosts が設定されているか確認
# または、環境変数でOllamaのURLを指定
export OLLAMA_BASE_URL=http://172.17.0.1:11434
docker-compose up -d
```

#### Ollamaモデルが見つからない

**症状:** `Error: model 'nomic-embed-text-v2-moe' not found`

**原因:** モデルがダウンロードされていない、または正しくロードされていない

**解決策:**

```bash
# モデル一覧確認
ollama list

# モデルをプル（オフライン環境では事前ダウンロードが必要）
ollama pull nomic-embed-text-v2-moe
ollama pull glm-ocr

# モデルが正しく配置されているか確認
ls ~/.ollama/models/
```

### 4.2 自己完結型パッケージ方式の問題

#### バイナリが実行できない

**症状:** `Permission denied` または `command not found`

**原因:** 実行権限がない、またはglibcのバージョンが古い

**解決策:**

```bash
# 実行権限を付与
chmod +x local-rag-mcp-server

# glibcバージョン確認
ldd --version
# 必要: glibc 2.31以上

# 依存ライブラリ確認
ldd local-rag-mcp-server
```

#### Ollama接続エラー

**症状:** `Connection refused: http://localhost:11434`

**原因:** Ollamaサーバーが起動していない

**解決策:**

```bash
# Ollamaサーバーを起動
./ollama serve &

# または、システムサービスとして起動
sudo systemctl start ollama

# 接続確認
curl http://localhost:11434/api/tags

# ポート確認
ss -tlnp | grep 11434
```

#### モデルが見つからない

**症状:** `Error: model not found`

**原因:** モデルが正しく配置されていない

**解決策:**

```bash
# モデルの場所確認
ls ~/.ollama/models/

# モデル一覧確認
./ollama list

# モデルを再配置
mkdir -p ~/.ollama/models
cp -r ollama_models/* ~/.ollama/models/

# または、モデルを再ダウンロード（オンライン環境）
./ollama pull nomic-embed-text-v2-moe
```

#### 設定ファイルエラー

**症状:** `Error parsing config.json`

**原因:** JSONフォーマットエラー

**解決策:**

```bash
# JSONの妥当性確認
python3 -c "import json; json.load(open('config.json'))"

# または
jq . config.json

# テンプレートから再作成
cp config.json.example config.json
nano config.json
```

### 4.3 共通の問題

#### ポートが使用中

**症状:** `Address already in use` または `Port 8000 is already in use`

**解決策:**

```bash
# 使用中のポート確認
ss -tlnp | grep 8000

# プロセスを特定
lsof -i :8000

# プロセスを終了
kill -9 <PID>

# または、別のポートを使用
# Docker方式: docker-compose.yml の ports を変更
# 自己完結型: --port 8001 オプションを使用
```

#### メモリ不足

**症状:** `Out of memory` またはシステムが遅い

**解決策:**

```bash
# メモリ使用量確認
free -h

# スワップを追加（一時的）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Dockerのメモリ制限（docker-compose.yml）
services:
  mcp-server:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### ドキュメントがインデックスされない

**症状:** 検索結果が空、またはドキュメントが見つからない

**解決策:**

```bash
# ドキュメントディレクトリの確認
ls -la documents/

# 設定ファイルのパス確認
cat config.json | grep source_docs_dir

# ChromaDBの状態確認
ls -la chroma_db/

# インデックスを再構築（Docker方式）
docker-compose exec mcp-server python update_index.py --rebuild

# インデックスを再構築（自己完結型）
./local-rag-mcp-server --rebuild-index
```

### 4.4 ログの確認方法

#### Docker方式

```bash
# 全体のログ
docker-compose logs

# 特定のサービスのログ
docker-compose logs mcp-server
docker-compose logs ollama

# リアルタイムログ
docker-compose logs -f mcp-server

# 直近のログ
docker-compose logs --tail=100 mcp-server

# タイムスタンプ付き
docker-compose logs -t mcp-server
```

#### 自己完結型パッケージ方式

```bash
# 標準出力をファイルにリダイレクト
./local-rag-mcp-server --transport sse --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# ログの確認
tail -f server.log

# systemdサービスの場合
sudo journalctl -u local-rag-mcp -f

# 特定期間のログ
sudo journalctl -u local-rag-mcp --since "1 hour ago"
```

---

## 5. 付録

### 5.1 必要なストレージ容量

#### Docker方式

**注意:** Docker方式ではOllamaは含まれず、外部サービスとして稼働します。

| コンポーネント | サイズ |
|---------------|--------|
| MCP Server イメージ | ~500MB |
| Docker ボリューム（データ） | ~1-5GB |
| **合計（最小）** | ~1.5GB |
| **合計（推奨）** | ~5GB |

**Ollama（別途必要）:**

| コンポーネント | サイズ |
|---------------|--------|
| Ollama バイナリ | ~50MB |
| nomic-embed-text-v2-moe モデル | ~274MB |
| glm-ocr モデル | ~2GB |

#### 自己完結型パッケージ方式

| コンポーネント | サイズ |
|---------------|--------|
| スタンドアロンバイナリ | ~100-250MB |
| データ（ChromaDB等） | ~1-5GB |
| **合計（最小）** | ~1GB |
| **合計（推奨）** | ~5GB |

**Ollama（別途必要）:**

| コンポーネント | サイズ |
|---------------|--------|
| Ollama バイナリ | ~50MB |
| nomic-embed-text-v2-moe モデル | ~274MB |
| glm-ocr モデル | ~2GB |

### 5.2 推奨USBメモリサイズ

| 方式 | 最小 | 推奨 |
|------|------|------|
| Docker方式（MCPサーバーのみ） | 4GB | 8GB |
| 自己完結型（MCPサーバーのみ） | 2GB | 4GB |
| Ollamaモデル込み | +4GB | +8GB |

**注意:** USBメモリは exFAT形式を推奨（4GB以上のファイルに対応）

### 5.3 ネットワーク設定

#### プロキシ環境（オンライン環境）

```bash
# HTTP プロキシ設定
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1

# Docker プロキシ設定
mkdir -p ~/.docker
cat > ~/.docker/config.json << EOF
{
  "proxies": {
    "default": {
      "httpProxy": "http://proxy.example.com:8080",
      "httpsProxy": "http://proxy.example.com:8080",
      "noProxy": "localhost,127.0.0.1"
    }
  }
}
EOF
```

#### オフライン環境でのプロキシ無効化

```bash
# プロキシ設定を削除
unset HTTP_PROXY HTTPS_PROXY NO_PROXY

# Docker設定から削除
cat > ~/.docker/config.json << EOF
{}
EOF
```

#### ファイアウォール設定

```bash
# 必要なポートを開放
# Ubuntu/Debian系:
sudo ufw allow 8000/tcp  # MCP Server
sudo ufw allow 11434/tcp # Ollama

# RHEL/CentOS/Rocky系:
# sudo firewall-cmd --add-port=8000/tcp --permanent
# sudo firewall-cmd --add-port=11434/tcp --permanent
# sudo firewall-cmd --reload

# または、ローカルのみに制限
sudo ufw allow from 127.0.0.1 to any port 8000
sudo ufw allow from 127.0.0.1 to any port 11434
```

### 5.4 システム要件詳細

#### 最小要件

| 項目 | Docker方式 | 自己完結型 |
|------|-----------|-----------|
| CPU | 2コア | 2コア |
| メモリ | 4GB | 2GB |
| ディスク | 10GB | 5GB |
| OS | Linux（Docker対応） | Linux（glibc 2.31+） |

#### 推奨要件

| 項目 | Docker方式 | 自己完結型 |
|------|-----------|-----------|
| CPU | 4コア | 4コア |
| メモリ | 8GB | 4GB |
| ディスク | 20GB | 10GB |
| OS | Linux（Ubuntu 22.04/Debian系、RHEL/CentOS系等） | Linux（Ubuntu 22.04/Debian系、RHEL/CentOS系等） |

**注意:** コマンド例は主にUbuntu/Debian系をベースにしています。RHEL/CentOS/Rocky系の場合は、パッケージマネージャを `apt` から `yum`/`dnf` に読み替えてください。

#### GPU使用時の追加要件

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA GPU（CUDA対応） |
| VRAM | 4GB以上 |
| ドライバ | NVIDIA Driver 470+ |
| Docker | NVIDIA Container Toolkit（Docker方式の場合） |

### 5.5 ファイル一覧

#### Docker方式パッケージ

**注意:** OllamaはDocker Composeに含まれず、外部サービスとして稼働します。

```
offline-package/
├── local-rag-mcp-server.tar    # MCP Server Dockerイメージ
├── docker-compose.yml          # Docker Compose設定
├── config.json.example         # 設定テンプレート
├── acl.json.example            # ACLテンプレート
├── Dockerfile                  # Dockerfile（参照用）
├── docker-entrypoint.sh        # エントリーポイントスクリプト
├── install-linux.sh            # インストールスクリプト
└── README.md                   # README
```

#### 自己完結型パッケージ

```
offline-package/
├── local-rag-mcp-server        # 実行ファイル
├── ollama                      # Ollamaバイナリ（オプション）
├── ollama_models/              # Ollamaモデル（オプション）
│   ├── manifests/              # モデルマニフェスト
│   └── blobs/                  # モデルデータ
├── config.json.example         # 設定テンプレート
├── acl.json.example            # ACLテンプレート
├── install.sh                  # インストールスクリプト
└── README.txt                  # README
```

**設定ファイルの説明:**

| ファイル | 説明 |
|---------|------|
| `config.json.example` | MCPサーバーの基本設定テンプレート。`config.json` としてコピーして使用 |
| `acl.json.example` | グループアクセス制御設定テンプレート。グループアクセス制御が必要な場合、`acl.json` としてコピーして使用 |

### 5.6 参考リンク

- [Docker公式ドキュメント](https://docs.docker.com/)
- [Docker Compose公式ドキュメント](https://docs.docker.com/compose/)
- [Ollama公式ドキュメント](https://ollama.com/docs)
- [Nuitka公式ドキュメント](https://nuitka.net/)
- [MCP Protocol仕様](https://modelcontextprotocol.io/)
- [オフラインインストール設計書](../plans/offline-installation-design.md)

### 5.7 サポート

問題が解決しない場合は、以下の情報を添えてIssueを作成してください：

1. 使用したインストール方式（Docker / 自己完結型）
2. OS バージョン（`cat /etc/os-release` の出力）
3. エラーメッセージ（ログ）
4. 再現手順
5. 設定ファイル（機密情報を除く）

---

*最終更新: 2026-03-17*