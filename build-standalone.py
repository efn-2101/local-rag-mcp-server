#!/usr/bin/env python3
"""
Nuitka Build Script for Local RAG MCP Server
=============================================

This script creates a standalone executable using Nuitka.
It handles:
- Python source compilation
- Package inclusion for complex dependencies
- Data file bundling
- Platform-specific configurations

Usage:
    python build-standalone.py [--platform linux|windows] [--onefile] [--clean]
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Project configuration
PROJECT_NAME = "local-rag-mcp-server"
PROJECT_VERSION = "1.0.0"
MAIN_SCRIPT = "server.py"

# Source files to include (Python modules)
SOURCE_FILES = [
    "server.py",
    "rag_engine.py",
    "file_converter.py",
    "ocr_engine.py",
    "file_watcher.py",
    "update_index.py",
    "stop.py",
    "_cleanup_db.py",
]

# Data files to include
DATA_FILES = [
    "config.json.example",
    "acl.json.example",
]

# Packages to include (for Nuitka)
# These are packages that need explicit inclusion due to dynamic imports
PACKAGES_TO_INCLUDE = [
    # Core dependencies
    "mcp",
    "mcp.server",
    "mcp.server.sse",
    "starlette",
    "starlette.applications",
    "starlette.routing",
    "starlette.responses",
    "starlette.middleware",
    "starlette.middleware.cors",
    "uvicorn",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.websockets",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    
    # RAG dependencies
    "ollama",
    "chromadb",
    "chromadb.api",
    "chromadb.config",
    "chromadb.db",
    
    # Search engines
    "rank_bm25",
    "flashrank",
    
    # Document processing
    "fitz",  # PyMuPDF
    "PIL",
    "PIL.Image",
    "openpyxl",
    "docx",
    "pptx",
    
    # Utilities
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "psutil",
    "urllib3",
    "charset_normalizer",
    
    # Additional dependencies for ChromaDB
    "sqlite3",
    "sqlite",
    
    # ONNX Runtime (for flashrank)
    "onnxruntime",
    
    # Additional packages that may be dynamically loaded
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "sniffio",
    "tenacity",
    "overrides",
    "typing_extensions",
    "pydantic",
    "pydantic_core",
    "annotated_types",
    "tqdm",
    "numpy",
    "pypdf",
    "pypdfium2",
]

# Packages that need recursive inclusion
PACKAGES_RECURSIVE = [
    "mcp",
    "starlette",
    "uvicorn",
    "chromadb",
    "flashrank",
    "onnxruntime",
]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.resolve()


def clean_build_directories(project_root: Path) -> None:
    """Clean build and dist directories."""
    print("Cleaning build directories...")
    
    dirs_to_clean = [
        project_root / "build",
        project_root / "dist",
        project_root / f"{PROJECT_NAME}.build",
        project_root / f"{PROJECT_NAME}.dist",
        project_root / f"{PROJECT_NAME}.onefile-build",
    ]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"  Removing: {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)
    
    # Also remove any .pyc files and __pycache__ directories
    for pycache in project_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    
    print("Clean complete.")


def build_nuitka_command(
    project_root: Path,
    platform: str,
    onefile: bool = True,
    output_dir: Optional[Path] = None
) -> List[str]:
    """Build the Nuitka command with all necessary options."""
    
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
    ]
    
    # Output configuration
    if onefile:
        cmd.append("--onefile")
    
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    else:
        output_dir = project_root / "dist"
        cmd.extend(["--output-dir", str(output_dir)])
    
    # Output filename
    cmd.extend(["--output-filename", PROJECT_NAME])
    
    # Python flags
    cmd.append("--python-flag=no_site")
    
    # Assume yes for downloads (Nuitka dependencies)
    cmd.append("--assume-yes-for-downloads")
    
    # Enable plugins for better compatibility
    cmd.extend([
        "--enable-plugin=pylint-warnings",
    ])
    
    # Include packages
    for package in PACKAGES_TO_INCLUDE:
        if package in PACKAGES_RECURSIVE:
            cmd.extend(["--include-package", package])
        else:
            cmd.extend(["--include-module", package])
    
    # Include data files
    for data_file in DATA_FILES:
        source_path = project_root / data_file
        if source_path.exists():
            # Format: source=destination
            cmd.extend([
                "--include-data-file",
                f"{source_path}={data_file}"
            ])
    
    # Include source Python files
    for source_file in SOURCE_FILES:
        source_path = project_root / source_file
        if source_path.exists() and source_file != MAIN_SCRIPT:
            # Include as data file for import
            cmd.extend([
                "--include-data-file",
                f"{source_path}={source_file}"
            ])
    
    # Platform-specific options
    if platform == "linux":
        cmd.extend([
            "--linux-icon=none",  # No icon for now
        ])
    elif platform == "windows":
        cmd.extend([
            "--windows-icon-from-ico=none",  # No icon for now
            "--windows-console-mode=force",  # Force console for debugging
        ])
    
    # Optimization flags
    cmd.extend([
        "--lto=yes",  # Link-time optimization
        "--assume-yes-for-downloads",
    ])
    
    # Add main script
    cmd.append(str(project_root / MAIN_SCRIPT))
    
    return cmd


def run_build(cmd: List[str], project_root: Path) -> int:
    """Execute the Nuitka build command."""
    print("=" * 60)
    print("Starting Nuitka Build")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    print("=" * 60)
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # Run Nuitka
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    return process.returncode


def create_distribution_package(project_root: Path, platform: str) -> None:
    """Create the final distribution package with all necessary files."""
    dist_dir = project_root / "dist"
    
    # Copy additional files to dist
    files_to_copy = [
        "config.json.example",
        "acl.json.example",
        "README.md",
        "LICENSE",
    ]
    
    for filename in files_to_copy:
        source = project_root / filename
        if source.exists():
            dest = dist_dir / filename
            print(f"Copying {source} -> {dest}")
            shutil.copy2(source, dest)
    
    # Create README for standalone package
    readme_content = f"""# {PROJECT_NAME} - Standalone Package

## Version
{PROJECT_VERSION}

## Requirements
- Operating System: {platform.capitalize()}
- Ollama server running on localhost:11434 (or configured URL)
- Required Ollama models:
  - nomic-embed-text-v2-moe (for embeddings)
  - glm-ocr (for OCR, optional)

## Setup

1. Copy `config.json.example` to `config.json`:
   ```bash
   cp config.json.example config.json
   ```

2. Edit `config.json` with your settings:
   - Set `source_docs_dir` to your documents directory
   - Set `ollama_base_url` to your Ollama server URL (default: http://localhost:11434)

3. Ensure Ollama is running with required models:
   ```bash
   ollama serve &
   ollama pull nomic-embed-text-v2-moe
   ollama pull glm-ocr
   ```

## Running

```bash
# Make executable (Linux/macOS)
chmod +x {PROJECT_NAME}

# Run the server
./{PROJECT_NAME} --transport sse --host 0.0.0.0 --port 8000
```

## Index Management

Before using the server, you need to create the initial index. In standalone environments,
use the included `update_index.py` script:

```bash
# Initial index creation or incremental update
python update_index.py

# Force full rebuild
python update_index.py --force
```

**Note:** The index update can be run while the server is running. Changes will be
reflected after the update completes.

## Stopping the Server

```bash
# If running in foreground
Ctrl+C

# If running in background, use stop.py
python stop.py

# Or find and kill the process
ps aux | grep {PROJECT_NAME}
kill <PID>
```

## Configuration

See `config.json.example` for all available configuration options.

## Troubleshooting

### "Ollama connection failed"
- Ensure Ollama is running: `ollama serve`
- Check the URL in config.json

### "Model not found"
- Pull the required model: `ollama pull <model-name>`

### "Permission denied"
- Make the file executable: `chmod +x {PROJECT_NAME}`

## Support

For issues and updates, visit the project repository.
"""
    
    readme_path = dist_dir / "README-standalone.txt"
    print(f"Creating {readme_path}")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create install script for Linux
    if platform == "linux":
        install_script = """#!/bin/bash
# Install script for Local RAG MCP Server Standalone Package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Local RAG MCP Server Installation ==="
echo ""

# Check for config file
if [ ! -f config.json ]; then
    echo "Creating config.json from template..."
    cp config.json.example config.json
    echo "Please edit config.json with your settings."
fi

# Make executable
chmod +x local-rag-mcp-server 2>/dev/null || true

echo ""
echo "=== Installation Complete ==="
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
"""
        install_path = dist_dir / "install.sh"
        print(f"Creating {install_path}")
        with open(install_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(install_script)
        # Make executable
        os.chmod(install_path, 0o755)
    
    print(f"\nDistribution package created in: {dist_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build standalone executable for Local RAG MCP Server using Nuitka"
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default="linux" if sys.platform != "win32" else "windows",
        help="Target platform (default: current platform)"
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        default=True,
        help="Create a single executable file (default: True)"
    )
    parser.add_argument(
        "--no-onefile",
        action="store_true",
        help="Create a directory with executable and dependencies"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directories before building"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the build"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    # Clean if requested
    if args.clean:
        clean_build_directories(project_root)
    
    # Determine onefile setting
    onefile = not args.no_onefile if args.no_onefile else args.onefile
    
    # Get output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Build command
    cmd = build_nuitka_command(project_root, args.platform, onefile, output_dir)
    
    # Run build
    returncode = run_build(cmd, project_root)
    
    if returncode != 0:
        print(f"\nBuild failed with return code: {returncode}")
        sys.exit(returncode)
    
    # Create distribution package
    create_distribution_package(project_root, args.platform)
    
    print("\n" + "=" * 60)
    print("Build completed successfully!")
    print("=" * 60)
    print(f"\nExecutable location: {project_root / 'dist' / PROJECT_NAME}")
    print("\nTo run the server:")
    print(f"  ./dist/{PROJECT_NAME} --transport sse --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()