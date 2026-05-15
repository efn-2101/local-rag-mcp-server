#!/usr/bin/env python3
"""
既存環境向け: source_docs_dir 内のファイルハッシュを一括生成し、
.source_hashes.json を作成する移行ツール。

既存環境で一度実行すれば、次回 sync_documents からハッシュベースの
OCRスキップ判定が有効になる。

使い方:
    python _generate_source_hashes.py

処理内容:
    1. config.json から source_docs_dir と db_dir を読み込む
    2. source_docs_dir 内の各ファイルについて、対応するMDファイルが存在するか確認
    3. MDファイルが存在するファイルのハッシュを計算
    4. db_dir 配下に .source_hashes.json を保存
"""

import os
import sys
import json
import hashlib
from pathlib import Path


def _normalize_path(path_str: str) -> str:
    """パス文字列を正規化する。"""
    normalized = path_str.replace("\\", "/")
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\t", " ")
    import re
    normalized = re.sub(r"[\u2000-\u200a]", " ", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    normalized = normalized.strip(" ")
    return normalized


def _compute_file_hash(file_path: Path) -> str:
    """ファイル内容のSHA-256ハッシュを計算する。"""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}", file=sys.stderr)
        return ""


def main():
    base_dir = Path(__file__).parent.absolute()

    # Load config
    config_locations = [
        base_dir / "config.json",
        Path("config.json"),
    ]

    config = None
    for loc in config_locations:
        if loc.exists():
            try:
                with open(loc, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"Config loaded from {loc}", file=sys.stderr)
                break
            except Exception as e:
                print(f"Error loading config from {loc}: {e}", file=sys.stderr)

    if config is None:
        print("Could not find or load config.json", file=sys.stderr)
        sys.exit(1)

    source_docs_dir_conf = config.get("source_docs_dir", "")
    db_dir_conf = config.get("db_dir", "")
    docs_dir_conf = config.get("docs_dir", "")

    if not source_docs_dir_conf:
        print("source_docs_dir is not configured. Nothing to do.", file=sys.stderr)
        sys.exit(0)

    source_dir = Path(source_docs_dir_conf).resolve()
    docs_dir = (base_dir / docs_dir_conf).resolve() if docs_dir_conf else None
    db_dir = (base_dir / db_dir_conf).resolve() if db_dir_conf else base_dir

    if not source_dir.exists():
        print(f"source_docs_dir does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    source_hashes_path = db_dir / ".source_hashes.json"

    # Load existing hashes if present
    source_hashes = {}
    if source_hashes_path.exists():
        try:
            with open(source_hashes_path, "r", encoding="utf-8") as f:
                source_hashes = json.load(f)
            print(f"Loaded existing hashes from {source_hashes_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading existing hashes: {e}", file=sys.stderr)

    generated = 0
    skipped = 0
    errors = 0

    for file_path in source_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("~$"):
            continue

        rel = file_path.relative_to(source_dir)
        rel_key = _normalize_path(str(rel))

        # Only generate hash if corresponding MD file exists
        if docs_dir:
            out_path = docs_dir / rel.with_suffix(".md")
            if not out_path.exists():
                print(f"Skipping (no MD yet): {rel}", file=sys.stderr)
                skipped += 1
                continue

        current_hash = _compute_file_hash(file_path)
        if current_hash:
            source_hashes[rel_key] = current_hash
            print(f"Generated hash: {rel}", file=sys.stderr)
            generated += 1
        else:
            print(f"Failed to hash: {rel}", file=sys.stderr)
            errors += 1

    # Save hashes
    try:
        with open(source_hashes_path, "w", encoding="utf-8") as f:
            json.dump(source_hashes, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {generated} hashes to {source_hashes_path}", file=sys.stderr)
        print(f"Skipped: {skipped}, Errors: {errors}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving hashes: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
