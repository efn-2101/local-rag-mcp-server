import json
import sys
from pathlib import Path
import ollama
try:
    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
from PIL import Image
import io
import numpy as np
import pymupdf  # PyMuPDF
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

class FileConverter:
    def __init__(self, config_path: str = "config.json"):
        # Resolve config_path relative to this script's directory
        base_dir = Path(__file__).parent.absolute()
        config_full_path = base_dir / config_path
        
        try:
            with open(config_full_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
             with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self.ocr_engine = self.config.get("ocr_engine", "ollama").lower()
        self.ocr_model = self.config.get("ocr_model", "glm-ocr:latest")
        self.paddleocr_use_gpu = self.config.get("paddleocr_use_gpu", False)
        
        self.ollama_client = ollama.Client(host=self.config.get("ollama_base_url", "http://localhost:11434"))
        
        # Additional text extensions from config
        self.extra_text_extensions = self.config.get("extra_text_extensions", [])
        
        self.paddle_ocr_instance = None
        if self.ocr_engine == "paddleocr":
            if not PADDLE_OCR_AVAILABLE:
                print("Warning: PaddleOCR is configured as the ocr_engine, but paddleocr is not installed. Falling back to ollama.", file=sys.stderr)
                self.ocr_engine = "ollama"
            else:
                try:
                    # Note: You may need to specify lang='japan' or lang='en' based on requirements
                    # Defaulting to 'japan' as the README is in Japanese.
                    # PaddleOCR v2.9+ via PaddleX might not accept use_gpu directly and uses device='gpu:0'
                    try:
                        self.paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='japan', use_gpu=self.paddleocr_use_gpu, show_log=False)
                    except (ValueError, TypeError) as e1:
                        # Fallback for PaddleOCR v2.9+ that uses PaddleX
                        # PaddleX pipeline wrapper may not support use_gpu and show_log kwargs
                        device = 'gpu' if getattr(self, 'paddleocr_use_gpu', False) else 'cpu'
                        try:
                            self.paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='japan', device=device)
                        except Exception as e2:
                            print(f"Fallback failed. Original error: {e1}, Fallback error: {e2}", file=sys.stderr)
                            raise e2
                except Exception as e:
                    print(f"Failed to initialize PaddleOCR: {e}. Falling back to ollama.", file=sys.stderr)
                    self.ocr_engine = "ollama"

    # OCR出力の最小有効文字数
    _OCR_MIN_CHARS = 5
    # 繰り返し文字でのOCR失敗パターンを検出する閾値 (同一文字の割合)
    _OCR_REPEAT_THRESHOLD = 0.6
    # Ollama OCRに送るプロンプト文字列（プロンプトエコー検出にも使用）
    _OCR_PROMPT = "この画像に含まれるテキストや数式、サンプルコードを詳細に抽出してください。マークダウン形式で出力してください。"

    def _is_ocr_output_valid(self, text: str) -> bool:
        """OCR出力が有効かどうかを検証する。
        以下の場合は異常と判定してFalseを返す:
        - 空文字・空白のみ
        - 有効文字数が極端に少ない
        - 1種の文字が過半数を超えて繰り返されている（モデル崩壊パターン）
        - プロンプト文字列をそのままオウム返ししている（Ollama v0.17系 + glm-ocrの既知不具合）
        """
        stripped = text.strip()
        if not stripped:
            return False
        # プロンプトエコー検出: 出力がプロンプト文字列を含むか完全一致している場合
        if self._OCR_PROMPT in stripped:
            print("OCR validation failed: prompt echo detected (model returned the prompt as output)", file=sys.stderr)
            return False
        # 有効文字数チェック（空白を除く）
        non_space = stripped.replace(' ', '').replace('\n', '').replace('\t', '')
        if len(non_space) < self._OCR_MIN_CHARS:
            print(f"OCR validation failed: output too short ({len(non_space)} chars)", file=sys.stderr)
            return False
        # 繰り返し文字のチェック（例: '。。。。。。。。。。' や '<image>' の連続）
        if len(non_space) > 10:
            most_common_char = max(set(non_space), key=non_space.count)
            ratio = non_space.count(most_common_char) / len(non_space)
            if ratio > self._OCR_REPEAT_THRESHOLD:
                print(f"OCR validation failed: repetitive output detected (char '{most_common_char}' = {ratio:.0%})", file=sys.stderr)
                return False
        return True

    def perform_ocr_from_bytes(self, image_data: bytes) -> str:
        """メモリ上の画像データからテキストを抽出する"""
        try:
            # 画像のリサイズ等は共通して行う (PaddleOCRも巨大すぎると重い)
            with Image.open(io.BytesIO(image_data)) as img:
                img = img.convert("RGB") # 3チャンネル化を強制
                max_dim = 1536 # Padding for OCR (Ollama had 1024, but Paddle can handle slightly larger standard sizes well)
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    print(f"Resizing image from {img.size} to {new_size}", file=sys.stderr)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                if self.ocr_engine == "paddleocr" and self.paddle_ocr_instance:
                    # PaddleOCRはNumpy配列を直接受け取れる (あるいはファイルパス)
                    img_array = np.array(img)
                    
                    try:
                        result = self.paddle_ocr_instance.ocr(img_array, cls=True)
                    except TypeError:
                        # PaddleOCR v2.9+ (PaddleX wrapper) drops the cls argument in predict()
                        result = self.paddle_ocr_instance.ocr(img_array)
                    
                    if not result or not result[0]:
                        return ""
                    
                    # 抽出結果のテキストを結合 (単純な行結合)
                    # result は [[[[[x,y],[x,y],[x,y],[x,y]], ('text', confidence)], ...]] の形式
                    # PaddleX wrapperの戻り値フォーマットチェック (dictが返る場合がある)
                    extracted_text = []
                    
                    if isinstance(result[0], dict) and 'rec_text' in result[0]:
                        # PaddleX pipeline return format
                        for text in result[0]['rec_text']:
                            extracted_text.append(text)
                    else:
                        # Classic PaddleOCR format
                        for line in result[0]:
                            if isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                                text = line[1][0]
                                extracted_text.append(text)
                            else:
                                # Fallback string casting
                                extracted_text.append(str(line))
                    
                    ocr_result = "\n".join(extracted_text)
                    if not self._is_ocr_output_valid(ocr_result):
                        print("PaddleOCR output validation failed. Returning empty.", file=sys.stderr)
                        return ""
                    return ocr_result

                else:
                    # Ollama (ggml) 用の処理
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    processed_image_data = buf.getvalue()

                    response = self.ollama_client.generate(
                        model=self.ocr_model,
                        prompt=self._OCR_PROMPT,
                        images=[processed_image_data]
                    )
                    ocr_result = response["response"]
                    if not self._is_ocr_output_valid(ocr_result):
                        print(f"Ollama OCR output validation failed (model={self.ocr_model}). Output may be abnormal. Returning empty.", file=sys.stderr)
                        return ""
                    return ocr_result

        except Exception as e:
            print(f"OCR error (bytes) [{self.ocr_engine}]: {e}", file=sys.stderr)
            return ""

    def perform_ocr(self, image_path: Path) -> str:
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return self.perform_ocr_from_bytes(image_data)
        except Exception as e:
            print(f"OCR error for {image_path}: {e}", file=sys.stderr)
            return ""

    def extract_text_from_pdf(self, pdf_path: Path, progress_callback=None) -> str:
        try:
            doc = pymupdf.open(pdf_path)
            pages_text = []
            total_pages = len(doc)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                print(f"Converting PDF page {i+1}/{total_pages}...", file=sys.stderr)
                if progress_callback:
                    progress_callback(i + 1, total_pages, pdf_path.name)
                page_text = self.perform_ocr_from_bytes(img_data)
                pages_text.append((i + 1, page_text))
            doc.close()

            # 全ページでOCRテキストが取得できなかった場合はファイル生成しない
            if not any(text.strip() for _, text in pages_text):
                print(f"OCR failed for all pages in {pdf_path.name}. Skipping file generation.", file=sys.stderr)
                return ""

            full_text = f"# PDF Content: {pdf_path.name}\n\n"
            for page_num, page_text in pages_text:
                full_text += f"\n## Page {page_num}\n\n{page_text}\n"
            return full_text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}", file=sys.stderr)
            return ""

    def extract_text_from_docx(self, file_path: Path) -> str:
        try:
            doc = Document(file_path)
            full_text = [f"# DOCX Content: {file_path.name}\n"]
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n\n".join(full_text)
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}", file=sys.stderr)
            return ""

    def extract_text_from_xlsx(self, file_path: Path) -> str:
        try:
            wb = load_workbook(file_path, data_only=True)
            text_parts = [f"# XLSX Content: {file_path.name}\n"]
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                text_parts.append(f"## Sheet: {sheet}")
                # Simple Markdown table conversion could be added here, currently tab-separated
                for row in ws.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    if row_text.strip().replace("|", ""):
                         text_parts.append(f"| {row_text} |")
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Error reading XLSX {file_path}: {e}", file=sys.stderr)
            return ""

    def extract_text_from_pptx(self, file_path: Path) -> str:
        try:
            prs = Presentation(file_path)
            text_parts = [f"# PPTX Content: {file_path.name}\n"]
            for i, slide in enumerate(prs.slides):
                text_parts.append(f"## Slide {i+1}")
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_parts.append(shape.text)
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Error reading PPTX {file_path}: {e}", file=sys.stderr)
            return ""

    def convert_file(self, file_path: Path, output_path: Path, progress_callback=None):
        """ファイルを変換してMarkdownとして保存する。
        戻り値:
            True  - 変換・保存に成功
            False - スキップ（未サポート形式・タイムスタンプが新しいので変換不要）
            None  - 変換を試みたがOCR失敗等でコンテンツが取得できなかった
        """
        if file_path.name.startswith("~$"):
            return False
            
        if output_path.exists():
            if output_path.stat().st_mtime > file_path.stat().st_mtime:
                print(f"Skipping {file_path.name} (uptodate)", file=sys.stderr)
                return False

        print(f"Converting {file_path.name}...", file=sys.stderr)
        attempted = False  # 変換を試みたかどうか
        content = None
        if file_path.suffix.lower() == ".pdf":
            attempted = True
            content = self.extract_text_from_pdf(file_path, progress_callback=progress_callback)
        elif file_path.suffix.lower() == ".docx":
            attempted = True
            content = self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() == ".xlsx":
            attempted = True
            content = self.extract_text_from_xlsx(file_path)
        elif file_path.suffix.lower() == ".pptx":
            attempted = True
            content = self.extract_text_from_pptx(file_path)
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            attempted = True
            content = self.perform_ocr(file_path)
        elif file_path.suffix.lower() in [
            # テキスト・マークダウン
            ".txt", ".md",
            # Web
            ".html", ".htm", ".css", ".js", ".ts",
            # データ形式
            ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".env",
            # Python
            ".py",
            # C / C++
            ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
            # Verilog / SystemVerilog
            ".v", ".sv", ".svh", ".vh",
            # Rust
            ".rs",
            # Go
            ".go",
            # Java / Kotlin
            ".java", ".kt",
            # その他
            ".sh", ".bash", ".bat", ".ps1", ".rb", ".php", ".swift", ".cs",
        ] + self.extra_text_extensions:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f"# Source: {file_path.name}\n\n" + f.read()
            except Exception:
                content = None
        
        if content:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved to {output_path}", file=sys.stderr)
            return True
        elif attempted:
            # 変換を試みたが内容が空（OCR失敗等）
            print(f"OCR/conversion produced no content for {file_path.name} (skipping file generation)", file=sys.stderr)
            return None
        else:
            print(f"Skipped {file_path.name} (unsupported or error)", file=sys.stderr)
            return False

def main():
    converter = FileConverter()
    
    # Input/Output directories
    base_dir = Path(__file__).parent.absolute()
    
    source_dir_name = converter.config.get("source_docs_dir", "documents")
    docs_dir = (base_dir / source_dir_name).resolve()
    
    output_dir_name = converter.config.get("docs_dir", "converted_docs")
    output_dir = (base_dir / output_dir_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {docs_dir}...", file=sys.stderr)
    if not docs_dir.exists():
        print(f"Directory {docs_dir} not found.", file=sys.stderr)
        return

    # 1. Sync: Convert new/modified files
    valid_markdown_files = set()
    
    for file_path in docs_dir.rglob("*"):
        if file_path.is_file():
            # Calculate output path preserving subdirectory structure
            rel_path = file_path.relative_to(docs_dir)
            # Replace extension with .md for the output file
            # If original file is .md, avoiding conflict is good, but typically we map .pdf -> .md
            # If we have foo.pdf, we get foo.md.
            output_rel_path = rel_path.with_suffix(".md")
            output_path = output_dir / output_rel_path
            
            # Convert (will check timestamp inside)
            converter.convert_file(file_path, output_path)
            
            # Keep track of valid output files to detect orphans
            valid_markdown_files.add(output_path.resolve())

    # 2. Cleanup: Remove orphaned markdown files in converted_docs
    # Only iterate if we effectively scanned documents.
    print(f"Checking for deleted files in {output_dir}...", file=sys.stderr)
    for md_file in output_dir.rglob("*.md"):
        if md_file.is_file():
            if md_file.resolve() not in valid_markdown_files:
                print(f"Removing orphaned file: {md_file}", file=sys.stderr)
                try:
                    md_file.unlink()
                    # Try to remove empty parent directories
                    parent = md_file.parent
                    while parent != output_dir:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                except Exception as e:
                    print(f"Error removing {md_file}: {e}", file=sys.stderr)

    print("File conversion completed.", file=sys.stderr)

if __name__ == "__main__":
    main()
