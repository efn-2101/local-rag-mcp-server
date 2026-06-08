"""
OCRエンジンの抽象化レイヤー
PaddleOCRとOllama glm-ocrを統一的なインターフェースで提供
"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import io

# PaddleOCRのオプショナルインポート
try:
    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False

import ollama
from PIL import Image
import numpy as np


class OCREngine(ABC):
    """OCRエンジンの抽象基底クラス"""
    
    @abstractmethod
    def perform_ocr_from_bytes(self, image_data: bytes) -> str:
        """メモリ上の画像データからテキストを抽出"""
        pass
    
    def perform_ocr(self, image_path: Path) -> str:
        """ファイルパスからテキストを抽出"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return self.perform_ocr_from_bytes(image_data)
        except Exception as e:
            print(f"OCR error for {image_path}: {e}", file=sys.stderr)
            return ""
    
    @abstractmethod
    def is_available(self) -> bool:
        """エンジンが利用可能かどうか"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """エンジン名を取得"""
        pass


class OllamaOCREngine(OCREngine):
    """Ollama glm-ocrを使用するOCRエンジン"""
    
    # OCR出力の最小有効文字数
    _OCR_MIN_CHARS = 5
    # 繰り返し文字でのOCR失敗パターンを検出する閾値
    _OCR_REPEAT_THRESHOLD = 0.6
    # OCRプロンプト
    _OCR_PROMPT = "この画像に含まれるテキストや数式、サンプルコードを詳細に抽出してください。マークダウン形式で出力してください。"
    
    def __init__(self, ollama_base_url: str, ocr_model: str = "glm-ocr:latest"):
        self.ollama_client = ollama.Client(host=ollama_base_url)
        self.ocr_model = ocr_model
    
    def is_available(self) -> bool:
        # Ollamaサーバーへの接続確認は実際の使用時に行う
        return True
    
    def get_name(self) -> str:
        return f"Ollama ({self.ocr_model})"
    
    def perform_ocr_from_bytes(self, image_data: bytes) -> str:
        try:
            # 画像の前処理
            processed_image_data = self._preprocess_image(image_data)
            
            response = self.ollama_client.generate(
                model=self.ocr_model,
                prompt=self._OCR_PROMPT,
                images=[processed_image_data]
            )
            ocr_result = response["response"]
            
            if not self._is_ocr_output_valid(ocr_result):
                print(f"Ollama OCR output validation failed (model={self.ocr_model}).", file=sys.stderr)
                return ""
            
            return ocr_result
        except Exception as e:
            print(f"OCR error (Ollama): {e}", file=sys.stderr)
            return ""
    
    def _preprocess_image(self, image_data: bytes) -> bytes:
        """画像のリサイズ等の前処理"""
        with Image.open(io.BytesIO(image_data)) as img:
            img = img.convert("RGB")
            max_dim = 1536
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                print(f"Resizing image from {img.size} to {new_size}", file=sys.stderr)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    
    def _is_ocr_output_valid(self, text: str) -> bool:
        """OCR出力が有効かどうかを検証"""
        stripped = text.strip()
        if not stripped:
            return False
        # BUG-011 fix: Use similarity-based prompt echo detection
        prompt_words = set(self._OCR_PROMPT.split())
        text_words = set(stripped.split())
        if prompt_words and len(prompt_words & text_words) / len(prompt_words) > 0.8:
            print("OCR validation failed: prompt echo detected", file=sys.stderr)
            return False
        non_space = stripped.replace(' ', '').replace('\n', '').replace('\t', '')
        if len(non_space) < self._OCR_MIN_CHARS:
            print(f"OCR validation failed: output too short ({len(non_space)} chars)", file=sys.stderr)
            return False
        if len(non_space) > 10:
            most_common_char = max(set(non_space), key=non_space.count)
            ratio = non_space.count(most_common_char) / len(non_space)
            if ratio > self._OCR_REPEAT_THRESHOLD:
                print(f"OCR validation failed: repetitive output detected", file=sys.stderr)
                return False
        return True


class PaddleOCREngine(OCREngine):
    """PaddleOCRを使用するOCRエンジン"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._instance = None
        self._actual_gpu = False  # BUG-04 fix: 実際の動作モードを保持

    def is_available(self) -> bool:
        return PADDLE_OCR_AVAILABLE

    def get_name(self) -> str:
        # BUG-04 fix: 設定と実際の動作モードを区別して表示
        return f"PaddleOCR (config=GPU={'enabled' if self.use_gpu else 'disabled'}, actual={'GPU' if self._actual_gpu else 'CPU'})"

    def _initialize(self) -> bool:
        """PaddleOCRインスタンスを初期化（BUG-04 fix: 3段階フォールバック）"""
        if self._instance is not None:
            return True

        if not PADDLE_OCR_AVAILABLE:
            return False

        # 試行1: use_gpu パラメータ + GPU設定
        if self.use_gpu:
            try:
                self._instance = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan',
                    use_gpu=True,
                    show_log=False
                )
                self._actual_gpu = True
                return True
            except Exception as e:
                print(f"PaddleOCR GPU init failed, falling back to CPU: {e}", file=sys.stderr)

        # 試行2: device='cpu' 明示指定
        try:
            self._instance = PaddleOCR(
                use_angle_cls=True,
                lang='japan',
                device='cpu'
            )
            self._actual_gpu = False
            return True
        except (ValueError, TypeError) as e:
            # 試行3: 旧 API (use_gpu=False)
            print(f"PaddleOCR device='cpu' init failed, trying legacy API: {e}", file=sys.stderr)
            try:
                self._instance = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan',
                    use_gpu=False,
                    show_log=False
                )
                self._actual_gpu = False
                return True
            except Exception as e2:
                print(f"PaddleOCR initialization completely failed: {e2}", file=sys.stderr)
                return False
    
    def perform_ocr_from_bytes(self, image_data: bytes) -> str:
        if not self._initialize():
            return ""
        
        try:
            # 画像の前処理
            img_array = self._preprocess_image(image_data)
            
            try:
                result = self._instance.ocr(img_array, cls=True)
            except TypeError:
                # PaddleOCR v2.9+ (PaddleX wrapper) drops the cls argument
                result = self._instance.ocr(img_array)
            
            if not result or not result[0]:
                return ""
            
            # 結果のテキストを結合
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
                        extracted_text.append(str(line))
            
            return "\n".join(extracted_text)
        except Exception as e:
            print(f"OCR error (PaddleOCR): {e}", file=sys.stderr)
            return ""
    
    def _preprocess_image(self, image_data: bytes):
        """画像の前処理（numpy配列を返す）"""
        with Image.open(io.BytesIO(image_data)) as img:
            img = img.convert("RGB")
            max_dim = 1536
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                print(f"Resizing image from {img.size} to {new_size}", file=sys.stderr)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return np.array(img)


class OCREngineFactory:
    """OCRエンジンのファクトリークラス"""
    
    @staticmethod
    def create_engine(config: dict) -> OCREngine:
        """
        設定に基づいてOCRエンジンを作成
        
        Args:
            config: 設定辞書。以下のキーを使用:
                - ocr_engine: "ollama" または "paddleocr"
                - ocr_model: Ollamaモデル名（デフォルト: "glm-ocr:latest"）
                - ollama_base_url: OllamaサーバーURL
                - paddleocr_use_gpu: PaddleOCRでGPUを使用するか
        
        Returns:
            OCREngine: 適切なOCRエンジンインスタンス
        """
        ocr_engine = config.get("ocr_engine", "ollama").lower()
        
        if ocr_engine == "paddleocr":
            if not PADDLE_OCR_AVAILABLE:
                print(
                    "Warning: PaddleOCR is configured as the ocr_engine, "
                    "but paddleocr is not installed. Falling back to ollama.",
                    file=sys.stderr
                )
                # フォールバック
                return OCREngineFactory._create_ollama_engine(config)
            
            use_gpu = config.get("paddleocr_use_gpu", False)
            return PaddleOCREngine(use_gpu=use_gpu)
        
        elif ocr_engine == "ollama":
            return OCREngineFactory._create_ollama_engine(config)
        
        else:
            print(
                f"Warning: Unknown ocr_engine '{ocr_engine}'. "
                "Falling back to ollama.",
                file=sys.stderr
            )
            return OCREngineFactory._create_ollama_engine(config)
    
    @staticmethod
    def _create_ollama_engine(config: dict) -> OllamaOCREngine:
        """Ollama OCRエンジンを作成"""
        ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")
        ocr_model = config.get("ocr_model", "glm-ocr:latest")
        return OllamaOCREngine(ollama_base_url=ollama_base_url, ocr_model=ocr_model)
    
    @staticmethod
    def get_available_engines() -> list:
        """利用可能なOCRエンジンのリストを取得"""
        engines = ["ollama"]  # Ollamaは常に利用可能
        if PADDLE_OCR_AVAILABLE:
            engines.append("paddleocr")
        return engines