# Changelog

## [Unreleased]

---

## [2026-03-04]

### Added
- **PaddleOCR 対応を追加**  
  代替 OCR エンジンとして **PaddleOCR** のサポートを追加しました。  
  `config.json` の `ocr_engine` パラメータで `"ollama"` と `"paddleocr"` を切り替えられます。  
  PaddleOCR の CPU / GPU 切り替えは `paddleocr_use_gpu` パラメータで制御できます。
- **OCR の異常出力の自動検知機能**  
  OCR の結果が空白、極端に短い文字列、または同一文字の異常な繰り返しになる問題を検知し、OCR処理を失敗（空文字として扱う）として安全にフォールバックする機能を追加しました。
- **PaddleOCR GPU セットアップガイドの追加**  
  GPU 環境で PaddleOCR を動作させるための手順書として `PADDLE_GPU_SETUP.md` を新規追加しました。

### Changed
- **README.md アーキテクチャ図の改善**  
  アーキテクチャ図に「ユーザー側」と「サーバー側」の境目を明記し、どこからどこまでがシステムの管理範囲なのかを視覚的に分かりやすく改善しました。
- **Ollama v0.17.x 系統への注意喚起の追加**  
  v0.17.x の `glm-ocr` 互換性問題（OCRエラー）に関する注意喚起と、ダウングレードや PaddleOCR への切り替え等の回避策を README に記載しました。
- **依存パッケージの更新**  
  `requirements.txt` に `paddleocr==2.7.3` および競合回避のための `numpy<2.0.0` を追加しました。
