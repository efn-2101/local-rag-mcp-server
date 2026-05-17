"""
コンテキスト予算管理モジュール

巨大ドキュメントのRAG返却時に、LLMのコンテキスト上限を超えないよう
テキストの長さを制限・圧縮する機能を提供する。

設計原則:
1. 文の途中で切らない（句読点・改行で分割）
2. 見出し単位で圧縮（セクションごとに判断）
3. 省略を明示（「[省略]」とマーク + 取得方法提示）
4. 詳細優先（常に1件以上の詳細を含める）
5. 段階的取得を促す（path#section で詳細取得可能と明示）
"""

import re
import sys
from typing import List, Optional, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# トークンカウント（tiktoken優先、フォールバックあり）
# ---------------------------------------------------------------------------

def _init_tokenizer():
    """トークナイザーを初期化。tiktokenがなければ簡易フォールバックを使用。"""
    try:
        import tiktoken
        # cl100k_baseはGPT-4/3.5-turboと互換性が高い
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except ImportError:
        print("[context_budget] tiktoken not found. Using fallback tokenizer (chars/2).", file=sys.stderr)
        # フォールバック: 日本語は1文字≒1.5トークン、英語は1単語≒1.3トークン
        # 安全側に倒して文字数/2とする
        return lambda text: max(1, len(text) // 2)


_count_tokens_func = _init_tokenizer()


def count_tokens(text: str) -> int:
    """テキストのトークン数を概算する。"""
    if not text:
        return 0
    return _count_tokens_func(text)


# ---------------------------------------------------------------------------
# コンテキスト予算計算
# ---------------------------------------------------------------------------

class ContextBudget:
    """
    LLMのコンテキスト上限に対する予算管理クラス。
    
    実効RAG予算 = モデル最大トークン数
                  - システムプロンプト予約
                  - 会話履歴予約
                  - 回答生成予約
                  - 安全マージン
    """
    
    # デフォルト設定（128Kモデルを基準）
    DEFAULT_MAX_CONTEXT = 128000
    DEFAULT_SYSTEM_RESERVE = 1000
    DEFAULT_GENERATION_RESERVE = 4000
    DEFAULT_SAFETY_MARGIN_RATIO = 0.1  # 10%
    
    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        system_reserve: Optional[int] = None,
        generation_reserve: Optional[int] = None,
        safety_margin_ratio: Optional[float] = None,
    ):
        self.max_context = max_context_tokens or self.DEFAULT_MAX_CONTEXT
        self.system_reserve = system_reserve or self.DEFAULT_SYSTEM_RESERVE
        self.generation_reserve = generation_reserve or self.DEFAULT_GENERATION_RESERVE
        self.safety_margin = int(self.max_context * (safety_margin_ratio or self.DEFAULT_SAFETY_MARGIN_RATIO))
    
    def get_available_budget(self, history_tokens: int = 0) -> int:
        """
        現在の会話履歴トークン数を考慮した、RAGテキストに使える予算を計算する。
        
        Args:
            history_tokens: 現在の会話履歴のトークン数（概算）
        
        Returns:
            RAGテキストに使える最大トークン数
        """
        reserved = (
            self.system_reserve +
            history_tokens +
            self.generation_reserve +
            self.safety_margin
        )
        return max(0, self.max_context - reserved)
    
    def __repr__(self):
        return (
            f"ContextBudget(max={self.max_context}, "
            f"available≈{self.get_available_budget()})"
        )


# ---------------------------------------------------------------------------
# Markdown構造解析
# ---------------------------------------------------------------------------

class MarkdownSection:
    """Markdownのセクション（見出し＋本文）を表す。"""
    
    def __init__(self, heading: str, level: int, content: str, line_start: int = 0):
        self.heading = heading      # 見出しテキスト（# 除く）
        self.level = level          # 見出しレベル（1=#, 2=##, ...）
        self.content = content      # 見出し行を含む本文
        self.line_start = line_start  # ファイル内の開始行番号
        self.token_count: Optional[int] = None  # 遅延計算
    
    def get_token_count(self) -> int:
        if self.token_count is None:
            self.token_count = count_tokens(self.content)
        return self.token_count
    
    def __repr__(self):
        return f"MarkdownSection(heading='{self.heading[:30]}...', level={self.level}, tokens={self.get_token_count()})"


def parse_markdown_sections(text: str) -> List[MarkdownSection]:
    """
    Markdownテキストを見出し単位でセクションに分割する。
    
    Args:
        text: Markdownテキスト
    
    Returns:
        MarkdownSectionのリスト（見出しレベル順に並ぶ）
    """
    lines = text.split('\n')
    sections: List[MarkdownSection] = []
    
    current_heading = ""
    current_level = 0
    current_content_lines: List[str] = []
    current_line_start = 0
    
    # 見出しパターン: # で始まり、後に空白が続く行
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    
    for i, line in enumerate(lines):
        match = heading_pattern.match(line)
        if match:
            # 前のセクションを保存
            if current_content_lines:
                content = '\n'.join(current_content_lines)
                sections.append(MarkdownSection(
                    heading=current_heading,
                    level=current_level,
                    content=content,
                    line_start=current_line_start,
                ))
            
            # 新しいセクション開始
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            current_content_lines = [line]
            current_line_start = i
        else:
            current_content_lines.append(line)
    
    # 最後のセクションを保存
    if current_content_lines:
        content = '\n'.join(current_content_lines)
        sections.append(MarkdownSection(
            heading=current_heading,
            level=current_level,
            content=content,
            line_start=current_line_start,
        ))
    
    # 見出しが1つもない場合は、全体を1つのセクションとして扱う
    if not sections and text.strip():
        sections.append(MarkdownSection(
            heading="(見出しなし)",
            level=0,
            content=text,
            line_start=0,
        ))
    
    return sections


def extract_section_by_heading(text: str, section_query: str) -> Optional[str]:
    """
    見出し名または見出し番号でセクションを抽出する。
    
    Args:
        text: Markdownテキスト
        section_query: 見出し名（例: "初期設定"）または番号（例: "2"）
    
    Returns:
        該当セクションのテキスト、見つからなければNone
    """
    sections = parse_markdown_sections(text)
    
    # 1. 完全一致で検索
    for sec in sections:
        if sec.heading.strip() == section_query.strip():
            return sec.content
    
    # 2. 部分一致で検索
    query_lower = section_query.strip().lower()
    for sec in sections:
        if query_lower in sec.heading.lower():
            return sec.content
    
    # 3. 番号指定の場合（"1. " や "1 " で始まる見出しを検索）
    if section_query.strip().isdigit():
        num = section_query.strip()
        for sec in sections:
            # "1. タイトル" または "1 タイトル" の形式を検索
            if re.match(rf'^\s*{re.escape(num)}[.\s]\s*', sec.heading):
                return sec.content
    
    return None


# ---------------------------------------------------------------------------
# 構造ベース圧縮（LLM不要）
# ---------------------------------------------------------------------------

# Common English abbreviations that end with a period
_ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'no', 'pp', 'et', 'al',
    'i.e', 'e.g', 'vs', 'vol', 'vols', 'inc', 'ltd', 'jr', 'sr',
    'st', 'ave', 'blvd', 'rd', 'dept', 'univ', 'corp', 'co',
    'fig', 'figs', 'et al', 'et al.', 'i.e.', 'e.g.', 'vs.',
    'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'no.',
}


def _is_abbreviation(word: str) -> bool:
    """単語が略称かどうかを判定（末尾の句読点を除去）"""
    cleaned = word.rstrip('.').lower()
    return cleaned in _ABBREVIATIONS


def _split_sentences(text: str) -> List[str]:
    """日本語・英語の文を分割する簡易実装。"""
    # 日本語: 。！？で分割
    # 英語: .!? で分割（但し Mr. Dr. 等は除外）
    sentences = []
    current = ""
    i = 0
    while i < len(text):
        current += text[i]
        if text[i] in '。！？':
            sentences.append(current.strip())
            current = ""
        elif text[i] in '.!?' and i > 0:
            # BUG-010 fix: Improved abbreviation detection
            # Get the last word before the punctuation
            text_before = current[:-1]  # Exclude the punctuation itself
            words = text_before.strip().split()
            last_word = words[-1] if words else ""
            if not _is_abbreviation(last_word):
                sentences.append(current.strip())
                current = ""
        i += 1
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _contains_important_keywords(text: str) -> bool:
    """重要そうなキーワードが含まれているかチェック。"""
    important_patterns = [
        r'重要', r'必須', r'必ず', r'注意', r'警告', r'危険',
        r'設定', r'手順', r'ステップ', r'方法',
        r'エラー', r'例外', r'トラブルシューティング',
        r'定義', r'仕様', r'要件', r'制約',
        r'\d+',  # 数字を含む
        r'[A-Z][a-z]+[A-Z]',  # キャメルケース（関数名等）
        r'`[^`]+`',  # インラインコード
    ]
    text_lower = text.lower()
    for pattern in important_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def compress_section_text(text: str, max_sentences: int = 3) -> str:
    """
    セクションのテキストを圧縮する。
    先頭文 + 重要キーワードを含む文 + 末尾文 を残す。
    
    Args:
        text: 圧縮対象のテキスト
        max_sentences: 残す最大文数
    
    Returns:
        圧縮されたテキスト
    """
    sentences = _split_sentences(text)
    
    if len(sentences) <= max_sentences:
        return text
    
    # 先頭文は必ず残す
    result = [sentences[0]]
    
    # 重要キーワードを含む文を優先的に追加
    for s in sentences[1:-1]:
        if _contains_important_keywords(s):
            result.append(s)
            if len(result) >= max_sentences:
                break
    
    # 末尾文を追加（まだ追加していれば）
    if len(sentences) > 1 and (len(result) < max_sentences or sentences[-1] not in result):
        result.append(sentences[-1])
    
    compressed = ' '.join(result)
    original_count = len(sentences)
    return f"{compressed}\n[このセクションは圧縮表示されています（原文{original_count}文から{len(result)}文を抽出）。詳細は get_document_content で該当セクションを指定して取得できます。]"


def compress_document_structure(
    text: str,
    budget_tokens: int,
    detail_level: str = "auto",
) -> Tuple[str, List[str], bool]:
    """
    ドキュメント全文を構造ベースで圧縮し、予算内に収める。
    
    Args:
        text: ドキュメント全文
        budget_tokens: 使用可能なトークン数
        detail_level: "auto" | "summary" | "full"
            - "full": 上限内に収まる限り全文を返す（超過時のみ圧縮）
            - "summary": 常に圧縮して返す
            - "auto": 上限に応じて自動判断
    
    Returns:
        (圧縮後のテキスト, 省略されたセクション名のリスト, 圧縮されたか)
    """
    sections = parse_markdown_sections(text)
    total_tokens = sum(s.get_token_count() for s in sections)
    
    # fullモードで予算内に収まる場合は全文返却
    if detail_level == "full" and total_tokens <= budget_tokens:
        return text, [], False
    
    # summaryモード、またはautoで予算超過の場合は圧縮
    if detail_level == "summary" or (detail_level == "auto" and total_tokens > budget_tokens):
        return _compress_sections(sections, budget_tokens)
    
    # autoモードで予算内に収まる場合
    return text, [], False


def _compress_sections(
    sections: List[MarkdownSection],
    budget_tokens: int,
) -> Tuple[str, List[str], bool]:
    """
    セクションを圧縮して予算内に収める内部実装。
    
    4段階フォールバック:
    1. 全文詳細（予算内に収まる場合）
    2. 重要セクション詳細 + 残り圧縮
    3. 最優先セクション詳細 + 残りタイトルのみ
    4. 全件タイトル + 要約のみ
    """
    if not sections:
        return "", [], False
    
    # 予算の安全マージン（ヘッダー等の固定分を考慮）
    effective_budget = max(int(budget_tokens * 0.9), 500)
    
    # ステージ1: 全文が収まるかチェック
    total = sum(s.get_token_count() for s in sections)
    if total <= effective_budget:
        full_text = '\n\n'.join(s.content for s in sections)
        return full_text, [], False
    
    # ステージ2: 重要セクションを優先して詳細、残りを圧縮
    # 重要度スコア: レベルが低い（大きな見出し）ほど重要
    scored = [(s, 100 - s.level * 10 + (50 if _contains_important_keywords(s.content) else 0)) for s in sections]
    scored.sort(key=lambda x: -x[1])  # スコア降順
    
    result_parts: List[str] = []
    omitted_sections: List[str] = []
    used_tokens = 0
    detailed_count = 0
    
    # 先頭セクションは必ず詳細に（要約地獄回避）
    first_section = sections[0]
    first_tokens = first_section.get_token_count()
    if first_tokens <= effective_budget * 0.5:  # 予算の半分まで許容
        result_parts.append(first_section.content)
        used_tokens += first_tokens
        detailed_count += 1
    
    # 残りのセクションを重要度順に処理
    for section, score in scored:
        if section == first_section:
            continue
        
        sec_tokens = section.get_token_count()
        
        # まだ詳細を入れられるか
        if detailed_count < 3 and used_tokens + sec_tokens <= effective_budget * 0.7:
            result_parts.append(section.content)
            used_tokens += sec_tokens
            detailed_count += 1
            continue
        
        # 圧縮版を試行
        compressed = compress_section_text(section.content, max_sentences=2)
        compressed_tokens = count_tokens(compressed)
        
        if used_tokens + compressed_tokens <= effective_budget * 0.85:
            result_parts.append(f"\n## {section.heading}\n{compressed}")
            used_tokens += compressed_tokens
            continue
        
        # タイトルのみ
        title_only = f"\n## {section.heading}\n[このセクションは省略されています。詳細は get_document_content で path=... section='{section.heading}' を指定して取得できます。]"
        title_tokens = count_tokens(title_only)
        
        if used_tokens + title_tokens <= effective_budget:
            result_parts.append(title_only)
            used_tokens += title_tokens
            omitted_sections.append(section.heading)
        else:
            # 予算を超えたら以降は完全省略
            omitted_sections.append(section.heading)
    
    # 省略セクション一覧を追加
    if omitted_sections:
        omitted_notice = (
            "\n\n[省略されたセクション一覧:]\n" +
            "\n".join(f"- {name}" for name in omitted_sections) +
            "\n\n[詳細取得方法:]\n" +
            "get_document_content(path=\"...\", section=\"セクション名\") で該当セクションの詳細を取得できます。"
        )
        result_parts.append(omitted_notice)
    
    final_text = '\n\n'.join(result_parts)
    return final_text, omitted_sections, True


# ---------------------------------------------------------------------------
# 検索結果の予算管理
# ---------------------------------------------------------------------------

def fit_search_results_to_budget(
    results: List[Dict[str, Any]],
    budget_tokens: int,
    format_template: str = "--- Result (Root: {root}, Category: {category}, Path: {path}) ---\n{content}\n",
) -> Tuple[str, bool, List[str]]:
    """
    検索結果をトークン予算内に収める。
    
    Args:
        results: 検索結果のリスト（各要素は dict with 'id', 'content', 'metadata'）
        budget_tokens: 使用可能なトークン数
        format_template: 結果のフォーマットテンプレート
    
    Returns:
        (フォーマット済みテキスト, 切り詰められたか, 省略された結果IDリスト)
    """
    if not results:
        return "", False, []
    
    effective_budget = max(int(budget_tokens * 0.9), 500)
    formatted_parts: List[str] = []
    total_tokens = 0
    omitted_ids: List[str] = []
    truncated = False
    
    for i, result in enumerate(results):
        path = result.get("id", "unknown")
        content = result.get("content", "")
        metadata = result.get("metadata", {})
        root = metadata.get("root_folder", "unknown")
        category = metadata.get("category", "unknown")
        
        formatted = format_template.format(
            path=path,
            content=content,
            root=root,
            category=category,
        )
        tokens = count_tokens(formatted)
        
        if total_tokens + tokens <= effective_budget:
            formatted_parts.append(formatted)
            total_tokens += tokens
        else:
            # 予算超過: 圧縮版を試行
            compressed_content = compress_section_text(content, max_sentences=2)
            compressed_formatted = format_template.format(
                path=path,
                content=compressed_content,
                root=root,
                category=category,
            )
            compressed_tokens = count_tokens(compressed_formatted)
            
            if total_tokens + compressed_tokens <= effective_budget:
                formatted_parts.append(compressed_formatted)
                total_tokens += compressed_tokens
                truncated = True
            else:
                # それでも超過する場合は以降をスキップ
                omitted_ids.append(path)
                truncated = True
    
    # 省略された結果がある場合は通知を追加
    if omitted_ids:
        notice = (
            f"\n[注意: 上記は検索結果の一部のみです（{len(results)}件中{len(results) - len(omitted_ids)}件を表示）。\n"
            f"特定のセクションの詳細が必要な場合は、get_document_content で該当パスを指定して取得できます。]"
        )
        notice_tokens = count_tokens(notice)
        if total_tokens + notice_tokens <= effective_budget:
            formatted_parts.append(notice)
    
    return '\n'.join(formatted_parts), truncated, omitted_ids


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    テキストを指定トークン数以内に切り詰める。
    文の途中で切らないよう、句読点で区切る。
    
    Args:
        text: 切り詰めるテキスト
        max_tokens: 最大トークン数
    
    Returns:
        切り詰められたテキスト
    """
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    
    # バイナリサーチで切り詰め位置を探す
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    
    # 文の途中で切らないよう、最後の句読点を探す
    truncated = text[:low]
    for punct in '。！？.!?\n':
        last_punct = truncated.rfind(punct)
        if last_punct > len(truncated) * 0.5:  # 半分以上は残す
            truncated = truncated[:last_punct + 1]
            break
    
    return truncated + "\n...[以下省略]..."


def get_document_stats(text: str) -> Dict[str, Any]:
    """ドキュメントの統計情報を取得する。"""
    sections = parse_markdown_sections(text)
    return {
        "total_chars": len(text),
        "total_tokens": count_tokens(text),
        "section_count": len(sections),
        "sections": [
            {
                "heading": s.heading,
                "level": s.level,
                "tokens": s.get_token_count(),
                "chars": len(s.content),
            }
            for s in sections
        ],
    }
