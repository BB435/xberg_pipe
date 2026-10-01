import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from xberg_pipe.models import (
    ChunkEmbedding,
    DocumentChunk,
    DocumentKeyword,
    DocumentSummary,
    DocumentText,
)

CHUNKER_VERSION = "ja-v1"
DEFAULT_TARGET_CHARS = 1_200
DEFAULT_OVERLAP_CHARS = 200

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n|$)")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_LATIN_LINE_HYPHEN = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])")
_SENTENCE = re.compile(r".*?(?:[。！？!?]+[」』】）》〕〉］）”’]*|$)", re.DOTALL)
_PREFERRED_BREAKS = ("\n", "。", "！", "？", "、", "，", ",", " ")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """日本語文書向けの文字数ベース設定."""

    target_chars: int = DEFAULT_TARGET_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS

    def __post_init__(self) -> None:
        if self.target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if not 0 <= self.overlap_chars < self.target_chars:
            raise ValueError("overlap_chars must be between 0 and target_chars")


@dataclass(slots=True)
class ChunkingResult:
    documents: int = 0
    chunks: int = 0


def clean_extracted_text(text: str) -> str:
    """原文を残したまま、検索・LLM入力を阻害するノイズを除去する."""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = _CONTROL_CHARS.sub("", text)
    text = _LATIN_LINE_HYPHEN.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _split_long_unit(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        search_start = max(limit // 2, 1)
        cut = max(
            remaining.rfind(mark, search_start, limit + 1) for mark in _PREFERRED_BREAKS
        )
        cut = cut + 1 if cut >= search_start else limit
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining.strip())
    return parts


def _semantic_units(text: str, limit: int) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [match.group().strip() for match in _SENTENCE.finditer(paragraph)]
        sentences = [sentence for sentence in sentences if sentence]
        for sentence in sentences or [paragraph]:
            units.extend(_split_long_unit(sentence, limit))
        # 段落境界を保持し、要約時に異なる話題が連結して見えないようにする。
        if units:
            units[-1] += "\n\n"
    return units


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """日本語の文・段落境界を優先し、重複を持つチャンクへ分割する."""

    config = config or ChunkingConfig()
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []

    units = _semantic_units(cleaned, config.target_chars)
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        if current and len("".join(current)) + len(unit) > config.target_chars:
            chunks.append("".join(current).strip())
            overlap: list[str] = []
            overlap_size = 0
            overlap_budget = min(config.overlap_chars, config.target_chars - len(unit))
            for previous in reversed(current):
                if overlap_size + len(previous) > overlap_budget:
                    break
                overlap.insert(0, previous)
                overlap_size += len(previous)
            current = overlap
        current.append(unit)

    if current:
        final_chunk = "".join(current).strip()
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)
    return chunks


def replace_document_chunks(
    session: Session,
    document_text: DocumentText,
    config: ChunkingConfig | None = None,
) -> int:
    """指定した抽出テキストの既存チャンクを置き換える."""

    session.flush()
    clear_document_derivatives(session, document_text)
    chunks = chunk_text(document_text.extracted_text, config)
    session.add_all(
        DocumentChunk(
            document_text=document_text,
            chunk_index=index,
            content=content,
            character_count=len(content),
            chunker_version=CHUNKER_VERSION,
        )
        for index, content in enumerate(chunks)
    )
    return len(chunks)


def clear_document_derivatives(session: Session, document_text: DocumentText) -> None:
    """抽出テキストから生成された派生データを削除する."""

    session.flush()
    old_chunk_ids = list(
        session.scalars(
            select(DocumentChunk.id).where(
                DocumentChunk.document_text_id == document_text.id
            )
        )
    )
    if old_chunk_ids:
        # sqlite-vecの仮想表には外部キー制約がないため、実体も明示的に消す。
        from xberg_pipe.vector_store import delete_chunk_vectors

        delete_chunk_vectors(session.connection(), old_chunk_ids)
    session.execute(
        delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(old_chunk_ids))
    )
    session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_text_id == document_text.id)
    )
    # 本文が変わった場合、以前のキーワードを残さない。
    session.execute(
        delete(DocumentKeyword).where(
            DocumentKeyword.document_text_id == document_text.id
        )
    )
    session.execute(
        delete(DocumentSummary).where(
            DocumentSummary.document_text_id == document_text.id
        )
    )


def rebuild_all_chunks(
    session: Session,
    config: ChunkingConfig | None = None,
    batch_size: int = 100,
) -> ChunkingResult:
    """SQLite内の全抽出テキストを再チャンキングする."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    result = ChunkingResult()
    document_texts = session.scalars(
        select(DocumentText)
        .order_by(DocumentText.id)
        .execution_options(yield_per=batch_size)
    )
    for document_text in document_texts:
        result.chunks += replace_document_chunks(session, document_text, config)
        result.documents += 1
        if result.documents % batch_size == 0:
            session.commit()
    session.commit()
    return result
