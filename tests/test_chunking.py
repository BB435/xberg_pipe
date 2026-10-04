from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import (
    MAX_DOWNSTREAM_CHUNKS,
    MAX_PROCESSING_CHARS,
    ChunkingConfig,
    chunk_text,
    clean_extracted_text,
    replace_document_chunks,
)
from xberg_pipe.models import Base, Document, DocumentChunk, DocumentText


def test_clean_extracted_text_normalizes_common_extraction_noise() -> None:
    source = "ＡＢＣ\r\n改行前のinter-\n national\x00\n\n\n次の段落　"

    assert clean_extracted_text(source) == "ABC\n改行前のinternational\n\n次の段落"


def test_chunk_text_prefers_japanese_sentence_boundaries_and_overlaps() -> None:
    source = "第一文です。第二文も重要です。第三文で詳しく説明します。第四文です。"
    chunks = chunk_text(source, ChunkingConfig(target_chars=25, overlap_chars=10))

    assert len(chunks) >= 2
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert chunks[0].endswith("。")
    assert any(
        sentence in chunks[1]
        for sentence in ("第二文も重要です。", "第三文で詳しく説明します。")
    )


def test_chunk_text_limits_long_input() -> None:
    chunks = chunk_text("あ" * MAX_PROCESSING_CHARS + "末尾は対象外です。")

    assert chunks
    assert "末尾" not in "".join(chunks)
    assert all(len(chunk) <= 1_200 for chunk in chunks)


def test_chunk_text_limits_number_of_chunks() -> None:
    chunks = chunk_text(
        "長い文。" * 1_000, ChunkingConfig(target_chars=10, overlap_chars=2)
    )

    assert len(chunks) == MAX_DOWNSTREAM_CHUNKS


def test_replace_document_chunks_replaces_old_rows() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        document = Document(id="a" * 64, extension=".txt", status="extracted")
        document_text = DocumentText(
            document=document,
            extractor="xberg",
            extracted_text="最初の文です。次の文です。最後の文です。",
        )
        session.add(document_text)
        replace_document_chunks(
            session,
            document_text,
            ChunkingConfig(target_chars=15, overlap_chars=5),
        )
        session.commit()

        document_text.extracted_text = "置き換え後の短い文章です。"
        assert replace_document_chunks(session, document_text) == 1
        session.commit()

        chunks = session.scalars(select(DocumentChunk)).all()
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].content == "置き換え後の短い文章です。"
