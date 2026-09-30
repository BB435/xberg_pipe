from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.models import Base, DocumentChunk, DocumentText
from xberg_pipe.scan_extractor import ScanExtractor


def test_save_only_persists_extracted_text(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("抽出対象の本文です。", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._save(source, "抽出された本文です。")
        session.commit()

        document_text = session.scalar(select(DocumentText))
        chunk_count = session.scalar(select(func.count()).select_from(DocumentChunk))
        assert document_text is not None
        assert document_text.extracted_text == "抽出された本文です。"
        assert chunk_count == 0

        replace_document_chunks(session, document_text)
        session.commit()
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 1


def test_changed_extracted_text_invalidates_chunks(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("同じファイル内容", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._save(source, "最初の抽出結果です。")
        session.flush()
        document_text = session.scalar(select(DocumentText))
        assert document_text is not None
        replace_document_chunks(session, document_text)
        session.commit()

        extractor._save(source, "更新された抽出結果です。")
        session.commit()

        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0
