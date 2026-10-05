import asyncio
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.keywording import replace_document_keywords
from xberg_pipe.models import (
    Base,
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentKeyword,
    DocumentSummary,
    DocumentText,
)
from xberg_pipe.provisional_summary import (
    replace_provisional_summary,
    summarize_extractively,
)
from xberg_pipe.scan_extractor import ScanExtractor


def test_scan_saves_postprocessing_with_extraction(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "document.txt"
    source.write_text("重要な結論です。", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    async def fake_extract_batch(_inputs, _config):
        return SimpleNamespace(
            results=[SimpleNamespace(content="重要な結論です。")], errors=[]
        )

    monkeypatch.setattr("xberg_pipe.scan_extractor.extract_batch", fake_extract_batch)

    def postprocess(session, document_text):
        replace_document_chunks(session, document_text)
        keywords = [("結論", 0.9)]
        replace_document_keywords(session, document_text, keywords, "test")
        replace_provisional_summary(
            session,
            document_text,
            summarize_extractively(document_text.extracted_text, keywords),
        )

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path, postprocess=postprocess)
        assert asyncio.run(extractor.scan()).extracted == 1
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 1
        assert session.scalar(select(DocumentKeyword)).keyword == "結論"
        assert session.scalar(select(DocumentSummary)).summary == "重要な結論です。"
        assert asyncio.run(extractor.scan()).skipped == 1


def test_save_persists_original_and_cleaned_text(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("抽出対象の本文です。", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._save(source, "抽 出 さ れ た\n本文です。")
        session.commit()

        document_text = session.scalar(select(DocumentText))
        chunk_count = session.scalar(select(func.count()).select_from(DocumentChunk))
        assert document_text is not None
        assert document_text.extracted_text == "抽 出 さ れ た\n本文です。"
        assert document_text.cleaned_text == "抽出された本文です。"
        assert chunk_count == 0

        replace_document_chunks(session, document_text)
        session.commit()
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 1
        assert session.scalar(select(DocumentChunk.content)) == "抽出された本文です。"


def test_custom_postprocess_does_not_require_a_summary(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("本文", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    calls = 0

    async def fake_extract_batch(_inputs, _config):
        nonlocal calls
        calls += 1
        return SimpleNamespace(results=[SimpleNamespace(content="本文")], errors=[])

    monkeypatch.setattr("xberg_pipe.scan_extractor.extract_batch", fake_extract_batch)
    with Session(engine) as session:
        extractor = ScanExtractor(
            session,
            tmp_path,
            postprocess=lambda current, text: replace_document_chunks(current, text),
        )
        assert asyncio.run(extractor.scan()).extracted == 1
        assert asyncio.run(extractor.scan()).skipped == 1
        assert calls == 1


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
        session.flush()
        chunk = session.scalar(select(DocumentChunk))
        assert chunk is not None
        document_text.keywords.append(
            DocumentKeyword(model_name="test", rank=1, keyword="旧", score=1.0)
        )
        document_text.summaries.append(
            DocumentSummary(
                model_name="test",
                endpoint="local",
                prompt_version="test",
                summary="古い要約",
            )
        )
        chunk.embeddings.append(ChunkEmbedding(model_name="test", dimensions=1))
        session.commit()

        extractor._save(source, "更新された抽出結果です。")
        session.commit()

        for model in (
            DocumentChunk,
            DocumentKeyword,
            DocumentSummary,
            ChunkEmbedding,
        ):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_failed_file_is_retried_even_when_metadata_is_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("再試行する内容", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._mark_failed(source)

        assert extractor._is_unchanged(source) is False


def test_failure_does_not_hide_existing_text_for_same_content(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("同一内容", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._save(source, "保存済みの抽出結果")
        session.commit()

        extractor._mark_failed(source)

        document = session.scalar(select(Document))
        assert document is not None
        assert document.status == "extracted"
        assert extractor._is_unchanged(source) is True


def test_unchanged_paths_reads_batch_in_one_query(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    paths = [tmp_path / f"document-{index}.txt" for index in range(3)]
    for path in paths:
        path.write_text("保存済み", encoding="utf-8")

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        for path in paths:
            extractor._save(path, "抽出済み")
        session.commit()

        queries: list[str] = []

        def count_query(_conn, _cursor, statement, _parameters, _context, _many):
            queries.append(statement)

        event.listen(engine, "before_cursor_execute", count_query)
        try:
            unchanged = extractor._unchanged_paths(paths)
        finally:
            event.remove(engine, "before_cursor_execute", count_query)

        assert unchanged == {str(path.resolve()) for path in paths}
        assert len(queries) == 1

        paths[0].write_text("更新後", encoding="utf-8")
        assert str(paths[0].resolve()) not in extractor._unchanged_paths(paths)


def test_changed_file_removes_orphaned_previous_document(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("変更前", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._save(source, "変更前の抽出結果")
        session.flush()
        previous_text = session.scalar(select(DocumentText))
        assert previous_text is not None
        replace_document_chunks(session, previous_text)
        session.commit()

        source.write_text("変更後のファイル", encoding="utf-8")
        extractor._save(source, "変更後の抽出結果")
        session.commit()

        assert session.scalar(select(func.count()).select_from(Document)) == 1
        assert session.scalar(select(func.count()).select_from(DocumentText)) == 1
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0
        assert session.scalar(select(DocumentText.extracted_text)) == "変更後の抽出結果"


def test_scan_removes_deleted_file_and_orphaned_document(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("削除対象", encoding="utf-8")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        extractor = ScanExtractor(session, tmp_path)
        extractor._save(source, "削除対象の抽出結果")
        session.commit()
        source.unlink()

        asyncio.run(extractor.scan())

        assert session.scalar(select(func.count()).select_from(Document)) == 0
        assert session.scalar(select(func.count()).select_from(DocumentText)) == 0
