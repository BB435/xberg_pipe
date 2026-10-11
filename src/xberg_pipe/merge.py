from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from xberg_pipe.db import create_db_engine, init_db
from xberg_pipe.keywording import replace_document_keywords
from xberg_pipe.models import (
    Document,
    DocumentChunk,
    DocumentKeyword,
    DocumentPath,
    DocumentSummary,
    DocumentText,
)


def merge_database(target_path: str | Path, source_paths: list[str | Path]) -> int:
    """独立 DB の文書と派生情報を文書ハッシュで統合する."""

    target_engine = create_db_engine(target_path)
    init_db(target_engine)
    target_resolved = Path(target_path).resolve()
    merged = 0
    try:
        for source_path in source_paths:
            source_resolved = Path(source_path).resolve()
            if source_resolved == target_resolved:
                continue
            if not source_resolved.is_file():
                raise FileNotFoundError(f"統合元DBが見つかりません: {source_resolved}")
            source_engine = create_db_engine(source_path)
            init_db(source_engine)
            try:
                with Session(source_engine) as source, Session(target_engine) as target:
                    for source_doc in source.scalars(select(Document)):
                        target_doc = target.get(Document, source_doc.id)
                        if target_doc is None:
                            target_doc = Document(
                                id=source_doc.id,
                                extension=source_doc.extension,
                                status=source_doc.status,
                            )
                            target.add(target_doc)
                            target.flush()
                        for source_path_row in source.scalars(
                            select(DocumentPath).where(
                                DocumentPath.document_id == source_doc.id
                            )
                        ):
                            path_row = target.scalar(
                                select(DocumentPath).where(
                                    DocumentPath.path == source_path_row.path
                                )
                            )
                            if path_row is None:
                                target.add(
                                    DocumentPath(
                                        document=target_doc,
                                        path=source_path_row.path,
                                        file_size=source_path_row.file_size,
                                        modified_at=source_path_row.modified_at,
                                    )
                                )
                        for source_text in source.scalars(
                            select(DocumentText).where(
                                DocumentText.document_id == source_doc.id
                            )
                        ):
                            target_text = target.scalar(
                                select(DocumentText).where(
                                    DocumentText.document_id == source_doc.id,
                                    DocumentText.extractor == source_text.extractor,
                                )
                            )
                            if target_text is None:
                                target_text = DocumentText(
                                    document=target_doc,
                                    extractor=source_text.extractor,
                                    extracted_text=source_text.extracted_text,
                                    cleaned_text=source_text.cleaned_text,
                                    content_hash=source_text.content_hash,
                                    light_keyword_count=source_text.light_keyword_count,
                                )
                                target.add(target_text)
                                target.flush()
                            source_chunks = source.scalars(
                                select(DocumentChunk).where(
                                    DocumentChunk.document_text_id == source_text.id
                                )
                            ).all()
                            for source_chunk in source_chunks:
                                chunk = target.scalar(
                                    select(DocumentChunk).where(
                                        DocumentChunk.document_text_id
                                        == target_text.id,
                                        DocumentChunk.chunk_index
                                        == source_chunk.chunk_index,
                                    )
                                )
                                if chunk is None:
                                    chunk = target.scalar(
                                        select(DocumentChunk).where(
                                            DocumentChunk.document_text_id
                                            == target_text.id,
                                            DocumentChunk.content_hash
                                            == source_chunk.content_hash,
                                        )
                                    )
                                if chunk is None:
                                    chunk = DocumentChunk(
                                        document_text=target_text,
                                        chunk_index=source_chunk.chunk_index,
                                        content=source_chunk.content,
                                        content_hash=source_chunk.content_hash,
                                        character_count=source_chunk.character_count,
                                        chunker_version=source_chunk.chunker_version,
                                    )
                                    target.add(chunk)
                                    target.flush()
                                # Embedding metadata is portable by content identity;
                                # vectors are regenerated locally by `embed`.
                            grouped: dict[str, list[tuple[str, float, int]]] = {}
                            source_links = source.scalars(
                                select(DocumentKeyword)
                                .options(selectinload(DocumentKeyword.term))
                                .where(
                                    DocumentKeyword.document_text_id == source_text.id
                                )
                            ).all()
                            for link in source_links:
                                grouped.setdefault(link.model_name, []).append(
                                    (link.keyword, link.score, link.rank)
                                )
                            for model, terms in grouped.items():
                                if (
                                    target.scalar(
                                        select(DocumentKeyword.id).where(
                                            DocumentKeyword.document_text_id
                                            == target_text.id,
                                            DocumentKeyword.model_name == model,
                                        )
                                    )
                                    is None
                                ):
                                    replace_document_keywords(
                                        target,
                                        target_text,
                                        [
                                            (term, score)
                                            for term, score, _rank in sorted(
                                                terms, key=lambda item: item[2]
                                            )
                                        ],
                                        model,
                                    )
                            for source_summary in source.scalars(
                                select(DocumentSummary).where(
                                    DocumentSummary.document_text_id == source_text.id
                                )
                            ):
                                existing = target.scalar(
                                    select(DocumentSummary).where(
                                        DocumentSummary.work_key
                                        == source_summary.work_key
                                    )
                                )
                                if existing is None:
                                    target.add(
                                        DocumentSummary(
                                            document_text=target_text,
                                            model_name=source_summary.model_name,
                                            endpoint=source_summary.endpoint,
                                            prompt_version=source_summary.prompt_version,
                                            summary=source_summary.summary,
                                            work_key=source_summary.work_key,
                                        )
                                    )
                        target.commit()
                        merged += 1
            finally:
                source_engine.dispose()
    finally:
        target_engine.dispose()
    return merged
