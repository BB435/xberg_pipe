from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from xberg_pipe.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentKeyword,
    DocumentPath,
    DocumentSummary,
    DocumentText,
)


def get_or_create_document(session: Session, sha256: str, extension: str) -> Document:
    document = session.get(Document, sha256)
    if document is not None:
        return document

    document = Document(id=sha256, extension=extension, status="hash")
    session.add(document)
    return document


def get_document_path(session: Session, path: str) -> DocumentPath | None:
    stmt = select(DocumentPath).where(DocumentPath.path == path)
    return session.scalar(stmt)


def get_document_text(document: Document, extractor: str) -> DocumentText | None:
    """文書に紐づく指定抽出器のテキストを返す."""

    return next((text for text in document.texts if text.extractor == extractor), None)


def delete_document_derivatives(session: Session, document_text: DocumentText) -> None:
    """抽出テキストから生成された全派生データを削除する."""

    session.flush()
    chunk_ids: Sequence[int] = tuple(
        session.scalars(
            select(DocumentChunk.id).where(
                DocumentChunk.document_text_id == document_text.id
            )
        )
    )
    if chunk_ids:
        # sqlite-vecの仮想表には外部キー制約がないため、実体も明示的に消す。
        from xberg_pipe.vector_store import delete_chunk_vectors

        delete_chunk_vectors(session.connection(), chunk_ids)
        session.execute(
            delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))
        )

    session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_text_id == document_text.id)
    )
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
