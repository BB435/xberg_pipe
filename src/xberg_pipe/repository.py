from sqlalchemy import select
from sqlalchemy.orm import Session

from xberg_pipe.models import Document, DocumentPath


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
