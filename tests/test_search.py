import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xberg_pipe.models import Base, Document, DocumentPath, DocumentText
from xberg_pipe.search import keyword_search


def test_keyword_search_uses_document_text_as_join_source() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        document = Document(id="a" * 64, extension=".txt", status="extracted")
        document.paths.append(
            DocumentPath(
                path="docs/a.txt", file_size=1, modified_at=dt.datetime(2026, 1, 1)
            )
        )
        session.add(
            DocumentText(
                document=document, extractor="xberg", extracted_text="人工知能の文書"
            )
        )
        session.commit()

        hits = keyword_search(session, "人工知能")

    assert len(hits) == 1
    assert hits[0].path == "docs/a.txt"
