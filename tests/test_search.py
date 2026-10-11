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


def test_keyword_operators_and_directory_boundary() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for index, (path, body) in enumerate(
            [
                ("docs/a.txt", "人工知能 研究"),
                ("docs/b.txt", "人工知能 広告"),
                ("docs2/c.txt", "人工知能 研究"),
            ]
        ):
            document = Document(id=str(index) * 64, extension=".txt")
            document.paths.append(
                DocumentPath(
                    path=path, file_size=1, modified_at=dt.datetime(2026, 1, 1)
                )
            )
            session.add(
                DocumentText(document=document, extractor="xberg", extracted_text=body)
            )
        session.commit()

        assert [
            hit.path
            for hit in keyword_search(session, "人工知能 研究 -広告", directory="docs")
        ] == ["docs/a.txt"]
        assert {
            hit.path
            for hit in keyword_search(
                session, "研究 広告", operator="or", directory="docs"
            )
        } == {"docs/a.txt", "docs/b.txt"}
        assert [
            hit.path
            for hit in keyword_search(session, '"人工知能 研究"', directory="docs")
        ] == ["docs/a.txt"]
