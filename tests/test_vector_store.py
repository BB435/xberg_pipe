from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

pytest.importorskip("sqlite_vec")

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.models import Base, Document, DocumentText
from xberg_pipe.vector_store import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbeddingConfig,
    rebuild_embeddings,
    search_embeddings,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 256
            if text.startswith(QUERY_PREFIX) or "人工知能" in text:
                vector[0] = 1.0
            else:
                vector[1] = 1.0
            assert text.startswith((DOCUMENT_PREFIX, QUERY_PREFIX))
            vectors.append(vector)
        return vectors


def test_rebuild_and_search_embeddings_with_sqlite_vec(tmp_path: Path) -> None:
    database = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{database.as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for index, text in enumerate(("人工知能の研究", "料理のレシピ")):
            document = Document(
                id=str(index) * 64, extension=".txt", status="extracted"
            )
            document_text = DocumentText(
                document=document, extractor="xberg", extracted_text=text
            )
            session.add(document_text)
            replace_document_chunks(session, document_text)
        session.commit()

    config = EmbeddingConfig(batch_size=2)
    assert rebuild_embeddings(database, config, FakeEncoder()).chunks == 2
    results = search_embeddings(
        database, "AI", top_k=1, config=config, encoder=FakeEncoder()
    )

    assert len(results) == 1
    assert results[0].content == "人工知能の研究"
    assert results[0].distance == pytest.approx(0.0)
