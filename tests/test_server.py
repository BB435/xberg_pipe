import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

pytest.importorskip("fastapi")
pytest.importorskip("sqlite_vec")
from fastapi.testclient import TestClient

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.keywording import DEFAULT_MODEL, LIGHT_MODEL
from xberg_pipe.models import (
    Base,
    Document,
    DocumentKeyword,
    DocumentPath,
    DocumentText,
)
from xberg_pipe.server import create_app
from xberg_pipe.vector_store import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbeddingConfig,
    rebuild_embeddings,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 256
            vector[0 if text.startswith(QUERY_PREFIX) or "人工知能" in text else 1] = (
                1.0
            )
            assert text.startswith((DOCUMENT_PREFIX, QUERY_PREFIX))
            vectors.append(vector)
        return vectors


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{database.as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        document = Document(id="a" * 64, extension=".txt", status="extracted")
        document.paths.append(
            DocumentPath(
                path="docs/ai.txt", file_size=10, modified_at=dt.datetime(2026, 1, 1)
            )
        )
        text = DocumentText(
            document=document,
            extractor="xberg",
            extracted_text="  人工知能の\n原文です。  続き",
        )
        session.add(text)
        replace_document_chunks(session, text)
        text.keywords.append(
            DocumentKeyword(model_name=LIGHT_MODEL, rank=1, keyword="AI", score=0.9)
        )
        session.commit()
    rebuild_embeddings(database, EmbeddingConfig(), FakeEncoder())
    return database


def test_vector_search_returns_original_text_preview(tmp_path: Path) -> None:
    client = TestClient(
        create_app(_database(tmp_path), preview_chars=8, encoder=FakeEncoder())
    )

    response = client.get("/api/search/vector", params={"q": "AI", "top_k": 1})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["path"] == "docs/ai.txt"
    assert result["preview"] == "人工知能の 原文"


def test_keyword_search_returns_keyword_and_preview(tmp_path: Path) -> None:
    client = TestClient(
        create_app(_database(tmp_path), preview_chars=100, encoder=FakeEncoder())
    )

    response = client.get("/api/search/keyword", params={"q": "AI"})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["keywords"] == ["AI"]
    assert result["preview"] == "人工知能の 原文です。 続き"


def test_keyword_search_prefers_refined_keywords(tmp_path: Path) -> None:
    database = _database(tmp_path)
    engine = create_engine(f"sqlite:///{database.as_posix()}")
    with Session(engine) as session:
        text = session.query(DocumentText).one()
        text.keywords.append(
            DocumentKeyword(
                model_name=DEFAULT_MODEL, rank=1, keyword="精密AI", score=0.8
            )
        )
        session.commit()
    client = TestClient(create_app(database, encoder=FakeEncoder()))

    response = client.get("/api/search/keyword", params={"q": "AI"})

    assert response.status_code == 200
    assert response.json()["results"][0]["keywords"] == ["精密AI"]


def test_search_page_is_available(tmp_path: Path) -> None:
    response = TestClient(create_app(_database(tmp_path), encoder=FakeEncoder())).get(
        "/"
    )

    assert response.status_code == 200
    assert "ベクトル検索" in response.text
