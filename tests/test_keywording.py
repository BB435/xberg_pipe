from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.keywording import (
    KeywordConfig,
    RuriKeyBertExtractor,
    aggregate_chunk_keywords,
    rebuild_all_keywords,
)
from xberg_pipe.models import Base, Document, DocumentKeyword, DocumentText


class FakeExtractor:
    def extract(self, text: str, top_n: int) -> list[tuple[str, float]]:
        if "人工知能" in text:
            return [("人工知能", 0.9), ("機械学習", 0.7)][:top_n]
        return [("自然言語処理", 0.8), ("機械学習", 0.6)][:top_n]


def test_keybert_accepts_fastembed_vectors_from_example_file(monkeypatch) -> None:
    class FakeModel:
        def embed(self, texts):
            for text in texts:
                vector = np.zeros(256, dtype=np.float32)
                vector[0] = len(text)
                vector[1] = 1
                yield vector

    monkeypatch.setattr(
        "xberg_pipe.fastembed_model.create_ruri_model",
        lambda model_name, device: FakeModel(),
    )
    example = Path(__file__).resolve().parents[1] / "example-docs" / "fake-text.txt"
    keywords = RuriKeyBertExtractor(KeywordConfig()).extract(
        example.read_text(encoding="utf-8"), top_n=5
    )

    assert keywords
    assert all(
        isinstance(keyword, str) and isinstance(score, float)
        for keyword, score in keywords
    )


def test_aggregate_chunk_keywords_rewards_document_wide_terms() -> None:
    keywords = aggregate_chunk_keywords(
        [
            [("人工知能", 0.8), ("機械学習", 0.7)],
            [("機械学習", 0.7), ("自然言語処理", 0.6)],
        ],
        top_n=2,
    )

    assert keywords == [("人工知能", 0.8), ("機械学習", 0.71)]


def test_rebuild_all_keywords_saves_file_level_result(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        document = Document(id="b" * 64, extension=".txt", status="extracted")
        document_text = DocumentText(
            document=document,
            extractor="xberg",
            extracted_text=("人工知能と機械学習について説明します。" * 80),
        )
        session.add(document_text)
        replace_document_chunks(session, document_text)
        session.commit()

        progress = []
        result = rebuild_all_keywords(
            session,
            config=KeywordConfig(model_name="test/model", top_n=3),
            extractor=FakeExtractor(),
            progress=lambda current: progress.append(
                (current.documents, current.keywords, current.skipped)
            ),
        )

        stored = session.scalars(
            select(DocumentKeyword).order_by(DocumentKeyword.rank)
        ).all()
        assert result.documents == 1
        assert result.keywords == 2
        assert [keyword.keyword for keyword in stored] == ["人工知能", "機械学習"]
        assert all(keyword.model_name == "test/model" for keyword in stored)
        assert progress == [(1, 2, 0)]
