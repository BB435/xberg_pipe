import warnings
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.keywording import (
    KeywordConfig,
    RuriKeyBertExtractor,
    YakeKeywordExtractor,
    aggregate_chunk_keywords,
    extract_document_keywords,
    rebuild_all_keywords,
    replace_document_keywords,
)
from xberg_pipe.models import Base, Document, DocumentKeyword, DocumentText, Keyword


class FakeExtractor:
    def extract(self, text: str, top_n: int) -> list[tuple[str, float]]:
        if "人工知能" in text:
            return [("人工知能", 0.9), ("機械学習", 0.7)][:top_n]
        return [("自然言語処理", 0.8), ("機械学習", 0.6)][:top_n]


def test_yake_keywords_extracts_japanese_phrases_without_keybert() -> None:
    keywords = YakeKeywordExtractor().extract(
        "人工知能の研究を進めます。人工知能の活用を検討します。", top_n=5
    )

    assert any(word == "人工知能" for word, _ in keywords)
    assert all(0 < score <= 1 for _, score in keywords)


def test_unchunked_document_is_processed_in_bounded_pieces() -> None:
    class RecordingExtractor:
        def __init__(self) -> None:
            self.lengths: list[int] = []

        def extract(self, text: str, top_n: int) -> list[tuple[str, float]]:
            self.lengths.append(len(text))
            return [("人工知能", 0.8)]

    document_text = DocumentText(extracted_text="人工知能について説明します。" * 1000)
    extractor = RecordingExtractor()
    keywords = extract_document_keywords(document_text, extractor, KeywordConfig())

    assert keywords == [("人工知能", 0.84)]
    assert len(extractor.lengths) > 1
    assert max(extractor.lengths) <= 1200


def test_direct_yake_extraction_limits_long_input() -> None:
    extractor = YakeKeywordExtractor()
    seen = []

    class RecordingTokenizer:
        def tokenize(self, text, _mode):
            seen.append(len(text))
            return []

    extractor._tokenizer = RecordingTokenizer()
    assert extractor.extract("あ" * 20_000, top_n=5) == []
    assert seen == [1_200]


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keywords = RuriKeyBertExtractor(KeywordConfig()).extract(
            example.read_text(encoding="utf-8"), top_n=5
        )

    assert keywords
    assert not any(
        "Upper case characters found in vocabulary" in str(w.message) for w in caught
    )
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
        assert (
            rebuild_all_keywords(
                session,
                config=KeywordConfig(model_name="test/model", top_n=3),
                extractor=FakeExtractor(),
            ).documents
            == 0
        )
        assert (
            rebuild_all_keywords(
                session,
                config=KeywordConfig(model_name="test/model", top_n=3),
                extractor=FakeExtractor(),
                all_items=True,
            ).documents
            == 1
        )


def test_replace_keywords_reuses_shared_term() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        texts = [
            DocumentText(
                document=Document(id=str(index) * 64, extension=".txt"),
                extractor="xberg",
                extracted_text="人工知能",
            )
            for index in (1, 2)
        ]
        session.add_all(texts)
        for text in texts:
            replace_document_keywords(session, text, [("人工知能", 0.8)], "test")
        session.commit()

        links = session.scalars(select(DocumentKeyword)).all()
        assert len(links) == 2
        assert links[0].keyword_id == links[1].keyword_id
        assert session.scalars(select(Keyword.value)).all() == ["人工知能"]

        replace_document_keywords(session, texts[0], [("人工知能", 0.9)], "test")
        session.commit()
        assert session.scalars(select(Keyword.value)).all() == ["人工知能"]
        assert len(session.scalars(select(DocumentKeyword)).all()) == 2


def test_replace_keywords_batches_shared_term_queries() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    statements = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_statement)
    with Session(engine) as session:
        text = DocumentText(
            document=Document(id="c" * 64, extension=".txt"),
            extractor="xberg",
            extracted_text="人工知能と機械学習",
        )
        session.add(text)
        replace_document_keywords(
            session,
            text,
            [("人工知能", 0.9), ("機械学習", 0.8), ("自然言語", 0.7)],
            "test",
        )
        session.flush()

        assert sum(sql.startswith("INSERT INTO keywords") for sql in statements) == 1
        assert sum(sql.startswith("SELECT keywords.value") for sql in statements) == 1
        assert len(session.scalars(select(DocumentKeyword)).all()) == 3
