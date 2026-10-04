from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from xberg_pipe.models import Base, Document, DocumentSummary, DocumentText
from xberg_pipe.provisional_summary import (
    MODEL_NAME,
    replace_provisional_summary,
    summarize_extractively,
)


def test_extractive_summary_uses_original_sentences_in_original_order() -> None:
    text = "背景を説明します。重要な結論は採用です。補足です。採用の条件は予算です。"

    summary = summarize_extractively(text, [("採用", 1.0)], max_chars=34)

    assert "重要な結論は採用です。" in summary
    assert "採用の条件は予算です。" in summary
    assert summary.index("重要な結論") < summary.index("採用の条件")


def test_provisional_summary_replaces_only_its_own_row(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        text = DocumentText(
            document=Document(id="a" * 64, extension=".txt", status="extracted"),
            extractor="xberg",
            extracted_text="本文",
        )
        session.add(text)
        session.add(
            DocumentSummary(
                document_text=text,
                model_name="llm",
                endpoint="local",
                prompt_version="test",
                summary="LLM要約",
            )
        )
        replace_provisional_summary(session, text, "暫定1")
        session.commit()
        replace_provisional_summary(session, text, "暫定2")
        session.commit()

        summaries = session.scalars(select(DocumentSummary)).all()
        assert {(item.model_name, item.summary) for item in summaries} == {
            (MODEL_NAME, "暫定2"),
            ("llm", "LLM要約"),
        }
