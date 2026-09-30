from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.models import Base, Document, DocumentSummary, DocumentText
from xberg_pipe.summarization import SummaryConfig, rebuild_all_summaries


class FakeSummaryClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        assert "原文にない情報" in system
        return f"要約{self.calls}"


def test_rebuild_all_summaries_saves_reduced_summary(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    client = FakeSummaryClient()

    with Session(engine) as session:
        document = Document(id="c" * 64, extension=".txt", status="extracted")
        document_text = DocumentText(
            document=document,
            extractor="xberg",
            extracted_text="最初の内容です。" * 100 + "\n\n" + "次の内容です。" * 100,
        )
        session.add(document_text)
        replace_document_chunks(session, document_text)
        session.commit()

        result = rebuild_all_summaries(
            session,
            SummaryConfig(model_name="local/test"),
            client=client,
        )

        summary = session.scalar(select(DocumentSummary))
        assert result.documents == 1
        assert result.failed == 0
        assert summary is not None
        assert summary.summary == f"要約{client.calls}"
        assert client.calls >= 3
