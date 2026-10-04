from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from xberg_pipe.chunking import replace_document_chunks
from xberg_pipe.models import (
    Base,
    Document,
    DocumentChunk,
    DocumentSummary,
    DocumentText,
)
from xberg_pipe.summarization import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_SUMMARY_MODEL,
    OllamaSummaryClient,
    SummaryConfig,
    rebuild_all_summaries,
    summarize_document,
)


class FakeSummaryClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        assert "原文にない情報" in system
        return f"要約{self.calls}"


class FakeOllamaClient:
    def __init__(self, content: str = " Ollamaの要約 ") -> None:
        self.content = content
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def test_summary_config_defaults_are_tuned_for_small_ollama_models() -> None:
    config = SummaryConfig()

    assert config.model_name == DEFAULT_SUMMARY_MODEL == "qwen3.5:4b"
    assert config.host == DEFAULT_OLLAMA_HOST
    assert config.context_window == DEFAULT_CONTEXT_WINDOW == 16_384
    assert config.max_output_tokens == 512
    assert config.reduce_group_size == 6
    assert config.think is False


def test_ollama_summary_client_passes_generation_options() -> None:
    ollama = FakeOllamaClient()
    config = SummaryConfig()
    client = OllamaSummaryClient(config, client=ollama)

    assert client.complete("システム", "本文") == "Ollamaの要約"
    assert ollama.calls == [
        {
            "model": "qwen3.5:4b",
            "messages": (
                {"role": "system", "content": "システム"},
                {"role": "user", "content": "本文"},
            ),
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 16_384,
                "num_predict": 512,
            },
            "keep_alive": "10m",
        }
    ]


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

        progress = []
        result = rebuild_all_summaries(
            session,
            SummaryConfig(model_name="local/test"),
            client=client,
            progress=lambda current: progress.append(
                (current.documents, current.failed, current.skipped)
            ),
        )

        summary = session.scalar(select(DocumentSummary))
        assert result.documents == 1
        assert result.failed == 0
        assert summary is not None
        assert summary.summary == f"要約{client.calls}"
        assert client.calls >= 3
        assert progress == [(1, 0, 0)]


def test_llm_summary_limits_existing_long_chunks() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def complete(self, _system: str, user: str) -> str:
            self.inputs.append(user)
            return "要約"

    text = DocumentText(extracted_text="本文")
    text.chunks = [
        DocumentChunk(chunk_index=index, content="あ" * 2_000, character_count=2_000)
        for index in range(60)
    ]
    client = RecordingClient()

    assert summarize_document(text, client, SummaryConfig()) == "要約"
    fragments = [item for item in client.inputs if item.startswith("次の文書断片")]
    assert len(fragments) == 50
    assert all(len(item.rsplit("\n\n", 1)[-1]) == 1_200 for item in fragments)
