from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from xberg_pipe.cli import build_parser, main
from xberg_pipe.keywording import LIGHT_MODEL
from xberg_pipe.models import DocumentChunk, DocumentKeyword, DocumentSummary


def test_summarize_defaults_to_small_ollama_model() -> None:
    args = build_parser().parse_args(["summarize"])

    assert args.model == "qwen3.5:4b"
    assert args.host == "http://127.0.0.1:11434"
    assert args.context_window == 16_384
    assert args.max_output_tokens == 512
    assert args.think is False


def test_init_db_and_stats(tmp_path: Path, capsys) -> None:
    database = tmp_path / "data" / "test.db"

    assert main(["--database", str(database), "init-db"]) == 0
    assert database.exists()
    assert main(["--database", str(database), "stats"]) == 0

    output = capsys.readouterr().out
    assert "SQLiteを初期化しました" in output
    assert "documents=0" in output
    assert "chunks=0" in output
    assert "keywords=0" in output


def test_keybert_refinement_is_a_separate_command() -> None:
    scan = build_parser().parse_args(["scan", "docs"])
    refine = build_parser().parse_args(["refine-keywords"])

    assert scan.command == "scan"
    assert refine.model


def test_scan_saves_light_keywords_without_keybert(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.txt"
    source.write_text("人工知能の研究です。", encoding="utf-8")
    database = tmp_path / "test.db"

    async def fake_extract_batch(_inputs, _config):
        return SimpleNamespace(
            results=[SimpleNamespace(content="人工知能の研究です。")], errors=[]
        )

    monkeypatch.setattr("xberg_pipe.scan_extractor.extract_batch", fake_extract_batch)
    assert main(["--database", str(database), "scan", str(tmp_path)]) == 0

    with Session(create_engine(f"sqlite:///{database.as_posix()}")) as session:
        assert session.scalar(select(DocumentChunk)) is not None
        keyword = session.scalar(select(DocumentKeyword))
        assert keyword is not None
        assert keyword.model_name == LIGHT_MODEL
        assert session.scalar(select(DocumentSummary)) is not None
