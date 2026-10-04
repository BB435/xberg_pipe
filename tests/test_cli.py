from pathlib import Path

from xberg_pipe.cli import build_parser, main


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


def test_scan_includes_postprocessing_options() -> None:
    args = build_parser().parse_args(["scan", "docs"])

    assert args.keyword_model
    assert args.device is None
