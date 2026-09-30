from pathlib import Path

from xberg_pipe.cli import main


def test_init_db_and_stats(tmp_path: Path, capsys) -> None:
    database = tmp_path / "data" / "test.db"

    assert main(["--database", str(database), "init-db"]) == 0
    assert database.exists()
    assert main(["--database", str(database), "stats"]) == 0

    output = capsys.readouterr().out
    assert "SQLiteを初期化しました" in output
    assert "documents=0" in output
    assert "chunks=0" in output


def test_chunk_empty_database(tmp_path: Path, capsys) -> None:
    database = tmp_path / "test.db"

    assert main(["--database", str(database), "chunk"]) == 0

    assert "処理文書=0 生成チャンク=0" in capsys.readouterr().out
