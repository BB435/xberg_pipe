from sqlalchemy import text

from xberg_pipe.db import create_db_engine


def test_create_db_engine_enables_sqlite_foreign_keys(tmp_path) -> None:
    engine = create_db_engine(tmp_path / "test.db")

    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1
