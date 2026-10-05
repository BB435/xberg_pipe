from sqlalchemy import text

from xberg_pipe.db import create_db_engine, init_db
from xberg_pipe.models import Base


def test_create_db_engine_enables_sqlite_foreign_keys(tmp_path) -> None:
    engine = create_db_engine(tmp_path / "test.db")

    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1


def test_init_db_backfills_existing_extracted_text(tmp_path) -> None:
    engine = create_db_engine(tmp_path / "test.db")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE document_texts DROP COLUMN cleaned_text"))
        connection.execute(
            text(
                "INSERT INTO documents (id, extension, status) VALUES ('old', '.txt', 'extracted')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_texts (document_id, extractor, extracted_text) VALUES ('old', 'xberg', '日 本 語')"
            )
        )

    init_db(engine)
    with engine.connect() as connection:
        assert (
            connection.scalar(text("SELECT cleaned_text FROM document_texts"))
            == "日本語"
        )
