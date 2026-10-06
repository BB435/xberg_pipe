from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from xberg_pipe.models import Base, DocumentText

DB_PATH = "data/app.db"


def create_db_engine(db_path: str | Path = DB_PATH) -> Engine:
    """SQLiteエンジンを作成し、保存先ディレクトリも用意する."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db_engine = create_engine(f"sqlite:///{path.as_posix()}")

    @event.listens_for(db_engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()

    return db_engine


def init_db(db_engine: Engine) -> None:
    Base.metadata.create_all(db_engine)
    columns = {
        column["name"] for column in inspect(db_engine).get_columns("document_texts")
    }
    if "cleaned_text" not in columns:
        with db_engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE document_texts ADD COLUMN cleaned_text TEXT")
            )
    if "light_keyword_count" not in columns:
        with db_engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE document_texts ADD COLUMN light_keyword_count INTEGER"
                )
            )
    from xberg_pipe.chunking import clean_extracted_text
    from xberg_pipe.repository import delete_document_derivatives

    with Session(db_engine) as session:
        for document_text in session.scalars(
            select(DocumentText).where(DocumentText.cleaned_text.is_(None))
        ):
            cleaned = clean_extracted_text(document_text.extracted_text)
            if cleaned != document_text.extracted_text:
                delete_document_derivatives(session, document_text)
            document_text.cleaned_text = cleaned
        session.commit()
