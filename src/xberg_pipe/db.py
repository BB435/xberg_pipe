import hashlib
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
    # 複数の処理プロセスが同じDBを使う場合の短い一時ロックを待つ。
    db_engine = create_engine(
        f"sqlite:///{path.as_posix()}", connect_args={"timeout": 30}
    )

    @event.listens_for(db_engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA busy_timeout = 30000")
        finally:
            cursor.close()

    return db_engine


def init_db(db_engine: Engine) -> None:
    Base.metadata.create_all(db_engine)
    initial_inspector = inspect(db_engine)
    initial_columns = {
        table: {column["name"] for column in initial_inspector.get_columns(table)}
        for table in ("document_texts", "document_chunks", "document_summaries")
    }
    initial_text_columns = initial_columns["document_texts"]
    missing_migration_columns = {
        table: {name for name in additions if name not in initial_columns[table]}
        for table, additions in {
            "document_texts": {"content_hash": "VARCHAR(64)"},
            "document_chunks": {"content_hash": "VARCHAR(64)"},
            "document_summaries": {"work_key": "VARCHAR(64)"},
        }.items()
    }
    migrations = {
        "document_texts": {"content_hash": "VARCHAR(64)"},
        "document_chunks": {"content_hash": "VARCHAR(64)"},
        "document_summaries": {"work_key": "VARCHAR(64)"},
    }
    with db_engine.begin() as connection:
        inspector = inspect(db_engine)
        for table, additions in migrations.items():
            columns = {column["name"] for column in inspector.get_columns(table)}
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                    )
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
    needs_backfill = any(missing_migration_columns.values()) or any(
        name not in initial_text_columns
        for name in ("cleaned_text", "light_keyword_count")
    )
    if not needs_backfill:
        return

    from xberg_pipe.chunking import clean_extracted_text
    from xberg_pipe.repository import delete_document_derivatives

    with Session(db_engine) as session:
        from xberg_pipe.models import DocumentChunk, DocumentSummary

        if "content_hash" in missing_migration_columns["document_texts"]:
            for item in session.scalars(
                select(DocumentText).execution_options(yield_per=500)
            ):
                item.content_hash = hashlib.sha256(
                    item.extracted_text.encode("utf-8")
                ).hexdigest()
        if "content_hash" in missing_migration_columns["document_chunks"]:
            for chunk in session.scalars(
                select(DocumentChunk).execution_options(yield_per=500)
            ):
                chunk.content_hash = hashlib.sha256(
                    chunk.content.encode("utf-8")
                ).hexdigest()
        if "work_key" in missing_migration_columns["document_summaries"]:
            for summary in session.scalars(
                select(DocumentSummary).execution_options(yield_per=500)
            ):
                summary.work_key = hashlib.sha256(
                    f"{summary.document_text.document_id}:{summary.model_name}:"
                    f"{summary.prompt_version}".encode()
                ).hexdigest()
        for document_text in session.scalars(
            select(DocumentText).where(DocumentText.cleaned_text.is_(None))
        ):
            cleaned = clean_extracted_text(document_text.extracted_text)
            if cleaned != document_text.extracted_text:
                delete_document_derivatives(session, document_text)
            document_text.cleaned_text = cleaned
        session.commit()
