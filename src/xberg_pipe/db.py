from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from xberg_pipe.models import Base

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
