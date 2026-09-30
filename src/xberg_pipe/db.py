from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from xberg_pipe.models import Base

DB_PATH = "data/app.db"


def create_db_engine(db_path: str | Path = DB_PATH) -> Engine:
    """SQLiteエンジンを作成し、保存先ディレクトリも用意する."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path.as_posix()}")


engine = create_db_engine()

LocalSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(db_engine: Engine = engine) -> None:
    Base.metadata.create_all(db_engine)
