from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xberg_pipe.models import Base

DB_PATH = "data/app.db"

engine = create_engine(f"sqlite:///{Path(DB_PATH).as_posix()}")

LocalSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    Base.metadata.create_all(engine)
