import sqlite3
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy.engine import Connection

from xberg_pipe.keywording import DEFAULT_MODEL

DOCUMENT_PREFIX = "検索文書: "
QUERY_PREFIX = "検索クエリ: "
MODEL_DIMENSIONS = {
    DEFAULT_MODEL: 256,
}
MODEL_TABLES = {
    DEFAULT_MODEL: "vec_ruri_v3_30m_onnx",
}


def delete_chunk_vectors(connection: Connection, chunk_ids: Sequence[int]) -> None:
    """SQLAlchemyのトランザクション内でsqlite-vec実体を削除する."""

    if not chunk_ids:
        return
    existing_tables = {
        row[0]
        for row in connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    vector_tables = set(MODEL_TABLES.values()) & existing_tables
    if not vector_tables:
        return

    try:
        import sqlite_vec
    except ImportError as error:
        raise RuntimeError(
            "既存ベクトルの無効化にはsqlite-vecが必要です。"
            "uv sync --extra vectors を実行してください。"
        ) from error

    raw_connection = connection.connection.driver_connection
    raw_connection.enable_load_extension(True)
    try:
        sqlite_vec.load(raw_connection)
    finally:
        raw_connection.enable_load_extension(False)

    for start in range(0, len(chunk_ids), 500):
        batch = chunk_ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        for table in vector_tables:
            connection.exec_driver_sql(
                f"DELETE FROM {table} WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            )


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL
    device: str | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_DIMENSIONS:
            raise ValueError(f"未対応の埋め込みモデルです: {self.model_name}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(slots=True)
class EmbeddingResult:
    chunks: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: int
    distance: float
    document_id: str
    path: str | None
    content: str


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class RuriEncoder:
    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from xberg_pipe.fastembed_model import create_ruri_model
        except ImportError as error:
            raise RuntimeError(
                "ベクトル生成依存がありません。uv sync --extra vectors を実行してください。"
            ) from error
        self.model = create_ruri_model(config.model_name, config.device)
        self.batch_size = config.batch_size

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return list(self.model.embed(list(texts), batch_size=self.batch_size))


def _serialize(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    try:
        import sqlite_vec
    except ImportError as error:
        raise RuntimeError(
            "sqlite-vecがありません。uv sync --extra vectors を実行してください。"
        ) from error
    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_vector_table(connection: sqlite3.Connection, model_name: str) -> str:
    table = MODEL_TABLES[model_name]
    dimensions = MODEL_DIMENSIONS[model_name]
    connection.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{dimensions}] "
        "distance_metric=cosine)"
    )
    return table


def rebuild_embeddings(
    db_path: str | Path,
    config: EmbeddingConfig | None = None,
    encoder: Encoder | None = None,
    progress: Callable[[EmbeddingResult], None] | None = None,
    all_items: bool = False,
) -> EmbeddingResult:
    config = config or EmbeddingConfig()
    connection = _connect(db_path)
    result = EmbeddingResult()
    try:
        table = _ensure_vector_table(connection, config.model_name)
        if all_items:
            connection.execute(f"DELETE FROM {table}")
            connection.execute(
                "DELETE FROM chunk_embeddings WHERE model_name = ?",
                (config.model_name,),
            )
        cursor = connection.execute(
            "SELECT chunks.id, chunks.content FROM document_chunks AS chunks "
            "LEFT JOIN chunk_embeddings AS stored ON stored.chunk_id = chunks.id "
            "AND stored.model_name = ? WHERE stored.id IS NULL ORDER BY chunks.id",
            (config.model_name,),
        )
        while rows := cursor.fetchmany(config.batch_size):
            if encoder is None:
                encoder = RuriEncoder(config)
            texts = [DOCUMENT_PREFIX + row[1] for row in rows]
            vectors = encoder.encode(texts)
            if any(
                len(vector) != MODEL_DIMENSIONS[config.model_name] for vector in vectors
            ):
                raise ValueError("埋め込みベクトルの次元数がモデル定義と一致しません")
            connection.executemany(
                f"INSERT INTO {table}(chunk_id, embedding) VALUES (?, ?)",
                [
                    (row[0], _serialize(vector))
                    for row, vector in zip(rows, vectors, strict=True)
                ],
            )
            connection.executemany(
                "INSERT INTO chunk_embeddings(chunk_id, model_name, dimensions) "
                "VALUES (?, ?, ?)",
                [
                    (row[0], config.model_name, MODEL_DIMENSIONS[config.model_name])
                    for row in rows
                ],
            )
            result.chunks += len(rows)
            if progress is not None:
                progress(result)
        connection.commit()
    finally:
        connection.close()
    return result


def search_embeddings(
    db_path: str | Path,
    query: str,
    top_k: int = 10,
    config: EmbeddingConfig | None = None,
    encoder: Encoder | None = None,
) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    config = config or EmbeddingConfig()
    encoder = encoder or RuriEncoder(config)
    query_vector = encoder.encode([QUERY_PREFIX + query])[0]
    connection = _connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        table = _ensure_vector_table(connection, config.model_name)
        rows = connection.execute(
            f"""
            WITH nearest AS (
                SELECT chunk_id, distance FROM {table}
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT nearest.chunk_id, nearest.distance, chunks.content,
                   texts.document_id,
                   (SELECT path FROM document_paths
                    WHERE document_id = texts.document_id ORDER BY id LIMIT 1) AS path
            FROM nearest
            JOIN document_chunks AS chunks ON chunks.id = nearest.chunk_id
            JOIN document_texts AS texts ON texts.id = chunks.document_text_id
            ORDER BY nearest.distance
            """,
            (_serialize(query_vector), top_k),
        ).fetchall()
        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                distance=row["distance"],
                document_id=row["document_id"],
                path=row["path"],
                content=row["content"],
            )
            for row in rows
        ]
    finally:
        connection.close()
