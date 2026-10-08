"""CLIとWebで共有する文書検索。"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session, aliased

from xberg_pipe.db import create_db_engine
from xberg_pipe.keywording import DEFAULT_MODEL, LIGHT_MODEL
from xberg_pipe.models import DocumentKeyword, DocumentPath, DocumentText, Keyword
from xberg_pipe.vector_store import EmbeddingConfig, Encoder, search_embeddings


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    path: str | None
    preview: str
    keywords: tuple[str, ...] = ()
    score: float | None = None
    distance: float | None = None
    chunk_id: int | None = None


def _preview(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _pattern(value: str) -> str:
    return (
        "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    )


def _validate(query: str, top_k: int, preview_chars: int) -> str:
    query = query.strip()
    if not query:
        raise ValueError("検索語を入力してください")
    if top_k <= 0 or preview_chars <= 0:
        raise ValueError("top_kとpreview_charsは正の整数を指定してください")
    return query


def keyword_search(
    session: Session,
    query: str,
    *,
    top_k: int = 10,
    preview_chars: int = 240,
    model_name: str = DEFAULT_MODEL,
    path_filter: str | None = None,
    extension: str | None = None,
) -> list[SearchHit]:
    """保存済みキーワードと本文から文書を探す。"""
    query = _validate(query, top_k, preview_chars)
    path_column = (
        select(DocumentPath.path)
        .where(DocumentPath.document_id == DocumentText.document_id)
        .order_by(DocumentPath.id)
        .limit(1)
        .scalar_subquery()
    )
    refined = aliased(DocumentKeyword)
    has_refined = (
        select(refined.id)
        .where(
            refined.document_text_id == DocumentText.id,
            refined.model_name == model_name,
        )
        .exists()
    )
    matching_score = func.max(
        case((Keyword.value.contains(query, autoescape=True), DocumentKeyword.score))
    )
    statement = (
        select(
            DocumentText.document_id,
            func.coalesce(DocumentText.cleaned_text, DocumentText.extracted_text).label(
                "body"
            ),
            path_column.label("path"),
            matching_score.label("score"),
            func.group_concat(Keyword.value, ", ").label("keywords"),
        )
        .select_from(DocumentText)
        .outerjoin(
            DocumentKeyword,
            and_(
                DocumentKeyword.document_text_id == DocumentText.id,
                or_(
                    DocumentKeyword.model_name == model_name,
                    and_(DocumentKeyword.model_name == LIGHT_MODEL, ~has_refined),
                ),
            ),
        )
        .outerjoin(Keyword, Keyword.id == DocumentKeyword.keyword_id)
        .where(
            or_(
                Keyword.value.contains(query, autoescape=True),
                func.coalesce(
                    DocumentText.cleaned_text, DocumentText.extracted_text
                ).like(_pattern(query), escape="\\"),
            )
        )
        .group_by(DocumentText.id)
        .order_by(matching_score.desc(), DocumentText.document_id)
        .limit(top_k)
    )
    if path_filter:
        statement = statement.where(
            path_column.like(_pattern(path_filter), escape="\\")
        )
    if extension:
        suffix = extension if extension.startswith(".") else f".{extension}"
        statement = statement.where(path_column.endswith(suffix, autoescape=True))
    return [
        SearchHit(
            document_id=row.document_id,
            path=row.path,
            preview=_preview(row.body, preview_chars),
            keywords=tuple(row.keywords.split(", ")) if row.keywords else (),
            score=row.score,
        )
        for row in session.execute(statement)
    ]


def search(
    database: str | Path,
    query: str,
    *,
    mode: str = "keyword",
    top_k: int = 10,
    preview_chars: int = 240,
    path_filter: str | None = None,
    extension: str | None = None,
    config: EmbeddingConfig | None = None,
    encoder: Encoder | None = None,
) -> list[SearchHit]:
    """検索結果を文書単位にまとめて返す。複合検索は順位を統合する。"""
    query = _validate(query, top_k, preview_chars)
    if mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError(f"未対応の検索モードです: {mode}")
    config = config or EmbeddingConfig()
    engine = create_db_engine(database)
    try:
        with Session(engine) as session:
            # 絞り込み後にも十分な候補を得るため、ベクトル側は多めに取得する。
            candidate_count = max(top_k * 5, 50)
            keywords = (
                keyword_search(
                    session,
                    query,
                    top_k=candidate_count,
                    preview_chars=preview_chars,
                    model_name=config.model_name,
                    path_filter=path_filter,
                    extension=extension,
                )
                if mode in {"keyword", "hybrid"}
                else []
            )
            vectors = []
            if mode in {"vector", "hybrid"}:
                matches = search_embeddings(
                    database,
                    query,
                    top_k=candidate_count,
                    config=config,
                    encoder=encoder,
                )
                ids = {match.document_id for match in matches}
                rows = session.execute(
                    select(
                        DocumentText.document_id,
                        func.coalesce(
                            DocumentText.cleaned_text, DocumentText.extracted_text
                        ),
                    ).where(DocumentText.document_id.in_(ids))
                )
                previews = {
                    document_id: _preview(body, preview_chars)
                    for document_id, body in rows
                }
                seen = set()
                for match in matches:
                    if match.document_id in seen:
                        continue
                    if path_filter and (
                        not match.path or path_filter not in match.path
                    ):
                        continue
                    suffix = (
                        extension
                        if not extension or extension.startswith(".")
                        else f".{extension}"
                    )
                    if suffix and (not match.path or not match.path.endswith(suffix)):
                        continue
                    seen.add(match.document_id)
                    vectors.append(
                        SearchHit(
                            document_id=match.document_id,
                            path=match.path,
                            preview=previews.get(match.document_id, ""),
                            distance=match.distance,
                            chunk_id=match.chunk_id,
                        )
                    )
            if mode == "keyword":
                return keywords[:top_k]
            if mode == "vector":
                return vectors[:top_k]
            ranks: dict[str, float] = {}
            hits: dict[str, SearchHit] = {}
            for results in (keywords, vectors):
                for rank, hit in enumerate(results, 1):
                    ranks[hit.document_id] = ranks.get(hit.document_id, 0) + 1 / (
                        60 + rank
                    )
                    prior = hits.get(hit.document_id)
                    hits[hit.document_id] = SearchHit(
                        hit.document_id,
                        hit.path or (prior.path if prior else None),
                        hit.preview or (prior.preview if prior else ""),
                        hit.keywords or (prior.keywords if prior else ()),
                        hit.score
                        if hit.score is not None
                        else (prior.score if prior else None),
                        hit.distance
                        if hit.distance is not None
                        else (prior.distance if prior else None),
                        hit.chunk_id
                        if hit.chunk_id is not None
                        else (prior.chunk_id if prior else None),
                    )
            return [
                hits[key]
                for key in sorted(ranks, key=lambda key: (-ranks[key], key))[:top_k]
            ]
    finally:
        engine.dispose()
