from __future__ import annotations

from pathlib import Path

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import aliased, sessionmaker

from xberg_pipe.db import create_db_engine, init_db
from xberg_pipe.keywording import DEFAULT_MODEL, LIGHT_MODEL
from xberg_pipe.models import DocumentKeyword, DocumentPath, DocumentText, Keyword
from xberg_pipe.vector_store import EmbeddingConfig, Encoder, search_embeddings

DEFAULT_PREVIEW_CHARS = 240


def _preview(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def _path_for_document(document_id: str) -> str | None:
    return (
        select(DocumentPath.path)
        .where(DocumentPath.document_id == document_id)
        .order_by(DocumentPath.id)
        .limit(1)
        .scalar_subquery()
    )


def create_app(
    database: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    encoder: Encoder | None = None,
):
    """検索用FastAPIアプリケーションを作成する."""

    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse
    except ImportError as error:
        raise RuntimeError(
            "サーバー依存がありません。uv sync --extra vis を実行してください。"
        ) from error

    if preview_chars <= 0:
        raise ValueError("preview_chars must be positive")

    database_path = Path(database)
    engine = create_db_engine(database_path)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    embedding_config = EmbeddingConfig(model_name=model_name, device=device)

    app = FastAPI(title="xberg-pipe search", version="0.1.0")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/search/vector")
    def vector_search(
        q: str = Query(min_length=1), top_k: int = Query(default=10, ge=1, le=100)
    ) -> dict[str, object]:
        try:
            matches = search_embeddings(
                database_path,
                q,
                top_k=top_k,
                config=embedding_config,
                encoder=encoder,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        document_ids = {match.document_id for match in matches}
        previews: dict[str, str] = {}
        if document_ids:
            with session_factory() as session:
                rows = session.execute(
                    select(
                        DocumentText.document_id,
                        func.coalesce(
                            DocumentText.cleaned_text, DocumentText.extracted_text
                        ),
                    ).where(DocumentText.document_id.in_(document_ids))
                )
                previews = {
                    document_id: _preview(text, preview_chars)
                    for document_id, text in rows
                }
        return {
            "query": q,
            "mode": "vector",
            "results": [
                {
                    "document_id": match.document_id,
                    "path": match.path,
                    "chunk_id": match.chunk_id,
                    "distance": match.distance,
                    "preview": previews.get(match.document_id, ""),
                }
                for match in matches
            ],
        }

    @app.get("/api/search/keyword")
    def keyword_search(
        q: str = Query(min_length=1), top_k: int = Query(default=10, ge=1, le=100)
    ) -> dict[str, object]:
        query = q.strip()
        if not query:
            raise HTTPException(status_code=422, detail="検索語を入力してください")
        pattern = (
            "%"
            + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            + "%"
        )
        path_column = _path_for_document(DocumentText.document_id)
        refined_keyword = aliased(DocumentKeyword)
        has_refined = (
            select(refined_keyword.id)
            .where(
                refined_keyword.document_text_id == DocumentText.id,
                refined_keyword.model_name == model_name,
            )
            .exists()
        )
        with session_factory() as session:
            matched_score = func.max(
                case(
                    (
                        Keyword.value.contains(query, autoescape=True),
                        DocumentKeyword.score,
                    ),
                    else_=None,
                )
            )
            rows = session.execute(
                select(
                    DocumentText.document_id,
                    func.coalesce(
                        DocumentText.cleaned_text, DocumentText.extracted_text
                    ).label("cleaned_text"),
                    path_column.label("path"),
                    matched_score.label("score"),
                    func.group_concat(Keyword.value, ", ").label("keywords"),
                )
                .select_from(DocumentText)
                .outerjoin(
                    DocumentKeyword,
                    and_(
                        DocumentKeyword.document_text_id == DocumentText.id,
                        or_(
                            DocumentKeyword.model_name == model_name,
                            and_(
                                DocumentKeyword.model_name == LIGHT_MODEL,
                                ~has_refined,
                            ),
                        ),
                    ),
                )
                .outerjoin(Keyword, Keyword.id == DocumentKeyword.keyword_id)
                .where(
                    or_(
                        Keyword.value.contains(query, autoescape=True),
                        func.coalesce(
                            DocumentText.cleaned_text, DocumentText.extracted_text
                        ).like(pattern, escape="\\"),
                    ),
                )
                .group_by(DocumentText.id)
                .order_by(matched_score.desc())
                .limit(top_k)
            ).all()
        return {
            "query": query,
            "mode": "keyword",
            "results": [
                {
                    "document_id": row.document_id,
                    "path": row.path,
                    "score": row.score,
                    "keywords": row.keywords.split(", ") if row.keywords else [],
                    "preview": _preview(row.cleaned_text, preview_chars),
                }
                for row in rows
            ],
        }

    return app


_INDEX_HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>xberg-pipe 検索</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:3rem auto;padding:0 1rem;color:#202124}
form{display:flex;gap:.6rem;flex-wrap:wrap}input{flex:1;min-width:16rem;padding:.7rem}
select,button{padding:.7rem}article{border-top:1px solid #ddd;padding:1rem 0}
.meta{color:#666;font-size:.9rem}.empty{color:#666;margin-top:2rem}
</style></head><body><h1>文書検索</h1>
<form id="search"><input id="q" required placeholder="検索語を入力">
<select id="mode"><option value="vector">ベクトル検索</option><option value="keyword">キーワード検索</option></select>
<button>検索</button></form><main id="results"></main>
<script>
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
document.querySelector('#search').addEventListener('submit',async e=>{e.preventDefault();
 const q=document.querySelector('#q').value,mode=document.querySelector('#mode').value,out=document.querySelector('#results');
 out.innerHTML='<p class="empty">検索中…</p>';
 try{const r=await fetch(`/api/search/${mode}?q=${encodeURIComponent(q)}`),data=await r.json();
 if(!r.ok)throw new Error(data.detail||'検索に失敗しました');
 out.innerHTML=data.results.length?data.results.map((x,i)=>`<article><div class="meta">${i+1}. ${esc(x.path||x.document_id)} ${x.distance!=null?' / distance '+x.distance.toFixed(4):''}</div><p>${esc(x.preview)}</p></article>`).join(''):'<p class="empty">該当する文書はありません。</p>';
 }catch(err){out.innerHTML=`<p class="empty">${esc(err.message)}</p>`}});
</script></body></html>"""
