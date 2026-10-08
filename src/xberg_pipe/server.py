from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from xberg_pipe.db import create_db_engine, init_db
from xberg_pipe.keywording import DEFAULT_MODEL
from xberg_pipe.search import search
from xberg_pipe.vector_store import EmbeddingConfig, Encoder

DEFAULT_PREVIEW_CHARS = 240


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
    embedding_config = EmbeddingConfig(model_name=model_name, device=device)

    app = FastAPI(title="xberg-pipe search", version="0.1.0")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _INDEX_HTML

    def run_search(
        mode: str, q: str, top_k: int, path: str | None, extension: str | None
    ):
        if not q.strip():
            raise HTTPException(status_code=422, detail="検索語を入力してください")
        try:
            hits = search(
                database_path,
                q,
                mode=mode,
                top_k=top_k,
                preview_chars=preview_chars,
                path_filter=path,
                extension=extension,
                config=embedding_config,
                encoder=encoder,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "query": q.strip(),
            "mode": mode,
            "results": [asdict(hit) for hit in hits],
        }

    @app.get("/api/search/{mode}")
    def search_api(
        mode: str,
        q: str = Query(min_length=1),
        top_k: int = Query(default=10, ge=1, le=100),
        path: str | None = None,
        extension: str | None = None,
    ) -> dict[str, object]:
        if mode not in {"keyword", "vector", "hybrid"}:
            raise HTTPException(status_code=404, detail="検索モードが見つかりません")
        return run_search(mode, q, top_k, path, extension)

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
<select id="mode"><option value="keyword">キーワード検索</option><option value="hybrid">複合検索</option><option value="vector">ベクトル検索</option></select>
<input id="path" placeholder="パスで絞り込み"><input id="extension" placeholder="拡張子 (例: pdf)">
<button>検索</button></form><main id="results"></main>
<script>
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
document.querySelector('#search').addEventListener('submit',async e=>{e.preventDefault();
 const q=document.querySelector('#q').value,mode=document.querySelector('#mode').value,out=document.querySelector('#results');
 out.innerHTML='<p class="empty">検索中…</p>';
 try{const params=new URLSearchParams({q});for(const id of ['path','extension']){const value=document.querySelector('#'+id).value.trim();if(value)params.set(id,value)}
 const r=await fetch(`/api/search/${mode}?${params}`),data=await r.json();
 if(!r.ok)throw new Error(data.detail||'検索に失敗しました');
 out.innerHTML=data.results.length?data.results.map((x,i)=>`<article><div class="meta">${i+1}. ${esc(x.path||x.document_id)} ${x.distance!=null?' / distance '+x.distance.toFixed(4):''}</div><p>${esc(x.preview)}</p></article>`).join(''):'<p class="empty">該当する文書はありません。</p>';
 }catch(err){out.innerHTML=`<p class="empty">${esc(err.message)}</p>`}});
</script></body></html>"""
