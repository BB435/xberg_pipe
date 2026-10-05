import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm.auto import tqdm
from xberg import ExtractionConfig, OcrConfig

from xberg_pipe.chunking import (
    CHUNKER_VERSION,
    MAX_DOWNSTREAM_CHUNKS,
    ChunkingConfig,
    chunk_text,
    replace_document_chunks,
)
from xberg_pipe.db import DB_PATH, create_db_engine, init_db
from xberg_pipe.keywording import (
    DEFAULT_MODEL,
    LIGHT_MODEL,
    RURI_MODELS,
    KeywordConfig,
)
from xberg_pipe.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentKeyword,
    DocumentPath,
    DocumentSummary,
    DocumentText,
)
from xberg_pipe.scan_extractor import BATCH_SIZE, ScanExtractor
from xberg_pipe.summarization import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_SUMMARY_MODEL,
)


def _advance_progress(bar: tqdm, completed: int, **details: int) -> None:
    bar.set_postfix(details, refresh=False)
    bar.update(completed - bar.n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xberg-pipe",
        description="文書の抽出、チャンク・キーフレーズ・暫定要約の保存を実行します。",
    )
    parser.add_argument(
        "--database",
        default=DB_PATH,
        metavar="PATH",
        help=f"SQLiteファイル (default: {DB_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="SQLiteテーブルを作成します。")

    scan = subparsers.add_parser(
        "scan", help="抽出からキーフレーズ・暫定要約まで実行します。"
    )
    scan.add_argument("root", type=Path, help="走査するディレクトリ")
    scan.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="一度に抽出するファイル数"
    )
    scan.add_argument(
        "--ocr-backend", default="paddleocr", help="xberg OCRバックエンド"
    )
    scan.add_argument(
        "--ocr-language",
        action="append",
        dest="ocr_languages",
        metavar="CODE",
        help="OCR言語。複数指定可 (default: jpn, en)",
    )
    ocr_mode = scan.add_mutually_exclusive_group()
    ocr_mode.add_argument(
        "--force-ocr", action="store_true", help="ネイティブテキストがあってもOCRする"
    )
    ocr_mode.add_argument("--disable-ocr", action="store_true", help="OCRを無効化する")

    refine = subparsers.add_parser(
        "refine-keywords", help="保存済み文書のキーフレーズをKeyBERTで精緻化します。"
    )
    refine.add_argument("--model", choices=RURI_MODELS, default=DEFAULT_MODEL)
    refine.add_argument("--device", help="例: cpu, cuda, cuda:0")
    refine.add_argument("--batch-size", type=int, default=20)

    summarize = subparsers.add_parser(
        "summarize", help="Ollamaでファイルごとの要約を生成します。"
    )
    summarize.add_argument(
        "--model",
        default=DEFAULT_SUMMARY_MODEL,
        help="Ollamaモデル (例: qwen3.5:4b, gemma4:e2b)",
    )
    summarize.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollamaホスト")
    summarize.add_argument("--timeout", type=int, default=180)
    summarize.add_argument("--max-output-tokens", type=int, default=512)
    summarize.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    summarize.add_argument("--temperature", type=float, default=0.1)
    summarize.add_argument("--reduce-group-size", type=int, default=6)
    summarize.add_argument("--think", action="store_true", help="思考モードを有効化")
    summarize.add_argument("--keep-alive", default="10m")

    embed = subparsers.add_parser(
        "embed", help="チャンクの検索用ベクトルをsqlite-vecへ保存します。"
    )
    embed.add_argument("--model", choices=RURI_MODELS, default=DEFAULT_MODEL)
    embed.add_argument("--device", help="例: cpu, cuda, cuda:0")
    embed.add_argument("--batch-size", type=int, default=32)

    search = subparsers.add_parser("search", help="sqlite-vecで意味検索します。")
    search.add_argument("query", help="検索文")
    search.add_argument("--model", choices=RURI_MODELS, default=DEFAULT_MODEL)
    search.add_argument("--device", help="例: cpu, cuda, cuda:0")
    search.add_argument("--top-k", type=int, default=10)

    serve = subparsers.add_parser(
        "serve", help="ベクトル検索・キーワード検索サーバーを起動します。"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", choices=RURI_MODELS, default=DEFAULT_MODEL)
    serve.add_argument("--device", help="例: cpu, cuda, cuda:0")
    serve.add_argument("--preview-chars", type=int, default=240)

    subparsers.add_parser("stats", help="SQLiteに保存された件数を表示します。")
    return parser


def _run_scan(args: argparse.Namespace, session: Session) -> int:
    from xberg_pipe import provisional_summary
    from xberg_pipe.keywording import (
        YakeKeywordExtractor,
        aggregate_chunk_keywords,
        replace_document_keywords,
    )

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"走査対象ディレクトリが見つかりません: {root}")

    extraction_config = ExtractionConfig(
        ocr=OcrConfig(
            enabled=not args.disable_ocr,
            backend=args.ocr_backend,
            language=args.ocr_languages or ["jpn", "en"],
        ),
        force_ocr=args.force_ocr,
    )
    keyword_config = KeywordConfig(model_name=LIGHT_MODEL)
    extractor = None

    def postprocess(session: Session, document_text: DocumentText) -> None:
        nonlocal extractor
        if extractor is None:
            extractor = YakeKeywordExtractor(keyword_config.max_ngram)
        chunks = chunk_text(document_text.processing_text)[:MAX_DOWNSTREAM_CHUNKS]
        session.flush()
        chunker_version = session.scalar(
            select(DocumentChunk.chunker_version)
            .where(DocumentChunk.document_text_id == document_text.id)
            .limit(1)
        )
        if chunker_version != CHUNKER_VERSION:
            replace_document_chunks(session, document_text, ChunkingConfig())
        keywords = aggregate_chunk_keywords(
            (extractor.extract(chunk, keyword_config.top_n * 3) for chunk in chunks),
            keyword_config.top_n,
        )
        replace_document_keywords(
            session, document_text, keywords, keyword_config.model_name
        )
        session.execute(
            delete(DocumentKeyword).where(
                DocumentKeyword.document_text_id == document_text.id,
                DocumentKeyword.model_name == "ja-morph-v1",
            )
        )
        provisional_summary.replace_provisional_summary(
            session,
            document_text,
            provisional_summary.summarize_extractively(
                document_text.processing_text, keywords
            ),
        )

    with tqdm(desc="走査中", unit="件") as bar:
        result = asyncio.run(
            ScanExtractor(
                session,
                root,
                extraction_config=extraction_config,
                batch_size=args.batch_size,
                postprocess=postprocess,
                required_summary=(
                    provisional_summary.MODEL_NAME,
                    provisional_summary.PROMPT_VERSION,
                ),
                progress=lambda current: _advance_progress(
                    bar,
                    current.extracted + current.skipped + current.failed,
                    抽出=current.extracted,
                    スキップ=current.skipped,
                    失敗=current.failed,
                ),
            ).scan()
        )
    print(f"抽出={result.extracted} スキップ={result.skipped} 失敗={result.failed}")
    return 1 if result.failed else 0


def _run_refine_keywords(args: argparse.Namespace, session: Session) -> int:
    from xberg_pipe.keywording import rebuild_all_keywords

    config = KeywordConfig(model_name=args.model, device=args.device)
    with tqdm(desc="精緻化中", unit="文書") as bar:
        result = rebuild_all_keywords(
            session,
            config=config,
            batch_size=args.batch_size,
            progress=lambda current: _advance_progress(
                bar,
                current.documents + current.skipped,
                キーフレーズ=current.keywords,
                スキップ=current.skipped,
            ),
        )
    print(
        f"処理文書={result.documents} 保存キーフレーズ={result.keywords} "
        f"スキップ={result.skipped}"
    )
    return 0


def _run_stats(session: Session) -> int:
    tables = (
        ("documents", Document),
        ("paths", DocumentPath),
        ("texts", DocumentText),
        ("chunks", DocumentChunk),
        ("keywords", DocumentKeyword),
        ("summaries", DocumentSummary),
        ("embeddings", ChunkEmbedding),
    )
    for label, model in tables:
        count = session.scalar(select(func.count()).select_from(model))
        print(f"{label}={count}")
    return 0


def _run_summarize(args: argparse.Namespace, session: Session) -> int:
    from xberg_pipe.summarization import SummaryConfig, rebuild_all_summaries

    config = SummaryConfig(
        model_name=args.model,
        host=args.host,
        timeout_seconds=args.timeout,
        max_output_tokens=args.max_output_tokens,
        context_window=args.context_window,
        temperature=args.temperature,
        reduce_group_size=args.reduce_group_size,
        think=args.think,
        keep_alive=args.keep_alive,
    )
    with tqdm(desc="要約生成中", unit="文書") as bar:
        result = rebuild_all_summaries(
            session,
            config,
            progress=lambda current: _advance_progress(
                bar,
                current.documents + current.failed + current.skipped,
                完了=current.documents,
                失敗=current.failed,
                スキップ=current.skipped,
            ),
        )
    print(f"処理文書={result.documents} 失敗={result.failed} スキップ={result.skipped}")
    return 1 if result.failed else 0


def _run_embed(args: argparse.Namespace) -> int:
    from xberg_pipe.vector_store import EmbeddingConfig, rebuild_embeddings

    config = EmbeddingConfig(
        model_name=args.model, device=args.device, batch_size=args.batch_size
    )
    with tqdm(desc="埋め込み生成中", unit="ベクトル") as bar:
        result = rebuild_embeddings(
            args.database,
            config,
            progress=lambda current: _advance_progress(bar, current.chunks),
        )
    print(f"生成ベクトル={result.chunks} モデル={config.model_name}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    from xberg_pipe.vector_store import EmbeddingConfig, search_embeddings

    config = EmbeddingConfig(model_name=args.model, device=args.device)
    results = search_embeddings(
        args.database, args.query, top_k=args.top_k, config=config
    )
    for rank, item in enumerate(results, start=1):
        preview = " ".join(item.content.split())[:160]
        print(
            f"{rank}. distance={item.distance:.4f} "
            f"document={item.document_id} path={item.path or '-'}"
        )
        print(f"   {preview}")
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "サーバー依存がありません。uv sync --extra vis を実行してください。"
        ) from error

    from xberg_pipe.server import create_app

    app = create_app(
        args.database,
        model_name=args.model,
        device=args.device,
        preview_chars=args.preview_chars,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        db_engine = create_db_engine(args.database)
        init_db(db_engine)
        if args.command == "init-db":
            print(f"SQLiteを初期化しました: {Path(args.database).resolve()}")
            return 0

        session_factory = sessionmaker(
            bind=db_engine, autoflush=False, expire_on_commit=False
        )
        with session_factory() as session:
            if args.command == "scan":
                return _run_scan(args, session)
            if args.command == "refine-keywords":
                return _run_refine_keywords(args, session)
            if args.command == "summarize":
                return _run_summarize(args, session)
            if args.command == "embed":
                return _run_embed(args)
            if args.command == "search":
                return _run_search(args)
            if args.command == "serve":
                return _run_serve(args)
            if args.command == "stats":
                return _run_stats(session)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 1
