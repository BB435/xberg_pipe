import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from xberg import ExtractionConfig, OcrConfig

from xberg_pipe.chunking import ChunkingConfig, rebuild_all_chunks
from xberg_pipe.db import DB_PATH, create_db_engine, init_db
from xberg_pipe.keywording import DEFAULT_MODEL, RURI_MODELS, KeywordConfig
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xberg-pipe",
        description="文書の抽出、チャンキング、SQLite保存を実行します。",
    )
    parser.add_argument(
        "--database",
        default=DB_PATH,
        metavar="PATH",
        help=f"SQLiteファイル (default: {DB_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="SQLiteテーブルを作成します。")

    scan = subparsers.add_parser("scan", help="ディレクトリを再帰走査して抽出します。")
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

    chunk = subparsers.add_parser(
        "chunk", help="保存済みテキストのチャンクを再生成します。"
    )
    chunk.add_argument("--target-chars", type=int, default=1_200)
    chunk.add_argument("--overlap-chars", type=int, default=200)
    chunk.add_argument("--batch-size", type=int, default=100)

    keywords = subparsers.add_parser(
        "keywords", help="KeyBERTでファイルごとのキーワードを抽出します。"
    )
    keywords.add_argument(
        "--model", choices=RURI_MODELS, default=DEFAULT_MODEL, help="Ruri埋め込みモデル"
    )
    keywords.add_argument("--top-n", type=int, default=10)
    keywords.add_argument("--max-ngram", type=int, default=3)
    keywords.add_argument("--max-candidates", type=int, default=500)
    keywords.add_argument("--diversity", type=float, default=0.35)
    keywords.add_argument("--device", help="例: cpu, cuda, cuda:0")
    keywords.add_argument("--batch-size", type=int, default=20)

    summarize = subparsers.add_parser(
        "summarize", help="ローカルLLMでファイルごとの要約を生成します。"
    )
    summarize.add_argument("--model", required=True, help="ローカルLLMのモデル名")
    summarize.add_argument(
        "--endpoint", default="http://127.0.0.1:11434/v1", help="OpenAI互換API"
    )
    summarize.add_argument("--api-key")
    summarize.add_argument("--timeout", type=int, default=120)
    summarize.add_argument("--max-output-tokens", type=int, default=800)

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

    subparsers.add_parser("stats", help="SQLiteに保存された件数を表示します。")
    return parser


def _run_scan(args: argparse.Namespace, session: Session) -> int:
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
    result = asyncio.run(
        ScanExtractor(
            session,
            root,
            extraction_config=extraction_config,
            batch_size=args.batch_size,
        ).scan()
    )
    print(
        f"検出={result.discovered} 抽出={result.extracted} "
        f"スキップ={result.skipped} 失敗={result.failed}"
    )
    return 1 if result.failed else 0


def _run_chunk(args: argparse.Namespace, session: Session) -> int:
    config = ChunkingConfig(
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
    )
    result = rebuild_all_chunks(session, config, batch_size=args.batch_size)
    print(f"処理文書={result.documents} 生成チャンク={result.chunks}")
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


def _run_keywords(args: argparse.Namespace, session: Session) -> int:
    # 重いML依存とモデルのロードは、このコマンドを選択したときだけ行う。
    from xberg_pipe.keywording import rebuild_all_keywords

    config = KeywordConfig(
        model_name=args.model,
        top_n=args.top_n,
        max_ngram=args.max_ngram,
        max_candidates=args.max_candidates,
        diversity=args.diversity,
        device=args.device,
    )
    result = rebuild_all_keywords(session, config=config, batch_size=args.batch_size)
    print(
        f"処理文書={result.documents} 保存キーワード={result.keywords} "
        f"スキップ={result.skipped}"
    )
    return 0


def _run_summarize(args: argparse.Namespace, session: Session) -> int:
    from xberg_pipe.summarization import SummaryConfig, rebuild_all_summaries

    config = SummaryConfig(
        model_name=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        max_output_tokens=args.max_output_tokens,
    )
    result = rebuild_all_summaries(session, config)
    print(f"処理文書={result.documents} 失敗={result.failed} スキップ={result.skipped}")
    return 1 if result.failed else 0


def _run_embed(args: argparse.Namespace) -> int:
    from xberg_pipe.vector_store import EmbeddingConfig, rebuild_embeddings

    config = EmbeddingConfig(
        model_name=args.model, device=args.device, batch_size=args.batch_size
    )
    result = rebuild_embeddings(args.database, config)
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
            if args.command == "chunk":
                return _run_chunk(args, session)
            if args.command == "keywords":
                return _run_keywords(args, session)
            if args.command == "summarize":
                return _run_summarize(args, session)
            if args.command == "embed":
                return _run_embed(args)
            if args.command == "search":
                return _run_search(args)
            if args.command == "stats":
                return _run_stats(session)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 1
