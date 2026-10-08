"""検索サーバー専用の起動コマンド。"""

import argparse
from collections.abc import Sequence

from xberg_pipe.db import DB_PATH
from xberg_pipe.keywording import DEFAULT_MODEL, RURI_MODELS
from xberg_pipe.server import DEFAULT_PREVIEW_CHARS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xberg-pipe-server", description="文書検索サーバーを起動します。"
    )
    parser.add_argument(
        "--database",
        default=DB_PATH,
        metavar="PATH",
        help=f"SQLiteファイル (default: {DB_PATH})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", choices=RURI_MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--device", help="例: cpu, cuda, cuda:0")
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("portは1～65535を指定してください")
    if args.preview_chars <= 0:
        parser.error("preview-charsは正の整数を指定してください")

    try:
        import uvicorn
    except ImportError:
        parser.error(
            "サーバー依存がありません。uv sync --extra vis を実行してください。"
        )

    from xberg_pipe.server import create_app

    try:
        app = create_app(
            args.database,
            model_name=args.model,
            device=args.device,
            preview_chars=args.preview_chars,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
