# Repository Guidelines

## 作業方針

- 変更前に関連コード・テストと `git status` を確認し、既存の変更を保持する。
- 依存関係とコマンドの定義は `pyproject.toml`、利用方法は `README.md` を参照する。
- 変更は目的に必要な範囲に絞る。公開コマンド・API・依存関係を変更したら README も更新する。
- 実行時のデータベース、ダウンロードしたモデル、ビルド成果物をコミットしない。

## 構成と責務

実装は `src/xberg_pipe/`、テストは `tests/` に置く。

| 対象 | 主な責務 |
| --- | --- |
| `cli.py` | 取り込み・後処理・検索の CLI |
| `scan_extractor.py` | ファイル検出・変更判定・テキスト抽出 |
| `models.py`, `db.py`, `repository.py` | SQLite スキーマ・初期化と既存 DB の更新・共有永続化処理 |
| `chunking.py` | 本文のクリーニング・日本語対応チャンク生成 |
| `keywording.py`, `provisional_summary.py` | YAKE/KeyBERT キーフレーズ・抽出的な暫定要約 |
| `summarization.py` | Ollama による要約 |
| `fastembed_model.py`, `vector_store.py` | 埋め込みモデル設定・sqlite-vec 保存と検索 |
| `search.py` | CLI と Web で共有する検索・絞り込み・順位統合 |
| `server.py`, `server_cli.py` | 検索画面と API・専用サーバー起動コマンド |

## 保持する動作

- `scan` は変更ファイルを抽出し、チャンク・軽量キーフレーズ・暫定要約を保存する。
  ML・LLM・ベクトル・サーバーの任意依存なしで利用できる状態を保つ。
- `refine-keywords`・`summarize`・`embed` は選択したモデル・版の未処理項目だけ処理し、
  `--all` で生成済み項目も再生成する。
- 本文やチャンク生成条件が変わる場合、チャンク・キーフレーズ・要約・埋め込みメタデータ・
  実ベクトルを一緒に無効化する。共有処理 `repository.delete_document_derivatives` を使う。
- 抽出原文を保持し、後処理には `DocumentText.processing_text` と既存の処理上限を使う。
- 検索の共通処理は `search.py` に置く。キーフレーズは選択モデルの精緻化結果を優先し、
  未生成なら軽量結果を使う。検索条件の変更では各検索モードへの適用範囲を明示する。
- スキーマ変更は新規 DB と既存 DB の両方を考慮し、更新処理を `db.py` に用意する。

## 開発と検証

Python 3.14 以上と `uv` を使用する。通常は `uv sync --group dev`、任意機能が必要な場合は
`uv sync --extra <extra> --group dev` を使う。extra は `keywords`・`vectors`・`llm`・`vis`・`all`。

```console
uv run xberg-pipe --database <path> scan <directory>
uv run xberg-pipe --database <path> search <query>
uv run xberg-pipe-server --database <path>
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

- `xberg-pipe` の `--database` はサブコマンドの前に指定する。既定は `data/app.db`。
- 4 スペース、型注釈、`pathlib.Path`、データ構造には型付き dataclass を使う。
  命名は `snake_case` / `PascalCase` / `UPPER_CASE`。重い任意依存は必要時に import する。
  docstring と利用者向けメッセージは簡潔な日本語にする。
- テストは `tests/test_<module>.py` の `test_<behavior>` に置く。`tmp_path` または
  インメモリ SQLite、偽の LLM・埋め込み実装を使い、実 DB の書き換えやモデル取得を避ける。
  任意拡張が必要なテストには `pytest.importorskip` を使う。
- 変更に関係するテストと Ruff を実行する。後処理変更では保存内容・本文変更時の無効化・
  増分再実行・`--all` を、検索変更では複数語・除外語・パス境界・特殊文字を確認する。
- ベクトル変更では `uv run --with sqlite-vec pytest -q tests/test_vector_store.py`、
  サーバー変更では `tests/test_server.py` と `tests/test_server_cli.py` も検証する。
  失敗・スキップ・未完了の検証は成功と区別して報告する。

## コミットと PR

コミット件名は短い Conventional Commits（例: `feat: add search filter`）。PR には動作への影響、
スキーマ・依存変更、検証結果、必要なモデル取得・移行手順を記載し、関連 issue があればリンクする。
