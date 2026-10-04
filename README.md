# xberg-pipe

ローカル文書を抽出し、キーワード検索・ベクトル検索するためのパイプラインです。

## 埋め込みモデル

キーフレーズ抽出とベクトル検索はFastEmbedで
`sirasagi62/ruri-v3-30m-ONNX`を使用します。初回実行時にモデルを取得します。
旧モデルで生成したベクトルは新しい検索用テーブルでは使用されません。
既存のデータベースでは`uv run xberg-pipe embed`を実行してベクトルを再生成してください。

## 文書の取り込み

`scan`で変更されたファイルのテキスト抽出、チャンク生成、KeyBERTによる
キーフレーズ抽出、暫定要約の保存を一度に行います。暫定要約は原文中の文を
重要度に応じて選ぶため、LLMや追加のモデルを使いません。

```console
uv sync --extra keywords --group dev
uv run xberg-pipe scan <directory>
```

既存のデータベースでも、暫定要約がまだない文書は次の`scan`で処理されます。

## 検索サーバー

```console
uv sync --extra all --group dev
uv run xberg-pipe --database data/app.db serve
```

起動後、`http://127.0.0.1:8000/` で検索画面を開けます。APIは
`/api/search/vector?q=検索文` と `/api/search/keyword?q=検索語` です。

## Ollamaによる要約

小型モデル向けの既定値は `qwen3.5:4b` です。Ollamaを起動してモデルを取得後、
保存済み文書の暫定要約とは別に、LLM要約を生成できます。

```console
uv sync --extra llm --group dev
ollama pull qwen3.5:4b
uv run xberg-pipe summarize
```

より軽量な `qwen3.5:2b` や、`gemma4:e2b` も `--model` で指定できます。
