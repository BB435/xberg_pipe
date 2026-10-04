# xberg-pipe

ローカル文書を抽出し、キーワード検索・ベクトル検索するためのパイプラインです。

## 埋め込みモデル

キーワード抽出とベクトル検索はFastEmbedで
`sirasagi62/ruri-v3-30m-ONNX`を使用します。初回実行時にモデルを取得します。
旧モデルで生成したベクトルは新しい検索用テーブルでは使用されません。
既存のデータベースでは`uv run xberg-pipe embed`を実行してベクトルを再生成してください。

## 検索サーバー

```console
uv sync --extra all --group dev
uv run xberg-pipe --database data/app.db serve
```

起動後、`http://127.0.0.1:8000/` で検索画面を開けます。APIは
`/api/search/vector?q=検索文` と `/api/search/keyword?q=検索語` です。

## Ollamaによる要約

小型モデル向けの既定値は `qwen3.5:4b` です。Ollamaを起動してモデルを取得後、
保存済み文書を要約できます。

```console
uv sync --extra llm --group dev
ollama pull qwen3.5:4b
uv run xberg-pipe summarize
```

より軽量な `qwen3.5:2b` や、`gemma4:e2b` も `--model` で指定できます。
