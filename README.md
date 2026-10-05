# xberg-pipe

ローカル文書を抽出し、キーワード検索・ベクトル検索するためのパイプラインです。

## 埋め込みモデル

KeyBERTによるキーフレーズ精緻化とベクトル検索はFastEmbedで
`sirasagi62/ruri-v3-30m-ONNX`を使用します。初回実行時にモデルを取得します。
旧モデルで生成したベクトルは新しい検索用テーブルでは使用されません。
既存のデータベースでは`uv run xberg-pipe embed`を実行してベクトルを再生成してください。

## 文書の取り込み

`scan`で変更されたファイルのテキスト抽出、チャンク生成、日本語を分かち書きしたYAKEによる
軽量なキーフレーズ抽出、暫定要約の保存を一度に行います。暫定要約は原文中の文を
重要度に応じて選ぶため、LLMや埋め込みモデルを使いません。
抽出本文は全量保存します。長文の後処理は先頭50,000文字まで、暫定要約は
先頭10,000文字・128文までに制限します。キーフレーズ抽出とLLM要約は
最大50チャンク、各チャンク1,200文字までを使用します。
既存文書も次回の`scan`でこの上限に合わせて再処理されます。

```console
uv sync --group dev
uv run xberg-pipe scan <directory>
```

KeyBERTでキーフレーズの精度を高める場合は、後から実行します。
検索には精緻化した結果を優先し、未処理の文書には軽量な結果を使用します。

```console
uv sync --extra keywords --group dev
uv run xberg-pipe refine-keywords
```

`refine-keywords`、`summarize`、`embed`は通常、選択したモデル・版の結果が
未保存の項目だけ処理します。保存済みの結果も含めて再生成する場合は、各コマンドに
`--all`を指定します（例: `uv run xberg-pipe embed --all`）。

既存のデータベースも次の`scan`で軽量なキーフレーズを生成します。

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
