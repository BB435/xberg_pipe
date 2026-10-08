import sys
from types import SimpleNamespace

import pytest

from xberg_pipe.cli import build_parser as build_pipeline_parser
from xberg_pipe.server_cli import build_parser, main


def test_server_has_separate_command() -> None:
    args = build_parser().parse_args(["--database", "docs.db", "--port", "9000"])

    assert args.database == "docs.db"
    assert args.port == 9000
    with pytest.raises(SystemExit):
        build_pipeline_parser().parse_args(["serve"])


def test_server_starts_with_selected_options(monkeypatch) -> None:
    calls = {}

    def fake_create_app(database, **options):
        calls["database"] = database
        calls["options"] = options
        return object()

    def fake_run(app, *, host, port):
        calls["host"] = host
        calls["port"] = port
        calls["app"] = app

    monkeypatch.setattr("xberg_pipe.server.create_app", fake_create_app)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    assert main(["--database", "docs.db", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert calls["database"] == "docs.db"
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9000
    assert calls["options"]["preview_chars"] == 240


@pytest.mark.parametrize("argument", [["--port", "0"], ["--preview-chars", "0"]])
def test_server_rejects_invalid_options(argument) -> None:
    with pytest.raises(SystemExit):
        main(argument)
