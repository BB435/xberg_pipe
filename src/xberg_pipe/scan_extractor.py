import datetime as dt
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from xberg import ExtractInput, extract_batch

from xberg_pipe.models import DocumentPath
from xberg_pipe.repository import get_document_path, get_or_create_document

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".docm",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ppt",
    ".pptx",
    ".pptm",
    ".txt",
    ".md",
    ".rtf",
    ".eml",
    ".msg",
}
BATCH_SIZE = 20

type FileCheckResult = Literal["insert", "update", "skip"]


def sha256_of(path: Path) -> str:
    """ファイルのSHA-256を計算する."""

    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def iter_files(root: Path) -> Iterator[list[Path]]:
    """ディレクトリを再帰探索し、一定件数ごとにファイル一覧を返す."""
    batch: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        batch.append(path)

        if len(batch) >= BATCH_SIZE:
            yield batch

    if batch:
        yield batch


class ScanExtractor:
    def __init__(self, session: Session, root: Path) -> None:
        self.session = session
        self.root = root

    def _is_unchanged(self, path: Path) -> bool:
        """ファイルが変化したかをチェックする."""
        document_path = get_document_path(self.session, str(path.resolve()))
        if document_path is None:
            return False
        stat = path.stat()
        modified_at = dt.datetime.fromtimestamp(stat.st_mtime)
        return bool(
            document_path.modified_at == modified_at
            or document_path.file_size == stat.st_size
        )

    def scan(self):
        """再帰的にファイルを探索してテキスト抽出を行う"""
        pass
