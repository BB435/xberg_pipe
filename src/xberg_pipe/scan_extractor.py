import datetime as dt
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from xberg import ExtractInput, ExtractionConfig, OcrConfig, extract_batch

from xberg_pipe.chunking import clear_document_derivatives
from xberg_pipe.models import Document, DocumentPath, DocumentText
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

EXTRACTOR_NAME = "xberg"
STATUS_EXTRACTED = "extracted"
STATUS_FAILED = "failed"


@dataclass(slots=True)
class ScanResult:
    discovered: int = 0
    skipped: int = 0
    extracted: int = 0
    failed: int = 0


def modified_at_of(path: Path) -> dt.datetime:
    """SQLiteに保存できるnaive UTCで更新日時を返す."""

    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).replace(tzinfo=None)


def sha256_of(path: Path) -> str:
    """ファイルのSHA-256を計算する."""

    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def iter_files(root: Path, batch_size: int = BATCH_SIZE) -> Iterator[list[Path]]:
    """ディレクトリを再帰探索し、一定件数ごとにファイル一覧を返す."""
    batch: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        batch.append(path)

        if len(batch) >= batch_size:
            yield batch
            batch.clear()

    if batch:
        yield batch


class ScanExtractor:
    def __init__(
        self,
        session: Session,
        root: Path,
        extraction_config: ExtractionConfig | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.session = session
        self.root = root
        self.batch_size = batch_size
        self.extraction_config = extraction_config or ExtractionConfig(
            ocr=OcrConfig(backend="paddleocr", language=["jpn", "en"])
        )

    def _is_unchanged(self, path: Path) -> bool:
        """ファイルが変化したかをチェックする."""
        document_path = get_document_path(self.session, str(path.resolve()))
        if document_path is None:
            return False
        stat = path.stat()
        modified_at = modified_at_of(path)
        return bool(
            document_path.modified_at == modified_at
            and document_path.file_size == stat.st_size
        )

    async def scan(self) -> ScanResult:
        """対象ファイルを再帰探索し、変更分の抽出結果をSQLiteへ保存する."""

        result = ScanResult()
        for paths in iter_files(self.root, self.batch_size):
            result.discovered += len(paths)
            changed_paths = [path for path in paths if not self._is_unchanged(path)]
            result.skipped += len(paths) - len(changed_paths)
            if not changed_paths:
                continue

            try:
                extraction = await extract_batch(
                    [
                        ExtractInput(kind="uri", uri=str(path.resolve()))
                        for path in changed_paths
                    ],
                    self.extraction_config,
                )
            except Exception:
                logger.exception(
                    f"Batch extraction failed ({len(changed_paths)} files)"
                )
                for path in changed_paths:
                    result.failed += 1
                    self._mark_failed(path)
                continue

            errors_by_index = {error.index: error for error in extraction.errors}
            documents = iter(extraction.results)
            for index, path in enumerate(changed_paths):
                error = errors_by_index.get(index)
                if error is not None:
                    result.failed += 1
                    logger.error(f"Extraction failed for {path}: {error.message}")
                    self._mark_failed(path)
                    continue

                try:
                    extracted = next(documents)
                    self._save(path, extracted.content)
                    self.session.commit()
                    result.extracted += 1
                except StopIteration:
                    result.failed += 1
                    logger.error(f"Extractor returned no result for {path}")
                    self._mark_failed(path)
                except Exception:
                    self.session.rollback()
                    result.failed += 1
                    logger.exception(f"Could not save extraction result for {path}")

        return result

    def _save(self, path: Path, text: str) -> None:
        document = get_or_create_document(
            self.session, sha256_of(path), path.suffix.lower()
        )
        document.status = STATUS_EXTRACTED

        self._save_path(document, path)

        document_text = next(
            (item for item in document.texts if item.extractor == EXTRACTOR_NAME), None
        )
        if document_text is None:
            document_text = DocumentText(extractor=EXTRACTOR_NAME, document=document)
            self.session.add(document_text)
        elif document_text.extracted_text != text:
            clear_document_derivatives(self.session, document_text)
        document_text.extracted_text = text

    def _mark_failed(self, path: Path) -> None:
        """抽出失敗を保存する."""

        try:
            document = get_or_create_document(
                self.session, sha256_of(path), path.suffix.lower()
            )
            document.status = STATUS_FAILED
            self._save_path(document, path)
            self.session.commit()
        except Exception:
            # DBエラー後のSessionでは後続処理できないため、ここだけは復旧する。
            self.session.rollback()
            logger.exception(f"Could not save extraction failure for {path}")

    def _save_path(self, document: Document, path: Path) -> None:
        resolved_path = str(path.resolve())
        stat = path.stat()
        document_path = get_document_path(self.session, resolved_path)
        if document_path is None:
            document_path = DocumentPath(path=resolved_path)
            self.session.add(document_path)
        document_path.document = document
        document_path.file_size = stat.st_size
        document_path.modified_at = modified_at_of(path)
