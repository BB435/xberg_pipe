import datetime as dt
import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import exists, func, select, true
from sqlalchemy.orm import selectinload
from xberg import ExtractInput, ExtractionConfig, OcrConfig, extract_batch

from xberg_pipe.chunking import clean_extracted_text
from xberg_pipe.keywording import LIGHT_MODEL
from xberg_pipe.models import (
    Document,
    DocumentChunk,
    DocumentKeyword,
    DocumentPath,
    DocumentSummary,
    DocumentText,
)
from xberg_pipe.repository import (
    delete_document_derivatives,
    get_document_path,
    get_document_text,
    get_or_create_document,
)

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
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not path.is_file():
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
        progress: Callable[[ScanResult], None] | None = None,
        postprocess: Callable[[Session, DocumentText], None] | None = None,
        required_summary: tuple[str, str] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.session = session
        self.root = root
        self.batch_size = batch_size
        self.extraction_config = extraction_config or ExtractionConfig(
            ocr=OcrConfig(backend="paddleocr", language=["jpn", "en"])
        )
        self.progress = progress
        self.postprocess = postprocess
        self.required_summary = required_summary

    def _is_unchanged(self, path: Path) -> bool:
        """ファイルが変化したかをチェックする."""
        document_path = get_document_path(self.session, str(path.resolve()))
        if document_path is None:
            return False
        stat = path.stat()
        modified_at = modified_at_of(path)
        return bool(
            document_path.document.status == STATUS_EXTRACTED
            and get_document_text(document_path.document, EXTRACTOR_NAME) is not None
            and document_path.modified_at == modified_at
            and document_path.file_size == stat.st_size
            and document_path.document_id == sha256_of(path)
        )

    def _unchanged_paths(
        self, paths: list[Path], resolved_paths: dict[Path, str] | None = None
    ) -> set[str]:
        """一つのバッチの更新判定に必要なメタデータをまとめて読む."""

        processed_clause = true()
        if self.required_summary is not None:
            model_name, prompt_version = self.required_summary
            processed_clause = exists().where(
                DocumentSummary.document_text_id == DocumentText.id,
                DocumentSummary.model_name == model_name,
                DocumentSummary.prompt_version == prompt_version,
            )
            has_chunks = exists().where(
                DocumentChunk.document_text_id == DocumentText.id
            )
            keyword_count = (
                select(func.count(DocumentKeyword.id))
                .where(
                    DocumentKeyword.document_text_id == DocumentText.id,
                    DocumentKeyword.model_name == LIGHT_MODEL,
                )
                .scalar_subquery()
            )
            processed_clause = (
                processed_clause
                & (
                    has_chunks
                    | (
                        func.trim(
                            func.coalesce(
                                DocumentText.cleaned_text, DocumentText.extracted_text
                            )
                        )
                        == ""
                    )
                )
                & DocumentText.light_keyword_count.is_not(None)
                & (DocumentText.light_keyword_count == keyword_count)
            )
        resolved = {
            resolved_paths[path]
            if resolved_paths is not None
            else str(path.resolve()): path
            for path in paths
        }
        rows = self.session.execute(
            select(
                DocumentPath.path,
                DocumentPath.modified_at,
                DocumentPath.file_size,
                Document.status,
                Document.id,
                DocumentText.id,
                processed_clause,
            )
            .join(DocumentPath.document)
            .outerjoin(
                DocumentText,
                (DocumentText.document_id == Document.id)
                & (DocumentText.extractor == EXTRACTOR_NAME),
            )
            .where(DocumentPath.path.in_(resolved))
        )
        return {
            stored_path
            for stored_path, modified_at, file_size, status, document_id, text_id, processed in rows
            if status == STATUS_EXTRACTED
            and text_id is not None
            and processed
            and (stat := resolved[stored_path].stat()).st_size == file_size
            and dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).replace(tzinfo=None)
            == modified_at
            and sha256_of(resolved[stored_path]) == document_id
        }

    async def scan(self) -> ScanResult:
        """対象ファイルを再帰探索し、変更分の抽出結果をSQLiteへ保存する."""

        result = ScanResult()
        seen_paths: set[str] = set()
        for paths in iter_files(self.root, self.batch_size):
            resolved = {path: str(path.resolve()) for path in paths}
            seen_paths.update(resolved.values())
            result.discovered += len(paths)
            unchanged = self._unchanged_paths(paths, resolved)
            changed_paths = [path for path in paths if resolved[path] not in unchanged]
            result.skipped += len(paths) - len(changed_paths)
            if not changed_paths:
                if self.progress is not None:
                    self.progress(result)
                continue

            source_stats = {}
            for path in changed_paths:
                stat = path.stat()
                source_stats[path] = (stat.st_size, stat.st_mtime_ns)
            try:
                extraction = await extract_batch(
                    [
                        ExtractInput(kind="uri", uri=resolved[path])
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
                    self._mark_failed(path, commit=False)
                self.session.commit()
                if self.progress is not None:
                    self.progress(result)
                continue

            errors_by_index = {error.index: error for error in extraction.errors}
            documents = iter(extraction.results)
            for index, path in enumerate(changed_paths):
                error = errors_by_index.get(index)
                if error is not None:
                    result.failed += 1
                    logger.error(f"Extraction failed for {path}: {error.message}")
                    self._mark_failed(path, commit=False)
                    continue

                try:
                    extracted = next(documents)
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != source_stats[path]:
                        result.failed += 1
                        logger.warning(f"File changed during extraction: {path}")
                        continue
                    with self.session.begin_nested():
                        document_text = self._save(path, extracted.content)
                        if self.postprocess is not None:
                            self.postprocess(self.session, document_text)
                    result.extracted += 1
                except StopIteration:
                    result.failed += 1
                    logger.error(f"Extractor returned no result for {path}")
                    self._mark_failed(path, commit=False)
                except Exception:
                    result.failed += 1
                    logger.exception(f"Could not save extraction result for {path}")
            self.session.commit()
            if self.progress is not None:
                self.progress(result)

        self._remove_missing_paths(seen_paths)
        self.session.commit()
        return result

    def _save(self, path: Path, text: str) -> DocumentText:
        document = get_or_create_document(
            self.session, sha256_of(path), path.suffix.lower()
        )
        document.status = STATUS_EXTRACTED

        previous_document = self._save_path(document, path)

        document_text = get_document_text(document, EXTRACTOR_NAME)
        if document_text is None:
            document_text = DocumentText(extractor=EXTRACTOR_NAME, document=document)
            self.session.add(document_text)
        elif document_text.extracted_text != text:
            delete_document_derivatives(self.session, document_text)
            document_text.light_keyword_count = None
        document_text.extracted_text = text
        document_text.cleaned_text = clean_extracted_text(text)
        self._delete_document_if_orphaned(previous_document, except_document=document)
        return document_text

    def _mark_failed(self, path: Path, *, commit: bool = True) -> None:
        """抽出失敗を保存する."""

        try:
            with self.session.begin_nested():
                document = get_or_create_document(
                    self.session, sha256_of(path), path.suffix.lower()
                )
                if get_document_text(document, EXTRACTOR_NAME) is None:
                    document.status = STATUS_FAILED
                previous_document = self._save_path(document, path)
                self._delete_document_if_orphaned(
                    previous_document, except_document=document
                )
            if commit:
                self.session.commit()
        except Exception:
            if commit:
                self.session.rollback()
            logger.exception(f"Could not save extraction failure for {path}")

    def _save_path(self, document: Document, path: Path) -> Document | None:
        resolved_path = str(path.resolve())
        stat = path.stat()
        document_path = get_document_path(self.session, resolved_path)
        if document_path is None:
            document_path = DocumentPath(path=resolved_path)
            self.session.add(document_path)
        previous_document = document_path.document
        document_path.document = document
        document_path.file_size = stat.st_size
        document_path.modified_at = modified_at_of(path)
        return previous_document

    def _remove_missing_paths(self, seen_paths: set[str]) -> None:
        """走査ルートから消えたパスと、参照されない文書を削除する."""

        root = self.root.resolve()
        prefix = str(root)
        if not prefix.endswith(os.sep):
            prefix += os.sep
        stored_paths = self.session.scalars(
            select(DocumentPath).where(
                DocumentPath.path.startswith(prefix, autoescape=True)
            )
        )
        missing_paths = []
        for document_path in stored_paths:
            stored_path = Path(document_path.path)
            try:
                under_root = stored_path.is_relative_to(root)
            except OSError, ValueError:
                under_root = False
            if not under_root or document_path.path in seen_paths:
                continue
            missing_paths.append(document_path)

        if not missing_paths:
            return
        document_ids = {path.document_id for path in missing_paths}
        for document_path in missing_paths:
            self.session.delete(document_path)
        self.session.flush()
        retained_ids = set(
            self.session.scalars(
                select(DocumentPath.document_id).where(
                    DocumentPath.document_id.in_(document_ids)
                )
            )
        )
        orphan_ids = document_ids - retained_ids
        for document in self.session.scalars(
            select(Document)
            .where(Document.id.in_(orphan_ids))
            .options(selectinload(Document.texts))
        ):
            for document_text in document.texts:
                delete_document_derivatives(self.session, document_text)
            self.session.delete(document)

    def _delete_document_if_orphaned(
        self, document: Document | None, except_document: Document | None = None
    ) -> None:
        """他のパスから参照されない旧文書と派生データを削除する."""

        if document is None or document is except_document:
            return
        self.session.flush()
        path_count = self.session.scalar(
            select(func.count())
            .select_from(DocumentPath)
            .where(DocumentPath.document_id == document.id)
        )
        if path_count:
            return
        for document_text in list(document.texts):
            delete_document_derivatives(self.session, document_text)
        self.session.delete(document)
