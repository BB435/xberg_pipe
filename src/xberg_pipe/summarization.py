from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from xberg_pipe.chunking import (
    DEFAULT_TARGET_CHARS,
    MAX_DOWNSTREAM_CHUNKS,
    chunk_text,
)
from xberg_pipe.models import DocumentSummary, DocumentText

PROMPT_VERSION = "ja-v1"
DEFAULT_SUMMARY_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_CONTEXT_WINDOW = 16_384
SYSTEM_PROMPT = (
    "あなたは日本語文書の要約者です。原文にない情報を加えず、固有名詞、数値、"
    "結論、重要な条件を保持して簡潔な日本語で要約してください。"
)


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    """小型Ollamaモデル向けの要約設定."""

    model_name: str = DEFAULT_SUMMARY_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    timeout_seconds: int = 180
    max_output_tokens: int = 512
    context_window: int = DEFAULT_CONTEXT_WINDOW
    temperature: float = 0.1
    reduce_group_size: int = 6
    think: bool = False
    keep_alive: str = "10m"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name is required")
        if not self.host.strip():
            raise ValueError("host is required")
        if (
            self.timeout_seconds <= 0
            or self.max_output_tokens <= 0
            or self.context_window <= 0
        ):
            raise ValueError(
                "timeout, max_output_tokens and context_window must be positive"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.reduce_group_size < 2:
            raise ValueError("reduce_group_size must be at least 2")
        if not self.keep_alive.strip():
            raise ValueError("keep_alive is required")


@dataclass(slots=True)
class SummaryResult:
    documents: int = 0
    failed: int = 0
    skipped: int = 0


class SummaryClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaMessage(Protocol):
    content: str | None


class OllamaChatResponse(Protocol):
    message: OllamaMessage


class OllamaChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        think: bool,
        options: Mapping[str, int | float],
        keep_alive: str,
    ) -> OllamaChatResponse: ...


class OllamaSummaryClient:
    """公式Ollama Pythonクライアントを利用する要約クライアント."""

    def __init__(
        self,
        config: SummaryConfig,
        *,
        client: OllamaChatClient | None = None,
    ) -> None:
        self.config = config
        if client is None:
            try:
                from ollama import Client
            except ImportError as error:
                raise RuntimeError(
                    "Ollama依存がありません。uv sync --extra llm を実行してください。"
                ) from error
            client = Client(host=config.host, timeout=config.timeout_seconds)
        self.client = client

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat(
            model=self.config.model_name,
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
            think=self.config.think,
            options={
                "temperature": self.config.temperature,
                "num_ctx": self.config.context_window,
                "num_predict": self.config.max_output_tokens,
            },
            keep_alive=self.config.keep_alive,
        )
        content = (response.message.content or "").strip()
        if not content:
            raise RuntimeError("Ollamaが空の要約を返しました")
        return content


def summarize_document(
    document_text: DocumentText,
    client: SummaryClient,
    config: SummaryConfig,
) -> str:
    chunks = [
        chunk.content[:DEFAULT_TARGET_CHARS]
        for chunk in document_text.chunks[:MAX_DOWNSTREAM_CHUNKS]
    ]
    if not chunks:
        chunks = chunk_text(document_text.processing_text)[:MAX_DOWNSTREAM_CHUNKS]
    if not chunks:
        return ""

    summaries = [
        client.complete(
            SYSTEM_PROMPT,
            "次の文書断片を、重要事項を落とさず箇条書きで要約してください。\n\n"
            + chunk,
        )
        for chunk in chunks
    ]
    while len(summaries) > 1:
        reduced: list[str] = []
        for start in range(0, len(summaries), config.reduce_group_size):
            group = summaries[start : start + config.reduce_group_size]
            reduced.append(
                client.complete(
                    SYSTEM_PROMPT,
                    "以下は同じ文書の部分要約です。重複を除き、文書全体の要約に"
                    "統合してください。\n\n" + "\n\n".join(group),
                )
            )
        summaries = reduced
    return summaries[0]


def replace_document_summary(
    session: Session,
    document_text: DocumentText,
    summary: str,
    config: SummaryConfig,
) -> None:
    session.execute(
        delete(DocumentSummary).where(
            DocumentSummary.document_text_id == document_text.id,
            DocumentSummary.model_name == config.model_name,
            DocumentSummary.prompt_version == PROMPT_VERSION,
        )
    )
    session.add(
        DocumentSummary(
            document_text=document_text,
            model_name=config.model_name,
            endpoint=config.host,
            prompt_version=PROMPT_VERSION,
            summary=summary,
        )
    )


def rebuild_all_summaries(
    session: Session,
    config: SummaryConfig,
    client: SummaryClient | None = None,
    progress: Callable[[SummaryResult], None] | None = None,
    all_items: bool = False,
) -> SummaryResult:
    client = client or OllamaSummaryClient(config)
    result = SummaryResult()
    statement = select(DocumentText.id).order_by(DocumentText.id)
    if not all_items:
        statement = statement.where(
            ~select(DocumentSummary.id)
            .where(
                DocumentSummary.document_text_id == DocumentText.id,
                DocumentSummary.model_name == config.model_name,
                DocumentSummary.prompt_version == PROMPT_VERSION,
            )
            .exists()
        )
    ids = session.scalars(statement).all()
    for document_text_id in ids:
        document_text = session.scalar(
            select(DocumentText)
            .options(selectinload(DocumentText.chunks))
            .where(DocumentText.id == document_text_id)
        )
        if document_text is None or not document_text.processing_text.strip():
            result.skipped += 1
            if progress is not None:
                progress(result)
            continue
        try:
            summary = summarize_document(document_text, client, config)
            replace_document_summary(session, document_text, summary, config)
            session.commit()
            result.documents += 1
        except Exception:
            session.rollback()
            result.failed += 1
            logger.exception(f"Could not summarize document_text={document_text_id}")
        if progress is not None:
            progress(result)
    return result
