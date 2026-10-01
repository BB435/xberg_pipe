import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from xberg_pipe.chunking import clean_extracted_text
from xberg_pipe.models import DocumentSummary, DocumentText

PROMPT_VERSION = "ja-v1"
SYSTEM_PROMPT = (
    "あなたは日本語文書の要約者です。原文にない情報を加えず、固有名詞、数値、"
    "結論、重要な条件を保持して簡潔な日本語で要約してください。"
)


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    model_name: str
    endpoint: str = "http://127.0.0.1:11434/v1"
    api_key: str | None = None
    timeout_seconds: int = 120
    max_output_tokens: int = 800
    temperature: float = 0.1
    reduce_group_size: int = 8

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name is required")
        if self.timeout_seconds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")
        if self.reduce_group_size < 2:
            raise ValueError("reduce_group_size must be at least 2")


@dataclass(slots=True)
class SummaryResult:
    documents: int = 0
    failed: int = 0
    skipped: int = 0


class SummaryClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleClient:
    """Ollama、vLLM、llama.cpp等のOpenAI互換ローカルAPIクライアント."""

    def __init__(self, config: SummaryConfig) -> None:
        self.config = config

    def complete(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self.config.endpoint.rstrip("/") + "/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.config.timeout_seconds
        ) as response:
            payload = json.load(response)
        content = payload["choices"][0]["message"]["content"].strip()
        if not content:
            raise RuntimeError("ローカルLLMが空の要約を返しました")
        return content


def summarize_document(
    document_text: DocumentText,
    client: SummaryClient,
    config: SummaryConfig,
) -> str:
    chunks = [chunk.content for chunk in document_text.chunks]
    if not chunks:
        cleaned = clean_extracted_text(document_text.extracted_text)
        chunks = [cleaned] if cleaned else []
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
            endpoint=config.endpoint,
            prompt_version=PROMPT_VERSION,
            summary=summary,
        )
    )


def rebuild_all_summaries(
    session: Session,
    config: SummaryConfig,
    client: SummaryClient | None = None,
    progress: Callable[[SummaryResult], None] | None = None,
) -> SummaryResult:
    client = client or OpenAICompatibleClient(config)
    result = SummaryResult()
    ids = session.scalars(select(DocumentText.id).order_by(DocumentText.id)).all()
    for document_text_id in ids:
        document_text = session.scalar(
            select(DocumentText)
            .options(selectinload(DocumentText.chunks))
            .where(DocumentText.id == document_text_id)
        )
        if document_text is None or not document_text.extracted_text.strip():
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
