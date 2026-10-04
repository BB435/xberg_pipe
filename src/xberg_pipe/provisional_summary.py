"""LLMを使わない暫定的な抽出要約。"""

import re
from collections import Counter
from collections.abc import Sequence

from sqlalchemy import delete
from sqlalchemy.orm import Session

from xberg_pipe.chunking import clean_extracted_text
from xberg_pipe.models import DocumentSummary, DocumentText

MODEL_NAME = "extractive-ja-v1"
_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]?", re.MULTILINE)


def summarize_extractively(
    text: str, keywords: Sequence[tuple[str, float]], max_chars: int = 400
) -> str:
    """頻出文字列とキーフレーズで文を順位付けし、原文順に返す。"""

    sentences = [
        match.group().strip()
        for match in _SENTENCE.finditer(clean_extracted_text(text))
    ]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return ""
    if sum(map(len, sentences)) <= max_chars:
        return "".join(sentences)

    counts = Counter(
        sentence[index : index + 2]
        for sentence in sentences
        for index in range(len(sentence) - 1)
        if not sentence[index : index + 2].isspace()
    )
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (
            -(
                sum(score for word, score in keywords if word in sentences[index]) * 3
                + sum(
                    counts[sentences[index][i : i + 2]]
                    for i in range(len(sentences[index]) - 1)
                )
                / max(len(sentences[index]), 1)
                + (1 if index == 0 else 0)
            ),
            index,
        ),
    )
    chosen: list[int] = []
    length = 0
    for index in ranked:
        size = len(sentences[index])
        if length + size <= max_chars or not chosen:
            chosen.append(index)
            length += size
    return "".join(sentences[index] for index in sorted(chosen))[:max_chars]


def replace_provisional_summary(
    session: Session, document_text: DocumentText, summary: str
) -> None:
    session.flush()
    session.execute(
        delete(DocumentSummary).where(
            DocumentSummary.document_text_id == document_text.id,
            DocumentSummary.model_name == MODEL_NAME,
        )
    )
    session.add(
        DocumentSummary(
            document_text=document_text,
            model_name=MODEL_NAME,
            endpoint="local",
            prompt_version="extractive-v2",
            summary=summary,
        )
    )
