"""LLMを使わない暫定的な抽出要約。"""

import re
from collections.abc import Sequence

from ja_stopword_filter import JaStopwordFilter
from sqlalchemy import delete
from sqlalchemy.orm import Session
from sudachipy import Dictionary, SplitMode
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from xberg_pipe.chunking import clean_extracted_text
from xberg_pipe.models import DocumentSummary, DocumentText

MODEL_NAME = "sumy-lexrank-ja-v1"
PROMPT_VERSION = "sumy-lexrank-v1"
LEGACY_MODEL_NAME = "extractive-ja-v1"
_SENTENCE = re.compile(r"[^。！？!?]+[。！？!?]*")


class RankedLexRankSummarizer(LexRankSummarizer):
    """LexRankの評価順を文字数制限付きの選択に渡す。"""

    @staticmethod
    def _get_best_sentences(sentences, count, rating):
        return tuple(
            sorted(sentences, key=lambda sentence: rating[sentence], reverse=True)
        )


class SudachiTokenizer(Tokenizer):
    """sumyの日本語文分割とSudachiPyの分かち書きを組み合わせる。"""

    def __init__(self) -> None:
        super().__init__("japanese")
        self._sudachi = Dictionary(dict="full").tokenizer()
        self._stopwords = JaStopwordFilter()

    def to_sentences(self, paragraph: str) -> tuple[str, ...]:
        return tuple(
            sentence.strip()
            for match in _SENTENCE.finditer(paragraph)
            if (sentence := match.group().strip())
        )

    def to_words(self, sentence: str) -> tuple[str, ...]:
        words = (
            word.normalized_form()
            for word in self._sudachi.tokenize(sentence, SplitMode.C)
            if word.part_of_speech()[0] in {"名詞", "形容詞", "動詞"}
        )
        return tuple(word for word in words if self._stopwords.remove([word]))


def summarize_extractively(
    text: str, keywords: Sequence[tuple[str, float]], max_chars: int = 400
) -> str:
    """sumyのLexRankで文を選び、原文順に返す。"""

    cleaned = clean_extracted_text(text)
    if not cleaned or max_chars <= 0:
        return ""
    parser = PlaintextParser.from_string(cleaned, SudachiTokenizer())
    sentences = [str(sentence).strip() for sentence in parser.document.sentences]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return ""
    if sum(map(len, sentences)) <= max_chars:
        return "".join(sentences)

    summarizer = RankedLexRankSummarizer()
    ranked_sentences = list(summarizer(parser.document, len(sentences)))
    ranks = {id(sentence): rank for rank, sentence in enumerate(ranked_sentences)}
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (
            -sum(score for word, score in keywords if word in sentences[index]),
            ranks[id(parser.document.sentences[index])],
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
            DocumentSummary.model_name.in_((MODEL_NAME, LEGACY_MODEL_NAME)),
        )
    )
    session.add(
        DocumentSummary(
            document_text=document_text,
            model_name=MODEL_NAME,
            endpoint="local",
            prompt_version=PROMPT_VERSION,
            summary=summary,
        )
    )
