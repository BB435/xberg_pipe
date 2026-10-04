import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql.selectable import Select

from xberg_pipe.chunking import chunk_text
from xberg_pipe.models import DocumentKeyword, DocumentText

DEFAULT_MODEL = "sirasagi62/ruri-v3-30m-ONNX"
RURI_MODELS = (DEFAULT_MODEL,)
TOPIC_PREFIX = "トピック: "
LIGHT_MODEL = "ja-yake-v1"

_TARGET_POS = {"名詞", "形容詞"}
_STOP_WORDS = {
    "こと",
    "これ",
    "それ",
    "ため",
    "もの",
    "よう",
    "ところ",
    "場合",
    "以下",
    "以上",
}


@dataclass(frozen=True, slots=True)
class KeywordConfig:
    model_name: str = DEFAULT_MODEL
    top_n: int = 10
    max_ngram: int = 3
    max_candidates: int = 500
    diversity: float = 0.35
    device: str | None = None

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.max_ngram <= 0:
            raise ValueError("max_ngram must be positive")
        if self.max_candidates < self.top_n:
            raise ValueError("max_candidates must be at least top_n")
        if not 0.0 <= self.diversity <= 1.0:
            raise ValueError("diversity must be between 0 and 1")


@dataclass(slots=True)
class KeywordResult:
    documents: int = 0
    keywords: int = 0
    skipped: int = 0


class KeywordExtractor(Protocol):
    def extract(self, text: str, top_n: int) -> list[tuple[str, float]]: ...


_JAPANESE_SPACES = re.compile(
    r"(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])"
)


class YakeKeywordExtractor:
    """日本語を分かち書きしてYAKEでキーフレーズを選ぶ。"""

    def __init__(self, max_ngram: int = 3) -> None:
        from fugashi import Tagger
        from yake import KeywordExtractor

        self._tagger = Tagger()
        self._extractor_type = KeywordExtractor
        self._max_ngram = max_ngram

    def extract(self, text: str, top_n: int) -> list[tuple[str, float]]:
        phrases: list[str] = []
        parts: list[str] = []
        for word in self._tagger(text):
            surface = word.surface.strip()
            if surface and getattr(word.feature, "pos1", "") in _TARGET_POS:
                parts.append(surface)
            elif parts:
                phrases.append("".join(parts))
                parts.clear()
        if parts:
            phrases.append("".join(parts))
        segmented = " ".join(phrases)
        if not segmented:
            return []
        extractor = self._extractor_type(lan="ja", n=self._max_ngram, top=top_n)
        keywords = extractor.extract_keywords(segmented)
        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        for phrase, score in keywords:
            phrase = _JAPANESE_SPACES.sub("", phrase).strip()
            if len(phrase) < 2 or phrase in seen or phrase not in text:
                continue
            seen.add(phrase)
            results.append((phrase, round(1 / (1 + float(score)), 6)))
        return results


class RuriKeyBertExtractor:
    """Ruri v3と日本語形態素候補を利用するKeyBERT抽出器."""

    def __init__(self, config: KeywordConfig) -> None:
        try:
            import numpy as np
            from fugashi import Tagger
            from keybert import KeyBERT
            from keybert.backend import BaseEmbedder
            from sklearn.feature_extraction.text import CountVectorizer

            from xberg_pipe.fastembed_model import create_ruri_model
        except ImportError as error:
            raise RuntimeError(
                "キーワード抽出依存がありません。uv sync --extra keywords を実行してください。"
            ) from error

        class RuriTopicEmbedder(BaseEmbedder):
            def __init__(self, model) -> None:
                super().__init__()
                self.model = model

            def embed(self, documents, verbose: bool = False):
                topical_documents = [TOPIC_PREFIX + document for document in documents]
                return np.stack(list(self.model.embed(topical_documents)))

        model = create_ruri_model(config.model_name, config.device)
        self._keybert = KeyBERT(model=RuriTopicEmbedder(model))
        self._tagger = Tagger()
        self._vectorizer_type = CountVectorizer
        self._config = config

    def _candidates(self, text: str) -> list[str]:
        tokens = [
            (word.surface.strip(), getattr(word.feature, "pos1", ""))
            for word in self._tagger(text)
        ]
        counts: Counter[str] = Counter()
        for start in range(len(tokens)):
            parts: list[str] = []
            for surface, pos in tokens[start : start + self._config.max_ngram]:
                if pos not in _TARGET_POS or not surface:
                    break
                parts.append(surface)
                candidate = "".join(parts)
                if (
                    len(candidate) >= 2
                    and candidate not in _STOP_WORDS
                    and not candidate.isnumeric()
                ):
                    counts[candidate] += 1
        return [word for word, _ in counts.most_common(self._config.max_candidates)]

    def extract(self, text: str, top_n: int) -> list[tuple[str, float]]:
        candidates = self._candidates(text)
        if not candidates:
            return []
        return self._keybert.extract_keywords(
            text,
            vectorizer=self._vectorizer_type(vocabulary=candidates, lowercase=False),
            top_n=min(top_n, len(candidates)),
            use_mmr=True,
            diversity=self._config.diversity,
        )


def aggregate_chunk_keywords(
    keyword_sets: Iterable[Sequence[tuple[str, float]]], top_n: int
) -> list[tuple[str, float]]:
    """チャンク別スコアを出現範囲も考慮してファイル単位へ集約する."""

    scores: dict[str, list[float]] = defaultdict(list)
    for keywords in keyword_sets:
        for keyword, score in keywords:
            scores[keyword].append(float(score))

    # 最大類似度を主とし、複数チャンクに現れる語をわずかに優遇する。
    aggregated = [
        (keyword, min(1.0, round(max(values) + min(len(values) - 1, 4) * 0.01, 6)))
        for keyword, values in scores.items()
    ]
    return sorted(aggregated, key=lambda item: (-item[1], item[0]))[:top_n]


def extract_document_keywords(
    document_text: DocumentText,
    extractor: KeywordExtractor,
    config: KeywordConfig,
) -> list[tuple[str, float]]:
    texts = (chunk.content for chunk in document_text.chunks)
    if not document_text.chunks:
        # 未チャンキング文書の全文をKeyBERTへ渡すとメモリ使用量が急増する。
        texts = iter(chunk_text(document_text.extracted_text))
    per_chunk_top_n = max(config.top_n * 3, config.top_n)
    return aggregate_chunk_keywords(
        (extractor.extract(text, per_chunk_top_n) for text in texts), config.top_n
    )


def replace_document_keywords(
    session: Session,
    document_text: DocumentText,
    keywords: Sequence[tuple[str, float]],
    model_name: str,
) -> int:
    session.flush()
    session.execute(
        delete(DocumentKeyword).where(
            DocumentKeyword.document_text_id == document_text.id,
            DocumentKeyword.model_name == model_name,
        )
    )
    session.add_all(
        DocumentKeyword(
            document_text=document_text,
            model_name=model_name,
            rank=rank,
            keyword=keyword,
            score=score,
        )
        for rank, (keyword, score) in enumerate(keywords, start=1)
    )
    return len(keywords)


def rebuild_all_keywords(
    session: Session,
    config: KeywordConfig | None = None,
    batch_size: int = 20,
    extractor: RuriKeyBertExtractor | None = None,
    progress: Callable[[KeywordResult], None] | None = None,
) -> KeywordResult:
    """保存済み文書をKeyBERTで処理し、ファイル単位のキーワードを保存する."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config: KeywordConfig = config or KeywordConfig()
    extractor: RuriKeyBertExtractor = extractor or RuriKeyBertExtractor(config)
    result = KeywordResult()
    statement: Select[DocumentText] = (
        select(DocumentText)
        .options(selectinload(DocumentText.chunks))
        .order_by(DocumentText.id)
        .execution_options(yield_per=batch_size)
    )
    for document_text in session.scalars(statement):
        if not document_text.extracted_text.strip():
            result.skipped += 1
            if progress is not None:
                progress(result)
            continue
        keywords: list[tuple[str, float]] = extract_document_keywords(
            document_text, extractor, config
        )
        result.keywords += replace_document_keywords(
            session, document_text, keywords, config.model_name
        )
        result.documents += 1
        if progress is not None:
            progress(result)
        if result.documents % batch_size == 0:
            session.commit()
    session.commit()
    return result
