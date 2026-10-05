import datetime as dt

from sqlalchemy import Float, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    extension: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="hash")

    paths: Mapped[list[DocumentPath]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    texts: Mapped[list[DocumentText]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentPath(Base):
    __tablename__ = "document_paths"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(unique=True)
    file_size: Mapped[int]
    modified_at: Mapped[dt.datetime]
    updated_at: Mapped[dt.datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="paths")


class DocumentText(Base):
    __tablename__ = "document_texts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    __table_args__ = (
        UniqueConstraint("document_id", "extractor", name="uq_document_id_extractor"),
        Index("idx_document_id_extractor", "document_id", "extractor"),
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    extractor: Mapped[str] = mapped_column(String(20))
    """抽出機"""
    extracted_text: Mapped[str] = mapped_column(Text)
    """抽出された原文"""
    cleaned_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    """後工程で使用するクリーニング済み本文"""

    @property
    def processing_text(self) -> str:
        return (
            self.cleaned_text if self.cleaned_text is not None else self.extracted_text
        )

    document: Mapped[Document] = relationship(back_populates="texts")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document_text",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )
    keywords: Mapped[list[DocumentKeyword]] = relationship(
        back_populates="document_text",
        cascade="all, delete-orphan",
        order_by="DocumentKeyword.rank",
    )
    summaries: Mapped[list[DocumentSummary]] = relationship(
        back_populates="document_text", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """LLM・キーワード抽出用のクリーニング済みテキスト断片."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_text_id", "chunk_index", name="uq_document_text_chunk_index"
        ),
        Index("idx_document_chunks_document_text", "document_text_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_text_id: Mapped[int] = mapped_column(
        ForeignKey("document_texts.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    character_count: Mapped[int]
    chunker_version: Mapped[str] = mapped_column(String(20))

    document_text: Mapped[DocumentText] = relationship(back_populates="chunks")
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class DocumentKeyword(Base):
    """抽出テキスト全体に対するKeyBERTキーワード."""

    __tablename__ = "document_keywords"
    __table_args__ = (
        UniqueConstraint(
            "document_text_id", "model_name", "rank", name="uq_text_model_rank"
        ),
        Index("idx_document_keywords_document_text", "document_text_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_text_id: Mapped[int] = mapped_column(
        ForeignKey("document_texts.id", ondelete="CASCADE")
    )
    model_name: Mapped[str] = mapped_column(String(200))
    rank: Mapped[int]
    keyword: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)

    document_text: Mapped[DocumentText] = relationship(back_populates="keywords")


class DocumentSummary(Base):
    """ローカルLLMで生成したファイル単位の要約."""

    __tablename__ = "document_summaries"
    __table_args__ = (
        UniqueConstraint(
            "document_text_id", "model_name", "prompt_version", name="uq_text_summary"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_text_id: Mapped[int] = mapped_column(
        ForeignKey("document_texts.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(200))
    endpoint: Mapped[str] = mapped_column(String(500))
    prompt_version: Mapped[str] = mapped_column(String(20))
    summary: Mapped[str] = mapped_column(Text)

    document_text: Mapped[DocumentText] = relationship(back_populates="summaries")


class ChunkEmbedding(Base):
    """sqlite-vec内のベクトルに対応する生成情報."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", "model_name", name="uq_chunk_embedding_model"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(200), index=True)
    dimensions: Mapped[int]

    chunk: Mapped[DocumentChunk] = relationship(back_populates="embeddings")
