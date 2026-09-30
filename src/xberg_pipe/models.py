import datetime as dt

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    extension: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="hash")

    paths: Mapped[list[DocumentPath]] = relationship(
        back_populates="document", cascade="all, delete"
    )
    texts: Mapped[list[DocumentText]] = relationship(
        back_populates="document", cascade="all, delete"
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
        server_default=func.now(), server_onupdate=func.now()
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
    extracted_text: Mapped[str]
    """テキスト"""

    document: Mapped[Document] = relationship(back_populates="texts")
