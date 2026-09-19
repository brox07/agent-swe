"""Relational state: what has been indexed, and how sync jobs progressed."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SyncStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Repository(Base):
    """A bind-mounted working tree.

    Departs from the spec's ``git_url``: nothing is cloned, so the authoritative
    locator is the mount path. The origin remote is recorded when one exists,
    for reference only.
    """

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    mount_path: Mapped[str] = mapped_column(Text, nullable=False)
    origin_url: Mapped[str | None] = mapped_column(Text)
    head_sha: Mapped[str | None] = mapped_column(String(64))
    last_synced_commit: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    files: Mapped[list[IndexedFile]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )


class IndexedFile(Base):
    """One row per indexed file, carrying the hash that drives incremental sync."""

    __tablename__ = "indexed_files"
    __table_args__ = (UniqueConstraint("repo_id", "file_path", name="uq_indexed_files_repo_path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    repository: Mapped[Repository] = relationship(back_populates="files")


class DocSource(Base):
    """One ingested documentation source: a book, an archive, a docs directory."""

    __tablename__ = "doc_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    framework: Mapped[str | None] = mapped_column(String(64))
    version_tag: Mapped[str | None] = mapped_column(String(32))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SyncJob(Base):
    """Progress record for an asynchronous sync.

    Indexing a real repository outlasts any MCP client timeout, so
    ``sync_repository`` returns one of these by id and the client polls
    ``get_sync_status``. Persisted so status survives a restart.
    """

    __tablename__ = "sync_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    repo_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[SyncStatus] = mapped_column(
        Enum(SyncStatus, name="sync_status"), default=SyncStatus.PENDING, nullable=False
    )
    phase: Mapped[str] = mapped_column(String(64), default="queued", nullable=False)
    files_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    files_done: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    files_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_upserted: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    chunks_deleted: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
