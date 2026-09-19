"""Initial schema: repositories, indexed files, doc sources, sync jobs.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repositories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("repo_name", sa.String(255), nullable=False, unique=True),
        # Bind-mounted working tree, not a clone URL: nothing is ever cloned, so
        # the mount path is the authoritative locator.
        sa.Column("mount_path", sa.Text(), nullable=False),
        sa.Column("origin_url", sa.Text(), nullable=True),
        sa.Column("head_sha", sa.String(64), nullable=True),
        sa.Column("last_synced_commit", sa.String(64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "indexed_files",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "repo_id",
            sa.Integer(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("repo_id", "file_path", name="uq_indexed_files_repo_path"),
    )
    # Every sync reads the full hash set for a repo.
    op.create_index("ix_indexed_files_repo_id", "indexed_files", ["repo_id"])

    op.create_table(
        "doc_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_url", sa.Text(), nullable=False, unique=True),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("framework", sa.String(64), nullable=True),
        sa.Column("version_tag", sa.String(32), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "sync_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("repo_name", sa.String(255), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "SUCCEEDED", "FAILED", name="sync_status"),
            nullable=False,
        ),
        sa.Column("phase", sa.String(64), nullable=False, server_default="queued"),
        sa.Column("files_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("files_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("files_skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks_upserted", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("chunks_deleted", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sync_jobs_repo_name", "sync_jobs", ["repo_name"])


def downgrade() -> None:
    op.drop_index("ix_sync_jobs_repo_name", table_name="sync_jobs")
    op.drop_table("sync_jobs")
    op.drop_table("doc_sources")
    op.drop_index("ix_indexed_files_repo_id", table_name="indexed_files")
    op.drop_table("indexed_files")
    op.drop_table("repositories")
    sa.Enum(name="sync_status").drop(op.get_bind(), checkfirst=True)
