"""doc_sources: title and chunk count

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("doc_sources", sa.Column("title", sa.Text(), nullable=True))
    op.add_column(
        "doc_sources",
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("doc_sources", "chunk_count")
    op.drop_column("doc_sources", "title")
