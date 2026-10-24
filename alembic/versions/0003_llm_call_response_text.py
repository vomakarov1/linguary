"""Debug text of LLM responses.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add a dedicated field for the model response."""
    with op.batch_alter_table("llm_call_logs") as batch_op:
        batch_op.add_column(sa.Column("response_text", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove the model response field."""
    with op.batch_alter_table("llm_call_logs") as batch_op:
        batch_op.drop_column("response_text")
