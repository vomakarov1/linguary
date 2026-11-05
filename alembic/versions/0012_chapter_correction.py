"""Persist current chapter correction checks without changing historical runs."""

from alembic import op
import sqlalchemy as sa


revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "correction_checks",
        sa.Column("run_id", sa.String(255), nullable=False),
        sa.Column("block_id", sa.String(255), nullable=False),
        sa.Column("section_id", sa.String(255), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("attempt_id", sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "block_id"),
        sa.ForeignKeyConstraint(["run_id"], ["processing_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["block_id"], ["document_blocks.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_correction_checks_run_section", "correction_checks", ["run_id", "section_id"])


def downgrade() -> None:
    op.drop_index("ix_correction_checks_run_section", table_name="correction_checks")
    op.drop_table("correction_checks")
