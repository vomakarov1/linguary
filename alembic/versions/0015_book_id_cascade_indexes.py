"""Add book_id cascade indexes on correction_candidates, llm_attempts, and processing_runs."""

from alembic import op


revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_correction_candidates_book_id",
        "correction_candidates",
        ["book_id"],
    )
    op.create_index("ix_llm_attempts_book_id", "llm_attempts", ["book_id"])
    op.create_index("ix_processing_runs_book_id", "processing_runs", ["book_id"])


def downgrade() -> None:
    op.drop_index("ix_processing_runs_book_id", table_name="processing_runs")
    op.drop_index("ix_llm_attempts_book_id", table_name="llm_attempts")
    op.drop_index(
        "ix_correction_candidates_book_id",
        table_name="correction_candidates",
    )
