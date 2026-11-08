"""Add foreign-key and lookup indexes for versioned work state tables."""

from alembic import op


revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_recovery_roots_book_id", "recovery_roots", ["book_id"])
    op.create_index("ix_recovery_scopes_root_id", "recovery_scopes", ["root_id"])
    op.create_index("ix_llm_attempts_scope_id", "llm_attempts", ["scope_id"])
    op.create_index("ix_llm_attempts_task_id", "llm_attempts", ["task_id"])
    op.create_index(
        "ix_work_bindings_book_id_entity_kind",
        "work_bindings",
        ["book_id", "entity_kind"],
    )
    op.create_index(
        "ix_artifact_revisions_book_id_run_id",
        "artifact_revisions",
        ["book_id", "run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_artifact_revisions_book_id_run_id", table_name="artifact_revisions")
    op.drop_index("ix_work_bindings_book_id_entity_kind", table_name="work_bindings")
    op.drop_index("ix_llm_attempts_task_id", table_name="llm_attempts")
    op.drop_index("ix_llm_attempts_scope_id", table_name="llm_attempts")
    op.drop_index("ix_recovery_scopes_root_id", table_name="recovery_scopes")
    op.drop_index("ix_recovery_roots_book_id", table_name="recovery_roots")
