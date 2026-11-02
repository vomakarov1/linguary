"""Persist translation operation policy without rewriting historical work."""

from alembic import context, op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

OLD_TASK_KEY = "uq_processing_tasks_book_id_task_type_scope_id"
NEW_TASK_KEY = "uq_processing_tasks_book_id_task_type_scope_id_operation_id_policy_version"


def upgrade() -> None:
    op.add_column("recovery_roots", sa.Column("policy_version", sa.String(), nullable=True))
    op.add_column("recovery_roots", sa.Column(
        "operation_kind", sa.String(), nullable=False, server_default="translation",
    ))
    # Recreating this parent table would cascade-delete scopes and attempts.
    # SQLite permits a nullable inline reference in ADD COLUMN without a copy.
    op.execute(sa.text(
        "ALTER TABLE recovery_roots ADD COLUMN superseded_by VARCHAR "
        "CONSTRAINT fk_recovery_roots_superseded_by_recovery_roots "
        "REFERENCES recovery_roots (id)"
    ))
    op.add_column("recovery_roots", sa.Column(
        "compatibility_json", sa.JSON(none_as_null=True), nullable=True,
    ))
    with op.batch_alter_table("processing_tasks", recreate="always") as batch:
        batch.add_column(sa.Column("operation_id", sa.String(), nullable=False, server_default=sa.text("''")))
        batch.add_column(sa.Column("policy_version", sa.String(), nullable=False, server_default=sa.text("''")))
        batch.drop_constraint(OLD_TASK_KEY, type_="unique")
        batch.create_unique_constraint(NEW_TASK_KEY, [
            "book_id", "task_type", "scope_id", "operation_id", "policy_version",
        ])
    op.execute(sa.text(
        "UPDATE processing_tasks SET operation_id = 'dispatch:' || id "
        "WHERE task_type = 'process_translation_unit'"
    ))
    op.execute(sa.text(
        "UPDATE processing_tasks SET operation_id = id, policy_version = 'translation-context-legacy-v1' "
        "WHERE task_type = 'regenerate_block'"
    ))


def _refuse_earlier_data_loss(connection) -> None:
    """Preserve older migrations' refusal before this revision performs any DDL."""
    target = context.get_revision_argument()
    if target == "0010":
        return
    checks = [
        ("block_results", "pronunciation_assistances_json", "нельзя удалить сохранённые произношения"),
    ]
    if target != "0009":
        checks.append(("glossary_candidates", "name_translation_evidence_json",
                       "нельзя удалить сохранённые свидетельства передачи имён"))
    if target not in ("0009", "0008"):
        checks.append(("block_results", "fragment_assistance_json",
                       "нельзя удалить сохранённую помощь к фрагментам"))
    if target not in ("0009", "0008", "0007", "0006"):
        checks.extend((table, column, "нельзя удалить данные содержательного плана или свидетельств")
                      for table, column in (("chapter_analyses", "content_plan_json"),
                                            ("block_results", "content_evidence_json")))
    for table, column, message in checks:
        if connection.execute(sa.text(
            f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)"
        )).scalar_one():
            raise RuntimeError(message)
    if target in ("0009", "0008", "0007"):
        return
    for table in ("book_conditions", "work_bindings", "artifact_revisions", "recovery_roots",
                  "recovery_scopes", "llm_attempts", "quality_reviews", "quality_reservations"):
        if connection.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar_one():
            raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM llm_call_logs "
        "WHERE attempt_id IS NOT NULL OR diagnostic_meta_json IS NOT NULL)"
    )).scalar_one():
        raise RuntimeError("нельзя удалить диагностику работы книги")


def downgrade() -> None:
    connection = op.get_bind()
    for query in (
        "SELECT EXISTS (SELECT 1 FROM recovery_roots WHERE policy_version IS NOT NULL "
        "OR operation_kind <> 'translation' OR superseded_by IS NOT NULL "
        "OR compatibility_json IS NOT NULL OR status = 'superseded')",
        "SELECT EXISTS (SELECT 1 FROM processing_tasks WHERE operation_id <> '' "
        "OR policy_version <> '' OR status = 'superseded')",
    ):
        if connection.execute(sa.text(query)).scalar_one():
            raise RuntimeError("нельзя удалить сохранённую историю операций перевода")
    _refuse_earlier_data_loss(connection)
    if connection.dialect.name == "sqlite" and connection.dialect.server_version_info < (3, 35, 0):
        raise RuntimeError("для безопасного отката политики нужен SQLite 3.35 или новее")
    with op.batch_alter_table("processing_tasks", recreate="always") as batch:
        batch.drop_constraint(NEW_TASK_KEY, type_="unique")
        batch.drop_column("operation_id")
        batch.drop_column("policy_version")
        batch.create_unique_constraint(OLD_TASK_KEY, ["book_id", "task_type", "scope_id"])
    for column in ("compatibility_json", "superseded_by", "operation_kind", "policy_version"):
        op.drop_column("recovery_roots", column)
