"""Persist name translation assessment observations on glossary candidates."""

from alembic import context, op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "glossary_candidates",
        sa.Column("name_translation_evidence_json", sa.JSON(none_as_null=True), nullable=True),
    )


def downgrade() -> None:
    connection = op.get_bind()
    if connection.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM glossary_candidates WHERE name_translation_evidence_json IS NOT NULL)"
    )).scalar_one():
        raise RuntimeError("нельзя удалить сохранённые свидетельства передачи имён")
    target = context.get_revision_argument()
    if target != "0008" and connection.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM block_results WHERE fragment_assistance_json IS NOT NULL)"
    )).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую помощь к фрагментам")
    if target not in ("0008", "0007", "0006"):
        for table, column in (
            ("chapter_analyses", "content_plan_json"),
            ("block_results", "content_evidence_json"),
        ):
            if connection.execute(sa.text(
                f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)"
            )).scalar_one():
                raise RuntimeError("нельзя удалить данные содержательного плана или свидетельств")
    if connection.dialect.name == "sqlite" and connection.dialect.server_version_info < (3, 35, 0):
        raise RuntimeError("для безопасного отката свидетельств нужен SQLite 3.35 или новее")
    op.drop_column("glossary_candidates", "name_translation_evidence_json")
