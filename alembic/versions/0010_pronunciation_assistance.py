"""Persist exact pronunciation occurrences and their evidence."""
from alembic import context, op
import sqlalchemy as sa

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('block_results', sa.Column('pronunciation_assistances_json', sa.JSON(none_as_null=True), nullable=True))


def downgrade() -> None:
    connection = op.get_bind()
    checks = [('block_results', 'pronunciation_assistances_json', 'нельзя удалить сохранённые произношения')]
    target = context.get_revision_argument()
    if target != '0009':
        checks.append(('glossary_candidates', 'name_translation_evidence_json',
                       'нельзя удалить сохранённые свидетельства передачи имён'))
    if target not in ('0009', '0008'):
        checks.append(('block_results', 'fragment_assistance_json',
                       'нельзя удалить сохранённую помощь к фрагментам'))
    if target not in ('0009', '0008', '0007', '0006'):
        checks.extend((table, column, 'нельзя удалить данные содержательного плана или свидетельств')
                      for table, column in (('chapter_analyses', 'content_plan_json'),
                                            ('block_results', 'content_evidence_json')))
    for table, column, message in checks:
        if connection.execute(sa.text(
                f'SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)')).scalar_one():
            raise RuntimeError(message)
    if connection.dialect.name == 'sqlite' and connection.dialect.server_version_info < (3, 35, 0):
        raise RuntimeError('для безопасного отката произношений нужен SQLite 3.35 или новее')
    op.drop_column('block_results', 'pronunciation_assistances_json')
