"""Optional typed assistance for source fragments."""
from alembic import context, op
import sqlalchemy as sa

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('block_results', sa.Column('fragment_assistance_json', sa.JSON(), nullable=True))


def downgrade() -> None:
    connection = op.get_bind()
    if connection.execute(sa.text(
        'SELECT EXISTS (SELECT 1 FROM block_results WHERE fragment_assistance_json IS NOT NULL)'
    )).scalar_one():
        raise RuntimeError('нельзя удалить сохранённую помощь к фрагментам')
    # Preserve the content migration's refusal before any destructive DDL.
    if context.get_revision_argument() not in ('0006', '0007'):
        for table, column in (('chapter_analyses', 'content_plan_json'), ('block_results', 'content_evidence_json')):
            if connection.execute(sa.text(
                f'SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)'
            )).scalar_one():
                raise RuntimeError('нельзя удалить данные содержательного плана или свидетельств')
    if connection.dialect.name == 'sqlite' and connection.dialect.server_version_info < (3, 35, 0):
        raise RuntimeError('для безопасного отката помощи нужен SQLite 3.35 или новее')
    op.drop_column('block_results', 'fragment_assistance_json')
