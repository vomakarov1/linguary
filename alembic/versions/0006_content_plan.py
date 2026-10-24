"""Optional content plan and educational assistance evidence."""

from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('chapter_analyses', sa.Column('content_plan_json', sa.JSON(), nullable=True))
    op.add_column('block_results', sa.Column('content_evidence_json', sa.JSON(), nullable=True))


def downgrade() -> None:
    connection = op.get_bind()
    for table, column in (
        ('chapter_analyses', 'content_plan_json'),
        ('block_results', 'content_evidence_json'),
    ):
        present = connection.execute(sa.text(
            f'SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)'
        )).scalar_one()
        if present:
            raise RuntimeError('нельзя удалить данные содержательного плана или свидетельств')
    if connection.dialect.name == 'sqlite' and connection.dialect.server_version_info < (3, 35, 0):
        raise RuntimeError('для безопасного отката содержательного плана нужен SQLite 3.35 или новее')
    # Recreating chapter_analyses in batch mode removes child rows via CASCADE.
    op.drop_column('block_results', 'content_evidence_json')
    op.drop_column('chapter_analyses', 'content_plan_json')
