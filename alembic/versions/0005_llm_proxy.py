"""Постоянные параметры прокси LLM."""
import json
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('system_settings', sa.Column('proxy_url', sa.Text(), nullable=True))


def downgrade() -> None:
    connection = op.get_bind()
    for row in connection.execute(sa.text('SELECT id, overrides_json FROM system_settings')):
        if row.overrides_json is not None:
            values = json.loads(row.overrides_json)
            values.pop('proxy_mode', None)
            connection.execute(sa.text('UPDATE system_settings SET overrides_json = :values WHERE id = :id'), {'values': json.dumps(values), 'id': row.id})
    with op.batch_alter_table('system_settings') as batch:
        batch.drop_column('proxy_url')
