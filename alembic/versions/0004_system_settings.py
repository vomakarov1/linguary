"""Постоянные настройки приложения."""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'system_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column('overrides_json', sa.JSON(none_as_null=True), nullable=True),
        sa.Column('api_key', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_system_settings'),
        sa.CheckConstraint('id = 1', name=op.f('ck_system_settings_singleton')),
        sa.CheckConstraint('revision > 0', name=op.f('ck_system_settings_revision_positive')),
    )


def downgrade() -> None:
    op.drop_table('system_settings')
