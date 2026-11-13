"""Persist block segments and per-segment results."""

from alembic import op
import sqlalchemy as sa


revision = '0017'
down_revision = '0016'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'document_blocks',
        sa.Column('segments_json', sa.JSON(), nullable=False, server_default='[]'),
    )
    op.add_column(
        'treatment_plan_items',
        sa.Column('segment_treatments_json', sa.JSON(), nullable=True),
    )
    op.add_column(
        'block_results',
        sa.Column('segment_treatments_json', sa.JSON(), nullable=True),
    )
    op.add_column(
        'block_results',
        sa.Column('segment_translations_json', sa.JSON(), nullable=True),
    )
    op.add_column(
        'glossary_candidates',
        sa.Column('description', sa.Text(), nullable=False, server_default=''),
    )


def downgrade() -> None:
    op.drop_column('glossary_candidates', 'description')
    op.drop_column('block_results', 'segment_translations_json')
    op.drop_column('block_results', 'segment_treatments_json')
    op.drop_column('treatment_plan_items', 'segment_treatments_json')
    op.drop_column('document_blocks', 'segments_json')
