"""Store accepted results of individual chapter analysis passes."""

from alembic import op
import sqlalchemy as sa


revision = '0016'
down_revision = '0015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'analysis_pass_results',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete='CASCADE'), nullable=False),
        sa.Column('section_id', sa.String(), sa.ForeignKey('sections.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pass_kind', sa.String(), nullable=False),
        sa.Column('scope_key', sa.String(), nullable=False),
        sa.Column('input_sha256', sa.String(), nullable=False),
        sa.Column('output_sha256', sa.String(), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('attempt_id', sa.String(), nullable=True),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('book_id', 'section_id', 'pass_kind', 'scope_key'),
    )
    op.create_index('ix_analysis_pass_results_book_id', 'analysis_pass_results', ['book_id'])


def downgrade() -> None:
    op.drop_index('ix_analysis_pass_results_book_id', table_name='analysis_pass_results')
    op.drop_table('analysis_pass_results')
