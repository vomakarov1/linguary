"""Store immutable correction proposals and their review decisions."""

from alembic import op
import sqlalchemy as sa

revision = '0013'
down_revision = '0012'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('correction_inputs',
        sa.Column('run_id', sa.String(), sa.ForeignKey('processing_runs.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('block_id', sa.String(), sa.ForeignKey('document_blocks.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('fingerprint', sa.String(64), nullable=False),
    )
    op.create_table('correction_candidates',
        sa.Column('candidate_id', sa.String(255), primary_key=True),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', sa.String(), sa.ForeignKey('processing_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('section_id', sa.String(255), nullable=False),
        sa.Column('block_id', sa.String(), sa.ForeignKey('document_blocks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('input_sha256', sa.String(64), nullable=False),
        sa.Column('input_generation', sa.Integer(), nullable=False),
        sa.Column('proposal_sha256', sa.String(64), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('group_id', sa.String(255), nullable=False),
        sa.Column('proposal_attempt_id', sa.String(255), nullable=False),
        sa.Column('state', sa.String(32), nullable=False),
        sa.Column('review_attempt_id', sa.String(255), nullable=True),
        sa.Column('review_json', sa.JSON(none_as_null=True), nullable=True),
        sa.Column('decision_json', sa.JSON(none_as_null=True), nullable=True),
        sa.UniqueConstraint('run_id', 'block_id', 'input_generation'),
    )
    op.create_index('ix_correction_candidates_run_section_state', 'correction_candidates',
                    ['run_id', 'section_id', 'state'])
    with op.batch_alter_table('correction_checks') as batch:
        batch.add_column(sa.Column('basis', sa.String(32), nullable=False, server_default='legacy'))
        batch.add_column(sa.Column('candidate_id', sa.String(255), nullable=True))


def downgrade():
    with op.batch_alter_table('correction_checks') as batch:
        batch.drop_column('candidate_id')
        batch.drop_column('basis')
    op.drop_index('ix_correction_candidates_run_section_state', table_name='correction_candidates')
    op.drop_table('correction_candidates')
    op.drop_table('correction_inputs')
