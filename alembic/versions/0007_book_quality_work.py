"""Durable book conditions, recovery, and optional review work."""
from alembic import op, context
import sqlalchemy as sa

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('book_conditions',
        sa.Column('id', sa.String(), primary_key=True, nullable=False),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column('schema_version', sa.Integer(), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('reproducibility', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('book_id', 'revision'),
        sa.CheckConstraint("revision >= 0", name="revision_nonnegative"),
        sa.CheckConstraint("schema_version >= 0", name="schema_version_nonnegative"),
    )
    op.create_table('work_bindings',
        sa.Column('entity_kind', sa.String(), primary_key=True, nullable=False),
        sa.Column('entity_id', sa.String(), primary_key=True, nullable=False),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('artifact_revision_id', sa.String(), sa.ForeignKey('artifact_revisions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('payload_json', sa.JSON(), nullable=False),
    )
    op.create_table('artifact_revisions',
        sa.Column('id', sa.String(), primary_key=True, nullable=False),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('run_id', sa.String(), nullable=True),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('scope_id', sa.String(), nullable=False),
        sa.Column('body_json', sa.JSON(), nullable=False),
        sa.Column('sha256', sa.String(), nullable=False),
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_table('recovery_roots',
        sa.Column('id', sa.String(), primary_key=True, nullable=False),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('root_key', sa.String(), unique=True, nullable=False),
        sa.Column('manifest_json', sa.JSON(), nullable=False),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('calls_used', sa.Integer(), nullable=False),
        sa.Column('extra_calls', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('reason', sa.String(), nullable=True),
        sa.CheckConstraint("generation >= 0", name="generation_nonnegative"),
        sa.CheckConstraint("calls_used >= 0", name="calls_used_nonnegative"),
        sa.CheckConstraint("extra_calls >= 0", name="extra_calls_nonnegative"),
    )
    op.create_table('recovery_scopes',
        sa.Column('id', sa.String(), primary_key=True, nullable=False),
        sa.Column('root_id', sa.String(), sa.ForeignKey('recovery_roots.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('parent_id', sa.String(), sa.ForeignKey('recovery_scopes.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('ordered_blocks_json', sa.JSON(), nullable=False),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('accepted_attempt_id', sa.String(), sa.ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.CheckConstraint("generation >= 0", name="generation_nonnegative"),
    )
    op.create_table('llm_attempts',
        sa.Column('id', sa.String(), primary_key=True, nullable=False),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('task_id', sa.String(), nullable=True),
        sa.Column('run_id', sa.String(), nullable=True),
        sa.Column('root_id', sa.String(), sa.ForeignKey('recovery_roots.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('scope_id', sa.String(), sa.ForeignKey('recovery_scopes.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('review_id', sa.String(), sa.ForeignKey('quality_reviews.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('context_json', sa.JSON(), nullable=False),
        sa.Column('outcome_json', sa.JSON(), nullable=False),
        sa.UniqueConstraint('root_id', 'ordinal'),
        sa.UniqueConstraint('review_id', 'ordinal'),
        sa.CheckConstraint("generation >= 0", name="generation_nonnegative"),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
    )
    op.create_table('quality_reviews',
        sa.Column('id', sa.String(), primary_key=True, nullable=False),
        sa.Column('book_id', sa.String(), sa.ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('conditions_id', sa.String(), sa.ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True),
        sa.Column('request_key', sa.String(), nullable=False),
        sa.Column('manifest_json', sa.JSON(), nullable=False),
        sa.Column('settings_json', sa.JSON(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column('findings_json', sa.JSON(), nullable=False),
        sa.Column('budget_limit', sa.Integer(), nullable=False),
        sa.Column('spent', sa.Integer(), nullable=False),
        sa.Column('reserved', sa.Integer(), nullable=False),
        sa.UniqueConstraint('book_id', 'request_key'),
        sa.CheckConstraint("revision >= 0", name="revision_nonnegative"),
        sa.CheckConstraint("budget_limit >= 0", name="budget_limit_nonnegative"),
        sa.CheckConstraint("spent >= 0", name="spent_nonnegative"),
        sa.CheckConstraint("reserved >= 0", name="reserved_nonnegative"),
    )
    op.create_table('quality_reservations',
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), primary_key=True, nullable=False),
        sa.Column('review_id', sa.String(), sa.ForeignKey('quality_reviews.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column('estimate', sa.Integer(), nullable=False),
        sa.Column('actual', sa.Integer(), nullable=True),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('price_json', sa.JSON(), nullable=False),
        sa.CheckConstraint("estimate >= 0", name="estimate_nonnegative"),
        sa.CheckConstraint("actual >= 0", name="actual_nonnegative"),
    )
    op.add_column('llm_call_logs', sa.Column('attempt_id', sa.String(), nullable=True))
    op.add_column('llm_call_logs', sa.Column('diagnostic_meta_json', sa.JSON(), nullable=True))
    op.create_index('uq_llm_call_logs_attempt_id', 'llm_call_logs', ['attempt_id'], unique=True)


def downgrade():
    connection = op.get_bind()
    destination = context.get_revision_argument()
    if destination not in ('0006', '0007'):
        for table, column in (('chapter_analyses', 'content_plan_json'), ('block_results', 'content_evidence_json')):
            if connection.execute(sa.text(f'SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)')).scalar_one():
                raise RuntimeError('нельзя удалить данные содержательного плана или свидетельств')
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM book_conditions)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM work_bindings)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM artifact_revisions)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM recovery_roots)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM recovery_scopes)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM llm_attempts)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM quality_reviews)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM quality_reservations)")).scalar_one():
        raise RuntimeError("нельзя удалить сохранённую работу книги")
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM llm_call_logs WHERE attempt_id IS NOT NULL OR diagnostic_meta_json IS NOT NULL)")).scalar_one():
        raise RuntimeError('нельзя удалить диагностику работы книги')
    op.drop_index('uq_llm_call_logs_attempt_id', table_name='llm_call_logs')
    op.drop_column('llm_call_logs', 'diagnostic_meta_json')
    op.drop_column('llm_call_logs', 'attempt_id')
    op.drop_table('quality_reservations')
    op.drop_table('work_bindings')
    op.drop_table('artifact_revisions')
    op.drop_table('recovery_scopes')
    op.drop_table('llm_attempts')
    op.drop_table('recovery_roots')
    op.drop_table('quality_reviews')
    op.drop_table('book_conditions')
