"""Initial MVP schema (SDD SQL-01).

Books, canonical document model, glossary, chapter analyses, treatment plan, results,
learning notes, warnings, local tasks, and technical LLM call log.

Revision ID: 0001
Revises:
Create Date: 2026-08-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""
    op.create_table('books',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('authors_json', sa.JSON(), nullable=False),
    sa.Column('identifier', sa.String(), nullable=True),
    sa.Column('publisher', sa.String(), nullable=True),
    sa.Column('source_language', sa.String(), nullable=False),
    sa.Column('target_language', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('source_epub_path', sa.String(), nullable=False),
    sa.Column('export_validation_status', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_books'))
    )
    with op.batch_alter_table('books', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_books_status'), ['status'], unique=False)

    op.create_table('import_warnings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('section_id', sa.String(), nullable=True),
    sa.Column('block_id', sa.String(), nullable=True),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_import_warnings_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_import_warnings'))
    )
    with op.batch_alter_table('import_warnings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_import_warnings_book_id'), ['book_id'], unique=False)

    op.create_table('llm_call_logs',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('task_id', sa.String(), nullable=True),
    sa.Column('stage', sa.String(), nullable=False),
    sa.Column('provider', sa.String(), nullable=False),
    sa.Column('model', sa.String(), nullable=False),
    sa.Column('started_at', sa.DateTime(), nullable=False),
    sa.Column('duration_ms', sa.Integer(), nullable=False),
    sa.Column('attempt', sa.Integer(), nullable=False),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('error_type', sa.String(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('error_detail', sa.Text(), nullable=True),
    sa.Column('prompt_tokens', sa.Integer(), nullable=True),
    sa.Column('completion_tokens', sa.Integer(), nullable=True),
    sa.Column('cost', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_llm_call_logs_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_llm_call_logs'))
    )
    with op.batch_alter_table('llm_call_logs', schema=None) as batch_op:
        batch_op.create_index('ix_llm_call_logs_book_id_stage', ['book_id', 'stage'], unique=False)
        batch_op.create_index('ix_llm_call_logs_task_id', ['task_id'], unique=False)

    op.create_table('processing_runs',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('glossary_snapshot_json', sa.JSON(), nullable=False),
    sa.Column('llm_settings_json', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_processing_runs_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_processing_runs'))
    )
    with op.batch_alter_table('processing_runs', schema=None) as batch_op:
        batch_op.create_index('uq_processing_runs_active_book', ['book_id'], unique=True, sqlite_where=sa.text("status = 'active'"))

    op.create_table('processing_tasks',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('task_type', sa.String(), nullable=False),
    sa.Column('scope_id', sa.String(), server_default='', nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('error_type', sa.String(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('error_detail', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_processing_tasks_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_processing_tasks')),
    sa.UniqueConstraint('book_id', 'task_type', 'scope_id', name=op.f('uq_processing_tasks_book_id_task_type_scope_id'))
    )
    with op.batch_alter_table('processing_tasks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_processing_tasks_status'), ['status'], unique=False)

    op.create_table('resources',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('source_path', sa.String(), nullable=False),
    sa.Column('media_type', sa.String(), nullable=False),
    sa.Column('stored_path', sa.String(), nullable=True),
    sa.Column('support_status', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_resources_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_resources')),
    sa.UniqueConstraint('book_id', 'source_path', name=op.f('uq_resources_book_id_source_path'))
    )
    op.create_table('sections',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('order', sa.Integer(), nullable=False),
    sa.Column('source_resource_path', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_sections_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_sections')),
    sa.UniqueConstraint('book_id', 'order', name=op.f('uq_sections_book_id_order'))
    )
    op.create_table('term_concepts',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('canonical_source_text', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_term_concepts_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_term_concepts'))
    )
    with op.batch_alter_table('term_concepts', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_term_concepts_book_id'), ['book_id'], unique=False)

    op.create_table('chapter_analyses',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('section_id', sa.String(), nullable=False),
    sa.Column('coverage_status', sa.String(), nullable=False),
    sa.Column('translation_instructions', sa.Text(), nullable=False),
    sa.Column('chapter_summary', sa.Text(), nullable=False),
    sa.Column('before_you_read_json', sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_chapter_analyses_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['section_id'], ['sections.id'], name=op.f('fk_chapter_analyses_section_id_sections'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chapter_analyses')),
    sa.UniqueConstraint('book_id', 'section_id', name=op.f('uq_chapter_analyses_book_id_section_id'))
    )
    op.create_table('document_blocks',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('book_id', sa.String(), nullable=False),
    sa.Column('section_id', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('order', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('inline_annotations_json', sa.JSON(), nullable=False),
    sa.Column('attributes_json', sa.JSON(), nullable=False),
    sa.Column('parent_id', sa.String(), nullable=True),
    sa.Column('source_reference_json', sa.JSON(), nullable=False),
    sa.Column('support_status', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_document_blocks_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['parent_id'], ['document_blocks.id'], name=op.f('fk_document_blocks_parent_id_document_blocks'), ondelete='CASCADE', initially='DEFERRED', deferrable=True),
    sa.ForeignKeyConstraint(['section_id'], ['sections.id'], name=op.f('fk_document_blocks_section_id_sections'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document_blocks')),
    sa.UniqueConstraint('section_id', 'order', name=op.f('uq_document_blocks_section_id_order'))
    )
    with op.batch_alter_table('document_blocks', schema=None) as batch_op:
        batch_op.create_index('ix_document_blocks_book_id_section_id', ['book_id', 'section_id'], unique=False)

    op.create_table('term_variants',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('concept_id', sa.String(), nullable=False),
    sa.Column('language', sa.String(), nullable=False),
    sa.Column('text', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['concept_id'], ['term_concepts.id'], name=op.f('fk_term_variants_concept_id_term_concepts'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_term_variants'))
    )
    with op.batch_alter_table('term_variants', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_term_variants_concept_id'), ['concept_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_term_variants_language'), ['language'], unique=False)

    op.create_table('block_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('block_id', sa.String(), nullable=False),
    sa.Column('treatment', sa.String(), nullable=False),
    sa.Column('translated_text', sa.Text(), nullable=True),
    sa.Column('glossary_hints_json', sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(['block_id'], ['document_blocks.id'], name=op.f('fk_block_results_block_id_document_blocks'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['processing_runs.id'], name=op.f('fk_block_results_run_id_processing_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_block_results')),
    sa.UniqueConstraint('run_id', 'block_id', name=op.f('uq_block_results_run_id_block_id'))
    )
    op.create_table('glossary_candidates',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('analysis_id', sa.String(), nullable=False),
    sa.Column('source_text', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('suggested_translation', sa.String(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('model_flagged_review', sa.Boolean(), nullable=False),
    sa.Column('requires_review', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['chapter_analyses.id'], name=op.f('fk_glossary_candidates_analysis_id_chapter_analyses'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_glossary_candidates'))
    )
    with op.batch_alter_table('glossary_candidates', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_glossary_candidates_analysis_id'), ['analysis_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_glossary_candidates_requires_review'), ['requires_review'], unique=False)

    op.create_table('language_notes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('block_id', sa.String(), nullable=False),
    sa.Column('note_id', sa.String(), nullable=False),
    sa.Column('original_excerpt', sa.Text(), nullable=False),
    sa.Column('translated_excerpt', sa.Text(), nullable=True),
    sa.Column('note', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['block_id'], ['document_blocks.id'], name=op.f('fk_language_notes_block_id_document_blocks'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['processing_runs.id'], name=op.f('fk_language_notes_run_id_processing_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_language_notes')),
    sa.UniqueConstraint('run_id', 'note_id', name=op.f('uq_language_notes_run_id_note_id'))
    )
    with op.batch_alter_table('language_notes', schema=None) as batch_op:
        batch_op.create_index('ix_language_notes_run_id_block_id', ['run_id', 'block_id'], unique=False)

    op.create_table('result_warnings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('block_id', sa.String(), nullable=False),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['block_id'], ['document_blocks.id'], name=op.f('fk_result_warnings_block_id_document_blocks'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['processing_runs.id'], name=op.f('fk_result_warnings_run_id_processing_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_result_warnings'))
    )
    with op.batch_alter_table('result_warnings', schema=None) as batch_op:
        batch_op.create_index('ix_result_warnings_run_id_block_id', ['run_id', 'block_id'], unique=False)

    op.create_table('treatment_plan_items',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('analysis_id', sa.String(), nullable=False),
    sa.Column('block_id', sa.String(), nullable=False),
    sa.Column('treatment', sa.String(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['chapter_analyses.id'], name=op.f('fk_treatment_plan_items_analysis_id_chapter_analyses'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_treatment_plan_items')),
    sa.UniqueConstraint('analysis_id', 'block_id', name=op.f('uq_treatment_plan_items_analysis_id_block_id'))
    )


def downgrade() -> None:
    """Roll back the migration."""
    op.drop_table('treatment_plan_items')
    with op.batch_alter_table('result_warnings', schema=None) as batch_op:
        batch_op.drop_index('ix_result_warnings_run_id_block_id')

    op.drop_table('result_warnings')
    with op.batch_alter_table('language_notes', schema=None) as batch_op:
        batch_op.drop_index('ix_language_notes_run_id_block_id')

    op.drop_table('language_notes')
    with op.batch_alter_table('glossary_candidates', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_glossary_candidates_requires_review'))
        batch_op.drop_index(batch_op.f('ix_glossary_candidates_analysis_id'))

    op.drop_table('glossary_candidates')
    op.drop_table('block_results')
    with op.batch_alter_table('term_variants', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_term_variants_language'))
        batch_op.drop_index(batch_op.f('ix_term_variants_concept_id'))

    op.drop_table('term_variants')
    with op.batch_alter_table('document_blocks', schema=None) as batch_op:
        batch_op.drop_index('ix_document_blocks_book_id_section_id')

    op.drop_table('document_blocks')
    op.drop_table('chapter_analyses')
    with op.batch_alter_table('term_concepts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_term_concepts_book_id'))

    op.drop_table('term_concepts')
    op.drop_table('sections')
    op.drop_table('resources')
    with op.batch_alter_table('processing_tasks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_processing_tasks_status'))

    op.drop_table('processing_tasks')
    with op.batch_alter_table('processing_runs', schema=None) as batch_op:
        batch_op.drop_index('uq_processing_runs_active_book', sqlite_where=sa.text("status = 'active'"))

    op.drop_table('processing_runs')
    with op.batch_alter_table('llm_call_logs', schema=None) as batch_op:
        batch_op.drop_index('ix_llm_call_logs_task_id')
        batch_op.drop_index('ix_llm_call_logs_book_id_stage')

    op.drop_table('llm_call_logs')
    with op.batch_alter_table('import_warnings', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_import_warnings_book_id'))

    op.drop_table('import_warnings')
    with op.batch_alter_table('books', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_books_status'))

    op.drop_table('books')
