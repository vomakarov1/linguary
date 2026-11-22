"""Persist per-run chapter note-number counters and backfill issued ordinals."""

import json
import re

from alembic import op
import sqlalchemy as sa


revision = '0018'
down_revision = '0017'
branch_labels = None
depends_on = None

_NOTE_ID = re.compile(r'^ln-ch(\d+)-(\d+)$')


def upgrade() -> None:
    op.create_table(
        'note_number_counters',
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('chapter_number', sa.Integer(), nullable=False),
        sa.Column('last_ordinal', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('run_id', 'chapter_number'),
        sa.ForeignKeyConstraint(
            ['run_id'], ['processing_runs.id'], ondelete='CASCADE',
        ),
        sa.CheckConstraint('chapter_number >= 1', name='chapter_number_positive'),
        sa.CheckConstraint('last_ordinal >= 0', name='last_ordinal_nonnegative'),
    )
    backfill_note_number_counters(op.get_bind())


def downgrade() -> None:
    op.drop_table('note_number_counters')


def backfill_note_number_counters(connection) -> None:
    """Seed counters from issued note ids. A higher existing counter is kept."""
    maxima: dict[tuple[str, int], int] = {}

    def absorb(run_id: object, payload: object) -> None:
        if not isinstance(run_id, str) or not run_id:
            return
        for chapter_number, ordinal in _note_ordinals(payload):
            key = (run_id, chapter_number)
            current = maxima.get(key)
            if current is None or ordinal > current:
                maxima[key] = ordinal

    for run_id, note_id in connection.execute(
        sa.text('SELECT run_id, note_id FROM language_notes')
    ):
        absorb(run_id, note_id)
    for run_id, payload in connection.execute(
        sa.text('SELECT run_id, payload_json FROM correction_candidates')
    ):
        absorb(run_id, payload)
    # artifact_revisions.run_id is not a foreign key. ResetProcessing can leave
    # rows whose run is gone; inserting those would fail with foreign_keys=ON.
    for run_id, body in connection.execute(
        sa.text(
            'SELECT run_id, body_json FROM artifact_revisions '
            'WHERE run_id IN (SELECT id FROM processing_runs)'
        )
    ):
        absorb(run_id, body)

    statement = sa.text(
        'INSERT INTO note_number_counters (run_id, chapter_number, last_ordinal) '
        'VALUES (:run_id, :chapter_number, :last_ordinal) '
        'ON CONFLICT (run_id, chapter_number) DO UPDATE '
        'SET last_ordinal = MAX(last_ordinal, excluded.last_ordinal)'
    )
    for (run_id, chapter_number), last_ordinal in maxima.items():
        connection.execute(
            statement,
            {
                'run_id': run_id,
                'chapter_number': chapter_number,
                'last_ordinal': last_ordinal,
            },
        )


def _note_ordinals(value: object):
    if isinstance(value, str):
        match = _NOTE_ID.fullmatch(value)
        if match is not None:
            chapter_number = int(match.group(1))
            # CHECK (chapter_number >= 1) rejects a zero chapter.
            if chapter_number >= 1:
                yield chapter_number, int(match.group(2))
            return
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return
        if parsed != value:
            yield from _note_ordinals(parsed)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _note_ordinals(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _note_ordinals(item)
