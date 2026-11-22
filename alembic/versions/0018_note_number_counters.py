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

    def absorb_pairs(run_id: object, pairs) -> None:
        if not isinstance(run_id, str) or not run_id:
            return
        for chapter_number, ordinal in pairs:
            key = (run_id, chapter_number)
            current = maxima.get(key)
            if current is None or ordinal > current:
                maxima[key] = ordinal

    for run_id, note_id in connection.execute(
        sa.text('SELECT run_id, note_id FROM language_notes')
    ):
        pair = _parse_note_id(note_id)
        if pair is not None:
            absorb_pairs(run_id, (pair,))
    for run_id, payload in connection.execute(
        sa.text('SELECT run_id, payload_json FROM correction_candidates')
    ):
        absorb_pairs(run_id, _structured_note_ordinals(payload))
    # artifact_revisions.run_id is not a foreign key. ResetProcessing can leave
    # rows whose run is gone; inserting those would fail with foreign_keys=ON.
    for run_id, body in connection.execute(
        sa.text(
            "SELECT run_id, body_json FROM artifact_revisions "
            "WHERE kind = 'block_result' AND run_id IN (SELECT id FROM processing_runs)"
        )
    ):
        absorb_pairs(run_id, _structured_note_ordinals(body))

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


def _parse_note_id(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    match = _NOTE_ID.fullmatch(value)
    if match is None:
        return None
    chapter_number = int(match.group(1))
    # CHECK (chapter_number >= 1) rejects a zero chapter.
    if chapter_number < 1:
        return None
    return chapter_number, int(match.group(2))


def _structured_note_ordinals(value: object):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return
    if not isinstance(value, dict):
        return
    containers = [value]
    for key in ('result', 'current_result'):
        nested = value.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        notes = container.get('language_notes')
        if not isinstance(notes, list):
            continue
        for note in notes:
            if isinstance(note, dict):
                pair = _parse_note_id(note.get('id'))
                if pair is not None:
                    yield pair
