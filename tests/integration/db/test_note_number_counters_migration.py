"""Migration 0018 backfills note_number_counters without orphaned run ids."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from linguary.adapters.db.models import Base

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[3] / "alembic/versions/0018_note_number_counters.py"
)


def test_migration_0018_backfills_notes_candidates_and_live_revisions(
    alembic_config, engine: Engine
) -> None:
    """Gaps, candidate payloads, and a live revision seed the max ordinal."""
    command.downgrade(alembic_config, "0017")
    with engine.begin() as connection:
        _insert_book_graph(connection)
        _insert_note(connection, "ln-ch01-001")
        _insert_note(connection, "ln-ch01-007")
        _insert_candidate(
            connection,
            {
                "result": {"language_notes": [{"id": "ln-ch01-009"}]},
                "current_result": {"language_notes": [{"id": "ln-ch02-004"}]},
            },
        )
        _insert_revision(
            connection,
            revision_id="artifact-live",
            run_id="run-1",
            body={
                "schema_version": 1,
                "result": {
                    "translated_text": '"ln-ch01-999"',
                    "language_notes": [
                        {"id": "ln-ch01-012"},
                        {"id": "custom-note"},
                    ],
                    "label": "see ln-ch01-099",
                },
            },
        )
        connection.execute(
            text(
                "INSERT INTO artifact_revisions ("
                "id, book_id, run_id, kind, scope_id, body_json, sha256, created_at"
                ") VALUES ("
                "'artifact-analysis', 'book-1', 'run-1', 'chapter_analysis', 'section-1', "
                ":body, :sha, '2026-10-10 00:00:00'"
                ")"
            ),
            {
                "body": json.dumps({"result": {"language_notes": [{"id": "ln-ch01-888"}]}}),
                "sha": "c" * 64,
            },
        )

    command.upgrade(alembic_config, "0018")

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT run_id, chapter_number, last_ordinal "
                "FROM note_number_counters ORDER BY chapter_number"
            )
        ).all()
    assert rows == [("run-1", 1, 12), ("run-1", 2, 4)]


def test_migration_0018_ignores_orphaned_revision_runs_under_fk_on(
    alembic_config, engine: Engine
) -> None:
    """A revision whose run was deleted does not violate the new foreign key."""
    command.downgrade(alembic_config, "0017")
    with engine.begin() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        _insert_book(connection)
        _insert_revision(
            connection,
            revision_id="artifact-orphan",
            run_id="missing-run",
            body={"result": {"language_notes": [{"id": "ln-ch01-015"}]}},
        )

    command.upgrade(alembic_config, "0018")

    with engine.connect() as connection:
        orphan_count = connection.execute(
            text(
                "SELECT COUNT(*) FROM note_number_counters WHERE run_id = 'missing-run'"
            )
        ).scalar_one()
    assert orphan_count == 0
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO note_number_counters (run_id, chapter_number, last_ordinal) "
                "VALUES ('missing-run', 1, 15)"
            )
        )


def test_migration_0018_backfill_is_idempotent_and_matches_models_and_downgrades(
    alembic_config, engine: Engine
) -> None:
    """Repeat backfill keeps a raised counter; downgrade drops the table cleanly."""
    from linguary.adapters.db.models import NoteNumberCounterRow
    import linguary.adapters.db.models as models

    migration = _load_migration()
    assert migration.revision == "0018"
    assert migration.down_revision == "0017"
    assert NoteNumberCounterRow.__tablename__ == "note_number_counters"
    assert "NoteNumberCounterRow" in models.__all__

    command.downgrade(alembic_config, "0017")
    with engine.begin() as connection:
        _insert_book_graph(connection)
        _insert_note(connection, "ln-ch01-003")
    command.upgrade(alembic_config, "0018")

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE note_number_counters SET last_ordinal = 40 "
                "WHERE run_id = 'run-1' AND chapter_number = 1"
            )
        )
        migration.backfill_note_number_counters(connection)
        migration.backfill_note_number_counters(connection)
        ordinal = connection.execute(
            text(
                "SELECT last_ordinal FROM note_number_counters "
                "WHERE run_id = 'run-1' AND chapter_number = 1"
            )
        ).scalar_one()
    assert ordinal == 40

    command.downgrade(alembic_config, "0017")
    with engine.connect() as connection:
        assert "note_number_counters" not in inspect(connection).get_table_names()
        assert connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one() == "0017"

    command.upgrade(alembic_config, "head")
    with engine.connect() as connection:
        assert "note_number_counters" in inspect(connection).get_table_names()
        assert connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one() == "0018"
        restored = connection.execute(
            text(
                "SELECT last_ordinal FROM note_number_counters "
                "WHERE run_id = 'run-1' AND chapter_number = 1"
            )
        ).scalar_one()
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)
    assert restored == 3
    assert not differences, f"schema diverged from models: {differences}"


def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "note_number_counters_0018", _MIGRATION_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"migration module is missing: {_MIGRATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _insert_book(connection: Connection) -> None:
    connection.execute(
        text(
            "INSERT INTO books (id, title, authors_json, source_language, "
            "target_language, status, source_epub_path, created_at) "
            "VALUES ('book-1', 'Book', '[]', 'en', 'ru', 'imported', "
            "'source.epub', '2026-10-10 00:00:00')"
        )
    )


def _insert_book_graph(connection: Connection) -> None:
    _insert_book(connection)
    connection.execute(
        text(
            "INSERT INTO sections (id, book_id, title, \"order\", source_resource_path) "
            "VALUES ('section-1', 'book-1', 'Chapter 1', 0, 'OEBPS/chapter1.xhtml')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO document_blocks ("
            "id, book_id, section_id, type, \"order\", text, inline_annotations_json, "
            "attributes_json, source_reference_json, support_status"
            ") VALUES ("
            "'block-1', 'book-1', 'section-1', 'paragraph', 0, 'Alpha.', '[]', "
            "'{}', "
            "'{\"resource_path\":\"OEBPS/chapter1.xhtml\",\"structural_path\":\"/p[1]\"}', "
            "'supported'"
            ")"
        )
    )
    connection.execute(
        text(
            "INSERT INTO processing_runs ("
            "id, book_id, status, glossary_snapshot_json, llm_settings_json, created_at"
            ") VALUES ("
            "'run-1', 'book-1', 'active', '[]', '{}', '2026-10-10 00:00:00'"
            ")"
        )
    )
    connection.execute(
        text(
            "INSERT INTO book_conditions ("
            "id, book_id, revision, schema_version, payload_json, reproducibility, created_at"
            ") VALUES ("
            "'conditions-1', 'book-1', 1, 1, '{\"schema_version\": 1}', 'complete', "
            "'2026-10-10 00:00:00'"
            ")"
        )
    )


def _insert_note(connection: Connection, note_id: str) -> None:
    connection.execute(
        text(
            "INSERT INTO language_notes (run_id, block_id, note_id, original_excerpt, note) "
            "VALUES ('run-1', 'block-1', :note_id, 'excerpt', 'заметка')"
        ),
        {"note_id": note_id},
    )


def _insert_candidate(connection: Connection, payload: dict) -> None:
    connection.execute(
        text(
            "INSERT INTO correction_candidates ("
            "candidate_id, book_id, run_id, conditions_id, section_id, block_id, "
            "input_sha256, input_generation, proposal_sha256, payload_json, "
            "group_id, proposal_attempt_id, state"
            ") VALUES ("
            "'candidate-1', 'book-1', 'run-1', 'conditions-1', 'section-1', 'block-1', "
            ":sha, 1, :sha, :payload, 'group-1', 'attempt-1', 'proposed'"
            ")"
        ),
        {"sha": "a" * 64, "payload": json.dumps(payload)},
    )


def _insert_revision(
    connection: Connection, *, revision_id: str, run_id: str, body: dict
) -> None:
    connection.execute(
        text(
            "INSERT INTO artifact_revisions ("
            "id, book_id, run_id, kind, scope_id, body_json, sha256, created_at"
            ") VALUES ("
            ":revision_id, 'book-1', :run_id, 'block_result', 'block-1', :body, :sha, "
            "'2026-10-10 00:00:00'"
            ")"
        ),
        {
            "revision_id": revision_id,
            "run_id": run_id,
            "body": json.dumps(body),
            "sha": "b" * 64,
        },
    )
