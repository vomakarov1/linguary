"""Translation operation metadata upgrades existing SQLite work without rewriting it."""

from __future__ import annotations

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, event, inspect, text
from sqlalchemy.exc import IntegrityError
from persistence_factories import make_book

from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base


@pytest.fixture
def legacy_database(alembic_config, database_path):
    command.upgrade(alembic_config, "0010")
    engine = create_database_engine(database_path)
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO books (id, title, authors_json, source_language, target_language,
                               status, source_epub_path, created_at)
            VALUES ('book', 'Book', '[]', 'en', 'ru', 'imported', 'source.epub', '2026-11-02')
        """))
        connection.execute(text("""
            INSERT INTO book_conditions VALUES ('conditions', 'book', 1, 1,
                '{"schema_version": 1,"template_hash":"original"}', 'verified', '2026-11-02')
        """))
        connection.execute(text("""
            INSERT INTO recovery_roots VALUES ('root', 'book', 'conditions', 'original-root-key',
                '{"schema_version": 1,"source_hash":"original","ordered_blocks":["block"]}',
                7, 4, 2, 'pending', NULL)
        """))
        connection.execute(text("""
            INSERT INTO recovery_scopes VALUES ('scope', 'root', NULL, '["block"]',
                8, 'accepted', 'attempt')
        """))
        connection.execute(text("""
            INSERT INTO llm_attempts VALUES ('attempt', 'book', 'conditions', 'chapter-task',
                'run', 'root', 'scope', NULL, 8, 3, 'accepted',
                '{"prompt_hash":"original"}', '{"text":"original result"}')
        """))
        connection.execute(text("""
            INSERT INTO artifact_revisions VALUES ('artifact', 'book', 'conditions', 'run',
                'translation', 'block', '{"text":"original result"}', 'original-hash',
                'attempt', '2026-11-02')
        """))
        connection.execute(text("""
            INSERT INTO work_bindings VALUES ('result', 'run:block', 'book', 'conditions',
                'attempt', 'artifact', '{"generation": 8}')
        """))
        connection.execute(text("""
            INSERT INTO llm_call_logs
                (id, book_id, task_id, stage, provider, model, started_at, duration_ms,
                 attempt, success, response_text, attempt_id, diagnostic_meta_json)
            VALUES ('call', 'book', 'chapter-task', 'translation', 'test', 'model',
                '2026-11-02', 42, 3, 1, 'Original response', 'attempt', '{"context":"original"}')
        """))
        for task_id, task_type, scope in (
            ("chapter-task", "process_translation_unit", "chapter"),
            ("regeneration-task", "regenerate_block", "block"),
            ("analysis-task", "analyze_chapter", "chapter"),
        ):
            connection.execute(text("""
                INSERT INTO processing_tasks VALUES (:id, 'book', :type, :scope,
                    'failed', 4, 'TransientError', 'Original failure', 'Original details', '2026-11-02')
            """), {"id": task_id, "type": task_type, "scope": scope})
    try:
        yield engine
    finally:
        engine.dispose()


def snapshot(engine, tables):
    with engine.connect() as connection:
        return {
            table: connection.execute(text(f"SELECT * FROM {table} ORDER BY 1")).all()
            for table in tables
        }


def assert_original_columns(engine, before):
    with engine.connect() as connection:
        for table, rows in before.items():
            columns = ", ".join(rows[0]._mapping)
            assert connection.execute(text(f"SELECT {columns} FROM {table} ORDER BY 1")).all() == rows
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []


def test_upgrade_0010_preserves_root_keys_and_artifacts(alembic_config, legacy_database):
    before = snapshot(legacy_database, (
        "book_conditions", "recovery_roots", "recovery_scopes", "artifact_revisions", "work_bindings",
    ))
    command.upgrade(alembic_config, "head")
    assert_original_columns(legacy_database, before)
    with legacy_database.connect() as connection:
        assert connection.execute(text("""
            SELECT policy_version, operation_kind, superseded_by, compatibility_json FROM recovery_roots
        """)).one() == (None, "translation", None, None)


def test_upgrade_preserves_tasks_attempts_and_foreign_keys(alembic_config, legacy_database):
    before = snapshot(legacy_database, ("processing_tasks", "llm_attempts", "llm_call_logs"))
    command.upgrade(alembic_config, "head")
    assert_original_columns(legacy_database, before)
    with legacy_database.connect() as connection:
        assert connection.execute(text("""
            SELECT id, operation_id, policy_version FROM processing_tasks ORDER BY id
        """)).all() == [
            ("analysis-task", "", ""),
            ("chapter-task", "dispatch:chapter-task", ""),
            ("regeneration-task", "regeneration-task", "translation-context-legacy-v1"),
        ]
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    with pytest.raises(IntegrityError), legacy_database.begin() as connection:
        connection.execute(text("UPDATE recovery_roots SET superseded_by='missing'"))
    with legacy_database.begin() as connection:
        connection.execute(text("DELETE FROM books WHERE id='book'"))
        for table in before:
            assert connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0


def test_extended_task_key_allows_new_operation_same_policy(engine, unit_of_work):
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
    with engine.begin() as connection:
        values = dict(book_id="book-1", task_type="regenerate_block", scope_id="block",
                      status="pending", attempts=0, created_at="2026-11-02",
                      policy_version="translation-context-v2")
        statement = text("""
            INSERT INTO processing_tasks
                (id, book_id, task_type, scope_id, status, attempts, created_at, operation_id, policy_version)
            VALUES (:id, :book_id, :task_type, :scope_id, :status, :attempts, :created_at, :operation_id, :policy_version)
        """)
        connection.execute(statement, dict(values, id="first", operation_id="first"))
        connection.execute(statement, dict(values, id="second", operation_id="second"))
        connection.execute(statement, dict(values, id="third", operation_id="first", policy_version="legacy"))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(statement, dict(values, id="duplicate", operation_id="first"))


def test_migrated_schema_matches_models(alembic_config, legacy_database):
    def compare_policy_defaults(context, inspected, metadata, old, new, rendered):
        if (metadata.table.name, metadata.name) in {
            ("recovery_roots", "operation_kind"),
            ("processing_tasks", "operation_id"),
            ("processing_tasks", "policy_version"),
        }:
            return None
        return False

    command.upgrade(alembic_config, "head")
    with legacy_database.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_server_default": compare_policy_defaults})
        assert compare_metadata(context, Base.metadata) == []


@pytest.mark.parametrize("mutation", [
    "UPDATE recovery_roots SET policy_version='translation-context-v2'",
    "UPDATE recovery_roots SET operation_kind='regeneration'",
    "UPDATE recovery_roots SET compatibility_json='{}'",
    "UPDATE recovery_roots SET superseded_by=id",
    "UPDATE recovery_roots SET status='superseded'",
    "UPDATE processing_tasks SET operation_id='new-operation'",
    "UPDATE processing_tasks SET policy_version='translation-context-v2'",
    "UPDATE processing_tasks SET status='superseded'",
])
def test_downgrade_refuses_to_discard_operation_history(alembic_config, legacy_database, mutation):
    command.upgrade(alembic_config, "head")
    with legacy_database.begin() as connection:
        connection.execute(text("UPDATE processing_tasks SET operation_id='', policy_version=''"))
        connection.execute(text(mutation))
    before = snapshot(legacy_database, ("recovery_roots", "processing_tasks", "alembic_version"))
    ddl = []

    def track_ddl(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("ALTER ", "DROP ", "CREATE ")):
            ddl.append(statement)

    event.listen(Engine, "before_cursor_execute", track_ddl)
    try:
        with pytest.raises(RuntimeError, match="историю операций"):
            command.downgrade(alembic_config, "0010")
    finally:
        event.remove(Engine, "before_cursor_execute", track_ddl)
    assert ddl == []
    assert snapshot(legacy_database, before) == before


def test_empty_policy_downgrade_and_upgrade_preserve_work(alembic_config, legacy_database):
    command.upgrade(alembic_config, "head")
    with legacy_database.begin() as connection:
        connection.execute(text("UPDATE processing_tasks SET operation_id='', policy_version=''"))
    before = snapshot(legacy_database, ("recovery_roots", "recovery_scopes", "llm_attempts", "artifact_revisions"))
    command.downgrade(alembic_config, "0010")
    assert "policy_version" not in {c["name"] for c in inspect(legacy_database).get_columns("recovery_roots")}
    command.upgrade(alembic_config, "head")
    assert_original_columns(legacy_database, before)


def test_downgrade_preserves_migrated_task_identity(alembic_config, legacy_database):
    command.upgrade(alembic_config, "head")
    before = snapshot(legacy_database, ("processing_tasks", "alembic_version"))
    with pytest.raises(RuntimeError, match="историю операций"):
        command.downgrade(alembic_config, "0010")
    assert snapshot(legacy_database, before) == before


def test_superseded_root_reference_keeps_book_delete_cascade(alembic_config, legacy_database):
    command.upgrade(alembic_config, "head")
    with legacy_database.begin() as connection:
        connection.execute(text("""
            INSERT INTO recovery_roots
                (id, book_id, conditions_id, root_key, manifest_json, generation, calls_used, extra_calls, status)
            VALUES ('successor', 'book', 'conditions', 'new-root-key', '{}', 0, 0, 0, 'pending')
        """))
        connection.execute(text("UPDATE recovery_roots SET superseded_by='successor' WHERE id='root'"))
        connection.execute(text("DELETE FROM books WHERE id='book'"))
        for table in ("recovery_roots", "recovery_scopes", "llm_attempts", "artifact_revisions", "work_bindings"):
            assert connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
