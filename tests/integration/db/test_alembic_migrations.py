"""Alembic migrations: application on an empty database and model compliance."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, JSON, inspect

from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base

#: Canonical SQL-01 tables.
EXPECTED_TABLES = (
    "books",
    "sections",
    "document_blocks",
    "resources",
    "import_warnings",
    "term_concepts",
    "term_variants",
    "glossary_candidates",
    "chapter_analyses",
    "treatment_plan_items",
    "processing_runs",
    "block_results",
    "language_notes",
    "result_warnings",
    "processing_tasks",
    "llm_call_logs",
    "system_settings",
)


def test_upgrade_creates_schema_on_empty_database(
    alembic_config: Config, database_path: Path
) -> None:
    """`alembic upgrade head` on an empty file creates all SQL-01 tables."""
    assert not database_path.exists()

    command.upgrade(alembic_config, "head")

    engine = create_database_engine(database_path)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert set(EXPECTED_TABLES) <= tables, (
        f"миграция не создала таблицы: {sorted(set(EXPECTED_TABLES) - tables)}"
    )


def test_migrated_schema_matches_models(engine: Engine) -> None:
    """Schema after migration matches ORM models: no discrepancies."""
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)
    assert not differences, f"схема разошлась с моделями: {differences}"


def test_downgrade_removes_schema(alembic_config: Config, engine: Engine) -> None:
    """Downgrade to base revision removes all schema tables."""
    command.downgrade(alembic_config, "base")

    tables = set(inspect(engine).get_table_names())
    assert not set(EXPECTED_TABLES) & tables, (
        f"после отката остались таблицы: {sorted(set(EXPECTED_TABLES) & tables)}"
    )


@pytest.mark.parametrize(
    ("table", "index_name"),
    [
        pytest.param("books", "ix_books_status", id="books-status"),
        pytest.param(
            "document_blocks",
            "ix_document_blocks_book_id_section_id",
            id="blocks-book-section",
        ),
        pytest.param(
            "glossary_candidates",
            "ix_glossary_candidates_requires_review",
            id="candidates-requires-review",
        ),
        pytest.param(
            "processing_runs", "uq_processing_runs_active_book", id="runs-active-book"
        ),
        pytest.param(
            "processing_tasks", "ix_processing_tasks_status", id="tasks-status"
        ),
        pytest.param(
            "llm_call_logs", "ix_llm_call_logs_book_id_stage", id="llm-book-stage"
        ),
        pytest.param(
            "result_warnings", "ix_result_warnings_run_id_block_id", id="warnings-run-block"
        ),
    ],
)
def test_migration_creates_required_index(
    engine: Engine, table: str, index_name: str
) -> None:
    """SQL-01 indexes are created by the migration under expected names."""
    names = {index["name"] for index in inspect(engine).get_indexes(table)}
    assert index_name in names, f"в таблице {table} нет индекса {index_name}: {sorted(names)}"


def test_upgrade_creates_system_settings(engine):
    from sqlalchemy import text
    assert 'system_settings' in inspect(engine).get_table_names()
    with engine.connect() as connection:
        assert connection.execute(text('SELECT count(*) FROM system_settings')).scalar() == 0


def test_upgrade_from_0003_preserves_books(alembic_config, database_path):
    from sqlalchemy import text
    command.upgrade(alembic_config, '0003')
    engine = create_database_engine(database_path)
    try:
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO books (id, title, authors_json, source_language, "
                "target_language, status, source_epub_path, created_at) "
                "VALUES ('existing', 'Book', '[]', 'en', 'ru', 'imported', "
                "'source.epub', '2026-10-10 00:00:00')"
            ))
            before = connection.execute(text('SELECT * FROM books')).all()
        command.upgrade(alembic_config, 'head')
        with engine.connect() as connection:
            assert connection.execute(text('SELECT * FROM books')).all() == before
    finally:
        engine.dispose()


@pytest.mark.parametrize('id,revision', [(2, 1), (1, 0)])
def test_settings_singleton_constraint(engine, id, revision):
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(text('INSERT INTO system_settings (id,revision) VALUES (:id,:revision)'), dict(id=id, revision=revision))


def test_downgrade_removes_system_settings(alembic_config, engine):
    command.downgrade(alembic_config, '0003')
    tables = inspect(engine).get_table_names()
    assert 'system_settings' not in tables
    assert 'books' in tables


def test_upgrade_0004_preserves_settings_and_adds_proxy_column(alembic_config, database_path):
    import json
    from sqlalchemy import text
    command.upgrade(alembic_config, '0004')
    engine = create_database_engine(database_path)
    values = dict(provider='openai_compatible', model='old', base_url='http://localhost', requests_per_minute=None, concurrency=1)
    try:
        with engine.begin() as connection:
            connection.execute(text('INSERT INTO system_settings (id, revision, overrides_json, api_key) VALUES (1, 7, :values, :key)'), {'values': json.dumps(values), 'key': 'key'})
        command.upgrade(alembic_config, 'head')
        with engine.begin() as connection:
            row = connection.execute(text('SELECT * FROM system_settings')).mappings().one()
            assert row['revision'] == 7 and row['api_key'] == 'key'
            assert row['proxy_url'] is None and json.loads(row['overrides_json']) == values
            connection.execute(text('UPDATE system_settings SET overrides_json=:values, proxy_url=:url'), {'values': json.dumps({**values, 'proxy_mode': 'custom'}), 'url': 'http://alice:password@localhost'})
        command.downgrade(alembic_config, '0004')
        assert 'proxy_url' not in {column['name'] for column in inspect(engine).get_columns('system_settings')}
        with engine.connect() as connection:
            row = connection.execute(text('SELECT * FROM system_settings')).mappings().one()
            assert row['revision'] == 7 and row['api_key'] == 'key'
            assert json.loads(row['overrides_json']) == values
    finally:
        engine.dispose()


@pytest.fixture
def content_rows(unit_of_work):
    from persistence_factories import (
        make_analysis, make_book, make_concept, make_document, make_run,
        make_translated_result,
    )
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.glossary.save(make_concept())
        uow.analyses.save('book-1', make_analysis())
        uow.runs.add(make_run())
        uow.results.replace_results('run-1', [make_translated_result()])


CONTENT_COLUMNS = (
    ('chapter_analyses', 'content_plan_json'),
    ('block_results', 'content_evidence_json'),
)


def test_content_upgrade_preserves_legacy_rows(alembic_config, engine, content_rows):
    from sqlalchemy import text
    from linguary.adapters.db.system_settings import SystemSettingsStore
    store = SystemSettingsStore(engine)
    settings = store.save(dict(provider='openai_compatible', model='test', base_url='http://localhost:8080/v1', requests_per_minute=None, concurrency=2), None, expected_revision=0)
    command.downgrade(alembic_config, '0005')
    with engine.connect() as connection:
        before = {
            name: connection.execute(text(f'SELECT * FROM {name}')).all()
            for name in ('chapter_analyses', 'block_results', 'treatment_plan_items', 'glossary_candidates', 'term_concepts', 'system_settings')
        }
    command.upgrade(alembic_config, '0006')
    with engine.connect() as connection:
        for name, rows in before.items():
            columns = list(rows[0]._mapping)
            after = connection.execute(text(f"SELECT {', '.join(columns)} FROM {name}")).all()
            assert after == rows
        for table, column in CONTENT_COLUMNS:
            assert connection.execute(text(
                f'SELECT {column} IS NULL, typeof({column}) FROM {table}'
            )).one() == (1, 'null')
    assert store.load() == settings


@pytest.mark.parametrize('table,column', CONTENT_COLUMNS)
@pytest.mark.parametrize('payload', ['null', '[]', '"private-book-text"', '0', 'true', '{broken-private-book-text'])
def test_content_json_rejects_nonobjects_on_read(engine, content_rows, table, column, payload):
    from sqlalchemy import select, text
    from linguary.application.ports.persistence import PersistenceError
    model = Base.metadata.tables[table]
    with engine.begin() as connection:
        connection.execute(text(f'UPDATE {table} SET {column} = :payload'), {'payload': payload})
    with engine.connect() as connection:
        with pytest.raises(PersistenceError) as caught:
            connection.execute(select(model.c[column])).all()
    assert 'private-book-text' not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize('table,column', CONTENT_COLUMNS)
def test_content_json_writes_none_as_sql_null(engine, content_rows, table, column):
    from sqlalchemy import select, text, update
    model = Base.metadata.tables[table]
    with engine.begin() as connection:
        connection.execute(update(model).values({column: {'version': 2}}))
        assert connection.execute(select(model.c[column])).scalar_one() == {'version': 2}
        connection.execute(update(model).values({column: None}))
        assert connection.execute(text(f'SELECT {column} IS NULL FROM {table}')).scalar_one() == 1
        assert connection.execute(select(model.c[column])).scalar_one() is None


@pytest.mark.parametrize('table,column', CONTENT_COLUMNS)
@pytest.mark.parametrize('payload', [[], 'private-book-text', 0, True, JSON.NULL])
def test_content_json_rejects_nonobjects_on_write(engine, content_rows, table, column, payload):
    from sqlalchemy import update
    from linguary.application.ports.persistence import PersistenceError
    with pytest.raises(PersistenceError) as caught, engine.begin() as connection:
        connection.execute(update(Base.metadata.tables[table]).values({column: payload}))
    assert 'private-book-text' not in str(caught.value)


@pytest.mark.parametrize('table,column', CONTENT_COLUMNS)
@pytest.mark.parametrize('payload', ['{}', 'null', '{"version": 999}', '{broken'])
def test_content_downgrade_refuses_all_nonnull_before_ddl(
    alembic_config, engine, content_rows, table, column, payload,
):
    from sqlalchemy import text
    with engine.begin() as connection:
        connection.execute(text(f'UPDATE {table} SET {column} = :payload'), {'payload': payload})
    from sqlalchemy import event
    ddl = []

    def track_ddl(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(('ALTER ', 'DROP ', 'CREATE ')):
            ddl.append(statement)

    event.listen(Engine, 'before_cursor_execute', track_ddl)
    try:
        with pytest.raises(RuntimeError, match='содержательного'):
            command.downgrade(alembic_config, '0005')
    finally:
        event.remove(Engine, 'before_cursor_execute', track_ddl)
    assert ddl == []
    with engine.connect() as connection:
        assert connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one() == '0006'
        assert connection.execute(text(f'SELECT {column} FROM {table}')).scalar_one() == payload
        for target, name in CONTENT_COLUMNS:
            assert name in {item['name'] for item in inspect(connection).get_columns(target)}


def test_content_null_downgrade_keeps_analysis_children(alembic_config, engine, content_rows, unit_of_work):
    from persistence_factories import make_analysis
    command.downgrade(alembic_config, '0005')
    command.upgrade(alembic_config, '0006')
    with unit_of_work.transaction() as uow:
        assert uow.analyses.get_for_section('book-1', 'section-1') == make_analysis()


def test_content_downgrade_old_sqlite_fails_before_ddl(engine, content_rows, monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[3] / 'alembic/versions/0006_content_plan.py'
    spec = importlib.util.spec_from_file_location('content_plan_migration', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.connect() as connection:
        monkeypatch.setattr(connection.dialect, 'server_version_info', (3, 34, 0))
        monkeypatch.setattr(migration.op, 'get_bind', lambda: connection)
        dropped = []
        monkeypatch.setattr(migration.op, 'drop_column', lambda *args: dropped.append(args))
        with pytest.raises(RuntimeError, match='SQLite 3.35'):
            migration.downgrade()
        assert dropped == []
        for target, name in CONTENT_COLUMNS:
            assert name in {item['name'] for item in inspect(connection).get_columns(target)}
