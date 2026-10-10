"""Миграции Alembic: применение на пустой базе и соответствие моделям."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect

from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base

#: Таблицы канонического блока SQL-01.
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
    """`alembic upgrade head` на пустом файле создаёт все таблицы SQL-01."""
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
    """Схема после миграции совпадает с ORM-моделями: расхождений нет."""
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)
    assert not differences, f"схема разошлась с моделями: {differences}"


def test_downgrade_removes_schema(alembic_config: Config, engine: Engine) -> None:
    """Откат к базовой ревизии убирает все таблицы схемы."""
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
    """Индексы SQL-01 создаются миграцией под ожидаемыми именами."""
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
