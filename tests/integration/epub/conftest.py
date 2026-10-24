"""Shared fixtures for EPUB import integration tests.

The database is brought up just like in storage tests: a dedicated file in a temporary
directory and schema created via Alembic migrations rather than from models. Fixtures
in `tests/integration/db` are scoped to their directory, so here they are configured
in the same manner rather than reinvented.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine

from linguary.adapters.db.engine import build_database_url, create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import EbookLibEpubReader
from linguary.application.ports.epub import EpubReadRequest, ImportedBook

#: Repository root: contains `alembic.ini` and the migration directory.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: EPUB test fixture suite (PRD § 28.1).
FIXTURE_DIRECTORY = REPO_ROOT / "tests" / "fixtures" / "epub"


@pytest.fixture
def reader() -> EbookLibEpubReader:
    """EPUB reader under test."""
    return EbookLibEpubReader()


@pytest.fixture
def read_fixture(
    reader: EbookLibEpubReader,
) -> Callable[..., ImportedBook]:
    """Reads a fixture book by its stem name without extension."""

    def read(name: str, *, book_id: str = "book-1", **overrides: str) -> ImportedBook:
        return reader.read(
            EpubReadRequest(
                book_id=book_id,
                source_path=str(FIXTURE_DIRECTORY / f"{name}.epub"),
                target_language=overrides.get("target_language", "ru"),
                source_language=overrides.get("source_language"),
            )
        )

    return read


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Path to the test database file."""
    return tmp_path / "linguary.db"


@pytest.fixture
def migrated_database(database_path: Path) -> Config:
    """Database with applied schema: runs `alembic upgrade head` on an empty file."""
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", build_database_url(database_path))
    command.upgrade(config, "head")
    return config


@pytest.fixture
def engine(migrated_database: Config, database_path: Path) -> Iterator[Engine]:
    """SQLite engine backed by a migrated database."""
    engine = create_database_engine(database_path)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def unit_of_work(engine: Engine) -> SqlAlchemyUnitOfWork:
    """Transactional boundary over the test database."""
    return SqlAlchemyUnitOfWork.for_engine(engine)
