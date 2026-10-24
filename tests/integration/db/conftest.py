"""Common fixtures for storage integration tests.

Each test database is a separate file in a temporary directory rather than `:memory:`: WAL mode
and partial unique indexes are verified only on a file-based database. The schema is created
by an Alembic migration, so all tests run against the exact schema the user gets,
and any discrepancy between models and migration is caught immediately.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine

from linguary.adapters.db.engine import build_database_url, create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork

#: Repository root: contains `alembic.ini` and the migrations directory.
REPO_ROOT = Path(__file__).resolve().parents[3]


def make_alembic_config(database_url: str) -> Config:
    """Alembic configuration pointing to the specified database."""
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Path to the test database file."""
    return tmp_path / "linguary.db"


@pytest.fixture
def database_url(database_path: Path) -> str:
    """Connection URL for the test database."""
    return build_database_url(database_path)


@pytest.fixture
def alembic_config(database_url: str) -> Config:
    """Alembic configuration for the test database."""
    return make_alembic_config(database_url)


@pytest.fixture
def migrated_database(alembic_config: Config) -> Config:
    """Database with applied schema: `alembic upgrade head` on an empty file."""
    command.upgrade(alembic_config, "head")
    return alembic_config


@pytest.fixture
def engine(migrated_database: Config, database_path: Path) -> Iterator[Engine]:
    """SQLite engine on top of a database with applied schema."""
    engine = create_database_engine(database_path)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def unit_of_work(engine: Engine) -> SqlAlchemyUnitOfWork:
    """Transactional boundary on top of the test database."""
    return SqlAlchemyUnitOfWork.for_engine(engine)
