"""Common fixtures for book processing integration tests.

The database is initialized in the same way as in storage and task tests: a separate file
in a temporary directory and the schema from the Alembic migration. Since pytest does not
import fixtures from neighboring directories, they are constructed consistently here.

Tasks receive predictable identifiers instead of `uuid4`: tests compare queue contents,
and random identifiers would make test output unreadable.
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
from linguary.adapters.runtime import SqliteTaskQueue, SqliteTaskStore

#: Repository root: contains `alembic.ini` and the migrations directory.
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Path to the test database file."""
    return tmp_path / "linguary.db"


@pytest.fixture
def migrated_database(database_path: Path) -> Config:
    """Database with applied schema: `alembic upgrade head` on an empty file."""
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", build_database_url(database_path))
    command.upgrade(config, "head")
    return config


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


@pytest.fixture
def processing_task_ids() -> Callable[[], str]:
    """Predictable task identifiers: `task-1`, `task-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"task-{counter}"

    return next_id


@pytest.fixture
def task_queue(engine: Engine, processing_task_ids: Callable[[], str]) -> SqliteTaskQueue:
    """Task queue visible to the application layer."""
    return SqliteTaskQueue(SqliteTaskStore.for_engine(engine, id_factory=processing_task_ids))
