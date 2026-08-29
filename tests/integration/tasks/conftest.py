"""Общие фикстуры интеграционных тестов очереди локальных задач.

База поднимается так же, как в тестах хранилища и импорта: отдельный файл во временной
директории и схема из миграции Alembic. Фикстуры соседних директорий pytest сюда не приносит,
поэтому они собраны тем же способом, а не изобретены заново.

Задачи получают предсказуемые идентификаторы вместо `uuid4`: тесты сравнивают порядок
захвата, и случайный идентификатор превратил бы сортировку в лотерею на совпавших отметках
времени создания.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine
from task_factories import make_book, sequential_task_ids

from linguary.adapters.db.engine import build_database_url, create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import SqliteTaskQueue, SqliteTaskStore

#: Корень репозитория: в нём лежат `alembic.ini` и каталог миграций.
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Путь к файлу базы данных теста."""
    return tmp_path / "linguary.db"


@pytest.fixture
def migrated_database(database_path: Path) -> Config:
    """База с применённой схемой: `alembic upgrade head` на пустом файле."""
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", build_database_url(database_path))
    command.upgrade(config, "head")
    return config


@pytest.fixture
def engine(migrated_database: Config, database_path: Path) -> Iterator[Engine]:
    """Движок SQLite поверх базы с применённой схемой."""
    engine = create_database_engine(database_path)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def unit_of_work(engine: Engine) -> SqlAlchemyUnitOfWork:
    """Транзакционная граница поверх базы теста."""
    return SqlAlchemyUnitOfWork.for_engine(engine)


@pytest.fixture
def task_ids() -> Callable[[], str]:
    """Предсказуемые идентификаторы задач: `task-1`, `task-2`, ..."""
    return sequential_task_ids()


@pytest.fixture
def task_store(engine: Engine, task_ids: Callable[[], str]) -> SqliteTaskStore:
    """Блокирующее хранилище очереди задач."""
    return SqliteTaskStore.for_engine(engine, id_factory=task_ids)


@pytest.fixture
def task_queue(task_store: SqliteTaskStore) -> SqliteTaskQueue:
    """Очередь задач, которую видит прикладной слой."""
    return SqliteTaskQueue(task_store)


@pytest.fixture
def stored_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Книга, к которой привязаны задачи теста."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
