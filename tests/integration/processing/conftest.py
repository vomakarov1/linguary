"""Общие фикстуры интеграционных тестов обработки книги.

База поднимается так же, как в тестах хранилища и задач: отдельный файл во временной
директории и схема из миграции Alembic. Фикстуры соседних директорий pytest сюда
не приносит, поэтому они собраны тем же способом, а не изобретены заново.

Задачи получают предсказуемые идентификаторы вместо `uuid4`: тест сравнивает состав
очереди, и случайный идентификатор сделал бы вывод нечитаемым.
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
def processing_task_ids() -> Callable[[], str]:
    """Предсказуемые идентификаторы задач: `task-1`, `task-2`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"task-{counter}"

    return next_id


@pytest.fixture
def task_queue(engine: Engine, processing_task_ids: Callable[[], str]) -> SqliteTaskQueue:
    """Очередь задач, которую видит прикладной слой."""
    return SqliteTaskQueue(SqliteTaskStore.for_engine(engine, id_factory=processing_task_ids))
