"""Общие фикстуры интеграционных тестов хранилища.

База каждого теста — отдельный файл во временной директории, а не `:memory:`: режим WAL
и частичные уникальные индексы проверяются только на файловой базе. Схема создаётся
миграцией Alembic, поэтому все тесты работают ровно с той схемой, которую получит
пользователь, а расхождение моделей и миграции ловится сразу.
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

#: Корень репозитория: в нём лежат `alembic.ini` и каталог миграций.
REPO_ROOT = Path(__file__).resolve().parents[3]


def make_alembic_config(database_url: str) -> Config:
    """Конфигурация Alembic, направленная на указанную базу."""
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Путь к файлу базы данных теста."""
    return tmp_path / "linguary.db"


@pytest.fixture
def database_url(database_path: Path) -> str:
    """URL подключения к базе теста."""
    return build_database_url(database_path)


@pytest.fixture
def alembic_config(database_url: str) -> Config:
    """Конфигурация Alembic для базы теста."""
    return make_alembic_config(database_url)


@pytest.fixture
def migrated_database(alembic_config: Config) -> Config:
    """База с применённой схемой: `alembic upgrade head` на пустом файле."""
    command.upgrade(alembic_config, "head")
    return alembic_config


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
