"""Общие фикстуры интеграционных тестов импорта EPUB.

База данных поднимается так же, как в тестах хранилища: отдельный файл во временной
директории и схема из миграции Alembic, а не из моделей. Фикстуры `tests/integration/db`
видны только своей директории, поэтому здесь они собраны тем же способом, а не изобретены
заново.
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

#: Корень репозитория: в нём лежат `alembic.ini` и каталог миграций.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Набор тестовых EPUB (PRD § 28.1).
FIXTURE_DIRECTORY = REPO_ROOT / "tests" / "fixtures" / "epub"


@pytest.fixture
def reader() -> EbookLibEpubReader:
    """Читатель EPUB, который проверяется этими тестами."""
    return EbookLibEpubReader()


@pytest.fixture
def read_fixture(
    reader: EbookLibEpubReader,
) -> Callable[..., ImportedBook]:
    """Читает книгу набора по её имени без расширения."""

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
