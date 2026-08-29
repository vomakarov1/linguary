"""Приведение базы к текущей схеме при запуске приложения.

Схему держит Alembic (`alembic/`), и приложение обязано применять её само: локальный
пользователь запускает Web UI или CLI, а не миграции руками. Повторный запуск ничего
не делает — миграции идемпотентны по своей природе.

Каталог миграций живёт в репозитории рядом с пакетом; путь вычисляется от модуля, чтобы
не зависеть от текущего рабочего каталога процесса.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from linguary.adapters.db.engine import build_database_url

#: Корень репозитория: `src/linguary/composition/migrations.py` лежит на три уровня ниже.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Каталог миграций Alembic.
MIGRATIONS_DIRECTORY = REPOSITORY_ROOT / "alembic"


class MigrationsUnavailableError(RuntimeError):
    """Каталог миграций не найден: применить схему нечем."""


def apply_migrations(database_path: Path | str) -> None:
    """Приводит базу к последней ревизии схемы."""
    if not MIGRATIONS_DIRECTORY.is_dir():
        raise MigrationsUnavailableError(
            f"каталог миграций {MIGRATIONS_DIRECTORY} не найден: "
            f"запустите приложение из каталога проекта"
        )
    command.upgrade(_config(database_path), "head")


def _config(database_path: Path | str) -> Config:
    """Конфигурация Alembic, направленная на базу приложения."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIRECTORY))
    config.set_main_option("sqlalchemy.url", build_database_url(database_path))
    return config


__all__ = [
    "MIGRATIONS_DIRECTORY",
    "REPOSITORY_ROOT",
    "MigrationsUnavailableError",
    "apply_migrations",
]
