"""Applying current schema to database upon application startup.

The schema is maintained by Alembic (`alembic/`), and the application must apply it itself:
a local user runs Web UI or CLI rather than running migrations manually. Repeated execution
does nothing — migrations are idempotent by design.

The migrations directory lives in the repository next to the package rather than inside it; the path
is computed relative to the module to avoid depending on the process working directory. Hence the
runtime constraint: the application runs from the repository source tree because an independently
installed wheel does not contain the migrations directory (README, 'Running').
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from linguary.adapters.db.engine import build_database_url

#: Source tree root: `src/linguary/composition/migrations.py` is three levels below.
#: Path is accurate specifically for running from repository — the only mode supported.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Alembic migrations directory.
MIGRATIONS_DIRECTORY = REPOSITORY_ROOT / "alembic"


class MigrationsUnavailableError(RuntimeError):
    """Migrations directory not found: nothing to apply schema with."""


def apply_migrations(database_path: Path | str) -> None:
    """Migrates database to latest schema revision."""
    if not MIGRATIONS_DIRECTORY.is_dir():
        raise MigrationsUnavailableError(
            f"каталог миграций {MIGRATIONS_DIRECTORY} не найден: приложение работает "
            f"из исходного дерева репозитория, где каталог alembic лежит рядом с пакетом"
        )
    command.upgrade(_config(database_path), "head")


def _config(database_path: Path | str) -> Config:
    """Alembic configuration pointing to application database."""
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
