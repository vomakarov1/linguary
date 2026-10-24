"""Alembic entrypoint: database connection and migration runner.

The database URL is not hardcoded in configuration or code: it is provided via
the `LINGUARY_DATABASE_URL` environment variable. The application sets it from
local configuration, and tests set it from a temporary directory.

`render_as_batch=True` is required for SQLite: SQLite cannot alter columns or constraints
in place, so Alembic recreates tables via a temporary copy.
"""

from __future__ import annotations

import os

from alembic import context
from alembic.autogenerate.api import AutogenContext
from sqlalchemy import engine_from_config, pool
from sqlalchemy.types import TypeDecorator

from linguary.adapters.db.engine import configure_sqlite_connections
from linguary.adapters.db.models import Base

#: Environment variable containing the SQLAlchemy connection URL.
DATABASE_URL_ENV = "LINGUARY_DATABASE_URL"

config = context.config
target_metadata = Base.metadata


def database_url() -> str:
    """Connection URL for the database targeted by migrations.

    A URL specified by caller code in configuration takes precedence: this allows
    the application and tests to specify their own database without relying on machine environment.
    """
    url = config.get_main_option("sqlalchemy.url") or os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(
            f"не задан адрес базы данных: укажите переменную окружения {DATABASE_URL_ENV}"
        )
    return url


def render_item(item_type: str, obj: object, autogen_context: AutogenContext) -> str | bool:
    """Renders custom column types using their underlying SQLAlchemy base type.

    Otherwise, the migration references `linguary.adapters.db.models` and breaks
    whenever the models module changes; migrations must remain self-contained.
    """
    if item_type == "type" and isinstance(obj, TypeDecorator):
        return f"sa.{type(obj.impl).__name__}()"
    return False


def run_migrations_offline() -> None:
    """Generate SQL migrations without connecting to the database."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations to a live database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    configure_sqlite_connections(connectable)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
