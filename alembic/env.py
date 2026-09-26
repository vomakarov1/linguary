"""Точка входа Alembic: подключение к базе и запуск миграций.

Путь к базе не зашит ни в конфигурацию, ни в код: он приходит переменной окружения
`LINGUARY_DATABASE_URL`. Приложение выставляет её из локальной конфигурации, тест —
из временной директории.

`render_as_batch=True` обязателен для SQLite: он не умеет менять колонки и ограничения
на месте, и Alembic пересоздаёт таблицу через временную копию.
"""

from __future__ import annotations

import os

from alembic import context
from alembic.autogenerate.api import AutogenContext
from sqlalchemy import engine_from_config, pool
from sqlalchemy.types import TypeDecorator

from linguary.adapters.db.engine import configure_sqlite_connections
from linguary.adapters.db.models import Base

#: Переменная окружения с URL подключения SQLAlchemy.
DATABASE_URL_ENV = "LINGUARY_DATABASE_URL"

config = context.config
target_metadata = Base.metadata


def database_url() -> str:
    """URL подключения к базе, к которой применяются миграции.

    Адрес, заданный вызывающим кодом в конфигурации, имеет приоритет: так приложение
    и тест задают собственную базу, не завися от окружения машины.
    """
    url = config.get_main_option("sqlalchemy.url") or os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(
            f"не задан адрес базы данных: укажите переменную окружения {DATABASE_URL_ENV}"
        )
    return url


def render_item(item_type: str, obj: object, autogen_context: AutogenContext) -> str | bool:
    """Записывает собственные типы колонок их базовым типом SQLAlchemy.

    Иначе миграция ссылается на `linguary.adapters.db.models` и перестаёт применяться,
    как только модуль моделей изменится: миграция обязана оставаться самодостаточной.
    """
    if item_type == "type" and isinstance(obj, TypeDecorator):
        return f"sa.{type(obj.impl).__name__}()"
    return False


def run_migrations_offline() -> None:
    """Генерация SQL миграций без подключения к базе."""
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
    """Применение миграций к реальной базе."""
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
