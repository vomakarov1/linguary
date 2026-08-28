"""Хранение данных: SQLite через SQLAlchemy, репозитории и unit of work.

Публичный вход адаптера: создать движок (`create_database_engine`) и получить транзакционную
границу (`SqlAlchemyUnitOfWork.for_engine`). Схема таблиц описана в `models`, преобразование
доменных сущностей — в `mapping`; прикладной слой их не импортирует и работает через порты
`linguary.application.ports.persistence`.
"""

from linguary.adapters.db.engine import (
    build_database_url,
    create_database_engine,
    create_session_factory,
)
from linguary.adapters.db.models import Base
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork

__all__ = [
    "Base",
    "SqlAlchemyUnitOfWork",
    "build_database_url",
    "create_database_engine",
    "create_session_factory",
]
