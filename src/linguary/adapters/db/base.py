"""Shared SQLAlchemy base metadata and custom column types."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, String
from sqlalchemy.engine import Dialect
from sqlalchemy.exc import DontWrapMixin
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

from linguary.application.ports.persistence import PersistenceError

#: Naming convention for keys and constraints. Without it, SQLite cannot alter or drop
#: an unnamed constraint in Alembic batch mode because the migration has no identifier to target.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class ContentJSONError(PersistenceError, DontWrapMixin):
    """Error for new JSON fields without embedding SQL parameters into the message."""


class ContentObjectJSON(TypeDecorator[dict[str, Any]]):
    """SQL NULL for legacy rows; non-null JSON must be a dict/object."""

    impl = JSON
    cache_ok = True

    def __init__(self) -> None:
        super().__init__(none_as_null=True)

    def process_bind_param(self, value, dialect):
        if value is not None and type(value) is not dict:
            raise ContentJSONError("некорректный объект содержательного плана или свидетельств")
        return value

    def result_processor(self, dialect, coltype):
        processor = super().result_processor(dialect, coltype)

        def process(raw):
            if raw is None:
                return None
            try:
                value = processor(raw) if processor is not None else raw
            except (ValueError, TypeError):
                raise ContentJSONError("повреждённый JSON содержательного плана или свидетельств") from None
            if type(value) is not dict:
                raise ContentJSONError("некорректный объект содержательного плана или свидетельств")
            return value

        return process


class UtcDateTime(TypeDecorator[datetime]):
    """Timestamp stored in UTC.

    SQLite does not store timezone offsets, so values are converted to UTC on write
    and returned with an explicit UTC timezone on read. Naive datetimes are assumed
    to be in UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class DecimalAsText(TypeDecorator[Decimal]):
    """Fixed-precision numeric value stored as text.

    SQLite numeric columns return `float`, causing call cost calculations to lose
    precision. Textual representation preserves precision without loss.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> str | None:
        return None if value is None else str(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> Decimal | None:
        return None if value is None else Decimal(value)


class Base(DeclarativeBase):
    """Common base class for schema ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "ContentJSONError",
    "ContentObjectJSON",
    "DecimalAsText",
    "UtcDateTime",
]
