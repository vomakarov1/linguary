"""Подключение к SQLite: движок, PRAGMA-настройки соединения и фабрика сессий.

Путь к файлу базы приходит параметром: приложение берёт его из локальной конфигурации,
тесты — из временной директории.

Каждое соединение настраивается отдельно, потому что PRAGMA в SQLite действуют на
соединение, а не на файл базы:

- `foreign_keys=ON` — внешние ключи в SQLite по умолчанию выключены, и без этой PRAGMA
  каскадное удаление и ссылочная целостность схемы просто не работают;
- `journal_mode=WAL` — режим журнала (ADR § 3): читатели не блокируют писателя, что нужно
  воркеру и Web UI, работающим с одной базой одновременно. Режим сохраняется в самом файле
  базы, но задавать его на соединении безопасно и делает поведение явным;
- `busy_timeout` — ожидание освобождения блокировки вместо немедленной ошибки `database is
  locked` при параллельной записи.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

#: Ожидание освобождения блокировки записи, миллисекунды.
DEFAULT_BUSY_TIMEOUT_MS = 5000

#: Значение пути, означающее базу в оперативной памяти.
IN_MEMORY = ":memory:"


def build_database_url(database_path: Path | str) -> str:
    """URL подключения SQLAlchemy к файлу базы или к базе в памяти."""
    if str(database_path) == IN_MEMORY:
        return "sqlite+pysqlite://"
    return f"sqlite+pysqlite:///{Path(database_path).resolve()}"


def create_database_engine(
    database_path: Path | str,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    echo: bool = False,
) -> Engine:
    """Создаёт движок SQLite с настроенными PRAGMA на каждом соединении."""
    engine = create_engine(build_database_url(database_path), echo=echo)
    configure_sqlite_connections(engine, busy_timeout_ms=busy_timeout_ms)
    return engine


def configure_sqlite_connections(
    engine: Engine, *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> None:
    """Подписывает движок на настройку PRAGMA при открытии соединения."""

    @event.listens_for(engine, "connect")
    def _apply_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        finally:
            cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Фабрика сессий для unit of work.

    `expire_on_commit=False` оставляет прочитанные строки пригодными к использованию
    после фиксации: репозитории возвращают доменные объекты, а не живые ORM-строки,
    и повторное обращение к базе после коммита им не нужно.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "IN_MEMORY",
    "build_database_url",
    "configure_sqlite_connections",
    "create_database_engine",
    "create_session_factory",
]
