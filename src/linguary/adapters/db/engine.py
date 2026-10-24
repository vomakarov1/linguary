"""SQLite connection: engine, connection PRAGMA configuration, and session factory.

The database file path is supplied as a parameter: the application reads it from the local
configuration, while tests supply it from a temporary directory.

Each connection is configured individually because SQLite PRAGMAs apply per connection,
not per database file:

- `foreign_keys=ON` — SQLite has foreign keys disabled by default, and without this PRAGMA,
  cascade deletion and schema referential integrity do not work;
- `journal_mode=WAL` — Write-Ahead Logging mode (ADR § 3): readers do not block writers, which is
  necessary for the worker and Web UI working concurrently on the same database. The mode is
  persisted in the database file itself, but setting it on connection is safe and makes the
  behavior explicit;
- `busy_timeout` — wait for write lock release instead of an immediate `database is locked` error
  under concurrent writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

#: Write lock wait timeout in milliseconds.
DEFAULT_BUSY_TIMEOUT_MS = 5000

#: Path value representing an in-memory database.
IN_MEMORY = ":memory:"


def build_database_url(database_path: Path | str) -> str:
    """SQLAlchemy connection URL for a database file or in-memory database."""
    if str(database_path) == IN_MEMORY:
        return "sqlite+pysqlite://"
    return f"sqlite+pysqlite:///{Path(database_path).resolve()}"


def create_database_engine(
    database_path: Path | str,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    echo: bool = False,
) -> Engine:
    """Create a SQLite engine configured with connection PRAGMAs."""
    engine = create_engine(build_database_url(database_path), echo=echo)
    configure_sqlite_connections(engine, busy_timeout_ms=busy_timeout_ms)
    return engine


def configure_sqlite_connections(
    engine: Engine, *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> None:
    """Register connection listeners on the engine to configure PRAGMAs upon connection."""

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
    """Session factory for the unit of work.

    `expire_on_commit=False` keeps retrieved rows usable after commit: repositories return
    domain objects rather than live ORM rows, and no subsequent database queries are needed after
    commit.
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
