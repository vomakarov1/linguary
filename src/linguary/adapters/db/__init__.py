"""Data storage: SQLite via SQLAlchemy, repositories, and unit of work.

The adapter's public entry point: create an engine (`create_database_engine`) and obtain a
transactional boundary (`SqlAlchemyUnitOfWork.for_engine`). The table schema is defined in `models`,
and domain entity transformations are in `mapping`; the application layer does not import them and
operates through `linguary.application.ports.persistence` ports.
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
