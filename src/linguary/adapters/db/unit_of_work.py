"""Unit of work on top of a SQLAlchemy session.

The transaction boundary is established by the application use case: it opens `transaction()`,
executes modifications via repositories, and exits the block. Exiting without exceptions commits
the changes, while exceptions roll back the entire transaction, ensuring that partially saved
results for a single `TranslationUnit` never occur (PRD § 17).

Repositories exist only within an active transaction: accessing them outside is a contract
violation rather than a silent opening of an implicit transaction.

A single instance serves the entire application, so the active session is stored not on the
object itself, but in the execution context (`ContextVar`): web requests in FastAPI worker threads,
background worker tasks, and use case coroutines run concurrently, each managing its own
transaction. A shared instance field would make them share a single transaction, causing concurrent
operations to fail with "transaction already opened".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from time import sleep

from linguary.adapters.db.work_state import SqlAlchemyWorkRepository
from sqlalchemy.orm import Session, sessionmaker

from linguary.adapters.db.engine import create_session_factory
from linguary.adapters.db.repositories import (
    SqlAlchemyBlockResultRepository,
    SqlAlchemyBookRepository,
    SqlAlchemyChapterAnalysisRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyGlossaryRepository,
    SqlAlchemyLLMCallLogRepository,
    SqlAlchemyProcessingRunRepository,
    SqlAlchemyTaskRepository,
)
from linguary.application.ports.persistence import PersistenceError

#: Instance counter: `ContextVar` name is used solely for diagnostic purposes.
_INSTANCES = count(1)


class SqlAlchemyUnitOfWork:
    """Transactional boundary for application use cases on top of SQLite."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._current: ContextVar[Session | None] = ContextVar(
            f"linguary_unit_of_work_{next(_INSTANCES)}", default=None
        )

    @classmethod
    def for_engine(cls, engine: Engine) -> SqlAlchemyUnitOfWork:
        """Create a unit of work with a default session factory for a given engine."""
        return cls(create_session_factory(engine))

    @contextmanager
    def transaction(self) -> Iterator[SqlAlchemyUnitOfWork]:
        """Open a transaction and make repositories accessible.

        Nested transactions within the same execution context remain prohibited: they would
        indicate that the use case failed to define an explicit transaction boundary. A concurrent
        transaction in another request or background task is not considered nested because it has
        its own context.
        """
        if self._current.get() is not None:
            raise PersistenceError(
                "транзакция уже открыта: вложенные транзакции не поддерживаются"
            )
        session = self._session_factory()
        token = self._current.set(session)
        try:
            with session.begin():
                # Acquire SQLite's writer lock before application work: retries never
                # replay user code or encompass a physical model request.
                for attempt in range(3):
                    try:
                        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                        break
                    except OperationalError as error:
                        if 'locked' not in str(error).lower() and 'busy' not in str(error).lower():
                            raise
                        if attempt == 2:
                            raise PersistenceError('база данных занята после ограниченного повтора') from None
                        sleep(0.01 * (attempt + 1))
                yield self
        finally:
            self._current.reset(token)
            session.close()

    @property
    def books(self) -> SqlAlchemyBookRepository:
        """Book repository."""
        return SqlAlchemyBookRepository(self._active_session())

    @property
    def documents(self) -> SqlAlchemyDocumentRepository:
        """Canonical document model repository."""
        return SqlAlchemyDocumentRepository(self._active_session())

    @property
    def glossary(self) -> SqlAlchemyGlossaryRepository:
        """Glossary repository."""
        return SqlAlchemyGlossaryRepository(self._active_session())

    @property
    def analyses(self) -> SqlAlchemyChapterAnalysisRepository:
        """Chapter analysis repository."""
        return SqlAlchemyChapterAnalysisRepository(self._active_session())

    @property
    def runs(self) -> SqlAlchemyProcessingRunRepository:
        """Processing run repository."""
        return SqlAlchemyProcessingRunRepository(self._active_session())

    @property
    def results(self) -> SqlAlchemyBlockResultRepository:
        """Block processing result repository."""
        return SqlAlchemyBlockResultRepository(self._active_session())

    @property
    def tasks(self) -> SqlAlchemyTaskRepository:
        """Local task repository."""
        return SqlAlchemyTaskRepository(self._active_session())

    @property
    def llm_calls(self) -> SqlAlchemyLLMCallLogRepository:
        """LLM call log repository."""
        return SqlAlchemyLLMCallLogRepository(self._active_session())

    @property
    def work(self) -> SqlAlchemyWorkRepository:
        """Versioned work, provenance, and monetary reservations."""
        return SqlAlchemyWorkRepository(self._active_session())

    def _active_session(self) -> Session:
        """Active transaction session in current execution context."""
        session = self._current.get()
        if session is None:
            raise PersistenceError(
                "репозиторий доступен только внутри transaction(): "
                "прикладной сценарий обязан задать транзакционную границу явно"
            )
        return session


__all__ = ["SqlAlchemyUnitOfWork"]
