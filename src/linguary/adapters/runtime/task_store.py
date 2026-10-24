"""SQLite-backed task queue: enqueueing, claiming, and persisting outcomes (ALG-02).

Encapsulates blocking database operations. Task claiming is the only place in the application
where a transaction opens with an explicit `BEGIN IMMEDIATE`: the writer acquires a write lock
immediately before candidate inspection, preventing two workers from reading the same row as available.
The conditional update verifying updated row count serves as a second barrier: even if a transaction
were ever opened deferred, a task claimed concurrently would not be claimed twice.

Task state transitions are performed by domain entities: this adapter is responsible for atomicity, not business rules.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from linguary.adapters.db import mapping
from linguary.adapters.db.engine import create_session_factory
from linguary.adapters.db.models import ProcessingTaskRow
from linguary.adapters.db.repositories import SqlAlchemyTaskRepository
from linguary.domain import ProcessingTask, TaskIdentity, TaskStatus

#: Explicit write transaction start. A deferred SQLite transaction acquires write locks only
#: on the first write, where upgrading locks after reading a candidate fails with an error
#: instead of waiting; therefore claiming begins in write mode immediately.
_BEGIN_IMMEDIATE = "BEGIN IMMEDIATE"


class SqliteTaskStore:
    """Blocking task queue operations backed by SQLite."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._id_factory = id_factory or _new_task_id

    @classmethod
    def for_engine(
        cls, engine: Engine, *, id_factory: Callable[[], str] | None = None
    ) -> SqliteTaskStore:
        """Task queue with default session factory for the given engine."""
        return cls(create_session_factory(engine), id_factory=id_factory)

    def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Enqueues a task; reenqueuing an existing identity creates no duplicates."""
        existing = self._find(identity)
        if existing is not None:
            return existing
        task = ProcessingTask(
            id=self._id_factory(),
            book_id=identity.book_id,
            task_type=identity.task_type,
            scope_id=identity.scope_id,
        )
        try:
            with self._session_factory.begin() as session:
                SqlAlchemyTaskRepository(session).add(task)
        except IntegrityError:
            # The key was claimed by another writer between lookup and insert: the queue is idempotent
            # on identity, so return the already-enqueued task.
            stored = self._find(identity)
            if stored is None:
                raise
            return stored
        return task

    def claim_next_pending(self) -> ProcessingTask | None:
        """Claims the next `pending` task, transitioning it to `running` (ALG-02)."""
        session = self._session_factory()
        try:
            session.connection().exec_driver_sql(_BEGIN_IMMEDIATE)
            candidate = self._next_pending(session)
            if candidate is None:
                session.commit()
                return None
            claimed = candidate.start()
            updated = session.execute(
                update(ProcessingTaskRow)
                .where(
                    ProcessingTaskRow.id == claimed.id,
                    ProcessingTaskRow.status == TaskStatus.PENDING.value,
                )
                .values(status=claimed.status.value, attempts=claimed.attempts)
            )
            session.commit()
            return claimed if updated.rowcount == 1 else None
        finally:
            session.close()

    def save(self, task: ProcessingTask) -> None:
        """Persists updated task state without resurrecting canceled tasks.

        Returning to glossary review deletes tasks from the canceled run, including
        any task currently being executed (PRD § 21.5). There is no longer a row to record
        its outcome: a missing row is normal completion for canceled work rather than a queue failure,
        and workers must not crash because of it.
        """
        with self._session_factory.begin() as session:
            repository = SqlAlchemyTaskRepository(session)
            if repository.get(task.id) is None:
                return
            repository.save(task)

    def list_running(self) -> tuple[ProcessingTask, ...]:
        """Tasks currently in `running` state in creation order."""
        with self._session_factory() as session:
            return self._list_by_status(session, TaskStatus.RUNNING)

    def _find(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Task by idempotent identity key."""
        with self._session_factory() as session:
            return SqlAlchemyTaskRepository(session).find_by_identity(identity)

    def _next_pending(self, session: Session) -> ProcessingTask | None:
        """Next pending task candidate: earliest `pending` task.

        The identifier in ordering serves as a stable tie-breaker: timestamps of tasks
        enqueued in a single batch may be identical down to database precision.
        """
        row = session.scalars(
            select(ProcessingTaskRow)
            .where(ProcessingTaskRow.status == TaskStatus.PENDING.value)
            .order_by(ProcessingTaskRow.created_at, ProcessingTaskRow.id)
            .limit(1)
        ).first()
        return None if row is None else mapping.to_task(row)

    def _list_by_status(
        self, session: Session, status: TaskStatus
    ) -> tuple[ProcessingTask, ...]:
        """Tasks in a specific status in creation order."""
        rows = session.scalars(
            select(ProcessingTaskRow)
            .where(ProcessingTaskRow.status == status.value)
            .order_by(ProcessingTaskRow.created_at, ProcessingTaskRow.id)
        ).all()
        return tuple(mapping.to_task(row) for row in rows)


def _new_task_id() -> str:
    """Unique identifier for a new task."""
    return str(uuid4())


__all__ = ["SqliteTaskStore"]
