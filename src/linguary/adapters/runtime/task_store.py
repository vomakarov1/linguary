"""SQLite-backed task queue: enqueueing, claiming, and persisting outcomes (ALG-02).

Encapsulates blocking database operations. Write operations (`enqueue`, `claim_next_pending`,
`save`) open explicit `BEGIN IMMEDIATE` transactions with bounded transient lock retry so
SQLite acquires the writer lock up front instead of upgrading a deferred read lock and failing
immediately under contention. The conditional update verifying updated row count in claiming
serves as a second barrier so a task claimed concurrently can never be claimed twice.

Task state transitions are performed by domain entities: this adapter is responsible for atomicity, not business rules.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
from time import sleep
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import Engine, case, exists, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from linguary.adapters.db import mapping
from linguary.adapters.db.engine import create_session_factory
from linguary.adapters.db.models import ProcessingTaskRow, WorkBindingRow
from linguary.adapters.db.repositories import SqlAlchemyTaskRepository
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import ProcessingTask, TaskIdentity, TaskStatus, TaskType

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
        self._wakeup_lock = threading.Lock()
        self._wakeup_callbacks: list[Callable[[], None]] = []

    def subscribe_wakeup(self, callback: Callable[[], None]) -> None:
        """Registers a thread-safe callback invoked when queue work may be available."""
        with self._wakeup_lock:
            if callback not in self._wakeup_callbacks:
                self._wakeup_callbacks.append(callback)

    def unsubscribe_wakeup(self, callback: Callable[[], None]) -> None:
        """Removes a previously registered wakeup callback."""
        with self._wakeup_lock:
            if callback in self._wakeup_callbacks:
                self._wakeup_callbacks.remove(callback)

    def notify_work_available(self) -> None:
        """Notifies registered wakeup callbacks that claimable work may exist."""
        with self._wakeup_lock:
            callbacks = tuple(self._wakeup_callbacks)
        for callback in callbacks:
            callback()

    def new_task_id(self) -> str:
        """Return an identifier using the same factory as persisted queue tasks."""
        return self._id_factory()

    @classmethod
    def for_engine(
        cls, engine: Engine, *, id_factory: Callable[[], str] | None = None
    ) -> SqliteTaskStore:
        """Task queue with default session factory for the given engine."""
        return cls(create_session_factory(engine), id_factory=id_factory)

    def _run_immediate_write[T](self, action: Callable[[Session], T]) -> T:
        """Execute a short write transaction under `BEGIN IMMEDIATE` with bounded lock retry."""
        session = self._session_factory()
        try:
            # A busy writer lock is transient. Roll back and retry the write itself;
            # matches unit_of_work transaction startup backoff and retry count.
            for attempt in range(1, 11):
                try:
                    session.connection().exec_driver_sql(_BEGIN_IMMEDIATE)
                    result = action(session)
                    session.commit()
                    return result
                except OperationalError as error:
                    session.rollback()
                    message = str(error).lower()
                    if "locked" not in message and "busy" not in message:
                        raise
                    if attempt == 10:
                        raise PersistenceError(
                            "база данных занята после ограниченного повтора"
                        ) from None
                    sleep(min(0.005 * attempt, 0.02))
                except BaseException:
                    session.rollback()
                    raise
        finally:
            session.close()
        raise AssertionError("unreachable")

    def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Enqueues a task; reenqueuing an existing identity creates no duplicates."""
        task_id: str | None = None

        def _insert(session: Session) -> tuple[ProcessingTask, bool]:
            nonlocal task_id
            repository = SqlAlchemyTaskRepository(session)
            existing = repository.find_by_identity(identity)
            if existing is not None:
                return existing, False
            if task_id is None:
                task_id = self._id_factory()
            task = ProcessingTask(
                id=task_id,
                book_id=identity.book_id,
                task_type=identity.task_type,
                scope_id=identity.scope_id,
                operation_id=identity.operation_id,
                policy_version=identity.policy_version,
            )
            repository.add(task)
            return task, True

        try:
            task, created = self._run_immediate_write(_insert)
        except IntegrityError:
            # The key was claimed by another writer between lookup and insert: the queue is idempotent
            # on identity, so return the already-enqueued task.
            stored = self._find(identity)
            if stored is None:
                raise
            if stored.status is TaskStatus.PENDING:
                self.notify_work_available()
            return stored
        if created or task.status is TaskStatus.PENDING:
            self.notify_work_available()
        return task

    def claim_next_pending(self) -> ProcessingTask | None:
        """Claims the next `pending` task, transitioning it to `running` (ALG-02)."""
        with self._session_factory() as read_session:
            if self._next_pending(read_session) is None:
                return None

        def _claim(session: Session) -> ProcessingTask | None:
            candidate = self._next_pending(session)
            if candidate is None:
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
            return claimed if cast(CursorResult[Any], updated).rowcount == 1 else None

        return self._run_immediate_write(_claim)

    def save(self, task: ProcessingTask) -> ProcessingTask:
        """Persists task state and returns its effective state.

        Returning to glossary review deletes tasks from the canceled run, including
        any task currently being executed (PRD § 21.5). There is no longer a row to record
        its outcome, so the passed task is returned for canceled work. If a newer operation
        superseded the task, its stored superseded state is returned.
        """

        def _save(session: Session) -> ProcessingTask:
            repository = SqlAlchemyTaskRepository(session)
            try:
                repository.save(task)
            except PersistenceError:
                current = repository.get(task.id)
                if current is None:
                    return task
                if current.status is TaskStatus.SUPERSEDED and current.identity == task.identity:
                    return current
                raise
            return task

        return self._run_immediate_write(_save)

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
        paused_book = exists(
            select(1).where(
                WorkBindingRow.entity_kind == "book_queue",
                WorkBindingRow.entity_id == ProcessingTaskRow.book_id,
                WorkBindingRow.book_id == ProcessingTaskRow.book_id,
                func.coalesce(func.json_extract(WorkBindingRow.payload_json, "$.paused"), 1) != 0,
            )
        )
        low_priority = (
            TaskType.PREPARE_ILLUSTRATION_REFERENCE.value,
            TaskType.PREPARE_CHAPTER_ILLUSTRATIONS.value,
        )
        priority_expr = case(
            (ProcessingTaskRow.task_type.in_(low_priority), 1),
            else_=0,
        )
        row = session.scalars(
            select(ProcessingTaskRow)
            .where(
                ProcessingTaskRow.status == TaskStatus.PENDING.value,
                ~paused_book,
            )
            .order_by(priority_expr, ProcessingTaskRow.created_at, ProcessingTaskRow.id)
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
