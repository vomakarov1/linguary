"""Asynchronous task queue on top of SQLite.

Implementation of port `linguary.application.task_runner.TaskQueue`. Database operations
are blocking and thus executed in a separate worker thread: the event loop running
workers, Web UI, and CLI is never stalled by SQLite queries. Engine connections
are thread-safe across threads — configured accordingly in `adapters.db.engine`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from sqlalchemy import Engine

from linguary.adapters.runtime.task_store import SqliteTaskStore
from linguary.domain import ProcessingTask, TaskIdentity


class SqliteTaskQueue:
    """Local task queue: async interface to blocking persistence store."""

    def __init__(self, store: SqliteTaskStore) -> None:
        self._store = store

    @classmethod
    def for_engine(
        cls, engine: Engine, *, id_factory: Callable[[], str] | None = None
    ) -> SqliteTaskQueue:
        """Task queue backed by an existing database engine."""
        return cls(SqliteTaskStore.for_engine(engine, id_factory=id_factory))

    async def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Enqueues a task; reenqueuing an existing identity creates no duplicates."""
        return await asyncio.to_thread(self._store.enqueue, identity)

    async def claim_next_pending(self) -> ProcessingTask | None:
        """Claims the next `pending` task, transitioning it to `running`."""
        return await asyncio.to_thread(self._store.claim_next_pending)

    async def save(self, task: ProcessingTask) -> None:
        """Persists the updated task state."""
        await asyncio.to_thread(self._store.save, task)

    async def list_running(self) -> tuple[ProcessingTask, ...]:
        """Tasks currently in `running` state in creation order."""
        return await asyncio.to_thread(self._store.list_running)


__all__ = ["SqliteTaskQueue"]
