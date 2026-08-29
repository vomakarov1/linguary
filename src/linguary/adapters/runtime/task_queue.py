"""Асинхронная очередь задач поверх SQLite.

Реализация порта `linguary.application.task_runner.TaskQueue`. Обращения к базе
блокирующие, поэтому выполняются в отдельном потоке: цикл событий, из которого работают
воркеры, Web UI и CLI, на время запроса к SQLite не останавливается. Соединения движка
пригодны к использованию из разных потоков — драйвер настроен на это в `adapters.db.engine`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from sqlalchemy import Engine

from linguary.adapters.runtime.task_store import SqliteTaskStore
from linguary.domain import ProcessingTask, TaskIdentity


class SqliteTaskQueue:
    """Очередь локальных задач: асинхронный вход в блокирующее хранилище."""

    def __init__(self, store: SqliteTaskStore) -> None:
        self._store = store

    @classmethod
    def for_engine(
        cls, engine: Engine, *, id_factory: Callable[[], str] | None = None
    ) -> SqliteTaskQueue:
        """Очередь поверх готового движка базы."""
        return cls(SqliteTaskStore.for_engine(engine, id_factory=id_factory))

    async def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Ставит задачу в очередь; повторная постановка того же ключа дубликата не создаёт."""
        return await asyncio.to_thread(self._store.enqueue, identity)

    async def claim_next_pending(self) -> ProcessingTask | None:
        """Захватывает очередную задачу `pending`, переводя её в `running`."""
        return await asyncio.to_thread(self._store.claim_next_pending)

    async def save(self, task: ProcessingTask) -> None:
        """Сохраняет новое состояние задачи."""
        await asyncio.to_thread(self._store.save, task)

    async def list_running(self) -> tuple[ProcessingTask, ...]:
        """Задачи в состоянии `running` в порядке их создания."""
        return await asyncio.to_thread(self._store.list_running)


__all__ = ["SqliteTaskQueue"]
