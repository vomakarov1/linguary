"""Локальная среда выполнения: очередь фоновых задач и воркеры над ней.

Публичный вход адаптера — `SqliteTaskQueue` (реализация порта
`linguary.application.task_runner.TaskQueue`), `TaskWorkerPool` с настройкой
`WorkerSettings` и `ExecutorLock` — отметка исполнителей, по которой видно, выполняет ли
задачи той же базы кто-то ещё. Блокирующие операции с базой вынесены в `SqliteTaskStore`:
очередь оборачивает их для цикла событий, а `to_async_handler` тем же способом подключает
синхронные прикладные сценарии.
"""

from linguary.adapters.runtime.executor_lock import ExecutorLock
from linguary.adapters.runtime.task_queue import SqliteTaskQueue
from linguary.adapters.runtime.task_store import SqliteTaskStore
from linguary.adapters.runtime.worker import (
    TaskWorkerPool,
    WorkerSettings,
    to_async_handler,
)

__all__ = [
    "ExecutorLock",
    "SqliteTaskQueue",
    "SqliteTaskStore",
    "TaskWorkerPool",
    "WorkerSettings",
    "to_async_handler",
]
