"""Local runtime environment: background task queue and worker pool.

The public adapter entry point consists of `SqliteTaskQueue` (implementing port
`linguary.application.task_runner.TaskQueue`), `TaskWorkerPool` configured with
`WorkerSettings`, and `ExecutorLock` — an executor marker indicating whether another
process is running tasks against the same database. Blocking database operations are encapsulated
in `SqliteTaskStore`: the queue wraps them for the event loop, and `to_async_handler` adapts
synchronous application use cases in the same manner.
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
