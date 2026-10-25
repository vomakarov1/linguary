"""Local task runner: claim, execute, and record task outcome (ALG-02).

The runner owns the state machine of a single task and nothing more. It fetches the next
task from the queue, looks up the handler for its task type, executes it, and records the
outcome — `succeeded` or `failed` with a typed error. Scheduling, concurrency, and SQLite
reside behind the `TaskQueue` port and in `linguary.adapters.runtime`: the application layer
is agnostic of them (ADR § 3).

State transitions originate in the domain (`ProcessingTask.start/succeed/fail/interrupt`)
rather than being duplicated here: the adapter guarantees atomic claim, but the transition
rules remain unified across the application.

The runner is async: in phases 3-4 it is driven by FastAPI and CLI, while `TranslationUnit`
processing awaits LLM HTTP responses; thus configurable parallelism (PRD § 25) amounts to
I/O waiting. Synchronous use cases are adapted via `to_async_handler` in the adapter layer.

Handler error classification is a separate policy (PRD § 16) not decided here: defaults
come from `linguary.application.retry_policy`, while app composition can substitute it via `classify_error`.
"""

from __future__ import annotations

import asyncio

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from linguary.application.retry_policy import classify_error as classify_task_error
from linguary.domain import (
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskIdentity,
    TaskType,
)

#: Handler for a single task type. It constructs the use case command itself: runner
#: only knows its idempotent key and cannot build input for ImportBook or ExportBook.
type TaskHandler = Callable[[ProcessingTask], Awaitable[None]]

#: Classifier mapping unexpected handler exceptions into typed task errors.
type TaskErrorClassifier = Callable[[Exception], TaskError]


class TaskQueue(Protocol):
    """Local task queue.

    Task enqueuing is idempotent by `book_id + task_type + scope_id`, and claiming
    is atomic: two attempts cannot claim the same task (ALG-02).
    """

    async def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Enqueues a task.

        If a task with this identity already exists, it is returned in its current status:
        successfully completed tasks are not re-executed, while failed tasks are restarted
        explicitly by the user (PRD § 18).
        """

    async def claim_next_pending(self) -> ProcessingTask | None:
        """Claims the next `pending` task, transitioning it to `running`.

        Returns the claimed task or `None` if the queue is empty.
        """

    async def save(self, task: ProcessingTask) -> None:
        """Persists the new status of a task.

        Saving does not resurrect a task removed from the queue: returning to glossary
        review removes tasks of the cancelled run along with active ones, leaving nowhere
        to record its outcome (PRD § 21.5).
        """

    async def list_running(self) -> tuple[ProcessingTask, ...]:
        """Tasks currently in `running` status in order of creation."""


class TaskRunner:
    """Executes queue tasks one at a time, adhering to the ALG-02 state machine."""

    def __init__(
        self,
        *,
        queue: TaskQueue,
        handlers: Mapping[TaskType, TaskHandler],
        classify_error: TaskErrorClassifier | None = None,
        maintenance: Callable[[], None] | None = None,
    ) -> None:
        self._queue = queue
        self._maintenance = maintenance
        self._handlers = dict(handlers)
        self._classify_error = classify_error or classify_task_error

    async def recover_interrupted(self) -> tuple[ProcessingTask, ...]:
        """Transitions `running` tasks to `failed` upon application startup (ALG-02).

        An application killed while executing a task leaves it in `running`, where
        no worker executes it. Such a task is marked interrupted, allowing the user
        to retry it (PRD § 17-18).
        """
        interrupted = tuple(task.interrupt() for task in await self._queue.list_running())
        for task in interrupted:
            await self._queue.save(task)
        return interrupted

    async def run_next(self) -> ProcessingTask | None:
        """Executes a single queue task and returns its completed state.

        `None` indicates no pending tasks in the queue. Cancellation of the outer
        coroutine is not caught: the task remains in `running` and will be marked
        interrupted on next application start — accurately reflecting that no final
        result occurred.
        """
        if self._maintenance is not None:
            await asyncio.to_thread(self._maintenance)
        task = await self._queue.claim_next_pending()
        if task is None:
            return None
        return await self._finish(task)

    async def _finish(self, task: ProcessingTask) -> ProcessingTask:
        """Executes claimed task and records its outcome."""
        handler = self._handlers.get(task.task_type)
        if handler is None:
            return await self._save(task.fail(_unregistered_handler_error(task)))
        try:
            await handler(task)
        except Exception as error:  # any use case exception results in a failed task
            return await self._save(task.fail(self._classify_error(error)))
        return await self._save(task.succeed())

    async def _save(self, task: ProcessingTask) -> ProcessingTask:
        """Persists the completed state of the task."""
        await self._queue.save(task)
        return task


def _unregistered_handler_error(task: ProcessingTask) -> TaskError:
    """Task error when no handler is registered for its task type.

    Such a task cannot be retried: the application was built without a handler for this
    type, and the result will not change until it is wired up.
    """
    return TaskError(
        type=ErrorType.PERMANENT,
        message=f"обработчик задач типа {task.task_type} не зарегистрирован",
        detail=f"task_type={task.task_type.value}",
    )


__all__ = [
    "TaskErrorClassifier",
    "TaskHandler",
    "TaskQueue",
    "TaskRunner",
]
