"""Retry failed task use case (PRD § 16, § 18).

The user returns a `failed` task to the queue: it transitions back to `pending`, the attempt
counter is preserved, and error details are cleared. Only tasks that make sense to retry can be
retried: a successfully completed task is not re-executed, and a task failed with a `PermanentError`
(unreadable EPUB, unsupported input) requires a different file rather than a retry attempt.
An LLM configuration error (`ConfigurationError`) is retryable: after fixing the API key or model,
the same book can resume processing.

Retrying does not duplicate results: the task use case is idempotent by its
`book_id + task_type + scope_id` key, and `TranslationUnit` results are saved in a single transaction
as a whole (PRD § 17).
"""

from __future__ import annotations

from dataclasses import dataclass

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import ProcessingTask, TaskStatus


class RetryTaskError(ApplicationError):
    """Task cannot be retried in its current state."""


class TaskNotFoundError(RetryTaskError):
    """Task with this ID was not found in persistence."""


@dataclass(frozen=True, slots=True)
class RetryTaskCommand:
    """Retry command: which task to return to the queue."""

    task_id: str


class RetryTask:
    """Returns a failed task to the queue: `failed -> pending`."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: RetryTaskCommand) -> ProcessingTask:
        """Transitions the task to `pending` and returns its updated state."""
        with self._unit_of_work.transaction() as transaction:
            task = transaction.tasks.get(command.task_id)
            if task is None:
                raise TaskNotFoundError(f"задачи {command.task_id} нет в хранилище")
            _ensure_retryable(task)
            restarted = task.retry()
            transaction.tasks.save(restarted)
        return restarted


def _ensure_retryable(task: ProcessingTask) -> None:
    """Ensures that retrying the task makes sense."""
    if task.status is TaskStatus.SUCCEEDED:
        raise RetryTaskError(
            f"задача {task.id} уже выполнена успешно и повторно не выполняется"
        )
    if task.status is not TaskStatus.FAILED:
        raise RetryTaskError(
            f"задача {task.id} в состоянии {task.status}: повторяют только неуспешные задачи"
        )
    if not task.is_retryable:
        raise RetryTaskError(
            f"задача {task.id} упала с невосстановимой ошибкой: нужен другой исходный файл, "
            f"а не повторная попытка"
        )


__all__ = [
    "RetryTask",
    "RetryTaskCommand",
    "RetryTaskError",
    "TaskNotFoundError",
]
