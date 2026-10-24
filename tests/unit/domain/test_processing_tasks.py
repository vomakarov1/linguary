"""Local processing tasks: idempotency key and states (PRD § 16-18)."""

from __future__ import annotations

import pytest

from linguary.domain import (
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskIdentity,
    TaskStatus,
    TaskTransitionError,
    TaskType,
)

TRANSIENT = TaskError(type=ErrorType.TRANSIENT, message="Провайдер вернул 429")
PERMANENT = TaskError(type=ErrorType.PERMANENT, message="Файл не является EPUB")


def make_task(
    *,
    status: TaskStatus = TaskStatus.PENDING,
    attempts: int = 0,
    last_error: TaskError | None = None,
    task_type: TaskType = TaskType.PROCESS_TRANSLATION_UNIT,
    scope_id: str = "unit-1",
) -> ProcessingTask:
    """A processing task in the given status."""
    return ProcessingTask(
        id="task-1",
        book_id="book-1",
        task_type=task_type,
        scope_id=scope_id,
        status=status,
        attempts=attempts,
        last_error=last_error,
    )


def test_identity_is_idempotency_key() -> None:
    """A task's idempotency key consists of book, type, and scope."""
    assert make_task().identity == TaskIdentity(
        book_id="book-1", task_type=TaskType.PROCESS_TRANSLATION_UNIT, scope_id="unit-1"
    )


def test_book_level_task_uses_empty_scope() -> None:
    """For book-level tasks, scope is an empty string, not None."""
    task = make_task(task_type=TaskType.IMPORT_BOOK, scope_id="")
    assert task.identity.scope_id == ""


def test_start_increments_attempts() -> None:
    """Claiming a task by a worker increments the attempt count."""
    task = make_task().start()
    assert task.status is TaskStatus.RUNNING
    assert task.attempts == 1


def test_succeeded_task_clears_error() -> None:
    """A succeeded task does not retain an error."""
    task = make_task().start().succeed()
    assert task.status is TaskStatus.SUCCEEDED
    assert task.last_error is None


def test_failed_task_keeps_typed_error() -> None:
    """A failed task retains a typed error for the interface."""
    task = make_task().start().fail(TRANSIENT)
    assert task.status is TaskStatus.FAILED
    assert task.last_error is TRANSIENT


def test_retry_returns_task_to_pending() -> None:
    """Retrying a failed task resets it to 'pending'."""
    task = make_task().start().fail(TRANSIENT).retry()
    assert task.status is TaskStatus.PENDING
    assert task.last_error is None
    assert task.attempts == 1


def test_permanent_error_is_not_retryable() -> None:
    """A task with a permanent error cannot be retried: a different file is required."""
    task = make_task(task_type=TaskType.IMPORT_BOOK, scope_id="").start().fail(PERMANENT)
    assert task.is_retryable is False
    with pytest.raises(TaskTransitionError):
        task.retry()


@pytest.mark.parametrize(
    "error_type",
    [
        pytest.param(ErrorType.TRANSIENT, id="transient"),
        pytest.param(ErrorType.MODEL_OUTPUT, id="model-output"),
        pytest.param(ErrorType.VALIDATION, id="validation"),
    ],
)
def test_recoverable_errors_are_retryable(error_type: ErrorType) -> None:
    """Transient errors, model output errors, and validation errors are retryable."""
    assert error_type.is_retryable is True


@pytest.mark.parametrize(
    ("status", "action"),
    [
        pytest.param(TaskStatus.RUNNING, "start", id="start-running"),
        pytest.param(TaskStatus.PENDING, "succeed", id="succeed-pending"),
        pytest.param(TaskStatus.SUCCEEDED, "retry", id="retry-succeeded"),
    ],
)
def test_forbidden_task_transition(status: TaskStatus, action: str) -> None:
    """A task state transition outside its state machine is rejected."""
    error = TRANSIENT if status is TaskStatus.FAILED else None
    task = make_task(status=status, last_error=error)
    with pytest.raises(TaskTransitionError):
        getattr(task, action)()


def test_interrupted_running_task_becomes_failed() -> None:
    """On application startup, an interrupted task is marked failed and remains retryable."""
    task = make_task().start().interrupt()
    assert task.status is TaskStatus.FAILED
    assert task.is_retryable is True


def test_error_is_stored_only_for_failed_task() -> None:
    """An error is not stored for a task that is not in the 'failed' state."""
    with pytest.raises(TaskTransitionError):
        make_task(status=TaskStatus.RUNNING, last_error=TRANSIENT)


def test_failed_task_requires_error() -> None:
    """A failed task without an error is invalid for the interface."""
    with pytest.raises(TaskTransitionError):
        make_task(status=TaskStatus.FAILED)
