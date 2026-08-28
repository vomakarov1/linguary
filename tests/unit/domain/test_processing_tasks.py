"""Локальные задачи обработки: идемпотентный ключ и состояния (PRD § 16-18)."""

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
    """Задача обработки в заданном состоянии."""
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
    """Идемпотентный ключ задачи — книга, тип и область действия."""
    assert make_task().identity == TaskIdentity(
        book_id="book-1", task_type=TaskType.PROCESS_TRANSLATION_UNIT, scope_id="unit-1"
    )


def test_book_level_task_uses_empty_scope() -> None:
    """У задач уровня книги область действия — пустая строка, а не `None`."""
    task = make_task(task_type=TaskType.IMPORT_BOOK, scope_id="")
    assert task.identity.scope_id == ""


def test_start_increments_attempts() -> None:
    """Захват задачи воркером увеличивает счётчик попыток."""
    task = make_task().start()
    assert task.status is TaskStatus.RUNNING
    assert task.attempts == 1


def test_succeeded_task_clears_error() -> None:
    """Успешная задача ошибки не хранит."""
    task = make_task().start().succeed()
    assert task.status is TaskStatus.SUCCEEDED
    assert task.last_error is None


def test_failed_task_keeps_typed_error() -> None:
    """Неуспешная задача хранит тип ошибки для интерфейса."""
    task = make_task().start().fail(TRANSIENT)
    assert task.status is TaskStatus.FAILED
    assert task.last_error is TRANSIENT


def test_retry_returns_task_to_pending() -> None:
    """Повтор неуспешной задачи возвращает её в `pending`."""
    task = make_task().start().fail(TRANSIENT).retry()
    assert task.status is TaskStatus.PENDING
    assert task.last_error is None
    assert task.attempts == 1


def test_permanent_error_is_not_retryable() -> None:
    """Задачу с постоянной ошибкой повторять нельзя: нужен другой файл."""
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
    """Временные ошибки, некорректный ответ модели и провал проверки повторяются."""
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
    """Переход состояния задачи вне её машины состояний отклоняется."""
    error = TRANSIENT if status is TaskStatus.FAILED else None
    task = make_task(status=status, last_error=error)
    with pytest.raises(TaskTransitionError):
        getattr(task, action)()


def test_interrupted_running_task_becomes_failed() -> None:
    """При старте приложения прерванная задача помечается неуспешной и повторяема."""
    task = make_task().start().interrupt()
    assert task.status is TaskStatus.FAILED
    assert task.is_retryable is True


def test_error_is_stored_only_for_failed_task() -> None:
    """Ошибка не хранится у задачи, которая не в состоянии `failed`."""
    with pytest.raises(TaskTransitionError):
        make_task(status=TaskStatus.RUNNING, last_error=TRANSIENT)


def test_failed_task_requires_error() -> None:
    """Неуспешная задача без ошибки не имеет смысла для интерфейса."""
    with pytest.raises(TaskTransitionError):
        make_task(status=TaskStatus.FAILED)
