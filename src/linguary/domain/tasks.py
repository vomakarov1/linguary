"""Local resumable book processing tasks.

A task is identified by the idempotent key `book_id + task_type + scope_id`, stores
an attempt counter, and tracks typed errors. A successfully completed task is not
re-executed after application restart (PRD § 16-18, ADR § 3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from linguary.domain.errors import TaskTransitionError


class TaskType(StrEnum):
    """Types of long-running operations executed by the local runner."""

    IMPORT_BOOK = "import_book"
    ANALYZE_CHAPTER = "analyze_chapter"
    PROCESS_TRANSLATION_UNIT = "process_translation_unit"
    REGENERATE_BLOCK = "regenerate_block"
    EXPORT_BOOK = "export_book"


class TaskStatus(StrEnum):
    """States of an individual task (PRD § 18)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ErrorType(StrEnum):
    """Processing error types (PRD § 16)."""

    TRANSIENT = "TransientError"
    MODEL_OUTPUT = "ModelOutputError"
    VALIDATION = "ValidationError"
    CONFIGURATION = "ConfigurationError"
    PERMANENT = "PermanentError"

    @property
    def is_retryable(self) -> bool:
        """Whether a task failed with this error can be retried.

        `PermanentError` (unreadable EPUB, unsupported input) cannot be retried:
        the user is prompted to upload a different file. `ConfigurationError` (invalid
        provider key, model not found) is not retried automatically, but the user
        can rerun the task after correcting LLM settings without needing another file.
        """
        return self is not ErrorType.PERMANENT

    @property
    def is_retried_automatically(self) -> bool:
        """Whether retrying the call without user intervention makes sense.

        Neither permanent nor configuration errors resolve upon retry: the former
        requires a different file, while the latter requires corrected configuration.
        """
        return self not in {ErrorType.PERMANENT, ErrorType.CONFIGURATION}


@dataclass(frozen=True, slots=True)
class TaskError:
    """Typed task error: error type, short message, and technical details."""

    type: ErrorType
    message: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise TaskTransitionError("сообщение об ошибке задачи пусто")


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Idempotent task key.

    For book-level tasks (import, export), `scope_id` is an empty string rather than `None`:
    otherwise, the SQLite unique index cannot distinguish such tasks.
    """

    book_id: str
    task_type: TaskType
    scope_id: str = ""

    def __post_init__(self) -> None:
        if not self.book_id.strip():
            raise TaskTransitionError("задача не привязана к книге")


@dataclass(frozen=True, slots=True)
class ProcessingTask:
    """Local book processing task.

    State transitions return a new instance: the entity is immutable, and persistence
    is handled by adapters via application use cases.
    """

    id: str
    book_id: str
    task_type: TaskType
    scope_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_error: TaskError | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise TaskTransitionError("идентификатор задачи пуст")
        if not self.book_id.strip():
            raise TaskTransitionError("задача не привязана к книге")
        if self.attempts < 0:
            raise TaskTransitionError("счётчик попыток задачи не может быть отрицательным")
        if self.status is TaskStatus.FAILED and self.last_error is None:
            raise TaskTransitionError(f"у неуспешной задачи {self.id} не указана ошибка")
        if self.status is not TaskStatus.FAILED and self.last_error is not None:
            raise TaskTransitionError(
                f"ошибка задачи {self.id} сохранена вне состояния failed: {self.status}"
            )

    @property
    def identity(self) -> TaskIdentity:
        """Idempotent task key."""
        return TaskIdentity(
            book_id=self.book_id, task_type=self.task_type, scope_id=self.scope_id
        )

    @property
    def is_retryable(self) -> bool:
        """Whether the user can rerun a failed task."""
        return (
            self.status is TaskStatus.FAILED
            and self.last_error is not None
            and self.last_error.type.is_retryable
        )

    def start(self) -> ProcessingTask:
        """Task claim by a worker: `pending -> running` incrementing attempts."""
        self._ensure_status(TaskStatus.PENDING, target=TaskStatus.RUNNING)
        return replace(self, status=TaskStatus.RUNNING, attempts=self.attempts + 1)

    def succeed(self) -> ProcessingTask:
        """Successful completion: `running -> succeeded`."""
        self._ensure_status(TaskStatus.RUNNING, target=TaskStatus.SUCCEEDED)
        return replace(self, status=TaskStatus.SUCCEEDED, last_error=None)

    def fail(self, error: TaskError) -> ProcessingTask:
        """Failed completion: `running -> failed` with a typed error."""
        self._ensure_status(TaskStatus.RUNNING, target=TaskStatus.FAILED)
        return replace(self, status=TaskStatus.FAILED, last_error=error)

    def retry(self) -> ProcessingTask:
        """Task retry by user: `failed -> pending`."""
        self._ensure_status(TaskStatus.FAILED, target=TaskStatus.PENDING)
        if not self.is_retryable:
            raise TaskTransitionError(
                f"задача {self.id} упала с невосстановимой ошибкой и повтору не подлежит"
            )
        return replace(self, status=TaskStatus.PENDING, last_error=None)

    def interrupt(self) -> ProcessingTask:
        """Interruption on application startup: `running -> failed` (ALG-02)."""
        return self.fail(
            TaskError(
                type=ErrorType.TRANSIENT,
                message="Задача прервана перезапуском приложения",
                detail="Interrupted",
            )
        )

    def _ensure_status(self, expected: TaskStatus, *, target: TaskStatus) -> None:
        if self.status is not expected:
            raise TaskTransitionError(
                f"переход задачи {self.id} {self.status} -> {target} запрещён: "
                f"ожидалось состояние {expected}"
            )
