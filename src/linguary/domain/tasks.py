"""Локальные возобновляемые задачи обработки книги.

Задача идентифицируется идемпотентным ключом `book_id + task_type + scope_id`, хранит
счётчик попыток и типизированную ошибку. Успешно завершённая задача после перезапуска
приложения повторно не выполняется (PRD § 16-18, ADR § 3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from linguary.domain.errors import TaskTransitionError


class TaskType(StrEnum):
    """Типы длительных операций, которые исполняет локальный runner."""

    IMPORT_BOOK = "import_book"
    ANALYZE_CHAPTER = "analyze_chapter"
    PROCESS_TRANSLATION_UNIT = "process_translation_unit"
    REGENERATE_BLOCK = "regenerate_block"
    EXPORT_BOOK = "export_book"


class TaskStatus(StrEnum):
    """Состояния отдельной задачи (PRD § 18)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ErrorType(StrEnum):
    """Типы ошибок обработки (PRD § 16)."""

    TRANSIENT = "TransientError"
    MODEL_OUTPUT = "ModelOutputError"
    VALIDATION = "ValidationError"
    CONFIGURATION = "ConfigurationError"
    PERMANENT = "PermanentError"

    @property
    def is_retryable(self) -> bool:
        """Можно ли повторить задачу, упавшую с такой ошибкой.

        `PermanentError` (нечитаемый EPUB, неподдержанный вход) повтору не подлежит:
        пользователю предлагается загрузить другой файл. `ConfigurationError` (ключ
        провайдера не принят, модель не найдена) автоматически не повторяется, но
        пользователь запускает задачу заново, исправив настройки LLM: книга при этом
        остаётся прежней, и другой файл не нужен.
        """
        return self is not ErrorType.PERMANENT

    @property
    def is_retried_automatically(self) -> bool:
        """Имеет ли смысл повторять вызов без вмешательства пользователя.

        Ни постоянная ошибка, ни ошибка настройки от новой попытки не исчезают: первая
        требует другого файла, вторая — исправленной конфигурации.
        """
        return self not in {ErrorType.PERMANENT, ErrorType.CONFIGURATION}


@dataclass(frozen=True, slots=True)
class TaskError:
    """Типизированная ошибка задачи: тип, короткое сообщение и техническая деталь."""

    type: ErrorType
    message: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise TaskTransitionError("сообщение об ошибке задачи пусто")


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Идемпотентный ключ задачи.

    Для задач уровня книги (импорт, экспорт) `scope_id` — пустая строка, а не `None`:
    иначе уникальный индекс в SQLite не различает такие задачи между собой.
    """

    book_id: str
    task_type: TaskType
    scope_id: str = ""

    def __post_init__(self) -> None:
        if not self.book_id.strip():
            raise TaskTransitionError("задача не привязана к книге")


@dataclass(frozen=True, slots=True)
class ProcessingTask:
    """Локальная задача обработки книги.

    Переходы состояния возвращают новый экземпляр: сущность неизменяема, а запись
    в хранилище выполняет adapters через прикладной сценарий.
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
        """Идемпотентный ключ задачи."""
        return TaskIdentity(
            book_id=self.book_id, task_type=self.task_type, scope_id=self.scope_id
        )

    @property
    def is_retryable(self) -> bool:
        """Может ли пользователь запустить неуспешную задачу повторно."""
        return (
            self.status is TaskStatus.FAILED
            and self.last_error is not None
            and self.last_error.type.is_retryable
        )

    def start(self) -> ProcessingTask:
        """Захват задачи воркером: `pending -> running` с увеличением попыток."""
        self._ensure_status(TaskStatus.PENDING, target=TaskStatus.RUNNING)
        return replace(self, status=TaskStatus.RUNNING, attempts=self.attempts + 1)

    def succeed(self) -> ProcessingTask:
        """Успешное завершение: `running -> succeeded`."""
        self._ensure_status(TaskStatus.RUNNING, target=TaskStatus.SUCCEEDED)
        return replace(self, status=TaskStatus.SUCCEEDED, last_error=None)

    def fail(self, error: TaskError) -> ProcessingTask:
        """Неуспешное завершение: `running -> failed` с типизированной ошибкой."""
        self._ensure_status(TaskStatus.RUNNING, target=TaskStatus.FAILED)
        return replace(self, status=TaskStatus.FAILED, last_error=error)

    def retry(self) -> ProcessingTask:
        """Повтор задачи пользователем: `failed -> pending`."""
        self._ensure_status(TaskStatus.FAILED, target=TaskStatus.PENDING)
        if not self.is_retryable:
            raise TaskTransitionError(
                f"задача {self.id} упала с невосстановимой ошибкой и повтору не подлежит"
            )
        return replace(self, status=TaskStatus.PENDING, last_error=None)

    def interrupt(self) -> ProcessingTask:
        """Прерывание при старте приложения: `running -> failed` (ALG-02)."""
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
