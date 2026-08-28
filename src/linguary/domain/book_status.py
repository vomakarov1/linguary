"""Состояния книги и таблица разрешённых переходов (ALG-06).

Состояние книги пересчитывает прикладной слой единой функцией после завершения каждой
задачи и каждого пользовательского действия; задачи и адаптеры состояние книги напрямую
не пишут. Домен владеет таблицей переходов и отвечает на вопрос, разрешён ли переход.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.errors import BookStatusTransitionError
from linguary.domain.tasks import ErrorType, TaskType


class BookStatus(StrEnum):
    """Пользовательские состояния книги (PRD § 18)."""

    IMPORTING = "IMPORTING"
    IMPORTED = "IMPORTED"
    ANALYZING = "ANALYZING"
    GLOSSARY_REVIEW = "GLOSSARY_REVIEW"
    READY_TO_PROCESS = "READY_TO_PROCESS"
    PROCESSING = "PROCESSING"
    READY_TO_EXPORT = "READY_TO_EXPORT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BookEvent(StrEnum):
    """События, которые меняют состояние книги (ALG-06)."""

    FILE_UPLOADED = "file_uploaded"
    IMPORT_SUCCEEDED = "import_succeeded"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    GLOSSARY_REVIEW_COMPLETED = "glossary_review_completed"
    PROCESSING_RUN_STARTED = "processing_run_started"
    PROCESSING_COMPLETED = "processing_completed"
    EXPORT_SUCCEEDED = "export_succeeded"
    BLOCK_REGENERATION_REQUESTED = "block_regeneration_requested"
    PROCESSING_RESET_CONFIRMED = "processing_reset_confirmed"
    IMPORT_PERMANENTLY_FAILED = "import_permanently_failed"


@dataclass(frozen=True, slots=True)
class BookStatusTransition:
    """Одна строка таблицы переходов.

    `source is None` означает, что книги ещё нет: загрузка файла её создаёт.
    """

    event: BookEvent
    source: BookStatus | None
    target: BookStatus


#: Полная таблица разрешённых переходов состояния книги (ALG-06).
#: Переходы вне этой таблицы запрещены и отклоняются ошибкой.
BOOK_STATUS_TRANSITIONS: tuple[BookStatusTransition, ...] = (
    BookStatusTransition(BookEvent.FILE_UPLOADED, None, BookStatus.IMPORTING),
    BookStatusTransition(
        BookEvent.IMPORT_SUCCEEDED, BookStatus.IMPORTING, BookStatus.IMPORTED
    ),
    BookStatusTransition(
        BookEvent.ANALYSIS_STARTED, BookStatus.IMPORTED, BookStatus.ANALYZING
    ),
    BookStatusTransition(
        BookEvent.ANALYSIS_COMPLETED, BookStatus.ANALYZING, BookStatus.GLOSSARY_REVIEW
    ),
    BookStatusTransition(
        BookEvent.GLOSSARY_REVIEW_COMPLETED,
        BookStatus.GLOSSARY_REVIEW,
        BookStatus.READY_TO_PROCESS,
    ),
    BookStatusTransition(
        BookEvent.PROCESSING_RUN_STARTED,
        BookStatus.READY_TO_PROCESS,
        BookStatus.PROCESSING,
    ),
    BookStatusTransition(
        BookEvent.PROCESSING_COMPLETED, BookStatus.PROCESSING, BookStatus.READY_TO_EXPORT
    ),
    BookStatusTransition(
        BookEvent.EXPORT_SUCCEEDED, BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED
    ),
    BookStatusTransition(
        BookEvent.BLOCK_REGENERATION_REQUESTED,
        BookStatus.READY_TO_EXPORT,
        BookStatus.PROCESSING,
    ),
    BookStatusTransition(
        BookEvent.BLOCK_REGENERATION_REQUESTED,
        BookStatus.COMPLETED,
        BookStatus.PROCESSING,
    ),
    BookStatusTransition(
        BookEvent.PROCESSING_RESET_CONFIRMED,
        BookStatus.PROCESSING,
        BookStatus.GLOSSARY_REVIEW,
    ),
    BookStatusTransition(
        BookEvent.PROCESSING_RESET_CONFIRMED,
        BookStatus.READY_TO_EXPORT,
        BookStatus.GLOSSARY_REVIEW,
    ),
    BookStatusTransition(
        BookEvent.PROCESSING_RESET_CONFIRMED,
        BookStatus.COMPLETED,
        BookStatus.GLOSSARY_REVIEW,
    ),
    BookStatusTransition(
        BookEvent.IMPORT_PERMANENTLY_FAILED, BookStatus.IMPORTING, BookStatus.FAILED
    ),
)


def is_transition_allowed(source: BookStatus | None, target: BookStatus) -> bool:
    """Разрешён ли переход между состояниями книги."""
    return any(
        transition.source == source and transition.target is target
        for transition in BOOK_STATUS_TRANSITIONS
    )


def allowed_targets(source: BookStatus | None) -> frozenset[BookStatus]:
    """Состояния, в которые книга может перейти из текущего."""
    return frozenset(
        transition.target
        for transition in BOOK_STATUS_TRANSITIONS
        if transition.source == source
    )


def apply_book_event(source: BookStatus | None, event: BookEvent) -> BookStatus:
    """Состояние книги после события.

    Событие, недопустимое в текущем состоянии, отклоняется ошибкой: прикладной слой
    превращает её в отказ пользовательского действия (в Web UI — 409).
    """
    for transition in BOOK_STATUS_TRANSITIONS:
        if transition.event is event and transition.source == source:
            return transition.target
    raise BookStatusTransitionError(
        f"событие {event} недопустимо в состоянии {source}"
    )


def ensure_transition_allowed(source: BookStatus | None, target: BookStatus) -> None:
    """Проверяет переход и отклоняет его, если такой строки в таблице нет."""
    if not is_transition_allowed(source, target):
        raise BookStatusTransitionError(f"переход {source} -> {target} запрещён")


def status_after_task_failure(
    current: BookStatus, *, task_type: TaskType, error_type: ErrorType
) -> BookStatus:
    """Состояние книги после неуспешной задачи (ALG-06).

    Неуспех повторяемой задачи состояние книги не меняет: задача показывается как
    `failed` с действием повтора. `FAILED` зарезервирован за постоянной ошибкой уровня
    книги — постоянной ошибкой импорта.
    """
    if error_type.is_retryable or task_type is not TaskType.IMPORT_BOOK:
        return current
    return apply_book_event(current, BookEvent.IMPORT_PERMANENTLY_FAILED)
