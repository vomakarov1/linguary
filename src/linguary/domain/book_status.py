"""Book states and allowed transitions table (ALG-06).

Book state is recomputed by the application layer using a single function after completing
each task and each user action; tasks and adapters do not write book state directly.
The domain owns the transition table and answers whether a transition is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.errors import BookStatusTransitionError
from linguary.domain.tasks import ErrorType, TaskType


class BookStatus(StrEnum):
    """User-facing book states (PRD § 18)."""

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
    """Events that trigger book state transitions (ALG-06)."""

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
    """A single row in the transition table.

    `source is None` indicates the book does not exist yet: file upload creates it.
    """

    event: BookEvent
    source: BookStatus | None
    target: BookStatus


#: Complete table of allowed book state transitions (ALG-06).
#: Transitions outside this table are forbidden and rejected with an error.
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
    """Whether a transition between book states is allowed."""
    return any(
        transition.source == source and transition.target is target
        for transition in BOOK_STATUS_TRANSITIONS
    )


def allowed_targets(source: BookStatus | None) -> frozenset[BookStatus]:
    """States to which the book can transition from the current state."""
    return frozenset(
        transition.target
        for transition in BOOK_STATUS_TRANSITIONS
        if transition.source == source
    )


def apply_book_event(source: BookStatus | None, event: BookEvent) -> BookStatus:
    """Book state after an event.

    An event invalid in the current state is rejected with an error: the application layer
    converts it into a user action rejection (HTTP 409 in Web UI).
    """
    for transition in BOOK_STATUS_TRANSITIONS:
        if transition.event is event and transition.source == source:
            return transition.target
    raise BookStatusTransitionError(
        f"событие {event} недопустимо в состоянии {source}"
    )


def ensure_transition_allowed(source: BookStatus | None, target: BookStatus) -> None:
    """Checks a transition and rejects it if no such row exists in the table."""
    if not is_transition_allowed(source, target):
        raise BookStatusTransitionError(f"переход {source} -> {target} запрещён")


def status_after_task_failure(
    current: BookStatus, *, task_type: TaskType, error_type: ErrorType
) -> BookStatus:
    """Book state after a failed task (ALG-06).

    Failure of a retryable task does not alter book state: the task is displayed as
    `failed` with a retry action. `FAILED` is reserved for permanent book-level errors—
    permanent import failures.
    """
    if error_type.is_retryable or task_type is not TaskType.IMPORT_BOOK:
        return current
    return apply_book_event(current, BookEvent.IMPORT_PERMANENTLY_FAILED)
