"""Book state machine: transition table ALG-06 (PRD § 18)."""

from __future__ import annotations

import pytest

from linguary.domain import (
    BookEvent,
    BookStatus,
    BookStatusTransitionError,
    ErrorType,
    TaskType,
    allowed_targets,
    apply_book_event,
    ensure_transition_allowed,
    is_transition_allowed,
    status_after_task_failure,
)


@pytest.mark.parametrize(
    ("source", "event", "target"),
    [
        pytest.param(None, BookEvent.FILE_UPLOADED, BookStatus.IMPORTING, id="upload-creates-book"),
        pytest.param(
            BookStatus.IMPORTING,
            BookEvent.IMPORT_SUCCEEDED,
            BookStatus.IMPORTED,
            id="import-succeeded",
        ),
        pytest.param(
            BookStatus.IMPORTED,
            BookEvent.ANALYSIS_STARTED,
            BookStatus.ANALYZING,
            id="analysis-started",
        ),
        pytest.param(
            BookStatus.ANALYZING,
            BookEvent.ANALYSIS_COMPLETED,
            BookStatus.GLOSSARY_REVIEW,
            id="analysis-completed",
        ),
        pytest.param(
            BookStatus.GLOSSARY_REVIEW,
            BookEvent.GLOSSARY_REVIEW_COMPLETED,
            BookStatus.READY_TO_PROCESS,
            id="glossary-review-completed",
        ),
        pytest.param(
            BookStatus.READY_TO_PROCESS,
            BookEvent.PROCESSING_RUN_STARTED,
            BookStatus.PROCESSING,
            id="processing-run-started",
        ),
        pytest.param(
            BookStatus.PROCESSING,
            BookEvent.PROCESSING_COMPLETED,
            BookStatus.READY_TO_EXPORT,
            id="processing-completed",
        ),
        pytest.param(
            BookStatus.READY_TO_EXPORT,
            BookEvent.EXPORT_SUCCEEDED,
            BookStatus.COMPLETED,
            id="export-succeeded",
        ),
        pytest.param(
            BookStatus.READY_TO_EXPORT,
            BookEvent.BLOCK_REGENERATION_REQUESTED,
            BookStatus.PROCESSING,
            id="regenerate-from-ready-to-export",
        ),
        pytest.param(
            BookStatus.COMPLETED,
            BookEvent.BLOCK_REGENERATION_REQUESTED,
            BookStatus.PROCESSING,
            id="regenerate-from-completed",
        ),
        pytest.param(
            BookStatus.PROCESSING,
            BookEvent.PROCESSING_RESET_CONFIRMED,
            BookStatus.GLOSSARY_REVIEW,
            id="reset-from-processing",
        ),
        pytest.param(
            BookStatus.READY_TO_EXPORT,
            BookEvent.PROCESSING_RESET_CONFIRMED,
            BookStatus.GLOSSARY_REVIEW,
            id="reset-from-ready-to-export",
        ),
        pytest.param(
            BookStatus.COMPLETED,
            BookEvent.PROCESSING_RESET_CONFIRMED,
            BookStatus.GLOSSARY_REVIEW,
            id="reset-from-completed",
        ),
        pytest.param(
            BookStatus.IMPORTING,
            BookEvent.IMPORT_PERMANENTLY_FAILED,
            BookStatus.FAILED,
            id="permanent-import-error",
        ),
    ],
)
def test_allowed_transition(
    source: BookStatus | None, event: BookEvent, target: BookStatus
) -> None:
    """Each row of table ALG-06 is executable and permitted."""
    assert apply_book_event(source, event) is target
    assert is_transition_allowed(source, target)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        pytest.param(BookStatus.IMPORTED, BookStatus.PROCESSING, id="skip-analysis"),
        pytest.param(BookStatus.COMPLETED, BookStatus.IMPORTING, id="back-to-import"),
        pytest.param(BookStatus.PROCESSING, BookStatus.FAILED, id="processing-to-failed"),
        pytest.param(BookStatus.FAILED, BookStatus.IMPORTED, id="out-of-failed"),
        pytest.param(BookStatus.ANALYZING, BookStatus.READY_TO_PROCESS, id="skip-glossary-review"),
    ],
)
def test_transition_outside_table_is_rejected(
    source: BookStatus, target: BookStatus
) -> None:
    """A transition outside the table is forbidden and rejected with an error."""
    assert not is_transition_allowed(source, target)
    with pytest.raises(BookStatusTransitionError):
        ensure_transition_allowed(source, target)


@pytest.mark.parametrize(
    ("source", "event"),
    [
        pytest.param(BookStatus.IMPORTED, BookEvent.EXPORT_SUCCEEDED, id="export-before-import"),
        pytest.param(
            BookStatus.PROCESSING,
            BookEvent.IMPORT_PERMANENTLY_FAILED,
            id="import-error-while-processing",
        ),
        pytest.param(
            BookStatus.GLOSSARY_REVIEW,
            BookEvent.BLOCK_REGENERATION_REQUESTED,
            id="regenerate-before-processing",
        ),
    ],
)
def test_event_in_wrong_status_is_rejected(source: BookStatus, event: BookEvent) -> None:
    """An event invalid in the current state is rejected with an error."""
    with pytest.raises(BookStatusTransitionError):
        apply_book_event(source, event)


def test_failed_is_reachable_only_from_importing() -> None:
    """'FAILED' is reserved for a permanent book-level error."""
    sources = [status for status in BookStatus if is_transition_allowed(status, BookStatus.FAILED)]
    assert sources == [BookStatus.IMPORTING]


def test_allowed_targets_lists_reachable_statuses() -> None:
    """From the processing state, a book transitions either to export or to glossary review."""
    assert allowed_targets(BookStatus.PROCESSING) == frozenset(
        {BookStatus.READY_TO_EXPORT, BookStatus.GLOSSARY_REVIEW}
    )


@pytest.mark.parametrize(
    "error_type",
    [
        pytest.param(ErrorType.TRANSIENT, id="transient"),
        pytest.param(ErrorType.MODEL_OUTPUT, id="model-output"),
        pytest.param(ErrorType.VALIDATION, id="validation"),
    ],
)
def test_retryable_task_failure_keeps_book_status(error_type: ErrorType) -> None:
    """Failure of a retryable task does not change the book status."""
    assert (
        status_after_task_failure(
            BookStatus.PROCESSING,
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            error_type=error_type,
        )
        is BookStatus.PROCESSING
    )


def test_permanent_import_error_fails_the_book() -> None:
    """A permanent import error transitions the book to 'FAILED'."""
    assert (
        status_after_task_failure(
            BookStatus.IMPORTING,
            task_type=TaskType.IMPORT_BOOK,
            error_type=ErrorType.PERMANENT,
        )
        is BookStatus.FAILED
    )


def test_permanent_error_of_other_task_keeps_book_status() -> None:
    """A permanent error in a non-import task does not fail the entire book."""
    assert (
        status_after_task_failure(
            BookStatus.PROCESSING,
            task_type=TaskType.EXPORT_BOOK,
            error_type=ErrorType.PERMANENT,
        )
        is BookStatus.PROCESSING
    )
