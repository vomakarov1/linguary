"""Book and task key factories for queue tests.

Constructs minimal valid input: domain invariants are validated separately,
and queue tests should not stumble on invariant violations. All default values
can be overridden via keyword arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count

from linguary.application.ports.persistence import BookRecord
from linguary.domain import BookMetadata, BookStatus, TaskIdentity, TaskType

BOOK_ID = "book-1"
SECTION_ID = "section-1"
CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def make_book(
    *,
    book_id: str = BOOK_ID,
    status: BookStatus = BookStatus.IMPORTING,
    source_epub_path: str = "/library/sample.epub",
) -> BookRecord:
    """Book record to which tasks are attached."""
    return BookRecord(
        id=book_id,
        metadata=BookMetadata(title="The Sample Book", authors=("Jane Doe",)),
        source_language="en",
        target_language="ru",
        status=status,
        source_epub_path=source_epub_path,
        created_at=CREATED_AT,
    )


def make_identity(
    *,
    book_id: str = BOOK_ID,
    task_type: TaskType = TaskType.ANALYZE_CHAPTER,
    scope_id: str = SECTION_ID,
) -> TaskIdentity:
    """Task idempotency key."""
    return TaskIdentity(book_id=book_id, task_type=task_type, scope_id=scope_id)


def sequential_task_ids(prefix: str = "task") -> Callable[[], str]:
    """Predictable task identifier generator."""
    numbers = count(1)
    return lambda: f"{prefix}-{next(numbers)}"
