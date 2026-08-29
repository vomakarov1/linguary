"""Фабрики книги и ключей задач для тестов очереди.

Собирают минимальный валидный вход: доменные инварианты проверены отдельно, и тест очереди
не должен спотыкаться на их нарушении. Все значения по умолчанию перекрываются именованными
аргументами.
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
    """Запись книги, к которой привязываются задачи."""
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
    """Идемпотентный ключ задачи."""
    return TaskIdentity(book_id=book_id, task_type=task_type, scope_id=scope_id)


def sequential_task_ids(prefix: str = "task") -> Callable[[], str]:
    """Генератор предсказуемых идентификаторов задач."""
    numbers = count(1)
    return lambda: f"{prefix}-{next(numbers)}"
