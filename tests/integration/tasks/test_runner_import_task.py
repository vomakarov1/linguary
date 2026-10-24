"""Task runner executes the real book import task and its retry (P2.T7).

This verifies the exact junction for which the runner exists: SQLite queue, worker,
synchronous application use case `ImportBook`, and the EPUB adapter work together.
The import task is created for a book already uploaded by the user in `IMPORTING`
status (PRD § 18), so the book record is created prior to enqueueing the task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest
from task_factories import BOOK_ID, make_book

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import EbookLibEpubReader
from linguary.adapters.runtime import (
    SqliteTaskQueue,
    TaskWorkerPool,
    to_async_handler,
)
from linguary.application.task_runner import TaskRunner
from linguary.application.use_cases.import_book import ImportBook, ImportBookCommand
from linguary.application.use_cases.retry_task import (
    RetryTask,
    RetryTaskCommand,
    RetryTaskError,
)
from linguary.domain import (
    BookStatus,
    ErrorType,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
)

#: Set of test EPUB fixtures (PRD § 28.1).
FIXTURE_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures" / "epub"

#: Import task identity: book-level task, so scope is empty.
IMPORT_IDENTITY = TaskIdentity(book_id=BOOK_ID, task_type=TaskType.IMPORT_BOOK)


def import_handler(unit_of_work: SqlAlchemyUnitOfWork) -> Callable[[ProcessingTask], None]:
    """Import task handler: constructs the use case command from the book record."""
    import_book = ImportBook(unit_of_work=unit_of_work, reader=EbookLibEpubReader())

    def handle(task: ProcessingTask) -> None:
        with unit_of_work.transaction() as uow:
            book = uow.books.get(task.book_id)
        assert book is not None
        import_book(
            ImportBookCommand(
                book_id=book.id,
                source_epub_path=book.source_epub_path,
                target_language=book.target_language,
                source_language=book.source_language,
            )
        )

    return handle


def flaky_import_handler(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> Callable[[ProcessingTask], None]:
    """Flaky import handler whose first attempt fails with a transient error.

    A task survives a transient failure and can be retried upon user action, unlike
    an unreadable EPUB which requires a different file rather than a retry (PRD § 16).
    """
    handle_import = import_handler(unit_of_work)
    attempts = count(1)

    def handle(task: ProcessingTask) -> None:
        if next(attempts) == 1:
            raise RuntimeError("чтение файла книги прервано")
        handle_import(task)

    return handle


def make_pool(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    handler: Callable[[ProcessingTask], None] | None = None,
) -> TaskWorkerPool:
    """Worker pool with a single handler for the book import task."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue,
            handlers={
                TaskType.IMPORT_BOOK: to_async_handler(
                    handler or import_handler(unit_of_work)
                )
            },
        )
    )


def store_book(unit_of_work: SqlAlchemyUnitOfWork, *, source_epub_path: str) -> None:
    """Stores an uploaded book in `IMPORTING` status in the persistence layer."""
    with unit_of_work.transaction() as uow:
        existing = uow.books.get(BOOK_ID)
        book = make_book(source_epub_path=source_epub_path)
        if existing is None:
            uow.books.add(book)
        else:
            uow.books.save(replace(book, created_at=existing.created_at))


async def test_import_task_imports_the_book(
    task_queue: SqliteTaskQueue, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    """Import task reaches `succeeded`, and the book becomes imported."""
    store_book(unit_of_work, source_epub_path=str(FIXTURE_DIRECTORY / "synthetic_story.epub"))
    await task_queue.enqueue(IMPORT_IDENTITY)

    finished = await make_pool(task_queue, unit_of_work).drain()

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED]
    with unit_of_work.transaction() as uow:
        book = uow.books.get(BOOK_ID)
        document = uow.documents.get(BOOK_ID)
    assert book is not None and book.status is BookStatus.IMPORTED
    assert document is not None
    assert len(document.sections) == 3


async def test_retried_import_task_does_not_duplicate_the_result(
    task_queue: SqliteTaskQueue, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    """Retry of a failed task imports the book once, without duplicate sections."""
    store_book(unit_of_work, source_epub_path=str(FIXTURE_DIRECTORY / "synthetic_story.epub"))
    task = await task_queue.enqueue(IMPORT_IDENTITY)
    pool = make_pool(task_queue, unit_of_work, flaky_import_handler(unit_of_work))
    failed = await pool.drain()
    assert [item.status for item in failed] == [TaskStatus.FAILED]

    RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))
    finished = await pool.drain()

    assert [item.status for item in finished] == [TaskStatus.SUCCEEDED]
    assert finished[0].attempts == 2
    with unit_of_work.transaction() as uow:
        document = uow.documents.get(BOOK_ID)
        tasks = uow.tasks.list_for_book(BOOK_ID)
    assert document is not None
    assert len(document.sections) == 3
    assert [item.id for item in tasks] == [task.id]


async def test_unreadable_epub_makes_the_task_not_retryable(
    task_queue: SqliteTaskQueue, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    """Unreadable EPUB produces a permanent error: requires another file, not a retry (ALG-05)."""
    store_book(unit_of_work, source_epub_path=str(FIXTURE_DIRECTORY / "missing.epub"))
    task = await task_queue.enqueue(IMPORT_IDENTITY)

    failed = await make_pool(task_queue, unit_of_work).drain()

    assert [item.status for item in failed] == [TaskStatus.FAILED]
    assert failed[0].last_error is not None
    assert failed[0].last_error.type is ErrorType.PERMANENT
    assert not failed[0].is_retryable
    with pytest.raises(RetryTaskError, match="нужен другой исходный файл"):
        RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))
