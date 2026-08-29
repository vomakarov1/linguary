"""Runner исполняет настоящую задачу импорта книги и её повтор (P2.T7).

Здесь проверяется стык, ради которого runner существует: очередь на SQLite, воркер,
синхронный прикладной сценарий `ImportBook` и адаптер EPUB работают вместе. Задача
импорта появляется у книги, уже загруженной пользователем и находящейся в состоянии
`IMPORTING` (PRD § 18), поэтому запись книги создаётся до постановки задачи.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

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
from linguary.application.use_cases.retry_task import RetryTask, RetryTaskCommand
from linguary.domain import (
    BookStatus,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
)

#: Набор тестовых EPUB (PRD § 28.1).
FIXTURE_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures" / "epub"

#: Ключ задачи импорта книги: задача уровня книги, поэтому область пуста.
IMPORT_IDENTITY = TaskIdentity(book_id=BOOK_ID, task_type=TaskType.IMPORT_BOOK)


def import_handler(unit_of_work: SqlAlchemyUnitOfWork) -> Callable[[ProcessingTask], None]:
    """Обработчик задачи импорта: собирает команду сценария из записи книги."""
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


def make_pool(
    task_queue: SqliteTaskQueue, unit_of_work: SqlAlchemyUnitOfWork
) -> TaskWorkerPool:
    """Воркер с единственным обработчиком — задачей импорта книги."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue,
            handlers={TaskType.IMPORT_BOOK: to_async_handler(import_handler(unit_of_work))},
        )
    )


def store_book(unit_of_work: SqlAlchemyUnitOfWork, *, source_epub_path: str) -> None:
    """Кладёт в хранилище загруженную книгу в состоянии `IMPORTING`."""
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
    """Задача импорта доходит до `succeeded`, а книга оказывается импортированной."""
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
    """Повтор упавшей задачи импортирует книгу один раз, без второго набора глав."""
    store_book(unit_of_work, source_epub_path=str(FIXTURE_DIRECTORY / "missing.epub"))
    task = await task_queue.enqueue(IMPORT_IDENTITY)
    pool = make_pool(task_queue, unit_of_work)
    failed = await pool.drain()
    assert [item.status for item in failed] == [TaskStatus.FAILED]

    store_book(unit_of_work, source_epub_path=str(FIXTURE_DIRECTORY / "synthetic_story.epub"))
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
