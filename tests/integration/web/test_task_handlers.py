"""Composition root registers a handler for each task type (ALG-02).

Before Phase 4, no handlers were registered, and a task of any type except import
failed with a PermanentError "handler not registered". This test verifies the opposite:
each queue task type reaches its use case, so task failures are explained by domain reasons
rather than missing handlers in the composed application.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import FIXTURE_DIRECTORY
from web_scenario import store_book, bound_queue

from linguary.composition import Application, drain_tasks
from linguary.domain import (
    BookStatus,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
)

#: Fixture suite book whose path is configured solely in the book record.
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"

#: Scope that does not exist in the book: the use case must explain this on its own,
#: rather than reporting that no handler of this type was registered.
MISSING_SCOPE = "scope-that-does-not-exist"


def error_message(task: ProcessingTask) -> str:
    """Error message of a finished task."""
    assert task.last_error is not None
    return task.last_error.message


@pytest.mark.parametrize(
    "task_type",
    [
        pytest.param(TaskType.IMPORT_BOOK, id="import"),
        pytest.param(TaskType.ANALYZE_CHAPTER, id="analyze"),
        pytest.param(TaskType.PROCESS_TRANSLATION_UNIT, id="process"),
        pytest.param(TaskType.REGENERATE_BLOCK, id="regenerate"),
        pytest.param(TaskType.EXPORT_BOOK, id="export"),
    ],
)
async def test_every_task_type_reaches_its_use_case(
    application: Application, task_type: TaskType
) -> None:
    """Each task type reaches its use case rather than a missing handler error."""
    scenario = store_book(application)
    await application.task_queue.enqueue(
        TaskIdentity(
            book_id=scenario.book_id, task_type=task_type, scope_id=MISSING_SCOPE
        )
    )

    finished = await drain_tasks(application)

    assert [task.status for task in finished] == [TaskStatus.FAILED]
    assert "не зарегистрирован" not in error_message(finished[0])


async def test_analysis_task_reports_the_missing_chapter(
    configured_application: Application,
) -> None:
    """Analysis task reaches chapter use case, which provides the rejection reason."""
    application = configured_application
    scenario = store_book(application, analysed_chapters=())
    await bound_queue(application).enqueue(
        TaskIdentity(
            book_id=scenario.book_id,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id=MISSING_SCOPE,
        )
    )

    finished = await drain_tasks(application)

    assert MISSING_SCOPE in error_message(finished[0])


async def test_export_task_reaches_the_export_use_case(
    application: Application,
) -> None:
    """Export task reaches `ExportBook`: rejection originates from book state.

    The destination file is not passed to the export task — it is selected by the data
    directory layout, and there is no other place in the application aware of it.
    """
    scenario = store_book(application)

    task = await application.task_queue.enqueue(
        TaskIdentity(book_id=scenario.book_id, task_type=TaskType.EXPORT_BOOK)
    )
    finished = await drain_tasks(application)

    assert task.task_type is TaskType.EXPORT_BOOK
    assert task.scope_id == ""
    assert [item.status for item in finished] == [TaskStatus.FAILED]
    assert "экспортировать целиком нельзя" in error_message(finished[0])


async def test_import_task_uses_the_path_from_the_book_record(
    application: Application,
) -> None:
    """Import task builds command from book record: path is taken from there.

    Runner only knows the task's idempotency key, so the use case command is built by
    the composition root — otherwise there would be nothing to import.
    """
    scenario = store_book(application, status=BookStatus.IMPORTING)
    with application.unit_of_work.transaction() as uow:
        book = uow.books.get(scenario.book_id)
        assert book is not None
        uow.books.save(replace(book, source_epub_path=str(SYNTHETIC_BOOK)))
    await application.task_queue.enqueue(
        TaskIdentity(book_id=scenario.book_id, task_type=TaskType.IMPORT_BOOK)
    )

    finished = await drain_tasks(application)

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED]
    with application.unit_of_work.transaction() as uow:
        imported = uow.books.get(scenario.book_id)
    assert imported is not None and imported.status is BookStatus.IMPORTED
