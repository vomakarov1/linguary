"""Состав зависимостей регистрирует обработчик каждого типа задач (ALG-02).

До фазы 4 обработчиков не регистрировал никто, и задача любого типа, кроме импорта,
честно падала с `PermanentError` «обработчик не зарегистрирован». Тест закрепляет
обратное: каждый тип задач очереди доходит до своего сценария, поэтому неуспех задачи
объясняется предметной причиной, а не отсутствием обработчика в собранном приложении.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import FIXTURE_DIRECTORY
from web_scenario import store_book

from linguary.composition import Application, drain_tasks
from linguary.domain import (
    BookStatus,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
)

#: Книга набора фикстур, путь к которой задаётся только в записи книги.
SYNTHETIC_BOOK = FIXTURE_DIRECTORY / "synthetic_story.epub"

#: Область задачи, которой в книге нет: сценарий обязан объяснить это по-своему,
#: а не сообщить, что обработчика такого типа в приложении не собрано.
MISSING_SCOPE = "scope-that-does-not-exist"


def error_message(task: ProcessingTask) -> str:
    """Сообщение об ошибке завершённой задачи."""
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
    """Задача каждого типа доходит до сценария, а не до отсутствующего обработчика."""
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
    application: Application,
) -> None:
    """Задача анализа доходит до сценария главы: он и объясняет отказ."""
    scenario = store_book(application, analysed_chapters=())
    await application.task_queue.enqueue(
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
    """Задача экспорта доходит до `ExportBook`: отказ приходит от состояния книги.

    Целевой файл экспортной задаче никто не передаёт — его выбирает раскладка каталога
    данных, и другого места, где о ней известно, в приложении нет.
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
    """Задача импорта собирает команду из записи книги: путь берётся оттуда.

    Runner знает о задаче только идемпотентный ключ, поэтому команду сценария строит
    состав зависимостей — иначе импортировать было бы нечего.
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
