"""Идемпотентный ключ задачи `book_id + task_type + scope_id`."""

from __future__ import annotations

import pytest
from persistence_factories import BOOK_ID, SECTION_ID, make_book, make_task
from sqlalchemy.exc import IntegrityError

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import TaskIdentity, TaskStatus, TaskType


@pytest.fixture
def stored_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Книга, к которой привязываются задачи теста."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())


def test_duplicate_identity_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Вторая задача с тем же ключом не сохраняется."""
    with unit_of_work.transaction() as uow:
        uow.tasks.add(make_task(task_id="task-1"))

    with pytest.raises(IntegrityError), unit_of_work.transaction() as uow:
        uow.tasks.add(make_task(task_id="task-2"))


def test_book_level_tasks_collide_on_empty_scope(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Две задачи экспорта одной книги различаются ключом, а не строкой `NULL`.

    `scope_id` задач уровня книги — пустая строка: в SQLite `NULL != NULL`, и на
    колонке с `NULL` уникальный индекс такие задачи бы не различил.
    """
    with unit_of_work.transaction() as uow:
        uow.tasks.add(make_task(task_id="export-1", task_type=TaskType.EXPORT_BOOK))

    with pytest.raises(IntegrityError), unit_of_work.transaction() as uow:
        uow.tasks.add(make_task(task_id="export-2", task_type=TaskType.EXPORT_BOOK))


@pytest.mark.parametrize(
    ("task_type", "scope_id"),
    [
        pytest.param(TaskType.IMPORT_BOOK, "", id="other-task-type"),
        pytest.param(TaskType.ANALYZE_CHAPTER, "section-2", id="other-scope"),
    ],
)
def test_different_identity_is_accepted(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
    task_type: TaskType,
    scope_id: str,
) -> None:
    """Задачи с разными ключами сосуществуют."""
    with unit_of_work.transaction() as uow:
        uow.tasks.add(
            make_task(
                task_id="analyze-1",
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id=SECTION_ID,
            )
        )
        uow.tasks.add(make_task(task_id="task-2", task_type=task_type, scope_id=scope_id))

    with unit_of_work.transaction() as uow:
        assert len(uow.tasks.list_for_book(BOOK_ID)) == 2


def test_task_is_found_by_identity(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Задача находится по идемпотентному ключу."""
    with unit_of_work.transaction() as uow:
        uow.tasks.add(
            make_task(
                task_id="analyze-1",
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id=SECTION_ID,
            )
        )

    identity = TaskIdentity(
        book_id=BOOK_ID, task_type=TaskType.ANALYZE_CHAPTER, scope_id=SECTION_ID
    )
    with unit_of_work.transaction() as uow:
        found = uow.tasks.find_by_identity(identity)
        missing = uow.tasks.find_by_identity(
            TaskIdentity(book_id=BOOK_ID, task_type=TaskType.EXPORT_BOOK)
        )

    assert found is not None
    assert found.id == "analyze-1"
    assert missing is None


def test_task_transition_is_stored(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Переход состояния сохраняется вместе со счётчиком попыток и ошибкой."""
    task = make_task()
    with unit_of_work.transaction() as uow:
        uow.tasks.add(task)

    failed = task.start().interrupt()
    with unit_of_work.transaction() as uow:
        uow.tasks.save(failed)

    with unit_of_work.transaction() as uow:
        stored = uow.tasks.get(task.id)

    assert stored is not None
    assert stored.status is TaskStatus.FAILED
    assert stored.attempts == 1
    assert stored.last_error == failed.last_error
    assert stored.is_retryable
