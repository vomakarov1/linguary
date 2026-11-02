"""Idempotency keys include translation operation and policy without changing other tasks."""

from __future__ import annotations

from dataclasses import replace

import pytest
from persistence_factories import BOOK_ID, SECTION_ID, make_book, make_task
from sqlalchemy.exc import IntegrityError

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import TaskIdentity, TaskStatus, TaskType


@pytest.fixture
def stored_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Book to which test tasks are attached."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())


def test_duplicate_identity_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """A second task with the same key is not stored."""
    with unit_of_work.transaction() as uow:
        uow.tasks.add(make_task(task_id="task-1"))

    with pytest.raises(IntegrityError), unit_of_work.transaction() as uow:
        uow.tasks.add(make_task(task_id="task-2"))


def test_book_level_tasks_collide_on_empty_scope(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Two export tasks for the same book collide on the key rather than NULL.

    Book-level tasks have an empty string as `scope_id`: in SQLite `NULL != NULL`,
    and a unique index on a column with `NULL` would not distinguish such tasks.
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
    """Tasks with different keys coexist."""
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
    """Task is found by its idempotency key."""
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
    """State transition is stored along with attempt counter and error."""
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


def test_same_scope_different_operations_coexist(unit_of_work, stored_book):
    first = replace(make_task(task_type=TaskType.REGENERATE_BLOCK, scope_id="block"),
                    operation_id="first", policy_version="translation-context-v2")
    second = replace(first, id="second-task", operation_id="second")
    another_policy = replace(first, id="legacy-task", policy_version="translation-context-legacy-v1")
    with unit_of_work.transaction() as uow:
        for task in (first, second, another_policy):
            uow.tasks.add(task)
    with unit_of_work.transaction() as uow:
        assert uow.tasks.list_for_book(BOOK_ID) == (first, second, another_policy)
        assert uow.tasks.get(first.id) == first
        uow.tasks.save(first.start())
    with unit_of_work.transaction() as uow:
        assert uow.tasks.get(first.id) == first.start()
    with pytest.raises(IntegrityError), unit_of_work.transaction() as uow:
        uow.tasks.add(replace(first, id="duplicate"))


@pytest.mark.parametrize("identity_change", [
    {"book_id": "another-book"},
    {"task_type": TaskType.EXPORT_BOOK},
    {"scope_id": "another-block"},
    {"operation_id": "absent"},
    {"policy_version": "absent"},
])
@pytest.mark.parametrize("stored_status", [TaskStatus.PENDING, TaskStatus.SUPERSEDED])
def test_find_task_never_returns_another_operation(
    unit_of_work, stored_book, identity_change, stored_status
):
    task = replace(make_task(task_type=TaskType.REGENERATE_BLOCK, scope_id="block"),
                   operation_id="first", policy_version="translation-context-v2",
                   status=stored_status)
    with unit_of_work.transaction() as uow:
        uow.tasks.add(task)
        uow.tasks.add(replace(task, id="second-task", operation_id="second"))
        uow.tasks.add(replace(task, id="legacy-task", policy_version="translation-context-legacy-v1"))
    with unit_of_work.transaction() as uow:
        assert uow.tasks.find_by_identity(task.identity) == task
        for identity in (
            replace(task.identity, operation_id="absent"),
            replace(task.identity, policy_version="absent"),
            replace(task.identity, operation_id="", policy_version=""),
        ):
            assert uow.tasks.find_by_identity(identity) is None
    changed = replace(task, **identity_change, attempts=1)
    with pytest.raises(PersistenceError, match="идентичност"), unit_of_work.transaction() as uow:
        uow.tasks.save(changed)
    with unit_of_work.transaction() as uow:
        assert uow.tasks.get(task.id) == task
        assert uow.tasks.find_by_identity(task.identity) == task
        assert uow.tasks.find_by_identity(changed.identity) is None


@pytest.mark.parametrize("late_status", [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.SUCCEEDED, TaskStatus.FAILED])
def test_late_task_save_cannot_undo_supersession(unit_of_work, stored_book, late_status):
    task = replace(make_task(task_type=TaskType.REGENERATE_BLOCK), operation_id="root",
                   policy_version="translation-context-v2")
    with unit_of_work.transaction() as uow:
        uow.tasks.add(task)
    with unit_of_work.transaction() as uow:
        stale = uow.tasks.get(task.id)
    superseded = replace(task, status=TaskStatus.SUPERSEDED, attempts=2)
    with unit_of_work.transaction() as uow:
        uow.tasks.save(superseded)
    late = {
        TaskStatus.PENDING: stale,
        TaskStatus.RUNNING: stale.start(),
        TaskStatus.SUCCEEDED: stale.start().succeed(),
        TaskStatus.FAILED: stale.start().interrupt(),
    }[late_status]
    with pytest.raises(PersistenceError, match="вытеснена"), unit_of_work.transaction() as uow:
        uow.tasks.save(late)
    with unit_of_work.transaction() as uow:
        # Repeated terminal writes are harmless, including their stale counters.
        uow.tasks.save(replace(superseded, attempts=0))
        assert uow.tasks.get(task.id) == superseded


@pytest.mark.parametrize("task_type", [
    TaskType.IMPORT_BOOK, TaskType.ANALYZE_CHAPTER, TaskType.EXPORT_BOOK, TaskType.QUALITY_REVIEW,
])
def test_nontranslation_task_idempotency_is_preserved(unit_of_work, stored_book, task_type):
    task = make_task(task_type=task_type)
    with unit_of_work.transaction() as uow:
        uow.tasks.add(task)
    with pytest.raises(IntegrityError), unit_of_work.transaction() as uow:
        uow.tasks.add(replace(task, id="duplicate"))
    with unit_of_work.transaction() as uow:
        assert uow.tasks.find_by_identity(task.identity) == task
        assert uow.tasks.get(task.id).operation_id == ""
        assert uow.tasks.get(task.id).policy_version == ""
