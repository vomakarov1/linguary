"""Очередь локальных задач на настоящем SQLite: постановка и захват (ALG-02, P2.T7)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from task_factories import BOOK_ID, make_identity

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import SqliteTaskQueue
from linguary.domain import ProcessingTask, TaskStatus, TaskType

#: Завершение захваченной задачи: успешное либо неуспешное.
FinishTask = Callable[[ProcessingTask], ProcessingTask]


async def test_enqueue_returns_existing_task_for_the_same_identity(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Повторная постановка ключа `book_id + task_type + scope_id` дубликата не создаёт."""
    first = await task_queue.enqueue(make_identity())

    second = await task_queue.enqueue(make_identity())

    assert second == first
    with unit_of_work.transaction() as uow:
        assert [task.id for task in uow.tasks.list_for_book(BOOK_ID)] == [first.id]


async def test_enqueue_distinguishes_tasks_by_type_and_scope(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Задачи разных типов и разных глав живут в очереди рядом."""
    await task_queue.enqueue(make_identity(scope_id="section-1"))
    await task_queue.enqueue(make_identity(scope_id="section-2"))
    await task_queue.enqueue(make_identity(task_type=TaskType.EXPORT_BOOK, scope_id=""))

    with unit_of_work.transaction() as uow:
        stored = uow.tasks.list_for_book(BOOK_ID)

    assert {(task.task_type, task.scope_id) for task in stored} == {
        (TaskType.ANALYZE_CHAPTER, "section-1"),
        (TaskType.ANALYZE_CHAPTER, "section-2"),
        (TaskType.EXPORT_BOOK, ""),
    }


async def test_claim_takes_the_earliest_pending_task(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Захват идёт в порядке постановки задач в очередь."""
    first = await task_queue.enqueue(make_identity(scope_id="section-1"))
    second = await task_queue.enqueue(make_identity(scope_id="section-2"))

    claimed = [await task_queue.claim_next_pending() for _ in range(3)]

    assert [task.id if task else None for task in claimed] == [first.id, second.id, None]


async def test_claim_marks_task_running_and_counts_attempt(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Захваченная задача становится `running`, а её попытка учтена."""
    task = await task_queue.enqueue(make_identity())

    claimed = await task_queue.claim_next_pending()

    assert claimed is not None
    assert (claimed.status, claimed.attempts) == (TaskStatus.RUNNING, 1)
    with unit_of_work.transaction() as uow:
        stored = uow.tasks.get(task.id)
    assert stored is not None
    assert (stored.status, stored.attempts) == (TaskStatus.RUNNING, 1)


async def test_claim_returns_none_on_empty_queue(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Пустая очередь не отдаёт задачу."""
    assert await task_queue.claim_next_pending() is None


@pytest.mark.parametrize(
    "finish",
    [
        pytest.param(lambda task: task.succeed(), id="succeeded"),
        pytest.param(lambda task: task.interrupt(), id="failed"),
    ],
)
async def test_finished_task_is_not_claimed_again(
    task_queue: SqliteTaskQueue, stored_book: None, finish: FinishTask
) -> None:
    """Завершённая задача в очередь не возвращается (PRD § 18).

    Успешная задача повторно не выполняется вовсе, неуспешную возвращает только явный
    повтор пользователем.
    """
    await task_queue.enqueue(make_identity())
    claimed = await task_queue.claim_next_pending()
    assert claimed is not None
    await task_queue.save(finish(claimed))

    assert await task_queue.claim_next_pending() is None


async def test_running_tasks_are_listed_for_recovery(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Незавершённые задачи видны отдельно: с них начинается восстановление после сбоя."""
    await task_queue.enqueue(make_identity(scope_id="section-1"))
    await task_queue.enqueue(make_identity(scope_id="section-2"))
    claimed = await task_queue.claim_next_pending()

    running = await task_queue.list_running()

    assert claimed is not None
    assert [task.id for task in running] == [claimed.id]
