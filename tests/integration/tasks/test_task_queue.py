"""Local task queue on real SQLite: enqueueing and claiming (ALG-02, P2.T7)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from task_factories import BOOK_ID, make_identity

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import SqliteTaskQueue
from linguary.domain import ProcessingTask, TaskStatus, TaskType

#: Finishing a claimed task: either succeeded or failed.
FinishTask = Callable[[ProcessingTask], ProcessingTask]


async def test_enqueue_returns_existing_task_for_the_same_identity(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Repeated enqueue of the same key `book_id + task_type + scope_id` does not create a duplicate."""
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
    """Tasks of different types and different chapters coexist in the queue."""
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
    """Claiming proceeds in the order tasks were enqueued."""
    first = await task_queue.enqueue(make_identity(scope_id="section-1"))
    second = await task_queue.enqueue(make_identity(scope_id="section-2"))

    claimed = [await task_queue.claim_next_pending() for _ in range(3)]

    assert [task.id if task else None for task in claimed] == [first.id, second.id, None]


async def test_claim_marks_task_running_and_counts_attempt(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Claimed task becomes `running` and its attempt is counted."""
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
    """Empty queue yields no task."""
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
    """Finished task is not claimed again (PRD § 18).

    A succeeded task is never executed again; a failed task is only returned
    via an explicit user retry.
    """
    await task_queue.enqueue(make_identity())
    claimed = await task_queue.claim_next_pending()
    assert claimed is not None
    await task_queue.save(finish(claimed))

    assert await task_queue.claim_next_pending() is None


async def test_running_tasks_are_listed_for_recovery(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Running tasks are listed separately: crash recovery begins with them."""
    await task_queue.enqueue(make_identity(scope_id="section-1"))
    await task_queue.enqueue(make_identity(scope_id="section-2"))
    claimed = await task_queue.claim_next_pending()

    running = await task_queue.list_running()

    assert claimed is not None
    assert [task.id for task in running] == [claimed.id]
