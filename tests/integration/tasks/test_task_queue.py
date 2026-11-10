"""Local task queue on real SQLite: enqueueing and claiming (ALG-02, P2.T7)."""

from __future__ import annotations

from collections.abc import Callable
import asyncio
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import Engine, event
from task_factories import BOOK_ID, CREATED_AT, make_identity

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import SqliteTaskQueue
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.domain.book_conditions import BookConditions

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


async def test_enqueue_preserves_operation_and_policy_identity(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    identity = replace(make_identity(task_type=TaskType.REGENERATE_BLOCK),
        operation_id="operation-1", policy_version="translation-context-v2")
    other_operation = replace(identity, operation_id="operation-2")
    other_policy = replace(identity, policy_version="translation-context-legacy-v1")
    tasks = [await task_queue.enqueue(key) for key in (identity, other_operation, other_policy)]

    assert len({task.id for task in tasks}) == 3
    for key, task in zip((identity, other_operation, other_policy), tasks, strict=True):
        assert task.identity == key
        assert await task_queue.enqueue(key) == task
    with unit_of_work.transaction() as transaction:
        assert {task.identity for task in transaction.tasks.list_for_book(BOOK_ID)} == {
            identity, other_operation, other_policy,
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


async def test_claim_next_pending_skips_begin_immediate_when_queue_empty_or_paused(
    engine: Engine,
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Idle or paused queue polling performs a read check without acquiring `BEGIN IMMEDIATE`."""
    begin_immediates: list[str] = []

    def record_sql(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            begin_immediates.append(statement)

    wakeups: list[int] = []
    callback = lambda: wakeups.append(1)

    event.listen(engine, "before_cursor_execute", record_sql)
    try:
        begin_immediates.clear()
        assert await task_queue.claim_next_pending() is None
        assert begin_immediates == []

        conditions_id = str(uuid4())
        with unit_of_work.transaction() as uow:
            uow.work.add_conditions(
                BookConditions(
                    id=conditions_id,
                    book_id=BOOK_ID,
                    revision=1,
                    payload={"schema_version": 1},
                    created_at=CREATED_AT,
                )
            )
            uow.work.set_book_paused(BOOK_ID, conditions_id, True)

        task_queue.subscribe_wakeup(callback)
        enqueued = await task_queue.enqueue(make_identity())
        assert len(wakeups) == 1

        begin_immediates.clear()
        assert await task_queue.claim_next_pending() is None
        assert begin_immediates == []

        with unit_of_work.transaction() as uow:
            uow.work.set_book_paused(BOOK_ID, conditions_id, False)

        begin_immediates.clear()
        claimed = await task_queue.claim_next_pending()
        assert claimed is not None
        assert claimed.id == enqueued.id
        assert claimed.status is TaskStatus.RUNNING
        assert len(begin_immediates) == 1

        begin_immediates.clear()
        assert await task_queue.claim_next_pending() is None
        assert begin_immediates == []

        second = await task_queue.enqueue(make_identity(scope_id="section-2"))
        concurrent_claims = await asyncio.gather(
            task_queue.claim_next_pending(),
            task_queue.claim_next_pending(),
        )
        assert [item.id for item in concurrent_claims if item is not None] == [second.id]

        task_queue.unsubscribe_wakeup(callback)
        task_queue.notify_work_available()
        assert len(wakeups) == 2
    finally:
        event.remove(engine, "before_cursor_execute", record_sql)
