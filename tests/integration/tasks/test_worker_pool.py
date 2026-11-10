"""Limiting worker parallelism and shutdown handling (PRD § 25, P2.T7)."""

from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import pytest
from task_factories import BOOK_ID, CREATED_AT, make_identity

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import (
    SqliteTaskQueue,
    SqliteTaskStore,
    TaskWorkerPool,
    WorkerSettings,
)
from linguary.application.task_runner import TaskHandler, TaskRunner
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.domain.book_conditions import BookConditions

#: Timeout limit in tests: safeguard against hanging, not part of the assertion.
TIMEOUT_SECONDS = 5.0

#: Waiting for a rendezvous that by test design must not occur.
IMPOSSIBLE_RENDEZVOUS_SECONDS = 0.2


class InFlightCounter:
    """In-flight counter for simultaneously executing handlers."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        """Marks the start of handler execution."""
        self.current += 1
        self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        """Marks the completion of handler execution."""
        self.current -= 1


def rendezvous_handler(
    barrier: asyncio.Barrier,
    counter: InFlightCounter | None = None,
    *,
    timeout: float = TIMEOUT_SECONDS,
) -> TaskHandler:
    """Handler that completes only together with all rendezvous participants.

    The rendezvous is a controlled synchronization point: the handler does not release
    the worker until the expected number of tasks are concurrently running.
    Concurrency is ensured by design rather than coroutine scheduling luck; if absent,
    the wait times out and the task fails.
    """

    async def handle(task: ProcessingTask) -> None:
        if counter is not None:
            counter.enter()
        try:
            await asyncio.wait_for(barrier.wait(), timeout=timeout)
        finally:
            if counter is not None:
                counter.leave()

    return handle


async def enqueue_tasks(queue: SqliteTaskQueue, count: int) -> None:
    """Enqueues `count` chapter analysis tasks."""
    for number in range(count):
        await queue.enqueue(make_identity(scope_id=f"section-{number}"))


def make_pool(
    task_queue: SqliteTaskQueue, handler: TaskHandler, settings: WorkerSettings | None = None
) -> TaskWorkerPool:
    """Worker pool over a queue with a single handler."""
    return TaskWorkerPool(
        TaskRunner(queue=task_queue, handlers={TaskType.ANALYZE_CHAPTER: handler}),
        settings,
    )


@pytest.mark.parametrize("concurrency", [1, 2, 3])
async def test_simultaneous_handlers_are_limited_by_the_setting(
    task_queue: SqliteTaskQueue, stored_book: None, concurrency: int
) -> None:
    """Simultaneously executed tasks are strictly bounded by concurrency setting."""
    counter = InFlightCounter()
    await enqueue_tasks(task_queue, concurrency * 2)
    pool = make_pool(
        task_queue,
        rendezvous_handler(asyncio.Barrier(concurrency), counter),
        WorkerSettings(concurrency=concurrency),
    )

    finished = await pool.drain()

    assert counter.peak == concurrency
    assert len(finished) == concurrency * 2
    assert {task.status for task in finished} == {TaskStatus.SUCCEEDED}


async def test_default_setting_runs_tasks_one_after_another(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """By default the second task waits for the first and does not run concurrently.

    Both tasks wait for a rendezvous of 2 participants. With default concurrency this
    is impossible, so the wait times out and both tasks fail: demonstrating that only
    one task was running at a time (PRD § 25).
    """
    await enqueue_tasks(task_queue, 2)
    pool = make_pool(
        task_queue,
        rendezvous_handler(
            asyncio.Barrier(2), timeout=IMPOSSIBLE_RENDEZVOUS_SECONDS
        ),
    )

    finished = await pool.drain()

    assert len(finished) == 2
    assert {task.status for task in finished} == {TaskStatus.FAILED}


async def test_pool_runs_until_stop_is_requested(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Continuously running worker waits for new tasks and shuts down on request."""
    executed = asyncio.Event()

    async def handle(task: ProcessingTask) -> None:
        executed.set()

    pool = make_pool(task_queue, handle, WorkerSettings(idle_delay_seconds=0.01))
    running = asyncio.create_task(pool.run_forever())

    await task_queue.enqueue(make_identity())
    await asyncio.wait_for(executed.wait(), timeout=TIMEOUT_SECONDS)
    pool.request_stop()
    await asyncio.wait_for(running, timeout=TIMEOUT_SECONDS)

    assert await task_queue.claim_next_pending() is None


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"concurrency": 0}, id="zero-concurrency"),
        pytest.param({"idle_delay_seconds": -1.0}, id="negative-delay"),
    ],
)
def test_worker_settings_reject_meaningless_values(settings: dict[str, float]) -> None:
    """Worker settings reject invalid values under which it cannot run."""
    with pytest.raises(ValueError):
        WorkerSettings(**settings)


class ControlledRunner:
    """Each claimed task remains busy until released by the test."""

    def __init__(self):
        self.started = asyncio.Queue()
        self.finished = asyncio.Queue()
        self.releases = {}
        self.claims = 0
        self.active = 0
        self.peak = 0

    async def run_next(self):
        number = self.claims
        self.claims += 1
        release = self.releases[number] = asyncio.Event()
        self.active += 1
        self.peak = max(self.active, self.peak)
        self.started.put_nowait(number)
        try:
            await release.wait()
            self.finished.put_nowait(number)
            return number
        finally:
            self.active -= 1


async def next_started(runner):
    return await asyncio.wait_for(runner.started.get(), TIMEOUT_SECONDS)


async def finish_claim(runner, number):
    runner.releases[number].set()
    assert await asyncio.wait_for(runner.finished.get(), TIMEOUT_SECONDS) == number


async def stop_controlled_pool(pool, running, runner):
    pool.request_stop()
    for release in runner.releases.values():
        release.set()
    await asyncio.wait_for(running, TIMEOUT_SECONDS)
    assert runner.active == 0


async def test_resize_up_starts_pending_tasks():
    runner = ControlledRunner()
    pool = TaskWorkerPool(runner)
    running = asyncio.create_task(pool.run_forever())
    try:
        assert await next_started(runner) == 0
        pool.set_concurrency(3)
        assert await next_started(runner) == 1
        assert await next_started(runner) == 2
        assert runner.active == 3 and pool.concurrency == 3
    finally:
        await stop_controlled_pool(pool, running, runner)


async def test_resize_down_does_not_cancel_inflight_tasks():
    runner = ControlledRunner()
    pool = TaskWorkerPool(runner, WorkerSettings(concurrency=3))
    running = asyncio.create_task(pool.run_forever())
    try:
        for number in range(3):
            assert await next_started(runner) == number
        pool.set_concurrency(1)
        assert runner.active == 3
        await finish_claim(runner, 1)
        await finish_claim(runner, 2)
        assert runner.claims == 3 and runner.active == 1
        await finish_claim(runner, 0)
        assert await next_started(runner) == 3
        assert runner.active == 1 and pool.concurrency == 1
    finally:
        await stop_controlled_pool(pool, running, runner)


async def test_resize_up_during_retirement_has_no_duplicate_workers():
    runner = ControlledRunner()
    pool = TaskWorkerPool(runner, WorkerSettings(concurrency=3))
    running = asyncio.create_task(pool.run_forever())
    try:
        for number in range(3):
            assert await next_started(runner) == number
        pool.set_concurrency(1)
        pool.set_concurrency(3)
        for number in range(3):
            await finish_claim(runner, number)
            assert await next_started(runner) == number + 3
        assert runner.active == 3 and runner.peak == 3 and runner.claims == 6
    finally:
        await stop_controlled_pool(pool, running, runner)


async def test_resize_after_stop_does_not_restart_workers():
    runner = ControlledRunner()
    pool = TaskWorkerPool(runner)
    running = asyncio.create_task(pool.run_forever())
    assert await next_started(runner) == 0
    pool.request_stop()
    pool.set_concurrency(5)
    await finish_claim(runner, 0)
    await asyncio.wait_for(running, TIMEOUT_SECONDS)
    assert pool.concurrency == 5 and runner.claims == 1
    pool.set_concurrency(7)
    assert runner.claims == 1


async def test_resize_while_idle_wakes_pool():
    class IdleRunner:
        def __init__(self):
            self.calls = asyncio.Queue()
        async def run_next(self):
            self.calls.put_nowait(None)
            return None
    runner = IdleRunner()
    pool = TaskWorkerPool(runner, WorkerSettings(idle_delay_seconds=3600))
    running = asyncio.create_task(pool.run_forever())
    try:
        await asyncio.wait_for(runner.calls.get(), TIMEOUT_SECONDS)
        pool.set_concurrency(2)
        await asyncio.wait_for(runner.calls.get(), TIMEOUT_SECONDS)
        await asyncio.wait_for(runner.calls.get(), TIMEOUT_SECONDS)
    finally:
        pool.request_stop()
        await asyncio.wait_for(running, TIMEOUT_SECONDS)


async def test_cancelling_pool_collects_all_worker_tasks():
    runner = ControlledRunner()
    pool = TaskWorkerPool(runner, WorkerSettings(concurrency=3))
    running = asyncio.create_task(pool.run_forever())
    for number in range(3):
        assert await next_started(runner) == number
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert runner.active == 0


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '2'])
def test_set_concurrency_rejects_invalid_values(value):
    pool = TaskWorkerPool(ControlledRunner())
    with pytest.raises(ValueError):
        pool.set_concurrency(value)
    assert pool.concurrency == 1


async def test_resize_down_prevents_extra_claims_while_old_slots_are_busy():
    runner = ControlledRunner()
    pool = TaskWorkerPool(runner, WorkerSettings(concurrency=3))
    running = asyncio.create_task(pool.run_forever())
    try:
        for number in range(3):
            assert await next_started(runner) == number
        pool.set_concurrency(1)
        await finish_claim(runner, 0)
        assert runner.claims == 3
        await finish_claim(runner, 1)
        assert runner.claims == 3
        await finish_claim(runner, 2)
        assert await next_started(runner) == 3
        assert runner.active == 1
    finally:
        await stop_controlled_pool(pool, running, runner)


async def test_worker_pool_wakes_thread_safely_on_enqueue_with_long_idle_delay(
    task_store: SqliteTaskStore,
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers parked on a long idle delay wake immediately on cross-thread `enqueue` and `notify()`."""
    handled: asyncio.Queue[str] = asyncio.Queue()
    empty_polls: asyncio.Queue[None] = asyncio.Queue()
    inject_during_claim = False
    original_claim = task_queue.claim_next_pending

    async def tracked_claim() -> ProcessingTask | None:
        nonlocal inject_during_claim
        claimed = await original_claim()
        if claimed is None:
            if inject_during_claim:
                inject_during_claim = False
                race_thread = threading.Thread(
                    target=lambda: task_store.enqueue(make_identity(scope_id="section-3"))
                )
                race_thread.start()
                race_thread.join(timeout=TIMEOUT_SECONDS)
                await asyncio.sleep(0)
            empty_polls.put_nowait(None)
        return claimed

    monkeypatch.setattr(task_queue, "claim_next_pending", tracked_claim)

    async def handle(task: ProcessingTask) -> None:
        handled.put_nowait(task.scope_id)

    pool = make_pool(task_queue, handle, WorkerSettings(idle_delay_seconds=3600.0))
    running = asyncio.create_task(pool.run_forever())
    try:
        await asyncio.wait_for(empty_polls.get(), timeout=TIMEOUT_SECONDS)
        await asyncio.sleep(0)

        enqueue_thread = threading.Thread(
            target=lambda: task_store.enqueue(make_identity(scope_id="section-1"))
        )
        enqueue_thread.start()
        enqueue_thread.join(timeout=TIMEOUT_SECONDS)

        assert await asyncio.wait_for(handled.get(), timeout=0.5) == "section-1"
        await asyncio.wait_for(empty_polls.get(), timeout=TIMEOUT_SECONDS)
        await asyncio.sleep(0)

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

        await task_queue.enqueue(make_identity(scope_id="section-2"))
        await asyncio.wait_for(empty_polls.get(), timeout=TIMEOUT_SECONDS)
        await asyncio.sleep(0)

        with unit_of_work.transaction() as uow:
            uow.work.set_book_paused(BOOK_ID, conditions_id, False)

        notify_thread = threading.Thread(target=pool.notify)
        notify_thread.start()
        notify_thread.join(timeout=TIMEOUT_SECONDS)

        assert await asyncio.wait_for(handled.get(), timeout=0.5) == "section-2"
        await asyncio.wait_for(empty_polls.get(), timeout=TIMEOUT_SECONDS)
        await asyncio.sleep(0)

        inject_during_claim = True
        pool.notify()
        assert await asyncio.wait_for(handled.get(), timeout=0.5) == "section-3"
    finally:
        pool.request_stop()
        await asyncio.wait_for(running, timeout=TIMEOUT_SECONDS)

    pool.notify()
    task_queue.notify_work_available()
    assert WorkerSettings().idle_delay_seconds == 0.5
