"""Ограничение параллелизма воркеров и их остановка (PRD § 25, P2.T7)."""

from __future__ import annotations

import asyncio

import pytest
from task_factories import make_identity

from linguary.adapters.runtime import SqliteTaskQueue, TaskWorkerPool, WorkerSettings
from linguary.application.task_runner import TaskHandler, TaskRunner
from linguary.domain import ProcessingTask, TaskStatus, TaskType

#: Предел ожидания в тестах: страховка от зависания, а не часть проверки.
TIMEOUT_SECONDS = 5.0

#: Ожидание встречи, которая по условию теста состояться не должна.
IMPOSSIBLE_RENDEZVOUS_SECONDS = 0.2


class InFlightCounter:
    """Счётчик одновременно выполняемых обработчиков."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        """Отмечает начало работы обработчика."""
        self.current += 1
        self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        """Отмечает завершение обработчика."""
        self.current -= 1


def rendezvous_handler(
    barrier: asyncio.Barrier,
    counter: InFlightCounter | None = None,
    *,
    timeout: float = TIMEOUT_SECONDS,
) -> TaskHandler:
    """Обработчик, который завершается только вместе с остальными участниками встречи.

    Встреча — управляемая точка синхронизации: обработчик не отпускает воркера, пока
    нужное число задач не окажется в работе одновременно. Одновременность обеспечена
    построением, а не удачным порядком выполнения корутин; если её нет, ожидание
    оборвётся по таймауту и задача станет неуспешной.
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
    """Ставит в очередь `count` задач анализа глав."""
    for number in range(count):
        await queue.enqueue(make_identity(scope_id=f"section-{number}"))


def make_pool(
    task_queue: SqliteTaskQueue, handler: TaskHandler, settings: WorkerSettings | None = None
) -> TaskWorkerPool:
    """Пул воркеров над очередью с единственным обработчиком."""
    return TaskWorkerPool(
        TaskRunner(queue=task_queue, handlers={TaskType.ANALYZE_CHAPTER: handler}),
        settings,
    )


@pytest.mark.parametrize("concurrency", [1, 2, 3])
async def test_simultaneous_handlers_are_limited_by_the_setting(
    task_queue: SqliteTaskQueue, stored_book: None, concurrency: int
) -> None:
    """Одновременно выполняется ровно столько задач, сколько разрешает настройка."""
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
    """По умолчанию вторая задача ждёт первую и одновременно с ней не выполняется.

    Обе задачи ждут встречи вдвоём. При параллелизме по умолчанию она невозможна, поэтому
    ожидание обрывается, и обе задачи становятся неуспешными: это и есть доказательство
    того, что одновременно работала одна задача (PRD § 25).
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
    """Постоянно работающий воркер ждёт новых задач и останавливается по запросу."""
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
    """Настройка воркера не принимает значений, при которых он не работает."""
    with pytest.raises(ValueError):
        WorkerSettings(**settings)


class ControlledRunner:
    """Каждый начатый захват остаётся занят до разрешения теста."""

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
