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
