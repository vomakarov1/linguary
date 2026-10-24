"""Local worker pool: concurrency limits and shutdown orchestration.

Concurrency is configurable with a default of one task (PRD § 25): sequential processing
is predictable, avoids LLM provider rate limits, and yields identical results because tasks
are data-independent. Each worker is an independent coroutine claiming its own task from the queue;
their count defines the maximum concurrent workload.

No external task broker is used in the MVP (ADR § 3), so workers wait for available tasks
using short database polling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from linguary.application.task_runner import TaskHandler, TaskRunner
from linguary.domain import ProcessingTask


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Local worker settings."""

    concurrency: int = 1
    idle_delay_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("параллелизм воркера должен быть положительным")
        if self.idle_delay_seconds < 0:
            raise ValueError("пауза опроса очереди не может быть отрицательной")


class TaskWorkerPool:
    """Worker pool managing a TaskRunner with concurrency limits."""

    def __init__(self, runner: TaskRunner, settings: WorkerSettings | None = None) -> None:
        self._runner = runner
        self._settings = settings or WorkerSettings()
        self._stop = asyncio.Event()
        self._concurrency = self._settings.concurrency
        self._changed = asyncio.Event()
        self._workers: dict[int, asyncio.Task[None]] = {}
        self._worker_wakes: dict[int, asyncio.Event] = {}
        self._running = False
        self._busy_slots: set[int] = set()
        self._capacity_waiters: set[int] = set()

    @property
    def concurrency(self) -> int:
        """Number of tasks the worker executes concurrently."""
        return self._concurrency

    def set_concurrency(self, value: int) -> None:
        """Adjusts target concurrency without cancelling in-flight work."""
        if type(value) is not int or value < 1:
            raise ValueError("параллелизм воркера должен быть положительным целым числом")
        self._concurrency = value
        if self._running and not self._stop.is_set():
            self._changed.set()
            for wake in self._worker_wakes.values():
                wake.set()

    async def drain(self) -> tuple[ProcessingTask, ...]:
        """Executes tasks until queue is empty and returns completed tasks.

        Batch run mode: used by CLI and tests where waiting for new tasks is unnecessary.
        """
        finished: list[ProcessingTask] = []
        await asyncio.gather(
            *(self._drain_worker(finished) for _ in range(self._concurrency))
        )
        return tuple(finished)

    async def run_forever(self) -> None:
        """Manages worker coroutines until stopped, then awaits their completion."""
        if self._running:
            raise RuntimeError("пул воркеров уже запущен")
        self._running = True
        self._stop.clear()
        try:
            while True:
                self._changed.clear()
                for slot, worker in tuple(self._workers.items()):
                    if worker.done():
                        del self._workers[slot]
                        del self._worker_wakes[slot]
                        worker.result()
                if not self._stop.is_set():
                    for slot in range(self._concurrency):
                        if slot not in self._workers:
                            wake = self._worker_wakes[slot] = asyncio.Event()
                            worker = asyncio.create_task(self._polling_worker(slot, wake))
                            self._workers[slot] = worker
                            worker.add_done_callback(self._worker_finished)
                elif not self._workers:
                    return
                await self._changed.wait()
        finally:
            self._running = False
            workers = tuple(self._workers.values())
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._workers.clear()
            self._worker_wakes.clear()
            self._busy_slots.clear()
            self._capacity_waiters.clear()

    def _worker_finished(self, worker: asyncio.Task[None]) -> None:
        self._changed.set()

    def request_stop(self) -> None:
        """Disallows new task claims; in-flight tasks finish without cancellation."""
        self._stop.set()
        self._changed.set()
        for wake in self._worker_wakes.values():
            wake.set()

    async def _drain_worker(self, finished: list[ProcessingTask]) -> None:
        """Drain worker coroutine: processes tasks while the queue has work."""
        while (task := await self._runner.run_next()) is not None:
            finished.append(task)

    async def _polling_worker(self, slot: int, wake: asyncio.Event) -> None:
        """Slot index is retained until completion of an already claimed task."""
        while not self._stop.is_set() and slot < self._concurrency:
            if len(self._busy_slots) >= self._concurrency:
                wake.clear()
                self._capacity_waiters.add(slot)
                try:
                    await wake.wait()
                finally:
                    self._capacity_waiters.discard(slot)
                continue
            self._busy_slots.add(slot)
            try:
                task = await self._runner.run_next()
            finally:
                self._busy_slots.remove(slot)
                for waiting_slot in self._capacity_waiters:
                    self._worker_wakes[waiting_slot].set()
            if task is None:
                await self._wait_for_work(wake)

    async def _wait_for_work(self, wake: asyncio.Event) -> None:
        """Queue polling is interrupted by shutdown or pool resize."""
        with suppress(TimeoutError):
            await asyncio.wait_for(wake.wait(), timeout=self._settings.idle_delay_seconds)
        wake.clear()


def to_async_handler(handler: Callable[[ProcessingTask], None]) -> TaskHandler:
    """Adapts a synchronous task handler into an asynchronous one.

    `ImportBook` and `ExportBook` use cases are synchronous and spend non-trivial time
    in EPUB parsing and SQLite transactions. Executed directly in a coroutine, they would block
    the event loop alongside other workers and the Web UI, and are therefore offloaded to a thread.
    """

    async def run(task: ProcessingTask) -> None:
        await asyncio.to_thread(handler, task)

    return run


__all__ = ["TaskWorkerPool", "WorkerSettings", "to_async_handler"]
