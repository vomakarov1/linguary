"""Локальный воркер: сколько задач выполняется одновременно и когда он останавливается.

Параллелизм — настройка со значением по умолчанию в одну задачу (PRD § 25): последовательная
обработка предсказуема, защищает от ограничений частоты запросов у провайдера LLM и не меняет
результат, потому что задачи независимы по данным. Каждый воркер — отдельная корутина, которая
берёт из очереди свою задачу; их число и есть предел одновременной работы.

Внешняя очередь задач в MVP не используется (ADR § 3), поэтому свободных задач воркер ждёт
коротким опросом базы.
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
    """Настройки локального воркера."""

    concurrency: int = 1
    idle_delay_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("параллелизм воркера должен быть положительным")
        if self.idle_delay_seconds < 0:
            raise ValueError("пауза опроса очереди не может быть отрицательной")


class TaskWorkerPool:
    """Набор воркеров над одним runner-ом с ограничением одновременной работы."""

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
        """Сколько задач воркер выполняет одновременно."""
        return self._concurrency

    def set_concurrency(self, value: int) -> None:
        """Меняет целевое число исполнителей без отмены занятой работы."""
        if type(value) is not int or value < 1:
            raise ValueError("параллелизм воркера должен быть положительным целым числом")
        self._concurrency = value
        if self._running and not self._stop.is_set():
            self._changed.set()
            for wake in self._worker_wakes.values():
                wake.set()

    async def drain(self) -> tuple[ProcessingTask, ...]:
        """Выполняет задачи, пока очередь не опустеет, и возвращает завершённые.

        Режим разовой прогонки: им пользуются CLI и тесты, где ждать новых задач незачем.
        """
        finished: list[ProcessingTask] = []
        await asyncio.gather(
            *(self._drain_worker(finished) for _ in range(self._concurrency))
        )
        return tuple(finished)

    async def run_forever(self) -> None:
        """Управляет исполнителями до остановки, затем собирает их завершение."""
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
        """Запрещает новые захваты; начатые задачи завершаются без отмены."""
        self._stop.set()
        self._changed.set()
        for wake in self._worker_wakes.values():
            wake.set()

    async def _drain_worker(self, finished: list[ProcessingTask]) -> None:
        """Воркер разовой прогонки: работает, пока в очереди есть задачи."""
        while (task := await self._runner.run_next()) is not None:
            finished.append(task)

    async def _polling_worker(self, slot: int, wake: asyncio.Event) -> None:
        """Номер слота сохраняется до завершения уже начатого захвата."""
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
        """Опрос очереди прерывается остановкой или изменением размера пула."""
        with suppress(TimeoutError):
            await asyncio.wait_for(wake.wait(), timeout=self._settings.idle_delay_seconds)
        wake.clear()


def to_async_handler(handler: Callable[[ProcessingTask], None]) -> TaskHandler:
    """Делает из синхронного обработчика асинхронный.

    Сценарии `ImportBook` и `ExportBook` синхронны и заметное время проводят в разборе
    EPUB и в SQLite. Выполненные прямо в корутине, они остановили бы цикл событий вместе
    с остальными воркерами и Web UI, поэтому исполняются в отдельном потоке.
    """

    async def run(task: ProcessingTask) -> None:
        await asyncio.to_thread(handler, task)

    return run


__all__ = ["TaskWorkerPool", "WorkerSettings", "to_async_handler"]
