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

    @property
    def concurrency(self) -> int:
        """Сколько задач воркер выполняет одновременно."""
        return self._settings.concurrency

    async def drain(self) -> tuple[ProcessingTask, ...]:
        """Выполняет задачи, пока очередь не опустеет, и возвращает завершённые.

        Режим разовой прогонки: им пользуются CLI и тесты, где ждать новых задач незачем.
        """
        finished: list[ProcessingTask] = []
        await asyncio.gather(
            *(self._drain_worker(finished) for _ in range(self._settings.concurrency))
        )
        return tuple(finished)

    async def run_forever(self) -> None:
        """Выполняет задачи, пока не запрошена остановка."""
        self._stop.clear()
        await asyncio.gather(
            *(self._polling_worker() for _ in range(self._settings.concurrency))
        )

    def request_stop(self) -> None:
        """Просит воркеры завершиться: текущие задачи доигрываются до конца."""
        self._stop.set()

    async def _drain_worker(self, finished: list[ProcessingTask]) -> None:
        """Воркер разовой прогонки: работает, пока в очереди есть задачи."""
        while (task := await self._runner.run_next()) is not None:
            finished.append(task)

    async def _polling_worker(self) -> None:
        """Воркер постоянной работы: пустую очередь опрашивает с паузой."""
        while not self._stop.is_set():
            if await self._runner.run_next() is None:
                await self._wait_for_work()

    async def _wait_for_work(self) -> None:
        """Пауза перед следующим опросом очереди, прерываемая остановкой."""
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self._stop.wait(), timeout=self._settings.idle_delay_seconds
            )


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
