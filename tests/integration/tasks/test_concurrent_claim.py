"""Конкурентный захват задачи: одну задачу не берут две попытки (ALG-02, P2.T7).

Тест детерминирован по построению, а не по везению планировщика:

- все попытки захвата стартуют из общей точки синхронизации (`threading.Barrier`), поэтому
  окно между чтением кандидата и условным обновлением у них общее;
- сценарий повторяется десятками раундов, и падение любого раунда валит тест;
- утверждения проверяют инвариант «ровно один победитель, ровно одна попытка», который
  обязан выполняться при любом порядке выполнения потоков, а не совпадение с ожидаемым
  порядком.

Захват идёт через блокирующее хранилище напрямую: асинхронная обёртка очереди сводится
к тому же вызову в отдельном потоке, и проверять конкурентность через цикл событий значило бы
проверять `asyncio.to_thread`, а не транзакцию SQLite.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from task_factories import make_identity

from linguary.adapters.runtime import SqliteTaskStore
from linguary.domain import ProcessingTask, TaskStatus

#: Число одновременных попыток захвата.
WORKERS = 8

#: Число раундов: один раунд — одна задача и общий старт всех попыток.
ROUNDS = 25


@pytest.fixture
def workers() -> Iterator[ThreadPoolExecutor]:
    """Пул потоков, из которых задачу пытаются захватить одновременно."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        yield pool


def claim_together(
    task_store: SqliteTaskStore, pool: ThreadPoolExecutor, attempts: int
) -> list[ProcessingTask]:
    """Запускает `attempts` попыток захвата с общего старта и возвращает захваченное."""
    barrier = threading.Barrier(attempts)

    def attempt() -> ProcessingTask | None:
        barrier.wait()
        return task_store.claim_next_pending()

    futures = [pool.submit(attempt) for _ in range(attempts)]
    return [claimed for future in futures if (claimed := future.result()) is not None]


def test_single_pending_task_is_claimed_by_exactly_one_attempt(
    task_store: SqliteTaskStore, workers: ThreadPoolExecutor, stored_book: None
) -> None:
    """Одну задачу забирает ровно одна попытка, и попытка учитывается один раз."""
    for round_number in range(ROUNDS):
        task = task_store.enqueue(make_identity(scope_id=f"section-{round_number}"))

        claimed = claim_together(task_store, workers, WORKERS)

        assert [item.id for item in claimed] == [task.id], f"раунд {round_number}"
        assert (claimed[0].status, claimed[0].attempts) == (TaskStatus.RUNNING, 1)


def test_every_pending_task_is_claimed_once(
    task_store: SqliteTaskStore, workers: ThreadPoolExecutor, stored_book: None
) -> None:
    """Очередь из нескольких задач раздаётся конкурентным попыткам без пересечений."""
    expected = {
        task_store.enqueue(make_identity(scope_id=f"section-{number}")).id
        for number in range(WORKERS)
    }

    claimed = claim_together(task_store, workers, WORKERS)

    assert sorted(item.id for item in claimed) == sorted(expected)
    assert {item.attempts for item in claimed} == {1}
    assert task_store.claim_next_pending() is None
