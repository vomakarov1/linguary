"""Один unit of work обслуживает одновременные транзакции разных исполнений.

Приложение собирается с единственным экземпляром транзакционной границы: с ним работают
и веб-запросы в рабочих потоках FastAPI, и фоновый воркер очереди. Транзакция принадлежит
исполнению, а не объекту, поэтому одновременная работа двух потоков — не вложенные
транзакции и отказа не вызывает.

Вложенная транзакция того же исполнения остаётся ошибкой: сценарий обязан выбирать
транзакционную границу явно.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.ports.persistence import PersistenceError

#: Сколько раз каждый исполнитель открывает и закрывает транзакцию.
_ROUNDS = 25


def read_books(unit_of_work: SqlAlchemyUnitOfWork) -> int:
    """Открывает транзакции подряд и возвращает число выполненных чтений."""
    for round_number in range(1, _ROUNDS + 1):
        with unit_of_work.transaction() as transaction:
            transaction.books.list_all()
    return round_number


def test_parallel_threads_do_not_share_one_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Транзакции разных потоков не мешают друг другу."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [pool.submit(read_books, unit_of_work) for _ in range(4)]
        assert [result.result() for result in results] == [_ROUNDS] * 4


async def test_parallel_tasks_do_not_share_one_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Транзакции разных задач цикла событий не мешают друг другу."""
    results = await asyncio.gather(
        *(asyncio.to_thread(read_books, unit_of_work) for _ in range(4))
    )

    assert list(results) == [_ROUNDS] * 4


def test_nested_transaction_is_still_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Вложенная транзакция одного исполнения по-прежнему отвергается."""
    with (
        unit_of_work.transaction(),
        pytest.raises(PersistenceError, match="вложенные транзакции"),
    ):
        with unit_of_work.transaction():
            pass


def test_repository_is_unavailable_after_the_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """После выхода из транзакции репозитории снова недоступны."""
    with unit_of_work.transaction() as transaction:
        transaction.books.list_all()

    with pytest.raises(PersistenceError, match="только внутри transaction"):
        unit_of_work.books.list_all()
