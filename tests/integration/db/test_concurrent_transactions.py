"""A single unit of work serves concurrent transactions across different executions.

The application is composed with a single instance of the transactional boundary: both
FastAPI worker threads serving web requests and the background queue worker interact with it.
A transaction belongs to the execution rather than the object, so simultaneous work from
two threads does not constitute nested transactions and does not cause a failure.

A nested transaction within the same execution remains an error: use cases must choose
transaction boundaries explicitly.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.ports.persistence import PersistenceError

#: How many times each executor opens and closes a transaction.
_ROUNDS = 25


def read_books(unit_of_work: SqlAlchemyUnitOfWork) -> int:
    """Opens transactions sequentially and returns the number of completed reads."""
    for round_number in range(1, _ROUNDS + 1):
        with unit_of_work.transaction() as transaction:
            transaction.books.list_all()
    return round_number


def test_parallel_threads_do_not_share_one_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Transactions across different threads do not interfere with each other."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [pool.submit(read_books, unit_of_work) for _ in range(4)]
        assert [result.result() for result in results] == [_ROUNDS] * 4


async def test_parallel_tasks_do_not_share_one_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Transactions across different event loop tasks do not interfere with each other."""
    results = await asyncio.gather(
        *(asyncio.to_thread(read_books, unit_of_work) for _ in range(4))
    )

    assert list(results) == [_ROUNDS] * 4


def test_nested_transaction_is_still_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Nested transaction within the same execution is still rejected."""
    with (
        unit_of_work.transaction(),
        pytest.raises(PersistenceError, match="вложенные транзакции"),
    ):
        with unit_of_work.transaction():
            pass


def test_repository_is_unavailable_after_the_transaction(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """After exiting a transaction, repositories become unavailable again."""
    with unit_of_work.transaction() as transaction:
        transaction.books.list_all()

    with pytest.raises(PersistenceError, match="только внутри transaction"):
        unit_of_work.books.list_all()
