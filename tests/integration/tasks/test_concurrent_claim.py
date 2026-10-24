"""Concurrent task claiming: no task is claimed by two attempts (ALG-02, P2.T7).

The test is deterministic by design, not reliant on scheduler luck:

- all claim attempts start from a shared synchronization barrier (`threading.Barrier`),
  so the window between candidate read and conditional update is shared among them;
- the scenario runs across dozens of rounds, and any round failure fails the test;
- assertions verify the invariant "exactly one winner, exactly one attempt", which
  must hold under any thread execution order, rather than matching a particular order.

Claiming goes through the blocking store directly: the async queue wrapper reduces
to the same call on a separate thread, and testing concurrency through the event loop
would test `asyncio.to_thread` rather than the SQLite transaction.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from task_factories import make_identity

from linguary.adapters.runtime import SqliteTaskStore
from linguary.domain import ProcessingTask, TaskStatus

#: Number of concurrent claim attempts.
WORKERS = 8

#: Number of rounds: one round per task with a shared start across all attempts.
ROUNDS = 25


@pytest.fixture
def workers() -> Iterator[ThreadPoolExecutor]:
    """Thread pool from which tasks are claimed concurrently."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        yield pool


def claim_together(
    task_store: SqliteTaskStore, pool: ThreadPoolExecutor, attempts: int
) -> list[ProcessingTask]:
    """Runs `attempts` claim attempts with a shared start and returns claimed tasks."""
    barrier = threading.Barrier(attempts)

    def attempt() -> ProcessingTask | None:
        barrier.wait()
        return task_store.claim_next_pending()

    futures = [pool.submit(attempt) for _ in range(attempts)]
    return [claimed for future in futures if (claimed := future.result()) is not None]


def test_single_pending_task_is_claimed_by_exactly_one_attempt(
    task_store: SqliteTaskStore, workers: ThreadPoolExecutor, stored_book: None
) -> None:
    """A single task is claimed by exactly one attempt, and the attempt is counted once."""
    for round_number in range(ROUNDS):
        task = task_store.enqueue(make_identity(scope_id=f"section-{round_number}"))

        claimed = claim_together(task_store, workers, WORKERS)

        assert [item.id for item in claimed] == [task.id], f"раунд {round_number}"
        assert (claimed[0].status, claimed[0].attempts) == (TaskStatus.RUNNING, 1)


def test_every_pending_task_is_claimed_once(
    task_store: SqliteTaskStore, workers: ThreadPoolExecutor, stored_book: None
) -> None:
    """A queue of multiple tasks is distributed across concurrent attempts without overlap."""
    expected = {
        task_store.enqueue(make_identity(scope_id=f"section-{number}")).id
        for number in range(WORKERS)
    }

    claimed = claim_together(task_store, workers, WORKERS)

    assert sorted(item.id for item in claimed) == sorted(expected)
    assert {item.attempts for item in claimed} == {1}
    assert task_store.claim_next_pending() is None
