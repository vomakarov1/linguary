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

import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from task_factories import BOOK_ID, make_book, make_identity, sequential_task_ids

from linguary.adapters.db.engine import (
    DEFAULT_BUSY_TIMEOUT_MS,
    create_database_engine,
)
from linguary.adapters.db.models import Base
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import SqliteTaskStore
from linguary.adapters.runtime import task_store as task_store_module
from linguary.application.illustrations import dispatch_scope_task
from linguary.application.ports.persistence import PersistenceError
from linguary.domain import ErrorType, ProcessingTask, TaskError, TaskStatus, TaskType

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

        assert [item.id for item in claimed] == [task.id], f"round {round_number}"
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


def test_engine_configures_15s_default_busy_timeout_and_pysqlite_timeout(
    tmp_path: Path,
) -> None:
    """`create_database_engine` defaults to 15s busy_timeout and propagates connect timeout."""
    assert DEFAULT_BUSY_TIMEOUT_MS == 15_000
    db_path = tmp_path / "busy-timeout.db"
    engine = create_database_engine(db_path)
    try:
        with engine.connect() as conn:
            busy_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar_one()
        assert busy_timeout == 15_000
    finally:
        engine.dispose()

    custom_engine = create_database_engine(db_path, busy_timeout_ms=250)
    try:
        with custom_engine.connect() as conn:
            custom_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar_one()
        assert custom_timeout == 250
    finally:
        custom_engine.dispose()


def test_task_store_enqueue_and_save_survive_transient_sqlite_write_lock(
    tmp_path: Path,
) -> None:
    """`SqliteTaskStore.enqueue` and `save` wait through a brief external write lock."""
    db_path = tmp_path / "transient-lock.db"
    engine = create_database_engine(db_path, busy_timeout_ms=200)
    Base.metadata.create_all(engine)
    try:
        uow = SqlAlchemyUnitOfWork.for_engine(engine)
        with uow.transaction() as tx:
            tx.books.add(make_book())
        store = SqliteTaskStore.for_engine(engine, id_factory=sequential_task_ids())

        def hold_write_lock(duration_seconds: float, acquired: threading.Event) -> None:
            raw = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
            try:
                raw.execute("BEGIN IMMEDIATE")
                acquired.set()
                time.sleep(duration_seconds)
                raw.execute("ROLLBACK")
            finally:
                raw.close()

        lock_acquired = threading.Event()
        holder = threading.Thread(target=hold_write_lock, args=(0.08, lock_acquired))
        holder.start()
        assert lock_acquired.wait(timeout=2.0)
        try:
            enqueued = store.enqueue(make_identity(scope_id="section-lock-1"))
        finally:
            holder.join(timeout=2.0)
        assert enqueued.status is TaskStatus.PENDING

        claimed = store.claim_next_pending()
        assert claimed is not None and claimed.id == enqueued.id

        lock_acquired2 = threading.Event()
        holder2 = threading.Thread(target=hold_write_lock, args=(0.08, lock_acquired2))
        holder2.start()
        assert lock_acquired2.wait(timeout=2.0)
        try:
            saved = store.save(claimed.succeed())
        finally:
            holder2.join(timeout=2.0)
        assert saved.status is TaskStatus.SUCCEEDED
    finally:
        engine.dispose()


def test_task_store_enqueue_and_save_raise_persistence_error_on_persistent_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent write lock exhausts 10 immediate-write attempts and raises `PersistenceError`."""
    db_path = tmp_path / "persistent-lock.db"
    engine = create_database_engine(db_path, busy_timeout_ms=1)
    Base.metadata.create_all(engine)
    sleeps: list[float] = []
    try:
        uow = SqlAlchemyUnitOfWork.for_engine(engine)
        with uow.transaction() as tx:
            tx.books.add(make_book())
        store = SqliteTaskStore.for_engine(engine, id_factory=sequential_task_ids())
        existing = store.enqueue(make_identity(scope_id="section-existing"))
        claimed = store.claim_next_pending()
        assert claimed is not None and claimed.id == existing.id

        monkeypatch.setattr(
            task_store_module,
            "sleep",
            lambda seconds: sleeps.append(round(seconds, 4)),
        )

        raw = sqlite3.connect(str(db_path), isolation_level=None, timeout=0.001)
        try:
            raw.execute("BEGIN IMMEDIATE")

            with pytest.raises(PersistenceError, match="база данных занята"):
                store.enqueue(make_identity(scope_id="section-blocked"))
            assert sleeps == [0.005, 0.01, 0.015, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

            sleeps.clear()
            with pytest.raises(PersistenceError, match="база данных занята"):
                store.save(claimed.succeed())
            assert sleeps == [0.005, 0.01, 0.015, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        finally:
            raw.execute("ROLLBACK")
            raw.close()
    finally:
        engine.dispose()


def test_task_repository_list_for_scope_filters_exact_book_type_and_scope(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """`SqlAlchemyTaskRepository.list_for_scope` returns only matching `(book_id, task_type, scope_id)` tasks."""
    with unit_of_work.transaction() as tx:
        tx.books.add(make_book(book_id="book-other"))
        t1 = ProcessingTask(
            id="t-1",
            book_id=BOOK_ID,
            task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
            scope_id="scope-a",
            operation_id="gen-1:a1",
            policy_version="illustration-v1",
        )
        t2 = ProcessingTask(
            id="t-2",
            book_id=BOOK_ID,
            task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
            scope_id="scope-a",
            operation_id="gen-1:a2",
            policy_version="illustration-v1",
        )
        other_scope = ProcessingTask(
            id="t-other-scope",
            book_id=BOOK_ID,
            task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
            scope_id="scope-b",
            operation_id="gen-1:a1",
            policy_version="illustration-v1",
        )
        other_type = ProcessingTask(
            id="t-other-type",
            book_id=BOOK_ID,
            task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
            scope_id="scope-a",
            operation_id="gen-1:a1",
            policy_version="illustration-v1",
        )
        other_book = ProcessingTask(
            id="t-other-book",
            book_id="book-other",
            task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
            scope_id="scope-a",
            operation_id="gen-1:a1",
            policy_version="illustration-v1",
        )
        for task in (t1, other_scope, other_type, other_book, t2):
            tx.tasks.add(task)

    with unit_of_work.read_transaction() as tx:
        matched = tx.tasks.list_for_scope(
            BOOK_ID, TaskType.PREPARE_ILLUSTRATION_REFERENCE, "scope-a"
        )
        assert [task.id for task in matched] == ["t-1", "t-2"]
        assert (
            tx.tasks.list_for_scope(
                BOOK_ID, TaskType.PREPARE_ILLUSTRATION_REFERENCE, "scope-missing"
            )
            == ()
        )


def test_dispatch_scope_task_uses_list_for_scope_and_supports_legacy_fake_fallback() -> None:
    """`dispatch_scope_task` queries `list_for_scope` instead of `list_for_book`, with fallback for fakes."""
    prior = ProcessingTask(
        id="task-prev",
        book_id=BOOK_ID,
        task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        scope_id="scope-1",
        operation_id="gen-1:a2",
        policy_version="illustration-v1",
    ).start().fail(TaskError(type=ErrorType.TRANSIENT, message="failed prior attempt"))
    list_for_scope_calls: list[tuple[str, TaskType, str]] = []
    added: list[ProcessingTask] = []
    saved: list[ProcessingTask] = []
    bound: list[tuple[str, str]] = []

    class _ScopedTasksRepo:
        def get(self, task_id: str) -> ProcessingTask | None:
            return prior if task_id == prior.id else None

        def list_for_book(self, book_id: str) -> tuple[ProcessingTask, ...]:
            raise AssertionError("list_for_book must not be called when list_for_scope exists")

        def list_for_scope(
            self, book_id: str, task_type: TaskType, scope_id: str
        ) -> tuple[ProcessingTask, ...]:
            list_for_scope_calls.append((book_id, task_type, scope_id))
            return (prior,)

        def save(self, task: ProcessingTask) -> None:
            saved.append(task)

        def add(self, task: ProcessingTask) -> None:
            added.append(task)

    scope = SimpleNamespace(id="scope-1", task_id=prior.id)
    tx = SimpleNamespace(
        tasks=_ScopedTasksRepo(),
        illustrations=SimpleNamespace(set_scope_task=lambda sid, tid: bound.append((sid, tid))),
    )
    dispatched = dispatch_scope_task(
        tx,  # type: ignore[arg-type]
        scope,  # type: ignore[arg-type]
        generation_id="gen-1",
        task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        id_factory=lambda: "task-next",
        book_id=BOOK_ID,
    )
    assert list_for_scope_calls == [
        (BOOK_ID, TaskType.PREPARE_ILLUSTRATION_REFERENCE, "scope-1")
    ]
    assert dispatched.id == "task-next"
    assert dispatched.operation_id == "gen-1:a3"
    assert len(saved) == 1 and saved[0].status is TaskStatus.SUPERSEDED
    assert bound == [("scope-1", "task-next")]

    # Fallback path for legacy unit-test fake repositories without `list_for_scope`.
    legacy_added: list[ProcessingTask] = []

    class _LegacyFakeTasksRepo:
        def get(self, task_id: str) -> ProcessingTask | None:
            return None

        def list_for_book(self, book_id: str) -> tuple[ProcessingTask, ...]:
            return (prior,)

        def save(self, task: ProcessingTask) -> None:
            pass

        def add(self, task: ProcessingTask) -> None:
            legacy_added.append(task)

    legacy_tx = SimpleNamespace(
        tasks=_LegacyFakeTasksRepo(),
        illustrations=SimpleNamespace(set_scope_task=lambda sid, tid: None),
    )
    legacy_scope = SimpleNamespace(id="scope-1", task_id=None)
    fallback_task = dispatch_scope_task(
        legacy_tx,  # type: ignore[arg-type]
        legacy_scope,  # type: ignore[arg-type]
        generation_id="gen-1",
        task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        id_factory=lambda: "task-fallback",
        book_id=BOOK_ID,
    )
    assert fallback_task.id == "task-fallback"
    assert fallback_task.operation_id == "gen-1:a3"
    assert [t.id for t in legacy_added] == ["task-fallback"]
