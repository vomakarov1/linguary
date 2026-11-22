"""Cancellation logging and transient SQLite lock handling for call diagnostics."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.adapters.runtime.worker import TaskWorkerPool, WorkerSettings
from linguary.application.block_processing import finalize_response, record_rejection
from linguary.application.diagnostics import diagnostics_for
from linguary.application.ports.llm import (
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.application.ports.persistence import PersistenceError
from linguary.application.reader_text import record_reader_warnings
from linguary.domain import LLMCallLog, LLMCallStage

_BUSY = PersistenceError("база данных занята после ограниченного повтора")


class _Calls:
    def __init__(self) -> None:
        self.entries: list[LLMCallLog] = []
        self.metadata: dict[str, dict[str, object]] = {}
        self.bodies: dict[str, str | None] = {}
        self.fail_updates = 0

    def add(self, entry: LLMCallLog, *, response_text: str | None = None) -> None:
        self.entries.append(entry)
        self.metadata[entry.id] = dict(entry.diagnostic_meta or {})
        self.bodies[entry.id] = response_text

    def diagnostic_record(self, *, call_id: str | None = None, attempt_id: str | None = None):
        entry = self._find(call_id=call_id, attempt_id=attempt_id)
        if entry is None:
            return None
        return {
            "call_id": entry.id,
            "attempt_id": entry.attempt_id,
            "book_id": entry.book_id,
            "started_at": entry.started_at,
            "metadata": self.metadata.get(entry.id),
            "body": self.bodies.get(entry.id),
        }

    def update_diagnostics(self, call_id: str, *, metadata: dict[str, object]) -> None:
        if self.fail_updates:
            self.fail_updates -= 1
            raise _BUSY
        if call_id not in self.metadata:
            raise PersistenceError("метаданные вызова отсутствуют")
        self.metadata[call_id] = metadata

    def store_diagnostic_body(self, call_id: str, body: str | None) -> None:
        self.bodies[call_id] = body

    def enforce_diagnostics_policy(self, policy: object, now: datetime, *, book_id: str | None = None) -> None:
        return None

    def _find(self, *, call_id: str | None, attempt_id: str | None) -> LLMCallLog | None:
        if attempt_id is not None:
            return next((entry for entry in self.entries if entry.attempt_id == attempt_id), None)
        if call_id is not None:
            return next((entry for entry in self.entries if entry.id == call_id), None)
        return None


class _UnitOfWork:
    def __init__(self) -> None:
        self.llm_calls = _Calls()
        self.fail_transactions = 0

    @contextmanager
    def transaction(self) -> Iterator[_UnitOfWork]:
        if self.fail_transactions:
            self.fail_transactions -= 1
            raise _BUSY
        yield self


class _Inner:
    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.calls = 0

    async def generate(self, request: LLMRequest[object]) -> LLMResponse[object]:
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _request() -> LLMRequest[object]:
    return LLMRequest.for_translation(instructions="Переведи", input_text="Текст").with_logging(
        book_id="book-1", stage=LLMCallStage.TRANSLATION_UNIT
    )


def _response() -> LLMResponse[object]:
    return LLMResponse(payload=object(), provider="fake", model="fake", raw_content="ответ")


def _client(inner: _Inner, unit_of_work: _UnitOfWork) -> LoggingLLMClient:
    sequence = iter(range(1, 100))
    return LoggingLLMClient(
        inner=inner,
        unit_of_work=unit_of_work,
        provider="fake",
        model="fake",
        clock=lambda: datetime(2026, 11, 22, tzinfo=UTC),
        id_factory=lambda: f"call-{next(sequence)}",
    )


async def test_logging_client_records_cancelled_call_and_reraises() -> None:
    """A cancelled model call is logged as an unknown failure and the cancellation propagates."""
    unit_of_work = _UnitOfWork()
    inner = _Inner([asyncio.CancelledError()])
    client = _client(inner, unit_of_work)

    with pytest.raises(asyncio.CancelledError):
        await client.generate(_request())

    assert inner.calls == 1
    (entry,) = unit_of_work.llm_calls.entries
    assert entry.success is False
    assert entry.diagnostic_meta is not None
    assert entry.diagnostic_meta["decision"] == "unknown"
    assert unit_of_work.llm_calls.metadata[entry.id]["decision"] == "unknown"
    assert client._observation_failed is False
    assert diagnostics_for(unit_of_work)._failed is False


async def test_cancelled_call_propagates_cancellation_even_when_logging_fails() -> None:
    """Logging failure inside CancelledError handler never masks cancellation."""
    unit_of_work = _UnitOfWork()
    unit_of_work.fail_transactions = 10
    inner = _Inner([asyncio.CancelledError()])
    client = _client(inner, unit_of_work)

    with pytest.raises(asyncio.CancelledError):
        await client.generate(_request())
    assert inner.calls == 1
    assert client._observation_failed is False
    assert diagnostics_for(unit_of_work)._failed is False

    unit_of_work_hard = _UnitOfWork()

    def broken_observer(*_args: object) -> None:
        raise RuntimeError("observer broken")

    inner_hard = _Inner([asyncio.CancelledError()])
    client_hard = LoggingLLMClient(
        inner=inner_hard,
        unit_of_work=unit_of_work_hard,
        provider="fake",
        model="fake",
        clock=lambda: datetime(2026, 11, 22, tzinfo=UTC),
        id_factory=lambda: "call-hard",
        on_logged=broken_observer,
    )
    with pytest.raises(asyncio.CancelledError):
        await client_hard.generate(_request())


async def test_transient_sqlite_busy_does_not_overwrite_outcomes_or_poison_diagnostics() -> None:
    """Transient SQLite busy in periodic_purge or _log preserves LLM outcomes and call_id=None safety."""
    unit_of_work = _UnitOfWork()
    unit_of_work.fail_transactions = 1
    inner = _Inner([_response(), _response()])
    client = _client(inner, unit_of_work)

    # First call: periodic_purge hits transient busy -> skipped, inner call runs and succeeds.
    first = await client.generate(_request())
    diagnostics = diagnostics_for(unit_of_work)
    assert inner.calls == 1
    assert first.call_id is not None
    assert diagnostics._last_cleanup is None
    assert client._observation_failed is False
    assert diagnostics._failed is False

    # Persistent transient busy across all 3 _log retries -> response returned with call_id=None.
    unit_of_work.llm_calls.fail_updates = 10
    second = await client.generate(_request())
    assert second.call_id is None
    assert client._observation_failed is False
    assert diagnostics._failed is False

    # Downstream helpers respect call_id=None without poisoning diagnostics.
    finalize_response(unit_of_work, second, "accepted")
    record_rejection(unit_of_work, second, None)
    record_reader_warnings(
        unit_of_work,
        second,
        [("translation", "I am sorry, as an AI language model")],
        reader_level="b1",
    )
    assert diagnostics._failed is False

    # Transient busy during error logging never overwrites LLMTransportError or LLMResponseFormatError.
    unit_of_work.fail_transactions = 20
    not_sent = LLMTransportError("connect failed", outcome="not_sent")
    rate_limited = LLMTransportError("429", outcome="known_transient", retry_after_seconds=12.0)
    fmt_error = LLMResponseFormatError("bad json")
    err_client = _client(_Inner([not_sent, rate_limited, fmt_error]), unit_of_work)

    with pytest.raises(LLMTransportError) as caught_not_sent:
        await err_client.generate(_request())
    assert caught_not_sent.value is not_sent
    assert caught_not_sent.value.outcome == "not_sent"

    with pytest.raises(LLMTransportError) as caught_429:
        await err_client.generate(_request())
    assert caught_429.value is rate_limited
    assert caught_429.value.outcome == "known_transient"
    assert caught_429.value.retry_after_seconds == 12.0

    with pytest.raises(LLMResponseFormatError) as caught_fmt:
        await err_client.generate(_request())
    assert caught_fmt.value is fmt_error
    assert caught_fmt.value.call_id is None
    assert diagnostics._failed is False


def test_sqlalchemy_uow_translates_commit_operational_locked_error(tmp_path) -> None:
    """OperationalError('database is locked') during commit translates into PersistenceError."""
    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError

    from linguary.adapters.db import Base, SqlAlchemyUnitOfWork, create_database_engine

    engine = create_database_engine(tmp_path / "lock.db")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    diagnostics = diagnostics_for(uow)
    try:
        with pytest.raises(PersistenceError, match="занята"):
            with uow.transaction() as tx:
                session = tx._active_session()

                def fail_commit(_session):
                    raise OperationalError("COMMIT", {}, Exception("database is locked"))

                event.listen(session, "before_commit", fail_commit)
        assert diagnostics._failed is False
    finally:
        engine.dispose()


async def test_worker_pool_survives_transient_sqlite_busy() -> None:
    """TaskWorkerPool.run_forever() keeps running after a transient SQLite lock error."""
    calls = 0
    done = asyncio.Event()

    class _FlakyRunner:
        async def run_next(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _BUSY
            if calls == 2:
                done.set()
                return object()
            return None

    pool = TaskWorkerPool(
        _FlakyRunner(),  # type: ignore[arg-type]
        WorkerSettings(concurrency=1, idle_delay_seconds=0.01),
    )
    task = asyncio.create_task(pool.run_forever())
    try:
        await asyncio.wait_for(done.wait(), timeout=2.0)
    finally:
        pool.request_stop()
        await asyncio.wait_for(task, timeout=2.0)
    assert calls >= 2
