"""Cancellation logging and transient SQLite lock handling for call diagnostics."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.application.diagnostics import diagnostics_for
from linguary.application.ports.llm import LLMRequest, LLMResponse
from linguary.application.ports.persistence import PersistenceError
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


async def test_transient_sqlite_busy_does_not_permanently_poison_diagnostics() -> None:
    """A temporary SQLite lock is re-raised and does not stop later diagnostic writes."""
    unit_of_work = _UnitOfWork()
    unit_of_work.fail_transactions = 1
    inner = _Inner([_response(), _response()])
    client = _client(inner, unit_of_work)

    with pytest.raises(PersistenceError, match="занята"):
        await client.generate(_request())
    diagnostics = diagnostics_for(unit_of_work)
    assert inner.calls == 0
    assert client._observation_failed is False
    assert diagnostics._failed is False

    unit_of_work.llm_calls.fail_updates = 1
    with pytest.raises(PersistenceError, match="занята"):
        await client.generate(_request())
    assert client._observation_failed is False
    assert diagnostics._failed is False
    diagnostics.ensure_available()

    response = await client.generate(_request())
    assert response.call_id is not None
    assert client._observation_failed is False
    assert diagnostics._failed is False
    assert any(
        entry.success and entry.id == response.call_id for entry in unit_of_work.llm_calls.entries
    )
