"""Correction send marks follow the same transport outcomes as processing."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.adapters.llm.rate_limit import LLMRateLimiter, RateLimitedLLMClient
from linguary.application.correction_dto import CORRECTION_PARAMETERS, CORRECTION_VERSIONS, CorrectionSnapshot
from linguary.application.ports.llm import (
    LLMRequest,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.application.recovery import RecoveryError
from linguary.composition.correction_runtime import CorrectionProvenanceClient
from linguary.domain import LLMCallStage, ProcessingRunStatus, ProcessingTask, TaskStatus, TaskType
from linguary.domain.book_conditions import WorkStateError, thaw
from linguary.domain.correction import CORRECTION_POLICY_VERSION, correction_task_identity
from linguary.domain.recovery import CallAttempt


def _snapshot() -> CorrectionSnapshot:
    parameters = {name: 1 for name in CORRECTION_PARAMETERS}
    parameters["max_output_tokens"] = None
    return CorrectionSnapshot(
        model="pinned-model",
        route={
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "proxy": "",
            "route_id": "route-1",
        },
        parameters=parameters,
        prompt="Проверь главу.",
        versions=CORRECTION_VERSIONS,
    )


class _Work:
    def __init__(self) -> None:
        self.attempts: dict[str, CallAttempt] = {}
        self.bindings: dict[tuple[str, str], dict] = {}

    def get_binding(self, kind, entity_id):
        return self.bindings.get((kind, entity_id))

    def get_attempt(self, attempt_id):
        return self.attempts.get(attempt_id)

    def add_attempt(self, attempt):
        self.attempts[attempt.id] = attempt

    def list_task_attempts(self, task_id):
        return tuple(item for item in self.attempts.values() if item.task_id == task_id)

    def transition_attempt(self, attempt_id, state, *, expected_state, outcome=None):
        current = self.attempts[attempt_id]
        if current.state != expected_state:
            raise WorkStateError("состояние попытки изменилось")
        self.attempts[attempt_id] = current.transition(state, outcome=outcome)
        return self.attempts[attempt_id]

    def cancel_unsent_attempt(self, attempt_id):
        current = self.attempts[attempt_id]
        if current.state != "created":
            raise WorkStateError("состояние попытки изменилось")
        self.attempts[attempt_id] = current.transition("cancelled")

    def rollback_not_sent_attempt(self, attempt_id):
        current = self.attempts[attempt_id]
        if current.state != "sent":
            raise WorkStateError("состояние попытки изменилось")
        self.attempts[attempt_id] = CallAttempt(
            current.id, current.book_id, state="cancelled", generation=current.generation,
            ordinal=current.ordinal, conditions_id=current.conditions_id, task_id=current.task_id,
            run_id=current.run_id, context=thaw(current.context), outcome=thaw(current.outcome),
        )


class _UoW:
    def __init__(self, work, run, task):
        self._tx = SimpleNamespace(
            work=work,
            runs=SimpleNamespace(get=lambda run_id: run if run_id == run.id else None),
            tasks=SimpleNamespace(get=lambda task_id: task if task_id == task.id else None),
        )

    @contextmanager
    def transaction(self):
        yield self._tx


class _Inner:
    def __init__(self, error):
        self.error = error
        self.calls = []

    async def generate(self, request):
        self.calls.append(request)
        raise self.error


def _client(error, *, limiter=None, prior=()):
    snapshot = _snapshot()
    book_id = "book-1"
    run_id = "run-1"
    section_id = "section-1"
    conditions_id = str(uuid4())
    identity = correction_task_identity(
        book_id=book_id, run_id=run_id, section_id=section_id, input_generation="gen-1",
        policy_version=CORRECTION_POLICY_VERSION,
    )
    task = ProcessingTask(
        id="task-1", book_id=book_id, task_type=TaskType.CHAPTER_CORRECTION, scope_id=section_id,
        status=TaskStatus.RUNNING, attempts=1, operation_id=identity.operation_id,
        policy_version=identity.policy_version,
    )
    run = SimpleNamespace(id=run_id, book_id=book_id, status=ProcessingRunStatus.ACTIVE)
    work = _Work()
    work.bindings[("run", run_id)] = {
        "book_id": book_id,
        "conditions_id": conditions_id,
        "payload_json": {"correction": snapshot.to_payload()},
    }
    work.bindings[("task", task.id)] = {
        "book_id": book_id,
        "conditions_id": conditions_id,
        "payload_json": {
            "run_id": run_id,
            "section_id": section_id,
            "input_generation": "gen-1",
        },
    }
    for attempt in prior:
        work.add_attempt(attempt)
    service = SimpleNamespace(
        uow=_UoW(work, run, task),
        require=lambda book, transaction=None: SimpleNamespace(id=conditions_id),
    )
    inner = _Inner(error)
    wrapped = inner if limiter is None else RateLimitedLLMClient(inner=inner, limiter=limiter)
    client = CorrectionProvenanceClient(wrapped, service, run_id, conditions_id, snapshot)
    request = LLMRequest.for_chapter_correction(
        instructions=snapshot.prompt, input_text="{}", ordered_block_ids=("block-1",),
    ).with_logging(
        book_id=book_id, stage=LLMCallStage.CHAPTER_CORRECTION, task_id=task.id,
        scope_id=section_id, ordered_block_ids=("block-1",), run_id=run_id,
    )
    return client, request, work, inner


async def _send(error):
    client, request, work, inner = _client(error)
    with pytest.raises(type(error)):
        await client.generate(request)
    return work, inner


@pytest.mark.asyncio
async def test_correction_not_sent_rolls_back_and_known_failures_stay_known():
    not_sent, _inner = await _send(LLMTransportError("соединение закрыто", outcome="not_sent"))
    assert [item.state for item in not_sent.attempts.values()] == ["cancelled"]

    transient, _inner = await _send(LLMTransportError("HTTP 429", outcome="known_transient"))
    assert [item.state for item in transient.attempts.values()] == ["failed"]

    formatted, _inner = await _send(LLMResponseFormatError("пустой ответ"))
    assert [item.state for item in formatted.attempts.values()] == ["failed"]

    unknown, _inner = await _send(LLMTransportError("чтение", outcome="unknown"))
    assert [item.state for item in unknown.attempts.values()] == ["unknown"]


@pytest.mark.asyncio
async def test_correction_cancel_while_queued_is_not_sent():
    limiter = LLMRateLimiter(max_concurrency=1)
    await limiter.acquire_concurrency()
    client, request, work, inner = _client(LLMTransportError("не должно уйти"), limiter=limiter)
    running = asyncio.create_task(client.generate(request))
    for _ in range(10):
        await asyncio.sleep(0)
        if work.attempts:
            break
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    limiter.release_concurrency()
    assert inner.calls == []
    assert [item.state for item in work.attempts.values()] == ["cancelled"]


@pytest.mark.asyncio
async def test_correction_refuses_unacknowledged_unknown_before_another_send():
    prior = CallAttempt(str(uuid4()), "book-1", state="unknown", task_id="task-1")
    client, request, work, inner = _client(LLMTransportError("ещё один"), prior=(prior,))
    with pytest.raises(RecoveryError, match="unknown_outcome"):
        await client.generate(request)
    assert inner.calls == []
    assert [item.state for item in work.attempts.values()] == ["unknown"]

    acknowledged = CallAttempt(
        str(uuid4()), "book-1", state="unknown", task_id="task-1",
        outcome={"schema_version": 1, "acknowledged_duplicate_billing_risk": True},
    )
    allowed, allowed_request, allowed_work, allowed_inner = _client(
        LLMTransportError("после подтверждения", outcome="known_transient"), prior=(acknowledged,),
    )
    with pytest.raises(LLMTransportError):
        await allowed.generate(allowed_request)
    assert len(allowed_inner.calls) == 1
    states = sorted(item.state for item in allowed_work.attempts.values())
    assert states == ["failed", "unknown"]
