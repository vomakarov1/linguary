from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.adapters.config.settings import AppSettings, LLMSettings
from linguary.composition.book_runtime import conditions_payload
from linguary.composition.correction_runtime import CorrectionRuntimeResolver, correction_snapshot
from linguary.domain import ProcessingRunStatus
from linguary.domain.book_conditions import BookConditions


class _FakeConfiguredClient:
    def __init__(self, llm_settings: LLMSettings) -> None:
        self.llm_settings = llm_settings
        self.closed = 0
        self.inner = SimpleNamespace()

    def client_for(self, model: str):
        return self.inner

    async def aclose(self) -> None:
        self.closed += 1


def _make_settings(api_key: str = "sk-corr-1") -> AppSettings:
    return AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="gpt-test",
            api_key=api_key,
            proxy_mode="direct",
        )
    )


@pytest.mark.asyncio
async def test_correction_runtime_resolver_reuses_client_and_retires_on_fingerprint_change() -> None:
    settings = _make_settings("sk-corr-1")
    book = SimpleNamespace(source_language="en", target_language="ru")
    conditions = BookConditions(
        id=str(uuid4()),
        book_id="book-1",
        revision=1,
        payload=conditions_payload(book, settings, 1, "content-plan-v3"),
        created_at=datetime.now(UTC),
    )
    snapshot = correction_snapshot(settings)
    run_id = "run-1"
    run = SimpleNamespace(id=run_id, book_id="book-1", status=ProcessingRunStatus.ACTIVE)
    binding = {
        "book_id": "book-1",
        "conditions_id": conditions.id,
        "payload_json": {"schema_version": 1, "correction": snapshot.to_payload()},
    }

    class _Runs:
        def get(self, target_run_id: str):
            return run if target_run_id == run_id else None

    class _Work:
        def get_binding(self, kind: str, entity_id: str):
            if kind == "run" and entity_id == run_id:
                return binding
            return None

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(runs=_Runs(), work=_Work())

    service = SimpleNamespace(
        uow=_UoW(),
        require=lambda book_id, transaction=None: conditions,
    )
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    created_clients: list[_FakeConfiguredClient] = []

    def client_factory(llm_settings: LLMSettings) -> _FakeConfiguredClient:
        client = _FakeConfiguredClient(llm_settings)
        created_clients.append(client)
        return client

    resolver = CorrectionRuntimeResolver(
        service,
        runtime,
        limiter=SimpleNamespace(),
        client_factory=client_factory,
    )

    async with resolver.lease(run_id) as rt1:
        assert len(created_clients) == 1
        assert created_clients[0].closed == 0

    assert created_clients[0].closed == 0

    async with resolver.lease(run_id) as rt2:
        assert rt2 is rt1
        assert len(created_clients) == 1
        assert created_clients[0].closed == 0

    assert created_clients[0].closed == 0

    async with resolver.lease(run_id) as rt_active:
        assert rt_active is rt1
        runtime.current = SimpleNamespace(settings=_make_settings("sk-corr-2"))

        async with resolver.lease(run_id) as rt_new:
            assert rt_new is not rt_active
            assert len(created_clients) == 2
            assert created_clients[0].closed == 0
            assert created_clients[1].closed == 0

        assert created_clients[0].closed == 0
        assert created_clients[1].closed == 0

    assert created_clients[0].closed == 1
    assert created_clients[1].closed == 0

    await resolver.aclose()
    assert created_clients[1].closed == 1


@pytest.mark.asyncio
async def test_correction_runtime_resolver_no_race_on_stale_close_and_resets_contextvars_and_evicts_old_runs() -> None:
    settings = _make_settings("sk-corr-v1")
    book = SimpleNamespace(source_language="en", target_language="ru")
    conditions = BookConditions(
        id=str(uuid4()),
        book_id="book-1",
        revision=1,
        payload=conditions_payload(book, settings, 1, "content-plan-v3"),
        created_at=datetime.now(UTC),
    )
    snapshot = correction_snapshot(settings)
    runs = {
        "run-1": SimpleNamespace(id="run-1", book_id="book-1", status=ProcessingRunStatus.ACTIVE),
        "run-2": SimpleNamespace(id="run-2", book_id="book-1", status=ProcessingRunStatus.ACTIVE),
    }
    bindings = {
        "run-1": {
            "book_id": "book-1",
            "conditions_id": conditions.id,
            "payload_json": {"schema_version": 1, "correction": snapshot.to_payload()},
        },
        "run-2": {
            "book_id": "book-1",
            "conditions_id": conditions.id,
            "payload_json": {"schema_version": 1, "correction": snapshot.to_payload()},
        },
    }

    class _Runs:
        def get(self, target_run_id: str):
            return runs.get(target_run_id)

    class _Work:
        def get_binding(self, kind: str, entity_id: str):
            if kind == "run":
                return bindings.get(entity_id)
            return None

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(runs=_Runs(), work=_Work())

    service = SimpleNamespace(
        uow=_UoW(),
        require=lambda book_id, transaction=None: conditions,
    )
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    close_started = asyncio.Event()
    allow_close_finish = asyncio.Event()

    class _SlowCloseClient(_FakeConfiguredClient):
        async def aclose(self) -> None:
            self.closed += 1
            if self.llm_settings.api_key == "sk-corr-v1" and self.closed == 1:
                close_started.set()
                await allow_close_finish.wait()

    created_clients: list[_SlowCloseClient] = []

    def client_factory(llm_settings: LLMSettings) -> _SlowCloseClient:
        client = _SlowCloseClient(llm_settings)
        created_clients.append(client)
        return client

    resolver = CorrectionRuntimeResolver(
        service,
        runtime,
        limiter=SimpleNamespace(),
        client_factory=client_factory,
    )

    async with resolver.lease("run-1") as rt1:
        rt1.llm_client._attempts.set(("att-1",))
        rt1.llm_client._responses.set((SimpleNamespace(id="resp-1"),))
        assert len(rt1.llm_client.attempts) == 1
        assert len(rt1.llm_client.responses) == 1

    assert rt1.llm_client.attempts == ()
    assert rt1.llm_client.responses == ()

    runtime.current = SimpleNamespace(settings=_make_settings("sk-corr-v2"))

    lease_a_entered = asyncio.Event()
    release_lease_a = asyncio.Event()

    async def worker_a() -> None:
        async with resolver.lease("run-1"):
            lease_a_entered.set()
            await release_lease_a.wait()

    task_a = asyncio.create_task(worker_a())
    await close_started.wait()

    async with resolver.lease("run-1"):
        allow_close_finish.set()
        await lease_a_entered.wait()

    release_lease_a.set()
    await task_a

    assert created_clients[0].closed == 1
    assert len(created_clients) == 2
    assert created_clients[1].closed == 0

    async with resolver.lease("run-2"):
        assert len(created_clients) == 3
        assert created_clients[1].closed == 1
        assert created_clients[2].closed == 0

    await resolver.aclose()
    assert all(client.closed == 1 for client in created_clients)

