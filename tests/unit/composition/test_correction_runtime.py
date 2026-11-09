from __future__ import annotations

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
