from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.adapters.config.settings import AppSettings, LLMSettings
from linguary.composition.book_runtime import BookRuntimeResolver, conditions_payload
from linguary.composition.container import Application
from linguary.composition.runtime_settings import RuntimeGeneration
from linguary.domain.book_conditions import BookConditions


class _FakeConfiguredClient:
    def __init__(self, llm_settings: LLMSettings) -> None:
        self.llm_settings = llm_settings
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def _make_settings(api_key: str = "sk-test-1") -> AppSettings:
    return AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="gpt-test",
            api_key=api_key,
            proxy_mode="direct",
        )
    )


def _make_conditions(settings: AppSettings, *, revision: int = 1) -> BookConditions:
    book = SimpleNamespace(source_language="en", target_language="ru")
    payload = conditions_payload(book, settings, revision, "content-plan-v3")
    return BookConditions(
        id=str(uuid4()),
        book_id="book-1",
        revision=revision,
        payload=payload,
        created_at=datetime.now(UTC),
    )


def _make_service(conditions: BookConditions):
    class _Work:
        def get_conditions(self, conditions_id: str):
            return conditions if conditions_id == conditions.id else None

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=_Work())

    return SimpleNamespace(uow=_UoW())


@pytest.mark.asyncio
async def test_book_runtime_resolver_reuses_client_across_leases_and_closes_on_aclose() -> None:
    settings = _make_settings("sk-initial")
    conditions = _make_conditions(settings)
    service = _make_service(conditions)
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    created_clients: list[_FakeConfiguredClient] = []

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        client = _FakeConfiguredClient(candidate.llm)
        created_clients.append(client)
        detector = object()
        return RuntimeGeneration(
            settings=candidate,
            configured_client=client,
            analysis=SimpleNamespace(language_detector=detector),
            processing=SimpleNamespace(),
            regeneration=SimpleNamespace(),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)

    async with resolver.lease(conditions.id, "analysis") as gen1:
        assert len(created_clients) == 1
        assert created_clients[0].closed == 0

    assert created_clients[0].closed == 0

    async with resolver.lease(conditions.id, "processing") as gen2:
        assert gen2 is gen1
        assert gen2.analysis.language_detector is gen1.analysis.language_detector
        assert len(created_clients) == 1
        assert created_clients[0].closed == 0

    assert created_clients[0].closed == 0

    await resolver.aclose()
    assert created_clients[0].closed == 1


@pytest.mark.asyncio
async def test_book_runtime_resolver_retires_old_generation_after_active_lease_finishes() -> None:
    settings = _make_settings("sk-v1")
    conditions = _make_conditions(settings)
    service = _make_service(conditions)
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    created_clients: list[_FakeConfiguredClient] = []

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        client = _FakeConfiguredClient(candidate.llm)
        created_clients.append(client)
        return RuntimeGeneration(
            settings=candidate,
            configured_client=client,
            analysis=SimpleNamespace(language_detector=object()),
            processing=SimpleNamespace(),
            regeneration=SimpleNamespace(),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)

    async with resolver.lease(conditions.id, "processing") as gen1:
        assert len(created_clients) == 1
        runtime.current = SimpleNamespace(settings=_make_settings("sk-v2"))

        async with resolver.lease(conditions.id, "processing") as gen2:
            assert gen2 is not gen1
            assert len(created_clients) == 2
            assert created_clients[0].closed == 0
            assert created_clients[1].closed == 0

        assert created_clients[0].closed == 0
        assert created_clients[1].closed == 0

    assert created_clients[0].closed == 1
    assert created_clients[1].closed == 0

    closed_order: list[str] = []

    async def close_system() -> None:
        closed_order.append("system")

    async def close_runtime() -> None:
        closed_order.append("runtime")

    async def close_quality() -> None:
        closed_order.append("quality")

    async def close_correction() -> None:
        closed_order.append("correction")

    app_stub = SimpleNamespace(
        system_settings=SimpleNamespace(aclose=close_system),
        runtime_settings=SimpleNamespace(aclose=close_runtime),
        book_runtime=resolver,
        correction=SimpleNamespace(resolver=SimpleNamespace(aclose=close_correction)),
        _close_quality=close_quality,
        dispose=lambda: closed_order.append("dispose"),
    )
    await Application._close(app_stub)

    assert created_clients[1].closed == 1
    assert "correction" in closed_order
