from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.adapters.config.settings import AppSettings, LLMSettings
from linguary.composition.book_runtime import conditions_payload
from linguary.composition.correction_runtime import (
    CORRECTION_POLICY_VERSION_V3,
    CORRECTION_PROMPT_V2,
    CORRECTION_REVIEW_PROMPT,
    CORRECTION_VERSIONS_V3,
    CorrectionRuntimeResolver,
    correction_snapshot,
)
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


@pytest.mark.asyncio
async def test_runtime_resolvers_lease_and_snapshot_use_read_transaction() -> None:
    settings = _make_settings("sk-corr-read-tx")
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
    read_calls = 0

    class _Runs:
        def get(self, target_run_id: str):
            return run if target_run_id == run_id else None

    class _Work:
        def get_binding(self, kind: str, entity_id: str):
            if kind == "run" and entity_id == run_id:
                return binding
            return None

    class _ReadOnlyUoW:
        @contextmanager
        def transaction(self):
            raise AssertionError("write transaction() must not be opened for read-only snapshot_for/lease")
            yield  # pragma: no cover

        @contextmanager
        def read_transaction(self):
            nonlocal read_calls
            read_calls += 1
            yield SimpleNamespace(runs=_Runs(), work=_Work())

    service = SimpleNamespace(
        uow=_ReadOnlyUoW(),
        require=lambda book_id, transaction=None: conditions,
    )
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))
    resolver = CorrectionRuntimeResolver(
        service,
        runtime,
        limiter=SimpleNamespace(),
        client_factory=_FakeConfiguredClient,
    )

    loaded_snapshot = resolver.snapshot_for(run_id)
    assert loaded_snapshot == snapshot
    assert read_calls == 1

    async with resolver.lease(run_id) as leased:
        assert leased.conditions_id == conditions.id
        assert leased.snapshot == snapshot
    assert read_calls == 2

    await resolver.aclose()


def test_correction_prompt_v2_covers_numbers_calques_notes_and_boundary_consistency() -> None:
    prompt = CORRECTION_PROMPT_V2
    for expected in (
        "two dozen",
        "две дюжины",
        "два десятка",
        "смысловые кальки",
        "ложных друзей",
        "фактическую точность заметок",
        "language_notes",
        "glossary_hints.note",
        "грамматическое согласование",
        "грамматический род животных",
        "«ты/вы»",
        "стыки соседних блоков",
        "пунктуации прямой речи",
    ):
        assert expected in prompt

    assert correction_snapshot(_make_settings()).prompt == CORRECTION_PROMPT_V2
    assert "условиям книги" not in prompt
    assert "условия книги" not in prompt


def test_correction_prompt_v2_and_a2_dialogue_rules_sync_post_speech_narrative_guidance() -> None:
    from linguary.application.reader_text import _A2_DIALOGUE_AND_CALQUE_RULES

    for text in (CORRECTION_PROMPT_V2, _A2_DIALOGUE_AND_CALQUE_RULES):
        assert "— Реплика!\\nАвторское предложение." in text
        assert "запрещено вводить самостоятельное авторское предложение через «— »" in text
        assert "— Реплика 1.\\nАвторское предложение.\\n— Реплика 2." in text
        assert "Авторское предложение. — Реплика 2." in text
        assert "кавычки-ёлочки с начала абзаца" in text


@pytest.mark.asyncio
async def test_correction_runtime_resolver_operational_overrides_legacy_route_and_fingerprint() -> None:
    from dataclasses import replace
    from linguary.application.book_conditions import ConditionsError
    from linguary.application.correction_dto import parse_correction_snapshot

    pinned_settings = AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="gpt-pinned",
            api_key="sk-corr-1",
            proxy_mode="direct",
            temperature=0.1,
            max_output_tokens=4096,
            timeout_seconds=180.0,
            max_attempts=2,
            input_price_per_mtok=0.5,
            output_price_per_mtok=1.5,
        )
    )
    book = SimpleNamespace(source_language="en", target_language="ru")
    conditions = BookConditions(
        id=str(uuid4()),
        book_id="book-1",
        revision=1,
        payload=conditions_payload(book, pinned_settings, 1, "content-plan-v3"),
        created_at=datetime.now(UTC),
    )
    snapshot = correction_snapshot(pinned_settings)
    # Simulate an old SQLite correction snapshot with legacy proxy.environment and route_id.
    legacy_payload = snapshot.to_payload()
    legacy_payload["route"] = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com/v1",
        "proxy": {"mode": "auto", "environment": {"HTTPS_PROXY": "http://old-proxy.example:8080"}},
        "route_id": "legacy-route-id",
    }
    legacy_snapshot = parse_correction_snapshot(legacy_payload)

    run_id = "run-wp5"
    run = SimpleNamespace(id=run_id, book_id="book-1", status=ProcessingRunStatus.ACTIVE)
    binding = {
        "book_id": "book-1",
        "conditions_id": conditions.id,
        "payload_json": {"schema_version": 1, "correction": legacy_snapshot.to_payload()},
    }

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(
                runs=SimpleNamespace(get=lambda rid: run if rid == run_id else None),
                work=SimpleNamespace(get_binding=lambda kind, eid: binding if (kind, eid) == ("run", run_id) else None),
            )

    updated_settings = replace(
        pinned_settings,
        llm=replace(
            pinned_settings.llm,
            model="gpt-new-ignored",
            temperature=0.8,
            proxy_mode="custom",
            proxy_url="http://new-proxy.example:3128",
            timeout_seconds=900.0,
            requests_per_minute=45,
            max_attempts=5,
            debug_llm_payloads=True,
            input_price_per_mtok=2.0,
            output_price_per_mtok=6.0,
        ),
    )
    service = SimpleNamespace(uow=_UoW(), require=lambda book_id, transaction=None: conditions)
    runtime = SimpleNamespace(current=SimpleNamespace(settings=updated_settings))
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

    # No TypeError from duplicate kwargs; operational settings come from current runtime.
    resolved = resolver.settings_for(legacy_snapshot)
    assert resolved.llm.model == "gpt-pinned"
    assert resolved.llm.temperature == 0.1
    assert resolved.llm.max_output_tokens == 4096
    assert resolved.llm.proxy_mode == "custom"
    assert resolved.llm.proxy_url == "http://new-proxy.example:3128"
    assert resolved.llm.timeout_seconds == 900.0
    assert resolved.llm.requests_per_minute == 45
    assert resolved.llm.max_attempts == 5
    assert resolved.llm.debug_llm_payloads is True
    assert resolved.llm.input_price_per_mtok == 2.0
    assert resolved.llm.output_price_per_mtok == 6.0

    # Changing timeout_seconds / max_attempts / proxy_mode / pricing retires cached client via _runtime_fingerprint.
    async with resolver.lease(run_id) as rt1:
        assert len(created_clients) == 1
        assert rt1.settings.llm.timeout_seconds == 900.0

    runtime.current = SimpleNamespace(
        settings=replace(updated_settings, llm=replace(updated_settings.llm, timeout_seconds=1200.0))
    )
    async with resolver.lease(run_id) as rt2:
        assert rt2 is not rt1
        assert len(created_clients) == 2
        assert created_clients[0].closed == 1
        assert rt2.settings.llm.timeout_seconds == 1200.0

    fp_base = resolver._runtime_fingerprint(rt2.settings)
    for mutated_llm in (
        replace(rt2.settings.llm, max_attempts=7),
        replace(rt2.settings.llm, proxy_mode="direct", proxy_url=None),
        replace(rt2.settings.llm, input_price_per_mtok=3.0, output_price_per_mtok=9.0),
    ):
        assert resolver._runtime_fingerprint(replace(rt2.settings, llm=mutated_llm)) != fp_base

    # Changing base_url or provider still raises ConditionsError('route_unavailable').
    runtime.current = SimpleNamespace(
        settings=replace(updated_settings, llm=replace(updated_settings.llm, base_url="https://other.example/v1"))
    )
    with pytest.raises(ConditionsError, match="route_unavailable"):
        resolver.settings_for(legacy_snapshot)

    runtime.current = SimpleNamespace(
        settings=replace(updated_settings, llm=replace(updated_settings.llm, provider="other_provider"))
    )
    with pytest.raises(ConditionsError, match="route_unavailable"):
        resolver.settings_for(legacy_snapshot)

    await resolver.aclose()
    assert created_clients[1].closed == 1


def test_correction_prompts_sync_orthography_warnings_and_review_criteria() -> None:
    for expected in (
        "гиперкоррекцию «ё»",
        "починённый",
        "починенный",
        "неестественный порядок слов",
        "в самый тот день",
        "в тот самый день",
        "validator_warnings",
    ):
        assert expected in CORRECTION_PROMPT_V2
    assert "условиям книги" not in CORRECTION_PROMPT_V2
    assert "условия книги" not in CORRECTION_PROMPT_V2

    for expected in (
        "candidate_id",
        "confirmed",
        "needs_human",
        "two dozen",
        "две дюжины",
        "гиперкоррекцию «ё»",
        "неестественный порядок слов",
        "language_notes",
        "glossary_hints.note",
        "грамматического рода животных",
        "«ты/вы»",
        "— Реплика!\\nАвторское предложение.",
        "— Реплика 1.\\nАвторское предложение.\\n— Реплика 2.",
        "Авторское предложение. — Реплика 2.",
        "кавычек-ёлочек с начала абзаца",
        "changes",
    ):
        assert expected in CORRECTION_REVIEW_PROMPT
    assert "proposal_sha256" not in CORRECTION_REVIEW_PROMPT
    assert CORRECTION_POLICY_VERSION_V3 == "chapter-correction-v3"
    assert CORRECTION_VERSIONS_V3 == {
        "context": 2,
        "schema": 2,
        "validation": 3,
        "grouping": 1,
        "review_schema": 1,
    }




