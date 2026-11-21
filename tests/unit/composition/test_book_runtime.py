from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.adapters.config.settings import AppSettings, LLMSettings
from linguary.application.book_conditions import ConditionsError
from linguary.composition.book_runtime import (
    BookRuntimeResolver,
    BoundTaskQueue,
    ProvenanceClient,
    conditions_payload,
)
from linguary.composition.container import Application
from linguary.composition.runtime_settings import RuntimeGeneration
from linguary.domain import ProcessingTask, TaskIdentity, TaskStatus, TaskType
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


@pytest.mark.asyncio
async def test_book_runtime_resolver_no_race_when_closing_stale_entry_during_concurrent_lease() -> None:
    settings_v1 = _make_settings("sk-v1")
    current_conditions = _make_conditions(settings_v1, revision=1)

    class _Work:
        def get_conditions(self, conditions_id: str):
            return current_conditions if conditions_id == current_conditions.id else None

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=_Work())

    service = SimpleNamespace(uow=_UoW())
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings_v1))

    close_started = asyncio.Event()
    allow_close_finish = asyncio.Event()

    class _SlowCloseClient(_FakeConfiguredClient):
        def __init__(self, llm_settings: LLMSettings, index: int) -> None:
            super().__init__(llm_settings)
            self.index = index

        async def aclose(self) -> None:
            self.closed += 1
            if self.index == 0 and self.closed == 1:
                close_started.set()
                await allow_close_finish.wait()

    created_clients: list[_SlowCloseClient] = []

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        client = _SlowCloseClient(candidate.llm, len(created_clients))
        created_clients.append(client)
        return RuntimeGeneration(
            settings=candidate,
            configured_client=client,
            analysis=SimpleNamespace(language_detector=object()),
            processing=SimpleNamespace(),
            regeneration=SimpleNamespace(),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)

    async with resolver.lease(current_conditions.id, "processing"):
        assert len(created_clients) == 1

    current_conditions = replace(current_conditions, revision=2)

    lease_a_entered = asyncio.Event()
    release_lease_a = asyncio.Event()

    async def worker_a() -> None:
        async with resolver.lease(current_conditions.id, "processing"):
            lease_a_entered.set()
            await release_lease_a.wait()

    task_a = asyncio.create_task(worker_a())
    await close_started.wait()

    async with resolver.lease(current_conditions.id, "processing"):
        allow_close_finish.set()
        await lease_a_entered.wait()

    release_lease_a.set()
    await task_a

    await resolver.aclose()
    assert all(client.closed == 1 for client in created_clients)
    assert resolver._cache == {}


@pytest.mark.asyncio
async def test_book_runtime_resolver_aclose_during_stale_entry_close_leaves_cache_empty() -> None:
    settings_v1 = _make_settings("sk-v1")
    current_conditions = _make_conditions(settings_v1, revision=1)

    class _Work:
        def get_conditions(self, conditions_id: str):
            return current_conditions if conditions_id == current_conditions.id else None

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=_Work())

    service = SimpleNamespace(uow=_UoW())
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings_v1))

    close_started = asyncio.Event()
    allow_close_finish = asyncio.Event()

    class _SlowCloseClient(_FakeConfiguredClient):
        def __init__(self, llm_settings: LLMSettings, index: int) -> None:
            super().__init__(llm_settings)
            self.index = index

        async def aclose(self) -> None:
            self.closed += 1
            if self.index == 0 and self.closed == 1:
                close_started.set()
                await allow_close_finish.wait()

    created_clients: list[_SlowCloseClient] = []

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        client = _SlowCloseClient(candidate.llm, len(created_clients))
        created_clients.append(client)
        return RuntimeGeneration(
            settings=candidate,
            configured_client=client,
            analysis=SimpleNamespace(),
            processing=SimpleNamespace(),
            regeneration=SimpleNamespace(),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)

    async with resolver.lease(current_conditions.id, "processing"):
        pass

    current_conditions = replace(current_conditions, revision=2)

    async def worker() -> None:
        async with resolver.lease(current_conditions.id, "processing"):
            pass

    task = asyncio.create_task(worker())
    await close_started.wait()
    await resolver.aclose()
    assert resolver._cache == {}
    allow_close_finish.set()
    await task

    assert resolver._cache == {}
    assert all(client.closed == 1 for client in created_clients)


@pytest.mark.asyncio
async def test_book_runtime_resolver_resets_provenance_client_contextvars_on_lease_exit() -> None:
    settings = _make_settings("sk-ctx")
    conditions = _make_conditions(settings)
    service = _make_service(conditions)
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    created_provenance: list[ProvenanceClient] = []

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        client = _FakeConfiguredClient(candidate.llm)
        prov = ProvenanceClient(SimpleNamespace(), service, conditions.id)
        created_provenance.append(prov)
        return RuntimeGeneration(
            settings=candidate,
            configured_client=client,
            analysis=SimpleNamespace(_analyze_chapter=SimpleNamespace(_llm_client=prov)),
            processing=SimpleNamespace(_process_chapter=SimpleNamespace(_call=SimpleNamespace(_llm_client=prov))),
            regeneration=SimpleNamespace(_regenerate_block=SimpleNamespace(_call=SimpleNamespace(_llm_client=prov))),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)

    async with resolver.lease(conditions.id, "processing"):
        prov = created_provenance[0]
        prov._attempts.set(("attempt-1",))
        prov._responses.set((SimpleNamespace(id="resp-1"),))
        assert len(prov.attempts) == 1
        assert len(prov.responses) == 1

    assert created_provenance[0].attempts == ()
    assert created_provenance[0].responses == ()

    async with resolver.lease(conditions.id, "processing"):
        assert created_provenance[0].attempts == ()
        assert created_provenance[0].responses == ()


@pytest.mark.asyncio
async def test_book_runtime_resolver_evicts_previous_conditions_of_same_book_and_tracks_worker_and_debug_settings() -> None:
    settings = _make_settings("sk-book")
    cond1 = _make_conditions(settings, revision=1)
    cond2 = _make_conditions(settings, revision=2)
    by_id = {cond1.id: cond1, cond2.id: cond2}

    class _Work:
        def get_conditions(self, conditions_id: str):
            return by_id.get(conditions_id)

    class _UoW:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=_Work())

    service = SimpleNamespace(uow=_UoW())
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    created_clients: list[_FakeConfiguredClient] = []

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        client = _FakeConfiguredClient(candidate.llm)
        created_clients.append(client)
        return RuntimeGeneration(
            settings=candidate,
            configured_client=client,
            analysis=SimpleNamespace(),
            processing=SimpleNamespace(),
            regeneration=SimpleNamespace(),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)

    async with resolver.lease(cond1.id, "analysis"):
        assert len(created_clients) == 1
        assert created_clients[0].closed == 0

    async with resolver.lease(cond2.id, "analysis") as gen_cond2:
        assert len(created_clients) == 2
        assert created_clients[0].closed == 1
        assert created_clients[1].closed == 0

    runtime.current = SimpleNamespace(
        settings=replace(settings, worker=replace(settings.worker, concurrency=4))
    )
    async with resolver.lease(cond2.id, "processing") as gen_concurrency:
        assert gen_concurrency is not gen_cond2
        assert len(created_clients) == 3
        assert created_clients[1].closed == 1
        assert created_clients[2].closed == 0

    runtime.current = SimpleNamespace(
        settings=replace(
            runtime.current.settings,
            llm=replace(runtime.current.settings.llm, debug_llm_payloads=True),
        )
    )
    async with resolver.lease(cond2.id, "processing") as gen_debug:
        assert gen_debug is not gen_concurrency
        assert gen_debug.settings.llm.debug_llm_payloads is True
        assert len(created_clients) == 4
        assert created_clients[2].closed == 1
        assert created_clients[3].closed == 0

    await resolver.aclose()
    assert created_clients[3].closed == 1


@pytest.mark.asyncio
async def test_runtime_resolvers_lease_and_snapshot_use_read_transaction() -> None:
    settings = _make_settings("sk-read-tx")
    conditions = _make_conditions(settings)
    read_calls = 0

    class _Work:
        def get_conditions(self, conditions_id: str):
            return conditions if conditions_id == conditions.id else None

    class _ReadOnlyUoW:
        @contextmanager
        def transaction(self):
            raise AssertionError("write transaction() must not be opened during BookRuntimeResolver.lease")
            yield  # pragma: no cover

        @contextmanager
        def read_transaction(self):
            nonlocal read_calls
            read_calls += 1
            yield SimpleNamespace(work=_Work())

    service = SimpleNamespace(uow=_ReadOnlyUoW())
    runtime = SimpleNamespace(current=SimpleNamespace(settings=settings))

    def factory(candidate: AppSettings, *, conditions=None) -> RuntimeGeneration:
        return RuntimeGeneration(
            settings=candidate,
            configured_client=_FakeConfiguredClient(candidate.llm),
            analysis=SimpleNamespace(),
            processing=SimpleNamespace(),
            regeneration=SimpleNamespace(),
        )

    resolver = BookRuntimeResolver(service, runtime, factory)
    async with resolver.lease(conditions.id, "processing") as gen:
        assert gen.settings.llm.api_key == "sk-read-tx"
    assert read_calls == 1
    await resolver.aclose()


@pytest.mark.asyncio
async def test_bound_task_queue_enqueue_notifies_work_available_after_commit() -> None:
    settings = _make_settings("sk-queue-notify")
    conditions = _make_conditions(settings)
    in_tx = False
    committed_count = 0
    stored_tasks: dict[TaskIdentity, ProcessingTask] = {}
    bindings: dict[tuple[str, str], dict] = {}
    active_conditions = conditions

    class _QueueTasks:
        def find_by_identity(self, identity: TaskIdentity):
            return stored_tasks.get(identity)

        def add(self, task: ProcessingTask) -> None:
            stored_tasks[task.identity] = task

    class _QueueWork:
        def get_binding(self, kind: str, entity_id: str):
            return bindings.get((kind, entity_id))

        def bind(self, kind: str, entity_id: str, *, book_id: str, conditions_id: str) -> None:
            bindings[(kind, entity_id)] = {"book_id": book_id, "conditions_id": conditions_id}

    class _QueueUoW:
        @contextmanager
        def transaction(self):
            nonlocal in_tx, committed_count
            in_tx = True
            try:
                yield SimpleNamespace(tasks=_QueueTasks(), work=_QueueWork())
            except BaseException:
                in_tx = False
                raise
            in_tx = False
            committed_count += 1

    notify_states: list[tuple[bool, int]] = []

    class _SpyQueue:
        def notify_work_available(self) -> None:
            notify_states.append((in_tx, committed_count))

    queue_service = SimpleNamespace(
        uow=_QueueUoW(),
        require=lambda book_id, transaction=None: active_conditions,
    )
    bound = BoundTaskQueue(_SpyQueue(), queue_service, conditions.id, id_factory=lambda: "task-1")
    identity = TaskIdentity(
        book_id="book-1",
        task_type=TaskType.ANALYZE_CHAPTER,
        scope_id="sec-1",
    )

    enqueued = await bound.enqueue(identity)
    assert enqueued.id == "task-1"
    assert enqueued.status is TaskStatus.PENDING
    assert notify_states == [(False, 1)]

    enqueued_again = await bound.enqueue(identity)
    assert enqueued_again.id == "task-1"
    assert notify_states == [(False, 1), (False, 2)]

    stored_tasks[identity] = replace(enqueued, status=TaskStatus.SUCCEEDED, attempts=1)
    completed = await bound.enqueue(identity)
    assert completed.status is TaskStatus.SUCCEEDED
    assert notify_states == [(False, 1), (False, 2)]

    active_conditions = _make_conditions(settings, revision=2)
    with pytest.raises(ConditionsError, match="conditions_revision_conflict"):
        await bound.enqueue(identity)
    assert notify_states == [(False, 1), (False, 2)]

    active_conditions = conditions
    plain_bound = BoundTaskQueue(SimpleNamespace(), queue_service, conditions.id, id_factory=lambda: "task-2")
    plain_task = await plain_bound.enqueue(
        TaskIdentity(book_id="book-1", task_type=TaskType.ANALYZE_CHAPTER, scope_id="sec-2")
    )
    assert plain_task.id == "task-2"



def test_conditions_pin_staged_choice_and_both_pass_models():
    settings = replace(_make_settings(), llm=replace(_make_settings().llm,
        staged_analysis=True, analysis_model='analysis', narrative_analysis_model='story',
        pedagogical_analysis_model='teaching'))
    conditions = _make_conditions(settings)
    assert conditions.payload['staged_analysis'] is True
    assert conditions.payload['models']['analysis_passes'] == {
        'narrative': 'story', 'pedagogical': 'teaching'}
    runtime = SimpleNamespace(current=SimpleNamespace(settings=replace(settings,
        llm=replace(settings.llm, narrative_analysis_model='new-story',
                    pedagogical_analysis_model='new-teaching', staged_analysis=False))))
    pinned = BookRuntimeResolver(_make_service(conditions), runtime, None).settings_for(conditions)
    assert pinned.llm.staged_analysis is True
    assert pinned.llm.effective_analysis_pass_model('narrative') == 'story'
    assert pinned.llm.effective_analysis_pass_model('pedagogical') == 'teaching'


def test_old_v3_conditions_keep_single_pass_without_new_templates_or_settings():
    from linguary.application.book_conditions import payload_digest
    from linguary.domain.book_conditions import thaw
    settings = _make_settings()
    conditions = _make_conditions(settings)
    payload = thaw(conditions.payload)
    payload.pop('staged_analysis', None)
    payload['models'].pop('analysis_passes', None)
    for key in ('staged_analysis', 'narrative_analysis_model', 'pedagogical_analysis_model'):
        payload['llm_parameters'].pop(key, None)
    for key in ('narrative_analysis', 'pedagogical_analysis'):
        payload['prompt_templates'].pop(key, None)
    payload['prompt_sha256'] = payload_digest(payload['prompt_templates'])
    old = replace(conditions, payload=payload)
    current = replace(settings, llm=replace(settings.llm, staged_analysis=True,
        narrative_analysis_model='new-story', pedagogical_analysis_model='new-teaching'))
    pinned = BookRuntimeResolver(_make_service(old),
        SimpleNamespace(current=SimpleNamespace(settings=current)), None).settings_for(old)
    assert pinned.llm.staged_analysis is False
    assert pinned.llm.effective_analysis_pass_model('narrative') == 'gpt-test'
    assert pinned.llm.effective_analysis_pass_model('pedagogical') == 'gpt-test'


def test_conditions_payload_disables_staged_analysis_for_non_v3_profile():
    settings = replace(
        _make_settings(),
        llm=replace(_make_settings().llm, staged_analysis=True),
    )
    book = SimpleNamespace(source_language="en", target_language="ru")
    legacy_payload = conditions_payload(book, settings, 1, "legacy-v1")
    v2_payload = conditions_payload(book, settings, 1, "content-plan-v2")
    v3_payload = conditions_payload(book, settings, 1, "content-plan-v3")
    assert legacy_payload["staged_analysis"] is False
    assert v2_payload["staged_analysis"] is False
    assert v3_payload["staged_analysis"] is True


def test_new_conditions_increment_context_and_prompts_versions_keeping_legacy_v5_compatible():
    from linguary.adapters.config.settings import ReadingSettings
    from linguary.application.book_conditions import (
        A2_CONTENT_PLAN_VERSIONS,
        A2_CONTENT_PLAN_V5_VERSIONS,
        A2_CONTENT_PLAN_V6_VERSIONS,
        payload_digest,
    )
    from linguary.application.context_builder import snapshot_prompt_templates
    from linguary.application.recovery import RecoveryError
    from linguary.application.translation_compatibility import _verify_conditions
    from linguary.composition.book_runtime import effective_content_plan_profile
    from linguary.domain.book_conditions import thaw

    settings = _make_settings()
    book = SimpleNamespace(source_language="en", target_language="ru")

    v3_payload = conditions_payload(book, settings, 1, "content-plan-v3")
    assert v3_payload["versions"] == {
        "runtime": 1,
        "context_builder": 4,
        "validators": 3,
        "schemas": 5,
        "prompts": 6,
    }
    assert v3_payload["prompt_templates"]["analysis_contract_version"] == 5

    legacy_b2_payload = conditions_payload(book, settings, 1, "legacy-v1")
    assert legacy_b2_payload["versions"] == {
        "runtime": 1,
        "context_builder": 1,
        "validators": 1,
        "schemas": 1,
        "prompts": 1,
    }
    assert "analysis_contract_version" not in legacy_b2_payload["prompt_templates"]

    def make_cond(payload: dict) -> BookConditions:
        return BookConditions(
            id=str(uuid4()),
            book_id="book-1",
            revision=1,
            payload=payload,
            created_at=datetime.now(UTC),
        )

    def check_compat(cond: BookConditions, *, expected_profile: str) -> None:
        resolver = BookRuntimeResolver(
            _make_service(cond),
            SimpleNamespace(current=SimpleNamespace(settings=settings)),
            None,
        )
        assert effective_content_plan_profile(cond) == expected_profile
        resolver.settings_for(cond)
        task = ProcessingTask(
            id="task-1",
            book_id="book-1",
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id="sec-1",
            status=TaskStatus.PENDING,
        )
        tx = SimpleNamespace(
            work=SimpleNamespace(
                get_conditions=lambda cid: cond if cid == cond.id else None,
                get_binding=lambda kind, eid: (
                    {"book_id": "book-1", "conditions_id": cond.id, "payload_json": {}}
                    if (kind, eid) == ("task", task.id)
                    else None
                ),
                latest_conditions=lambda bid: cond if bid == "book-1" else None,
            ),
            tasks=SimpleNamespace(get=lambda tid: task if tid == task.id else None),
        )
        _verify_conditions(tx, task, cond)

    # 1. New v6 content-plan-v3 (b2 and a2)
    check_compat(make_cond(v3_payload), expected_profile="content-plan-v3")
    v3_a2_settings = replace(settings, reading=ReadingSettings(reader_level="a2"))
    v3_a2_payload = conditions_payload(book, v3_a2_settings, 1, "content-plan-v3")
    v3_a2_payload["versions"]["validators"] = 4
    check_compat(make_cond(v3_a2_payload), expected_profile="content-plan-v3")

    # 2. Historical v5 content-plan-v3 (b2 and a2)
    v5_b2_payload = thaw(v3_payload)
    v5_b2_payload["versions"] = {
        "runtime": 1,
        "context_builder": 3,
        "validators": 3,
        "schemas": 5,
        "prompts": 5,
    }
    check_compat(make_cond(v5_b2_payload), expected_profile="content-plan-v3")

    v5_a2_payload = thaw(v3_a2_payload)
    v5_a2_payload["versions"] = {
        "runtime": 1,
        "context_builder": 3,
        "validators": 4,
        "schemas": 5,
        "prompts": 5,
    }
    check_compat(make_cond(v5_a2_payload), expected_profile="content-plan-v3")

    # 3. A2 on legacy-v1: new v6, historical v5, historical v3
    legacy_a2_base = conditions_payload(book, v3_a2_settings, 1, "legacy-v1")
    a2_v5_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="legacy-v1",
        validators_version=4,
        name_translation_policy={"version": 1, "strategy": "sound"},
        analysis_contract_version=5,
    )
    for a2_versions in (A2_CONTENT_PLAN_V5_VERSIONS, A2_CONTENT_PLAN_V6_VERSIONS):
        a2_payload = thaw(legacy_a2_base)
        a2_payload["versions"] = dict(a2_versions)
        a2_payload["prompt_templates"] = a2_v5_templates
        a2_payload["prompt_sha256"] = payload_digest(a2_v5_templates)
        check_compat(make_cond(a2_payload), expected_profile="content-plan-v3")

    a2_v3_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="legacy-v1",
        validators_version=4,
        analysis_contract_version=None,
    )
    a2_v3_payload = thaw(legacy_a2_base)
    a2_v3_payload["versions"] = dict(A2_CONTENT_PLAN_VERSIONS)
    a2_v3_payload["prompt_templates"] = a2_v3_templates
    a2_v3_payload["prompt_sha256"] = payload_digest(a2_v3_templates)
    check_compat(make_cond(a2_v3_payload), expected_profile="content-plan-v3")

    # 4. B2 legacy-v1 remains untouched
    check_compat(make_cond(legacy_b2_payload), expected_profile="legacy-v1")

    # 5. Mismatched mixed versions (e.g. context_builder=4 with prompts=5) are rejected
    bad_payload = thaw(v3_payload)
    bad_payload["versions"] = {
        "runtime": 1,
        "context_builder": 4,
        "validators": 3,
        "schemas": 5,
        "prompts": 5,
    }
    bad_cond = make_cond(bad_payload)
    resolver = BookRuntimeResolver(
        _make_service(bad_cond),
        SimpleNamespace(current=SimpleNamespace(settings=settings)),
        None,
    )
    with pytest.raises(ConditionsError, match="conditions_unavailable"):
        resolver.settings_for(bad_cond)
    task = ProcessingTask(
        id="task-bad",
        book_id="book-1",
        task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        scope_id="sec-1",
        status=TaskStatus.PENDING,
    )
    tx = SimpleNamespace(
        work=SimpleNamespace(
            get_conditions=lambda cid: bad_cond if cid == bad_cond.id else None,
            get_binding=lambda kind, eid: (
                {"book_id": "book-1", "conditions_id": bad_cond.id, "payload_json": {}}
                if (kind, eid) == ("task", task.id)
                else None
            ),
            latest_conditions=lambda bid: bad_cond if bid == "book-1" else None,
        ),
        tasks=SimpleNamespace(get=lambda tid: task if tid == task.id else None),
    )
    with pytest.raises(RecoveryError, match="conditions_unavailable"):
        _verify_conditions(tx, task, bad_cond)


