"""Changing settings via HTTP while real task queue and owned clients are running."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import replace
from threading import Event

import httpx
import pytest
from web_scenario import BOOK_ID, SOURCE_TEXT, TRANSLATED_TEXT, start_run, store_book

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.runtime import ExecutorLock
from linguary.composition import build_application
from linguary.domain import BookStatus, TaskIdentity, TaskStatus, TaskType
from linguary.web import create_app


async def wait(event):
    await asyncio.wait_for(event.wait(), 5)


def payload(**changes):
    values = dict(
        provider="openai_compatible", model="new-model", api_key="new-generation-secret",
        base_url="https://new-provider.test/v2", requests_per_minute=None,
        concurrency=1, expected_revision=0,
    )
    values.update(changes)
    return values


class Provider:
    """HTTP call and close barriers; observation records do not expose keys."""

    def __init__(self):
        self.calls = []
        self.entered = defaultdict(asyncio.Event)
        self.release = defaultdict(asyncio.Event)
        self.blocked = {0}
        self.handles = []
        self.closed = defaultdict(asyncio.Event)
        self.clock = lambda: 0.0
        self.real_http = httpx.AsyncClient

    async def respond(self, request):
        body = json.loads(request.content)
        model = body["model"]
        expected_key = "old-generation-secret" if model == "old-model" else "new-generation-secret"
        if request.headers["Authorization"] != f"Bearer {expected_key}":
            raise AssertionError("Generation sent an invalid authorization key")
        index = len(self.calls)
        self.calls.append((str(request.url), model, self.clock()))
        self.entered[index].set()
        if index in self.blocked:
            await self.release[index].wait()
        return httpx.Response(200, json={
            "model": model,
            "choices": [{"message": {"content": json.dumps({
                "blocks": [{"source_block_id": "b1", "treatment": "TRANSLATE",
                            "translated_text": TRANSLATED_TEXT, "glossary_hints": []}],
                "language_notes": [],
            })}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.007},
        })

    def install(self, monkeypatch):
        provider = self

        class OwnedHTTP(self.real_http):
            async def aclose(self):
                await super().aclose()
                provider.closed[provider.handles.index(self)].set()

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(self.respond)
            client = OwnedHTTP(*args, **kwargs)
            self.handles.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.fixture
def provider(monkeypatch):
    instance = Provider()
    instance.install(monkeypatch)
    return instance


def build(tmp_path, *, concurrency=1, rpm=None, model="old-model", key="old-generation-secret"):
    settings = AppSettings(
        llm=LLMSettings(
            model=model, api_key=key, base_url="https://old-provider.test/v1",
            requests_per_minute=rpm, max_unit_source_chars=30, max_attempts=1,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    return build_application(settings=replace(
        settings, worker=replace(settings.worker, concurrency=concurrency),
    ))


def observe_tasks(app, monkeypatch):
    completed = defaultdict(asyncio.Event)
    outcomes = {}
    original = app.task_queue.save

    async def save(task):
        await original(task)
        if task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
            outcomes[task.id] = task
            completed[task.id].set()

    monkeypatch.setattr(app.task_queue, "save", save)
    return completed, outcomes


async def enqueue(app, scenario, index):
    return await app.task_queue.enqueue(TaskIdentity(
        book_id=BOOK_ID, task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        scope_id=scenario.section(index).id,
    ))


@asynccontextmanager
async def working_web(app, provider):
    web = create_app(application=app, run_background_worker=True)
    try:
        async with web.router.lifespan_context(web):
            async with provider.real_http(
                transport=httpx.ASGITransport(app=web), base_url=app.settings.server.base_url,
            ) as client:
                yield client
    finally:
        await app.aclose()


async def test_owned_generations_use_new_url_model_key_and_close_after_release(
    tmp_path, monkeypatch, provider,
):
    app = build(tmp_path)
    completed, outcomes = observe_tasks(app, monkeypatch)
    scenario = store_book(app, status=BookStatus.PROCESSING, chapters=((SOURCE_TEXT,) * 2,) * 2)
    start_run(app, scenario)
    async with working_web(app, provider) as web:
        old = await enqueue(app, scenario, 0)
        await wait(provider.entered[0])
        saved = await web.post("/system/config", json=payload())
        assert saved.status_code == 200
        assert len(provider.handles) == 2
        assert not provider.handles[0].is_closed
        new = await enqueue(app, scenario, 1)
        provider.release[0].set()
        await wait(completed[old.id])
        await wait(completed[new.id])
        await wait(provider.closed[0])
        assert not provider.handles[1].is_closed
        assert [call[:2] for call in provider.calls] == [
            ("https://old-provider.test/v1/chat/completions", "old-model"),
            ("https://old-provider.test/v1/chat/completions", "old-model"),
            ("https://new-provider.test/v2/chat/completions", "new-model"),
            ("https://new-provider.test/v2/chat/completions", "new-model"),
        ]
        assert all(task.status is TaskStatus.SUCCEEDED for task in outcomes.values())
        with app.unit_of_work.transaction() as uow:
            logs = uow.llm_calls.list_for_book(BOOK_ID)
        assert [log.model for log in logs] == ["old-model", "old-model", "new-model", "new-model"]
        assert all(str(log.cost) == "0.007" for log in logs)
    assert all(handle.is_closed for handle in provider.handles)


async def test_save_during_active_task_preserves_old_and_new_models(tmp_path, monkeypatch, provider):
    app = build(tmp_path, concurrency=2)
    completed, outcomes = observe_tasks(app, monkeypatch)
    scenario = store_book(app, status=BookStatus.PROCESSING, chapters=((SOURCE_TEXT,),) * 2)
    start_run(app, scenario)
    async with working_web(app, provider) as web:
        old = await enqueue(app, scenario, 0)
        await wait(provider.entered[0])
        assert (await web.post("/system/config", json=payload(concurrency=2))).status_code == 200
        new = await enqueue(app, scenario, 1)
        await wait(completed[new.id])
        assert not completed[old.id].is_set()
        assert not provider.handles[0].is_closed
        provider.release[0].set()
        await wait(completed[old.id])
        assert [call[1] for call in provider.calls] == ["old-model", "new-model"]
        assert all(task.status is TaskStatus.SUCCEEDED for task in outcomes.values())


async def test_resize_during_processing_finishes_all_tasks(tmp_path, monkeypatch, provider):
    provider.blocked = {0, 1, 2, 3}
    app = build(tmp_path, concurrency=2)
    completed, outcomes = observe_tasks(app, monkeypatch)
    scenario = store_book(app, status=BookStatus.PROCESSING, chapters=((SOURCE_TEXT,),) * 4)
    start_run(app, scenario)
    async with working_web(app, provider) as web:
        tasks = [await enqueue(app, scenario, index) for index in range(4)]
        await wait(provider.entered[1])
        assert (await web.post("/system/config", json=payload())).status_code == 200
        assert len(provider.calls) == 2
        provider.release[0].set()
        await wait(completed[tasks[0].id])
        assert len(provider.calls) == 2
        provider.release[1].set()
        await wait(provider.entered[2])
        assert (await web.post("/system/config", json=payload(concurrency=3, expected_revision=1))).status_code == 200
        await wait(provider.entered[3])
        provider.release[2].set()
        provider.release[3].set()
        for task in tasks:
            await wait(completed[task.id])
        assert len(outcomes) == 4
        assert all(task.status is TaskStatus.SUCCEEDED for task in outcomes.values())
        assert len(provider.calls) == 4
        assert app.worker_pool.concurrency == 3


async def test_save_does_not_rerun_recovery_or_drop_executor_lock(tmp_path, monkeypatch, provider):
    app = build(tmp_path)
    completed, outcomes = observe_tasks(app, monkeypatch)
    recovery_calls = []
    original = app.runner.recover_interrupted

    async def recover():
        recovery_calls.append(True)
        return await original()

    monkeypatch.setattr(app.runner, "recover_interrupted", recover)
    stable = (app.task_queue, app.runner, app.worker_pool, app.rate_limiter)
    scenario = store_book(app, status=BookStatus.PROCESSING, chapters=((SOURCE_TEXT,),))
    start_run(app, scenario)
    async with working_web(app, provider) as web:
        task = await enqueue(app, scenario, 0)
        await wait(provider.entered[0])
        for revision in range(2):
            assert (await web.post("/system/config", json=payload(expected_revision=revision))).status_code == 200
            assert stable == (app.task_queue, app.runner, app.worker_pool, app.rate_limiter)
            other = ExecutorLock(app.paths.executor_lock)
            try:
                assert other.join() is False
            finally:
                other.leave()
            with app.unit_of_work.transaction() as uow:
                assert uow.tasks.get(task.id).status is TaskStatus.RUNNING
        assert recovery_calls == [True]
        provider.release[0].set()
        await wait(completed[task.id])
        assert outcomes[task.id].status is TaskStatus.SUCCEEDED
    other = ExecutorLock(app.paths.executor_lock)
    try:
        assert other.join() is True
    finally:
        other.leave()


async def test_new_generation_shares_rate_limit_with_old(tmp_path, monkeypatch, provider):
    app = build(tmp_path, concurrency=2, rpm=60)
    completed, outcomes = observe_tasks(app, monkeypatch)
    now = [0.0]
    waiting = asyncio.Event()
    advance = asyncio.Event()
    delays = []

    async def sleep(seconds):
        delays.append(seconds)
        waiting.set()
        await advance.wait()

    monkeypatch.setattr(app.rate_limiter, "_clock", lambda: now[0])
    monkeypatch.setattr(app.rate_limiter, "_sleep", sleep)
    provider.clock = lambda: now[0]
    scenario = store_book(app, status=BookStatus.PROCESSING, chapters=((SOURCE_TEXT,),) * 2)
    start_run(app, scenario)
    async with working_web(app, provider) as web:
        old = await enqueue(app, scenario, 0)
        await wait(provider.entered[0])
        assert (await web.post("/system/config", json=payload(concurrency=2, requests_per_minute=60))).status_code == 200
        new = await enqueue(app, scenario, 1)
        await wait(waiting)
        assert delays == [1.0]
        app.rate_limiter.hold(3)
        assert len(provider.calls) == 1
        assert not provider.handles[0].is_closed
        now[0] = 3
        advance.set()
        await wait(completed[new.id])
        provider.release[0].set()
        await wait(completed[old.id])
        assert [(call[1], call[2]) for call in provider.calls] == [("old-model", 0), ("new-model", 3)]
        assert all(task.status is TaskStatus.SUCCEEDED for task in outcomes.values())


@pytest.mark.parametrize("transaction_fails", [False, True])
async def test_shutdown_with_cancelled_post_keeps_database_runtime_consistent(
    tmp_path, monkeypatch, provider, transaction_fails,
):
    app = build(tmp_path)
    started = asyncio.Event()
    close_started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    unhandled = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    original_save = app.settings_store.save
    original_begin = app.system_settings.begin_close
    disposed = []
    original_dispose = app._dispose

    def save(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if transaction_fails:
            raise RuntimeError("transaction failed")
        return original_save(*args, **kwargs)

    def begin_close():
        original_begin()
        close_started.set()

    def dispose():
        disposed.append(True)
        original_dispose()

    monkeypatch.setattr(app.settings_store, "save", save)
    monkeypatch.setattr(app.system_settings, "begin_close", begin_close)
    object.__setattr__(app, "_dispose", dispose)
    monkeypatch.setattr("linguary.web.app.build_application", lambda: app)
    web = create_app(run_background_worker=True)
    lifespan = web.router.lifespan_context(web)
    await lifespan.__aenter__()
    try:
        async with provider.real_http(
            transport=httpx.ASGITransport(app=web), base_url=app.settings.server.base_url,
        ) as client:
            observer = asyncio.create_task(client.post("/system/config", json=payload()))
            await wait(started)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            closing = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await wait(close_started)
            assert not closing.done() and not disposed
            assert len(provider.handles) == 2
            assert not any(handle.is_closed for handle in provider.handles)
            release.set()
            await asyncio.wait_for(closing, 5)
        if transaction_fails:
            assert app.settings_store.load() is None
            assert app.runtime_settings.revision == 0
            assert app.settings.llm.model == "old-model"
            assert len(await app.system_settings.drain()) == 1
        else:
            assert app.runtime_settings.revision == app.settings_store.load().revision == 1
            assert app.settings.llm.model == "new-model"
        assert all(handle.is_closed for handle in provider.handles)
        assert disposed == [True]
        assert not app.system_settings._operations and not app.runtime_settings._close_tasks
        assert not app.worker_pool._workers and not unhandled
    finally:
        release.set()
        await app.aclose()
        loop.set_exception_handler(previous_handler)
