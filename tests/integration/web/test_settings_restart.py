"""Сохранённые настройки и сброс при повторном запуске на том же каталоге данных."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest
from web_scenario import BOOK_ID, SOURCE_TEXT, TRANSLATED_TEXT, start_run, store_book

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.composition import build_application
from linguary.domain import BookStatus, TaskIdentity, TaskStatus, TaskType
from linguary.web import create_app


def baseline(tmp_path, *, model="startup-model", key="startup-secret", rpm=10, concurrency=1):
    settings = AppSettings(
        llm=LLMSettings(
            model=model, api_key=key, base_url="https://startup-provider.test/v1",
            requests_per_minute=rpm, timeout_seconds=17,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    return replace(settings, worker=replace(settings.worker, concurrency=concurrency))


def payload(**changes):
    values = dict(
        provider="openai_compatible", model="saved-model", api_key="saved-secret",
        base_url="https://saved-provider.test/v2", requests_per_minute=30,
        concurrency=3, expected_revision=0,
    )
    values.update(changes)
    return values


@asynccontextmanager
async def running(settings, *, llm_client=None):
    app = build_application(settings=settings, llm_client=llm_client or FakeLLMClient())
    web = create_app(application=app, run_background_worker=True)
    try:
        async with web.router.lifespan_context(web):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=web), base_url=app.settings.server.base_url,
            ) as client:
                yield app, client
    finally:
        await app.aclose()


def assert_key_is_private(response, key):
    assert (key not in response.text) is True
    assert "api_key" not in response.json()["llm"]


async def test_persistent_settings_survive_application_restart(tmp_path):
    async with running(baseline(tmp_path)) as (first, web):
        response = await web.post("/system/config", json=payload())
        assert response.status_code == 200
        assert_key_is_private(response, "saved-secret")
        persisted = response.json()
        first_queue = first.task_queue
        database_path = first.paths.database_path
    different_startup = baseline(
        tmp_path, model="different-startup-model", key="different-startup-secret",
        rpm=7, concurrency=2,
    )
    async with running(different_startup) as (restarted, web):
        response = await web.get("/system/config")
        assert response.status_code == 200
        assert response.json() == persisted
        assert_key_is_private(response, "saved-secret")
        assert restarted.paths.database_path == database_path
        assert restarted.task_queue is not first_queue
        assert (restarted.settings.llm.api_key == "saved-secret") is True
        assert restarted.settings.llm.timeout_seconds == different_startup.llm.timeout_seconds
        assert restarted.worker_pool.concurrency == 3
        assert restarted.rate_limiter.requests_per_minute == 30
        assert restarted.runtime_settings.revision == restarted.settings_store.load().revision == 1


@pytest.mark.parametrize("configured_baseline", [False, True])
async def test_reset_survives_restart_and_reenables_baseline(tmp_path, configured_baseline):
    startup_key = "startup-secret" if configured_baseline else None
    async with running(baseline(tmp_path, key=startup_key)) as (first, web):
        assert (await web.post("/system/config", json=payload())).status_code == 200
        reset = await web.post("/system/config/reset", json={"expected_revision": 1})
        assert reset.status_code == 200
        assert reset.json()["revision"] == 2
        assert reset.json()["llm"]["model"] == "startup-model"
        assert reset.json()["llm"]["is_ready"] is configured_baseline
        assert first.settings_store.load().overrides is None
        assert first.settings_store.load().api_key is None
    new_startup_key = "restarted-baseline-secret" if configured_baseline else None
    new_startup = baseline(
        tmp_path, model="restarted-baseline-model", key=new_startup_key,
        rpm=7, concurrency=2,
    )
    async with running(new_startup) as (restarted, web):
        response = await web.get("/system/config")
        assert response.status_code == 200
        data = response.json()
        assert data["revision"] == 2
        assert data["llm"]["model"] == "restarted-baseline-model"
        assert data["llm"]["requests_per_minute"] == 7
        assert data["worker"]["concurrency"] == 2
        assert data["llm"]["is_ready"] is configured_baseline
        assert data["llm"]["has_api_key"] is configured_baseline
        assert restarted.settings_store.load().overrides is None
        assert restarted.settings_store.load().api_key is None
        assert ("saved-secret" not in response.text) is True
        if new_startup_key:
            assert_key_is_private(response, new_startup_key)
            assert (restarted.settings.llm.api_key == new_startup_key) is True
        else:
            assert restarted.settings.llm.api_key is None


async def test_unconfigured_llm_can_be_configured_without_restart(tmp_path, monkeypatch):
    fake = FakeLLMClient.replying([FakeReply(content=json.dumps({
        "blocks": [{"source_block_id": "b1", "treatment": "TRANSLATE",
                    "translated_text": TRANSLATED_TEXT, "glossary_hints": []}],
        "language_notes": [],
    }))])
    async with running(baseline(tmp_path, model="", key=None, rpm=None), llm_client=fake) as (app, web):
        before = await web.get("/system/config")
        assert before.json()["llm"]["is_ready"] is False
        stable_queue = app.task_queue
        stable_runner = app.runner
        completed = asyncio.Event()
        original_save = app.task_queue.save
        outcomes = []

        async def save(task):
            await original_save(task)
            outcomes.append(task)
            completed.set()

        monkeypatch.setattr(app.task_queue, "save", save)
        configured = await web.post("/system/config", json=payload(requests_per_minute=None))
        assert configured.status_code == 200
        assert configured.json()["llm"]["is_ready"] is True
        assert_key_is_private(configured, "saved-secret")
        assert fake.call_count == 0
        scenario = store_book(app, status=BookStatus.PROCESSING, chapters=((SOURCE_TEXT,),))
        start_run(app, scenario)
        task = await app.task_queue.enqueue(TaskIdentity(
            book_id=BOOK_ID, task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id=scenario.section().id,
        ))
        await asyncio.wait_for(completed.wait(), 5)
        assert app.task_queue is stable_queue and app.runner is stable_runner
        assert outcomes[0].id == task.id and outcomes[0].status is TaskStatus.SUCCEEDED
        assert fake.call_count == 1
        with app.unit_of_work.transaction() as uow:
            (entry,) = uow.llm_calls.list_for_book(BOOK_ID)
        assert entry.success and entry.model == "fake-model"
        assert app.settings.llm.model == "saved-model"


async def test_large_positive_rate_survives_same_database_restart(tmp_path):
    rate = 10**400
    async with running(baseline(tmp_path)) as (first, web):
        response = await web.post("/system/config", json=payload(requests_per_minute=rate))
        assert response.status_code == 200
        persisted = response.json()
        database_path = first.paths.database_path
    async with running(baseline(tmp_path, rpm=7)) as (restarted, web):
        response = await web.get("/system/config")
        assert response.status_code == 200
        assert response.json() == persisted
        assert restarted.paths.database_path == database_path
        assert restarted.settings.llm.requests_per_minute == rate
        assert restarted.rate_limiter.requests_per_minute == rate
        assert restarted.settings_store.load().overrides["requests_per_minute"] == rate
        assert restarted.runtime_settings.revision == restarted.settings_store.load().revision == 1
        await restarted.rate_limiter.acquire()
        await restarted.rate_limiter.acquire()
