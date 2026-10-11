"""Атомарное сохранение, сброс и отмена наблюдателя операции."""
import asyncio
from threading import Event

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.system_settings import SystemSettingsUpdate
from linguary.adapters.db.system_settings import SettingsRevisionConflict, SettingsStoreError
from linguary.adapters.llm import FakeLLMClient
from linguary.composition import build_application
from linguary.composition.system_settings import SettingsClosingError, SettingsNotAppliedError


def update(**values):
    fields = dict(provider='openai_compatible', model='new', base_url='http://localhost:8080/v1', requests_per_minute=30, concurrency=3)
    fields.update(values)
    return SystemSettingsUpdate(**fields)


@pytest.fixture
def application(tmp_path):
    app = build_application(settings=AppSettings(llm=LLMSettings(model='initial', api_key='initial-test-key'), storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=FakeLLMClient())
    yield app
    app.dispose()


async def test_save_prepares_before_commit(application, monkeypatch):
    app = application
    prepared = []
    original_prepare, original_save = app.runtime_settings.prepare, app.settings_store.save
    def prepare(settings):
        candidate = original_prepare(settings)
        prepared.append(candidate)
        return candidate
    def save(*args, **kwargs):
        assert prepared and app.settings.llm.model == 'initial'
        return original_save(*args, **kwargs)
    monkeypatch.setattr(app.runtime_settings, 'prepare', prepare)
    monkeypatch.setattr(app.settings_store, 'save', save)
    await app.system_settings.save(update(), expected_revision=0)
    assert app.runtime_settings.current is prepared[0]
    await app.aclose()


async def test_prepare_error_preserves_database_and_runtime(application, monkeypatch):
    app = application
    def fail(settings):
        raise RuntimeError('private-test-key')
    monkeypatch.setattr(app.runtime_settings, 'prepare', fail)
    with pytest.raises(SettingsNotAppliedError) as caught:
        await app.system_settings.save(update(), expected_revision=0)
    assert 'private-test-key' not in str(caught.value)
    assert app.settings_store.load() is None and app.settings.llm.model == 'initial'
    await app.aclose()


async def test_commit_error_discards_candidate(application, monkeypatch):
    app = application
    candidates, closed = [], []
    original = app.runtime_settings.prepare
    def prepare(settings):
        candidate = original(settings)
        async def close():
            closed.append(candidate)
        monkeypatch.setattr(candidate.configured_client, 'aclose', close)
        candidates.append(candidate)
        return candidate
    def fail(*args, **kwargs):
        raise SettingsStoreError('write failed')
    monkeypatch.setattr(app.runtime_settings, 'prepare', prepare)
    monkeypatch.setattr(app.settings_store, 'save', fail)
    with pytest.raises(SettingsNotAppliedError):
        await app.system_settings.save(update(), expected_revision=0)
    assert closed == candidates
    assert app.settings_store.load() is None and app.settings.llm.model == 'initial'
    await app.aclose()


async def test_success_publishes_settings_limiter_and_pool(application):
    app = application
    await app.system_settings.save(update(), expected_revision=0)
    saved = app.settings_store.load()
    assert saved.revision == app.runtime_settings.revision == 1
    assert saved.overrides['model'] == app.settings.llm.model == 'new'
    assert app.rate_limiter.requests_per_minute == 30
    assert app.worker_pool.concurrency == app.settings.worker.concurrency == 3
    assert saved.api_key == 'initial-test-key'
    await app.aclose()


async def test_reset_uses_startup_baseline(application):
    app = application
    await app.system_settings.save(update(api_key='replacement-test-key'), expected_revision=0)
    await app.system_settings.reset(expected_revision=1)
    assert app.settings == app.runtime_settings.initial_settings
    assert app.settings_store.load().overrides is None
    assert app.settings_store.load().api_key is None
    assert app.runtime_settings.revision == 2
    assert app.rate_limiter.requests_per_minute is None and app.worker_pool.concurrency == 1
    await app.aclose()


async def test_two_posts_with_same_revision_have_one_winner(application):
    app = application
    results = await asyncio.gather(app.system_settings.save(update(), expected_revision=0), app.system_settings.save(update(model='other'), expected_revision=0), return_exceptions=True)
    assert results.count(None) == 1
    assert sum(isinstance(result, SettingsRevisionConflict) for result in results) == 1
    assert app.settings_store.load().revision == 1
    await app.aclose()


async def test_cancelled_request_completes_commit_and_publish(application, monkeypatch):
    app = application
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    original = app.settings_store.save
    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(app.settings_store, 'save', blocked)
    waiting = asyncio.create_task(app.system_settings.save(update(), expected_revision=0))
    try:
        await asyncio.wait_for(started.wait(), 5)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        release.set()
    await app.system_settings.drain()
    assert app.settings_store.load().revision == app.runtime_settings.revision == 1
    assert app.settings.llm.model == 'new'
    await app.aclose()


async def test_rejected_save_and_reset_after_close(application):
    app = application
    app.begin_close()
    for operation in (app.system_settings.save(update(), expected_revision=0), app.system_settings.reset(expected_revision=0)):
        with pytest.raises(SettingsClosingError):
            await operation
    assert not app.system_settings._operations
    await app.aclose()


async def test_drain_collects_errors_of_cancelled_http_waiters(application, monkeypatch):
    app = application
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    unhandled = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    def fail(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        raise RuntimeError('private-test-key')
    monkeypatch.setattr(app.settings_store, 'save', fail)
    waiting = asyncio.create_task(app.system_settings.save(update(), expected_revision=0))
    try:
        await asyncio.wait_for(started.wait(), 5)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release.set()
        errors = await app.system_settings.drain()
        assert len(errors) == 1 and 'private-test-key' not in str(errors)
        assert not app.system_settings._operations and not unhandled
    finally:
        release.set()
        await app.aclose()
        loop.set_exception_handler(old_handler)


async def test_persisted_settings_are_loaded_after_restart(application):
    app = application
    baseline = app.runtime_settings.initial_settings
    await app.system_settings.save(update(api_key='replacement-test-key'), expected_revision=0)
    await app.aclose()
    restarted = build_application(settings=baseline, llm_client=FakeLLMClient())
    try:
        assert restarted.settings.llm.model == 'new'
        assert restarted.settings.llm.api_key == 'replacement-test-key'
        assert restarted.runtime_settings.revision == 1
    finally:
        await restarted.aclose()
