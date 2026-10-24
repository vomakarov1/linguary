"""Application shutdown waits for cancelled observers and accepted changes."""
import asyncio
from threading import Event

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.settings import SettingsError
from linguary.adapters.config.system_settings import SystemSettingsUpdate
from linguary.composition import build_application
from linguary.composition.system_settings import SettingsClosingError
from linguary.web.app import create_app


def update():
    return SystemSettingsUpdate(provider='openai_compatible', model='saved', base_url='http://localhost:8080/v1', requests_per_minute=30, concurrency=3)


def build(tmp_path):
    return build_application(settings=AppSettings(llm=LLMSettings(model='initial', api_key='initial-test-key'), storage=StorageSettings(data_root=str(tmp_path / 'data'))))


def block_transaction(app, monkeypatch, *, reset=False, fail=False):
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    name = 'reset' if reset else 'save'
    original = getattr(app.settings_store, name)
    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if fail:
            raise RuntimeError('private-test-key')
        return original(*args, **kwargs)
    monkeypatch.setattr(app.settings_store, name, blocked)
    return started, release


def track_close(app, monkeypatch):
    started = asyncio.Event()
    disposed = []
    handles = [app.runtime_settings.current.configured_client._client._http]
    original_begin = app.system_settings.begin_close
    original_prepare = app.runtime_settings.prepare
    original_dispose = app._dispose
    def begin():
        original_begin()
        started.set()
    def prepare(settings):
        candidate = original_prepare(settings)
        if candidate.configured_client._client is not None:
            handles.append(candidate.configured_client._client._http)
        return candidate
    def dispose():
        disposed.append(True)
        original_dispose()
    monkeypatch.setattr(app.system_settings, 'begin_close', begin)
    monkeypatch.setattr(app.runtime_settings, 'prepare', prepare)
    object.__setattr__(app, '_dispose', dispose)
    return started, disposed, handles


@pytest.mark.parametrize('reset', [False, True])
async def test_cancelled_change_with_blocked_transaction_shutdown_waits(tmp_path, monkeypatch, reset):
    app = build(tmp_path)
    close_started, disposed, handles = track_close(app, monkeypatch)
    if reset:
        await app.system_settings.save(update(), expected_revision=0)
    started, release = block_transaction(app, monkeypatch, reset=reset)
    operation = app.system_settings.reset(expected_revision=1) if reset else app.system_settings.save(update(), expected_revision=0)
    observer = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(started.wait(), 5)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        closing = asyncio.create_task(app.aclose())
        await asyncio.wait_for(close_started.wait(), 5)
        assert not closing.done() and not disposed
        assert not app.runtime_settings.current.configured_client._client._http.is_closed
        release.set()
        await asyncio.wait_for(closing, 5)
        assert app.settings_store.load().revision == app.runtime_settings.revision == (2 if reset else 1)
        assert all(handle.is_closed for handle in handles)
        assert disposed == [True]
        assert not app.system_settings._operations
        assert not app.runtime_settings._close_tasks
        await app.aclose()
        assert disposed == [True]
    finally:
        release.set()
        await app.aclose()


async def test_shutdown_collects_transaction_failure_and_candidate(tmp_path, monkeypatch):
    app = build(tmp_path)
    close_started, disposed, handles = track_close(app, monkeypatch)
    started, release = block_transaction(app, monkeypatch, fail=True)
    loop = asyncio.get_running_loop()
    unhandled = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    observer = asyncio.create_task(app.system_settings.save(update(), expected_revision=0))
    try:
        await asyncio.wait_for(started.wait(), 5)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        closing = asyncio.create_task(app.aclose())
        await close_started.wait()
        assert not closing.done() and not disposed
        release.set()
        await asyncio.wait_for(closing, 5)
        assert app.settings_store.load() is None
        assert app.runtime_settings.revision == 0
        assert all(handle.is_closed for handle in handles)
        assert not unhandled and not app.system_settings._operations
        errors = await app.system_settings.drain()
        assert len(errors) == 1 and 'private-test-key' not in str(errors)
    finally:
        release.set()
        await app.aclose()
        loop.set_exception_handler(previous_handler)


async def test_shutdown_waits_for_accepted_lock_waiters(tmp_path, monkeypatch):
    app = build(tmp_path)
    close_started, disposed, handles = track_close(app, monkeypatch)
    started, release = block_transaction(app, monkeypatch)
    first = asyncio.create_task(app.system_settings.save(update(), expected_revision=0))
    try:
        await asyncio.wait_for(started.wait(), 5)
        second = asyncio.create_task(app.system_settings.reset(expected_revision=1))
        await asyncio.sleep(0)
        assert len(app.system_settings._operations) == 2
        closing = asyncio.create_task(app.aclose())
        await close_started.wait()
        assert not closing.done() and not disposed
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second, closing), 5)
        assert app.runtime_settings.revision == 2
        assert app.settings_store.load().overrides is None
        assert all(handle.is_closed for handle in handles)
    finally:
        release.set()
        await app.aclose()


async def test_shutdown_rejects_new_save_and_reset(tmp_path):
    app = build(tmp_path)
    app.begin_close()
    try:
        with pytest.raises(SettingsClosingError):
            await app.system_settings.save(update(), expected_revision=0)
        with pytest.raises(SettingsClosingError):
            await app.system_settings.reset(expected_revision=0)
        assert not app.system_settings._operations
    finally:
        await app.aclose()


async def test_shutdown_collects_all_clients_and_internal_tasks(tmp_path, monkeypatch):
    app = build(tmp_path)
    close_started, disposed, handles = track_close(app, monkeypatch)
    old_client = app.runtime_settings.current.configured_client
    original = old_client.aclose
    async def fail_close():
        await original()
        raise RuntimeError('private-test-key')
    monkeypatch.setattr(old_client, 'aclose', fail_close)
    await app.system_settings.save(update(), expected_revision=0)
    with pytest.raises(SettingsError) as caught:
        await app.aclose()
    assert 'private-test-key' not in str(caught.value)
    assert all(handle.is_closed for handle in handles)
    assert disposed == [True]
    assert not app.system_settings._operations and not app.runtime_settings._close_tasks


@pytest.mark.parametrize('run_background_worker', [False, True])
async def test_injected_application_lifespan_drains_operations(tmp_path, monkeypatch, run_background_worker):
    app = build(tmp_path)
    close_started, disposed, handles = track_close(app, monkeypatch)
    started, release = block_transaction(app, monkeypatch)
    web = create_app(application=app, run_background_worker=run_background_worker)
    lifespan = web.router.lifespan_context(web)
    await lifespan.__aenter__()
    observer = asyncio.create_task(app.system_settings.save(update(), expected_revision=0))
    try:
        await asyncio.wait_for(started.wait(), 5)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        closing = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.wait_for(close_started.wait(), 5)
        assert not closing.done() and not disposed
        release.set()
        await asyncio.wait_for(closing, 5)
        assert app.settings_store.load().revision == app.runtime_settings.revision == 1
        assert not app.system_settings._operations
        assert not app.worker_pool._workers and app.worker_pool._stop.is_set()
        assert not disposed
        assert not app.runtime_settings.current.configured_client._client._http.is_closed
    finally:
        release.set()
        await app.aclose()
    assert all(handle.is_closed for handle in handles)
