"""Settings HTTP contracts, request origin, and safe resubmission HTML."""
import asyncio
from dataclasses import asdict
from fastapi.testclient import TestClient
import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.composition import build_application
from linguary.web import create_app
from linguary.web.routes import system


def payload(**changes):
    values = dict(provider='openai_compatible', model='new-model', base_url='http://localhost:8080/v1', requests_per_minute=30, concurrency=3, expected_revision=0)
    values.update(changes)
    return values


def test_save_updates_get_and_persists(configured_client, configured_application):
    response = configured_client.post('/system/config', json=payload())
    assert response.status_code == 200
    data = response.json()
    assert data['revision'] == 1 and data['llm']['model'] == 'new-model'
    assert data['worker']['concurrency'] == 3 and data['llm']['requests_per_minute'] == 30
    assert configured_client.get('/system/config').json() == data
    assert configured_application.settings_store.load().overrides['model'] == 'new-model'


def test_save_system_settings_updates_split_models_and_temperature(configured_client, configured_application):
    values = payload(
        analysis_model='analysis-v1',
        processing_model='processing-v1',
        temperature=0.8,
        timeout_seconds=120.0,
    )
    response = configured_client.post('/system/config', json=values)
    assert response.status_code == 200
    data = response.json()
    assert data['llm']['analysis_model'] == 'analysis-v1'
    assert data['llm']['processing_model'] == 'processing-v1'
    assert data['llm']['temperature'] == 0.8
    assert data['llm']['timeout_seconds'] == 120.0

    get_data = configured_client.get('/system/config').json()
    assert get_data['llm']['analysis_model'] == 'analysis-v1'
    assert get_data['llm']['processing_model'] == 'processing-v1'
    assert get_data['llm']['temperature'] == 0.8
    assert get_data['llm']['timeout_seconds'] == 120.0
    assert configured_application.settings.llm.analysis_model == 'analysis-v1'
    assert configured_application.settings.llm.processing_model == 'processing-v1'
    assert configured_application.settings.llm.temperature == 0.8
    assert configured_application.settings.llm.timeout_seconds == 120.0
    stored_overrides = configured_application.settings_store.load().overrides
    assert stored_overrides.get('analysis_model') == 'analysis-v1'
    assert stored_overrides.get('processing_model') == 'processing-v1'
    assert stored_overrides.get('temperature') == 0.8
    assert stored_overrides.get('timeout_seconds') == 120.0


def test_save_system_settings_resets_temperature_when_empty_or_none(configured_client, configured_application):
    first = configured_client.post('/system/config', json=payload(temperature=0.8, expected_revision=0))
    assert first.status_code == 200
    assert first.json()['llm']['temperature'] == 0.8
    assert configured_application.settings.llm.temperature == 0.8
    assert configured_application.settings_store.load().overrides['temperature'] == 0.8

    second = configured_client.post('/system/config', json=payload(temperature=None, expected_revision=1))
    assert second.status_code == 200
    assert second.json()['llm']['temperature'] is None
    assert configured_application.settings.llm.temperature is None
    assert configured_application.settings_store.load().overrides['temperature'] is None

    third = configured_client.post('/system/config', json=payload(temperature=0.5, expected_revision=2))
    assert third.status_code == 200
    assert third.json()['llm']['temperature'] == 0.5

    form_data = payload(temperature='', expected_revision=3)
    fourth = configured_client.post(
        '/system/config',
        data=form_data,
        headers={'content-type': 'application/x-www-form-urlencoded'},
    )
    assert fourth.status_code in (200, 303)
    assert configured_application.settings.llm.temperature is None
    assert configured_application.settings_store.load().overrides['temperature'] is None

    configured_client.post('/system/config', json=payload(temperature=0.7, expected_revision=4))
    assert configured_application.settings.llm.temperature == 0.7
    no_temp_payload = payload(expected_revision=5)
    configured_client.post('/system/config', json=no_temp_payload)
    assert configured_application.settings.llm.temperature == 0.7



@pytest.mark.parametrize('key', [None, '', '   '])
def test_empty_key_keeps_current_key(configured_client, configured_application, key):
    values = payload()
    if key is not None:
        values['api_key'] = key
    assert configured_client.post('/system/config', json=values).status_code == 200
    assert configured_application.settings.llm.api_key == 'test-key'


@pytest.mark.parametrize('model,key,ready', [('model', 'new-test-key', True), ('', 'new-test-key', False), ('model', '', False), ('', '', False)])
def test_complete_and_incomplete_save_json_return_actual_readiness(client, model, key, ready):
    response = client.post('/system/config', json=payload(model=model, api_key=key))
    assert response.status_code == 200
    assert response.json()['llm']['is_ready'] is ready
    assert 'new-test-key' not in response.text


def test_valid_but_unusable_key_saves_without_network_call(tmp_path, monkeypatch):
    import httpx
    async def forbidden(*args, **kwargs):
        raise AssertionError('Settings must not call the provider')
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    app = build_application(settings=AppSettings(storage=StorageSettings(data_root=str(tmp_path / 'data'))))
    try:
        with TestClient(create_app(application=app, run_background_worker=False), base_url='http://localhost') as client:
            response = client.post('/system/config', json=payload(api_key='unusable-test-key'))
            assert response.status_code == 200 and response.json()['llm']['is_ready'] is True
    finally:
        asyncio.run(app.aclose())


def test_reset_without_baseline_key_is_not_ready(client):
    assert client.post('/system/config', json=payload(api_key='new-test-key')).status_code == 200
    response = client.post('/system/config/reset', json={'expected_revision': 1})
    assert response.status_code == 200
    assert response.json()['llm']['is_ready'] is False
    assert response.json()['llm']['has_api_key'] is False


def test_reset_clears_overrides_and_restores_baseline(configured_client, configured_application):
    assert configured_client.post('/system/config', json=payload(api_key='new-test-key')).status_code == 200
    response = configured_client.post('/system/config/reset', json={'expected_revision': 1})
    assert response.status_code == 200 and response.json()['revision'] == 2
    assert response.json()['llm']['model'] == 'fake-model'
    assert configured_application.settings_store.load().overrides is None
    assert configured_application.settings.llm.api_key == 'test-key'


@pytest.mark.parametrize('reset', [False, True])
def test_stale_revision_returns_conflict(configured_client, configured_application, reset):
    configured_client.post('/system/config', json=payload())
    before = configured_application.settings_store.load()
    response = configured_client.post('/system/config/reset' if reset else '/system/config', json={'expected_revision': 0} if reset else payload(model='stale'))
    assert response.status_code == 409
    assert response.json()['code'] == 'revision_conflict'
    assert configured_application.settings_store.load() == before


def test_same_origin_posts_are_accepted(client):
    response = client.post('/system/config', json=payload(), headers={'Origin': 'http://127.0.0.1:8000', 'Sec-Fetch-Site': 'same-origin'})
    assert response.status_code == 200


@pytest.mark.parametrize('headers', [{'Origin': 'https://foreign.example'}, {'Sec-Fetch-Site': 'cross-site'}])
def test_foreign_origin_and_fetch_site_posts_are_rejected_without_changes(client, application, headers):
    response = client.post('/system/config', content='malformed secret body', headers={**headers, 'Content-Type': 'application/json'})
    assert response.status_code == 403
    assert application.settings_store.load() is None


def test_foreign_host_is_rejected(client, application):
    assert client.post('/system/config', json=payload(), headers={'Host': 'foreign.example'}).status_code == 400
    assert application.settings_store.load() is None


def test_posts_without_origin_headers_keep_current_access_model(client):
    assert client.post('/system/config', json=payload()).status_code == 200


@pytest.mark.parametrize('value', [0, True, 'private-test-key'])
def test_errors_never_expose_key(client, value):
    response = client.post('/system/config', json=payload(concurrency=value, api_key='private-test-key'))
    assert response.status_code == 422
    assert 'private-test-key' not in response.text
    assert set(response.json()) == {'code', 'message', 'field_errors'}


def test_closing_rejects_only_unaccepted_operation_with_503(client, application):
    application.begin_close()
    response = client.post('/system/config', json=payload())
    assert response.status_code == 503 and response.json()['code'] == 'settings_closing'
    assert not application.system_settings._operations


@pytest.mark.parametrize('reset,marker', [(False, 'saved'), (True, 'reset')])
def test_html_success_has_fixed_redirect_without_referer(client, reset, marker):
    response = client.post('/system/config/reset' if reset else '/system/config', json={'expected_revision': 0} if reset else payload(), headers={'Accept': 'text/html'}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers['location'] == '/system/config?result=' + marker


@pytest.mark.parametrize('status', [422, 409, 500])
def test_html_errors_pass_safe_context_without_referer(configured_client, configured_application, monkeypatch, status):
    contexts = []
    original = system.render
    def record(request, template, **context):
        contexts.append(context)
        return original(request, template, **context)
    monkeypatch.setattr(system, 'render', record)
    if status == 409:
        configured_client.post('/system/config', json=payload())
    if status == 500:
        def fail(settings):
            raise RuntimeError('private-test-key')
        monkeypatch.setattr(configured_application.runtime_settings, 'prepare', fail)
    response = configured_client.post('/system/config', json=payload(model='edited-model', concurrency=0 if status == 422 else 3, api_key='private-test-key'), headers={'Accept': 'text/html'})
    assert response.status_code == status
    values = contexts[-1]['form_values']
    assert values.model == 'edited-model' and values.expected_revision == 0
    assert 'api_key' not in asdict(values)
    assert 'private-test-key' not in response.text + str(asdict(values))
    assert contexts[-1]['config_url'] == '/system/config'
    assert contexts[-1]['can_submit'] is (status != 409)


def test_unsupported_media_type_is_415(client):
    response = client.post('/system/config', content='secret-body', headers={'Content-Type': 'text/plain'})
    assert response.status_code == 415
    assert response.json()['code'] == 'unsupported_media_type'
    assert 'secret-body' not in response.text


async def test_closing_rejects_only_new_operation_but_completes_accepted_http_request(application, monkeypatch):
    import httpx
    from threading import Event
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    original = application.settings_store.save
    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(application.settings_store, 'save', blocked)
    web = create_app(application=application, run_background_worker=False)
    async with web.router.lifespan_context(web):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url='http://localhost') as client:
            first = asyncio.create_task(client.post('/system/config', json=payload()))
            try:
                await asyncio.wait_for(started.wait(), 5)
                application.begin_close()
                rejected = await client.post('/system/config/reset', json={'expected_revision': 0})
                assert rejected.status_code == 503
                assert len(application.system_settings._operations) == 1
                release.set()
                response = await asyncio.wait_for(first, 5)
                assert response.status_code == 200 and response.json()['revision'] == 1
            finally:
                release.set()
                await application.aclose()


def test_html_missing_revision_disables_resubmission(client, monkeypatch):
    contexts = []
    original = system.render
    def record(request, template, **context):
        contexts.append(context)
        return original(request, template, **context)
    monkeypatch.setattr(system, 'render', record)
    values = payload()
    del values['expected_revision']
    response = client.post('/system/config', json=values, headers={'Accept': 'text/html'})
    assert response.status_code == 422
    assert contexts[-1]['form_values'].expected_revision is None
    assert contexts[-1]['can_submit'] is False


def test_cleanup_error_after_publication_does_not_return_false_failure(configured_client, configured_application, monkeypatch):
    async def fail_close():
        raise RuntimeError('private-test-key')
    monkeypatch.setattr(configured_application.runtime_settings.current.configured_client, 'aclose', fail_close)
    response = configured_client.post('/system/config', json=payload())
    assert response.status_code == 200 and response.json()['revision'] == 1
    assert 'private-test-key' not in response.text


def test_large_positive_rate_json_save_matches_database_and_runtime(configured_client, configured_application):
    rate = 10**400
    response = configured_client.post('/system/config', json=payload(requests_per_minute=rate))
    assert response.status_code == 200
    data = response.json()
    stored = configured_application.settings_store.load()
    assert data['revision'] == stored.revision == configured_application.runtime_settings.revision == 1
    assert data['llm']['requests_per_minute'] == stored.overrides['requests_per_minute'] == rate
    assert configured_application.settings.llm.requests_per_minute == rate
    assert configured_application.rate_limiter.requests_per_minute == rate
    assert configured_client.get('/system/config').json() == data
