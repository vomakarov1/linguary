"""Прокси: безопасный HTTP-контракт и реальная локальная маршрутизация."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest
from linguary.adapters.config import LLMSettings
from linguary.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
def payload(**changes):
    return dict(provider='openai_compatible', model='new-model', base_url='http://localhost:8080/v1', requests_per_minute=30, concurrency=3, expected_revision=0) | changes

SECRET = 'http://alice:p%40ss@localhost:8123'


def test_proxy_get_success_json_only_exposes_mode_and_presence(configured_client, configured_application):
    response = configured_client.post('/system/config', json=payload(proxy_mode='custom', proxy_url=SECRET))
    assert response.status_code == 200
    assert response.json()['llm']['proxy_mode'] == 'custom'
    assert response.json()['llm']['has_proxy_url'] is True
    assert configured_application.settings_store.load().proxy_url == SECRET
    assert SECRET not in response.text
    for secret in (SECRET, 'alice', 'p%40ss', 'p@ss'):
        assert secret not in configured_client.get('/system/config', headers={'Accept': 'text/html'}).text
    response = configured_client.post('/system/config', json=payload(expected_revision=1))
    assert response.status_code == 200
    assert configured_application.settings.llm.proxy_url == SECRET


@pytest.mark.parametrize('html', [False, True])
def test_custom_without_url_html_json_422(configured_client, html):
    response = configured_client.post('/system/config', json=payload(proxy_mode='custom'), headers={'Accept': 'text/html' if html else 'application/json'})
    assert response.status_code == 422
    assert 'proxy_url' in response.text


def test_proxy_url_credentials_in_other_fields_are_not_reflected(configured_client):
    response = configured_client.post('/system/config', json=payload(proxy_mode='custom', proxy_url=SECRET, model='p@ss'), headers={'Accept': 'text/html'})
    assert response.status_code == 422
    assert 'p@ss' not in response.text
    assert SECRET not in response.text


@contextmanager
def server(*, entered=None, release=None, status=200, response_body=b'{"choices":[]}'):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers.get('Proxy-Authorization')))
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            if entered is not None:
                entered.set()
            if release is not None:
                assert release.wait(5)
            body = response_body
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}', requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize('mode,no_proxy,expected', [('auto', '', 'proxy'), ('auto', '127.0.0.1', 'origin'), ('direct', '', 'origin'), ('custom', '*', 'proxy')])
async def test_real_httpx_proxy_routing(monkeypatch, mode, no_proxy, expected):
    with server() as (origin, origin_requests), server() as (proxy, proxy_requests):
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy'):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv('HTTP_PROXY', proxy)
        monkeypatch.setenv('ALL_PROXY', proxy)
        monkeypatch.setenv('NO_PROXY', no_proxy)
        settings = LLMSettings(model='test', api_key='key', base_url=origin, proxy_mode=mode, proxy_url=proxy.replace('://', '://alice:p%40ss@') if mode == 'custom' else None)
        client = OpenAICompatibleLLMClient(settings=settings)
        try:
            assert await client._post({'test': True}) == {'choices': []}
        finally:
            await client.aclose()
        assert bool(proxy_requests) is (expected == 'proxy')
        assert bool(origin_requests) is (expected == 'origin')
        if mode == 'custom':
            assert proxy_requests[0][1] == 'Basic YWxpY2U6cEBzcw=='


async def test_proxy_generation_save_keeps_active_task_old_and_routes_new_task_new(tmp_path):
    import httpx
    from linguary.adapters.config import AppSettings, StorageSettings
    from linguary.composition import build_application
    from linguary.web import create_app
    entered, release = Event(), Event()
    with server(entered=entered, release=release) as (old_proxy, old_requests), server() as (new_proxy, new_requests):
        app = build_application(settings=AppSettings(llm=LLMSettings(model='test', api_key='key', base_url='http://origin.invalid', proxy_mode='custom', proxy_url=old_proxy), storage=StorageSettings(data_root=str(tmp_path))))
        try:
            web = create_app(application=app, run_background_worker=False)
            async with web.router.lifespan_context(web):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url='http://localhost') as http:
                    with app.runtime_settings.lease() as old_generation:
                        old_client = old_generation.configured_client._client
                        active = asyncio.create_task(old_client._post({'old': True}))
                        try:
                            assert await asyncio.to_thread(entered.wait, 5)
                            saved = await http.post('/system/config', json=payload(base_url='http://origin.invalid', proxy_mode='custom', proxy_url=new_proxy))
                            assert saved.status_code == 200
                            assert not old_client._http.is_closed
                            new_client = app.runtime_settings.current.configured_client._client
                            assert new_client is not old_client
                            assert await new_client._post({'new': True}) == {'choices': []}
                            assert len(old_requests) == len(new_requests) == 1
                            release.set()
                            assert await asyncio.wait_for(active, 5) == {'choices': []}
                        finally:
                            release.set()
                            await asyncio.gather(active, return_exceptions=True)
                    await asyncio.sleep(0)
                    assert old_client._http.is_closed
                    assert not new_client._http.is_closed
        finally:
            release.set()
            await app.aclose()


@pytest.mark.parametrize('html', [False, True])
def test_proxy_form_and_json_have_same_semantics(configured_client, configured_application, html):
    for revision, mode, url in [(0, 'custom', SECRET), (1, 'custom', ''), (2, 'direct', ''), (3, 'auto', '')]:
        values = payload(expected_revision=revision, proxy_mode=mode, proxy_url=url)
        if html:
            values['requests_per_minute'] = '30'
            response = configured_client.post('/system/config', data=values, headers={'Accept': 'text/html'}, follow_redirects=False)
            assert response.status_code == 303
        else:
            response = configured_client.post('/system/config', json=values)
            assert response.status_code == 200
        assert configured_application.settings.llm.proxy_mode == mode
        assert configured_application.settings.llm.proxy_url == (SECRET if mode == 'custom' else None)


def test_failed_proxy_save_keeps_database_and_client(configured_client, configured_application, monkeypatch):
    from linguary.adapters.db.system_settings import SettingsStoreError
    old = configured_application.runtime_settings.current
    def fail(*args, **kwargs):
        raise SettingsStoreError('Не удалось записать настройки')
    monkeypatch.setattr(configured_application.settings_store, 'save', fail)
    response = configured_client.post('/system/config', json=payload(proxy_mode='custom', proxy_url=SECRET))
    assert response.status_code == 500
    assert configured_application.runtime_settings.current is old
    assert configured_application.settings_store.load() is None
    assert SECRET not in response.text


@pytest.mark.parametrize('field,value', [('proxy_mode', 'unsupported'), ('proxy_mode', 3), ('proxy_url', {'secret': SECRET}), ('proxy_url', [SECRET])])
def test_proxy_mode_unknown_and_url_wrong_type_are_safe(configured_client, field, value):
    response = configured_client.post('/system/config', json=payload(proxy_mode='custom', proxy_url=SECRET) | {field: value}, headers={'Accept': 'text/html'})
    assert response.status_code == 422
    assert SECRET not in response.text


@pytest.mark.parametrize('status', [422, 409, 500])
def test_proxy_url_input_password_empty_on_errors(configured_client, configured_application, monkeypatch, status):
    import re
    if status == 409:
        configured_client.post('/system/config', json=payload())
    if status == 500:
        def fail(settings):
            raise RuntimeError(SECRET)
        monkeypatch.setattr(configured_application.runtime_settings, 'prepare', fail)
    response = configured_client.post('/system/config', json=payload(proxy_mode='custom', proxy_url=SECRET, concurrency=0 if status == 422 else 3), headers={'Accept': 'text/html'})
    assert response.status_code == status
    field = re.search(r'<input[^>]+name="proxy_url"[^>]*>', response.text).group()
    assert 'type="password"' in field and 'value=' not in field
    for secret in (SECRET, 'alice', 'p%40ss', 'p@ss'):
        assert secret not in response.text


def test_proxy_duplicate_fields_do_not_echo_first_secret(configured_client):
    import json
    values = payload(proxy_mode='custom', model='first-password')
    body = json.dumps(values)[:-1] + ',"proxy_url":"http://alice:first-password@localhost","proxy_url":"http://bob:second-password@localhost"}'
    response = configured_client.post('/system/config', content=body, headers={'Content-Type': 'application/json', 'Accept': 'text/html'})
    assert response.status_code == 422
    assert 'first-password' not in response.text


@pytest.mark.parametrize('mode', ['custom', 'auto'])
async def test_real_proxy_rejected_basic_credentials_are_redacted(monkeypatch, mode):
    from linguary.application.ports.llm import LLMProviderError
    with server(status=407, response_body=b'Rejected Basic YWxpY2U6cEBzcw==') as (proxy, requests):
        url = proxy.replace('://', '://alice:p%40ss@')
        monkeypatch.setenv('HTTP_PROXY', url)
        monkeypatch.setenv('NO_PROXY', '')
        monkeypatch.delenv('http_proxy', raising=False)
        monkeypatch.delenv('no_proxy', raising=False)
        client = OpenAICompatibleLLMClient(settings=LLMSettings(model='test', api_key='key', base_url='http://origin.invalid', proxy_mode=mode, proxy_url=url if mode == 'custom' else None))
        try:
            with pytest.raises(LLMProviderError) as caught:
                await client._post({'test': True})
            assert 'YWxpY2U6cEBzcw==' not in str(caught.value)
            assert '[REDACTED]' in str(caught.value)
            assert requests[0][1] == 'Basic YWxpY2U6cEBzcw=='
        finally:
            await client.aclose()


@pytest.mark.parametrize('status', [409, 500])
async def test_cross_field_basic_credentials_are_removed_after_generation_changes(tmp_path, monkeypatch, status):
    import httpx
    from linguary.adapters.config import AppSettings, StorageSettings
    from linguary.adapters.db.system_settings import SettingsStoreError
    from linguary.composition import build_application
    from linguary.web import create_app
    from linguary.web.routes import system
    entered, release = asyncio.Event(), asyncio.Event()
    original = system.read_settings_update
    async def paused(request, **kwargs):
        parsed = await original(request, **kwargs)
        if request.headers.get('X-Test-Pause'):
            entered.set()
            await release.wait()
        return parsed
    monkeypatch.setattr(system, 'read_settings_update', paused)
    app = build_application(settings=AppSettings(llm=LLMSettings(model='test', api_key='key'), storage=StorageSettings(data_root=str(tmp_path))))
    try:
        web = create_app(application=app, run_background_worker=False)
        async with web.router.lifespan_context(web):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url='http://localhost') as http:
                active = asyncio.create_task(http.post('/system/config', json=payload(model='YWxpY2U6cEBzcw==', expected_revision=0 if status == 409 else 1), headers={'Accept': 'text/html', 'X-Test-Pause': '1'}))
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    saved = await http.post('/system/config', json=payload(proxy_mode='custom', proxy_url=SECRET))
                    assert saved.status_code == 200
                    if status == 500:
                        def fail(*args, **kwargs):
                            raise SettingsStoreError('Не удалось записать настройки')
                        monkeypatch.setattr(app.settings_store, 'save', fail)
                    release.set()
                    response = await asyncio.wait_for(active, 5)
                    assert response.status_code == status
                    assert 'YWxpY2U6cEBzcw==' not in response.text
                    assert SECRET not in response.text
                    assert app.settings_store.load().revision == 1
                    assert app.settings.llm.proxy_url == SECRET
                finally:
                    release.set()
                    await asyncio.gather(active, return_exceptions=True)
    finally:
        await app.aclose()
