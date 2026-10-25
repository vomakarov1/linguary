from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import importlib
import json
import pytest
from fastapi.testclient import TestClient
from web_scenario import store_book
from linguary.adapters.config import ServerSettings
from linguary.application.diagnostics import DiagnosticsPolicy
from linguary.composition import build_application
from linguary.domain import LLMCallLog, LLMCallStage
from linguary.web import create_app

BODY = '<script>alert("private diagnostic")</script>'


def seed(app):
    scenario = store_book(app)
    with app.unit_of_work.transaction() as tx:
        tx.llm_calls.add(LLMCallLog('call-local', scenario.book_id, LLMCallStage.CHAPTER_ANALYSIS,
            'fake', 'fake', datetime.now(UTC), 1, 1, True, cost=Decimal('0.001')),
            response_text=BODY)
    return scenario, f'/books/{scenario.book_id}/diagnostics/call-local'


def connect(app, peer='127.0.0.1', host='localhost'):
    return TestClient(create_app(application=app, run_background_worker=False),
        base_url='http://localhost:8000', headers={'host': f'{host}:8000'}, client=(peer, 50000))


@pytest.mark.parametrize('peer', ['192.0.2.10', '172.17.0.1', '10.0.0.1', 'testclient'])
def test_remote_peer_with_forged_local_host_denied(application, peer):
    _, path = seed(application)
    with connect(application, peer) as client:
        response = client.get(path)
        assert response.status_code == 403 and 'private diagnostic' not in response.text
        assert client.delete(path).status_code == 403


@pytest.mark.parametrize('header', ['forwarded', 'x-forwarded-for', 'x-forwarded-host', 'x-forwarded-proto', 'x-forwarded-port'])
def test_forwarded_headers_denied(application, header):
    _, path = seed(application)
    with connect(application) as client:
        response = client.get(path, headers={header: '127.0.0.1'})
        assert response.status_code == 403 and 'private diagnostic' not in response.text


@pytest.mark.parametrize('peer,host', [('127.0.0.1', 'localhost'), ('::1', '[::1]')])
def test_local_metadata_and_text_read_delete_idempotent(application, peer, host):
    scenario, path = seed(application)
    with connect(application, peer, host) as client:
        response = client.get(path)
        assert response.status_code == 200, response.text
        assert response.json()['body'] == BODY
        assert response.json()['body_status'] == 'available'
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['x-content-type-options'] == 'nosniff'
        plain = client.get(path, headers={'accept': 'text/plain'})
        assert plain.headers['content-type'].startswith('text/plain') and plain.text == BODY
        assert client.delete(path).status_code == 204
        assert client.delete(path).status_code == 204
        assert client.get(path).json()['body_status'] == 'deleted'
        assert client.get(path + '-missing').status_code == 404
        store_book(application, book_id='other')
        assert client.get('/books/other/diagnostics/call-local').status_code == 404


@pytest.mark.parametrize('headers', [{'origin': 'http://attacker.invalid'}, {'sec-fetch-site': 'cross-site'},
    {'origin': 'null'}, {'sec-fetch-site': 'same-site'}])
def test_even_get_requires_same_origin(application, headers):
    _, path = seed(application)
    with connect(application) as client:
        response = client.get(path, headers=headers)
        assert response.status_code == 403 and BODY not in response.text


def test_loopback_only_when_server_host_is_network(settings, llm_client):
    app = build_application(settings=replace(settings, server=ServerSettings(host='0.0.0.0')), llm_client=llm_client)
    try:
        _, path = seed(app)
        with connect(app, '172.17.0.1') as client:
            assert client.get(path).status_code == 403
        with connect(app) as client:
            assert client.get(path).status_code == 200
        with connect(app, host='attacker.invalid') as client:
            assert client.get(path).status_code == 403
    finally:
        app.dispose()


@pytest.mark.parametrize('policy,status', [(DiagnosticsPolicy(bodies_enabled=False), 'disabled'),
    (DiagnosticsPolicy(retention_seconds=1), 'expired')])
def test_http_respects_existing_body_policy(application, policy, status):
    _, path = seed(application)
    application.diagnostics.policy = policy
    application.diagnostics.clock = lambda: datetime.now(UTC) + timedelta(seconds=5)
    with connect(application) as client:
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()['body_status'] == status and response.json()['body'] is None
        assert 'private diagnostic' not in response.text


def test_cli_inside_container_reads_and_purges(application, monkeypatch, capsys):
    scenario, _ = seed(application)
    cli = importlib.import_module('linguary.cli.app')
    monkeypatch.setattr(cli, '_application', lambda args: application)
    monkeypatch.setattr(application.__class__, 'aclose', _no_close)
    assert cli.main(['diagnostics', 'list', scenario.book_id]) == 0
    listing = capsys.readouterr().out
    assert 'call-local' in listing and 'private diagnostic' not in listing
    assert cli.main(['diagnostics', 'show', scenario.book_id, 'call-local']) == 0
    assert json.loads(capsys.readouterr().out)['body'] == BODY
    assert cli.main(['diagnostics', 'purge', scenario.book_id, '--call-id', 'call-local']) == 0
    capsys.readouterr()
    assert cli.main(['diagnostics', 'show', scenario.book_id, 'call-local']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['body'] is None and result['body_status'] == 'deleted'
    assert cli.main(['diagnostics', 'show', 'foreign', 'call-local']) == 1
    assert BODY not in capsys.readouterr().err


async def _no_close(self):
    pass


def test_cli_serve_disables_proxy_peer_rewriting(tmp_path, monkeypatch):
    cli = importlib.import_module('linguary.cli.app')
    config = tmp_path / 'config.toml'
    config.write_text('[server]\nhost="0.0.0.0"\n')
    seen = {}
    monkeypatch.setenv(cli.CONFIG_PATH_VARIABLE, '')
    monkeypatch.setattr(cli.uvicorn, 'run', lambda *args, **kwargs: seen.update(kwargs))
    assert cli.main(['--config', str(config), 'serve']) == 0
    assert seen['proxy_headers'] is False
