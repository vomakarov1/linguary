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


def test_list_for_book_returns_dict_meta_without_input_text_preserving_body_status_and_read_uses_get_for_book(
    application, monkeypatch
):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from sqlalchemy import event
    from linguary.adapters.db import mapping
    from linguary.adapters.db.models import LLMCallLogRow
    from linguary.adapters.db.repositories import SqlAlchemyLLMCallLogRepository
    from linguary.application.diagnostics import Diagnostics
    from linguary.domain import TokenUsage

    scenario = store_book(application)
    now = datetime.now(UTC)
    expires_at = (now + timedelta(days=7)).isoformat()
    input_secret = 'SECRET REPAIR INPUT TEXT'
    with application.unit_of_work.transaction() as tx:
        tx.llm_calls.add(
            LLMCallLog(
                id='call-with-input',
                book_id=scenario.book_id,
                stage=LLMCallStage.TRANSLATION_UNIT,
                provider='fake',
                model='fake',
                started_at=now,
                duration_ms=12,
                attempt=2,
                success=True,
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=40,
                    cached_prompt_tokens=64,
                    reasoning_tokens=16,
                ),
                cost=Decimal('0.002'),
                diagnostic_meta={
                    'schema_version': 1,
                    'decision': 'accepted',
                    'input_text': input_secret,
                    'usage_details': {
                        'cached_prompt_tokens': 64,
                        'reasoning_tokens': 16,
                    },
                    'body_meta': {
                        'body_status': 'available',
                        'stored_bytes': len(BODY.encode('utf-8')),
                        'expires_at': expires_at,
                    },
                },
            ),
            response_text=BODY,
        )
        tx.llm_calls.add(
            LLMCallLog(
                id='call-null-meta',
                book_id=scenario.book_id,
                stage=LLMCallStage.CHAPTER_ANALYSIS,
                provider='fake',
                model='fake',
                started_at=now + timedelta(seconds=1),
                duration_ms=5,
                attempt=1,
                success=True,
                diagnostic_meta=None,
            ),
            response_text=None,
        )

    executed_sql: list[str] = []
    with application.unit_of_work.transaction() as tx:
        engine = tx._active_session().get_bind()

        def _capture_sql(conn, cursor, statement, parameters, context, executemany):
            executed_sql.append(statement)

        event.listen(engine, 'before_cursor_execute', _capture_sql)
        try:
            entries = tx.llm_calls.list_for_book(scenario.book_id)
        finally:
            event.remove(engine, 'before_cursor_execute', _capture_sql)

        assert len(entries) == 2
        assert type(entries[0].diagnostic_meta) is dict
        assert 'input_text' not in entries[0].diagnostic_meta
        assert entries[0].diagnostic_meta['body_meta']['body_status'] == 'available'
        assert entries[0].usage == TokenUsage(
            prompt_tokens=100,
            completion_tokens=40,
            cached_prompt_tokens=64,
            reasoning_tokens=16,
        )
        assert entries[1].diagnostic_meta is None

        full_entry = tx.llm_calls.get_for_book(scenario.book_id, 'call-with-input')
        assert full_entry is not None
        assert full_entry.diagnostic_meta is not None
        assert full_entry.diagnostic_meta['input_text'] == input_secret
        assert tx.llm_calls.get_for_book('other-book', 'call-with-input') is None
        assert tx.llm_calls.get_for_book(scenario.book_id, 'missing-call') is None

    select_sql = [s for s in executed_sql if s.lstrip().upper().startswith('SELECT')]
    assert len(select_sql) == 1
    assert 'response_text' not in select_sql[0].lower()
    assert 'json_remove' in select_sql[0].lower()

    # Defensive json.loads when SQLite/driver returns a raw JSON string
    str_meta_row = LLMCallLogRow(
        id='call-str-meta',
        book_id=scenario.book_id,
        task_id=None,
        stage=LLMCallStage.TRANSLATION_UNIT.value,
        provider='fake',
        model='fake',
        started_at=now,
        duration_ms=1,
        attempt=1,
        success=True,
        error_type=None,
        error_message=None,
        error_detail=None,
        prompt_tokens=10,
        completion_tokens=5,
        cost=None,
        attempt_id=None,
        diagnostic_meta_json='{"schema_version": 1, "usage_details": {"cached_prompt_tokens": 4}}',  # type: ignore[arg-type]
    )
    decoded = mapping.to_llm_call(str_meta_row)
    assert type(decoded.diagnostic_meta) is dict
    assert decoded.usage == TokenUsage(
        prompt_tokens=10,
        completion_tokens=5,
        cached_prompt_tokens=4,
        reasoning_tokens=None,
    )

    listed = application.diagnostics.list(scenario.book_id)
    assert len(listed) == 2
    assert type(listed[0]['diagnostic_meta']) is dict
    assert 'input_text' not in listed[0]['diagnostic_meta']
    assert listed[0]['diagnostic_meta']['body_meta']['body_status'] == 'available'
    assert listed[0]['usage']['cached_prompt_tokens'] == 64
    assert listed[0]['usage']['reasoning_tokens'] == 16

    get_calls: list[tuple[str, str]] = []
    orig_get_for_book = SqlAlchemyLLMCallLogRepository.get_for_book

    def _spy_get_for_book(self_repo, b_id: str, c_id: str):
        get_calls.append((b_id, c_id))
        return orig_get_for_book(self_repo, b_id, c_id)

    def _forbid_list_for_book(self_repo, b_id: str):
        raise AssertionError('Diagnostics.read must use get_for_book instead of list_for_book')

    monkeypatch.setattr(SqlAlchemyLLMCallLogRepository, 'get_for_book', _spy_get_for_book)
    monkeypatch.setattr(SqlAlchemyLLMCallLogRepository, 'list_for_book', _forbid_list_for_book)

    with connect(application) as client:
        response = client.get(f'/books/{scenario.book_id}/diagnostics/call-with-input')
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['body'] == BODY
        assert payload['body_status'] == 'available'
        assert payload['metadata']['diagnostic_meta']['input_text'] == input_secret
    assert get_calls == [(scenario.book_id, 'call-with-input')]

    # Fallback to list_for_book for test doubles without get_for_book
    stub_entry = LLMCallLog(
        id='call-stub',
        book_id=scenario.book_id,
        stage=LLMCallStage.CHAPTER_ANALYSIS,
        provider='fake',
        model='fake',
        started_at=now,
        duration_ms=1,
        attempt=1,
        success=True,
        diagnostic_meta={'schema_version': 1, 'body_meta': {'body_status': 'available'}},
    )

    class _StubCallsRepo:
        def diagnostic_record(self, *, call_id=None, attempt_id=None):
            if call_id == 'call-stub':
                return {
                    'call_id': 'call-stub',
                    'book_id': scenario.book_id,
                    'metadata': stub_entry.diagnostic_meta,
                    'body': 'stub-body',
                }
            return None

        def enforce_diagnostics_policy(self, policy, current, *, book_id=None):
            return None

        def list_for_book(self, b_id: str):
            return (stub_entry,) if b_id == scenario.book_id else ()

    class _StubUoW:
        books = SimpleNamespace(get=lambda b_id: object() if b_id == scenario.book_id else None)
        llm_calls = _StubCallsRepo()

        @contextmanager
        def transaction(self):
            yield self

    stub_diag = Diagnostics(_StubUoW(), clock=lambda: now)
    stub_read = stub_diag.read(scenario.book_id, 'call-stub')
    assert stub_read['body'] == 'stub-body'
    assert stub_read['metadata']['call_id'] == 'call-stub'

