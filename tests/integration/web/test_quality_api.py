from dataclasses import replace
from decimal import Decimal
from uuid import uuid4
import pytest
from web_scenario import store_book, start_run
from linguary.application.quality_budget import QualityRoute
from linguary.domain.recovery import RecoveryRoot, RecoveryScope, CallAttempt


def setup_review(client, app):
    scenario = store_book(app)
    response = client.post(f'/books/{scenario.book_id}/conditions', json=dict(
        action='continue_legacy', expected_revision=0, settings_revision=0, confirmed=True))
    assert response.status_code == 201
    app.quality_review.routes['test', 'review'] = QualityRoute('review', 'test', 'fixture-v1',
        Decimal('1'), Decimal('2'), Decimal('0'), 100, lambda request: 1000, 'synthetic-proof', True)
    manifest = app.quality_review.preview(scenario.book_id)
    command = dict(request_key='explicit-review', model='review', route_id='test', budget_microusd=100000,
        scope='all', expected_manifest_sha=manifest['manifest_sha'])
    return scenario, command


@pytest.mark.parametrize('field', ['model', 'route_id', 'budget_microusd', 'scope', 'expected_manifest_sha', 'request_key'])
def test_missing_model_budget_scope_422(configured_client, configured_application, field):
    scenario, command = setup_review(configured_client, configured_application)
    del command[field]
    assert configured_client.post(f'/books/{scenario.book_id}/quality-review', json=command).status_code == 422
    assert configured_application.quality_review.get(scenario.book_id)['state'] == 'not-requested'


def test_request_key_is_idempotent(configured_client, configured_application, llm_client):
    scenario, command = setup_review(configured_client, configured_application)
    path = f'/books/{scenario.book_id}/quality-review'
    before = len(llm_client.requests)
    first = configured_client.post(path, json=command)
    assert first.status_code == 202, first.text
    assert first.json()['state'] == 'running'
    second = configured_client.post(path, json=command)
    assert second.status_code == 202 and second.json()['review_id'] == first.json()['review_id']
    conflict = configured_client.post(path, json={**command, 'budget_microusd': 200000})
    assert conflict.status_code == 409 and conflict.json()['detail']['reason_code'] == 'request_key_conflict'
    read = configured_client.get(path, params={'review_id': first.json()['review_id']})
    assert read.status_code == 200 and read.json()['review_id'] == first.json()['review_id']
    with configured_application.unit_of_work.transaction() as tx:
        assert len(tx.tasks.list_for_book(scenario.book_id)) == 1
    assert len(llm_client.requests) == before


@pytest.mark.parametrize('change,reason', [({'route_id': 'missing'}, 'route_unavailable'),
    ({'expected_manifest_sha': '0' * 64}, 'manifest_changed'), ({'budget_microusd': 1}, 'budget')])
def test_review_errors_never_imply_success(configured_client, configured_application, change, reason):
    scenario, command = setup_review(configured_client, configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', json={**command, **change})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['reason_code'] == reason


def test_legacy_off_read_and_foreign_review_not_found(client, application, llm_client):
    one = store_book(application)
    two = store_book(application, book_id='other')
    path = f'/books/{one.book_id}/quality-review'
    response = client.get(path)
    assert response.status_code == 200 and response.json()['state'] == 'not-requested'
    assert response.json()['available_routes'] == []
    assert client.get(path, params={'review_id': str(uuid4())}).status_code == 404
    assert client.get('/books/missing/quality-review').status_code == 404
    assert llm_client.requests == []


def test_html_review_uses_same_scenario(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data=command,
        headers={'accept': 'text/html'}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers['location'] == f'/books/{scenario.book_id}'
    assert configured_application.quality_review.get(scenario.book_id)['state'] == 'running'


def test_cross_origin_post_denied(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', json=command,
        headers={'origin': 'https://foreign.example'})
    assert response.status_code == 403
    assert configured_application.quality_review.get(scenario.book_id)['state'] == 'not-requested'


def recovery_fixture(client, app, *, unknown=False):
    scenario, _ = setup_review(client, app)
    start_run(app, scenario)
    with app.unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(scenario.book_id)
        root = RecoveryRoot(str(uuid4()), scenario.book_id, conditions.id, str(uuid4()),
            dict(schema_version=1, active_run_id=scenario.run_id, ordered_blocks=list(scenario.block_ids()),
                historical_coverage=[], policy=dict(max_calls=12, max_scope_calls=3)),
            calls_used=12, status='stopped', reason='interrupted' if unknown else 'limit')
        tx.work.add_root(root)
        scope = RecoveryScope(str(uuid4()), root.id, scenario.block_ids())
        tx.work.add_scope(scope)
        if unknown:
            tx.work.add_attempt(CallAttempt(str(uuid4()), scenario.book_id, state='unknown',
                conditions_id=conditions.id, root_id=root.id, scope_id=scope.id, run_id=scenario.run_id))
    return scenario, root


def test_extension_not_implicit_retry(configured_client, configured_application):
    scenario, root = recovery_fixture(configured_client, configured_application)
    path = f'/books/{scenario.book_id}/recovery/{root.id}/limit'
    payload = dict(additional_calls=2, expected_generation=0, confirmed=True)
    response = configured_client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['limit'] == 14 and response.json()['calls_used'] == 12
    assert response.json()['generation'] == 1
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.tasks.list_for_book(scenario.book_id) == ()
    stale = configured_client.post(path, json=payload)
    assert stale.status_code == 409 and stale.json()['detail']['reason_code'] == 'generation_conflict'
    assert configured_client.post(path, json={**payload, 'confirmed': False}).status_code == 422
    assert configured_client.post(f'/books/missing/recovery/{root.id}/limit', json=payload).status_code == 404
    store_book(configured_application, book_id='other')
    assert configured_client.post(f'/books/other/recovery/{root.id}/limit', json=payload).status_code == 404


def test_extension_never_clears_unknown(configured_client, configured_application):
    scenario, root = recovery_fixture(configured_client, configured_application, unknown=True)
    response = configured_client.post(f'/books/{scenario.book_id}/recovery/{root.id}/limit',
        json=dict(additional_calls=2, expected_generation=0, confirmed=True))
    assert response.status_code == 409 and response.json()['detail']['reason_code'] == 'unknown'
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.get_root(root.id) == root


@pytest.mark.parametrize('change', [{'budget_microusd': 0}, {'budget_microusd': -1},
    {'budget_microusd': True}, {'budget_microusd': 1.5}, {'currency': 'EUR'}, {'scope': []},
    {'scope': ['duplicate', 'duplicate']}, {'model': ' '}, {'base_url': 'https://foreign.invalid'}])
def test_invalid_explicit_parameters_never_enqueue(configured_client, configured_application, change):
    scenario, command = setup_review(configured_client, configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', json={**command, **change})
    assert response.status_code == 422
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.tasks.list_for_book(scenario.book_id) == ()


def test_unproven_price_has_no_review_or_task(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    route = configured_application.quality_review.routes['test', 'review']
    configured_application.quality_review.routes['test', 'review'] = replace(route, proof='')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', json=command)
    assert response.status_code == 409 and response.json()['detail']['reason_code'] == 'price_unavailable'
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == () and tx.tasks.list_for_book(scenario.book_id) == ()
