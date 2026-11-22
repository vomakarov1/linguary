from dataclasses import asdict, replace
from decimal import Decimal
from uuid import uuid4
import pytest
from web_scenario import continue_legacy, store_book, start_run
from linguary.application.book_conditions import payload_digest
from linguary.application.quality_budget import QualityRoute
from linguary.application.recovery import NOTES_POLICY, RecoveryPolicy, root_manifest
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.application.use_cases.extend_recovery_limit import (
    ExtendRecoveryLimit,
    ExtendRecoveryLimitCommand,
)
from linguary.domain import BookStatus
from linguary.domain.recovery import RecoveryRoot, RecoveryScope, CallAttempt


def setup_review(client, app, book_id=None):
    scenario = store_book(app) if book_id is None else store_book(app, book_id=book_id)
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
    store_book(application, book_id='other')
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


def recovery_fixture(client, app, *, unknown=False, book_id=None):
    scenario, _ = setup_review(client, app, book_id)
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


def test_stalled_processing_without_failed_tasks_stops_auto_refresh(
    configured_client, configured_application,
):
    scenario = store_book(
        configured_application, status=BookStatus.PROCESSING, book_id='stalled-processing',
    )
    detail = configured_client.get(f'/books/{scenario.book_id}')
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body['book']['status'] == 'PROCESSING'
    assert body['progress']['failures'] == []
    assert body['progress']['pending_tasks'] == 0
    assert body['progress']['running_tasks'] == 0
    page = configured_client.get(
        f'/books/{scenario.book_id}', headers={'accept': 'text/html'},
    )
    assert page.status_code == 200, page.text
    assert 'http-equiv="refresh"' not in page.text
    assert 'страница обновится автоматически' not in page.text


def restartable_unknown_root(app, *, book_id, reason='unknown_outcome'):
    scenario = store_book(app, status=BookStatus.PROCESSING, book_id=book_id)
    continue_legacy(app, book_id)
    start_run(app, scenario)
    ordered = tuple(scenario.block_ids(0))
    with app.unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(scenario.book_id)
        run = tx.runs.active_for_book(scenario.book_id)
        analysis = tx.analyses.get_for_section(scenario.book_id, scenario.section(0).id)
        policy = RecoveryPolicy(**asdict(app.settings.recovery))
        manifest = root_manifest(
            book_id=scenario.book_id,
            conditions_id=conditions.id,
            glossary_sha=payload_digest(asdict(run.glossary_snapshot)),
            analysis_sha=payload_digest(asdict(analysis)),
            plan_sha=None,
            ordered_blocks=ordered,
            treatments=tuple('TRANSLATE' for _ in ordered),
            boundaries=tuple((block_id,) for block_id in ordered),
            atomic_intervals=tuple((block_id,) for block_id in ordered),
            max_source_chars=4000,
            policy=policy,
            block_chars={block_id: 20 for block_id in ordered},
        )
        manifest.update(
            notes_policy=NOTES_POLICY, active_run_id=run.id, historical_coverage=[],
        )
        root = RecoveryRoot(
            str(uuid4()), scenario.book_id, conditions.id, str(uuid4()), manifest,
            status='stopped', reason=reason, operation_kind='translation',
            policy_version=CURRENT_TRANSLATION_POLICY,
        )
        tx.work.add_root(root)
        for block_id in ordered:
            tx.work.add_scope(RecoveryScope(str(uuid4()), root.id, (block_id,)))
    return scenario, root


def test_web_restart_route_requires_separate_unknown_acknowledgement(
    configured_client, configured_application,
):
    scenario, root = restartable_unknown_root(
        configured_application, book_id='unknown-restart',
    )
    path = f'/books/{scenario.book_id}/recovery/{root.id}/restart'
    denied = configured_client.post(path, json=dict(
        expected_generation=root.generation, confirmed=True, acknowledge_unknown_outcome=False,
    ))
    assert denied.status_code == 409, denied.text
    assert denied.json()['detail']['reason_code'] == 'unknown'
    with configured_application.unit_of_work.transaction() as tx:
        stored = tx.work.get_root(root.id)
        assert stored.superseded_by is None and stored.reason == 'unknown_outcome'
    allowed = configured_client.post(path, json=dict(
        expected_generation=root.generation, confirmed=True, acknowledge_unknown_outcome=True,
    ))
    assert allowed.status_code == 202, allowed.text
    assert allowed.json()['successor_root_id']

    html_scenario, html_root = restartable_unknown_root(
        configured_application, book_id='unknown-restart-html',
    )
    html_denied = configured_client.post(
        f'/books/{html_scenario.book_id}/recovery/{html_root.id}/restart',
        data=dict(
            expected_generation=html_root.generation, confirmed='true',
            acknowledge_unknown_outcome='false',
        ),
        headers={'accept': 'text/html'}, follow_redirects=False,
    )
    assert html_denied.status_code == 409, html_denied.text
    html_allowed = configured_client.post(
        f'/books/{html_scenario.book_id}/recovery/{html_root.id}/restart',
        data=dict(
            expected_generation=html_root.generation, confirmed='true',
            acknowledge_unknown_outcome='true',
        ),
        headers={'accept': 'text/html'}, follow_redirects=False,
    )
    assert html_allowed.status_code == 303, html_allowed.text
    assert html_allowed.headers['location'] == f'/books/{html_scenario.book_id}'


def _plant_attempt(app, root, *, state, claim=False):
    with app.unit_of_work.transaction() as tx:
        scope = next(item for item in tx.work.list_scopes(root.id) if item.ordered_blocks)
        if claim:
            scope = tx.work.claim_scope(scope.id, expected_generation=scope.generation)
        tx.work.add_attempt(CallAttempt(
            str(uuid4()), root.book_id, state=state, generation=scope.generation,
            conditions_id=root.conditions_id, root_id=root.id, scope_id=scope.id,
            run_id=root.manifest['active_run_id'],
        ))


def test_web_restart_refuses_in_flight_send_without_acknowledgement(
    configured_client, configured_application,
):
    sent_scenario, sent_root = restartable_unknown_root(
        configured_application, book_id='inflight-sent', reason='limit',
    )
    _plant_attempt(configured_application, sent_root, state='sent')
    sent_path = f'/books/{sent_scenario.book_id}/recovery/{sent_root.id}/restart'
    denied = configured_client.post(sent_path, json=dict(
        expected_generation=sent_root.generation, confirmed=True, acknowledge_unknown_outcome=False,
    ))
    assert denied.status_code == 409, denied.text
    assert denied.json()['detail']['reason_code'] == 'state_conflict'
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.get_root(sent_root.id).superseded_by is None
    acknowledged = configured_client.post(sent_path, json=dict(
        expected_generation=sent_root.generation, confirmed=True, acknowledge_unknown_outcome=True,
    ))
    assert acknowledged.status_code == 202, acknowledged.text
    assert acknowledged.json()['successor_root_id']

    claimed_scenario, claimed_root = restartable_unknown_root(
        configured_application, book_id='inflight-claimed', reason='scope_limit',
    )
    _plant_attempt(configured_application, claimed_root, state='created', claim=True)
    claimed_path = f'/books/{claimed_scenario.book_id}/recovery/{claimed_root.id}/restart'
    claimed = configured_client.post(claimed_path, json=dict(
        expected_generation=claimed_root.generation, confirmed=True,
        acknowledge_unknown_outcome=False,
    ))
    assert claimed.status_code == 409, claimed.text
    assert claimed.json()['detail']['reason_code'] == 'state_conflict'
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.get_root(claimed_root.id).superseded_by is None
    claimed_ack = configured_client.post(claimed_path, json=dict(
        expected_generation=claimed_root.generation, confirmed=True,
        acknowledge_unknown_outcome=True,
    ))
    assert claimed_ack.status_code == 202, claimed_ack.text


def test_web_extend_limit_uses_application_use_case_and_preserves_reason_codes(
    configured_client, configured_application, monkeypatch,
):
    scenario, root = recovery_fixture(configured_client, configured_application, book_id='limit-book')
    unknown_scenario, unknown_root = recovery_fixture(
        configured_client, configured_application, unknown=True, book_id='unknown-limit-book',
    )
    seen = []
    use_case = configured_application.extend_recovery_limit
    original = ExtendRecoveryLimit.__call__

    def spy(self, command):
        if self is use_case:
            seen.append(command)
        return original(self, command)

    monkeypatch.setattr(ExtendRecoveryLimit, '__call__', spy)
    payload = dict(additional_calls=2, expected_generation=0, confirmed=True)
    missing = configured_client.post(f'/books/missing/recovery/{root.id}/limit', json=payload)
    assert missing.status_code == 404, missing.text
    assert missing.json()['detail']['reason_code'] == 'not_found'
    store_book(configured_application, book_id='other')
    foreign = configured_client.post(f'/books/other/recovery/{root.id}/limit', json=payload)
    assert foreign.status_code == 404, foreign.text
    assert foreign.json()['detail']['reason_code'] == 'not_found'
    extended = configured_client.post(
        f'/books/{scenario.book_id}/recovery/{root.id}/limit', json=payload,
    )
    assert extended.status_code == 200, extended.text
    assert extended.json()['limit'] == 14 and extended.json()['generation'] == 1
    stale = configured_client.post(
        f'/books/{scenario.book_id}/recovery/{root.id}/limit', json=payload,
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()['detail']['reason_code'] == 'generation_conflict'
    unknown = configured_client.post(
        f'/books/{unknown_scenario.book_id}/recovery/{unknown_root.id}/limit', json=payload,
    )
    assert unknown.status_code == 409, unknown.text
    assert unknown.json()['detail']['reason_code'] == 'unknown'
    assert [type(command) for command in seen] == [ExtendRecoveryLimitCommand] * 5
    assert [(command.book_id, command.root_id) for command in seen] == [
        ('missing', root.id),
        ('other', root.id),
        (scenario.book_id, root.id),
        (scenario.book_id, root.id),
        (unknown_scenario.book_id, unknown_root.id),
    ]


def test_unproven_price_has_no_review_or_task(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    route = configured_application.quality_review.routes['test', 'review']
    configured_application.quality_review.routes['test', 'review'] = replace(route, proof='')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', json=command)
    assert response.status_code == 409 and response.json()['detail']['reason_code'] == 'price_unavailable'
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == () and tx.tasks.list_for_book(scenario.book_id) == ()


def test_web_strict_json_content_type_invalid_limit_422_and_stopped_other_unknown_restart(
    configured_client, configured_application,
):
    import json
    from linguary.domain import ErrorType, ProcessingTask, TaskError, TaskStatus, TaskType

    # 1. Overflow sum in POST .../limit returns 422 invalid_limit
    lim_scenario, lim_root = recovery_fixture(configured_client, configured_application, book_id='overflow-limit')
    overflow = configured_client.post(
        f'/books/{lim_scenario.book_id}/recovery/{lim_root.id}/limit',
        json=dict(additional_calls=9223372036854775807, expected_generation=0, confirmed=True),
    )
    assert overflow.status_code == 422, overflow.text
    assert overflow.json()['detail']['reason_code'] == 'invalid_limit'

    # 2. Strict JSON validation on /restart + case-insensitive Content-Type + stopped other reason with unknown attempt
    other_scenario, other_root = restartable_unknown_root(
        configured_application, book_id='other-stop-unknown', reason='validation_error',
    )
    _plant_attempt(configured_application, other_root, state='unknown')
    path = f'/books/{other_scenario.book_id}/recovery/{other_root.id}/restart'

    bad_bool = configured_client.post(
        path,
        json=dict(expected_generation=other_root.generation, confirmed=True, acknowledge_unknown_outcome='yes'),
    )
    assert bad_bool.status_code == 422, bad_bool.text

    no_ack = configured_client.post(
        path,
        json=dict(expected_generation=other_root.generation, confirmed=True, acknowledge_unknown_outcome=False),
    )
    assert no_ack.status_code == 409, no_ack.text
    assert no_ack.json()['detail']['reason_code'] == 'unknown'

    ok = configured_client.post(
        path,
        content=json.dumps(dict(
            expected_generation=other_root.generation, confirmed=True, acknowledge_unknown_outcome=True,
        )),
        headers={'Content-Type': 'Application/JSON; charset=utf-8'},
    )
    assert ok.status_code == 202, ok.text
    assert ok.json()['successor_root_id']

    # 3. /tasks/{id}/retry with Application/JSON empty body vs invalid JSON / non-bool flag
    with configured_application.unit_of_work.transaction() as tx:
        retryable_task = ProcessingTask(
            str(uuid4()), other_scenario.book_id, TaskType.ANALYZE_CHAPTER, 'section-1',
            status=TaskStatus.FAILED, attempts=1,
            last_error=TaskError(ErrorType.TRANSIENT, 'временный сбой'),
        )
        tx.tasks.add(retryable_task)
        task_id = retryable_task.id

    bad_json = configured_client.post(
        f'/tasks/{task_id}/retry',
        content='{broken',
        headers={'Content-Type': 'Application/JSON'},
    )
    assert bad_json.status_code == 422

    bad_flag = configured_client.post(
        f'/tasks/{task_id}/retry',
        content=json.dumps({'acknowledge_unknown_outcome': 'true'}),
        headers={'Content-Type': 'Application/JSON; charset=utf-8'},
    )
    assert bad_flag.status_code == 422

    empty_json = configured_client.post(
        f'/tasks/{task_id}/retry',
        content=b'',
        headers={'Content-Type': 'Application/JSON'},
    )
    assert empty_json.status_code == 200, empty_json.text
    assert empty_json.json()['status'] == 'pending'
