from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4
import pytest
from web_scenario import SOURCE_TEXT, store_book, start_run, translated_result
from test_quality_api import setup_review, recovery_fixture
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.domain import (
    BookStatus, ErrorType, LLMCallLog, LLMCallStage, ProcessingTask, TaskError, TaskStatus, TaskType,
)
from linguary.domain.quality_review import QualityReview
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope


def ready_book(app):
    scenario = store_book(app, status=BookStatus.READY_TO_EXPORT)
    start_run(app, scenario, results=[translated_result(block) for block in scenario.block_ids()])
    return scenario


def test_epub_ready_with_review_off_or_findings(client, application, llm_client):
    scenario = ready_book(application)
    for endpoint in ('', '/result'):
        response = client.get(f'/books/{scenario.book_id}{endpoint}')
        assert response.status_code == 200
        value = response.json()
        assert value['epub_ready'] is True
        assert value['quality_review']['state'] == 'not-requested'
        assert value['reproducibility'] == 'unknown'
        assert value['conditions']['reader_level'] is None
    assert llm_client.requests == []


@pytest.mark.parametrize('results', [0, 1])
def test_incomplete_book_not_ready(client, application, results):
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(application, scenario, results=[translated_result(block) for block in scenario.block_ids()[:results]])
    assert client.get(f'/books/{scenario.book_id}').json()['epub_ready'] is False
    assert client.get(f'/books/{scenario.book_id}/result').json()['epub_ready'] is False


@pytest.mark.parametrize('state', ['limit', 'unknown', 'stale', 'completed'])
def test_limit_unknown_stale_distinct(configured_client, configured_application, llm_client, state):
    scenario, _command = setup_review(configured_client, configured_application)
    manifest = configured_application.quality_review.preview(scenario.book_id)
    with configured_application.unit_of_work.transaction() as tx:
        review = QualityReview(str(uuid4()), scenario.book_id, 'projection', manifest,
            dict(schema_version=1), 10000, manifest['conditions_id'], state=state,
            findings=({'issue': 'Possible error'},) if state == 'completed' else ())
        tx.work.add_review(review)
        tx.work.update_review(review.id, state=state,
            findings=review.findings, checked=[a['artifact_sha'] for a in manifest['artifacts']])
    for path in ('', '/result'):
        value = configured_client.get(f'/books/{scenario.book_id}{path}').json()
        assert value['quality_review']['state'] == ('completed-with-findings' if state == 'completed' else state)
        assert value['reproducibility'] == 'legacy_partial'
        assert value['conditions']['reader_level'] == configured_application.settings.reading.normalized_level
        assert value['epub_ready'] is False
    assert llm_client.requests == []


def test_recovery_counters_read_only_and_separate_costs(configured_client, configured_application, llm_client):
    scenario, root = recovery_fixture(configured_client, configured_application)
    with configured_application.unit_of_work.transaction() as tx:
        tx.llm_calls.add(LLMCallLog('ordinary', scenario.book_id, LLMCallStage.BLOCK_REGENERATION,
            'fake', 'fake', datetime.now(UTC), 1, 1, True, cost=Decimal('0.1')))
        tx.llm_calls.add(LLMCallLog('optional', scenario.book_id, LLMCallStage.QUALITY_REVIEW,
            'fake', 'fake', datetime.now(UTC), 1, 1, True, cost=Decimal('0.2')),
            response_text='PRIVATE DIAGNOSTIC BODY MUST NOT LEAK')
    value = configured_client.get(f'/books/{scenario.book_id}').json()
    recovery = value['recovery']
    assert recovery['total'] == 2 and recovery['covered'] == 0
    assert recovery['calls_used'] == 12 and recovery['limit'] == 12
    assert recovery['reason'] == 'limit' and recovery['can_extend'] is True and recovery['can_retry'] is False
    assert value['ordinary_cost']['known_spent_usd'] == '0.1'
    assert value['ordinary_cost']['regeneration_spent_usd'] == '0.1'
    assert 'PRIVATE DIAGNOSTIC' not in str(value)
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.get_root(root.id) == root
        assert tx.tasks.list_for_book(scenario.book_id) == ()
    assert llm_client.requests == []


def test_legacy_client_fields_unchanged(client, application):
    scenario = ready_book(application)
    detail = client.get(f'/books/{scenario.book_id}').json()
    assert {'book', 'progress', 'export_validation_status', 'warning_count'} <= detail.keys()
    result = client.get(f'/books/{scenario.book_id}/result').json()
    assert {'book_id', 'status', 'warnings', 'chapters', 'export_validation_status'} <= result.keys()
    assert result['chapters'][0]['blocks'][0]['translated_text']
    with application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id) is None


@pytest.mark.parametrize('state', ['running', 'unknown', 'limit', 'completed'])
def test_actual_epub_export_with_active_optional_task(configured_client, configured_application, llm_client, tmp_path, state):
    from zipfile import ZipFile
    from linguary.application.use_cases.export_book import ExportBookCommand
    from linguary.domain import ProcessingTask, TaskType
    scenario, _ = setup_review(configured_client, configured_application)
    start_run(
        configured_application,
        scenario,
        results=[translated_result(block) for block in scenario.block_ids()],
    )
    with configured_application.unit_of_work.transaction() as tx:
        book = tx.books.get(scenario.book_id)
        tx.books.save(replace(book, status=BookStatus.READY_TO_EXPORT))
    manifest = configured_application.quality_review.preview(scenario.book_id)
    review = QualityReview(str(uuid4()), scenario.book_id, 'actual-export', manifest,
        dict(schema_version=1), 100000, manifest['conditions_id'], state=state,
        findings=({'issue': 'Possible problem'},) if state == 'completed' else ())
    with configured_application.unit_of_work.transaction() as tx:
        tx.work.add_review(review)
        tx.work.update_review(review.id, state=state, findings=review.findings,
            checked=[a['artifact_sha'] for a in manifest['artifacts']])
        tx.tasks.add(ProcessingTask(id='active-optional-task', book_id=scenario.book_id,
            task_type=TaskType.QUALITY_REVIEW, scope_id=review.id).start())
    before_calls = len(llm_client.requests)
    value = configured_client.get(f'/books/{scenario.book_id}').json()
    assert value['epub_ready'] is True
    assert value['quality_review']['state'] == ('completed-with-findings' if state == 'completed' else state)
    assert value['progress']['running_tasks'] == 1
    queued = configured_client.post(f'/books/{scenario.book_id}/export')
    assert queued.status_code == 202, queued.text
    path = tmp_path / 'actual.epub'
    exported = configured_application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    assert exported.path == str(path)
    with ZipFile(path) as epub:
        assert epub.read('mimetype') == b'application/epub+zip'
        assert any('Общепризнанная истина' in epub.read(name).decode('utf-8') for name in epub.namelist() if name.endswith('.xhtml'))
    assert len(llm_client.requests) == before_calls


def test_incomplete_legacy_export_is_409(client, application, tmp_path):
    from linguary.application.use_cases.export_book import ExportBookCommand, ExportBookError
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(application, scenario, results=[translated_result(scenario.block_ids()[0])])
    assert client.get(f'/books/{scenario.book_id}').json()['epub_ready'] is False
    rejected = client.post(f'/books/{scenario.book_id}/export')
    assert rejected.status_code == 409, rejected.text
    path = tmp_path / 'partial.epub'
    with pytest.raises(ExportBookError):
        application.export_book(ExportBookCommand(scenario.book_id, str(path)))
    assert not path.exists()


def test_complete_root_projection_does_not_mark_it_complete(configured_client, configured_application):
    from linguary.domain.recovery import RecoveryRoot
    scenario, _ = setup_review(configured_client, configured_application)
    start_run(configured_application, scenario, results=[translated_result(block) for block in scenario.block_ids()])
    with configured_application.unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(scenario.book_id)
        book = tx.books.get(scenario.book_id)
        tx.books.save(replace(book, status=BookStatus.READY_TO_EXPORT))
        root = RecoveryRoot(str(uuid4()), scenario.book_id, conditions.id, str(uuid4()),
            dict(schema_version=1, active_run_id=scenario.run_id, ordered_blocks=list(scenario.block_ids()),
                historical_coverage=list(scenario.block_ids()), policy=dict(max_calls=12, max_scope_calls=3)))
        tx.work.add_root(root)
    for path in ('', '/result'):
        value = configured_client.get(f'/books/{scenario.book_id}{path}').json()
        assert value['epub_ready'] is True and value['recovery']['covered'] == 2
        assert value['recovery']['roots'][0]['complete'] is True
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.get_root(root.id) == root


def _prepared_book(client, app, *, chapters=None):
    scenario = store_book(app, chapters=chapters) if chapters is not None else store_book(app)
    response = client.post(
        f'/books/{scenario.book_id}/conditions',
        json=dict(action='continue_legacy', expected_revision=0, settings_revision=0, confirmed=True),
    )
    assert response.status_code == 201, response.text
    start_run(app, scenario)
    with app.unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(scenario.book_id)
    return scenario, conditions


def _manifest(run_id, blocks, *, max_calls=3, max_scope_calls=3):
    return dict(
        schema_version=1, active_run_id=run_id, ordered_blocks=list(blocks), historical_coverage=[],
        policy=dict(max_calls=max_calls, max_scope_calls=max_scope_calls, max_depth=3, max_leaves=8),
    )


def _save_root(tx, scenario, conditions, blocks, *, status, reason, calls_used=0, max_calls=3,
               max_scope_calls=3, scope_status='pending'):
    root = RecoveryRoot(
        str(uuid4()), scenario.book_id, conditions.id, str(uuid4()),
        _manifest(scenario.run_id, blocks, max_calls=max_calls, max_scope_calls=max_scope_calls),
        calls_used=calls_used, status=status, reason=reason, policy_version=CURRENT_TRANSLATION_POLICY,
    )
    scope = RecoveryScope(str(uuid4()), root.id, tuple(blocks), status=scope_status)
    tx.work.add_root(root)
    tx.work.add_scope(scope)
    return root, scope


def _save_attempt(tx, scenario, conditions, root, scope, *, state, ordinal):
    attempt = CallAttempt(
        str(uuid4()), scenario.book_id, state=state, ordinal=ordinal, generation=scope.generation,
        conditions_id=conditions.id, root_id=root.id, scope_id=scope.id, run_id=scenario.run_id,
    )
    tx.work.add_attempt(attempt)
    return attempt


def _save_failed_task(tx, scenario, conditions, roots, *, section_id, task_type):
    task = ProcessingTask(
        str(uuid4()), scenario.book_id, task_type, section_id, status=TaskStatus.FAILED, attempts=1,
        last_error=TaskError(ErrorType.TRANSIENT, 'временный сбой обработки', 'detail'),
        operation_id=str(uuid4()),
    )
    tx.tasks.add(task)
    tx.work.bind(
        'task', task.id, book_id=scenario.book_id, conditions_id=conditions.id,
        payload={'schema_version': 1, 'operation_refs': [
            {'root_id': root.id, 'policy_version': root.policy_version} for root in roots
        ]},
    )
    return task


def test_in_flight_sent_attempt_not_shown_as_unknown(configured_client, configured_application):
    scenario, conditions = _prepared_book(configured_client, configured_application)
    blocks = scenario.block_ids()
    with configured_application.unit_of_work.transaction() as tx:
        root, scope = _save_root(
            tx, scenario, conditions, blocks, status='running', reason=None,
            max_calls=3, max_scope_calls=2,
        )
        _save_attempt(tx, scenario, conditions, root, scope, state='cancelled', ordinal=1)
        _save_attempt(tx, scenario, conditions, root, scope, state='sent', ordinal=2)
    value = configured_client.get(f'/books/{scenario.book_id}').json()
    recovery = value['recovery']
    projected = recovery['roots'][0]
    assert projected['in_flight'] is True
    assert projected['has_unknown_outcome'] is False
    assert projected['reason'] is None
    assert projected['status'] == 'running'
    assert projected['can_retry'] is False
    assert projected['section_id'] == scenario.section().id
    assert projected['chapter_title'] == 'Chapter 1'
    assert recovery['reason'] is None
    assert recovery['can_retry'] is False


def test_unknown_and_stopped_roots_expose_can_restart_and_chapter_metadata(
    configured_client, configured_application,
):
    scenario, conditions = _prepared_book(configured_client, configured_application)
    blocks = scenario.block_ids()
    stopped_reasons = (
        'limit', 'scope_limit', 'unknown_outcome', 'interrupted', 'unsafe_partition',
        'output_truncated', 'context_capacity', 'provider_refusal',
    )
    with configured_application.unit_of_work.transaction() as tx:
        stopped = {
            reason: _save_root(
                tx, scenario, conditions, blocks, status='stopped', reason=reason,
            )[0].id
            for reason in stopped_reasons
        }
        pending_unknown, pending_scope = _save_root(
            tx, scenario, conditions, blocks, status='pending', reason=None,
        )
        _save_attempt(
            tx, scenario, conditions, pending_unknown, pending_scope, state='unknown', ordinal=1,
        )
        stuck_id = _save_root(
            tx, scenario, conditions, blocks, status='pending', reason=None, scope_status='stopped',
        )[0].id
        disallowed_id = _save_root(
            tx, scenario, conditions, blocks, status='stopped', reason='transient_error',
        )[0].id
        unknown_on_other_stop, other_scope = _save_root(
            tx, scenario, conditions, blocks, status='stopped', reason='validation_error',
        )
        _save_attempt(
            tx, scenario, conditions, unknown_on_other_stop, other_scope, state='unknown', ordinal=1,
        )
    value = configured_client.get(f'/books/{scenario.book_id}').json()
    roots = {item['root_id']: item for item in value['recovery']['roots']}
    for reason, root_id in stopped.items():
        projected = roots[root_id]
        assert projected['can_restart'] is True
        assert projected['status'] == 'stopped'
        assert projected['section_id'] == scenario.section().id
        assert projected['chapter_title'] == 'Chapter 1'
        assert projected['in_flight'] is False
        assert projected['has_unknown_outcome'] is (reason in ('unknown_outcome', 'interrupted'))
        if reason in ('unknown_outcome', 'interrupted'):
            assert projected['reason'] == reason
    pending = roots[pending_unknown.id]
    assert pending['status'] == 'pending'
    assert pending['has_unknown_outcome'] is True
    assert pending['reason'] == 'unknown'
    assert pending['can_restart'] is True
    assert pending['can_retry'] is False
    assert pending['section_id'] == scenario.section().id
    assert pending['chapter_title'] == 'Chapter 1'
    assert pending['in_flight'] is False
    stuck = roots[stuck_id]
    assert stuck['can_restart'] is True
    assert stuck['can_retry'] is False
    assert stuck['has_unknown_outcome'] is False
    assert stuck['status'] == 'pending'
    assert roots[disallowed_id]['can_restart'] is False
    assert roots[disallowed_id]['has_unknown_outcome'] is False
    unknown_stop = roots[unknown_on_other_stop.id]
    assert unknown_stop['has_unknown_outcome'] is True
    assert unknown_stop['can_restart'] is True
    assert unknown_stop['reason'] == 'unknown'


def test_one_unknown_root_does_not_disable_can_retry_for_other_chapters(
    configured_client, configured_application,
):
    scenario, conditions = _prepared_book(
        configured_client, configured_application, chapters=((SOURCE_TEXT,), (SOURCE_TEXT,)),
    )
    with configured_application.unit_of_work.transaction() as tx:
        unknown_root, unknown_scope = _save_root(
            tx, scenario, conditions, scenario.block_ids(0), status='stopped', reason='unknown_outcome',
            calls_used=1,
        )
        _save_attempt(
            tx, scenario, conditions, unknown_root, unknown_scope, state='unknown', ordinal=1,
        )
        healthy_root, _healthy_scope = _save_root(
            tx, scenario, conditions, scenario.block_ids(1), status='pending', reason='transient_error',
        )
        blocked = _save_failed_task(
            tx, scenario, conditions, (unknown_root,), section_id=scenario.section(0).id,
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        )
        allowed = _save_failed_task(
            tx, scenario, conditions, (healthy_root,), section_id=scenario.section(1).id,
            task_type=TaskType.REGENERATE_BLOCK,
        )
    value = configured_client.get(f'/books/{scenario.book_id}').json()
    roots = {item['root_id']: item for item in value['recovery']['roots']}
    failures = {item['task_id']: item for item in value['progress']['failures']}
    assert roots[unknown_root.id]['chapter_title'] == 'Chapter 1'
    assert roots[unknown_root.id]['section_id'] == scenario.section(0).id
    assert roots[unknown_root.id]['has_unknown_outcome'] is True
    assert roots[unknown_root.id]['can_retry'] is False
    assert roots[unknown_root.id]['can_restart'] is True
    assert roots[healthy_root.id]['chapter_title'] == 'Chapter 2'
    assert roots[healthy_root.id]['section_id'] == scenario.section(1).id
    assert roots[healthy_root.id]['has_unknown_outcome'] is False
    assert roots[healthy_root.id]['in_flight'] is False
    assert roots[healthy_root.id]['can_retry'] is True
    assert value['recovery']['can_retry'] is True
    assert failures[blocked.id]['is_retryable'] is False
    assert failures[allowed.id]['is_retryable'] is True


def test_recovery_reason_prioritizes_unknown_and_interrupted_over_limit(
    configured_client, configured_application,
):
    scenario, conditions = _prepared_book(
        configured_client, configured_application, chapters=((SOURCE_TEXT,), (SOURCE_TEXT,)),
    )
    with configured_application.unit_of_work.transaction() as tx:
        _save_root(
            tx, scenario, conditions, scenario.block_ids(0), status='stopped', reason='limit',
            calls_used=3, max_calls=3,
        )
        _save_root(
            tx, scenario, conditions, scenario.block_ids(1), status='stopped', reason='interrupted',
            calls_used=1, max_calls=3,
        )
    value = configured_client.get(f'/books/{scenario.book_id}').json()
    assert value['recovery']['reason'] == 'interrupted'


def test_stopped_regeneration_root_does_not_block_epub_ready_or_export(
    configured_client, configured_application, tmp_path,
):
    from dataclasses import replace
    from linguary.application.book_overview import book_overview
    from linguary.application.use_cases.export_book import ExportBookCommand
    from linguary.domain import BookStatus
    from linguary.domain.recovery import RecoveryRoot, RecoveryScope
    from web_scenario import translated_result

    scenario, conditions = _prepared_book(configured_client, configured_application)
    blocks = scenario.block_ids()
    with configured_application.unit_of_work.transaction() as tx:
        tx.results.replace_results(
            scenario.run_id,
            tuple(translated_result(block_id) for block_id in blocks),
        )
        # Add a stopped regeneration root without any unfinished regeneration tasks.
        regen_root = RecoveryRoot(
            str(uuid4()), scenario.book_id, conditions.id, str(uuid4()),
            {
                'schema_version': 1,
                'active_run_id': scenario.run_id,
                'ordered_blocks': [blocks[0]],
                'atomic_intervals': [[blocks[0]]],
                'historical_coverage': [],
                'policy': {'max_calls': 3, 'max_scope_calls': 3, 'max_depth': 1, 'max_leaves': 1},
            },
            calls_used=1, status='stopped', reason='provider_error',
            policy_version=CURRENT_TRANSLATION_POLICY, operation_kind='regeneration',
        )
        tx.work.add_root(regen_root)
        tx.work.add_scope(RecoveryScope(str(uuid4()), regen_root.id, (blocks[0],), status='stopped'))
        book = tx.books.get(scenario.book_id)
        tx.books.save(replace(book, status=BookStatus.READY_TO_EXPORT))

    overview = book_overview(configured_application.unit_of_work, scenario.book_id)
    assert overview.epub_ready is True
    exported = configured_application.export_book(
        ExportBookCommand(book_id=scenario.book_id, target_path=str(tmp_path / 'out.epub'))
    )
    assert exported.path == str(tmp_path / 'out.epub')
