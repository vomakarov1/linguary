from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4
import pytest
from web_scenario import store_book, start_run, translated_result
from test_quality_api import setup_review, recovery_fixture
from linguary.domain import BookStatus, LLMCallLog, LLMCallStage
from linguary.domain.quality_review import QualityReview


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
    scenario, command = setup_review(configured_client, configured_application)
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
    start_run(configured_application, scenario, results=[translated_result(block) for block in scenario.block_ids()])
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
