from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from lxml import html
from web_scenario import start_run, translated_result
from test_quality_api import setup_review
from test_local_diagnostics import connect, seed, BODY
from linguary.adapters.config import ServerSettings
from linguary.application.diagnostics import DiagnosticsPolicy
from linguary.composition import build_application
from linguary.domain import BookStatus
from linguary.domain.quality_review import QualityReview
from linguary.web import create_app

HTML = {'accept': 'text/html'}
FRAGMENT = {**HTML, 'x-linguary-fragment': 'result-header'}


@pytest.mark.parametrize('state', ['running', 'unknown', 'limit', 'stale', 'completed'])
def test_polling_retains_quality_and_export_states(configured_client, configured_application, state):
    scenario, _ = setup_review(configured_client, configured_application)
    start_run(configured_application, scenario, results=[translated_result(b) for b in scenario.block_ids()])
    with configured_application.unit_of_work.transaction() as tx:
        book = tx.books.get(scenario.book_id)
        tx.books.save(replace(book, status=BookStatus.READY_TO_EXPORT))
    manifest = configured_application.quality_review.preview(scenario.book_id)
    finding = dict(issue='<script>unsafe()</script>', suggestion='Проверьте время', uncertainty='Возможная ошибка', evidence_anchors=[])
    with configured_application.unit_of_work.transaction() as tx:
        review = QualityReview(str(uuid4()), scenario.book_id, 'poll', manifest, dict(schema_version=1),
            1234567, manifest['conditions_id'], state=state, findings=(finding,) if state == 'completed' else (),
            spent=100, reserved=200 if state == 'unknown' else 0)
        tx.work.add_review(review)
        tx.work.update_review(review.id, state=state, findings=review.findings,
            checked=[a['artifact_sha'] for a in manifest['artifacts']])
    page = configured_client.get(f'/books/{scenario.book_id}/result', headers=HTML)
    fragment = configured_client.get(f'/books/{scenario.book_id}/result', headers=FRAGMENT)
    assert page.status_code == fragment.status_code == 200
    for response in (page, fragment):
        assert 'Добровольная вычитка' in response.text and 'EPUB готов к сборке' in response.text
        assert '1.234567 USD' in response.text
        assert '<script>unsafe()' not in response.text
    assert 'name="budget_usd"' not in fragment.text
    assert 'Удержанный резерв: 0.000200 USD' in fragment.text if state == 'unknown' else True
    if state == 'completed':
        assert '&lt;script&gt;unsafe()&lt;/script&gt;' in fragment.text
        assert 'Вычитка завершена с замечаниями' in fragment.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.tasks.list_for_book(scenario.book_id) == ()


def test_diagnostic_html_escaped(application):
    _, path = seed(application)
    with connect(application) as client:
        response = client.get(path, headers=HTML)
        assert response.status_code == 200 and response.headers['content-type'].startswith('text/html')
        assert '<script>alert(' not in response.text and '&lt;script&gt;' in response.text
        assert response.headers['cache-control'] == 'no-store'
    with connect(application, '172.17.0.1') as client:
        assert client.get(path, headers=HTML).status_code == 403


@pytest.mark.parametrize('status,policy', [('disabled', DiagnosticsPolicy(bodies_enabled=False)),
    ('expired', DiagnosticsPolicy(retention_seconds=1)), ('deleted', None), ('unavailable', None)])
def test_missing_payload_never_says_false_rejection(application, status, policy):
    _, path = seed(application)
    if policy:
        application.diagnostics.policy = policy
        application.diagnostics.clock = lambda: datetime.now(UTC) + timedelta(seconds=5)
    elif status == 'deleted':
        application.diagnostics.purge('book-web-1', call_id='call-local')
    else:
        with application.unit_of_work.transaction() as tx:
            tx.llm_calls.store_diagnostic_body('call-local', None)
    with connect(application) as client:
        response = client.get(path, headers=HTML)
        assert response.status_code == 200
        assert 'По отсутствующему телу нельзя оценить обоснованность отказа' in response.text
        assert BODY not in response.text and 'private diagnostic' not in response.text
        expected = {'disabled': 'Хранение тел отключено', 'expired': 'Срок хранения истёк',
            'deleted': 'Тело удалено', 'unavailable': 'Тело не сохранено'}[status]
        assert expected in response.text


def test_no_body_in_result_fragments_and_docker_uses_cli(settings, llm_client):
    app = build_application(settings=replace(settings, server=ServerSettings(host='0.0.0.0')), llm_client=llm_client)
    try:
        scenario, path = seed(app)
        with app.unit_of_work.transaction() as tx:
            tx.llm_calls.update_diagnostics('call-local', metadata=dict(schema_version=1, decision='rejected',
                body_meta=dict(body_status='available')))
        with TestClient(create_app(application=app, run_background_worker=False), base_url='http://localhost:8000', client=('172.17.0.1', 50000)) as client:
            for endpoint, headers in [('', HTML), ('/result', HTML), ('/result', FRAGMENT)]:
                response = client.get(f'/books/{scenario.book_id}{endpoint}', headers=headers)
                assert 'private diagnostic' not in response.text
                assert path not in response.text
                if headers is HTML:
                    assert f'linguary diagnostics show {scenario.book_id} call-local' in response.text
            assert client.get(path, headers=HTML).status_code == 403
        assert llm_client.requests == []
    finally:
        app.dispose()


def test_real_rendered_form_uses_registered_route_scope_and_exact_usd(configured_client, configured_application, llm_client):
    scenario, _ = setup_review(configured_client, configured_application)
    response = configured_client.get(f'/books/{scenario.book_id}', headers=HTML)
    tree = html.fromstring(response.text)
    assert len(tree.xpath('//title')) == 1 and len(tree.xpath('//title/*')) == 0
    form = tree.xpath('//form[contains(@action,"/quality-review")]')[0]
    values = {field.get('name'): field.get('value') for field in form.xpath('.//input[@type="hidden"]')}
    values['scope'] = form.xpath('.//input[@name="scope"]')[0].get('value')
    values['route_selection'] = form.xpath('.//select[@name="route_selection"]/option[@value!=""]')[0].get('value')
    values['budget_usd'] = '0,100001'
    assert not form.xpath('ancestor::*[@data-result-header]')
    for name in ('scope', 'budget_usd', 'route_selection'):
        field = form.xpath(f'.//*[@name="{name}"]')[0]
        assert field.get('required') is not None and field.xpath('ancestor::label')
    sent = configured_client.post(form.get('action'), data=values, headers=HTML, follow_redirects=False)
    assert sent.status_code == 303, sent.text
    assert configured_client.post(form.get('action'), data=values, headers=HTML, follow_redirects=False).status_code == 303
    with configured_application.unit_of_work.transaction() as tx:
        review, = tx.work.list_reviews(scenario.book_id)
        assert review.budget_limit == 100001 and review.settings['model'] == 'review'
        assert len(tx.tasks.list_for_book(scenario.book_id)) == 1
    assert llm_client.requests == []


def test_diagnostic_html_discloses_truncation_and_redaction(application):
    _, path = seed(application)
    with application.unit_of_work.transaction() as tx:
        record = tx.llm_calls.diagnostic_record(call_id='call-local')
        metadata = dict(record['metadata'] or {})
        metadata['body_meta'] = {**metadata.get('body_meta', {}), 'truncated': True, 'redacted': True}
        tx.llm_calls.update_diagnostics('call-local', metadata=metadata)
    with connect(application) as client:
        response = client.get(path, headers=HTML)
        assert response.status_code == 200
        assert 'Ответ усечён по пределу хранения' in response.text
        assert 'Секретные данные удалены из диагностической копии' in response.text
