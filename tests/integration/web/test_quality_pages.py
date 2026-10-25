from dataclasses import asdict, replace
import json
import pytest
from fastapi.testclient import TestClient
from web_scenario import store_book, start_run, translated_result
from test_quality_api import setup_review, recovery_fixture
from linguary.adapters.config import load_settings
from linguary.adapters.config.settings import RecoverySettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.composition import build_application, drain_tasks
from linguary.domain import BookStatus, TaskStatus, LLMCallStage
from linguary.web import create_app

HTML = {'accept': 'text/html'}


def test_unknown_legacy_level_label(client, application):
    scenario = store_book(application)
    response = client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Уровень читателя неизвестен' in response.text
    assert 'Сохранённые условия отсутствуют' in response.text
    assert 'Продолжить со старыми материалами' in response.text
    assert 'Новый анализ заменит текущий анализ' in response.text
    assert 'Подтверждаю новый анализ' in response.text
    with application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id) is None and tx.tasks.list_for_book(scenario.book_id) == ()


def test_review_form_off_and_requires_budget(configured_client, configured_application, llm_client):
    scenario, _ = setup_review(configured_client, configured_application)
    response = configured_client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Добровольная вычитка выключена' in response.text
    assert 'name="budget_usd"' in response.text and 'name="budget_microusd"' not in response.text
    assert 'Предел расходов, USD' in response.text
    assert 'route_selection' in response.text and 'review' in response.text
    assert 'Оценка одного обращения' in response.text
    assert 'не гарантирует' in response.text
    assert 'Исправление оплачивается отдельно' in response.text
    assert 'name="scope"' in response.text
    assert llm_client.requests == []


@pytest.mark.parametrize('budget', ['0', '-1', 'NaN', 'Infinity', '0.0000001', '1e999999999'])
def test_invalid_html_usd_never_enqueues(configured_client, configured_application, budget):
    scenario, command = setup_review(configured_client, configured_application)
    command.pop('budget_microusd')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review',
        data={**command, 'budget_usd': budget}, headers=HTML)
    assert response.status_code == 422
    assert 'Введите положительную сумму USD' in response.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == ()


def test_html_usd_exactly_preserves_user_limit(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    command.pop('budget_microusd')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review',
        data={**command, 'budget_usd': '0.001201'}, headers=HTML, follow_redirects=False)
    assert response.status_code == 303
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id)[0].budget_limit == 1201


def test_recovery_limit_explained(configured_client, configured_application):
    scenario, root = recovery_fixture(configured_client, configured_application)
    response = configured_client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Покрытие: 0 из 2' in response.text
    assert 'Обращения: 12 из 12' in response.text
    assert 'Достигнут предел обращений' in response.text
    assert f'/recovery/{root.id}/limit' in response.text
    assert 'Подтверждаю увеличение предела' in response.text
    assert 'не запускает повтор автоматически' in response.text


def test_production_v3_env_is_explicit_before_http_analysis(settings, monkeypatch):
    monkeypatch.setenv('LINGUARY_CONTENT_PLAN_PROFILE', 'content-plan-v3')
    app = build_application(settings=replace(settings, llm=replace(settings.llm, model='fake', api_key='synthetic')),
        llm_client=FakeLLMClient())
    try:
        scenario = store_book(app, status=BookStatus.IMPORTED, analysed_chapters=())
        with TestClient(create_app(application=app, run_background_worker=False), base_url=app.settings.server.base_url) as client:
            response = client.post(f'/books/{scenario.book_id}/analyze')
            assert response.status_code == 202, response.text
        with app.unit_of_work.transaction() as tx:
            conditions = tx.work.latest_conditions(scenario.book_id)
            assert conditions.payload['profile'] == 'content-plan-v3'
            assert conditions.payload['versions']['schemas'] == 3
            assert conditions.payload['recovery_policy'] == asdict(app.settings.recovery)
            assert tx.work.get_binding('task', response.json()['id'])['conditions_id'] == conditions.id
    finally:
        app.dispose()


async def test_recovery_env_is_pinned_across_restart_and_actual_calls(tmp_path, monkeypatch):
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from linguary.application.use_cases.retry_task import RetryTaskCommand
    config = tmp_path / 'config.toml'
    config.write_text(f'[storage]\ndata_root={json.dumps(str(tmp_path / "data"))}\n[llm]\nmodel="fake"\napi_key="synthetic"\n')
    initial = load_settings(config_path=config, environ={'LINGUARY_RECOVERY_MAX_CALLS': '1', 'LINGUARY_RECOVERY_MAX_SCOPE_CALLS': '1'})
    fake = FakeLLMClient()
    app = build_application(settings=initial, llm_client=fake)
    scenario = store_book(app, status=BookStatus.IMPORTED, analysed_chapters=())
    fake.replies = [FakeReply.of_payload(dict(translation_instructions='Сохраняй смысл.', chapter_summary='A scene.',
        glossary_candidates=[], before_you_read=[], treatment_plan=[dict(source_block_id=f'b{index}', treatment='TRANSLATE') for index, _ in enumerate(scenario.block_ids(), 1)])), FakeReply(content='{')]
    await app.analysis.start(scenario.book_id)
    analysed = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in analysed), [task.last_error for task in analysed]
    with app.unit_of_work.transaction() as tx:
        conditions = tx.work.latest_conditions(scenario.book_id)
        assert conditions.payload['recovery_policy']['max_calls'] == 1
    app.start_processing()(StartProcessingCommand(scenario.book_id))
    task, = await app.processing.start(scenario.book_id)
    await drain_tasks(app)
    with app.unit_of_work.transaction() as tx:
        root, = tx.work.list_roots(scenario.book_id)
        assert root.calls_used == 1 and root.reason == 'limit'
    assert sum(request.stage == LLMCallStage.TRANSLATION_UNIT for request in fake.requests) == 1
    await app.aclose()
    changed = load_settings(config_path=config, environ={'LINGUARY_RECOVERY_MAX_CALLS': '99', 'LINGUARY_RECOVERY_MAX_SCOPE_CALLS': '99'})
    restarted_fake = FakeLLMClient()
    restarted = build_application(settings=changed, llm_client=restarted_fake)
    try:
        restarted.retry_task(RetryTaskCommand(task.id))
        await drain_tasks(restarted)
        with restarted.unit_of_work.transaction() as tx:
            assert tx.work.get_root(root.id) == root
            current = tx.work.latest_conditions(scenario.book_id)
        assert restarted.book_runtime.settings_for(current).recovery.max_calls == 1
        legacy = replace(current, payload={k: v for k, v in current.payload.items() if k != 'recovery_policy'})
        assert restarted.book_runtime.settings_for(legacy).recovery == RecoverySettings()
        assert restarted_fake.requests == []
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('change,message', [({'route_id': 'missing'}, 'Выбранная модель или зарегистрированный маршрут недоступны'),
    ({'expected_manifest_sha': '0' * 64}, 'Материалы книги изменились'), ({'budget_microusd': 1}, 'Указанного предела расходов недостаточно')])
def test_html_review_conflicts_are_russian_and_do_not_start(configured_client, configured_application, change, message):
    scenario, command = setup_review(configured_client, configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data={**command, **change}, headers=HTML)
    assert response.status_code == 409
    assert message in response.text and 'reason_code' not in response.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == () and tx.tasks.list_for_book(scenario.book_id) == ()


def test_html_unproven_price_is_russian(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    route = configured_application.quality_review.routes['test', 'review']
    configured_application.quality_review.routes['test', 'review'] = replace(route, proof='')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data=command, headers=HTML)
    assert response.status_code == 409 and 'Нет подтверждённой цены' in response.text and 'reason_code' not in response.text


def test_reanalyze_confirmation_and_html_stale_revision(configured_client, configured_application, llm_client):
    from lxml import html
    scenario, _ = setup_review(configured_client, configured_application)
    tree = html.fromstring(configured_client.get(f'/books/{scenario.book_id}', headers=HTML).text)
    form = tree.xpath('//form[input[@name="action" and @value="reanalyze"]]')[0]
    confirmation = form.xpath('.//input[@name="confirmed"]')[0]
    assert confirmation.get('required') is not None and confirmation.get('checked') is None
    values = {field.get('name'): field.get('value') for field in form.xpath('.//input[@type="hidden"]')}
    response = configured_client.post(form.get('action'), data={**values, 'confirmed': 'false'}, headers=HTML)
    assert response.status_code == 422 and 'reason_code' not in response.text
    response = configured_client.post(form.get('action'), data={**values, 'expected_revision': '0', 'confirmed': 'true'}, headers=HTML)
    assert response.status_code == 409 and 'Условия книги изменились' in response.text and 'reason_code' not in response.text
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id).revision == 1
        assert tx.analyses.list_for_book(scenario.book_id) and tx.tasks.list_for_book(scenario.book_id) == ()
    assert llm_client.requests == []


def test_no_internal_ids_in_learning_text(client, application):
    from lxml import html
    scenario = store_book(application, status=BookStatus.READY_TO_EXPORT)
    start_run(application, scenario, results=[translated_result(b) for b in scenario.block_ids()])
    tree = html.fromstring(client.get(f'/books/{scenario.book_id}/result', headers=HTML).text)
    layers = tree.xpath('//p[@class="source" or @class="translation"]')
    visible = ' '.join(layer.text_content() for layer in layers)
    assert 'It is a truth universally acknowledged.' in visible and 'Общепризнанная истина' in visible
    assert all(block_id not in visible for block_id in scenario.block_ids())
    assert all(marker not in visible for marker in ('source_block_id', 'context_facts', 'required_points', 'artifact_sha'))


def test_scope_selection_get_is_read_only(configured_client, configured_application, llm_client):
    from lxml import html
    scenario, _ = setup_review(configured_client, configured_application)
    expected = configured_application.quality_review.preview(scenario.book_id, scenario.block_ids())
    response = configured_client.get(f'/books/{scenario.book_id}/result', params={'review_scope': scenario.section().id}, headers=HTML)
    assert response.status_code == 200
    tree = html.fromstring(response.text)
    form = tree.xpath('//form[contains(@action,"/quality-review")]')[0]
    assert json.loads(form.xpath('.//input[@name="scope"]')[0].get('value')) == list(scenario.block_ids())
    assert form.xpath('.//input[@name="expected_manifest_sha"]')[0].get('value') == expected['manifest_sha']
    assert not form.xpath('ancestor::*[@data-result-header]')
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.latest_conditions(scenario.book_id).revision == 1
        assert tx.tasks.list_for_book(scenario.book_id) == () and tx.work.list_reviews(scenario.book_id) == ()
    assert llm_client.requests == []


@pytest.mark.parametrize('status', [BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED])
def test_incomplete_html_does_not_claim_current_epub_is_ready(client, application, status):
    scenario = store_book(application, status=status)
    start_run(application, scenario, results=[translated_result(scenario.block_ids()[0])])
    response = client.get(f'/books/{scenario.book_id}', headers=HTML)
    assert 'Книга готова к сборке EPUB' not in response.text
    assert 'Итоговый EPUB собран.' not in response.text
    assert 'полный EPUB пока недоступен' in response.text


def test_saved_conditions_visible_on_detail_result_and_poll(configured_client, configured_application):
    scenario, _ = setup_review(configured_client, configured_application)
    for endpoint, headers in [('', HTML), ('/result', HTML),
            ('/result', {**HTML, 'x-linguary-fragment': 'result-header'})]:
        response = configured_client.get(f'/books/{scenario.book_id}{endpoint}', headers=headers)
        assert response.status_code == 200
        assert 'Редакция условий: 1' in response.text
        assert 'Профиль обработки: legacy-v1' in response.text
        assert 'Модель анализа: fake-model' in response.text
        assert 'Модель обработки: fake-model' in response.text


def test_html_uploaded_budget_is_rejected_without_queue(configured_client, configured_application):
    scenario, command = setup_review(configured_client, configured_application)
    command.pop('budget_microusd')
    response = configured_client.post(f'/books/{scenario.book_id}/quality-review', data=command,
        files={'budget_usd': ('budget.txt', b'1.00', 'text/plain')}, headers=HTML)
    assert response.status_code == 422
    with configured_application.unit_of_work.transaction() as tx:
        assert tx.work.list_reviews(scenario.book_id) == ()
        assert tx.tasks.list_for_book(scenario.book_id) == ()
