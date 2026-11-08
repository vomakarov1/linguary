from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from lxml import html
import pytest
from sqlalchemy import text
from web_scenario import store_book, start_run, translated_result
from linguary.domain import BookStatus
from linguary.domain.book_conditions import BookConditions


def legacy_book(application, *, results=False):
    scenario = store_book(application, status=BookStatus.COMPLETED if results else BookStatus.GLOSSARY_REVIEW)
    source = application.paths.source_epub(scenario.book_id)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'isolated historical book source')
    with application.unit_of_work.transaction() as transaction:
        book = transaction.books.get(scenario.book_id)
        transaction.books.save(replace(book, source_epub_path=str(source)))
    if results:
        start_run(application, scenario, results=[translated_result(scenario.block_ids()[0])])
    return scenario


def choice(client, scenario, action='continue_legacy', revision=0):
    return client.post(f'/books/{scenario.book_id}/conditions', json=dict(action=action, expected_revision=revision, settings_revision=0, confirmed=True))


def test_legacy_resume_requires_explicit_choice(configured_client, configured_application):
    scenario = legacy_book(configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/process')
    assert response.status_code == 409
    selected = choice(configured_client, scenario)
    assert selected.status_code == 201, selected.text
    assert selected.json()['reproducibility'] == 'legacy_partial'
    assert configured_client.post(f'/books/{scenario.book_id}/process').status_code == 202


def test_reanalyze_archives_before_reset(configured_client, configured_application):
    scenario = legacy_book(configured_application, results=True)
    response = choice(configured_client, scenario, 'reanalyze')
    assert response.status_code == 201, response.text
    with configured_application.unit_of_work.transaction() as transaction:
        assert transaction.books.get(scenario.book_id).status == BookStatus.IMPORTED
        assert transaction.analyses.list_for_book(scenario.book_id) == ()
        assert transaction.runs.active_for_book(scenario.book_id) is None
        session = configured_application.unit_of_work._active_session()
        assert session.scalar(text('select count(*) from artifact_revisions')) == 2
        assert session.scalar(text('select count(*) from work_bindings')) == 0


def test_duplicate_revision_conflict(configured_client, configured_application):
    scenario = legacy_book(configured_application)
    assert choice(configured_client, scenario).status_code == 201
    response = choice(configured_client, scenario, 'reanalyze')
    assert response.status_code == 409
    assert response.json()['detail']['reason_code'] == 'conditions_revision_conflict'


@pytest.mark.parametrize('confirmed', [False, None])
def test_conditions_choice_requires_confirmation(configured_client, configured_application, confirmed):
    scenario = legacy_book(configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/conditions', json=dict(action='reanalyze', expected_revision=0, settings_revision=0, confirmed=confirmed))
    assert response.status_code == 422


def test_missing_book_404(configured_client):
    response = configured_client.post('/books/missing/conditions', json=dict(action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True))
    assert response.status_code == 404


def test_legacy_read_without_choice(configured_client, configured_application):
    scenario = legacy_book(configured_application)
    assert configured_client.get(f'/books/{scenario.book_id}').status_code == 200
    with configured_application.unit_of_work.transaction() as transaction:
        assert transaction.work.latest_conditions(scenario.book_id) is None


def test_conditions_not_saved_is_distinct_from_legacy(configured_client, configured_application):
    scenario = legacy_book(configured_application)
    response = configured_client.get(f'/books/{scenario.book_id}', headers={'Accept': 'text/html'})
    assert 'Условия ещё не сохранены' in response.text
    assert configured_client.get(f'/books/{scenario.book_id}').json()['conditions']['name_translation_state'] == 'not_saved'
    assert choice(configured_client, scenario).status_code == 201
    response = configured_client.get(f'/books/{scenario.book_id}', headers={'Accept': 'text/html'})
    assert 'Сохранённые условия без способа передачи имён' in response.text
    assert 'Сохранённый способ передачи имён: По звучанию' not in response.text
    legacy = configured_client.get(f'/books/{scenario.book_id}').json()['conditions']
    assert legacy['name_translation_state'] == 'legacy' and legacy['name_translation_policy'] is None


def test_continue_legacy_says_name_strategy_is_unchanged(configured_client, configured_application):
    scenario = legacy_book(configured_application)
    text = configured_client.get(f'/books/{scenario.book_id}', headers={'Accept': 'text/html'}).text
    assert 'текущая общая стратегия имён не применяется' in text


def test_saved_book_mode_differs_from_global(configured_client, configured_application):
    scenario = legacy_book(configured_application)
    assert choice(configured_client, scenario, 'reanalyze').status_code == 201
    config = configured_client.post('/system/config', json=dict(
        provider='openai_compatible', model='fake-model', base_url=configured_application.settings.llm.base_url,
        requests_per_minute=None, concurrency=configured_application.settings.worker.concurrency,
        expected_revision=0, name_translation_strategy='meaning',
    ))
    assert config.status_code == 200, config.text
    assert config.json()['reading']['name_translation_strategy'] == 'meaning'
    detail = configured_client.get(f'/books/{scenario.book_id}')
    assert detail.json()['conditions']['name_translation_state'] == 'configured'
    assert detail.json()['conditions']['name_translation_policy'] == {'version': 1, 'strategy': 'sound'}
    assert 'По звучанию' in configured_client.get(f'/books/{scenario.book_id}', headers={'Accept': 'text/html'}).text


def test_legacy_choice_does_not_invent_historical_provenance(configured_client, configured_application):
    scenario = legacy_book(configured_application, results=True)
    assert choice(configured_client, scenario).status_code == 201
    response = choice(configured_client, scenario, 'reanalyze', revision=1)
    assert response.status_code == 201, response.text
    with configured_application.unit_of_work.transaction():
        session = configured_application.unit_of_work._active_session()
        assert session.scalar(text('select count(*) from artifact_revisions where conditions_id is not null')) == 0


def reanalyze_form(client, scenario, page=''):
    response = client.get(f'/books/{scenario.book_id}{page}', headers={'Accept': 'text/html'})
    assert response.status_code == 200, response.text
    tree = html.fromstring(response.text)
    return tree.xpath('//form[input[@name="action" and @value="reanalyze"]]')[0]


@pytest.mark.parametrize('english_help', [False, True])
def test_pronunciation_form_separate_toggle(configured_client, configured_application, english_help):
    scenario = legacy_book(configured_application)
    form = reanalyze_form(configured_client, scenario)
    toggle = form.xpath('.//input[@name="pronunciation_help_enabled"]')[0]
    assert toggle.get('type') == 'checkbox' and toggle.get('value') == 'true'
    assert form.xpath('.//input[@name="english_help"]')
    response = configured_client.post(form.get('action'), data=dict(
        action='reanalyze', expected_revision=0, settings_revision=0, confirmed='true',
        english_help=str(english_help).lower(), pronunciation_help_enabled='true',
        pronunciation_variant='uk', reader_level='b2',
    ), headers={'Accept': 'text/html'}, follow_redirects=False)
    assert response.status_code == 303, response.text
    with configured_application.unit_of_work.transaction() as transaction:
        saved = transaction.work.latest_conditions(scenario.book_id)
        assert saved.pronunciation_help_enabled is True
        assert saved.pronunciation_variant == 'uk'
        assert saved.payload['english_help'] is english_help


@pytest.mark.parametrize('variant', [None, 'uk', 'us'])
def test_pronunciation_form_uk_us_and_disabled_state(configured_client, configured_application, variant):
    scenario = legacy_book(configured_application)
    if variant is not None:
        response = configured_client.post(f'/books/{scenario.book_id}/conditions', json=dict(
            action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True,
            pronunciation_help_enabled=True, pronunciation_variant=variant,
        ))
        assert response.status_code == 201, response.text
    form = reanalyze_form(configured_client, scenario)
    toggle = form.xpath('.//input[@name="pronunciation_help_enabled"]')[0]
    select = form.xpath('.//select[@data-pronunciation-variant]')[0]
    assert (toggle.get('checked') is not None) is (variant is not None)
    assert (select.getparent().get('hidden') is not None) is (variant is None)
    assert (select.get('required') is not None) is (variant is not None)
    assert select.get('name') == ('pronunciation_variant' if variant is not None else None)
    assert dict(form.form_values()).get('pronunciation_variant') == variant
    assert [option.get('value') for option in select.xpath('./option')] == ['', 'uk', 'us']
    assert select.xpath('./option[@selected]')[0].get('value') == (variant or '')
    assert 'Британский (UK)' in select.text_content()
    assert 'Американский (US)' in select.text_content()


@pytest.mark.parametrize('page', ['', '/result'])
def test_pronunciation_form_non_english_original_on_detail_and_result(configured_client, configured_application, page):
    scenario = legacy_book(configured_application, results=True)
    with configured_application.unit_of_work.transaction() as transaction:
        book = transaction.books.get(scenario.book_id)
        transaction.books.save(replace(book, source_language='fr', target_language='en'))
    conditions = configured_client.get(f'/books/{scenario.book_id}{page}').json()['conditions']
    assert conditions['source_language'] == 'fr'
    form = reanalyze_form(configured_client, scenario, page)
    assert not form.xpath('.//*[@name="pronunciation_help_enabled" or @name="pronunciation_variant"]')
    assert not form.xpath('.//select[@data-pronunciation-variant]')
    assert form.xpath('.//input[@name="english_help"]')


@pytest.mark.parametrize('page', ['', '/result'])
@pytest.mark.parametrize('variant', ['uk', 'us'])
def test_pronunciation_form_restores_saved_choice_on_detail_and_result(configured_client, configured_application, page, variant):
    scenario = legacy_book(configured_application)
    response = configured_client.post(f'/books/{scenario.book_id}/conditions', json=dict(
        action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True,
        pronunciation_help_enabled=True, pronunciation_variant=variant,
    ))
    assert response.status_code == 201, response.text
    start_run(configured_application, scenario, results=[translated_result(scenario.block_ids()[0])])
    conditions = configured_client.get(f'/books/{scenario.book_id}{page}').json()['conditions']
    assert conditions['source_language'] == 'en'
    assert conditions['pronunciation_help_enabled'] is True
    assert conditions['pronunciation_variant'] == variant
    form = reanalyze_form(configured_client, scenario, page)
    assert form.xpath('.//input[@name="pronunciation_help_enabled"]')[0].get('checked') is not None
    assert form.xpath('.//select[@name="pronunciation_variant"]/option[@selected]')[0].get('value') == variant


@pytest.mark.parametrize('page', ['', '/result'])
@pytest.mark.parametrize('saved_snapshot', [False, True])
def test_pronunciation_form_legacy_defaults(configured_client, configured_application, page, saved_snapshot):
    scenario = legacy_book(configured_application, results=True)
    if saved_snapshot:
        with configured_application.unit_of_work.transaction() as transaction:
            transaction.work.add_conditions(BookConditions(
                id=str(uuid4()), book_id=scenario.book_id, revision=1,
                payload={'schema_version': 1}, created_at=datetime.now(UTC),
                reproducibility='legacy_partial',
            ))
    conditions = configured_client.get(f'/books/{scenario.book_id}{page}').json()['conditions']
    assert conditions['source_language'] == 'en'
    assert conditions['pronunciation_help_enabled'] is False
    assert conditions['pronunciation_variant'] is None
    form = reanalyze_form(configured_client, scenario, page)
    assert form.xpath('.//input[@name="pronunciation_help_enabled"]')[0].get('checked') is None
    select = form.xpath('.//select[@data-pronunciation-variant]')[0]
    assert select.getparent().get('hidden') is not None
    assert select.get('name') is None
    assert 'pronunciation_variant' not in dict(form.form_values())
    with configured_application.unit_of_work.transaction() as transaction:
        saved = transaction.work.latest_conditions(scenario.book_id)
        assert (saved is not None) is saved_snapshot
        if saved is not None:
            assert 'pronunciation_help_enabled' not in saved.payload


def test_pronunciation_form_keeps_existing_conditions(configured_client, configured_application):
    scenario = legacy_book(configured_application, results=True)
    assert choice(configured_client, scenario).status_code == 201
    form = reanalyze_form(configured_client, scenario)
    values = {field.get('name'): field.get('value') for field in form.xpath('.//input[@type="hidden"]')}
    assert values == {'action': 'reanalyze', 'expected_revision': '1', 'settings_revision': '0'}
    confirmation = form.xpath('.//input[@name="confirmed"]')[0]
    assert confirmation.get('required') is not None and confirmation.get('checked') is None
    assert 'Архив версий сохранится' in form.getparent().text_content()
    assert [option.get('value') for option in form.xpath('.//select[@name="reader_level"]/option')] == ['a2', 'b1', 'b2', 'c1']
    values.update(confirmed='true', english_help='true', reader_level='a2',
                  pronunciation_help_enabled='true', pronunciation_variant='us')
    response = configured_client.post(form.get('action'), data=values,
        headers={'Accept': 'text/html'}, follow_redirects=False)
    assert response.status_code == 303, response.text
    with configured_application.unit_of_work.transaction() as transaction:
        saved = transaction.work.latest_conditions(scenario.book_id)
        assert saved.revision == 2
        assert saved.payload['english_help'] is True
        assert saved.payload['reader_level'] == 'a2'
        assert saved.pronunciation_variant == 'us'
    values['expected_revision'] = '2'
    values.pop('pronunciation_help_enabled')
    values.pop('pronunciation_variant')
    response = configured_client.post(form.get('action'), data=values,
        headers={'Accept': 'text/html'}, follow_redirects=False)
    assert response.status_code == 303, response.text
    with configured_application.unit_of_work.transaction() as transaction:
        saved = transaction.work.latest_conditions(scenario.book_id)
        assert saved.revision == 3
        assert saved.pronunciation_help_enabled is False
        assert saved.pronunciation_variant is None


def test_c1_reader_level_supported_in_conditions_and_english_help(configured_client, configured_application):
    from linguary.application.reader_text import learner_instructions

    assert "C1" in learner_instructions("c1")

    scenario = legacy_book(configured_application)
    form = reanalyze_form(configured_client, scenario)
    assert [option.get('value') for option in form.xpath('.//select[@name="reader_level"]/option')] == ['a2', 'b1', 'b2', 'c1']

    response = configured_client.post(
        f'/books/{scenario.book_id}/conditions',
        json=dict(
            action='reanalyze',
            expected_revision=0,
            settings_revision=0,
            confirmed=True,
            reader_level='c1',
            english_help=False,
        ),
    )
    assert response.status_code == 201, response.text
    with configured_application.unit_of_work.transaction() as transaction:
        saved = transaction.work.latest_conditions(scenario.book_id)
        assert saved.payload['reader_level'] == 'c1'
        assert saved.payload['english_help'] is False

    response_eh = configured_client.post(
        f'/books/{scenario.book_id}/conditions',
        json=dict(
            action='reanalyze',
            expected_revision=1,
            settings_revision=0,
            confirmed=True,
            reader_level='c1',
            english_help=True,
        ),
    )
    assert response_eh.status_code == 201, response_eh.text
    with configured_application.unit_of_work.transaction() as transaction:
        saved_eh = transaction.work.latest_conditions(scenario.book_id)
        assert saved_eh.payload['reader_level'] == 'c1'
        assert saved_eh.payload['english_help'] is True
        assert 'Для C1' in saved_eh.payload['prompt_templates']['english_help_processing']


def test_build_application_defaults_to_content_plan_v3(tmp_path, monkeypatch):
    from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, ServerSettings, StorageSettings
    from linguary.adapters.llm import FakeLLMClient
    from linguary.composition import build_application

    monkeypatch.delenv("LINGUARY_CONTENT_PLAN_PROFILE", raising=False)
    app_settings = AppSettings(
        llm=LLMSettings(model="fake-model", api_key="test-key"),
        checks=CheckSettings(),
        storage=StorageSettings(data_root=str(tmp_path / "default-v3-data")),
        server=ServerSettings(),
    )
    app = build_application(settings=app_settings, llm_client=FakeLLMClient())
    try:
        scenario = legacy_book(app)
        with app.unit_of_work.transaction() as transaction:
            book = transaction.books.get(scenario.book_id)
            conditions = app.book_conditions.create(transaction, book, app.runtime_settings.revision)
            assert conditions.payload["profile"] == "content-plan-v3"
    finally:
        app.dispose()

