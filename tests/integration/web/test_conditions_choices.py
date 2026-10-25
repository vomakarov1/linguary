from dataclasses import replace
import pytest
from sqlalchemy import text
from web_scenario import store_book, start_run, translated_result
from linguary.domain import BookStatus


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


def test_legacy_choice_does_not_invent_historical_provenance(configured_client, configured_application):
    scenario = legacy_book(configured_application, results=True)
    assert choice(configured_client, scenario).status_code == 201
    response = choice(configured_client, scenario, 'reanalyze', revision=1)
    assert response.status_code == 201, response.text
    with configured_application.unit_of_work.transaction():
        session = configured_application.unit_of_work._active_session()
        assert session.scalar(text('select count(*) from artifact_revisions where conditions_id is not null')) == 0
