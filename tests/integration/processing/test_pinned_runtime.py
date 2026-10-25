from dataclasses import replace
import pytest
from linguary.adapters.config.settings import ReadingSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.application.book_conditions import ConditionsError
from linguary.composition import drain_tasks
from linguary.domain import TaskStatus
from test_book_conditions import make_app, store_imported, reply


async def test_pending_task_after_settings_change_and_restart(tmp_path, monkeypatch):
    from linguary.application import context_builder
    app = make_app(tmp_path)
    store_imported(app)
    await app.analysis.start('book-1')
    conditions = app.book_conditions.require('book-1')
    saved = conditions.payload['prompt_templates']['analysis']
    await app.aclose()
    monkeypatch.setattr(context_builder, '_ANALYSIS_INSTRUCTIONS', 'CHANGED IN NEXT RELEASE')
    fake = FakeLLMClient([reply()])
    restarted = make_app(tmp_path, fake)
    candidate = replace(restarted.settings, reading=ReadingSettings(reader_level='c1'),
                        llm=replace(restarted.settings.llm, model='new-model', api_key='rotated-secret'))
    restarted.runtime_settings.publish(restarted.runtime_settings.prepare(candidate), 1)
    try:
        completed = await drain_tasks(restarted)
        assert len(completed) == 1 and completed[0].status == TaskStatus.SUCCEEDED
        assert fake.requests[0].instructions == saved
        with restarted.unit_of_work.transaction() as transaction:
            attempt = transaction.work.get_attempt(transaction.work.get_binding('analysis', transaction.analyses.list_for_book('book-1')[0].id)['attempt_id'])
            assert attempt.context['designated_models']['analysis'] == 'pinned-model'
            assert attempt.outcome['model'] != 'new-model'
        assert restarted.book_conditions.require('book-1').payload['reader_level'] == 'b2'
    finally:
        await restarted.aclose()


async def test_changed_proxy_route_rejected_without_request(tmp_path):
    fake = FakeLLMClient([reply()])
    app = make_app(tmp_path, fake)
    try:
        store_imported(app)
        task = await app.analysis.start('book-1')
        candidate = replace(app.settings, llm=replace(app.settings.llm, proxy_mode='custom', proxy_url='http://user:password@proxy.example:80'))
        app.runtime_settings.publish(app.runtime_settings.prepare(candidate), 1)
        with pytest.raises(ConditionsError, match='route_unavailable'):
            await app.analysis.run_task(task)
        assert fake.requests == []
    finally:
        await app.aclose()


async def test_missing_revision_has_no_request(tmp_path):
    fake = FakeLLMClient([reply()])
    app = make_app(tmp_path, fake)
    try:
        from uuid import uuid4
        async with app.book_runtime.lease(str(uuid4()), 'analysis'):
            pytest.fail('missing version leased')
    except ConditionsError as error:
        assert error.reason_code == 'conditions_unavailable'
        assert fake.requests == []
    finally:
        await app.aclose()


async def test_auto_proxy_route_fingerprint_excludes_credentials(tmp_path, monkeypatch):
    from linguary.composition.book_runtime import route_snapshot
    from linguary.adapters.config import LLMSettings
    monkeypatch.setenv('HTTPS_PROXY', 'http://alice:first@proxy.example:80')
    first = route_snapshot(LLMSettings(proxy_mode='auto'))
    monkeypatch.setenv('HTTPS_PROXY', 'http://bob:second@proxy.example:80')
    assert route_snapshot(LLMSettings(proxy_mode='auto')) == first
    assert 'alice' not in str(first) and 'first' not in str(first)
    monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.example:81')
    assert route_snapshot(LLMSettings(proxy_mode='auto')) != first
