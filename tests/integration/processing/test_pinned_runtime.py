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


async def test_changed_proxy_allowed_while_changed_base_url_or_provider_rejected_without_request(tmp_path):
    fake = FakeLLMClient([reply()])
    app = make_app(tmp_path, fake)
    try:
        store_imported(app)
        task = await app.analysis.start('book-1')
        # Proxy change is operational and must NOT block pinned conditions.
        with_proxy = replace(
            app.settings,
            llm=replace(app.settings.llm, proxy_mode='custom', proxy_url='http://user:password@proxy.example:80'),
        )
        app.runtime_settings.publish(app.runtime_settings.prepare(with_proxy), 1)
        conditions = app.book_conditions.require('book-1')
        resolved = app.book_runtime.settings_for(conditions)
        assert resolved.llm.proxy_mode == 'custom'
        assert resolved.llm.proxy_url == 'http://user:password@proxy.example:80'

        # Changing base_url or provider still rejects with route_unavailable without sending a request.
        changed_url = replace(
            app.settings,
            llm=replace(app.settings.llm, base_url='https://other-provider.example/v1'),
        )
        app.runtime_settings.publish(app.runtime_settings.prepare(changed_url), 2)
        with pytest.raises(ConditionsError, match='route_unavailable'):
            await app.analysis.run_task(task)
        assert fake.requests == []

        changed_provider = replace(
            app.settings,
            llm=replace(app.settings.llm, provider='other_provider'),
        )
        app.runtime_settings.publish(app.runtime_settings.prepare(changed_provider), 3)
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


async def test_auto_proxy_route_fingerprint_uses_public_url_and_ignores_env_proxies(tmp_path, monkeypatch):
    from linguary.composition.book_runtime import route_snapshot
    from linguary.adapters.config import LLMSettings
    monkeypatch.setenv('HTTPS_PROXY', 'http://alice:first@proxy.example:80')
    first = route_snapshot(
        LLMSettings(
            base_url='https://alice:secret@api.example.com:8443/v1?token=xyz',
            proxy_mode='auto',
        )
    )
    assert first['base_url'] == 'https://api.example.com:8443/v1'
    assert first['proxy'] == {'mode': 'auto'}
    assert 'environment' not in first['proxy']
    assert 'alice' not in str(first) and 'secret' not in str(first) and 'token' not in str(first)
    monkeypatch.setenv('HTTPS_PROXY', 'http://bob:second@proxy.example:81')
    second = route_snapshot(
        LLMSettings(
            base_url='https://bob:other@api.example.com:8443/v1?token=abc',
            proxy_mode='auto',
        )
    )
    assert second == first


async def test_operational_llm_overrides_and_legacy_env_proxy_conditions_compatibility(tmp_path):
    from linguary.domain.book_conditions import BookConditions, thaw
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        current = app.book_conditions.require('book-1')

        # Simulate an old SQLite book_conditions row containing legacy proxy.environment and old route_id.
        legacy_payload = thaw(current.payload)
        legacy_payload['route'] = {
            'provider': legacy_payload['route']['provider'],
            'base_url': legacy_payload['route']['base_url'],
            'proxy': {
                'mode': 'auto',
                'environment': {'HTTPS_PROXY': 'http://legacy-proxy.example:8080'},
            },
            'route_id': 'legacy-route-digest-with-env-proxy',
        }
        legacy_payload['llm_parameters']['timeout_seconds'] = 180.0
        legacy_payload['llm_parameters']['max_attempts'] = 2
        legacy_payload['llm_parameters']['requests_per_minute'] = 10
        legacy_payload['llm_parameters']['temperature'] = 0.2
        legacy_conditions = replace(current, payload=legacy_payload)
        assert isinstance(legacy_conditions, BookConditions)

        updated = replace(
            app.settings,
            llm=replace(
                app.settings.llm,
                model='new-model-ignored',
                temperature=0.9,
                proxy_mode='custom',
                proxy_url='http://new-proxy.example:3128',
                timeout_seconds=900.0,
                requests_per_minute=60,
                max_attempts=5,
                debug_llm_payloads=True,
                input_price_per_mtok=1.25,
                output_price_per_mtok=5.0,
            ),
        )
        app.runtime_settings.publish(app.runtime_settings.prepare(updated), 1)

        resolved = app.book_runtime.settings_for(legacy_conditions)
        # Operational settings come from current runtime settings.
        assert resolved.llm.proxy_mode == 'custom'
        assert resolved.llm.proxy_url == 'http://new-proxy.example:3128'
        assert resolved.llm.timeout_seconds == 900.0
        assert resolved.llm.requests_per_minute == 60
        assert resolved.llm.max_attempts == 5
        assert resolved.llm.debug_llm_payloads is True
        assert resolved.llm.input_price_per_mtok == 1.25
        assert resolved.llm.output_price_per_mtok == 5.0
        # Content-affecting settings remain pinned to the book's conditions.
        assert resolved.llm.model == 'pinned-model'
        assert resolved.llm.temperature == 0.2

        # _runtime_fingerprint changes when operational fields change.
        fp_before = app.book_runtime._runtime_fingerprint(resolved)
        for mutated_llm in (
            replace(resolved.llm, timeout_seconds=1200.0),
            replace(resolved.llm, requests_per_minute=30),
            replace(resolved.llm, max_attempts=7),
            replace(resolved.llm, proxy_mode='direct', proxy_url=None),
            replace(resolved.llm, input_price_per_mtok=2.0, output_price_per_mtok=8.0),
        ):
            assert app.book_runtime._runtime_fingerprint(replace(resolved, llm=mutated_llm)) != fp_before
    finally:
        await app.aclose()


async def test_english_help_survives_restart_and_old_snapshot_defaults_off(tmp_path):
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    from linguary.application.context_builder import ContextBuilder
    app = make_app(tmp_path)
    store_imported(app)
    saved = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True, english_help=True, reader_level='a2')
    await app.aclose()
    restarted = make_app(tmp_path)
    try:
        conditions = restarted.book_conditions.require('book-1')
        assert conditions.id == saved.id
        assert restarted.book_runtime.settings_for(conditions).reading.normalized_level == 'a2'
        assert ContextBuilder(content_plan_profile='content-plan-v3', prompt_templates=conditions.payload['prompt_templates'])._english_help
        assert not ContextBuilder()._english_help
    finally:
        await restarted.aclose()


async def test_snapshot_without_english_help_retains_original_level(tmp_path):
    from linguary.domain.book_conditions import BookConditions, thaw
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        current = app.book_conditions.require('book-1')
        payload = thaw(current.payload)
        payload.pop('english_help', None)
        historical = replace(current, payload=payload)
        assert isinstance(historical, BookConditions)
        assert app.book_runtime.settings_for(historical).reading.normalized_level == payload['reader_level']
        assert 'english_help' not in historical.payload['prompt_templates']
    finally:
        await app.aclose()


async def test_start_processing_port_change_rolls_back_and_keeps_correction_route_aligned(tmp_path):
    """WP1: changing local port before process rejects atomically; restoring port allows processing + correction."""
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from linguary.domain import BookStatus

    fake = FakeLLMClient([reply()])
    app = make_app(tmp_path, fake)
    try:
        port_a = replace(app.settings, llm=replace(app.settings.llm, base_url='http://127.0.0.1:51247/v1'))
        app.runtime_settings.publish(app.runtime_settings.prepare(port_a), 1)
        store_imported(app)
        await app.analysis.start('book-1')
        completed = await drain_tasks(app)
        assert len(completed) == 1 and completed[0].status == TaskStatus.SUCCEEDED

        # Simulate ephemeral port shift before launching processing.
        port_b = replace(app.settings, llm=replace(app.settings.llm, base_url='http://127.0.0.1:51594/v1'))
        app.runtime_settings.publish(app.runtime_settings.prepare(port_b), 2)
        with pytest.raises(ConditionsError, match='route_unavailable'):
            app.start_processing()(StartProcessingCommand(book_id='book-1'))

        with app.unit_of_work.transaction() as tx:
            assert tx.runs.active_for_book('book-1') is None
            assert tx.books.get('book-1').status in {BookStatus.GLOSSARY_REVIEW, BookStatus.READY_TO_PROCESS}

        # Restore original port and start processing cleanly.
        app.runtime_settings.publish(app.runtime_settings.prepare(port_a), 3)
        run = app.start_processing()(StartProcessingCommand(book_id='book-1'))
        conditions = app.book_conditions.require('book-1')
        with app.unit_of_work.transaction() as tx:
            binding = tx.work.get_binding('run', run.id)
        assert binding is not None
        assert binding['payload_json']['correction']['route'] == dict(conditions.payload['route'])
    finally:
        await app.aclose()

