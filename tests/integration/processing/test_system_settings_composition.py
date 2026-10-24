"""Application build loads saved settings and preserves long-lived services."""
from dataclasses import replace
from pathlib import Path
import traceback

import pytest
from sqlalchemy import inspect

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.adapters.config.settings import SettingsError
from linguary.adapters.config.system_settings import SystemSettingsUpdate, candidate_settings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.system_settings import SystemSettingsStore
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.runtime import WorkerSettings
from linguary.composition import build_application, drain_tasks
from processing_scenario import BOOK_ID, SOURCE_TEXT, store_processing_book, translated_reply


def settings(tmp_path):
    return AppSettings(llm=LLMSettings(model='baseline', api_key='baseline-test-key'), storage=StorageSettings(data_root=str(tmp_path / 'data')))


def overrides():
    return dict(provider='openai_compatible', model='saved', base_url='http://localhost:8080/v1', requests_per_minute=None, concurrency=3)


async def test_saved_settings_override_environment_on_start(tmp_path):
    baseline = settings(tmp_path)
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    app.settings_store.save(overrides(), 'saved-test-key', expected_revision=0)
    await app.aclose()
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    try:
        assert app.settings.llm.model == 'saved'
        assert app.settings.llm.api_key == 'saved-test-key'
        assert app.settings.worker.concurrency == app.worker_pool.concurrency == 3
        assert app.runtime_settings.revision == 1
        assert app.runtime_settings.initial_settings == baseline
        assert app.settings.storage is baseline.storage
    finally:
        await app.aclose()


async def test_invalid_baseline_reports_safe_error_before_database_overlay(tmp_path, monkeypatch):
    baseline = settings(tmp_path)
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    app.settings_store.save(overrides(), 'saved-test-key', expected_revision=0)
    await app.aclose()
    secret = 'private-test-key'
    bad_value = 'not-an-integer-private-value'
    monkeypatch.setenv('LINGUARY_STORAGE_DATA_ROOT', baseline.storage.data_root)
    monkeypatch.setenv('LINGUARY_LLM_API_KEY', secret)
    monkeypatch.setenv('LINGUARY_WORKER_CONCURRENCY', bad_value)
    with pytest.raises(SettingsError) as caught:
        build_application(llm_client=FakeLLMClient())
    formatted = ''.join(traceback.format_exception(caught.value))
    assert secret not in formatted and bad_value not in formatted
    assert caught.value.__cause__ is None


async def test_invalid_saved_settings_fail_safely(tmp_path):
    baseline = settings(tmp_path)
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    app.settings_store.save({**overrides(), 'base_url': 'http://user:private-test-key@host'}, 'private-test-key', expected_revision=0)
    await app.aclose()
    with pytest.raises(SettingsError) as caught:
        build_application(settings=baseline, llm_client=FakeLLMClient())
    assert 'private-test-key' not in ''.join(traceback.format_exception(caught.value))


async def test_build_application_applies_settings_migration(tmp_path):
    app = build_application(settings=settings(tmp_path), llm_client=FakeLLMClient())
    engine = create_database_engine(app.paths.database_path)
    try:
        assert 'system_settings' in inspect(engine).get_table_names()
    finally:
        engine.dispose()
        await app.aclose()


async def test_injected_llm_client_survives_reconfiguration(tmp_path):
    class Injected:
        def __init__(self):
            self.inner = FakeLLMClient.always(translated_reply())
            self.closed = 0
        async def generate(self, request):
            return replace(await self.inner.generate(request), model='saved')
        async def aclose(self):
            self.closed += 1
    client = Injected()
    app = build_application(settings=settings(tmp_path), llm_client=client)
    try:
        scenario = store_processing_book(app.unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT), (SOURCE_TEXT, SOURCE_TEXT)))
        await app.processing.start(BOOK_ID)
        previous_runner, previous_queue, previous_pool = app.runner, app.task_queue, app.worker_pool
        new_settings = candidate_settings(app.settings, SystemSettingsUpdate(**overrides()))
        app.runtime_settings.publish(app.runtime_settings.prepare(new_settings), 1)
        finished = await drain_tasks(app)
        assert len(finished) == 2 and all(task.last_error is None for task in finished)
        assert client.inner.call_count == 2
        assert app.runner is previous_runner and app.task_queue is previous_queue and app.worker_pool is previous_pool
        with app.unit_of_work.transaction() as transaction:
            logs = transaction.llm_calls.list_for_book(BOOK_ID)
        assert [log.model for log in logs] == ['saved', 'saved']
    finally:
        await app.aclose()
    assert client.closed == 0


async def test_explicit_worker_settings_remain_startup_baseline(tmp_path):
    app = build_application(settings=settings(tmp_path), llm_client=FakeLLMClient(), worker_settings=WorkerSettings(concurrency=4, idle_delay_seconds=0.2))
    try:
        assert app.worker_pool.concurrency == 4
        assert app.runtime_settings.initial_settings.worker.concurrency == 4
    finally:
        await app.aclose()


async def test_unedited_sections_remain_from_baseline(tmp_path):
    baseline = replace(settings(tmp_path), llm=replace(settings(tmp_path).llm, timeout_seconds=73, temperature=0.4, max_attempts=2))
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    app.settings_store.save(overrides(), 'saved-test-key', expected_revision=0)
    await app.aclose()
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    try:
        assert app.settings.llm.timeout_seconds == 73
        assert app.settings.llm.temperature == 0.4 and app.settings.llm.max_attempts == 2
        assert app.settings.checks is baseline.checks
        assert app.settings.server is baseline.server and app.settings.storage is baseline.storage
    finally:
        await app.aclose()


async def test_processing_run_snapshot_is_not_rewritten(tmp_path):
    app = build_application(settings=settings(tmp_path), llm_client=FakeLLMClient())
    try:
        scenario = store_processing_book(app.unit_of_work)
        with app.unit_of_work.transaction() as transaction:
            before = transaction.runs.get(scenario.run_id)
        snapshot = app.start_processing()._llm_settings_snapshot
        app.runtime_settings.publish(app.runtime_settings.prepare(candidate_settings(app.settings, SystemSettingsUpdate(**overrides()))), 1)
        with app.unit_of_work.transaction() as transaction:
            after = transaction.runs.get(scenario.run_id)
        assert before.llm_settings_snapshot == after.llm_settings_snapshot
        assert snapshot.model == 'baseline'
        assert app.start_processing()._llm_settings_snapshot.model == 'saved'
    finally:
        await app.aclose()


async def test_generation_logging_uses_actual_model(tmp_path):
    from linguary.application.ports.llm import LLMConfigurationError
    class FailingClient:
        async def generate(self, request):
            raise LLMConfigurationError('Model not available')
    app = build_application(settings=settings(tmp_path), llm_client=FailingClient())
    try:
        store_processing_book(app.unit_of_work)
        await app.processing.start(BOOK_ID)
        app.runtime_settings.publish(app.runtime_settings.prepare(candidate_settings(app.settings, SystemSettingsUpdate(**overrides()))), 1)
        await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            logs = transaction.llm_calls.list_for_book(BOOK_ID)
        assert len(logs) == 1 and logs[0].model == 'saved'
    finally:
        await app.aclose()


def test_cli_build_loads_saved_settings(tmp_path, capsys):
    from linguary.cli import main
    baseline = settings(tmp_path)
    app = build_application(settings=baseline, llm_client=FakeLLMClient())
    app.settings_store.save(overrides(), 'saved-test-key', expected_revision=0)
    app.dispose()
    config = tmp_path / 'config.toml'
    config.write_text(f'[storage]\ndata_root = "{baseline.storage.data_root}"\n[llm]\nmodel = "baseline"\n')
    assert main(['--config', str(config), 'doctor']) == 0
    output = capsys.readouterr().out
    assert 'LLM: ready' in output
    assert 'saved-test-key' not in output


async def test_owned_client_is_prepared_without_network_and_closed(tmp_path):
    app = build_application(settings=settings(tmp_path))
    client = app.runtime_settings.current.configured_client
    assert client._client is not None
    http = client._client._http
    assert not http.is_closed
    await app.aclose()
    assert http.is_closed


async def test_runtime_rebuild_preserves_content_plan_profile(tmp_path):
    from linguary.application.context_builder import CONTENT_PLAN_PROMPT_VERSION
    from linguary.domain import BookMetadata, CanonicalDocument, DocumentSection, DocumentBlock, BlockType, SupportStatus, SourceReference
    app = build_application(settings=settings(tmp_path), llm_client=FakeLLMClient(), content_plan_profile='content-plan-v2')
    try:
        block = DocumentBlock.create(document_id='book', section_id='s', order=0, type=BlockType.PARAGRAPH, text='Echo calls.', support_status=SupportStatus.SUPPORTED, source_reference=SourceReference(resource_path='s.xhtml', structural_path='/p'))
        section = DocumentSection(id='s', document_id='book', title='Chapter', order=0, source_resource_path='s.xhtml', blocks=(block,))
        document = CanonicalDocument(id='book', metadata=BookMetadata(title='Story', authors=()), source_language='en', target_language='ru', sections=(section,))
        app.runtime_settings.publish(app.runtime_settings.prepare(candidate_settings(app.settings, SystemSettingsUpdate(**overrides()))), 1)
        context = app.runtime_settings.current.analysis._analyze_chapter._context_builder.for_chapter_analysis(document=document, section=section)
        assert context.request.contract_version == 2
        assert CONTENT_PLAN_PROMPT_VERSION in context.request.instructions
    finally:
        await app.aclose()
