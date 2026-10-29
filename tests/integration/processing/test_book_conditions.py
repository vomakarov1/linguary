"""Pinned conditions are exercised through the real composition and queue."""
from dataclasses import replace
from datetime import UTC, datetime
import json

import pytest
from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.config.system_settings import SystemSettingsUpdate
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.book_conditions import ConditionsError
from linguary.application.book_conditions import payload_digest
from linguary.application.context_builder import snapshot_prompt_templates
from linguary.application.use_cases.reset_processing import ChooseBookConditions
from linguary.application.ports.persistence import BookRecord
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.composition import build_application, drain_tasks
from linguary.domain import BookStatus, BookMetadata, TaskStatus
from processing_scenario import build_document


def make_app(tmp_path, fake=None):
    return build_application(settings=AppSettings(llm=LLMSettings(model='pinned-model', api_key='synthetic-secret', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake or FakeLLMClient())


def store_imported(app, chapters=(('The cat sat.',),)):
    source = app.paths.source_epub('book-1')
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'isolated synthetic source')
    document = build_document(chapters)
    with app.unit_of_work.transaction() as transaction:
        transaction.books.add(BookRecord('book-1', BookMetadata('Synthetic'), 'en', 'ru', BookStatus.IMPORTED, str(source), datetime.now(UTC)))
        transaction.documents.save(document)
    return document


def reply():
    return FakeReply.of_payload({'treatment_plan': [{'source_block_id': 'b1', 'treatment': 'READ_AS_IS'}],
        'glossary_candidates': [], 'translation_instructions': 'Сохраняй смысл.', 'chapter_summary': 'The cat sat.', 'before_you_read': []})


async def test_freeze_precedes_first_analysis(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        task = await app.analysis.start('book-1')
        with app.unit_of_work.transaction() as transaction:
            conditions = transaction.work.latest_conditions('book-1')
            assert transaction.work.get_binding('task', task.id)['conditions_id'] == conditions.id
            assert 'synthetic-secret' not in json.dumps(conditions.to_dict())
            assert 'glossary_snapshot' not in conditions.payload
    finally:
        await app.aclose()


async def test_analysis_result_has_attempt_and_next_chapter_keeps_conditions(tmp_path):
    fake = FakeLLMClient([reply(), reply()])
    app = make_app(tmp_path, fake)
    try:
        store_imported(app, (('The cat sat.',), ('The dog ran.',)))
        await app.analysis.start('book-1')
        completed = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in completed), [(task.status, task.last_error) for task in completed]
        with app.unit_of_work.transaction() as transaction:
            conditions = transaction.work.latest_conditions('book-1')
            analyses = transaction.analyses.list_for_book('book-1')
            assert len(analyses) == 2
            for analysis in analyses:
                binding = transaction.work.get_binding('analysis', analysis.id)
                assert binding['conditions_id'] == conditions.id
                assert transaction.work.get_attempt(binding['attempt_id']).state == 'accepted'
                assert transaction.work.get_artifact(binding['artifact_revision_id'])
    finally:
        await app.aclose()


async def test_glossary_frozen_at_start(tmp_path):
    app = make_app(tmp_path, FakeLLMClient([reply()]))
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        with app.unit_of_work.transaction() as transaction:
            binding = transaction.work.get_binding('run', run.id)
            assert binding['payload_json']['glossary_snapshot'] == {'entries': []}
            assert len(binding['payload_json']['glossary_sha256']) == 64
    finally:
        await app.aclose()


async def test_source_digest_mismatch_stops(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        app.paths.source_epub('book-1').write_bytes(b'changed')
        with pytest.raises(ConditionsError, match='source_digest_mismatch'):
            await app.analysis.start('book-1')
    finally:
        await app.aclose()


def choose_legacy_fixture(application, book_id='book-1'):
    """Explicit historical continuation for isolated pre-existing test projections."""
    from dataclasses import replace
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    source = application.paths.source_epub(book_id)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'isolated legacy fixture source')
    with application.unit_of_work.transaction() as transaction:
        book = transaction.books.get(book_id)
        transaction.books.save(replace(book, source_epub_path=str(source)))
        previous = transaction.work.latest_conditions(book_id)
    if previous is None:
        ChooseBookConditions(application.book_conditions, application.book_runtime)(book_id,
            action='continue_legacy', expected_revision=0, settings_revision=application.runtime_settings.revision, confirmed=True)
    else:
        with application.unit_of_work.transaction() as transaction:
            for task in transaction.tasks.list_for_book(book_id):
                if transaction.work.get_binding('task', task.id) is None:
                    transaction.work.bind('task', task.id, book_id=book_id, conditions_id=previous.id)


async def test_late_old_analysis_cannot_write(tmp_path):
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    fake = FakeLLMClient([reply()])

    class ReanalyzingClient:
        async def generate(self, request):
            response = await fake.generate(request)
            ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze',
                expected_revision=1, settings_revision=0, confirmed=True)
            return response

    app = make_app(tmp_path, ReanalyzingClient())
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.FAILED
        assert 'conditions_revision_conflict' in completed[0].last_error.message
        with app.unit_of_work.transaction() as transaction:
            assert transaction.analyses.list_for_book('book-1') == ()
            assert transaction.work.latest_conditions('book-1').revision == 2
    finally:
        await app.aclose()


@pytest.mark.parametrize('level', ['a2', 'b1', 'b2'])
async def test_explicit_english_help_conditions_are_revision_pinned(tmp_path, level):
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    app = make_app(tmp_path)
    try:
        store_imported(app)
        choice = ChooseBookConditions(app.book_conditions, app.book_runtime)
        conditions = choice('book-1', action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True, english_help=True, reader_level=level)
        assert conditions.payload['english_help'] is True
        assert conditions.payload['reader_level'] == level
        assert conditions.payload['versions']['schemas'] == 4
        assert conditions.payload['versions']['prompts'] == 4
        with pytest.raises(ConditionsError, match='conditions_revision_conflict'):
            choice('book-1', action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True, english_help=False)
        assert app.book_conditions.require('book-1').id == conditions.id
    finally:
        await app.aclose()


async def test_new_conditions_capture_sound(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        conditions = app.book_conditions.require('book-1')
        assert conditions.payload['name_translation_policy'] == {'version': 1, 'strategy': 'sound'}
        assert app.book_runtime.settings_for(conditions).reading.name_translation_strategy == 'sound'
    finally:
        await app.aclose()


async def test_global_change_does_not_mutate_book(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        old = app.book_conditions.require('book-1')
        await app.system_settings.save(SystemSettingsUpdate(
            provider='openai_compatible', model='pinned-model', base_url='https://api.openai.com/v1',
            requests_per_minute=None, concurrency=1, name_translation_strategy='meaning',
        ), expected_revision=0)
        assert app.runtime_settings.current.settings.reading.name_translation_strategy == 'meaning'
        assert app.book_conditions.require('book-1').id == old.id
        assert app.book_runtime.settings_for(old).reading.name_translation_strategy == 'sound'
        fresh = ChooseBookConditions(app.book_conditions, app.book_runtime)(
            'book-1', action='reanalyze', expected_revision=1, settings_revision=1, confirmed=True)
        assert fresh.payload['name_translation_policy'] == {'version': 1, 'strategy': 'meaning'}
        assert old.payload['name_translation_policy'] == {'version': 1, 'strategy': 'sound'}
    finally:
        await app.aclose()


async def test_reanalysis_captures_new_policy(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        previous = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='continue_legacy', expected_revision=0, settings_revision=0, confirmed=True)
        assert 'name_translation_policy' not in previous.payload
        current = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=1, settings_revision=0, confirmed=True)
        assert current.revision == 2
        assert current.payload['name_translation_policy']['strategy'] == 'sound'
        assert previous.payload['prompt_sha256'] == payload_digest(previous.payload['prompt_templates'])
    finally:
        await app.aclose()


async def test_continue_legacy_has_no_policy(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='continue_legacy', expected_revision=0, settings_revision=0, confirmed=True)
        assert 'name_translation_policy' not in conditions.payload
    finally:
        await app.aclose()


async def test_old_snapshots_keep_exact_prompts(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='continue_legacy', expected_revision=0, settings_revision=0, confirmed=True)
        assert conditions.payload['prompt_templates'] == snapshot_prompt_templates(source_language='en', target_language='ru', reader_level='b2', profile=conditions.payload['profile'])
    finally:
        await app.aclose()


async def test_unknown_policy_version_refused(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        conditions = app.book_conditions.require('book-1')
        from dataclasses import replace as dataclass_replace
        payload = dict(conditions.payload)
        payload['name_translation_policy'] = {'version': 2, 'strategy': 'sound'}
        changed = dataclass_replace(conditions, payload=payload)
        with pytest.raises(ConditionsError, match='conditions_unavailable'):
            app.book_runtime.settings_for(changed)
    finally:
        await app.aclose()


@pytest.mark.parametrize('level', ['a2', 'b1', 'b2'])
async def test_english_help_and_reader_level_keep_policy(tmp_path, level):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True, english_help=True, reader_level=level)
        assert conditions.payload['name_translation_policy'] == {'version': 1, 'strategy': 'sound'}
        assert conditions.payload['prompt_sha256'] == payload_digest(conditions.payload['prompt_templates'])
    finally:
        await app.aclose()
