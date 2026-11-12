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
        assert conditions.payload['versions']['schemas'] == 5
        assert conditions.payload['versions']['prompts'] == 5
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


@pytest.mark.parametrize('invalid_policy', [None, [], 'sound', {'version': 1}, {'version': 1, 'strategy': 'sound', 'extra': True}])
async def test_explicit_null_policy_refused(tmp_path, invalid_policy):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        conditions = app.book_conditions.require('book-1')
        changed = replace(conditions, payload={**conditions.payload, 'name_translation_policy': invalid_policy})
        with pytest.raises(ConditionsError, match='conditions_unavailable'):
            app.book_runtime.settings_for(changed)
        with pytest.raises(ConditionsError, match='conditions_unavailable'):
            app.book_runtime.factory(app.book_runtime.settings_for(conditions), conditions=changed)
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


@pytest.mark.parametrize('variant', ['uk', 'us'])
async def test_pronunciation_conditions_uk_us(tmp_path, variant):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)(
            'book-1', action='reanalyze', expected_revision=0, settings_revision=0,
            confirmed=True, pronunciation_help_enabled=True, pronunciation_variant=variant)
        assert conditions.pronunciation_help_enabled is True
        assert conditions.pronunciation_variant == variant
        assert conditions.payload['english_help'] is False
        assert conditions.payload['reader_level'] == 'b2'
        assert conditions.payload['prompt_templates']['pronunciation_candidates_version'] == 1
        assert 'pronunciation_processing' in conditions.payload['prompt_templates']
        assert conditions.payload['prompt_sha256'] == payload_digest(conditions.payload['prompt_templates'])
        assert conditions.payload['pronunciation_versions'] == {
            'selection_policy': 2, 'form_rules': 1,
            'source': 'ipa-dict@972cd67dd5f96bc5660e28070cde26e3928abcd7:parser-1'}
        assert app.book_conditions.require('book-1') == conditions
        app.book_runtime.settings_for(conditions)
    finally:
        await app.aclose()


async def test_legacy_conditions_disable_pronunciation(tmp_path):
    from linguary.domain.book_conditions import BookConditions
    app = make_app(tmp_path)
    try:
        store_imported(app)
        await app.analysis.start('book-1')
        original = app.book_conditions.require('book-1').to_dict()
        for key in ('pronunciation_help_enabled', 'pronunciation_variant', 'pronunciation_versions'):
            original['payload'].pop(key, None)
        legacy = BookConditions.from_dict(original)
        assert legacy.pronunciation_help_enabled is False
        assert legacy.pronunciation_variant is None
        assert legacy.to_dict() == original
        app.book_runtime.settings_for(legacy)
    finally:
        await app.aclose()


async def test_pronunciation_rejects_non_english(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        with app.unit_of_work.transaction() as transaction:
            book = transaction.books.get('book-1')
            transaction.books.save(replace(book, source_language='fr'))
        with pytest.raises(ConditionsError, match='pronunciation_requires_english'):
            ChooseBookConditions(app.book_conditions, app.book_runtime)(
                'book-1', action='reanalyze', expected_revision=0, settings_revision=0,
                confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.latest_conditions('book-1') is None
    finally:
        await app.aclose()


@pytest.mark.parametrize('variant', [None, 'au', 'UK'])
async def test_pronunciation_rejects_invalid_variant(tmp_path, variant):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        with pytest.raises(ConditionsError, match='invalid_pronunciation_variant'):
            ChooseBookConditions(app.book_conditions, app.book_runtime)(
                'book-1', action='reanalyze', expected_revision=0, settings_revision=0,
                confirmed=True, pronunciation_help_enabled=True, pronunciation_variant=variant)
    finally:
        await app.aclose()


async def test_pronunciation_conditions_revision_conflict(tmp_path):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        choice = ChooseBookConditions(app.book_conditions, app.book_runtime)
        old = choice('book-1', action='reanalyze', expected_revision=0, settings_revision=0,
            confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        with pytest.raises(ConditionsError, match='conditions_revision_conflict'):
            choice('book-1', action='reanalyze', expected_revision=0, settings_revision=0,
                confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='us')
        new = choice('book-1', action='reanalyze', expected_revision=1, settings_revision=0,
            confirmed=True, pronunciation_help_enabled=False, pronunciation_variant='us')
        assert new.revision == 2 and new.pronunciation_variant is None
        assert old.pronunciation_variant == 'uk'
    finally:
        await app.aclose()


async def test_pronunciation_version_change_creates_revision(tmp_path, monkeypatch):
    import linguary.application.book_conditions as module
    app = make_app(tmp_path)
    try:
        store_imported(app)
        choice = ChooseBookConditions(app.book_conditions, app.book_runtime)
        old = choice('book-1', action='reanalyze', expected_revision=0, settings_revision=0,
            confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        monkeypatch.setattr(module, 'PRONUNCIATION_VERSIONS', {
            'selection_policy': 3, 'form_rules': 1, 'source': 'disabled'})
        new = choice('book-1', action='reanalyze', expected_revision=1, settings_revision=0,
            confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        assert old.revision == 1 and new.revision == 2
        assert old.payload['pronunciation_versions']['selection_policy'] == 2
        assert new.payload['pronunciation_versions']['selection_policy'] == 3
    finally:
        await app.aclose()


@pytest.mark.parametrize('encoding', ['json', 'form'])
async def test_pronunciation_conditions_http_contract(tmp_path, encoding):
    from httpx import ASGITransport, AsyncClient
    from linguary.web import create_app
    app = make_app(tmp_path)
    try:
        store_imported(app)
        web = create_app(application=app, run_background_worker=False)
        web.state.linguary_application = app
        choice = dict(action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True,
                      pronunciation_help_enabled=True, pronunciation_variant='uk')
        async with AsyncClient(transport=ASGITransport(app=web), base_url='http://127.0.0.1') as client:
            async def submit(book_id='book-1', **changes):
                data = {**choice, **changes}
                return await client.post(f'/books/{book_id}/conditions',
                    **({'json': data} if encoding == 'json' else {'data': data}),
                    headers={'accept': 'application/json' if encoding == 'json' else 'text/html'})
            assert (await submit(pronunciation_variant='au')).status_code == 422
            assert (await submit(pronunciation_variant=None)).status_code == 422
            assert (await submit(confirmed=False)).status_code == 422
            assert (await submit(book_id='missing')).status_code == 404
            response = await submit()
            assert response.status_code == (201 if encoding == 'json' else 303)
            if encoding == 'json':
                assert response.json()['revision'] == 1
            assert (await submit()).status_code == 409
            with app.unit_of_work.transaction() as transaction:
                book = transaction.books.get('book-1')
                transaction.books.save(replace(book, source_language='fr'))
            assert (await submit(expected_revision=1)).status_code == 422
    finally:
        await app.aclose()


async def test_source_digest_uses_stat_cache_across_calls_and_invalidates_on_file_change(tmp_path, monkeypatch):
    from pathlib import Path
    import linguary.composition.container as container_module

    app = make_app(tmp_path)
    try:
        store_imported(app)
        source = app.paths.source_epub('book-1').resolve()
        cache = getattr(container_module, '_FILE_SHA256_CACHE', None)
        if cache is not None:
            cache.clear()

        read_count = 0
        real_read_bytes = Path.read_bytes

        def counting_read_bytes(self):
            nonlocal read_count
            if self.resolve() == source:
                read_count += 1
            return real_read_bytes(self)

        monkeypatch.setattr(Path, 'read_bytes', counting_read_bytes)

        write_tx_entered = 0
        in_write_tx = False
        real_transaction = app.unit_of_work.transaction

        from contextlib import contextmanager

        @contextmanager
        def tracking_transaction():
            nonlocal write_tx_entered, in_write_tx
            write_tx_entered += 1
            in_write_tx = True
            try:
                with real_transaction() as tx:
                    yield tx
            finally:
                in_write_tx = False

        digest_in_write_tx = []
        real_source_digest = app.book_conditions.source_digest

        def tracking_source_digest(book):
            digest_in_write_tx.append(in_write_tx)
            return real_source_digest(book)

        monkeypatch.setattr(app.unit_of_work, 'transaction', tracking_transaction)
        monkeypatch.setattr(app.book_conditions, 'source_digest', tracking_source_digest)

        conditions_id = app.book_conditions.freeze_before_analysis('book-1', 0)
        assert read_count == 1
        assert digest_in_write_tx == [False]

        tx_before_checks = write_tx_entered
        assert app.book_conditions.require('book-1').id == conditions_id
        assert app.book_conditions.require('book-1').id == conditions_id
        assert app.book_conditions.freeze_before_analysis('book-1', 0) == conditions_id
        assert write_tx_entered == tx_before_checks
        assert read_count == 1

        with app.unit_of_work.transaction() as tx:
            book = tx.books.get('book-1')
            pinned = app.book_conditions.create(
                tx,
                book,
                0,
                source_sha256=app.book_conditions.require('book-1', transaction=tx).payload['source_sha256'],
                seed_first_chapter=True,
            )
            assert pinned.payload['checks']['seed_first_chapter'] is True
        assert read_count == 1

        source.write_bytes(b'changed synthetic source bytes')
        with pytest.raises(ConditionsError, match='source_digest_mismatch'):
            app.book_conditions.require('book-1')
        assert read_count == 2

        from types import SimpleNamespace
        from linguary.composition.book_runtime import ProvenanceClient

        def fail_lookup(_id):
            raise AssertionError('unexpected duplicate lookup')

        dummy_tx = SimpleNamespace(work=SimpleNamespace(get_root=fail_lookup, get_scope=fail_lookup))
        dummy_root = SimpleNamespace(superseded_by=None, status='running', generation=1)
        dummy_scope = SimpleNamespace(generation=2, status='claimed')
        dummy_ctx = SimpleNamespace(root_id='r1', scope_id='s1', generation=2)
        dummy_attempt = SimpleNamespace(context={'root_generation': 1})
        provenance = ProvenanceClient(FakeLLMClient(), app.book_conditions, conditions_id)
        assert provenance._has_operation_ownership(
            dummy_tx, dummy_ctx, dummy_attempt, root=dummy_root, scope=dummy_scope
        ) is True
    finally:
        await app.aclose()


async def test_complete_analysis_updates_archived_analysis_binding_without_breaking_reanalyze(tmp_path):
    shared_byr = [{'expression': 'entailed estate', 'explanation': 'имение с ограничением наследования'}]
    reply_with_byr = FakeReply.of_payload({
        'treatment_plan': [{'source_block_id': 'b1', 'treatment': 'READ_AS_IS'}],
        'glossary_candidates': [],
        'translation_instructions': 'Сохраняй смысл.',
        'chapter_summary': 'The cat sat.',
        'before_you_read': shared_byr,
    })
    app = make_app(tmp_path, FakeLLMClient([reply_with_byr, reply_with_byr]))
    try:
        store_imported(app, (('Entailed estate in chapter one.',), ('Entailed estate in chapter two.',)))
        await app.analysis.start('book-1')
        completed = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in completed)

        with app.unit_of_work.transaction() as tx:
            conditions_v1 = tx.work.latest_conditions('book-1')
            analyses = tx.analyses.list_for_book('book-1')
            assert len(analyses) == 2
            ch1, ch2 = analyses
            assert len(ch1.before_you_read) == 1
            assert ch2.before_you_read == ()

            binding_1 = tx.work.get_binding('analysis', ch1.id)
            binding_2 = tx.work.get_binding('analysis', ch2.id)
            assert binding_1['conditions_id'] == conditions_v1.id
            assert binding_2['conditions_id'] == conditions_v1.id
            assert binding_2['attempt_id'] is not None
            assert tx.work.get_attempt(binding_2['attempt_id']).state == 'accepted'

            archived_2 = tx.work.get_artifact(binding_2['artifact_revision_id'])
            assert archived_2['body']['analysis']['before_you_read'] == []
            refreshed_revision_id = binding_2['artifact_revision_id']
            refreshed_attempt_id = binding_2['attempt_id']

        conditions_v2 = ChooseBookConditions(app.book_conditions, app.book_runtime)(
            'book-1',
            action='reanalyze',
            expected_revision=1,
            settings_revision=0,
            confirmed=True,
        )
        assert conditions_v2.revision == 2

        with app.unit_of_work.transaction() as tx:
            assert tx.analyses.list_for_book('book-1') == ()
            binding_after_reanalyze = tx.work.get_binding('analysis', ch2.id)
            assert binding_after_reanalyze['conditions_id'] == conditions_v1.id
            assert binding_after_reanalyze['attempt_id'] == refreshed_attempt_id
            assert binding_after_reanalyze['artifact_revision_id'] == refreshed_revision_id
    finally:
        await app.aclose()


async def test_ensure_initial_for_analysis_creates_or_reuses_matching_conditions_and_validates_variant(tmp_path):
    from linguary.application.book_overview import BookNotFoundError

    app = make_app(tmp_path)
    try:
        with pytest.raises(BookNotFoundError):
            app.book_conditions.ensure_initial_for_analysis('missing-book', reader_level='a2')

        store_imported(app)

        def reject_conditions(_conditions):
            raise ConditionsError('conditions_unavailable')

        with pytest.raises(ConditionsError, match='conditions_unavailable'):
            app.book_conditions.ensure_initial_for_analysis(
                'book-1',
                reader_level='a2',
                validate_conditions=reject_conditions,
            )
        with app.unit_of_work.transaction() as tx:
            assert tx.work.latest_conditions('book-1') is None

        created = app.book_conditions.ensure_initial_for_analysis(
            'book-1',
            reader_level='a2',
            english_help=True,
            pronunciation_help_enabled=True,
            pronunciation_variant='uk',
            seed_first_chapter=True,
            default_seed_first_chapter=False,
            validate_conditions=app.book_runtime.settings_for,
        )
        assert created.revision == 1
        assert created.payload['reader_level'] == 'a2'
        assert created.payload['english_help'] is True
        assert created.pronunciation_help_enabled is True
        assert created.pronunciation_variant == 'uk'
        assert created.payload['checks']['seed_first_chapter'] is True

        reused = app.book_conditions.ensure_initial_for_analysis(
            'book-1',
            reader_level='a2',
            english_help=None,
            pronunciation_help_enabled=None,
            pronunciation_variant=None,
            seed_first_chapter=None,
            default_seed_first_chapter=False,
            validate_conditions=app.book_runtime.settings_for,
        )
        assert reused.id == created.id
        assert reused.revision == 1

        for bad_variant in ('', 'au'):
            with pytest.raises(ConditionsError, match='invalid_pronunciation_variant'):
                app.book_conditions.ensure_initial_for_analysis(
                    'book-1',
                    pronunciation_help_enabled=None,
                    pronunciation_variant=bad_variant,
                    default_seed_first_chapter=False,
                    validate_conditions=app.book_runtime.settings_for,
                )
        assert app.book_conditions.require('book-1').id == created.id

        updated = app.book_conditions.ensure_initial_for_analysis(
            'book-1',
            pronunciation_help_enabled=None,
            pronunciation_variant='us',
            default_seed_first_chapter=False,
            validate_conditions=app.book_runtime.settings_for,
        )
        assert updated.revision == 2
        assert updated.pronunciation_help_enabled is True
        assert updated.pronunciation_variant == 'us'

        await app.analysis.start('book-1')
        reused_analyzing = app.book_conditions.ensure_initial_for_analysis(
            'book-1',
            reader_level='a2',
            validate_conditions=app.book_runtime.settings_for,
        )
        assert reused_analyzing.id == updated.id
        with pytest.raises(ConditionsError, match='conditions_already_selected'):
            app.book_conditions.ensure_initial_for_analysis(
                'book-1',
                reader_level='b1',
                validate_conditions=app.book_runtime.settings_for,
            )
    finally:
        await app.aclose()


async def test_staged_runtime_uses_pinned_models_and_reuses_narrative_on_reanalyze(tmp_path, monkeypatch):
    from linguary.adapters.llm import OpenAICompatibleLLMClient
    from linguary.domain import AnalysisPassKind
    from linguary.application.book_overview import book_overview
    from linguary.application.use_cases.reset_processing import ResetProcessing, ResetProcessingCommand
    bodies = []

    async def post(client, body):
        bodies.append(body)
        kind = body['response_format']['json_schema']['name']
        if kind == 'chapter_analysis_narrative':
            payload = dict(translation_instructions='Keep the voice.', chapter_summary='The cat sat.',
                glossary_candidates=[], context_facts=[], scenes=[dict(start_block_id='b1',
                    end_block_id='b1', summary='The cat sits.', location=None, participants=[],
                    visual_facts=[], visual_assumptions=[])])
        elif kind == 'chapter_analysis_pedagogical':
            payload = dict(treatment_plan=[dict(source_block_id='b1', treatment='READ_AS_IS')],
                before_you_read=[], difficulties=[], before_you_read_evidence=[])
            if sum(b['response_format']['json_schema']['name'] == kind for b in bodies) == 1:
                payload['treatment_plan'] = []  # Valid schema, rejected coverage, then retry.
            if 'reader_chapter_summary' in body['response_format']['json_schema']['schema']['properties']:
                payload['reader_chapter_summary'] = 'The cat sat.\n\nКот сидел.'
        else:
            raise AssertionError(f'Unexpected request kind: {kind}')
        usage = dict(prompt_tokens=100, completion_tokens=25)
        if kind == 'chapter_analysis_narrative':
            usage['cost'] = 0.02
        return dict(model=body['model'], usage=usage,
            choices=[dict(finish_reason='stop', message=dict(content=json.dumps(payload)))])

    monkeypatch.setattr(OpenAICompatibleLLMClient, '_post', post)
    settings = AppSettings(llm=LLMSettings(model='base', analysis_model='analysis',
        narrative_analysis_model='story', pedagogical_analysis_model='teaching',
        max_output_tokens=16000, api_key='synthetic-secret', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data')))
    app = build_application(settings=settings, content_plan_profile='content-plan-v3')
    try:
        document = store_imported(app)
        await app.analysis.start('book-1')
        done = await drain_tasks(app)
        assert all(t.status is TaskStatus.SUCCEEDED for t in done), [(t.status, t.last_error) for t in done]
        assert [(b['response_format']['json_schema']['name'], b['model'], b['max_completion_tokens'])
                for b in bodies] == [('chapter_analysis_narrative', 'story', 16000),
                                     ('chapter_analysis_pedagogical', 'teaching', 16000),
                                     ('chapter_analysis_pedagogical', 'teaching', 16000)]
        with app.unit_of_work.transaction() as tx:
            calls = tx.llm_calls.list_for_book('book-1')
            for call in calls:
                assert tx.work.get_attempt(call.attempt_id).context['assigned_model'] == call.model
            first = tx.analyses.get_pass('book-1', document.sections[0].id, AnalysisPassKind.NARRATIVE)
            assert first is not None
        old_conditions = app.book_conditions.require('book-1')
        # Current global model must never alter an already pinned route.
        await app.system_settings.save(SystemSettingsUpdate(provider='openai_compatible', model='global',
            base_url='https://api.openai.com/v1', requests_per_minute=None, concurrency=1), expected_revision=0)
        pinned = app.book_runtime.settings_for(old_conditions)
        assert pinned.llm.staged_analysis is True
        assert pinned.llm.effective_analysis_pass_model('narrative') == 'story'
        choice = ChooseBookConditions(app.book_conditions, app.book_runtime)
        for revision, level, help_mode in [(1, 'a2', False), (2, 'b2', True)]:
            choice('book-1', action='reanalyze', expected_revision=revision,
                settings_revision=1, confirmed=True, reader_level=level, english_help=help_mode)
            before = len(bodies)
            await app.analysis.start('book-1')
            done = await drain_tasks(app)
            assert all(t.status is TaskStatus.SUCCEEDED for t in done), [(t.status, t.last_error) for t in done]
            assert [b['response_format']['json_schema']['name'] for b in bodies[before:]] == ['chapter_analysis_pedagogical']
            with app.unit_of_work.transaction() as tx:
                assert tx.analyses.get_pass('book-1', document.sections[0].id, AnalysisPassKind.NARRATIVE) == first
                analysis = tx.analyses.list_for_book('book-1')[0]
                assert analysis.chapter_summary == ('The cat sat.\n\nКот сидел.' if level == 'a2' else 'The cat sat.')
        overview = book_overview(app.unit_of_work, 'book-1')
        stats = overview.ordinary_cost['analysis']
        assert stats['calls'] == 5
        assert stats['prompt_tokens'] == 500
        assert stats['completion_tokens'] == 125
        assert stats['cost_usd'] is None
        assert stats['known_spent_usd'] == '0.02'
        assert stats['unknown_calls'] == 4
        assert stats['passes']['narrative']['cost_usd'] == '0.02'
        assert stats['passes']['pedagogical']['cost_usd'] is None
        assert stats['passes']['narrative']['calls'] == 1
        assert stats['passes']['pedagogical']['calls'] == 4
        with app.unit_of_work.transaction() as tx:
            tx.books.save(replace(tx.books.get('book-1'), status=BookStatus.PROCESSING))
            passes = tx.analyses.list_passes_for_section('book-1', document.sections[0].id)
        ResetProcessing(unit_of_work=app.unit_of_work)(ResetProcessingCommand('book-1'))
        with app.unit_of_work.transaction() as tx:
            assert tx.analyses.list_passes_for_section('book-1', document.sections[0].id) == passes
            assert tx.analyses.list_for_book('book-1')[0] == analysis
    finally:
        await app.aclose()


@pytest.mark.parametrize('historical_snapshot', [False, True])
async def test_continue_legacy_with_current_opt_in_keeps_single_pass(tmp_path, historical_snapshot):
    settings = AppSettings(llm=LLMSettings(model='base', api_key='synthetic-secret',
        staged_analysis=True, narrative_analysis_model='story', pedagogical_analysis_model='teaching',
        proxy_mode='direct'), storage=StorageSettings(data_root=str(tmp_path / 'data')))
    fake = FakeLLMClient([FakeReply.of_payload(dict(
        treatment_plan=[dict(source_block_id='b1', treatment='READ_AS_IS')],
        glossary_candidates=[], translation_instructions='Keep the voice.', chapter_summary='The cat sat.',
        before_you_read=[], content_plan=dict(version=3, profile='content-plan-v3',
            difficulties=[], context_facts=[], before_you_read_evidence=[])))])
    app = build_application(settings=settings, llm_client=fake, content_plan_profile='content-plan-v3')
    try:
        store_imported(app)
        conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1',
            action='continue_legacy', expected_revision=0, settings_revision=0, confirmed=True)
        assert conditions.payload['staged_analysis'] is False
        if historical_snapshot:
            from linguary.domain.book_conditions import thaw
            from uuid import uuid4
            payload = thaw(conditions.payload)
            payload.pop('staged_analysis')
            payload['models'].pop('analysis_passes')
            for key in ('staged_analysis', 'narrative_analysis_model', 'pedagogical_analysis_model'):
                payload['llm_parameters'].pop(key)
            for key in ('narrative_analysis', 'pedagogical_analysis'):
                payload['prompt_templates'].pop(key)
            payload['prompt_sha256'] = payload_digest(payload['prompt_templates'])
            conditions = replace(conditions, id=str(uuid4()), revision=2, payload=payload)
            with app.unit_of_work.transaction() as tx:
                tx.work.add_conditions(conditions)
        await app.analysis.start('book-1')
        done = await drain_tasks(app)
        assert all(t.status is TaskStatus.SUCCEEDED for t in done)
        assert [r.stage.value for r in fake.requests] == ['chapter_analysis']
        assert app.book_conditions.require('book-1').id == conditions.id
        from linguary.application.book_overview import book_overview
        stats = book_overview(app.unit_of_work, 'book-1').ordinary_cost['analysis']
        assert stats['calls'] == stats['passes']['single']['calls'] == 1
        assert stats['passes']['narrative']['calls'] == stats['passes']['pedagogical']['calls'] == 0
    finally:
        await app.aclose()
