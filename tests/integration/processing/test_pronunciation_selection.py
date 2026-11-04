"""Synthetic pronunciations pass through real JSON, pinned runtime and SQLite."""
from dataclasses import replace
import json

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeReply
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from linguary.application.llm_dto import PronunciationAnchorResponse, PronunciationCandidateResponse, PronunciationContextResponse
from linguary.application.ports.pronunciation import InflectionRule, SourceLookupResult
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.reset_processing import ChooseBookConditions
from linguary.application.validators.pronunciation import group_pronunciations, verify_candidates
from linguary.composition import build_application, drain_tasks
from linguary.domain import TaskStatus
from test_book_conditions import store_imported
from test_pronunciation_source import entry


class SyntheticSource:
    configured = True

    def __init__(self, entries=()):
        self.entries = tuple(entries)
        self.calls = []
        self.failures = set()

    async def lookup(self, form, variant):
        self.calls.append((form, variant))
        if form in self.failures:
            raise TimeoutError('SOURCE_PRIVATE_ERROR_MARKER')
        matches = tuple(item for item in self.entries if item.form == form and item.variant == variant)
        return SourceLookupResult.found(matches) if matches else SourceLookupResult.missing()


def candidate(form='zorp', *, block='b1', start=0, sense='invented action', grammar='present', proper=False):
    return PronunciationCandidateResponse(PronunciationAnchorResponse(block, start, start + len(form), form), form,
        PronunciationContextResponse(sense, 'verb', grammar, proper), 'unpredictable_spelling')


def candidate_json(item):
    from dataclasses import asdict
    return asdict(item)


def analysis_reply(blocks=1, profile='legacy-v1', treatment='READ_AS_IS'):
    data = dict(glossary_candidates=[], translation_instructions='Сохраняй смысл.',
        chapter_summary='An invented action happens.\n\nПроисходит вымышленное действие.',
        treatment_plan=[dict(source_block_id=f'b{i}', treatment=treatment) for i in range(1, blocks + 1)], before_you_read=[])
    if profile != 'legacy-v1':
        data['content_plan'] = dict(version=3, profile=profile, difficulties=[], before_you_read_evidence=[], context_facts=[])
    return FakeReply.of_payload(data)


def processing_reply(candidates=(), *, blocks=1, english_help=False, profile='legacy-v1', treatment='READ_AS_IS', translated_text=None, glossary_hints=()):
    data = dict(blocks=[dict(source_block_id=f'b{i}', treatment=treatment, translated_text=translated_text, glossary_hints=list(glossary_hints))
                       for i in range(1, blocks + 1)], language_notes=[],
                pronunciation_candidates=[candidate_json(item) for item in candidates])
    if profile != 'legacy-v1' or english_help:
        data['content_evidence'] = []
    if english_help:
        data['fragment_assistances'] = []
    return FakeReply.of_payload(data)


async def prepare_app(tmp_path, source, candidates=(), *, text='zorp', english_help=False, profile='legacy-v1', replies=(), treatment='READ_AS_IS', translated_text=None, glossary_hints=(), processing_response=None, analysis_response=None, max_attempts=1, llm_client=None):
    fake = llm_client or FakeLLMClient([analysis_response or analysis_reply(profile=profile, treatment=treatment), processing_response or processing_reply(candidates,
        english_help=english_help, profile=profile, treatment=treatment, translated_text=translated_text, glossary_hints=glossary_hints), *replies])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct', max_attempts=max_attempts),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake,
        pronunciation_source=source, content_plan_profile=profile)
    document = store_imported(app, ((text,),))
    conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)(
        'book-1', action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True,
        pronunciation_help_enabled=True, pronunciation_variant='uk', english_help=english_help)
    await app.analysis.start('book-1')
    tasks = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
    run = app.start_processing()(StartProcessingCommand('book-1'))
    await app.processing.start('book-1')
    tasks = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
    return app, fake, document, run, conditions


@pytest.mark.parametrize('english_help,profile', [(False, 'legacy-v1'), (False, 'content-plan-v3'), (True, 'content-plan-v3')])
async def test_pronunciation_read_as_is_without_translation(tmp_path, english_help, profile):
    source = SyntheticSource([entry()])
    app, fake, document, run, conditions = await prepare_app(tmp_path, source, [candidate()], english_help=english_help, profile=profile)
    try:
        with app.unit_of_work.transaction() as tx:
            result, = tx.results.list_for_run(run.id)
        accepted, = result.pronunciation_assistances
        assert accepted.anchor.block_id == document.sections[0].blocks[0].id
        assert accepted.ipa == 'zɔːp'
        assert result.translated_text is None and result.glossary_hints == result.fragment_assistances == ()
        assert source.calls == [('zorp', 'uk')]
        assert fake.scripted_requests[-1].pronunciation_candidates_version == 1
        assert conditions.payload['prompt_templates']['pronunciation_candidates_version'] == 1
    finally:
        await app.aclose()


async def test_pronunciation_homographs_and_inflections():
    present = entry()
    past = replace(present, ipa='zɛp', grammatical_form='past', evidence=replace(present.evidence, dictionary_ipa='zɛp', sense_id='past'))
    rule = InflectionRule('synthetic-plural', '1', 'plural', 's', 's')
    plural = replace(present, form='zorps', ipa='zɔːps', grammatical_form='plural', inflection_rule=rule,
        evidence=replace(present.evidence, rule_id=rule.id, rule_version=rule.version))
    source = SyntheticSource([present, past, plural])
    candidates = [candidate(), candidate(start=5, grammar='past'), candidate('zorps', start=10, grammar='plural')]
    selected = await verify_candidates(candidates, source=source, texts={'b1': 'zorp zorp zorps'}, variant='uk', scope_id='run')
    assert [item.ipa for item in selected.accepted] == ['zɔːp', 'zɛp', 'zɔːps']
    assert len({item.hint_id for item in selected.accepted}) == 3
    assert selected.accepted[-1].evidence.rule_id == 'synthetic-plural'
    source.entries = (replace(plural, ipa='unproven'),)
    assert not (await verify_candidates(candidates[-1:], source=source, texts={'b1': 'zorp zorp zorps'}, variant='uk', scope_id='run')).accepted


async def test_pronunciation_rejects_unproven_names_and_ambiguity():
    source = SyntheticSource([entry()])
    assert not (await verify_candidates([candidate(proper=True)], source=source, texts={'b1': 'zorp'}, variant='uk', scope_id='run')).accepted
    source.entries = (replace(entry(), proper_name_confirmed=True),)
    assert (await verify_candidates([candidate(proper=True)], source=source, texts={'b1': 'zorp'}, variant='uk', scope_id='run')).accepted
    source.entries += (replace(source.entries[0], ipa='zɛp', evidence=replace(source.entries[0].evidence, dictionary_ipa='zɛp')),)
    assert not (await verify_candidates([candidate()], source=source, texts={'b1': 'zorp'}, variant='uk', scope_id='run')).accepted


async def test_pronunciation_no_level_uses_context_and_rejects_length_or_rarity():
    source = SyntheticSource([entry()])
    items = [candidate(), replace(candidate(start=5), selection_reason='rare'),
             replace(candidate(start=10), context=replace(candidate().context, meaning_in_context=None))]
    selected = await verify_candidates(items, source=source, texts={'b1': 'zorp zorp zorp'}, variant='uk', scope_id='run')
    assert len(selected.accepted) == 1
    assert {item.reason for item in selected.skipped} == {'selection_reason', 'unknown_context'}


async def test_lexicon_unique_exact_form_accepted():
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    item = replace(candidate('queue'), context=PronunciationContextResponse(None, None, None, False))
    selected = await verify_candidates([item], source=IpaDictPronunciationSource(),
        texts={'b1': 'queue'}, variant='uk', scope_id='run')
    accepted, = selected.accepted
    assert accepted.ipa == 'kjˈuː'
    assert accepted.evidence.kind == 'lexicon_form'
    assert accepted.evidence.sense_id is accepted.evidence.part_of_speech is None


@pytest.mark.parametrize('form,proper', [('read', False), ('the', False), ('queue', True)])
async def test_lexicon_multiple_ipa_and_proper_name_skipped(form, proper):
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    selected = await verify_candidates([candidate(form, proper=proper)], source=IpaDictPronunciationSource(),
        texts={'b1': form}, variant='us', scope_id='run')
    assert selected.accepted == ()
    assert selected.skipped[0].reason == 'missing_or_ambiguous'


async def test_lexicon_missing_inflected_form_not_invented():
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    selected = await verify_candidates([candidate('queuexyz', grammar='plural')], source=IpaDictPronunciationSource(),
        texts={'b1': 'queuexyz'}, variant='uk', scope_id='run')
    assert selected.accepted == ()


async def test_lexicon_fake_semantic_fields_rejected():
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    item, = (await IpaDictPronunciationSource().lookup('queue', 'uk')).entries
    object.__setattr__(item, 'sense', 'invented action')
    selected = await verify_candidates([candidate('queue')], source=SyntheticSource([item]),
        texts={'b1': 'queue'}, variant='uk', scope_id='run')
    assert selected.accepted == ()


async def test_ipa_dict_failure_preserves_previous():
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    from linguary.application.validators.pronunciation import preserve_unavailable
    item = candidate('queue')
    source = IpaDictPronunciationSource()
    selected = await verify_candidates([item], source=source, texts={'b1': 'queue'}, variant='uk', scope_id='run')
    failing = SyntheticSource()
    failing.failures.add('queue')
    failed = await verify_candidates([item], source=failing,
        texts={'b1': 'queue'}, variant='uk', scope_id='run')
    assert preserve_unavailable(failed, selected.accepted, texts={'b1': 'queue'}, variant='uk') == selected.accepted


@pytest.mark.parametrize('source_version', ['disabled', 'ipa-dict@unknown:parser-1'])
async def test_old_disabled_and_unknown_conditions_never_enable_ipa_dict(tmp_path, monkeypatch, source_version):
    from linguary.application import book_conditions
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    calls = []
    async def forbidden_lookup(*args):
        calls.append(args)
        raise AssertionError('unsupported conditions must not use the current snapshot')
    monkeypatch.setattr(IpaDictPronunciationSource, 'lookup', forbidden_lookup)
    with monkeypatch.context() as patch:
        patch.setattr(book_conditions, 'PRONUNCIATION_VERSIONS', {
            'selection_policy': 1, 'form_rules': 1, 'source': source_version})
        app, _, _, run, conditions = await prepare_app(tmp_path, None, [candidate('queue')], text='queue',
            treatment='TRANSLATE', translated_text='Очередь.')
    try:
        with app.unit_of_work.transaction() as tx:
            result, = tx.results.list_for_run(run.id)
        assert result.pronunciation_assistances == ()
        assert result.translated_text == 'Очередь.'
        assert conditions.payload['pronunciation_versions']['source'] == source_version
        assert calls == []
    finally:
        await app.aclose()


async def test_ipa_dict_runtime_process_regenerate_recovery(tmp_path):
    from linguary.adapters.pronunciation.source import SOURCE_VERSION
    def translation(count, *, bad=False, candidates=()):
        return FakeReply.of_payload(dict(blocks=[dict(source_block_id=f'b{i}', treatment='TRANSLATE',
            translated_text='«Он увидел очередь дома.' if bad and i == count else 'Он увидел очередь дома.',
            glossary_hints=[]) for i in range(1, count + 1)], language_notes=[],
            pronunciation_candidates=[candidate_json(item) for item in candidates]))
    items = [candidate('queue', start=7)]
    fake = FakeLLMClient([analysis_reply(2, treatment='TRANSLATE'), translation(2, bad=True),
        translation(2, bad=True), translation(1, candidates=items), translation(1, candidates=items),
        translation(1, candidates=items)])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake)
    try:
        document = store_imported(app, (('He saw queue at home.', 'He saw queue at home.'),))
        conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze',
            expected_revision=0, settings_revision=0, confirmed=True,
            pronunciation_help_enabled=True, pronunciation_variant='uk')
        assert conditions.payload['pronunciation_versions']['source'] == SOURCE_VERSION
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        tasks = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
        await regenerate(app, document.sections[0].blocks[0].id)
        with app.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(run.id)
        assert len(results) == 2
        assert all(result.pronunciation_assistances[0].evidence.source_version == SOURCE_VERSION for result in results)
        assert all(result.pronunciation_assistances[0].ipa == 'kjˈuː' for result in results)
        assert len(fake.scripted_requests) == 6
    finally:
        await app.aclose()


async def test_ipa_dict_content_never_reaches_model(tmp_path):
    from linguary.adapters.pronunciation.source import SOURCE_VERSION
    app, fake, _, run, _ = await prepare_app(tmp_path, None, [candidate('queue')], text='queue')
    try:
        with app.unit_of_work.transaction() as tx:
            accepted, = tx.results.list_for_run(run.id)[0].pronunciation_assistances
        for request in fake.requests:
            assert SOURCE_VERSION not in request.instructions + request.input_text
            assert accepted.ipa not in request.instructions + request.input_text
            assert accepted.evidence.source_uri not in request.instructions + request.input_text
    finally:
        await app.aclose()


async def test_pronunciation_primary_ignores_completion_order():
    source = SyntheticSource([entry()])
    later = (await verify_candidates([candidate(block='later')], source=source, texts={'later': 'zorp'}, variant='uk', scope_id='run')).accepted[0]
    earlier = (await verify_candidates([candidate(block='earlier')], source=source, texts={'earlier': 'zorp'}, variant='uk', scope_id='run')).accepted[0]
    assert earlier.hint_id == later.hint_id
    group, = group_pronunciations([later, earlier], block_order={'earlier': 0, 'later': 1})
    assert group.primary == earlier
    assert group.occurrences == (earlier, later)


async def test_pronunciation_duplicate_and_conflicting_anchors_are_order_independent():
    source = SyntheticSource([entry()])
    same = await verify_candidates([candidate(), candidate()], source=source, texts={'b1': 'zorp'}, variant='uk', scope_id='run')
    assert len(same.accepted) == 1 and len(source.calls) == 1
    different = replace(candidate(), context=replace(candidate().context, grammatical_form='past'))
    for items in ([candidate(), different], [different, candidate()]):
        assert not (await verify_candidates(items, source=source, texts={'b1': 'zorp'}, variant='uk', scope_id='run')).accepted


async def regenerate(app, block_id):
    await app.regeneration.start('book-1', block_id)
    tasks = await drain_tasks(app)
    assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]


async def test_pronunciation_partial_failure_keeps_confirmed_new_and_previous_failed_anchors(tmp_path):
    original = entry()
    second = entry(form='bip', ipa='bɪp')
    source = SyntheticSource([original, second])
    items = [candidate(), candidate('bip', start=5)]
    app, fake, document, run, _ = await prepare_app(tmp_path, source, items, text='zorp bip', replies=[
        processing_reply(items), processing_reply(items), processing_reply([])])
    block_id = document.sections[0].blocks[0].id
    try:
        with app.unit_of_work.transaction() as tx:
            old = tx.results.get(run.id, block_id).pronunciation_assistances
        source.entries = (replace(original, ipa='ˈzɔːp', evidence=replace(original.evidence, dictionary_ipa='ˈzɔːp')),
                          replace(second, ipa='ˈbɪp', evidence=replace(second.evidence, dictionary_ipa='ˈbɪp')))
        source.failures = {'bip'}
        await regenerate(app, block_id)
        with app.unit_of_work.transaction() as tx:
            partial = tx.results.get(run.id, block_id)
        assert [item.ipa for item in partial.pronunciation_assistances] == ['ˈzɔːp', 'bɪp']
        assert partial.pronunciation_assistances[1] == old[1]
        assert partial.pronunciation_assistances[0].hint_id != old[0].hint_id
        source.failures.clear()
        source.entries = source.entries[:1]
        await regenerate(app, block_id)
        with app.unit_of_work.transaction() as tx:
            checked = tx.results.get(run.id, block_id).pronunciation_assistances
        assert [item.form for item in checked] == ['zorp']
        source.configured = False
        await regenerate(app, block_id)
        with app.unit_of_work.transaction() as tx:
            assert tx.results.get(run.id, block_id).pronunciation_assistances == checked
            assert tx.documents.get('book-1').sections[0].blocks[0].text == 'zorp bip'
        assert len(fake.scripted_requests) == 5
        assert all('SOURCE_PRIVATE' not in request.instructions + request.input_text for request in fake.requests)
    finally:
        await app.aclose()


async def test_pronunciation_dictionary_never_reaches_model(tmp_path):
    private = replace(entry(), ipa='ˈzɔːpˌɪvɪdəns', evidence=replace(entry().evidence,
        dictionary_ipa='ˈzɔːpˌɪvɪdəns', attribution='SOURCE_PRIVATE_ATTRIBUTION', entry_id='SOURCE_PRIVATE_ENTRY'))
    source = SyntheticSource([private])
    app, fake, document, _, _ = await prepare_app(tmp_path, source, [candidate()], replies=[processing_reply([candidate()])])
    try:
        source.failures = {'zorp'}
        await regenerate(app, document.sections[0].blocks[0].id)
        for request in fake.requests:
            assert 'SOURCE_PRIVATE' not in request.input_text + request.instructions
            assert private.ipa not in request.input_text + request.instructions
    finally:
        await app.aclose()


async def test_pronunciation_candidate_invalid_does_not_discard_translation(tmp_path):
    source = SyntheticSource([entry()])
    bad = [replace(candidate(), form='invented'), candidate(block='b2'),
           replace(candidate(start=99), anchor=PronunciationAnchorResponse('b1', 99, 103, 'zorp'))]
    app, fake, document, run, _ = await prepare_app(tmp_path, source, bad,
        text='He saw zorp at home.', treatment='TRANSLATE', translated_text='Он увидел выдумку дома.',
        glossary_hints=[dict(original='zorp', translation='выдумка', note='Условное название вымышленного объекта.')])
    try:
        with app.unit_of_work.transaction() as tx:
            result, = tx.results.list_for_run(run.id)
        assert result.pronunciation_assistances == ()
        assert result.translated_text == 'Он увидел выдумку дома.'
        assert result.glossary_hints[0].original == 'zorp'
        assert len(fake.scripted_requests) == 2
        assert source.calls == []
        assert result.block_id == document.sections[0].blocks[0].id
    finally:
        await app.aclose()


async def test_pronunciation_old_snapshot_unchanged(tmp_path, monkeypatch):
    from linguary.application import context_builder
    from linguary.application.book_conditions import payload_digest
    original_snapshot = context_builder.snapshot_prompt_templates
    def previous_snapshot(**kwargs):
        kwargs.pop('pronunciation_help_enabled', None)
        return original_snapshot(**kwargs)
    plain = FakeReply.of_payload(dict(blocks=[dict(source_block_id='b1', treatment='READ_AS_IS',
        translated_text=None, glossary_hints=[])], language_notes=[]))
    source = SyntheticSource([entry()])
    fake = FakeLLMClient([analysis_reply(), plain, plain])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake, pronunciation_source=source)
    try:
        document = store_imported(app, (('zorp',),))
        with monkeypatch.context() as patch:
            patch.setattr(context_builder, 'snapshot_prompt_templates', previous_snapshot)
            conditions = ChooseBookConditions(app.book_conditions, app.book_runtime)(
                'book-1', action='reanalyze', expected_revision=0, settings_revision=0,
                confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        snapshot_hash = conditions.payload['prompt_sha256']
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        tasks = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
        block_id = document.sections[0].blocks[0].id
        old = (await verify_candidates([candidate(block=block_id)], source=source,
            texts={block_id: 'zorp'}, variant='uk', scope_id=run.id)).accepted
        with app.unit_of_work.transaction() as tx:
            previous = tx.results.get(run.id, block_id)
            tx.results.replace_results(run.id, [replace(previous, pronunciation_assistances=old)])
        source.calls.clear()
        await regenerate(app, block_id)
        with app.unit_of_work.transaction() as tx:
            current = tx.work.latest_conditions('book-1')
            assert current == conditions
            assert tx.results.get(run.id, block_id).pronunciation_assistances == old
        assert source.calls == []
        assert all(request.pronunciation_candidates_version is None for request in fake.requests)
        assert current.payload['prompt_sha256'] == snapshot_hash == payload_digest(current.payload['prompt_templates'])
        assert 'pronunciation_candidates_version' not in current.payload['prompt_templates']
    finally:
        await app.aclose()


async def test_pronunciation_candidate_mode_pinned_to_conditions(tmp_path):
    source = SyntheticSource([entry()])
    app, fake, document, _, conditions = await prepare_app(tmp_path, source, [candidate()], replies=[processing_reply([candidate()])])
    try:
        changed = replace(app.settings, reading=replace(app.settings.reading, reader_level='a2'))
        app.runtime_settings.publish(app.runtime_settings.prepare(changed), 1)
        await regenerate(app, document.sections[0].blocks[0].id)
        assert fake.scripted_requests[-1].pronunciation_candidates_version == 1
        assert conditions.payload['prompt_templates']['pronunciation_processing'] in fake.scripted_requests[-1].instructions
        assert 'Уровень читателя: b2.' in fake.scripted_requests[-1].instructions
    finally:
        await app.aclose()


async def test_pronunciation_candidates_regeneration_and_recovery(tmp_path):
    from linguary.domain import LLMCallStage
    def translation(count, *, bad=False, candidates=()):
        data = dict(blocks=[dict(source_block_id=f'b{i}', treatment='TRANSLATE',
            translated_text='«Он увидел выдумку дома.' if bad and i == count else 'Он увидел выдумку дома.',
            glossary_hints=[]) for i in range(1, count + 1)], language_notes=[],
            pronunciation_candidates=[candidate_json(item) for item in candidates])
        return FakeReply.of_payload(data)
    private = replace(entry(), ipa='ˈzɔːpˌɪvɪdəns', evidence=replace(entry().evidence,
        dictionary_ipa='ˈzɔːpˌɪvɪdəns', attribution='SOURCE_PRIVATE_ATTRIBUTION', entry_id='SOURCE_PRIVATE_ENTRY'))
    source = SyntheticSource([private])
    fake = FakeLLMClient([analysis_reply(2, treatment='TRANSLATE'), translation(2, bad=True), translation(2, bad=True),
        translation(1, candidates=[candidate(start=7)]), translation(1, candidates=[candidate(start=7)])])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake, pronunciation_source=source)
    try:
        document = store_imported(app, (('He saw zorp at home.', 'He saw zorp at home.'),))
        ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=0,
            settings_revision=0, confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        tasks = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
        with app.unit_of_work.transaction() as tx:
            results = tx.results.list_for_run(run.id)
        assert len(results) == 2 and all(len(result.pronunciation_assistances) == 1 for result in results)
        assert results[0].pronunciation_assistances[0].hint_id == results[1].pronunciation_assistances[0].hint_id
        assert source.calls == [('zorp', 'uk'), ('zorp', 'uk')]
        children = [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT and len(request.ordered_block_ids) == 1]
        assert len(children) == 2
        assert all(request.pronunciation_candidates_version == 1 for request in children)
        assert {request.ordered_block_ids for request in children} == {
            (block.id,) for block in document.sections[0].blocks
        }
        assert all('Контекст исходной сцены' not in request.input_text for request in children)
        assert all('SOURCE_PRIVATE' not in request.instructions + request.input_text
                   and private.ipa not in request.instructions + request.input_text for request in fake.requests)
        assert {result.pronunciation_assistances[0].anchor.block_id for result in results} == {block.id for block in document.sections[0].blocks}
    finally:
        await app.aclose()


async def test_pronunciation_rejects_read_inside_bread_before_lookup():
    source = SyntheticSource([entry(form='read')])
    selected = await verify_candidates([candidate('read', start=1)], source=source,
        texts={'b1': 'bread'}, variant='uk', scope_id='run')
    assert selected.accepted == ()
    assert [item.reason for item in selected.skipped] == ['partial_word']
    assert source.calls == []


@pytest.mark.parametrize('text,start,end', [
    ('reading', 0, 4), ('bread', 1, 5), ('école', 0, 1), ('école', 1, 5),
    ('e\u0301cole', 0, 1), ('e\u0301cole', 1, 6), ('e\u0301cole', 2, 6),
    ('read\u0301', 0, 4), ('\u0301read', 0, 5), ('\u0301', 0, 1),
])
async def test_pronunciation_rejects_prefix_suffix_and_combining_mark_splits(text, start, end):
    source = SyntheticSource()
    selected = await verify_candidates([candidate(text[start:end], start=start)], source=source,
        texts={'b1': text}, variant='uk', scope_id='run')
    assert not selected.accepted
    assert [item.reason for item in selected.skipped] == ['partial_word']
    assert source.calls == []


@pytest.mark.parametrize('joiner', ["'", '’', 'ʼ', '-', '‐', '‑'])
@pytest.mark.parametrize('left,right', [('read', 'only'), ('réad', 'ónly'), ('re\u0301ad', 'o\u0301nly')])
async def test_pronunciation_rejects_apostrophe_and_hyphen_word_parts(joiner, left, right):
    source = SyntheticSource()
    text = left + joiner + right
    selected = await verify_candidates([candidate(left), candidate(right, start=len(left) + 1)],
        source=source, texts={'b1': text}, variant='uk', scope_id='run')
    assert [item.reason for item in selected.skipped] == ['partial_word', 'partial_word']
    assert source.calls == []


@pytest.mark.parametrize('text,form,start', [
    ('read', 'read', 0), ("'read'", 'read', 1), ('“read”', 'read', 1),
    ('read, only', 'read', 0), ('only. read', 'read', 6),
    ('read–only', 'only', 5), ('read—only', 'read', 0),
    ("can't", "can't", 0), ('canʼt', 'canʼt', 0), ('read-only', 'read-only', 0),
    ('café', 'café', 0), ('cafe\u0301', 'cafe\u0301', 0), ('cafe\u0301-re\u0301ad', 'cafe\u0301-re\u0301ad', 0),
])
async def test_pronunciation_accepts_complete_unicode_form_and_quoted_word(text, form, start):
    source = SyntheticSource([entry(form=form)])
    selected = await verify_candidates([candidate(form, start=start)], source=source,
        texts={'b1': text}, variant='uk', scope_id='run')
    assert len(selected.accepted) == 1
    assert source.calls == [(form, 'uk')]


@pytest.mark.parametrize('form', ['read‐only', 'read‑only'])
async def test_pronunciation_complete_form_does_not_expand_lookup_support(form):
    from linguary.application.validators.pronunciation import is_complete_pronunciation_form
    assert is_complete_pronunciation_form(form, 0, len(form))
    source = SyntheticSource()
    selected = await verify_candidates([candidate(form)], source=source,
        texts={'b1': form}, variant='uk', scope_id='run')
    assert [item.reason for item in selected.skipped] == ['invalid_form']
    assert source.calls == []


SYNTHETIC_BOOK_TEXT = 'He saw zorp bread.'
SYNTHETIC_TRANSLATION = 'Он увидел выдумку и хлеб.'
SYNTHETIC_MEANING = 'Слово обозначает вымышленный объект.'


def raw_mixed_reply(*, blocks=1, bad_translation=False, profile='legacy-v1'):
    data = json.loads(processing_reply([candidate(start=7)], blocks=blocks,
        profile=profile, treatment='TRANSLATE', translated_text=SYNTHETIC_TRANSLATION,
        glossary_hints=[dict(original='zorp', translation='выдумка', note='Условное название вымышленного объекта.')]).content)
    valid = candidate_json(candidate(start=7))
    forbidden = candidate_json(candidate('bread', start=12))
    forbidden.update(ipa='REJECTED_PRIVATE_IPA', source='REJECTED_PRIVATE_SOURCE', url='https://rejected.invalid')
    data['pronunciation_candidates'] = [valid, None, 'malformed', {}, forbidden,
        valid | {'anchor': valid['anchor'] | {'start': True}},
        valid | {'context': valid['context'] | {'source': 'REJECTED_PRIVATE_CONTEXT'}},
        candidate_json(candidate('read', start=13))]
    data['language_notes'] = [dict(source_block_id='b1', original_excerpt='zorp',
        translated_excerpt='выдумка', note=SYNTHETIC_MEANING)]
    if bad_translation:
        data['blocks'][-1]['translated_text'] = '«' + SYNTHETIC_TRANSLATION
    if profile != 'legacy-v1':
        data['content_evidence'] = [dict(difficulty_id='d', point_id='p', artifact_kind='language_note.note',
            source_block_id='b1', artifact_index=0, body_excerpt=SYNTHETIC_MEANING)]
    return FakeReply.of_payload(data)


def meaning_analysis_reply():
    data = json.loads(analysis_reply(profile='content-plan-v3', treatment='TRANSLATE').content)
    data['content_plan']['difficulties'] = [dict(id='d', owner_block_id='b1', kind='vocabulary',
        priority='critical', help_kind='explanation',
        anchors=[dict(block_id='b1', start=7, end=11, quote='zorp')],
        required_points=[dict(id='p', description='Explain the invented object')])]
    return FakeReply.of_payload(data)


async def test_pronunciation_raw_malformed_candidates_preserve_translation_and_meaning(tmp_path):
    source = SyntheticSource([entry(), entry(form='read'), entry(form='bread')])
    class MeaningFake(FakeLLMClient):
        async def generate(self, request):
            if request.pronunciation_candidates_version == 1:
                marker = 'Обязательные аспекты и точные связанные привязки:\n'
                plan, _ = json.JSONDecoder().raw_decode(request.input_text.split(marker, 1)[1])
                difficulty, = plan['difficulties']
                point, = difficulty['required_points']
                data = json.loads(raw_mixed_reply(profile='content-plan-v3').content)
                data['content_evidence'][0].update(difficulty_id=difficulty['id'], point_id=point['id'])
                self.replies = [*self.replies, FakeReply.of_payload(data)]
            return await super().generate(request)
    fake = MeaningFake([meaning_analysis_reply()])
    app, fake, document, run, _ = await prepare_app(tmp_path, source, text=SYNTHETIC_BOOK_TEXT,
        profile='content-plan-v3', treatment='TRANSLATE', llm_client=fake, max_attempts=3)
    try:
        with app.unit_of_work.transaction() as tx:
            saved, = tx.results.list_for_run(run.id)
        assert saved.translated_text == SYNTHETIC_TRANSLATION
        assert saved.glossary_hints[0].translation == 'выдумка'
        assert saved.language_notes[0].note == SYNTHETIC_MEANING
        assert saved.content_evidence.evidence[0].body_excerpt == SYNTHETIC_MEANING
        assert [item.form for item in saved.pronunciation_assistances] == ['zorp']
        assert saved.pronunciation_assistances[0].anchor.block_id == document.sections[0].blocks[0].id
        assert source.calls == [('zorp', 'uk')]
        assert len(fake.scripted_requests) == 2
    finally:
        await app.aclose()


async def test_pronunciation_mixed_candidates_process_regenerate_and_recovery(tmp_path):
    from linguary.domain import LLMCallStage
    source = SyntheticSource([entry(), entry(form='read'), entry(form='bread')])
    fake = FakeLLMClient([analysis_reply(2, treatment='TRANSLATE'),
        raw_mixed_reply(blocks=2, bad_translation=True), raw_mixed_reply(blocks=2, bad_translation=True),
        raw_mixed_reply(), raw_mixed_reply(), raw_mixed_reply()])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake, pronunciation_source=source)
    try:
        document = store_imported(app, ((SYNTHETIC_BOOK_TEXT, SYNTHETIC_BOOK_TEXT),))
        ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=0,
            settings_revision=0, confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        tasks = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in tasks), [task.last_error for task in tasks]
        block_id = document.sections[0].blocks[0].id
        await regenerate(app, block_id)
        with app.unit_of_work.transaction() as tx:
            saved = tx.results.list_for_run(run.id)
        assert len(saved) == 2
        assert all(item.translated_text == SYNTHETIC_TRANSLATION and item.glossary_hints[0].translation == 'выдумка'
                   and item.language_notes[0].note == SYNTHETIC_MEANING for item in saved)
        assert all([help.form for help in item.pronunciation_assistances] == ['zorp'] for item in saved)
        assert source.calls == [('zorp', 'uk')] * 3
        assert len(fake.scripted_requests) == 6
        children = [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT
                    and len(request.ordered_block_ids) == 1]
        assert len(children) == 2
        assert len(fake.scripted_requests[-1].ordered_block_ids) == 1
        assert fake.scripted_requests[-1].pronunciation_candidates_version == 1
        assert all(request.pronunciation_candidates_version == 1 for request in children)
        assert all('REJECTED_PRIVATE' not in request.instructions + request.input_text for request in fake.requests)
    finally:
        await app.aclose()


@pytest.mark.parametrize('items', [
    [None], ['invalid'], [candidate_json(candidate(start=7)) | {'ipa': 'REJECTED_PRIVATE_IPA'}],
    [candidate_json(candidate(start=7)) | {'anchor': dict(block_id='b1', start=True, end=11, quote='zorp')}],
    [candidate_json(candidate('read', start=13))],
])
async def test_pronunciation_candidate_rejections_do_not_retry_translation(tmp_path, items):
    data = json.loads(raw_mixed_reply().content)
    data['pronunciation_candidates'] = items
    source = SyntheticSource([entry(form='read')])
    app, fake, _, run, _ = await prepare_app(tmp_path, source, text=SYNTHETIC_BOOK_TEXT,
        treatment='TRANSLATE', processing_response=FakeReply.of_payload(data), max_attempts=3)
    try:
        with app.unit_of_work.transaction() as tx:
            saved, = tx.results.list_for_run(run.id)
        assert saved.translated_text == SYNTHETIC_TRANSLATION
        assert saved.pronunciation_assistances == ()
        assert len(fake.scripted_requests) == 2
        assert source.calls == []
    finally:
        await app.aclose()


async def test_pronunciation_rejected_content_never_reaches_model(tmp_path):
    private = replace(entry(), ipa='ˈzɔːpˌɪvɪdəns', evidence=replace(entry().evidence,
        dictionary_ipa='ˈzɔːpˌɪvɪdəns', attribution='SOURCE_PRIVATE_ATTRIBUTION', entry_id='SOURCE_PRIVATE_ENTRY'))
    source = SyntheticSource([private])
    app, fake, document, _, _ = await prepare_app(tmp_path, source, text=SYNTHETIC_BOOK_TEXT,
        treatment='TRANSLATE', processing_response=raw_mixed_reply(),
        replies=[raw_mixed_reply(bad_translation=True), raw_mixed_reply(), raw_mixed_reply()], max_attempts=3)
    try:
        source.failures = {'zorp'}
        await regenerate(app, document.sections[0].blocks[0].id)
        for request in fake.requests:
            outgoing = request.instructions + request.input_text
            assert all(marker not in outgoing for marker in ('SOURCE_PRIVATE', 'REJECTED_PRIVATE', private.ipa))
        assert len(fake.scripted_requests) >= 4
    finally:
        await app.aclose()


async def test_pronunciation_invalid_translation_with_valid_candidate_saves_nothing(tmp_path):
    data = json.loads(raw_mixed_reply().content)
    data['pronunciation_candidates'] = [candidate_json(candidate(start=7))]
    data['blocks'][0]['translated_text'] = None
    invalid = FakeReply.of_payload(data)
    source = SyntheticSource([entry()])
    fake = FakeLLMClient([analysis_reply(treatment='TRANSLATE')], default=invalid)
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct', max_attempts=1),
        storage=StorageSettings(data_root=str(tmp_path / 'data'))), llm_client=fake, pronunciation_source=source)
    try:
        store_imported(app, ((SYNTHETIC_BOOK_TEXT,),))
        ChooseBookConditions(app.book_conditions, app.book_runtime)('book-1', action='reanalyze', expected_revision=0,
            settings_revision=0, confirmed=True, pronunciation_help_enabled=True, pronunciation_variant='uk')
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        tasks = await drain_tasks(app)
        assert any(task.status == TaskStatus.FAILED for task in tasks)
        with app.unit_of_work.transaction() as tx:
            assert tx.results.list_for_run(run.id) == ()
        assert source.calls == []
    finally:
        await app.aclose()
