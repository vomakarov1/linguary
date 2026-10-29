"""Synthetic pronunciations pass through real JSON, pinned runtime and SQLite."""
from dataclasses import replace

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.llm import FakeLLMClient, FakeReply
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


async def prepare_app(tmp_path, source, candidates=(), *, text='zorp', english_help=False, profile='legacy-v1', replies=(), treatment='READ_AS_IS', translated_text=None, glossary_hints=()):
    fake = FakeLLMClient([analysis_reply(profile=profile, treatment=treatment), processing_reply(candidates,
        english_help=english_help, profile=profile, treatment=treatment, translated_text=translated_text, glossary_hints=glossary_hints), *replies])
    app = build_application(settings=AppSettings(llm=LLMSettings(model='synthetic', api_key='synthetic', proxy_mode='direct', max_attempts=1),
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
        assert fake.requests[-1].pronunciation_candidates_version == 1
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
        assert len(fake.requests) == 5
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
        assert len(fake.requests) == 2
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
        assert fake.requests[-1].pronunciation_candidates_version == 1
        assert conditions.payload['prompt_templates']['pronunciation_processing'] in fake.requests[-1].instructions
        assert 'Уровень читателя: b2.' in fake.requests[-1].instructions
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
        assert all('Контекст исходной сцены' in request.input_text for request in children)
        assert all('SOURCE_PRIVATE' not in request.instructions + request.input_text
                   and private.ipa not in request.instructions + request.input_text for request in fake.requests)
        assert {result.pronunciation_assistances[0].anchor.block_id for result in results} == {block.id for block in document.sections[0].blocks}
    finally:
        await app.aclose()
