"""Staged analysis uses durable pass results and independent input dependencies."""
from dataclasses import replace

import pytest
from analysis_fakes import (
    BOOK_ID, FakeLanguageDetector, analysis_reply, make_analysis_block,
    make_analysis_document, make_analysis_section, make_book_record,
)

from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.context_builder import ContextBuilder, snapshot_prompt_templates
from linguary.application.ports.llm import LLMResponseFormatError, LLMTransportError
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.use_cases.analyze_chapter import AnalyzeChapter, AnalyzeChapterCommand
from linguary.domain import (
    AnalysisPassKind, BlockSegment, BlockTreatment, BlockType, LLMCallStage,
)

NARRATIVE = LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE
PEDAGOGICAL = LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL
COMMAND = AnalyzeChapterCommand(BOOK_ID, 'section-1')


def narrative(aliases=('b1',), *, scenes=True, facts=()):
    return FakeReply.of_payload(dict(
        translation_instructions='Keep the voice.', chapter_summary='Holmes opens the door.',
        glossary_candidates=[], context_facts=list(facts),
        scenes=[dict(start_block_id=aliases[0], end_block_id=aliases[-1],
                     summary='Holmes arrives.', location='Baker Street', participants=['Holmes'],
                     visual_facts=[], visual_assumptions=[])] if scenes else [],
    ))


def pedagogical(aliases=('b1',), *, summary=None, **changes):
    payload = dict(treatment_plan=[dict(source_block_id=a, treatment='READ_AS_IS') for a in aliases],
                   before_you_read=[], difficulties=[], before_you_read_evidence=[])
    if summary is not None:
        payload['reader_chapter_summary'] = summary
    payload.update(changes)
    return FakeReply.of_payload(payload)


@pytest.fixture
def stored_book(tmp_path):
    engine = create_database_engine(tmp_path / 'book.db')
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    section = make_analysis_section(texts=('Holmes opens the door.',))
    document = make_analysis_document((section,))
    with uow.transaction() as tx:
        tx.books.add(make_book_record())
        tx.documents.save(document)
    yield engine, uow, document
    engine.dispose()


def builder(level='b2', *, help=False, notes=8, **kwargs):
    return ContextBuilder(content_plan_profile='content-plan-v3', reader_level=level,
        language_notes_per_chapter=notes, prompt_templates=snapshot_prompt_templates(
            source_language='en', target_language='ru', reader_level=level,
            profile='content-plan-v3', english_help=help), **kwargs)


def scenario(uow, client, *, context=None, attempts=1, **kwargs):
    class OutsideTransactionClient:
        def __getattr__(self, name):
            return getattr(client, name)

        async def generate(self, request):
            assert uow._current.get() is None, 'LLM request held a database transaction'
            return await client.generate(request)

    return AnalyzeChapter(unit_of_work=uow, llm_client=OutsideTransactionClient(),
        language_detector=FakeLanguageDetector(), retry_policy=RetryPolicy(max_attempts=attempts),
        glossary_confidence_threshold=.8, context_builder=context or builder(),
        staged_analysis=True, **kwargs)


async def test_second_pass_failure_preserves_first_pass_and_new_use_case_skips_it(stored_book):
    engine, uow, document = stored_book
    client = FakeLLMClient([narrative(), pedagogical(aliases=())])
    with pytest.raises(ResultValidationError, match='не покрывает'):
        await scenario(uow, client)(COMMAND)
    with uow.transaction() as tx:
        assert tx.analyses.get_for_section(BOOK_ID, COMMAND.section_id) is None
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE)
        assert saved is not None
        assert saved.payload['schema_version'] == 1
    restarted = SqlAlchemyUnitOfWork.for_engine(engine)
    retry = FakeLLMClient([pedagogical()])
    result = await scenario(restarted, retry)(COMMAND)
    assert [r.stage for r in retry.requests] == [PEDAGOGICAL]
    assert result.analysis.is_complete
    assert len(result.analysis.scenes) == 1
    assert result.analysis.scenes[0].start_block_id == document.sections[0].blocks[0].id
    with restarted.transaction() as tx:
        assert tx.analyses.get_for_section(BOOK_ID, COMMAND.section_id) == result.analysis
        assert tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.PEDAGOGICAL)
    no_calls = FakeLLMClient()
    cached = await scenario(restarted, no_calls)(COMMAND)
    assert cached.attempts == 0
    assert cached.analysis.content_plan == result.analysis.content_plan


@pytest.mark.parametrize('level,help,notes', [('a2', False, 8), ('b1', True, 8), ('b2', False, 0)])
async def test_reader_settings_recompute_only_pedagogy(stored_book, level, help, notes):
    _, uow, _ = stored_book
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]))(COMMAND)
    summary = 'Holmes opens the door.' if level == 'a2' else None
    client = FakeLLMClient([pedagogical(summary=summary)])
    result = await scenario(uow, client, context=builder(level, help=help, notes=notes))(COMMAND)
    assert [r.stage for r in client.requests] == [PEDAGOGICAL]
    assert result.analysis.chapter_summary == (summary or 'Holmes opens the door.')
    assert client.requests[0].contract_version == (2 if level == 'a2' else 1)


def test_narrative_context_and_fingerprint_ignore_reader_only_settings(stored_book):
    _, _, document = stored_book
    section = document.sections[0]
    contexts = [b.for_chapter_narrative_analysis(document, section) for b in
                (builder(), builder('a2'), builder('b1', help=True), builder(notes=0))]
    assert len({(c.request.instructions, c.request.input_text) for c in contexts}) == 1
    hashes = [b.narrative_input_sha256(document, section) for b in
              (builder(), builder('a2'), builder('b1', help=True), builder(notes=0))]
    assert len(set(hashes)) == 1
    assert 'reader_chapter_summary' not in contexts[0].request.instructions


async def test_raw_source_change_invalidates_both_passes(stored_book):
    _, uow, document = stored_book
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]))(COMMAND)
    section = document.sections[0]
    changed = replace(section, blocks=(replace(section.blocks[0], text='Holmes shuts the door.'),))
    with uow.transaction() as tx:
        tx.documents.save(replace(document, sections=(changed,)))
    client = FakeLLMClient([narrative(), pedagogical()])
    await scenario(uow, client)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL]


async def test_long_chapter_merges_scene_across_fragment_alias_restarts(stored_book):
    engine, uow, _ = stored_book
    section = make_analysis_section(texts=tuple(f'Holmes opens door {n}.' for n in range(6)))
    document = make_analysis_document((section,))
    with uow.transaction() as tx:
        tx.documents.save(document)
    facts = [dict(kind='gender', anchors=[dict(block_id='b1', quote='Holmes')],
                  subject_anchor=dict(block_id='b1', quote='Holmes'), value='male', certainty='explicit')]
    client = FakeLLMClient([*[narrative(('b1', 'b2'), facts=facts) for _ in range(3)],
                            LLMTransportError('offline')])
    with pytest.raises(LLMTransportError, match='offline'):
        await scenario(uow, client, max_fragment_blocks=2)(COMMAND)
    retry = FakeLLMClient([pedagogical(('b1', 'b2')) for _ in range(3)])
    result = await scenario(SqlAlchemyUnitOfWork.for_engine(engine), retry, max_fragment_blocks=2)(COMMAND)
    assert [r.stage for r in retry.requests] == [PEDAGOGICAL] * 3
    scene, = result.analysis.scenes
    assert (scene.start_block_id, scene.end_block_id) == (section.blocks[0].id, section.blocks[-1].id)
    assert {f.subject_anchor.block_id for f in result.analysis.content_plan.context_facts} == {
        section.blocks[n].id for n in (0, 2, 4)}
    assert result.analysis.treatment_plan.block_ids == tuple(b.id for b in section.blocks)


@pytest.mark.parametrize('truncated_stage', [NARRATIVE, PEDAGOGICAL])
async def test_truncation_splits_only_the_affected_pass(stored_book, truncated_stage):
    _, uow, _ = stored_book
    section = make_analysis_section(texts=tuple(f'Holmes opens door {n}.' for n in range(4)))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    truncated = LLMResponseFormatError('too long', truncated=True)
    replies = ([truncated, narrative(('b1', 'b2')), narrative(('b1', 'b2')),
                pedagogical(('b1', 'b2', 'b3', 'b4'))] if truncated_stage == NARRATIVE else
               [narrative(('b1', 'b2', 'b3', 'b4')), truncated,
                pedagogical(('b1', 'b2')), pedagogical(('b1', 'b2'))])
    client = FakeLLMClient(replies)
    result = await scenario(uow, client)(COMMAND)
    assert result.attempts == 4
    assert [r.stage for r in client.requests] == (
        [NARRATIVE] * 3 + [PEDAGOGICAL] if truncated_stage == NARRATIVE else
        [NARRATIVE] + [PEDAGOGICAL] * 3)
    assert result.analysis.is_complete
    assert len(result.analysis.scenes) == 1


async def test_narrative_coverage_failure_retries_with_repair(stored_book):
    _, uow, _ = stored_book
    client = FakeLLMClient([narrative(scenes=False), narrative(), pedagogical()])
    result = await scenario(uow, client, attempts=2)(COMMAND)
    assert result.attempts == 3
    assert client.requests[1].repair_instructions
    assert len(result.analysis.scenes) == 1


async def test_default_is_one_pass_even_for_v3(stored_book):
    _, uow, _ = stored_book
    client = FakeLLMClient([analysis_reply(aliases=('b1',), before_you_read=[],
                                         content_plan=dict(version=3, profile='content-plan-v3',
                                         difficulties=[], before_you_read_evidence=[], context_facts=[]))])
    result = await AnalyzeChapter(unit_of_work=uow, llm_client=client,
        language_detector=FakeLanguageDetector(), retry_policy=RetryPolicy(max_attempts=1),
        glossary_confidence_threshold=.8, context_builder=builder())(COMMAND)
    assert result.analysis.is_complete
    assert [r.stage for r in client.requests] == [LLMCallStage.CHAPTER_ANALYSIS]


def difficulty(alias='b1', quote='Holmes', *, kind='explanation'):
    return dict(id='d1', anchors=[dict(block_id=alias, quote=quote)], owner_block_id=alias,
                kind='grammar', priority='critical', help_kind=kind,
                required_points=[dict(id='p1', description='Объяснить условие')])


@pytest.mark.parametrize('invalid', [
    dict(difficulties=[difficulty(quote='missing')]),
    dict(difficulties=[difficulty(kind='lexical_hint')]),
    dict(
        difficulties=[difficulty(kind='lexical_hint')],
        before_you_read=[dict(expression='Holmes', explanation='Главный сыщик истории.')],
    ),
])
async def test_pedagogical_invalid_quotes_and_channels_preserve_narrative(stored_book, invalid):
    _, uow, _ = stored_book
    client = FakeLLMClient([narrative(), pedagogical(**invalid), pedagogical()])
    result = await scenario(uow, client, attempts=2)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL, PEDAGOGICAL]
    assert client.requests[2].repair_instructions
    assert result.analysis.is_complete


async def test_a2_retries_reader_summary_without_repeating_narrative(stored_book):
    _, uow, _ = stored_book
    summary = 'Holmes opens the door.'
    client = FakeLLMClient([
        narrative(),
        pedagogical(summary='Holmes opens the door.\n\nХолмс открывает дверь.'),
        pedagogical(summary=summary),
    ])
    result = await scenario(uow, client, context=builder('a2'), attempts=2)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL, PEDAGOGICAL]
    assert result.analysis.chapter_summary == summary
    assert result.analysis.content_plan.version == 3


async def test_a2_staged_accepts_legacy_two_paragraph_summary_with_frozen_template(stored_book):
    _, uow, _ = stored_book
    legacy_summary = 'Holmes opens the door.\n\nХолмс открывает дверь.'
    templates = dict(snapshot_prompt_templates(
        source_language='en', target_language='ru', reader_level='a2', profile='content-plan-v3',
    ))
    templates['pedagogical_analysis'] += '\nВерни ровно два абзаца через пустую строку.'
    ctx_builder = ContextBuilder(
        content_plan_profile='content-plan-v3',
        reader_level='a2',
        language_notes_per_chapter=8,
        prompt_templates=templates,
    )
    client = FakeLLMClient([narrative(), pedagogical(summary=legacy_summary)])
    result = await scenario(uow, client, context=ctx_builder, attempts=1)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL]
    assert result.analysis.chapter_summary == legacy_summary


async def test_notes_only_sections_require_empty_scenes(stored_book):
    _, uow, _ = stored_book
    section = make_analysis_section(texts=('Holmes opens the door.',))
    section = replace(section, blocks=(replace(section.blocks[0], type=BlockType.FOOTNOTE),))
    assert section.is_notes_only
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    client = FakeLLMClient([narrative(), narrative(scenes=False), pedagogical()])
    result = await scenario(uow, client, attempts=2)(COMMAND)
    assert result.analysis.scenes == ()
    assert [r.stage for r in client.requests] == [NARRATIVE, NARRATIVE, PEDAGOGICAL]
    assert 'служебный раздел' in client.requests[0].instructions


async def test_front_matter_remains_read_as_is_without_llm(stored_book):
    _, uow, _ = stored_book
    section = make_analysis_section(title='Title page', texts=('Title and publisher.',))
    section = replace(section, source_resource_path='titlepage.xhtml')
    document = make_analysis_document((section,))
    assert document.is_front_matter(section)
    with uow.transaction() as tx:
        tx.documents.save(document)
    client = FakeLLMClient()
    result = await scenario(uow, client)(COMMAND)
    assert result.attempts == 0
    assert result.analysis.scenes == ()
    assert result.analysis.content_plan.version == 3
    context = builder().for_chapter_narrative_analysis(document, section)
    assert 'служебный раздел' in context.request.instructions


def install_conditions(uow, *, revision=1, narrative_model='narrative-model', pedagogical_model='teaching-model',
                       temperature=None, api_key='first-secret'):
    from datetime import UTC, datetime
    from uuid import uuid4
    from linguary.domain.book_conditions import BookConditions
    conditions = BookConditions(str(uuid4()), BOOK_ID, revision,
        dict(schema_version=1, models=dict(analysis='fallback', processing='processing', analysis_passes=dict(
            narrative=narrative_model, pedagogical=pedagogical_model)),
            route=dict(provider='fake', base_url='https://public.invalid/v1', route_id='public-route'),
            llm_parameters=dict(api_key=api_key, temperature=temperature)), datetime.now(UTC))
    with uow.transaction() as tx:
        tx.work.add_conditions(conditions)
    return conditions


@pytest.mark.parametrize('change,stages', [
    (dict(pedagogical_model='new-teacher'), [PEDAGOGICAL]),
    (dict(narrative_model='new-narrator'), [NARRATIVE]),
    (dict(temperature=.2), [NARRATIVE, PEDAGOGICAL]),
    (dict(api_key='rotated-secret'), []),
])
async def test_pass_models_and_generation_parameters_invalidate_independently(stored_book, change, stages):
    _, uow, _ = stored_book
    install_conditions(uow)
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]))(COMMAND)
    install_conditions(uow, revision=2, **change)
    # Narrative model changed but its output did not: accepted pedagogy remains reusable.
    client = FakeLLMClient([narrative() if s == NARRATIVE else pedagogical() for s in stages])
    await scenario(uow, client)(COMMAND)
    assert [r.stage for r in client.requests] == stages


async def test_saved_pass_has_full_stable_dto_and_rejects_corrupt_checksum(stored_book):
    from sqlalchemy import update
    from linguary.adapters.db.models import AnalysisPassResultRow
    from linguary.application.use_cases.analyze_chapter import AnalyzeChapterError
    engine, uow, document = stored_book
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]))(COMMAND)
    with uow.transaction() as tx:
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE)
    assert saved.payload['response']['scenes'][0]['start_block_id'] == document.sections[0].blocks[0].id
    broken = dict(saved.payload)
    broken['response'] = {**broken['response'], 'chapter_summary': 'Corrupted.'}
    with engine.begin() as connection:
        connection.execute(update(AnalysisPassResultRow).where(AnalysisPassResultRow.id == saved.id)
                           .values(payload_json=broken))
    with pytest.raises(AnalyzeChapterError, match='сохранённый проход'):
        await scenario(uow, FakeLLMClient())(COMMAND)


async def test_invalid_cached_schema_is_reported_without_using_partial_dto(stored_book):
    from sqlalchemy import update
    from linguary.adapters.db.models import AnalysisPassResultRow
    from linguary.application.use_cases.analyze_chapter import AnalyzeChapterError
    from linguary.domain.content_plan import canonical_sha
    engine, uow, _ = stored_book
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]))(COMMAND)
    with uow.transaction() as tx:
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE)
    broken = dict(schema_version=2, response={})
    with engine.begin() as connection:
        connection.execute(update(AnalysisPassResultRow).where(AnalysisPassResultRow.id == saved.id)
                           .values(payload_json=broken, output_sha256=canonical_sha(broken)))
    with pytest.raises(AnalyzeChapterError, match='сохранённый проход'):
        await scenario(uow, FakeLLMClient())(COMMAND)


async def test_fragment_pedagogical_quotes_and_byr_survive_cache_round_trip(stored_book):
    _, uow, _ = stored_book
    section = make_analysis_section(texts=('Holmes opens a door.', 'Watson opens a door.'))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    def taught(name):
        return pedagogical(difficulties=[difficulty(quote=name)],
            before_you_read=[dict(expression=name, explanation='Это имя действующего лица.')],
            before_you_read_evidence=[dict(difficulty_id='d1', point_id='p1',
                artifact_kind='before_you_read.explanation', source_block_id='b1', artifact_index=0,
                body_excerpt='Это имя действующего лица.')])
    client = FakeLLMClient([narrative(), narrative(), taught('Holmes'), taught('Watson')])
    first = await scenario(uow, client, max_fragment_blocks=1)(COMMAND)
    second = await scenario(uow, FakeLLMClient(), max_fragment_blocks=1)(COMMAND)
    assert first.analysis.content_plan == second.analysis.content_plan
    assert len(second.analysis.content_plan.difficulties) == 2
    assert len(second.analysis.content_plan.before_you_read_evidence) == 2
    assert len(second.analysis.scenes) == 1


@pytest.mark.parametrize('level,help', [('a2', False), ('b1', True)])
async def test_real_reanalyze_with_own_glossary_suggestion_reuses_narrative(tmp_path, level, help):
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    from linguary.composition import build_application
    from linguary.domain import ConceptType, TermConcept, TermStatus, TermVariant
    from linguary.domain.book_conditions import thaw
    app = build_application(settings=AppSettings(
        llm=LLMSettings(model='pinned', api_key='synthetic-secret', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'application'))),
        llm_client=FakeLLMClient(), content_plan_profile='content-plan-v3')
    try:
        section = make_analysis_section(texts=('Holmes opens the door.',))
        source = app.paths.source_epub(BOOK_ID)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'synthetic source')
        with app.unit_of_work.transaction() as tx:
            tx.books.add(replace(make_book_record(), source_epub_path=str(source)))
            tx.documents.save(make_analysis_document((section,)))
        choose = ChooseBookConditions(app.book_conditions, app.book_runtime)
        conditions = choose(BOOK_ID, action='reanalyze', expected_revision=0,
                            settings_revision=0, confirmed=True, reader_level='b2')
        def pinned_context(c):
            return ContextBuilder(content_plan_profile='content-plan-v3',
                reader_level=c.payload['reader_level'], prompt_templates=thaw(c.payload['prompt_templates']))
        import json
        raw = json.loads(narrative().content)
        raw['glossary_candidates'] = [dict(source_text='Holmes', type='person',
            suggested_translation='Холмс', confidence=.9, model_flagged_review=False,
            note='', name_translation_assessment=None)]
        await scenario(app.unit_of_work, FakeLLMClient([FakeReply.of_payload(raw), pedagogical()]),
                       context=pinned_context(conditions))(COMMAND)
        with app.unit_of_work.transaction() as tx:
            tx.glossary.save(TermConcept('own-suggestion', BOOK_ID, ConceptType.PERSON,
                'Holmes', TermStatus.SUGGESTED,
                variants=(TermVariant('own-suggestion', 'ru', 'Холмс'),)))
        fresh = choose(BOOK_ID, action='reanalyze', expected_revision=1,
                       settings_revision=0, confirmed=True, reader_level=level, english_help=help)
        summary = 'Holmes opens the door.' if level == 'a2' else None
        client = FakeLLMClient([pedagogical(summary=summary)])
        result = await scenario(app.unit_of_work, client, context=pinned_context(fresh))(COMMAND)
        assert [r.stage for r in client.requests] == [PEDAGOGICAL]
        assert result.analysis.is_complete
        assert result.analysis.glossary_candidates[0].source_text == 'Holmes'
    finally:
        await app.aclose()


async def test_pedagogical_note_budget_retries_locally(stored_book):
    _, uow, _ = stored_book
    client = FakeLLMClient([narrative(), pedagogical(difficulties=[difficulty()]), pedagogical()])
    await scenario(uow, client, context=builder(notes=0), attempts=2)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL, PEDAGOGICAL]
    assert 'обязательные заметки не помещаются' in client.requests[-1].repair_instructions[0]


@pytest.mark.parametrize('pass_kind', [NARRATIVE, PEDAGOGICAL])
async def test_service_markers_are_rejected_before_pass_is_saved(stored_book, pass_kind):
    import json
    _, uow, _ = stored_book
    if pass_kind == NARRATIVE:
        bad = json.loads(narrative().content)
        bad['chapter_summary'] = 'Holmes opens [b1].'
        replies = [FakeReply.of_payload(bad), narrative(), pedagogical()]
    else:
        replies = [narrative(), pedagogical(summary='Holmes opens [b1].'),
                   pedagogical(summary='Holmes opens the door.')]
    client = FakeLLMClient(replies)
    await scenario(uow, client, context=builder('a2') if pass_kind == PEDAGOGICAL else builder(),
                   attempts=2)(COMMAND)
    assert client.requests[1 if pass_kind == NARRATIVE else 2].repair_instructions


async def test_narrative_attempt_is_accepted_even_if_second_pass_fails(stored_book):
    from sqlalchemy import select
    from linguary.adapters.db.models import LLMAttemptRow
    from linguary.adapters.llm.logging import LoggingLLMClient
    from linguary.application.book_conditions import BookConditionsService
    from linguary.composition.book_runtime import ProvenanceClient
    engine, uow, _ = stored_book
    conditions = install_conditions(uow)
    # Use a real conditions service and logging/provenance decorators, with deterministic source hashing.
    from linguary.domain.book_conditions import BookConditions
    from dataclasses import replace
    from uuid import uuid4
    conditions = replace(conditions, id=str(uuid4()), revision=2,
                         payload={**dict(conditions.payload), 'source_sha256': 'source'})
    with uow.transaction() as tx:
        tx.work.add_conditions(conditions)
    service = BookConditionsService(uow, None, lambda: 0, lambda _: 'source')
    client = ProvenanceClient(LoggingLLMClient(FakeLLMClient([narrative(), pedagogical(aliases=())]),
        unit_of_work=uow, provider='fake', model='pinned'), service, conditions.id)
    with pytest.raises(ResultValidationError, match='не покрывает'):
        await scenario(uow, client)(COMMAND)
    with engine.connect() as connection:
        states = connection.execute(select(LLMAttemptRow.state, LLMAttemptRow.context_json)).all()
    assert {row.context_json['stage']: row.state for row in states} == {
        NARRATIVE.value: 'accepted', PEDAGOGICAL.value: 'rejected'}


async def test_late_narrative_cannot_persist_after_conditions_change(stored_book):
    from linguary.application.book_conditions import ConditionsError
    _, uow, _ = stored_book
    install_conditions(uow)
    inner = FakeLLMClient([narrative()])
    class NewConditionsDuringRequest:
        async def generate(self, request):
            response = await inner.generate(request)
            install_conditions(uow, revision=2)
            return response
    with pytest.raises(ConditionsError, match='conditions_revision_conflict'):
        await scenario(uow, NewConditionsDuringRequest())(COMMAND)
    with uow.transaction() as tx:
        assert tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE) is None
        assert tx.analyses.get_for_section(BOOK_ID, COMMAND.section_id) is None


async def test_glossary_decision_change_invalidates_both_passes(stored_book):
    from linguary.domain import ConceptType, TermConcept, TermStatus, TermVariant
    from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy
    _, uow, _ = stored_book
    policy = NameTranslationPolicy(1, NameTranslationStrategy.SOUND)
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]),
                   context=builder(name_translation_policy=policy))(COMMAND)
    with uow.transaction() as tx:
        tx.glossary.save(TermConcept('approved', BOOK_ID, ConceptType.PERSON, 'Holmes', TermStatus.APPROVED,
            variants=(TermVariant('approved', 'ru', 'Холмс', TermStatus.APPROVED),)))
    client = FakeLLMClient([narrative(), pedagogical()])
    await scenario(uow, client, context=builder(name_translation_policy=policy))(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL]
    assert 'Холмс' in client.requests[0].input_text


async def test_narrative_fact_conflicts_with_glossary_and_retries(stored_book):
    from linguary.domain import ConceptType, TermConcept, TermStatus, TermVariant
    _, uow, _ = stored_book
    with uow.transaction() as tx:
        tx.glossary.save(TermConcept('approved', BOOK_ID, ConceptType.PERSON, 'Holmes', TermStatus.APPROVED,
            note='Женский род.', variants=(TermVariant('approved', 'ru', 'Холмс', TermStatus.APPROVED),)))
    fact = dict(kind='gender', anchors=[dict(block_id='b1', quote='Holmes')],
                subject_anchor=dict(block_id='b1', quote='Holmes'), value='male', certainty='explicit')
    client = FakeLLMClient([narrative(facts=[fact]), narrative(), pedagogical()])
    await scenario(uow, client, attempts=2)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, NARRATIVE, PEDAGOGICAL]
    assert client.requests[1].repair_instructions


async def test_cached_visual_facts_keep_verbatim_anchors(stored_book):
    import json
    _, uow, document = stored_book
    raw = json.loads(narrative().content)
    raw['scenes'][0]['visual_facts'] = [dict(anchor=dict(block_id='b1', quote='door'),
        subject='room', detail='A door is present.')]
    raw['scenes'][0]['visual_assumptions'] = ['The door might be wooden.']
    first = await scenario(uow, FakeLLMClient([FakeReply.of_payload(raw), pedagogical()]))(COMMAND)
    second = await scenario(uow, FakeLLMClient())(COMMAND)
    fact, = second.analysis.scenes[0].visual_facts
    block = document.sections[0].blocks[0]
    assert fact.anchor.block_id == block.id
    assert block.text[fact.anchor.start:fact.anchor.end] == 'door'
    assert second.analysis.scenes == first.analysis.scenes
    assert second.analysis.scenes[0].visual_assumptions == ('The door might be wooden.',)


@pytest.mark.parametrize('method', ['for_chapter_narrative_analysis', 'for_chapter_pedagogical_analysis'])
@pytest.mark.parametrize('count,order', [(0, 0), (1, 1)])
def test_staged_context_rejects_invalid_fragment_coordinates(stored_book, method, count, order):
    from linguary.application.analysis_merge import ChapterFragment
    from linguary.application.context_builder import ContextBuildError
    _, _, document = stored_book
    section = document.sections[0]
    kwargs = dict(fragment=ChapterFragment(order, section.blocks), fragment_count=count)
    if method == 'for_chapter_pedagogical_analysis':
        kwargs['narrative_summary'] = 'Holmes opens the door.'
    with pytest.raises(ContextBuildError, match='фрагмент'):
        getattr(builder(), method)(document, section, **kwargs)


async def test_existing_upstream_suggestion_is_not_mistaken_for_own_output(stored_book):
    import json
    from linguary.domain import ConceptType, TermConcept, TermStatus, TermVariant
    from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy
    _, uow, _ = stored_book
    with uow.transaction() as tx:
        tx.glossary.save(TermConcept('upstream', BOOK_ID, ConceptType.PERSON, 'Holmes', TermStatus.SUGGESTED,
            variants=(TermVariant('upstream', 'ru', 'Холмс'),)))
    raw = json.loads(narrative().content)
    raw['glossary_candidates'] = [dict(source_text='Holmes', type='person', suggested_translation='Холмс',
        confidence=.9, model_flagged_review=False, note='', name_translation_assessment=None)]
    context = builder(name_translation_policy=NameTranslationPolicy(1, NameTranslationStrategy.SOUND))
    await scenario(uow, FakeLLMClient([FakeReply.of_payload(raw), pedagogical()]), context=context)(COMMAND)
    cached = await scenario(uow, FakeLLMClient(), context=context)(COMMAND)
    assert cached.attempts == 0


async def test_overwritten_pass_is_archived_with_original_conditions(stored_book):
    from sqlalchemy import select
    from linguary.adapters.db.models import ArtifactRevisionRow
    engine, uow, _ = stored_book
    original = install_conditions(uow)
    await scenario(uow, FakeLLMClient([narrative(), pedagogical()]))(COMMAND)
    with uow.transaction() as tx:
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE)
    install_conditions(uow, revision=2, narrative_model='new-narrator')
    await scenario(uow, FakeLLMClient([narrative()]))(COMMAND)
    with engine.connect() as connection:
        archived, = connection.execute(select(ArtifactRevisionRow.conditions_id, ArtifactRevisionRow.body_json)
            .where(ArtifactRevisionRow.kind == 'analysis_pass')).all()
    assert archived.conditions_id == original.id
    assert archived.body_json['input_sha256'] == saved.input_sha256
    assert archived.body_json['output_sha256'] == saved.output_sha256
    assert archived.body_json['payload'] == saved.payload


async def test_transient_pedagogical_failure_uses_backoff_without_repeating_narrative(stored_book):
    _, uow, _ = stored_book
    delays = []
    async def sleep(seconds):
        delays.append(seconds)
    client = FakeLLMClient([narrative(), LLMTransportError('temporary'), pedagogical()])
    result = await scenario(uow, client, attempts=2, sleep=sleep)(COMMAND)
    assert delays == [1.0]
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL, PEDAGOGICAL]
    assert result.attempts == 3
    assert result.analysis.is_complete


async def test_late_pedagogical_split_preserves_byr_parent_quota_with_evidence(stored_book):
    import re
    _, uow, _ = stored_book
    section = make_analysis_section(texts=tuple(
        ' '.join(f'phrase{index}{n}' for n in range(6)) for index in range(6)))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    blocks = {block.id: block for block in section.blocks}
    class BudgetFollowingClient:
        def __init__(self):
            self.requests, self.byr_quotas = [], []
            self.did_split = False
        async def generate(self, request):
            self.requests.append(request)
            aliases = tuple(f'b{n}' for n in range(1, len(request.ordered_block_ids) + 1))
            if request.stage == NARRATIVE:
                reply = narrative(aliases)
            else:
                budget = int(re.search(r'не больше (\d+) BYR', request.input_text).group(1))
                self.byr_quotas.append(budget)
                if not self.did_split and request.ordered_block_ids[0] == section.blocks[3].id:
                    self.did_split = True
                    raise LLMResponseFormatError('late cutoff', truncated=True)
                first = blocks[request.ordered_block_ids[0]]
                expressions = first.text.split()[:budget]
                entries, difficulties, evidence = [], [], []
                for n, expression in enumerate(expressions):
                    explanation = f'Это выражение передаёт отдельный смысл: вариант {n}.'
                    entries.append(dict(expression=expression, explanation=explanation))
                    difficulties.append({**difficulty(quote=expression), 'id': f'd{n}'})
                    evidence.append(dict(difficulty_id=f'd{n}', point_id='p1',
                        artifact_kind='before_you_read.explanation', source_block_id='b1',
                        artifact_index=n, body_excerpt=explanation))
                reply = pedagogical(aliases, before_you_read=entries, difficulties=difficulties,
                                    before_you_read_evidence=evidence)
            return await FakeLLMClient([reply]).generate(request)
    client = BudgetFollowingClient()
    result = await scenario(uow, client, max_fragment_blocks=3, attempts=2)(COMMAND)
    assert client.byr_quotas == [5, 5, 3, 2]
    assert len(result.analysis.before_you_read) == 10
    assert len(result.analysis.content_plan.before_you_read_evidence) == 10
    assert len(result.analysis.content_plan.difficulties) == 10
    assert result.analysis.is_complete


async def test_late_pedagogical_split_preserves_parent_note_budget(stored_book):
    import re
    _, uow, _ = stored_book
    # Nine blocks stay under the 24_000-character cap, so the chapter still splits as [9, 9].
    # A block that receives a note stays above half of an 8_000-character processing group,
    # so two such notes are not asked to share one group.
    lead = 'Holmes opens the door. '
    noted = {0, 1, 2, 3, 9, 10, 13, 14}
    section = make_analysis_section(texts=tuple(
        lead + 'x' * (4500 - len(lead) if index in noted else 0)
        for index in range(18)
    ))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    class NoteBudgetFollowingClient:
        def __init__(self):
            self.requests, self.note_quotas = [], []
            self.did_split = False
        async def generate(self, request):
            self.requests.append(request)
            aliases = tuple(f'b{n}' for n in range(1, len(request.ordered_block_ids) + 1))
            if request.stage == NARRATIVE:
                reply = narrative(aliases)
            else:
                budget = int(re.search(r'Бюджет обязательных заметок этого анализа: (\d+)', request.input_text).group(1))
                self.note_quotas.append(budget)
                if not self.did_split and request.ordered_block_ids[0] == section.blocks[9].id:
                    self.did_split = True
                    raise LLMResponseFormatError('late cutoff', truncated=True)
                reply = pedagogical(aliases, difficulties=[
                    {**difficulty(alias=alias), 'id': f'd{index}'}
                    for index, alias in enumerate(aliases[:budget])])
            return await FakeLLMClient([reply]).generate(request)
    client = NoteBudgetFollowingClient()
    result = await scenario(uow, client, max_fragment_blocks=9, attempts=2)(COMMAND)
    assert client.note_quotas == [4, 4, 2, 2]
    assert len({d.owner_block_id for d in result.analysis.content_plan.difficulties}) == 8
    assert result.analysis.is_complete


@pytest.mark.parametrize('conflict_stage', [NARRATIVE, PEDAGOGICAL])
@pytest.mark.parametrize('conflict', ['conditions', 'source', 'task'])
async def test_pass_save_conflicts_stale_only_unaccepted_attempts(stored_book, conflict_stage, conflict):
    from dataclasses import replace
    from sqlalchemy import select
    from uuid import uuid4
    from linguary.adapters.db.models import LLMAttemptRow, LLMCallLogRow
    from linguary.adapters.llm.logging import LoggingLLMClient
    from linguary.application.book_conditions import BookConditionsService, ConditionsError
    from linguary.composition.book_runtime import ProvenanceClient
    from linguary.domain import ProcessingTask, TaskType
    engine, uow, _ = stored_book
    conditions = install_conditions(uow)
    conditions = replace(conditions, id=str(uuid4()), revision=2,
                         payload={**dict(conditions.payload), 'source_sha256': 'source'})
    task = ProcessingTask(id='analysis-task', task_type=TaskType.ANALYZE_CHAPTER,
                                  book_id=BOOK_ID, scope_id=COMMAND.section_id)
    with uow.transaction() as tx:
        tx.work.add_conditions(conditions)
        tx.tasks.add(task)
    digest = ['source']
    service = BookConditionsService(uow, None, lambda: 0, lambda _: digest[0])
    inner = FakeLLMClient([narrative(), pedagogical()])
    class ConflictingClient:
        async def generate(self, request):
            response = await inner.generate(request)
            if request.stage == conflict_stage:
                if conflict == 'conditions':
                    changed = replace(conditions, id=str(uuid4()), revision=3)
                    with uow.transaction() as tx:
                        tx.work.add_conditions(changed)
                elif conflict == 'source':
                    digest[0] = 'changed-source'
                else:
                    with uow.transaction() as tx:
                        tx.tasks.delete(task.id)
            return response
    client = ProvenanceClient(LoggingLLMClient(ConflictingClient(), unit_of_work=uow,
        provider='fake', model='pinned'), service, conditions.id)
    expected = dict(conditions='conditions_revision_conflict', source='source_digest_mismatch',
                    task='task_no_longer_active')[conflict]
    with pytest.raises(ConditionsError, match=expected):
        await scenario(uow, client)(replace(COMMAND, task_id=task.id))
    with engine.connect() as connection:
        rows = connection.execute(select(LLMAttemptRow.state, LLMAttemptRow.context_json,
                                         LLMAttemptRow.outcome_json)).all()
        logs = connection.execute(select(LLMCallLogRow.stage, LLMCallLogRow.diagnostic_meta_json)).all()
    states = {row.context_json['stage']: (row.state, row.outcome_json.get('decision')) for row in rows}
    assert states[NARRATIVE.value] == (('validated', 'stale') if conflict_stage == NARRATIVE else ('accepted', 'accepted'))
    if conflict_stage == PEDAGOGICAL:
        assert states[PEDAGOGICAL.value] == ('validated', 'stale')
    decisions = {row.stage: row.diagnostic_meta_json['decision'] for row in logs}
    assert decisions[conflict_stage.value] == 'stale'
    if conflict_stage == PEDAGOGICAL:
        assert decisions[NARRATIVE.value] == 'accepted'
    with uow.transaction() as tx:
        assert tx.analyses.get_for_section(BOOK_ID, COMMAND.section_id) is None
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE)
        assert (saved is not None) == (conflict_stage == PEDAGOGICAL)
        assert tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.PEDAGOGICAL) is None


@pytest.mark.parametrize('suggestion_origin', ['suggested', 'skip', 'skip_without_note', 'initial', 'edited_note'])
async def test_reader_reanalyze_recognizes_independently_merged_own_glossary_suggestion(tmp_path, suggestion_origin):
    import json
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.application.use_cases.approve_glossary import (
        ApproveGlossary, GlossaryDecision, GlossaryDecisionCommand,
    )
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    from linguary.composition import build_application
    from linguary.domain import BookStatus, ConceptType, TermConcept, TermStatus, TermVariant
    from linguary.domain.book_conditions import thaw

    class FragmentClient:
        def __init__(self):
            self.requests = []
        async def generate(self, request):
            count = sum(r.stage == NARRATIVE for r in self.requests)
            self.requests.append(request)
            if request.stage == NARRATIVE:
                raw = json.loads(narrative().content)
                raw['glossary_candidates'] = [dict(source_text='Holmes', type='person',
                    suggested_translation='Холмс' if count % 2 == 0 else 'Хоумс',
                    confidence=.95 if count % 2 == 0 else .85,
                    model_flagged_review=False, note='' if count % 2 == 0 else 'Частный сыщик.',
                    name_translation_assessment=None)]
                reply = FakeReply.of_payload(raw)
            else:
                reply = pedagogical()
            return await FakeLLMClient([reply]).generate(request)

    app = build_application(settings=AppSettings(
        llm=LLMSettings(model='pinned', api_key='synthetic-secret', proxy_mode='direct'),
        storage=StorageSettings(data_root=str(tmp_path / 'application'))),
        llm_client=FakeLLMClient(), content_plan_profile='content-plan-v3')
    try:
        section = make_analysis_section(texts=('Holmes opens the door.', 'Holmes enters the room.'))
        source = app.paths.source_epub(BOOK_ID)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'synthetic source')
        def suggestion(note):
            return TermConcept('suggested-holmes', BOOK_ID, ConceptType.PERSON, 'Holmes',
                TermStatus.SUGGESTED, note=note,
                variants=(TermVariant('suggested-holmes', 'ru', 'Холмс'),))
        with app.unit_of_work.transaction() as tx:
            tx.books.add(replace(make_book_record(), source_epub_path=str(source)))
            tx.documents.save(make_analysis_document((section,)))
            if suggestion_origin == 'initial':
                tx.glossary.save(suggestion('Частный сыщик.'))
        choose = ChooseBookConditions(app.book_conditions, app.book_runtime)
        conditions = choose(BOOK_ID, action='reanalyze', expected_revision=0, settings_revision=0,
                            confirmed=True, reader_level='b2')
        def pinned_context(c):
            return ContextBuilder(content_plan_profile='content-plan-v3',
                reader_level=c.payload['reader_level'], prompt_templates=thaw(c.payload['prompt_templates']))
        first = await scenario(app.unit_of_work, FragmentClient(), max_fragment_blocks=1,
                               context=pinned_context(conditions))(COMMAND)
        candidate, = first.analysis.glossary_candidates
        assert (candidate.suggested_translation, candidate.note) == ('Холмс', 'Частный сыщик.')
        if suggestion_origin in {'skip', 'skip_without_note'}:
            with app.unit_of_work.transaction() as tx:
                tx.books.save(replace(tx.books.get(BOOK_ID), status=BookStatus.GLOSSARY_REVIEW))
            saved = ApproveGlossary(unit_of_work=app.unit_of_work, glossary_confidence_threshold=.8)(
                GlossaryDecisionCommand(BOOK_ID, GlossaryDecision.SKIP, source_text='Holmes',
                    note=candidate.note if suggestion_origin == 'skip' else None))
            assert saved.status is TermStatus.SUGGESTED
            assert saved.variant_for('ru').text == 'Холмс'
            assert saved.note == (candidate.note if suggestion_origin == 'skip' else '')
        elif suggestion_origin != 'initial':
            with app.unit_of_work.transaction() as tx:
                tx.glossary.save(suggestion('Уточнение пользователя.' if suggestion_origin == 'edited_note'
                                            else candidate.note))
        fresh = choose(BOOK_ID, action='reanalyze', expected_revision=1, settings_revision=0,
                       confirmed=True, reader_level='b1')
        retry = FragmentClient()
        result = await scenario(app.unit_of_work, retry, max_fragment_blocks=1,
                                context=pinned_context(fresh))(COMMAND)
        assert [r.stage for r in retry.requests] == (
            [NARRATIVE, NARRATIVE, PEDAGOGICAL, PEDAGOGICAL] if suggestion_origin == 'edited_note'
            else [PEDAGOGICAL, PEDAGOGICAL])
        assert result.analysis.is_complete
        if suggestion_origin == 'initial':
            with app.unit_of_work.transaction() as tx:
                saved_pass = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.NARRATIVE)
            assert list(saved_pass.payload['input_suggestions'][0]) == ['holmes', 'Холмс', 'Частный сыщик.']
    finally:
        await app.aclose()


def test_pedagogical_context_facts_use_fragment_aliases_and_strip_offsets(stored_book):
    import json
    from linguary.application.analysis_merge import ChapterFragment
    from linguary.domain.content_plan import ContextFact, SourceAnchor

    _, _, _ = stored_book
    section = make_analysis_section(texts=('Holmes enters.', 'She smiles.', 'Watson nods.'))
    document = make_analysis_document((section,))
    b0, b1, b2 = section.blocks
    facts = (
        ContextFact(
            id='a' * 64,
            kind='gender',
            anchors=(SourceAnchor(b0.id, 0, 6, 'Holmes'), SourceAnchor(b1.id, 0, 3, 'She')),
            subject_anchor=SourceAnchor(b1.id, 0, 3, 'She'),
            value='female',
            certainty='explicit',
        ),
        ContextFact(
            id='b' * 64,
            kind='gender',
            anchors=(SourceAnchor(b2.id, 0, 6, 'Watson'),),
            subject_anchor=SourceAnchor(b2.id, 0, 6, 'Watson'),
            value='male',
            certainty='explicit',
        ),
    )
    fragment = ChapterFragment(order=1, blocks=(b1,))
    context = builder().for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary='Holmes enters and she smiles.',
        context_facts=facts,
        fragment=fragment,
        fragment_count=3,
    )
    prompt = context.request.input_text
    assert b0.id not in prompt
    assert b1.id not in prompt
    assert b2.id not in prompt
    assert 'a' * 64 not in prompt
    payload_str = prompt.split('Принятый повествовательный анализ:\n', 1)[1]
    parsed = json.loads(payload_str)
    assert len(parsed['context_facts']) == 1
    fact_json = parsed['context_facts'][0]
    assert fact_json['id'] == 'f1'
    assert fact_json['subject_anchor'] == {'block_id': 'b1', 'quote': 'She'}
    assert sorted(fact_json['anchors'], key=lambda a: a['block_id']) == [
        {'block_id': 'b1', 'quote': 'She'},
        {'block_id': 'related-source-1', 'quote': 'Holmes'},
    ]


async def test_multi_fragment_narrative_findings_keep_all_instructions_and_sanitize_scene(stored_book):
    import json
    _, uow, _ = stored_book
    section = make_analysis_section(texts=('Holmes opens door 1.', 'Note text.', 'Holmes opens door 2.'))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    raw_1 = json.loads(narrative(('b1',)).content)
    raw_1['translation_instructions'] = 'Keep instruction one.'
    raw_1['scenes'][0]['visual_facts'] = [
        dict(anchor=dict(block_id='b1', quote='door'), subject='door', detail='Open door.')
    ]
    raw_2 = json.loads(narrative(('b1',)).content)
    raw_2['translation_instructions'] = 'Keep instruction two.'
    client = FakeLLMClient([
        FakeReply.of_payload(raw_1),
        FakeReply.of_payload(raw_2),
        narrative(('b1',)),
        pedagogical(('b1',)),
        pedagogical(('b1',)),
        pedagogical(('b1',)),
    ])
    await scenario(uow, client, max_fragment_blocks=1)(COMMAND)
    third_narrative_prompt = client.requests[2].input_text
    assert 'Keep instruction one.' in third_narrative_prompt
    assert 'Keep instruction two.' in third_narrative_prompt
    assert 'Последняя сцена: {' in third_narrative_prompt
    for block in section.blocks:
        assert block.id not in third_narrative_prompt
    assert 'scene_id' not in third_narrative_prompt


async def test_multi_fragment_pedagogical_remaps_stem_duplicate_byr_with_different_explanation_and_retries_merge_conflict(stored_book):
    _, uow, _ = stored_book
    section = make_analysis_section(texts=('Holmes opens a door.', 'Holmes shuts a door.'))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))

    frag1_ped = pedagogical(
        ('b1',),
        difficulties=[difficulty(quote='Holmes')],
        before_you_read=[dict(expression='Holmes', explanation='Первое объяснение имени сыщика.')],
        before_you_read_evidence=[dict(
            difficulty_id='d1', point_id='p1', artifact_kind='before_you_read.explanation',
            source_block_id='b1', artifact_index=0, body_excerpt='Первое объяснение имени сыщика.',
        )],
    )
    frag2_ped_stem_diff_explanation = pedagogical(
        ('b1',),
        difficulties=[difficulty(quote='Holmes')],
        before_you_read=[dict(expression='Holmes', explanation='Второе независимое объяснение имени.')],
        before_you_read_evidence=[dict(
            difficulty_id='d1', point_id='p1', artifact_kind='before_you_read.explanation',
            source_block_id='b1', artifact_index=0, body_excerpt='Второе независимое объяснение имени.',
        )],
    )
    client = FakeLLMClient([
        narrative(('b1',)),
        narrative(('b1',)),
        frag1_ped,
        frag2_ped_stem_diff_explanation,
    ])
    result = await scenario(uow, client, max_fragment_blocks=1)(COMMAND)
    assert len(result.analysis.before_you_read) == 1
    assert len(result.analysis.content_plan.before_you_read_evidence) == 2
    assert all(
        ev.body_excerpt == 'Первое объяснение имени сыщика.'
        for ev in result.analysis.content_plan.before_you_read_evidence
    )


async def test_multi_fragment_pass_persists_completed_fragments_and_cleans_up_on_completion(stored_book):
    _, uow, _ = stored_book
    section = make_analysis_section(texts=('Holmes opens door 1.', 'Holmes opens door 2.'))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    client = FakeLLMClient([
        narrative(('b1',)),
        narrative(('b1',)),
        pedagogical(('b1',)),
        LLMTransportError('second pedagogical fragment failed'),
    ])
    with pytest.raises(LLMTransportError, match='second pedagogical fragment failed'):
        await scenario(uow, client, max_fragment_blocks=1)(COMMAND)
    with uow.transaction() as tx:
        passes = tx.analyses.list_passes_for_section(BOOK_ID, COMMAND.section_id)
        assert [(p.pass_kind, bool(p.scope_key)) for p in passes] == [
            (AnalysisPassKind.NARRATIVE, False),
            (AnalysisPassKind.PEDAGOGICAL, True),
        ]
    retry = FakeLLMClient([pedagogical(('b1',))])
    result = await scenario(uow, retry, max_fragment_blocks=1)(COMMAND)
    assert [r.stage for r in retry.requests] == [PEDAGOGICAL]
    assert result.attempts == 1
    with uow.transaction() as tx:
        passes_after = tx.analyses.list_passes_for_section(BOOK_ID, COMMAND.section_id)
        assert [(p.pass_kind, p.scope_key) for p in passes_after] == [
            (AnalysisPassKind.NARRATIVE, ''),
            (AnalysisPassKind.PEDAGOGICAL, ''),
        ]


async def test_multi_chapter_reanalyze_after_skip_reuses_narrative_across_all_chapters(tmp_path):
    import json
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.application.use_cases.approve_glossary import (
        ApproveGlossary, GlossaryDecision, GlossaryDecisionCommand,
    )
    from linguary.application.use_cases.reset_processing import ChooseBookConditions
    from linguary.composition import build_application
    from linguary.domain import BookStatus
    from linguary.domain.book_conditions import thaw

    app = build_application(
        settings=AppSettings(
            llm=LLMSettings(model='pinned', api_key='synthetic-secret', proxy_mode='direct'),
            storage=StorageSettings(data_root=str(tmp_path / 'application')),
        ),
        llm_client=FakeLLMClient(),
        content_plan_profile='content-plan-v3',
    )
    try:
        sec1 = make_analysis_section(section_id='section-1', order=0, texts=('Holmes opens the door.',))
        sec2 = make_analysis_section(section_id='section-2', order=1, texts=('Holmes enters the room.',))
        source = app.paths.source_epub(BOOK_ID)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'synthetic source')
        with app.unit_of_work.transaction() as tx:
            tx.books.add(replace(make_book_record(), source_epub_path=str(source)))
            tx.documents.save(make_analysis_document((sec1, sec2)))
        choose = ChooseBookConditions(app.book_conditions, app.book_runtime)
        conditions = choose(
            BOOK_ID, action='reanalyze', expected_revision=0, settings_revision=0, confirmed=True, reader_level='b2'
        )

        def pinned_context(c):
            return ContextBuilder(
                content_plan_profile='content-plan-v3',
                reader_level=c.payload['reader_level'],
                prompt_templates=thaw(c.payload['prompt_templates']),
            )

        raw1 = json.loads(narrative().content)
        raw1['glossary_candidates'] = [dict(
            source_text='Holmes', type='person', suggested_translation='Холмс',
            confidence=.95, model_flagged_review=False, note='Частный сыщик.',
            name_translation_assessment=None,
        )]
        await scenario(
            app.unit_of_work,
            FakeLLMClient([FakeReply.of_payload(raw1), pedagogical()]),
            context=pinned_context(conditions),
        )(AnalyzeChapterCommand(BOOK_ID, 'section-1'))
        await scenario(
            app.unit_of_work,
            FakeLLMClient([narrative(), pedagogical()]),
            context=pinned_context(conditions),
        )(AnalyzeChapterCommand(BOOK_ID, 'section-2'))

        with app.unit_of_work.transaction() as tx:
            tx.books.save(replace(tx.books.get(BOOK_ID), status=BookStatus.GLOSSARY_REVIEW))
        ApproveGlossary(unit_of_work=app.unit_of_work, glossary_confidence_threshold=.8)(
            GlossaryDecisionCommand(BOOK_ID, GlossaryDecision.SKIP, source_text='Holmes', note=None)
        )

        fresh = choose(
            BOOK_ID, action='reanalyze', expected_revision=1, settings_revision=0, confirmed=True, reader_level='b1'
        )
        client1 = FakeLLMClient([pedagogical()])
        await scenario(app.unit_of_work, client1, context=pinned_context(fresh))(
            AnalyzeChapterCommand(BOOK_ID, 'section-1')
        )
        assert [r.stage for r in client1.requests] == [PEDAGOGICAL]

        client2 = FakeLLMClient([pedagogical()])
        await scenario(app.unit_of_work, client2, context=pinned_context(fresh))(
            AnalyzeChapterCommand(BOOK_ID, 'section-2')
        )
        assert [r.stage for r in client2.requests] == [PEDAGOGICAL]
    finally:
        await app.aclose()


def test_save_pass_updates_loaded_row_in_same_sqlalchemy_session(stored_book):
    from linguary.domain import AnalysisPassResult
    _, uow, _ = stored_book
    first = AnalysisPassResult(
        id='pass-1', book_id=BOOK_ID, section_id='section-1', pass_kind=AnalysisPassKind.NARRATIVE,
        scope_key='', input_sha256='1' * 64, output_sha256='a' * 64, conditions_id=None,
        attempt_id='att-1', payload={'schema_version': 1, 'response': {'v': 1}},
    )
    second = AnalysisPassResult(
        id='pass-2', book_id=BOOK_ID, section_id='section-1', pass_kind=AnalysisPassKind.NARRATIVE,
        scope_key='', input_sha256='2' * 64, output_sha256='b' * 64, conditions_id=None,
        attempt_id='att-2', payload={'schema_version': 1, 'response': {'v': 2}},
    )
    with uow.transaction() as tx:
        tx.analyses.save_pass(first)
        loaded_1 = tx.analyses.get_pass(BOOK_ID, 'section-1', AnalysisPassKind.NARRATIVE)
        assert loaded_1.input_sha256 == '1' * 64
        tx.analyses.save_pass(second)
        loaded_2 = tx.analyses.get_pass(BOOK_ID, 'section-1', AnalysisPassKind.NARRATIVE)
        assert loaded_2.input_sha256 == '2' * 64
        assert loaded_2.attempt_id == 'att-2'


async def test_pedagogical_pass_receives_current_chapter_narrative_glossary_and_name_policy(stored_book):
    import json
    from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy

    _, uow, _ = stored_book
    section = make_analysis_section(
        texts=('Holmes lives on Baker Street.', 'Watson arrives at noon.')
    )
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))

    policy = NameTranslationPolicy(1, NameTranslationStrategy.SOUND)
    templates = snapshot_prompt_templates(
        source_language='en',
        target_language='ru',
        reader_level='a2',
        profile='content-plan-v3',
        name_translation_policy=policy,
    )
    ctx_builder = ContextBuilder(
        content_plan_profile='content-plan-v3',
        reader_level='a2',
        prompt_templates=templates,
        name_translation_policy=policy,
    )

    raw_1 = json.loads(narrative(('b1',)).content)
    raw_1['chapter_summary'] = 'Holmes waits on Baker Street.'
    raw_1['glossary_candidates'] = [
        dict(
            source_text='Baker Street',
            type='place',
            suggested_translation='Бейкер-стрит',
            confidence=0.95,
            model_flagged_review=False,
            note='Улица в Лондоне.',
            name_translation_assessment=None,
        )
    ]
    raw_2 = json.loads(narrative(('b1',)).content)
    raw_2['chapter_summary'] = 'Holmes waits on Baker Street and Watson arrives.'

    summary_1 = 'Holmes waits on Baker Street.'
    summary_2 = 'Holmes waits on Baker Street and Watson arrives.'
    client = FakeLLMClient([
        FakeReply.of_payload(raw_1),
        FakeReply.of_payload(raw_2),
        pedagogical(('b1',), summary=summary_1),
        pedagogical(('b1',), summary=summary_2),
    ])
    await scenario(uow, client, context=ctx_builder, max_fragment_blocks=1)(COMMAND)

    ped_requests = [r for r in client.requests if r.stage == PEDAGOGICAL]
    assert len(ped_requests) == 2
    for req in ped_requests:
        assert 'Передавай имена и названия по звучанию' in req.instructions
        assert 'name_translation_assessment' not in req.instructions
        assert '- Baker Street [place] -> Бейкер-стрит (Улица в Лондоне.)' in req.input_text


def save_segmented_block(uow, parts, sentences):
    """Replace the stored chapter with one block split into these spans."""
    text = "".join(parts)
    offset = 0
    spans = []
    for part, sentence in zip(parts, sentences, strict=True):
        spans.append(BlockSegment(offset, offset + len(part), sentence))
        offset += len(part)
    block = replace(
        make_analysis_block(section_id="section-1", order=0, text=text),
        segments=tuple(spans),
    )
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((make_analysis_section(blocks=(block,)),)))
    return block


def plan_reply(treatment, segments):
    return pedagogical(treatment_plan=[dict(
        source_block_id="b1", treatment=treatment, segments=segments,
    )])


def _plan_item(analysis, block_id):
    return next(item for item in analysis.treatment_plan.items if item.block_id == block_id)


async def test_segment_treatments_build_mixed_plan_item(stored_book):
    """Segment modes are stored in span order, and the block mode is the strongest."""
    _, uow, _ = stored_book
    block = save_segmented_block(
        uow, ("Holmes speaks. ", "Watson listens."), (0, 1),
    )
    result = await scenario(uow, FakeLLMClient([
        narrative(),
        plan_reply("TRANSLATE", [
            dict(index=2, treatment="TRANSLATE"),
            dict(index=1, treatment="READ_AS_IS"),
        ]),
    ]))(COMMAND)
    item = _plan_item(result.analysis, block.id)
    assert item.treatment is BlockTreatment.TRANSLATE
    assert item.segment_treatments == (
        BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE,
    )


async def test_different_treatments_in_same_sentence_rejected_by_merge(stored_book):
    """Spans of one sentence cannot keep different modes."""
    _, uow, _ = stored_book
    save_segmented_block(
        uow, ("Holmes speaks softly ", "and then stops."), (0, 0),
    )
    client = FakeLLMClient([
        narrative(),
        plan_reply("TRANSLATE", [
            dict(index=1, treatment="READ_AS_IS"),
            dict(index=2, treatment="TRANSLATE"),
        ]),
    ])
    with pytest.raises(ResultValidationError, match="разные режимы"):
        await scenario(uow, client)(COMMAND)


async def test_partial_segment_coverage_rejected(stored_book):
    """A missing span, a repeated index, and an index past the spans are rejected."""
    _, uow, _ = stored_book
    save_segmented_block(
        uow, ("One. ", "Two. ", "Three."), (0, 1, 2),
    )
    cases = (
        ([dict(index=1, treatment="READ_AS_IS"), dict(index=3, treatment="TRANSLATE")],
         "не все сегменты"),
        ([dict(index=1, treatment="READ_AS_IS"), dict(index=1, treatment="TRANSLATE")],
         "повторно"),
        ([dict(index=1, treatment="READ_AS_IS"), dict(index=4, treatment="TRANSLATE")],
         "вне диапазона"),
    )
    for index, (segments, message) in enumerate(cases):
        client = FakeLLMClient(
            [plan_reply("TRANSLATE", segments)] if index else
            [narrative(), plan_reply("TRANSLATE", segments)]
        )
        with pytest.raises(ResultValidationError, match=message):
            await scenario(uow, client)(COMMAND)


async def test_segments_for_short_block_rejected(stored_book):
    """A one-span block, including a block with no stored segments, rejects segments."""
    _, uow, _ = stored_book
    client = FakeLLMClient([
        narrative(),
        plan_reply("READ_AS_IS", [dict(index=1, treatment="READ_AS_IS")]),
    ])
    with pytest.raises(ResultValidationError, match="одного сегмента"):
        await scenario(uow, client)(COMMAND)


async def test_uniform_segments_collapse_to_block_treatment(stored_book):
    """Identical segment modes collapse, and the block mode stays that mode."""
    _, uow, _ = stored_book
    block = save_segmented_block(
        uow, ("Holmes speaks. ", "Watson listens."), (0, 1),
    )
    result = await scenario(uow, FakeLLMClient([
        narrative(),
        plan_reply("GLOSSARY_ONLY", [
            dict(index=2, treatment="GLOSSARY_ONLY"),
            dict(index=1, treatment="GLOSSARY_ONLY"),
        ]),
    ]))(COMMAND)
    item = _plan_item(result.analysis, block.id)
    assert item.treatment is BlockTreatment.GLOSSARY_ONLY
    assert item.segment_treatments == ()
    assert result.analysis.treatment_plan.segment_treatments_for(block.id, 2) == (
        BlockTreatment.GLOSSARY_ONLY, BlockTreatment.GLOSSARY_ONLY,
    )


async def test_description_kept_for_person_and_place_only(stored_book):
    """A candidate description survives only for person and place."""
    import json

    _, uow, _ = stored_book
    candidates = [
        dict(source_text="Holmes", type="person", suggested_translation="Холмс",
             confidence=0.95, model_flagged_review=False, note="",
             description="Детектив с Бейкер-стрит."),
        dict(source_text="Baker Street", type="place", suggested_translation="Бейкер-стрит",
             confidence=0.95, model_flagged_review=False, note="",
             description="Улица в Лондоне."),
        dict(source_text="actual", type="term", suggested_translation="действительный",
             confidence=0.95, model_flagged_review=False, note="",
             description="ложный друг: не «актуальный»."),
        dict(source_text="Scotland Yard", type="organization",
             suggested_translation="Скотленд-Ярд", confidence=0.95,
             model_flagged_review=False, note="", description="Лондонская полиция."),
        dict(source_text="speckled band", type="fictional_concept",
             suggested_translation="пёстрая лента", confidence=0.95,
             model_flagged_review=False, note="", description="Вымышленный предмет."),
    ]
    raw = json.loads(narrative().content)
    raw["glossary_candidates"] = candidates
    result = await scenario(uow, FakeLLMClient([
        FakeReply.of_payload(raw), pedagogical(),
    ]))(COMMAND)
    by_source = {item.source_text: item.description for item in result.analysis.glossary_candidates}
    assert by_source == {
        "Holmes": "Детектив с Бейкер-стрит.",
        "Baker Street": "Улица в Лондоне.",
        "actual": "",
        "Scotland Yard": "",
        "speckled band": "",
    }


A2_SUMMARY = 'A speaker asks for a small box.'
_LONG_QUOTE = '“Please bring the small red box from the hall before noon.”'
_LONG_CRATE = '“Please carry the small red crate from the hall before noon.”'


def _block(order, text, *, type=BlockType.PARAGRAPH, segments=()):
    block = make_analysis_block(section_id='section-1', order=order, text=text, type=type)
    return replace(block, segments=segments) if segments else block


def _store(uow, blocks):
    section = make_analysis_section(blocks=tuple(blocks))
    with uow.transaction() as tx:
        tx.documents.save(make_analysis_document((section,)))
    return section


def _aliases(count):
    return tuple(f'b{index}' for index in range(1, count + 1))


def _plan(treatments):
    return [dict(source_block_id=alias, treatment=treatment) for alias, treatment in treatments]


def _vocab(alias, quote, difficulty_id, *, help_kind='lexical_hint', kind='vocabulary'):
    return dict(id=difficulty_id, anchors=[dict(block_id=alias, quote=quote)], owner_block_id=alias,
                kind=kind, priority='useful', help_kind=help_kind,
                required_points=[dict(id='p1', description='Объяснить значение слова в этой реплике.')])


def _modes(analysis, blocks):
    plan = analysis.treatment_plan
    return {block.text: plan.treatment_for(block.id) for block in blocks}


async def test_a2_dialogue_longer_than_eight_words_promoted_to_translate(stored_book):
    """A quoted A2 block or dialogue segment longer than eight words becomes TRANSLATE."""
    from sqlalchemy import update

    from linguary.adapters.db.models import AnalysisPassResultRow
    from linguary.application.analysis_pass_cache import decode_pass, encode_pass
    from linguary.domain import BlockSegment
    from linguary.domain.content_plan import canonical_sha

    engine, uow, _ = stored_book
    quote_block = _block(0, _LONG_QUOTE, type=BlockType.QUOTE)
    segment_text = _LONG_QUOTE + ' The cat slept.'
    split = len(_LONG_QUOTE)
    segment_block = _block(1, segment_text, segments=(
        BlockSegment(0, split, 0), BlockSegment(split, len(segment_text), 1),
    ))
    _store(uow, (quote_block, segment_block))

    class FragmentClient:
        async def generate(self, request):
            aliases = _aliases(len(request.ordered_block_ids))
            if request.stage == NARRATIVE:
                reply = narrative(aliases)
            elif request.ordered_block_ids == (segment_block.id,):
                reply = pedagogical(summary=A2_SUMMARY, treatment_plan=[dict(
                    source_block_id='b1', treatment='GLOSSARY_ONLY',
                    segments=[dict(index=1, treatment='GLOSSARY_ONLY'),
                              dict(index=2, treatment='READ_AS_IS')])])
            else:
                reply = pedagogical(aliases, summary=A2_SUMMARY)
            return await FakeLLMClient([reply]).generate(request)

    result = await scenario(uow, FragmentClient(), context=builder('a2'), max_fragment_blocks=1)(COMMAND)
    plan = result.analysis.treatment_plan
    assert plan.treatment_for(quote_block.id) is BlockTreatment.TRANSLATE
    assert plan.treatment_for(segment_block.id) is BlockTreatment.TRANSLATE
    assert plan.segment_treatments_for(segment_block.id, 2) == (
        BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS,
    )
    with uow.transaction() as tx:
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.PEDAGOGICAL)
    by_id = {item['source_block_id']: item for item in saved.payload['response']['treatment_plan']}
    assert by_id[quote_block.id]['treatment'] == 'TRANSLATE'
    assert by_id[segment_block.id]['treatment'] == 'TRANSLATE'
    assert [segment['treatment'] for segment in by_id[segment_block.id]['segments']] == [
        'TRANSLATE', 'READ_AS_IS',
    ]

    decoded = decode_pass(saved.payload, narrative=False)
    reverted = replace(decoded, treatment_plan=tuple(
        replace(item, treatment=BlockTreatment.READ_AS_IS, segments=())
        if item.source_block_id == quote_block.id else item
        for item in decoded.treatment_plan))
    encoded = encode_pass(reverted)
    with engine.begin() as connection:
        connection.execute(update(AnalysisPassResultRow).where(AnalysisPassResultRow.id == saved.id)
                           .values(payload_json=encoded, output_sha256=canonical_sha(encoded)))
    cached = await scenario(uow, FakeLLMClient(), context=builder('a2'), max_fragment_blocks=1)(COMMAND)
    assert cached.attempts == 0
    assert cached.analysis.treatment_plan.treatment_for(quote_block.id) is BlockTreatment.TRANSLATE

    one_pass_block = _block(0, _LONG_QUOTE)
    section = _store(uow, (one_pass_block,))
    from analysis_fakes import analysis_reply
    one_pass = FakeLLMClient([analysis_reply(
        aliases=('b1',), treatment='READ_AS_IS', chapter_summary=A2_SUMMARY, before_you_read=[],
        content_plan=dict(version=3, profile='content-plan-v3', difficulties=[],
                          before_you_read_evidence=[], context_facts=[]))])
    single = await AnalyzeChapter(unit_of_work=uow, llm_client=one_pass,
        language_detector=FakeLanguageDetector(), retry_policy=RetryPolicy(max_attempts=1),
        glossary_confidence_threshold=.8, context_builder=builder('a2'))(COMMAND)
    assert single.analysis.treatment_plan.treatment_for(section.blocks[0].id) is BlockTreatment.TRANSLATE


async def test_a2_dialogue_with_conditional_or_reporting_difficulty_promoted_to_translate(stored_book):
    """A short quote is translated for a subordinator or a non-basic reporting verb.

    Remark vocabulary promotes a reply only when it is longer than eight words
    or the speech verb is non-basic. A short elementary reply keeps its mode.
    """
    _, uow, _ = stored_book
    cases = (
        ('‘Stay here if you can.’', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Come in,’ she demanded.', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Yes,’ he murmured.', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Stop,’ she snapped.', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Yes,’ he whispered.', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Yes,’ he said grudgingly.', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('“Call me when you arrive.”', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('"Bring it because I asked."', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Leave unless I call.’', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Ask whether he came today.’', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Wait although it is late.’', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Stay though it is cold.’', 'READ_AS_IS', BlockTreatment.TRANSLATE, None),
        ('‘Hello,’ he said about the ledger.', 'GLOSSARY_ONLY', BlockTreatment.GLOSSARY_ONLY,
         _vocab('b13', 'ledger', 'd-ledger')),
        ('‘Hello,’ he said about the actual.', 'GLOSSARY_ONLY', BlockTreatment.GLOSSARY_ONLY,
         _vocab('b14', 'actual', 'd-actual', kind='false_friend')),
        ('‘Hello ledger,’ he said.', 'GLOSSARY_ONLY', BlockTreatment.GLOSSARY_ONLY,
         _vocab('b15', 'ledger', 'd-inside')),
        ('‘Hello,’ he said about the tense.', 'READ_AS_IS', BlockTreatment.READ_AS_IS,
         _vocab('b16', 'tense', 'd-tense', help_kind='explanation', kind='grammar')),
    )
    blocks = tuple(_block(index, text) for index, (text, *_rest) in enumerate(cases))
    _store(uow, blocks)
    aliases = _aliases(len(blocks))
    difficulties = [case[3] for case in cases if case[3] is not None]
    taught = pedagogical(summary=A2_SUMMARY, treatment_plan=_plan(zip(aliases, (case[1] for case in cases), strict=True)),
                         difficulties=difficulties)
    result = await scenario(uow, FakeLLMClient([narrative(aliases), taught]),
                            context=builder('a2'))(COMMAND)
    assert _modes(result.analysis, blocks) == {block.text: case[2] for block, case in zip(blocks, cases, strict=True)}
    by_quote = {}
    for item in result.analysis.content_plan.difficulties:
        by_quote.setdefault(item.anchors[0].quote, []).append(item.help_kind)
    assert by_quote['ledger'] == ['lexical_hint', 'lexical_hint']
    assert by_quote['actual'] == ['lexical_hint']
    assert by_quote['tense'] == ['explanation']


async def test_a2_promoted_glossary_only_dialogue_switches_lexical_hint_to_explanation(stored_book):
    """Only a GLOSSARY_ONLY dialogue that becomes TRANSLATE changes lexical_hint to explanation."""
    _, uow, _ = stored_book
    promoted = _block(0, _LONG_QUOTE)
    covered = _block(1, _LONG_CRATE)
    kept = _block(2, '‘Go now!’ he said loudly.')
    _store(uow, (promoted, covered, kept))
    explanation = 'Это слово называет ящик, а не животное.'
    taught = pedagogical(summary=A2_SUMMARY, treatment_plan=_plan((
        ('b1', 'GLOSSARY_ONLY'), ('b2', 'READ_AS_IS'), ('b3', 'GLOSSARY_ONLY'),
    )), difficulties=[
        _vocab('b1', 'box', 'd-box'),
        _vocab('b2', 'crate', 'd-crate'),
        _vocab('b3', 'Go', 'd-go'),
    ], before_you_read=[dict(expression='crate', explanation=explanation)],
        before_you_read_evidence=[dict(difficulty_id='d-crate', point_id='p1',
            artifact_kind='before_you_read.explanation', source_block_id='b2', artifact_index=0,
            body_excerpt=explanation)])
    result = await scenario(uow, FakeLLMClient([narrative(('b1', 'b2', 'b3')), taught]),
                            context=builder('a2'))(COMMAND)
    plan = result.analysis.treatment_plan
    assert plan.treatment_for(promoted.id) is BlockTreatment.TRANSLATE
    assert plan.treatment_for(covered.id) is BlockTreatment.TRANSLATE
    assert plan.treatment_for(kept.id) is BlockTreatment.GLOSSARY_ONLY
    by_quote = {item.anchors[0].quote: item.help_kind for item in result.analysis.content_plan.difficulties}
    assert by_quote == {'box': 'explanation', 'crate': 'lexical_hint', 'Go': 'lexical_hint'}
    with uow.transaction() as tx:
        saved = tx.analyses.get_pass(BOOK_ID, COMMAND.section_id, AnalysisPassKind.PEDAGOGICAL)
    saved_help = {item['anchors'][0]['quote']: item['help_kind']
                  for item in saved.payload['response']['difficulties']}
    assert saved_help['box'] == 'explanation'
    assert saved_help['crate'] == 'lexical_hint'

    one_pass_block = _block(0, _LONG_QUOTE)
    section = _store(uow, (one_pass_block,))
    from analysis_fakes import analysis_reply
    one_pass = FakeLLMClient([analysis_reply(
        aliases=('b1',), treatment='GLOSSARY_ONLY', chapter_summary=A2_SUMMARY, before_you_read=[],
        content_plan=dict(version=3, profile='content-plan-v3', before_you_read_evidence=[], context_facts=[],
            difficulties=[_vocab('b1', 'box', 'd-box')]))])
    single = await AnalyzeChapter(unit_of_work=uow, llm_client=one_pass,
        language_detector=FakeLanguageDetector(), retry_policy=RetryPolicy(max_attempts=1),
        glossary_confidence_threshold=.8, context_builder=builder('a2'))(COMMAND)
    assert single.analysis.treatment_plan.treatment_for(section.blocks[0].id) is BlockTreatment.TRANSLATE
    assert single.analysis.content_plan.difficulties[0].help_kind == 'explanation'


async def test_a2_elementary_short_dialogues_and_non_dialogue_lines_keep_mode(stored_book):
    """Short elementary replies and lines without dialogue quotes keep their mode."""
    import linguary.application.analysis_merge as analysis_merge

    assert hasattr(analysis_merge, 'normalize_a2_dialogue_response')
    _, uow, _ = stored_book
    cases = (
        ('‘Go now!’ he said loudly.', 'READ_AS_IS', BlockType.PARAGRAPH),
        ('‘Up now!’', 'GLOSSARY_ONLY', BlockType.PARAGRAPH),
        ('‘Yes.’', 'READ_AS_IS', BlockType.PARAGRAPH),
        ('‘Please come over here right now,’ she said.', 'READ_AS_IS', BlockType.PARAGRAPH),
        ('The old gardener walked slowly across the empty yard before noon.', 'GLOSSARY_ONLY', BlockType.PARAGRAPH),
        ('He demanded an answer if the old gardener walked slowly across the yard.', 'GLOSSARY_ONLY', BlockType.PARAGRAPH),
        ('It blinked.', 'READ_AS_IS', BlockType.PARAGRAPH),
        (_LONG_QUOTE, 'READ_AS_IS', BlockType.LIST_ITEM),
    )
    blocks = tuple(_block(index, text, type=block_type) for index, (text, _mode, block_type) in enumerate(cases))
    _store(uow, blocks)
    aliases = _aliases(len(blocks))
    original = _plan(zip(aliases, (case[1] for case in cases), strict=True))
    expected = {block.text: BlockTreatment(case[1]) for block, case in zip(blocks, cases, strict=True)}
    result = await scenario(uow, FakeLLMClient([
        narrative(aliases), pedagogical(summary=A2_SUMMARY, treatment_plan=original),
    ]), context=builder('a2'))(COMMAND)
    assert _modes(result.analysis, blocks) == expected

    kept = await scenario(uow, FakeLLMClient([pedagogical(treatment_plan=original)]),
                          context=builder('b1'))(COMMAND)
    assert _modes(kept.analysis, blocks) == expected

    english_block = _block(0, _LONG_QUOTE)
    _store(uow, (english_block,))
    english = await scenario(uow, FakeLLMClient([narrative(), pedagogical(summary=A2_SUMMARY)]),
                             context=builder('a2', help=True))(COMMAND)
    assert english.analysis.treatment_plan.treatment_for(english_block.id) is BlockTreatment.READ_AS_IS


def test_a2_split_sentence_segments_share_the_promoted_mode():
    """One promoted piece of a split sentence lifts every piece of that sentence.

    A later sentence keeps the mode the model assigned. A lexical hint on the
    promoted tail or on another segment of a block that becomes TRANSLATE
    becomes an explanation. A quote split across two segments of one sentence is
    evaluated across the full sentence span. Assembly and validation must accept
    the plan.
    """
    from linguary.application.analysis_merge import (
        FragmentAnalysis,
        normalize_a2_dialogue_response,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        PedagogicalAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
        SegmentTreatmentResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    tail = 'He waited by the lane '
    quote = '‘Stay here if you can.’'
    other = ' ‘Yes.’'
    text = tail + quote + other
    quote_at = len(tail)
    other_at = quote_at + len(quote)
    assert text[quote_at:other_at] == quote
    assert 'lane' in tail and 'lane' not in quote and 'lane' not in other
    block = _block(0, text, segments=(
        BlockSegment(0, quote_at, 0),
        BlockSegment(quote_at, other_at, 0),
        BlockSegment(other_at, len(text), 1),
    ))
    split_open = '‘Stay here '
    split_close = 'if you can,’ she demanded.'
    split_tail = ' The cat slept.'
    split_text = split_open + split_close + split_tail
    first_cut = len(split_open)
    second_cut = first_cut + len(split_close)
    split_block = _block(1, split_text, segments=(
        BlockSegment(0, first_cut, 0),
        BlockSegment(first_cut, second_cut, 0),
        BlockSegment(second_cut, len(split_text), 1),
    ))
    point = (RequiredPoint('p1', 'Объяснить значение слова в этой реплике.'),)
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.GLOSSARY_ONLY, (
                SegmentTreatmentResponse(1, BlockTreatment.GLOSSARY_ONLY),
                SegmentTreatmentResponse(2, BlockTreatment.GLOSSARY_ONLY),
                SegmentTreatmentResponse(3, BlockTreatment.READ_AS_IS),
            )),
            TreatmentPlanItemResponse('b2', BlockTreatment.GLOSSARY_ONLY, (
                SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
                SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                SegmentTreatmentResponse(3, BlockTreatment.GLOSSARY_ONLY),
            )),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                'd-lane',
                (QuotedSourceAnchorResponse('b1', 'lane'),),
                'b1',
                'vocabulary',
                'useful',
                'lexical_hint',
                point,
            ),
            QuotedDifficultyResponse(
                'd-slept',
                (QuotedSourceAnchorResponse('b2', 'slept'),),
                'b2',
                'vocabulary',
                'useful',
                'lexical_hint',
                point,
            ),
        ),
    )
    aliases = {'b1': block.id, 'b2': split_block.id}
    normalized = normalize_a2_dialogue_response(
        payload, (block, split_block), aliases, reader_level='a2', english_help=False,
    )
    assert [segment.treatment for segment in normalized.treatment_plan[0].segments] == [
        BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS,
    ]
    assert [segment.treatment for segment in normalized.treatment_plan[1].segments] == [
        BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY,
    ]
    assert [item.help_kind for item in normalized.difficulties] == ['explanation', 'explanation']
    combined = ChapterAnalysisResponse(
        translation_instructions='Keep the voice.',
        chapter_summary='A speaker waits by the lane.',
        treatment_plan=normalized.treatment_plan,
        content_plan=QuotedContentPlanResponse(difficulties=normalized.difficulties),
    )
    texts = {block.id: block.text, split_block.id: split_block.text}
    visible = {'b1': block.text, 'b2': split_block.text}
    validated = validate_content_plan(
        combined, aliases, texts, visible_texts_by_alias=visible, blocks=(block, split_block),
    )
    assert validated is not None
    assert [item.help_kind for item in validated.difficulties] == ['explanation', 'explanation']
    fragment = FragmentAnalysis.of_response(
        combined,
        order=0,
        block_ids_by_alias=aliases,
        texts=texts,
        visible_texts_by_alias=visible,
        blocks=(block, split_block),
    )
    assert fragment.treatments[0].treatment is BlockTreatment.TRANSLATE
    assert fragment.treatments[0].segment_treatments == (
        BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS,
    )
    assert fragment.treatments[1].treatment is BlockTreatment.TRANSLATE
    assert fragment.treatments[1].segment_treatments == (
        BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY,
    )


def test_a2_apostrophe_inside_single_quotes_is_not_the_closer():
    """An apostrophe inside, before, or after a word does not end ‘...’.

    ‘Don’t go if I call.’, ‘Get ’em if you can.’, and ‘Check the boys’ room if you can.’
    are translated. ‘I’m here.’ and ‘No,’ she said sharply. ‘Why?’ stay short exceptions.
    ‘Don’t touch the ledger.’ and ‘Check the boys’ ledger.’ keep ledger inside the reply.
    A vocabulary anchor in a short elementary remark stays glossary-only.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse, QuotedDifficultyResponse, QuotedSourceAnchorResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    conditional = '‘Don’t go if I call.’'
    short = '‘I’m here.’'
    interior = '‘Don’t touch the ledger.’'
    remark = '‘Don’t,’ he said about the ledger.'
    elision = '‘Get ’em if you can.’'
    possessive = '‘Check the boys’ room if you can.’'
    possessive_interior = '‘Check the boys’ ledger.’'
    two_quotes = '‘No,’ she said sharply. ‘Why?’'
    typographic_remark = '‘No,’ he didn’t answer.'
    word_close_remark = '‘Boys’ if he didn’t say.'
    unclosed_many_apostrophes = '‘' + ' '.join(["don’t"] * 28)
    blocks = (
        _block(0, conditional),
        _block(1, short),
        _block(2, interior),
        _block(3, remark),
        _block(4, elision),
        _block(5, possessive),
        _block(6, possessive_interior),
        _block(7, two_quotes),
        _block(8, typographic_remark),
        _block(9, word_close_remark),
        _block(10, unclosed_many_apostrophes),
    )
    point = (RequiredPoint('p1', 'Объяснить значение слова в этой реплике.'),)

    def vocab(difficulty_id, alias, quote='ledger'):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse(alias, quote),), alias,
            'vocabulary', 'useful', 'lexical_hint', point,
        )

    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b2', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b3', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b4', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b5', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b6', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b7', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b8', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b9', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b10', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b11', BlockTreatment.READ_AS_IS),
        ),
        difficulties=(
            vocab('d-inside', 'b3'),
            vocab('d-remark', 'b4'),
            vocab('d-possessive-inside', 'b7'),
            vocab('d-typo-remark', 'b9', quote="didn't"),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
    ]
    assert [item.help_kind for item in normalized.difficulties] == [
        'lexical_hint',
        'lexical_hint',
        'lexical_hint',
        'lexical_hint',
    ]


def test_short_elementary_dialogue_with_adverb_difficulty_stays_or_demotes_to_glossary_only():
    """A remark adverb does not translate a short elementary A2 reply.

    ‘Shoo!’ said Mr Dursley loudly. and ‘No,’ she said sharply. ‘Why?’ stay
    GLOSSARY_ONLY when the only difficulty is that adverb. A model TRANSLATE on
    the same replies is demoted, and those vocabulary items become lexical hints.
    The same reply with no difficulty is demoted to READ_AS_IS.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse, QuotedDifficultyResponse, QuotedSourceAnchorResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    shoo = '‘Shoo!’ said Mr Dursley loudly.'
    sharply = '‘No,’ she said sharply. ‘Why?’'
    bare = '‘Yes,’ he said.'
    blocks = (
        _block(0, shoo),
        _block(1, shoo),
        _block(2, sharply),
        _block(3, sharply),
        _block(4, bare),
    )
    point = (RequiredPoint('p1', 'Объяснить значение слова в этой реплике.'),)

    def vocab(difficulty_id, alias, quote, help_kind):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse(alias, quote),), alias,
            'vocabulary', 'useful', help_kind, point,
        )

    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b2', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b3', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b4', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b5', BlockTreatment.TRANSLATE),
        ),
        difficulties=(
            vocab('d-shoo-translate', 'b1', 'loudly', 'explanation'),
            vocab('d-shoo-glossary', 'b2', 'loudly', 'lexical_hint'),
            vocab('d-sharp-translate', 'b3', 'sharply', 'explanation'),
            vocab('d-sharp-glossary', 'b4', 'sharply', 'lexical_hint'),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
    ]
    assert [item.help_kind for item in normalized.difficulties] == [
        'lexical_hint',
        'lexical_hint',
        'lexical_hint',
        'lexical_hint',
    ]


def test_short_dialogue_with_non_elementary_speech_verb_still_promotes_to_translate():
    """Demanded, snapped, a question tag, and a reply over eight words are translated."""
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import PedagogicalAnalysisResponse, TreatmentPlanItemResponse

    demanded = '‘Are you up yet?’ she demanded.'
    snapped = '‘What did you say?’ his aunt snapped through the door.'
    tag = '‘No problems, were there?’'
    long_reply = '‘Please bring the small red box from the hall,’ she said.'
    blocks = (
        _block(0, demanded),
        _block(1, snapped),
        _block(2, tag),
        _block(3, long_reply),
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(f'b{index}', BlockTreatment.READ_AS_IS)
            for index in range(1, 5)
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    ]


def test_normalize_a2_dialogue_response_clears_before_you_read():
    """A2 without English help drops before_you_read when nothing cites it.

    PedagogicalAnalysisResponse and ChapterAnalysisResponse both lose the list
    when before_you_read_evidence is empty. Evidence keeps the list.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        BeforeYouReadEntryResponse, ChapterAnalysisResponse, HelpEvidenceResponse,
        PedagogicalAnalysisResponse, QuotedContentPlanResponse, TreatmentPlanItemResponse,
    )

    block = _block(0, 'It blinked.')
    aliases = {'b1': block.id}
    entry = BeforeYouReadEntryResponse('Holmes', 'Главный сыщик истории.')
    evidence = HelpEvidenceResponse(
        'd1', 'p1', 'before_you_read.explanation', 'b1', 0, entry.explanation,
    )
    pedagogical_cleared = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),),
            before_you_read=(entry,),
        ),
        (block,), aliases, reader_level='a2', english_help=False,
    )
    assert pedagogical_cleared.before_you_read == ()

    pedagogical_kept = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),),
            before_you_read=(entry,),
            before_you_read_evidence=(evidence,),
        ),
        (block,), aliases, reader_level='a2', english_help=False,
    )
    assert pedagogical_kept.before_you_read == (entry,)

    chapter_cleared = normalize_a2_dialogue_response(
        ChapterAnalysisResponse(
            translation_instructions='Keep the voice.',
            chapter_summary='A speaker waits.',
            treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),),
            before_you_read=(entry,),
            content_plan=QuotedContentPlanResponse(),
        ),
        (block,), aliases, reader_level='a2', english_help=False,
    )
    assert chapter_cleared.before_you_read == ()

    chapter_kept = normalize_a2_dialogue_response(
        ChapterAnalysisResponse(
            translation_instructions='Keep the voice.',
            chapter_summary='A speaker waits.',
            treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),),
            before_you_read=(entry,),
            content_plan=QuotedContentPlanResponse(before_you_read_evidence=(evidence,)),
        ),
        (block,), aliases, reader_level='a2', english_help=False,
    )
    assert chapter_kept.before_you_read == (entry,)


def test_cried_and_shouted_stay_glossary_only_like_other_basic_speech_verbs():
    """cried and shouted are basic: a light mode is not raised, and TRANSLATE drops.

    ‘Are you up yet?’ she demanded. still promotes. A reply longer than eight
    words still becomes TRANSLATE, and a model TRANSLATE on that reply stays.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse, QuotedDifficultyResponse, QuotedSourceAnchorResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    cried = '‘Yes,’ he cried.'
    shouted = '‘Yes,’ he shouted.'
    demanded = '‘Are you up yet?’ she demanded.'
    long_reply = '‘Please bring the small red box from the hall,’ she said.'
    blocks = (
        _block(0, cried),
        _block(1, shouted),
        _block(2, cried),
        _block(3, cried),
        _block(4, demanded),
        _block(5, long_reply),
        _block(6, long_reply),
    )
    point = (RequiredPoint('p1', 'Объяснить значение слова в этой реплике.'),)
    difficulty = QuotedDifficultyResponse(
        'd-cried', (QuotedSourceAnchorResponse('b4', 'Yes'),), 'b4',
        'vocabulary', 'useful', 'explanation', point,
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b2', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse('b3', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b4', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b5', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b6', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b7', BlockTreatment.TRANSLATE),
        ),
        difficulties=(difficulty,),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    ]
    assert normalized.difficulties[0].help_kind == 'lexical_hint'


def test_segmented_elementary_translate_reply_demotes_with_its_segments():
    """Per-sentence TRANSLATE on a short elementary reply drops with the block."""
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse, QuotedDifficultyResponse, QuotedSourceAnchorResponse,
        SegmentTreatmentResponse, TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    text = '‘No,’ she said sharply. ‘Why?’'
    second = text.rindex('‘')
    block = _block(0, text, segments=(
        BlockSegment(0, second, 0),
        BlockSegment(second, len(text), 1),
    ))
    point = (RequiredPoint('p1', 'Объяснить значение слова в этой реплике.'),)
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE, (
                SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                SegmentTreatmentResponse(2, BlockTreatment.TRANSLATE),
            )),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                'd-sharply', (QuotedSourceAnchorResponse('b1', 'sharply'),), 'b1',
                'vocabulary', 'useful', 'explanation', point,
            ),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, (block,), {'b1': block.id}, reader_level='a2', english_help=False,
    )
    item = normalized.treatment_plan[0]
    assert item.treatment is BlockTreatment.GLOSSARY_ONLY
    assert [segment.treatment for segment in item.segments] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
    ]
    assert normalized.difficulties[0].help_kind == 'lexical_hint'


def test_secondary_anchor_on_demoted_reply_does_not_retarget_foreign_owner():
    """A vocabulary difficulty owned by a TRANSLATE block keeps `explanation`."""
    from linguary.application.analysis_merge import (
        normalize_a2_dialogue_response,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    b1 = _block(0, 'He walked past the house, looking sharply at the cat on the wall.')
    b2 = _block(1, '‘No,’ she said sharply. ‘Why?’')
    point = (RequiredPoint('p1', 'Объяснить значение наречия sharply.'),)
    payload = ChapterAnalysisResponse(
        chapter_summary='Summary.',
        translation_instructions='Keep tone.',
        before_you_read=(),
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b2', BlockTreatment.TRANSLATE),
        ),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile='content-plan-v3',
            difficulties=(
                QuotedDifficultyResponse(
                    'd-sharply',
                    (
                        QuotedSourceAnchorResponse('b1', 'sharply'),
                        QuotedSourceAnchorResponse('b2', 'sharply'),
                    ),
                    'b1',
                    'vocabulary',
                    'useful',
                    'explanation',
                    point,
                ),
            ),
        ),
    )
    aliases = {'b1': b1.id, 'b2': b2.id}
    normalized = normalize_a2_dialogue_response(
        payload, (b1, b2), aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
    ]
    assert normalized.content_plan.difficulties[0].help_kind == 'explanation'
    plan = validate_content_plan(
        normalized,
        aliases,
        {b1.id: b1.text, b2.id: b2.text},
        visible_texts_by_alias={'b1': b1.text, 'b2': b2.text},
        blocks=(b1, b2),
    )
    assert plan is not None


def test_a2_chapter_title_heading_promotes_to_translate_and_drops_from_no_one_grammar_note():
    """A meaningful chapter title is translated and loses an unevidenced grammar note.

    — CHAPTER THREE — and a bare roman numeral stay READ_AS_IS. The Letters from
    No One becomes TRANSLATE. Grammar quote from no one is dropped only when
    nothing cites it. The same note with evidence, a vocabulary note, and the
    same title in a paragraph stay.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        HelpEvidenceResponse, PedagogicalAnalysisResponse, QuotedDifficultyResponse,
        QuotedSourceAnchorResponse, TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    numbered = '— CHAPTER THREE —'
    roman = 'II'
    title = 'The Letters from No One'
    blocks = (
        _block(0, numbered, type=BlockType.HEADING),
        _block(1, roman, type=BlockType.HEADING),
        _block(2, title, type=BlockType.HEADING),
        _block(3, title),
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    point = (RequiredPoint('p1', 'Объяснить конструкцию в заголовке.'),)

    def grammar(difficulty_id, alias, quote='from no one'):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse(alias, quote),), alias,
            'grammar', 'useful', 'explanation', point,
        )

    payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(alias, BlockTreatment.READ_AS_IS) for alias in aliases
        ),
        difficulties=(
            grammar('d-from', 'b3'),
            grammar('d-from-evidenced', 'b3'),
            QuotedDifficultyResponse(
                'd-letters', (QuotedSourceAnchorResponse('b3', 'Letters'),), 'b3',
                'vocabulary', 'useful', 'lexical_hint', point,
            ),
            grammar('d-paragraph', 'b4'),
        ),
        before_you_read_evidence=(
            HelpEvidenceResponse(
                'd-from-evidenced', 'p1', 'before_you_read.explanation', 'b3', 0,
                'Предлог from в заголовке.',
            ),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
    ]
    assert {item.id: item.help_kind for item in normalized.difficulties} == {
        'd-from-evidenced': 'explanation',
        'd-letters': 'explanation',
        'd-paragraph': 'explanation',
    }

    untouched = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='b1', english_help=False,
    )
    assert [item.treatment for item in untouched.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
    ]
    assert [item.id for item in untouched.difficulties] == [
        'd-from', 'd-from-evidenced', 'd-letters', 'd-paragraph',
    ]


def test_a2_short_lines_with_winked_groaned_chuckled_or_fancy_seeing_you_do_not_stay_read_as_is():
    """Short READ_AS_IS lines with the listed B1+ words or phrases leave that mode.

    No vocabulary difficulty means TRANSLATE. A vocabulary difficulty means
    GLOSSARY_ONLY and lexical_hint. Elementary replies, a line over ten words,
    and a line that is already GLOSSARY_ONLY stay as they are.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse, QuotedDifficultyResponse, QuotedSourceAnchorResponse,
        SegmentTreatmentResponse, TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    translate_lines = (
        'It winked.',
        'Harry groaned.',
        'Uncle Vernon chuckled.',
        '‘Fancy seeing you here, Professor McGonagall.’',
        '‘Thirty-nine, sweetums,’ said Aunt Petunia.',
        'How very wrong he was.',
        'He peered.',
        'They sniggered.',
        'She moaned.',
        '‘I suppose so.’',
    )
    stay_lines = (
        '‘Up! Get up! Now!’',
        '‘A what?’',
        '‘Nearly,’ said Harry.',
        '‘Shoo!’ said Mr Dursley loudly.',
        '‘No,’ she said sharply. ‘Why?’',
        'It blinked.',
        'Fancy that.',
        'How are you?',
        'I suppose he left.',
        'It winked at the boy who stood alone by the door.',
    )
    glossary_line = 'It winked.'
    already_glossary = 'He peered.'
    winked = 'It winked.'
    cut = len('It ')
    blocks = tuple(_block(index, text) for index, text in enumerate(translate_lines))
    glossary_block = _block(len(blocks), glossary_line)
    stay_blocks = tuple(
        _block(len(blocks) + 1 + index, text) for index, text in enumerate(stay_lines)
    )
    kept_glossary = _block(len(blocks) + 1 + len(stay_blocks), already_glossary)
    split = _block(
        len(blocks) + 2 + len(stay_blocks), winked, segments=(
            BlockSegment(0, cut, 0),
            BlockSegment(cut, len(winked), 0),
        ),
    )
    ordered = blocks + (glossary_block,) + stay_blocks + (kept_glossary, split)
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(ordered)}
    glossary_alias = f'b{len(blocks) + 1}'
    split_alias = f'b{len(ordered)}'
    point = (RequiredPoint('p1', 'Объяснить значение слова в этой реплике.'),)
    payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(f'b{index + 1}', BlockTreatment.READ_AS_IS)
            for index in range(len(ordered) - 2)
        ) + (
            TreatmentPlanItemResponse(f'b{len(ordered) - 1}', BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse(split_alias, BlockTreatment.READ_AS_IS, (
                SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
                SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
            )),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                'd-winked', (QuotedSourceAnchorResponse(glossary_alias, 'winked'),),
                glossary_alias, 'vocabulary', 'useful', 'explanation', point,
            ),
            QuotedDifficultyResponse(
                'd-fancy', (QuotedSourceAnchorResponse('b4', 'Fancy seeing you here'),),
                'b4', 'idiom', 'useful', 'lexical_hint', point,
            ),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, ordered, aliases, reader_level='a2', english_help=False,
    )
    expected = (
        [BlockTreatment.TRANSLATE] * len(translate_lines)
        + [BlockTreatment.GLOSSARY_ONLY]
        + [BlockTreatment.READ_AS_IS] * len(stay_lines)
        + [BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE]
    )
    assert [item.treatment for item in normalized.treatment_plan] == expected
    assert [segment.treatment for segment in normalized.treatment_plan[-1].segments] == [
        BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE,
    ]
    assert normalized.difficulties[0].help_kind == 'lexical_hint'
    assert normalized.difficulties[1].help_kind == 'explanation'


def test_a2_translate_block_drops_trivial_sounding_relieved_grammar_difficulty():
    """A block that ends as TRANSLATE loses unevidenced grammar quote sounding relieved.

    Evidence keeps the note. A block that stays READ_AS_IS keeps it. Another
    grammar quote on a translated block stays. The same drop applies when the
    difficulty is stored on a chapter content plan.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse, HelpEvidenceResponse, PedagogicalAnalysisResponse,
        QuotedContentPlanResponse, QuotedDifficultyResponse, QuotedSourceAnchorResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    short = '‘Hagrid,’ said Dumbledore, sounding relieved.'
    promoted = (
        '‘Hagrid,’ said Dumbledore, sounding relieved. '
        '‘At last. And where did you get that motorbike?’'
    )
    wrong = 'How very wrong he was, sounding relieved.'
    blocks = (
        _block(0, short),
        _block(1, short),
        _block(2, short),
        _block(3, promoted),
        _block(4, wrong),
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    point = (RequiredPoint('p1', 'Объяснить конструкцию в этой реплике.'),)

    def grammar(difficulty_id, alias, quote):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse(alias, quote),), alias,
            'grammar', 'useful', 'explanation', point,
        )

    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b2', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b3', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b4', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b5', BlockTreatment.READ_AS_IS),
        ),
        difficulties=(
            grammar('d-relieved', 'b1', 'sounding relieved'),
            grammar('d-relieved-evidenced', 'b2', 'sounding relieved'),
            grammar('d-said', 'b2', 'said'),
            grammar('d-stay', 'b3', 'sounding relieved'),
            grammar('d-promoted', 'b4', 'sounding relieved'),
            grammar('d-wrong', 'b5', 'sounding relieved'),
        ),
        before_you_read_evidence=(
            HelpEvidenceResponse(
                'd-relieved-evidenced', 'p1', 'before_you_read.explanation', 'b2', 0,
                'Говорящий успокаивается.',
            ),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    ]
    assert {item.id: item.anchors[0].quote for item in normalized.difficulties} == {
        'd-relieved-evidenced': 'sounding relieved',
        'd-said': 'said',
        'd-stay': 'sounding relieved',
    }

    chapter_block = _block(0, short)
    relieved = grammar('d-relieved', 'b1', 'sounding relieved')
    chapter = normalize_a2_dialogue_response(
        ChapterAnalysisResponse(
            translation_instructions='Keep the voice.',
            chapter_summary='Dumbledore speaks.',
            treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),),
            content_plan=QuotedContentPlanResponse(difficulties=(relieved,)),
        ),
        (chapter_block,), {'b1': chapter_block.id}, reader_level='a2', english_help=False,
    )
    assert chapter.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert chapter.content_plan.difficulties == ()


def test_a2_short_replies_with_cannon_budge_lump_or_eagerly_keep_translate():
    """B1+ words and non-basic remark adverbs keep a short reply in TRANSLATE.

    ‘Where’s the cannon?’ he said stupidly., ‘Budge up, yeh great lump,’ said
    the stranger., and ‘Kept what from me?’ said Harry eagerly. are not demoted.
    Up, A what, Nearly, Shoo, and No/Why are not promoted. A model TRANSLATE on
    Nearly, Shoo, or No/Why still drops to READ_AS_IS. B1 leaves those lines alone.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import PedagogicalAnalysisResponse, TreatmentPlanItemResponse

    translate_lines = (
        '‘Where’s the cannon?’ he said stupidly.',
        '‘Budge up, yeh great lump,’ said the stranger.',
        '‘Kept what from me?’ said Harry eagerly.',
        '‘Kept what from me?’ said Harry.',
        '‘Yes,’ he said stupidly.',
        '‘Yes,’ he said eagerly.',
        '‘Yes,’ he said gingerly.',
        '‘Yes,’ he said furiously.',
        '‘Yes,’ he said nervously.',
        '‘Yes,’ he said casually.',
        '‘Yes,’ he said sternly.',
        '‘Yes,’ he said firmly.',
        '‘Yes,’ he said anxiously.',
        '‘Yes,’ he said bitterly.',
        '‘Yes,’ he said wearily.',
        '‘Yes,’ he said sleepily.',
        '‘Yes,’ he said coldly.',
        '‘Cannon,’ he said.',
        '‘Budge,’ he said.',
        '‘Lump,’ he said.',
        '‘Shack,’ he said.',
        '‘Shivered,’ he said.',
        '‘Yeh,’ he said.',
        '‘Yer,’ he said.',
        '‘Ter,’ he said.',
        '‘Summat,’ he said.',
        '‘Fer,’ he said.',
        '‘Meself,’ he said.',
        '‘Mighta,’ he said.',
        '‘Dunno,’ he said.',
        '‘Shan’t,’ he said.',
        '‘Blimey,’ he said.',
        '‘Codswallop,’ he said.',
    )
    stay_read_as_is = (
        '‘Up! Get up! Now!’',
        '‘A what?’',
        '‘Nearly,’ said Harry.',
        '‘Shoo!’ said Mr Dursley loudly.',
        '‘No,’ she said sharply. ‘Why?’',
    )
    demote_to_read_as_is = (
        '‘Nearly,’ said Harry.',
        '‘Shoo!’ said Mr Dursley loudly.',
        '‘No,’ she said sharply. ‘Why?’',
    )
    lines = translate_lines + stay_read_as_is + demote_to_read_as_is
    blocks = tuple(_block(index, text) for index, text in enumerate(lines))
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    translate_until = len(translate_lines)
    stay_until = translate_until + len(stay_read_as_is)
    payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(
                f'b{index + 1}',
                BlockTreatment.READ_AS_IS if translate_until <= index < stay_until else BlockTreatment.TRANSLATE,
            )
            for index in range(len(blocks))
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == (
        [BlockTreatment.TRANSLATE] * translate_until
        + [BlockTreatment.READ_AS_IS] * len(stay_read_as_is)
        + [BlockTreatment.READ_AS_IS] * len(demote_to_read_as_is)
    )

    untouched = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='b1', english_help=False,
    )
    assert [item.treatment for item in untouched.treatment_plan] == [
        item.treatment for item in payload.treatment_plan
    ]


def test_a2_numbered_chapter_heading_forced_to_read_as_is():
    """A numbered heading the model marked TRANSLATE is forced to READ_AS_IS.

    — CHAPTER THREE — and a bare roman numeral lose an unevidenced grammar note.
    Evidence keeps its note. A meaningful title is still translated. The same
    title in a paragraph is left alone. B1 does not rewrite the plan.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        HelpEvidenceResponse, PedagogicalAnalysisResponse, QuotedDifficultyResponse,
        QuotedSourceAnchorResponse, TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    numbered = '— CHAPTER THREE —'
    roman = 'IV'
    title = 'The Letters from No One'
    blocks = (
        _block(0, numbered, type=BlockType.HEADING),
        _block(1, roman, type=BlockType.HEADING),
        _block(2, title, type=BlockType.HEADING),
        _block(3, title),
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    point = (RequiredPoint('p1', 'Объяснить номер главы.'),)

    def grammar(difficulty_id, alias, quote):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse(alias, quote),), alias,
            'grammar', 'useful', 'explanation', point,
        )

    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b2', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b3', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b4', BlockTreatment.READ_AS_IS),
        ),
        difficulties=(
            grammar('d-numbered', 'b1', 'THREE'),
            grammar('d-numbered-evidenced', 'b1', 'CHAPTER'),
            grammar('d-roman', 'b2', 'IV'),
        ),
        before_you_read_evidence=(
            HelpEvidenceResponse(
                'd-numbered-evidenced', 'p1', 'before_you_read.explanation', 'b1', 0,
                'Номер главы уже объяснён.',
            ),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
    ]
    assert [item.id for item in normalized.difficulties] == ['d-numbered-evidenced']

    untouched = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='b1', english_help=False,
    )
    assert [item.treatment for item in untouched.treatment_plan] == [
        item.treatment for item in payload.treatment_plan
    ]
    assert [item.id for item in untouched.difficulties] == [
        'd-numbered', 'd-numbered-evidenced', 'd-roman',
    ]

    from linguary.application.analysis_merge import (
        combine_staged_analysis_responses,
        validate_content_plan,
    )
    from linguary.application.llm_dto import NarrativeAnalysisResponse

    vocab_on_heading = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.GLOSSARY_ONLY),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                'd-ch-vocab',
                (QuotedSourceAnchorResponse('b1', 'CHAPTER'),),
                'b1',
                'vocabulary',
                'useful',
                'lexical_hint',
                point,
            ),
        ),
    )
    heading_blocks = (blocks[0],)
    heading_aliases = {'b1': blocks[0].id}
    normalized_vocab = normalize_a2_dialogue_response(
        vocab_on_heading, heading_blocks, heading_aliases, reader_level='a2', english_help=False,
    )
    assert normalized_vocab.treatment_plan[0].treatment is BlockTreatment.READ_AS_IS
    assert normalized_vocab.difficulties == ()
    assert validate_content_plan(
        combine_staged_analysis_responses(
            NarrativeAnalysisResponse('Keep the voice.', 'Chapter heading.'),
            normalized_vocab,
        ),
        heading_aliases,
        {blocks[0].id: blocks[0].text},
        visible_texts_by_alias={'b1': blocks[0].text},
        blocks=heading_blocks,
    ) is not None


def test_a2_translate_blocks_drop_trivial_ch5_grammar_and_vocabulary_notes_and_prune_forward_speech_style_spoiler():
    """TRANSLATE drops unevidenced Ch5 grammar and vocabulary notes on the closed list.

    could tell stays, including when the same note also anchors cupboard.
    Evidence and a READ_AS_IS block keep the note. A lexical
    hint is not an explanation. speech_style loses a later-block quote of any
    length, including "mighta sat" and «mighta», and dialect lemmas that first
    appear later, and keeps at least one point.
    """
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        HelpEvidenceResponse, PedagogicalAnalysisResponse, QuotedDifficultyResponse,
        QuotedSourceAnchorResponse, TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    blocks = (
        _block(0, 'The hut stood by the sea.'),
        _block(1, '‘Budge up, yeh great lump, and sit down here,’ said the stranger to the boy.'),
        _block(2, '‘I mighta sat on it,’ he said. Got summat fer meself las\' night.'),
        _block(3, 'Harry kept his eyes shut tight in the cupboard under the stairs.'),
        _block(4, 'The small cupboard was dark.'),
        _block(5, 'He waited until morning.'),
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    point = (RequiredPoint('p1', 'Объяснить конструкцию в этом блоке.'),)

    def note(difficulty_id, alias, quote, kind, help_kind='explanation'):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse(alias, quote),), alias,
            kind, 'useful', help_kind, point,
        )

    def speech(difficulty_id, points):
        return QuotedDifficultyResponse(
            difficulty_id, (QuotedSourceAnchorResponse('b2', 'yeh'),), 'b2',
            'speech_style', 'useful', 'explanation', points,
        )

    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b2', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b3', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b4', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b5', BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse('b6', BlockTreatment.TRANSLATE),
        ),
        difficulties=(
            note('d-cupboard', 'b4', 'cupboard', 'vocabulary'),
            note('d-in-cupboard', 'b4', 'in my cupboard', 'grammar'),
            note('d-eyes', 'b4', 'kept his eyes shut tight', 'grammar'),
            note('d-giant', 'b4', 'a giant called Hagrid', 'vocabulary'),
            note('d-called', 'b4', 'called Hagrid', 'grammar'),
            note('d-school', 'b4', 'was going to a school', 'vocabulary'),
            note('d-wizards', 'b4', 'I was going to a school for wizards', 'grammar'),
            note('d-open', 'b4', 'when I open my eyes', 'grammar'),
            note('d-could', 'b4', 'could tell', 'vocabulary'),
            note('d-could-grammar', 'b4', 'could tell', 'grammar'),
            QuotedDifficultyResponse(
                'd-mixed',
                (
                    QuotedSourceAnchorResponse('b4', 'could tell'),
                    QuotedSourceAnchorResponse('b4', 'cupboard'),
                ),
                'b4', 'vocabulary', 'useful', 'explanation', point,
            ),
            note('d-lexical', 'b4', 'cupboard', 'vocabulary', 'lexical_hint'),
            note('d-read', 'b5', 'in my cupboard', 'grammar'),
            note('d-evidenced', 'b6', 'cupboard', 'vocabulary'),
            speech('d-style', (
                RequiredPoint('p-yeh', 'В этой реплике yeh значит you.'),
                RequiredPoint('p-quoted-you', 'В этой реплике «yeh» значит «you».'),
                RequiredPoint('p-quote', 'Позже он скажет "I mighta sat on it" про торт.'),
                RequiredPoint('p-two-words', 'Короткая цитата "mighta sat" из будущего блока.'),
                RequiredPoint('p-one-word', 'Одно слово «mighta» из будущего блока.'),
                RequiredPoint('p-guillemet', 'Он говорит «I mighta sat on it» про торт.'),
                RequiredPoint(
                    'p-lemmas',
                    "summat = something, fer = for, las' = last, meself = myself",
                ),
                RequiredPoint(
                    'p-curly-lemmas',
                    'summat = something, fer = for, las’ = last, meself = myself',
                ),
                RequiredPoint('p-yeh-lemma', 'yeh = you'),
                RequiredPoint('p-mixed', 'yeh = you, summat = something'),
                RequiredPoint('p-codswallop', 'codswallop = nonsense'),
                RequiredPoint('p-current-quote', 'Реплика "yeh great lump" звучит грубо.'),
                RequiredPoint('p-unknown-quote', 'Цитата "flying motorbike cake" нигде не стоит.'),
            )),
            speech('d-style-min', (
                RequiredPoint('p-only-quote', 'Спойлер "I mighta sat on it" про торт.'),
                RequiredPoint('p-only-lemma', 'summat = something, fer = for'),
            )),
            speech('d-style-one', (
                RequiredPoint('p-single', 'Единственный спойлер "I mighta sat on it".'),
            )),
        ),
        before_you_read_evidence=(
            HelpEvidenceResponse(
                'd-evidenced', 'p1', 'before_you_read.explanation', 'b6', 0,
                'Шкаф уже объяснён.',
            ),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
    ]
    kept = {item.id: item for item in normalized.difficulties}
    assert set(kept) == {
        'd-could',
        'd-could-grammar',
        'd-mixed',
        'd-lexical',
        'd-read',
        'd-evidenced',
        'd-style',
        'd-style-min',
        'd-style-one',
    }
    assert [point.id for point in kept['d-style'].required_points] == [
        'p-yeh',
        'p-quoted-you',
        'p-yeh-lemma',
        'p-mixed',
        'p-codswallop',
        'p-current-quote',
        'p-unknown-quote',
    ]
    assert [point.id for point in kept['d-style-min'].required_points] == ['p-only-quote']
    assert [point.id for point in kept['d-style-one'].required_points] == ['p-single']
    assert kept['d-lexical'].help_kind == 'lexical_hint'


def test_combine_staged_analysis_marks_speaker_explicit_when_remark_names_character():
    """‘Why?’ said Harry. turns an inferred speaker into explicit.

    Harry and Harry Potter both match that remark. Mr. Dursley and Mrs. Weasley
    match it too: the title period is not the end of the remark. A name before
    Dr. in the opening remark matches as well. The same name only inside the
    quote, a remark that names nobody, and Harry on a later line stay inferred.
    Without blocks the fact is unchanged. content_plan_from_response applies
    the same rule from the block text.
    """
    from linguary.application.analysis_merge import (
        combine_staged_analysis_responses,
        content_plan_from_response,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse, NarrativeAnalysisResponse, PedagogicalAnalysisResponse,
        QuotedContentPlanResponse, QuotedContextFactResponse, QuotedSourceAnchorResponse,
        TreatmentPlanItemResponse,
    )

    harry = _block(0, '‘Why?’ said Harry.')
    named = _block(1, 'Harry said, ‘Why?’')
    potter = _block(2, '‘Why?’ said Harry.')
    stranger = _block(3, '‘Why?’ said the stranger.')
    inside = _block(4, '‘Harry?’ she said.')
    later = _block(5, '‘No,’ she said sharply. ‘Why?’ said Harry.')
    dursley = _block(6, '‘Why?’ said Mr. Dursley.')
    weasley = _block(7, '‘Why?’ said Mrs. Weasley.')
    titled = _block(8, 'Harry saw Dr. and said, ‘Why?’')
    addressee = _block(9, '‘Look at that,’ she said to Mr. Dursley.')
    dodged = _block(10, '‘Get out!’ Harry dodged past Uncle Vernon.')
    backed = _block(11, '‘Leave me alone.’ Harry backed away.')
    blocks = (
        harry, named, potter, stranger, inside, later,
        dursley, weasley, titled, addressee, dodged, backed,
    )

    def speaker(block, fact_id, value, quote, *, kind='speaker', certainty='inferred'):
        anchor = QuotedSourceAnchorResponse(block.id, quote)
        return QuotedContextFactResponse(fact_id, kind, (anchor,), anchor, value, certainty)

    facts = (
        speaker(harry, 'f-harry', 'Harry', 'Why?'),
        speaker(named, 'f-named', 'Harry', 'Why?'),
        speaker(potter, 'f-potter', 'Harry Potter', 'Why?'),
        speaker(dursley, 'f-dursley', 'Mr. Dursley', 'Why?'),
        speaker(weasley, 'f-weasley', 'Mrs. Weasley', 'Why?'),
        speaker(titled, 'f-titled', 'Harry', 'Why?'),
        speaker(stranger, 'f-stranger', 'Harry', 'Why?'),
        speaker(inside, 'f-inside', 'Harry', 'Harry?'),
        speaker(later, 'f-later', 'Harry', 'Why?'),
        speaker(addressee, 'f-addressee', 'Mrs. Dursley', 'Look at that,'),
        speaker(dodged, 'f-dodged', 'Uncle Vernon', 'Get out!'),
        speaker(backed, 'f-backed', 'Harry', 'Leave me alone.'),
        speaker(harry, 'f-gender', 'male', 'said', kind='gender'),
        speaker(harry, 'f-explicit', 'Harry', 'Why?', certainty='explicit'),
    )
    narrative = NarrativeAnalysisResponse(
        'Keep the voice.', 'Harry asks why.', context_facts=facts,
    )
    pedagogical = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(block.id, BlockTreatment.READ_AS_IS) for block in blocks
        ),
    )
    combined = combine_staged_analysis_responses(
        narrative, pedagogical, resolved_blocks=blocks,
    )
    assert [(fact.id, fact.certainty) for fact in combined.content_plan.context_facts] == [
        ('f-harry', 'explicit'),
        ('f-named', 'explicit'),
        ('f-potter', 'explicit'),
        ('f-dursley', 'explicit'),
        ('f-weasley', 'explicit'),
        ('f-titled', 'explicit'),
        ('f-stranger', 'inferred'),
        ('f-inside', 'inferred'),
        ('f-later', 'inferred'),
        ('f-addressee', 'inferred'),
        ('f-dodged', 'inferred'),
        ('f-backed', 'inferred'),
        ('f-gender', 'inferred'),
        ('f-explicit', 'explicit'),
    ]
    unchanged = combine_staged_analysis_responses(narrative, pedagogical)
    assert [fact.certainty for fact in unchanged.content_plan.context_facts] == [
        'inferred', 'inferred', 'inferred', 'inferred', 'inferred', 'inferred',
        'inferred', 'inferred', 'inferred', 'inferred', 'inferred', 'inferred',
        'inferred', 'explicit',
    ]

    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}

    def quoted_speaker(alias, fact_id, value, quote, *, kind='speaker', certainty='inferred'):
        anchor = QuotedSourceAnchorResponse(alias, quote)
        return QuotedContextFactResponse(fact_id, kind, (anchor,), anchor, value, certainty)

    plan = content_plan_from_response(
        ChapterAnalysisResponse(
            translation_instructions='Keep the voice.',
            chapter_summary='Harry asks why.',
            treatment_plan=tuple(
                TreatmentPlanItemResponse(alias, BlockTreatment.READ_AS_IS) for alias in aliases
            ),
            content_plan=QuotedContentPlanResponse(context_facts=(
                quoted_speaker('b1', 'f-harry', 'Harry', 'Why?'),
                quoted_speaker('b2', 'f-named', 'Harry', 'Why?'),
                quoted_speaker('b3', 'f-potter', 'Harry Potter', 'Why?'),
                quoted_speaker('b7', 'f-dursley', 'Mr. Dursley', 'Why?'),
                quoted_speaker('b8', 'f-weasley', 'Mrs. Weasley', 'Why?'),
                quoted_speaker('b9', 'f-titled', 'Harry', 'Why?'),
                quoted_speaker('b4', 'f-stranger', 'Harry', 'Why?'),
                quoted_speaker('b5', 'f-inside', 'Harry', 'Harry?'),
                quoted_speaker('b6', 'f-later', 'Harry', 'Why?'),
                quoted_speaker('b1', 'f-gender', 'male', 'said', kind='gender'),
            )),
        ),
        aliases,
        {block.id: block.text for block in blocks},
        visible_texts_by_alias={alias: block.text for alias, block in zip(aliases, blocks, strict=True)},
        blocks=blocks,
    )
    assert plan is not None
    grouped = {}
    for fact in plan.context_facts:
        grouped.setdefault(fact.subject_anchor.block_id, set()).add(
            (fact.kind, fact.value, fact.certainty),
        )
    assert grouped == {
        harry.id: {('speaker', 'Harry', 'explicit'), ('gender', 'male', 'inferred')},
        named.id: {('speaker', 'Harry', 'explicit')},
        potter.id: {('speaker', 'Harry Potter', 'explicit')},
        dursley.id: {('speaker', 'Mr. Dursley', 'explicit')},
        weasley.id: {('speaker', 'Mrs. Weasley', 'explicit')},
        titled.id: {('speaker', 'Harry', 'explicit')},
        stranger.id: {('speaker', 'Harry', 'inferred')},
        inside.id: {('speaker', 'Harry', 'inferred')},
        later.id: {('speaker', 'Harry', 'inferred')},
    }


def test_pruned_speech_point_drops_its_evidence_and_the_plan_still_builds():
    """A forward speech point cited by BYR evidence is removed with that evidence.

    The difficulty stays. Evidence for the kept point stays. A pedagogical
    payload and a chapter content plan both still build a content plan.
    """
    from linguary.application.analysis_merge import (
        combine_staged_analysis_responses,
        normalize_a2_dialogue_response,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        BeforeYouReadEntryResponse, ChapterAnalysisResponse, HelpEvidenceResponse,
        NarrativeAnalysisResponse, PedagogicalAnalysisResponse, QuotedContentPlanResponse,
        QuotedDifficultyResponse, QuotedSourceAnchorResponse, TreatmentPlanItemResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    blocks = (
        _block(0, '‘Budge up, yeh great lump,’ said the stranger.'),
        _block(1, '‘I mighta sat on it,’ he said.'),
    )
    aliases = {f'b{index + 1}': block.id for index, block in enumerate(blocks)}
    speech = QuotedDifficultyResponse(
        'd-style', (QuotedSourceAnchorResponse('b1', 'yeh'),), 'b1',
        'speech_style', 'useful', 'explanation', (
            RequiredPoint('p-kept', 'В этой реплике yeh значит you.'),
            RequiredPoint('p-spoiler', 'Позже он скажет "I mighta sat on it".'),
        ),
    )
    explanation = 'Диалект этой реплики уже объяснён.'
    entry = BeforeYouReadEntryResponse('yeh', explanation)
    evidence = (
        HelpEvidenceResponse(
            'd-style', 'p-kept', 'before_you_read.explanation', 'b1', 0, explanation,
        ),
        HelpEvidenceResponse(
            'd-style', 'p-spoiler', 'before_you_read.explanation', 'b1', 0, explanation,
        ),
    )
    treatment = (
        TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),
        TreatmentPlanItemResponse('b2', BlockTreatment.TRANSLATE),
    )

    def assert_plan(payload):
        plan = validate_content_plan(
            payload,
            aliases,
            {block.id: block.text for block in blocks},
            visible_texts_by_alias={
                alias: block.text for alias, block in zip(aliases, blocks, strict=True)
            },
            blocks=blocks,
        )
        assert plan is not None
        assert [point.id for point in plan.difficulties[0].required_points] == ['p-kept']
        assert [item.point_id for item in plan.before_you_read_evidence] == ['p-kept']

    normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=treatment,
            before_you_read=(entry,),
            difficulties=(speech,),
            before_you_read_evidence=evidence,
        ),
        blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [point.id for point in normalized.difficulties[0].required_points] == ['p-kept']
    assert [item.point_id for item in normalized.before_you_read_evidence] == ['p-kept']
    assert_plan(combine_staged_analysis_responses(
        NarrativeAnalysisResponse('Keep the voice.', 'A stranger speaks.'),
        normalized,
    ))

    chapter = normalize_a2_dialogue_response(
        ChapterAnalysisResponse(
            translation_instructions='Keep the voice.',
            chapter_summary='A stranger speaks.',
            treatment_plan=treatment,
            before_you_read=(entry,),
            content_plan=QuotedContentPlanResponse(
                difficulties=(speech,),
                before_you_read_evidence=evidence,
            ),
        ),
        blocks, aliases, reader_level='a2', english_help=False,
    )
    assert [point.id for point in chapter.content_plan.difficulties[0].required_points] == ['p-kept']
    assert [item.point_id for item in chapter.content_plan.before_you_read_evidence] == ['p-kept']
    assert_plan(chapter)



