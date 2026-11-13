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
    summary = 'Holmes opens the door.\n\nХолмс открывает дверь.' if level == 'a2' else None
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
    summary = 'Holmes opens the door.\n\nХолмс открывает дверь.'
    client = FakeLLMClient([narrative(), pedagogical(summary='English only.'), pedagogical(summary=summary)])
    result = await scenario(uow, client, context=builder('a2'), attempts=2)(COMMAND)
    assert [r.stage for r in client.requests] == [NARRATIVE, PEDAGOGICAL, PEDAGOGICAL]
    assert result.analysis.chapter_summary == summary
    assert result.analysis.content_plan.version == 3


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
        summary = 'Holmes opens the door.\n\nХолмс открывает дверь.' if level == 'a2' else None
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
        replies = [narrative(), pedagogical(summary='Holmes opens [b1].\n\nХолмс открывает дверь.'),
                   pedagogical(summary='Holmes opens the door.\n\nХолмс открывает дверь.')]
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
    section = make_analysis_section(texts=tuple('Holmes opens the door. ' + 'x' * 5001 for _ in range(18)))
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

    summary_1 = 'Holmes waits on Baker Street.\n\nХолмс ждёт на Бейкер-стрит.'
    summary_2 = (
        'Holmes waits on Baker Street and Watson arrives.\n\n'
        'Холмс ждёт на Бейкер-стрит, и приезжает Уотсон.'
    )
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


