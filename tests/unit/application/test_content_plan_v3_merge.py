from dataclasses import replace
import pytest
from linguary.domain.content_plan import (
    ContextFact,
    Difficulty,
    HelpEvidence,
    RequiredPoint,
    SourceAnchor,
    ContentPlan,
    artifact_key,
)
from linguary.domain import DomainError
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    ChapterAnalysisResponse,
    ContentPlanResponse,
)
from linguary.application.analysis_merge import content_plan_from_response, FragmentAnalysis, _merged_content_plan, AnalysisMergeError, validate_fact_glossary
from linguary.adapters.db.mapping import _encode_content_plan, _decode_content_plan
from linguary.application.ports.persistence import PersistenceError


def response(value='Mara', certainty='explicit'):
    anchor = SourceAnchor('b1', 0, 4, 'Mara')
    fact = ContextFact('f', 'speaker', (anchor,), anchor, value, certainty)
    return ChapterAnalysisResponse('Сохраняй смысл.', 'Mara said hello.', content_plan=ContentPlanResponse(version=3, profile='content-plan-v3', context_facts=(fact,)))


def test_fragment_alias_and_fact_merge():
    plan = content_plan_from_response(response(), {'b1': 'stable'}, {'stable': 'Mara said hello.'})
    fact, = plan.context_facts
    assert fact.subject_anchor.block_id == 'stable'
    assert fact.anchors[0].block_id == 'stable'
    assert _merged_content_plan((FragmentAnalysis(0, 'one', 'summary', content_plan=plan), FragmentAnalysis(1, 'two', 'summary', content_plan=plan))).context_facts == plan.context_facts


def test_conflicting_speaker_rejected():
    first = content_plan_from_response(response(), {'b1': 'stable'}, {'stable': 'Mara said hello.'})
    second = content_plan_from_response(response('Other'), {'b1': 'stable'}, {'stable': 'Mara said hello.'})
    with pytest.raises((DomainError, AnalysisMergeError)):
        _merged_content_plan((FragmentAnalysis(0, 'one', 'summary', content_plan=first), FragmentAnalysis(1, 'two', 'summary', content_plan=second)))


def test_glossary_wins_over_inference():
    plan = content_plan_from_response(response('Other', 'inferred'), {'b1': 'stable'}, {'stable': 'Mara said hello.'})
    with pytest.raises(AnalysisMergeError):
        validate_fact_glossary(plan, {'Mara': {'speaker': 'Mara'}})


def test_v1_v2_v3_persistence():
    plans = [None, ContentPlan(), content_plan_from_response(response(), {'b1': 'stable'}, {'stable': 'Mara said hello.'})]
    for plan in plans:
        assert _decode_content_plan(_encode_content_plan(plan)) == plan
    assert 'context_facts' not in _encode_content_plan(ContentPlan())


def test_fragment_merge_remaps_evidence_to_retained_duplicate_byr():
    explanation = 'Речь звучит резко.'

    def fragment(order, block_id, expression, point_id):
        difficulty = Difficulty(
            f'd-{order}',
            (SourceAnchor(block_id, 0, len(expression), expression),),
            block_id,
            'idiom',
            'critical',
            'explanation',
            (RequiredPoint(point_id, 'Объяснить значение выражения'),),
        )
        evidence = HelpEvidence(
            difficulty.id,
            point_id,
            'before_you_read.explanation',
            block_id,
            artifact_key(
                'before_you_read.explanation',
                block_id,
                {'expression': expression, 'explanation': explanation},
            ),
            explanation,
        )
        return FragmentAnalysis(
            order,
            '',
            '',
            before_you_read=(BeforeYouReadEntryResponse(expression, explanation),),
            content_plan=ContentPlan(
                version=3,
                profile='content-plan-v3',
                difficulties=(difficulty,),
                before_you_read_evidence=(evidence,),
            ),
        )

    plan = _merged_content_plan(
        (
            fragment(0, 'first', 'made a grab', 'point-first'),
            fragment(1, 'second', 'made a grab at', 'point-second'),
        )
    )

    assert plan is not None
    assert len(plan.before_you_read_evidence) == 2
    for item in plan.before_you_read_evidence:
        assert item.artifact_key == artifact_key(
            'before_you_read.explanation',
            item.block_id,
            {'expression': 'made a grab', 'explanation': explanation},
        )


def test_unknown_plan_version_not_legacy():
    with pytest.raises(PersistenceError):
        _decode_content_plan({'version': 99, 'profile': 'future', 'difficulties': [], 'before_you_read_evidence': []})


def test_v3_does_not_relocate_inexact_anchor():
    with pytest.raises((DomainError, AnalysisMergeError)):
        content_plan_from_response(response(), {'b1': 'stable'}, {'stable': 'x Mara said hello.'})


def test_fragment_merge_keeps_distinct_phrasal_verbs_and_their_evidence():
    def fragment(order, block_id, expression, explanation, point_id):
        difficulty = Difficulty(
            f'd-{order}',
            (SourceAnchor(block_id, 0, len(expression), expression),),
            block_id,
            'idiom',
            'critical',
            'explanation',
            (RequiredPoint(point_id, 'Объяснить значение выражения'),),
        )
        evidence = HelpEvidence(
            difficulty.id,
            point_id,
            'before_you_read.explanation',
            block_id,
            artifact_key(
                'before_you_read.explanation',
                block_id,
                {'expression': expression, 'explanation': explanation},
            ),
            explanation,
        )
        return FragmentAnalysis(
            order,
            '',
            '',
            before_you_read=(BeforeYouReadEntryResponse(expression, explanation),),
            content_plan=ContentPlan(
                version=3,
                profile='content-plan-v3',
                difficulties=(difficulty,),
                before_you_read_evidence=(evidence,),
            ),
        )

    plan = _merged_content_plan(
        (
            fragment(0, 'first', 'take off', 'Снять одежду или взлететь.', 'point-first'),
            fragment(1, 'second', 'take place', 'Происходить, состояться.', 'point-second'),
        )
    )

    assert plan is not None
    assert len(plan.before_you_read_evidence) == 2
    assert {item.body_excerpt for item in plan.before_you_read_evidence} == {
        'Снять одежду или взлететь.',
        'Происходить, состояться.',
    }


def test_content_plan_v3_binds_case_and_apostrophe_normalized_byr_and_note_evidence():
    from linguary.application.analysis_merge import before_you_read_entries, validate_content_plan
    from linguary.application.llm_dto import (
        LanguageNoteResponse,
        TranslationResultResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.application.validators.content_plan import (
        resolve_response_evidence,
        validate_content_coverage,
        validate_note_sources,
    )
    from linguary.domain import BlockProcessingResult, BlockTreatment, LanguageNote

    aliases = {'b1': 'block-1', 'b2': 'block-2'}
    texts = {
        'block-1': 'Don’t make a grab at the bottle.',
        'block-2': 'One’s cheeks burned with shame.',
    }
    d1 = Difficulty(
        'd1',
        (SourceAnchor('b1', 0, 20, 'Don’t make a grab at'),),
        'b1',
        'idiom',
        'critical',
        'explanation',
        (RequiredPoint('p1', 'Объяснить идиому'),),
    )
    d2 = Difficulty(
        'd2',
        (SourceAnchor('b2', 0, 19, 'One’s cheeks burned'),),
        'b2',
        'idiom',
        'critical',
        'explanation',
        (RequiredPoint('p2', 'Объяснить образ стыда'),),
    )
    payload = ChapterAnalysisResponse(
        'Сохраняй смысл.',
        'Герой смутился.',
        treatment_plan=(
            TreatmentPlanItemResponse('b1', BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse('b2', BlockTreatment.READ_AS_IS),
        ),
        before_you_read=(
            BeforeYouReadEntryResponse(
                "don't make a grab at",
                'Не пытайся резко схватить рукой.',
            ),
        ),
        content_plan=ContentPlanResponse(
            version=3,
            profile='content-plan-v3',
            difficulties=(d1, d2),
        ),
    )

    plan = validate_content_plan(payload, aliases, texts)
    assert plan is not None
    assert len(plan.before_you_read_evidence) == 1

    selected_byr = before_you_read_entries(payload.before_you_read, strict=True)
    note_body = 'Щёки горели от стыда и неловкости.'
    results = (
        BlockProcessingResult(
            'block-1',
            BlockTreatment.TRANSLATE,
            translated_text='Не хватай бутылку.',
        ),
        BlockProcessingResult(
            'block-2',
            BlockTreatment.READ_AS_IS,
            language_notes=(
                LanguageNote('n1', 'block-2', "“one's cheeks burned”", note_body),
            ),
        ),
    )
    validate_note_sources(results, texts)

    translation_payload = TranslationResultResponse(
        language_notes=(
            LanguageNoteResponse('block-2', "“one's cheeks burned”", note_body),
        ),
    )
    evidence = resolve_response_evidence(
        plan,
        translation_payload,
        results,
        selected_byr,
        owner_block_ids=('block-1', 'block-2'),
    )
    checked = validate_content_coverage(
        plan,
        results,
        selected_byr,
        evidence,
        owner_block_ids=('block-1', 'block-2'),
        source_texts=texts,
        validators_version=4,
    )
    assert len(checked) == 2


