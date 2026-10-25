from dataclasses import replace
import pytest
from linguary.domain.content_plan import ContextFact, SourceAnchor, ContentPlan
from linguary.domain import DomainError
from linguary.application.llm_dto import ChapterAnalysisResponse, ContentPlanResponse
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


def test_unknown_plan_version_not_legacy():
    with pytest.raises(PersistenceError):
        _decode_content_plan({'version': 99, 'profile': 'future', 'difficulties': [], 'before_you_read_evidence': []})


def test_v3_does_not_relocate_inexact_anchor():
    with pytest.raises((DomainError, AnalysisMergeError)):
        content_plan_from_response(response(), {'b1': 'stable'}, {'stable': 'x Mara said hello.'})
