from dataclasses import asdict, replace
import pytest
from linguary.domain.content_plan import ContentPlan, ContentEvidence, ContextFact, SourceAnchor, Scene, SceneVisualFact
from linguary.domain.errors import DomainError
from linguary.domain import AnalysisPassKind, AnalysisPassResult, ChapterAnalysis, LLMCallStage, TreatmentPlan


def fact(value='Mara', certainty='explicit'):
    anchor = SourceAnchor('b', 0, 4, 'Mara')
    return ContextFact('speaker', 'speaker', (anchor,), anchor, value, certainty)


def test_v2_hash_unchanged():
    assert ContentPlan().sha256 == 'ac4d7b04db0f1a3cf997957eb108c7c5b9401459aa5bb81ba49c0abb4d8cb67a'
    with pytest.raises(DomainError):
        ContentPlan(context_facts=(fact(),))


def test_v3_context_fact_roundtrip():
    plan = ContentPlan(version=3, profile='content-plan-v3', context_facts=(fact(),))
    assert plan.sha256 != replace(plan, context_facts=(fact('Other'),)).sha256
    assert ContentPlan(version=3, profile='content-plan-v3', context_facts=tuple(replace(item) for item in plan.context_facts)) == plan


def test_unknown_attribution_not_fact():
    assert fact(None, 'unknown').value is None
    with pytest.raises(DomainError):
        fact('Mara', 'unknown')
    with pytest.raises(DomainError):
        fact(None, 'inferred')


def test_invalid_anchor_rejected():
    with pytest.raises(DomainError):
        fact().validate({'b': 'Other'})
    with pytest.raises(DomainError):
        replace(fact(), anchors=(SourceAnchor('foreign', 0, 4, 'Mara'),)).validate({'b': 'Mara'})


def test_structural_success_not_semantic_success():
    plan = ContentPlan(version=3, profile='content-plan-v3')
    assert ContentEvidence(plan.sha256, version=3).semantic_correctness == 'not_evaluated'
    with pytest.raises(DomainError):
        ContentEvidence(plan.sha256, version=3, semantic_correctness='correct')


def test_nonempty_v2_canonical_json_and_hash_unchanged():
    import json
    from linguary.domain.content_plan import Difficulty, RequiredPoint
    from linguary.adapters.db.mapping import _encode_content_plan
    difficulty = Difficulty('d', (SourceAnchor('block', 0, 4, 'echo'),), 'block', 'wordplay', 'critical', 'explanation', (RequiredPoint('p', 'Explain both meanings'),))
    plan = ContentPlan(difficulties=(difficulty,))
    expected = '{"before_you_read_evidence":[],"difficulties":[{"anchors":[{"block_id":"block","end":4,"quote":"echo","start":0}],"help_kind":"explanation","id":"d","kind":"wordplay","owner_block_id":"block","priority":"critical","required_points":[{"description":"Explain both meanings","id":"p"}]}],"profile":"content-plan-v2","version":2}'
    assert json.dumps(_encode_content_plan(plan), ensure_ascii=False, sort_keys=True, separators=(',', ':')) == expected
    assert plan.sha256 == 'bab24d97bb74d6fb69d968fe88b067fb117d0276565e6d970c22805fcc5ad1d1'


def test_unknown_fact_serialization_omits_value():
    from linguary.adapters.db.mapping import _encode_content_plan, _decode_content_plan
    plan = ContentPlan(version=3, profile='content-plan-v3', context_facts=(fact(None, 'unknown'),))
    encoded = _encode_content_plan(plan)
    assert 'value' not in encoded['context_facts'][0]
    assert _decode_content_plan(encoded) == plan



def test_fact_subject_and_evidence_can_use_distinct_exact_anchors():
    subject = SourceAnchor('dialogue', 0, 4, 'Shoo')
    evidence = SourceAnchor('attribution', 0, 4, 'Mara')
    value = ContextFact('speaker', 'speaker', (evidence,), subject, 'Mara', 'explicit')
    value.validate({'dialogue': 'Shoo!', 'attribution': 'Mara said'})


def scene(**changes):
    anchor = SourceAnchor('b1', 0, 10, 'Dark room.')
    return Scene(**{
        'id': 's1', 'order': 0, 'start_block_id': 'b1', 'end_block_id': 'b3',
        'summary': 'Holmes waits for a visitor.', 'location': 'Baker Street',
        'participants': ['Holmes', 'Watson'],
        'visual_facts': [SceneVisualFact(anchor, 'Holmes', 'stands at the window')],
        'visual_assumptions': ['fog outside'], **changes,
    })


def test_scene_and_visual_fact_enforce_invariants_and_canonical_sha():
    value = scene()
    visual_fact, = value.visual_facts
    visual_fact.validate({'b1': 'Dark room.'})
    assert visual_fact.to_dict() == {
        'anchor': {'block_id': 'b1', 'start': 0, 'end': 10, 'quote': 'Dark room.'},
        'subject': 'Holmes', 'detail': 'stands at the window',
    }
    with pytest.raises(DomainError):
        visual_fact.validate({'b1': 'Other room'})
    assert value.participants == ('Holmes', 'Watson')
    assert value.visual_assumptions == ('fog outside',)
    assert isinstance(value.visual_facts, tuple)
    assert value.canonical_id == replace(value, id='generated-again').canonical_id
    assert value.canonical_id != replace(value, summary='Watson waits.').canonical_id
    plan = ContentPlan(version=3, profile='content-plan-v3', scenes=[value])
    assert plan.scenes == (value,)
    assert plan.sha256 != replace(plan, scenes=()).sha256
    assert plan.sha256 != replace(plan, scenes=(replace(value, visual_assumptions=()),)).sha256
    with pytest.raises(DomainError):
        ContentPlan(scenes=(value,))


@pytest.mark.parametrize('changes', [
    {'id': ' '}, {'order': -1}, {'start_block_id': ''},
    {'end_block_id': ' '}, {'summary': ''},
])
def test_scene_rejects_invalid_fields(changes):
    with pytest.raises(DomainError):
        scene(**changes)


@pytest.mark.parametrize('changes', [{'subject': ' '}, {'detail': ''}])
def test_scene_visual_fact_rejects_empty_description(changes):
    with pytest.raises(DomainError):
        SceneVisualFact(**{'anchor': SourceAnchor('b1', 0, 4, 'Mara'),
                           'subject': 'Mara', 'detail': 'wears a coat', **changes})


def test_empty_scene_map_preserves_existing_v3_hashes():
    assert ContentPlan(version=3, profile='content-plan-v3').sha256 == '3162daf4e0f678838765db3539b60622d5d203ccb5250977dd95d68979b752fb'
    assert ContentPlan(version=3, profile='content-plan-v3', context_facts=(fact(),)).sha256 == '0bc4ff32614b9e8545f059d1fb78cecdb30a4e562007097650f8c52d1c0d1a6a'


def test_chapter_analysis_inherits_plan_scenes_unless_explicitly_overridden():
    value = scene()
    kwargs = dict(id='analysis', section_id='chapter',
                  treatment_plan=TreatmentPlan(section_id='chapter', items=()),
                  translation_instructions='Keep the tone.', chapter_summary='Holmes waits.',
                  expected_block_ids=(),
                  content_plan=ContentPlan(version=3, profile='content-plan-v3', scenes=(value,)))
    assert ChapterAnalysis.create(**kwargs).scenes == (value,)
    assert ChapterAnalysis.create(**kwargs, scenes=()).scenes == ()
    assert ChapterAnalysis.create(**{**kwargs, 'content_plan': None}).scenes == ()


def pass_result(**changes):
    return AnalysisPassResult(**{
        'id': 'pass-1', 'book_id': 'book-1', 'section_id': 'sec-1',
        'pass_kind': AnalysisPassKind.NARRATIVE, 'scope_key': '',
        'input_sha256': 'a' * 64, 'output_sha256': 'b' * 64,
        'conditions_id': 'cond-1', 'attempt_id': 'att-1',
        'payload': {'schema_version': 1}, **changes,
    })


def test_analysis_pass_result_validates_hashes_and_pass_kind():
    record = pass_result()
    assert record.pass_kind is AnalysisPassKind.NARRATIVE
    assert pass_result(pass_kind='pedagogical').pass_kind is AnalysisPassKind.PEDAGOGICAL
    assert LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE.value == 'chapter_analysis_narrative'
    assert LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL.value == 'chapter_analysis_pedagogical'


@pytest.mark.parametrize('changes', [
    {'id': ''}, {'book_id': ' '}, {'section_id': ''},
    {'input_sha256': 'a' * 63}, {'input_sha256': 'g' * 64},
    {'output_sha256': 'b' * 65}, {'output_sha256': '!' * 64},
    {'pass_kind': 'unsupported'},
])
def test_analysis_pass_result_rejects_invalid_identifiers_hashes_and_kind(changes):
    with pytest.raises(DomainError):
        pass_result(**changes)
