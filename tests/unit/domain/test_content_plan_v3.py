from dataclasses import asdict, replace
import pytest
from linguary.domain.content_plan import ContentPlan, ContentEvidence, ContextFact, SourceAnchor
from linguary.domain.errors import DomainError


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
