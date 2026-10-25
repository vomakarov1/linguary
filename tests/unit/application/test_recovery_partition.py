from dataclasses import replace
import pytest
from linguary.application.recovery import dependency_intervals, merge_unit_boundaries, local_failure_signature, root_manifest
from linguary.domain.content_plan import ContentPlan, Difficulty, SourceAnchor, RequiredPoint
from linguary.domain.llm_log import Violation


def plan():
    return ContentPlan(difficulties=(Difficulty('d', (SourceAnchor('b1',0,1,'a'), SourceAnchor('b3',0,1,'c')), 'b1', 'grammar', 'critical', 'explanation', (RequiredPoint('p','Explain relation'),)),))


def test_indivisible_cross_block_dependency():
    intervals = dependency_intervals(('b1','b2','b3','b4'), plan())
    assert intervals == (('b1','b2','b3'), ('b4',))
    assert merge_unit_boundaries((('b1','b2'), ('b3','b4')), intervals) == (('b1','b2','b3','b4'),)


def test_global_or_unknown_error_not_split():
    assert local_failure_signature(Violation('quotes','translation_unit','global'), ('b1',)) is None
    assert local_failure_signature(Violation('quotes','translation_unit','unknown'), ('b1',)) is None
    assert local_failure_signature(Violation('quotes','translation_unit','local',('foreign',)), ('b1',)) is None
    assert local_failure_signature(Violation('quotes','translation_unit','local',('b1',)), ('b1',))


def test_original_group_boundary_frozen():
    kwargs = dict(book_id='book', conditions_id='conditions', glossary_sha='g', analysis_sha='a', plan_sha=None,
        ordered_blocks=('b1','b2'), treatments=('TRANSLATE','TRANSLATE'), boundaries=(('b1','b2'),),
        atomic_intervals=(('b1',),('b2',)), max_source_chars=100)
    first = root_manifest(**kwargs)
    assert first['root_key'] == root_manifest(**kwargs)['root_key']
    assert 'unit_id' not in first
    assert first['root_key'] != root_manifest(**{**kwargs,'max_source_chars': 101})['root_key']
