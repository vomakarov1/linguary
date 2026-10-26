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


@pytest.mark.parametrize('cross_block', [False, True])
def test_prepare_roots_splits_technical_group_only_between_plan_dependencies(cross_block):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from linguary.application.recovery import Recovery
    from linguary.domain import BlockTreatment, BlockType

    roots, scopes = [], []
    class WorkStore:
        def latest_conditions(self, book_id):
            return SimpleNamespace(id='00000000-0000-0000-0000-000000000001', payload={'checks': {'language_notes_per_chapter': 2}})
        def find_root_by_key(self, key):
            return None
        def add_root(self, root):
            roots.append(root)
        def list_scopes(self, root_id):
            return tuple(s for s in scopes if s.root_id == root_id)
        def add_scope(self, scope):
            scopes.append(scope)
    class Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(work=store)
    store = WorkStore()
    from linguary.domain import GlossarySnapshot
    from linguary.domain.analysis import ChapterAnalysis
    from linguary.domain import TreatmentPlan, TreatmentPlanItem, CoverageStatus
    ids = ('b1', 'b2', 'b3', 'b4')
    treatment = TreatmentPlan('section', tuple(TreatmentPlanItem(b, BlockTreatment.TRANSLATE) for b in ids))
    analysis = ChapterAnalysis(id='analysis', section_id='section', coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=treatment, translation_instructions='Translate.', chapter_summary='Scene.',
        content_plan=plan() if cross_block else None)
    blocks = tuple(SimpleNamespace(id=b, text='abcd', parent_id=None,
        type=BlockType.PARAGRAPH, attributes={'split_group': 'original-paragraph'}) for b in ids)
    units = (SimpleNamespace(block_ids=ids, items=tuple(SimpleNamespace(block=b) for b in blocks)),)
    work = SimpleNamespace(document=SimpleNamespace(id='book'), analysis=analysis,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()), processed_block_ids=set())
    Recovery(Uow()).prepare_roots(work, units, max_source_chars=100, validator=None)
    assert len(roots) == 1
    atoms = tuple(tuple(group) for group in roots[0].manifest['atomic_intervals'])
    assert atoms == ((('b1', 'b2', 'b3'), ('b4',)) if cross_block else tuple((b,) for b in ids))
    assert scopes[0].ordered_blocks == ids
    assert tuple(b.attributes['split_group'] for b in blocks) == ('original-paragraph',) * 4
