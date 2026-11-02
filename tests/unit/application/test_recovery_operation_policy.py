from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.application.recovery import Recovery, RecoveryError
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY as CURRENT, LEGACY_TRANSLATION_POLICY as LEGACY
from linguary.domain import BlockTreatment, BlockType, GlossarySnapshot, TreatmentPlan, TreatmentPlanItem, CoverageStatus
from linguary.domain.analysis import ChapterAnalysis


def fixture(*, processed=(), atomic=False, task=True):
    roots, scopes, bindings = {}, [], {}
    conditions = SimpleNamespace(id=str(uuid4()), payload={'checks': {'language_notes_per_chapter': 2}})
    class Store:
        def latest_conditions(self, book_id): return conditions
        def add_root(self, root): roots[root.id] = root
        def get_root(self, root_id): return roots.get(root_id)
        def list_scopes(self, root_id): return tuple(s for s in scopes if s.root_id == root_id)
        def add_scope(self, scope): scopes.append(scope)
        def get_binding(self, kind, entity_id): return bindings.get(entity_id)
        def bind_operation_refs(self, task_id, **kwargs):
            bindings[task_id] = {'conditions_id': conditions.id, 'payload_json': {'schema_version': 1, 'operation_refs': list(kwargs['operation_refs'])}}
    store = Store()
    class Uow:
        @contextmanager
        def transaction(self): yield SimpleNamespace(work=store)
    ids = ('b1', 'b2')
    plan = TreatmentPlan('section', tuple(TreatmentPlanItem(b, BlockTreatment.TRANSLATE) for b in ids))
    analysis = ChapterAnalysis(id='analysis', section_id='section', coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=plan, translation_instructions='Translate.', chapter_summary='Scene.')
    blocks = tuple(SimpleNamespace(id=b, text='abcd', parent_id='same-parent' if atomic else None, type=BlockType.PARAGRAPH) for b in ids)
    units = (SimpleNamespace(block_ids=ids, items=tuple(SimpleNamespace(block=b) for b in blocks)),)
    work = SimpleNamespace(document=SimpleNamespace(id='book'), analysis=analysis,
        run=SimpleNamespace(id='run', glossary_snapshot=GlossarySnapshot()), processed_block_ids=set(processed), task_id='task' if task else None)
    recovery = Recovery(Uow())
    recovery._validate_historical = lambda *args: None
    return recovery, work, units, roots, scopes, bindings


def prepare(recovery, work, units, **kwargs):
    return recovery.prepare_roots(work, units, max_source_chars=100, validator=None, **kwargs)


def test_new_root_for_old_book_uses_current_policy():
    recovery, work, units, roots, scopes, bindings = fixture(processed=('b1',))
    root_id, = prepare(recovery, work, units)
    root = roots[root_id]
    assert root.policy_version == CURRENT
    assert root.id in root.root_key and CURRENT in root.root_key
    assert root.manifest['historical_coverage'] == ('b1',)
    assert scopes[0].ordered_blocks == ('b2',)
    assert bindings['task']['payload_json']['operation_refs'] == [{'root_id': root_id, 'policy_version': CURRENT}]


def test_resume_split_and_limit_extension_keep_pinned_policy(monkeypatch):
    recovery, work, units, roots, _, bindings = fixture()
    root_id, = prepare(recovery, work, units)
    old = replace(roots[root_id], policy_version=LEGACY, extra_calls=4, calls_used=2)
    roots[root_id] = old
    bindings['task']['payload_json']['operation_refs'][0]['policy_version'] = LEGACY
    monkeypatch.setattr('linguary.application.recovery.CURRENT_TRANSLATION_POLICY', 'future-unknown')
    assert prepare(recovery, work, units) == (root_id,)
    assert roots[root_id] == old


def test_explicit_restart_same_policy_gets_new_identity():
    recovery, work, units, roots, _, _ = fixture(task=False)
    first, = prepare(recovery, work, units)
    second, = prepare(recovery, work, units)
    assert first != second and roots[first].root_key != roots[second].root_key
    assert roots[first].policy_version == roots[second].policy_version == CURRENT


def test_historical_coverage_retains_original_provenance():
    recovery, work, units, roots, _, _ = fixture(processed=('b1',))
    root_id, = prepare(recovery, work, units)
    assert roots[root_id].manifest['historical_coverage'] == ('b1',)
    assert prepare(recovery, work, units) == (root_id,)


def test_partial_atomic_interval_rejects_restart():
    recovery, work, units, roots, _, _ = fixture(processed=('b1',), atomic=True)
    with pytest.raises(RecoveryError, match='unsafe_partition'):
        prepare(recovery, work, units)
    assert roots == {}


def test_resume_requires_exact_version_and_root():
    recovery, work, units, roots, _, _ = fixture()
    root_id, = prepare(recovery, work, units)
    with pytest.raises(RecoveryError, match='conditions_unavailable'):
        prepare(recovery, work, units, operation_refs=({'root_id': root_id, 'policy_version': LEGACY},))
    with pytest.raises(RecoveryError, match='conditions_unavailable'):
        prepare(recovery, work, units, operation_refs=({'root_id': str(uuid4()), 'policy_version': CURRENT},))


def test_regeneration_root_targets_one_completed_block_without_neighbours():
    recovery, work, units, roots, scopes, _ = fixture(processed=('b1', 'b2'), task=False)
    unit = SimpleNamespace(block_ids=('b1',), items=(units[0].items[0],))
    root_id, = prepare(recovery, work, (unit,), operation_kind='regeneration')
    assert roots[root_id].operation_kind == 'regeneration'
    assert roots[root_id].manifest['historical_coverage'] == ()
    assert scopes[0].ordered_blocks == ('b1',)
