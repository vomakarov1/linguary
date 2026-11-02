from dataclasses import asdict, replace
from uuid import uuid4

import pytest

from linguary.application.book_conditions import payload_digest
from linguary.application.translation_policy import LEGACY_TRANSLATION_POLICY as LEGACY
from linguary.domain import ProcessingTask, TaskType
from linguary.domain.recovery import RecoveryRoot, RecoveryScope
from linguary.application.recovery import RecoveryError, root_manifest
from test_book_conditions import make_app, store_imported


def setup_old(app, *, regeneration=False, missing=False):
    doc = store_imported(app)
    block_id = doc.sections[0].blocks[0].id
    conditions = app.book_conditions.freeze_before_analysis('book-1', 0)
    from linguary.domain import ProcessingRun, ProcessingRunStatus, GlossarySnapshot, LLMSettingsSnapshot
    with app.unit_of_work.transaction() as tx:
        saved = tx.work.get_conditions(conditions)
        run = ProcessingRun('run-legacy', 'book-1', ProcessingRunStatus.ACTIVE, GlossarySnapshot(), LLMSettingsSnapshot('fake', 'model', .2, None))
        tx.runs.add(run)
        op = str(uuid4())
        task = ProcessingTask(str(uuid4()), 'book-1', TaskType.REGENERATE_BLOCK if regeneration else TaskType.PROCESS_TRANSLATION_UNIT,
            block_id if regeneration else doc.sections[0].id, operation_id=op if regeneration else 'dispatch:'+str(uuid4()), policy_version=LEGACY if regeneration else '')
        tx.tasks.add(task)
        tx.work.bind('task', task.id, book_id='book-1', conditions_id=saved.id)
        from linguary.domain import ChapterAnalysis, CoverageStatus, TreatmentPlan, TreatmentPlanItem, BlockTreatment
        analysis = ChapterAnalysis(id='analysis-old', section_id=doc.sections[0].id,
            coverage_status=CoverageStatus.COMPLETE, treatment_plan=TreatmentPlan(doc.sections[0].id,
                (TreatmentPlanItem(block_id, BlockTreatment.TRANSLATE),)),
            translation_instructions='Translate.', chapter_summary='A cat sits.')
        tx.analyses.save('book-1', analysis)
        root = None
        if not missing and not regeneration:
            manifest = root_manifest(book_id='book-1', conditions_id=saved.id, glossary_sha=payload_digest({'entries': []}),
                analysis_sha=payload_digest(asdict(analysis)), plan_sha=None, ordered_blocks=(block_id,), treatments=('TRANSLATE',),
                boundaries=((block_id,),), atomic_intervals=((block_id,),), max_source_chars=100)
            root = RecoveryRoot(op, 'book-1', saved.id, manifest['root_key'],
                {**manifest, 'active_run_id': run.id, 'notes_policy': 'grounded-notes-v1', 'historical_coverage': []})
            tx.work.add_root(root)
            tx.work.add_scope(RecoveryScope(str(uuid4()), root.id, (block_id,)))
    return task, saved, root


def ensure(app, task, conditions):
    from linguary.application.translation_compatibility import ensure_legacy_operation
    with app.unit_of_work.transaction() as tx:
        return ensure_legacy_operation(tx, task=task, conditions=conditions)


async def test_old_root_resumes_legacy_with_verified_snapshot(tmp_path):
    app = make_app(tmp_path)
    try:
        task, conditions, root = setup_old(app)
        refs = ensure(app, task, conditions)
        assert refs == ({'root_id': root.id, 'policy_version': LEGACY},)
        with app.unit_of_work.transaction() as tx:
            actual = tx.work.get_root(root.id)
            assert actual.policy_version == LEGACY
            assert actual.compatibility['prompt_sha256'] == conditions.payload['prompt_sha256']
            assert actual.manifest == root.manifest and actual.root_key == root.root_key
    finally:
        await app.aclose()


async def test_old_regeneration_task_gets_legacy_root_before_resume(tmp_path):
    app = make_app(tmp_path)
    try:
        task, conditions, _ = setup_old(app, regeneration=True)
        refs = ensure(app, task, conditions)
        assert refs == ({'root_id': task.operation_id, 'policy_version': LEGACY},)
        with app.unit_of_work.transaction() as tx:
            root = tx.work.get_root(task.operation_id)
            assert root.operation_kind == 'regeneration' and root.policy_version == LEGACY
            assert tx.work.list_scopes(root.id)[0].ordered_blocks == (task.scope_id,)
    finally:
        await app.aclose()


async def test_missing_root_pending_old_task_never_silently_adopts_v2(tmp_path):
    app = make_app(tmp_path)
    try:
        task, conditions, _ = setup_old(app, missing=True)
        with pytest.raises(RecoveryError, match='conditions_unavailable'):
            ensure(app, task, conditions)
    finally:
        await app.aclose()


@pytest.mark.parametrize('corruption', ['hash', 'versions', 'book', 'binding'])
async def test_unverifiable_legacy_stops_without_call(tmp_path, corruption):
    app = make_app(tmp_path)
    try:
        task, conditions, root = setup_old(app)
        if corruption in {'hash', 'versions'}:
            payload = dict(conditions.payload)
            payload['prompt_sha256' if corruption == 'hash' else 'versions'] = 'bad'
            conditions = replace(conditions, payload=payload)
        elif corruption == 'book':
            conditions = replace(conditions, book_id='other')
        else:
            task = replace(task, id='unknown-task')
        with pytest.raises(RecoveryError, match='conditions_unavailable'):
            ensure(app, task, conditions)
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(root.id).policy_version is None
    finally:
        await app.aclose()


async def test_compatibility_is_idempotent_and_keeps_snapshot_hashes(tmp_path):
    app = make_app(tmp_path)
    try:
        task, conditions, root = setup_old(app)
        assert ensure(app, task, conditions) == ensure(app, task, conditions)
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_conditions(conditions.id) == conditions
            assert tx.work.get_root(root.id).root_key == root.root_key
            assert tx.work.get_root(root.id).manifest == root.manifest
    finally:
        await app.aclose()


@pytest.mark.parametrize('ambiguous', [False, True])
async def test_rootless_regeneration_only_adopts_proven_attempts(tmp_path, ambiguous):
    from linguary.domain.recovery import CallAttempt
    app = make_app(tmp_path)
    try:
        task, conditions, _ = setup_old(app, regeneration=True)
        attempt = CallAttempt(str(uuid4()), 'book-1', state='failed', conditions_id=conditions.id,
            task_id=task.id, run_id='run-legacy', context={'schema_version': 1,
                'conditions_id': conditions.id, 'stage': 'block_regeneration', 'task_id': task.id,
                'run_id': 'run-legacy', 'ordered_block_ids': ['foreign' if ambiguous else task.scope_id],
                'prompt_hash': 'a'*64})
        with app.unit_of_work.transaction() as tx:
            tx.work.add_attempt(attempt)
        if ambiguous:
            with pytest.raises(RecoveryError, match='conditions_unavailable'):
                ensure(app, task, conditions)
        else:
            assert ensure(app, task, conditions) == ensure(app, task, conditions)
        with app.unit_of_work.transaction() as tx:
            actual = tx.work.get_attempt(attempt.id)
            assert actual.root_id == (None if ambiguous else task.operation_id)
            assert actual.context == attempt.context
    finally:
        await app.aclose()
