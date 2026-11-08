from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from linguary.application.recovery import RecoveryError
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY as CURRENT
from linguary.domain import ProcessingTask, TaskType
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.recovery import RecoveryScope
from test_book_work_store import prepare, uid


def owner(uow):
    conditions, old, scope = prepare(uow)
    manifest = {'schema_version': 1, 'active_run_id': 'run-1', 'ordered_blocks': ['block-1'],
        'atomic_intervals': [['block-1']], 'historical_coverage': [],
        'policy': {'max_calls': 12, 'max_scope_calls': 3}}
    old = replace(old, id=uid(), root_key=uid(), manifest=manifest, policy_version=CURRENT)
    scope = RecoveryScope(uid(), old.id, ('block-1',))
    task = ProcessingTask(uid(), 'book-1', TaskType.REGENERATE_BLOCK, 'block-1', operation_id=old.id, policy_version=CURRENT)
    with uow.transaction() as tx:
        tx.work.add_root(old)
        tx.work.add_scope(scope)
        tx.tasks.add(task)
        ctx = tx.work.begin_recovery_attempt(old.id, scope.id, task_id=task.id, run_id='run-1',
            stage='block_regeneration', assigned_model='model', prompt_hash='a'*64, policy_version=CURRENT)
    return old, scope, task, ctx


def successor(old):
    new = replace(old, id=uid(), root_key=uid())
    scope = RecoveryScope(uid(), new.id, ('block-1',))
    task = ProcessingTask(uid(), 'book-1', TaskType.REGENERATE_BLOCK, 'block-1', operation_id=new.id, policy_version=CURRENT)
    return dict(expected_generation=old.generation, successor=new, scopes=(scope,), task=task)


def test_restart_revokes_old_attempt_before_successor_can_send(unit_of_work):
    old, _, _, ctx = owner(unit_of_work)
    args = successor(old)
    with unit_of_work.transaction() as tx:
        tx.work.restart_root(old.id, **args)
        assert tx.work.get_root(old.id).superseded_by == args['successor'].id
        assert tx.work.get_attempt(ctx.attempt_id).state == 'cancelled'
        assert tx.tasks.get(args['task'].id) == args['task']
        assert tx.work.get_binding('task', args['task'].id)['payload_json']['operation_refs'] == [
            {'root_id': args['successor'].id, 'policy_version': CURRENT}]
        with pytest.raises(RecoveryError):
            tx.work.begin_recovery_attempt(old.id, tx.work.get_attempt(ctx.attempt_id).scope_id,
                task_id='late', run_id='run-1', stage='translation_unit', assigned_model='model', prompt_hash='a'*64, policy_version=CURRENT)


def test_concurrent_restarts_have_one_successor(unit_of_work):
    old, *_ = owner(unit_of_work)
    barrier = Barrier(2)
    def restart(_):
        args = successor(old)
        barrier.wait(timeout=5)
        try:
            with unit_of_work.transaction() as tx:
                tx.work.restart_root(old.id, **args)
            return True
        except WorkStateError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(restart, range(2))) == [False, True]


def test_restart_transaction_failure_preserves_old_owner(unit_of_work):
    old, scope, task, ctx = owner(unit_of_work)
    args = successor(old)
    args['task'] = replace(args['task'], id=task.id)
    with pytest.raises(Exception), unit_of_work.transaction() as tx:
        tx.work.restart_root(old.id, **args)
    with unit_of_work.transaction() as tx:
        assert tx.work.get_root(old.id).superseded_by is None
        assert tx.work.get_root(args['successor'].id) is None
        assert tx.work.get_scope(scope.id).status == 'claimed'
        assert tx.work.get_attempt(ctx.attempt_id).state == 'created'


def test_late_validated_response_cannot_commit(unit_of_work):
    old, _, _, ctx = owner(unit_of_work)
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        assert tx.work.owns_recovery_attempt(ctx.attempt_id)
        tx.work.restart_root(old.id, **successor(old))
        assert not tx.work.owns_recovery_attempt(ctx.attempt_id)
        with pytest.raises(WorkStateError):
            tx.work.mark_recovery_accepted(ctx.attempt_id)


def test_mark_recovery_accepted_handles_none_outcome(unit_of_work):
    old, scope, task, ctx = owner(unit_of_work)
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        # Simulate attempt with outcome=None
        from linguary.adapters.db.models import LLMAttemptRow
        from sqlalchemy import update
        tx.work._session.execute(update(LLMAttemptRow).where(LLMAttemptRow.id == ctx.attempt_id).values(outcome_json=None))
        tx.work.mark_recovery_accepted(ctx.attempt_id)
        attempt = tx.work.get_attempt(ctx.attempt_id)
        assert attempt.state == 'accepted'
        assert attempt.outcome['decision'] == 'accepted'


@pytest.mark.parametrize('mismatched_source', [None, 'binding', 'artifact'])
def test_replace_rooted_result_binding_checks_persisted_prompt_hash(unit_of_work, mismatched_source):
    root, _, _, ctx = owner(unit_of_work)
    provenance = {'schema_version': 1, 'root_id': root.id, 'policy_version': CURRENT,
        'prompt_hash': 'a'*64, 'attempt_id': ctx.attempt_id}
    binding_provenance = dict(provenance)
    artifact_provenance = dict(provenance)
    if mismatched_source == 'binding':
        binding_provenance['prompt_hash'] = 'b'*64
    elif mismatched_source == 'artifact':
        artifact_provenance['prompt_hash'] = 'b'*64
    artifact_id = uid()
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        tx.work.archive_artifacts([dict(id=artifact_id, book_id='book-1',
            conditions_id=root.conditions_id, run_id='run-1', attempt_id=ctx.attempt_id,
            kind='block_result', scope_id='block-1',
            body={'schema_version': 1, 'provenance': artifact_provenance})])
        args = dict(book_id='book-1', conditions_id=root.conditions_id,
            attempt_id=ctx.attempt_id, artifact_revision_id=artifact_id,
            expected_attempt_id=None, provenance=binding_provenance)
        if mismatched_source is not None:
            with pytest.raises(WorkStateError, match='происхождение новой версии не совпадает'):
                tx.work.replace_result_binding('run-1:block-1', **args)
        else:
            tx.work.replace_result_binding('run-1:block-1', **args)
    with unit_of_work.transaction() as tx:
        binding = tx.work.get_binding('result', 'run-1:block-1')
        if mismatched_source is not None:
            assert binding is None
        else:
            assert binding['attempt_id'] == ctx.attempt_id
            assert binding['artifact_revision_id'] == artifact_id
            assert binding['payload_json'] == provenance
            assert tx.work.get_artifact(artifact_id)['body']['provenance'] == provenance


def test_policy_mismatch_cannot_claim_or_accept(unit_of_work):
    old, _, _, ctx = owner(unit_of_work)
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        from linguary.adapters.db.models import RecoveryRootRow
        tx.work._session.get(RecoveryRootRow, old.id).policy_version = 'unknown'
        tx.work._session.flush()
        assert not tx.work.owns_recovery_attempt(ctx.attempt_id)


def test_superseded_roots_are_history_not_pending_work(unit_of_work):
    old, *_ = owner(unit_of_work)
    args = successor(old)
    with unit_of_work.transaction() as tx:
        tx.work.restart_root(old.id, **args)
        active = tx.work.list_active_roots('book-1', run_id='run-1')
        assert [r.id for r in active] == [args['successor'].id]
        assert not tx.work.root_coverage_complete(old.id)
        assert old.id in [r.id for r in tx.work.list_roots('book-1')]


def test_restart_one_root_keeps_other_dispatcher_roots_resumable(unit_of_work):
    old, *_ = owner(unit_of_work)
    other = replace(old, id=uid(), root_key=uid(), manifest={**dict(old.manifest), 'ordered_blocks': ['block-2'], 'atomic_intervals': [['block-2']]})
    dispatch = ProcessingTask(uid(), 'book-1', TaskType.PROCESS_TRANSLATION_UNIT, 'section-1', operation_id='dispatch:'+uid())
    with unit_of_work.transaction() as tx:
        tx.work.add_root(other)
        tx.tasks.add(dispatch)
        tx.work.bind('task', dispatch.id, book_id='book-1', conditions_id=old.conditions_id,
            payload={'schema_version': 1, 'operation_refs': [{'root_id': old.id, 'policy_version': CURRENT}, {'root_id': other.id, 'policy_version': CURRENT}]})
        tx.work.restart_root(old.id, **successor(old))
        binding = tx.work.get_binding('task', dispatch.id)
        assert binding['payload_json']['operation_refs'] == [{'root_id': other.id, 'policy_version': CURRENT}]
        assert tx.tasks.get(dispatch.id) == dispatch
        assert tx.work.get_root(other.id) == other


def test_policy_mismatch_cannot_claim(unit_of_work):
    old, _, task, _ = owner(unit_of_work)
    scope = RecoveryScope(uid(), old.id, ('block-2',))
    with unit_of_work.transaction() as tx:
        tx.work.add_scope(scope)
        with pytest.raises(RecoveryError, match='conditions_unavailable'):
            tx.work.begin_recovery_attempt(old.id, scope.id, task_id=task.id, run_id='run-1',
                stage='translation_unit', assigned_model='model', prompt_hash='a'*64,
                policy_version='translation-context-legacy-v1')
        assert tx.work.get_scope(scope.id).status == 'pending'


def test_late_validated_response_cannot_use_accept_scope(unit_of_work):
    from persistence_factories import make_translated_result
    old, scope, _, ctx = owner(unit_of_work)
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        tx.work.restart_root(old.id, **successor(old))
        with pytest.raises(WorkStateError):
            tx.work.accept_scope(scope.id, attempt_id=ctx.attempt_id, expected_generation=1,
                results=[make_translated_result()])
        assert tx.results.list_for_run('run-1') == ()


def test_inactive_run_cannot_accept(unit_of_work):
    from linguary.adapters.db.models import ProcessingRunRow
    old, _, _, ctx = owner(unit_of_work)
    with unit_of_work.transaction() as tx:
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        tx.work._session.get(ProcessingRunRow, 'run-1').status = 'cancelled'
        tx.work._session.flush()
        assert not tx.work.owns_recovery_attempt(ctx.attempt_id)


def test_restart_preserves_accepted_historical_provenance(unit_of_work):
    from persistence_factories import make_translated_result
    old, scope, task, ctx = owner(unit_of_work)
    with unit_of_work.transaction() as tx:
        from linguary.adapters.db.models import RecoveryRootRow
        row = tx.work._session.get(RecoveryRootRow, old.id)
        row.manifest_json = {**row.manifest_json, 'ordered_blocks': ['block-1', 'block-2'],
            'atomic_intervals': [['block-1'], ['block-2']]}
        tx.work._session.flush()
        tx.work.transition_attempt(ctx.attempt_id, 'sent', expected_state='created')
        tx.work.transition_attempt(ctx.attempt_id, 'validated', expected_state='sent')
        tx.work.accept_scope(scope.id, attempt_id=ctx.attempt_id, expected_generation=1, results=[make_translated_result()])
        old = tx.work.get_root(old.id)
        before = tx.work.get_binding('result', 'run-1:block-1')
        assert before['payload_json']['root_id'] == old.id
        assert before['payload_json']['policy_version'] == CURRENT
        artifact = tx.work.get_artifact(before['artifact_revision_id'])
        assert artifact['body']['provenance'] == before['payload_json']
        new = replace(old, id=uid(), root_key=uid(), calls_used=0, status='pending',
            manifest={**dict(old.manifest), 'historical_coverage': ['block-1']})
        next_scope = RecoveryScope(uid(), new.id, ('block-2',))
        next_task = replace(task, id=uid(), operation_id=new.id)
        tx.work.restart_root(old.id, expected_generation=old.generation, successor=new,
            scopes=(next_scope,), task=next_task)
        assert tx.work.get_binding('result', 'run-1:block-1') == before
        assert tx.work.get_attempt(ctx.attempt_id).state == 'accepted'
        assert tx.work.get_scope(scope.id).status == 'accepted'
        assert tx.work.get_root(new.id).manifest['historical_coverage'] == ('block-1',)
