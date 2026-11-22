"""Ordinary RetryTask must not clear an unknown outcome or a limit stop."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from linguary.application.ports.persistence import BookRecord
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.application.use_cases.retry_task import RetryTask, RetryTaskCommand, RetryTaskError
from linguary.domain import (
    BookMetadata,
    BookStatus,
    ErrorType,
    GlossarySnapshot,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    TaskError,
    TaskStatus,
    TaskType,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope


def _manifest(run_id, blocks, *, max_calls):
    return {
        'schema_version': 1,
        'active_run_id': run_id,
        'ordered_blocks': list(blocks),
        'atomic_intervals': [[block] for block in blocks],
        'historical_coverage': [],
        'policy': {
            'max_calls': max_calls,
            'max_scope_calls': 3,
            'max_depth': 3,
            'max_leaves': 8,
        },
    }


def _open_book(unit_of_work, book_id):
    run = ProcessingRun(
        str(uuid4()), book_id, ProcessingRunStatus.ACTIVE, GlossarySnapshot(),
        LLMSettingsSnapshot(provider='fake', model='pinned-model'),
    )
    conditions = BookConditions(str(uuid4()), book_id, 1, {'schema_version': 1}, datetime.now(UTC))
    with unit_of_work.transaction() as tx:
        tx.books.add(BookRecord(
            book_id, BookMetadata(title='Synthetic'), 'en', 'ru', BookStatus.PROCESSING,
            '/library/synthetic.epub', datetime.now(UTC),
        ))
        tx.runs.add(run)
        tx.work.add_conditions(conditions)
    return run, conditions


def _root(tx, conditions, run, blocks, *, status, reason, calls_used, max_calls, operation_kind='translation'):
    root = RecoveryRoot(
        str(uuid4()), conditions.book_id, conditions.id, str(uuid4()),
        _manifest(run.id, blocks, max_calls=max_calls), calls_used=calls_used, status=status,
        reason=reason, policy_version=CURRENT_TRANSLATION_POLICY, operation_kind=operation_kind,
    )
    scope = RecoveryScope(str(uuid4()), root.id, tuple(blocks))
    tx.work.add_root(root)
    tx.work.add_scope(scope)
    return root, scope


def _attempt(tx, conditions, root, scope, task, *, state, ordinal=1):
    attempt = CallAttempt(
        str(uuid4()), conditions.book_id, state=state, generation=scope.generation, ordinal=ordinal,
        conditions_id=conditions.id, task_id=task.id, run_id=root.manifest['active_run_id'],
        root_id=root.id, scope_id=scope.id,
    )
    tx.work.add_attempt(attempt)
    return attempt


def _failed_task(tx, conditions, roots, *, task_type):
    if task_type is TaskType.REGENERATE_BLOCK:
        operation_id = roots[0].id
        policy_version = CURRENT_TRANSLATION_POLICY
        scope_id = roots[0].manifest['ordered_blocks'][0]
    else:
        operation_id = 'dispatch:' + str(uuid4())
        policy_version = ''
        scope_id = 'section-1'
    task = ProcessingTask(
        str(uuid4()), conditions.book_id, task_type, scope_id, status=TaskStatus.FAILED, attempts=2,
        last_error=TaskError(ErrorType.TRANSIENT, 'временный сбой обработки', 'detail'),
        operation_id=operation_id, policy_version=policy_version,
    )
    tx.tasks.add(task)
    tx.work.bind(
        'task', task.id, book_id=conditions.book_id, conditions_id=conditions.id,
        payload={'schema_version': 1, 'operation_refs': [
            {'root_id': root.id, 'policy_version': root.policy_version} for root in roots
        ]},
    )
    return task


def _retry(unit_of_work, task):
    return RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))


def test_retry_task_allows_paused_transient_root_with_remaining_budget(unit_of_work):
    run, conditions = _open_book(unit_of_work, 'book-paused')
    with unit_of_work.transaction() as tx:
        root, scope = _root(
            tx, conditions, run, ('block-1',), status='pending', reason='transient_error',
            calls_used=1, max_calls=3,
        )
        task = _failed_task(tx, conditions, (root,), task_type=TaskType.PROCESS_TRANSLATION_UNIT)
        rejected = _attempt(tx, conditions, root, scope, task, state='rejected')
        before = tx.work.get_root(root.id)

    restarted = _retry(unit_of_work, task)

    assert (restarted.status, restarted.attempts, restarted.last_error) == (TaskStatus.PENDING, 2, None)
    with unit_of_work.transaction() as tx:
        stored = tx.work.get_root(root.id)
        assert (stored.status, stored.reason, stored.calls_used, stored.generation) == (
            'pending', 'transient_error', 1, before.generation,
        )
        assert stored == before
        assert tx.work.get_attempt(rejected.id).state == 'rejected'
        assert tx.tasks.get(task.id).status is TaskStatus.PENDING


def test_retry_task_rejects_unknown_outcome_and_limit_stopped_roots(unit_of_work):
    run, conditions = _open_book(unit_of_work, 'book-blocked')
    with unit_of_work.transaction() as tx:
        unknown_root, unknown_scope = _root(
            tx, conditions, run, ('block-unknown',), status='stopped', reason='unknown_outcome',
            calls_used=1, max_calls=3,
        )
        unknown_task = _failed_task(
            tx, conditions, (unknown_root,), task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        )
        unknown_attempt = _attempt(
            tx, conditions, unknown_root, unknown_scope, unknown_task, state='unknown',
        )
        limit_root, limit_scope = _root(
            tx, conditions, run, ('block-limit',), status='stopped', reason='limit',
            calls_used=3, max_calls=3, operation_kind='regeneration',
        )
        limit_task = _failed_task(
            tx, conditions, (limit_root,), task_type=TaskType.REGENERATE_BLOCK,
        )
        _attempt(tx, conditions, limit_root, limit_scope, limit_task, state='failed')
        unknown_before = tx.work.get_root(unknown_root.id)
        limit_before = tx.work.get_root(limit_root.id)

    with pytest.raises(RetryTaskError, match='явный перезапуск'):
        _retry(unit_of_work, unknown_task)
    with pytest.raises(RetryTaskError, match='продление лимита'):
        _retry(unit_of_work, limit_task)

    with unit_of_work.transaction() as tx:
        assert tx.work.get_root(unknown_root.id) == unknown_before
        assert tx.work.get_attempt(unknown_attempt.id).state == 'unknown'
        assert tx.tasks.get(unknown_task.id).status is TaskStatus.FAILED
        assert tx.work.get_root(limit_root.id) == limit_before
        assert tx.tasks.get(limit_task.id).status is TaskStatus.FAILED
    with pytest.raises(RetryTaskError, match='явный перезапуск'):
        RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(
            task_id=unknown_task.id, acknowledge_unknown_outcome=True,
        ))
    with unit_of_work.transaction() as tx:
        assert tx.tasks.get(unknown_task.id).status is TaskStatus.FAILED
        assert tx.work.get_attempt(unknown_attempt.id).state == 'unknown'


def test_chapter_correction_unknown_retry_requires_acknowledgement(unit_of_work):
    from linguary.domain.book_conditions import thaw

    run, conditions = _open_book(unit_of_work, 'book-correction')
    with unit_of_work.transaction() as tx:
        task = ProcessingTask(
            str(uuid4()), conditions.book_id, TaskType.CHAPTER_CORRECTION, 'section-1',
            status=TaskStatus.FAILED, attempts=1,
            last_error=TaskError(ErrorType.TRANSIENT, 'временный сбой обработки', 'detail'),
            operation_id=str(uuid4()), policy_version='chapter-correction-v1',
        )
        tx.tasks.add(task)
        attempt = CallAttempt(
            str(uuid4()), conditions.book_id, state='unknown', task_id=task.id, run_id=run.id,
            conditions_id=conditions.id,
        )
        tx.work.add_attempt(attempt)
        attempt_id = attempt.id

    with pytest.raises(RetryTaskError, match='явный перезапуск'):
        _retry(unit_of_work, task)
    with unit_of_work.transaction() as tx:
        assert tx.tasks.get(task.id).status is TaskStatus.FAILED
        assert tx.work.get_attempt(attempt_id).state == 'unknown'

    restarted = RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(
        task_id=task.id, acknowledge_unknown_outcome=True,
    ))
    assert restarted.status is TaskStatus.PENDING
    with unit_of_work.transaction() as tx:
        stored = tx.work.get_attempt(attempt_id)
        assert stored.state == 'unknown'
        assert thaw(stored.outcome)['acknowledged_duplicate_billing_risk'] is True
        assert tx.tasks.get(task.id).status is TaskStatus.PENDING


def test_retry_task_allows_partial_chapter_when_one_root_retryable(unit_of_work):
    run, conditions = _open_book(unit_of_work, 'book-partial')
    with unit_of_work.transaction() as tx:
        blocked, blocked_scope = _root(
            tx, conditions, run, ('block-unknown',), status='stopped', reason='unknown_outcome',
            calls_used=1, max_calls=3,
        )
        paused, _paused_scope = _root(
            tx, conditions, run, ('block-paused',), status='pending', reason='transient_error',
            calls_used=1, max_calls=3,
        )
        task = _failed_task(tx, conditions, (blocked, paused), task_type=TaskType.PROCESS_TRANSLATION_UNIT)
        unknown_attempt = _attempt(tx, conditions, blocked, blocked_scope, task, state='unknown')
        blocked_before = tx.work.get_root(blocked.id)
        paused_before = tx.work.get_root(paused.id)

    restarted = _retry(unit_of_work, task)

    assert restarted.status is TaskStatus.PENDING
    assert restarted.attempts == 2
    with unit_of_work.transaction() as tx:
        assert tx.work.get_root(blocked.id) == blocked_before
        assert tx.work.get_attempt(unknown_attempt.id).state == 'unknown'
        assert tx.work.get_root(paused.id) == paused_before
        assert tx.tasks.get(task.id).status is TaskStatus.PENDING


def test_retry_task_allows_complete_roots_and_rejects_in_flight_or_exhausted_scopes(unit_of_work):
    from linguary.adapters.db.models import RecoveryScopeRow

    run, conditions = _open_book(unit_of_work, 'book-strict-guards')
    with unit_of_work.transaction() as tx:
        # 1. FAILED task whose bound root is already complete -> retryable
        complete_root, complete_scope = _root(
            tx, conditions, run, ('block-complete',), status='complete', reason=None,
            calls_used=1, max_calls=3,
        )
        complete_task = _failed_task(
            tx, conditions, (complete_root,), task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        )
        accepted_att = _attempt(tx, conditions, complete_root, complete_scope, complete_task, state='accepted')
        scope_row = tx.work._session.get(RecoveryScopeRow, complete_scope.id)
        scope_row.status = 'accepted'
        scope_row.accepted_attempt_id = accepted_att.id

        # 2. pending root with stopped scope -> rejected
        stopped_scope_root, stopped_scope = _root(
            tx, conditions, run, ('block-stopped-scope',), status='pending', reason=None,
            calls_used=1, max_calls=3,
        )
        tx.work._session.get(RecoveryScopeRow, stopped_scope.id).status = 'stopped'
        stopped_scope_task = _failed_task(
            tx, conditions, (stopped_scope_root,), task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        )

        # 3. pending root with in-flight sent attempt -> rejected
        sent_root, sent_scope = _root(
            tx, conditions, run, ('block-sent',), status='running', reason=None,
            calls_used=1, max_calls=3,
        )
        sent_task = _failed_task(
            tx, conditions, (sent_root,), task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        )
        _attempt(tx, conditions, sent_root, sent_scope, sent_task, state='sent')

        # 4. pending root with exhausted max_scope_calls -> rejected
        exhausted_root, exhausted_scope = _root(
            tx, conditions, run, ('block-exhausted',), status='pending', reason=None,
            calls_used=3, max_calls=10,
        )
        exhausted_task = _failed_task(
            tx, conditions, (exhausted_root,), task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        )
        for idx in range(1, 4):
            _attempt(tx, conditions, exhausted_root, exhausted_scope, exhausted_task, state='failed', ordinal=idx)

    assert _retry(unit_of_work, complete_task).status is TaskStatus.PENDING
    with pytest.raises(RetryTaskError):
        _retry(unit_of_work, stopped_scope_task)
    with pytest.raises(RetryTaskError):
        _retry(unit_of_work, sent_task)
    with pytest.raises(RetryTaskError):
        _retry(unit_of_work, exhausted_task)
