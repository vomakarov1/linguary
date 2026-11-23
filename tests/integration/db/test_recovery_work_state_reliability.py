"""Generation fences, honest unknown outcomes, and recovery reconcile."""

from dataclasses import replace

import pytest

from linguary.adapters.db.models import LLMAttemptRow, RecoveryRootRow
from linguary.application.recovery import RecoveryError
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY as CURRENT
from linguary.domain import ErrorType, ProcessingTask, TaskError, TaskStatus, TaskType
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.llm_log import Violation
from linguary.domain.recovery import RecoveryRoot, RecoveryScope
from test_book_work_store import prepare, uid


PROMPT = "a" * 64
BLOCKS = ("block-1", "block-2")
LOCAL = Violation("same-rule", "block_regeneration", scope="local", block_ids=("block-1",))


def _manifest(blocks=BLOCKS, *, max_calls=12, max_scope_calls=6):
    return {
        "schema_version": 1,
        "active_run_id": "run-1",
        "ordered_blocks": list(blocks),
        "atomic_intervals": [[block] for block in blocks],
        "historical_coverage": [],
        "policy": {
            "max_calls": max_calls,
            "max_scope_calls": max_scope_calls,
            "max_depth": 3,
            "max_leaves": 8,
        },
    }


def _plant(tx, conditions, blocks=BLOCKS, *, scope_blocks=None, status="pending", max_calls=12, max_scope_calls=6):
    root = RecoveryRoot(
        uid(), "book-1", conditions.id, uid(),
        _manifest(blocks, max_calls=max_calls, max_scope_calls=max_scope_calls),
        policy_version=CURRENT, status=status,
    )
    scope = RecoveryScope(uid(), root.id, tuple(scope_blocks or blocks))
    task = ProcessingTask(
        uid(), "book-1", TaskType.REGENERATE_BLOCK, scope.ordered_blocks[0],
        operation_id=root.id, policy_version=CURRENT,
    )
    tx.work.add_root(root)
    tx.work.add_scope(scope)
    tx.tasks.add(task)
    return root, scope, task


def _begin(tx, root, scope, task):
    return tx.work.begin_recovery_attempt(
        root.id, scope.id, task_id=task.id, run_id="run-1", stage="block_regeneration",
        assigned_model="model", prompt_hash=PROMPT, policy_version=CURRENT,
    )


def _reject(tx, attempt_id, generation):
    if tx.work.get_attempt(attempt_id).state == "created":
        tx.work.mark_attempt_sent(attempt_id, expected_root_generation=generation)
    tx.work.transition_attempt(attempt_id, "rejected", expected_state="sent")
    return tx.work.record_recovery_failure(attempt_id, LOCAL, outcome="known")


def _dispatcher(tx, conditions, refs, *, failed):
    error = TaskError(ErrorType.TRANSIENT, "чужая ошибка", "detail") if failed else None
    task = ProcessingTask(
        uid(), "book-1", TaskType.PROCESS_TRANSLATION_UNIT, "section-1",
        status=TaskStatus.FAILED if failed else TaskStatus.PENDING,
        attempts=1 if failed else 0, last_error=error, operation_id="dispatch:" + uid(),
    )
    tx.tasks.add(task)
    tx.work.bind(
        "task", task.id, book_id="book-1", conditions_id=conditions.id,
        payload={"schema_version": 1, "operation_refs": list(refs)},
    )
    return task


def _successor(old):
    audit = {
        "schema_version": 1,
        "predecessor_root_id": old.id,
        "observed_generation": old.generation,
        "previous_reason": old.reason,
        "had_unknown_outcome": False,
        "acknowledged_duplicate_billing_risk": True,
        "restarted_at": "2026-11-22T00:00:00+00:00",
    }
    new = replace(
        old, id=uid(), root_key=uid(), calls_used=0, extra_calls=0, generation=0,
        status="pending", reason=None, superseded_by=None, compatibility=audit,
    )
    scope = RecoveryScope(uid(), new.id, tuple(old.manifest["ordered_blocks"]))
    task = ProcessingTask(
        uid(), "book-1", TaskType.REGENERATE_BLOCK, scope.ordered_blocks[0],
        operation_id=new.id, policy_version=CURRENT,
    )
    return audit, dict(expected_generation=old.generation, successor=new, scopes=(scope,), task=task)


def test_stop_root_and_split_scope_reject_stale_generation(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        root, scope, _task = _plant(tx, conditions)
        other, other_scope, other_task = _plant(tx, conditions, scope_blocks=("block-1",))
        generation = root.generation
        with pytest.raises(WorkStateError):
            tx.work.stop_root(root.id, "limit", expected_generation=generation + 1)
        with pytest.raises(WorkStateError):
            tx.work.pause_root(root.id, "transient_error", expected_generation=generation + 1)
        with pytest.raises(WorkStateError):
            tx.work.split_scope(
                scope.id, expected_generation=scope.generation,
                children=(("block-1",), ("block-2",)), expected_root_generation=generation + 1,
            )
        assert tx.work.get_root(root.id).generation == generation
        assert tx.work.get_root(root.id).status == "pending"
        assert tx.work.get_root(root.id).reason is None
        stored = tx.work.get_scope(scope.id)
        assert (stored.status, stored.generation) == ("pending", scope.generation)
        assert len(tx.work.list_scopes(root.id)) == 1

        tx.work.split_scope(
            scope.id, expected_generation=scope.generation,
            children=(("block-1",), ("block-2",)), expected_root_generation=generation,
        )
        parent = tx.work.get_scope(scope.id)
        assert parent.status == "split"
        assert parent.generation == scope.generation + 1
        assert len(tx.work.list_scopes(root.id)) == 3

        tx.work.pause_root(root.id, "transient_error", expected_generation=generation)
        paused = tx.work.get_root(root.id)
        assert (paused.status, paused.reason, paused.generation) == ("pending", "transient_error", generation + 1)
        with pytest.raises(WorkStateError):
            tx.work.stop_root(root.id, "limit", expected_generation=generation)
        tx.work.stop_root(root.id, "limit", expected_generation=paused.generation)
        stopped = tx.work.get_root(root.id)
        assert (stopped.status, stopped.reason, stopped.generation) == ("stopped", "limit", generation + 2)

        ctx = _begin(tx, other, other_scope, other_task)
        calls = tx.work.get_root(other.id).calls_used
        live = tx.work.get_root(other.id).generation
        tx.work.stop_root(other.id, "provider_refusal", expected_generation=live)
        with pytest.raises(RecoveryError):
            tx.work.mark_attempt_sent(ctx.attempt_id, expected_root_generation=live)
        assert tx.work.get_attempt(ctx.attempt_id).state == "created"
        assert tx.work.get_root(other.id).calls_used == calls
        assert tx.work.get_root(other.id).generation == live + 1

        finished = tx.work.get_root(other.id)
        tx.work._session.get(RecoveryRootRow, other.id).status = "complete"
        tx.work._session.flush()
        with pytest.raises(WorkStateError):
            tx.work.stop_root(other.id, "limit", expected_generation=finished.generation)
        assert tx.work.get_root(other.id).status == "complete"


def test_record_recovery_failure_never_downgrades_unknown_to_failed(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        unsent_root, unsent_scope, unsent_task = _plant(tx, conditions, scope_blocks=("block-1",))
        unsent = _begin(tx, unsent_root, unsent_scope, unsent_task)
        unsent_calls = tx.work.get_root(unsent_root.id).calls_used
        unsent_generation = tx.work.get_root(unsent_root.id).generation
        scope_generation = tx.work.get_scope(unsent_scope.id).generation
        assert tx.work.record_recovery_failure(unsent.attempt_id, None, outcome="not_sent") is False
        assert tx.work.get_attempt(unsent.attempt_id).state == "cancelled"
        released = tx.work.get_scope(unsent_scope.id)
        assert released.status == "pending"
        assert released.generation == scope_generation + 1
        assert tx.work.get_root(unsent_root.id).calls_used == unsent_calls
        assert tx.work.get_root(unsent_root.id).generation == unsent_generation

        fresh, fresh_scope, fresh_task = _plant(tx, conditions, scope_blocks=("block-2",))
        sent = _begin(tx, fresh, fresh_scope, fresh_task)
        tx.work.transition_attempt(sent.attempt_id, "sent", expected_state="created")
        before = tx.work.get_root(fresh.id)
        scope_before = tx.work.get_scope(fresh_scope.id)
        assert tx.work.record_recovery_failure(sent.attempt_id, LOCAL, outcome="unknown") is False
        attempt = tx.work.get_attempt(sent.attempt_id)
        assert attempt.state == "unknown"
        scope = tx.work.get_scope(fresh_scope.id)
        assert scope.status == "stopped"
        assert scope.generation == scope_before.generation + 1
        root = tx.work.get_root(fresh.id)
        assert (root.status, root.reason) == ("stopped", "unknown_outcome")
        assert root.generation == before.generation + 1
        assert root.calls_used == before.calls_used

        downgraded, down_scope, down_task = _plant(tx, conditions, blocks=("block-1",), scope_blocks=("block-1",))
        created = _begin(tx, downgraded, down_scope, down_task)
        tx.work.transition_attempt(created.attempt_id, "sent", expected_state="created")
        tx.work.transition_attempt(created.attempt_id, "unknown", expected_state="sent")
        down_before = tx.work.get_root(downgraded.id)
        down_scope_before = tx.work.get_scope(down_scope.id)
        assert tx.work.record_recovery_failure(created.attempt_id, LOCAL, outcome="known") is False
        kept = tx.work.get_attempt(created.attempt_id)
        assert kept.state == "unknown"
        stopped = tx.work.get_scope(down_scope.id)
        assert stopped.status == "stopped"
        assert stopped.generation == down_scope_before.generation + 1
        down_root = tx.work.get_root(downgraded.id)
        assert (down_root.status, down_root.reason) == ("stopped", "unknown_outcome")
        assert down_root.generation == down_before.generation + 1
        assert down_root.calls_used == down_before.calls_used


def test_cancel_unsent_attempt_keeps_unique_ordinal_and_budget(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        root, scope, task = _plant(tx, conditions)
        generation = root.generation
        first = _begin(tx, root, scope, task)
        assert tx.work.get_attempt(first.attempt_id).ordinal == 1
        assert tx.work.get_root(root.id).calls_used == 0
        tx.work.cancel_unsent_attempt(first.attempt_id)
        assert tx.work.get_attempt(first.attempt_id).state == "cancelled"
        assert tx.work.get_attempt(first.attempt_id).ordinal == 1
        assert tx.work.get_scope(scope.id).status == "pending"
        assert tx.work.get_root(root.id).calls_used == 0
        assert tx.work.scope_attempt_count(scope.id) == 0

        second = _begin(tx, root, scope, task)
        second_attempt = tx.work.get_attempt(second.attempt_id)
        assert second_attempt.ordinal == 2
        assert second_attempt.state == "created"
        assert tx.work.get_root(root.id).calls_used == 0
        assert tx.work.scope_attempt_count(scope.id) == 1
        tx.work.mark_attempt_sent(second.attempt_id, expected_root_generation=generation)
        sent = tx.work.get_attempt(second.attempt_id)
        assert sent.state == "sent"
        assert sent.ordinal == 2
        running = tx.work.get_root(root.id)
        assert (running.calls_used, running.status, running.reason) == (1, "running", None)
        assert _reject(tx, second.attempt_id, generation) is False

        sandwiched = _begin(tx, root, scope, task)
        assert tx.work.get_attempt(sandwiched.attempt_id).ordinal == 3
        tx.work.cancel_unsent_attempt(sandwiched.attempt_id)
        assert tx.work.scope_attempt_count(scope.id) == 1
        assert tx.work.get_root(root.id).calls_used == 1

        fourth = _begin(tx, root, scope, task)
        assert tx.work.get_attempt(fourth.attempt_id).ordinal == 4
        assert _reject(tx, fourth.attempt_id, generation) is True
        assert tx.work.scope_attempt_count(scope.id) == 2
        assert tx.work.get_root(root.id).calls_used == 2
        states = {item.ordinal: item.state for item in tx.work.list_root_attempts(root.id)}
        assert states == {1: "cancelled", 2: "rejected", 3: "cancelled", 4: "rejected"}

        local_root, local_scope, local_task = _plant(tx, conditions, scope_blocks=("block-1",))
        local = tx.work.begin_local_acceptance(
            local_root.id, local_scope.id, task_id=local_task.id, run_id="run-1",
            stage="block_regeneration", assigned_model="model", prompt_hash=PROMPT,
            policy_version=CURRENT,
        )
        accepted = tx.work.get_attempt(local.attempt_id)
        assert accepted.state == "validated"
        assert accepted.ordinal == 1
        assert tx.work.get_root(local_root.id).calls_used == 0
        assert tx.work.get_scope(local_scope.id).status == "claimed"
        assert accepted.generation == tx.work.get_scope(local_scope.id).generation
        assert accepted.context["root_generation"] == local_root.generation


def test_restart_root_cleans_up_failed_multiroot_dispatcher_and_persists_audit(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)

    def refs(left, right):
        return [
            {"root_id": left.id, "policy_version": CURRENT},
            {"root_id": right.id, "policy_version": CURRENT},
        ]

    with unit_of_work.transaction() as tx:
        old, scope, task = _plant(tx, conditions, scope_blocks=("block-1",))
        _begin(tx, old, scope, task)
        complete, _, _ = _plant(tx, conditions, scope_blocks=("block-2",), status="complete")
        failed = _dispatcher(tx, conditions, refs(old, complete), failed=True)
        audit, args = _successor(old)
        tx.work.restart_root(old.id, **args)
        stored = tx.tasks.get(failed.id)
        assert stored.status is TaskStatus.SUCCEEDED
        assert stored.last_error is None
        assert tx.work.get_binding("task", failed.id)["payload_json"]["operation_refs"] == [
            {"root_id": complete.id, "policy_version": CURRENT},
        ]
        successor = tx.work.get_root(args["successor"].id)
        assert successor.compatibility == audit
        assert tx.work.get_root(old.id).superseded_by == successor.id

        waiting, waiting_scope, waiting_task = _plant(tx, conditions, scope_blocks=("block-1",))
        _begin(tx, waiting, waiting_scope, waiting_task)
        pending_root, _, _ = _plant(tx, conditions, scope_blocks=("block-2",))
        retryable = _dispatcher(tx, conditions, refs(waiting, pending_root), failed=True)
        _, waiting_args = _successor(waiting)
        tx.work.restart_root(waiting.id, **waiting_args)
        resumed = tx.tasks.get(retryable.id)
        assert resumed.status is TaskStatus.PENDING
        assert resumed.last_error is None
        assert tx.work.get_binding("task", retryable.id)["payload_json"]["operation_refs"] == [
            {"root_id": pending_root.id, "policy_version": CURRENT},
        ]

        kept, kept_scope, kept_task = _plant(tx, conditions, scope_blocks=("block-1",))
        _begin(tx, kept, kept_scope, kept_task)
        done, _, _ = _plant(tx, conditions, scope_blocks=("block-2",), status="complete")
        quiet = _dispatcher(tx, conditions, refs(kept, done), failed=False)
        _, kept_args = _successor(kept)
        tx.work.restart_root(kept.id, **kept_args)
        assert tx.tasks.get(quiet.id) == quiet
        assert tx.work.get_binding("task", quiet.id)["payload_json"]["operation_refs"] == [
            {"root_id": done.id, "policy_version": CURRENT},
        ]


def test_reconcile_recovery_attempts_recovers_claimed_scope_with_rejected_attempt(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        root, left, task = _plant(tx, conditions, scope_blocks=("block-1",))
        right = RecoveryScope(uid(), root.id, ("block-2",))
        tx.work.add_scope(right)
        claimed_generation = {}
        for scope in (left, right):
            ctx = _begin(tx, root, scope, task)
            tx.work.transition_attempt(ctx.attempt_id, "sent", expected_state="created")
            tx.work.transition_attempt(ctx.attempt_id, "rejected", expected_state="sent")
            claimed_generation[scope.id] = tx.work.get_scope(scope.id).generation
        generation = tx.work.get_root(root.id).generation
        tx.work.reconcile_recovery_attempts()
        assert tx.work.get_root(root.id).generation == generation + 1
        assert tx.work.get_root(root.id).status != "stopped"
        for scope in (left, right):
            stored = tx.work.get_scope(scope.id)
            assert stored.status == "pending"
            assert stored.generation == claimed_generation[scope.id] + 1
        resumed = _begin(tx, root, left, task)
        assert tx.work.get_attempt(resumed.attempt_id).ordinal == 3

        sent_root, sent_left, sent_task = _plant(tx, conditions, scope_blocks=("block-1",))
        sent_right = RecoveryScope(uid(), sent_root.id, ("block-2",))
        tx.work.add_scope(sent_right)
        sent_scope_generation = {}
        for scope in (sent_left, sent_right):
            ctx = _begin(tx, sent_root, scope, sent_task)
            tx.work.transition_attempt(ctx.attempt_id, "sent", expected_state="created")
            sent_scope_generation[scope.id] = tx.work.get_scope(scope.id).generation
        sent_generation = tx.work.get_root(sent_root.id).generation
        tx.work.reconcile_recovery_attempts()
        recovered = tx.work.get_root(sent_root.id)
        assert recovered.generation == sent_generation + 1
        assert (recovered.status, recovered.reason) == ("stopped", "interrupted")
        for scope in (sent_left, sent_right):
            stored = tx.work.get_scope(scope.id)
            assert stored.status == "stopped"
            assert stored.generation == sent_scope_generation[scope.id] + 1
        attempts = tx.work.list_root_attempts(sent_root.id)
        assert {item.state for item in attempts} == {"unknown"}
        assert {item.outcome["decision"] for item in attempts} == {"interrupted"}

        stale_root, stale_scope, stale_task = _plant(tx, conditions, scope_blocks=("block-1",))
        stale = _begin(tx, stale_root, stale_scope, stale_task)
        row = tx.work._session.get(LLMAttemptRow, stale.attempt_id)
        row.state = "unknown"
        row.outcome_json = {"schema_version": 1, "decision": "stale", "marker": "keep"}
        context = dict(row.context_json)
        context["root_generation"] = 99
        row.context_json = context
        from linguary.adapters.db.models import RecoveryScopeRow
        tx.work._session.get(RecoveryScopeRow, stale_scope.id).status = "stopped"
        tx.work._session.flush()
        stale_generation = tx.work.get_root(stale_root.id).generation
        stale_status = tx.work.get_root(stale_root.id).status
        tx.work.reconcile_recovery_attempts()
        kept = tx.work.get_attempt(stale.attempt_id)
        assert kept.state == "unknown"
        assert kept.outcome["decision"] == "stale"
        assert kept.outcome["marker"] == "keep"
        assert tx.work.get_root(stale_root.id).generation == stale_generation
        assert tx.work.get_root(stale_root.id).status == stale_status


def test_reconcile_recovery_attempts_handles_claimed_edge_cases_and_correction_reconcile(unit_of_work):
    from linguary.adapters.db.models import RecoveryRootRow
    from linguary.domain import ProcessingTask, TaskStatus, TaskType
    from linguary.domain.recovery import CallAttempt

    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        # 1. claimed + unknown (interrupted) -> scope=stopped, root=('stopped', 'interrupted')
        u_root, u_scope, u_task = _plant(tx, conditions, scope_blocks=("block-1",))
        u_ctx = _begin(tx, u_root, u_scope, u_task)
        u_row = tx.work._session.get(LLMAttemptRow, u_ctx.attempt_id)
        u_row.state = "unknown"
        u_row.outcome_json = {"schema_version": 1, "decision": "interrupted"}
        tx.work._session.flush()

        # 2. claimed + stale created -> created->cancelled, scope->pending
        sc_root, sc_scope, sc_task = _plant(tx, conditions, scope_blocks=("block-1",))
        sc_ctx = _begin(tx, sc_root, sc_scope, sc_task)
        sc_row = tx.work._session.get(LLMAttemptRow, sc_ctx.attempt_id)
        sc_row.context_json = {**dict(sc_row.context_json), "root_generation": 999}
        tx.work._session.flush()

        # 3. claimed with no attempts (latest is None) -> scope->pending
        _empty_root, empty_scope, _ = _plant(tx, conditions, scope_blocks=("block-1",))
        tx.work.claim_scope(empty_scope.id, expected_generation=empty_scope.generation)

        # 4. claimed on stopped root and superseded root -> scope->stopped without mutating root
        stopped_root, stopped_scope, _ = _plant(tx, conditions, scope_blocks=("block-1",))
        tx.work.claim_scope(stopped_scope.id, expected_generation=stopped_scope.generation)
        tx.work.stop_root(stopped_root.id, "limit")
        stopped_gen = tx.work.get_root(stopped_root.id).generation

        sup_root, sup_scope, _ = _plant(tx, conditions, scope_blocks=("block-1",))
        tx.work.claim_scope(sup_scope.id, expected_generation=sup_scope.generation)
        tx.work._session.get(RecoveryRootRow, sup_root.id).superseded_by = stopped_root.id
        tx.work._session.flush()
        sup_gen = tx.work.get_root(sup_root.id).generation

        tx.work.reconcile_recovery_attempts()

        assert tx.work.get_scope(u_scope.id).status == "stopped"
        assert (tx.work.get_root(u_root.id).status, tx.work.get_root(u_root.id).reason) == ("stopped", "interrupted")

        assert tx.work.get_attempt(sc_ctx.attempt_id).state == "cancelled"
        assert tx.work.get_scope(sc_scope.id).status == "pending"

        assert tx.work.get_scope(empty_scope.id).status == "pending"

        assert tx.work.get_scope(stopped_scope.id).status == "stopped"
        assert tx.work.get_root(stopped_root.id).generation == stopped_gen
        assert tx.work.get_root(stopped_root.id).reason == "limit"

        assert tx.work.get_scope(sup_scope.id).status == "stopped"
        assert tx.work.get_root(sup_root.id).generation == sup_gen
        assert tx.work.get_root(sup_root.id).superseded_by == stopped_root.id

        # 5. reconcile_correction_attempts settles correction attempts and skips chapter_analysis attempts
        corr_task = ProcessingTask(
            id="corr-task-1", book_id="book-1", task_type=TaskType.CHAPTER_CORRECTION,
            scope_id="section-1", status=TaskStatus.RUNNING,
        )
        analysis_task = ProcessingTask(
            id="analysis-task-1", book_id="book-1", task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-1", status=TaskStatus.RUNNING,
        )
        tx.tasks.add(corr_task)
        tx.tasks.add(analysis_task)
        corr_sent = CallAttempt(
            uid(), "book-1", state="sent", conditions_id=conditions.id, task_id=corr_task.id,
            context={"schema_version": 1, "stage": "chapter_correction"},
        )
        corr_created = CallAttempt(
            uid(), "book-1", state="created", conditions_id=conditions.id, task_id=corr_task.id,
            context={"schema_version": 1, "stage": "chapter_correction_review"},
        )
        corr_validated = CallAttempt(
            uid(), "book-1", state="validated", conditions_id=conditions.id, task_id=corr_task.id,
            context={"schema_version": 1},
        )
        analysis_sent = CallAttempt(
            uid(), "book-1", state="sent", conditions_id=conditions.id, task_id=analysis_task.id,
            context={"schema_version": 1, "stage": "chapter_analysis"},
        )
        for att in (corr_sent, corr_created, corr_validated, analysis_sent):
            tx.work.add_attempt(att)

        updated = tx.work.reconcile_correction_attempts()
        assert updated == 3
        assert tx.work.get_attempt(corr_sent.id).state == "unknown"
        assert tx.work.get_attempt(corr_sent.id).outcome["decision"] == "interrupted"
        assert tx.work.get_attempt(corr_created.id).state == "cancelled"
        assert tx.work.get_attempt(corr_validated.id).state == "rejected"
        assert tx.work.get_attempt(analysis_sent.id).state == "sent"


def test_rollback_not_sent_attempt_releases_only_a_matching_charge(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        root, scope, task = _plant(tx, conditions, scope_blocks=("block-1",))
        charged = _begin(tx, root, scope, task)
        generation = tx.work.get_root(root.id).generation
        scope_generation = tx.work.get_scope(scope.id).generation
        tx.work.mark_attempt_sent(charged.attempt_id, expected_root_generation=generation)
        assert tx.work.get_root(root.id).calls_used == 1
        tx.work.rollback_not_sent_attempt(charged.attempt_id)
        released = tx.work.get_scope(scope.id)
        stored = tx.work.get_root(root.id)
        assert tx.work.get_attempt(charged.attempt_id).state == "cancelled"
        assert stored.calls_used == 0
        assert stored.generation == generation
        assert (released.status, released.generation) == ("pending", scope_generation + 1)
        tx.work.reconcile_recovery_attempts()
        assert tx.work.get_attempt(charged.attempt_id).state == "cancelled"
        reconciled = tx.work.get_root(root.id)
        assert reconciled.reason != "unknown_outcome"
        assert reconciled.calls_used == 0

        mismatched, mismatched_scope, mismatched_task = _plant(tx, conditions, scope_blocks=("block-2",))
        sent = _begin(tx, mismatched, mismatched_scope, mismatched_task)
        live = tx.work.get_root(mismatched.id).generation
        tx.work.mark_attempt_sent(sent.attempt_id, expected_root_generation=live)
        tx.work.pause_root(mismatched.id, "transient_error", expected_generation=live)
        charged_calls = tx.work.get_root(mismatched.id).calls_used
        tx.work.rollback_not_sent_attempt(sent.attempt_id)
        assert tx.work.get_attempt(sent.attempt_id).state == "cancelled"
        assert tx.work.get_root(mismatched.id).calls_used == charged_calls == 1


def test_begin_recovery_attempt_labels_scope_exhaustion_as_scope_limit(unit_of_work):
    conditions, _, _ = prepare(unit_of_work)
    with unit_of_work.transaction() as tx:
        root, scope, task = _plant(
            tx, conditions, scope_blocks=("block-1",), max_scope_calls=1, max_calls=12,
        )
        ctx = _begin(tx, root, scope, task)
        generation = tx.work.get_root(root.id).generation
        tx.work.mark_attempt_sent(ctx.attempt_id, expected_root_generation=generation)
        tx.work.record_recovery_failure(ctx.attempt_id, LOCAL, outcome="known")
        assert _begin(tx, tx.work.get_root(root.id), tx.work.get_scope(scope.id), task) is None
        stopped = tx.work.get_root(root.id)
        assert (stopped.status, stopped.reason) == ("stopped", "scope_limit")

        limited, limited_scope, limited_task = _plant(
            tx, conditions, scope_blocks=("block-2",), max_scope_calls=6, max_calls=1,
        )
        call = _begin(tx, limited, limited_scope, limited_task)
        call_generation = tx.work.get_root(limited.id).generation
        tx.work.mark_attempt_sent(call.attempt_id, expected_root_generation=call_generation)
        tx.work.record_recovery_failure(call.attempt_id, LOCAL, outcome="known")
        assert _begin(tx, tx.work.get_root(limited.id), tx.work.get_scope(limited_scope.id), limited_task) is None
        assert tx.work.get_root(limited.id).reason == "limit"
