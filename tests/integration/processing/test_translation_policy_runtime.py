"""Pinned operation identity survives queueing and runtime reconstruction."""

from dataclasses import replace
from hashlib import sha256
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from linguary.application.book_conditions import ConditionsError
from linguary.application.ports.llm import LLMRequest, LLMResponse
from linguary.application.recovery import RecoveryError, root_manifest
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY, LEGACY_TRANSLATION_POLICY
from linguary.composition.book_runtime import BoundTaskQueue, ProvenanceClient
from linguary.domain import (
    GlossarySnapshot, LLMCallStage, LLMSettingsSnapshot, ProcessingRun,
    ProcessingRunStatus, ProcessingTask, TaskIdentity, TaskType,
)
from linguary.domain.recovery import RecoveryRoot, RecoveryScope
from test_book_conditions import make_app, store_imported


def pinned_request(app, policy_version=LEGACY_TRANSLATION_POLICY):
    document = store_imported(app)
    block_id = document.sections[0].blocks[0].id
    with app.unit_of_work.transaction() as transaction:
        conditions = app.book_conditions.create(
            transaction, transaction.books.get('book-1'), 0, continue_legacy=True,
        )
        run = ProcessingRun('run-pinned', 'book-1', ProcessingRunStatus.ACTIVE,
            GlossarySnapshot(), LLMSettingsSnapshot('fake', 'model', .2, None))
        transaction.runs.add(run)
        root_id = str(uuid4())
        manifest = root_manifest(book_id='book-1', conditions_id=conditions.id,
            glossary_sha='a' * 64, analysis_sha='b' * 64, plan_sha=None,
            ordered_blocks=(block_id,), treatments=('TRANSLATE',),
            boundaries=((block_id,),), atomic_intervals=((block_id,),), max_source_chars=100)
        transaction.work.add_root(RecoveryRoot(root_id, 'book-1', conditions.id,
            root_id, {**manifest, 'active_run_id': run.id}, policy_version=policy_version))
        scope = RecoveryScope(str(uuid4()), root_id, (block_id,))
        transaction.work.add_scope(scope)
        task = ProcessingTask(str(uuid4()), 'book-1', TaskType.REGENERATE_BLOCK,
            block_id, operation_id=root_id, policy_version=policy_version)
        transaction.tasks.add(task)
        transaction.work.bind('task', task.id, book_id='book-1', conditions_id=conditions.id)
        context = transaction.work.begin_recovery_attempt(root_id, scope.id,
            task_id=task.id, run_id=run.id, stage='block_regeneration',
            assigned_model=conditions.payload['models']['processing'],
            prompt_hash=sha256(b'Translate.').hexdigest(), policy_version=policy_version)
    request = LLMRequest.for_translation(instructions='Translate.', input_text='The cat sat.').with_logging(
        book_id='book-1', stage=LLMCallStage.BLOCK_REGENERATION, task_id=task.id,
        scope_id=scope.id, run_id=run.id, ordered_block_ids=(block_id,))
    return conditions, replace(request, call_context=context)


@pytest.mark.parametrize('bound', [False, True])
async def test_both_queues_preserve_operation_identity(tmp_path, bound):
    app = make_app(tmp_path)
    try:
        store_imported(app)
        conditions_id = app.book_conditions.freeze_before_analysis('book-1', 0)
        queue = BoundTaskQueue(app.task_queue, app.book_conditions, conditions_id) if bound else app.task_queue
        identity = TaskIdentity('book-1', TaskType.REGENERATE_BLOCK, 'block-1', 'operation-1', CURRENT_TRANSLATION_POLICY)
        identities = (identity, replace(identity, operation_id='operation-2'),
            replace(identity, policy_version=LEGACY_TRANSLATION_POLICY),
            TaskIdentity('book-1', TaskType.PROCESS_TRANSLATION_UNIT, 'chapter-1', 'dispatch:1', ''))
        tasks = [await queue.enqueue(key) for key in identities]
        assert len({task.id for task in tasks}) == len(identities)
        for task, key in zip(tasks, identities, strict=True):
            assert task.identity == key
            assert await queue.enqueue(key) == task
        with app.unit_of_work.transaction() as transaction:
            assert {task.identity for task in transaction.tasks.list_for_book('book-1')} == set(identities)
    finally:
        await app.aclose()


@pytest.mark.parametrize('policy_version', [LEGACY_TRANSLATION_POLICY, CURRENT_TRANSLATION_POLICY])
async def test_runtime_uses_root_policy_with_old_book_templates(tmp_path, policy_version):
    app = make_app(tmp_path)
    try:
        conditions, request = pinned_request(app, policy_version)
        app.book_runtime.settings_for(conditions)
        inner = AsyncMock()
        inner.generate.return_value = LLMResponse(payload=object(), provider='fake', model='model', call_context=request.call_context)
        client = ProvenanceClient(inner, app.book_conditions, conditions.id)
        response = await client.generate(request)
        assert inner.generate.call_args.args[0].call_context.policy_version == policy_version
        assert response.call_context == request.call_context
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_conditions(conditions.id).to_dict() == conditions.to_dict()
            assert transaction.work.get_attempt(request.call_context.attempt_id).state == 'validated'
    finally:
        await app.aclose()


async def test_application_restart_keeps_legacy_execution(tmp_path):
    app = make_app(tmp_path)
    conditions, request = pinned_request(app)
    await app.aclose()
    restarted = make_app(tmp_path)
    try:
        restarted.book_runtime.settings_for(conditions)
        inner = AsyncMock()
        inner.generate.return_value = LLMResponse(payload=object(), provider='fake', model='model', call_context=request.call_context)
        await ProvenanceClient(inner, restarted.book_conditions, conditions.id).generate(request)
        assert inner.generate.call_args.args[0].call_context.policy_version == LEGACY_TRANSLATION_POLICY
        with restarted.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(request.call_context.root_id).policy_version == LEGACY_TRANSLATION_POLICY
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('change', [
    {'policy_version': CURRENT_TRANSLATION_POLICY}, {'policy_version': 'unavailable'},
    {'policy_version': None}, {'root_id': str(uuid4())},
    {'prompt_hash': '0' * 64}, {'ordered_block_ids': ('foreign-block',)},
    {'conditions_id': str(uuid4())}, {'attempt_id': str(uuid4())},
])
async def test_provenance_rejects_policy_or_root_mismatch_before_send(tmp_path, change):
    app = make_app(tmp_path)
    try:
        conditions, request = pinned_request(app)
        request = replace(request, call_context=replace(request.call_context, **change))
        inner = AsyncMock()
        with pytest.raises((ConditionsError, RecoveryError), match='conditions_unavailable'):
            await ProvenanceClient(inner, app.book_conditions, conditions.id).generate(request)
        inner.generate.assert_not_awaited()
    finally:
        await app.aclose()


@pytest.mark.parametrize('ownership_transferred', [False, True])
async def test_provenance_rejects_response_context_mismatch(tmp_path, ownership_transferred):
    app = make_app(tmp_path)
    try:
        conditions, request = pinned_request(app)
        inner = AsyncMock()
        async def mismatched_response(_request):
            if ownership_transferred:
                with app.unit_of_work.transaction() as transaction:
                    transaction.work.invalidate_run_coverage('book-1', request.run_id)
            return LLMResponse(payload=object(), provider='fake', model='model',
                call_context=replace(request.call_context, policy_version=CURRENT_TRANSLATION_POLICY))
        inner.generate.side_effect = mismatched_response
        with pytest.raises((ConditionsError, RecoveryError), match='conditions_unavailable'):
            await ProvenanceClient(inner, app.book_conditions, conditions.id).generate(request)
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_attempt(request.call_context.attempt_id).state != 'validated'
    finally:
        await app.aclose()


async def test_provenance_rejects_revoked_ownership_before_send(tmp_path):
    app = make_app(tmp_path)
    try:
        conditions, request = pinned_request(app)
        with app.unit_of_work.transaction() as transaction:
            transaction.work.invalidate_run_coverage('book-1', request.run_id)
        inner = AsyncMock()
        with pytest.raises(RecoveryError, match='interrupted'):
            await ProvenanceClient(inner, app.book_conditions, conditions.id).generate(request)
        inner.generate.assert_not_awaited()
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_attempt(request.call_context.attempt_id).state == 'created'
    finally:
        await app.aclose()


async def test_valid_response_after_ownership_transfer_is_returned_for_stale_audit(tmp_path, monkeypatch):
    from linguary.adapters.llm import FakeLLMClient
    from test_recovery_commit import analysis_reply, prepare, translation_reply

    class TransferringClient(FakeLLMClient):
        async def generate(self, request):
            response = await super().generate(request)
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                with app.unit_of_work.transaction() as transaction:
                    transaction.work.invalidate_run_coverage('book-1', request.run_id)
                    context = request.call_context
                    transaction.work.begin_recovery_attempt(context.root_id, context.scope_id,
                        task_id=request.task_id, run_id=request.run_id, stage=request.stage.value,
                        assigned_model=context.assigned_model, prompt_hash=context.prompt_hash,
                        policy_version=context.policy_version)
            return response

    app = make_app(tmp_path, TransferringClient([analysis_reply(2), translation_reply(2)]))
    returned_states = []
    original_generate = ProvenanceClient.generate

    async def observe_return(client, request):
        response = await original_generate(client, request)
        if request.stage == LLMCallStage.TRANSLATION_UNIT:
            assert response.call_context == request.call_context
            with app.unit_of_work.transaction() as transaction:
                attempt = transaction.work.get_attempt(response.call_context.attempt_id)
                returned_states.append(attempt.state)
                assert not transaction.work.owns_recovery_attempt(attempt.id)
        return response

    monkeypatch.setattr(ProvenanceClient, 'generate', observe_return)
    try:
        _, run, task = await prepare(app)
        with pytest.raises(RecoveryError, match='interrupted'):
            await app.processing.run_task(task)
        assert returned_states == ['sent']
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.list_for_run(run.id) == ()
            (call,) = [call for call in transaction.llm_calls.list_for_book('book-1')
                       if call.stage == LLMCallStage.TRANSLATION_UNIT]
            assert call.diagnostic_meta['decision'] == 'stale'
            attempt = transaction.work.get_attempt(call.attempt_id)
            assert attempt.state not in {'validated', 'accepted'}
            root = transaction.work.get_root(attempt.root_id)
            assert transaction.work.list_scopes(root.id)[0].accepted_attempt_id is None
    finally:
        await app.aclose()


async def test_provenance_rechecks_root_policy_after_response(tmp_path):
    from linguary.adapters.db.models import RecoveryRootRow
    from sqlalchemy import update

    app = make_app(tmp_path)
    try:
        conditions, request = pinned_request(app)
        async def change_policy(_request):
            with app.unit_of_work.transaction() as transaction:
                transaction.work._session.execute(update(RecoveryRootRow).where(
                    RecoveryRootRow.id == request.call_context.root_id).values(policy_version='unavailable'))
            return LLMResponse(payload=object(), provider='fake', model='model', call_context=request.call_context)
        inner = AsyncMock()
        inner.generate.side_effect = change_policy
        with pytest.raises((ConditionsError, RecoveryError), match='conditions_unavailable'):
            await ProvenanceClient(inner, app.book_conditions, conditions.id).generate(request)
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_attempt(request.call_context.attempt_id).state != 'validated'
    finally:
        await app.aclose()
