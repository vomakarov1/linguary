"""Chapter dispatchers retain exact operations and their context rules."""

from dataclasses import replace
from uuid import uuid4

import pytest

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.recovery import RecoveryError
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY, LEGACY_TRANSLATION_POLICY
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.application.use_cases.retry_task import RetryTaskCommand, RetryTaskError
from linguary.composition import drain_tasks
from linguary.domain import BookStatus, LLMCallStage, ProcessingTask, TaskStatus
from linguary.domain.recovery import RecoveryScope
from test_book_conditions import make_app, store_imported
from test_recovery_commit import analysis_reply, prepare, translation_reply
from test_translation_legacy_compatibility import setup_old


async def prepare_chapter(app, texts, *, max_chars):
    document = store_imported(app, (texts,))
    await app.analysis.start('book-1')
    assert all(task.status == TaskStatus.SUCCEEDED for task in await drain_tasks(app))
    run = app.start_processing()(StartProcessingCommand('book-1'))
    conditions = app.book_conditions.require('book-1')
    async with app.book_runtime.lease(conditions.id, 'processing') as generation:
        generation.processing._process_chapter._max_unit_source_chars = max_chars
        task, = await generation.processing.start('book-1')
    return document, run, task


def refs_for(app, task):
    with app.unit_of_work.transaction() as transaction:
        return transaction.work.get_binding('task', task.id)['payload_json']['operation_refs']


async def test_chapter_translation_uses_bounded_source_neighbours(tmp_path):
    texts = tuple((f'Scene {index}. ' + 'He returned home. ' * 160).strip() for index in range(7))
    fake = FakeLLMClient([analysis_reply(7), *(translation_reply(1) for _ in texts)])
    app = make_app(tmp_path, fake)
    try:
        document, run, task = await prepare_chapter(app, texts, max_chars=1)
        assert task.operation_id.startswith('dispatch:') and task.policy_version == ''
        refs = refs_for(app, task)
        assert len(refs) == 7
        assert {ref['policy_version'] for ref in refs} == {CURRENT_TRANSLATION_POLICY}
        assert (await app.processing.start('book-1')) == (task,)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        requests = [r for r in fake.requests if r.stage == LLMCallStage.TRANSLATION_UNIT]
        middle = requests[3]
        assert texts[3] in middle.input_text
        assert texts[2] in middle.input_text and texts[4] in middle.input_text
        assert all(text not in middle.input_text for text in (texts[0], texts[1], texts[5], texts[6]))
        assert middle.ordered_block_ids == (document.sections[0].blocks[3].id,)
        assert refs_for(app, task) == refs
    finally:
        await app.aclose()


async def test_dispatcher_refs_are_committed_before_queue_exposes_task(tmp_path, monkeypatch):
    from linguary.composition.book_runtime import BoundTaskQueue

    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        document = store_imported(app, (('First scene.', 'Second scene.'),))
        await app.analysis.start('book-1')
        assert all(task.status == TaskStatus.SUCCEEDED for task in await drain_tasks(app))
        app.start_processing()(StartProcessingCommand('book-1'))

        enqueue = BoundTaskQueue.enqueue
        claimed_ids = []

        async def claim_immediately(queue, identity):
            task = await enqueue(queue, identity)
            claimed = await queue.queue.claim_next_pending()
            assert claimed is not None and claimed.id == task.id
            with app.unit_of_work.transaction() as transaction:
                binding = transaction.work.get_binding('task', claimed.id)
                assert binding is not None
                refs = binding['payload_json']['operation_refs']
                assert refs
                roots = [transaction.work.get_root(ref['root_id']) for ref in refs]
                assert all(root is not None and root.manifest['ordered_blocks'] for root in roots)
            claimed_ids.append(claimed.id)
            await queue.save(replace(claimed, status=TaskStatus.PENDING))
            return task

        monkeypatch.setattr(BoundTaskQueue, 'enqueue', claim_immediately)
        tasks = await app.processing.start('book-1')
        assert [task.id for task in tasks] == claimed_ids
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            refs = transaction.work.get_binding('task', tasks[0].id)['payload_json']['operation_refs']
            assert {ref['root_id'] for ref in refs}
            assert len(document.sections[0].blocks) == 2
    finally:
        await app.aclose()


async def test_child_request_does_not_append_unbounded_root_in_v2(tmp_path):
    class SplittingClient(FakeLLMClient):
        async def generate(self, request):
            count = len(request.ordered_block_ids)
            self.default = analysis_reply(7) if request.stage == LLMCallStage.CHAPTER_ANALYSIS else translation_reply(count, bad=count > 3)
            return await super().generate(request)

    fake = SplittingClient()
    app = make_app(tmp_path, fake)
    try:
        texts = tuple(f'Unique scene {index}. He returned home.' for index in range(7))
        document, run, task = await prepare_chapter(app, texts, max_chars=12000)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        child = next(r for r in fake.requests if r.stage == LLMCallStage.TRANSLATION_UNIT and len(r.ordered_block_ids) == 3)
        assert 'Контекст исходной сцены' not in child.input_text
        assert texts[3] in child.input_text and texts[4] in child.input_text
        assert texts[5] not in child.input_text and texts[6] not in child.input_text
    finally:
        await app.aclose()


async def test_retry_after_upgrade_resumes_exact_original_roots(tmp_path, monkeypatch):
    fake = FakeLLMClient([analysis_reply(2), FakeReply(content='{'), translation_reply(2)])
    app = make_app(tmp_path, fake)
    settings = replace(app.settings, recovery=replace(app.settings.recovery, max_calls=1))
    app.runtime_settings.publish(app.runtime_settings.prepare(settings), 1)
    try:
        document, run, task = await prepare(app)
        refs = refs_for(app, task)
        assert (await drain_tasks(app))[0].status == TaskStatus.FAILED
        with app.unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(refs[0]['root_id'])
            transaction.work.extend_recovery_limit(root.id, additional_calls=1, expected_generation=root.generation)
        await app.aclose()
        fake = FakeLLMClient([translation_reply(2)])
        app = make_app(tmp_path, fake)
        monkeypatch.setattr('linguary.application.recovery.CURRENT_TRANSLATION_POLICY', 'future-unavailable')
        app.retry_task(RetryTaskCommand(task.id))
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        assert refs_for(app, task) == refs
        with app.unit_of_work.transaction() as transaction:
            roots = transaction.work.list_roots('book-1', run_id=run.id)
            assert [root.id for root in roots] == [ref['root_id'] for ref in refs]
        assert fake.requests[-1].call_context.policy_version == CURRENT_TRANSLATION_POLICY
    finally:
        await app.aclose()


async def test_limit_extension_does_not_change_policy(tmp_path, monkeypatch):
    fake = FakeLLMClient([analysis_reply(2), FakeReply(content='{'), translation_reply(2)])
    app = make_app(tmp_path, fake)
    settings = replace(app.settings, recovery=replace(app.settings.recovery, max_calls=1))
    app.runtime_settings.publish(app.runtime_settings.prepare(settings), 1)
    try:
        document, run, task = await prepare(app)
        assert (await drain_tasks(app))[0].status == TaskStatus.FAILED
        refs = refs_for(app, task)
        with app.unit_of_work.transaction() as transaction:
            before = transaction.work.get_root(refs[0]['root_id'])
            transaction.work.extend_recovery_limit(before.id, additional_calls=1, expected_generation=before.generation)
        monkeypatch.setattr('linguary.application.recovery.CURRENT_TRANSLATION_POLICY', LEGACY_TRANSLATION_POLICY)
        app.retry_task(RetryTaskCommand(task.id))
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
        with app.unit_of_work.transaction() as transaction:
            after = transaction.work.get_root(before.id)
            assert after.policy_version == before.policy_version == CURRENT_TRANSLATION_POLICY
            assert after.calls_used == 2
        assert refs_for(app, task) == refs
    finally:
        await app.aclose()


async def test_superseded_dispatcher_cannot_recreate_old_work(tmp_path):
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        document, run, task = await prepare(app)
        refs = refs_for(app, task)
        with app.unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(refs[0]['root_id'])
            successor = replace(root, id=str(uuid4()), root_key=str(uuid4()))
            next_task = ProcessingTask(str(uuid4()), task.book_id, task.task_type, task.scope_id,
                operation_id='dispatch:' + str(uuid4()))
            transaction.work.restart_root(root.id, expected_generation=root.generation,
                successor=successor, scopes=(RecoveryScope(str(uuid4()), successor.id, tuple(root.manifest['ordered_blocks'])),), task=next_task)
        with pytest.raises(RetryTaskError, match='вытеснен'):
            app.retry_task(RetryTaskCommand(task.id))
        with pytest.raises(RecoveryError, match='conditions_unavailable'):
            await app.processing.run_task(task)
        assert len(fake.requests) == 1
        assert (await app.processing.start('book-1')) == (next_task,)
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
        assert fake.requests[-1].call_context.root_id == successor.id
    finally:
        await app.aclose()


async def test_explicit_root_ids_limit_resume_without_rebinding_dispatcher(tmp_path):
    fake = FakeLLMClient([analysis_reply(3), *(translation_reply(1) for _ in range(3))])
    app = make_app(tmp_path, fake)
    try:
        document, run, task = await prepare_chapter(app, ('First scene.', 'Second scene.', 'Third scene.'), max_chars=1)
        refs = refs_for(app, task)
        conditions = app.book_conditions.require('book-1')
        async with app.book_runtime.lease(conditions.id, 'processing') as generation:
            result = await generation.processing._process_chapter(ProcessChapterCommand(
                'book-1', task.scope_id, task.id, root_ids=(refs[1]['root_id'],)))
        assert result.units == result.processed_units == result.saved_blocks == 1
        assert fake.requests[-1].ordered_block_ids == (document.sections[0].blocks[1].id,)
        assert refs_for(app, task) == refs
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
        assert len(fake.requests) == 4
    finally:
        await app.aclose()


@pytest.mark.parametrize('missing', [False, True])
async def test_historical_chapter_dispatch_verifies_legacy_before_request(tmp_path, missing):
    fake = FakeLLMClient([translation_reply(1)])
    app = make_app(tmp_path, fake)
    try:
        task, conditions, root = setup_old(app, missing=missing)
        with app.unit_of_work.transaction() as transaction:
            transaction.books.save(replace(transaction.books.get('book-1'), status=BookStatus.PROCESSING))
        completed = await drain_tasks(app)
        if missing:
            assert completed[0].status == TaskStatus.FAILED
            assert 'conditions_unavailable' in completed[0].last_error.message
            assert fake.requests == []
        else:
            assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
            assert fake.requests[0].call_context.policy_version == LEGACY_TRANSLATION_POLICY
            with app.unit_of_work.transaction() as transaction:
                assert transaction.work.get_root(root.id).compatibility is not None
    finally:
        await app.aclose()
