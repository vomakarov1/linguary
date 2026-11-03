"""Explicitly restart only the unfinished part of a pinned chapter operation."""

import asyncio

import pytest

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.ports.llm import LLMTransportError
from linguary.application.use_cases.restart_translation import (
    RestartTranslationCommand,
    RestartTranslationConflict,
    RestartTranslationError,
)
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.composition import drain_tasks
from linguary.domain import LLMCallStage, TaskStatus
from test_book_conditions import make_app
from test_recovery_commit import analysis_reply, prepare, translation_reply


async def make_partial_root(tmp_path):
    """Split a two-block root, accept its first child, and exhaust the second."""
    fake = FakeLLMClient([
        analysis_reply(2),
        translation_reply(2, bad=True),
        translation_reply(2, bad=True),
        translation_reply(1),
        LLMTransportError("synthetic transport failure"),
        LLMTransportError("synthetic transport failure"),
        LLMTransportError("synthetic transport failure"),
    ])
    app = make_app(tmp_path, fake)
    document, run, task = await prepare(app)
    completed = await drain_tasks(app)
    assert completed[0].status is TaskStatus.FAILED
    with app.unit_of_work.transaction() as transaction:
        root, = transaction.work.list_roots("book-1", run_id=run.id)
        scopes = transaction.work.list_scopes(root.id)
        accepted = tuple(scope for scope in scopes if scope.status == "accepted")
        unfinished = tuple(scope for scope in scopes if scope.status not in {"accepted", "split"})
        assert len(accepted) == len(unfinished) == 1
        assert root.reason == "limit"
    return app, fake, document, run, task, root, accepted[0], unfinished[0]


def restart(app, root):
    return app.restart_translation(RestartTranslationCommand(
        book_id=root.book_id,
        root_id=root.id,
        expected_generation=root.generation,
    ))


async def test_restart_only_unfinished_safe_scope_with_current_policy(tmp_path):
    app, fake, document, run, old_task, root, accepted, unfinished = await make_partial_root(tmp_path)
    try:
        result = restart(app, root)
        with app.unit_of_work.transaction() as transaction:
            old = transaction.work.get_root(root.id)
            successor = transaction.work.get_root(result.successor_root_id)
            new_task = transaction.tasks.get(result.task_id)
            binding = transaction.work.get_binding("task", result.task_id)
            scopes = transaction.work.list_scopes(successor.id)
            assert old.superseded_by == successor.id
            assert old.generation == root.generation + 1
            assert successor.policy_version == CURRENT_TRANSLATION_POLICY
            assert successor.manifest["historical_coverage"] == accepted.ordered_blocks
            assert [scope.ordered_blocks for scope in scopes] == [unfinished.ordered_blocks]
            assert new_task is not None and new_task.status is TaskStatus.PENDING
            assert binding["payload_json"]["operation_refs"] == [
                {"root_id": successor.id, "policy_version": CURRENT_TRANSLATION_POLICY}
            ]
            assert transaction.tasks.get(old_task.id).status is TaskStatus.SUPERSEDED
        assert len(fake.requests) == 7
        assert document.sections[0].blocks
    finally:
        await app.aclose()


async def test_restart_keeps_accepted_text_and_historical_origin(tmp_path):
    app, fake, document, run, _, root, accepted, unfinished = await make_partial_root(tmp_path)
    try:
        accepted_block = accepted.ordered_blocks[0]
        with app.unit_of_work.transaction() as transaction:
            before_result = transaction.results.get(run.id, accepted_block)
            before_binding = transaction.work.get_binding("result", f"{run.id}:{accepted_block}")
            before_artifact = transaction.work.get_artifact(before_binding["artifact_revision_id"])
        result = restart(app, root)
        fake.default = translation_reply(1)
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.SUCCEEDED, completed[-1].last_error
        with app.unit_of_work.transaction() as transaction:
            successor = transaction.work.get_root(result.successor_root_id)
            current_result = transaction.results.get(run.id, accepted_block)
            current_binding = transaction.work.get_binding("result", f"{run.id}:{accepted_block}")
            assert transaction.work.get_artifact(before_binding["artifact_revision_id"]) == before_artifact
            assert successor.manifest["historical_coverage"] == accepted.ordered_blocks
            assert current_result == before_result
            assert current_binding == before_binding
            assert transaction.results.get(run.id, unfinished.ordered_blocks[0]) is not None
        new_requests = [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT][6:]
        assert [request.ordered_block_ids for request in new_requests] == [unfinished.ordered_blocks]
        assert document.sections[0].blocks[0].id == accepted_block
    finally:
        await app.aclose()


async def test_restart_then_crash_resumes_new_operation(tmp_path):
    app, fake, _, run, _, root, _, _ = await make_partial_root(tmp_path)
    result = restart(app, root)
    with app.unit_of_work.transaction() as transaction:
        successor = transaction.work.get_root(result.successor_root_id)
        saved_task = transaction.tasks.get(result.task_id)
        assert saved_task.status is TaskStatus.PENDING
        assert saved_task.operation_id.startswith("dispatch:")
    await app.aclose()

    resumed_fake = FakeLLMClient.always(translation_reply(1))
    resumed = make_app(tmp_path, resumed_fake)
    try:
        with resumed.unit_of_work.transaction() as transaction:
            assert transaction.tasks.get(result.task_id).status is TaskStatus.PENDING
            assert transaction.work.get_root(result.successor_root_id).id == successor.id
        completed = await drain_tasks(resumed)
        assert completed[0].id == result.task_id
        assert completed[0].status is TaskStatus.SUCCEEDED, completed[0].last_error
        with resumed.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(result.successor_root_id).status == "complete"
            assert transaction.runs.active_for_book("book-1").id == run.id
    finally:
        await resumed.aclose()


async def test_concurrent_old_response_cannot_overwrite_successor(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    class PausingClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT and not entered.is_set():
                entered.set()
                await release.wait()
            return await super().generate(request)

    fake = PausingClient([analysis_reply(1), translation_reply(1)])
    app = make_app(tmp_path, fake)
    document, run, task = await prepare(app, blocks=1)
    claimed = await app.task_queue.claim_next_pending()
    assert claimed is not None and claimed.id == task.id
    running = asyncio.create_task(app.processing.run_task(claimed))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with app.unit_of_work.transaction() as transaction:
            old_root, = transaction.work.list_roots("book-1", run_id=run.id)
        result = restart(app, old_root)
        release.set()
        await asyncio.wait_for(running, 5)
        fake.default = translation_reply(1)
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.get(run.id, document.sections[0].blocks[0].id) is None
            assert transaction.tasks.get(task.id).status is TaskStatus.SUPERSEDED
            successor = transaction.work.get_root(result.successor_root_id)
            assert successor.status == "pending"
        completed = await drain_tasks(app)
        assert completed[0].id == result.task_id
        assert completed[0].status is TaskStatus.SUCCEEDED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            current = transaction.results.get(run.id, document.sections[0].blocks[0].id)
            binding = transaction.work.get_binding("result", f"{run.id}:{current.block_id}")
            assert binding["payload_json"]["root_id"] == result.successor_root_id
    finally:
        release.set()
        if not running.done():
            await running
        await app.aclose()


async def test_restart_completed_block_requires_regeneration(tmp_path):
    fake = FakeLLMClient([analysis_reply(1), translation_reply(1)])
    app = make_app(tmp_path, fake)
    document, run, _ = await prepare(app, blocks=1)
    try:
        assert (await drain_tasks(app))[0].status is TaskStatus.SUCCEEDED
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots("book-1", run_id=run.id)
            before_result = transaction.results.get(run.id, document.sections[0].blocks[0].id)
        with pytest.raises(RestartTranslationError, match="повторную генерацию"):
            restart(app, root)
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(root.id).superseded_by is None
            assert transaction.results.get(run.id, document.sections[0].blocks[0].id) == before_result
    finally:
        await app.aclose()


async def test_restart_rejects_foreign_book_root(tmp_path):
    fake = FakeLLMClient([analysis_reply(1), FakeReply.of_payload({"blocks": [], "language_notes": []})])
    app = make_app(tmp_path, fake)
    _, run, _ = await prepare(app, blocks=1)
    try:
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots("book-1", run_id=run.id)
        with pytest.raises(RestartTranslationError, match="не найден"):
            app.restart_translation(RestartTranslationCommand(
                book_id="another-book", root_id=root.id, expected_generation=root.generation,
            ))
    finally:
        await app.aclose()


async def test_restart_rejects_stale_generation(tmp_path):
    app, _, _, _, _, root, _, _ = await make_partial_root(tmp_path)
    try:
        with pytest.raises(RestartTranslationConflict):
            app.restart_translation(RestartTranslationCommand(
                root.book_id, root.id, root.generation + 1,
            ))
    finally:
        await app.aclose()
