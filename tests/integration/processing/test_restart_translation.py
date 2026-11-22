"""Explicitly restart only the unfinished part of a pinned chapter operation."""

import asyncio
from contextlib import contextmanager

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.adapters.db.work_state import SqlAlchemyWorkRepository
from linguary.adapters.llm import FakeReply
from tests.correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from linguary.application.block_processing import TranslationCall
from linguary.application.ports.llm import LLMTransportError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.restart_translation import (
    RestartTranslationCommand,
    RestartTranslationConflict,
    RestartTranslationError,
)
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.composition import build_application, drain_tasks
from linguary.domain import LLMCallStage, TaskStatus
from test_book_conditions import make_app, store_imported
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
        assert root.reason == "scope_limit"
    return app, fake, document, run, task, root, accepted[0], unfinished[0]


def restart(app, root, *, acknowledge=False):
    return app.restart_translation(RestartTranslationCommand(
        book_id=root.book_id,
        root_id=root.id,
        expected_generation=root.generation,
        acknowledge_unknown_outcome=acknowledge,
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
    running = asyncio.create_task(app.runner.run_next())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with app.unit_of_work.transaction() as transaction:
            old_root, = transaction.work.list_roots("book-1", run_id=run.id)
        result = restart(app, old_root, acknowledge=True)
        release.set()
        old_completed = await asyncio.wait_for(running, 5)
        assert old_completed is not None
        assert old_completed.status is TaskStatus.SUPERSEDED
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


async def test_restart_one_root_keeps_other_dispatch_roots_running(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    class PausingClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                if not entered.is_set():
                    entered.set()
                    await release.wait()
                steps = list(self.replies)
                index = len(self.requests)
                while len(steps) <= index:
                    steps.append(FakeReply.of_payload({"blocks": [], "language_notes": []}))
                steps[index] = FakeReply.of_payload({
                    "blocks": [
                        {
                            "source_block_id": f"b{block_index}",
                            "treatment": "TRANSLATE",
                            "translated_text": "Кот был здесь.",
                            "glossary_hints": [],
                        }
                        for block_index, _ in enumerate(request.ordered_block_ids, start=1)
                    ],
                    "language_notes": [],
                })
                self.replies = tuple(steps)
            return await super().generate(request)

    fake = PausingClient([analysis_reply(2)])
    app = build_application(settings=AppSettings(
        llm=LLMSettings(
            model="pinned-model",
            api_key="synthetic-secret",
            proxy_mode="direct",
            max_unit_source_chars=15,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    ), llm_client=fake, content_plan_profile="legacy-v1")
    try:
        document = store_imported(app, (("The cat sat.", "The dog ran."),))
        await app.analysis.start("book-1")
        analysis = await drain_tasks(app)
        assert analysis[0].status is TaskStatus.SUCCEEDED
        run = app.start_processing()(StartProcessingCommand("book-1"))
        dispatchers = await app.processing.start("book-1")
        dispatcher = dispatchers[0]
        with app.unit_of_work.transaction() as transaction:
            binding = transaction.work.get_binding("task", dispatcher.id)
            refs = binding["payload_json"]["operation_refs"]
            assert len(refs) == 2
            restarted_root = transaction.work.get_root(refs[0]["root_id"])
            retained_root = transaction.work.get_root(refs[1]["root_id"])

        running = asyncio.create_task(app.runner.run_next())
        await asyncio.wait_for(entered.wait(), 5)
        successor = restart(app, restarted_root, acknowledge=True)
        release.set()
        completed_dispatcher = await asyncio.wait_for(running, 5)
        assert completed_dispatcher is not None
        assert completed_dispatcher.id == dispatcher.id
        assert completed_dispatcher.status is TaskStatus.SUCCEEDED, completed_dispatcher.last_error

        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(retained_root.id).status == "complete"
            assert transaction.tasks.get(dispatcher.id).status is TaskStatus.SUCCEEDED
            assert transaction.tasks.get(successor.task_id).status is TaskStatus.PENDING

        completed_successor = await drain_tasks(app)
        assert completed_successor[0].id == successor.task_id
        assert completed_successor[0].status is TaskStatus.SUCCEEDED, completed_successor[0].last_error
        with app.unit_of_work.transaction() as transaction:
            successor_root = transaction.work.get_root(successor.successor_root_id)
            assert successor_root.status == "complete"
            for block_id, root_id in (
                (document.sections[0].blocks[0].id, successor_root.id),
                (document.sections[0].blocks[1].id, retained_root.id),
            ):
                binding = transaction.work.get_binding("result", f"{run.id}:{block_id}")
                assert binding["payload_json"]["root_id"] == root_id
    finally:
        release.set()
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


async def test_restart_translation_allows_accepted_scope_updated_by_chapter_correction(tmp_path):
    from dataclasses import asdict
    from uuid import uuid4
    from linguary.domain.recovery import CallAttempt

    app, _, _, run, _, root, accepted, unfinished = await make_partial_root(tmp_path)
    try:
        accepted_block = accepted.ordered_blocks[0]
        correction_attempt_id = str(uuid4())
        correction_artifact_id = str(uuid4())
        with app.unit_of_work.transaction() as transaction:
            before_result = transaction.results.get(run.id, accepted_block)
            before_binding = transaction.work.get_binding("result", f"{run.id}:{accepted_block}")
            transaction.work.add_attempt(
                CallAttempt(
                    correction_attempt_id,
                    root.book_id,
                    state="accepted",
                    generation=0,
                    ordinal=1,
                    conditions_id=root.conditions_id,
                    run_id=run.id,
                    context={
                        "schema_version": 1,
                        "stage": "chapter_correction",
                        "ordered_block_ids": [accepted_block],
                        "policy_version": "correction-v1",
                    },
                )
            )
            transaction.work.archive_artifacts(
                [
                    {
                        "id": correction_artifact_id,
                        "book_id": root.book_id,
                        "conditions_id": root.conditions_id,
                        "run_id": run.id,
                        "attempt_id": correction_attempt_id,
                        "kind": "block_result",
                        "scope_id": accepted_block,
                        "body": {
                            "schema_version": 1,
                            "result": asdict(before_result),
                            "source_revision": before_binding["artifact_revision_id"],
                            "source_attempt": before_binding["attempt_id"],
                        },
                    }
                ]
            )
            transaction.work.replace_result_binding(
                f"{run.id}:{accepted_block}",
                book_id=root.book_id,
                conditions_id=root.conditions_id,
                attempt_id=correction_attempt_id,
                artifact_revision_id=correction_artifact_id,
                expected_attempt_id=before_binding["attempt_id"],
                expected_artifact_revision_id=before_binding["artifact_revision_id"],
                provenance={"schema_version": 1},
            )

        result = restart(app, root)
        with app.unit_of_work.transaction() as transaction:
            successor = transaction.work.get_root(result.successor_root_id)
            scopes = transaction.work.list_scopes(successor.id)
            assert successor.manifest["historical_coverage"] == accepted.ordered_blocks
            assert [scope.ordered_blocks for scope in scopes] == [unfinished.ordered_blocks]
    finally:
        await app.aclose()


def _split_chapter_app(tmp_path, fake, *, concurrency):
    """Two short blocks become two roots; worker concurrency is also the root limit."""
    return build_application(settings=AppSettings(
        llm=LLMSettings(
            model="pinned-model",
            api_key="synthetic-secret",
            proxy_mode="direct",
            max_unit_source_chars=15,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
        worker=WorkerPoolSettings(concurrency=concurrency),
    ), llm_client=fake, content_plan_profile="legacy-v1")


def _queue_translation(fake, request):
    steps = list(fake.replies)
    index = len(fake.requests)
    while len(steps) <= index:
        steps.append(FakeReply.of_payload({"blocks": [], "language_notes": []}))
    steps[index] = FakeReply.of_payload({
        "blocks": [
            {
                "source_block_id": f"b{block_index}",
                "treatment": "TRANSLATE",
                "translated_text": "Кот был здесь.",
                "glossary_hints": [],
            }
            for block_index, _block_id in enumerate(request.ordered_block_ids, start=1)
        ],
        "language_notes": [],
    })
    fake.replies = tuple(steps)


async def test_restart_one_root_at_concurrency_2_keeps_sibling_running_and_saved(tmp_path):
    both_entered = asyncio.Event()
    restarted_returned = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    gates = {}
    seen = []
    restarted = {"id": None}

    class PausingClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage != LLMCallStage.TRANSLATION_UNIT:
                return await super().generate(request)
            block_id = request.ordered_block_ids[0]
            gate = asyncio.Event()
            gates[block_id] = gate
            seen.append(block_id)
            if len(seen) == 2:
                both_entered.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                if block_id != restarted["id"]:
                    sibling_cancelled.set()
                raise
            _queue_translation(self, request)
            response = await super().generate(request)
            if block_id == restarted["id"]:
                restarted_returned.set()
            return response

    fake = PausingClient([analysis_reply(2)])
    app = _split_chapter_app(tmp_path, fake, concurrency=2)
    running = None
    try:
        document = store_imported(app, (("The cat sat.", "The dog ran."),))
        await app.analysis.start("book-1")
        analysis = await drain_tasks(app)
        assert analysis[0].status is TaskStatus.SUCCEEDED, analysis[0].last_error
        run = app.start_processing()(StartProcessingCommand("book-1"))
        dispatcher = (await app.processing.start("book-1"))[0]
        with app.unit_of_work.transaction() as transaction:
            refs = transaction.work.get_binding("task", dispatcher.id)["payload_json"]["operation_refs"]
            assert len(refs) == 2

        running = asyncio.create_task(app.runner.run_next())
        await asyncio.wait_for(both_entered.wait(), 5)
        with app.unit_of_work.transaction() as transaction:
            roots = [
                transaction.work.get_root(ref["root_id"])
                for ref in transaction.work.get_binding("task", dispatcher.id)["payload_json"]["operation_refs"]
            ]
        restarted_root = next(root for root in roots if seen[0] in root.manifest["ordered_blocks"])
        sibling_root = next(root for root in roots if root.id != restarted_root.id)
        sibling_block = seen[1]
        restarted["id"] = seen[0]
        successor = restart(app, restarted_root, acknowledge=True)
        gates[seen[0]].set()
        await asyncio.wait_for(restarted_returned.wait(), 5)
        await asyncio.sleep(0)
        if not sibling_cancelled.is_set() and not running.done():
            gates[sibling_block].set()
        completed = await asyncio.wait_for(running, 5)
        assert not sibling_cancelled.is_set()
        assert completed.id == dispatcher.id
        assert completed.status is TaskStatus.SUCCEEDED, completed.last_error
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(sibling_root.id).status == "complete"
            assert transaction.tasks.get(successor.task_id).status is TaskStatus.PENDING
            binding = transaction.work.get_binding("result", f"{run.id}:{sibling_block}")
            assert binding["payload_json"]["root_id"] == sibling_root.id
            assert transaction.results.get(run.id, document.sections[0].blocks[0].id) is not None or (
                transaction.results.get(run.id, document.sections[0].blocks[1].id) is not None
            )
            assert transaction.results.get(run.id, sibling_block) is not None
    finally:
        for gate in gates.values():
            gate.set()
        if running is not None and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await app.aclose()


async def test_restart_waiting_root_at_concurrency_1_does_not_fail_dispatcher(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    entered_block = {}

    class PausingClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                if not entered.is_set():
                    entered_block["id"] = request.ordered_block_ids[0]
                    entered.set()
                    await release.wait()
                _queue_translation(self, request)
            return await super().generate(request)

    fake = PausingClient([analysis_reply(2)])
    app = _split_chapter_app(tmp_path, fake, concurrency=1)
    running = None
    try:
        document = store_imported(app, (("The cat sat.", "The dog ran."),))
        await app.analysis.start("book-1")
        analysis = await drain_tasks(app)
        assert analysis[0].status is TaskStatus.SUCCEEDED, analysis[0].last_error
        run = app.start_processing()(StartProcessingCommand("book-1"))
        dispatcher = (await app.processing.start("book-1"))[0]
        running = asyncio.create_task(app.runner.run_next())
        await asyncio.wait_for(entered.wait(), 5)
        with app.unit_of_work.transaction() as transaction:
            roots = [
                transaction.work.get_root(ref["root_id"])
                for ref in transaction.work.get_binding("task", dispatcher.id)["payload_json"]["operation_refs"]
            ]
            assert len(roots) == 2
        running_root = next(root for root in roots if entered_block["id"] in root.manifest["ordered_blocks"])
        waiting_root = next(root for root in roots if root.id != running_root.id)
        successor = restart(app, waiting_root)
        release.set()
        completed = await asyncio.wait_for(running, 5)
        assert completed.id == dispatcher.id
        assert completed.status is TaskStatus.SUCCEEDED, completed.last_error
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(running_root.id).status == "complete"
            assert transaction.work.get_root(waiting_root.id).superseded_by == successor.successor_root_id
            assert transaction.tasks.get(successor.task_id).status is TaskStatus.PENDING
            assert transaction.results.get(run.id, entered_block["id"]) is not None
            waiting_block = waiting_root.manifest["ordered_blocks"][0]
            assert transaction.results.get(run.id, waiting_block) is None
            assert document.sections[0].blocks
    finally:
        release.set()
        if running is not None and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await app.aclose()


async def test_local_read_as_is_commits_in_single_transaction(tmp_path):
    fake = FakeLLMClient([FakeReply.of_payload(dict(
        translation_instructions="Сохраняй смысл.",
        chapter_summary="The cat sat.",
        glossary_candidates=[],
        treatment_plan=[dict(source_block_id="b1", treatment="READ_AS_IS")],
        before_you_read=[],
    ))])
    app = make_app(tmp_path, fake)
    depth = {"n": 0}
    events = []
    original_tx = app.unit_of_work.transaction
    original_local = SqlAlchemyWorkRepository.begin_local_acceptance
    original_recovery = SqlAlchemyWorkRepository.begin_recovery_attempt
    original_save = TranslationCall.save

    @contextmanager
    def tracing_transaction():
        depth["n"] += 1
        try:
            with original_tx() as transaction:
                yield transaction
        finally:
            depth["n"] -= 1

    def trace_local(self, *args, **kwargs):
        events.append(("local", depth["n"]))
        return original_local(self, *args, **kwargs)

    def trace_recovery(self, *args, **kwargs):
        events.append(("recovery", depth["n"]))
        return original_recovery(self, *args, **kwargs)

    def trace_save(self, *args, **kwargs):
        events.append(("save", depth["n"], kwargs.get("transaction") is not None, kwargs.get("chapter_num")))
        return original_save(self, *args, **kwargs)

    try:
        document = store_imported(app, (("The cat sat.",),))
        await app.analysis.start("book-1")
        analysis = await drain_tasks(app)
        assert analysis[0].status is TaskStatus.SUCCEEDED, analysis[0].last_error
        run = app.start_processing()(StartProcessingCommand("book-1"))
        await app.processing.start("book-1")
        app.unit_of_work.transaction = tracing_transaction
        SqlAlchemyWorkRepository.begin_local_acceptance = trace_local
        SqlAlchemyWorkRepository.begin_recovery_attempt = trace_recovery
        TranslationCall.save = trace_save
        completed = await drain_tasks(app)
        assert completed[0].status is TaskStatus.SUCCEEDED, completed[0].last_error
        acceptance = [event for event in events if event[0] in {"local", "recovery"}]
        saves = [event for event in events if event[0] == "save"]
        assert acceptance == [("local", saves[0][1] if saves else None)]
        assert len(saves) == 1
        assert saves[0][1] > 0 and saves[0][2] is True and saves[0][3] == 1
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots("book-1", run_id=run.id)
            attempts = transaction.work.list_root_attempts(root.id)
            assert root.calls_used == 0
            assert transaction.work.root_coverage_complete(root.id)
            assert [attempt.state for attempt in attempts] == ["accepted"]
            assert transaction.results.get(run.id, document.sections[0].blocks[0].id) is not None
        assert not any(request.stage == LLMCallStage.TRANSLATION_UNIT for request in fake.requests)
    finally:
        app.unit_of_work.transaction = original_tx
        SqlAlchemyWorkRepository.begin_local_acceptance = original_local
        SqlAlchemyWorkRepository.begin_recovery_attempt = original_recovery
        TranslationCall.save = original_save
        await app.aclose()


async def test_discarded_result_never_marks_incomplete_chapter_succeeded(tmp_path, monkeypatch):
    fake = FakeLLMClient([
        analysis_reply(1),
        analysis_reply(1),
        translation_reply(1),
        translation_reply(1),
    ])
    app = make_app(tmp_path, fake)
    original_save = TranslationCall.save
    state = {"calls": 0}

    def discard_first(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return False
        return original_save(self, *args, **kwargs)

    try:
        document = store_imported(app, (("The cat sat.",), ("The dog ran.",)))
        await app.analysis.start("book-1")
        analysis = await drain_tasks(app)
        assert all(task.status is TaskStatus.SUCCEEDED for task in analysis), [task.last_error for task in analysis]
        run = app.start_processing()(StartProcessingCommand("book-1"))
        dispatchers = await app.processing.start("book-1")
        assert len(dispatchers) == 2
        monkeypatch.setattr(TranslationCall, "save", discard_first)
        completed = await app.runner.run_next()
        assert completed is not None
        assert completed.status is TaskStatus.FAILED, completed.status
        with app.unit_of_work.transaction() as transaction:
            section = next(section for section in document.sections if section.id == completed.scope_id)
            block_id = section.blocks[0].id
            root = next(
                root for root in transaction.work.list_roots("book-1", run_id=run.id)
                if block_id in root.manifest["ordered_blocks"]
            )
            assert root.superseded_by is None
            assert transaction.runs.get(run.id).status == "active"
            assert not transaction.work.root_coverage_complete(root.id, update_state=False)
            assert transaction.results.get(run.id, block_id) is None
            other = next(task for task in dispatchers if task.id != completed.id)
            assert transaction.tasks.get(other.id).status is TaskStatus.PENDING
    finally:
        await app.aclose()


@pytest.mark.parametrize("error_kind", ["provider", "capacity", "truncated", "rate_limit", "read_timeout"])
async def test_restarted_root_failing_at_concurrency_2_isolates_sibling(tmp_path, error_kind):
    """When root 1 is restarted while in-flight and then fails with a terminal error, sibling root 2 still succeeds."""
    from linguary.application.ports.llm import (
        LLMContextCapacityError,
        LLMProviderError,
        LLMResponseFormatError,
    )
    from linguary.domain.llm_log import Violation

    both_entered = asyncio.Event()
    gates = {}
    seen = []
    restarted_block = {"id": None}

    class FailingPausingClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage != LLMCallStage.TRANSLATION_UNIT:
                return await super().generate(request)
            block_id = request.ordered_block_ids[0]
            gate = asyncio.Event()
            gates[block_id] = gate
            seen.append(block_id)
            if len(seen) == 2:
                both_entered.set()
            await gate.wait()
            if block_id == restarted_block["id"]:
                if error_kind == "provider":
                    raise LLMProviderError("400 rejected")
                if error_kind == "capacity":
                    raise LLMContextCapacityError("context length exceeded")
                if error_kind == "truncated":
                    raise LLMResponseFormatError(
                        "truncated",
                        truncated=True,
                        violation=Violation("provider_truncated", "translation_unit", safe_message="truncated"),
                    )
                if error_kind == "read_timeout":
                    raise LLMTransportError("read timeout", outcome="unknown")
                raise LLMTransportError("HTTP 429", outcome="known_transient")
            _queue_translation(self, request)
            return await super().generate(request)

    fake = FailingPausingClient([analysis_reply(2)])
    app = build_application(settings=AppSettings(
        llm=LLMSettings(
            model="pinned-model",
            api_key="synthetic-secret",
            proxy_mode="direct",
            max_unit_source_chars=15,
            max_attempts=1,
        ),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
        worker=WorkerPoolSettings(concurrency=2),
    ), llm_client=fake, content_plan_profile="legacy-v1")
    running = None
    try:
        store_imported(app, (("The cat sat.", "The dog ran."),))
        await app.analysis.start("book-1")
        assert (await drain_tasks(app))[0].status is TaskStatus.SUCCEEDED
        run = app.start_processing()(StartProcessingCommand("book-1"))
        dispatcher = (await app.processing.start("book-1"))[0]
        running = asyncio.create_task(app.runner.run_next())
        await asyncio.wait_for(both_entered.wait(), 5)
        with app.unit_of_work.transaction() as tx:
            roots = [
                tx.work.get_root(ref["root_id"])
                for ref in tx.work.get_binding("task", dispatcher.id)["payload_json"]["operation_refs"]
            ]
        restarted_root = next(r for r in roots if seen[0] in r.manifest["ordered_blocks"])
        sibling_root = next(r for r in roots if r.id != restarted_root.id)
        sibling_block = seen[1]
        restarted_block["id"] = seen[0]

        successor = restart(app, restarted_root, acknowledge=True)
        gates[seen[0]].set()
        gates[sibling_block].set()
        completed = await asyncio.wait_for(running, 5)
        assert completed.id == dispatcher.id
        assert completed.status is TaskStatus.SUCCEEDED, completed.last_error
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(sibling_root.id).status == "complete"
            assert tx.work.get_root(restarted_root.id).superseded_by == successor.successor_root_id
            assert tx.results.get(run.id, sibling_block) is not None
    finally:
        for gate in gates.values():
            gate.set()
        if running is not None and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await app.aclose()


async def test_stale_generation_failure_after_reconcile_does_not_overwrite_root(tmp_path):
    """An attempt started at generation G failing after reconcile (G+1) does not mutate the root."""
    from linguary.application.ports.llm import LLMProviderError

    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowFailClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                entered.set()
                await release.wait()
                raise LLMProviderError("400 rejected")
            return await super().generate(request)

    fake = SlowFailClient([analysis_reply(1)])
    app = make_app(tmp_path, fake)
    running = None
    try:
        _doc, run, _task = await prepare(app, blocks=1)
        running = asyncio.create_task(app.runner.run_next())
        await asyncio.wait_for(entered.wait(), 5)
        with app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            gen_before = root.generation
            tx.work.reconcile_recovery_attempts()
            reconciled = tx.work.get_root(root.id)
            gen_after = reconciled.generation
            status_after = reconciled.status
            reason_after = reconciled.reason
        assert gen_after == gen_before + 1
        assert (status_after, reason_after) == ("stopped", "interrupted")
        release.set()
        await asyncio.wait_for(running, 5)
        with app.unit_of_work.transaction() as tx:
            final_root = tx.work.get_root(root.id)
        assert final_root.generation == gen_after
        assert final_root.status == status_after
        assert final_root.reason == reason_after
    finally:
        release.set()
        if running is not None and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await app.aclose()

