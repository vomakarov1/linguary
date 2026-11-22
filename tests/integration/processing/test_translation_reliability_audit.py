"""End-to-end regression for translation reliability AC-1..AC-11.

Real SQLite, Alembic, and controllable async clocks. No external LLM.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import math
import sqlite3
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from alembic import command
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, OperationalError
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from linguary.adapters.config.settings import LLMSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.models import Base, DocumentBlockRow
from linguary.adapters.llm import FakeReply
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.adapters.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    _raise_for_status,
)
from linguary.adapters.llm.rate_limit import LLMRateLimiter
from linguary.adapters.runtime import ExecutorLock, SqliteTaskStore
from linguary.application.block_processing import TranslationCall, complete_processing
from linguary.application.book_overview import book_overview
from linguary.application.context_builder import ContextBuilder
from linguary.application.diagnostics import Diagnostics
from linguary.application.llm_dto import (
    BlockResultResponse,
    FragmentAssistanceResponse,
    LanguageNoteResponse,
    SegmentTranslationResponse,
    TranslationResultResponse,
)
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
    LLMResponseKind,
    LLMTransportError,
)
from linguary.application.ports.persistence import BookRecord, PersistenceError
from linguary.application.book_conditions import payload_digest
from linguary.application.recovery import NOTES_POLICY, RecoveryError, RecoveryPolicy, root_manifest
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.translation_policy import (
    CURRENT_TRANSLATION_POLICY,
    TRANSLATION_CONTEXT_POLICY_V2,
)
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.correct_chapter import CorrectChapter
from linguary.application.use_cases.extend_recovery_limit import (
    ExtendRecoveryLimitCommand,
    ExtendRecoveryLimitConflict,
)
from linguary.application.use_cases.restart_translation import RestartTranslationError
from linguary.application.use_cases.retry_task import RetryTaskCommand, RetryTaskError
from linguary.composition import drain_tasks
from linguary.composition.book_runtime import ProvenanceClient
from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    ErrorType,
    LLMCallStage,
    ProcessingTask,
    TaskError,
    TaskIdentity,
    TaskStatus,
    TaskType,
    TranslationUnit,
)
from linguary.domain.book_conditions import WorkStateError
from linguary.domain.llm_log import Violation
from linguary.domain.learning import LanguageNote
from linguary.domain.processing import BlockProcessingResult
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope
from linguary.web.routes.books import _is_stalled, _needs_refresh
from linguary.web.schemas import BookDetailResponse
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    make_validator,
    store_processing_book,
    translated_reply,
)
from test_book_conditions import make_app, store_imported
from test_concurrent_note_ids_and_policies import (
    _ADDED_NOTE,
    _KEPT_NOTE,
    _NEW_EXCERPT,
    _OLD_EXCERPT,
    _correction_inputs,
    _counter_in,
    test_concurrent_chapter_regeneration_and_correction_never_collide_on_note_ids as _concurrent_note_ids,
    test_regenerate_block_does_not_duplicate_live_v2_operation_under_v3 as _live_v2_regeneration,
)
from test_recovery_commit import analysis_reply, prepare, translation_reply
from test_recovery_lifecycle import (
    _UNKNOWN_RESTART,
    _assert_restart_audit,
    _restart,
    _stop_as,
)
from test_restart_translation import _queue_translation, _split_chapter_app, restart
from test_send_boundary_and_rate_limit import _arm_rate_limit, _open_gate, _running_attempt
from test_translation_context_capacity import no_sleep

_MIGRATION_PATH = Path(__file__).resolve().parents[3] / "alembic/versions/0018_note_number_counters.py"
_QUOTE = "One.\nTwo.\nThree."
_POEM = "Rose is red.\nViolet is blue."
_SEGMENTED = "Alpha stays.\nSecond stays.\nGamma goes.\nDelta goes."
_DIALOGUE = 'She said "hello" softly.'
_QUOTED_REFUSAL = 'He said "I can\'t fulfill this request." once.'
_REFUSAL = "I’m sorry, but I cannot fulfill this."
_SHORT_RU = "Он вернулся домой."


class ControlledClock:
    """Waits resolve only when the test advances the clock."""

    def __init__(self):
        self.now = 0.0
        self.waiters = []
        self.waiting = asyncio.Queue()

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + seconds, future))
        self.waiting.put_nowait(self.now + seconds)
        await future

    def advance(self, now):
        self.now = now
        for deadline, future in self.waiters:
            if deadline <= now and not future.done():
                future.set_result(None)


async def _settle(clock: ControlledClock, admitted: list[float]) -> None:
    snapshot = None
    quiet = 0
    for _ in range(40):
        await asyncio.sleep(0)
        current = (len(admitted), clock.waiting.qsize(), clock.now)
        if current == snapshot:
            quiet += 1
            if quiet >= 5:
                return
        else:
            snapshot = current
            quiet = 0


def _publish_attempts(app, attempts: int):
    tuned = replace(app.settings, llm=replace(app.settings.llm, max_attempts=attempts))
    app.runtime_settings.publish(app.runtime_settings.prepare(tuned), app.runtime_settings.revision + 1)
    return tuned


def _translation_requests(fake):
    return [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT]


def _chat_document(*, finish_reason, content=None, refusal=None):
    message = {"role": "assistant", "content": content}
    if refusal is not None:
        message["refusal"] = refusal
    return {
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


class _HttpStub:
    def __init__(self, document=None, error=None):
        self.document = document
        self.error = error
        self.posts = 0

    async def post(self, *args, **kwargs):
        self.posts += 1
        if self.error is not None:
            raise self.error
        return httpx.Response(200, json=self.document)


def _gateway(stub):
    settings = LLMSettings(model="pinned-model", api_key="synthetic-secret", proxy_mode="direct")
    return OpenAICompatibleLLMClient(settings=settings, http_client=stub)


def _shape_blocks(app, shapes):
    with app.unit_of_work.transaction() as tx:
        session = tx.results._session
        for block_id, block_type, segments in shapes:
            row = session.get(DocumentBlockRow, block_id)
            assert row is not None
            row.type = block_type
            if segments is not None:
                row.segments_json = segments


def _open_library(app, texts):
    scenario = store_processing_book(app.unit_of_work, chapters=(texts,))
    source = app.paths.source_epub("book-1")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"isolated synthetic source")
    with app.unit_of_work.transaction() as tx:
        book = replace(tx.books.get("book-1"), source_epub_path=str(source))
        tx.books.save(book)
        conditions = app.book_conditions.create(tx, tx.books.get("book-1"), 0, continue_legacy=True)
        versions = dict(conditions.payload["versions"])
    return scenario, conditions, versions


def _open_call(app, scenario, conditions, client, block_id, *, policy_version, max_attempts=1):
    with app.unit_of_work.transaction() as tx:
        document = tx.documents.get("book-1")
        section = document.section(scenario.section().id)
        block = next(item for item in section.blocks if item.id == block_id)
        analysis = tx.analyses.get_for_section("book-1", section.id)
        run = tx.runs.active_for_book("book-1")
        unit = TranslationUnit.of_blocks(id=f"unit-{block_id}", blocks=(block,), plan=analysis.treatment_plan)
        manifest = root_manifest(
            book_id="book-1", conditions_id=conditions.id, glossary_sha="a" * 64, analysis_sha="b" * 64,
            plan_sha=None, ordered_blocks=unit.block_ids,
            treatments=tuple(item.treatment.value for item in unit.items),
            boundaries=(unit.block_ids,), atomic_intervals=tuple((item,) for item in unit.block_ids),
            max_source_chars=1000, policy=RecoveryPolicy(max_calls=12),
        )
        root = RecoveryRoot(
            str(uuid4()), "book-1", conditions.id, str(uuid4()),
            {**manifest, "active_run_id": run.id, "historical_coverage": []},
            policy_version=policy_version,
        )
        scope = RecoveryScope(str(uuid4()), root.id, unit.block_ids)
        tx.work.add_root(root)
        tx.work.add_scope(scope)
    context = ContextBuilder().for_translation_unit(
        document=document, section=section, unit=unit, analysis=analysis,
        glossary=run.glossary_snapshot, policy_version=policy_version,
    )
    context = replace(
        context,
        request=context.request.with_logging(book_id="book-1", stage=LLMCallStage.TRANSLATION_UNIT),
    )
    logger = LoggingLLMClient(client, app.unit_of_work, "fake", "pinned-model")
    wrapped = ProvenanceClient(logger, app.book_conditions, conditions.id)
    call = TranslationCall(
        unit_of_work=app.unit_of_work, llm_client=wrapped,
        retry_policy=RetryPolicy(max_attempts=max_attempts, backoff_seconds=0),
        validator=make_validator(), sleep=no_sleep,
    )
    issued = {"n": 0}

    def note_ids():
        issued["n"] += 1
        return f"temp-{issued['n']}"

    return SimpleNamespace(
        call=call, context=context, root=root, scope=scope, unit=unit, run=run,
        request_args=dict(unit=unit, target_language="ru", note_ids=note_ids, recovery=(root.id, scope.id)),
    )


class _PayloadClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    @property
    def call_count(self):
        return len(self.requests)

    async def generate(self, request):
        self.requests.append(request)
        step = self.payloads.pop(0)
        if isinstance(step, Exception):
            raise step
        return LLMResponse(
            payload=step, provider="fake", model="pinned-model", call_context=request.call_context,
        )


def _block_payload(text, *, segments=()):
    if segments:
        return TranslationResultResponse(blocks=(
            BlockResultResponse("b1", BlockTreatment.TRANSLATE, None, (), segments),
        ))
    return TranslationResultResponse(blocks=(
        BlockResultResponse("b1", BlockTreatment.TRANSLATE, text),
    ))


def _manifest(run_id, blocks, *, max_calls=3, max_scope_calls=3):
    return dict(
        schema_version=1, active_run_id=run_id, ordered_blocks=list(blocks), historical_coverage=[],
        policy=dict(max_calls=max_calls, max_scope_calls=max_scope_calls, max_depth=3, max_leaves=8),
    )


def _restartable_manifest(tx, *, conditions_id, run_id, blocks, section_id):
    run = tx.runs.get(run_id)
    analysis = tx.analyses.get_for_section("book-1", section_id)
    plan = analysis.content_plan
    manifest = root_manifest(
        book_id="book-1", conditions_id=conditions_id,
        glossary_sha=payload_digest(asdict(run.glossary_snapshot)),
        analysis_sha=payload_digest(asdict(analysis)),
        plan_sha=None if plan is None else plan.sha256,
        ordered_blocks=blocks, treatments=tuple("TRANSLATE" for _ in blocks),
        boundaries=tuple((block_id,) for block_id in blocks),
        atomic_intervals=tuple((block_id,) for block_id in blocks),
        max_source_chars=4000, policy=RecoveryPolicy(),
        block_chars={block_id: 20 for block_id in blocks},
    )
    manifest.update(notes_policy=NOTES_POLICY, active_run_id=run_id, historical_coverage=[])
    return manifest


def _plant_root(tx, *, conditions_id, run_id, blocks, status, reason=None, calls_used=0, kind="translation", manifest=None):
    root = RecoveryRoot(
        str(uuid4()), "book-1", conditions_id, str(uuid4()),
        manifest if manifest is not None else _manifest(run_id, blocks),
        calls_used=calls_used, status=status, reason=reason,
        policy_version=CURRENT_TRANSLATION_POLICY, operation_kind=kind,
    )
    scope = RecoveryScope(str(uuid4()), root.id, tuple(blocks))
    tx.work.add_root(root)
    tx.work.add_scope(scope)
    return root, scope


def _plant_attempt(tx, root, scope, run_id, *, state, ordinal):
    attempt = CallAttempt(
        str(uuid4()), "book-1", state=state, generation=scope.generation, ordinal=ordinal,
        conditions_id=root.conditions_id, run_id=run_id, root_id=root.id, scope_id=scope.id,
        context={
            "schema_version": 1,
            "root_generation": root.generation,
            "policy_version": root.policy_version,
        },
    )
    tx.work.add_attempt(attempt)
    return attempt


def _failed_task(tx, root, section_id):
    task = ProcessingTask(
        str(uuid4()), "book-1", TaskType.PROCESS_TRANSLATION_UNIT, section_id,
        status=TaskStatus.FAILED, attempts=1,
        last_error=TaskError(ErrorType.TRANSIENT, "временный сбой обработки", "detail"),
        operation_id=str(uuid4()),
    )
    tx.tasks.add(task)
    tx.work.bind(
        "task", task.id, book_id="book-1", conditions_id=root.conditions_id,
        payload={"schema_version": 1, "operation_refs": [
            {"root_id": root.id, "policy_version": root.policy_version},
        ]},
    )
    return task


async def _saved_translation(tmp_path):
    app = make_app(tmp_path)
    from test_translation_context_capacity import capacity_call
    work = capacity_call(app, FakeLLMClient([translated_reply(("b1",))]), neighbours=False)
    result = await work.call.request(work.context, **work.request_args)
    assert work.call.save(book_id="book-1", run_id=work.run.id, results=result.results)
    return app, work


def _add_root(app, work, *, kind, status, reason=None):
    root = replace(
        work.root, id=str(uuid4()), root_key=str(uuid4()), status=status, reason=reason,
        operation_kind=kind, superseded_by=None, generation=0,
    )
    scope = replace(work.scope, id=str(uuid4()), root_id=root.id, status="pending", parent_id=None)
    with app.unit_of_work.transaction() as tx:
        tx.work.add_root(root)
        tx.work.add_scope(scope)
    return root


async def test_ac1_parallel_restart_isolation(tmp_path):
    """Restart at concurrency 1 and 2 keeps siblings, and a stopped regeneration does not block readiness."""
    both_entered = asyncio.Event()
    restarted_returned = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    gates = {}
    seen = []
    restarted_block = {"id": None}

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
                if block_id != restarted_block["id"]:
                    sibling_cancelled.set()
                raise
            _queue_translation(self, request)
            response = await super().generate(request)
            if block_id == restarted_block["id"]:
                restarted_returned.set()
            return response

    fake = PausingClient([analysis_reply(2)])
    app = _split_chapter_app(tmp_path / "c2", fake, concurrency=2)
    running = None
    try:
        document = store_imported(app, (("The cat sat.", "The dog ran."),))
        await app.analysis.start("book-1")
        analysis = await drain_tasks(app)
        assert analysis[0].status is TaskStatus.SUCCEEDED, analysis[0].last_error
        run = app.start_processing()(StartProcessingCommand("book-1"))
        dispatcher = (await app.processing.start("book-1"))[0]
        running = asyncio.create_task(app.runner.run_next())
        await asyncio.wait_for(both_entered.wait(), 5)
        with app.unit_of_work.transaction() as tx:
            roots = [
                tx.work.get_root(ref["root_id"])
                for ref in tx.work.get_binding("task", dispatcher.id)["payload_json"]["operation_refs"]
            ]
        restarted_root = next(root for root in roots if seen[0] in root.manifest["ordered_blocks"])
        sibling_root = next(root for root in roots if root.id != restarted_root.id)
        sibling_block = seen[1]
        restarted_block["id"] = seen[0]
        successor = restart(app, restarted_root)
        gates[seen[0]].set()
        await asyncio.wait_for(restarted_returned.wait(), 5)
        await asyncio.sleep(0)
        if not sibling_cancelled.is_set() and not running.done():
            gates[sibling_block].set()
        completed = await asyncio.wait_for(running, 5)
        assert not sibling_cancelled.is_set()
        assert completed.id == dispatcher.id
        assert completed.status is TaskStatus.SUCCEEDED, completed.last_error
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(sibling_root.id).status == "complete"
            assert tx.tasks.get(successor.task_id).status is TaskStatus.PENDING
            binding = tx.work.get_binding("result", f"{run.id}:{sibling_block}")
            assert binding["payload_json"]["root_id"] == sibling_root.id
            assert tx.results.get(run.id, sibling_block) is not None
            assert tx.results.get(run.id, seen[0]) is None
            assert document.sections[0].blocks
    finally:
        for gate in gates.values():
            gate.set()
        if running is not None and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await app.aclose()

    entered = asyncio.Event()
    release = asyncio.Event()
    entered_block = {}

    class WaitingClient(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                if not entered.is_set():
                    entered_block["id"] = request.ordered_block_ids[0]
                    entered.set()
                    await release.wait()
                _queue_translation(self, request)
            return await super().generate(request)

    waiting_fake = WaitingClient([analysis_reply(2)])
    waiting_app = _split_chapter_app(tmp_path / "c1", waiting_fake, concurrency=1)
    waiting_run = None
    try:
        store_imported(waiting_app, (("The cat sat.", "The dog ran."),))
        await waiting_app.analysis.start("book-1")
        analysis = await drain_tasks(waiting_app)
        assert analysis[0].status is TaskStatus.SUCCEEDED, analysis[0].last_error
        run = waiting_app.start_processing()(StartProcessingCommand("book-1"))
        dispatcher = (await waiting_app.processing.start("book-1"))[0]
        waiting_run = asyncio.create_task(waiting_app.runner.run_next())
        await asyncio.wait_for(entered.wait(), 5)
        with waiting_app.unit_of_work.transaction() as tx:
            roots = [
                tx.work.get_root(ref["root_id"])
                for ref in tx.work.get_binding("task", dispatcher.id)["payload_json"]["operation_refs"]
            ]
            assert len(roots) == 2
        running_root = next(root for root in roots if entered_block["id"] in root.manifest["ordered_blocks"])
        waiting_root = next(root for root in roots if root.id != running_root.id)
        successor = restart(waiting_app, waiting_root)
        release.set()
        completed = await asyncio.wait_for(waiting_run, 5)
        assert completed.id == dispatcher.id
        assert completed.status is TaskStatus.SUCCEEDED, completed.last_error
        waiting_block = waiting_root.manifest["ordered_blocks"][0]
        with waiting_app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(running_root.id).status == "complete"
            assert tx.work.get_root(waiting_root.id).superseded_by == successor.successor_root_id
            assert tx.tasks.get(successor.task_id).status is TaskStatus.PENDING
            assert tx.results.get(run.id, entered_block["id"]) is not None
            assert tx.results.get(run.id, waiting_block) is None
        followed = await drain_tasks(waiting_app)
        assert any(task.status is TaskStatus.SUCCEEDED for task in followed), [task.last_error for task in followed]
        with waiting_app.unit_of_work.transaction() as tx:
            assert tx.results.get(run.id, waiting_block) is not None
            assert tx.books.get("book-1").status is BookStatus.READY_TO_EXPORT
            assert tx.work.get_root(waiting_root.id).superseded_by == successor.successor_root_id
    finally:
        release.set()
        if waiting_run is not None and not waiting_run.done():
            waiting_run.cancel()
            await asyncio.gather(waiting_run, return_exceptions=True)
        await waiting_app.aclose()

    ready_app, work = await _saved_translation(tmp_path / "regen")
    try:
        pending = _add_root(ready_app, work, kind="regeneration", status="pending")
        with pytest.raises(RecoveryError, match="unsafe_partition"):
            complete_processing(
                ready_app.unit_of_work, book_id="book-1", finished_task_id=None,
                task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
            )
        with ready_app.unit_of_work.transaction() as tx:
            assert tx.books.get("book-1").status is BookStatus.PROCESSING
            tx.work.stop_root(pending.id, "limit", expected_generation=pending.generation)
        complete_processing(
            ready_app.unit_of_work, book_id="book-1", finished_task_id=None,
            task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
        )
        with ready_app.unit_of_work.transaction() as tx:
            assert tx.books.get("book-1").status is BookStatus.READY_TO_EXPORT
            assert tx.results.list_for_run(work.run.id)
    finally:
        await ready_app.aclose()

    blocked_app, blocked = await _saved_translation(tmp_path / "translation-stop")
    try:
        stopped = _add_root(blocked_app, blocked, kind="translation", status="stopped", reason="limit")
        with pytest.raises(RecoveryError, match="unsafe_partition"):
            complete_processing(
                blocked_app.unit_of_work, book_id="book-1", finished_task_id=None,
                task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
            )
        with blocked_app.unit_of_work.transaction() as tx:
            assert tx.books.get("book-1").status is BookStatus.PROCESSING
            assert tx.work.get_root(stopped.id).status == "stopped"
            assert tx.results.list_for_run(blocked.run.id)
    finally:
        await blocked_app.aclose()


async def _pause_then_continue(tmp_path, error, reason, *, attempts):
    fake = FakeLLMClient([analysis_reply(2), *([error] * attempts), translation_reply(2)])
    app = make_app(tmp_path, fake)
    tuned = _publish_attempts(app, attempts)
    try:
        _doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[0].status is TaskStatus.FAILED, completed[0].last_error
        with app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            attempts_state = [item.state for item in tx.work.list_root_attempts(root.id)]
            assert (root.status, root.reason) == ("pending", reason)
            assert root.reason != "unsafe_partition"
            assert root.calls_used == attempts
            assert attempts_state == ["failed"] * attempts
            assert all(item != "unknown" for item in attempts_state)
            assert root.calls_used < root.manifest["policy"]["max_scope_calls"]
            assert tuned.llm.max_attempts == attempts
            root_id, generation, extra = root.id, root.generation, root.extra_calls
            failed_task = tx.tasks.get(task.id)
            assert failed_task.attempts >= 1
        assert len(_translation_requests(fake)) == attempts
        continued = app.retry_task(RetryTaskCommand(task_id=task.id))
        assert continued.status is TaskStatus.PENDING
        assert continued.attempts == failed_task.attempts
        finished = await drain_tasks(app)
        continued_task = next(item for item in finished if item.id == task.id)
        assert continued_task.status is TaskStatus.SUCCEEDED, continued_task.last_error
        assert continued_task.attempts == failed_task.attempts + 1
        with app.unit_of_work.transaction() as tx:
            stored = tx.work.get_root(root_id)
            assert stored.id == root_id
            assert stored.status == "complete"
            assert stored.calls_used == attempts + 1
            assert stored.extra_calls == extra
            assert stored.generation >= generation
            assert len(tx.results.list_for_run(run.id)) == 2
            assert [item.state for item in tx.work.list_root_attempts(root_id) if item.state == "failed"] == (
                ["failed"] * attempts
            )
        assert len(_translation_requests(fake)) == attempts + 1
    finally:
        await app.aclose()


async def test_ac2_max_attempts_1_manual_continue(tmp_path):
    """Known 401/402/429/503 and a short attempt budget pause for manual continue without burning the root."""
    classified = {
        401: LLMConfigurationError,
        402: LLMProviderError,
        429: LLMTransportError,
        503: LLMTransportError,
    }
    outcomes = {429: "known_transient", 503: "known_transient"}
    for status, error_type in classified.items():
        with pytest.raises(error_type) as caught:
            _raise_for_status(
                httpx.Response(status), provider="openai_compatible", api_key="synthetic-secret",
            )
        if status in outcomes:
            assert caught.value.outcome == outcomes[status]
    cases = (
        (LLMTransportError("HTTP 503", outcome="known_transient"), "transient_error", 1),
        (LLMConfigurationError("HTTP 401"), "configuration_error", 1),
        (LLMProviderError("HTTP 402"), "provider_error", 1),
        (LLMTransportError("HTTP 429", outcome="known_transient"), "transient_error", 1),
        (LLMTransportError("HTTP 503", outcome="known_transient"), "transient_error", 2),
    )
    for index, (error, reason, attempts) in enumerate(cases):
        await _pause_then_continue(tmp_path / f"pause-{index}", error, reason, attempts=attempts)

    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path / "cancel", fake)
    try:
        _doc, run, _task = await prepare(app)
        entered, _release = _arm_rate_limit(app)
        worker = asyncio.create_task(drain_tasks(app))
        await asyncio.wait_for(entered.wait(), 3)
        root, attempt = await _running_attempt(app, run.id)
        assert attempt.state == "created"
        assert root.calls_used == 0
        worker.cancel()
        await asyncio.wait({worker})
        with app.unit_of_work.transaction() as tx:
            saved = tx.work.get_attempt(attempt.id)
            stored = tx.work.get_root(root.id)
        assert saved.state in {"created", "cancelled"}
        assert stored.calls_used == 0
        assert _translation_requests(fake) == []
    finally:
        await app.aclose()


async def test_ac3_unknown_outcome_explicit_restart_and_fencing(tmp_path):
    """Unknown sends are not retried, need an audited restart, and reject late writes of the old generation."""
    for status in (502, 504, 524):
        with pytest.raises(LLMTransportError) as caught:
            _raise_for_status(
                httpx.Response(status), provider="openai_compatible", api_key="synthetic-secret",
            )
        assert caught.value.outcome == "unknown"
    request = LLMRequest(kind=LLMResponseKind.TRANSLATION_RESULT, instructions="translate", input_text="text")
    for error_type in (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError):
        stub = _HttpStub(error=error_type("synthetic"))
        client = _gateway(stub)
        with pytest.raises(LLMTransportError) as caught:
            await client.generate(request)
        assert caught.value.outcome == "unknown"
        assert stub.posts == 1
        await client.aclose()

    fake = FakeLLMClient([
        analysis_reply(2),
        LLMTransportError("read timeout", outcome="unknown"),
        translation_reply(2),
    ])
    app = make_app(tmp_path / "unknown", fake)
    try:
        _doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[0].status is TaskStatus.FAILED, completed[0].last_error
        with app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            attempt, = tx.work.list_root_attempts(root.id)
            assert (root.status, root.reason, root.calls_used) == ("stopped", "unknown_outcome", 1)
            assert attempt.state == "unknown"
            root_id, attempt_id = root.id, attempt.id
        assert len(_translation_requests(fake)) == 1
        with pytest.raises(RetryTaskError, match="явный перезапуск"):
            app.retry_task(RetryTaskCommand(task_id=task.id))
        assert len(_translation_requests(fake)) == 1
        with pytest.raises(RestartTranslationError, match=_UNKNOWN_RESTART):
            _restart(app, root)
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(root_id).superseded_by is None
            assert tx.work.get_attempt(attempt_id).state == "unknown"
        opened = _restart(app, root, acknowledge=True)
        with app.unit_of_work.transaction() as tx:
            successor = tx.work.get_root(opened.successor_root_id)
            predecessor = tx.work.get_root(root_id)
            _assert_restart_audit(
                successor, predecessor=root_id, generation=predecessor.generation - 1,
                reason="unknown_outcome", had_unknown=True, acknowledged=True,
            )
            assert predecessor.superseded_by == successor.id
            successor_id, successor_status = successor.id, successor.status
            scope = next(item for item in tx.work.list_scopes(root_id) if item.ordered_blocks)
        with pytest.raises(WorkStateError, match="generation_conflict"):
            with app.unit_of_work.transaction() as tx:
                tx.work.stop_root(root_id, "limit", expected_generation=root.generation)
        with pytest.raises(WorkStateError, match="generation_conflict"):
            with app.unit_of_work.transaction() as tx:
                current = tx.work.get_scope(scope.id)
                tx.work.split_scope(
                    current.id, expected_generation=current.generation, children=(current.ordered_blocks,),
                )
        with app.unit_of_work.transaction() as tx:
            assert tx.work.record_recovery_failure(
                attempt_id,
                Violation("provider_truncated", "translation_unit", scope="unit", safe_message="late write"),
            ) is False
            assert tx.work.get_attempt(attempt_id).state == "unknown"
        with pytest.raises(RecoveryError, match="interrupted"):
            with app.unit_of_work.transaction() as tx:
                tx.work.mark_attempt_sent(attempt_id, expected_root_generation=root.generation)
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_attempt(attempt_id).state == "unknown"
            stored = tx.work.get_root(successor_id)
            assert (stored.id, stored.status) == (successor_id, successor_status)
        current = stored
        refused = _stop_as(app, current, "provider_error")
        with pytest.raises(RestartTranslationError, match="нельзя безопасно"):
            _restart(app, refused)
        current = refused
        for reason in (
            "output_truncated", "context_capacity", "provider_refusal", "limit", "scope_limit", "unsafe_partition",
        ):
            current = _stop_as(app, current, reason)
            restarted = _restart(app, current)
            with app.unit_of_work.transaction() as tx:
                followed = tx.work.get_root(restarted.successor_root_id)
                _assert_restart_audit(
                    followed, predecessor=current.id, generation=current.generation,
                    reason=reason, had_unknown=False, acknowledged=False,
                )
                assert tx.work.get_root(current.id).superseded_by == followed.id
                current = followed
    finally:
        await app.aclose()

    limited = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    limited_app = make_app(tmp_path / "limiter", limited)
    try:
        _doc, run, _task = await prepare(limited_app)
        entered, release = _arm_rate_limit(limited_app)
        worker = asyncio.create_task(drain_tasks(limited_app))
        await asyncio.wait_for(entered.wait(), 3)
        root, attempt = await _running_attempt(limited_app, run.id)
        assert attempt.state == "created" and root.calls_used == 0
        limited_app.restart_translation(__import__(
            "linguary.application.use_cases.restart_translation", fromlist=["RestartTranslationCommand"],
        ).RestartTranslationCommand(
            book_id="book-1", root_id=root.id, expected_generation=root.generation,
        ))
        _open_gate(limited_app, release)
        await asyncio.wait_for(worker, 5)
        sent = [
            item for item in _translation_requests(limited)
            if item.call_context is not None and item.call_context.attempt_id == attempt.id
        ]
        assert sent == []
        with limited_app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(root.id).calls_used == 0
            assert tx.work.get_attempt(attempt.id).state not in {"sent", "unknown"}
    finally:
        await limited_app.aclose()


def _load_migration():
    spec = importlib.util.spec_from_file_location("audit_note_counters_0018", _MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _insert_migration_graph(connection):
    connection.execute(text(
        "INSERT INTO books (id, title, authors_json, source_language, target_language, status, "
        "source_epub_path, created_at) VALUES ('book-mig', 'Book', '[]', 'en', 'ru', 'imported', "
        "'source.epub', '2026-10-10 00:00:00')"
    ))
    connection.execute(text(
        "INSERT INTO sections (id, book_id, title, \"order\", source_resource_path) "
        "VALUES ('section-mig', 'book-mig', 'Chapter 1', 0, 'OEBPS/chapter1.xhtml')"
    ))
    connection.execute(text(
        "INSERT INTO document_blocks (id, book_id, section_id, type, \"order\", text, "
        "inline_annotations_json, attributes_json, source_reference_json, support_status) VALUES ("
        "'block-mig', 'book-mig', 'section-mig', 'paragraph', 0, 'Alpha.', '[]', '{}', "
        "'{\"resource_path\":\"OEBPS/chapter1.xhtml\",\"structural_path\":\"/p[1]\"}', 'supported')"
    ))
    connection.execute(text(
        "INSERT INTO processing_runs (id, book_id, status, glossary_snapshot_json, llm_settings_json, created_at) "
        "VALUES ('run-mig', 'book-mig', 'active', '[]', '{}', '2026-10-10 00:00:00')"
    ))
    connection.execute(text(
        "INSERT INTO book_conditions (id, book_id, revision, schema_version, payload_json, reproducibility, created_at) "
        "VALUES ('conditions-mig', 'book-mig', 1, 1, '{\"schema_version\": 1}', 'complete', '2026-10-10 00:00:00')"
    ))
    for note_id in ("ln-ch01-001", "ln-ch01-007"):
        connection.execute(text(
            "INSERT INTO language_notes (run_id, block_id, note_id, original_excerpt, note) "
            "VALUES ('run-mig', 'block-mig', :note_id, 'excerpt', 'заметка')"
        ), {"note_id": note_id})
    connection.execute(text(
        "INSERT INTO correction_candidates (candidate_id, book_id, run_id, conditions_id, section_id, block_id, "
        "input_sha256, input_generation, proposal_sha256, payload_json, group_id, proposal_attempt_id, state) "
        "VALUES ('candidate-mig', 'book-mig', 'run-mig', 'conditions-mig', 'section-mig', 'block-mig', "
        ":sha, 1, :sha, :payload, 'group-mig', 'attempt-mig', 'proposed')"
    ), {
        "sha": "a" * 64,
        "payload": json.dumps({
            "result": {"language_notes": [{"id": "ln-ch01-009"}]},
            "current_result": {"language_notes": [{"id": "ln-ch02-004"}]},
        }),
    })
    connection.execute(text(
        "INSERT INTO artifact_revisions (id, book_id, run_id, kind, scope_id, body_json, sha256, created_at) "
        "VALUES ('artifact-mig', 'book-mig', 'run-mig', 'block_result', 'block-mig', :body, :sha, "
        "'2026-10-10 00:00:00')"
    ), {
        "body": json.dumps({
            "schema_version": 1,
            "result": {"language_notes": [{"id": "ln-ch01-012"}, {"id": "custom-note"}], "label": "see ln-ch01-099"},
        }),
        "sha": "b" * 64,
    })
    connection.execute(text(
        "INSERT INTO artifact_revisions (id, book_id, run_id, kind, scope_id, body_json, sha256, created_at) "
        "VALUES ('artifact-orphan', 'book-mig', 'missing-run', 'block_result', 'block-mig', :body, :sha, "
        "'2026-10-10 00:00:00')"
    ), {
        "body": json.dumps({"result": {"language_notes": [{"id": "ln-ch01-015"}]}}),
        "sha": "c" * 64,
    })


def _counts(connection):
    tables = ("books", "language_notes", "correction_candidates", "artifact_revisions")
    return tuple(connection.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one() for name in tables)


async def test_ac4_ac11_concurrent_note_ids_and_migration(migrated_database, engine, unit_of_work):
    """Migration 0018 backfills live ids, and concurrent allocation does not collide or burn a rollback."""
    migration = _load_migration()
    assert migration.revision == "0018"
    assert migration.down_revision == "0017"
    command.downgrade(migrated_database, "0017")
    with engine.begin() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        _insert_migration_graph(connection)
        before = _counts(connection)
    assert before == (1, 2, 1, 2)
    command.upgrade(migrated_database, "0018")
    with engine.connect() as connection:
        assert _counts(connection) == before
        rows = connection.execute(text(
            "SELECT run_id, chapter_number, last_ordinal FROM note_number_counters ORDER BY chapter_number"
        )).all()
        orphan = connection.execute(text(
            "SELECT COUNT(*) FROM note_number_counters WHERE run_id = 'missing-run'"
        )).scalar_one()
    assert rows == [("run-mig", 1, 12), ("run-mig", 2, 4)]
    assert orphan == 0
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO note_number_counters (run_id, chapter_number, last_ordinal) "
            "VALUES ('missing-run', 1, 15)"
        ))
    with engine.begin() as connection:
        connection.execute(text(
            "UPDATE note_number_counters SET last_ordinal = 40 "
            "WHERE run_id = 'run-mig' AND chapter_number = 1"
        ))
        migration.backfill_note_number_counters(connection)
        migration.backfill_note_number_counters(connection)
        ordinal = connection.execute(text(
            "SELECT last_ordinal FROM note_number_counters "
            "WHERE run_id = 'run-mig' AND chapter_number = 1"
        )).scalar_one()
    assert ordinal == 40
    command.downgrade(migrated_database, "0017")
    with engine.connect() as connection:
        assert "note_number_counters" not in sa_inspect(connection).get_table_names()
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "0017"
        assert _counts(connection) == before
    command.upgrade(migrated_database, "head")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "0018"
        restored = connection.execute(text(
            "SELECT last_ordinal FROM note_number_counters "
            "WHERE run_id = 'run-mig' AND chapter_number = 1"
        )).scalar_one()
        orphan = connection.execute(text(
            "SELECT COUNT(*) FROM note_number_counters WHERE run_id = 'missing-run'"
        )).scalar_one()
    assert restored == 12
    assert orphan == 0
    engine.dispose()

    await _concurrent_note_ids(unit_of_work, engine)
    boxes = []

    def take():
        found = []
        with unit_of_work.transaction() as tx:
            found.extend(tx.results.allocate_note_ids(RUN_ID, 2, 1))
        boxes.append(tuple(found))

    threads = [threading.Thread(target=take) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert all(not thread.is_alive() for thread in threads)
    issued = [note_id for batch in boxes for note_id in batch]
    assert sorted(issued) == ["ln-ch02-001", "ln-ch02-002"]
    with unit_of_work.transaction() as tx:
        assert tx.results._session.get(
            __import__("linguary.adapters.db.models", fromlist=["NoteNumberCounterRow"]).NoteNumberCounterRow,
            (RUN_ID, 2),
        ).last_ordinal == 2
    with pytest.raises(RuntimeError, match="rollback allocation"):
        with unit_of_work.transaction() as tx:
            rolled = tx.results.allocate_note_ids(RUN_ID, 3, 1)
            assert rolled == ("ln-ch03-001",)
            raise RuntimeError("rollback allocation")
    with unit_of_work.transaction() as tx:
        again = tx.results.allocate_note_ids(RUN_ID, 3, 1)
        assert again == ("ln-ch03-001",)
        stored = {
            note.id for result in tx.results.list_for_run(RUN_ID) for note in result.language_notes
        }
    assert "ln-ch03-001" not in stored


async def _expect_validation(app, scenario, conditions, block_id, payload, *, policy, reason, fragment):
    client = _PayloadClient([payload])
    work = _open_call(app, scenario, conditions, client, block_id, policy_version=policy)
    with app.unit_of_work.transaction() as tx:
        before = tx.results.get(work.run.id, block_id)
    with pytest.raises(RecoveryError, match=reason) as caught:
        await work.call.request(work.context, **work.request_args)
    assert caught.value.reason == reason
    assert isinstance(caught.value.original_error, ResultValidationError)
    assert fragment in str(caught.value.original_error)
    assert caught.value.reason not in {"unsafe_partition", "output_truncated"}
    assert client.call_count == 1
    with app.unit_of_work.transaction() as tx:
        root = tx.work.get_root(work.root.id)
        assert root.calls_used == 1
        assert len(tx.work.list_scopes(root.id)) == 1
        assert tx.work.list_scopes(root.id)[0].status != "split"
        assert tx.results.get(work.run.id, block_id) == before
    return caught


async def _expect_saved(app, scenario, conditions, block_id, payload, *, policy):
    client = _PayloadClient([payload])
    work = _open_call(app, scenario, conditions, client, block_id, policy_version=policy)
    result = await work.call.request(work.context, **work.request_args)
    assert work.call.save(book_id="book-1", run_id=work.run.id, results=result.results, chapter_num=1)
    with app.unit_of_work.transaction() as tx:
        assert tx.results.get(work.run.id, block_id) is not None
        assert tx.work.get_root(work.root.id).calls_used == 1
    return work


async def test_ac5_ac6_ac7_ac9_validation_and_policy_compat(tmp_path, unit_of_work, task_queue):
    """v2 keeps lines, truncation wins, service text is rejected, and a live v2 regeneration is not duplicated."""
    gamma = _SEGMENTED.find("Gamma")
    texts = (_QUOTE, _POEM, _SEGMENTED, _DIALOGUE, _QUOTED_REFUSAL)
    app = make_app(tmp_path / "policy")
    try:
        scenario, conditions, versions = _open_library(app, texts)
        blocks = scenario.block_ids()
        _shape_blocks(app, (
            (blocks[0], BlockType.QUOTE.value, None),
            (blocks[1], BlockType.POEM.value, None),
            (blocks[2], BlockType.PARAGRAPH.value, [[0, gamma, 0], [gamma, len(_SEGMENTED), 1]]),
        ))
        quote, poem, segmented, dialogue, quoted = blocks
        short = _block_payload("Один.")
        await _expect_validation(
            app, scenario, conditions, quote, short, policy=CURRENT_TRANSLATION_POLICY,
            reason="validation_error", fragment="абзацев",
        )
        await _expect_saved(
            app, scenario, conditions, quote, short, policy=TRANSLATION_CONTEXT_POLICY_V2,
        )
        poem_short = _block_payload("Роза красная.\n")
        await _expect_validation(
            app, scenario, conditions, poem, poem_short, policy=CURRENT_TRANSLATION_POLICY,
            reason="validation_error", fragment="непустых строк",
        )
        await _expect_saved(
            app, scenario, conditions, poem, poem_short, policy=TRANSLATION_CONTEXT_POLICY_V2,
        )
        partial_segments = _block_payload("", segments=(
            SegmentTranslationResponse(1, "Альфа.\nВторая."),
            SegmentTranslationResponse(2, "Гамма."),
        ))
        await _expect_validation(
            app, scenario, conditions, segmented, partial_segments, policy=CURRENT_TRANSLATION_POLICY,
            reason="validation_error", fragment="сегмента",
        )
        await _expect_saved(
            app, scenario, conditions, segmented, partial_segments, policy=TRANSLATION_CONTEXT_POLICY_V2,
        )
        await _expect_saved(
            app, scenario, conditions, dialogue, _block_payload("Она сказала.\n— И замолчала."),
            policy=CURRENT_TRANSLATION_POLICY,
        )
        for content, truncated in (("", True), ("{", True)):
            client = FakeLLMClient([FakeReply(content=content, truncated=True)])
            work = _open_call(
                app, scenario, conditions, client, quoted, policy_version=CURRENT_TRANSLATION_POLICY,
            )
            with pytest.raises(RecoveryError, match="output_truncated") as caught:
                await work.call.request(work.context, **work.request_args)
            assert caught.value.reason == "output_truncated"
            assert client.call_count == 1
            with app.unit_of_work.transaction() as tx:
                root = tx.work.get_root(work.root.id)
                assert (root.status, root.reason, root.calls_used) == ("stopped", "output_truncated", 1)
                assert tx.work.list_scopes(root.id)[0].status != "split"
                assert tx.results.get(work.run.id, quoted) is None
        truncated_line = FakeReply(
            content=json.dumps({
                "blocks": [{
                    "source_block_id": "b1", "treatment": "TRANSLATE",
                    "translated_text": "Один.", "glossary_hints": [],
                }],
                "language_notes": [],
            }),
            truncated=True,
        )
        client = FakeLLMClient([truncated_line])
        work = _open_call(app, scenario, conditions, client, quote, policy_version=CURRENT_TRANSLATION_POLICY)
        with pytest.raises(RecoveryError, match="output_truncated") as caught:
            await work.call.request(work.context, **work.request_args)
        assert caught.value.reason == "output_truncated"
        assert "абзацев" not in str(caught.value.original_error)
        assert client.call_count == 1
        refusal = LLMResponseFormatError(
            "провайдер отказался", truncated=False,
            violation=Violation("provider_refusal", "translation_unit", scope="unit", safe_message="отказ"),
        )
        filtered = LLMResponseFormatError(
            "фильтр", truncated=False,
            violation=Violation(
                "provider_content_filter", "translation_unit", scope="unit", safe_message="фильтр",
            ),
        )
        for error in (refusal, filtered):
            client = FakeLLMClient([error, translated_reply(("b1",))])
            work = _open_call(
                app, scenario, conditions, client, dialogue, policy_version=CURRENT_TRANSLATION_POLICY,
            )
            with pytest.raises(RecoveryError, match="provider_refusal") as caught:
                await work.call.request(work.context, **work.request_args)
            assert caught.value.reason == "provider_refusal"
            assert client.call_count == 1
            with app.unit_of_work.transaction() as tx:
                root = tx.work.get_root(work.root.id)
                assert root.reason == "provider_refusal"
                assert len(tx.work.list_scopes(root.id)) == 1
                assert tx.work.list_scopes(root.id)[0].status != "split"
        help_payload = TranslationResultResponse(
            blocks=(BlockResultResponse("b1", BlockTreatment.TRANSLATE, _SHORT_RU),),
            fragment_assistances=(FragmentAssistanceResponse("d-1", ("p-1",), "lexical_hint", _REFUSAL),),
        )
        await _expect_validation(
            app, scenario, conditions, dialogue, help_payload, policy=CURRENT_TRANSLATION_POLICY,
            reason="validation_error", fragment="cannot",
        )
        await _expect_saved(
            app, scenario, conditions, quoted,
            _block_payload('Он сказал: "I can\'t fulfill this request."'),
            policy=CURRENT_TRANSLATION_POLICY,
        )
        note_payload = TranslationResultResponse(
            blocks=(BlockResultResponse("b1", BlockTreatment.TRANSLATE, _SHORT_RU),),
            language_notes=(LanguageNoteResponse("b1", "said", "Короткая заметка о речи.", _REFUSAL),),
        )
        await _expect_validation(
            app, scenario, conditions, quoted, note_payload, policy=CURRENT_TRANSLATION_POLICY,
            reason="validation_error", fragment="cannot",
        )
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_conditions(conditions.id).payload["versions"] == versions
    finally:
        await app.aclose()

    split_fake = FakeLLMClient([
        analysis_reply(2),
        translated_reply(("b1", "b2"), truncated=True),
        translation_reply(1),
        translation_reply(1),
    ])
    split_app = make_app(tmp_path / "split", split_fake)
    try:
        _doc, _run, _task = await prepare(split_app)
        finished = await drain_tasks(split_app)
        assert finished[-1].status is TaskStatus.SUCCEEDED, finished[-1].last_error
        lengths = [len(request.ordered_block_ids) for request in _translation_requests(split_fake)]
        assert lengths == [2, 1, 1]
        assert _translation_requests(split_fake)[0].ordered_block_ids != (
            _translation_requests(split_fake)[1].ordered_block_ids
        )
    finally:
        await split_app.aclose()

    gateway_request = LLMRequest(
        kind=LLMResponseKind.TRANSLATION_RESULT, instructions="translate", input_text="text",
    )
    cases = (
        (_chat_document(finish_reason="length", content=None), "provider_truncated", True),
        (_chat_document(finish_reason="stop", content=None, refusal="no"), "provider_refusal", False),
        (_chat_document(finish_reason="content_filter", content="x"), "provider_content_filter", False),
    )
    for document, rule, truncated in cases:
        stub = _HttpStub(document=document)
        client = _gateway(stub)
        with pytest.raises(LLMResponseFormatError) as caught:
            await client.generate(gateway_request)
        assert stub.posts == 1
        assert caught.value.truncated is truncated
        assert caught.value.violation.rule_code == rule
        await client.aclose()

    interval_fake = FakeLLMClient([analysis_reply(10)])
    interval_app = make_app(tmp_path / "intervals", interval_fake)
    try:
        texts = (
            "Alpha.", "Poem one.", "Poem two.", "Quote one.", "Quote two.",
            "List one.", "List two.", "Cell one.", "Cell two.", "Omega.",
        )
        types = (
            BlockType.PARAGRAPH, BlockType.POEM, BlockType.POEM, BlockType.QUOTE, BlockType.QUOTE,
            BlockType.LIST_ITEM, BlockType.LIST_ITEM, BlockType.TABLE, BlockType.TABLE, BlockType.PARAGRAPH,
        )
        document = store_imported(interval_app, (texts,))
        ids = [block.id for block in document.sections[0].blocks]
        _shape_blocks(interval_app, tuple((block_id, block_type.value, None) for block_id, block_type in zip(ids, types)))
        await interval_app.analysis.start("book-1")
        analysed = await drain_tasks(interval_app)
        assert analysed[0].status is TaskStatus.SUCCEEDED, analysed[0].last_error
        run = interval_app.start_processing()(StartProcessingCommand("book-1"))
        await interval_app.processing.start("book-1")
        with interval_app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            intervals = tuple(tuple(group) for group in root.manifest["atomic_intervals"])
        assert intervals == (
            (ids[0],), (ids[1], ids[2]), (ids[3], ids[4]), (ids[5], ids[6]), (ids[7], ids[8]), (ids[9],),
        )
    finally:
        await interval_app.aclose()

    await _live_v2_regeneration(unit_of_work, task_queue)

    correction_app = make_app(tmp_path / "correction")
    try:
        scenario = store_processing_book(correction_app.unit_of_work, chapters=((SOURCE_TEXT,),))
        block_id = scenario.block_ids()[0]
        carried = "I can't fulfill this request."
        previous = BlockProcessingResult(
            block_id=block_id, treatment=BlockTreatment.TRANSLATE, translated_text=TRANSLATED_TEXT,
            language_notes=(LanguageNote("ln-ch01-001", block_id, _OLD_EXCERPT, _KEPT_NOTE, carried),),
        )
        with correction_app.unit_of_work.transaction() as tx:
            tx.results.replace_results(RUN_ID, (previous,))
            document = tx.documents.get(BOOK_ID)
            section = document.section(scenario.section().id)
            analysis = tx.analyses.get_for_section(BOOK_ID, section.id)
            run = tx.runs.get(RUN_ID)
        inputs = _correction_inputs(document, section, analysis, run, {block_id: previous})
        handler = CorrectChapter(
            unit_of_work=correction_app.unit_of_work, llm_client=None,
            snapshot=inputs.snapshot, validator=make_validator(),
        )
        from linguary.application.correction_dto import (
            ChapterCorrectionResponse,
            CorrectedBlockResponse,
        )
        kept = ChapterCorrectionResponse(blocks=(CorrectedBlockResponse(
            block_id, TRANSLATED_TEXT, (), (
                LanguageNoteResponse(block_id, _OLD_EXCERPT, _KEPT_NOTE, carried),
                LanguageNoteResponse(block_id, _NEW_EXCERPT, _ADDED_NOTE, "Пересказ фрагмента."),
            ),
        ),))
        failing = ChapterCorrectionResponse(blocks=(CorrectedBlockResponse(
            block_id, TRANSLATED_TEXT, (), (
                LanguageNoteResponse(block_id, _OLD_EXCERPT, _KEPT_NOTE, carried),
                LanguageNoteResponse(block_id, _NEW_EXCERPT, _ADDED_NOTE, _REFUSAL),
            ),
        ),))
        with correction_app.unit_of_work.transaction() as tx:
            corrected, = handler._candidates(tx, inputs, kept)
            assert [note.id for note in corrected.language_notes] == ["ln-ch01-001", "ln-ch01-002"]
            assert corrected.language_notes[0].translated_excerpt == carried
            assert _counter_in(tx) == 2
            with pytest.raises(ResultValidationError, match="cannot"):
                handler._candidates(tx, inputs, failing)
            assert _counter_in(tx) == 2
    finally:
        await correction_app.aclose()


async def test_ac8_rate_limit_pacing_after_pause():
    """60 RPM plus hold(20) admits at 20, 21, 22, and 23, without leaking a cancelled slot."""
    clock = ControlledClock()
    limiter = LLMRateLimiter(60, clock=clock, sleep=clock.sleep)
    limiter.hold(20.0)
    admitted: list[float] = []

    async def admit():
        await limiter.acquire()
        admitted.append(clock())

    waiters = [asyncio.create_task(admit()) for _ in range(3)]
    fourth = None
    try:
        await _settle(clock, admitted)
        clock.advance(20.0)
        await _settle(clock, admitted)
        clock.advance(20.01)
        fourth = asyncio.create_task(admit())
        await _settle(clock, admitted)
        pending = [*waiters, fourth]
        for _ in range(10):
            if all(task.done() for task in pending):
                break
            await _settle(clock, admitted)
            if clock.waiting.empty():
                continue
            deadline = clock.waiting.get_nowait()
            if deadline > clock.now:
                clock.advance(deadline)
            await _settle(clock, admitted)
        assert admitted == [20.0, 21.0, 22.0, 23.0]
    finally:
        for task in (*waiters, fourth):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*waiters, *([] if fourth is None else [fourth]), return_exceptions=True)

    fifo = ControlledClock()
    fifo_limiter = LLMRateLimiter(60, clock=fifo, sleep=fifo.sleep)
    fifo_admitted = []

    async def fifo_admit():
        await fifo_limiter.acquire()
        fifo_admitted.append(fifo())

    await fifo_admit()
    first = asyncio.create_task(fifo_admit())
    middle = None
    last = None
    try:
        assert await fifo.waiting.get() == 1.0
        middle = asyncio.create_task(fifo_admit())
        last = asyncio.create_task(fifo_admit())
        await asyncio.sleep(0)
        assert fifo.waiting.empty()
        middle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middle
        fifo.advance(1.0)
        await first
        assert fifo_admitted == [0.0, 1.0]
        assert await fifo.waiting.get() == 2.0
        fifo.advance(2.0)
        await last
        assert fifo_admitted == [0.0, 1.0, 2.0]
    finally:
        for task in (first, middle, last):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(first, *([middle] if middle else []), *([last] if last else []), return_exceptions=True)

    open_clock = ControlledClock()
    open_limiter = LLMRateLimiter(None, clock=open_clock, sleep=open_clock.sleep)
    open_limiter.hold(math.inf)
    immediate = asyncio.create_task(open_limiter.acquire())
    await asyncio.sleep(0)
    assert immediate.done()
    await immediate
    open_limiter.hold(10**9)
    capped = asyncio.create_task(open_limiter.acquire())
    assert await open_clock.waiting.get() == 120.0
    open_clock.advance(120.0)
    await capped

    shared = LLMRateLimiter(None, max_concurrency=1, clock=open_clock, sleep=open_clock.sleep)
    await shared.acquire_concurrency()
    queued = asyncio.create_task(shared.acquire_concurrency())
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    third = asyncio.create_task(shared.acquire_concurrency())
    await asyncio.sleep(0)
    assert not third.done()
    shared.release_concurrency()
    await asyncio.wait_for(third, 1)
    shared.release_concurrency()

    paced = ControlledClock()
    changing = LLMRateLimiter(1, clock=paced, sleep=paced.sleep)
    changed = []

    async def change_admit():
        await changing.acquire()
        changed.append(paced())

    await change_admit()
    head = asyncio.create_task(change_admit())
    follower = None
    try:
        assert await paced.waiting.get() == 60.0
        follower = asyncio.create_task(change_admit())
        await asyncio.sleep(0)
        paced.advance(10.0)
        changing.set_limit(60)
        for _ in range(20):
            if head.done():
                break
            await asyncio.sleep(0)
        await head
        assert changed == [0.0, 10.0]
        assert await paced.waiting.get() == 11.0
        paced.advance(11.0)
        await follower
        assert changed == [0.0, 10.0, 11.0]
    finally:
        for task in (head, follower):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(head, *([] if follower is None else [follower]), return_exceptions=True)


async def test_ac10_crash_recovery_and_overview_projection(tmp_path, monkeypatch):
    """Crash recovery runs only for a lone executor, and overview, retry, and SQLite locks stay explainable."""
    app = make_app(tmp_path / "lock", FakeLLMClient())
    try:
        store_imported(app)
        with app.unit_of_work.transaction() as tx:
            tx.tasks.add(ProcessingTask(
                id="task-interrupted", book_id="book-1", task_type=TaskType.ANALYZE_CHAPTER,
                scope_id="section-1",
            ).start())
        await drain_tasks(app)
        with app.unit_of_work.transaction() as tx:
            recovered = tx.tasks.get("task-interrupted")
        assert recovered.status is TaskStatus.FAILED
        assert recovered.last_error is not None and recovered.last_error.detail == "Interrupted"
        with app.unit_of_work.transaction() as tx:
            tx.tasks.add(ProcessingTask(
                id="task-live", book_id="book-1", task_type=TaskType.EXPORT_BOOK, scope_id="export",
            ).start())
        other = ExecutorLock(app.paths.executor_lock)
        assert other.join() is True
        other.share()
        try:
            await drain_tasks(app)
        finally:
            other.leave()
        with app.unit_of_work.transaction() as tx:
            assert tx.tasks.get("task-live").status is TaskStatus.RUNNING
    finally:
        await app.aclose()

    import linguary.cli.app as cli_app
    assert cli_app.drain_tasks is drain_tasks
    assert "await drain_tasks(application)" in inspect.getsource(cli_app._worker)

    recovery_app = make_app(tmp_path / "recover")
    try:
        scenario, conditions, _versions = _open_library(recovery_app, ("He returned home.", "She waited there."))
        first, second = scenario.block_ids()
        sections = [scenario.section(0).id]
        with recovery_app.unit_of_work.transaction() as tx:
            tx.results.replace_results(scenario.run_id, (
                BlockProcessingResult(first, BlockTreatment.TRANSLATE, _SHORT_RU),
            ))
            local_root, local_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(first,),
                status="pending",
                manifest=_restartable_manifest(
                    tx, conditions_id=conditions.id, run_id=scenario.run_id,
                    blocks=(first,), section_id=sections[0],
                ),
            )
            tx.work.begin_local_acceptance(
                local_root.id, local_scope.id, task_id=None, run_id=scenario.run_id,
                stage="translation_unit", assigned_model="pinned-model", prompt_hash="a" * 64,
                policy_version=local_root.policy_version,
            )
            rejected_root, rejected_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(second,),
                status="pending",
            )
            claimed = tx.work.claim_scope(rejected_scope.id, expected_generation=rejected_scope.generation)
            _plant_attempt(
                tx, rejected_root, claimed, scenario.run_id, state="rejected", ordinal=1,
            )
            local_generation = local_root.generation
            rejected_generation = rejected_root.generation
        holder = ExecutorLock(recovery_app.paths.executor_lock)
        assert holder.join() is True
        holder.share()
        try:
            await drain_tasks(recovery_app)
        finally:
            holder.leave()
        with recovery_app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(local_root.id).status == "pending"
            assert tx.work.get_root(local_root.id).generation == local_generation
            assert any(
                item.state == "validated" for item in tx.work.list_root_attempts(local_root.id)
            )
        await drain_tasks(recovery_app)
        with recovery_app.unit_of_work.transaction() as tx:
            recovered_root = tx.work.get_root(local_root.id)
            assert (recovered_root.status, recovered_root.reason) == ("stopped", "interrupted")
            assert recovered_root.generation == local_generation + 1
            history = tx.work.list_root_attempts(local_root.id)
            assert history and all(item.state != "validated" for item in history)
            assert any(item.state == "rejected" for item in history)
            reopened = tx.work.get_root(rejected_root.id)
            assert reopened.status != "stopped"
            assert reopened.generation == rejected_generation + 1
            assert tx.work.get_scope(claimed.id).status == "pending"
            stored = tx.results.get(scenario.run_id, first)
            assert stored is not None and stored.translated_text == _SHORT_RU
            sent_root, sent_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(first,),
                status="running",
            )
            _plant_attempt(tx, sent_root, sent_scope, scenario.run_id, state="sent", ordinal=1)
            unknown_root, unknown_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(first,),
                status="stopped", reason="unknown_outcome", calls_used=1,
            )
            _plant_attempt(tx, unknown_root, unknown_scope, scenario.run_id, state="unknown", ordinal=1)
            healthy_root, _healthy_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(second,),
                status="pending", reason="transient_error",
            )
            blocked = _failed_task(tx, unknown_root, sections[0])
            allowed = _failed_task(tx, healthy_root, sections[0])
            limit_root, _limit_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(second,),
                status="stopped", reason="limit", calls_used=3,
            )
            _plant_attempt(tx, limit_root, _limit_scope, scenario.run_id, state="unknown", ordinal=1)
            stale_root, _stale_scope = _plant_root(
                tx, conditions_id=conditions.id, run_id=scenario.run_id, blocks=(second,),
                status="stopped", reason="limit", calls_used=3,
            )
        overview = book_overview(recovery_app.unit_of_work, "book-1")
        roots = {item["root_id"]: item for item in overview.recovery["roots"]}
        assert roots[sent_root.id]["in_flight"] is True
        assert roots[sent_root.id]["has_unknown_outcome"] is False
        assert roots[sent_root.id]["reason"] != "unknown"
        assert roots[unknown_root.id]["can_retry"] is False
        assert roots[unknown_root.id]["has_unknown_outcome"] is True
        assert roots[healthy_root.id]["can_retry"] is True
        assert overview.recovery["can_retry"] is True
        detail = BookDetailResponse.of_overview(overview)
        failures = {item.task_id: item for item in detail.progress.failures}
        assert failures[blocked.id].is_retryable is False
        assert failures[allowed.id].is_retryable is True
        with pytest.raises(RetryTaskError, match="явный перезапуск"):
            recovery_app.retry_task(RetryTaskCommand(task_id=blocked.id))
        retried = recovery_app.retry_task(RetryTaskCommand(task_id=allowed.id))
        assert retried.status is TaskStatus.PENDING
        with pytest.raises(RestartTranslationError, match=_UNKNOWN_RESTART):
            _restart(recovery_app, recovered_root)
        acknowledged = _restart(recovery_app, recovered_root, acknowledge=True)
        assert acknowledged.successor_root_id != recovered_root.id
        with recovery_app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(limit_root.id).extra_calls == 0
            assert tx.work.get_root(stale_root.id).extra_calls == 0
        with pytest.raises(ExtendRecoveryLimitConflict) as unknown_limit:
            recovery_app.extend_recovery_limit(ExtendRecoveryLimitCommand(
                "book-1", limit_root.id, 2, limit_root.generation,
            ))
        assert unknown_limit.value.reason_code == "unknown"
        with pytest.raises(ExtendRecoveryLimitConflict) as stale_limit:
            recovery_app.extend_recovery_limit(ExtendRecoveryLimitCommand(
                "book-1", stale_root.id, 2, stale_root.generation + 5,
            ))
        assert stale_limit.value.reason_code == "generation_conflict"
        with recovery_app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(limit_root.id).extra_calls == 0
            assert tx.work.get_root(stale_root.id).extra_calls == 0
            assert tx.results.get(scenario.run_id, first).translated_text == _SHORT_RU
    finally:
        await recovery_app.aclose()

    stalled = make_app(tmp_path / "stalled")
    try:
        store_processing_book(stalled.unit_of_work)
        response = BookDetailResponse.of_overview(book_overview(stalled.unit_of_work, "book-1"))
        assert response.book.status == "PROCESSING"
        assert response.progress.pending_tasks == 0
        assert response.progress.running_tasks == 0
        assert not response.progress.failures
        assert _is_stalled(response) is True
        assert _needs_refresh(response) is False
    finally:
        await stalled.aclose()

    diag_app = make_app(tmp_path / "diag")
    try:
        diagnostics = Diagnostics(diag_app.unit_of_work)
        calls = {"n": 0}

        def finalize(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PersistenceError("база данных занята")
            return "stored"

        diagnostics._finalize_attempt = finalize
        with pytest.raises(PersistenceError, match="занята"):
            diagnostics.finalize_attempt("attempt-1", "rejected")
        assert diagnostics._failed is False
        assert diagnostics.finalize_attempt("attempt-1", "rejected") == "stored"
    finally:
        await diag_app.aclose()

    busy_engine = create_database_engine(tmp_path / "busy.db")
    Base.metadata.create_all(busy_engine)
    from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
    from datetime import UTC, datetime
    busy_uow = SqlAlchemyUnitOfWork.for_engine(busy_engine)
    with busy_uow.transaction() as tx:
        tx.books.add(BookRecord(
            "book-busy", BookMetadata("Book", ("Author",)), "en", "ru", BookStatus.PROCESSING,
            "book.epub", datetime.now(UTC),
        ))
    store = SqliteTaskStore.for_engine(busy_engine, id_factory=iter(["task-1", "task-2"]).__next__)
    store.enqueue(TaskIdentity(book_id="book-busy", task_type=TaskType.EXPORT_BOOK))
    sleeps = []
    script = {"remaining": 1, "message": "database is busy"}
    import linguary.adapters.runtime.task_store as task_store_module
    monkeypatch.setattr(task_store_module, "sleep", lambda seconds: sleeps.append(seconds), raising=False)
    original_exec = Connection.exec_driver_sql

    def exec_driver_sql(self, statement, *args, **kwargs):
        sql = str(statement)
        if "BEGIN IMMEDIATE" in sql and script["remaining"] > 0:
            script["remaining"] -= 1
            raise OperationalError(sql, {}, sqlite3.OperationalError(script["message"]))
        return original_exec(self, statement, *args, **kwargs)

    monkeypatch.setattr(Connection, "exec_driver_sql", exec_driver_sql)
    claimed = store.claim_next_pending()
    assert claimed is not None and claimed.status is TaskStatus.RUNNING
    assert sleeps == [0.005]
    store.enqueue(TaskIdentity(book_id="book-busy", task_type=TaskType.IMPORT_BOOK))
    sleeps.clear()
    script["remaining"] = 1
    script["message"] = "disk I/O error"
    with pytest.raises(OperationalError, match="disk I/O error"):
        store.claim_next_pending()
    assert sleeps == []
    busy_engine.dispose()
