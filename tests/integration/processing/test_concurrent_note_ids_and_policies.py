"""Shared note counters and regeneration policy compatibility."""

from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace
import threading
from uuid import uuid4

import pytest
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    _ensure_test_translation_conditions,
    make_validator,
    store_processing_book,
)
from sqlalchemy.orm import Session
from test_regenerate_block import make_regenerate_block

from linguary.adapters.db.models import NoteNumberCounterRow
from linguary.adapters.llm import FakeLLMClient
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS_V3,
    ChapterCorrectionResponse,
    CorrectionSnapshotV2,
    CorrectedBlockResponse,
)
from linguary.application.llm_dto import LanguageNoteResponse
from linguary.application.retry_policy import ResultValidationError
from linguary.application.translation_policy import TRANSLATION_CONTEXT_POLICY_V2
from linguary.application.use_cases.correct_chapter import (
    CorrectChapter,
    prepare_correction_candidates,
)
from linguary.application.use_cases.regenerate_block import (
    BlockRegenerationSequence,
    RegenerateBlockCommand,
)
from linguary.domain import BlockTreatment, ProcessingTask, TaskStatus, TaskType
from linguary.domain.content_plan import SourceAnchor, canonical_sha
from linguary.domain.correction import (
    CORRECTION_POLICY_VERSION_V3,
    CorrectionCandidate,
)
from linguary.domain.learning import FragmentAssistance, LanguageNote
from linguary.domain.processing import BlockProcessingResult
from linguary.domain.recovery import RecoveryRoot, RecoveryScope

_OLD_EXCERPT = "a truth universally acknowledged"
_NEW_EXCERPT = "a man must want a wife"
_KEPT_NOTE = "Ироническая формула зачина романа."
_ADDED_NOTE = "Заметка о том, что речь идёт о браке."
_REGEN_NOTE = "Повторное объяснение того же оборота."
_REGEN_TEXT = "Всем известно: одинокий богач непременно ищет себе жену."


def _counter(engine, run_id: str = RUN_ID, chapter: int = 1) -> int | None:
    with Session(engine) as session:
        row = session.get(NoteNumberCounterRow, (run_id, chapter))
        return None if row is None else row.last_ordinal


def _counter_in(tx, run_id: str = RUN_ID, chapter: int = 1) -> int | None:
    row = tx.results._session.get(NoteNumberCounterRow, (run_id, chapter))
    return None if row is None else row.last_ordinal


def _snapshot() -> CorrectionSnapshotV2:
    return CorrectionSnapshotV2(
        model="gpt-test",
        route={
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "proxy": "",
            "route_id": "r1",
        },
        parameters={
            "temperature": 0.2,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V3,
        policy_version=CORRECTION_POLICY_VERSION_V3,
    )


def _conditions(*, english_help: bool = False):
    return SimpleNamespace(
        id="cond-recheck",
        payload={
            "schema_version": 1,
            "versions": {"validators": 3},
            "reader_level": "b1",
            "english_help": english_help,
        },
    )


def _correction_inputs(document, section, analysis, run, results, *, english_help: bool = False):
    from linguary.application.correction import CorrectionInputs

    return CorrectionInputs(
        document=document,
        section=section,
        analysis=analysis,
        run=run,
        conditions=_conditions(english_help=english_help),
        snapshot=_snapshot(),
        blocks=section.ordered_blocks,
        results=dict(results),
        bindings={block.id: None for block in section.ordered_blocks},
    )


def _notes_response(block_id: str, excerpts: tuple[tuple[str, str], ...]) -> ChapterCorrectionResponse:
    return ChapterCorrectionResponse(
        blocks=(
            CorrectedBlockResponse(
                block_id,
                TRANSLATED_TEXT,
                (),
                tuple(
                    LanguageNoteResponse(block_id, excerpt, note, "Пересказ фрагмента.")
                    for excerpt, note in excerpts
                ),
            ),
        )
    )


class _BarrierClient:
    def __init__(self, inner, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._barrier = barrier

    async def generate(self, request):
        self._barrier.wait(timeout=10)
        return await self._inner.generate(request)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _stored_note_ids(unit_of_work) -> list[str]:
    with unit_of_work.transaction() as tx:
        return [note.id for result in tx.results.list_for_run(RUN_ID) for note in result.language_notes]


async def test_concurrent_chapter_regeneration_and_correction_never_collide_on_note_ids(
    unit_of_work, engine
) -> None:
    """Regeneration and correction take distinct counter ids for one chapter."""
    scenario = store_processing_book(unit_of_work)
    first, second = scenario.block_ids()
    with unit_of_work.transaction() as tx:
        tx.results.replace_results(
            RUN_ID,
            (
                BlockProcessingResult(
                    block_id=first,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text=TRANSLATED_TEXT,
                    language_notes=(
                        LanguageNote("ln-ch01-001", first, _OLD_EXCERPT, _KEPT_NOTE),
                    ),
                ),
                BlockProcessingResult(
                    block_id=second,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text=TRANSLATED_TEXT,
                ),
            ),
        )
        document = tx.documents.get(BOOK_ID)
        section = document.section(scenario.section().id)
        analysis = tx.analyses.get_for_section(BOOK_ID, section.id)
        run = tx.runs.get(RUN_ID)
        stored = {item.block_id: item for item in tx.results.list_for_run(RUN_ID)}
    inputs = _correction_inputs(document, section, analysis, run, stored)
    handler = CorrectChapter(
        unit_of_work=unit_of_work,
        llm_client=None,
        snapshot=inputs.snapshot,
        validator=make_validator(),
    )
    response = _notes_response(first, ((_OLD_EXCERPT, _KEPT_NOTE), (_NEW_EXCERPT, _ADDED_NOTE)))
    barrier = threading.Barrier(2)
    holder: dict[str, object] = {}

    def correct() -> None:
        try:
            barrier.wait(timeout=10)
            with unit_of_work.transaction() as tx:
                corrected, = handler._candidates(tx, inputs, response)
            holder["ids"] = [note.id for note in corrected.language_notes]
        except Exception as exc:
            # The main thread checks this after joining the worker.
            holder["error"] = exc

    thread = threading.Thread(target=correct)
    thread.start()
    client = _BarrierClient(
        FakeLLMClient.always(
            __import__(
                "processing_scenario", fromlist=["translation_reply"]
            ).translation_reply(
                [__import__("processing_scenario", fromlist=["block_reply"]).block_reply(
                    "b1", translated_text=_REGEN_TEXT
                )],
                notes=(
                    __import__("processing_scenario", fromlist=["note_reply"]).note_reply(
                        "b1", original_excerpt=_NEW_EXCERPT, note=_REGEN_NOTE
                    ),
                ),
            )
        ),
        barrier,
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)
    await regenerate(RegenerateBlockCommand(BOOK_ID, second))
    thread.join(timeout=15)

    assert "error" not in holder
    assert not thread.is_alive()
    regenerated = _stored_note_ids(unit_of_work)
    corrected_ids = holder["ids"]
    assert corrected_ids[0] == "ln-ch01-001"
    # The untouched block and the correction both keep ln-ch01-001. Fresh ids must not repeat.
    fresh = [note_id for note_id in (*regenerated, *corrected_ids) if note_id != "ln-ch01-001"]
    assert len(fresh) == len(set(fresh))
    assert _counter(engine) == 3
    assert set(regenerated) | set(corrected_ids) == {"ln-ch01-001", "ln-ch01-002", "ln-ch01-003"}


async def test_regenerate_block_does_not_duplicate_live_v2_operation_under_v3(
    unit_of_work, task_queue
) -> None:
    """A live v2 regeneration stays the active operation after the policy moves to v3."""
    scenario = store_processing_book(unit_of_work)
    block_id = scenario.block_ids()[0]
    _, conditions_id = _ensure_test_translation_conditions(unit_of_work)
    root_id = str(uuid4())
    with unit_of_work.transaction() as tx:
        tx.work.add_root(
            RecoveryRoot(
                root_id,
                BOOK_ID,
                conditions_id,
                str(uuid4()),
                {
                    "schema_version": 1,
                    "active_run_id": RUN_ID,
                    "ordered_blocks": [block_id],
                },
                status="running",
                policy_version=TRANSLATION_CONTEXT_POLICY_V2,
                operation_kind="regeneration",
            )
        )
        tx.work.add_scope(RecoveryScope(str(uuid4()), root_id, (block_id,)))
        tx.tasks.add(
            ProcessingTask(
                id="live-v2-task",
                book_id=BOOK_ID,
                task_type=TaskType.REGENERATE_BLOCK,
                scope_id=block_id,
                status=TaskStatus.RUNNING,
                operation_id=root_id,
                policy_version=TRANSLATION_CONTEXT_POLICY_V2,
            )
        )
        tx.work.bind_operation_refs(
            "live-v2-task",
            book_id=BOOK_ID,
            conditions_id=conditions_id,
            operation_refs=({"root_id": root_id, "policy_version": TRANSLATION_CONTEXT_POLICY_V2},),
        )
    regenerate, _ = make_regenerate_block(
        unit_of_work, FakeLLMClient.always(__import__("json").dumps({}))
    )
    sequence = BlockRegenerationSequence(
        unit_of_work=unit_of_work, queue=task_queue, regenerate_block=regenerate
    )

    task = await sequence.start(BOOK_ID, block_id)

    assert task.id == "live-v2-task"
    assert task.policy_version == TRANSLATION_CONTEXT_POLICY_V2
    with unit_of_work.transaction() as tx:
        roots = tx.work.list_roots(BOOK_ID, run_id=RUN_ID)
        tasks = [
            item
            for item in tx.tasks.list_for_book(BOOK_ID)
            if item.task_type is TaskType.REGENERATE_BLOCK and item.scope_id == block_id
        ]
    assert [item.id for item in roots] == [root_id]
    assert [item.id for item in tasks] == ["live-v2-task"]


async def test_new_regeneration_supersedes_stopped_regeneration_root(
    unit_of_work, task_queue
) -> None:
    """A new regeneration root replaces the stopped root of the same block and run."""
    scenario = store_processing_book(unit_of_work)
    block_id = scenario.block_ids()[0]
    _, conditions_id = _ensure_test_translation_conditions(unit_of_work)
    stopped_id = str(uuid4())
    with unit_of_work.transaction() as tx:
        tx.work.add_root(
            RecoveryRoot(
                stopped_id,
                BOOK_ID,
                conditions_id,
                str(uuid4()),
                {
                    "schema_version": 1,
                    "active_run_id": RUN_ID,
                    "ordered_blocks": [block_id],
                },
                status="stopped",
                reason="limit",
                policy_version=TRANSLATION_CONTEXT_POLICY_V2,
                operation_kind="regeneration",
            )
        )
        tx.work.add_scope(RecoveryScope(str(uuid4()), stopped_id, (block_id,)))
    regenerate, _ = make_regenerate_block(unit_of_work, FakeLLMClient([]))
    sequence = BlockRegenerationSequence(
        unit_of_work=unit_of_work, queue=task_queue, regenerate_block=regenerate
    )

    task = await sequence.start(BOOK_ID, block_id)

    with unit_of_work.transaction() as tx:
        stopped = tx.work.get_root(stopped_id)
        successor = tx.work.get_root(task.operation_id)
    assert successor is not None
    assert successor.id != stopped_id
    assert successor.operation_kind == "regeneration"
    assert successor.superseded_by is None
    assert stopped.superseded_by == successor.id


async def test_failed_regeneration_superseded_by_new_regeneration_reaches_ready_to_export(
    tmp_path,
) -> None:
    """Failed regeneration task_1 is marked SUPERSEDED with cleared error when task_2 starts, and book returns to READY_TO_EXPORT."""
    from linguary.application.book_overview import book_overview
    from linguary.application.ports.llm import LLMProviderError
    from linguary.composition import drain_tasks
    from linguary.domain import BookStatus
    from tests.correction_fakes import CorrectionAwareFakeLLMClient
    from test_book_conditions import make_app
    from test_recovery_commit import analysis_reply, prepare, translation_reply

    fake = CorrectionAwareFakeLLMClient([
        analysis_reply(1),
        translation_reply(1),
        LLMProviderError("hard provider failure"),
        translation_reply(1),
    ])
    app = make_app(tmp_path, fake)
    try:
        document, _run, _ = await prepare(app, blocks=1)
        block_id = document.sections[0].blocks[0].id
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.SUCCEEDED

        # First regeneration fails -> stopped regeneration root + FAILED task
        task_1 = await app.regeneration.start("book-1", block_id)
        failed = await drain_tasks(app)
        assert failed[-1].id == task_1.id
        assert failed[-1].status is TaskStatus.FAILED
        with app.unit_of_work.transaction() as tx:
            root_1 = tx.work.get_root(task_1.operation_id)
            assert root_1.operation_kind == "regeneration"
            tx.work.stop_root(root_1.id, "provider_error", expected_generation=root_1.generation)
            stopped_gen = tx.work.get_root(root_1.id).generation

        # Start second regeneration -> supersedes root_1 and task_1 (clearing error fields)
        task_2 = await app.regeneration.start("book-1", block_id)
        assert task_2.id != task_1.id
        with app.unit_of_work.transaction() as tx:
            old_task = tx.tasks.get(task_1.id)
            old_root = tx.work.get_root(root_1.id)
            new_root = tx.work.get_root(task_2.operation_id)
            assert old_task.status is TaskStatus.SUPERSEDED
            assert old_task.last_error is None
            assert old_root.superseded_by == new_root.id
            assert old_root.generation == stopped_gen + 1

        # Complete task_2 -> book is READY_TO_EXPORT and epub_ready is True
        finished = await drain_tasks(app)
        assert any(t.id == task_2.id and t.status is TaskStatus.SUCCEEDED for t in finished)
        assert all(t.status is TaskStatus.SUCCEEDED for t in finished), [t.last_error for t in finished]
        with app.unit_of_work.transaction() as tx:
            assert tx.books.get("book-1").status is BookStatus.READY_TO_EXPORT
        overview = book_overview(app.unit_of_work, "book-1")
        assert overview.epub_ready is True
    finally:
        await app.aclose()


def test_correction_preserves_old_ids_and_candidate_recheck_burns_zero_counter(
    unit_of_work, engine
) -> None:
    """Kept notes reuse ids; a late rejection and a candidate recheck do not spend numbers."""
    scenario = store_processing_book(unit_of_work, chapters=((SOURCE_TEXT,),))
    block_id = scenario.block_ids()[0]
    previous = BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=TRANSLATED_TEXT,
        language_notes=(LanguageNote("ln-ch01-001", block_id, _OLD_EXCERPT, _KEPT_NOTE),),
    )
    with unit_of_work.transaction() as tx:
        tx.results.replace_results(RUN_ID, (previous,))
        document = tx.documents.get(BOOK_ID)
        section = document.section(scenario.section().id)
        analysis = tx.analyses.get_for_section(BOOK_ID, section.id)
        run = tx.runs.get(RUN_ID)
    inputs = _correction_inputs(
        document, section, analysis, run, {block_id: previous}
    )
    handler = CorrectChapter(
        unit_of_work=unit_of_work,
        llm_client=None,
        snapshot=inputs.snapshot,
        validator=make_validator(),
    )
    quote = _NEW_EXCERPT
    start = SOURCE_TEXT.index(quote)
    assisted = previous.__class__(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=TRANSLATED_TEXT,
        language_notes=previous.language_notes,
        fragment_assistances=(
            FragmentAssistance(
                "d-marriage",
                ("p1",),
                "sentence_rephrase",
                "Короткое пояснение смысла.",
                (SourceAnchor(block_id, start, start + len(quote), quote),),
            ),
        ),
    )
    failing = _correction_inputs(
        document,
        section,
        analysis,
        run,
        {block_id: assisted},
        english_help=True,
    )
    response = _notes_response(
        block_id, ((_OLD_EXCERPT, _KEPT_NOTE), (_NEW_EXCERPT, _ADDED_NOTE))
    )
    with unit_of_work.transaction() as tx:
        corrected, = handler._candidates(tx, inputs, response)
        assert [note.id for note in corrected.language_notes] == ["ln-ch01-001", "ln-ch01-002"]
        assert _counter_in(tx) == 2
        with pytest.raises(ResultValidationError, match="дублирует защищённую помощь"):
            handler._candidates(tx, failing, response)
        assert _counter_in(tx) == 2
        payload = json.loads(json.dumps({
            "schema_version": 1,
            "book_id": BOOK_ID,
            "run_id": RUN_ID,
            "conditions_id": "cond-recheck",
            "section_id": section.id,
            "block_id": block_id,
            "input_sha256": inputs.checksums[block_id],
            "input_generation": 1,
            "group_id": "group-1",
            "proposal_attempt_id": "attempt-1",
            "original": {"text": SOURCE_TEXT},
            "current_result": asdict(previous),
            "result": asdict(corrected),
            "proposal": {
                "block_id": block_id,
                "status": "corrected",
                "translated_text": TRANSLATED_TEXT,
                "glossary_hints": [],
                "language_notes": [
                    {
                        "source_block_id": block_id,
                        "original_excerpt": excerpt,
                        "note": note,
                        "translated_excerpt": "Пересказ фрагмента.",
                    }
                    for excerpt, note in ((_OLD_EXCERPT, _KEPT_NOTE), (_NEW_EXCERPT, _ADDED_NOTE))
                ],
                "issues": [
                    {
                        "field_path": "translated_text",
                        "source_quote": "It",
                        "current_evidence": {
                            "kind": "quote",
                            "field_path": "translated_text",
                            "quote": "Общепризнанная",
                        },
                        "problem": "Неточность",
                        "minimal_fix_reason": "Уточнить",
                    }
                ],
            },
        }))
        candidate = CorrectionCandidate(
            candidate_id="cand-1",
            book_id=BOOK_ID,
            run_id=RUN_ID,
            conditions_id="cond-recheck",
            section_id=section.id,
            block_id=block_id,
            input_sha256=inputs.checksums[block_id],
            input_generation=1,
            proposal_sha256=canonical_sha(payload),
            payload=payload,
            group_id="group-1",
            proposal_attempt_id="attempt-1",
        )
        reprepared, = prepare_correction_candidates(
            tx, inputs, (candidate,), make_validator()
        )
        assert [note.id for note in reprepared.language_notes] == ["ln-ch01-001", "ln-ch01-002"]
        assert _counter_in(tx) == 2
    assert _counter(engine) == 2
