"""Unit tests for note numbering (`story-v1`), writer synchronization, and v3 `translated_excerpt` validation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from processing_fakes import make_validator
from test_approve_glossary import (
    BOOK_ID,
    CONFIDENCE_THRESHOLD,
    CORRECTION_SETTINGS,
    LLM_SETTINGS,
    ProcessingWork,
    approve,
    command,
    glossary_book,
    hint_candidate,
    make_analysis_candidate,
)

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.block_processing import TranslationCall, note_chapter_number
from linguary.application.context_builder import ContextBuilder
from linguary.application.correction import correction_inputs
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS,
    ChapterCorrectionResponse,
    CorrectedBlockResponse,
    CorrectionSnapshot,
)
from linguary.application.llm_dto import LanguageNoteResponse
from linguary.application.recovery import RecoveryError
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.translation_policy import (
    CURRENT_TRANSLATION_POLICY,
    TRANSLATION_RELIABILITY_POLICY_V3,
)
from linguary.application.use_cases.approve_glossary import (
    GlossaryDecision,
    StartProcessing,
    StartProcessingCommand,
)
from linguary.application.use_cases.correct_chapter import CorrectChapter
from linguary.application.use_cases.process_chapter import ProcessChapter, ProcessChapterCommand
from linguary.application.use_cases.regenerate_block import RegenerateBlock, RegenerateBlockCommand
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    ChapterAnalysis,
    CoverageStatus,
    DocumentBlock,
    DocumentSection,
    GlossarySnapshot,
    LLMCallStage,
    ProcessingRun,
    ProcessingRunStatus,
    SourceReference,
    TranslationUnit,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.content_plan import ContentPlan
from linguary.domain.llm_log import CallContext
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope


def _doc_block(
    section_id: str,
    order: int,
    text: str,
    *,
    kind: BlockType = BlockType.PARAGRAPH,
    segments: tuple[BlockSegment, ...] = (),
) -> DocumentBlock:
    return DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=section_id,
        type=kind,
        order=order,
        text=text,
        source_reference=SourceReference(f"{section_id}.xhtml", f"/body/p[{order + 1}]"),
        segments=segments,
    )


def _doc_section(
    section_id: str,
    order: int,
    title: str,
    blocks: tuple[DocumentBlock, ...],
    *,
    resource_path: str | None = None,
) -> DocumentSection:
    return DocumentSection(
        id=section_id,
        document_id=BOOK_ID,
        title=title,
        order=order,
        source_resource_path=resource_path or f"{section_id}.xhtml",
        blocks=blocks,
    )


def _book_document(sections: tuple[DocumentSection, ...]) -> CanonicalDocument:
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Story Book", authors=("Author",)),
        source_language="en",
        target_language="ru",
        sections=sections,
    )


def test_note_chapter_number_reading_order_and_story_v1() -> None:
    """`story-v1` numbers story chapters 1..N and non-story sections N+1..N+M (>=1 even when N==0)."""
    title = _doc_section("title", 0, "Title Page", (_doc_block("title", 0, "Title"),), resource_path="title.xhtml")
    ch1 = _doc_section("ch1", 1, "Chapter 1", (_doc_block("ch1", 0, "First story paragraph."),))
    notes = _doc_section(
        "notes",
        2,
        "Notes",
        (_doc_block("notes", 0, "Footnote text.", kind=BlockType.FOOTNOTE),),
        resource_path="notes.xhtml",
    )
    ch2 = _doc_section("ch2", 3, "Chapter 2", (_doc_block("ch2", 0, "Second story paragraph."),))
    document = _book_document((title, ch1, notes, ch2))

    assert document.is_front_matter(title) is True
    assert notes.is_notes_only is True
    assert document.is_front_matter(ch1) is False and ch1.is_notes_only is False
    assert document.is_front_matter(ch2) is False and ch2.is_notes_only is False

    # Default / legacy "reading-order" preserves spine order
    assert [note_chapter_number(document, sec) for sec in document.ordered_sections] == [1, 2, 3, 4]
    assert [
        note_chapter_number(document, sec, note_numbering="reading-order")
        for sec in document.ordered_sections
    ] == [1, 2, 3, 4]

    # "story-v1": story chapters -> 1..N (1, 2); non-story -> N + non_story_index (3, 4)
    assert note_chapter_number(document, ch1, note_numbering="story-v1") == 1
    assert note_chapter_number(document, ch2, note_numbering="story-v1") == 2
    assert note_chapter_number(document, title, note_numbering="story-v1") == 3
    assert note_chapter_number(document, notes, note_numbering="story-v1") == 4

    # When N == 0 (only front-matter and notes-only sections), numbers still start at >= 1
    non_story_only = _book_document((title, notes))
    assert note_chapter_number(non_story_only, title, note_numbering="story-v1") == 1
    assert note_chapter_number(non_story_only, notes, note_numbering="story-v1") == 2


def test_start_processing_records_story_v1_note_numbering_in_run_binding() -> None:
    """`StartProcessing` records `'note_numbering': 'story-v1'` in the `run` binding JSON payload."""
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())
    approve(unit_of_work)(command(GlossaryDecision.ACCEPT))
    conditions = SimpleNamespace(
        id="conditions-1",
        payload={
            "route": {"provider": LLM_SETTINGS.provider},
            "models": {"processing": LLM_SETTINGS.model},
            "llm_parameters": {
                "temperature": LLM_SETTINGS.temperature,
                "max_output_tokens": LLM_SETTINGS.max_output_tokens,
            },
        },
    )
    work = ProcessingWork(conditions)
    unit_of_work.work = work
    starter = StartProcessing(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        llm_settings_snapshot=LLM_SETTINGS,
        correction_snapshot=CORRECTION_SETTINGS,
        book_conditions=SimpleNamespace(require=lambda *args, **kwargs: conditions),
    )

    run = starter(StartProcessingCommand(book_id=BOOK_ID))

    binding = work.bindings[("run", run.id)]
    assert binding["payload"]["note_numbering"] == "story-v1"


class _InMemoryNoteNumberingUow:
    """Minimal in-memory UoW verifying ProcessChapter, RegenerateBlock, and CorrectChapter note numbering."""

    def __init__(
        self,
        document: CanonicalDocument,
        analyses: dict[str, ChapterAnalysis],
        run: ProcessingRun,
        *,
        run_binding: dict | None,
        with_work: bool = True,
    ) -> None:
        self._document = document
        self._analyses = analyses
        self._run = run
        self._results: dict[str, BlockProcessingResult] = {}
        self._note_counters: dict[tuple[str, int], int] = {}
        self._bindings: dict[tuple[str, str], dict] = {}
        if run_binding is not None:
            self._bindings[("run", run.id)] = run_binding
        self._conditions = SimpleNamespace(
            id=str(uuid4()),
            payload={
                "schema_version": 1,
                "reader_level": "b1",
                "versions": {"validators": 3},
                "models": {"processing": "gpt-test"},
                "checks": {"language_notes_per_chapter": 10},
            },
        )
        self._roots: dict[str, RecoveryRoot] = {}
        self._scopes: dict[str, RecoveryScope] = {}
        self._attempts: dict[str, CallAttempt] = {}
        self._attempt_counts: dict[str, int] = {}
        self._scope_ordinals: dict[str, int] = {}
        self._tasks: list = []
        self._artifacts: list[dict] = []
        self._with_work = with_work
        self._uow = self

    @contextmanager
    def transaction(self):
        yield self._tx()

    def _tx(self):
        parent = self

        class _Docs:
            def get(self, book_id: str) -> CanonicalDocument | None:
                return parent._document if book_id == parent._document.id else None

        class _Analyses:
            def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
                return tuple(parent._analyses.values())

            def get_for_section(self, book_id: str, section_id: str) -> ChapterAnalysis | None:
                return parent._analyses.get(section_id)

        class _Runs:
            def active_for_book(self, book_id: str) -> ProcessingRun | None:
                return parent._run if book_id == parent._document.id else None

        class _Results:
            def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
                return tuple(parent._results.values())

            def list_for_section(self, run_id: str, section_id: str) -> tuple[BlockProcessingResult, ...]:
                sec_ids = {b.id for b in parent._document.section(section_id).blocks}
                return tuple(r for r in parent._results.values() if r.block_id in sec_ids)

            def get_many(self, run_id: str, block_ids: tuple[str, ...]) -> dict[str, BlockProcessingResult]:
                return {bid: parent._results[bid] for bid in block_ids if bid in parent._results}

            def get(self, run_id: str, block_id: str) -> BlockProcessingResult | None:
                return parent._results.get(block_id)

            def replace_results(self, run_id: str, items: tuple[BlockProcessingResult, ...]) -> None:
                for item in items:
                    parent._results[item.block_id] = item

            def allocate_note_ids(self, run_id: str, chapter_num: int, count: int) -> tuple[str, ...]:
                assert chapter_num >= 1
                key = (run_id, chapter_num)
                base = parent._note_counters.get(key, 0)
                issued = tuple(f"ln-ch{chapter_num:02d}-{base + i:03d}" for i in range(1, count + 1))
                parent._note_counters[key] = base + count
                return issued

        class _Work:
            def latest_conditions(self, book_id: str):
                return parent._conditions

            def get_conditions(self, conditions_id: str):
                return parent._conditions

            def get_binding(self, kind: str, entity_id: str):
                return parent._bindings.get((kind, entity_id))

            def bind(self, kind: str, entity_id: str, *, book_id: str, conditions_id: str, payload: dict, **kwargs) -> None:
                parent._bindings[(kind, entity_id)] = {
                    "book_id": book_id,
                    "conditions_id": conditions_id,
                    "payload_json": payload,
                }

            def result_bindings(self, run_id: str, block_ids=None) -> dict[str, dict | None]:
                keys = block_ids if block_ids is not None else [b.id for b in parent._document.blocks()]
                return {bid: parent._bindings.get(("result", f"{run_id}:{bid}")) for bid in keys}

            def replace_result_binding(self, key: str, *, book_id: str, conditions_id: str, attempt_id: str, artifact_revision_id: str, **kwargs) -> None:
                parent._bindings[("result", key)] = {
                    "book_id": book_id,
                    "conditions_id": conditions_id,
                    "attempt_id": attempt_id,
                    "artifact_revision_id": artifact_revision_id,
                    "payload_json": {"schema_version": 1},
                }

            def archive_artifacts(self, items: list[dict]) -> tuple[str, ...]:
                ids = []
                for item in items:
                    aid = f"art-{len(parent._artifacts) + 1}"
                    parent._artifacts.append(item)
                    ids.append(aid)
                return tuple(ids)

            def find_root_by_key(self, root_key: str):
                return None

            def add_root(self, root: RecoveryRoot) -> None:
                parent._roots[root.id] = root

            def get_root(self, root_id: str) -> RecoveryRoot | None:
                return parent._roots.get(root_id)

            def list_active_roots(self, book_id: str, *, run_id: str | None = None) -> tuple[RecoveryRoot, ...]:
                return tuple(r for r in parent._roots.values() if r.superseded_by is None)

            def list_scopes(self, root_id: str) -> tuple[RecoveryScope, ...]:
                return tuple(s for s in parent._scopes.values() if s.root_id == root_id)

            def add_scope(self, scope: RecoveryScope) -> None:
                parent._scopes[scope.id] = scope

            def get_scope(self, scope_id: str) -> RecoveryScope | None:
                return parent._scopes.get(scope_id)

            def begin_recovery_attempt(self, root_id: str, scope_id: str, **kwargs):
                root = parent._roots[root_id]
                scope = parent._scopes[scope_id]
                if scope.status != "pending":
                    return None
                parent._scope_ordinals[root_id] = parent._scope_ordinals.get(root_id, 0) + 1
                claimed = scope.transition("claimed")
                parent._scopes[scope_id] = claimed
                parent._attempt_counts[scope_id] = parent._attempt_counts.get(scope_id, 0) + 1
                attempt_id = str(uuid4())
                ctx = CallContext(
                    conditions_id=parent._conditions.id,
                    stage=kwargs["stage"],
                    scope_id=scope_id,
                    ordered_block_ids=scope.ordered_blocks,
                    attempt_id=attempt_id,
                    generation=claimed.generation,
                    assigned_model=kwargs["assigned_model"],
                    prompt_hash=kwargs["prompt_hash"],
                    run_id=kwargs.get("run_id"),
                    task_id=kwargs.get("task_id"),
                    root_id=root_id,
                    policy_version=kwargs["policy_version"],
                )
                parent._attempts[attempt_id] = CallAttempt(
                    id=attempt_id,
                    book_id=root.book_id,
                    state="created",
                    generation=claimed.generation,
                    ordinal=parent._scope_ordinals[root_id],
                    conditions_id=parent._conditions.id,
                    task_id=kwargs.get("task_id"),
                    run_id=kwargs.get("run_id"),
                    root_id=root_id,
                    scope_id=scope_id,
                    context={**ctx.to_dict(), "root_generation": root.generation},
                )
                return ctx

            def get_attempt(self, attempt_id: str) -> CallAttempt | None:
                return parent._attempts.get(attempt_id)

            def transition_attempt(self, attempt_id: str, state: str, *, expected_state=None, outcome=None) -> CallAttempt:
                attempt = parent._attempts[attempt_id]
                updated = replace(attempt, state=state, outcome=outcome or attempt.outcome)
                parent._attempts[attempt_id] = updated
                return updated

            def audit_stale_attempt(self, attempt_id: str, *, call_id: str | None = None) -> None:
                pass

            def owns_recovery_attempt(self, attempt_id: str) -> bool:
                attempt = parent._attempts.get(attempt_id)
                if attempt is None or attempt.scope_id is None:
                    return False
                scope = parent._scopes.get(attempt.scope_id)
                return scope is not None and scope.status == "claimed"

            def mark_recovery_accepted(self, attempt_id: str) -> None:
                attempt = parent._attempts[attempt_id]
                if attempt.scope_id is not None:
                    scope = parent._scopes[attempt.scope_id]
                    parent._scopes[attempt.scope_id] = scope.transition("accepted", attempt_id=attempt_id)
                parent._attempts[attempt_id] = replace(attempt, state="accepted")

            def record_recovery_failure(self, attempt_id: str, violation=None, *, outcome: str = "known") -> bool:
                attempt = parent._attempts[attempt_id]
                if attempt.scope_id is not None:
                    scope = parent._scopes[attempt.scope_id]
                    if scope.status == "claimed":
                        parent._scopes[attempt.scope_id] = scope.transition("pending")
                return False

            def pause_root(self, root_id: str, reason: str, *, expected_generation: int | None = None) -> None:
                root = parent._roots[root_id]
                parent._roots[root_id] = replace(root, reason=reason)

            def stop_root(self, root_id: str, reason: str, *, expected_generation: int | None = None) -> None:
                root = parent._roots[root_id]
                parent._roots[root_id] = replace(root, status="stopped", reason=reason, generation=root.generation + 1)

            def root_coverage_complete(self, root_id: str) -> bool:
                root = parent._roots.get(root_id)
                if root is None or root.status == "stopped" or root.superseded_by is not None:
                    return False
                scopes = [s for s in parent._scopes.values() if s.root_id == root_id and s.status != "split"]
                return bool(scopes) and all(s.status == "accepted" for s in scopes)

            def stale_reviews(self, book_id: str) -> None:
                pass

            def delete_correction_checks(self, run_id: str, *, block_ids=None) -> None:
                pass

            def correction_input_generations(self, run_id: str) -> dict:
                return {}

            def correction_checks(self, run_id: str, section_id: str) -> tuple:
                return ()

            def observe_correction_input(self, run_id: str, block_id: str, generation: str) -> None:
                pass

            def list_correction_candidates(self, run_id: str, **kwargs) -> tuple:
                return ()

            def stale_correction_candidates(self, *args, **kwargs) -> None:
                pass

            def list_task_attempts(self, task_id: str) -> tuple:
                return ()

            def replace_correction_checks(self, run_id: str, section_id: str, checks) -> None:
                pass

            def scope_attempt_count(self, scope_id: str) -> int:
                return parent._attempt_counts.get(scope_id, 0)

        class _Tasks:
            def list_for_book(self, book_id: str) -> tuple:
                return tuple(parent._tasks)

            def find_by_identity(self, identity):
                return None

            def add(self, task) -> None:
                parent._tasks.append(task)

            def save(self, task) -> None:
                pass

        class _LLMCalls:
            def diagnostic_record(self, **kwargs):
                return {
                    "call_id": "call-1",
                    "book_id": BOOK_ID,
                    "started_at": datetime.now(UTC),
                    "body": None,
                    "metadata": {},
                }

            def update_diagnostics(self, call_id: str, *, metadata: dict) -> None:
                pass

            def store_diagnostic_body(self, call_id: str, body: str | None) -> None:
                pass

            def enforce_diagnostics_policy(self, policy, now, *, book_id: str | None = None) -> None:
                pass

            def mark_rejected(self, call_id: str, *, error=None, response_text=None) -> None:
                pass

            def list_for_book(self, book_id: str):
                return ()

        ns = SimpleNamespace(
            documents=_Docs(),
            analyses=_Analyses(),
            runs=_Runs(),
            results=_Results(),
            tasks=_Tasks(),
            llm_calls=_LLMCalls(),
        )
        if self._with_work:
            ns.work = _Work()
        return ns


def _make_run() -> ProcessingRun:
    return ProcessingRun(
        id="run-1",
        book_id=BOOK_ID,
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=GlossarySnapshot(),
        llm_settings_snapshot=LLM_SETTINGS,
    )


def _correction_snapshot() -> CorrectionSnapshot:
    return CorrectionSnapshot(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Check translation.",
        versions=CORRECTION_VERSIONS,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run_payload", "expected_ch1_prefix", "expected_notes_prefix"),
    [
        ({"schema_version": 1, "note_numbering": "story-v1"}, "ln-ch01-", "ln-ch04-"),
        ({"schema_version": 1}, "ln-ch02-", "ln-ch03-"),
        (None, "ln-ch02-", "ln-ch03-"),
    ],
)
async def test_note_numbering_synchronized_across_all_three_writers_and_legacy_fallback(
    run_payload: dict | None,
    expected_ch1_prefix: str,
    expected_notes_prefix: str,
) -> None:
    """ProcessChapter, RegenerateBlock, and CorrectChapter stay synchronized on story-v1 vs legacy reading-order."""
    title_sec = _doc_section(
        "title", 0, "Title Page", (_doc_block("title", 0, "Book Title"),), resource_path="title.xhtml"
    )
    ch1_b1 = _doc_block("ch1", 0, "It is a truth universally acknowledged.")
    ch1_sec = _doc_section("ch1", 1, "Chapter 1", (ch1_b1,))
    notes_b1 = _doc_block("notes", 0, "Author footnote about customs.", kind=BlockType.FOOTNOTE)
    notes_sec = _doc_section("notes", 2, "Notes", (notes_b1,), resource_path="notes.xhtml")
    ch2_sec = _doc_section("ch2", 3, "Chapter 2", (_doc_block("ch2", 0, "Another chapter paragraph."),))
    document = _book_document((title_sec, ch1_sec, notes_sec, ch2_sec))

    analyses = {
        sec.id: ChapterAnalysis(
            id=f"an-{sec.id}",
            section_id=sec.id,
            coverage_status=CoverageStatus.COMPLETE,
            treatment_plan=TreatmentPlan.of_treatments(
                sec.id,
                tuple((b.id, BlockTreatment.TRANSLATE) for b in sec.blocks),
            ),
            translation_instructions="Keep style.",
            chapter_summary="Summary.",
        )
        for sec in (ch1_sec, notes_sec, ch2_sec)
    }
    run = _make_run()
    run_binding = (
        None
        if run_payload is None
        else {
            "book_id": BOOK_ID,
            "conditions_id": "",
            "payload_json": {**run_payload, "correction": _correction_snapshot().to_payload()},
        }
    )
    uow = _InMemoryNoteNumberingUow(document, analyses, run, run_binding=run_binding)
    if run_binding is not None:
        uow._bindings[("run", run.id)]["conditions_id"] = uow._conditions.id

    class _SyncClient:
        async def generate(self, request):
            reply = FakeReply.of_payload(
                {
                    "blocks": [
                        {
                            "source_block_id": "b1",
                            "treatment": "TRANSLATE",
                            "translated_text": "Общепризнанная истина в примечании.",
                            "glossary_hints": [],
                        }
                    ],
                    "language_notes": [
                        {
                            "source_block_id": "b1",
                            "original_excerpt": "truth"
                            if request.ordered_block_ids == (ch1_b1.id,)
                            else "customs",
                            "translated_excerpt": None,
                            "note": "Устойчивый оборот.",
                        }
                    ],
                }
            )
            resp = await FakeLLMClient([reply]).generate(request)
            return replace(resp, call_context=request.call_context)

    client = _SyncClient()
    process = ProcessChapter(
        unit_of_work=uow,  # type: ignore[arg-type]
        llm_client=client,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
    )
    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id=ch1_sec.id))
    assert uow._results[ch1_b1.id].language_notes[0].id == f"{expected_ch1_prefix}001"

    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id=notes_sec.id))
    assert uow._results[notes_b1.id].language_notes[0].id == f"{expected_notes_prefix}001"

    regen = RegenerateBlock(
        unit_of_work=uow,  # type: ignore[arg-type]
        llm_client=client,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
    )
    await regen(RegenerateBlockCommand(book_id=BOOK_ID, block_id=ch1_b1.id))
    assert uow._results[ch1_b1.id].language_notes[0].id == f"{expected_ch1_prefix}002"

    # CorrectChapter: via correction_inputs(tx, ...) when binding exists, else fallback to reading-order
    with uow.transaction() as tx:
        if run_binding is not None:
            loaded_inputs = correction_inputs(tx, BOOK_ID, ch1_sec.id)
            assert loaded_inputs is not None
        else:
            from linguary.application.correction import CorrectionInputs

            loaded_inputs = CorrectionInputs(
                document=document,
                section=ch1_sec,
                analysis=analyses[ch1_sec.id],
                run=run,
                conditions=uow._conditions,
                snapshot=_correction_snapshot(),
                blocks=(ch1_b1,),
                results={ch1_b1.id: uow._results[ch1_b1.id]},
                bindings={ch1_b1.id: None},
            )
        corrector = CorrectChapter(
            unit_of_work=uow,
            llm_client=None,
            snapshot=_correction_snapshot(),
            validator=make_validator(),
        )
        corr_resp = ChapterCorrectionResponse(
            blocks=(
                CorrectedBlockResponse(
                    ch1_b1.id,
                    "Общепризнанная истина в примечании.",
                    (),
                    (
                        LanguageNoteResponse(ch1_b1.id, "truth", "Устойчивый оборот."),
                        LanguageNoteResponse(ch1_b1.id, "universally", "Наречие степени."),
                    ),
                ),
            )
        )
        corrected, = corrector._candidates(tx, loaded_inputs, corr_resp)
    assert [n.id for n in corrected.language_notes] == [
        f"{expected_ch1_prefix}002",
        f"{expected_ch1_prefix}003",
    ]


def test_correct_chapter_handles_fake_transaction_without_work() -> None:
    """Fakes lacking `transaction.work` never raise AttributeError and default to reading-order."""
    title_sec = _doc_section(
        "title", 0, "Title Page", (_doc_block("title", 0, "Book Title"),), resource_path="title.xhtml"
    )
    ch1_b1 = _doc_block("ch1", 0, "It is a truth universally acknowledged.")
    ch1_sec = _doc_section("ch1", 1, "Chapter 1", (ch1_b1,))
    document = _book_document((title_sec, ch1_sec))
    analysis = ChapterAnalysis(
        id="an-ch1",
        section_id=ch1_sec.id,
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=TreatmentPlan.of_treatments(ch1_sec.id, ((ch1_b1.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
    )
    run = _make_run()
    from linguary.application.correction import CorrectionInputs

    prev = BlockProcessingResult(
        block_id=ch1_b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Общепризнанная истина.",
    )
    inputs = CorrectionInputs(
        document=document,
        section=ch1_sec,
        analysis=analysis,
        run=run,
        conditions=SimpleNamespace(id="cond-1", payload={"schema_version": 1, "versions": {"validators": 3}}),
        snapshot=_correction_snapshot(),
        blocks=(ch1_b1,),
        results={ch1_b1.id: prev},
        bindings={ch1_b1.id: None},
    )
    fake_tx = SimpleNamespace()  # intentionally no `.work` attribute
    corrector = CorrectChapter(
        unit_of_work=None,
        llm_client=None,
        snapshot=_correction_snapshot(),
        validator=make_validator(),
    )
    resp = ChapterCorrectionResponse(
        blocks=(
            CorrectedBlockResponse(
                ch1_b1.id,
                "Общепризнанная истина.",
                (),
                (LanguageNoteResponse(ch1_b1.id, "truth", "Устойчивый оборот."),),
            ),
        )
    )
    corrected, = corrector._candidates(fake_tx, inputs, resp)
    assert corrected.language_notes[0].id == "ln-ch02-001"


@pytest.mark.asyncio
async def test_translation_call_v3_checks_translated_excerpt_only_under_policy_level_3() -> None:
    """`TranslationCall` enables `check_translated_excerpt=True` under v3 policy and ignores GLOSSARY_ONLY segments."""
    mixed_text = "He saw the cloak. She raised the wand."
    cut = mixed_text.index(" She raised")
    block = _doc_block(
        "ch1",
        0,
        mixed_text,
        segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(mixed_text), 1)),
    )
    section = _doc_section("ch1", 0, "Chapter 1", (block,))
    document = _book_document((section,))
    plan = TreatmentPlan(
        section.id,
        (
            TreatmentPlanItem(
                block.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
                (0, 1),
            ),
        ),
    )
    unit = TranslationUnit.of_blocks(id="u1", blocks=(block,), plan=plan)
    content_plan = ContentPlan(version=3, profile="content-plan-v3")
    analysis = ChapterAnalysis.create(
        id="an-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(block.id,),
        content_plan=content_plan,
    )

    def _run_call(policy_version: str, notes: list[dict]):
        uow = _InMemoryNoteNumberingUow(document, {section.id: analysis}, _make_run(), run_binding=None)
        root_id = str(uuid4())
        scope_id = str(uuid4())
        root = RecoveryRoot(
            id=root_id,
            book_id=BOOK_ID,
            conditions_id=uow._conditions.id,
            root_key="rk-1",
            manifest={
                "schema_version": 1,
                "ordered_blocks": [block.id],
                "policy": {"max_calls": 3, "max_scope_calls": 3},
                "active_run_id": "run-1",
                "notes_policy": "grounded-notes-v1",
            },
            policy_version=policy_version,
        )
        scope = RecoveryScope(id=scope_id, root_id=root.id, ordered_blocks=(block.id,))
        uow._roots[root.id] = root
        uow._scopes[scope.id] = scope
        ctx = ContextBuilder(content_plan_profile="content-plan-v3").for_translation_unit(
            document=document,
            section=section,
            unit=unit,
            analysis=analysis,
            glossary=GlossarySnapshot(),
            policy_version=policy_version,
        )
        ctx = replace(
            ctx,
            request=ctx.request.with_logging(book_id=BOOK_ID, stage=LLMCallStage.TRANSLATION_UNIT),
        )

        class _ReplyClient:
            async def generate(self, request):
                reply = FakeReply.of_payload(
                    {
                        "blocks": [
                            {
                                "source_block_id": "b1",
                                "treatment": "TRANSLATE",
                                "translated_text": None,
                                "glossary_hints": [
                                    {"original": "cloak", "translation": "плащ", "note": "Длинная накидка."}
                                ],
                                "segments": [{"index": 2, "translated_text": "Она подняла палочку."}],
                            }
                        ],
                        "language_notes": notes,
                        "content_evidence": [],
                    }
                )
                resp = await FakeLLMClient([reply]).generate(request)
                return replace(resp, call_context=request.call_context)

        call = TranslationCall(
            unit_of_work=uow,  # type: ignore[arg-type]
            llm_client=_ReplyClient(),  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_attempts=1),
            validator=make_validator(),
            sleep=lambda _: None,  # type: ignore[arg-type]
        )
        return call.request(
            ctx,
            unit=unit,
            target_language="ru",
            note_ids=lambda: "ln-ch01-001",
            recovery=(root.id, scope.id),
            source_texts={block.id: block.text},
        )

    # Hallucinated translated_excerpt on TRANSLATE segment fails under v3 (policy_level == 3)
    hallucinated_on_translate = [
        {
            "source_block_id": "b1",
            "original_excerpt": "raised the wand",
            "translated_excerpt": "опустила книгу",
            "note": "Идиоматический образ.",
        }
    ]
    with pytest.raises(RecoveryError) as exc_info:
        await _run_call(CURRENT_TRANSLATION_POLICY, hallucinated_on_translate)
    orig = exc_info.value.original_error
    assert isinstance(orig, ResultValidationError)
    assert orig.violation is not None
    assert orig.violation.rule_code == "note_translated_excerpt_missing"

    # Under v2 policy (policy_level == 2 < 3), translated_excerpt is NOT checked
    v2_res = await _run_call(TRANSLATION_RELIABILITY_POLICY_V3, hallucinated_on_translate)
    assert len(v2_res.results) == 1

    # Note anchored in GLOSSARY_ONLY segment of mixed block does NOT falsely fail under v3!
    note_in_glossary_only = [
        {
            "source_block_id": "b1",
            "original_excerpt": "the cloak",
            "translated_excerpt": "мантия с капюшоном",
            "note": "Устаревший предмет одежды.",
        }
    ]
    mixed_res = await _run_call(CURRENT_TRANSLATION_POLICY, note_in_glossary_only)
    assert len(mixed_res.results[0].language_notes) == 1
