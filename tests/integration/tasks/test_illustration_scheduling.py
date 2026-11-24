"""Integration tests for illustration recovery, pause/resume, reanalysis and container wiring (`P4.T3`)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime.executor_lock import ExecutorLock
from linguary.application.illustration_dto import (
    IllustrationAliasResponse,
    IllustrationCandidatesResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationFactScopeResponse,
    IllustrationMentionDTO,
    IllustrationSelectionResponse,
)
from linguary.application.illustrations import dispatch_scope_task
from linguary.application.llm_dto import (
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedSceneResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.llm import LLMRequest, LLMResponse, LLMResponseKind
from linguary.application.ports.persistence import BookRecord
from linguary.application.use_cases.analyze_chapter import ChapterAnalysisSequence
from linguary.application.use_cases.pause_book import (
    PauseBookCommand,
    PauseBookError,
    ResumeBookCommand,
)
from linguary.application.use_cases.reset_processing import (
    ChooseBookConditions,
    ResetProcessingCommand,
)
from linguary.cli.app import main as cli_main
from linguary.composition import build_application, drain_tasks, run_worker
from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    CoverageStatus,
    DocumentBlock,
    DocumentSection,
    LLMCallStage,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TaskType,
    TokenUsage,
    TreatmentPlan,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.illustrations import (
    IllustrationAttemptState,
    IllustrationConflictError,
    IllustrationGenerationState,
    IllustrationReasonCode,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
)


class _DummyDetector:
    def detect(self, text: str) -> str:
        return "en" if any("a" <= ch.lower() <= "z" for ch in text) else "ru"


def _extract_json_payload(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    return json.loads(text[start : end + 1])


class _StagedIllustrationFakeLLM:
    """Deterministic fake LLM supporting staged narrative/pedagogical and illustration stages."""

    supports_staged_analysis = True

    def __init__(self, *, forbid_narrative: bool = False) -> None:
        self.forbid_narrative = forbid_narrative
        self.calls: list[LLMCallStage] = []

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        stage = request.stage or LLMCallStage.CHAPTER_ANALYSIS
        self.calls.append(stage)
        usage = TokenUsage(prompt_tokens=15, completion_tokens=15)
        if request.kind is LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE:
            if self.forbid_narrative:
                raise AssertionError("cached NARRATIVE pass must not call the LLM")
            aliases = tuple(alias for alias, _ in request.block_aliases) or ("b1",)
            payload: Any = NarrativeAnalysisResponse(
                translation_instructions="Сохраняй спокойный повествовательный тон.",
                chapter_summary="Alice opened the heavy oak door into the quiet garden.",
                scenes=(
                    QuotedSceneResponse(
                        start_block_id=aliases[0],
                        end_block_id=aliases[-1],
                        summary="Алиса открыла тяжёлую дубовую дверь в тихий сад.",
                        location="Сад",
                        participants=("Alice",),
                    ),
                ),
            )
            return LLMResponse(
                payload=payload,
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL:
            aliases = tuple(alias for alias, _ in request.block_aliases) or ("b1",)
            items = tuple(
                TreatmentPlanItemResponse(source_block_id=bid, treatment=BlockTreatment.TRANSLATE)
                for bid in aliases
            )
            payload = PedagogicalAnalysisResponse(treatment_plan=items)
            return LLMResponse(
                payload=payload,
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            data = _extract_json_payload(request.input_text)
            units = data.get("units", ())
            unit_ids = tuple(
                str(u.get("source_unit_id") or u.get("unit_id") or u.get("id"))
                for u in units
                if isinstance(u, dict)
            )
            first_blk_id = f"{request.book_id}:b0"
            first_quote = "Alice"
            for u in units:
                for blk in u.get("blocks", ()):
                    first_blk_id = str(blk.get("block_id") or blk.get("id"))
                    txt = str(blk.get("text", ""))
                    first_quote = "Alice" if "Alice" in txt else txt[:20]
                    break
                break
            payload = IllustrationExtractResponse(
                mentions=(
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Alice",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=first_blk_id,
                        quote=first_quote,
                    ),
                ),
                facts=(),
                covered_source_unit_ids=unit_ids,
                complete=True,
            )
            return LLMResponse(
                payload=payload, provider="openai_compatible", model="gpt-5-mini", usage=usage
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            return LLMResponse(
                payload=IllustrationAliasResponse(decisions=()),
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            data = _extract_json_payload(request.input_text)
            entity = data.get("entity", {})
            eid = str(entity.get("entity_id") or data.get("entity_id") or "ent-alice")
            allowed_traits = data.get("allowed_traits") or ["clothing"]
            trait_name = str(allowed_traits[0])
            return LLMResponse(
                payload=IllustrationDesignResponse(
                    entity_id=eid,
                    traits=(
                        IllustrationDesignTraitDTO(
                            trait=trait_name,
                            description_ru="Тёмный дорожный плащ.",
                            description_en="Dark travelling cloak.",
                            assumption_ru="Предположение по контексту главы.",
                        ),
                    ),
                ),
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_FACT_SCOPE:
            data = _extract_json_payload(request.input_text)
            ev_id = str(data.get("event_id") or "ev-1")
            return LLMResponse(
                payload=IllustrationFactScopeResponse(
                    event_id=ev_id,
                    applicable_fact_ids=(),
                    excluded_fact_ids=(),
                    compact_description_ru="Алиса.",
                    compact_description_en="Alice.",
                    complete=True,
                ),
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            data = _extract_json_payload(request.input_text)
            scene_id = str(data.get("scene_id") or "s1")
            unit_ids = tuple(
                str(b["source_unit_id"])
                for b in data.get("blocks", ())
                if isinstance(b, dict) and "source_unit_id" in b
            )
            return LLMResponse(
                payload=IllustrationCandidatesResponse(
                    scene_id=scene_id,
                    events=(),
                    zero_reason_ru="Спокойный эпизод без выразительной сцены.",
                    covered_source_unit_ids=unit_ids,
                    complete=True,
                ),
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            return LLMResponse(
                payload=IllustrationSelectionResponse(
                    selected_candidate_ids=(),
                    zero_reason_ru="Спокойный эпизод без выразительной сцены.",
                ),
                provider="openai_compatible",
                model="gpt-5-mini",
                usage=usage,
            )
        raise AssertionError(f"unexpected request kind: {request.kind}")


def _make_source_epub(tmp_path: Path, name: str = "book.epub") -> Path:
    path = tmp_path / name
    path.write_bytes(b"PK\x03\x04dummy-epub-content")
    return path


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_book(
    app: Any,
    book_id: str,
    *,
    source_path: Path,
    status: BookStatus = BookStatus.GLOSSARY_REVIEW,
    block_text: str = "Alice opened the heavy oak door into the quiet garden.",
) -> tuple[str, str, BookConditions]:
    sec_id = f"{book_id}-sec-0"
    blk_id = f"{book_id}:b0"
    sec = DocumentSection(
        id=sec_id,
        document_id=book_id,
        title="Chapter 1",
        order=0,
        source_resource_path="Text/ch1.xhtml",
        blocks=(
            DocumentBlock(
                id=blk_id,
                section_id=sec_id,
                type=BlockType.PARAGRAPH,
                order=0,
                text=block_text,
                source_reference=SourceReference("Text/ch1.xhtml", "p[0]"),
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Novel {book_id}"),
        source_language="en",
        target_language="ru",
        sections=(sec,),
    )
    book = BookRecord(
        id=book_id,
        metadata=doc.metadata,
        source_language="en",
        target_language="ru",
        status=status,
        source_epub_path=str(source_path),
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    with app.unit_of_work.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
        cond = app.book_conditions.create(tx, book, app.runtime_settings.revision)
    return sec_id, blk_id, cond


def _enable_illustrations(app: Any, book_id: str) -> str:
    rev = app.illustrations.save_settings(
        book_id,
        expected_revision=0,
        settings=IllustrationSettings(
            enabled=True,
            max_per_chapter=2,
            model_reference="gpt-5-mini",
            model_prompts="gpt-5-mini",
        ),
    )
    return rev.id


def _make_app(tmp_path: Path, *, llm_client: Any | None = None, name: str = "data") -> Any:
    return build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / name)),
            llm=LLMSettings(
                api_key="test-key",
                model="gpt-5-mini",
                staged_analysis=True,
                max_attempts=1,
            ),
        ),
        llm_client=llm_client or _StagedIllustrationFakeLLM(),
        content_plan_profile="content-plan-v3",
    )


def _seed_valid_narrative_pass(app: Any, book_id: str, sec_id: str, blk_id: str) -> None:
    """Run real AnalyzeChapter on the chapter so its NARRATIVE pass is 100% genuine."""
    del sec_id, blk_id
    with app.unit_of_work.transaction() as tx:
        book = tx.books.get(book_id)
        assert book is not None
        if book.status is not BookStatus.IMPORTED:
            tx.books.save(replace(book, status=BookStatus.IMPORTED))
    asyncio.run(app.analysis.start(book_id))
    tasks = asyncio.run(drain_tasks(app))
    assert any(t.task_type is TaskType.ANALYZE_CHAPTER and t.status is TaskStatus.SUCCEEDED for t in tasks), tasks


def test_book_pause_blocks_both_types_in_glossary_review_and_completed(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    src_sha = _source_sha(src)
    app = _make_app(tmp_path)
    try:
        for book_id, status in (
            ("book-gr", BookStatus.GLOSSARY_REVIEW),
            ("book-done", BookStatus.COMPLETED),
        ):
            sec_id, _, _ = _seed_book(app, book_id, source_path=src, status=status)
            # Without illustration work, pausing in GLOSSARY_REVIEW / COMPLETED is rejected.
            with pytest.raises(PauseBookError):
                app.pause_book(PauseBookCommand(book_id=book_id))

            settings_id = _enable_illustrations(app, book_id)
            with app.unit_of_work.transaction() as tx:
                gen, _ = tx.illustrations.create_or_get_generation(
                    book_id,
                    settings_id=settings_id,
                    request_key=f"req-{book_id}",
                    source_sha256=src_sha,
                    source_authority={},
                    requested_sections=(sec_id,),
                )
                tx.illustrations.update_generation_state(
                    book_id, gen.id, state=IllustrationGenerationState.QUEUED
                )
                sc_ref, sc_ch = tx.illustrations.upsert_scopes(
                    gen.id,
                    [
                        IllustrationScopeSpec(
                            stage=IllustrationScopeStage.EXTRACT,
                            scope_key="extract:p0",
                            input_sha256="b" * 64,
                            summary={},
                        ),
                        IllustrationScopeSpec(
                            stage=IllustrationScopeStage.CHAPTER,
                            scope_key=sec_id,
                            section_id=sec_id,
                            section_order=0,
                            section_title="Chapter 1",
                            input_sha256="c" * 64,
                            summary={},
                        ),
                    ],
                )
                dispatch_scope_task(
                    tx,
                    sc_ref,
                    generation_id=gen.id,
                    task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                    id_factory=app.task_queue.new_task_id,
                    book_id=book_id,
                )
                dispatch_scope_task(
                    tx,
                    sc_ch,
                    generation_id=gen.id,
                    task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                    id_factory=app.task_queue.new_task_id,
                    book_id=book_id,
                )

            res = app.pause_book(PauseBookCommand(book_id=book_id))
            assert res.pause_requested is True
            assert res.is_paused is True

            # Queue refuses to claim BOTH illustration task types while paused.
            claimed = asyncio.run(app.task_queue.claim_next_pending())
            assert claimed is None

            # Illustration view reflects paused state via is_book_paused.
            view = app.illustrations.get(book_id)
            assert view.state == "paused"
            assert view.reason_code == IllustrationReasonCode.BOOK_PAUSED.value
    finally:
        asyncio.run(app.aclose())


def test_primary_finish_and_cli_export_do_not_clear_pause_with_pending_illustrations(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    src = _make_source_epub(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[storage]
data_root = "{data_root.as_posix()}"

[llm]
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key = "test-key"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(data_root)),
            llm=LLMSettings(api_key="test-key", model="gpt-5-mini"),
        ),
        llm_client=_StagedIllustrationFakeLLM(),
        content_plan_profile="content-plan-v3",
    )
    try:
        sec_id, _, cond = _seed_book(
            app, "book-keep-pause", source_path=src, status=BookStatus.READY_TO_EXPORT
        )
        settings_id = _enable_illustrations(app, "book-keep-pause")
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-keep-pause",
                settings_id=settings_id,
                request_key="req-kp",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=(sec_id,),
            )
            tx.illustrations.update_generation_state(
                "book-keep-pause", gen.id, state=IllustrationGenerationState.QUEUED
            )
            sc = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p0",
                        input_sha256="b" * 64,
                        summary={},
                    )
                ],
            )[0]
            dispatch_scope_task(
                tx,
                sc,
                generation_id=gen.id,
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                id_factory=app.task_queue.new_task_id,
                book_id="book-keep-pause",
            )
            tx.work.set_book_paused("book-keep-pause", cond.id, True)

        # 1) ResetProcessing must NOT clear pause while illustration tasks are pending.
        app.reset_processing(ResetProcessingCommand(book_id="book-keep-pause"))
        with app.unit_of_work.read_transaction() as tx:
            assert tx.work.is_book_paused("book-keep-pause") is True

        # 2) Processing completion helper must NOT clear pause.
        from linguary.application.block_processing import complete_processing_in_transaction

        with app.unit_of_work.transaction() as tx:
            book = tx.books.get("book-keep-pause")
            assert book is not None
            tx.books.save(replace(book, status=BookStatus.PROCESSING))
            complete_processing_in_transaction(
                tx,
                book_id="book-keep-pause",
                finished_task_id=None,
                task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
            )
        with app.unit_of_work.read_transaction() as tx:
            assert tx.work.is_book_paused("book-keep-pause") is True
            book_after = tx.books.get("book-keep-pause")
            assert book_after is not None and book_after.status is BookStatus.READY_TO_EXPORT

        # 3) Analysis completion helper must NOT clear pause.
        with app.unit_of_work.transaction() as tx:
            book = tx.books.get("book-keep-pause")
            assert book is not None
            tx.books.save(replace(book, status=BookStatus.ANALYZING))
            tx.analyses.save(
                "book-keep-pause",
                ChapterAnalysis(
                    id="an-kp-0",
                    section_id=sec_id,
                    coverage_status=CoverageStatus.COMPLETE,
                    treatment_plan=TreatmentPlan(section_id=sec_id, items=()),
                    translation_instructions="Переводи точно.",
                    chapter_summary="Краткое содержание главы.",
                ),
            )
        seq = ChapterAnalysisSequence(
            unit_of_work=app.unit_of_work,
            queue=app.task_queue,
            analyze_chapter=app.runtime_settings.current.analysis._analyze_chapter,  # type: ignore[attr-defined]
        )
        seq._try_complete_analysis("book-keep-pause")
        with app.unit_of_work.read_transaction() as tx:
            assert tx.work.is_book_paused("book-keep-pause") is True
            book = tx.books.get("book-keep-pause")
            assert book is not None and book.status is BookStatus.GLOSSARY_REVIEW
        with app.unit_of_work.transaction() as tx:
            book = tx.books.get("book-keep-pause")
            assert book is not None
            tx.books.save(replace(book, status=BookStatus.READY_TO_EXPORT))
    finally:
        asyncio.run(app.aclose())

    # 4) CLI `export --mode structure_only` must NOT clear pause while illustration tasks exist.
    out_epub = tmp_path / "exported.epub"
    rc = cli_main(
        [
            "--config",
            str(cfg),
            "export",
            "book-keep-pause",
            "--mode",
            "structure_only",
            "--output",
            str(out_epub),
        ]
    )
    assert rc == 0
    assert out_epub.exists()
    engine = create_database_engine(data_root / "linguary.db")
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    try:
        with uow.read_transaction() as tx:
            assert tx.work.is_book_paused("book-keep-pause") is True
    finally:
        engine.dispose()


def test_resume_reconciles_book_and_notifies_after_commit(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        sec_id, blk_id, cond = _seed_book(
            app, "book-res", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-res", sec_id, blk_id)
        settings_id = _enable_illustrations(app, "book-res")

        # Insert waiting_source generation manually and pause the book.
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-res",
                settings_id=settings_id,
                request_key="req-res",
                source_sha256=_source_sha(src),
                source_authority={},
                requested_sections=(sec_id,),
            )
            tx.work.set_book_paused("book-res", cond.id, True)

        observed_events: list[tuple[str, bool]] = []
        orig_notify = app.task_queue.notify_work_available

        def spy_notify() -> None:
            # Verify pause is ALREADY committed as False when notify fires!
            with app.unit_of_work.read_transaction() as rtx:
                observed_events.append(("notify", rtx.work.is_book_paused("book-res")))
            orig_notify()

        app.task_queue.notify_work_available = spy_notify  # type: ignore[method-assign]
        app.illustration_coordinator._notify_work_available = spy_notify  # type: ignore[attr-defined]
        # Exercise the real wired app.resume_book() directly!
        res = app.resume_book(ResumeBookCommand(book_id="book-res"))
        assert res.pause_requested is False
        assert observed_events
        assert all(paused is False for _, paused in observed_events)

        with app.unit_of_work.read_transaction() as tx:
            gen_after = tx.illustrations.get_generation("book-res", gen.id)
            assert gen_after is not None
            assert gen_after.manifest_sha256 is not None
            assert gen_after.state is IllustrationGenerationState.QUEUED
            scopes = tx.illustrations.list_scopes(gen.id)
            assert any(s.stage is IllustrationScopeStage.EXTRACT for s in scopes)
    finally:
        asyncio.run(app.aclose())


def test_notify_only_after_commit(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        sec_id, blk_id, _ = _seed_book(
            app, "book-notif", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-notif", sec_id, blk_id)
        _enable_illustrations(app, "book-notif")

        notifications: list[int] = []
        orig_notify = app.task_queue.notify_work_available

        def spy_notify() -> None:
            # Must see the newly committed generation and tasks in a separate read tx.
            with app.unit_of_work.read_transaction() as rtx:
                head = rtx.illustrations.get_head("book-notif")
                assert head.active_generation_id is not None
                tasks = [
                    t
                    for t in rtx.tasks.list_for_book("book-notif")
                    if t.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE
                ]
                notifications.append(len(tasks))
            orig_notify()

        app.illustrations._notify_work_available = spy_notify  # type: ignore[attr-defined]
        asyncio.run(
            app.illustrations.prepare(
                "book-notif",
                expected_revision=1,
            )
        )
        assert notifications and notifications[-1] >= 1

        # If prepare fails with conflict (rollback), notify MUST NOT fire.
        before_count = len(notifications)
        with pytest.raises(IllustrationConflictError):
            asyncio.run(app.illustrations.prepare("book-notif", expected_revision=999))
        assert len(notifications) == before_count
    finally:
        asyncio.run(app.aclose())


def test_restart_between_narrative_save_and_reconcile(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        sec_id, blk_id, _ = _seed_book(
            app, "book-gap", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-gap", sec_id, blk_id)
        settings_id = _enable_illustrations(app, "book-gap")

        # Simulate crash between NARRATIVE save and coordinator.reconcile():
        # generation sits in WAITING_SOURCE even though NARRATIVE is already stored.
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-gap",
                settings_id=settings_id,
                request_key="req-gap",
                source_sha256=_source_sha(src),
                source_authority={},
                requested_sections=(sec_id,),
            )
            assert gen.state is IllustrationGenerationState.WAITING_SOURCE

        # Restart: drain_tasks runs recover_interrupted() -> after_recovery -> reconciles and drains!
        asyncio.run(drain_tasks(app))

        with app.unit_of_work.read_transaction() as tx:
            gen_after = tx.illustrations.get_generation("book-gap", gen.id)
            assert gen_after is not None
            assert gen_after.manifest_sha256 is not None
            assert gen_after.state is IllustrationGenerationState.READY
    finally:
        asyncio.run(app.aclose())


def test_after_recovery_releases_reserved_redispaches_pending_and_completes_on_drain_while_sent_interrupted(
    tmp_path: Path,
) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        # Book A ("book-rec"): genuine NARRATIVE, reserved attempt + RUNNING task (:a1)
        sec_a, blk_a, _ = _seed_book(
            app, "book-rec", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-rec", sec_a, blk_a)
        _enable_illustrations(app, "book-rec")
        # Prepare creates gen + extract scope + task :a1
        view_a = asyncio.run(app.illustrations.prepare("book-rec", expected_revision=1))
        gen_a_id = view_a.active_generation_id
        assert gen_a_id is not None

        with app.unit_of_work.transaction() as tx:
            scopes_a = tx.illustrations.list_scopes(gen_a_id, stage=IllustrationScopeStage.EXTRACT)
            assert len(scopes_a) == 1
            sc_a = scopes_a[0]
            assert sc_a.task_id is not None
            task_a = tx.tasks.get(sc_a.task_id)
            assert task_a is not None
            # Simulate task was claimed (RUNNING) and entered TX1 (reserved), then process crashed!
            tx.tasks.save(task_a.start())
            att_a = tx.illustrations.claim_scope(
                sc_a.id,
                expected_generation_id=gen_a_id,
                task_id=task_a.id,
                max_calls=3,
            )
            assert att_a.state is IllustrationAttemptState.RESERVED

        # Book B ("book-sent"): sent attempt + RUNNING task (:a1)
        sec_b, blk_b, _ = _seed_book(
            app, "book-sent", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-sent", sec_b, blk_b)
        _enable_illustrations(app, "book-sent")
        view_b = asyncio.run(app.illustrations.prepare("book-sent", expected_revision=1))
        gen_b_id = view_b.active_generation_id
        assert gen_b_id is not None

        with app.unit_of_work.transaction() as tx:
            scopes_b = tx.illustrations.list_scopes(gen_b_id, stage=IllustrationScopeStage.EXTRACT)
            sc_b = scopes_b[0]
            assert sc_b.task_id is not None
            task_b = tx.tasks.get(sc_b.task_id)
            assert task_b is not None
            tx.tasks.save(task_b.start())
            att_b = tx.illustrations.claim_scope(
                sc_b.id,
                expected_generation_id=gen_b_id,
                task_id=task_b.id,
                max_calls=3,
            )
            tx.illustrations.mark_attempt_sent(
                att_b.id,
                request_sha256="d" * 64,
                request_meta={"wire_bytes": 128},
            )

        # Now run exclusive drain_tasks(): recovers BOTH books and executes Book A to completion!
        asyncio.run(drain_tasks(app))

        with app.unit_of_work.read_transaction() as tx:
            # Book A: reserved was cancelled & released, :a1 superseded, :a2 ran and ACCEPTED!
            old_task_a = tx.tasks.get(task_a.id)
            assert old_task_a is not None and old_task_a.status is TaskStatus.SUPERSEDED
            a_tasks = [
                t
                for t in tx.tasks.list_for_book("book-rec")
                if t.scope_id == sc_a.id
            ]
            assert any(t.operation_id == f"{gen_a_id}:a2" and t.status is TaskStatus.SUCCEEDED for t in a_tasks)
            sc_a_after = tx.illustrations.get_scope(sc_a.id)
            assert sc_a_after is not None and sc_a_after.state is IllustrationScopeState.ACCEPTED
            gen_a_after = tx.illustrations.get_generation("book-rec", gen_a_id)
            assert gen_a_after is not None
            assert gen_a_after.calls_reserved == 0
            assert gen_a_after.state is IllustrationGenerationState.READY

            # Book B: sent became interrupted, calls_sent stayed 1, scope FAILED/interrupted, NOT auto-retried!
            sc_b_after = tx.illustrations.get_scope(sc_b.id)
            assert sc_b_after is not None
            assert sc_b_after.state is IllustrationScopeState.FAILED
            assert sc_b_after.reason_code == IllustrationReasonCode.INTERRUPTED.value
            gen_b_after = tx.illustrations.get_generation("book-sent", gen_b_id)
            assert gen_b_after is not None
            assert gen_b_after.calls_sent == 1
            assert gen_b_after.state is IllustrationGenerationState.FAILED
            b_tasks = [
                t for t in tx.tasks.list_for_book("book-sent") if t.scope_id == sc_b.id
            ]
            assert len(b_tasks) == 1
            assert b_tasks[0].status is TaskStatus.FAILED
    finally:
        asyncio.run(app.aclose())


def test_after_recovery_closes_orphaned_failed_task_of_accepted_scope(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        sec_id, _, _ = _seed_book(app, "book-orph", source_path=src, status=BookStatus.COMPLETED)
        settings_id = _enable_illustrations(app, "book-orph")
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-orph",
                settings_id=settings_id,
                request_key="req-orph",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=(sec_id,),
            )
            tx.illustrations.update_generation_state(
                "book-orph", gen.id, state=IllustrationGenerationState.RUNNING
            )
            sc = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p0",
                        input_sha256="b" * 64,
                        summary={},
                    )
                ],
            )[0]
            task = dispatch_scope_task(
                tx,
                sc,
                generation_id=gen.id,
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                id_factory=app.task_queue.new_task_id,
                book_id="book-orph",
            )
            # Simulate crash AFTER TX3 accept_scope but BEFORE TaskRunner._save(succeed):
            tx.tasks.save(task.start())
            tx.illustrations.accept_scope(
                sc.id,
                attempt_id=None,
                expected_input_sha256=sc.input_sha256,
                payload={"complete": True, "entities": [], "facts": []},
                output_sha256="e" * 64,
                summary={},
            )

        asyncio.run(app.runner.recover_interrupted())

        with app.unit_of_work.read_transaction() as tx:
            task_after = tx.tasks.get(task.id)
            assert task_after is not None
            assert task_after.status is TaskStatus.SUPERSEDED
            assert task_after.last_error is None
    finally:
        asyncio.run(app.aclose())


def test_two_executors_drain_tasks_and_run_worker_no_false_interrupt(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        # Prepare Book 2 & Book 3 narrative passes BEFORE creating Book 1's live RUNNING task!
        sec2, blk2, cond2 = _seed_book(
            app, "book-wait", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-wait", sec2, blk2)
        settings2 = _enable_illustrations(app, "book-wait")

        sec3, blk3, _ = _seed_book(
            app, "book-wait-rw", source_path=src, status=BookStatus.ANALYZING
        )
        _seed_valid_narrative_pass(app, "book-wait-rw", sec3, blk3)
        settings3 = _enable_illustrations(app, "book-wait-rw")

        # Book 1: live RUNNING task + SENT attempt owned by executor #1
        sec1, _, _ = _seed_book(app, "book-live", source_path=src, status=BookStatus.COMPLETED)
        settings1 = _enable_illustrations(app, "book-live")
        with app.unit_of_work.transaction() as tx:
            gen1, _ = tx.illustrations.create_or_get_generation(
                "book-live",
                settings_id=settings1,
                request_key="req-live",
                source_sha256="a" * 64,
                source_authority={},
                requested_sections=(sec1,),
            )
            tx.illustrations.update_generation_state(
                "book-live", gen1.id, state=IllustrationGenerationState.RUNNING
            )
            sc1 = tx.illustrations.upsert_scopes(
                gen1.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p0",
                        input_sha256="b" * 64,
                        summary={},
                    )
                ],
            )[0]
            live_task = dispatch_scope_task(
                tx,
                sc1,
                generation_id=gen1.id,
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                id_factory=app.task_queue.new_task_id,
                book_id="book-live",
            )
            tx.tasks.save(live_task.start())
            att1 = tx.illustrations.claim_scope(
                sc1.id,
                expected_generation_id=gen1.id,
                task_id=live_task.id,
                max_calls=3,
            )
            tx.illustrations.mark_attempt_sent(
                att1.id, request_sha256="f" * 64, request_meta={"wire_bytes": 64}
            )

        # Hold ExecutorLock as executor #1 (shared mode -> second executor sees alone=False)
        primary_lock = ExecutorLock(app.paths.executor_lock)
        assert primary_lock.join() is True
        primary_lock.share()
        try:
            # Create waiting_source generation for book-wait right before second drain_tasks
            with app.unit_of_work.transaction() as tx:
                gen2, _ = tx.illustrations.create_or_get_generation(
                    "book-wait",
                    settings_id=settings2,
                    request_key="req-wait-drain",
                    source_sha256=_source_sha(src),
                    source_authority={},
                    requested_sections=(sec2,),
                )
                tx.work.set_book_paused("book-wait", cond2.id, True)  # pause so drain doesn't execute it

            # 1) Second executor: drain_tasks() with alone=False
            asyncio.run(drain_tasks(app))
            with app.unit_of_work.read_transaction() as tx:
                # Live executor's task & scope MUST NOT be interrupted!
                t1 = tx.tasks.get(live_task.id)
                assert t1 is not None and t1.status is TaskStatus.RUNNING
                s1 = tx.illustrations.get_scope(sc1.id)
                assert s1 is not None and s1.state is IllustrationScopeState.RUNNING
                # Book 2 WAS reconciled by reconcile_active()!
                g2 = tx.illustrations.get_generation("book-wait", gen2.id)
                assert g2 is not None and g2.manifest_sha256 is not None

            # 2) Second executor: run_worker() with alone=False (pre-stopped so it exits after reconcile_active)
            with app.unit_of_work.transaction() as tx:
                gen3, _ = tx.illustrations.create_or_get_generation(
                    "book-wait-rw",
                    settings_id=settings3,
                    request_key="req-wait-rw",
                    source_sha256=_source_sha(src),
                    source_authority={},
                    requested_sections=(sec3,),
                )
            app.worker_pool.request_stop()
            asyncio.run(run_worker(app))
            with app.unit_of_work.read_transaction() as tx:
                t1_after = tx.tasks.get(live_task.id)
                assert t1_after is not None and t1_after.status is TaskStatus.RUNNING
                s1_after = tx.illustrations.get_scope(sc1.id)
                assert s1_after is not None and s1_after.state is IllustrationScopeState.RUNNING
                g3 = tx.illustrations.get_generation("book-wait-rw", gen3.id)
                assert g3 is not None and g3.manifest_sha256 is not None
        finally:
            primary_lock.leave()
    finally:
        asyncio.run(app.aclose())


def test_reanalysis_revokes_and_supersedes_but_translation_reset_preserves(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    app = _make_app(tmp_path)
    try:
        sec_id, _, cond = _seed_book(
            app, "book-reana", source_path=src, status=BookStatus.READY_TO_EXPORT
        )
        settings_id = _enable_illustrations(app, "book-reana")
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-reana",
                settings_id=settings_id,
                request_key="req-reana",
                source_sha256=_source_sha(src),
                source_authority={},
                requested_sections=(sec_id,),
            )
            tx.illustrations.update_generation_state(
                "book-reana", gen.id, state=IllustrationGenerationState.QUEUED
            )
            sc = tx.illustrations.upsert_scopes(
                gen.id,
                [
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.EXTRACT,
                        scope_key="extract:p0",
                        input_sha256="b" * 64,
                        summary={},
                    )
                ],
            )[0]
            task = dispatch_scope_task(
                tx,
                sc,
                generation_id=gen.id,
                task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                id_factory=app.task_queue.new_task_id,
                book_id="book-reana",
            )

        # 1) Translation reset preserves active illustration generation and its pending task.
        app.reset_processing(ResetProcessingCommand(book_id="book-reana"))
        with app.unit_of_work.read_transaction() as tx:
            head = tx.illustrations.get_head("book-reana")
            assert head.active_generation_id == gen.id
            t_after_reset = tx.tasks.get(task.id)
            assert t_after_reset is not None and t_after_reset.status is TaskStatus.PENDING

        # 2) ChooseBookConditions(action='reanalyze') revokes generation and supersedes tasks in same tx!
        choose = ChooseBookConditions(app.book_conditions, app.book_runtime)
        new_cond = choose(
            "book-reana",
            action="reanalyze",
            expected_revision=cond.revision,
            settings_revision=app.runtime_settings.revision,
            confirmed=True,
        )
        assert new_cond.revision == cond.revision + 1
        with app.unit_of_work.read_transaction() as tx:
            head = tx.illustrations.get_head("book-reana")
            assert head.active_generation_id is None
            gen_after = tx.illustrations.get_generation("book-reana", gen.id)
            assert gen_after is not None
            assert gen_after.state is IllustrationGenerationState.REVOKED
            assert gen_after.reason_code == IllustrationReasonCode.SOURCE_CHANGED.value
            t_after_reana = tx.tasks.get(task.id)
            assert t_after_reana is not None and t_after_reana.status is TaskStatus.SUPERSEDED
    finally:
        asyncio.run(app.aclose())


def test_reanalysis_cached_narrative_resumes_waiting_generation(tmp_path: Path) -> None:
    src = _make_source_epub(tmp_path)
    fake = _StagedIllustrationFakeLLM()
    app = _make_app(tmp_path, llm_client=fake)
    try:
        sec_id, blk_id, cond = _seed_book(
            app, "book-cache", source_path=src, status=BookStatus.ANALYZING
        )
        # Initial analysis produces cached NARRATIVE pass
        _seed_valid_narrative_pass(app, "book-cache", sec_id, blk_id)
        assert LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE in fake.calls

        # Reanalyze under new conditions with same NARRATIVE input hash
        choose = ChooseBookConditions(app.book_conditions, app.book_runtime)
        choose(
            "book-cache",
            action="reanalyze",
            expected_revision=cond.revision,
            settings_revision=app.runtime_settings.revision,
            confirmed=True,
        )
        settings_id = _enable_illustrations(app, "book-cache")

        # Create a WAITING_SOURCE generation before re-running AnalyzeChapter
        with app.unit_of_work.transaction() as tx:
            gen, _ = tx.illustrations.create_or_get_generation(
                "book-cache",
                settings_id=settings_id,
                request_key="req-cache",
                source_sha256=_source_sha(src),
                source_authority={},
                requested_sections=(sec_id,),
            )
            assert gen.state is IllustrationGenerationState.WAITING_SOURCE

        # Forbid any NARRATIVE LLM calls: AnalyzeChapter MUST hit cache AND still fire on_narrative_ready!
        fake.forbid_narrative = True
        fake.calls.clear()
        asyncio.run(app.analysis.start("book-cache"))
        # Run just the ANALYZE_CHAPTER task (not the newly enqueued illustration task yet)
        first = asyncio.run(app.runner.run_next())
        assert first is not None and first.task_type is TaskType.ANALYZE_CHAPTER
        assert first.status is TaskStatus.SUCCEEDED
        assert LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE not in fake.calls

        with app.unit_of_work.read_transaction() as tx:
            gen_after = tx.illustrations.get_generation("book-cache", gen.id)
            assert gen_after is not None
            assert gen_after.manifest_sha256 is not None
            assert gen_after.state is IllustrationGenerationState.QUEUED
    finally:
        asyncio.run(app.aclose())


def test_application_close_disposes_illustration_runtime(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    closed: list[bool] = []
    orig_close = app._close_illustrations

    async def spy_close() -> None:
        closed.append(True)
        if orig_close is not None:
            await orig_close()

    object.__setattr__(app, "_close_illustrations", spy_close)
    asyncio.run(app.aclose())
    assert closed == [True]
