"""Integration tests for WP-2 (CHAPTER lifecycle) and WP-6 (LLM diagnostics & attempt outcomes)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import event, text

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, ReadingSettings, StorageSettings
from linguary.adapters.db.engine import create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import LoggingLLMClient
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.analysis_pass_cache import encode_pass
from linguary.composition.book_runtime import conditions_payload
from linguary.application.context_builder import ContextBuilder
from linguary.application.diagnostics import Diagnostics, DiagnosticsPolicy
from linguary.application.illustration_chapter import IllustrationChapter
from linguary.application.illustration_coordinator import IllustrationCoordinator
from linguary.application.illustration_dto import (
    IllustrationCandidatesResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationMentionDTO,
    IllustrationSelectionResponse,
)
from linguary.application.illustration_reference import IllustrationReference
from linguary.application.illustration_source import (
    IllustrationSource,
    _pass_model_dependencies,
)
from linguary.application.illustrations import (
    IllustrationService,
    has_unfinished_illustration_work,
)
from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
from linguary.application.ports.illustrations import IllustrationRuntimePort, RequestWireSizer
from linguary.application.ports.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseKind,
    LLMTransportError,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.token_budget import narrative_max_fragment_blocks_for
from linguary.composition.illustration_runtime import IllustrationWireSizer
from linguary.composition.migrations import apply_migrations
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    LLMCallStage,
    SourceReference,
    SupportStatus,
    TaskType,
    TokenUsage,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import canonical_sha
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy
from linguary.domain.illustrations import (
    IllustrationAttemptState,
    IllustrationGenerationState,
    IllustrationPolicy,
    IllustrationReasonCode,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationSettingsRevision,
)


class _LoggedStubRuntime(IllustrationRuntimePort):
    """Runtime wrapping a scripted handler inside real `LoggingLLMClient`."""

    def __init__(
        self,
        uow: SqlAlchemyUnitOfWork,
        handler: Callable[[LLMRequest[Any], int], Any],
        *,
        debug_payloads: bool = False,
    ) -> None:
        self._uow = uow
        self._handler = handler
        self._debug_payloads = debug_payloads
        self.calls = 0

    def is_provider_ready(self) -> bool:
        return True

    def default_model(self) -> str | None:
        return "gpt-5-mini"

    def route_id(self) -> str:
        return "test-route"

    def sizer_for(self, revision: IllustrationSettingsRevision, stage: LLMCallStage) -> RequestWireSizer:
        del revision, stage
        return IllustrationWireSizer(
            settings=LLMSettings(
                model="gpt-5-mini",
                api_key="k",
                temperature=None,
                max_output_tokens=16384,
            ),
            include_openrouter_provider=False,
        )

    def client_for(self, revision: IllustrationSettingsRevision, stage: LLMCallStage) -> Any:
        del revision, stage
        parent = self

        class _Inner:
            async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
                parent.calls += 1
                result = parent._handler(request, parent.calls)
                if isinstance(result, Exception):
                    raise result
                if isinstance(result, tuple):
                    payload, raw_content = result
                else:
                    payload = result
                    raw_content = '{"stub":"response"}'
                return LLMResponse(
                    payload=payload,
                    provider="fake",
                    model="gpt-5-mini",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
                    raw_content=raw_content,
                    call_context=request.call_context,
                )

        return LoggingLLMClient(
            inner=_Inner(),  # type: ignore[arg-type]
            unit_of_work=self._uow,
            provider="fake",
            model="gpt-5-mini",
            debug_payloads=self._debug_payloads,
        )


def _setup_two_chapter_app(
    tmp_path: Path,
    handler: Callable[[LLMRequest[Any], int], Any],
    *,
    policy: IllustrationPolicy | None = None,
    sec0_scenes: int = 1,
    debug_payloads: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "integ.db"
    apply_migrations(db)
    engine = create_database_engine(db)
    uow = SqlAlchemyUnitOfWork.for_engine(engine)
    Diagnostics(uow, policy=DiagnosticsPolicy())

    book_id = "book-wp2"
    sec0_blocks = tuple(
        DocumentBlock(
            id=f"sec0-b{i}",
            section_id="sec-0",
            type=BlockType.PARAGRAPH,
            order=i,
            text=f"Alice walks through courtyard scene {i}.",
            source_reference=SourceReference("Text/ch1.xhtml", f"p[{i}]"),
            support_status=SupportStatus.SUPPORTED,
        )
        for i in range(sec0_scenes)
    )
    sec1_blocks = (
        DocumentBlock(
            id="sec1-b0",
            section_id="sec-1",
            type=BlockType.PARAGRAPH,
            order=0,
            text="Alice reaches the calm river bank.",
            source_reference=SourceReference("Text/ch2.xhtml", "p[0]"),
            support_status=SupportStatus.SUPPORTED,
        ),
    )
    doc = CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title="Two Chapters"),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id="sec-0",
                document_id=book_id,
                title="Chapter 1",
                order=0,
                source_resource_path="Text/ch1.xhtml",
                blocks=sec0_blocks,
            ),
            DocumentSection(
                id="sec-1",
                document_id=book_id,
                title="Chapter 2",
                order=1,
                source_resource_path="Text/ch2.xhtml",
                blocks=sec1_blocks,
            ),
        ),
    )
    epub_file = tmp_path / "two.epub"
    epub_file.write_bytes(b"fake-two-epub")
    book = BookRecord(
        id=book_id,
        metadata=BookMetadata(title="Two Chapters"),
        source_language="en",
        target_language="ru",
        status=BookStatus.GLOSSARY_REVIEW,
        source_epub_path=str(epub_file),
        created_at=datetime(2026, 11, 25, tzinfo=UTC),
    )
    app_settings = AppSettings(
        llm=LLMSettings(
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
            api_key="test-key",
            max_output_tokens=16384,
            staged_analysis=True,
        ),
        checks=CheckSettings(sequential_analysis=False),
        reading=ReadingSettings(reader_level="b1", name_translation_strategy="sound"),
        storage=StorageSettings(data_root=str(tmp_path / "data")),
    )
    conditions_id = str(uuid4())
    cond_payload = conditions_payload(book, app_settings, revision=1, profile="content-plan-v3")
    cond_payload["source_sha256"] = "a" * 64
    prompts_version = int(cond_payload["versions"]["prompts"])
    ctx_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        name_translation_policy=NameTranslationPolicy(
            version=1, strategy=NameTranslationStrategy.SOUND
        ),
        reader_level="b1",
    )
    model_deps = _pass_model_dependencies(cond_payload)
    max_frag_blocks = narrative_max_fragment_blocks_for(cond_payload["llm_parameters"])

    with uow.transaction() as tx:
        tx.books.add(book)
        tx.documents.save(doc)
        tx.work.add_conditions(
            BookConditions(
                id=conditions_id,
                book_id=book_id,
                revision=1,
                payload=cond_payload,
                created_at=datetime(2026, 11, 25, tzinfo=UTC),
            )
        )
        narr0 = NarrativeAnalysisResponse(
            translation_instructions="Сохраняй стиль.",
            chapter_summary="Alice crosses the courtyard.",
            glossary_candidates=(),
            context_facts=(),
            scenes=tuple(
                QuotedSceneResponse(
                    start_block_id=f"sec0-b{i}",
                    end_block_id=f"sec0-b{i}",
                    summary=f"Сцена {i}",
                    location="Двор",
                    participants=("Alice",),
                    visual_facts=(),
                    visual_assumptions=(),
                    continues_previous_scene=False,
                )
                for i in range(sec0_scenes)
            ),
        )
        narr1 = NarrativeAnalysisResponse(
            translation_instructions="Сохраняй стиль.",
            chapter_summary="Alice reaches the river bank.",
            glossary_candidates=(),
            context_facts=(),
            scenes=(
                QuotedSceneResponse(
                    start_block_id="sec1-b0",
                    end_block_id="sec1-b0",
                    summary="Берег реки",
                    location="Река",
                    participants=("Alice",),
                    visual_facts=(),
                    visual_assumptions=(),
                    continues_previous_scene=False,
                ),
            ),
        )
        for sec, narr in zip(doc.ordered_sections, (narr0, narr1), strict=True):
            encoded = encode_pass(narr)
            encoded["input_suggestions"] = []
            encoded["input_context"] = {
                "schema_version": 1,
                "sequential_analysis": False,
                "parallel_batch": False,
                "seed_active": False,
                "user_decisions": [],
                "nonbinding_suggestions": [],
                "prompts_version": prompts_version,
            }
            input_sha = ctx_builder.narrative_input_sha256(
                doc,
                sec,
                (),
                (),
                (),
                model_dependencies=model_deps,
                max_fragment_blocks=max_frag_blocks,
            )
            tx.analyses.save_pass(
                AnalysisPassResult(
                    id=f"pass-{sec.id}",
                    book_id=book_id,
                    section_id=sec.id,
                    pass_kind=AnalysisPassKind.NARRATIVE,
                    scope_key="",
                    input_sha256=input_sha,
                    output_sha256=canonical_sha(encoded),
                    conditions_id=conditions_id,
                    attempt_id=None,
                    payload=encoded,
                )
            )

    source = IllustrationSource(
        language_detector=Py3LangidDetector(),
        source_digest_fn=lambda _p: "a" * 64,
    )
    runtime = _LoggedStubRuntime(uow, handler, debug_payloads=debug_payloads)
    chapter = IllustrationChapter(uow, source=source, runtime=runtime, sleep_fn=lambda _: None)
    reference = IllustrationReference(
        uow,
        source=source,
        runtime=runtime,
        sleep_fn=lambda _: None,
        plan_chapter_scopes=chapter.plan_chapter_scopes,
    )
    coordinator = IllustrationCoordinator(
        uow,
        source=source,
        plan_reference_scopes=reference.plan_scopes,
        plan_chapter_scopes=chapter.plan_chapter_scopes,
    )
    service = IllustrationService(
        uow,
        provider_ready_fn=runtime.is_provider_ready,
        default_model_fn=runtime.default_model,
        source_snapshot_fn=source.snapshot,
        reconcile_in_tx_fn=coordinator.reconcile_in_transaction,
    )
    settings = IllustrationSettings(
        enabled=True,
        max_per_chapter=2,
        model_reference="gpt-5-mini",
        model_prompts="gpt-5-mini",
        policy=policy or IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=3),
    )
    rev = service.save_settings(book_id, expected_revision=0, settings=settings)
    return engine, uow, source, runtime, reference, chapter, coordinator, service, book_id, rev


async def _drain(
    uow: SqlAlchemyUnitOfWork,
    book_id: str,
    reference: IllustrationReference,
    chapter: IllustrationChapter,
    *,
    max_steps: int = 50,
) -> None:
    for _ in range(max_steps):
        with uow.read_transaction() as tx:
            pending = [t for t in tx.tasks.list_for_book(book_id) if t.status.value == "pending"]
        if not pending:
            return
        for task in pending:
            if task.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE:
                await reference.run_task(task)
            elif task.task_type is TaskType.PREPARE_CHAPTER_ILLUSTRATIONS:
                await chapter.run_task(task)


def _ok_extract_and_design(req: LLMRequest[Any]) -> Any | None:
    data = json.loads(req.input_text)
    if req.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
        units = data["units"]
        first_blk = units[0]["blocks"][0]["block_id"]
        return IllustrationExtractResponse(
            mentions=(
                IllustrationMentionDTO(
                    mention_id="m1",
                    name="Alice",
                    entity_kind="person",
                    alias_of_hint=None,
                    quote_block_id=first_blk,
                    quote="Alice",
                ),
            ),
            facts=(),
            covered_source_unit_ids=tuple(u["source_unit_id"] for u in units),
            complete=True,
        )
    if req.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
        return IllustrationDesignResponse(
            entity_id=data["entity_id"],
            traits=(
                IllustrationDesignTraitDTO(
                    trait="palette_value",
                    description_ru="Контрастная штриховка тушью",
                    description_en="high-contrast black ink hatching",
                    assumption_ru="Монохромный стиль",
                ),
            ),
        )
    if req.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
        return IllustrationSelectionResponse(
            selected_candidate_ids=(),
            zero_reason_ru="Спокойная сцена.",
        )
    return None


# ── WP-2 Integration Scenarios ──────────────────────────────────────────────


async def test_wp2_failed_candidate_marks_chapter_and_generation_failed_and_retry_recovers(
    tmp_path: Path,
) -> None:
    fail_sec0 = {"active": True}

    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        base = _ok_extract_and_design(req)
        if base is not None:
            return base
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            if data["section_id"] == "sec-0" and fail_sec0["active"]:
                raise LLMProviderError("провайдер отверг запрос: 404 - No endpoints found")
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Нет яркого кадра.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(f"unexpected {req.kind}")

    engine, uow, _src, _rt, ref, ch, _coord, svc, book_id, rev = _setup_two_chapter_app(
        tmp_path, _handler
    )
    view0 = await svc.prepare(book_id, expected_revision=rev.revision)
    gen_id = view0.active_generation_id
    assert gen_id is not None

    await _drain(uow, book_id, ref, ch)

    # Generation is FAILED, chapter 1 is failed with route_unavailable, chapter 2 is ready.
    with uow.read_transaction() as tx:
        gen = tx.illustrations.get_generation(book_id, gen_id)
        assert gen is not None
        assert gen.state is IllustrationGenerationState.FAILED
        assert gen.reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value
        assert has_unfinished_illustration_work(tx, book_id) is False

    # SQL check: get_chapter_header_page must NOT select payload_json even with pending_sids!
    executed_sql: list[str] = []

    def _capture_sql(_conn, _cursor, statement, _params, _ctx, _execmany):
        executed_sql.append(statement)

    event.listen(engine, "before_cursor_execute", _capture_sql)
    try:
        view_failed = svc.get(book_id, section_id="sec-0")
    finally:
        event.remove(engine, "before_cursor_execute", _capture_sql)

    assert view_failed.state == "failed"
    assert view_failed.reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value
    headers_by_sid = {h.section_id: h for h in view_failed.chapter_headers}
    assert headers_by_sid["sec-0"].state == "failed"
    assert headers_by_sid["sec-0"].reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value
    assert headers_by_sid["sec-1"].state == "ready"
    assert view_failed.selected_chapter is not None
    assert view_failed.selected_chapter.state == "failed"
    assert view_failed.selected_chapter.reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value
    # Header queries must not select payload_json (only get_scope for selected_chapter loads payload)
    header_queries = [
        sql for sql in executed_sql if "illustration_scopes" in sql and "stage !=" in sql
    ]
    assert header_queries
    assert all("payload_json" not in sql for sql in header_queries)

    # Retry must requeue ONLY the failed CANDIDATES scope, never create a task for virtual CHAPTER.
    fail_sec0["active"] = False
    await svc.retry(book_id, expected_revision=rev.revision, expected_generation_id=gen_id)
    with uow.read_transaction() as tx:
        ch_scope = tx.illustrations.list_scopes(
            gen_id, stage=IllustrationScopeStage.CHAPTER, section_id="sec-0"
        )[0]
        assert ch_scope.task_id is None
        pending_tasks = [t for t in tx.tasks.list_for_book(book_id) if t.status.value == "pending"]
        assert len(pending_tasks) == 1
        assert pending_tasks[0].scope_id != ch_scope.id

    await _drain(uow, book_id, ref, ch)
    view_ready = svc.get(book_id, section_id="sec-0")
    assert view_ready.state == "ready"
    assert view_ready.selected_chapter is not None
    assert view_ready.selected_chapter.state == "ready"


async def test_wp2_parallel_pending_scope_holds_chapter_queued_until_last_finishes(
    tmp_path: Path,
) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        base = _ok_extract_and_design(req)
        if base is not None:
            return base
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            if data["blocks"][0]["block_id"] == "sec0-b0":
                raise LLMProviderError("провайдер отверг запрос: 404 - No endpoints found")
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Нет кадра.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(f"unexpected {req.kind}")

    _engine, uow, _src, _rt, ref, ch, _coord, svc, book_id, rev = _setup_two_chapter_app(
        tmp_path, _handler, sec0_scenes=2
    )
    view0 = await svc.prepare(book_id, expected_revision=rev.revision, section_id="sec-0")
    gen_id = view0.active_generation_id
    assert gen_id is not None

    # Finish reference phase so two CANDIDATES scopes exist for sec-0.
    for _ in range(10):
        with uow.read_transaction() as tx:
            ref_tasks = [
                t
                for t in tx.tasks.list_for_book(book_id)
                if t.status.value == "pending"
                and t.task_type is TaskType.PREPARE_ILLUSTRATION_REFERENCE
            ]
        if not ref_tasks:
            break
        for t in ref_tasks:
            await ref.run_task(t)

    with uow.read_transaction() as tx:
        cand_tasks = sorted(
            [
                t
                for t in tx.tasks.list_for_book(book_id)
                if t.status.value == "pending"
                and t.task_type is TaskType.PREPARE_CHAPTER_ILLUSTRATIONS
            ],
            key=lambda t: t.id,
        )
    assert len(cand_tasks) == 2

    # Run only the failing scene first; second scene is still PENDING -> chapter & generation stay active!
    with uow.read_transaction() as tx:
        scopes_by_id = {s.id: s for s in tx.illustrations.list_scopes(gen_id)}
    failing_task = next(
        t for t in cand_tasks if "sec0-b0" in scopes_by_id[t.scope_id].summary["ordered_block_ids"]  # type: ignore[index]
    )
    other_task = next(t for t in cand_tasks if t.id != failing_task.id)

    await ch.run_task(failing_task)
    mid_view = svc.get(book_id, section_id="sec-0")
    assert mid_view.state == "queued"
    assert mid_view.selected_chapter is not None
    assert mid_view.selected_chapter.state == "queued"

    # Now finish the second scene -> no active executable scopes remain -> both transition to failed!
    await ch.run_task(other_task)
    final_view = svc.get(book_id, section_id="sec-0")
    assert final_view.state == "failed"
    assert final_view.selected_chapter is not None
    assert final_view.selected_chapter.state == "failed"
    assert final_view.selected_chapter.reason_code == IllustrationReasonCode.ROUTE_UNAVAILABLE.value


async def test_wp2_prepare_cached_second_chapter_when_first_failed_returns_to_failed(
    tmp_path: Path,
) -> None:
    def _handler(req: LLMRequest[Any], _n: int) -> Any:
        base = _ok_extract_and_design(req)
        if base is not None:
            return base
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            if data["section_id"] == "sec-0":
                raise LLMProviderError("провайдер отверг запрос: 404 - No endpoints found")
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Нет кадра.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(f"unexpected {req.kind}")

    _engine, uow, _src, _rt, ref, ch, _coord, svc, book_id, rev = _setup_two_chapter_app(
        tmp_path, _handler
    )
    # 1. Generate sec-1 cleanly in generation 1 so CHAPTER(sec-1) is cached in DB.
    v1 = await svc.prepare(book_id, expected_revision=rev.revision, section_id="sec-1")
    await _drain(uow, book_id, ref, ch)
    assert svc.get(book_id).state == "ready"
    assert v1.active_generation_id is not None

    # 2. Revoke gen 1 and start gen 2 with only sec-0 (which fails).
    with uow.transaction() as tx:
        tx.illustrations.revoke_generation(
            book_id, v1.active_generation_id, reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value
        )
    v2 = await svc.prepare(book_id, expected_revision=rev.revision, section_id="sec-0")
    gen2_id = v2.active_generation_id
    assert gen2_id is not None
    await _drain(uow, book_id, ref, ch)
    assert svc.get(book_id).state == "failed"

    # 3. Request sec-1 on gen 2 -> instantly reused from gen 1 cache (dispatched=False).
    # Generation MUST NOT stay stuck in QUEUED; finalize_generation_state_in_tx returns it to FAILED.
    v_after = await svc.prepare(
        book_id,
        expected_revision=rev.revision,
        expected_generation_id=gen2_id,
        section_id="sec-1",
    )
    assert v_after.state == "failed"
    with uow.read_transaction() as tx:
        gen2 = tx.illustrations.get_generation(book_id, gen2_id)
        assert gen2 is not None
        assert gen2.state is IllustrationGenerationState.FAILED


async def test_wp2_recover_interrupted_transitions_pending_chapter_with_failed_child_to_failed(
    tmp_path: Path,
) -> None:
    _engine, uow, source, _rt, _ref, _ch, coord, _svc, book_id, rev = _setup_two_chapter_app(
        tmp_path, lambda _r, _n: None
    )
    with uow.transaction() as tx:
        snap = source.snapshot(tx, book_id)
        gen, _ = tx.illustrations.create_or_get_generation(
            book_id,
            settings_id=rev.id,
            request_key="req-rec",
            source_sha256=snap.source_sha256 or "",
            source_authority=snap.source_authority or {},
            requested_sections=("sec-0",),
        )
        assert snap.manifest is not None and snap.manifest_sha256 is not None
        tx.illustrations.seal_manifest(
            book_id, gen.id, manifest=snap.manifest, manifest_sha256=snap.manifest_sha256
        )
        tx.illustrations.seal_registry(book_id, gen.id, registry_sha256="r" * 64)
        tx.illustrations.update_generation_state(
            book_id, gen.id, state=IllustrationGenerationState.RUNNING, reason_code=None
        )
        tx.illustrations.upsert_scopes(
            gen.id,
            [
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.CHAPTER,
                    scope_key="chapter:sec-0",
                    input_sha256="c" * 64,
                    section_id="sec-0",
                    section_order=0,
                    section_title="Chapter 1",
                    state=IllustrationScopeState.PENDING,
                ),
                IllustrationScopeSpec(
                    stage=IllustrationScopeStage.CANDIDATES,
                    scope_key="cand:sec-0:s000:p000",
                    input_sha256="d" * 64,
                    section_id="sec-0",
                    section_order=0,
                    section_title="Chapter 1",
                    state=IllustrationScopeState.FAILED,
                    reason_code=IllustrationReasonCode.PROVIDER_ERROR.value,
                ),
            ],
        )
        gen_id = gen.id

    coord.recover_interrupted()
    with uow.read_transaction() as tx:
        gen_after = tx.illustrations.get_generation(book_id, gen_id)
        assert gen_after is not None
        assert gen_after.state is IllustrationGenerationState.FAILED
        assert gen_after.reason_code == IllustrationReasonCode.PROVIDER_ERROR.value


# ── WP-6 Integration Scenarios (Diagnostics & outcome_json) ─────────────────


async def test_wp6_diagnostics_rejected_accepted_stale_and_no_overwrite_after_transport_error(
    tmp_path: Path,
) -> None:
    # Scenario A: 1st EXTRACT attempt semantically rejected (bad covered_source_unit_ids),
    # 2nd attempt fails with LLMTransportError before receiving a response ->
    # 1st llm_call_log stays rejected (success=False, body retained even with debug_payloads=False),
    # attempt 1 outcome_json has reason_code + detail without "Исправь ответ: ".
    def _handler_reject_then_transport(req: LLMRequest[Any], call_no: int) -> Any:
        if call_no == 1:
            return (
                IllustrationExtractResponse(
                    mentions=(),
                    facts=(),
                    covered_source_unit_ids=(),
                    complete=True,
                ),
                '{"mentions":[],"facts":[],"covered_source_unit_ids":[],"complete":true}',
            )
        raise LLMTransportError("connection dropped before response")

    _eng, uow, _src, _rt, ref, ch, _coord, svc, book_id, rev = _setup_two_chapter_app(
        tmp_path / "case_a",
        _handler_reject_then_transport,
        policy=IllustrationPolicy(max_repair_turns=1, max_calls_per_scope_activation=2),
        debug_payloads=False,
    )
    view = await svc.prepare(book_id, expected_revision=rev.revision, section_id="sec-0")
    gen_id = view.active_generation_id
    assert gen_id is not None
    await _drain(uow, book_id, ref, ch)

    with uow.read_transaction() as tx:
        calls = tx.llm_calls.list_for_book(book_id)
        extract_calls = [c for c in calls if c.stage is LLMCallStage.ILLUSTRATION_REFERENCE]
        assert len(extract_calls) >= 2
        first_call = extract_calls[0]
        assert first_call.success is False
        rec = tx.llm_calls.diagnostic_record(attempt_id=first_call.attempt_id)
        assert rec is not None
        assert rec["metadata"]["decision"] == "rejected"
        assert rec["body"] is not None and "covered_source_unit_ids" in rec["body"]

        extract_scope = tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.EXTRACT)[0]
    with _eng.connect() as conn:
        attempt_rows = conn.execute(
            text(
                "SELECT state, outcome_json FROM illustration_attempts "
                "WHERE scope_id = :sid ORDER BY ordinal ASC"
            ),
            {"sid": extract_scope.id},
        ).all()
    assert len(attempt_rows) == 2
    att0_state, att0_outcome_raw = attempt_rows[0]
    assert att0_state == IllustrationAttemptState.REJECTED.value
    att0_outcome = (
        json.loads(att0_outcome_raw) if isinstance(att0_outcome_raw, str) else att0_outcome_raw
    )
    assert att0_outcome is not None
    assert att0_outcome["reason_code"] == IllustrationReasonCode.MODEL_OUTPUT_INVALID.value
    assert "detail" in att0_outcome
    assert not att0_outcome["detail"].startswith("Исправь ответ:")
    assert len(att0_outcome["detail"].encode("utf-8")) <= 256

    # Scenario B: Accepted scope records decision == "accepted".
    def _handler_accept(req: LLMRequest[Any], _n: int) -> Any:
        base = _ok_extract_and_design(req)
        if base is not None:
            return base
        data = json.loads(req.input_text)
        if req.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            return IllustrationCandidatesResponse(
                scene_id=data["scene_id"],
                events=(),
                zero_reason_ru="Нет событий.",
                covered_source_unit_ids=tuple(b["source_unit_id"] for b in data["blocks"]),
                complete=True,
            )
        raise AssertionError(f"unexpected {req.kind}")

    _eng_b, uow_b, _src_b, _rt_b, ref_b, ch_b, _coord_b, svc_b, book_b, rev_b = (
        _setup_two_chapter_app(tmp_path / "case_b", _handler_accept, debug_payloads=False)
    )
    await svc_b.prepare(book_b, expected_revision=rev_b.revision, section_id="sec-0")
    await _drain(uow_b, book_b, ref_b, ch_b)
    with uow_b.read_transaction() as tx:
        calls_b = tx.llm_calls.list_for_book(book_b)
        assert calls_b
        for c in calls_b:
            rec_b = tx.llm_calls.diagnostic_record(attempt_id=c.attempt_id)
            assert rec_b is not None
            assert rec_b["metadata"]["decision"] == "accepted"

    # Scenario C: Stale attempt (generation revoked mid-call) records decision == "stale".
    gen_c_holder: dict[str, str] = {}

    def _handler_revoke_mid_call(req: LLMRequest[Any], _n: int) -> Any:
        with uow_c.transaction() as tx:
            tx.illustrations.revoke_generation(
                book_c,
                gen_c_holder["id"],
                reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value,
            )
        return _ok_extract_and_design(req)

    _eng_c, uow_c, _src_c, _rt_c, ref_c, ch_c, _coord_c, svc_c, book_c, rev_c = (
        _setup_two_chapter_app(tmp_path / "case_c", _handler_revoke_mid_call, debug_payloads=False)
    )
    view_c = await svc_c.prepare(book_c, expected_revision=rev_c.revision, section_id="sec-0")
    assert view_c.active_generation_id is not None
    gen_c_holder["id"] = view_c.active_generation_id
    await _drain(uow_c, book_c, ref_c, ch_c)
    with uow_c.read_transaction() as tx:
        calls_c = tx.llm_calls.list_for_book(book_c)
        assert len(calls_c) == 1
        rec_c = tx.llm_calls.diagnostic_record(attempt_id=calls_c[0].attempt_id)
        assert rec_c is not None
        assert rec_c["metadata"]["decision"] == "stale"
