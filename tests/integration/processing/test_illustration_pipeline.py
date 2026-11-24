"""End-to-end integration tests for illustration pipeline invariants and isolation (`P5.T1`)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import event

from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
from linguary.application.book_overview import book_overview
from linguary.application.illustration_dto import (
    IllustrationAliasResponse,
    IllustrationCandidatesResponse,
    IllustrationDesignResponse,
    IllustrationDesignTraitDTO,
    IllustrationExtractResponse,
    IllustrationMentionDTO,
    IllustrationSelectionResponse,
)
from linguary.application.llm_dto import (
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedSceneResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.epub import ExportedBook
from linguary.application.ports.export_validation import ExportValidationReport
from linguary.application.ports.llm import (
    LLMRequest,
    LLMResponse,
    LLMResponseKind,
    LLMTransportError,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.use_cases.export_book import (
    ExportBook,
    ExportBookCommand,
    ExportMode,
)
from linguary.composition import build_application, drain_tasks
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    CoverageStatus,
    DocumentBlock,
    DocumentSection,
    GlossarySnapshot,
    LLMCallLog,
    LLMCallStage,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    SourceReference,
    SupportStatus,
    TaskType,
    TokenUsage,
    TreatmentPlan,
)
from linguary.domain.content_plan import ContentPlan
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
)
from linguary.web.app import create_app


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    return json.loads(text[start : end + 1])


class _PipelineFakeLLM:
    supports_staged_analysis = True

    def __init__(self, *, fail_section_once: str | None = None) -> None:
        self.fail_section_once = fail_section_once
        self._failed_once = False
        self.calls: list[tuple[LLMResponseKind, str | None]] = []

    async def generate(self, request: LLMRequest[Any]) -> LLMResponse[Any]:
        self.calls.append((request.kind, request.scope_id))
        usage = TokenUsage(prompt_tokens=20, completion_tokens=20)
        if request.kind is LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE:
            aliases = tuple(alias for alias, _ in request.block_aliases) or ("b1",)
            return LLMResponse(
                payload=NarrativeAnalysisResponse(
                    translation_instructions="Сохраняй ровный повествовательный стиль.",
                    chapter_summary="Alice crossed the courtyard toward the old clocktower.",
                    scenes=(
                        QuotedSceneResponse(
                            start_block_id=aliases[0],
                            end_block_id=aliases[-1],
                            summary="Алиса пересекает двор у старой часовой башни.",
                            location="Двор",
                            participants=("Alice",),
                        ),
                    ),
                ),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL:
            aliases = tuple(alias for alias, _ in request.block_aliases) or ("b1",)
            return LLMResponse(
                payload=PedagogicalAnalysisResponse(
                    treatment_plan=tuple(
                        TreatmentPlanItemResponse(
                            source_block_id=a, treatment=BlockTreatment.TRANSLATE
                        )
                        for a in aliases
                    )
                ),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_EXTRACT:
            data = _extract_json(request.input_text)
            units = data.get("units", ())
            unit_ids = tuple(str(u["source_unit_id"]) for u in units if isinstance(u, dict))
            first_blk_id = None
            for u in units:
                for blk in u.get("blocks", ()):
                    first_blk_id = str(blk["block_id"])
                    break
                if first_blk_id:
                    break
            mentions = (
                (
                    IllustrationMentionDTO(
                        mention_id="m1",
                        name="Alice",
                        entity_kind="person",
                        alias_of_hint=None,
                        quote_block_id=first_blk_id,
                        quote="Alice",
                    ),
                )
                if first_blk_id
                else ()
            )
            return LLMResponse(
                payload=IllustrationExtractResponse(
                    mentions=mentions,
                    facts=(),
                    covered_source_unit_ids=unit_ids,
                    complete=True,
                ),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_ALIAS:
            return LLMResponse(
                payload=IllustrationAliasResponse(decisions=()),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_DESIGN:
            data = _extract_json(request.input_text)
            entity = data.get("entity", {})
            eid = str(entity.get("entity_id") or data.get("entity_id") or "ent-1")
            trait = str((data.get("allowed_traits") or ["clothing"])[0])
            return LLMResponse(
                payload=IllustrationDesignResponse(
                    entity_id=eid,
                    traits=(
                        IllustrationDesignTraitDTO(
                            trait=trait,
                            description_ru="Тёмный плащ.",
                            description_en="Dark cloak.",
                            assumption_ru="Предположение.",
                        ),
                    ),
                ),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
            data = _extract_json(request.input_text)
            sec_id = str(data.get("section_id") or "")
            if self.fail_section_once == sec_id and not self._failed_once:
                self._failed_once = True
                raise LLMTransportError("simulated chapter 2 outage")
            scene_id = str(data.get("scene_id") or "s000")
            unit_ids = tuple(
                str(b["source_unit_id"])
                for b in data.get("blocks", ())
                if isinstance(b, dict)
            )
            return LLMResponse(
                payload=IllustrationCandidatesResponse(
                    scene_id=scene_id,
                    events=(),
                    zero_reason_ru="Обычный переходный эпизод без выразительной сцены.",
                    covered_source_unit_ids=unit_ids,
                    complete=True,
                ),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        if request.kind is LLMResponseKind.ILLUSTRATION_SELECTION:
            return LLMResponse(
                payload=IllustrationSelectionResponse(
                    selected_candidate_ids=(),
                    zero_reason_ru="Спокойная глава.",
                ),
                provider="fake",
                model="gpt-5-mini",
                usage=usage,
            )
        raise AssertionError(f"unexpected kind: {request.kind}")


def _make_doc(book_id: str, chapter_count: int) -> CanonicalDocument:
    sections = []
    for idx in range(chapter_count):
        sec_id = f"{book_id}-sec-{idx:02d}"
        sections.append(
            DocumentSection(
                id=sec_id,
                document_id=book_id,
                title=f"Chapter {idx + 1}",
                order=idx,
                source_resource_path=f"Text/ch{idx + 1}.xhtml",
                blocks=(
                    DocumentBlock(
                        id=f"{book_id}:b{idx}",
                        section_id=sec_id,
                        type=BlockType.PARAGRAPH,
                        order=0,
                        text=f"Alice crossed the quiet stone courtyard in chapter {idx + 1}.",
                        source_reference=SourceReference(f"Text/ch{idx + 1}.xhtml", "p[0]"),
                        support_status=SupportStatus.SUPPORTED,
                    ),
                ),
            )
        )
    return CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=f"Book {book_id}"),
        source_language="en",
        target_language="ru",
        sections=tuple(sections),
    )


def test_limit_and_model_change_preserves_translation_ids_hashes_and_export(tmp_path: Path) -> None:
    fake_epub = tmp_path / "src.epub"
    fake_epub.write_bytes(b"PK\x03\x04pipeline-epub")
    fake_llm = _PipelineFakeLLM()
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(api_key="k", model="gpt-5-mini", staged_analysis=True, max_attempts=1),
        ),
        llm_client=fake_llm,
        content_plan_profile="content-plan-v3",
    )
    try:
        doc = _make_doc("book-preserve", 1)
        sec = doc.sections[0]
        blk = sec.blocks[0]
        book = BookRecord(
            id=doc.id,
            metadata=doc.metadata,
            source_language="en",
            target_language="ru",
            status=BookStatus.READY_TO_EXPORT,
            source_epub_path=str(fake_epub),
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
        )
        plan = ContentPlan(version=3, profile="content-plan-v3")
        run = ProcessingRun(
            id="run-1",
            book_id=doc.id,
            status=ProcessingRunStatus.ACTIVE,
            glossary_snapshot=GlossarySnapshot(entries=()),
            llm_settings_snapshot=LLMSettingsSnapshot(
                provider="openai_compatible",
                model="gpt-5-mini",
            ),
        )
        block_res = BlockProcessingResult(
            block_id=blk.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Алиса пересекла тихий каменный двор в первой главе.",
        )
        with app.unit_of_work.transaction() as tx:
            tx.books.add(book)
            tx.documents.save(doc)
            app.book_conditions.create(tx, book, app.runtime_settings.revision)
            tx.analyses.save(
                doc.id,
                ChapterAnalysis(
                    id="an-1",
                    section_id=sec.id,
                    coverage_status=CoverageStatus.COMPLETE,
                    treatment_plan=TreatmentPlan(section_id=sec.id, items=()),
                    translation_instructions="Переводи точно.",
                    chapter_summary="Alice crossed the courtyard.",
                    content_plan=plan,
                ),
            )
            tx.runs.add(run)
            tx.results.replace_results(run.id, (block_res,))
            tx.llm_calls.add(
                LLMCallLog(
                    id="log-ord-1",
                    book_id=doc.id,
                    stage=LLMCallStage.CHAPTER_ANALYSIS,
                    provider="openai_compatible",
                    model="gpt-5-mini",
                    started_at=datetime(2026, 11, 12, tzinfo=UTC),
                    duration_ms=50,
                    attempt=1,
                    success=True,
                    usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
                    cost=Decimal("0.25"),
                )
            )

        plan_sha_before = plan.sha256
        ov_before = book_overview(app.unit_of_work, doc.id)

        # Update max_per_chapter & model_prompts via real HTTP API-02
        web_app = create_app(application=app, run_background_worker=False)
        with TestClient(web_app, base_url="http://127.0.0.1:8000") as client:
            resp = client.post(
                f"/books/{doc.id}/illustrations/settings",
                json={
                    "expected_revision": 0,
                    "enabled": True,
                    "max_per_chapter": 1,
                    "model_reference": "gpt-5-mini",
                    "model_prompts": "gpt-5-pro",
                },
                headers={"Accept": "application/json"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["settings"]["style"] == "monochrome_ink"
            assert body["settings"]["max_per_chapter"] == 1
            assert body["settings"]["model_prompts"] == "gpt-5-pro"

            resp2 = client.post(
                f"/books/{doc.id}/illustrations/settings",
                json={
                    "expected_revision": 1,
                    "enabled": True,
                    "max_per_chapter": 3,
                    "model_reference": "gpt-5-mini",
                    "model_prompts": "gpt-5-mini",
                },
                headers={"Accept": "application/json"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["settings"]["max_per_chapter"] == 3

        # Verify translation IDs, ContentPlan.sha256, ordinary_cost, source resources & export untouched!
        with app.unit_of_work.read_transaction() as tx:
            analyses_after = tx.analyses.list_for_book(doc.id)
            assert len(analyses_after) == 1
            assert analyses_after[0].content_plan is not None
            assert analyses_after[0].content_plan.sha256 == plan_sha_before
            results_after = tx.results.list_for_run(run.id)
            assert [r.block_id for r in results_after] == [blk.id]
            assert results_after[0].translated_text == block_res.translated_text
            book_after = tx.books.get(doc.id)
            assert book_after is not None and book_after.source_epub_path == str(fake_epub)

        ov_after = book_overview(app.unit_of_work, doc.id)
        assert ov_after.ordinary_cost == ov_before.ordinary_cost

        class _StubWriter:
            def write(self, request: Any) -> ExportedBook:
                Path(request.target_path).write_bytes(b"epub-preserved")
                return ExportedBook(path=request.target_path, warnings=())

        class _StubValidator:
            def validate(self, _request: Any) -> ExportValidationReport:
                return ExportValidationReport(available=False, unavailable_reason="stub")

        out_path = tmp_path / "export.epub"
        exp = ExportBook(
            unit_of_work=app.unit_of_work,
            writer=_StubWriter(),  # type: ignore[arg-type]
            validator=_StubValidator(),  # type: ignore[arg-type]
        )
        exp_res = exp(
            ExportBookCommand(
                book_id=doc.id, target_path=str(out_path), mode=ExportMode.STRUCTURE_ONLY
            )
        )
        assert exp_res.book_id == doc.id and out_path.exists()
    finally:
        asyncio.run(app.aclose())


def test_no_hidden_paid_reanalysis(tmp_path: Path) -> None:
    fake_epub = tmp_path / "legacy.epub"
    fake_epub.write_bytes(b"PK\x03\x04legacy-epub")
    fake_llm = _PipelineFakeLLM()
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(api_key="k", model="gpt-5-mini", staged_analysis=True, max_attempts=1),
        ),
        llm_client=fake_llm,
        content_plan_profile="legacy-v1",
    )
    try:
        doc = _make_doc("book-legacy", 1)
        book = BookRecord(
            id=doc.id,
            metadata=doc.metadata,
            source_language="en",
            target_language="ru",
            status=BookStatus.GLOSSARY_REVIEW,
            source_epub_path=str(fake_epub),
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
        )
        with app.unit_of_work.transaction() as tx:
            tx.books.add(book)
            tx.documents.save(doc)
            app.book_conditions.create(tx, book, app.runtime_settings.revision)

        # Saving settings does NOT trigger analysis or LLM calls
        app.illustrations.save_settings(
            doc.id,
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        view = app.illustrations.get(doc.id)
        assert view.state == "needs_analysis"
        assert "prepare" not in view.allowed_actions

        # Draining queue produces 0 analysis tasks and 0 LLM calls
        asyncio.run(drain_tasks(app))
        assert fake_llm.calls == []
        with app.unit_of_work.read_transaction() as tx:
            assert not any(
                t.task_type is TaskType.ANALYZE_CHAPTER
                for t in tx.tasks.list_for_book(doc.id)
            )
    finally:
        asyncio.run(app.aclose())


def test_many_chapters_bounded_queries(tmp_path: Path) -> None:
    fake_epub = tmp_path / "big.epub"
    fake_epub.write_bytes(b"PK\x03\x04big-epub")
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(api_key="k", model="gpt-5-mini", staged_analysis=True, max_attempts=1),
        ),
        llm_client=_PipelineFakeLLM(),
        content_plan_profile="content-plan-v3",
    )
    try:
        doc = _make_doc("book-many", 25)
        book = BookRecord(
            id=doc.id,
            metadata=doc.metadata,
            source_language="en",
            target_language="ru",
            status=BookStatus.GLOSSARY_REVIEW,
            source_epub_path=str(fake_epub),
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
        )
        with app.unit_of_work.transaction() as tx:
            tx.books.add(book)
            tx.documents.save(doc)
            app.book_conditions.create(tx, book, app.runtime_settings.revision)

        engine = app.unit_of_work._session_factory.kw["bind"]  # type: ignore[attr-defined]
        sql_statements: list[str] = []

        def _before_cursor_execute(
            _conn: Any, _cursor: Any, statement: str, *_rest: Any
        ) -> None:
            sql_statements.append(statement)

        event.listen(engine, "before_cursor_execute", _before_cursor_execute)
        try:
            view = app.illustrations.get(doc.id, offset=10, limit=5)
        finally:
            event.remove(engine, "before_cursor_execute", _before_cursor_execute)

        assert len(view.chapter_headers) == 5
        assert view.pagination.total == 25
        assert view.pagination.offset == 10
        assert view.pagination.next_offset == 15
        assert view.pagination.previous_offset == 5
        assert view.selected_chapter is None
        # Bounded SQL count — NOT proportional to 25 chapters (no per-chapter N+1)
        assert len(sql_statements) < 25
    finally:
        asyncio.run(app.aclose())


def test_cancellation_and_restart_retains_accepted_parts(tmp_path: Path) -> None:
    fake_epub = tmp_path / "partial.epub"
    fake_epub.write_bytes(b"PK\x03\x04partial-epub")
    # Fail chapter 2 ONCE on CANDIDATES, while reference & chapter 1 succeed!
    fake_llm = _PipelineFakeLLM(fail_section_once="book-partial-sec-01")
    app = build_application(
        settings=AppSettings(
            storage=StorageSettings(data_root=str(tmp_path / "data")),
            llm=LLMSettings(api_key="k", model="gpt-5-mini", staged_analysis=True, max_attempts=1),
        ),
        llm_client=fake_llm,
        content_plan_profile="content-plan-v3",
    )
    try:
        doc = _make_doc("book-partial", 2)
        book = BookRecord(
            id=doc.id,
            metadata=doc.metadata,
            source_language="en",
            target_language="ru",
            status=BookStatus.IMPORTED,
            source_epub_path=str(fake_epub),
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
        )
        with app.unit_of_work.transaction() as tx:
            tx.books.add(book)
            tx.documents.save(doc)
            app.book_conditions.create(tx, book, app.runtime_settings.revision)

        asyncio.run(app.analysis.start(doc.id))
        asyncio.run(drain_tasks(app))

        app.illustrations.save_settings(
            doc.id,
            expected_revision=0,
            settings=IllustrationSettings(
                enabled=True,
                max_per_chapter=2,
                model_reference="gpt-5-mini",
                model_prompts="gpt-5-mini",
            ),
        )
        # First attempt: reference + chapter 0 accepted, chapter 1 fails with transient error (after 3 activation calls)
        # To fail immediately on 1st attempt without succeeding on 2nd attempt in same activation, make fail_section_once fail 3 times!
        fake_llm._failed_once = False
        fail_remaining = [3]
        orig_generate = fake_llm.generate

        async def _flaky_generate(request: LLMRequest[Any]) -> LLMResponse[Any]:
            if request.kind is LLMResponseKind.ILLUSTRATION_CANDIDATES:
                data = _extract_json(request.input_text)
                if data.get("section_id") == "book-partial-sec-01" and fail_remaining[0] > 0:
                    fail_remaining[0] -= 1
                    raise LLMTransportError("simulated ch2 outage")
            return await orig_generate(request)

        fake_llm.generate = _flaky_generate  # type: ignore[method-assign]
        fake_llm.fail_section_once = None

        view_start = asyncio.run(app.illustrations.prepare(doc.id, expected_revision=1))
        gen_id = view_start.active_generation_id
        assert gen_id is not None
        asyncio.run(drain_tasks(app))

        with app.unit_of_work.read_transaction() as tx:
            scopes_mid = tx.illustrations.list_scopes(gen_id)
            ch0_scope = next(
                s
                for s in scopes_mid
                if s.stage is IllustrationScopeStage.CHAPTER and s.section_id == "book-partial-sec-00"
            )
            assert ch0_scope.state is IllustrationScopeState.ACCEPTED
            failed_ch1 = [
                s
                for s in scopes_mid
                if s.section_id == "book-partial-sec-01" and s.state is IllustrationScopeState.FAILED
            ]
            assert failed_ch1

        # Clear calls list, then retry failed chapter 1 -> MUST NOT re-run reference or chapter 0!
        fake_llm.calls.clear()
        asyncio.run(
            app.illustrations.retry(
                doc.id,
                expected_generation_id=gen_id,
                expected_revision=1,
            )
        )
        asyncio.run(drain_tasks(app))

        # Reference & chapter 0 were retained without re-calling LLM!
        assert not any(
            kind in {LLMResponseKind.ILLUSTRATION_EXTRACT, LLMResponseKind.ILLUSTRATION_DESIGN}
            for kind, _ in fake_llm.calls
        )
        with app.unit_of_work.read_transaction() as tx:
            gen_after = tx.illustrations.get_generation(doc.id, gen_id)
            assert gen_after is not None and gen_after.state is IllustrationGenerationState.READY
            ch1_scope = next(
                s
                for s in tx.illustrations.list_scopes(gen_id, stage=IllustrationScopeStage.CHAPTER)
                if s.section_id == "book-partial-sec-01"
            )
            assert ch1_scope.state is IllustrationScopeState.ACCEPTED
    finally:
        asyncio.run(app.aclose())
