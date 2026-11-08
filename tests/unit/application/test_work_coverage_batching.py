from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from linguary.adapters.db.engine import create_database_engine, create_session_factory
from linguary.adapters.db.models import Base
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.block_processing import (
    TranslationCall,
    complete_processing_in_transaction,
)
from linguary.application.context_builder import ContextBuilder
from linguary.application.llm_dto import (
    BlockResultResponse,
    ChapterAnalysisResponse,
    QuotedContentPlanResponse,
    TranslationResultResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.llm import (
    LLMClient,
    LLMResponse,
)
from linguary.application.ports.persistence import BookRecord
from linguary.application.retry_policy import RetryPolicy
from linguary.application.use_cases.analyze_chapter import AnalyzeChapter
from linguary.application.use_cases.reset_processing import _archive_run_results
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    LLMCallStage,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TaskType,
    TermConcept,
    TermStatus,
    TermVariant,
    TokenUsage,
    TranslationUnit,
    TranslationUnitItem,
    TreatmentPlan,
)
from linguary.domain.book_conditions import BookConditions
from linguary.domain.content_plan import ContentPlan
from linguary.domain.llm_log import CallContext
from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope


NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


def _uid() -> str:
    return str(uuid4())


def test_root_coverage_complete_uses_batched_bindings_and_single_block_results_query(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_database_engine(tmp_path / "batch.sqlite")
    Base.metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork(create_session_factory(engine))

    blocks = tuple(
        DocumentBlock(
            id=f"b{i}",
            section_id="sec-1",
            type=BlockType.PARAGRAPH,
            order=i - 1,
            text=f"Paragraph {i}.",
            source_reference=SourceReference(
                resource_path="ch1.xhtml",
                structural_path=f"/html/body/p[{i}]",
                element_id=f"p{i}",
            ),
            support_status=SupportStatus.SUPPORTED,
        )
        for i in range(1, 5)
    )
    document = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(
            title="Book",
            authors=("Author",),
        ),
        source_language="en",
        target_language="ru",
        sections=(
            DocumentSection(
                id="sec-1",
                document_id="book-1",
                title="Chapter 1",
                order=0,
                source_resource_path="ch1.xhtml",
                blocks=blocks,
            ),
        ),
    )
    cond_id = _uid()
    root_id = _uid()
    scope_id = _uid()
    attempt_id = _uid()
    conditions = BookConditions(
        id=cond_id,
        book_id="book-1",
        revision=1,
        payload={
            "schema_version": 1,
            "models": {"processing": "test-model"},
            "versions": {"validators": 1},
        },
        created_at=NOW,
    )
    run = ProcessingRun(
        id="run-1",
        book_id="book-1",
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=GlossarySnapshot(entries=()),
        llm_settings_snapshot=LLMSettingsSnapshot(provider="test", model="test-model"),
    )
    block_ids = tuple(b.id for b in blocks)
    root = RecoveryRoot(
        id=root_id,
        book_id="book-1",
        conditions_id=cond_id,
        root_key="rk-1",
        manifest={
            "schema_version": 1,
            "active_run_id": "run-1",
            "ordered_blocks": list(block_ids),
            "atomic_intervals": [list(block_ids)],
            "historical_coverage": [],
            "policy": {"max_calls": 6, "max_scope_calls": 2},
        },
        generation=0,
        calls_used=1,
        extra_calls=0,
        status="running",
        policy_version="translation-context-v2",
    )
    scope = RecoveryScope(
        id=scope_id,
        root_id=root_id,
        ordered_blocks=block_ids,
        generation=0,
        status="pending",
    )
    attempt = CallAttempt(
        id=attempt_id,
        book_id="book-1",
        state="validated",
        generation=1,
        ordinal=1,
        conditions_id=cond_id,
        run_id="run-1",
        root_id=root_id,
        scope_id=scope_id,
        context={
            "schema_version": 1,
            "conditions_id": cond_id,
            "stage": "translation_unit",
            "scope_id": scope_id,
            "ordered_block_ids": list(block_ids),
            "attempt_id": attempt_id,
            "generation": 1,
            "assigned_model": "test-model",
            "prompt_hash": "a" * 64,
            "run_id": "run-1",
            "root_id": root_id,
            "policy_version": "translation-context-v2",
            "root_generation": 0,
        },
    )
    results = tuple(
        BlockProcessingResult(
            block_id=b.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text=f"Абзац {b.order + 1}.",
        )
        for b in blocks
    )

    with uow.transaction() as tx:
        tx.books.add(
            BookRecord(
                id="book-1",
                metadata=document.metadata,
                source_language="en",
                target_language="ru",
                status=BookStatus.PROCESSING,
                source_epub_path="book.epub",
                created_at=NOW,
            )
        )
        tx.documents.save(document)
        tx.work.add_conditions(conditions)
        tx.runs.add(run)
        tx.work.bind("run", "run-1", book_id="book-1", conditions_id=cond_id)
        tx.work.add_root(root)
        tx.work.add_scope(scope)
        tx.work.claim_scope(scope_id, expected_generation=0)
        tx.work.add_attempt(attempt)

        work = tx.work
        archive_calls = 0
        original_archive = work.archive_artifacts

        def counting_archive(artifacts):
            nonlocal archive_calls
            archive_calls += 1
            return original_archive(artifacts)

        monkeypatch.setattr(work, "archive_artifacts", counting_archive)
        work.accept_scope(
            scope_id,
            attempt_id=attempt_id,
            expected_generation=1,
            results=results,
        )
        assert archive_calls == 1

    with uow.transaction() as tx:
        work = tx.work
        get_binding_calls = 0
        result_bindings_calls = 0
        get_attempt_calls = 0

        orig_get_binding = work.get_binding
        orig_result_bindings = work.result_bindings
        orig_get_attempt = work.get_attempt

        def spy_get_binding(kind, entity_id):
            nonlocal get_binding_calls
            get_binding_calls += 1
            return orig_get_binding(kind, entity_id)

        def spy_result_bindings(run_id):
            nonlocal result_bindings_calls
            result_bindings_calls += 1
            return orig_result_bindings(run_id)

        def spy_get_attempt(att_id):
            nonlocal get_attempt_calls
            get_attempt_calls += 1
            return orig_get_attempt(att_id)

        monkeypatch.setattr(work, "get_binding", spy_get_binding)
        monkeypatch.setattr(work, "result_bindings", spy_result_bindings)
        monkeypatch.setattr(work, "get_attempt", spy_get_attempt)

        from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository

        monkeypatch.setattr(
            SqlAlchemyBlockResultRepository,
            "get",
            lambda self, run_id, block_id: (_ for _ in ()).throw(
                AssertionError("Поштучный results.get не должен вызываться при наличии SQL-сессии")
            ),
        )

        assert work.root_coverage_complete(root_id) is True
        assert result_bindings_calls == 1
        assert get_binding_calls == 0
        assert get_attempt_calls == 1

    with uow.transaction() as tx:
        from linguary.adapters.db.models import DocumentBlockRow, LLMAttemptRow
        from linguary.domain.correction import CorrectionCheck

        corr_attempt_id = _uid()
        corr_attempt = CallAttempt(
            id=corr_attempt_id,
            book_id="book-1",
            state="accepted",
            generation=1,
            ordinal=1,
            conditions_id=cond_id,
            run_id="run-1",
            root_id=None,
            scope_id=None,
            context={
                "schema_version": 1,
                "conditions_id": cond_id,
                "stage": "chapter_correction",
                "scope_id": "sec-1",
                "ordered_block_ids": list(block_ids),
                "attempt_id": corr_attempt_id,
                "generation": 1,
                "assigned_model": "test-model",
                "prompt_hash": "b" * 64,
                "run_id": "run-1",
            },
        )
        tx.work.add_attempt(corr_attempt)

        work = tx.work
        orig_session_get = work._session.get

        def guarded_get(entity, ident, *args, **kwargs):
            if entity in (DocumentBlockRow, LLMAttemptRow):
                raise AssertionError(
                    f"session.get({entity.__name__}) не должен вызываться в replace_correction_checks"
                )
            return orig_session_get(entity, ident, *args, **kwargs)

        monkeypatch.setattr(work._session, "get", guarded_get)
        checks = tuple(
            CorrectionCheck(
                run_id="run-1",
                block_id=b.id,
                section_id="sec-1",
                input_sha256="c" * 64,
                attempt_id=corr_attempt_id,
                basis="unchanged",
                candidate_id=None,
            )
            for b in blocks
        )
        work.replace_correction_checks("run-1", "sec-1", checks)
        saved_checks = work.correction_checks("run-1", "sec-1")
        assert len(saved_checks) == len(blocks)
        assert all(c.attempt_id == corr_attempt_id for c in saved_checks)

    with uow.transaction() as tx:
        from linguary.adapters.db.work_state import SqlAlchemyWorkRepository

        archive_calls = 0
        orig_archive = SqlAlchemyWorkRepository.archive_artifacts

        def spy_archive(self, artifacts):
            nonlocal archive_calls
            archive_calls += 1
            return orig_archive(self, artifacts)

        monkeypatch.setattr(SqlAlchemyWorkRepository, "archive_artifacts", spy_archive)
        monkeypatch.setattr(
            SqlAlchemyWorkRepository,
            "get_binding",
            lambda self, kind, entity_id: (
                {"conditions_id": cond_id, "attempt_id": None, "payload_json": {"schema_version": 1}}
                if kind == "run"
                else (_ for _ in ()).throw(
                    AssertionError("Поштучный get_binding('result', ...) не должен вызываться в _archive_run_results")
                )
            ),
        )
        _archive_run_results(tx, "book-1", run)
        assert archive_calls == 1

    engine.dispose()


async def test_translation_call_request_reuses_loaded_conditions_and_glossary_without_extra_transactions() -> None:
    block = DocumentBlock(
        id="b1",
        section_id="sec-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text="Winston Smith entered.",
        source_reference=SourceReference(
            resource_path="ch1.xhtml",
            structural_path="/html/body/p[1]",
            element_id="p1",
        ),
        support_status=SupportStatus.SUPPORTED,
    )
    unit = TranslationUnit(
        id="unit-1",
        section_id="sec-1",
        items=(TranslationUnitItem(block=block, treatment=BlockTreatment.TRANSLATE),),
    )
    glossary = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Winston Smith",
                translation="Уинстон Смит",
                type=ConceptType.PERSON,
                status=TermStatus.LOCKED,
            ),
        )
    )
    run = ProcessingRun(
        id="run-1",
        book_id="book-1",
        status=ProcessingRunStatus.ACTIVE,
        glossary_snapshot=glossary,
        llm_settings_snapshot=LLMSettingsSnapshot(provider="test", model="test-model"),
    )
    cond_id = _uid()
    att_id = _uid()
    conditions = BookConditions(
        id=cond_id,
        book_id="book-1",
        revision=1,
        payload={
            "schema_version": 1,
            "reader_level": "b2",
            "models": {"processing": "test-model"},
            "versions": {"validators": 3},
        },
        created_at=NOW,
    )
    call_context = CallContext(
        conditions_id=cond_id,
        stage="translation_unit",
        scope_id="unit-1",
        ordered_block_ids=("b1",),
        attempt_id=att_id,
        generation=1,
        assigned_model="test-model",
        prompt_hash="a" * 64,
        run_id="run-1",
    )

    builder = ContextBuilder(content_plan_profile="content-plan-v3")
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Ch 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=(block,),
    )
    document = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(
            title="Book",
            authors=("Author",),
        ),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )
    analysis = ChapterAnalysis.create(
        id="an-1",
        section_id="sec-1",
        treatment_plan=TreatmentPlan.of_treatments("sec-1", (("b1", BlockTreatment.TRANSLATE),)),
        translation_instructions="Translate.",
        chapter_summary="Summary.",
        expected_block_ids=["b1"],
        glossary_candidates=(),
        before_you_read=(),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(),
            before_you_read_evidence=(),
            context_facts=(),
        ),
    )
    context = builder.for_translation_unit(
        document=document,
        section=section,
        analysis=analysis,
        unit=unit,
        glossary=glossary,
    )
    from dataclasses import replace

    context = replace(
        context,
        request=replace(
            context.request.with_logging(
                book_id="book-1",
                stage=LLMCallStage.TRANSLATION_UNIT,
                task_id="task-1",
                scope_id="unit-1",
                ordered_block_ids=("b1",),
            ),
            call_context=call_context,
        ),
    )

    tx_count = 0
    result_bindings_calls = 0
    get_binding_calls = 0
    active_for_book_calls = 0
    get_conditions_calls = 0

    class _FakeWork:
        def result_bindings(self, run_id: str):
            nonlocal result_bindings_calls
            result_bindings_calls += 1
            return {"b1": {"attempt_id": "prev-att"}}

        def get_binding(self, kind: str, entity_id: str):
            nonlocal get_binding_calls
            get_binding_calls += 1
            return None

        def get_conditions(self, conditions_id: str):
            nonlocal get_conditions_calls
            get_conditions_calls += 1
            return conditions

    class _FakeRuns:
        def active_for_book(self, book_id: str):
            nonlocal active_for_book_calls
            active_for_book_calls += 1
            return run

    class _FakeTx:
        def __init__(self) -> None:
            self.work = _FakeWork()
            self.runs = _FakeRuns()

    class _FakeUoW:
        @contextmanager
        def transaction(self):
            nonlocal tx_count
            tx_count += 1
            yield _FakeTx()

    class _FakeLLM(LLMClient):
        async def generate(self, request):
            return LLMResponse(
                payload=TranslationResultResponse(
                    blocks=(
                        BlockResultResponse(
                            source_block_id="b1",
                            treatment=BlockTreatment.TRANSLATE,
                            translated_text="Уинстон Смит вошёл.",
                        ),
                    )
                ),
                provider="test",
                model="test-model",
                truncated=False,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=_FakeUoW(),  # type: ignore[arg-type]
        llm_client=_FakeLLM(),
        retry_policy=RetryPolicy(max_attempts=2),
        validator=TranslationResultValidator(
            target_language_min_share=0.5,
            language_check_min_length=20,
            repetition_ngram_size=3,
            repetition_min_repeats=4,
        ),
        sleep=lambda _: None,  # type: ignore[arg-type]
    )

    res = await call.request(
        context,
        unit=unit,
        target_language="ru",
        note_ids=lambda: "ln-ch01-001",
        glossary=glossary,
        source_texts={"b1": "Winston Smith entered."},
    )
    assert len(res.results) == 1
    assert result_bindings_calls == 1
    assert get_binding_calls == 0
    assert active_for_book_calls == 1
    assert get_conditions_calls == 1
    assert tx_count == 1


async def test_analyze_chapter_request_reuses_loaded_glossary_concepts() -> None:
    block = DocumentBlock(
        id="b1",
        section_id="sec-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text="Winston Smith entered.",
        source_reference=SourceReference(
            resource_path="ch1.xhtml",
            structural_path="/html/body/p[1]",
            element_id="p1",
        ),
        support_status=SupportStatus.SUPPORTED,
    )
    concept = TermConcept(
        id="concept-1",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Winston Smith",
        status=TermStatus.LOCKED,
        variants=(
            TermVariant(
                concept_id="concept-1",
                language="ru",
                text="Уинстон Смит",
                status=TermStatus.LOCKED,
            ),
        ),
    )

    glossary_calls = 0

    class _FakeGlossary:
        def list_for_book(self, book_id: str):
            nonlocal glossary_calls
            glossary_calls += 1
            return (concept,)

    class _FakeTx:
        def __init__(self) -> None:
            self.glossary = _FakeGlossary()

    class _FakeUoW:
        @contextmanager
        def transaction(self):
            yield _FakeTx()

    class _FakeLLM(LLMClient):
        async def generate(self, request):
            return LLMResponse(
                payload=ChapterAnalysisResponse(
                    treatment_plan=(
                        TreatmentPlanItemResponse(
                            source_block_id="b1",
                            treatment=BlockTreatment.TRANSLATE,
                        ),
                    ),
                    translation_instructions="Translate carefully.",
                    chapter_summary="Short summary.",
                    glossary_candidates=(),
                    before_you_read=(),
                    content_plan=QuotedContentPlanResponse(
                        difficulties=(),
                        before_you_read_evidence=(),
                        context_facts=(),
                    ),
                ),
                provider="test",
                model="test-model",
                truncated=False,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            )

    builder = ContextBuilder(content_plan_profile="content-plan-v3")
    use_case = AnalyzeChapter(
        unit_of_work=_FakeUoW(),  # type: ignore[arg-type]
        llm_client=_FakeLLM(),
        language_detector=MagicMock(),
        retry_policy=RetryPolicy(max_attempts=1),
        glossary_confidence_threshold=0.8,
        context_builder=builder,
    )
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Ch 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=(block,),
    )
    document = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(
            title="Book",
            authors=("Author",),
        ),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )
    from dataclasses import replace

    context = builder.for_chapter_analysis(
        document=document,
        section=section,
        accumulated_candidates=(),
    )
    context = replace(
        context,
        request=context.request.with_logging(
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_ANALYSIS,
            task_id="task-1",
            scope_id="sec-1",
            ordered_block_ids=("b1",),
        ),
    )

    payload, attempts, _ = await use_case._request_analysis(
        context,
        blocks=(block,),
        source_language="en",
        concepts=(concept,),
    )
    assert attempts == 1
    assert payload.chapter_summary == "Short summary."
    assert glossary_calls == 0


def test_complete_processing_passes_section_id_to_reconcile_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconcile_calls: list[tuple[str, str | None]] = []

    import linguary.application.correction as correction_mod

    monkeypatch.setattr(
        correction_mod,
        "reconcile_corrections",
        lambda tx, book_id, *, section_id=None: reconcile_calls.append((book_id, section_id)),
    )
    monkeypatch.setattr(
        correction_mod,
        "correction_coverage_complete",
        lambda tx, book_id: False,
    )

    task = ProcessingTask(
        id="task-1",
        book_id="book-1",
        task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        scope_id="sec-1",
        status=TaskStatus.RUNNING,
        attempts=1,
    )

    book = BookRecord(
        id="book-1",
        metadata=BookMetadata(
            title="Book",
            authors=("Author",),
        ),
        source_language="en",
        target_language="ru",
        status=BookStatus.PROCESSING,
        source_epub_path="book.epub",
        created_at=NOW,
    )

    class _FakeTasks:
        def get(self, task_id: str):
            return task if task_id == "task-1" else None

        def save(self, updated):
            pass

        def list_for_book(self, book_id: str):
            return ()

    class _FakeBooks:
        def get(self, book_id: str):
            return book

    class _FakeRuns:
        def active_for_book(self, book_id: str):
            return None

    class _FakeTx:
        tasks = _FakeTasks()
        books = _FakeBooks()
        runs = _FakeRuns()

    complete_processing_in_transaction(
        _FakeTx(),
        book_id="book-1",
        finished_task_id="task-1",
        task_types=(TaskType.PROCESS_TRANSLATION_UNIT,),
    )
    assert reconcile_calls == [("book-1", "sec-1")]
