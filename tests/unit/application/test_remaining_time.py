"""Remaining time estimation for book analysis and processing (QA-13)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from linguary.application.book_overview import (
    BookProgress,
    RunningChapter,
    _remaining_time,
    book_overview,
    book_result,
    estimate_remaining_seconds,
)
from linguary.application.ports.persistence import BookRecord
from linguary.domain import (
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    LLMCallLog,
    LLMCallStage,
    ProcessingTask,
    SourceReference,
    SupportStatus,
    TaskStatus,
    TaskType,
)
from linguary.web.labels import remaining_time_label


def test_no_calls_yet_gives_no_estimate() -> None:
    """Before the first model call there is nothing to measure: no estimate rather than zero."""
    assert (
        estimate_remaining_seconds(
            durations_ms=(), completed_blocks=0, remaining_blocks=100
        )
        is None
    )


def test_estimate_is_median_call_times_remaining_calls() -> None:
    """Median call duration * remaining calls: one call per 10-block chapter."""
    seconds = estimate_remaining_seconds(
        durations_ms=(40_000, 50_000, 60_000),
        completed_blocks=30,
        remaining_blocks=100,
    )

    assert seconds == 500


def test_one_slow_retry_does_not_stretch_the_estimate() -> None:
    """Median rather than mean: a single delayed retry does not inflate the estimate."""
    seconds = estimate_remaining_seconds(
        durations_ms=(50_000, 50_000, 900_000),
        completed_blocks=30,
        remaining_blocks=30,
    )

    assert seconds == 150


def test_parallelism_divides_the_estimate() -> None:
    """Multiple chapters in parallel proportionally reduce wait time."""
    seconds = estimate_remaining_seconds(
        durations_ms=(40_000,),
        completed_blocks=10,
        remaining_blocks=40,
        parallelism=4,
    )

    assert seconds == 40


def test_rate_limit_bounds_the_gain_of_parallelism() -> None:
    """Rate limit bounds the parallelism speedup."""
    seconds = estimate_remaining_seconds(
        durations_ms=(10_000,),
        completed_blocks=10,
        remaining_blocks=40,
        parallelism=8,
        requests_per_minute=6,
    )

    assert seconds == 40


def test_finished_stage_needs_no_more_time() -> None:
    """When no work remains, there is nothing left to wait for."""
    assert (
        estimate_remaining_seconds(
            durations_ms=(40_000,), completed_blocks=10, remaining_blocks=0
        )
        == 0
    )


@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        pytest.param(None, "Оценка оставшегося времени появится после первой главы", id="unknown"),
        pytest.param(20, "Осталось меньше минуты", id="under-a-minute"),
        pytest.param(61, "Осталось примерно 2 мин", id="rounded-up"),
        pytest.param(3600, "Осталось примерно 1 ч", id="whole-hour"),
        pytest.param(5000, "Осталось примерно 1 ч 24 мин", id="hours-and-minutes"),
    ],
)
def test_remaining_time_is_named_roughly(seconds: int | None, label: str) -> None:
    """The estimate is rough, so it is phrased as approximate and rounded up to minutes."""
    assert remaining_time_label(seconds) == label


def _sample_document(*, chapters: int = 3) -> CanonicalDocument:
    """Document with `chapters` sections of one textual block each."""
    sections = tuple(
        DocumentSection(
            id=f"sec-{idx}",
            document_id="book-1",
            title=f"Chapter {idx}",
            order=idx - 1,
            source_resource_path=f"OEBPS/ch{idx}.xhtml",
            blocks=(
                DocumentBlock(
                    id=f"blk-{idx}",
                    section_id=f"sec-{idx}",
                    type=BlockType.PARAGRAPH,
                    order=0,
                    text=f"Paragraph {idx}.",
                    source_reference=SourceReference(
                        resource_path=f"OEBPS/ch{idx}.xhtml",
                        structural_path="/html/body/p[1]",
                    ),
                    support_status=SupportStatus.SUPPORTED,
                ),
            ),
        )
        for idx in range(1, chapters + 1)
    )
    return CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Sample", authors=("Author",)),
        source_language="en",
        target_language="ru",
        sections=sections,
    )


def test_remaining_time_uses_worker_concurrency_for_chapter_analysis() -> None:
    """Chapter analysis remaining time divides by `min(concurrency, active)`."""
    document = _sample_document(chapters=3)
    tasks = (
        ProcessingTask(
            id="t-1",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-1",
            status=TaskStatus.SUCCEEDED,
        ),
        ProcessingTask(
            id="t-2",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-2",
            status=TaskStatus.RUNNING,
        ),
        ProcessingTask(
            id="t-3",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-3",
            status=TaskStatus.PENDING,
        ),
    )
    calls = (
        LLMCallLog(
            id="call-1",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_ANALYSIS,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
            duration_ms=40_000,
            attempt=1,
            success=True,
            task_id="t-1",
            cost=Decimal("0.01"),
        ),
    )
    progress = BookProgress(
        total_sections=3,
        analyzed_sections=1,
        planned_blocks=1,
        processed_blocks=0,
    )

    estimate = _remaining_time(
        status=BookStatus.ANALYZING,
        progress=progress,
        document=document,
        tasks=tasks,
        calls=calls,
        concurrency=4,
        requests_per_minute=None,
    )

    assert estimate is not None
    assert estimate.stage is LLMCallStage.CHAPTER_ANALYSIS
    assert estimate.seconds == 40


def test_book_overview_queries_llm_calls_once_per_transaction() -> None:
    """`book_overview` fetches `llm_calls.list_for_book` once and shares it with `_quality_projection`."""
    document = _sample_document(chapters=3)
    book = BookRecord(
        id="book-1",
        metadata=document.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.ANALYZING,
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
    )
    tasks = (
        ProcessingTask(
            id="t-1",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-1",
            status=TaskStatus.SUCCEEDED,
        ),
        ProcessingTask(
            id="t-2",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-2",
            status=TaskStatus.RUNNING,
        ),
    )
    calls = (
        LLMCallLog(
            id="call-1",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_ANALYSIS,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
            duration_ms=30_000,
            attempt=1,
            success=True,
            task_id="t-1",
            cost=Decimal("0.05"),
        ),
    )
    llm_call_queries: list[str] = []

    def list_for_book(book_id: str):
        llm_call_queries.append(book_id)
        return calls

    tx = SimpleNamespace(
        books=SimpleNamespace(get=lambda book_id: book),
        documents=SimpleNamespace(
            get=lambda book_id: document,
            import_warnings=lambda book_id: (),
        ),
        analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
        tasks=SimpleNamespace(list_for_book=lambda book_id: tasks),
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        results=SimpleNamespace(
            count_for_run=lambda run_id: 0,
            list_for_run=lambda run_id: (),
        ),
        llm_calls=SimpleNamespace(list_for_book=list_for_book),
        work=SimpleNamespace(
            latest_conditions=lambda book_id: None,
            list_roots=lambda book_id, run_id=None: (),
            list_reviews=lambda book_id: (),
        ),
    )

    class _StubUoW:
        @contextmanager
        def transaction(self):
            yield tx

    overview = book_overview(_StubUoW(), "book-1", concurrency=2)  # type: ignore[arg-type]

    assert llm_call_queries == ["book-1"]
    assert overview.ordinary_cost["known_spent_usd"] == "0.05"


def test_book_overview_marks_chapter_active_and_estimates_remaining_time_during_chapter_correction() -> None:
    """When translation units are done and chapter correction is running, overview shows active chapter and estimate."""
    document = _sample_document(chapters=3)
    book = BookRecord(
        id="book-1",
        metadata=document.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.PROCESSING,
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
    )
    tasks = (
        ProcessingTask(
            id="tu-1",
            book_id="book-1",
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id="sec-1",
            status=TaskStatus.SUCCEEDED,
        ),
        ProcessingTask(
            id="tc-1",
            book_id="book-1",
            task_type=TaskType.CHAPTER_CORRECTION,
            scope_id="sec-1",
            status=TaskStatus.SUCCEEDED,
        ),
        ProcessingTask(
            id="tc-2",
            book_id="book-1",
            task_type=TaskType.CHAPTER_CORRECTION,
            scope_id="sec-2",
            status=TaskStatus.RUNNING,
        ),
        ProcessingTask(
            id="tc-3",
            book_id="book-1",
            task_type=TaskType.CHAPTER_CORRECTION,
            scope_id="sec-3",
            status=TaskStatus.PENDING,
        ),
    )
    calls = (
        LLMCallLog(
            id="call-c1",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_CORRECTION,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
            duration_ms=20_000,
            attempt=1,
            success=True,
            task_id="tc-1",
            cost=Decimal("0.02"),
        ),
        LLMCallLog(
            id="call-c2",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_CORRECTION_REVIEW,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 1, tzinfo=UTC),
            duration_ms=20_000,
            attempt=1,
            success=True,
            task_id="tc-1",
            cost=Decimal("0.01"),
        ),
    )
    tx = SimpleNamespace(
        books=SimpleNamespace(get=lambda book_id: book),
        documents=SimpleNamespace(
            get=lambda book_id: document,
            import_warnings=lambda book_id: (),
        ),
        analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
        tasks=SimpleNamespace(list_for_book=lambda book_id: tasks),
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        results=SimpleNamespace(
            count_for_run=lambda run_id: 3,
            list_for_run=lambda run_id: (),
        ),
        llm_calls=SimpleNamespace(list_for_book=lambda book_id: calls),
        work=SimpleNamespace(
            latest_conditions=lambda book_id: None,
            list_roots=lambda book_id, run_id=None: (),
            list_reviews=lambda book_id: (),
        ),
    )

    class _StubUoW:
        @contextmanager
        def transaction(self):
            yield tx

    overview = book_overview(_StubUoW(), "book-1", concurrency=2)  # type: ignore[arg-type]

    assert overview.progress.running_chapters == (
        RunningChapter(chapter_number=2, title="Chapter 2"),
    )
    assert overview.progress.remaining_time is not None
    assert overview.progress.remaining_time.stage is LLMCallStage.CHAPTER_CORRECTION
    assert overview.progress.remaining_time.seconds == 40

    fallback = _remaining_time(
        status=BookStatus.PROCESSING,
        progress=overview.progress,
        document=None,
        tasks=tasks,
        calls=calls,
        concurrency=2,
        requests_per_minute=None,
    )
    assert fallback is not None
    assert fallback.stage is LLMCallStage.CHAPTER_CORRECTION
    assert fallback.seconds == 40


def test_book_result_without_quality_skips_quality_projection_queries() -> None:
    """`book_result(..., include_quality=False)` and `_result_block_responses` skip quality queries."""
    from linguary.web.routes.result import _result_block_responses

    document = _sample_document(chapters=1)
    book = BookRecord(
        id="book-1",
        metadata=document.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.PROCESSING,
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("quality projection query should not run when include_quality=False")

    tx = SimpleNamespace(
        books=SimpleNamespace(get=lambda book_id: book),
        documents=SimpleNamespace(
            get=lambda book_id: document,
            import_warnings=_unexpected,
        ),
        analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
        tasks=SimpleNamespace(list_for_book=lambda book_id: ()),
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        results=SimpleNamespace(list_for_run=lambda run_id: ()),
        llm_calls=SimpleNamespace(list_for_book=_unexpected),
        work=SimpleNamespace(
            latest_conditions=_unexpected,
            list_roots=_unexpected,
            list_reviews=_unexpected,
            get_binding=lambda kind, entity_id: None,
            result_bindings=lambda run_id: {},
        ),
    )

    class _StubUoW:
        @contextmanager
        def transaction(self):
            yield tx

    uow = _StubUoW()
    summary = book_result(uow, "book-1", include_quality=False)  # type: ignore[arg-type]
    assert len(summary.chapters) == 1
    assert summary.warnings.total == 0

    app = SimpleNamespace(unit_of_work=uow)
    groups = _result_block_responses(app, "book-1", ("blk-1",))  # type: ignore[arg-type]
    assert len(groups) == 1
    assert groups[0].block_id == "blk-1"


def test_book_overview_uses_read_transaction_and_skips_full_results_while_processing() -> None:
    """`book_overview` uses `read_transaction` and `block_ids_for_run` without loading full results while processing."""
    document = _sample_document(chapters=1)
    book = BookRecord(
        id="book-1",
        metadata=document.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.PROCESSING,
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
    )
    tasks = (
        ProcessingTask(
            id="t-1",
            book_id="book-1",
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id="sec-1",
            status=TaskStatus.RUNNING,
        ),
    )
    run = SimpleNamespace(id="run-1", book_id="book-1", status="active")
    block_ids_queries: list[str] = []
    read_tx_entered = 0

    def _block_ids_for_run(run_id: str) -> frozenset[str]:
        block_ids_queries.append(run_id)
        return frozenset({"blk-1"})

    def _unexpected_list_for_run(run_id: str):
        raise AssertionError("list_for_run should not be called while book is still processing")

    tx = SimpleNamespace(
        books=SimpleNamespace(get=lambda book_id: book),
        documents=SimpleNamespace(
            get=lambda book_id: document,
            import_warnings=lambda book_id: (),
        ),
        analyses=SimpleNamespace(list_for_book=lambda book_id: ()),
        tasks=SimpleNamespace(list_for_book=lambda book_id: tasks),
        runs=SimpleNamespace(active_for_book=lambda book_id: run),
        results=SimpleNamespace(
            count_for_run=lambda run_id: 1,
            block_ids_for_run=_block_ids_for_run,
            list_for_run=_unexpected_list_for_run,
        ),
        llm_calls=SimpleNamespace(list_for_book=lambda book_id: ()),
        work=SimpleNamespace(
            latest_conditions=lambda book_id: None,
            list_roots=lambda book_id, run_id=None: (),
            list_reviews=lambda book_id: (),
        ),
    )

    class _ReadOnlyStubUoW:
        @contextmanager
        def transaction(self):
            raise AssertionError("write transaction() should not be used for read-only book_overview")
            yield tx

        @contextmanager
        def read_transaction(self):
            nonlocal read_tx_entered
            read_tx_entered += 1
            yield tx

    overview = book_overview(_ReadOnlyStubUoW(), "book-1", concurrency=1)  # type: ignore[arg-type]

    assert read_tx_entered == 1
    assert block_ids_queries == ["run-1"]
    assert overview.epub_ready is False
    assert overview.recovery["covered"] == 1

    from linguary.adapters.db.repositories import SqlAlchemyBlockResultRepository
    from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork

    driver_sql_calls: list[str] = []

    class _FakeSession:
        @contextmanager
        def begin(self):
            yield

        def connection(self):
            return SimpleNamespace(exec_driver_sql=driver_sql_calls.append)

        def scalars(self, _stmt):
            return SimpleNamespace(all=lambda: ["blk-1", "blk-2"])

        def close(self):
            pass

    real_uow = SqlAlchemyUnitOfWork(_FakeSession)  # type: ignore[arg-type]
    with real_uow.read_transaction() as read_tx:
        assert isinstance(read_tx.results, SqlAlchemyBlockResultRepository)
        assert read_tx.results.block_ids_for_run("run-1") == frozenset({"blk-1", "blk-2"})
    assert driver_sql_calls == []


def test_quality_projection_batches_root_scopes_and_attempts_queries() -> None:
    """`_quality_projection` batches scopes, attempts, and result bindings across all recovery roots."""
    from linguary.adapters.db.work_state import SqlAlchemyWorkRepository
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    root_1_id = "00000000-0000-0000-0000-000000000001"
    root_2_id = "00000000-0000-0000-0000-000000000002"
    scope_1_id = "00000000-0000-0000-0000-000000000011"
    scope_2_id = "00000000-0000-0000-0000-000000000012"
    att_1_id = "00000000-0000-0000-0000-000000000021"
    att_2_id = "00000000-0000-0000-0000-000000000022"
    cond_id = "00000000-0000-0000-0000-000000000031"
    run_id = "00000000-0000-0000-0000-000000000041"
    book_id = "00000000-0000-0000-0000-000000000051"

    document = _sample_document(chapters=2)
    book = BookRecord(
        id=book_id,
        metadata=document.metadata,
        source_language="en",
        target_language="ru",
        status=BookStatus.PROCESSING,
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
    )
    conditions = SimpleNamespace(
        id=cond_id,
        book_id=book_id,
        revision=1,
        payload={
            "reader_level": "B2",
            "english_help": False,
            "profile": "standard",
            "models": {},
            "versions": {"validators": 1},
        },
        pronunciation_help_enabled=False,
        pronunciation_variant=None,
        reproducibility="strict",
    )
    run = SimpleNamespace(id=run_id, book_id=book_id, status="active")
    roots = (
        RecoveryRoot(
            id=root_1_id,
            book_id=book_id,
            conditions_id=cond_id,
            root_key="rk-1",
            manifest={
                "schema_version": 1,
                "active_run_id": run_id,
                "ordered_blocks": ["blk-1"],
                "historical_coverage": [],
                "policy": {"max_calls": 4, "max_scope_calls": 2},
            },
            generation=1,
            calls_used=1,
            extra_calls=0,
            status="running",
            reason=None,
            policy_version="translation_recovery_v1",
        ),
        RecoveryRoot(
            id=root_2_id,
            book_id=book_id,
            conditions_id=cond_id,
            root_key="rk-2",
            manifest={
                "schema_version": 1,
                "active_run_id": run_id,
                "ordered_blocks": ["blk-2"],
                "historical_coverage": [],
                "policy": {"max_calls": 4, "max_scope_calls": 2},
            },
            generation=1,
            calls_used=1,
            extra_calls=0,
            status="running",
            reason=None,
            policy_version="translation_recovery_v1",
        ),
    )
    scopes_by_root = {
        root_1_id: (
            RecoveryScope(
                id=scope_1_id,
                root_id=root_1_id,
                ordered_blocks=("blk-1",),
                generation=1,
                status="accepted",
                accepted_attempt_id=att_1_id,
            ),
        ),
        root_2_id: (
            RecoveryScope(
                id=scope_2_id,
                root_id=root_2_id,
                ordered_blocks=("blk-2",),
                generation=1,
                status="accepted",
                accepted_attempt_id=att_2_id,
            ),
        ),
    }
    attempts_by_root = {
        root_1_id: (
            CallAttempt(
                id=att_1_id,
                book_id=book_id,
                state="accepted",
                generation=1,
                ordinal=1,
                conditions_id=cond_id,
                run_id=run_id,
                root_id=root_1_id,
                scope_id=scope_1_id,
                context={"schema_version": 1, "stage": "translation_unit", "policy_version": "translation_recovery_v1"},
            ),
        ),
        root_2_id: (
            CallAttempt(
                id=att_2_id,
                book_id=book_id,
                state="accepted",
                generation=1,
                ordinal=1,
                conditions_id=cond_id,
                run_id=run_id,
                root_id=root_2_id,
                scope_id=scope_2_id,
                context={"schema_version": 1, "stage": "translation_unit", "policy_version": "translation_recovery_v1"},
            ),
        ),
    }

    batch_scopes_calls: list[tuple[str, ...]] = []
    batch_attempts_calls: list[tuple[str, ...]] = []
    bindings_calls: list[str] = []

    work_repo = SqlAlchemyWorkRepository(session=SimpleNamespace())

    def _list_scopes_for_roots(root_ids):
        batch_scopes_calls.append(tuple(root_ids))
        return scopes_by_root

    def _list_attempts_for_roots(root_ids):
        batch_attempts_calls.append(tuple(root_ids))
        return attempts_by_root

    def _result_bindings(r_id: str):
        bindings_calls.append(r_id)
        return {
            "blk-1": {
                "book_id": book_id,
                "conditions_id": cond_id,
                "attempt_id": att_1_id,
                "artifact_revision_id": "art-1",
            },
            "blk-2": {
                "book_id": book_id,
                "conditions_id": cond_id,
                "attempt_id": att_2_id,
                "artifact_revision_id": "art-2",
            },
        }

    def _unexpected(*args, **kwargs):
        raise AssertionError("per-root query should not be called when batch data is preloaded")

    work = SimpleNamespace(
        latest_conditions=lambda b_id: conditions,
        list_roots=lambda b_id, run_id=None: roots,
        list_scopes_for_roots=_list_scopes_for_roots,
        list_attempts_for_roots=_list_attempts_for_roots,
        result_bindings=_result_bindings,
        list_scopes=_unexpected,
        list_root_attempts=_unexpected,
        get_root=_unexpected,
        get_attempt=_unexpected,
        root_coverage_complete=work_repo.root_coverage_complete,
        list_reviews=lambda b_id: (),
    )
    tx = SimpleNamespace(
        books=SimpleNamespace(get=lambda b_id: book),
        documents=SimpleNamespace(
            get=lambda b_id: document,
            import_warnings=lambda b_id: (),
        ),
        analyses=SimpleNamespace(list_for_book=lambda b_id: ()),
        tasks=SimpleNamespace(
            list_for_book=lambda b_id: (
                ProcessingTask(
                    id="t-1",
                    book_id=book_id,
                    task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                    scope_id="sec-1",
                    status=TaskStatus.RUNNING,
                ),
            )
        ),
        runs=SimpleNamespace(active_for_book=lambda b_id: run),
        results=SimpleNamespace(
            count_for_run=lambda r_id: 2,
            block_ids_for_run=lambda r_id: frozenset({"blk-1", "blk-2"}),
            list_for_run=_unexpected,
        ),
        llm_calls=SimpleNamespace(list_for_book=lambda b_id: ()),
        work=work,
    )

    class _ReadOnlyStubUoW:
        @contextmanager
        def read_transaction(self):
            yield tx

        @contextmanager
        def transaction(self):
            yield tx

    overview = book_overview(_ReadOnlyStubUoW(), book_id, concurrency=1)  # type: ignore[arg-type]

    assert batch_scopes_calls == [(root_1_id, root_2_id)]
    assert batch_attempts_calls == [(root_1_id, root_2_id)]
    assert bindings_calls == [run_id]
    assert [item["complete"] for item in overview.recovery["roots"]] == [True, True]
    assert overview.recovery["covered"] == 2

    regen_att_id = "00000000-0000-0000-0000-000000000099"
    fetched_missing: list[str] = []

    def _get_attempt_fallback(attempt_id: str):
        fetched_missing.append(attempt_id)
        return CallAttempt(
            id=attempt_id,
            book_id=book_id,
            state="accepted",
            generation=1,
            ordinal=2,
            conditions_id=cond_id,
            run_id=run_id,
            context={"schema_version": 1, "stage": "block_regeneration"},
        )

    work_repo.get_attempt = _get_attempt_fallback  # type: ignore[method-assign]
    assert (
        work_repo.root_coverage_complete(
            root_1_id,
            update_state=False,
            root=roots[0],
            scopes=scopes_by_root[root_1_id],
            bindings={
                "blk-1": {
                    "book_id": book_id,
                    "conditions_id": cond_id,
                    "attempt_id": regen_att_id,
                    "artifact_revision_id": "art-regen",
                }
            },
            attempts_by_id={att_1_id: attempts_by_root[root_1_id][0]},
            present_blocks={"blk-1"},
            conditions=conditions,
            run=run,
        )
        is True
    )
    assert fetched_missing == [regen_att_id]


def test_remaining_time_and_stage_estimate_account_for_staged_analysis() -> None:
    from linguary.application.book_overview import _stage_estimate

    document = _sample_document(chapters=3)
    tasks = (
        ProcessingTask(
            id="t-1",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-1",
            status=TaskStatus.SUCCEEDED,
        ),
        ProcessingTask(
            id="t-2",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-2",
            status=TaskStatus.RUNNING,
        ),
        ProcessingTask(
            id="t-3",
            book_id="book-1",
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="sec-3",
            status=TaskStatus.PENDING,
        ),
    )
    calls = (
        LLMCallLog(
            id="call-1-nar",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 0, tzinfo=UTC),
            duration_ms=20_000,
            attempt=1,
            success=True,
            task_id="t-1",
        ),
        LLMCallLog(
            id="call-1-ped",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 1, tzinfo=UTC),
            duration_ms=30_000,
            attempt=1,
            success=True,
            task_id="t-1",
        ),
        # In-flight narrative pass of running task t-2 must not inflate completed chapter durations.
        LLMCallLog(
            id="call-2-nar",
            book_id="book-1",
            stage=LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
            provider="openai",
            model="gpt-test",
            started_at=datetime(2026, 11, 8, 12, 2, tzinfo=UTC),
            duration_ms=20_000,
            attempt=1,
            success=True,
            task_id="t-2",
        ),
    )
    progress = BookProgress(
        total_sections=3,
        analyzed_sections=1,
        planned_blocks=1,
        processed_blocks=0,
    )
    estimate = _remaining_time(
        status=BookStatus.ANALYZING,
        progress=progress,
        document=document,
        tasks=tasks,
        calls=calls,
        concurrency=1,
        requests_per_minute=None,
    )
    assert estimate is not None
    assert estimate.seconds == 100

    single_est = _stage_estimate(
        status=BookStatus.IMPORTED,
        document=document,
        analyses=(),
        tasks=(),
        done_block_ids=set(),
        input_price_per_mtok=None,
        output_price_per_mtok=None,
        staged_analysis=False,
    )
    staged_est = _stage_estimate(
        status=BookStatus.IMPORTED,
        document=document,
        analyses=(),
        tasks=(),
        done_block_ids=set(),
        input_price_per_mtok=None,
        output_price_per_mtok=None,
        staged_analysis=True,
    )
    assert staged_est["input_tokens"] == 2 * single_est["input_tokens"]




