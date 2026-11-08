"""Remaining time estimation for book analysis and processing (QA-13)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from linguary.application.book_overview import (
    BookProgress,
    _remaining_time,
    book_overview,
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

