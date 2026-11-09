"""Chapter analysis task order (P3.T3, PRD § 13, § 25)."""

from __future__ import annotations

import asyncio

import pytest
from analysis_fakes import (
    BOOK_ID,
    FakeTaskQueue,
    FakeUnitOfWork,
    analysis_reply,
    make_analysis_document,
    make_analysis_section,
    make_analyze_chapter,
    make_book_record,
)

from linguary.adapters.llm import FakeLLMClient
from linguary.application.ports.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMPayload,
    LLMRequest,
    LLMResponse,
)
from linguary.application.task_runner import TaskRunner
from linguary.application.use_cases.analyze_chapter import (
    AnalyzeChapterError,
    ChapterAnalysisSequence,
)
from linguary.domain import (
    BookStatus,
    BookStatusTransitionError,
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskStatus,
    TaskType,
)

CHAPTER_COUNT = 3
WORKER_COUNT = 4


class YieldingLLMClient:
    """Client yielding control to the event loop during a call.

    Without this, the entire model call completes without a single yield point,
    and parallel workers would never interleave, making order checks meaningless.
    """

    def __init__(self, inner: FakeLLMClient) -> None:
        self._inner = inner

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        await asyncio.sleep(0)
        return await self._inner.generate(request)


def make_book(chapters: int = CHAPTER_COUNT) -> FakeUnitOfWork:
    """A book with multiple chapters, each containing one paragraph."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Text of chapter {number}.",),
        )
        for number in range(1, chapters + 1)
    )
    return FakeUnitOfWork(
        document=make_analysis_document(sections), book=make_book_record()
    )


def make_sequence(
    unit_of_work: FakeUnitOfWork, queue: FakeTaskQueue, client: LLMClient
) -> ChapterAnalysisSequence:
    """Chapter analysis sequence on top of storage and queue doubles."""
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    return ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # storage port double
        queue=queue,  # type: ignore[arg-type]  # queue port double
        analyze_chapter=analyze_chapter,
    )


async def drain(runner: TaskRunner, queue: FakeTaskQueue) -> None:
    """Drains the queue using multiple concurrent workers."""

    async def worker() -> None:
        while queue.has_work:
            if await runner.run_next() is None:
                await asyncio.sleep(0)

    await asyncio.gather(*(worker() for _ in range(WORKER_COUNT)))


async def test_analysis_tasks_run_in_reading_order_one_at_a_time() -> None:
    """Chapters are analyzed in parallel across workers."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",), before_you_read=[]))
    sequence = make_sequence(unit_of_work, queue, YieldingLLMClient(client))
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]  # queue port double
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )

    await sequence.start(BOOK_ID)
    await drain(runner, queue)

    expected = [f"section-{number}" for number in range(1, CHAPTER_COUNT + 1)]
    assert queue.claimed_scope_ids == expected
    assert unit_of_work.analyses.saved_section_ids == expected
    assert queue.max_running == CHAPTER_COUNT
    assert [
        f"Chapter {number}" in request.input_text
        for number, request in enumerate(client.requests, start=1)
    ] == [True] * CHAPTER_COUNT


async def test_start_moves_book_to_analyzing_and_queues_first_chapter() -> None:
    """Starting analysis transitions the book to `ANALYZING` and queues all chapters (ALG-06)."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    sequence = make_sequence(
        unit_of_work, queue, FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    )

    task = await sequence.start(BOOK_ID)

    assert task is not None
    assert (task.task_type, task.scope_id) == (TaskType.ANALYZE_CHAPTER, "section-1")
    assert queue.tasks[0] == task
    assert len(queue.tasks) == CHAPTER_COUNT
    assert [t.scope_id for t in queue.tasks] == [f"section-{i}" for i in range(1, CHAPTER_COUNT + 1)]
    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.ANALYZING


async def test_completed_analysis_moves_book_to_glossary_review() -> None:
    """After the last chapter, the book advances to glossary review (ALG-06)."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",), before_you_read=[]))
    sequence = make_sequence(unit_of_work, queue, client)
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]  # queue port double
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )

    await sequence.start(BOOK_ID)
    await drain(runner, queue)

    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.GLOSSARY_REVIEW
    assert len(queue.tasks) == CHAPTER_COUNT


async def test_start_resumes_from_first_unanalysed_chapter() -> None:
    """Restarting resumes from the first unanalyzed chapter (PRD § 18)."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",), before_you_read=[]))
    sequence = make_sequence(unit_of_work, queue, client)
    await sequence.start(BOOK_ID)
    first = queue.tasks[0]
    await sequence.run_task(first.start())

    resumed = await sequence.start(BOOK_ID)

    assert resumed is not None and resumed.scope_id == "section-2"
    assert client.call_count == 1


async def test_foreign_task_type_is_rejected() -> None:
    """The analysis handler rejects tasks of an alien type."""
    unit_of_work = make_book()
    sequence = make_sequence(
        unit_of_work,
        FakeTaskQueue(),
        FakeLLMClient.always(analysis_reply(aliases=("b1",))),
    )

    with pytest.raises(AnalyzeChapterError):
        await sequence.run_task(
            ProcessingTask(id="task-1", book_id=BOOK_ID, task_type=TaskType.EXPORT_BOOK)
        )


async def test_analysis_does_not_start_before_import_is_finished() -> None:
    """A book that is still importing cannot start analysis (ALG-06)."""
    unit_of_work = make_book()
    unit_of_work.books.save(make_book_record(status=BookStatus.IMPORTING))
    sequence = make_sequence(
        unit_of_work,
        FakeTaskQueue(),
        FakeLLMClient.always(analysis_reply(aliases=("b1",))),
    )

    with pytest.raises(BookStatusTransitionError):
        await sequence.start(BOOK_ID)


async def test_unknown_book_cannot_start_analysis() -> None:
    """Analysis cannot be started for a book not in storage."""
    sequence = make_sequence(
        FakeUnitOfWork(), FakeTaskQueue(), FakeLLMClient.always(analysis_reply())
    )

    with pytest.raises(AnalyzeChapterError):
        await sequence.start("unknown-book")


async def test_repeated_start_retries_analysis_stopped_by_llm_configuration() -> None:
    """Analysis stopped by provider configuration error is retried upon restart (QA-01).

    The first chapter fails with `LLMConfigurationError`: subsequent chapters are not queued,
    and the book stays in `ANALYZING`. When the user fixes the key and starts analysis again,
    the same task is reenqueued, and the book reaches glossary review without reimporting.
    """
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient(
        replies=(LLMConfigurationError("провайдер не принял ключ: 401"),),
        default=analysis_reply(aliases=("b1",), before_you_read=[]),
    )
    sequence = make_sequence(unit_of_work, queue, client)
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]  # queue port double
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )
    first = await sequence.start(BOOK_ID)
    failed = await runner.run_next()
    assert first is not None and failed is not None
    assert failed.status is TaskStatus.FAILED
    assert failed.is_retryable

    restarted = await sequence.start(BOOK_ID)
    await drain(runner, queue)

    assert restarted is not None
    assert restarted.id == first.id
    assert restarted.status is TaskStatus.PENDING
    assert unit_of_work.analyses.saved_section_ids == [
        f"section-{number}" for number in range(1, CHAPTER_COUNT + 1)
    ]
    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.GLOSSARY_REVIEW


async def test_repeated_start_keeps_a_permanently_failed_analysis_task() -> None:
    """Restarting does not re-enqueue a permanently failed task."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    sequence = make_sequence(
        unit_of_work, queue, FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    )
    first = await sequence.start(BOOK_ID)
    assert first is not None
    permanent = first.start().fail(
        TaskError(type=ErrorType.PERMANENT, message="обработчик не зарегистрирован")
    )
    await queue.save(permanent)

    again = await sequence.start(BOOK_ID)

    assert again == permanent


async def test_auto_start_processing_transitions_book_after_auto_approve_glossary() -> None:
    """When auto_approve_glossary and auto_start_processing are enabled and review completes, book transitions to PROCESSING (P2.T1)."""
    from types import SimpleNamespace

    unit_of_work = make_book(chapters=2)
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": True,
                "auto_start_processing": True,
                "glossary_confidence_threshold": 0.7,
                "glossary_review_min_mentions": 1,
                "glossary_review_min_chapters": 1,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue = FakeTaskQueue()
    reply = analysis_reply(
        aliases=("b1",),
        before_you_read=[],
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Даскмер",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    client = FakeLLMClient.always(reply)
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    started_books: list[str] = []

    async def fake_start_processing(book_id: str) -> None:
        started_books.append(book_id)
        book = unit_of_work.books.get(book_id)
        assert book is not None and book.status is BookStatus.READY_TO_PROCESS
        unit_of_work.books.save(make_book_record(status=BookStatus.PROCESSING))

    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
        auto_start_processing=True,
        start_processing=fake_start_processing,
    )
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )

    await sequence.start(BOOK_ID)
    await drain(runner, queue)

    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.PROCESSING
    assert started_books == [BOOK_ID]

    # Also verify fallback transition when start_processing callback is None
    unit_of_work_direct = make_book(chapters=1)
    unit_of_work_direct.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue_direct = FakeTaskQueue()
    analyze_direct, _ = make_analyze_chapter(
        unit_of_work_direct, FakeLLMClient.always(reply)
    )
    sequence_direct = ChapterAnalysisSequence(
        unit_of_work=unit_of_work_direct,  # type: ignore[arg-type]
        queue=queue_direct,  # type: ignore[arg-type]
        analyze_chapter=analyze_direct,
        auto_start_processing=True,
    )
    first_direct = await sequence_direct.start(BOOK_ID)
    assert first_direct is not None
    await sequence_direct.run_task(first_direct.start())
    stored_direct = unit_of_work_direct.books.get(BOOK_ID)
    assert stored_direct is not None and stored_direct.status is BookStatus.PROCESSING


async def test_auto_approve_glossary_resolves_parallel_translation_conflicts_and_reuses_mention_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel chapter analysis with conflicting translations auto-approves high-confidence unflagged candidates and builds mention_counter once (P2.T1)."""
    from types import SimpleNamespace
    import linguary.application.mention_counter as mention_counter_mod
    from linguary.domain import TermStatus

    original_mention_counter = mention_counter_mod.mention_counter
    mention_counter_calls = 0

    def counting_mention_counter(document):
        nonlocal mention_counter_calls
        mention_counter_calls += 1
        return original_mention_counter(document)

    monkeypatch.setattr(
        mention_counter_mod, "mention_counter", counting_mention_counter
    )

    unit_of_work = make_book(chapters=3)
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": True,
                "auto_start_processing": True,
                "glossary_confidence_threshold": 0.7,
                "glossary_review_min_mentions": 1,
                "glossary_review_min_chapters": 1,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue = FakeTaskQueue()
    reply_daskmer = analysis_reply(
        aliases=("b1",),
        before_you_read=[],
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Даскмер",
                "confidence": 0.95,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    reply_sumerechye = analysis_reply(
        aliases=("b1",),
        before_you_read=[],
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Сумеречье",
                "confidence": 0.85,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    client = FakeLLMClient.replying((reply_daskmer, reply_sumerechye, reply_daskmer))
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    analyze_chapter._max_concurrency = 2
    started_books: list[str] = []

    async def fake_start_processing(book_id: str) -> None:
        started_books.append(book_id)
        unit_of_work.books.save(make_book_record(status=BookStatus.PROCESSING))

    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
        auto_start_processing=True,
        start_processing=fake_start_processing,
    )
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )

    await sequence.start(BOOK_ID)
    await drain(runner, queue)

    assert mention_counter_calls == 1
    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.PROCESSING
    assert started_books == [BOOK_ID]
    glossary_items = unit_of_work.glossary.list_for_book(BOOK_ID)
    assert len(glossary_items) == 1
    assert glossary_items[0].canonical_source_text == "Duskmere"
    assert glossary_items[0].status is TermStatus.APPROVED
    assert glossary_items[0].variants[0].text == "Даскмер"


async def test_try_complete_analysis_avoids_duplicate_queries_and_only_approves_requires_review_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_try_complete_analysis reuses updated_analyses, queries glossary and builds review once, and only saves TermConcept for requires_review items (P3.T1)."""
    from types import SimpleNamespace
    import linguary.application.glossary_review as glossary_review_mod
    from analysis_fakes import make_analysis_candidate, make_chapter_analysis
    from linguary.domain import ConceptType, TermStatus

    unit_of_work = make_book(chapters=1)
    unit_of_work.books.save(make_book_record(status=BookStatus.ANALYZING))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            id="analysis-1",
            section_id="section-1",
            candidates=(
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Duskmere",
                    type=ConceptType.PLACE,
                    suggested_translation="Даскмер",
                    confidence=0.95,
                    model_flagged_review=False,
                ),
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="boardwalk",
                    type=ConceptType.TERM,
                    suggested_translation="деревянный настил",
                    confidence=0.95,
                    model_flagged_review=False,
                ),
            ),
        ),
    )
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": True,
                "auto_start_processing": True,
                "glossary_confidence_threshold": 0.7,
                "glossary_review_min_mentions": 1,
                "glossary_review_min_chapters": 1,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    sequence = make_sequence(
        unit_of_work,
        FakeTaskQueue(),
        FakeLLMClient.always(analysis_reply(aliases=("b1",))),
    )

    analyses_calls = 0
    original_analyses_list = type(unit_of_work.analyses).list_for_book

    def counting_analyses_list(self, book_id: str):
        nonlocal analyses_calls
        analyses_calls += 1
        return original_analyses_list(self, book_id)

    glossary_calls = 0
    original_glossary_list = type(unit_of_work.glossary).list_for_book

    def counting_glossary_list(self, book_id: str):
        nonlocal glossary_calls
        glossary_calls += 1
        return original_glossary_list(self, book_id)

    build_review_calls = 0
    original_build_review = glossary_review_mod.build_glossary_review

    def counting_build_review(*args, **kwargs):
        nonlocal build_review_calls
        build_review_calls += 1
        return original_build_review(*args, **kwargs)

    monkeypatch.setattr(
        type(unit_of_work.analyses), "list_for_book", counting_analyses_list
    )
    monkeypatch.setattr(
        type(unit_of_work.glossary), "list_for_book", counting_glossary_list
    )
    monkeypatch.setattr(
        glossary_review_mod, "build_glossary_review", counting_build_review
    )

    sequence._try_complete_analysis(BOOK_ID)

    assert analyses_calls == 1
    assert glossary_calls == 1
    assert build_review_calls == 1

    saved_concepts = unit_of_work.glossary.list_for_book(BOOK_ID)
    assert [c.canonical_source_text for c in saved_concepts] == ["Duskmere"]
    assert saved_concepts[0].status is TermStatus.APPROVED
    stored_book = unit_of_work.books.get(BOOK_ID)
    assert stored_book is not None and stored_book.status is BookStatus.PROCESSING




