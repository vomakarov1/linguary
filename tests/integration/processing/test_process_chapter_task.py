"""Chapter processing tasks and book processing completion (P3.T6, ALG-02, ALG-06).

This verifies the junction for which the use case exists: SQLite queue, worker,
`ChapterProcessingSequence`, and `FakeLLMClient` working together. The handler for task type
`PROCESS_TRANSLATION_UNIT` is registered in the same way as the phase 4 composition root
will do it — with `ChapterProcessingSequence.run_task`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from processing_scenario import (
    BOOK_ID,
    SOURCE_TEXT,
    make_process_chapter,
    store_processing_book,
    translated_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.runtime import SqliteTaskQueue, TaskWorkerPool, WorkerSettings
from linguary.application.ports.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMPayload,
    LLMRequest,
    LLMResponse,
)
from linguary.application.task_runner import TaskRunner
from linguary.application.use_cases.process_chapter import (
    ChapterProcessingSequence,
    ProcessChapterError,
)
from linguary.domain import (
    BookStatus,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
)

#: Two chapters of two paragraphs each: a book whose processing can be completed.
TWO_CHAPTERS = ((SOURCE_TEXT, SOURCE_TEXT), (SOURCE_TEXT, SOURCE_TEXT))


@dataclass(slots=True)
class ChapterAwareLLMClient(LLMClient):
    """Client that fails calls for one chapter and answers the rest.

    Ensures that failure of one chapter is not failure of the whole book (PRD § 16):
    the remaining chapters must finish successfully.
    """

    failing_chapter: str
    inner: FakeLLMClient

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        if self.failing_chapter in request.input_text:
            raise LLMConfigurationError("модель для этой главы не настроена")
        return await self.inner.generate(request)


def make_sequence(
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    client: LLMClient,
) -> ChapterProcessingSequence:
    """Chapter processing task sequence on top of real queue."""
    process, _ = make_process_chapter(unit_of_work, client)
    return ChapterProcessingSequence(
        unit_of_work=unit_of_work, queue=task_queue, process_chapter=process
    )


@dataclass(slots=True)
class SlowLLMClient(LLMClient):
    """Client replying with network latency and counting concurrent calls.

    An instantaneous response does not yield to the event loop, causing workers
    to execute sequentially without revealing parallelism.
    """

    inner: FakeLLMClient
    delay_seconds: float = 0.02
    in_flight: int = 0
    max_in_flight: int = 0

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay_seconds)
            return await self.inner.generate(request)
        finally:
            self.in_flight -= 1


@dataclass(slots=True)
class SlowSavingQueue:
    """Queue where task completion is persisted with a delay.

    Reproduces the window between handler completion and task completion write by runner:
    during parallel processing a neighboring chapter finishes in this window.
    """

    inner: SqliteTaskQueue
    delay_seconds: float = 0.05
    saved: list[str] = field(default_factory=list)

    async def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        return await self.inner.enqueue(identity)

    async def claim_next_pending(self) -> ProcessingTask | None:
        return await self.inner.claim_next_pending()

    async def save(self, task: ProcessingTask) -> None:
        await asyncio.sleep(self.delay_seconds)
        await self.inner.save(task)
        self.saved.append(task.id)

    async def list_running(self) -> tuple[ProcessingTask, ...]:
        return await self.inner.list_running()


def make_pool(
    task_queue: SqliteTaskQueue | SlowSavingQueue,
    sequence: ChapterProcessingSequence,
    *,
    concurrency: int = 1,
) -> TaskWorkerPool:
    """Worker with a single handler — chapter processing task."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue,  # type: ignore[arg-type]
            handlers={TaskType.PROCESS_TRANSLATION_UNIT: sequence.run_task},
        ),
        WorkerSettings(concurrency=concurrency),
    )


def book_status(unit_of_work: SqlAlchemyUnitOfWork) -> BookStatus:
    """Current status of the book."""
    with unit_of_work.transaction() as uow:
        book = uow.books.get(BOOK_ID)
    assert book is not None
    return book.status


async def test_start_enqueues_a_task_for_every_chapter_with_text(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Only chapters with text get a task: cover has nothing to process."""
    store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT,), (None,), (SOURCE_TEXT,))
    )
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())

    tasks = await sequence.start(BOOK_ID)

    assert [task.scope_id for task in tasks] == ["section-1", "section-3"]
    assert all(task.task_type is TaskType.PROCESS_TRANSLATION_UNIT for task in tasks)


async def test_repeated_start_does_not_duplicate_tasks(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Task enqueueing is idempotent on book, type, and chapter key (ALG-02)."""
    store_processing_book(unit_of_work, chapters=TWO_CHAPTERS)
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())

    first = await sequence.start(BOOK_ID)
    second = await sequence.start(BOOK_ID)

    assert [task.id for task in first] == [task.id for task in second]
    with unit_of_work.transaction() as uow:
        assert len(uow.tasks.list_for_book(BOOK_ID)) == 2


async def test_processed_book_becomes_ready_to_export(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Processing all chapters transitions the book to `READY_TO_EXPORT` (ALG-06)."""
    scenario = store_processing_book(unit_of_work, chapters=TWO_CHAPTERS)
    sequence = make_sequence(
        unit_of_work, task_queue, FakeLLMClient.always(translated_reply())
    )
    await sequence.start(BOOK_ID)

    finished = await make_pool(task_queue, sequence).drain()

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED] * 2
    assert book_status(unit_of_work) is BookStatus.READY_TO_EXPORT
    with unit_of_work.transaction() as uow:
        results = uow.results.list_for_run(scenario.run_id)
        assert uow.runs.active_for_book(BOOK_ID) is not None
    assert len(results) == 4


async def test_parallel_processing_handles_every_chapter_once(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Multiple workers on one database process each chapter exactly once (PRD § 25).

    Task claiming and result writing happen concurrently from multiple workers:
    no chapter should be taken by two workers, and no record lost.
    """
    chapters = tuple((SOURCE_TEXT, SOURCE_TEXT) for _ in range(6))
    scenario = store_processing_book(unit_of_work, chapters=chapters)
    client = SlowLLMClient(inner=FakeLLMClient.always(translated_reply()))
    sequence = make_sequence(unit_of_work, task_queue, client)
    await sequence.start(BOOK_ID)

    finished = await make_pool(task_queue, sequence, concurrency=4).drain()

    assert client.max_in_flight > 1
    assert sorted(task.scope_id for task in finished) == sorted(
        section.id for section in scenario.document.ordered_sections
    )
    assert all(task.status is TaskStatus.SUCCEEDED for task in finished)
    assert all(task.attempts == 1 for task in finished)
    assert client.inner.call_count == len(chapters)
    with unit_of_work.transaction() as uow:
        results = uow.results.list_for_run(scenario.run_id)
    assert len(results) == 2 * len(chapters)
    assert book_status(unit_of_work) is BookStatus.READY_TO_EXPORT


async def test_chapters_finishing_together_complete_the_book(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """The last chapters finishing concurrently still complete processing (ALG-06).

    Each chapter checks whether unfinished peers remain before its own result is written.
    If peers check each other in this window, both see the other as running,
    leaving the book permanently in `PROCESSING`.
    """
    store_processing_book(unit_of_work, chapters=TWO_CHAPTERS)
    queue = SlowSavingQueue(task_queue)
    sequence = make_sequence(
        unit_of_work, queue, FakeLLMClient.always(translated_reply())  # type: ignore[arg-type]
    )
    await sequence.start(BOOK_ID)

    finished = await make_pool(queue, sequence, concurrency=2).drain()

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED] * 2
    assert book_status(unit_of_work) is BookStatus.READY_TO_EXPORT


async def test_failed_chapter_keeps_the_book_in_processing(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Failure of one chapter does not finish book processing and does not stop others."""
    scenario = store_processing_book(unit_of_work, chapters=TWO_CHAPTERS)
    sequence = make_sequence(
        unit_of_work,
        task_queue,
        ChapterAwareLLMClient(
            failing_chapter="Chapter 2",
            inner=FakeLLMClient.always(translated_reply()),
        ),
    )
    await sequence.start(BOOK_ID)

    finished = await make_pool(task_queue, sequence).drain()

    assert sorted(task.status for task in finished) == [
        TaskStatus.FAILED,
        TaskStatus.SUCCEEDED,
    ]
    assert book_status(unit_of_work) is BookStatus.PROCESSING
    with unit_of_work.transaction() as uow:
        results = uow.results.list_for_run(scenario.run_id)
    assert [item.block_id for item in results] == list(scenario.block_ids(0))


async def test_start_without_an_active_run_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Processing cannot be started until glossary review creates a run (PRD § 12)."""
    store_processing_book(unit_of_work)
    with unit_of_work.transaction() as uow:
        uow.runs.delete("run-1")
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())

    with pytest.raises(ProcessChapterError, match="активного запуска"):
        await sequence.start(BOOK_ID)


async def test_task_of_another_type_is_not_processed_as_a_chapter(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Chapter handler rejects tasks of an unexpected type."""
    store_processing_book(unit_of_work)
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())
    task = ProcessingTask(
        id="task-9",
        book_id=BOOK_ID,
        task_type=TaskType.ANALYZE_CHAPTER,
        scope_id="section-1",
        status=TaskStatus.RUNNING,
    )

    with pytest.raises(ProcessChapterError, match="не является обработкой главы"):
        await sequence.run_task(task)
