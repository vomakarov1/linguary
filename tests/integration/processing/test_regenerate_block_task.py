"""Block regeneration task and book state (P3.T7, ALG-02, ALG-06).

Tests the integration point that motivates this scenario: SQLite queue, worker pool,
`BlockRegenerationSequence`, and `FakeLLMClient` work together. The `REGENERATE_BLOCK`
task handler is registered just as the phase 4 composition root will register it —
via `BlockRegenerationSequence.run_task`.
"""

from __future__ import annotations

import pytest
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    block_reply,
    make_process_chapter,
    make_regenerate_block,
    store_processing_book,
    translated_reply,
    translation_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.runtime import SqliteTaskQueue, TaskWorkerPool
from linguary.application.ports.llm import LLMClient
from linguary.application.task_runner import TaskRunner
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.application.use_cases.regenerate_block import (
    BlockRegenerationSequence,
    RegenerateBlockError,
)
from linguary.domain import (
    BookStatus,
    ProcessingTask,
    TaskStatus,
    TaskType,
)

#: Another translation of the same paragraph: shows the block result is indeed replaced.
REGENERATED_TEXT = "Всем известно: одинокий богач непременно ищет себе жену."

#: Second regeneration of the same block: should reach storage just like the first.
TWICE_REGENERATED_TEXT = "Известная всем истина: богатый холостяк ищет себе жену."


def make_sequence(
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    client: LLMClient,
) -> BlockRegenerationSequence:
    """Block regeneration sequence on top of the real task queue."""
    regenerate, _ = make_regenerate_block(unit_of_work, client)
    return BlockRegenerationSequence(
        unit_of_work=unit_of_work, queue=task_queue, regenerate_block=regenerate
    )


def make_pool(
    task_queue: SqliteTaskQueue, sequence: BlockRegenerationSequence
) -> TaskWorkerPool:
    """Worker pool with a single handler — block regeneration task."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue, handlers={TaskType.REGENERATE_BLOCK: sequence.run_task}
        )
    )


def book_status(unit_of_work: SqlAlchemyUnitOfWork) -> BookStatus:
    """Current book status."""
    with unit_of_work.transaction() as uow:
        book = uow.books.get(BOOK_ID)
    assert book is not None
    return book.status


def translations(unit_of_work: SqlAlchemyUnitOfWork) -> list[str | None]:
    """Translations of run blocks in document order."""
    with unit_of_work.transaction() as uow:
        return [item.translated_text for item in uow.results.list_for_run(RUN_ID)]


async def process_chapter(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Brings the first chapter to completion: regeneration replaces already finished results."""
    process, _ = make_process_chapter(
        unit_of_work, FakeLLMClient.always(translated_reply())
    )
    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1"))


def regenerated_reply(text: str = REGENERATED_TEXT) -> FakeLLMClient:
    """Client answering with a new translation of the single requested block."""
    return FakeLLMClient.always(
        translation_reply([block_reply("b1", translated_text=text)])
    )


async def test_request_moves_the_book_through_processing_back_to_export(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Regeneration transitions book to `PROCESSING` and brings it back (ALG-06)."""
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    await process_chapter(unit_of_work)
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())

    task = await sequence.start(BOOK_ID, scenario.block_ids()[1])

    assert (task.task_type, task.scope_id, task.status) == (
        TaskType.REGENERATE_BLOCK,
        scenario.block_ids()[1],
        TaskStatus.PENDING,
    )
    assert book_status(unit_of_work) is BookStatus.PROCESSING

    finished = await make_pool(task_queue, sequence).drain()

    assert [item.status for item in finished] == [TaskStatus.SUCCEEDED]
    assert book_status(unit_of_work) is BookStatus.READY_TO_EXPORT
    assert translations(unit_of_work) == [TRANSLATED_TEXT, REGENERATED_TEXT]


async def test_repeated_request_of_a_scheduled_block_does_not_duplicate_the_task(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """While a block task is queued, a second request does not create a duplicate (ALG-02)."""
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())

    first = await sequence.start(BOOK_ID, scenario.block_ids()[0])
    second = await sequence.start(BOOK_ID, scenario.block_ids()[0])

    assert first.id == second.id
    with unit_of_work.transaction() as uow:
        assert len(uow.tasks.list_for_book(BOOK_ID)) == 1


async def test_block_can_be_regenerated_twice(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """A second request for the same block runs rather than returning previous result.

    The task idempotency key lives while the task lives: a completed block task is
    cleared, otherwise block regeneration would only work once (PRD § 21.5).
    """
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    await process_chapter(unit_of_work)
    block_id = scenario.block_ids()[0]
    first = make_sequence(unit_of_work, task_queue, regenerated_reply())
    await first.start(BOOK_ID, block_id)
    await make_pool(task_queue, first).drain()

    second = make_sequence(
        unit_of_work, task_queue, regenerated_reply(TWICE_REGENERATED_TEXT)
    )
    task = await second.start(BOOK_ID, block_id)
    finished = await make_pool(task_queue, second).drain()

    assert task.status is TaskStatus.PENDING
    assert [item.status for item in finished] == [TaskStatus.SUCCEEDED]
    assert translations(unit_of_work) == [TWICE_REGENERATED_TEXT, TRANSLATED_TEXT]


async def test_request_during_processing_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """While the book is being processed, block replacement is rejected (PRD § 21.5)."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT), (SOURCE_TEXT,))
    )
    with unit_of_work.transaction() as uow:
        uow.tasks.add(
            ProcessingTask(
                id="task-processing",
                book_id=BOOK_ID,
                task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id="section-2",
            )
        )
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())

    with pytest.raises(RegenerateBlockError, match="ещё обрабатывается"):
        await sequence.start(BOOK_ID, scenario.block_ids()[0])

    with unit_of_work.transaction() as uow:
        assert [item.task_type for item in uow.tasks.list_for_book(BOOK_ID)] == [
            TaskType.PROCESS_TRANSLATION_UNIT
        ]


async def test_two_regenerations_return_the_book_to_export_once_both_finish(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """The book waits for `READY_TO_EXPORT` until the last regeneration finishes."""
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    await process_chapter(unit_of_work)
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())
    await sequence.start(BOOK_ID, scenario.block_ids()[0])
    await sequence.start(BOOK_ID, scenario.block_ids()[1])

    finished = await make_pool(task_queue, sequence).drain()

    assert [item.status for item in finished] == [TaskStatus.SUCCEEDED] * 2
    assert book_status(unit_of_work) is BookStatus.READY_TO_EXPORT
    assert translations(unit_of_work) == [REGENERATED_TEXT, REGENERATED_TEXT]


async def test_task_of_another_type_is_not_processed_as_a_regeneration(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """The regeneration handler rejects a task of an alien type."""
    store_processing_book(unit_of_work)
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())
    task = ProcessingTask(
        id="task-9",
        book_id=BOOK_ID,
        task_type=TaskType.PROCESS_TRANSLATION_UNIT,
        scope_id="section-1",
        status=TaskStatus.RUNNING,
    )

    with pytest.raises(RegenerateBlockError, match="не является повторной генерацией"):
        await sequence.run_task(task)
