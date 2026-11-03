"""Block regeneration task and book state (P3.T7, ALG-02, ALG-06).

Tests the integration point that motivates this scenario: SQLite queue, worker pool,
`BlockRegenerationSequence`, and `FakeLLMClient` work together. The `REGENERATE_BLOCK`
task handler is registered just as the phase 4 composition root will register it —
via `BlockRegenerationSequence.run_task`.
"""

from __future__ import annotations

import asyncio
import pytest
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    block_reply,
    store_processing_book,
    translated_reply,
    translation_reply,
)
from test_regenerate_block import make_process_chapter, make_regenerate_block

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.runtime import SqliteTaskQueue, TaskWorkerPool
from linguary.application.ports.llm import LLMClient
from linguary.application.task_runner import TaskRunner
from linguary.application.translation_policy import (
    CURRENT_TRANSLATION_POLICY,
    LEGACY_TRANSLATION_POLICY,
)
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
        root, = uow.work.list_roots(BOOK_ID, run_id=RUN_ID)
        assert root.id == first.operation_id


async def test_double_click_deduplicates_active_operation(unit_of_work, task_queue, monkeypatch):
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())
    other_sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())
    enqueue = task_queue.enqueue

    async def delayed_enqueue(identity):
        await asyncio.sleep(0)
        return await enqueue(identity)

    monkeypatch.setattr(task_queue, "enqueue", delayed_enqueue)

    first, second = await asyncio.gather(
        other_sequence.start(BOOK_ID, scenario.block_ids()[0]),
        sequence.start(BOOK_ID, scenario.block_ids()[0]),
    )

    assert first.id == second.id
    assert first.operation_id == second.operation_id
    with unit_of_work.transaction() as uow:
        assert len(uow.tasks.list_for_book(BOOK_ID)) == 1
        (root,) = uow.work.list_roots(BOOK_ID, run_id=RUN_ID)
        assert root.id == first.operation_id
        (scope,) = uow.work.list_scopes(root.id)
        assert scope.ordered_blocks == (scenario.block_ids()[0],)
        assert uow.work.get_binding("task", first.id)["payload_json"]["operation_refs"] == [
            {"root_id": root.id, "policy_version": root.policy_version}
        ]


@pytest.mark.parametrize(
    ("repository", "method"),
    [("work", "add_root"), ("work", "add_scope"), ("tasks", "add"),
     ("work", "bind_operation_refs"), ("books", "save")],
)
async def test_regeneration_start_transaction_failure_rolls_back_root_scope_task_binding_and_book_status(
    unit_of_work, task_queue, monkeypatch, repository, method
):
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())
    with unit_of_work.transaction() as uow:
        repository_type = type(getattr(uow, repository))
    original = getattr(repository_type, method)

    def fail_after_write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(repository_type, method, fail_after_write)
    with pytest.raises(RuntimeError, match="injected failure"):
        await sequence.start(BOOK_ID, scenario.block_ids()[0])
    with unit_of_work.transaction() as uow:
        assert uow.work.list_roots(BOOK_ID, run_id=RUN_ID) == ()
        assert uow.tasks.list_for_book(BOOK_ID) == ()
        assert uow.books.get(BOOK_ID).status is BookStatus.READY_TO_EXPORT


async def test_committed_regeneration_is_claimable_without_enqueue(unit_of_work, task_queue, monkeypatch):
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    sequence = make_sequence(unit_of_work, task_queue, regenerated_reply())

    async def fail_before_enqueue(identity):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(task_queue, "enqueue", fail_before_enqueue)
    with pytest.raises(RuntimeError, match="queue unavailable"):
        await sequence.start(BOOK_ID, scenario.block_ids()[0])
    task = await task_queue.claim_next_pending()
    assert task is not None and task.status is TaskStatus.RUNNING
    with unit_of_work.transaction() as uow:
        root = uow.work.get_root(task.operation_id)
        assert root is not None
        assert len(uow.work.list_scopes(root.id)) == 1
        assert uow.work.get_binding("task", task.id)["payload_json"]["operation_refs"] == [
            {"root_id": root.id, "policy_version": root.policy_version}
        ]
        assert uow.books.get(BOOK_ID).status is BookStatus.PROCESSING


async def test_second_completed_regeneration_has_new_identity(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """A second request for the same block runs rather than returning previous result.

    Each explicit repeat has its own operation; completed tasks remain in history.
    """
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    await process_chapter(unit_of_work)
    block_id = scenario.block_ids()[0]
    first = make_sequence(unit_of_work, task_queue, regenerated_reply())
    first_task = await first.start(BOOK_ID, block_id)
    await make_pool(task_queue, first).drain()

    second = make_sequence(
        unit_of_work, task_queue, regenerated_reply(TWICE_REGENERATED_TEXT)
    )
    task = await second.start(BOOK_ID, block_id)
    finished = await make_pool(task_queue, second).drain()

    assert task.status is TaskStatus.PENDING
    assert [item.status for item in finished] == [TaskStatus.SUCCEEDED]
    assert translations(unit_of_work) == [TWICE_REGENERATED_TEXT, TRANSLATED_TEXT]
    assert task.operation_id != first_task.operation_id
    assert task.policy_version == first_task.policy_version == CURRENT_TRANSLATION_POLICY
    with unit_of_work.transaction() as uow:
        assert uow.tasks.get(first_task.id).status is TaskStatus.SUCCEEDED
        assert uow.work.get_root(first_task.operation_id).operation_kind == "regeneration"
        assert uow.work.get_root(task.operation_id).operation_kind == "regeneration"


async def test_regeneration_has_own_root_and_current_policy(unit_of_work, task_queue):
    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    await process_chapter(unit_of_work)
    client = regenerated_reply()
    sequence = make_sequence(unit_of_work, task_queue, client)

    task = await sequence.start(BOOK_ID, scenario.block_ids()[0])

    assert client.call_count == 0
    assert task.operation_id
    assert task.policy_version == CURRENT_TRANSLATION_POLICY
    with unit_of_work.transaction() as uow:
        root = uow.work.get_root(task.operation_id)
        assert root.operation_kind == "regeneration"
        assert root.policy_version == task.policy_version
        assert tuple(root.manifest["ordered_blocks"]) == (scenario.block_ids()[0],)
        assert uow.work.get_binding("task", task.id)["payload_json"]["operation_refs"] == [
            {"root_id": root.id, "policy_version": root.policy_version}
        ]


async def test_regeneration_resume_keeps_policy_after_upgrade(unit_of_work, task_queue, monkeypatch):
    from linguary.application import recovery

    scenario = store_processing_book(unit_of_work, status=BookStatus.READY_TO_EXPORT)
    client = regenerated_reply()
    sequence = make_sequence(unit_of_work, task_queue, client)
    monkeypatch.setattr(recovery, "CURRENT_TRANSLATION_POLICY", LEGACY_TRANSLATION_POLICY)
    task = await sequence.start(BOOK_ID, scenario.block_ids()[0])
    monkeypatch.setattr(recovery, "CURRENT_TRANSLATION_POLICY", CURRENT_TRANSLATION_POLICY)

    finished = await make_pool(task_queue, sequence).drain()

    assert [item.status for item in finished] == [TaskStatus.SUCCEEDED]
    assert client.last_request.call_context.root_id == task.operation_id
    assert task.policy_version == LEGACY_TRANSLATION_POLICY
    assert client.last_request.call_context.policy_version == LEGACY_TRANSLATION_POLICY
    with unit_of_work.transaction() as uow:
        assert len(uow.work.list_roots(BOOK_ID, run_id=RUN_ID)) == 1


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
