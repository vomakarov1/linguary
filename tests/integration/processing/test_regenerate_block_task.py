"""Задача повторной генерации блока и состояние книги (P3.T7, ALG-02, ALG-06).

Здесь проверяется стык, ради которого сценарий существует: очередь на SQLite, воркер,
`BlockRegenerationSequence` и `FakeLLMClient` работают вместе. Обработчик типа задачи
`REGENERATE_BLOCK` регистрируется так же, как это сделает composition root фазы 4, —
готовым `BlockRegenerationSequence.run_task`.
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

#: Другой перевод того же абзаца: по нему видно, что результат блока действительно заменён.
REGENERATED_TEXT = "Всем известно: одинокий богач непременно ищет себе жену."

#: Второй повтор того же блока: он должен дойти до хранилища так же, как первый.
TWICE_REGENERATED_TEXT = "Известная всем истина: богатый холостяк ищет себе жену."


def make_sequence(
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    client: LLMClient,
) -> BlockRegenerationSequence:
    """Последовательность задач повторной генерации поверх настоящей очереди."""
    regenerate, _ = make_regenerate_block(unit_of_work, client)
    return BlockRegenerationSequence(
        unit_of_work=unit_of_work, queue=task_queue, regenerate_block=regenerate
    )


def make_pool(
    task_queue: SqliteTaskQueue, sequence: BlockRegenerationSequence
) -> TaskWorkerPool:
    """Воркер с единственным обработчиком — задачей повторной генерации."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue, handlers={TaskType.REGENERATE_BLOCK: sequence.run_task}
        )
    )


def book_status(unit_of_work: SqlAlchemyUnitOfWork) -> BookStatus:
    """Текущее состояние книги."""
    with unit_of_work.transaction() as uow:
        book = uow.books.get(BOOK_ID)
    assert book is not None
    return book.status


def translations(unit_of_work: SqlAlchemyUnitOfWork) -> list[str | None]:
    """Переводы блоков запуска в порядке документа."""
    with unit_of_work.transaction() as uow:
        return [item.translated_text for item in uow.results.list_for_run(RUN_ID)]


async def process_chapter(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Доводит первую главу до результата: повторная генерация заменяет уже готовое."""
    process, _ = make_process_chapter(
        unit_of_work, FakeLLMClient.always(translated_reply())
    )
    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1"))


def regenerated_reply(text: str = REGENERATED_TEXT) -> FakeLLMClient:
    """Клиент, отвечающий новым переводом единственного блока запроса."""
    return FakeLLMClient.always(
        translation_reply([block_reply("b1", translated_text=text)])
    )


async def test_request_moves_the_book_through_processing_back_to_export(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Повторная генерация переводит книгу в `PROCESSING` и возвращает её обратно (ALG-06)."""
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
    """Пока задача блока ждёт очереди, второй запрос новой задачи не создаёт (ALG-02)."""
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
    """Второй запрос по тому же блоку выполняется, а не возвращает прошлый результат.

    Идемпотентный ключ задачи живёт, пока живёт задача: завершённая задача блока
    снимается, иначе повторная генерация работала бы ровно один раз (PRD § 21.5).
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
    """Пока книга обрабатывается, заменять результат блока нечем (PRD § 21.5)."""
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
    """Книга ждёт `READY_TO_EXPORT`, пока не закончилась последняя повторная генерация."""
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
    """Обработчик повторной генерации отвергает задачу чужого типа."""
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
