"""Задачи обработки глав и завершение обработки книги (P3.T6, ALG-02, ALG-06).

Здесь проверяется стык, ради которого сценарий существует: очередь на SQLite, воркер,
`ChapterProcessingSequence` и `FakeLLMClient` работают вместе. Обработчик типа задачи
`PROCESS_TRANSLATION_UNIT` регистрируется так же, как это сделает composition root
фазы 4, — готовым `ChapterProcessingSequence.run_task`.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from linguary.adapters.runtime import SqliteTaskQueue, TaskWorkerPool
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
    TaskStatus,
    TaskType,
)

#: Две главы по два абзаца: книга, обработку которой можно довести до конца.
TWO_CHAPTERS = ((SOURCE_TEXT, SOURCE_TEXT), (SOURCE_TEXT, SOURCE_TEXT))


@dataclass(slots=True)
class ChapterAwareLLMClient(LLMClient):
    """Клиент, который срывает вызовы одной главы и отвечает на остальные.

    Нужен ровно для того, чтобы неуспех одной главы не был неуспехом всей книги
    (PRD § 16): остальные главы обязаны дойти до результата.
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
    """Последовательность задач обработки поверх настоящей очереди."""
    process, _ = make_process_chapter(unit_of_work, client)
    return ChapterProcessingSequence(
        unit_of_work=unit_of_work, queue=task_queue, process_chapter=process
    )


def make_pool(
    task_queue: SqliteTaskQueue, sequence: ChapterProcessingSequence
) -> TaskWorkerPool:
    """Воркер с единственным обработчиком — задачей обработки главы."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue,
            handlers={TaskType.PROCESS_TRANSLATION_UNIT: sequence.run_task},
        )
    )


def book_status(unit_of_work: SqlAlchemyUnitOfWork) -> BookStatus:
    """Текущее состояние книги."""
    with unit_of_work.transaction() as uow:
        book = uow.books.get(BOOK_ID)
    assert book is not None
    return book.status


async def test_start_enqueues_a_task_for_every_chapter_with_text(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Задачу получают только главы с текстом: обложке обрабатывать нечего."""
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
    """Постановка задач идемпотентна по ключу книги, типа и главы (ALG-02)."""
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
    """Обработка всех глав переводит книгу в `READY_TO_EXPORT` (ALG-06)."""
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


async def test_failed_chapter_keeps_the_book_in_processing(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Неуспех одной главы не завершает обработку книги и не останавливает остальные."""
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
    """Обработка не ставится, пока проверка глоссария не завершена запуском (PRD § 12)."""
    store_processing_book(unit_of_work)
    with unit_of_work.transaction() as uow:
        uow.runs.delete("run-1")
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())

    with pytest.raises(ProcessChapterError, match="активного запуска"):
        await sequence.start(BOOK_ID)


async def test_task_of_another_type_is_not_processed_as_a_chapter(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Обработчик главы отвергает задачу чужого типа."""
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
