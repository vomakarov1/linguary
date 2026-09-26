"""Задачи обработки глав и завершение обработки книги (P3.T6, ALG-02, ALG-06).

Здесь проверяется стык, ради которого сценарий существует: очередь на SQLite, воркер,
`ChapterProcessingSequence` и `FakeLLMClient` работают вместе. Обработчик типа задачи
`PROCESS_TRANSLATION_UNIT` регистрируется так же, как это сделает composition root
фазы 4, — готовым `ChapterProcessingSequence.run_task`.
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


@dataclass(slots=True)
class SlowLLMClient(LLMClient):
    """Клиент, отвечающий с задержкой сети и считающий одновременные вызовы.

    Мгновенный ответ не отдаёт управление циклу событий, и воркеры отработали бы
    по очереди: параллельной обработки тест бы не увидел.
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
    """Очередь, итог задачи в которой фиксируется с задержкой.

    Так воспроизводится окно между концом обработчика и записью итога задачи runner-ом:
    при параллельной обработке в это окно успевает закончиться соседняя глава.
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
    """Воркер с единственным обработчиком — задачей обработки главы."""
    return TaskWorkerPool(
        TaskRunner(
            queue=task_queue,  # type: ignore[arg-type]
            handlers={TaskType.PROCESS_TRANSLATION_UNIT: sequence.run_task},
        ),
        WorkerSettings(concurrency=concurrency),
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


async def test_parallel_processing_handles_every_chapter_once(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Несколько воркеров над одной базой обрабатывают каждую главу ровно один раз (PRD § 25).

    Захват задач и запись результатов идут в SQLite одновременно из нескольких
    воркеров: ни одна глава не должна достаться двоим, ни одна запись — потеряться.
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
    """Последние главы, закончившиеся одновременно, всё равно завершают обработку (ALG-06).

    Каждая глава проверяет, остались ли незавершённые соседи, пока её собственный итог
    ещё не записан. Если соседи проверяют друг друга в это окно, каждая видит другую
    выполняющейся, и книга навсегда остаётся в `PROCESSING`.
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
