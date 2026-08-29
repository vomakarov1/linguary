"""Возврат к проверке глоссария: сброс результатов обработки (P3.T7, PRD § 11, § 21.5).

Тесты идут на настоящем SQLite: сброс — транзакция, которая удаляет запуск с его
результатами и задачи отменённой работы, и проверять её имеет смысл там же, где она
выполняется у пользователя.
"""

from __future__ import annotations

import pytest
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    InterruptingLLMClient,
    make_process_chapter,
    note_reply,
    store_processing_book,
    translated_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.runtime import SqliteTaskQueue, TaskWorkerPool
from linguary.application.task_runner import TaskRunner
from linguary.application.use_cases.approve_glossary import (
    StartProcessing,
    StartProcessingCommand,
)
from linguary.application.use_cases.process_chapter import (
    ChapterProcessingSequence,
    ProcessChapterCommand,
)
from linguary.application.use_cases.reset_processing import (
    ResetProcessing,
    ResetProcessingCommand,
    ResetProcessingError,
)
from linguary.domain import (
    BookStatus,
    BookStatusTransitionError,
    ErrorType,
    LLMSettingsSnapshot,
    ProcessingTask,
    TaskError,
    TaskIdentity,
    TaskStatus,
    TaskType,
)

#: Две главы по два абзаца: книга, обработку которой можно довести до конца.
TWO_CHAPTERS = ((SOURCE_TEXT, SOURCE_TEXT), (SOURCE_TEXT, SOURCE_TEXT))


def make_sequence(
    unit_of_work: SqlAlchemyUnitOfWork,
    task_queue: SqliteTaskQueue,
    client: FakeLLMClient | InterruptingLLMClient,
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


async def process_first_chapter(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Обрабатывает первую главу книги: у запуска появляются результаты и заметки."""
    process, _ = make_process_chapter(
        unit_of_work,
        FakeLLMClient.always(translated_reply(notes=(note_reply("b1"),))),
    )
    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1"))


async def test_reset_removes_results_and_keeps_the_analysis(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Сброс удаляет запуск с результатами, сохраняя импорт, блоки и анализ глав."""
    scenario = store_processing_book(unit_of_work)
    await process_first_chapter(unit_of_work)

    result = ResetProcessing(unit_of_work=unit_of_work)(
        ResetProcessingCommand(book_id=BOOK_ID)
    )

    assert (result.run_id, result.removed_results) == (RUN_ID, 2)
    with unit_of_work.transaction() as uow:
        assert uow.runs.get(RUN_ID) is None
        assert uow.runs.active_for_book(BOOK_ID) is None
        assert uow.results.list_for_run(RUN_ID) == ()
        assert [item.section_id for item in uow.analyses.list_for_book(BOOK_ID)] == [
            "section-1"
        ]
        document = uow.documents.get(BOOK_ID)
    assert document is not None
    assert [block.id for block in document.blocks()] == list(scenario.block_ids())
    assert book_status(unit_of_work) is BookStatus.GLOSSARY_REVIEW


async def test_reset_removes_processing_tasks_and_keeps_the_rest(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Сброс снимает задачи обработки и повторной генерации, анализ глав не трогает."""
    store_processing_book(unit_of_work, chapters=TWO_CHAPTERS)
    await task_queue.enqueue(
        TaskIdentity(
            book_id=BOOK_ID, task_type=TaskType.ANALYZE_CHAPTER, scope_id="section-1"
        )
    )
    await task_queue.enqueue(
        TaskIdentity(
            book_id=BOOK_ID, task_type=TaskType.REGENERATE_BLOCK, scope_id="block-1"
        )
    )
    sequence = make_sequence(unit_of_work, task_queue, FakeLLMClient())
    await sequence.start(BOOK_ID)

    result = ResetProcessing(unit_of_work=unit_of_work)(
        ResetProcessingCommand(book_id=BOOK_ID)
    )

    assert result.removed_tasks == 3
    with unit_of_work.transaction() as uow:
        remaining = uow.tasks.list_for_book(BOOK_ID)
    assert [task.task_type for task in remaining] == [TaskType.ANALYZE_CHAPTER]


async def test_processed_book_can_be_processed_again_after_reset(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """После сброса обработка запускается заново и снова доходит до результатов.

    Идемпотентный ключ задачи живёт, пока живёт задача: успешная задача отменённого
    запуска не дала бы поставить такую же задачу второй раз (PRD § 18.1).
    """
    store_processing_book(unit_of_work, chapters=TWO_CHAPTERS)
    sequence = make_sequence(
        unit_of_work, task_queue, FakeLLMClient.always(translated_reply())
    )
    await sequence.start(BOOK_ID)
    await make_pool(task_queue, sequence).drain()
    ResetProcessing(unit_of_work=unit_of_work)(ResetProcessingCommand(book_id=BOOK_ID))

    run = StartProcessing(
        unit_of_work=unit_of_work,
        glossary_confidence_threshold=0.7,
        llm_settings_snapshot=LLMSettingsSnapshot(provider="fake", model="fake-model"),
        id_factory=lambda: "run-2",
    )(StartProcessingCommand(book_id=BOOK_ID))
    await sequence.start(BOOK_ID)
    finished = await make_pool(task_queue, sequence).drain()

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED] * 2
    assert book_status(unit_of_work) is BookStatus.READY_TO_EXPORT
    with unit_of_work.transaction() as uow:
        assert len(uow.results.list_for_run(run.id)) == 4


async def test_late_result_of_a_running_task_is_not_saved(
    unit_of_work: SqlAlchemyUnitOfWork, task_queue: SqliteTaskQueue
) -> None:
    """Сброс во время выполняющейся задачи: её поздний результат не сохраняется (ALG-04)."""
    store_processing_book(unit_of_work)
    reset = ResetProcessing(unit_of_work=unit_of_work)
    client = InterruptingLLMClient(
        inner=FakeLLMClient.always(translated_reply()),
        interrupt=lambda: reset(ResetProcessingCommand(book_id=BOOK_ID)),
    )
    sequence = make_sequence(unit_of_work, task_queue, client)
    await sequence.start(BOOK_ID)

    finished = await make_pool(task_queue, sequence).drain()

    assert [task.status for task in finished] == [TaskStatus.SUCCEEDED]
    assert book_status(unit_of_work) is BookStatus.GLOSSARY_REVIEW
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == ()
        assert uow.tasks.list_for_book(BOOK_ID) == ()


async def test_reset_of_a_book_without_processing_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Книга, обработка которой не начиналась, сбрасывать нечего (ALG-06)."""
    store_processing_book(unit_of_work, status=BookStatus.GLOSSARY_REVIEW)

    with pytest.raises(BookStatusTransitionError, match="недопустимо"):
        ResetProcessing(unit_of_work=unit_of_work)(
            ResetProcessingCommand(book_id=BOOK_ID)
        )

    with unit_of_work.transaction() as uow:
        assert uow.runs.get(RUN_ID) is not None


async def test_reset_of_an_unknown_book_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Сброс книги, которой нет в хранилище, отклоняется прикладной ошибкой."""
    with pytest.raises(ResetProcessingError, match="нет в хранилище"):
        ResetProcessing(unit_of_work=unit_of_work)(
            ResetProcessingCommand(book_id="book-404")
        )


async def test_failed_task_of_a_cancelled_run_is_removed_too(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Неуспешная задача отменённого запуска тоже снимается: повторять её нечего."""
    store_processing_book(unit_of_work)
    with unit_of_work.transaction() as uow:
        uow.tasks.add(
            ProcessingTask(
                id="task-failed",
                book_id=BOOK_ID,
                task_type=TaskType.PROCESS_TRANSLATION_UNIT,
                scope_id="section-1",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=TaskError(
                    type=ErrorType.TRANSIENT, message="Провайдер не ответил"
                ),
            )
        )

    result = ResetProcessing(unit_of_work=unit_of_work)(
        ResetProcessingCommand(book_id=BOOK_ID)
    )

    assert result.removed_tasks == 1
    with unit_of_work.transaction() as uow:
        assert uow.tasks.get("task-failed") is None
