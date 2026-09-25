"""Порядок задач анализа глав (P3.T3, PRD § 13, § 25)."""

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
    """Клиент, отдающий управление циклу событий внутри вызова.

    Без этого весь вызов модели проходит без единой точки переключения, и параллельные
    воркеры никогда не пересекаются — проверка порядка стала бы бессмысленной.
    """

    def __init__(self, inner: FakeLLMClient) -> None:
        self._inner = inner

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        await asyncio.sleep(0)
        return await self._inner.generate(request)


def make_book(chapters: int = CHAPTER_COUNT) -> FakeUnitOfWork:
    """Книга из нескольких глав по одному абзацу в каждой."""
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
    """Последовательность анализа глав поверх двойников хранилища и очереди."""
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    return ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # двойник порта хранения
        queue=queue,  # type: ignore[arg-type]  # двойник порта очереди
        analyze_chapter=analyze_chapter,
    )


async def drain(runner: TaskRunner, queue: FakeTaskQueue) -> None:
    """Выполняет очередь несколькими одновременными воркерами."""

    async def worker() -> None:
        while queue.has_work:
            if await runner.run_next() is None:
                await asyncio.sleep(0)

    await asyncio.gather(*(worker() for _ in range(WORKER_COUNT)))


async def test_analysis_tasks_run_in_reading_order_one_at_a_time() -> None:
    """Главы анализируются по одной и строго в порядке чтения книги."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    sequence = make_sequence(unit_of_work, queue, YieldingLLMClient(client))
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]  # двойник порта очереди
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )

    await sequence.start(BOOK_ID)
    await drain(runner, queue)

    expected = [f"section-{number}" for number in range(1, CHAPTER_COUNT + 1)]
    assert queue.claimed_scope_ids == expected
    assert unit_of_work.analyses.saved_section_ids == expected
    assert queue.max_running == 1
    assert [
        f"Chapter {number}" in request.input_text
        for number, request in enumerate(client.requests, start=1)
    ] == [True] * CHAPTER_COUNT


async def test_start_moves_book_to_analyzing_and_queues_first_chapter() -> None:
    """Запуск анализа переводит книгу в `ANALYZING` и ставит первую главу (ALG-06)."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    sequence = make_sequence(
        unit_of_work, queue, FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    )

    task = await sequence.start(BOOK_ID)

    assert task is not None
    assert (task.task_type, task.scope_id) == (TaskType.ANALYZE_CHAPTER, "section-1")
    assert queue.tasks == [task]
    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.ANALYZING


async def test_completed_analysis_moves_book_to_glossary_review() -> None:
    """После последней главы книга уходит на проверку глоссария (ALG-06)."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    sequence = make_sequence(unit_of_work, queue, client)
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]  # двойник порта очереди
        handlers={TaskType.ANALYZE_CHAPTER: sequence.run_task},
    )

    await sequence.start(BOOK_ID)
    await drain(runner, queue)

    stored = unit_of_work.books.get(BOOK_ID)
    assert stored is not None and stored.status is BookStatus.GLOSSARY_REVIEW
    assert len(queue.tasks) == CHAPTER_COUNT


async def test_start_resumes_from_first_unanalysed_chapter() -> None:
    """Повторный запуск продолжает с первой непроанализированной главы (PRD § 18)."""
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    sequence = make_sequence(unit_of_work, queue, client)
    await sequence.start(BOOK_ID)
    first = queue.tasks[0]
    await sequence.run_task(first.start())

    resumed = await sequence.start(BOOK_ID)

    assert resumed is not None and resumed.scope_id == "section-2"
    assert client.call_count == 1


async def test_foreign_task_type_is_rejected() -> None:
    """Обработчик анализа не берётся выполнять задачу чужого типа."""
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
    """Книга, которая ещё импортируется, к анализу не допускается (ALG-06)."""
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
    """Анализ книги, которой нет в хранилище, не запускается."""
    sequence = make_sequence(
        FakeUnitOfWork(), FakeTaskQueue(), FakeLLMClient.always(analysis_reply())
    )

    with pytest.raises(AnalyzeChapterError):
        await sequence.start("unknown-book")


async def test_repeated_start_retries_analysis_stopped_by_llm_configuration() -> None:
    """Анализ, упавший на ключе провайдера, продолжается повторным запуском (QA-01).

    Первая глава падает с `LLMConfigurationError`: остальные главы в очередь не ставятся,
    и книга остаётся в `ANALYZING`. Пользователь исправляет ключ и снова запускает анализ —
    та же задача возвращается в очередь, и книга доходит до проверки глоссария без
    повторной загрузки файла.
    """
    unit_of_work = make_book()
    queue = FakeTaskQueue()
    client = FakeLLMClient(
        replies=(LLMConfigurationError("провайдер не принял ключ: 401"),),
        default=analysis_reply(aliases=("b1",)),
    )
    sequence = make_sequence(unit_of_work, queue, client)
    runner = TaskRunner(
        queue=queue,  # type: ignore[arg-type]  # двойник порта очереди
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
    """Повторный запуск не возвращает в очередь задачу с постоянной ошибкой."""
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
