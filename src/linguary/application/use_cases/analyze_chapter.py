"""Сценарий анализа главы и последовательность задач анализа (PRD § 13-14, ADR § 3).

`AnalyzeChapter` выполняет один запрос к модели на главу и сохраняет `ChapterAnalysis`:
кандидатов глоссария, инструкции перевода главы, план обработки блоков, `Before You Read`
и краткую сводку событий главы. Отдельных подготовительных запросов — на глоссарий,
на план, на сводку — не выполняется, и отдельного прохода ради пересказа предыдущей главы
тоже нет: раздел `Previously in Chapter ...` собирает экспортёр из сводки главы `N-1`
(PRD § 19.2).

Вызов модели выполняется вне транзакции, а результат сохраняется одной короткой
транзакцией (ADR § 3). Повторы ведёт `RetryPolicy` (ALG-05): цикл здесь выдерживает
паузу перед повтором временной ошибки и дописывает уточняющую инструкцию к исходному
запросу, не подменяя собранный контекст.

`ChapterAnalysisSequence` отвечает за порядок: главы анализируются строго последовательно
в порядке чтения, потому что контекст главы включает кандидатов предыдущих глав (PRD § 25).
Порядок обеспечивается не блокировкой воркера, а очередью: у книги одновременно существует
не более одной задачи `ANALYZE_CHAPTER`, и следующая ставится только после сохранения
предыдущего анализа. Настройка параллелизма воркера при этом остаётся общей и на анализ
не влияет: параллельно могут идти только задачи разных книг.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

from linguary.application.context_builder import (
    ChapterAnalysisContext,
    ContextBuilder,
    textual_blocks,
)
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import ChapterAnalysisResponse
from linguary.application.ports.llm import LLMClient, LLMRequest
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.task_runner import TaskQueue
from linguary.domain import (
    BeforeYouReadEntry,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    ProcessingTask,
    TaskIdentity,
    TaskType,
    TokenUsage,
    TreatmentPlan,
    apply_book_event,
)

#: Пауза между попытками вызова модели. Прикладной слой циклом событий не владеет,
#: но выдержать паузу обязан он: политика повторов её только вычисляет (ALG-05).
type Sleep = Callable[[float], Awaitable[None]]

#: Фабрика идентификаторов сохраняемых записей анализа.
type IdFactory = Callable[[], str]


class AnalyzeChapterError(ApplicationError):
    """Главу нельзя проанализировать в её текущем состоянии."""


@dataclass(frozen=True, slots=True)
class AnalyzeChapterCommand:
    """Команда анализа: какую главу какой книги анализировать."""

    book_id: str
    section_id: str


@dataclass(frozen=True, slots=True)
class AnalyzeChapterResult:
    """Результат анализа главы вместе с техническими данными вызова.

    `attempts` и `usage` нужны для ориентиров жизнеспособности (PRD § 29.1): сколько
    обращений к модели стоила глава и какой объём токенов провайдер за них насчитал.
    Пустая глава анализа не требует, и вызова модели у неё нет: `attempts` равен нулю.
    """

    analysis: ChapterAnalysis
    attempts: int = 0
    usage: TokenUsage | None = None


class AnalyzeChapter:
    """Анализирует одну главу и сохраняет её `ChapterAnalysis`."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        glossary_confidence_threshold: float,
        context_builder: ContextBuilder | None = None,
        id_factory: IdFactory | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._retry_policy = retry_policy
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id
        self._sleep = sleep or asyncio.sleep

    async def __call__(self, command: AnalyzeChapterCommand) -> AnalyzeChapterResult:
        """Анализирует главу и сохраняет результат."""
        document, accumulated = self._load(command)
        section = _section_of(document, command)
        blocks = textual_blocks(section)
        if not blocks:
            analysis = _empty_analysis(section, id=self._id_factory())
            self._save(command.book_id, analysis)
            return AnalyzeChapterResult(analysis=analysis)
        context = self._context_builder.for_chapter_analysis(
            document=document, section=section, accumulated_candidates=accumulated
        )
        payload, attempts, usage = await self._request_analysis(context)
        analysis = self._to_analysis(
            payload, context=context, section=section, blocks=blocks
        )
        self._save(command.book_id, analysis)
        return AnalyzeChapterResult(analysis=analysis, attempts=attempts, usage=usage)

    def _load(
        self, command: AnalyzeChapterCommand
    ) -> tuple[CanonicalDocument, tuple[GlossaryCandidate, ...]]:
        """Читает книгу и кандидатов глоссария уже проанализированных глав."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {command.book_id} нет в хранилище")
            accumulated = tuple(
                candidate
                for analysis in transaction.analyses.list_for_book(command.book_id)
                if analysis.section_id != command.section_id
                for candidate in analysis.glossary_candidates
            )
        return document, accumulated

    async def _request_analysis(
        self, context: ChapterAnalysisContext
    ) -> tuple[ChapterAnalysisResponse, int, TokenUsage | None]:
        """Выполняет вызов модели с повторами по политике (ALG-05).

        Исчерпав попытки, сценарий поднимает исходную ошибку: типизировать её и перевести
        задачу в `failed` — работа runner-а, а не этого цикла.
        """
        request: LLMRequest[ChapterAnalysisResponse] = context.request
        attempts = 0
        while True:
            attempts += 1
            try:
                response = await self._llm_client.generate(request)
                _ensure_plan_covers_chapter(response.payload, context)
            except Exception as error:
                decision = self._retry_policy.decide(error, attempts=attempts)
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds > 0:
                    await self._sleep(decision.backoff_seconds)
                request = decision.next_request(request)
                continue
            return response.payload, attempts, response.usage

    def _to_analysis(
        self,
        payload: ChapterAnalysisResponse,
        *,
        context: ChapterAnalysisContext,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
    ) -> ChapterAnalysis:
        """Переводит ответ модели в доменный анализ главы.

        Псевдонимы запроса возвращаются к стабильным идентификаторам блоков, а план
        собирается в порядке исходного документа: ответ модели о порядке блоков главы
        не решает.
        """
        analysis_id = self._id_factory()
        treatments = {
            item.source_block_id: item.treatment for item in payload.treatment_plan
        }
        return ChapterAnalysis.create(
            id=analysis_id,
            section_id=section.id,
            treatment_plan=TreatmentPlan.of_treatments(
                section.id,
                (
                    (context.block_id_for(alias), treatments[alias])
                    for alias in context.aliases
                ),
            ),
            translation_instructions=payload.translation_instructions,
            chapter_summary=payload.chapter_summary,
            expected_block_ids=[block.id for block in blocks],
            glossary_candidates=self._candidates(payload, analysis_id=analysis_id),
            before_you_read=_before_you_read(payload),
        )

    def _candidates(
        self, payload: ChapterAnalysisResponse, *, analysis_id: str
    ) -> tuple[GlossaryCandidate, ...]:
        """Кандидаты глоссария главы с вычисленным `requires_review` (DTO-01).

        Пометка модели остаётся входным сигналом: требует ли кандидат внимания
        пользователя, решает приложение по правилам PRD § 12.
        """
        return tuple(
            GlossaryCandidate.create(
                id=self._id_factory(),
                analysis_id=analysis_id,
                source_text=candidate.source_text,
                type=candidate.type,
                suggested_translation=candidate.suggested_translation,
                confidence=candidate.confidence,
                model_flagged_review=candidate.model_flagged_review,
                confidence_threshold=self._glossary_confidence_threshold,
            )
            for candidate in payload.glossary_candidates
        )

    def _save(self, book_id: str, analysis: ChapterAnalysis) -> None:
        """Сохраняет анализ главы одной короткой транзакцией."""
        with self._unit_of_work.transaction() as transaction:
            transaction.analyses.save(book_id, analysis)


class ChapterAnalysisSequence:
    """Ставит задачи анализа глав строго по одной, в порядке чтения (PRD § 13, § 25)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        analyze_chapter: AnalyzeChapter,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._analyze_chapter = analyze_chapter

    async def start(self, book_id: str) -> ProcessingTask | None:
        """Запускает анализ книги: переводит её в `ANALYZING` и ставит первую главу.

        Возвращает `None`, если анализировать больше нечего: книга без глав или уже
        полностью проанализированная. Повторный запуск после перезапуска приложения
        продолжает с первой непроанализированной главы.
        """
        self._begin_analysis(book_id)
        return await self._enqueue_next(book_id)

    async def run_task(self, task: ProcessingTask) -> None:
        """Обработчик задачи `ANALYZE_CHAPTER` для локального runner-а (ALG-02).

        Следующая глава ставится в очередь только после сохранения анализа текущей:
        так набор накопленных кандидатов детерминирован и не зависит от параллелизма.
        """
        if task.task_type is not TaskType.ANALYZE_CHAPTER:
            raise AnalyzeChapterError(
                f"задача {task.id} типа {task.task_type} не является анализом главы"
            )
        await self._analyze_chapter(
            AnalyzeChapterCommand(book_id=task.book_id, section_id=task.scope_id)
        )
        await self._enqueue_next(task.book_id)

    async def _enqueue_next(self, book_id: str) -> ProcessingTask | None:
        """Ставит задачу следующей непроанализированной главы либо завершает анализ."""
        section_id = self._next_section_id(book_id)
        if section_id is None:
            self._complete_analysis(book_id)
            return None
        return await self._queue.enqueue(
            TaskIdentity(
                book_id=book_id,
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id=section_id,
            )
        )

    def _next_section_id(self, book_id: str) -> str | None:
        """Первая глава книги в порядке чтения, для которой анализа ещё нет."""
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
            analyzed = {
                analysis.section_id
                for analysis in transaction.analyses.list_for_book(book_id)
            }
        return next(
            (
                section.id
                for section in document.ordered_sections
                if section.id not in analyzed
            ),
            None,
        )

    def _begin_analysis(self, book_id: str) -> None:
        """Переводит книгу в `ANALYZING` (ALG-06).

        Продолжение уже идущего анализа состояние не меняет. Книга, которая до анализа
        ещё не дошла или уже ушла дальше, получает отказ от домена: переходы вне таблицы
        ALG-06 запрещены, и в Web UI такой отказ становится `409`.
        """
        with self._unit_of_work.transaction() as transaction:
            book = self._book(transaction, book_id)
            if book.status is BookStatus.ANALYZING:
                return
            transaction.books.save(
                _with_status(
                    book, apply_book_event(book.status, BookEvent.ANALYSIS_STARTED)
                )
            )

    def _complete_analysis(self, book_id: str) -> None:
        """Переводит проанализированную книгу к проверке глоссария (ALG-06).

        Книга, которая уже прошла проверку глоссария, остаётся как есть: повторный
        анализ отдельной главы не отматывает её назад по процессу.
        """
        with self._unit_of_work.transaction() as transaction:
            book = self._book(transaction, book_id)
            if book.status is not BookStatus.ANALYZING:
                return
            transaction.books.save(
                _with_status(
                    book, apply_book_event(book.status, BookEvent.ANALYSIS_COMPLETED)
                )
            )

    def _book(self, transaction: UnitOfWork, book_id: str) -> BookRecord:
        """Запись книги в открытой транзакции."""
        book = transaction.books.get(book_id)
        if book is None:
            raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
        return book


def _section_of(
    document: CanonicalDocument, command: AnalyzeChapterCommand
) -> DocumentSection:
    """Глава книги, которую требует команда."""
    for section in document.sections:
        if section.id == command.section_id:
            return section
    raise AnalyzeChapterError(
        f"главы {command.section_id} нет в книге {command.book_id}"
    )


def _ensure_plan_covers_chapter(
    payload: ChapterAnalysisResponse, context: ChapterAnalysisContext
) -> None:
    """Блокирующая проверка плана обработки (PRD § 15).

    План обязан покрыть каждый переданный блок ровно один раз и не содержать чужих.
    Порядок ответа не проверяется: план сохраняется в порядке исходного документа,
    и перестановка строк ответа результату не вредит.
    """
    expected = context.expectation.block_ids
    returned = payload.planned_block_ids
    unknown = sorted(set(returned) - set(expected))
    if unknown:
        raise ResultValidationError(
            f"план обработки главы {context.section_id} содержит чужие блоки: {unknown}",
            expectation=context.expectation,
        )
    missing = sorted(set(expected) - set(returned))
    if missing:
        raise ResultValidationError(
            f"план обработки главы {context.section_id} не покрывает блоки: {missing}",
            expectation=context.expectation,
        )
    duplicated = sorted({block for block in returned if returned.count(block) > 1})
    if duplicated:
        raise ResultValidationError(
            f"блоки главы {context.section_id} получили по два режима обработки: "
            f"{duplicated}",
            expectation=context.expectation,
        )


def _empty_analysis(section: DocumentSection, *, id: str) -> ChapterAnalysis:
    """Полный анализ главы без текста: обложка, вклейка, страница с иллюстрацией.

    Передавать модели нечего, а глава должна быть проанализирована: иначе книга никогда
    не дойдёт до проверки глоссария. План такой главы пуст и покрывает все её текстовые
    блоки — их нет.
    """
    return ChapterAnalysis.create(
        id=id,
        section_id=section.id,
        treatment_plan=TreatmentPlan(section_id=section.id),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(),
    )


def _before_you_read(
    payload: ChapterAnalysisResponse,
) -> tuple[BeforeYouReadEntry, ...]:
    """Материал `Before You Read` главы (PRD § 19.1)."""
    return tuple(
        BeforeYouReadEntry(
            expression=entry.expression, explanation=entry.explanation
        )
        for entry in payload.before_you_read
    )


def _with_status(book: BookRecord, status: BookStatus) -> BookRecord:
    """Запись книги с новым состоянием."""
    return replace(book, status=status)


def _new_id() -> str:
    """Идентификатор новой записи анализа."""
    return str(uuid4())


__all__ = [
    "AnalyzeChapter",
    "AnalyzeChapterCommand",
    "AnalyzeChapterError",
    "AnalyzeChapterResult",
    "ChapterAnalysisSequence",
    "IdFactory",
    "Sleep",
]
