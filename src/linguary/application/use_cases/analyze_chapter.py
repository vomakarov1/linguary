"""Сценарий анализа главы и последовательность задач анализа (PRD § 13-14, ADR § 3).

`AnalyzeChapter` выполняет один запрос к модели на главу и сохраняет `ChapterAnalysis`:
кандидатов глоссария, инструкции перевода главы, план обработки блоков, `Before You Read`
и краткую сводку событий главы. Отдельных подготовительных запросов — на глоссарий,
на план, на сводку — не выполняется, и отдельного прохода ради пересказа предыдущей главы
тоже нет: раздел `Previously in Chapter ...` собирает экспортёр из сводки главы `N-1`
(PRD § 19.2).

Глава, которая в один запрос не помещается, — резервный сценарий (PRD § 14): она уходит
в модель последовательными фрагментами, и её анализ собирается из ответов фрагментов
(`analysis_merge`, ALG-03). Порог дробления — настройка, а не свойство сценария: он зависит
от предела вывода конкретной модели.

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
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

from linguary.application.analysis_merge import (
    DEFAULT_MAX_FRAGMENT_BLOCKS,
    ChapterFragment,
    FragmentAnalysis,
    before_you_read_entries,
    merge_chapter_analysis,
    plan_fragments,
)
from linguary.application.block_labels import without_block_labels
from linguary.application.context_builder import (
    ChapterAnalysisContext,
    ContextBuilder,
    FragmentFindings,
    textual_blocks,
)
from linguary.application.errors import ApplicationError
from linguary.application.glossary_review import candidates_with_decisions
from linguary.application.llm_dto import ChapterAnalysisResponse
from linguary.application.llm_rejections import record_rejection
from linguary.application.ports.llm import LLMClient, LLMRequest
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators.mixed_script import mixed_script_word
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
    TaskStatus,
    TaskType,
    TokenUsage,
    LLMCallStage,
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
    task_id: str | None = None


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
        max_fragment_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS,
        context_builder: ContextBuilder | None = None,
        id_factory: IdFactory | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._retry_policy = retry_policy
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._max_fragment_blocks = max_fragment_blocks
        self._context_builder = context_builder or ContextBuilder()
        self._id_factory = id_factory or _new_id
        self._sleep = sleep or asyncio.sleep

    async def __call__(self, command: AnalyzeChapterCommand) -> AnalyzeChapterResult:
        """Анализирует главу и сохраняет результат.

        Глава, которая помещается в один запрос, им и анализируется. Более длинная
        уходит в модель последовательными фрагментами, и её анализ собирается из их
        ответов (PRD § 14, ALG-03).
        """
        document, accumulated = self._load(command)
        section = _section_of(document, command)
        blocks = textual_blocks(section)
        if not blocks:
            analysis = _empty_analysis(section, id=self._id_factory())
            self._save(command.book_id, analysis)
            return AnalyzeChapterResult(analysis=analysis)
        fragments = plan_fragments(blocks, max_blocks=self._max_fragment_blocks)
        if len(fragments) == 1:
            analysis, attempts, usage = await self._analyze_whole_chapter(
                document=document,
                section=section,
                blocks=blocks,
                accumulated=accumulated,
                command=command,
            )
        else:
            analysis, attempts, usage = await self._analyze_by_fragments(
                document=document,
                section=section,
                blocks=blocks,
                fragments=fragments,
                accumulated=accumulated,
                command=command,
            )
        self._save(command.book_id, analysis)
        return AnalyzeChapterResult(analysis=analysis, attempts=attempts, usage=usage)

    async def _analyze_whole_chapter(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        accumulated: Sequence[GlossaryCandidate],
        command: AnalyzeChapterCommand,
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        """Основной сценарий: вся глава одним запросом (PRD § 14)."""
        context = self._context_builder.for_chapter_analysis(
            document=document, section=section, accumulated_candidates=accumulated
        )
        context = replace(
            context,
            request=context.request.with_logging(
                book_id=command.book_id,
                stage=LLMCallStage.CHAPTER_ANALYSIS,
                task_id=command.task_id,
            ),
        )
        payload, attempts, usage = await self._request_analysis(context)
        analysis = self._to_analysis(
            payload, context=context, section=section, blocks=blocks
        )
        return analysis, attempts, usage

    async def _analyze_by_fragments(
        self,
        *,
        document: CanonicalDocument,
        section: DocumentSection,
        blocks: Sequence[DocumentBlock],
        fragments: Sequence[ChapterFragment],
        accumulated: Sequence[GlossaryCandidate],
        command: AnalyzeChapterCommand,
    ) -> tuple[ChapterAnalysis, int, TokenUsage | None]:
        """Резервный сценарий: длинная глава по фрагментам (PRD § 14, ALG-03).

        Фрагменты запрашиваются подряд, каждый со своими блоками и своим циклом повторов.
        Фрагмент, не выполненный за отведённые попытки, поднимает исходную ошибку: анализ
        главы не сохраняется вовсе, а не сохраняется неполным, иначе неполнота досталась
        бы пользователю как готовый результат (ALG-03).
        """
        analysed: list[FragmentAnalysis] = []
        attempts = 0
        usages: list[TokenUsage | None] = []
        for fragment in fragments:
            previous = None
            if analysed:
                previous = FragmentFindings(
                    summary=next(
                        (
                            part.chapter_summary
                            for part in reversed(analysed)
                            if part.chapter_summary
                        ),
                        "",
                    ),
                    instructions="\n".join(
                        part.translation_instructions
                        for part in analysed
                        if part.translation_instructions
                    ),
                    expressions=tuple(
                        entry.expression
                        for part in analysed
                        for entry in part.before_you_read
                    ),
                )
            context = self._context_builder.for_chapter_fragment(
                document=document,
                section=section,
                fragment=fragment,
                fragment_count=len(fragments),
                accumulated_candidates=accumulated,
                findings=previous,
            )
            context = replace(
                context,
                request=context.request.with_logging(
                    book_id=command.book_id,
                    stage=LLMCallStage.CHAPTER_ANALYSIS,
                    task_id=command.task_id,
                ),
            )
            payload, fragment_attempts, usage = await self._request_analysis(context)
            attempts += fragment_attempts
            usages.append(usage)
            analysed.append(
                FragmentAnalysis.of_response(
                    payload,
                    order=fragment.order,
                    block_ids_by_alias=context.block_ids_by_alias,
                )
            )
        analysis = merge_chapter_analysis(
            analysed,
            id=self._id_factory(),
            section_id=section.id,
            expected_block_ids=[block.id for block in blocks],
            glossary_confidence_threshold=self._glossary_confidence_threshold,
            id_factory=self._id_factory,
        )
        return analysis, attempts, _total_usage(usages)

    def _load(
        self, command: AnalyzeChapterCommand
    ) -> tuple[CanonicalDocument, tuple[GlossaryCandidate, ...]]:
        """Читает книгу и накопленные решения глоссария книги.

        Решение пользователя по кандидату заменяет вариант модели: анализ повторяет
        накопленные решения дословно и не должен возвращать исправленный вариант.

        Кандидаты предыдущих глав идут все, с вариантом модели там, где решения нет.
        Кандидаты самой главы и следующих глав — только с решением пользователя:
        решение действует на всю книгу, и повторный анализ главы после проверки
        глоссария обязан его видеть (PRD § 11), а вариант модели следующей главы
        этой главе ничего не даёт. Порядок чтения сохраняет сам `list_for_book`.
        """
        with self._unit_of_work.transaction() as transaction:
            document = transaction.documents.get(command.book_id)
            if document is None:
                raise AnalyzeChapterError(f"книги {command.book_id} нет в хранилище")
            current_order = _section_of(document, command).order
            orders = {section.id: section.order for section in document.sections}
            analyses = transaction.analyses.list_for_book(command.book_id)
            accumulated = candidates_with_decisions(
                (
                    candidate
                    for analysis in analyses
                    if orders.get(analysis.section_id, current_order) < current_order
                    for candidate in analysis.glossary_candidates
                ),
                transaction.glossary.list_for_book(command.book_id),
                language=document.target_language,
                decided_only=(
                    candidate
                    for analysis in analyses
                    if orders.get(analysis.section_id, current_order) >= current_order
                    for candidate in analysis.glossary_candidates
                ),
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
            response = None
            try:
                response = await self._llm_client.generate(request)
                _ensure_plan_covers_chapter(response.payload, context)
                _ensure_candidates_single_script(response.payload, context)
            except Exception as error:
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
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
            translation_instructions=without_block_labels(payload.translation_instructions),
            chapter_summary=without_block_labels(payload.chapter_summary),
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
                note=candidate.note,
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

        Если анализ этой главы уже падал с ошибкой, которую имеет смысл повторить
        (например, провайдер не принял ключ, а пользователь его исправил), повторный
        запуск возвращает её задачу в очередь: иначе книга осталась бы в `ANALYZING`
        без единой задачи в работе. Задача с постоянной ошибкой остаётся как есть.
        """
        self._begin_analysis(book_id)
        task = await self._enqueue_next(book_id)
        if task is not None and task.status is TaskStatus.FAILED and task.is_retryable:
            task = task.retry()
            await self._queue.save(task)
        return task

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
            AnalyzeChapterCommand(
                book_id=task.book_id,
                section_id=task.scope_id,
                task_id=task.id,
            )
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
        missing_block_ids = [context.block_id_for(alias) for alias in missing]
        raise ResultValidationError(
            f"план обработки главы {context.section_id} не покрывает блоки: "
            f"{missing_block_ids}",
            expectation=context.expectation,
        )
    duplicated = sorted({block for block in returned if returned.count(block) > 1})
    if duplicated:
        raise ResultValidationError(
            f"блоки главы {context.section_id} получили по два режима обработки: "
            f"{duplicated}",
            expectation=context.expectation,
        )


def _ensure_candidates_single_script(
    payload: ChapterAnalysisResponse, context: ChapterAnalysisContext
) -> None:
    """Вариант перевода кандидата глоссария не содержит слов из смеси алфавитов (ALG-04).

    Вариант глоссария уходит во все главы книги, поэтому слово вроде «PГЛАФ» отбраковывается
    здесь же, до сохранения анализа, а не всплывает потом в каждом блоке с этим термином.
    """
    for candidate in payload.glossary_candidates:
        word = mixed_script_word(candidate.suggested_translation)
        if word is not None:
            raise ResultValidationError(
                f"вариант перевода кандидата глоссария {candidate.source_text!r} "
                f"содержит слово из смеси латиницы и кириллицы: {word!r}. Каждое слово "
                f"пиши одним алфавитом",
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
    return before_you_read_entries(payload.before_you_read)


def _total_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    """Суммарный расход токенов главы: провайдер считает каждый вызов отдельно.

    Провайдер, не сообщающий объём токенов, оставляет `None` — суммой это не подменяется:
    ноль означал бы бесплатный вызов, а не отсутствие данных (PRD § 24).
    """
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        prompt_tokens=_total_tokens(usage.prompt_tokens for usage in reported),
        completion_tokens=_total_tokens(usage.completion_tokens for usage in reported),
    )


def _total_tokens(values: Iterable[int | None]) -> int | None:
    """Сумма тех значений, которые провайдер сообщил."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


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
