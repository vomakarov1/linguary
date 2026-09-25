"""Чтение состояния книги для интерфейса: карточка, прогресс, план и предупреждения.

Web UI и CLI показывают одно и то же (PRD § 18, § 21.1, § 21.4): сколько блоков обработано,
как распределились режимы, какая глава идёт сейчас, какие задачи неуспешны и что импорт
и обработка успели сказать о книге. Считается это по задачам, сохранённому плану обработки
и результатам блоков — то есть по хранилищу, а не по ходу выполнения.

Модуль читает через порты и ничего не меняет. Он живёт в прикладном слое, а не в routes:
подсчёт прогресса — прикладное правило, и повторять его в каждом представлении нельзя.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BookImportWarning,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    CoverageStatus,
    DocumentBlock,
    DocumentSection,
    ErrorType,
    ProcessingTask,
    ResultWarning,
    TaskStatus,
    TaskType,
)


class BookNotFoundError(ApplicationError):
    """Книги с таким идентификатором в хранилище нет."""


class ProcessingPlanUnavailableError(ApplicationError):
    """План обработки ещё не готов: ни одна глава книги не проанализирована."""


class ResultBlockNotFoundError(ApplicationError):
    """Запрошенного блока нет в каноническом документе книги."""


#: Типы задач, чья глава показывается как текущая (PRD § 21.4). Импорт и экспорт —
#: задачи уровня книги, отдельной главы у них нет.
_CHAPTER_TASK_TYPES = (TaskType.ANALYZE_CHAPTER, TaskType.PROCESS_TRANSLATION_UNIT)


@dataclass(frozen=True, slots=True)
class TaskFailure:
    """Неуспешная задача книги вместе с действием, доступным пользователю (PRD § 16)."""

    task_id: str
    task_type: TaskType
    scope_id: str
    error_type: ErrorType
    message: str
    detail: str | None
    attempts: int
    is_retryable: bool


@dataclass(frozen=True, slots=True)
class BookProgress:
    """Прогресс обработки книги в терминах экрана обработки (PRD § 21.4)."""

    total_sections: int = 0
    analyzed_sections: int = 0
    planned_blocks: int = 0
    processed_blocks: int = 0
    treatment_counts: Mapping[BlockTreatment, int] = field(default_factory=dict)
    current_chapter: int | None = None
    current_chapter_title: str | None = None
    failures: tuple[TaskFailure, ...] = ()
    pending_tasks: int = 0
    running_tasks: int = 0

    @property
    def percent_complete(self) -> int:
        """Доля обработанных блоков плана, целыми процентами.

        Книга без плана считается необработанной, а не завершённой: план появляется
        при анализе, и до него обрабатывать нечего.
        """
        if self.planned_blocks <= 0:
            return 0
        return round(100 * self.processed_blocks / self.planned_blocks)


@dataclass(frozen=True, slots=True)
class BookOverview:
    """Карточка книги: состояние, прогресс, счётчики режимов и ошибки (API-01)."""

    book: BookRecord
    progress: BookProgress
    warning_count: int = 0


@dataclass(frozen=True, slots=True)
class PlanBlockSummary:
    """Один блок главы с выбранным режимом обработки."""

    block: DocumentBlock
    treatment: BlockTreatment | None


@dataclass(frozen=True, slots=True)
class ChapterPlanSummary:
    """Сводка плана обработки одной главы (PRD § 21, UX notes)."""

    section_id: str
    chapter_number: int
    title: str
    coverage_status: CoverageStatus
    textual_blocks: int
    treatment_counts: Mapping[BlockTreatment, int]
    blocks: tuple[PlanBlockSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessingPlanSummary:
    """Диагностическая сводка плана обработки книги по главам."""

    chapters: tuple[ChapterPlanSummary, ...]
    treatment_counts: Mapping[BlockTreatment, int]
    analyzed_sections: int
    total_sections: int

    @property
    def is_complete(self) -> bool:
        """Проанализированы ли все главы книги и полон ли план каждой из них."""
        return self.analyzed_sections == self.total_sections and all(
            chapter.coverage_status is CoverageStatus.COMPLETE
            for chapter in self.chapters
        )


@dataclass(frozen=True, slots=True)
class ResultWarningEntry:
    """Замечание к результату обработки вместе с его местом в книге (PRD § 15)."""

    warning: ResultWarning
    block_id: str
    section_id: str | None
    chapter_number: int | None


@dataclass(frozen=True, slots=True)
class BookWarnings:
    """Предупреждения импорта и обработки книги (PRD § 21.6)."""

    import_warnings: tuple[BookImportWarning, ...] = ()
    result_warnings: tuple[ResultWarningEntry, ...] = ()

    @property
    def total(self) -> int:
        """Сколько всего предупреждений накопила книга."""
        return len(self.import_warnings) + len(self.result_warnings)


@dataclass(frozen=True, slots=True)
class ResultBlockSummary:
    """Исходный блок, запланированный режим и сохранённый результат обработки."""

    block: DocumentBlock
    chapter_number: int
    treatment: BlockTreatment | None = None
    result: BlockProcessingResult | None = None


@dataclass(frozen=True, slots=True)
class ResultChapterSummary:
    """Глава результата с блоками в порядке чтения."""

    section_id: str
    chapter_number: int
    title: str
    blocks: tuple[ResultBlockSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class BookResultSummary:
    """Результат книги по главам вместе с предупреждениями и состоянием экспорта."""

    book: BookRecord
    chapters: tuple[ResultChapterSummary, ...] = ()
    warnings: BookWarnings = field(default_factory=BookWarnings)


def list_books(
    unit_of_work: UnitOfWork, *, status: BookStatus | None = None
) -> tuple[BookRecord, ...]:
    """Книги в порядке добавления, при необходимости отфильтрованные по состоянию."""
    with unit_of_work.transaction() as transaction:
        books = transaction.books.list_all()
    if status is None:
        return books
    return tuple(book for book in books if book.status is status)


def book_overview(unit_of_work: UnitOfWork, book_id: str) -> BookOverview:
    """Карточка книги: состояние, прогресс, счётчики режимов и неуспешные задачи."""
    with unit_of_work.transaction() as transaction:
        book = _book(transaction, book_id)
        document = transaction.documents.get(book_id)
        analyses = transaction.analyses.list_for_book(book_id)
        tasks = transaction.tasks.list_for_book(book_id)
        run = transaction.runs.active_for_book(book_id)
        processed_blocks = (
            transaction.results.count_for_run(run.id) if run is not None else 0
        )
        warning_count = len(transaction.documents.import_warnings(book_id))
    return BookOverview(
        book=book,
        progress=_progress(
            document=document,
            analyses=analyses,
            tasks=tasks,
            processed_blocks=processed_blocks,
        ),
        warning_count=warning_count,
    )


def processing_plan(unit_of_work: UnitOfWork, book_id: str) -> ProcessingPlanSummary:
    """Сводка плана обработки по главам.

    План — результат анализа, поэтому до анализа его не существует: сводка отвергается,
    а не отдаётся пустой. Пустая сводка выглядела бы как книга без единого текстового
    блока (в Web UI это `409`).
    """
    with unit_of_work.transaction() as transaction:
        _book(transaction, book_id)
        document = transaction.documents.get(book_id)
        analyses = transaction.analyses.list_for_book(book_id)
    if not analyses:
        raise ProcessingPlanUnavailableError(
            f"план обработки книги {book_id} ещё не готов: главы не проанализированы"
        )
    sections = document.ordered_sections if document is not None else ()
    chapters = tuple(
        _chapter_plan(section=section, number=number, analysis=analysis)
        for number, section, analysis in _analysed_sections(sections, analyses)
    )
    return ProcessingPlanSummary(
        chapters=chapters,
        treatment_counts=_merge_counts(
            chapter.treatment_counts for chapter in chapters
        ),
        analyzed_sections=len(chapters),
        total_sections=len(sections),
    )


def book_warnings(unit_of_work: UnitOfWork, book_id: str) -> BookWarnings:
    """Предупреждения импорта и замечания к результатам обработки книги."""
    with unit_of_work.transaction() as transaction:
        _book(transaction, book_id)
        import_warnings = transaction.documents.import_warnings(book_id)
        document = transaction.documents.get(book_id)
        run = transaction.runs.active_for_book(book_id)
        results = transaction.results.list_for_run(run.id) if run is not None else ()
    return _warnings_of(
        document=document, import_warnings=import_warnings, results=results
    )


def book_result(unit_of_work: UnitOfWork, book_id: str) -> BookResultSummary:
    """Результат обработки по блокам в постоянном порядке исходного документа.

    Документ, план и результаты читаются один раз в одной транзакции: предупреждения
    книги собираются из тех же данных, а не отдельным вторым чтением.
    """
    with unit_of_work.transaction() as transaction:
        book = _book(transaction, book_id)
        document = transaction.documents.get(book_id)
        analyses = transaction.analyses.list_for_book(book_id)
        import_warnings = transaction.documents.import_warnings(book_id)
        run = transaction.runs.active_for_book(book_id)
        stored_results = (
            transaction.results.list_for_run(run.id) if run is not None else ()
        )
    treatments = {
        item.block_id: item.treatment
        for analysis in analyses
        for item in analysis.treatment_plan.items
    }
    results = {result.block_id: result for result in stored_results}
    chapters = (
        tuple(
            ResultChapterSummary(
                section_id=section.id,
                chapter_number=number,
                title=section.title,
                blocks=tuple(
                    ResultBlockSummary(
                        block=block,
                        chapter_number=number,
                        treatment=treatments.get(block.id),
                        result=results.get(block.id),
                    )
                    for block in section.ordered_blocks
                ),
            )
            for number, section in enumerate(document.ordered_sections, start=1)
        )
        if document is not None
        else ()
    )
    return BookResultSummary(
        book=book,
        chapters=chapters,
        warnings=_warnings_of(
            document=document,
            import_warnings=import_warnings,
            results=stored_results,
        ),
    )


def result_block(
    unit_of_work: UnitOfWork, book_id: str, block_id: str
) -> ResultBlockSummary:
    """Один блок результата для HTMX-фрагмента или JSON-ответа перегенерации.

    Читается только запрошенный блок: фрагмент опрашивается раз в полторы-две секунды
    до конца перегенерации, и собирать ради него результат всей книги нельзя.
    """
    with unit_of_work.transaction() as transaction:
        _book(transaction, book_id)
        document = transaction.documents.get(book_id)
        located = _locate_block(document, block_id)
        if located is None:
            raise ResultBlockNotFoundError(f"блока {block_id} нет в книге {book_id}")
        chapter_number, section_id, block = located
        analysis = transaction.analyses.get_for_section(book_id, section_id)
        run = transaction.runs.active_for_book(book_id)
        result = transaction.results.get(run.id, block_id) if run is not None else None
    return ResultBlockSummary(
        block=block,
        chapter_number=chapter_number,
        treatment=_planned_treatment(analysis, block_id),
        result=result,
    )


def _warnings_of(
    *,
    document: CanonicalDocument | None,
    import_warnings: Sequence[BookImportWarning],
    results: Sequence[BlockProcessingResult],
) -> BookWarnings:
    """Предупреждения книги по уже прочитанным документу и результатам обработки."""
    sections_by_block = _sections_by_block(document)
    numbers = _chapter_numbers(document)
    return BookWarnings(
        import_warnings=tuple(import_warnings),
        result_warnings=tuple(
            ResultWarningEntry(
                warning=warning,
                block_id=result.block_id,
                section_id=sections_by_block.get(result.block_id),
                chapter_number=numbers.get(sections_by_block.get(result.block_id, "")),
            )
            for result in results
            for warning in result.warnings
        ),
    )


def _locate_block(
    document: CanonicalDocument | None, block_id: str
) -> tuple[int, str, DocumentBlock] | None:
    """Блок книги вместе с номером и идентификатором его главы."""
    if document is None:
        return None
    for number, section in enumerate(document.ordered_sections, start=1):
        for block in section.ordered_blocks:
            if block.id == block_id:
                return number, section.id, block
    return None


def _planned_treatment(
    analysis: ChapterAnalysis | None, block_id: str
) -> BlockTreatment | None:
    """Режим обработки блока из плана его главы либо `None`, если плана ещё нет."""
    if analysis is None:
        return None
    return next(
        (
            item.treatment
            for item in analysis.treatment_plan.items
            if item.block_id == block_id
        ),
        None,
    )


def _progress(
    *,
    document: CanonicalDocument | None,
    analyses: Sequence[ChapterAnalysis],
    tasks: Sequence[ProcessingTask],
    processed_blocks: int,
) -> BookProgress:
    """Собирает прогресс книги по её плану, задачам и сохранённым результатам."""
    sections = document.ordered_sections if document is not None else ()
    numbers = _chapter_numbers(document)
    titles = {section.id: section.title for section in sections}
    running = next(
        (
            task
            for task in tasks
            if task.status is TaskStatus.RUNNING
            and task.task_type in _CHAPTER_TASK_TYPES
        ),
        None,
    )
    current_section = running.scope_id if running is not None else None
    return BookProgress(
        total_sections=len(sections),
        analyzed_sections=len(analyses),
        planned_blocks=sum(
            len(analysis.treatment_plan.textual_block_ids) for analysis in analyses
        ),
        processed_blocks=processed_blocks,
        treatment_counts=_merge_counts(
            _treatment_counts(analysis) for analysis in analyses
        ),
        current_chapter=(
            numbers.get(current_section) if current_section is not None else None
        ),
        current_chapter_title=(
            titles.get(current_section) if current_section is not None else None
        ),
        failures=tuple(
            _failure(task) for task in tasks if task.status is TaskStatus.FAILED
        ),
        pending_tasks=sum(1 for task in tasks if task.status is TaskStatus.PENDING),
        running_tasks=sum(1 for task in tasks if task.status is TaskStatus.RUNNING),
    )


def _failure(task: ProcessingTask) -> TaskFailure:
    """Неуспешная задача в терминах интерфейса."""
    error = task.last_error
    if error is None:  # состояние `failed` без ошибки домен не допускает
        raise ApplicationError(f"у неуспешной задачи {task.id} нет ошибки")
    return TaskFailure(
        task_id=task.id,
        task_type=task.task_type,
        scope_id=task.scope_id,
        error_type=error.type,
        message=error.message,
        detail=error.detail,
        attempts=task.attempts,
        is_retryable=task.is_retryable,
    )


def _chapter_plan(
    *, section: DocumentSection, number: int, analysis: ChapterAnalysis
) -> ChapterPlanSummary:
    """Сводка плана одной главы."""
    treatments = {
        item.block_id: item.treatment for item in analysis.treatment_plan.items
    }
    return ChapterPlanSummary(
        section_id=analysis.section_id,
        chapter_number=number,
        title=section.title,
        coverage_status=analysis.coverage_status,
        textual_blocks=len(analysis.treatment_plan.textual_block_ids),
        treatment_counts=_treatment_counts(analysis),
        blocks=tuple(
            PlanBlockSummary(block=block, treatment=treatments.get(block.id))
            for block in section.ordered_blocks
        ),
    )


def _analysed_sections(
    sections: Sequence[DocumentSection], analyses: Sequence[ChapterAnalysis]
) -> tuple[tuple[int, DocumentSection, ChapterAnalysis], ...]:
    """Проанализированные главы в порядке чтения вместе с их номерами.

    Анализ без главы в документе пропускается: такого состояния хранилище не создаёт,
    но сводка не должна падать, если книгу переимпортировали.
    """
    by_section = {analysis.section_id: analysis for analysis in analyses}
    return tuple(
        (number, section, by_section[section.id])
        for number, section in enumerate(sections, start=1)
        if section.id in by_section
    )


def _treatment_counts(analysis: ChapterAnalysis) -> Mapping[BlockTreatment, int]:
    """Сколько блоков главы получило каждый режим обработки (PRD § 10.1)."""
    counts: dict[BlockTreatment, int] = {}
    for item in analysis.treatment_plan.items:
        counts[item.treatment] = counts.get(item.treatment, 0) + 1
    return counts


def _merge_counts(
    counts: Iterable[Mapping[BlockTreatment, int]],
) -> Mapping[BlockTreatment, int]:
    """Складывает счётчики режимов нескольких глав в один."""
    total: dict[BlockTreatment, int] = {}
    for chapter_counts in counts:
        for treatment, value in chapter_counts.items():
            total[treatment] = total.get(treatment, 0) + value
    return total


def _chapter_numbers(document: CanonicalDocument | None) -> Mapping[str, int]:
    """Номера глав книги в порядке чтения по идентификаторам разделов."""
    if document is None:
        return {}
    return {
        section.id: number
        for number, section in enumerate(document.ordered_sections, start=1)
    }


def _sections_by_block(document: CanonicalDocument | None) -> Mapping[str, str]:
    """Глава каждого блока книги."""
    if document is None:
        return {}
    return {
        block.id: section.id
        for section in document.ordered_sections
        for block in section.ordered_blocks
    }


def _book(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Запись книги в открытой транзакции."""
    book = transaction.books.get(book_id)
    if book is None:
        raise BookNotFoundError(f"книги {book_id} нет в хранилище")
    return book


__all__ = [
    "BookNotFoundError",
    "BookOverview",
    "BookProgress",
    "BookResultSummary",
    "BookWarnings",
    "ChapterPlanSummary",
    "PlanBlockSummary",
    "ProcessingPlanSummary",
    "ProcessingPlanUnavailableError",
    "ResultBlockNotFoundError",
    "ResultBlockSummary",
    "ResultChapterSummary",
    "ResultWarningEntry",
    "TaskFailure",
    "book_overview",
    "book_result",
    "book_warnings",
    "list_books",
    "processing_plan",
    "result_block",
]
