"""Reading book status for UI: overview card, progress, plan, and warnings.

Web UI and CLI present identical state views (PRD § 18, § 21.1, § 21.4): processed block count,
treatment distribution, currently running chapters, remaining time estimates, failed tasks,
and import/processing warnings. Computed from tasks, saved treatment plans, block results,
and LLM call logs — querying persisted storage rather than runtime process memory.

The module queries through ports without mutating state. It resides in the application layer
rather than routes: computing progress is an application policy that must not be repeated across views.
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from linguary.application.context_builder import textual_blocks
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
    LLMCallLog,
    LLMCallStage,
    ProcessingTask,
    ResultWarning,
    TaskStatus,
    TaskType,
)


class BookNotFoundError(ApplicationError):
    """Book with specified ID does not exist in storage."""


class ProcessingPlanUnavailableError(ApplicationError):
    """Processing plan is not yet available: no chapters have been analyzed."""


class ResultBlockNotFoundError(ApplicationError):
    """Requested block not found in the book's canonical document."""


#: Task types whose chapter is displayed as active/current (PRD § 21.4).
#: Import and export are book-level tasks without a specific chapter.
_CHAPTER_TASK_TYPES = (TaskType.ANALYZE_CHAPTER, TaskType.PROCESS_TRANSLATION_UNIT)

#: Stage for which time is estimated based on book status, and its associated task type.
#: Single block regeneration is an ad-hoc action and does not need time estimation.
_ESTIMATED_STAGES: Mapping[BookStatus, tuple[LLMCallStage, TaskType]] = {
    BookStatus.ANALYZING: (LLMCallStage.CHAPTER_ANALYSIS, TaskType.ANALYZE_CHAPTER),
    BookStatus.PROCESSING: (
        LLMCallStage.TRANSLATION_UNIT,
        TaskType.PROCESS_TRANSLATION_UNIT,
    ),
}

#: Seconds per minute: request rate limit is configured in requests per minute.
_SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class TaskFailure:
    """Failed book task along with user-actionable retry status (PRD § 16)."""

    task_id: str
    task_type: TaskType
    scope_id: str
    error_type: ErrorType
    message: str
    detail: str | None
    attempts: int
    is_retryable: bool


@dataclass(frozen=True, slots=True)
class RunningChapter:
    """Chapter whose task is currently executing."""

    chapter_number: int
    title: str


@dataclass(frozen=True, slots=True)
class RemainingTime:
    """Coarse estimate of time remaining for the current book processing stage.

    `seconds` is empty until the stage completes at least one model call: without measurements,
    it is cleaner to indicate that estimates will appear later rather than fabricate numbers.
    """

    stage: LLMCallStage
    seconds: int | None = None

    @property
    def is_known(self) -> bool:
        """Whether sufficient data exists for estimation."""
        return self.seconds is not None


@dataclass(frozen=True, slots=True)
class BookProgress:
    """Book processing progress in terms of the processing screen (PRD § 21.4)."""

    total_sections: int = 0
    analyzed_sections: int = 0
    planned_blocks: int = 0
    processed_blocks: int = 0
    treatment_counts: Mapping[BlockTreatment, int] = field(default_factory=dict)
    running_chapters: tuple[RunningChapter, ...] = ()
    remaining_time: RemainingTime | None = None
    failures: tuple[TaskFailure, ...] = ()
    pending_tasks: int = 0
    running_tasks: int = 0

    @property
    def current_chapter(self) -> int | None:
        """Chapter number of the first active chapter."""
        return self.running_chapters[0].chapter_number if self.running_chapters else None

    @property
    def current_chapter_title(self) -> str | None:
        """Title of the first active chapter."""
        return self.running_chapters[0].title if self.running_chapters else None

    @property
    def percent_complete(self) -> int:
        """Percentage of planned blocks processed, rounded to integer.

        A book without a plan is considered unprocessed rather than complete:
        the plan emerges during analysis, and before then there is nothing to process.
        """
        if self.planned_blocks <= 0:
            return 0
        return round(100 * self.processed_blocks / self.planned_blocks)


@dataclass(frozen=True, slots=True)
class BookOverview:
    """Book card: status, progress, treatment counters, and errors (API-01)."""

    book: BookRecord
    progress: BookProgress
    warning_count: int = 0
    conditions: Mapping = field(default_factory=dict)
    reproducibility: str = 'unknown'
    recovery: Mapping = field(default_factory=dict)
    quality_review: Mapping = field(default_factory=lambda: {'state': 'not-requested'})
    epub_ready: bool = False
    ordinary_cost: Mapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanBlockSummary:
    """A single chapter block with its assigned treatment mode."""

    block: DocumentBlock
    treatment: BlockTreatment | None


@dataclass(frozen=True, slots=True)
class ChapterPlanSummary:
    """Treatment plan summary for a single chapter (PRD § 21, UX notes)."""

    section_id: str
    chapter_number: int
    title: str
    coverage_status: CoverageStatus
    textual_blocks: int
    treatment_counts: Mapping[BlockTreatment, int]
    blocks: tuple[PlanBlockSummary, ...] = ()
    is_notes: bool = False


@dataclass(frozen=True, slots=True)
class ProcessingPlanSummary:
    """Diagnostic summary of book processing plan broken down by chapter."""

    chapters: tuple[ChapterPlanSummary, ...]
    treatment_counts: Mapping[BlockTreatment, int]
    analyzed_sections: int
    total_sections: int

    @property
    def is_complete(self) -> bool:
        """Whether all chapters are analyzed and their plans are complete."""
        return self.analyzed_sections == self.total_sections and all(
            chapter.coverage_status is CoverageStatus.COMPLETE
            for chapter in self.chapters
        )


@dataclass(frozen=True, slots=True)
class ResultWarningEntry:
    """Processing result warning along with its location in the book (PRD § 15)."""

    warning: ResultWarning
    block_id: str
    section_id: str | None
    chapter_number: int | None


@dataclass(frozen=True, slots=True)
class BookWarnings:
    """Import and processing warnings for a book (PRD § 21.6)."""

    import_warnings: tuple[BookImportWarning, ...] = ()
    result_warnings: tuple[ResultWarningEntry, ...] = ()

    @property
    def total(self) -> int:
        """Total number of warnings accumulated by the book."""
        return len(self.import_warnings) + len(self.result_warnings)


@dataclass(frozen=True, slots=True)
class ResultBlockSummary:
    """Source block, planned treatment, and persisted processing result."""

    block: DocumentBlock
    chapter_number: int
    treatment: BlockTreatment | None = None
    result: BlockProcessingResult | None = None


@dataclass(frozen=True, slots=True)
class ResultChapterSummary:
    """Result chapter with blocks in reading order."""

    section_id: str
    chapter_number: int
    title: str
    blocks: tuple[ResultBlockSummary, ...] = ()
    is_notes: bool = False


@dataclass(frozen=True, slots=True)
class BookResultSummary:
    """Book result grouped by chapter along with warnings and export status."""

    book: BookRecord
    chapters: tuple[ResultChapterSummary, ...] = ()
    warnings: BookWarnings = field(default_factory=BookWarnings)
    conditions: Mapping = field(default_factory=dict)
    reproducibility: str = 'unknown'
    recovery: Mapping = field(default_factory=dict)
    quality_review: Mapping = field(default_factory=lambda: {'state': 'not-requested'})
    epub_ready: bool = False
    ordinary_cost: Mapping = field(default_factory=dict)


def _quality_projection(transaction, book, document, analyses, run, tasks):
    """Current technical coverage and optional review, without accepting or scheduling work."""
    from linguary.application.use_cases.quality_review import quality_review_state
    from linguary.application.use_cases.export_book import _ensure_ready, _validate_content_plans, ExportBookError
    from linguary.application.retry_policy import ResultValidationError
    from linguary.domain.book_conditions import thaw
    conditions = transaction.work.latest_conditions(book.id)
    summary = dict(conditions_id=None, revision=None, reader_level=None, profile=None, models={}, state='unknown')
    if conditions is not None:
        summary.update(conditions_id=conditions.id, revision=conditions.revision,
            reader_level=conditions.payload.get('reader_level'), profile=conditions.payload.get('profile'),
            models=thaw(conditions.payload.get('models', {})), state='saved')
    results = () if run is None else transaction.results.list_for_run(run.id)
    textual = {block.id for section in document.sections for block in textual_blocks(section)} if document else set()
    result_ids = {result.block_id for result in results}
    roots = () if run is None else transaction.work.list_roots(book.id, run_id=run.id)
    root_summaries = []
    durable_coverage = set()
    for root in roots:
        scopes = transaction.work.list_scopes(root.id)
        attempts = transaction.work.list_root_attempts(root.id)
        unknown = any(attempt.state in ('sent', 'unknown') for attempt in attempts)
        current = conditions is not None and root.conditions_id == conditions.id
        scope_counts = {scope.id: sum(attempt.scope_id == scope.id for attempt in attempts) for scope in scopes}
        retryable = any(scope.status == 'pending' and scope_counts[scope.id] < root.manifest['policy']['max_scope_calls']
            for scope in scopes)
        busy = any(scope.status == 'claimed' for scope in scopes)
        covered = set(root.manifest.get('historical_coverage', ()))
        for scope in scopes:
            if scope.status == 'accepted':
                covered.update(scope.ordered_blocks)
        covered.intersection_update(result_ids)
        covered.intersection_update(root.manifest['ordered_blocks'])
        if not current:
            covered.clear()
        durable_coverage.update(covered)
        limit = root.manifest['policy']['max_calls'] + root.extra_calls
        complete = transaction.work.root_coverage_complete(root.id, update_state=False)
        root_summaries.append(dict(root_id=root.id, generation=root.generation,
            covered=len(covered), total=len(root.manifest['ordered_blocks']), calls_used=root.calls_used,
            limit=limit, reason='conditions_unavailable' if not current else 'unknown' if unknown else root.reason,
            can_retry=bool(current and not unknown and not busy and retryable and root.calls_used < limit
                and root.status in ('pending', 'running')),
            can_extend=bool(current and not unknown and not busy and retryable
                and root.status == 'stopped' and root.reason == 'limit'), complete=complete))
    recovery = dict(covered=len(durable_coverage) if roots else len(result_ids & textual), total=len(textual),
        calls_used=sum(item['calls_used'] for item in root_summaries),
        limit=sum(item['limit'] for item in root_summaries),
        reason=next((item['reason'] for item in root_summaries if item['reason'] == 'unknown'),
            next((item['reason'] for item in root_summaries if item['reason']), None)),
        can_retry=any(item['can_retry'] for item in root_summaries),
        can_extend=any(item['can_extend'] for item in root_summaries), roots=root_summaries)
    epub_ready = bool(document and document.sections and run is not None and textual <= result_ids
        and all(item['complete'] for item in root_summaries))
    if epub_ready:
        try:
            _ensure_ready(book, tasks)
            _validate_content_plans(document, analyses, results)
        except (ExportBookError, ResultValidationError):
            epub_ready = False
    ordinary = [call for call in transaction.llm_calls.list_for_book(book.id) if call.stage != LLMCallStage.QUALITY_REVIEW]
    known_cost = sum((call.cost for call in ordinary if call.cost is not None), Decimal('0'))
    regeneration_cost = sum((call.cost for call in ordinary if call.cost is not None
        and call.stage == LLMCallStage.BLOCK_REGENERATION), Decimal('0'))
    ordinary_cost = dict(currency='USD', known_spent_usd=str(known_cost),
        unknown_calls=sum(call.cost is None for call in ordinary), regeneration_spent_usd=str(regeneration_cost),
        regeneration_unknown_calls=sum(call.cost is None for call in ordinary if call.stage == LLMCallStage.BLOCK_REGENERATION))
    return dict(conditions=summary, reproducibility='unknown' if conditions is None else conditions.reproducibility,
        recovery=recovery, quality_review=quality_review_state(transaction, book.id),
        epub_ready=epub_ready, ordinary_cost=ordinary_cost)


def list_books(
    unit_of_work: UnitOfWork, *, status: BookStatus | None = None
) -> tuple[BookRecord, ...]:
    """Books ordered by addition, optionally filtered by status."""
    with unit_of_work.transaction() as transaction:
        books = transaction.books.list_all()
    if status is None:
        return books
    return tuple(book for book in books if book.status is status)


def book_overview(
    unit_of_work: UnitOfWork,
    book_id: str,
    *,
    concurrency: int = 1,
    requests_per_minute: int | None = None,
) -> BookOverview:
    """Book card: status, progress, treatment counters, and failed tasks.

    `concurrency` and `requests_per_minute` — worker concurrency and rate limits:
    without them, remaining time estimation cannot determine how many calls run in parallel.
    """
    with unit_of_work.transaction() as transaction:
        book = _book(transaction, book_id)
        document = transaction.documents.get(book_id)
        analyses = transaction.analyses.list_for_book(book_id)
        tasks = transaction.tasks.list_for_book(book_id)
        run = transaction.runs.active_for_book(book_id)
        processed_blocks = (
            transaction.results.count_for_run(run.id) if run is not None else 0
        )
        quality = _quality_projection(transaction, book, document, analyses, run, tasks)
        warning_count = len(transaction.documents.import_warnings(book_id))
        calls = (
            transaction.llm_calls.list_for_book(book_id)
            if book.status in _ESTIMATED_STAGES
            else ()
        )
    progress = _progress(
        document=document,
        analyses=analyses,
        tasks=tasks,
        processed_blocks=processed_blocks,
    )
    return BookOverview(
        book=book,
        progress=replace(
            progress,
            remaining_time=_remaining_time(
                status=book.status,
                progress=progress,
                document=document,
                tasks=tasks,
                calls=calls,
                concurrency=concurrency,
                requests_per_minute=requests_per_minute,
            ),
        ),
        warning_count=warning_count,
        **quality,
    )


def estimate_remaining_seconds(
    *,
    durations_ms: Sequence[int],
    completed_blocks: int,
    remaining_blocks: int,
    parallelism: int = 1,
    requests_per_minute: int | None = None,
) -> int | None:
    """Coarse estimate of remaining stage duration in seconds (QA-13).

    Median duration of model calls is multiplied by remaining calls and divided by
    effective concurrency. Remaining calls are projected through block counts:
    call/block ratio observed so far is extrapolated to remaining blocks, accounting for
    multi-block groups and chapter fragment splits. Rate limits prevent parallelism from
    compressing call intervals below `60 / requests_per_minute`. Median is used rather than
    mean so individual timeouts do not skew the entire estimate.

    Returns `None` if there is insufficient data (no successful calls or zero completed blocks).
    """
    if not durations_ms or completed_blocks <= 0:
        return None
    if remaining_blocks <= 0:
        return 0
    remaining_calls = remaining_blocks * len(durations_ms) / completed_blocks
    seconds_per_call = statistics.median(durations_ms) / 1000 / max(1, parallelism)
    if requests_per_minute is not None:
        seconds_per_call = max(
            seconds_per_call, _SECONDS_PER_MINUTE / requests_per_minute
        )
    return math.ceil(remaining_calls * seconds_per_call)


def processing_plan(unit_of_work: UnitOfWork, book_id: str) -> ProcessingPlanSummary:
    """Processing plan summary grouped by chapter.

    The plan is produced by analysis and does not exist before it: an unanalyzed book
    raises an error rather than returning an empty summary, which would look like a book
    with zero textual blocks (HTTP 409 in Web UI).
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
    """Import warnings and block processing result warnings for a book."""
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
    """Processing results by block in stable original document order.

    Document, plan, and results are read in a single transaction: book warnings are
    derived from the same dataset without a secondary query.
    """
    with unit_of_work.transaction() as transaction:
        book = _book(transaction, book_id)
        document = transaction.documents.get(book_id)
        analyses = transaction.analyses.list_for_book(book_id)
        import_warnings = transaction.documents.import_warnings(book_id)
        run = transaction.runs.active_for_book(book_id)
        quality = _quality_projection(transaction, book, document, analyses, run, transaction.tasks.list_for_book(book_id))
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
                is_notes=section.is_notes_only,
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
        **quality,
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
    """A single result block for HTMX fragment polling or JSON regeneration responses.

    Only the requested block is queried: this endpoint is polled every 1.5-2 seconds
    during block regeneration, so loading whole-book results is inappropriate.
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
    """Book warnings computed from already retrieved document and results."""
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
    """Book block along with chapter number and chapter section ID."""
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
    """Block treatment mode from chapter plan or `None` if plan not yet ready."""
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
    """Assembles book progress from its plan, tasks, and stored results."""
    sections = document.ordered_sections if document is not None else ()
    numbers = _chapter_numbers(document)
    titles = {section.id: section.title for section in sections}
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
        running_chapters=_running_chapters(tasks, numbers=numbers, titles=titles),
        failures=tuple(
            _failure(task) for task in tasks if task.status is TaskStatus.FAILED
        ),
        pending_tasks=sum(1 for task in tasks if task.status is TaskStatus.PENDING),
        running_tasks=sum(1 for task in tasks if task.status is TaskStatus.RUNNING),
    )


def _running_chapters(
    tasks: Sequence[ProcessingTask],
    *,
    numbers: Mapping[str, int],
    titles: Mapping[str, str],
) -> tuple[RunningChapter, ...]:
    """All chapters whose tasks are executing right now, in reading order.

    With parallel execution (PRD § 25), several chapters run concurrently; showing only one
    would conceal the others.
    """
    sections = {
        task.scope_id
        for task in tasks
        if task.status is TaskStatus.RUNNING
        and task.task_type in _CHAPTER_TASK_TYPES
        and task.scope_id in numbers
    }
    return tuple(
        RunningChapter(chapter_number=numbers[section_id], title=titles[section_id])
        for section_id in sorted(sections, key=numbers.__getitem__)
    )


def _remaining_time(
    *,
    status: BookStatus,
    progress: BookProgress,
    document: CanonicalDocument | None,
    tasks: Sequence[ProcessingTask],
    calls: Sequence[LLMCallLog],
    concurrency: int,
    requests_per_minute: int | None,
) -> RemainingTime | None:
    """Remaining time estimation for analysis or processing while stage is active.

    Stages with no pending or running tasks are not estimated: they are either complete
    or paused by error. Only calls from currently existing tasks are counted: returning to
    glossary review prunes cancelled run tasks, so earlier calls do not skew remaining work.
    """
    if status not in _ESTIMATED_STAGES:
        return None
    stage, task_type = _ESTIMATED_STAGES[status]
    stage_tasks = [task for task in tasks if task.task_type is task_type]
    active = sum(
        1
        for task in stage_tasks
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
    )
    if active == 0:
        return None
    task_ids = {task.id for task in stage_tasks}
    durations = [
        call.duration_ms
        for call in calls
        if call.stage is stage and call.success and call.task_id in task_ids
    ]
    if stage is LLMCallStage.CHAPTER_ANALYSIS:
        completed = progress.planned_blocks
        remaining = _textual_block_count(document) - completed
        parallelism = 1
    else:
        completed = progress.processed_blocks
        remaining = progress.planned_blocks - completed
        parallelism = min(concurrency, active)
    return RemainingTime(
        stage=stage,
        seconds=estimate_remaining_seconds(
            durations_ms=durations,
            completed_blocks=completed,
            remaining_blocks=remaining,
            parallelism=parallelism,
            requests_per_minute=requests_per_minute,
        ),
    )


def _textual_block_count(document: CanonicalDocument | None) -> int:
    """Number of blocks in the book passed to the model."""
    if document is None:
        return 0
    return sum(len(textual_blocks(section)) for section in document.ordered_sections)


def _failure(task: ProcessingTask) -> TaskFailure:
    """Failed task in UI representation."""
    error = task.last_error
    if error is None:  # domain invariant forbids `failed` status without an error
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
    """Plan summary for a single chapter."""
    treatments = {
        item.block_id: item.treatment for item in analysis.treatment_plan.items
    }
    return ChapterPlanSummary(
        section_id=analysis.section_id,
        chapter_number=number,
        title=section.title,
        is_notes=section.is_notes_only,
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
    """Analyzed chapters in reading order with their sequence numbers.

    Analyses without matching document chapters are skipped: storage does not produce this state,
    but summaries should not crash if a book was re-imported.
    """
    by_section = {analysis.section_id: analysis for analysis in analyses}
    return tuple(
        (number, section, by_section[section.id])
        for number, section in enumerate(sections, start=1)
        if section.id in by_section
    )


def _treatment_counts(analysis: ChapterAnalysis) -> Mapping[BlockTreatment, int]:
    """Number of chapter blocks assigned each treatment mode (PRD § 10.1)."""
    counts: dict[BlockTreatment, int] = {}
    for item in analysis.treatment_plan.items:
        counts[item.treatment] = counts.get(item.treatment, 0) + 1
    return counts


def _merge_counts(
    counts: Iterable[Mapping[BlockTreatment, int]],
) -> Mapping[BlockTreatment, int]:
    """Aggregates treatment counters across multiple chapters."""
    total: dict[BlockTreatment, int] = {}
    for chapter_counts in counts:
        for treatment, value in chapter_counts.items():
            total[treatment] = total.get(treatment, 0) + value
    return total


def _chapter_numbers(document: CanonicalDocument | None) -> Mapping[str, int]:
    """Chapter numbers in reading order by section ID."""
    if document is None:
        return {}
    return {
        section.id: number
        for number, section in enumerate(document.ordered_sections, start=1)
    }


def _sections_by_block(document: CanonicalDocument | None) -> Mapping[str, str]:
    """Section ID for each block in the book."""
    if document is None:
        return {}
    return {
        block.id: section.id
        for section in document.ordered_sections
        for block in section.ordered_blocks
    }


def _book(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Book record within an open transaction."""
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
    "RemainingTime",
    "ResultBlockNotFoundError",
    "ResultBlockSummary",
    "ResultChapterSummary",
    "ResultWarningEntry",
    "RunningChapter",
    "TaskFailure",
    "book_overview",
    "book_result",
    "book_warnings",
    "estimate_remaining_seconds",
    "list_books",
    "processing_plan",
    "result_block",
]
