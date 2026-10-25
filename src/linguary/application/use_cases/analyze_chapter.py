"""Chapter analysis use case and analysis task sequence (PRD § 13-14, ADR § 3).

`AnalyzeChapter` executes a single model request per chapter and saves `ChapterAnalysis`:
glossary candidates, chapter translation instructions, block treatment plan, `Before You Read`,
and a brief chapter event summary. No separate preparatory requests are made (for glossary,
plan, or summary), and there is no separate pass for recounting previous chapter events:
the `Previously in Chapter ...` section is assembled by the exporter from chapter `N-1`'s summary
(PRD § 19.2).

A chapter that does not fit into a single request is a fallback scenario (PRD § 14): it is sent
to the model in sequential fragments, and its analysis is assembled from the fragment responses
(`analysis_merge`, ALG-03). The chunking threshold is a configuration setting, not a property
of the use case: it depends on the output limit of the specific model.

The model call is executed outside a transaction, while the result is saved in a single short
transaction (ADR § 3). Retries are managed by `RetryPolicy` (ALG-05): the loop here waits
for backoff before retrying transient errors and appends clarification instructions to the
original request without replacing the assembled context.

`ChapterAnalysisSequence` manages ordering: chapters are analyzed strictly sequentially
in reading order because chapter context includes candidates from previous chapters (PRD § 25).
Ordering is enforced via a queue rather than worker locks: a book has at most one active
`ANALYZE_CHAPTER` task at a time, and the next is enqueued only after saving the previous
analysis. The worker parallelism setting remains shared and does not affect analysis ordering:
only tasks for different books can run in parallel.
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
    content_plan_from_response,
    validate_content_plan,
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
from linguary.domain.errors import DomainError
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

#: Sleep between model call attempts. The application layer does not own the event loop,
#: but must execute the sleep: the retry policy only calculates the duration (ALG-05).
type Sleep = Callable[[float], Awaitable[None]]

#: Factory for IDs of persisted analysis records.
type IdFactory = Callable[[], str]


class AnalyzeChapterError(ApplicationError):
    """The chapter cannot be analyzed in its current state."""


@dataclass(frozen=True, slots=True)
class AnalyzeChapterCommand:
    """Analysis command: which chapter of which book to analyze."""

    book_id: str
    section_id: str
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnalyzeChapterResult:
    """Chapter analysis result along with technical call metadata.

    `attempts` and `usage` are used for observability metrics (PRD § 29.1): how many
    model calls the chapter took and the token usage reported by the provider.
    An empty chapter requires no analysis and has no model call: `attempts` is 0.
    """

    analysis: ChapterAnalysis
    attempts: int = 0
    usage: TokenUsage | None = None


class AnalyzeChapter:
    """Analyzes a single chapter and persists its `ChapterAnalysis`."""

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
        """Analyzes a chapter and persists the result.

        A chapter fitting in a single request is analyzed directly. A longer chapter
        is sent to the model in sequential fragments, and its analysis is assembled
        from their responses (PRD § 14, ALG-03).
        """
        document, accumulated = self._load(command)
        section = _section_of(document, command)
        blocks = textual_blocks(section)
        if not blocks:
            analysis = _empty_analysis(section, id=self._id_factory())
            self._save(command.book_id, analysis, task_id=command.task_id)
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
        self._save(command.book_id, analysis, task_id=command.task_id)
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
        """Main scenario: entire chapter in a single request (PRD § 14)."""
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
        payload, attempts, usage = await self._request_analysis(context, blocks=blocks)
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
        """Fallback scenario: long chapter processed by fragments (PRD § 14, ALG-03).

        Fragments are requested sequentially, each with its own blocks and retry loop.
        A fragment that fails after all attempts raises the original error: the chapter
        analysis is not saved at all rather than saved incomplete, preventing incomplete
        results from reaching the user (ALG-03).
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
            payload, fragment_attempts, usage = await self._request_analysis(context, blocks=fragment.blocks)
            attempts += fragment_attempts
            usages.append(usage)
            analysed.append(
                FragmentAnalysis.of_response(
                    payload,
                    order=fragment.order,
                    block_ids_by_alias=context.block_ids_by_alias,
                    texts={b.id: b.text for b in fragment.blocks},
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
        """Reads the book and accumulated glossary decisions for the book.

        The user's decision on a candidate overrides the model's suggestion: analysis repeats
        accumulated decisions verbatim and must not return a modified version.

        Candidates from previous chapters are all included, with the model suggestion where no decision exists.
        Candidates from the current and subsequent chapters are included only with user decisions:
        decisions apply book-wide, and re-analyzing a chapter after glossary review must see them (PRD § 11),
        while model suggestions for subsequent chapters provide no value to this chapter.
        `list_for_book` itself preserves reading order.
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
        self, context: ChapterAnalysisContext, *, blocks: Sequence[DocumentBlock]
    ) -> tuple[ChapterAnalysisResponse, int, TokenUsage | None]:
        """Executes model call with retries per policy (ALG-05).

        When attempts are exhausted, raises the original error: typing the error and
        transitioning the task to `failed` is the responsibility of the runner, not this loop.
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
                try:
                    validate_content_plan(
                        response.payload, context.block_ids_by_alias, {b.id: b.text for b in blocks},
                        before_you_read_limit=context.before_you_read_limit,
                        language_notes_per_chapter=context.language_notes_per_chapter,
                        note_limits_by_block=context.note_limits_by_block,
                        note_group_limits=context.note_group_limits,
                    )
                except (ApplicationError, DomainError) as error:
                    raise ResultValidationError(str(error), expectation=context.expectation) from error
            except Exception as error:
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                    if hasattr(self._llm_client, 'reject_last'):
                        self._llm_client.reject_last()
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
        """Converts model response to a domain chapter analysis.

        Request aliases are mapped back to stable block IDs, and the plan
        is assembled in original document order: the model response does not decide
        chapter block ordering.
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
            content_plan=content_plan_from_response(
                payload, context.block_ids_by_alias, {b.id: b.text for b in blocks}
            ),
        )

    def _candidates(
        self, payload: ChapterAnalysisResponse, *, analysis_id: str
    ) -> tuple[GlossaryCandidate, ...]:
        """Glossary candidates of the chapter with computed `requires_review` (DTO-01).

        The model's flag remains an input signal: whether a candidate requires user review
        is decided by the application under PRD § 12 rules.
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

    def _save(self, book_id: str, analysis: ChapterAnalysis, *, task_id=None) -> None:
        """Saves chapter analysis in a single short transaction."""
        with self._unit_of_work.transaction() as transaction:
            conditions_id = getattr(self._llm_client, 'conditions_id', None)
            if conditions_id is not None:
                from linguary.application.book_conditions import BookConditionsService, ConditionsError
                conditions = transaction.work.latest_conditions(book_id)
                if conditions is None or conditions.id != conditions_id:
                    raise ConditionsError('conditions_revision_conflict')
                book = transaction.books.get(book_id)
                self._llm_client.service.verify_source(book, conditions)
                if task_id is not None and transaction.tasks.get(task_id) is None:
                    raise ConditionsError('task_no_longer_active')
                attempts = self._llm_client.attempts
                attempt_id = attempts[-1] if attempts else None
                BookConditionsService.archive_analysis(transaction, book_id, analysis, conditions_id, attempt_id)
                for accepted in attempts:
                    current = transaction.work.get_attempt(accepted)
                    if current.state == 'validated':
                        transaction.work.transition_attempt(accepted, 'accepted', expected_state='validated')
            transaction.analyses.save(book_id, analysis)


class ChapterAnalysisSequence:
    """Enqueues chapter analysis tasks strictly one at a time, in reading order (PRD § 13, § 25)."""

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
        """Starts book analysis: transitions book to `ANALYZING` and enqueues the first chapter.

        Returns `None` if there is nothing left to analyze: a book without chapters or one
        already fully analyzed. Re-running after application restart resumes from the first
        unanalyzed chapter.

        If analysis of this chapter previously failed with a retryable error
        (e.g., provider rejected API key, and user fixed it), restarting returns its task
        to the queue: otherwise the book would remain in `ANALYZING` without any active task.
        A task with a permanent error is left as is.
        """
        self._begin_analysis(book_id)
        task = await self._enqueue_next(book_id)
        if task is not None and task.status is TaskStatus.FAILED and task.is_retryable:
            task = task.retry()
            await self._queue.save(task)
        return task

    async def run_task(self, task: ProcessingTask) -> None:
        """Handler for `ANALYZE_CHAPTER` task for the local runner (ALG-02).

        The next chapter is enqueued only after saving the current chapter's analysis:
        this ensures the set of accumulated candidates is deterministic and independent of concurrency.
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
        """Enqueues the next unanalyzed chapter task or completes analysis."""
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
        """First chapter in reading order for which no analysis exists yet."""
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
        """Transitions book to `ANALYZING` (ALG-06).

        Continuing an already running analysis does not change state. A book that
        has not yet reached analysis or has already progressed past it is rejected by domain:
        transitions outside table ALG-06 are prohibited, returning `409` in Web UI.
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
        """Transitions analyzed book to glossary review (ALG-06).

        A book that has already completed glossary review remains unchanged: re-analyzing
        an individual chapter does not rewind it backward in the process.
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
        """Book record in open transaction."""
        book = transaction.books.get(book_id)
        if book is None:
            raise AnalyzeChapterError(f"книги {book_id} нет в хранилище")
        return book


def _section_of(
    document: CanonicalDocument, command: AnalyzeChapterCommand
) -> DocumentSection:
    """Book chapter required by command."""
    for section in document.sections:
        if section.id == command.section_id:
            return section
    raise AnalyzeChapterError(
        f"главы {command.section_id} нет в книге {command.book_id}"
    )


def _ensure_plan_covers_chapter(
    payload: ChapterAnalysisResponse, context: ChapterAnalysisContext
) -> None:
    """Blocking validation of treatment plan (PRD § 15).

    The plan must cover every provided block exactly once and contain no foreign blocks.
    Response ordering is not checked: the plan is persisted in document order,
    and permuted response rows do not affect the result.
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
    """Glossary candidate translation suggestion contains no mixed-script words (ALG-04).

    The glossary candidate is propagated to all book chapters, so a mixed-script word
    is rejected here before analysis is saved, rather than surfacing later in every block.
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
    """Complete analysis for a textless chapter: cover, insert, illustration page.

    There is nothing to pass to the model, but the chapter must be marked as analyzed:
    otherwise the book would never reach glossary review. The plan for such a chapter
    is empty and covers all its textual blocks (none exist).
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
    """Chapter `Before You Read` content (PRD § 19.1)."""
    return before_you_read_entries(payload.before_you_read)


def _total_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    """Total token usage for chapter: provider counts each call separately.

    A provider not reporting token usage leaves `None` — it is not replaced with a sum:
    zero would mean a free call, not missing data (PRD § 24).
    """
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        prompt_tokens=_total_tokens(usage.prompt_tokens for usage in reported),
        completion_tokens=_total_tokens(usage.completion_tokens for usage in reported),
    )


def _total_tokens(values: Iterable[int | None]) -> int | None:
    """Sum of values reported by the provider."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _with_status(book: BookRecord, status: BookStatus) -> BookRecord:
    """Book record with new status."""
    return replace(book, status=status)


def _new_id() -> str:
    """ID for new analysis record."""
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
