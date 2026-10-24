"""Shared steps for processing book blocks: model invocation and result storage (ALG-04).

Chapter processing by translation units (`use_cases/process_chapter.py`) and single block
regeneration (`use_cases/regenerate_block.py`) differ in context composition and work granularity.
Everything else is shared and resides here: model call retry loop per policy, blocking response
validations, short save transaction checking active run, learning note numbering, and book completion.

The transaction intentionally does not span the model call (ADR § 3): while the response is in flight,
the user may return to glossary review; saving checks the active run against the initial run, writing
nothing if they diverge.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, replace

from linguary.application.context_builder import TranslationContext
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import ExplainedExpressions
from linguary.application.llm_rejections import record_rejection
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators import TranslationResultValidator
from linguary.application.validators.content_plan import resolve_response_evidence, validate_content_coverage
from linguary.domain.content_plan import ContentEvidence
from linguary.domain import (
    BlockProcessingResult,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    DocumentSection,
    TaskStatus,
    TaskType,
    TokenUsage,
    TranslationUnit,
    apply_book_event,
    build_language_note_id,
)

#: Pause between model call attempts: calculated by retry policy, observed
#: by use case — the application layer does not own the event loop (ALG-05).
type Sleep = Callable[[float], Awaitable[None]]


class BlockProcessingError(ApplicationError):
    """Book blocks cannot be processed in its current status."""


@dataclass(frozen=True, slots=True)
class TranslationCallResult:
    """Validated result of a single call along with its cost.

    `attempts` counts model requests rather than task executions: a single block group
    may cost multiple attempts, and the provider charges for each (PRD § 24).
    """

    results: tuple[BlockProcessingResult, ...]
    attempts: int = 0
    usage: TokenUsage | None = None


class TranslationCall:
    """Executes model call with prepared context and atomically records the result (ALG-04)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        sleep: Sleep,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._retry_policy = retry_policy
        self._validator = validator
        self._sleep = sleep

    async def request(
        self,
        context: TranslationContext,
        *,
        unit: TranslationUnit,
        target_language: str,
        note_ids: Callable[[], str],
    ) -> TranslationCallResult:
        """Calls the model and validates the response, retrying per policy (ALG-04, ALG-05).

        Validations run inside the loop: rejecting a response is standard cause for retrying
        with clarification rather than failing the task. Once attempts are exhausted, the call
        raises the underlying error: classifying it and failing the task is the runner's responsibility.

        Hints and notes duplicating chapter `Before You Read` are filtered out of the validated
        result (PRD § 19.3): duplication is not grounds for rejecting the response.
        """
        request = context.request
        explained = ExplainedExpressions(
            context.explained_expressions, explanations=context.explained_entries
        )
        attempts = 0
        while True:
            attempts += 1
            response = None
            try:
                response = await self._llm_client.generate(request)
                mapped = context.to_block_ids(response.payload)
                results = self._validator.validate(
                    mapped,
                    unit=unit,
                    target_language=target_language,
                    note_id_factory=note_ids,
                    truncated=response.truncated,
                    expectation=context.expectation,
                )
                if context.content_plan is not None:
                    owners = tuple(r.block_id for r in results)
                    evidence = resolve_response_evidence(context.content_plan, mapped, results, context.before_you_read, owner_block_ids=owners)
                    results = tuple(explained.without_repeats_v2(r) for r in results)
                    checked = validate_content_coverage(context.content_plan, results, context.before_you_read, evidence, owner_block_ids=owners)
                    results = tuple(replace(r, content_evidence=ContentEvidence(context.content_plan.sha256, tuple(e for e in checked if e.block_id == r.block_id))) for r in results)
                else:
                    results = tuple(explained.without_repeats(r) for r in results)
            except Exception as error:
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                decision = self._retry_policy.decide(
                    _worded_for_the_model(error, context), attempts=attempts
                )
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds > 0:
                    await self._sleep(decision.backoff_seconds)
                request = decision.next_request(request)
                continue
            return TranslationCallResult(
                results=results,
                attempts=attempts,
                usage=response.usage,
            )

    def save(
        self, *, book_id: str, run_id: str, results: Sequence[BlockProcessingResult]
    ) -> bool:
        """Atomically replaces results of the specified blocks (ALG-04, PRD § 17).

        Returns `False` if there is no active run to write to, or if the run ID does not
        match. This discards stale results of calls started before returning to glossary review.
        """
        with self._unit_of_work.transaction() as transaction:
            run = transaction.runs.active_for_book(book_id)
            if run is None or run.id != run_id:
                return False
            transaction.results.replace_results(run_id, results)
        return True


def note_id_factory(*, chapter_number: int, used: Collection[str]) -> Callable[[], str]:
    """Factory for chapter learning note IDs formatted as `ln-ch03-001` (PRD § 19.3).

    IDs are unique within a run, so numbering continues past already stored notes of this
    chapter: neither re-running a partially processed chapter nor regenerating an individual
    block will collide with existing notes. Gaps in numbering are permitted — IDs must be
    unique, not strictly consecutive.
    """
    ordinal = 0

    def next_id() -> str:
        nonlocal ordinal
        while True:
            ordinal += 1
            note_id = build_language_note_id(
                chapter_number=chapter_number, ordinal=ordinal
            )
            if note_id not in used:
                return note_id

    return next_id


def chapter_number(document: CanonicalDocument, section: DocumentSection) -> int:
    """Chapter number in reading order: included in note IDs (PRD § 19.3)."""
    sections = document.ordered_sections
    return next(
        (
            number
            for number, other in enumerate(sections, start=1)
            if other.id == section.id
        ),
        section.order + 1,
    )


def complete_processing(
    unit_of_work: UnitOfWork,
    *,
    book_id: str,
    finished_task_id: str | None,
    task_types: Collection[TaskType],
) -> None:
    """Transitions the book to `READY_TO_EXPORT` when all work is finished (ALG-06).

    A task executing right now is still marked `running`: the runner records its final status
    only after the handler returns. Therefore its success is recorded right here, in the same
    transaction with its siblings: otherwise during concurrent processing (PRD § 25), two
    chapters completing almost simultaneously see each other still running, and the book remains
    in `PROCESSING` forever. Re-recording the same outcome by the runner is a no-op.
    A book that left `PROCESSING` does not change status: while the call was in flight,
    the user could have returned to glossary review.
    """
    with unit_of_work.transaction() as transaction:
        _record_success(transaction, finished_task_id)
        book = book_record(transaction, book_id)
        if book.status is not BookStatus.PROCESSING:
            return
        unfinished = tuple(
            task
            for task in transaction.tasks.list_for_book(book_id)
            if task.task_type in task_types
            and task.id != finished_task_id
            and task.status is not TaskStatus.SUCCEEDED
        )
        if unfinished:
            return
        transaction.books.save(
            replace(
                book,
                status=apply_book_event(book.status, BookEvent.PROCESSING_COMPLETED),
            )
        )


def _record_success(transaction: UnitOfWork, task_id: str | None) -> None:
    """Marks running task as successful if it still exists and is running.

    Tasks of a cancelled run are deleted along with it, so a missing row is not an error.
    """
    if task_id is None:
        return
    task = transaction.tasks.get(task_id)
    if task is not None and task.status is TaskStatus.RUNNING:
        transaction.tasks.save(task.succeed())


def book_record(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Book record within an active transaction."""
    book = transaction.books.get(book_id)
    if book is None:
        raise BlockProcessingError(f"книги {book_id} нет в хранилище")
    return book


def total_usage(usages: Iterable[TokenUsage | None]) -> TokenUsage | None:
    """Total token usage: provider tracks each call separately.

    A provider reporting no token counts leaves `None`: zero would signify a free call
    rather than missing data (PRD § 24).
    """
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        prompt_tokens=_total_tokens(usage.prompt_tokens for usage in reported),
        completion_tokens=_total_tokens(usage.completion_tokens for usage in reported),
    )


def _total_tokens(values: Iterable[int | None]) -> int | None:
    """Sum of token counts reported by the provider."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _worded_for_the_model(error: Exception, context: TranslationContext) -> Exception:
    """Same error rephrased using block aliases seen by the model (ALG-05).

    The rejection reason is inserted into the retry clarification instruction, while
    validators formulate it with stable block IDs. The expected block list remains as
    populated by the validator. Outwardly — into logs and task state — the original error
    with real block IDs is propagated.
    """
    if not isinstance(error, ResultValidationError):
        return error
    return ResultValidationError(
        context.to_aliases(str(error)), expectation=error.expectation
    )


__all__ = [
    "BlockProcessingError",
    "Sleep",
    "TranslationCall",
    "TranslationCallResult",
    "book_record",
    "chapter_number",
    "complete_processing",
    "note_id_factory",
    "total_usage",
]
