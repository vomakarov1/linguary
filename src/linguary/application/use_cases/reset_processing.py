"""Reset of book processing results: return to glossary review (PRD § 11, § 21.5, § 22).

The user performs this reset when noticing an unsatisfactory translation choice affecting
the entire book: isolated errors in individual blocks are fixed by block regeneration. The reset
removes the processing run along with its results — translations, vocabulary hints, `Language Notes`,
and warnings, — while preserving the import, stable `DocumentBlock` entities, and chapter analyses:
re-analyzing chapters is the most expensive phase and should not be re-executed due to editing a single term.

Everything is performed in a single transaction: the run, its results, processing and regeneration
tasks, and book status are updated together. A model call started prior to reset is not covered
by the transaction; its late result is discarded by the active run check in the save transaction
(ALG-04): the run with which the chapter was started no longer exists.

The reset requires no separate "unfreezing" of the glossary: decision edits are allowed in
`GLOSSARY_REVIEW` and `READY_TO_PROCESS`, and the book transitions into the former (ALG-06).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import BookEvent, TaskType, apply_book_event

#: Tasks removed by reset: they belong to the cancelled run and should not
#: execute after reset (PRD § 21.5).
RESET_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK}
)


class ResetProcessingError(ApplicationError):
    """Book processing cannot be reset."""


@dataclass(frozen=True, slots=True)
class ResetProcessingCommand:
    """Return to glossary review command.

    User confirmation is managed by the UI: it displays what will be deleted
    (PRD § 11) and invokes the use case only after explicit consent.
    """

    book_id: str


@dataclass(frozen=True, slots=True)
class ResetProcessingResult:
    """What the reset removed: displays the cost of the confirmed action to the user.

    `run_id` is empty if the book had no active run: in that case reset only returns
    the book to glossary review.
    """

    book_id: str
    run_id: str | None = None
    removed_results: int = 0
    removed_tasks: int = 0


class ResetProcessing:
    """Deletes book processing results and returns the book to `GLOSSARY_REVIEW`."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: ResetProcessingCommand) -> ResetProcessingResult:
        """Resets book processing in a single transaction (PRD § 17, § 21.5).

        A book whose status does not permit reset receives a domain rejection
        (HTTP 409 in Web UI): there is nothing to reset before processing has started.
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            status = apply_book_event(book.status, BookEvent.PROCESSING_RESET_CONFIRMED)
            run = transaction.runs.active_for_book(book.id)
            removed_results = (
                transaction.results.count_for_run(run.id) if run is not None else 0
            )
            removed_tasks = _remove_tasks(transaction, book.id)
            if run is not None:
                _archive_run_results(transaction, book.id, run)
                transaction.work.invalidate_run_coverage(book.id, run.id)
                transaction.work.stale_reviews(book.id)
                transaction.runs.delete(run.id)
            transaction.books.save(replace(book, status=status))
        return ResetProcessingResult(
            book_id=book.id,
            run_id=run.id if run is not None else None,
            removed_results=removed_results,
            removed_tasks=removed_tasks,
        )


def _remove_tasks(transaction: UnitOfWork, book_id: str) -> int:
    """Cancels processing and block regeneration tasks for the book (PRD § 18.1).

    Tasks in any status are removed, not just pending or running: an idempotent task
    key lives as long as the task itself, and a completed task from a cancelled run
    would prevent enqueuing the same task for a new run — leaving the book unable
    to be processed a second time. Import, chapter analysis, and export tasks are not
    touched: their results are preserved.
    """
    removed = tuple(
        task
        for task in transaction.tasks.list_for_book(book_id)
        if task.task_type in RESET_TASK_TYPES
    )
    for task in removed:
        transaction.tasks.delete(task.id)
    return len(removed)


def _book(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Book record within an active transaction."""
    book = transaction.books.get(book_id)
    if book is None:
        raise ResetProcessingError(f"книги {book_id} нет в хранилище")
    return book


__all__ = [
    "RESET_TASK_TYPES",
    "ResetProcessing",
    "ResetProcessingCommand",
    "ResetProcessingError",
    "ResetProcessingResult",
]


class ChooseBookConditions:
    """Explicitly continue historical work or atomically start a new analysis revision."""
    def __init__(self, service, resolver):
        self.service, self.resolver = service, resolver

    def __call__(self, book_id, *, action, expected_revision, settings_revision, confirmed):
        from dataclasses import asdict
        from linguary.application.book_conditions import ConditionsError, BookConditionsService
        if confirmed is not True or action not in {'continue_legacy', 'reanalyze'}:
            raise ConditionsError('invalid_conditions_choice')
        with self.service.uow.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise ConditionsError('book_not_found')
            previous = transaction.work.latest_conditions(book_id)
            if (0 if previous is None else previous.revision) != expected_revision:
                raise ConditionsError('conditions_revision_conflict')
            if action == 'continue_legacy' and previous is not None:
                raise ConditionsError('conditions_already_selected')
            if book.status.value in {'IMPORTING', 'FAILED'}:
                raise ConditionsError('invalid_book_state')
            conditions = self.service.create(transaction, book, settings_revision,
                reproducibility='legacy_partial' if action == 'continue_legacy' else 'complete')
            # Validate the complete executable route before committing any revision.
            self.resolver.settings_for(conditions)
            run = transaction.runs.active_for_book(book_id)
            if action == 'reanalyze':
                for analysis in transaction.analyses.list_for_book(book_id):
                    binding = transaction.work.get_binding('analysis', analysis.id)
                    BookConditionsService.archive_analysis(transaction, book_id, analysis,
                        binding['conditions_id'] if binding else None,
                        binding['attempt_id'] if binding else None)
                if run is not None:
                    _archive_run_results(transaction, book_id, run)
                    transaction.work.invalidate_run_coverage(book_id, run.id)
                    transaction.work.stale_reviews(book_id)
                    transaction.runs.delete(run.id)
                transaction.analyses.delete_for_book(book_id)
                for task in transaction.tasks.list_for_book(book_id):
                    if task.task_type in {TaskType.ANALYZE_CHAPTER, TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK, TaskType.EXPORT_BOOK}:
                        transaction.tasks.delete(task.id)
                transaction.books.save(replace(book, status=apply_book_event(book.status, BookEvent.REANALYSIS_CONFIRMED)))
            else:
                # Bind only future execution; old analyses and results stay unknown.
                if run is not None:
                    transaction.work.bind('run', run.id, book_id=book_id, conditions_id=conditions.id,
                        payload={'schema_version': 1, 'legacy_partial': True})
                for task in transaction.tasks.list_for_book(book_id):
                    if task.task_type in {TaskType.ANALYZE_CHAPTER, TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK}:
                        transaction.work.bind('task', task.id, book_id=book_id, conditions_id=conditions.id,
                            payload={'schema_version': 1, 'legacy_partial': True})
            return conditions



def _archive_run_results(transaction, book_id, run):
    """Historical projections retain unknown conditions after explicit continuation."""
    from dataclasses import asdict
    run_binding = transaction.work.get_binding('run', run.id)
    for result in transaction.results.list_for_run(run.id):
        binding = transaction.work.get_binding('result', f'{run.id}:{result.block_id}')
        provenance = binding or run_binding
        known = provenance is not None and not provenance['payload_json'].get('legacy_partial', False)
        transaction.work.archive_artifacts([dict(book_id=book_id,
            conditions_id=provenance['conditions_id'] if known else None,
            attempt_id=binding['attempt_id'] if binding else None,
            run_id=run.id, kind='block_result', scope_id=result.block_id,
            body={'schema_version': 1, 'result': asdict(result)})])
