"""Book export to EPUB use case.

The use case collects exporter inputs — canonical document, resources, processing results,
and learning materials, — invokes the writer port, and, if configured, the external validation port.
It is agnostic of EPUB container details, XHTML, filesystem, and SQLite (ADR § 3): book assembly
and its built-in technical checks reside in the adapter, while the use case owns export availability
rules and decisions based on the external validator's report.

Export operates in two modes (PRD § 20). Full mode builds the book with translations, vocabulary
hints, and learning materials, and is available only for books without mandatory unfinished tasks.
`structure_only` mode builds the same book without processing results: it verifies the technical
`import -> export` cycle without LLM calls and serves as a CLI diagnostic.
"""

from __future__ import annotations

from linguary.application.validators.content_plan import validate_content_coverage
from linguary.application.retry_policy import ResultValidationError

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from linguary.application.errors import ApplicationError
from linguary.application.ports.epub import (
    EpubWriteRequest,
    EpubWriter,
    ExportWarning,
)
from linguary.application.ports.export_validation import (
    ExportValidationRequest,
    ExportValidator,
)
from linguary.application.ports.persistence import (
    BookRecord,
    ExportValidationStatus,
    UnitOfWork,
)
from linguary.application.task_runner import TaskQueue
from linguary.domain import (
    BlockProcessingResult,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    LanguageNote,
    LearningMaterial,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
    TaskType,
    apply_book_event,
)


class ExportBookError(ApplicationError):
    """Export is not possible in the book's current status."""


class ExportBookNotFoundError(ExportBookError):
    """Book for export or download was not found."""


class ExportArtifactNotFoundError(ExportBookError):
    """Successful export task did not leave the expected file."""


class ExportValidationRejectedError(ExportBookError):
    """External validation found errors in the assembled book (PRD § 20).

    The file remains on disk alongside the validator report: the user needs to see
    what failed, while the export task receives `failed`.
    """


class ExportMode(StrEnum):
    """Book export modes (PRD § 20)."""

    FULL = "full"
    STRUCTURE_ONLY = "structure_only"


#: Book statuses from which full export is allowed. Re-exporting a completed
#: book is allowed: the user may have lost the file or wanted to rebuild it.
_EXPORTABLE_STATUSES = frozenset({BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED})

#: Tasks whose results are included in the final EPUB. An unfinished task of this type
#: blocks full export; the export task itself, of course, does not block.
_MANDATORY_TASK_TYPES = frozenset(
    {
        TaskType.IMPORT_BOOK,
        TaskType.ANALYZE_CHAPTER,
        TaskType.PROCESS_TRANSLATION_UNIT,
        TaskType.REGENERATE_BLOCK,
    }
)

#: Statuses of an unfinished task.
_UNFINISHED_STATUSES = frozenset(
    {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED}
)

_ACTIVE_EXPORT_TASK_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.RUNNING})


@dataclass(frozen=True, slots=True)
class ExportBookCommand:
    """Export command: which book, where, and in which mode to build."""

    book_id: str
    target_path: str
    mode: ExportMode = ExportMode.FULL
    strict_validation: bool = False
    validation_report_path: str | None = None


@dataclass(frozen=True, slots=True)
class ExportBookResult:
    """Export result: assembled file, build warnings, and external validation outcome."""

    book_id: str
    path: str
    mode: ExportMode
    validation_status: ExportValidationStatus
    warnings: tuple[ExportWarning, ...] = ()
    validation_report_path: str | None = None


class ExportBook:
    """Assembles the final book EPUB and records external validation outcome."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        writer: EpubWriter,
        validator: ExportValidator | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._writer = writer
        self._validator = validator

    def __call__(self, command: ExportBookCommand) -> ExportBookResult:
        """Assembles the book, validates it, and saves the validation outcome."""
        plan = self._plan(command)
        exported = self._writer.write(
            EpubWriteRequest(
                document=plan.document,
                target_path=command.target_path,
                stored_resource_paths=plan.stored_resource_paths,
                source_epub_path=plan.source_epub_path,
                block_results=plan.block_results,
                learning_materials=plan.learning_materials,
            )
        )
        validation = self._validate(exported.path, command)
        if command.mode is ExportMode.FULL:
            self._record(command.book_id, validation=validation)
        if validation.status is ExportValidationStatus.FAILED:
            raise ExportValidationRejectedError(
                f"внешняя проверка книги {command.book_id} нашла ошибки: "
                + "; ".join(validation.errors)
            )
        return ExportBookResult(
            book_id=command.book_id,
            path=exported.path,
            mode=command.mode,
            validation_status=validation.status,
            warnings=exported.warnings,
            validation_report_path=validation.report_path,
        )

    def ensure_ready(self, book_id: str) -> None:
        """Checks whether full export is possible before enqueuing a background task."""
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise ExportBookNotFoundError(f"книги {book_id} нет в хранилище")
            document = transaction.documents.get(book_id)
            if document is None or not document.sections:
                raise ExportBookError(
                    f"книга {book_id} не импортирована: экспортировать нечего"
                )
            _ensure_ready(book, transaction.tasks.list_for_book(book_id))
            run = transaction.runs.active_for_book(book_id)
            if run is None:
                raise ExportBookError(
                    f"у книги {book_id} нет запуска обработки: полный экспорт "
                    f"собирать не из чего"
                )
            _ensure_recovery_complete(transaction, book_id, run.id)

    def downloadable_task(self, book_id: str) -> ProcessingTask:
        """Latest successful task whose file is available for download."""
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise ExportBookNotFoundError(f"книги {book_id} нет в хранилище")
            export_tasks = tuple(
                task
                for task in transaction.tasks.list_for_book(book_id)
                if task.task_type is TaskType.EXPORT_BOOK
            )
        latest = export_tasks[-1] if export_tasks else None
        if (
            book.status is not BookStatus.COMPLETED
            or latest is None
            or latest.status is not TaskStatus.SUCCEEDED
        ):
            raise ExportBookError(
                f"готового файла книги {book_id} ещё нет: экспорт не завершён"
            )
        return latest

    def _plan(self, command: ExportBookCommand) -> _ExportPlan:
        """Reads everything needed by the exporter in a single short transaction."""
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(command.book_id)
            if book is None:
                raise ExportBookNotFoundError(
                    f"книги {command.book_id} нет в хранилище"
                )
            document = transaction.documents.get(command.book_id)
            if document is None or not document.sections:
                raise ExportBookError(
                    f"книга {command.book_id} не импортирована: экспортировать нечего"
                )
            plan = _ExportPlan(
                document=document,
                stored_resource_paths=transaction.documents.stored_resource_paths(
                    command.book_id
                ),
                source_epub_path=book.source_epub_path,
            )
            if command.mode is ExportMode.STRUCTURE_ONLY:
                return plan
            _ensure_ready(book, transaction.tasks.list_for_book(command.book_id))
            run = transaction.runs.active_for_book(command.book_id)
            if run is None:
                raise ExportBookError(
                    f"у книги {command.book_id} нет запуска обработки: полный экспорт "
                    f"собирать не из чего"
                )
            _ensure_recovery_complete(transaction, command.book_id, run.id)
            results = transaction.results.list_for_run(run.id)
            analyses = transaction.analyses.list_for_book(command.book_id)
            _validate_content_plans(document, analyses, results)
            return replace(
                plan,
                block_results={result.block_id: result for result in results},
                learning_materials=_learning_materials(
                    document=document,
                    analyses=analyses,
                    results=results,
                ),
            )

    def _validate(self, path: str, command: ExportBookCommand) -> _ValidationOutcome:
        """Outcome of external validation of the assembled file (PRD § 20).

        If the validator is missing or unavailable, the book is marked unvalidated and export
        remains successful. Warnings alone do not block export; strict mode treats them
        as errors.
        """
        if self._validator is None:
            return _ValidationOutcome(status=ExportValidationStatus.NOT_VALIDATED)
        report = self._validator.validate(
            ExportValidationRequest(
                epub_path=path,
                report_path=command.validation_report_path,
                strict=command.strict_validation,
            )
        )
        if not report.available:
            return _ValidationOutcome(
                status=ExportValidationStatus.NOT_VALIDATED,
                report_path=report.report_path,
            )
        blocking = tuple(
            issue.message
            for issue in (
                report.issues if command.strict_validation else report.errors
            )
        )
        return _ValidationOutcome(
            status=(
                ExportValidationStatus.FAILED
                if blocking
                else ExportValidationStatus.PASSED
            ),
            errors=blocking,
            report_path=report.report_path,
        )

    def _record(self, book_id: str, *, validation: _ValidationOutcome) -> None:
        """Saves validation outcome and, on success, completes the book (ALG-06)."""
        succeeded = validation.status is not ExportValidationStatus.FAILED
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise ExportBookNotFoundError(f"книги {book_id} нет в хранилище")
            transaction.books.save(
                replace(
                    book,
                    status=_status_after_export(book, succeeded=succeeded),
                    export_validation_status=validation.status,
                )
            )


class ExportBookSequence:
    """Validates export readiness and manages its idempotent background task."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        queue: TaskQueue,
        export_book: ExportBook,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._queue = queue
        self._export_book = export_book

    async def start(self, book_id: str) -> ProcessingTask:
        """Enqueues export after checks, replacing only a completed old task."""
        self._export_book.ensure_ready(book_id)
        identity = TaskIdentity(book_id=book_id, task_type=TaskType.EXPORT_BOOK)
        with self._unit_of_work.transaction() as transaction:
            existing = transaction.tasks.find_by_identity(identity)
            if (
                existing is not None
                and existing.status in _ACTIVE_EXPORT_TASK_STATUSES
            ):
                return existing
            if existing is not None:
                transaction.tasks.delete(existing.id)
        return await self._queue.enqueue(identity)


@dataclass(frozen=True, slots=True)
class _ExportPlan:
    """Exporter input assembled from persistence."""

    document: CanonicalDocument
    stored_resource_paths: Mapping[str, str]
    source_epub_path: str
    block_results: Mapping[str, BlockProcessingResult] = field(default_factory=dict)
    learning_materials: Mapping[str, LearningMaterial] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ValidationOutcome:
    """External validation outcome structured for the use case."""

    status: ExportValidationStatus
    errors: tuple[str, ...] = ()
    report_path: str | None = None


def _ensure_ready(book: BookRecord, tasks: Sequence[ProcessingTask]) -> None:
    """Ensures the book can be fully exported (PRD § 20, ADR § 3).

    Partial export with unhandled mandatory blocks is not supported in MVP,
    so an unfinished task affecting the final EPUB stops export.
    """
    if book.status not in _EXPORTABLE_STATUSES:
        raise ExportBookError(
            f"книгу в состоянии {book.status} экспортировать целиком нельзя: "
            f"обработка ещё не завершена"
        )
    unfinished = tuple(
        task
        for task in tasks
        if task.task_type in _MANDATORY_TASK_TYPES
        and task.status in _UNFINISHED_STATUSES
    )
    if unfinished:
        raise ExportBookError(
            f"у книги {book.id} есть незавершённые обязательные задачи: "
            + ", ".join(f"{task.task_type} ({task.status})" for task in unfinished)
        )


def _learning_materials(
    *,
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
    results: Sequence[BlockProcessingResult],
) -> Mapping[str, LearningMaterial]:
    """Chapter learning materials (PRD § 19).

    `Before You Read` and chapter summary are stored in its analysis, while `Language Notes`
    arrive with block results; chapter materials are assembled here rather than stored
    in a separate table.
    """
    sections = {block.id: block.section_id for block in document.blocks()}
    notes: dict[str, list[LanguageNote]] = {}
    for result in results:
        for note in result.language_notes:
            section_id = sections.get(note.block_id)
            if section_id is not None:
                notes.setdefault(section_id, []).append(note)
    materials = {
        analysis.section_id: LearningMaterial.for_chapter(
            section_id=analysis.section_id,
            before_you_read=analysis.before_you_read,
            chapter_summary=analysis.chapter_summary,
            language_notes=notes.pop(analysis.section_id, []),
        )
        for analysis in analyses
    }
    materials.update(
        {
            section_id: LearningMaterial(
                section_id=section_id, language_notes=tuple(section_notes)
            )
            for section_id, section_notes in notes.items()
        }
    )
    return materials


def _status_after_export(book: BookRecord, *, succeeded: bool) -> BookStatus:
    """Book status after export (ALG-06).

    Successful export completes the book; re-export of an already completed book preserves
    its status, as does an export rejected by external validation.
    """
    if not succeeded or book.status is not BookStatus.READY_TO_EXPORT:
        return book.status
    return apply_book_event(book.status, BookEvent.EXPORT_SUCCEEDED)


__all__ = [
    "ExportBook",
    "ExportBookCommand",
    "ExportBookError",
    "ExportBookNotFoundError",
    "ExportBookResult",
    "ExportBookSequence",
    "ExportArtifactNotFoundError",
    "ExportMode",
    "ExportValidationRejectedError",
]


def _validate_content_plans(document, analyses, results):
    """Rechecks stored bodies of the active run before assembling the book."""
    sections = {s.id: s for s in document.sections}
    blocks = tuple(document.blocks())
    for analysis in analyses:
        plan = analysis.content_plan
        if plan is None:
            continue
        owners = {b.id for b in blocks if b.section_id == analysis.section_id}
        actual = tuple(r for r in results if r.block_id in owners)
        expected = {item.block_id: item.treatment for item in analysis.treatment_plan.items if item.treatment.is_textual}
        required_owners = {d.owner_block_id for d in plan.difficulties}
        if not (required_owners | expected.keys()) <= {r.block_id for r in actual}:
            raise ResultValidationError('отсутствует результат владельца содержательного плана')
        byr = () if sections[analysis.section_id].is_notes_only else analysis.before_you_read
        if not actual:
            raise ResultValidationError('отсутствуют свидетельства содержательного плана')
        for result in actual:
            if result.block_id not in expected or result.treatment != expected[result.block_id]:
                raise ResultValidationError('режим результата не соответствует плану обработки')
            if result.content_evidence is None:
                raise ResultValidationError('отсутствуют свидетельства содержательного плана')
            validate_content_coverage(plan, (result,), byr, result.content_evidence, owner_block_ids=(result.block_id,))



def _ensure_recovery_complete(transaction, book_id, run_id):
    for root in transaction.work.list_roots(book_id, run_id=run_id):
        if not transaction.work.root_coverage_complete(root.id):
            raise ExportBookError('восстановление книги не завершено: полный экспорт недоступен')
