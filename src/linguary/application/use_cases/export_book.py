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
    DomainError,
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
                verified_body_sections=plan.verified_body_sections,
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
            results = transaction.results.list_for_run(run.id)
            _ensure_result_coverage(document, results)
            conditions = transaction.work.latest_conditions(book_id)
            _ensure_content_plans(
                document,
                transaction.analyses.list_for_book(book_id),
                results,
                english_help=bool(conditions and conditions.payload.get("english_help")),
            )

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
            _ensure_result_coverage(document, results)
            analyses = transaction.analyses.list_for_book(command.book_id)
            conditions = transaction.work.latest_conditions(command.book_id)
            _ensure_content_plans(
                document,
                analyses,
                results,
                english_help=bool(conditions and conditions.payload.get("english_help")),
            )
            return replace(
                plan,
                block_results={result.block_id: result for result in results},
                learning_materials=_learning_materials(
                    document=document,
                    analyses=analyses,
                    results=results,
                ),
                verified_body_sections=frozenset(a.section_id for a in analyses
                    if a.content_plan is not None and a.content_plan.version == 3),
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
    verified_body_sections: frozenset[str] = frozenset()


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


def _ensure_content_plans(document, analyses, results, *, english_help=False):
    """Rechecks stored bodies of the active run before assembling the book."""
    try:
        _validate_content_plans(document, analyses, results, english_help=english_help)
    except (ResultValidationError, DomainError) as error:
        raise ExportBookError(str(error)) from error


def _validate_content_plans(document, analyses, results, *, english_help=False):
    sections = {s.id: s for s in document.sections}
    blocks = tuple(document.blocks())
    for analysis in analyses:
        plan = analysis.content_plan
        if plan is None:
            continue
        if english_help and plan.version != 3:
            raise ResultValidationError("английская помощь требует план версии 3")
        owners = {b.id for b in blocks if b.section_id == analysis.section_id}
        actual = tuple(r for r in results if r.block_id in owners)
        if plan.version == 3:
            _validate_v3_bodies(analysis, actual, {b.id: b.text for b in blocks})
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
            if english_help:
                if result.content_evidence:
                    raise ResultValidationError('английская помощь не должна дублировать свидетельства старого формата')
            elif not result.content_evidence:
                raise ResultValidationError('отсутствуют свидетельства содержательного плана')
            else:
                validate_content_coverage(plan, (result,), byr, result.content_evidence, owner_block_ids=(result.block_id,))
        if english_help:
            _validate_fragment_assistance_coverage(plan, actual)


def _validate_fragment_assistance_coverage(plan, results):
    """Checks persisted English-help bodies against the plan's exact obligations."""
    expected = {difficulty.id: difficulty for difficulty in plan.difficulties}
    found = {}
    for result in results:
        for assistance in result.fragment_assistances:
            difficulty = expected.get(assistance.difficulty_id)
            if difficulty is None or difficulty.owner_block_id != result.block_id:
                raise ResultValidationError('помощь относится к чужой трудности')
            if assistance.difficulty_id in found:
                raise ResultValidationError('повтор помощи к одной трудности')
            required_points = {point.id for point in difficulty.required_points}
            if (len(assistance.point_ids) != len(set(assistance.point_ids))
                    or set(assistance.point_ids) != required_points):
                raise ResultValidationError('помощь не покрывает обязательные смысловые пункты')
            if assistance.anchors != difficulty.anchors:
                raise ResultValidationError('помощь не совпадает с точными привязками плана')
            found[assistance.difficulty_id] = assistance
    if set(found) != set(expected):
        raise ResultValidationError('не все трудности получили помощь простым английским')


def _validate_v3_bodies(analysis, results, texts):
    from linguary.application.reader_text import ReaderTextPolicy, source_quote_fragments
    from linguary.application.validators.content_plan import validate_note_sources
    policy = ReaderTextPolicy()
    plan = analysis.content_plan
    for difficulty in plan.difficulties:
        for anchor in difficulty.anchors:
            anchor.validate(texts)
    for fact in plan.context_facts:
        fact.validate(texts)
    policy.validate_final('summary', analysis.chapter_summary, tuple(e.expression for e in analysis.before_you_read))
    for entry in analysis.before_you_read:
        policy.validate_final('before_you_read.explanation', entry.explanation, (entry.expression,))
        if policy.prepare('before_you_read.explanation', entry.explanation, (entry.expression,)) != entry.explanation:
            raise ResultValidationError('учебное тело не прошло окончательную очистку')
    protected = source_quote_fragments(texts.values())
    if policy.prepare_results(results, protected) != tuple(results):
        raise ResultValidationError('учебное тело не прошло окончательную очистку')
    validate_note_sources(results, texts)



def _ensure_recovery_complete(transaction, book_id, run_id):
    for root in transaction.work.list_roots(book_id, run_id=run_id):
        if not transaction.work.root_coverage_complete(root.id):
            raise ExportBookError('восстановление книги не завершено: полный экспорт недоступен')


def _ensure_result_coverage(document, results):
    """Full export requires all existing supported textual blocks, including legacy runs."""
    from linguary.application.context_builder import textual_blocks
    expected = {block.id for section in document.sections for block in textual_blocks(section)}
    if not expected <= {result.block_id for result in results}:
        raise ExportBookError('обязательные текстовые блоки обработаны не полностью')
