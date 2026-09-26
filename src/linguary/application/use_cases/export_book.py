"""Сценарий экспорта книги в EPUB.

Сценарий собирает вход экспортёра — каноническую модель, ресурсы, результаты обработки
и учебные материалы, — вызывает порт записи и, если подключён, порт внешней проверки.
О контейнере EPUB, XHTML, файловой системе и SQLite он не знает (ADR § 3): сборка книги
и её встроенные технические проверки живут в адаптере, а сценарий владеет правилами
доступности экспорта и решением по отчёту внешнего валидатора.

Экспорт работает в двух режимах (PRD § 20). Полный режим собирает книгу с переводами,
словарными подсказками и учебными материалами и доступен только для книги без обязательных
незавершённых задач. Режим `structure_only` собирает ту же книгу без результатов обработки:
он проверяет технический цикл `import -> export` без LLM и служит диагностикой через CLI.
"""

from __future__ import annotations

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
    """Экспорт невозможен в текущем состоянии книги."""


class ExportBookNotFoundError(ExportBookError):
    """Книга для экспорта или скачивания не найдена."""


class ExportArtifactNotFoundError(ExportBookError):
    """Успешная экспортная задача не оставила ожидаемый файл."""


class ExportValidationRejectedError(ExportBookError):
    """Внешняя проверка нашла в собранной книге ошибки (PRD § 20).

    Файл остаётся на диске вместе с отчётом валидатора: пользователю нужно видеть,
    что именно не сошлось, а экспортная задача получает `failed`.
    """


class ExportMode(StrEnum):
    """Режимы экспорта книги (PRD § 20)."""

    FULL = "full"
    STRUCTURE_ONLY = "structure_only"


#: Состояния книги, из которых доступен полный экспорт. Повторный экспорт завершённой
#: книги допустим: пользователь мог потерять файл или захотеть собрать его заново.
_EXPORTABLE_STATUSES = frozenset({BookStatus.READY_TO_EXPORT, BookStatus.COMPLETED})

#: Задачи, чей результат входит в итоговый EPUB. Незавершённая задача такого типа
#: блокирует полный экспорт; сама экспортная задача, разумеется, не блокирует.
_MANDATORY_TASK_TYPES = frozenset(
    {
        TaskType.IMPORT_BOOK,
        TaskType.ANALYZE_CHAPTER,
        TaskType.PROCESS_TRANSLATION_UNIT,
        TaskType.REGENERATE_BLOCK,
    }
)

#: Состояния незавершённой задачи.
_UNFINISHED_STATUSES = frozenset(
    {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED}
)

_ACTIVE_EXPORT_TASK_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.RUNNING})


@dataclass(frozen=True, slots=True)
class ExportBookCommand:
    """Команда экспорта: какую книгу, куда и в каком режиме собирать."""

    book_id: str
    target_path: str
    mode: ExportMode = ExportMode.FULL
    strict_validation: bool = False
    validation_report_path: str | None = None


@dataclass(frozen=True, slots=True)
class ExportBookResult:
    """Результат экспорта: собранный файл, замечания сборки и итог внешней проверки."""

    book_id: str
    path: str
    mode: ExportMode
    validation_status: ExportValidationStatus
    warnings: tuple[ExportWarning, ...] = ()
    validation_report_path: str | None = None


class ExportBook:
    """Собирает итоговый EPUB книги и фиксирует результат внешней проверки."""

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
        """Собирает книгу, проверяет её и сохраняет итог проверки."""
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
        """Проверяет полный экспорт до постановки фоновой задачи."""
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
            if transaction.runs.active_for_book(book_id) is None:
                raise ExportBookError(
                    f"у книги {book_id} нет запуска обработки: полный экспорт "
                    f"собирать не из чего"
                )

    def downloadable_task(self, book_id: str) -> ProcessingTask:
        """Последняя успешная задача, чей файл разрешено скачать."""
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
        """Читает всё, что нужно экспортёру, одной короткой транзакцией."""
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
            results = transaction.results.list_for_run(run.id)
            return replace(
                plan,
                block_results={result.block_id: result for result in results},
                learning_materials=_learning_materials(
                    document=document,
                    analyses=transaction.analyses.list_for_book(command.book_id),
                    results=results,
                ),
            )

    def _validate(self, path: str, command: ExportBookCommand) -> _ValidationOutcome:
        """Итог внешней проверки собранного файла (PRD § 20).

        Валидатора нет или он недоступен — книга помечается как непроверенная, и экспорт
        остаётся успешным. Предупреждения сами по себе экспорт не блокируют; строгий режим
        приравнивает их к ошибкам.
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
        """Сохраняет итог проверки и, при успехе, завершает книгу (ALG-06)."""
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
    """Проверяет экспорт и управляет его идемпотентной фоновой задачей."""

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
        """Ставит экспорт после проверки, заменяя только завершённую старую задачу."""
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
    """Вход экспортёра, собранный из хранилища."""

    document: CanonicalDocument
    stored_resource_paths: Mapping[str, str]
    source_epub_path: str
    block_results: Mapping[str, BlockProcessingResult] = field(default_factory=dict)
    learning_materials: Mapping[str, LearningMaterial] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ValidationOutcome:
    """Итог внешней проверки в виде, нужном сценарию."""

    status: ExportValidationStatus
    errors: tuple[str, ...] = ()
    report_path: str | None = None


def _ensure_ready(book: BookRecord, tasks: Sequence[ProcessingTask]) -> None:
    """Проверяет, что книгу можно экспортировать полностью (PRD § 20, ADR § 3).

    Частичный экспорт с необработанными обязательными блоками в MVP не поддерживается,
    поэтому незавершённая задача, влияющая на итоговый EPUB, экспорт останавливает.
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
    """Учебные материалы по главам (PRD § 19).

    `Before You Read` и сводка главы лежат в её анализе, а `Language Notes` приходят
    вместе с результатами блоков, поэтому материалы главы собираются здесь, а не
    хранятся отдельной таблицей.
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
    """Состояние книги после экспорта (ALG-06).

    Успешный экспорт завершает книгу; повторный экспорт уже завершённой книги оставляет
    её состояние прежним, как и экспорт, отвергнутый внешней проверкой.
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
