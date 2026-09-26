"""Сценарий экспорта книги: режимы, готовность и внешняя проверка (P2.T5)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from conftest import FIXTURE_DIRECTORY

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import (
    EbookLibEpubReader,
    EpubExportWriter,
    FilesystemEpubResourceStore,
)
from linguary.application.ports.export_validation import (
    ExportIssueSeverity,
    ExportValidationIssue,
    ExportValidationReport,
    ExportValidationRequest,
)
from linguary.application.ports.persistence import ExportValidationStatus
from linguary.application.use_cases.export_book import (
    ExportBook,
    ExportBookCommand,
    ExportBookError,
    ExportMode,
    ExportValidationRejectedError,
)
from linguary.application.use_cases.import_book import ImportBook, ImportBookCommand
from linguary.domain import (
    BookStatus,
    ErrorType,
    GlossarySnapshot,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    TaskError,
    TaskStatus,
    TaskType,
)

BOOK_ID = "book-1"


@dataclass(frozen=True, slots=True)
class FakeValidator:
    """Внешний валидатор с заранее известным ответом."""

    report: ExportValidationReport

    def validate(self, request: ExportValidationRequest) -> ExportValidationReport:
        """Отдаёт заготовленный отчёт независимо от содержимого книги."""
        return replace(self.report, report_path=request.report_path)


@pytest.fixture
def imported(unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path) -> str:
    """Импортированная книга набора в хранилище теста."""
    ImportBook(
        unit_of_work=unit_of_work,
        reader=EbookLibEpubReader(),
        resource_store=FilesystemEpubResourceStore(root=tmp_path / "storage"),
    )(
        ImportBookCommand(
            book_id=BOOK_ID,
            source_epub_path=str(FIXTURE_DIRECTORY / "synthetic_story.epub"),
            target_language="ru",
        )
    )
    return BOOK_ID


def export(
    unit_of_work: SqlAlchemyUnitOfWork, *, validator: FakeValidator | None = None
) -> ExportBook:
    """Сценарий экспорта, настроенный на хранилище теста."""
    return ExportBook(
        unit_of_work=unit_of_work, writer=EpubExportWriter(), validator=validator
    )


def command(tmp_path: Path, **overrides: object) -> ExportBookCommand:
    """Команда экспорта в файл внутри директории теста."""
    arguments: dict[str, object] = {
        "book_id": BOOK_ID,
        "target_path": str(tmp_path / "export" / "book.epub"),
    }
    arguments.update(overrides)
    return ExportBookCommand(**arguments)  # type: ignore[arg-type]


def make_ready(
    unit_of_work: SqlAlchemyUnitOfWork,
    *,
    status: BookStatus = BookStatus.READY_TO_EXPORT,
    with_run: bool = True,
    tasks: tuple[ProcessingTask, ...] = (),
) -> None:
    """Доводит книгу до состояния, в котором доступен полный экспорт."""
    with unit_of_work.transaction() as transaction:
        book = transaction.books.get(BOOK_ID)
        assert book is not None
        transaction.books.save(replace(book, status=status))
        if with_run:
            transaction.runs.add(
                ProcessingRun(
                    id="run-1",
                    book_id=BOOK_ID,
                    status=ProcessingRunStatus.ACTIVE,
                    glossary_snapshot=GlossarySnapshot(),
                    llm_settings_snapshot=LLMSettingsSnapshot(
                        provider="fake", model="fake-1"
                    ),
                )
            )
        for task in tasks:
            transaction.tasks.add(task)


def status_of(unit_of_work: SqlAlchemyUnitOfWork) -> tuple[BookStatus, object]:
    """Состояние книги и итог её последней внешней проверки."""
    with unit_of_work.transaction() as transaction:
        book = transaction.books.get(BOOK_ID)
        assert book is not None
        return book.status, book.export_validation_status


def test_structure_only_export_needs_no_processing_results(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Технический цикл `import -> export` проходит без LLM и без обработки."""
    result = export(unit_of_work)(
        command(tmp_path, mode=ExportMode.STRUCTURE_ONLY)
    )

    assert Path(result.path).is_file()
    assert result.validation_status is ExportValidationStatus.NOT_VALIDATED
    assert status_of(unit_of_work) == (BookStatus.IMPORTED, None)


def test_full_export_is_rejected_before_processing_is_finished(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Полный экспорт недоступен книге, которая ещё обрабатывается."""
    make_ready(unit_of_work, status=BookStatus.PROCESSING)

    with pytest.raises(ExportBookError, match="обработка ещё не завершена"):
        export(unit_of_work)(command(tmp_path))


def test_full_export_is_rejected_while_mandatory_tasks_are_unfinished(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Незавершённая обязательная задача блокирует полный экспорт (PRD § 20)."""
    make_ready(
        unit_of_work,
        tasks=(
            ProcessingTask(
                id="task-1",
                book_id=BOOK_ID,
                task_type=TaskType.ANALYZE_CHAPTER,
                scope_id="section-1",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=TaskError(
                    type=ErrorType.TRANSIENT, message="Провайдер недоступен"
                ),
            ),
        ),
    )

    with pytest.raises(ExportBookError, match="незавершённые обязательные задачи"):
        export(unit_of_work)(command(tmp_path))


def test_finished_export_task_does_not_block_a_repeated_export(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Сама экспортная задача экспорт не блокирует."""
    make_ready(
        unit_of_work,
        tasks=(
            ProcessingTask(
                id="task-1",
                book_id=BOOK_ID,
                task_type=TaskType.EXPORT_BOOK,
                status=TaskStatus.RUNNING,
                attempts=1,
            ),
        ),
    )

    result = export(unit_of_work)(command(tmp_path))

    assert Path(result.path).is_file()


def test_successful_full_export_completes_the_book(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Успешный экспорт переводит книгу в завершённое состояние (ALG-06)."""
    make_ready(unit_of_work)

    result = export(unit_of_work)(command(tmp_path))

    assert result.mode is ExportMode.FULL
    assert status_of(unit_of_work) == (
        BookStatus.COMPLETED,
        ExportValidationStatus.NOT_VALIDATED,
    )


def test_unavailable_validator_leaves_the_book_unchecked(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Недоступный EPUBCheck экспорт не ломает: книга помечается непроверенной."""
    make_ready(unit_of_work)
    validator = FakeValidator(
        ExportValidationReport(available=False, unavailable_reason="Java не найдена")
    )

    result = export(unit_of_work, validator=validator)(command(tmp_path))

    assert result.validation_status is ExportValidationStatus.NOT_VALIDATED
    assert status_of(unit_of_work) == (
        BookStatus.COMPLETED,
        ExportValidationStatus.NOT_VALIDATED,
    )


def test_clean_validator_report_marks_the_book_as_checked(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Доступный валидатор без ошибок даёт книге отметку об успешной проверке."""
    make_ready(unit_of_work)
    validator = FakeValidator(ExportValidationReport(available=True))

    result = export(unit_of_work, validator=validator)(
        command(tmp_path, validation_report_path=str(tmp_path / "report.json"))
    )

    assert result.validation_status is ExportValidationStatus.PASSED
    assert result.validation_report_path == str(tmp_path / "report.json")
    assert status_of(unit_of_work) == (
        BookStatus.COMPLETED,
        ExportValidationStatus.PASSED,
    )


def test_validator_warnings_do_not_block_the_export(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Предупреждения внешней проверки экспорт не блокируют (PRD § 20)."""
    make_ready(unit_of_work)
    validator = FakeValidator(
        ExportValidationReport(
            available=True,
            issues=(
                ExportValidationIssue(
                    severity=ExportIssueSeverity.WARNING,
                    message="Обложка книги не объявлена",
                ),
            ),
        )
    )

    result = export(unit_of_work, validator=validator)(command(tmp_path))

    assert result.validation_status is ExportValidationStatus.PASSED


def test_strict_mode_turns_warnings_into_a_failed_export(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """В строгом режиме предупреждение делает проверку неуспешной."""
    make_ready(unit_of_work)
    validator = FakeValidator(
        ExportValidationReport(
            available=True,
            issues=(
                ExportValidationIssue(
                    severity=ExportIssueSeverity.WARNING,
                    message="Обложка книги не объявлена",
                ),
            ),
        )
    )

    with pytest.raises(ExportValidationRejectedError):
        export(unit_of_work, validator=validator)(
            command(tmp_path, strict_validation=True)
        )


def test_validator_errors_reject_the_export(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Ошибки внешней проверки делают экспорт неуспешным, а отметку — видимой."""
    make_ready(unit_of_work)
    validator = FakeValidator(
        ExportValidationReport(
            available=True,
            issues=(
                ExportValidationIssue(
                    severity=ExportIssueSeverity.ERROR,
                    message="RSC-005: несоответствие схеме",
                ),
            ),
        )
    )

    with pytest.raises(ExportValidationRejectedError, match="RSC-005"):
        export(unit_of_work, validator=validator)(command(tmp_path))

    assert status_of(unit_of_work) == (
        BookStatus.READY_TO_EXPORT,
        ExportValidationStatus.FAILED,
    )


def test_export_of_an_unknown_book_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Экспортировать книгу, которой нет в хранилище, нельзя."""
    with pytest.raises(ExportBookError, match="нет в хранилище"):
        export(unit_of_work)(command(tmp_path))
