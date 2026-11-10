"""Book export use case: modes, readiness, and external validation (P2.T5)."""

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
from linguary.application.context_builder import textual_blocks
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
    BlockProcessingResult,
    BlockTreatment,
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
    """External validator with a predetermined response."""

    report: ExportValidationReport

    def validate(self, request: ExportValidationRequest) -> ExportValidationReport:
        """Returns the prepared report regardless of book contents."""
        return replace(self.report, report_path=request.report_path)


@pytest.fixture
def imported(unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path) -> str:
    """Imported fixture book in the test repository."""
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
    """Export use case configured for the test store."""
    return ExportBook(
        unit_of_work=unit_of_work, writer=EpubExportWriter(), validator=validator
    )


def command(tmp_path: Path, **overrides: object) -> ExportBookCommand:
    """Export command to a file within test directory."""
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
    """Brings the book to a state where full export is available."""
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
            document = transaction.documents.get(BOOK_ID)
            transaction.results.replace_results('run-1', [
                BlockProcessingResult(block.id, BlockTreatment.READ_AS_IS)
                for section in document.sections for block in textual_blocks(section)
            ])
        for task in tasks:
            transaction.tasks.add(task)


def status_of(unit_of_work: SqlAlchemyUnitOfWork) -> tuple[BookStatus, object]:
    """Book status and the outcome of its latest external validation."""
    with unit_of_work.transaction() as transaction:
        book = transaction.books.get(BOOK_ID)
        assert book is not None
        return book.status, book.export_validation_status


def test_structure_only_export_needs_no_processing_results(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Technical `import -> export` cycle succeeds without LLM and without processing."""
    result = export(unit_of_work)(
        command(tmp_path, mode=ExportMode.STRUCTURE_ONLY)
    )

    assert Path(result.path).is_file()
    assert result.validation_status is ExportValidationStatus.NOT_VALIDATED
    assert status_of(unit_of_work) == (BookStatus.IMPORTED, None)


def test_full_export_is_rejected_before_processing_is_finished(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Full export is unavailable for a book that is still being processed."""
    make_ready(unit_of_work, status=BookStatus.PROCESSING)

    with pytest.raises(ExportBookError, match="обработка ещё не завершена"):
        export(unit_of_work)(command(tmp_path))


def test_full_export_is_rejected_while_mandatory_tasks_are_unfinished(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    """Unfinished mandatory tasks block full export (PRD § 20)."""
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
    """The export task itself does not block export."""
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
    """Successful export transitions the book to completed state (ALG-06)."""
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
    """Unavailable EPUBCheck does not fail export: the book is marked unvalidated."""
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
    """Available validator with no issues grants the book a passed status."""
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
    """External validation warnings do not block export (PRD § 20)."""
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
    """In strict mode, a warning marks validation as failed."""
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
    """External validation errors reject export and record failed status."""
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
    """Export of a book absent from the repository is rejected."""
    with pytest.raises(ExportBookError, match="нет в хранилище"):
        export(unit_of_work)(command(tmp_path))


def test_export_rejects_missing_or_stale_v2_evidence():
    from types import SimpleNamespace
    from linguary.application.use_cases.export_book import _validate_content_plans
    from linguary.application.retry_policy import ResultValidationError
    from linguary.domain import BlockProcessingResult, BlockTreatment
    from linguary.domain.content_plan import ContentPlan, ContentEvidence
    p = ContentPlan()
    document = SimpleNamespace(blocks=lambda: (SimpleNamespace(id='b', section_id='s'),), sections=(SimpleNamespace(id='s', is_notes_only=False),))
    analysis = SimpleNamespace(section_id='s', content_plan=p, before_you_read=(), treatment_plan=SimpleNamespace(items=(SimpleNamespace(block_id='b', treatment=BlockTreatment.READ_AS_IS), SimpleNamespace(block_id='missing', treatment=BlockTreatment.READ_AS_IS))))
    result = BlockProcessingResult('b', BlockTreatment.READ_AS_IS)
    for value in (None, ContentEvidence('0' * 64)):
        with pytest.raises(ResultValidationError):
            _validate_content_plans(document, (analysis,), (replace(result, content_evidence=value),))
    valid = replace(result, content_evidence=ContentEvidence(p.sha256))
    with pytest.raises(ResultValidationError):
        _validate_content_plans(document, (analysis,), (valid,))
    analysis.treatment_plan = SimpleNamespace(items=analysis.treatment_plan.items[:1])
    _validate_content_plans(document, (analysis,), (valid,))


@pytest.mark.asyncio
async def test_export_book_sequence_returns_active_task_without_rechecking_readiness(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork
) -> None:
    from linguary.application.use_cases.export_book import ExportBookSequence

    active_task = ProcessingTask(
        id="export-task-active",
        book_id=BOOK_ID,
        task_type=TaskType.EXPORT_BOOK,
        status=TaskStatus.RUNNING,
        attempts=1,
    )
    with unit_of_work.transaction() as transaction:
        transaction.tasks.add(active_task)

    ensure_ready_calls = 0

    class SpyExportBook:
        def ensure_ready(self, book_id: str) -> None:
            nonlocal ensure_ready_calls
            ensure_ready_calls += 1
            raise AssertionError("ensure_ready must not be called when export task is already active")

    class ForbiddenQueue:
        async def enqueue(self, identity):
            raise AssertionError("enqueue must not be called when export task is already active")

    sequence = ExportBookSequence(
        unit_of_work=unit_of_work,
        queue=ForbiddenQueue(),  # type: ignore[arg-type]
        export_book=SpyExportBook(),  # type: ignore[arg-type]
    )
    returned = await sequence.start(BOOK_ID)
    assert returned.id == active_task.id
    assert returned.status is TaskStatus.RUNNING
    assert ensure_ready_calls == 0


def test_export_book_reuses_validated_export_inputs_helper(
    imported: str, unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_ready(unit_of_work)
    exporter = export(unit_of_work)
    assert hasattr(exporter, "_load_validated_export_inputs")

    calls: list[str] = []
    original = exporter._load_validated_export_inputs

    def spy_load(transaction, book_id: str):
        calls.append(book_id)
        return original(transaction, book_id)

    monkeypatch.setattr(exporter, "_load_validated_export_inputs", spy_load)

    exporter.ensure_ready(BOOK_ID)
    assert calls == [BOOK_ID]

    exporter(command(tmp_path, mode=ExportMode.STRUCTURE_ONLY))
    assert calls == [BOOK_ID]

    exporter(command(tmp_path, mode=ExportMode.FULL))
    assert calls == [BOOK_ID, BOOK_ID]


def test_export_ensure_ready_and_downloadable_task_use_read_transaction(
    imported: str,
    engine,
    unit_of_work: SqlAlchemyUnitOfWork,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import event

    export_task = ProcessingTask(
        id="export-task-ok",
        book_id=BOOK_ID,
        task_type=TaskType.EXPORT_BOOK,
        status=TaskStatus.SUCCEEDED,
        attempts=1,
    )
    make_ready(
        unit_of_work,
        status=BookStatus.COMPLETED,
        tasks=(export_task,),
    )
    exporter = export(unit_of_work)

    tx_calls: list[str] = []
    original_tx = unit_of_work.transaction
    original_read_tx = unit_of_work.read_transaction

    def spy_transaction():
        tx_calls.append("write")
        return original_tx()

    def spy_read_transaction():
        tx_calls.append("read")
        return original_read_tx()

    monkeypatch.setattr(unit_of_work, "transaction", spy_transaction)
    monkeypatch.setattr(unit_of_work, "read_transaction", spy_read_transaction)

    executed_sql: list[str] = []

    def capture_sql(_conn, _cursor, statement: str, *_args) -> None:
        executed_sql.append(statement.strip().upper())

    event.listen(engine, "before_cursor_execute", capture_sql)
    try:
        exporter.ensure_ready(BOOK_ID)
        downloaded = exporter.downloadable_task(BOOK_ID)
    finally:
        event.remove(engine, "before_cursor_execute", capture_sql)

    assert downloaded.id == export_task.id
    assert tx_calls == ["read", "read"]
    assert not any("BEGIN IMMEDIATE" in sql for sql in executed_sql)
    assert not any(sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in executed_sql)

    fallback_calls = 0

    class _LegacyUoWProxy:
        def transaction(self):
            nonlocal fallback_calls
            fallback_calls += 1
            return original_read_tx()

    legacy_exporter = ExportBook(
        unit_of_work=_LegacyUoWProxy(),  # type: ignore[arg-type]
        writer=EpubExportWriter(),
    )
    legacy_exporter.ensure_ready(BOOK_ID)
    assert legacy_exporter.downloadable_task(BOOK_ID).id == export_task.id
    assert fallback_calls == 2

