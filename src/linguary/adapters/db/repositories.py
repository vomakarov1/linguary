"""SQLAlchemy-based storage port implementations.

Each repository operates on an already open session: the transaction boundary is defined
by `SqlAlchemyUnitOfWork`, not individual operations. Repositories accept and return
domain entities; database table rows are never exposed.

Technical timestamps (`created_at`) belong to the persistence layer: domain entities do not
contain them, so repositories assign them upon insertion and preserve them upon updates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from linguary.adapters.db import mapping
from linguary.adapters.db.models import (
    BlockResultRow,
    BookRow,
    ChapterAnalysisRow,
    DocumentBlockRow,
    GlossaryCandidateRow,
    ImportWarningRow,
    LanguageNoteRow,
    LLMCallLogRow,
    ProcessingRunRow,
    ProcessingTaskRow,
    ResourceRow,
    ResultWarningRow,
    SectionRow,
    TermConceptRow,
    TermVariantRow,
    TreatmentPlanItemRow,
)
from linguary.application.ports.persistence import BookRecord, PersistenceError
from linguary.domain import (
    BlockProcessingResult,
    BookImportWarning,
    CanonicalDocument,
    ChapterAnalysis,
    LLMCallLog,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    TaskIdentity,
    TermConcept,
    TaskError,
)


class SqlAlchemyBookRepository:
    """Books and their user-facing state."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, book: BookRecord) -> None:
        """Add a new book."""
        self._session.add(mapping.to_book_row(book))
        self._session.flush()

    def save(self, book: BookRecord) -> None:
        """Overwrite an existing book completely."""
        self._session.merge(mapping.to_book_row(book))
        self._session.flush()

    def get(self, book_id: str) -> BookRecord | None:
        """Get a book by identifier or return `None` if not found."""
        row = self._session.get(BookRow, book_id)
        return None if row is None else mapping.to_book_record(row)

    def list_all(self) -> tuple[BookRecord, ...]:
        """List all books in insertion order."""
        rows = self._session.scalars(
            select(BookRow).order_by(BookRow.created_at, BookRow.id)
        ).all()
        return tuple(mapping.to_book_record(row) for row in rows)


class SqlAlchemyDocumentRepository:
    """Canonical document model and import warnings."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        document: CanonicalDocument,
        *,
        stored_resource_paths: Mapping[str, str] | None = None,
    ) -> None:
        """Save the document completely, replacing previous sections, blocks, and resources.

        Sections are written in a separate step: due to the self-referential
        `document_blocks.parent_id`, SQLAlchemy cannot automatically order table inserts,
        and blocks would otherwise be sent to the database before their sections.
        """
        paths = stored_resource_paths or {}
        self._session.execute(
            delete(SectionRow).where(SectionRow.book_id == document.id)
        )
        self._session.execute(
            delete(ResourceRow).where(ResourceRow.book_id == document.id)
        )
        self._session.flush()
        for section in document.ordered_sections:
            self._session.add(mapping.to_section_row(document.id, section))
        self._session.flush()
        for section in document.ordered_sections:
            for block in section.ordered_blocks:
                self._session.add(mapping.to_block_row(document.id, block))
        for resource in document.resources:
            self._session.add(
                mapping.to_resource_row(
                    document.id, resource, paths.get(resource.source_path)
                )
            )
        self._session.flush()

    def get(self, book_id: str) -> CanonicalDocument | None:
        """Get canonical document model for a book or `None` if the book does not exist."""
        book = self._session.get(BookRow, book_id)
        if book is None:
            return None
        sections = self._session.scalars(
            select(SectionRow).where(SectionRow.book_id == book_id)
        ).all()
        blocks = self._session.scalars(
            select(DocumentBlockRow).where(DocumentBlockRow.book_id == book_id)
        ).all()
        resources = self._session.scalars(
            select(ResourceRow)
            .where(ResourceRow.book_id == book_id)
            .order_by(ResourceRow.source_path)
        ).all()
        return mapping.to_document(book, sections, blocks, resources)

    def stored_resource_paths(self, book_id: str) -> Mapping[str, str]:
        """Map of stored book resource paths by their source paths."""
        rows = self._session.execute(
            select(ResourceRow.source_path, ResourceRow.stored_path).where(
                ResourceRow.book_id == book_id, ResourceRow.stored_path.is_not(None)
            )
        ).all()
        return {source_path: stored_path for source_path, stored_path in rows}

    def add_import_warnings(
        self, book_id: str, warnings: Sequence[BookImportWarning]
    ) -> None:
        """Add book import warnings."""
        self._session.add_all(
            mapping.to_import_warning_row(book_id, warning) for warning in warnings
        )
        self._session.flush()

    def import_warnings(self, book_id: str) -> tuple[BookImportWarning, ...]:
        """Get book import warnings."""
        rows = self._session.scalars(
            select(ImportWarningRow)
            .where(ImportWarningRow.book_id == book_id)
            .order_by(ImportWarningRow.id)
        ).all()
        return tuple(mapping.to_import_warning(row) for row in rows)


class SqlAlchemyGlossaryRepository:
    """Book glossary concepts together with their language variants."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, concept: TermConcept) -> None:
        """Save a concept with its variants, replacing any existing record."""
        self._session.execute(
            delete(TermVariantRow).where(TermVariantRow.concept_id == concept.id)
        )
        self._session.merge(mapping.to_concept_row(concept))
        self._session.add_all(
            mapping.to_variant_row(variant) for variant in concept.variants
        )
        self._session.flush()

    def get(self, concept_id: str) -> TermConcept | None:
        """Get a concept by identifier or return `None`."""
        row = self._session.get(TermConceptRow, concept_id)
        return None if row is None else mapping.to_concept(row, self._variants(concept_id))

    def list_for_book(self, book_id: str) -> tuple[TermConcept, ...]:
        """List all glossary concepts for a book."""
        rows = self._session.scalars(
            select(TermConceptRow)
            .where(TermConceptRow.book_id == book_id)
            .order_by(TermConceptRow.canonical_source_text, TermConceptRow.id)
        ).all()
        return tuple(mapping.to_concept(row, self._variants(row.id)) for row in rows)

    def delete(self, concept_id: str) -> None:
        """Delete a concept along with its variants."""
        self._session.execute(
            delete(TermConceptRow).where(TermConceptRow.id == concept_id)
        )
        self._session.flush()

    def _variants(self, concept_id: str) -> Sequence[TermVariantRow]:
        """Get concept variants in stable order."""
        return self._session.scalars(
            select(TermVariantRow)
            .where(TermVariantRow.concept_id == concept_id)
            .order_by(TermVariantRow.id)
        ).all()


class SqlAlchemyChapterAnalysisRepository:
    """Chapter analysis: treatment plan, glossary candidates, and chapter materials."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, book_id: str, analysis: ChapterAnalysis) -> None:
        """Save chapter analysis, replacing any previous analysis of the same chapter."""
        self._session.execute(
            delete(ChapterAnalysisRow).where(
                ChapterAnalysisRow.book_id == book_id,
                ChapterAnalysisRow.section_id == analysis.section_id,
            )
        )
        self._session.flush()
        self._session.add(mapping.to_analysis_row(book_id, analysis))
        self._session.add_all(mapping.to_plan_item_rows(analysis))
        self._session.add_all(
            mapping.to_candidate_row(candidate)
            for candidate in analysis.glossary_candidates
        )
        self._session.flush()

    def get_for_section(self, book_id: str, section_id: str) -> ChapterAnalysis | None:
        """Get analysis for a specific chapter or `None` if the chapter has not been analyzed."""
        row = self._session.scalars(
            select(ChapterAnalysisRow).where(
                ChapterAnalysisRow.book_id == book_id,
                ChapterAnalysisRow.section_id == section_id,
            )
        ).one_or_none()
        return None if row is None else self._to_analysis(row)

    def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
        """List analyses for all analyzed chapters of a book in reading order."""
        rows = self._session.scalars(
            select(ChapterAnalysisRow)
            .join(SectionRow, SectionRow.id == ChapterAnalysisRow.section_id)
            .where(ChapterAnalysisRow.book_id == book_id)
            .order_by(SectionRow.order)
        ).all()
        return tuple(self._to_analysis(row) for row in rows)

    def _to_analysis(self, row: ChapterAnalysisRow) -> ChapterAnalysis:
        """Assemble domain analysis along with treatment plan and candidates."""
        plan_items = self._session.scalars(
            select(TreatmentPlanItemRow)
            .where(TreatmentPlanItemRow.analysis_id == row.id)
            .order_by(TreatmentPlanItemRow.position)
        ).all()
        candidates = self._session.scalars(
            select(GlossaryCandidateRow)
            .where(GlossaryCandidateRow.analysis_id == row.id)
            .order_by(GlossaryCandidateRow.id)
        ).all()
        return mapping.to_analysis(row, plan_items, candidates)


    def delete_for_book(self, book_id: str) -> None:
        """Remove current analysis projections, preserving work archives."""
        self._session.execute(delete(ChapterAnalysisRow).where(ChapterAnalysisRow.book_id == book_id))
        self._session.flush()


class SqlAlchemyProcessingRunRepository:
    """Book processing runs."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, run: ProcessingRun) -> None:
        """Add a new processing run."""
        self._session.add(mapping.to_run_row(run, datetime.now(UTC)))
        self._session.flush()

    def save(self, run: ProcessingRun) -> None:
        """Update an existing processing run, preserving its creation timestamp."""
        row = mapping.to_run_row(run, datetime.now(UTC))
        result = self._session.execute(
            update(ProcessingRunRow)
            .where(ProcessingRunRow.id == run.id)
            .values(
                status=row.status,
                glossary_snapshot_json=row.glossary_snapshot_json,
                llm_settings_json=row.llm_settings_json,
            )
        )
        if result.rowcount != 1:
            raise PersistenceError(f"запуск обработки {run.id} отсутствует в хранилище")
        self._session.flush()

    def get(self, run_id: str) -> ProcessingRun | None:
        """Get a processing run by identifier or return `None`."""
        row = self._session.get(ProcessingRunRow, run_id)
        return None if row is None else mapping.to_run(row)

    def active_for_book(self, book_id: str) -> ProcessingRun | None:
        """Get the single active run for a book or return `None`.

        Uniqueness is guaranteed by the schema's partial unique index.
        """
        row = self._session.scalars(
            select(ProcessingRunRow).where(
                ProcessingRunRow.book_id == book_id,
                ProcessingRunRow.status == ProcessingRunStatus.ACTIVE.value,
            )
        ).one_or_none()
        return None if row is None else mapping.to_run(row)

    def delete(self, run_id: str) -> None:
        """Delete a run along with block results, language notes, and warnings."""
        self._session.execute(
            delete(ProcessingRunRow).where(ProcessingRunRow.id == run_id)
        )
        self._session.flush()


class SqlAlchemyBlockResultRepository:
    """Block processing results along with language notes and warnings."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_results(
        self, run_id: str, results: Sequence[BlockProcessingResult]
    ) -> None:
        """Replace results for the specified blocks in a single operation (PRD § 17)."""
        block_ids = [result.block_id for result in results]
        if not block_ids:
            return
        for table in (BlockResultRow, LanguageNoteRow, ResultWarningRow):
            self._session.execute(
                delete(table).where(
                    table.run_id == run_id, table.block_id.in_(block_ids)
                )
            )
        self._session.flush()
        for result in results:
            self._session.add(mapping.to_result_row(run_id, result))
            self._session.add_all(mapping.to_note_rows(run_id, result))
            self._session.add_all(mapping.to_result_warning_rows(run_id, result))
        self._session.flush()

    def get(self, run_id: str, block_id: str) -> BlockProcessingResult | None:
        """Get the processing result of a single block within a run."""
        row = self._session.scalars(
            select(BlockResultRow).where(
                BlockResultRow.run_id == run_id, BlockResultRow.block_id == block_id
            )
        ).one_or_none()
        if row is None:
            return None
        return mapping.to_block_result(
            row, self._notes(run_id, block_id), self._warnings(run_id, block_id)
        )

    def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
        """List all results of a run in source document block order.

        Language notes and warnings are fetched via two bulk queries for the entire run
        and grouped in memory: per-result queries would otherwise turn book loading
        into thousands of database roundtrips.
        """
        rows = self._session.scalars(
            select(BlockResultRow)
            .join(DocumentBlockRow, DocumentBlockRow.id == BlockResultRow.block_id)
            .join(SectionRow, SectionRow.id == DocumentBlockRow.section_id)
            .where(BlockResultRow.run_id == run_id)
            .order_by(SectionRow.order, DocumentBlockRow.order)
        ).all()
        notes = _by_block(self._run_notes(run_id))
        warnings = _by_block(self._run_warnings(run_id))
        return tuple(
            mapping.to_block_result(
                row,
                notes.get(row.block_id, ()),
                warnings.get(row.block_id, ()),
            )
            for row in rows
        )

    def count_for_run(self, run_id: str) -> int:
        """Count how many blocks in the run have already been processed.

        The book card is updated via polling and only requires the result count,
        so counting is executed by the database rather than domain object assembly.
        """
        return (
            self._session.scalar(
                select(func.count())
                .select_from(BlockResultRow)
                .where(BlockResultRow.run_id == run_id)
            )
            or 0
        )

    def _notes(self, run_id: str, block_id: str) -> Sequence[LanguageNoteRow]:
        """Get block language notes in stable order."""
        return self._session.scalars(
            select(LanguageNoteRow)
            .where(LanguageNoteRow.run_id == run_id, LanguageNoteRow.block_id == block_id)
            .order_by(LanguageNoteRow.id)
        ).all()

    def _warnings(self, run_id: str, block_id: str) -> Sequence[ResultWarningRow]:
        """Get block result warnings in stable order."""
        return self._session.scalars(
            select(ResultWarningRow)
            .where(
                ResultWarningRow.run_id == run_id, ResultWarningRow.block_id == block_id
            )
            .order_by(ResultWarningRow.id)
        ).all()

    def _run_notes(self, run_id: str) -> Sequence[LanguageNoteRow]:
        """Get language notes for all blocks in a run in the same order as per-block notes."""
        return self._session.scalars(
            select(LanguageNoteRow)
            .where(LanguageNoteRow.run_id == run_id)
            .order_by(LanguageNoteRow.id)
        ).all()

    def _run_warnings(self, run_id: str) -> Sequence[ResultWarningRow]:
        """Get warnings for all blocks in a run in the same order as per-block warnings."""
        return self._session.scalars(
            select(ResultWarningRow)
            .where(ResultWarningRow.run_id == run_id)
            .order_by(ResultWarningRow.id)
        ).all()


class SqlAlchemyTaskRepository:
    """Local resumable processing tasks."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, task: ProcessingTask) -> None:
        """Add a task; duplicate idempotency keys are rejected by the store."""
        self._session.add(mapping.to_task_row(task, datetime.now(UTC)))
        self._session.flush()

    def save(self, task: ProcessingTask) -> None:
        """Save updated task state without modifying its creation timestamp."""
        row = mapping.to_task_row(task, datetime.now(UTC))
        result = self._session.execute(
            update(ProcessingTaskRow)
            .where(ProcessingTaskRow.id == task.id)
            .values(
                status=row.status,
                attempts=row.attempts,
                error_type=row.error_type,
                error_message=row.error_message,
                error_detail=row.error_detail,
            )
        )
        if result.rowcount != 1:
            raise PersistenceError(f"задача {task.id} отсутствует в хранилище")
        self._session.flush()

    def get(self, task_id: str) -> ProcessingTask | None:
        """Get a task by identifier or return `None`."""
        row = self._session.get(ProcessingTaskRow, task_id)
        return None if row is None else mapping.to_task(row)

    def find_by_identity(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Find a task by its idempotency key `book_id + task_type + scope_id`."""
        row = self._session.scalars(
            select(ProcessingTaskRow).where(
                ProcessingTaskRow.book_id == identity.book_id,
                ProcessingTaskRow.task_type == identity.task_type.value,
                ProcessingTaskRow.scope_id == identity.scope_id,
            )
        ).one_or_none()
        return None if row is None else mapping.to_task(row)

    def list_for_book(self, book_id: str) -> tuple[ProcessingTask, ...]:
        """List tasks for a book in creation order."""
        rows = self._session.scalars(
            select(ProcessingTaskRow)
            .where(ProcessingTaskRow.book_id == book_id)
            .order_by(ProcessingTaskRow.created_at, ProcessingTaskRow.id)
        ).all()
        return tuple(mapping.to_task(row) for row in rows)

    def delete(self, task_id: str) -> None:
        """Delete a task along with its idempotency key."""
        self._session.execute(
            delete(ProcessingTaskRow).where(ProcessingTaskRow.id == task_id)
        )
        self._session.flush()


class SqlAlchemyLLMCallLogRepository:
    """Technical log of LLM calls."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, entry: LLMCallLog, *, response_text: str | None = None) -> None:
        """Add a log entry."""
        row = mapping.to_llm_call_row(entry)
        row.response_text = response_text
        self._session.add(row)
        self._session.flush()

    def mark_rejected(
        self, call_id: str, *, error: TaskError, response_text: str | None
    ) -> None:
        """Mark a rejected response, preserving token usage and cost."""
        row = self._session.get(LLMCallLogRow, call_id)
        if row is None:
            raise PersistenceError(f"запись вызова LLM {call_id} не найдена")
        row.success = False
        row.error_type = error.type.value
        row.error_message = error.message
        row.error_detail = error.detail
        row.response_text = response_text
        self._session.flush()

    def response_text_for(self, call_id: str) -> str | None:
        """Retrieve stored response text if it was recorded."""
        row = self._session.get(LLMCallLogRow, call_id)
        return None if row is None else row.response_text

    def list_for_book(self, book_id: str) -> tuple[LLMCallLog, ...]:
        """List log entries for a book in invocation order."""
        rows = self._session.scalars(
            select(LLMCallLogRow)
            .where(LLMCallLogRow.book_id == book_id)
            .order_by(LLMCallLogRow.started_at, LLMCallLogRow.id)
        ).all()
        return tuple(mapping.to_llm_call(row) for row in rows)


def _by_block[RowT: (LanguageNoteRow, ResultWarningRow)](
    rows: Iterable[RowT],
) -> Mapping[str, Sequence[RowT]]:
    """Rows grouped by block while preserving original order."""
    grouped: dict[str, list[RowT]] = {}
    for row in rows:
        grouped.setdefault(row.block_id, []).append(row)
    return grouped


__all__ = [
    "SqlAlchemyBlockResultRepository",
    "SqlAlchemyBookRepository",
    "SqlAlchemyChapterAnalysisRepository",
    "SqlAlchemyDocumentRepository",
    "SqlAlchemyGlossaryRepository",
    "SqlAlchemyLLMCallLogRepository",
    "SqlAlchemyProcessingRunRepository",
    "SqlAlchemyTaskRepository",
]
