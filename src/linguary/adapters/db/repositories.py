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
import json

from sqlalchemy import JSON, LargeBinary, cast, delete, func, or_, select, update
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from linguary.adapters.db import mapping
from linguary.adapters.db.models import (
    BlockResultRow,
    BookRow,
    ChapterAnalysisRow,
    AnalysisPassResultRow,
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
    AnalysisPassKind,
    AnalysisPassResult,
    LLMCallLog,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    TaskIdentity,
    TaskStatus,
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
        if not rows:
            return ()
        concept_ids = [row.id for row in rows]
        variant_rows = self._session.scalars(
            select(TermVariantRow)
            .where(TermVariantRow.concept_id.in_(concept_ids))
            .order_by(TermVariantRow.id)
        ).all()
        variants_by_concept: dict[str, list[TermVariantRow]] = {}
        for variant in variant_rows:
            variants_by_concept.setdefault(variant.concept_id, []).append(variant)
        return tuple(
            mapping.to_concept(row, variants_by_concept.get(row.id, ()))
            for row in rows
        )

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
        if not rows:
            return ()
        analysis_ids = [row.id for row in rows]
        plan_rows = self._session.scalars(
            select(TreatmentPlanItemRow)
            .where(TreatmentPlanItemRow.analysis_id.in_(analysis_ids))
            .order_by(TreatmentPlanItemRow.position)
        ).all()
        candidate_rows = self._session.scalars(
            select(GlossaryCandidateRow)
            .where(GlossaryCandidateRow.analysis_id.in_(analysis_ids))
            .order_by(GlossaryCandidateRow.id)
        ).all()
        plans_by_analysis: dict[str, list[TreatmentPlanItemRow]] = {}
        for item in plan_rows:
            plans_by_analysis.setdefault(item.analysis_id, []).append(item)
        candidates_by_analysis: dict[str, list[GlossaryCandidateRow]] = {}
        for candidate in candidate_rows:
            candidates_by_analysis.setdefault(candidate.analysis_id, []).append(candidate)
        return tuple(
            mapping.to_analysis(
                row,
                plans_by_analysis.get(row.id, ()),
                candidates_by_analysis.get(row.id, ()),
            )
            for row in rows
        )

    def save_pass(self, result: AnalysisPassResult) -> None:
        row = mapping.to_analysis_pass_row(result)
        now = datetime.now(UTC)
        values = {column.name: getattr(row, column.name) for column in AnalysisPassResultRow.__table__.columns
                  if column.name not in {"created_at", "updated_at"}}
        statement = insert(AnalysisPassResultRow).values(**values, created_at=now, updated_at=now)
        statement = statement.on_conflict_do_update(
            index_elements=["book_id", "section_id", "pass_kind", "scope_key"],
            set_={**{key: getattr(statement.excluded, key) for key in values}, "updated_at": now},
        )
        self._session.execute(statement)
        self._session.flush()
        self._expunge_pass_rows(
            book_id=result.book_id,
            section_id=result.section_id,
            pass_kind=result.pass_kind.value,
            scope_key=result.scope_key,
        )

    def get_pass(self, book_id: str, section_id: str, pass_kind: AnalysisPassKind,
                 *, scope_key: str = "") -> AnalysisPassResult | None:
        row = self._session.scalars(select(AnalysisPassResultRow).where(
            AnalysisPassResultRow.book_id == book_id, AnalysisPassResultRow.section_id == section_id,
            AnalysisPassResultRow.pass_kind == pass_kind.value, AnalysisPassResultRow.scope_key == scope_key,
        )).one_or_none()
        return None if row is None else mapping.to_analysis_pass_result(row)

    def list_passes_for_section(self, book_id: str, section_id: str) -> tuple[AnalysisPassResult, ...]:
        rows = self._session.scalars(select(AnalysisPassResultRow).where(
            AnalysisPassResultRow.book_id == book_id, AnalysisPassResultRow.section_id == section_id,
        ).order_by(AnalysisPassResultRow.pass_kind, AnalysisPassResultRow.scope_key)).all()
        return tuple(mapping.to_analysis_pass_result(row) for row in rows)

    def list_passes_for_book(
        self, book_id: str, *, pass_kind: AnalysisPassKind | None = None, scope_key: str | None = ""
    ) -> tuple[AnalysisPassResult, ...]:
        statement = select(AnalysisPassResultRow).where(AnalysisPassResultRow.book_id == book_id)
        if pass_kind is not None:
            statement = statement.where(AnalysisPassResultRow.pass_kind == pass_kind.value)
        if scope_key is not None:
            statement = statement.where(AnalysisPassResultRow.scope_key == scope_key)
        rows = self._session.scalars(
            statement.order_by(AnalysisPassResultRow.section_id, AnalysisPassResultRow.pass_kind, AnalysisPassResultRow.scope_key)
        ).all()
        return tuple(mapping.to_analysis_pass_result(row) for row in rows)

    def delete_fragment_passes_for_section(
        self, book_id: str, section_id: str, pass_kind: AnalysisPassKind
    ) -> None:
        statement = delete(AnalysisPassResultRow).where(
            AnalysisPassResultRow.book_id == book_id,
            AnalysisPassResultRow.section_id == section_id,
            AnalysisPassResultRow.pass_kind == pass_kind.value,
            AnalysisPassResultRow.scope_key != "",
        )
        self._session.execute(statement)
        self._session.flush()
        self._expunge_pass_rows(book_id=book_id, section_id=section_id, pass_kind=pass_kind.value)

    def delete_passes_for_book(self, book_id: str, *, pass_kind: AnalysisPassKind | None = None) -> None:
        statement = delete(AnalysisPassResultRow).where(AnalysisPassResultRow.book_id == book_id)
        if pass_kind is not None:
            statement = statement.where(AnalysisPassResultRow.pass_kind == pass_kind.value)
        self._session.execute(statement)
        self._session.flush()
        self._expunge_pass_rows(
            book_id=book_id,
            pass_kind=pass_kind.value if pass_kind is not None else None,
        )

    def _expunge_pass_rows(
        self,
        *,
        book_id: str,
        section_id: str | None = None,
        pass_kind: str | None = None,
        scope_key: str | None = None,
    ) -> None:
        for obj in list(self._session.identity_map.values()):
            if not isinstance(obj, AnalysisPassResultRow) or obj.book_id != book_id:
                continue
            if section_id is not None and obj.section_id != section_id:
                continue
            if pass_kind is not None and obj.pass_kind != pass_kind:
                continue
            if scope_key is not None and obj.scope_key != scope_key:
                continue
            self._session.expunge(obj)

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
        pronunciation_results = [result for result in results if result.pronunciation_assistances]
        if pronunciation_results:
            blocks = self._session.scalars(select(DocumentBlockRow).where(
                DocumentBlockRow.id.in_([result.block_id for result in pronunciation_results])
            )).all()
            run = self._session.get(ProcessingRunRow, run_id)
            texts = {block.id: block.text for block in blocks if run is not None and block.book_id == run.book_id}
            for result in pronunciation_results:
                for assistance in result.pronunciation_assistances:
                    assistance.anchor.validate(texts)
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

    def get_many(
        self, run_id: str, block_ids: Sequence[str]
    ) -> dict[str, BlockProcessingResult]:
        """Get processing results for multiple blocks within a run in a single batch."""
        unique_ids = tuple(dict.fromkeys(block_ids))
        if not unique_ids:
            return {}
        rows = self._session.scalars(
            select(BlockResultRow).where(
                BlockResultRow.run_id == run_id,
                BlockResultRow.block_id.in_(unique_ids),
            )
        ).all()
        if not rows:
            return {}
        found_ids = tuple(row.block_id for row in rows)
        notes = _by_block(
            self._session.scalars(
                select(LanguageNoteRow)
                .where(
                    LanguageNoteRow.run_id == run_id,
                    LanguageNoteRow.block_id.in_(found_ids),
                )
                .order_by(LanguageNoteRow.id)
            ).all()
        )
        warnings = _by_block(
            self._session.scalars(
                select(ResultWarningRow)
                .where(
                    ResultWarningRow.run_id == run_id,
                    ResultWarningRow.block_id.in_(found_ids),
                )
                .order_by(ResultWarningRow.id)
            ).all()
        )
        return {
            row.block_id: mapping.to_block_result(
                row,
                notes.get(row.block_id, ()),
                warnings.get(row.block_id, ()),
            )
            for row in rows
        }

    def pronunciation_assistances_for_blocks(
        self, run_id: str, block_ids: Sequence[str]
    ) -> dict[str, tuple]:
        """Return non-empty pronunciation assistances for the specified blocks without loading full results."""
        unique_ids = tuple(dict.fromkeys(block_ids))
        if not unique_ids:
            return {}
        out: dict[str, tuple] = {}
        for offset in range(0, len(unique_ids), 500):
            batch = unique_ids[offset : offset + 500]
            rows = self._session.execute(
                select(
                    BlockResultRow.block_id,
                    BlockResultRow.pronunciation_assistances_json,
                ).where(
                    BlockResultRow.run_id == run_id,
                    BlockResultRow.block_id.in_(batch),
                    BlockResultRow.pronunciation_assistances_json.is_not(None),
                )
            ).all()
            for block_id, raw_json in rows:
                decoded = mapping._decode_pronunciation(raw_json)
                if decoded:
                    out[block_id] = decoded
        return out

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

    def list_for_section(
        self, run_id: str, section_id: str
    ) -> tuple[BlockProcessingResult, ...]:
        """List results of a single chapter within a run in source document block order."""
        rows = self._session.scalars(
            select(BlockResultRow)
            .join(DocumentBlockRow, DocumentBlockRow.id == BlockResultRow.block_id)
            .where(
                BlockResultRow.run_id == run_id,
                DocumentBlockRow.section_id == section_id,
            )
            .order_by(DocumentBlockRow.order)
        ).all()
        notes = _by_block(self._section_notes(run_id, section_id))
        warnings = _by_block(self._section_warnings(run_id, section_id))
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

    def block_ids_for_run(self, run_id: str) -> frozenset[str]:
        """Return processed block IDs for a run without loading full block results."""
        return frozenset(
            self._session.scalars(
                select(BlockResultRow.block_id).where(BlockResultRow.run_id == run_id)
            ).all()
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

    def _section_notes(
        self, run_id: str, section_id: str
    ) -> Sequence[LanguageNoteRow]:
        """Get language notes for all blocks of a chapter in stable order."""
        return self._session.scalars(
            select(LanguageNoteRow)
            .join(DocumentBlockRow, DocumentBlockRow.id == LanguageNoteRow.block_id)
            .where(
                LanguageNoteRow.run_id == run_id,
                DocumentBlockRow.section_id == section_id,
            )
            .order_by(LanguageNoteRow.id)
        ).all()

    def _section_warnings(
        self, run_id: str, section_id: str
    ) -> Sequence[ResultWarningRow]:
        """Get warnings for all blocks of a chapter in stable order."""
        return self._session.scalars(
            select(ResultWarningRow)
            .join(DocumentBlockRow, DocumentBlockRow.id == ResultWarningRow.block_id)
            .where(
                ResultWarningRow.run_id == run_id,
                DocumentBlockRow.section_id == section_id,
            )
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
        """Save state with unchanged identity unless the task has been superseded."""
        row = mapping.to_task_row(task, datetime.now(UTC))
        result = self._session.execute(
            update(ProcessingTaskRow)
            .where(
                ProcessingTaskRow.id == task.id,
                ProcessingTaskRow.book_id == task.book_id,
                ProcessingTaskRow.task_type == task.task_type.value,
                ProcessingTaskRow.scope_id == task.scope_id,
                ProcessingTaskRow.operation_id == task.operation_id,
                ProcessingTaskRow.policy_version == task.policy_version,
                ProcessingTaskRow.status != TaskStatus.SUPERSEDED.value,
            )
            .values(
                status=row.status,
                attempts=row.attempts,
                error_type=row.error_type,
                error_message=row.error_message,
                error_detail=row.error_detail,
            )
        )
        if result.rowcount != 1:
            stored = self._session.execute(
                select(
                    ProcessingTaskRow.book_id,
                    ProcessingTaskRow.task_type,
                    ProcessingTaskRow.scope_id,
                    ProcessingTaskRow.operation_id,
                    ProcessingTaskRow.policy_version,
                    ProcessingTaskRow.status,
                ).where(ProcessingTaskRow.id == task.id)
            ).one_or_none()
            if stored is None:
                raise PersistenceError(f"задача {task.id} отсутствует в хранилище")
            if stored[:5] != (
                task.book_id,
                task.task_type.value,
                task.scope_id,
                task.operation_id,
                task.policy_version,
            ):
                raise PersistenceError(f"идентичность задачи {task.id} не совпадает с сохранённой")
            if stored.status == TaskStatus.SUPERSEDED.value:
                if task.status is TaskStatus.SUPERSEDED:
                    return
                raise PersistenceError(f"задача {task.id} вытеснена другой операцией")
            raise PersistenceError(f"задача {task.id} отсутствует в хранилище")
        self._session.flush()

    def get(self, task_id: str) -> ProcessingTask | None:
        """Get a task by identifier or return `None`."""
        row = self._session.get(ProcessingTaskRow, task_id)
        return None if row is None else mapping.to_task(row)

    def find_by_identity(self, identity: TaskIdentity) -> ProcessingTask | None:
        """Find a task by its complete operation and policy identity."""
        row = self._session.scalars(
            select(ProcessingTaskRow).where(
                ProcessingTaskRow.book_id == identity.book_id,
                ProcessingTaskRow.task_type == identity.task_type.value,
                ProcessingTaskRow.scope_id == identity.scope_id,
                ProcessingTaskRow.operation_id == identity.operation_id,
                ProcessingTaskRow.policy_version == identity.policy_version,
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

    def __init__(self, session: Session, *, policy=None, clock=None) -> None:
        self._session = session
        self.policy, self.clock = policy, clock

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
        if row is None:
            return None
        from linguary.application.diagnostics import DiagnosticsPolicy
        self.enforce_diagnostics_policy(self.policy or DiagnosticsPolicy(), self._now(), book_id=row.book_id)
        return row.response_text

    def list_for_book(self, book_id: str) -> tuple[LLMCallLog, ...]:
        """List log entries for a book in invocation order."""
        rows = self._session.execute(
            select(
                LLMCallLogRow.id,
                LLMCallLogRow.book_id,
                LLMCallLogRow.task_id,
                LLMCallLogRow.stage,
                LLMCallLogRow.provider,
                LLMCallLogRow.model,
                LLMCallLogRow.started_at,
                LLMCallLogRow.duration_ms,
                LLMCallLogRow.attempt,
                LLMCallLogRow.success,
                LLMCallLogRow.error_type,
                LLMCallLogRow.error_message,
                LLMCallLogRow.error_detail,
                LLMCallLogRow.prompt_tokens,
                LLMCallLogRow.completion_tokens,
                LLMCallLogRow.cost,
                LLMCallLogRow.attempt_id,
                func.json_remove(
                    LLMCallLogRow.diagnostic_meta_json,
                    "$.input_text",
                    type_=JSON(none_as_null=True),
                ).label("diagnostic_meta_json"),
            )
            .where(LLMCallLogRow.book_id == book_id)
            .order_by(LLMCallLogRow.started_at, LLMCallLogRow.id)
        ).all()
        return tuple(
            mapping.to_llm_call(
                row,
                diagnostic_meta=(
                    json.loads(row.diagnostic_meta_json)
                    if isinstance(row.diagnostic_meta_json, str)
                    else row.diagnostic_meta_json
                ),
            )
            for row in rows
        )

    def get_for_book(self, book_id: str, call_id: str) -> LLMCallLog | None:
        """Get a single log entry for a book with full diagnostic metadata, or `None`."""
        row = self._session.scalars(
            select(LLMCallLogRow).where(
                LLMCallLogRow.book_id == book_id,
                LLMCallLogRow.id == call_id,
            )
        ).one_or_none()
        return None if row is None else mapping.to_llm_call(row)


    def diagnostic_record(self, *, call_id=None, attempt_id=None):
        query = select(LLMCallLogRow)
        if attempt_id is not None:
            query = query.where(LLMCallLogRow.attempt_id == attempt_id)
        elif call_id is not None:
            query = query.where(LLMCallLogRow.id == call_id)
        else:
            return None
        row = self._session.scalar(query)
        if row is None:
            return None
        return {'call_id': row.id, 'attempt_id': row.attempt_id, 'book_id': row.book_id,
                'started_at': row.started_at, 'metadata': row.diagnostic_meta_json, 'body': row.response_text}

    def update_diagnostics(self, call_id, *, metadata):
        row = self._session.get(LLMCallLogRow, call_id)
        if row is None:
            raise PersistenceError('метаданные вызова отсутствуют')
        from copy import deepcopy
        from linguary.domain import ErrorType
        row.diagnostic_meta_json = deepcopy(metadata)
        decision = metadata.get('decision')
        if decision == 'accepted':
            row.success = True
            row.error_type = row.error_message = row.error_detail = None
        elif decision in {'rejected', 'stale'}:
            row.success = False
            row.error_type = row.error_type or ErrorType.VALIDATION.value
            row.error_message = row.error_message or ('Ответ устарел' if decision == 'stale' else 'Ответ не прошёл проверку')
        self._session.flush()

    def store_diagnostic_body(self, call_id, body):
        with self._session.begin_nested():
            row = self._session.get(LLMCallLogRow, call_id)
            if row is None:
                raise PersistenceError('метаданные вызова отсутствуют')
            row.response_text = body
            self._session.flush()

    def purge_diagnostic_bodies(self, book_id, *, call_id=None):
        has_body = or_(
            LLMCallLogRow.response_text.is_not(None),
            func.json_type(LLMCallLogRow.diagnostic_meta_json, '$.input_text').is_not(None),
        )
        query = select(LLMCallLogRow).where(LLMCallLogRow.book_id == book_id, has_body)
        if call_id is not None:
            query = query.where(LLMCallLogRow.id == call_id)
        rows = self._session.scalars(query).all()
        for row in rows:
            self._remove_body(row, 'deleted')
        self._session.flush()
        return len(rows)

    def _remove_body(self, row, status):
        metadata = dict(row.diagnostic_meta_json or {'schema_version': 1})
        metadata.pop('input_text', None)
        body_meta = dict(metadata.get('body_meta') or {})
        body_meta.update(body_status=status, stored_bytes=0)
        metadata['body_meta'] = body_meta
        row.diagnostic_meta_json = metadata
        row.response_text = None

    def enforce_diagnostics_policy(self, policy, now, *, book_id=None):
        from datetime import timedelta
        from linguary.application.diagnostics import bounded_body

        has_response_expr = LLMCallLogRow.response_text.is_not(None)
        has_input_expr = func.json_type(
            LLMCallLogRow.diagnostic_meta_json, '$.input_text'
        ).is_not(None)
        has_body = or_(has_response_expr, has_input_expr)
        response_bytes_expr = func.coalesce(
            func.length(cast(LLMCallLogRow.response_text, LargeBinary)), 0
        )
        input_bytes_expr = func.coalesce(
            func.length(
                cast(
                    func.json_extract(
                        LLMCallLogRow.diagnostic_meta_json, '$.input_text'
                    ),
                    LargeBinary,
                )
            ),
            0,
        )
        query = select(
            LLMCallLogRow.id,
            LLMCallLogRow.book_id,
            LLMCallLogRow.started_at,
            has_response_expr.label('has_response'),
            has_input_expr.label('has_input'),
            response_bytes_expr.label('response_bytes'),
            input_bytes_expr.label('input_bytes'),
            func.json_extract(
                LLMCallLogRow.diagnostic_meta_json, '$.body_meta.body_status'
            ).label('body_status'),
            func.json_extract(
                LLMCallLogRow.diagnostic_meta_json, '$.body_meta.stored_bytes'
            ).label('stored_bytes'),
            func.json_type(
                LLMCallLogRow.diagnostic_meta_json, '$.body_meta.stored_bytes'
            ).label('stored_bytes_type'),
            func.json_extract(
                LLMCallLogRow.diagnostic_meta_json, '$.body_meta.expires_at'
            ).label('expires_at'),
        ).where(has_body)
        if book_id is not None:
            query = query.where(LLMCallLogRow.book_id == book_id)
        projected = self._session.execute(
            query.order_by(LLMCallLogRow.started_at, LLMCallLogRow.id)
        ).all()
        if not projected:
            return

        remove_reasons: dict[str, str] = {}
        need_process: list[tuple[object, str, bool, list[object]]] = []
        surviving_by_book: dict[str, list[list[object]]] = {}
        for item in projected:
            status = item.body_status
            if status in {'expired', 'deleted', 'disabled'} or (
                status == 'unavailable' and bool(item.has_response)
            ):
                remove_reasons[item.id] = status
                continue
            if not policy.bodies_enabled:
                remove_reasons[item.id] = 'disabled'
                continue
            expires = item.started_at + timedelta(seconds=policy.retention_seconds)
            if now >= expires:
                remove_reasons[item.id] = 'expired'
                continue
            expires_at = expires.isoformat()
            has_resp = bool(item.has_response)
            has_inp = bool(item.has_input)
            resp_bytes = int(item.response_bytes or 0)
            inp_bytes = int(item.input_bytes or 0)
            if has_resp:
                already_bounded = (
                    status == 'available'
                    and item.stored_bytes_type == 'integer'
                    and isinstance(item.stored_bytes, int)
                    and not isinstance(item.stored_bytes, bool)
                    and item.stored_bytes <= policy.max_body_bytes
                    and (not has_inp or inp_bytes <= policy.max_body_bytes)
                )
            else:
                already_bounded = inp_bytes <= policy.max_body_bytes
            slot: list[object] = [item.id, (resp_bytes if has_resp else 0) + inp_bytes]
            surviving_by_book.setdefault(item.book_id, []).append(slot)
            if not already_bounded or item.expires_at != expires_at:
                need_process.append((item, expires_at, already_bounded, slot))

        rows_by_id: dict[str, LLMCallLogRow] = {}
        if need_process:
            process_ids = [item.id for item, _, _, _ in need_process]
            for row in self._session.scalars(
                select(LLMCallLogRow).where(LLMCallLogRow.id.in_(process_ids))
            ).all():
                rows_by_id[row.id] = row
            for item, expires_at, already_bounded, slot in need_process:
                row = rows_by_id.get(item.id)
                if row is None:
                    continue
                metadata = dict(row.diagnostic_meta_json or {'schema_version': 1})
                previous = dict(metadata.get('body_meta') or {})
                if isinstance(metadata.get('input_text'), str):
                    if len(metadata['input_text'].encode('utf-8')) > policy.max_body_bytes:
                        metadata['input_text'], _ = bounded_body(
                            metadata['input_text'], max_bytes=policy.max_body_bytes
                        )
                new_inp_bytes = (
                    len(metadata['input_text'].encode('utf-8'))
                    if isinstance(metadata.get('input_text'), str)
                    else 0
                )
                if row.response_text is not None:
                    if already_bounded:
                        metadata['body_meta'] = {**previous, 'expires_at': expires_at}
                        row.diagnostic_meta_json = metadata
                        slot[1] = len(row.response_text.encode('utf-8')) + new_inp_bytes
                    else:
                        body, body_meta = bounded_body(
                            row.response_text, max_bytes=policy.max_body_bytes
                        )
                        new_resp_bytes = len(body.encode('utf-8'))
                        metadata['body_meta'] = {
                            **body_meta,
                            **previous,
                            'body_status': 'available',
                            'stored_bytes': new_resp_bytes,
                            'expires_at': expires_at,
                        }
                        metadata['body_meta']['truncated'] = (
                            previous.get('truncated', False) or body_meta['truncated']
                        )
                        metadata['body_meta']['redacted'] = (
                            previous.get('redacted', False) or body_meta['redacted']
                        )
                        row.response_text, row.diagnostic_meta_json = body, metadata
                        slot[1] = new_resp_bytes + new_inp_bytes
                else:
                    metadata['body_meta'] = {**previous, 'expires_at': expires_at}
                    row.diagnostic_meta_json = metadata
                    slot[1] = new_inp_bytes

        for slots in surviving_by_book.values():
            total = sum(int(weight) for _, weight in slots)
            for call_id, weight in slots:
                if total <= policy.max_book_bytes:
                    break
                total -= int(weight)
                remove_reasons[str(call_id)] = 'deleted'

        if remove_reasons:
            missing_ids = [cid for cid in remove_reasons if cid not in rows_by_id]
            if missing_ids:
                for row in self._session.scalars(
                    select(LLMCallLogRow).where(LLMCallLogRow.id.in_(missing_ids))
                ).all():
                    rows_by_id[row.id] = row
            for cid, reason in remove_reasons.items():
                row = rows_by_id.get(cid)
                if row is not None:
                    self._remove_body(row, reason)

        if need_process or remove_reasons:
            self._session.flush()

    def _now(self):
        return self.clock() if self.clock is not None else datetime.now(UTC)


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
