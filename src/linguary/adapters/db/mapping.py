"""Mapping between domain entities and database table rows.

The domain consists of immutable dataclasses and has no knowledge of SQLAlchemy, so the
"domain - table" mapping is extracted into separate pure functions: they do not touch the
session and perform no queries. Repositories compose read and write operations from them.

Domain `StrEnum` values match the PRD external representation and are stored as raw strings;
upon reading, strings are parsed back into enum members.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from dataclasses import asdict
from typing import Any

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
from linguary.application.ports.persistence import BookRecord, ExportValidationStatus, PersistenceError
from linguary.domain.content_plan import (
    ContentPlan, ContentEvidence, Difficulty, SourceAnchor, RequiredPoint, HelpEvidence,
)
from linguary.domain.errors import DomainError
from linguary.domain import (
    BeforeYouReadEntry,
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookImportWarning,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    CoverageStatus,
    DocumentBlock,
    DocumentResource,
    DocumentSection,
    ErrorType,
    GlossaryCandidate,
    GlossaryHint,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    ImportWarningKind,
    InlineAnnotation,
    InlineAnnotationType,
    LanguageNote,
    LLMCallLog,
    LLMCallStage,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    ResultWarning,
    ResultWarningKind,
    SourceReference,
    SupportStatus,
    TaskError,
    TaskStatus,
    TaskType,
    TermConcept,
    TermStatus,
    TermVariant,
    TokenUsage,
    TreatmentPlan,
    TreatmentPlanItem,
)

# --- book -------------------------------------------------------------------------


def to_book_row(record: BookRecord) -> BookRow:
    """Convert an application book record to a database row."""
    validation = record.export_validation_status
    return BookRow(
        id=record.id,
        title=record.metadata.title,
        authors_json=list(record.metadata.authors),
        identifier=record.metadata.identifier,
        publisher=record.metadata.publisher,
        source_language=record.source_language,
        target_language=record.target_language,
        status=record.status.value,
        source_epub_path=record.source_epub_path,
        export_validation_status=None if validation is None else validation.value,
        created_at=record.created_at,
    )


def to_book_record(row: BookRow) -> BookRecord:
    """Convert a database row to an application book record."""
    validation = row.export_validation_status
    return BookRecord(
        id=row.id,
        metadata=BookMetadata(
            title=row.title,
            authors=tuple(row.authors_json),
            identifier=row.identifier,
            publisher=row.publisher,
        ),
        source_language=row.source_language,
        target_language=row.target_language,
        status=BookStatus(row.status),
        source_epub_path=row.source_epub_path,
        created_at=row.created_at,
        export_validation_status=(
            None if validation is None else ExportValidationStatus(validation)
        ),
    )


# --- canonical document model -------------------------------------------------


def to_section_row(book_id: str, section: DocumentSection) -> SectionRow:
    """Convert a document section to a database row."""
    return SectionRow(
        id=section.id,
        book_id=book_id,
        title=section.title,
        order=section.order,
        source_resource_path=section.source_resource_path,
    )


def to_block_row(book_id: str, block: DocumentBlock) -> DocumentBlockRow:
    """Convert a document block to a database row."""
    return DocumentBlockRow(
        id=block.id,
        book_id=book_id,
        section_id=block.section_id,
        type=block.type.value,
        order=block.order,
        text=block.text,
        inline_annotations_json=[
            {
                "type": annotation.type.value,
                "start": annotation.start,
                "end": annotation.end,
                "target": annotation.target,
            }
            for annotation in block.inline_annotations
        ],
        attributes_json=dict(block.attributes),
        parent_id=block.parent_id,
        source_reference_json={
            "resource_path": block.source_reference.resource_path,
            "structural_path": block.source_reference.structural_path,
            "element_id": block.source_reference.element_id,
        },
        support_status=block.support_status.value,
    )


def to_block(row: DocumentBlockRow) -> DocumentBlock:
    """Convert a database row to a domain document block."""
    return DocumentBlock(
        id=row.id,
        section_id=row.section_id,
        type=BlockType(row.type),
        order=row.order,
        text=row.text,
        source_reference=SourceReference(**row.source_reference_json),
        support_status=SupportStatus(row.support_status),
        inline_annotations=tuple(
            InlineAnnotation(
                type=InlineAnnotationType(payload["type"]),
                start=payload["start"],
                end=payload["end"],
                target=payload["target"],
            )
            for payload in row.inline_annotations_json
        ),
        parent_id=row.parent_id,
        attributes=dict(row.attributes_json),
    )


def to_resource_row(
    book_id: str, resource: DocumentResource, stored_path: str | None
) -> ResourceRow:
    """Convert a book resource to a database row along with its stored path."""
    return ResourceRow(
        id=resource.id,
        book_id=book_id,
        source_path=resource.source_path,
        media_type=resource.media_type,
        stored_path=stored_path,
        support_status=resource.support_status.value,
    )


def to_resource(row: ResourceRow) -> DocumentResource:
    """Convert a database row to a domain document resource."""
    return DocumentResource(
        id=row.id,
        source_path=row.source_path,
        media_type=row.media_type,
        support_status=SupportStatus(row.support_status),
    )


def to_document(
    book: BookRow,
    sections: Sequence[SectionRow],
    blocks: Sequence[DocumentBlockRow],
    resources: Sequence[ResourceRow],
) -> CanonicalDocument:
    """Reconstruct a canonical document model from rows across all its tables."""
    blocks_by_section: dict[str, list[DocumentBlockRow]] = {}
    for block in blocks:
        blocks_by_section.setdefault(block.section_id, []).append(block)
    return CanonicalDocument(
        id=book.id,
        metadata=BookMetadata(
            title=book.title,
            authors=tuple(book.authors_json),
            identifier=book.identifier,
            publisher=book.publisher,
        ),
        source_language=book.source_language,
        target_language=book.target_language,
        sections=tuple(
            DocumentSection(
                id=section.id,
                document_id=book.id,
                title=section.title,
                order=section.order,
                source_resource_path=section.source_resource_path,
                blocks=tuple(
                    to_block(block)
                    for block in sorted(
                        blocks_by_section.get(section.id, []), key=lambda row: row.order
                    )
                ),
            )
            for section in sorted(sections, key=lambda row: row.order)
        ),
        resources=tuple(to_resource(resource) for resource in resources),
    )


def to_import_warning_row(book_id: str, warning: BookImportWarning) -> ImportWarningRow:
    """Convert an import warning to a database row."""
    return ImportWarningRow(
        book_id=book_id,
        section_id=warning.section_id,
        block_id=warning.block_id,
        kind=warning.kind.value,
        message=warning.message,
    )


def to_import_warning(row: ImportWarningRow) -> BookImportWarning:
    """Convert a database row to a domain import warning."""
    return BookImportWarning(
        kind=ImportWarningKind(row.kind),
        message=row.message,
        section_id=row.section_id,
        block_id=row.block_id,
    )


# --- glossary ---------------------------------------------------------------------


def to_concept_row(concept: TermConcept) -> TermConceptRow:
    """Convert a glossary term concept to a database row."""
    return TermConceptRow(
        id=concept.id,
        book_id=concept.book_id,
        type=concept.type.value,
        canonical_source_text=concept.canonical_source_text,
        status=concept.status.value,
        note=concept.note,
    )


def to_variant_row(variant: TermVariant) -> TermVariantRow:
    """Convert a term variant to a database row."""
    return TermVariantRow(
        concept_id=variant.concept_id,
        language=variant.language,
        text=variant.text,
        status=variant.status.value,
    )


def to_concept(row: TermConceptRow, variants: Sequence[TermVariantRow]) -> TermConcept:
    """Convert a concept row and variant rows to a domain term concept."""
    return TermConcept(
        id=row.id,
        book_id=row.book_id,
        type=ConceptType(row.type),
        canonical_source_text=row.canonical_source_text,
        status=TermStatus(row.status),
        note=row.note or "",
        variants=tuple(
            TermVariant(
                concept_id=variant.concept_id,
                language=variant.language,
                text=variant.text,
                status=TermStatus(variant.status),
            )
            for variant in variants
        ),
    )


# --- chapter analysis ------------------------------------------------------------------


def to_analysis_row(book_id: str, analysis: ChapterAnalysis) -> ChapterAnalysisRow:
    """Convert chapter analysis to a database row with chapter study materials."""
    return ChapterAnalysisRow(
        id=analysis.id,
        book_id=book_id,
        section_id=analysis.section_id,
        coverage_status=analysis.coverage_status.value,
        translation_instructions=analysis.translation_instructions,
        chapter_summary=analysis.chapter_summary,
        content_plan_json=_encode_content_plan(analysis.content_plan),
        before_you_read_json=[
            {"expression": entry.expression, "explanation": entry.explanation}
            for entry in analysis.before_you_read
        ],
    )


def to_plan_item_rows(analysis: ChapterAnalysis) -> list[TreatmentPlanItemRow]:
    """Convert chapter treatment plan items to database rows preserving order."""
    return [
        TreatmentPlanItemRow(
            analysis_id=analysis.id,
            block_id=item.block_id,
            treatment=item.treatment.value,
            position=position,
        )
        for position, item in enumerate(analysis.treatment_plan.items)
    ]


def to_candidate_row(candidate: GlossaryCandidate) -> GlossaryCandidateRow:
    """Convert a glossary candidate to a database row."""
    return GlossaryCandidateRow(
        id=candidate.id,
        analysis_id=candidate.analysis_id,
        source_text=candidate.source_text,
        type=candidate.type.value,
        suggested_translation=candidate.suggested_translation,
        confidence=candidate.confidence,
        model_flagged_review=candidate.model_flagged_review,
        requires_review=candidate.requires_review,
        note=candidate.note,
    )


def to_analysis(
    row: ChapterAnalysisRow,
    plan_items: Sequence[TreatmentPlanItemRow],
    candidates: Sequence[GlossaryCandidateRow],
) -> ChapterAnalysis:
    """Reconstruct domain chapter analysis from database rows."""
    return ChapterAnalysis(
        id=row.id,
        section_id=row.section_id,
        coverage_status=CoverageStatus(row.coverage_status),
        treatment_plan=TreatmentPlan(
            section_id=row.section_id,
            items=tuple(
                TreatmentPlanItem(
                    block_id=item.block_id, treatment=BlockTreatment(item.treatment)
                )
                for item in sorted(plan_items, key=lambda item: item.position)
            ),
        ),
        translation_instructions=row.translation_instructions,
        chapter_summary=row.chapter_summary,
        glossary_candidates=tuple(
            GlossaryCandidate(
                id=candidate.id,
                analysis_id=candidate.analysis_id,
                source_text=candidate.source_text,
                type=ConceptType(candidate.type),
                suggested_translation=candidate.suggested_translation,
                confidence=candidate.confidence,
                model_flagged_review=candidate.model_flagged_review,
                requires_review=candidate.requires_review,
                note=candidate.note or "",
            )
            for candidate in candidates
        ),
        content_plan=_decode_content_plan(row.content_plan_json),
        before_you_read=tuple(
            BeforeYouReadEntry(
                expression=entry["expression"], explanation=entry["explanation"]
            )
            for entry in row.before_you_read_json
        ),
    )


# --- processing run and results ----------------------------------------------


def to_run_row(run: ProcessingRun, created_at: datetime) -> ProcessingRunRow:
    """Convert a processing run to a database row with glossary and LLM settings snapshots."""
    settings = run.llm_settings_snapshot
    return ProcessingRunRow(
        id=run.id,
        book_id=run.book_id,
        status=run.status.value,
        glossary_snapshot_json=[
            {
                "source_text": entry.source_text,
                "translation": entry.translation,
                "type": entry.type.value,
                "status": entry.status.value,
                "note": entry.note,
            }
            for entry in run.glossary_snapshot.entries
        ],
        llm_settings_json={
            "provider": settings.provider,
            "model": settings.model,
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
        },
        created_at=created_at,
    )


def to_run(row: ProcessingRunRow) -> ProcessingRun:
    """Convert a database row to a domain processing run."""
    return ProcessingRun(
        id=row.id,
        book_id=row.book_id,
        status=ProcessingRunStatus(row.status),
        glossary_snapshot=GlossarySnapshot(
            entries=tuple(
                GlossarySnapshotEntry(
                    source_text=entry["source_text"],
                    translation=entry["translation"],
                    type=ConceptType(entry["type"]),
                    status=TermStatus(entry["status"]),
                    note=entry.get("note", ""),
                )
                for entry in row.glossary_snapshot_json
            )
        ),
        llm_settings_snapshot=LLMSettingsSnapshot(**row.llm_settings_json),
    )


def to_result_row(run_id: str, result: BlockProcessingResult) -> BlockResultRow:
    """Convert a block processing result to a database row."""
    return BlockResultRow(
        run_id=run_id,
        block_id=result.block_id,
        treatment=result.treatment.value,
        translated_text=result.translated_text,
        content_evidence_json=_encode_content_evidence(result.content_evidence),
        glossary_hints_json=[
            {"original": hint.original, "translation": hint.translation, "note": hint.note}
            for hint in result.glossary_hints
        ],
    )


def to_note_rows(run_id: str, result: BlockProcessingResult) -> list[LanguageNoteRow]:
    """Convert language notes attached to a block result into database rows."""
    return [
        LanguageNoteRow(
            run_id=run_id,
            block_id=note.block_id,
            note_id=note.id,
            original_excerpt=note.original_excerpt,
            translated_excerpt=note.translated_excerpt,
            note=note.note,
        )
        for note in result.language_notes
    ]


def to_result_warning_rows(
    run_id: str, result: BlockProcessingResult
) -> list[ResultWarningRow]:
    """Convert non-blocking warnings of a block result into database rows."""
    return [
        ResultWarningRow(
            run_id=run_id,
            block_id=result.block_id,
            kind=warning.kind.value,
            message=warning.message,
        )
        for warning in result.warnings
    ]


def to_block_result(
    row: BlockResultRow,
    notes: Iterable[LanguageNoteRow],
    warnings: Iterable[ResultWarningRow],
) -> BlockProcessingResult:
    """Reconstruct a domain block processing result with language notes and warnings."""
    return BlockProcessingResult(
        block_id=row.block_id,
        treatment=BlockTreatment(row.treatment),
        translated_text=row.translated_text,
        content_evidence=_decode_content_evidence(row.content_evidence_json),
        glossary_hints=tuple(
            GlossaryHint(
                original=hint["original"],
                translation=hint["translation"],
                note=hint["note"],
            )
            for hint in row.glossary_hints_json
        ),
        language_notes=tuple(
            LanguageNote(
                id=note.note_id,
                block_id=note.block_id,
                original_excerpt=note.original_excerpt,
                translated_excerpt=note.translated_excerpt,
                note=note.note,
            )
            for note in notes
        ),
        warnings=tuple(
            ResultWarning(kind=ResultWarningKind(warning.kind), message=warning.message)
            for warning in warnings
        ),
    )


# --- tasks and LLM log ------------------------------------------------------------


def to_task_row(task: ProcessingTask, created_at: datetime) -> ProcessingTaskRow:
    """Convert a processing task to a database row."""
    error = task.last_error
    return ProcessingTaskRow(
        id=task.id,
        book_id=task.book_id,
        task_type=task.task_type.value,
        scope_id=task.scope_id,
        status=task.status.value,
        attempts=task.attempts,
        error_type=None if error is None else error.type.value,
        error_message=None if error is None else error.message,
        error_detail=None if error is None else error.detail,
        created_at=created_at,
    )


def to_task(row: ProcessingTaskRow) -> ProcessingTask:
    """Convert a database row to a domain processing task."""
    return ProcessingTask(
        id=row.id,
        book_id=row.book_id,
        task_type=TaskType(row.task_type),
        scope_id=row.scope_id,
        status=TaskStatus(row.status),
        attempts=row.attempts,
        last_error=_to_task_error(row.error_type, row.error_message, row.error_detail),
    )


def to_llm_call_row(entry: LLMCallLog) -> LLMCallLogRow:
    """Convert an LLM call log entry to a database row.

    Full prompt and completion texts are omitted from the log: schema has no columns for them.
    """
    error = entry.error
    usage = entry.usage
    return LLMCallLogRow(
        id=entry.id,
        book_id=entry.book_id,
        task_id=entry.task_id,
        stage=entry.stage.value,
        provider=entry.provider,
        model=entry.model,
        started_at=entry.started_at,
        duration_ms=entry.duration_ms,
        attempt=entry.attempt,
        success=entry.success,
        error_type=None if error is None else error.type.value,
        error_message=None if error is None else error.message,
        error_detail=None if error is None else error.detail,
        prompt_tokens=None if usage is None else usage.prompt_tokens,
        completion_tokens=None if usage is None else usage.completion_tokens,
        cost=entry.cost,
    )


def to_llm_call(row: LLMCallLogRow) -> LLMCallLog:
    """Convert a database row to a domain LLM call log entry."""
    has_usage = row.prompt_tokens is not None or row.completion_tokens is not None
    return LLMCallLog(
        id=row.id,
        book_id=row.book_id,
        stage=LLMCallStage(row.stage),
        provider=row.provider,
        model=row.model,
        started_at=row.started_at,
        duration_ms=row.duration_ms,
        attempt=row.attempt,
        success=row.success,
        task_id=row.task_id,
        error=_to_task_error(row.error_type, row.error_message, row.error_detail),
        usage=(
            TokenUsage(
                prompt_tokens=row.prompt_tokens, completion_tokens=row.completion_tokens
            )
            if has_usage
            else None
        ),
        cost=row.cost,
    )


def _to_task_error(
    error_type: str | None, message: str | None, detail: str | None
) -> TaskError | None:
    """Construct a typed task error from three columns or return `None` if no error occurred."""
    if error_type is None or message is None:
        return None
    return TaskError(type=ErrorType(error_type), message=message, detail=detail)


# --- content plan: dedicated strict v2 storage format -----------------


def _object(value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError("неверные поля объекта")
    return value


def _string(value: object) -> str:
    if type(value) is not str:
        raise ValueError("ожидалась строка")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("ожидалось целое число")
    return value


def _array(value: object) -> list[Any]:
    if type(value) is not list:
        raise ValueError("ожидался список")
    return value


def _help_evidence(value: object) -> HelpEvidence:
    fields = {"difficulty_id", "point_id", "artifact_kind", "block_id", "artifact_key", "body_excerpt"}
    data = _object(value, fields)
    return HelpEvidence(**{name: _string(data[name]) for name in fields})


def _difficulty(value: object) -> Difficulty:
    fields = {"id", "anchors", "owner_block_id", "kind", "priority", "help_kind", "required_points"}
    data = _object(value, fields)
    anchors = []
    for item in _array(data["anchors"]):
        anchor = _object(item, {"block_id", "start", "end", "quote"})
        anchors.append(SourceAnchor(
            block_id=_string(anchor["block_id"]), start=_integer(anchor["start"]),
            end=_integer(anchor["end"]), quote=_string(anchor["quote"]),
        ))
    points = []
    for item in _array(data["required_points"]):
        point = _object(item, {"id", "description"})
        points.append(RequiredPoint(_string(point["id"]), _string(point["description"])))
    return Difficulty(
        **{name: _string(data[name]) for name in ("id", "owner_block_id", "kind", "priority", "help_kind")},
        anchors=tuple(anchors), required_points=tuple(points),
    )


def _decode_content_plan(value: object) -> ContentPlan | None:
    if value is None:
        return None
    try:
        data = _object(value, {"version", "profile", "difficulties", "before_you_read_evidence"})
        return ContentPlan(
            version=_integer(data["version"]), profile=_string(data["profile"]),
            difficulties=tuple(_difficulty(item) for item in _array(data["difficulties"])),
            before_you_read_evidence=tuple(_help_evidence(item) for item in _array(data["before_you_read_evidence"])),
        )
    except (ValueError, TypeError, DomainError):
        raise PersistenceError("повреждённый сохранённый содержательный план") from None


def _decode_content_evidence(value: object) -> ContentEvidence | tuple[()]:
    if value is None:
        return ()
    try:
        data = _object(value, {"version", "plan_sha256", "evidence", "structural_coverage", "semantic_correctness"})
        return ContentEvidence(
            version=_integer(data["version"]), plan_sha256=_string(data["plan_sha256"]),
            evidence=tuple(_help_evidence(item) for item in _array(data["evidence"])),
            structural_coverage=_string(data["structural_coverage"]),
            semantic_correctness=_string(data["semantic_correctness"]),
        )
    except (ValueError, TypeError, DomainError):
        raise PersistenceError("повреждённые сохранённые свидетельства помощи") from None


def _json_collections(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_collections(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_collections(item) for item in value]
    return value


def _encode_content_plan(value: ContentPlan | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not ContentPlan:
        raise PersistenceError("некорректный содержательный план для сохранения")
    data = _json_collections(asdict(value))
    _decode_content_plan(data)
    return data


def _encode_content_evidence(value: ContentEvidence | tuple[()]) -> dict[str, Any] | None:
    if type(value) is tuple and value == ():
        return None
    if type(value) is not ContentEvidence:
        raise PersistenceError("некорректные свидетельства помощи для сохранения")
    data = _json_collections(asdict(value))
    _decode_content_evidence(data)
    return data


def conditions_to_row(conditions):
    """Map immutable conditions without inventing legacy metadata."""
    from linguary.adapters.db.models import BookConditionsRow
    from linguary.domain.book_conditions import thaw
    return BookConditionsRow(
        id=conditions.id, book_id=conditions.book_id, revision=conditions.revision,
        schema_version=conditions.schema_version, payload_json=thaw(conditions.payload),
        reproducibility=conditions.reproducibility, created_at=conditions.created_at,
    )


def conditions_from_row(row):
    """Absent conditions remain absent on historical books."""
    from linguary.domain.book_conditions import BookConditions
    if row is None:
        return None
    return BookConditions(row.id, row.book_id, row.revision, row.payload_json,
                          row.created_at, row.reproducibility, row.schema_version)
