"""Преобразование доменных сущностей в строки таблиц и обратно.

Домен состоит из неизменяемых dataclass-ов и ничего не знает о SQLAlchemy, поэтому связь
«домен — таблица» вынесена в отдельные чистые функции: они не трогают сессию и не делают
запросов. Репозитории собирают из них операции чтения и записи.

Значения доменных `StrEnum` совпадают с внешним представлением PRD и хранятся строками как
есть; при чтении строка снова разбирается в перечисление.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

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
from linguary.application.ports.persistence import BookRecord, ExportValidationStatus
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

# --- книга -------------------------------------------------------------------------


def to_book_row(record: BookRecord) -> BookRow:
    """Строка книги из прикладной записи."""
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
    """Прикладная запись книги из строки таблицы."""
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


# --- каноническая модель документа -------------------------------------------------


def to_section_row(book_id: str, section: DocumentSection) -> SectionRow:
    """Строка раздела."""
    return SectionRow(
        id=section.id,
        book_id=book_id,
        title=section.title,
        order=section.order,
        source_resource_path=section.source_resource_path,
    )


def to_block_row(book_id: str, block: DocumentBlock) -> DocumentBlockRow:
    """Строка блока документа."""
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
    """Доменный блок из строки таблицы."""
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
    """Строка ресурса книги вместе с путём его сохранённой копии."""
    return ResourceRow(
        id=resource.id,
        book_id=book_id,
        source_path=resource.source_path,
        media_type=resource.media_type,
        stored_path=stored_path,
        support_status=resource.support_status.value,
    )


def to_resource(row: ResourceRow) -> DocumentResource:
    """Доменный ресурс из строки таблицы."""
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
    """Каноническая модель книги, собранная из строк всех её таблиц."""
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
    """Строка предупреждения импорта."""
    return ImportWarningRow(
        book_id=book_id,
        section_id=warning.section_id,
        block_id=warning.block_id,
        kind=warning.kind.value,
        message=warning.message,
    )


def to_import_warning(row: ImportWarningRow) -> BookImportWarning:
    """Доменное предупреждение импорта из строки таблицы."""
    return BookImportWarning(
        kind=ImportWarningKind(row.kind),
        message=row.message,
        section_id=row.section_id,
        block_id=row.block_id,
    )


# --- глоссарий ---------------------------------------------------------------------


def to_concept_row(concept: TermConcept) -> TermConceptRow:
    """Строка понятия глоссария."""
    return TermConceptRow(
        id=concept.id,
        book_id=concept.book_id,
        type=concept.type.value,
        canonical_source_text=concept.canonical_source_text,
        status=concept.status.value,
        note=concept.note,
    )


def to_variant_row(variant: TermVariant) -> TermVariantRow:
    """Строка языкового варианта понятия."""
    return TermVariantRow(
        concept_id=variant.concept_id,
        language=variant.language,
        text=variant.text,
        status=variant.status.value,
    )


def to_concept(row: TermConceptRow, variants: Sequence[TermVariantRow]) -> TermConcept:
    """Доменное понятие глоссария вместе с его вариантами."""
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


# --- анализ главы ------------------------------------------------------------------


def to_analysis_row(book_id: str, analysis: ChapterAnalysis) -> ChapterAnalysisRow:
    """Строка анализа главы вместе с учебными материалами главы."""
    return ChapterAnalysisRow(
        id=analysis.id,
        book_id=book_id,
        section_id=analysis.section_id,
        coverage_status=analysis.coverage_status.value,
        translation_instructions=analysis.translation_instructions,
        chapter_summary=analysis.chapter_summary,
        before_you_read_json=[
            {"expression": entry.expression, "explanation": entry.explanation}
            for entry in analysis.before_you_read
        ],
    )


def to_plan_item_rows(analysis: ChapterAnalysis) -> list[TreatmentPlanItemRow]:
    """Строки плана обработки блоков главы с сохранением их порядка."""
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
    """Строка кандидата глоссария."""
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
    """Доменный анализ главы из строк его таблиц."""
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
        before_you_read=tuple(
            BeforeYouReadEntry(
                expression=entry["expression"], explanation=entry["explanation"]
            )
            for entry in row.before_you_read_json
        ),
    )


# --- запуск обработки и его результаты ----------------------------------------------


def to_run_row(run: ProcessingRun, created_at: datetime) -> ProcessingRunRow:
    """Строка запуска обработки со снимками глоссария и настроек LLM."""
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
    """Доменный запуск обработки из строки таблицы."""
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
    """Строка результата обработки блока."""
    return BlockResultRow(
        run_id=run_id,
        block_id=result.block_id,
        treatment=result.treatment.value,
        translated_text=result.translated_text,
        glossary_hints_json=[
            {"original": hint.original, "translation": hint.translation, "note": hint.note}
            for hint in result.glossary_hints
        ],
    )


def to_note_rows(run_id: str, result: BlockProcessingResult) -> list[LanguageNoteRow]:
    """Строки учебных заметок, привязанных к блоку результата."""
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
    """Строки непреградных замечаний к результату блока."""
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
    """Доменный результат обработки блока вместе с заметками и замечаниями."""
    return BlockProcessingResult(
        block_id=row.block_id,
        treatment=BlockTreatment(row.treatment),
        translated_text=row.translated_text,
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


# --- задачи и журнал LLM ------------------------------------------------------------


def to_task_row(task: ProcessingTask, created_at: datetime) -> ProcessingTaskRow:
    """Строка локальной задачи."""
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
    """Доменная задача из строки таблицы."""
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
    """Строка журнала вызовов LLM.

    Полные тексты запроса и ответа в журнал не попадают: колонок под них в схеме нет.
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
    """Доменная запись журнала вызовов LLM из строки таблицы."""
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
    """Типизированная ошибка из трёх колонок либо `None`, если ошибки не было."""
    if error_type is None or message is None:
        return None
    return TaskError(type=ErrorType(error_type), message=message, detail=detail)
