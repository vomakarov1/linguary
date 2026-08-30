"""Фабрики доменных объектов для тестов хранилища.

Собирают минимальные, но валидные сущности: доменные инварианты проверены пакетом B,
и тест хранилища не должен спотыкаться на их нарушении. Все значения по умолчанию
перекрываются именованными аргументами.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from linguary.application.ports.persistence import BookRecord
from linguary.domain import (
    BeforeYouReadEntry,
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
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

BOOK_ID = "book-1"
SECTION_ID = "section-1"
BLOCK_ID = "block-1"
SECOND_BLOCK_ID = "block-2"
RUN_ID = "run-1"
ANALYSIS_ID = "analysis-1"

#: Фиксированный момент времени: тесты не зависят от текущих часов.
CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def make_book(
    *,
    book_id: str = BOOK_ID,
    status: BookStatus = BookStatus.IMPORTING,
    title: str = "The Sample Book",
) -> BookRecord:
    """Запись книги."""
    return BookRecord(
        id=book_id,
        metadata=BookMetadata(
            title=title,
            authors=("Jane Doe", "John Roe"),
            identifier="urn:uuid:sample",
            publisher="Sample Press",
        ),
        source_language="en",
        target_language="ru",
        status=status,
        source_epub_path="/library/sample.epub",
        created_at=CREATED_AT,
    )


def make_block(
    *,
    block_id: str = BLOCK_ID,
    section_id: str = SECTION_ID,
    order: int = 0,
    text: str = "It was a bright cold day in April.",
    parent_id: str | None = None,
) -> DocumentBlock:
    """Текстовый блок со встроенной разметкой и служебными признаками."""
    return DocumentBlock(
        id=block_id,
        section_id=section_id,
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path="OEBPS/chapter1.xhtml",
            structural_path=f"/html/body/p[{order + 1}]",
            element_id=f"p{order + 1}",
        ),
        support_status=SupportStatus.SUPPORTED,
        inline_annotations=(
            InlineAnnotation(type=InlineAnnotationType.ITALIC, start=0, end=2),
            InlineAnnotation(
                type=InlineAnnotationType.LINK, start=3, end=6, target="#note-1"
            ),
        ),
        parent_id=parent_id,
        attributes={"epub:type": "bodymatter"},
    )


def make_document(*, book_id: str = BOOK_ID) -> CanonicalDocument:
    """Документ из одной главы с двумя блоками и одним ресурсом."""
    section = DocumentSection(
        id=SECTION_ID,
        document_id=book_id,
        title="Chapter 1",
        order=0,
        source_resource_path="OEBPS/chapter1.xhtml",
        blocks=(
            make_block(),
            make_block(block_id=SECOND_BLOCK_ID, order=1, text="Winston Smith slipped in."),
        ),
    )
    return CanonicalDocument(
        id=book_id,
        metadata=make_book(book_id=book_id).metadata,
        source_language="en",
        target_language="ru",
        sections=(section,),
        resources=(
            DocumentResource(
                id="resource-1",
                source_path="OEBPS/images/cover.jpg",
                media_type="image/jpeg",
                support_status=SupportStatus.SUPPORTED,
            ),
        ),
    )


def make_concept(*, book_id: str = BOOK_ID, concept_id: str = "concept-1") -> TermConcept:
    """Понятие глоссария с двумя языковыми вариантами."""
    return TermConcept(
        id=concept_id,
        book_id=book_id,
        type=ConceptType.PERSON,
        canonical_source_text="Winston Smith",
        status=TermStatus.LOCKED,
        note="мужской род",
        variants=(
            TermVariant(
                concept_id=concept_id,
                language="ru",
                text="Уинстон Смит",
                status=TermStatus.LOCKED,
            ),
            TermVariant(
                concept_id=concept_id,
                language="en",
                text="Winston Smith",
                status=TermStatus.APPROVED,
            ),
        ),
    )


def make_analysis(*, analysis_id: str = ANALYSIS_ID) -> ChapterAnalysis:
    """Полный анализ главы: план, кандидаты и материалы `Before You Read`."""
    return ChapterAnalysis(
        id=analysis_id,
        section_id=SECTION_ID,
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=TreatmentPlan(
            section_id=SECTION_ID,
            items=(
                TreatmentPlanItem(block_id=BLOCK_ID, treatment=BlockTreatment.TRANSLATE),
                TreatmentPlanItem(
                    block_id=SECOND_BLOCK_ID, treatment=BlockTreatment.GLOSSARY_ONLY
                ),
            ),
        ),
        translation_instructions="Сохранять сухой газетный тон рассказчика.",
        chapter_summary="Уинстон возвращается домой в холодный апрельский день.",
        glossary_candidates=(
            GlossaryCandidate(
                id="candidate-1",
                analysis_id=analysis_id,
                source_text="Winston Smith",
                type=ConceptType.PERSON,
                suggested_translation="Уинстон Смит",
                confidence=0.92,
                model_flagged_review=False,
                requires_review=True,
                note="мужской род",
            ),
        ),
        before_you_read=(
            BeforeYouReadEntry(
                expression="bright cold day",
                explanation="Сочетание описывает ясный морозный день.",
            ),
        ),
    )


def make_run(
    *,
    run_id: str = RUN_ID,
    book_id: str = BOOK_ID,
    status: ProcessingRunStatus = ProcessingRunStatus.ACTIVE,
) -> ProcessingRun:
    """Запуск обработки со снимками глоссария и настроек LLM."""
    return ProcessingRun(
        id=run_id,
        book_id=book_id,
        status=status,
        glossary_snapshot=GlossarySnapshot(
            entries=(
                GlossarySnapshotEntry(
                    source_text="Winston Smith",
                    translation="Уинстон Смит",
                    type=ConceptType.PERSON,
                    status=TermStatus.LOCKED,
                    note="мужской род",
                ),
            )
        ),
        llm_settings_snapshot=LLMSettingsSnapshot(
            provider="openai", model="gpt-test", temperature=0.2, max_output_tokens=4096
        ),
    )


def make_translated_result(*, block_id: str = BLOCK_ID) -> BlockProcessingResult:
    """Результат режима `TRANSLATE` с подсказками, заметкой и замечанием."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Был ясный холодный апрельский день.",
        glossary_hints=(
            GlossaryHint(original="bright", translation="ясный", note="о погоде"),
        ),
        language_notes=(
            LanguageNote(
                id="ln-ch01-001",
                block_id=block_id,
                original_excerpt="bright cold day",
                note="Два прилагательных подряд без союза — обычная английская конструкция.",
                translated_excerpt="ясный холодный день",
            ),
        ),
        warnings=(
            ResultWarning(
                kind=ResultWarningKind.REPETITION, message="Повтор слова «день» в абзаце."
            ),
        ),
    )


def make_glossary_only_result(
    *, block_id: str = SECOND_BLOCK_ID
) -> BlockProcessingResult:
    """Результат режима `GLOSSARY_ONLY`: подсказки без полного перевода."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="slipped in", translation="проскользнул"),),
    )


def make_task(
    *,
    task_id: str = "task-1",
    book_id: str = BOOK_ID,
    task_type: TaskType = TaskType.IMPORT_BOOK,
    scope_id: str = "",
    status: TaskStatus = TaskStatus.PENDING,
    attempts: int = 0,
    last_error: TaskError | None = None,
) -> ProcessingTask:
    """Локальная задача обработки."""
    return ProcessingTask(
        id=task_id,
        book_id=book_id,
        task_type=task_type,
        scope_id=scope_id,
        status=status,
        attempts=attempts,
        last_error=last_error,
    )


def make_llm_call(
    *, call_id: str = "call-1", book_id: str = BOOK_ID, success: bool = True
) -> LLMCallLog:
    """Запись журнала вызовов LLM."""
    return LLMCallLog(
        id=call_id,
        book_id=book_id,
        stage=LLMCallStage.TRANSLATION_UNIT,
        provider="openai",
        model="gpt-test",
        started_at=CREATED_AT,
        duration_ms=1234,
        attempt=1,
        success=success,
        task_id="task-1",
        error=(
            None
            if success
            else TaskError(
                type=ErrorType.TRANSIENT, message="Таймаут провайдера", detail="504"
            )
        ),
        usage=TokenUsage(prompt_tokens=1500, completion_tokens=800),
        cost=Decimal("0.0123"),
    )
