"""Book states rendered by Web UI screens.

Screens require a book progressed to a specific pipeline stage: analysed, processing,
with a failed task, or with import warnings. This module seeds such states in the real
application storage via a single helper so that tests only describe what they verify.

Intentionally omitted the `test_` prefix so pytest does not collect it as a test file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from linguary.application.ports.persistence import BookRecord
from linguary.composition import Application
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookImportWarning,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    ErrorType,
    GlossaryCandidate,
    GlossaryHint,
    ImportWarningKind,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingTask,
    ResultWarning,
    ResultWarningKind,
    SourceReference,
    SupportStatus,
    TaskError,
    TaskType,
    TreatmentPlan,
)
from linguary.domain.processing import GlossarySnapshot

BOOK_ID = "book-web-1"
RUN_ID = "run-web-1"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "ru"

#: Fixed timestamp: test does not depend on current system time.
CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

#: Source paragraph and its translation.
SOURCE_TEXT = "It is a truth universally acknowledged."
TRANSLATED_TEXT = "Общепризнанная истина гласит следующее."


@dataclass(frozen=True, slots=True)
class WebScenario:
    """Book in repository paired with its canonical document model."""

    document: CanonicalDocument
    book_id: str = BOOK_ID
    run_id: str = RUN_ID

    def section(self, index: int = 0) -> DocumentSection:
        """Chapter section in reading order."""
        return self.document.ordered_sections[index]

    def block_ids(self, index: int = 0) -> tuple[str, ...]:
        """Block IDs of the chapter in original document order."""
        return tuple(block.id for block in self.section(index).ordered_blocks)


def store_book(
    application: Application,
    *,
    status: BookStatus = BookStatus.GLOSSARY_REVIEW,
    chapters: Sequence[Sequence[str]] = ((SOURCE_TEXT, SOURCE_TEXT),),
    treatments: Mapping[int, Sequence[BlockTreatment]] | None = None,
    analysed_chapters: Sequence[int] | None = None,
    book_id: str = BOOK_ID,
    title: str = "Pride and Prejudice",
    import_warnings: Sequence[BookImportWarning] = (),
    glossary_candidates: Mapping[int, Sequence[GlossaryCandidate]] | None = None,
    notes_chapters: Sequence[int] = (),
) -> WebScenario:
    """Stores a book in application repository along with its analysis and warnings.

    `treatments` specifies block treatments by 0-based chapter index; without them all
    blocks default to full translation. `analysed_chapters` restricts analysis to given
    chapters: an empty sequence represents an unanalysed book. `notes_chapters` specifies
    chapter indices composed entirely of footnotes.
    """
    document = _document(
        book_id=book_id, title=title, chapters=chapters, notes_chapters=notes_chapters
    )
    with application.unit_of_work.transaction() as uow:
        uow.books.add(_book(book_id=book_id, title=title, status=status))
        uow.documents.save(document)
        for index, section in enumerate(document.ordered_sections):
            if analysed_chapters is not None and index not in analysed_chapters:
                continue
            uow.analyses.save(
                book_id,
                _analysis(
                    section,
                    treatments=(treatments or {}).get(index),
                    glossary_candidates=(glossary_candidates or {}).get(index, ()),
                ),
            )
        if import_warnings:
            uow.documents.add_import_warnings(book_id, list(import_warnings))
    return WebScenario(document=document, book_id=book_id)


def glossary_candidate(
    *,
    book_id: str = BOOK_ID,
    section_index: int = 0,
    candidate_id: str = "candidate-mr-bennet",
    source_text: str = "Mr. Bennet",
    suggested_translation: str = "мистер Беннет",
    concept_type: ConceptType = ConceptType.PERSON,
    confidence: float = 0.95,
    model_flagged_review: bool = False,
) -> GlossaryCandidate:
    """Glossary candidate belonging to the analysis of the given chapter."""
    section_id = section_id_of(book_id, section_index)
    return GlossaryCandidate.create(
        id=candidate_id,
        analysis_id=f"analysis-{section_id}",
        source_text=source_text,
        type=concept_type,
        suggested_translation=suggested_translation,
        confidence=confidence,
        model_flagged_review=model_flagged_review,
        confidence_threshold=0.8,
    )


def start_run(
    application: Application,
    scenario: WebScenario,
    *,
    results: Sequence[BlockProcessingResult] = (),
) -> None:
    """Opens a processing run for the book and saves prepared block results."""
    with application.unit_of_work.transaction() as uow:
        uow.runs.add(
            ProcessingRun(
                id=scenario.run_id,
                book_id=scenario.book_id,
                status=ProcessingRunStatus.ACTIVE,
                glossary_snapshot=GlossarySnapshot(),
                llm_settings_snapshot=LLMSettingsSnapshot(
                    provider="fake", model="fake-model"
                ),
            )
        )
        if results:
            uow.results.replace_results(scenario.run_id, list(results))


def store_task(
    application: Application,
    *,
    task: ProcessingTask,
) -> ProcessingTask:
    """Stores a book task directly into storage: screens display its state."""
    with application.unit_of_work.transaction() as uow:
        uow.tasks.add(task)
    return task


def running_task(
    scenario: WebScenario,
    *,
    task_id: str = "task-running",
    section_index: int = 0,
    task_type: TaskType = TaskType.PROCESS_TRANSLATION_UNIT,
) -> ProcessingTask:
    """In-progress chapter task: its chapter displays as active."""
    return ProcessingTask(
        id=task_id,
        book_id=scenario.book_id,
        task_type=task_type,
        scope_id=scenario.section(section_index).id,
    ).start()


def failed_task(
    scenario: WebScenario,
    *,
    task_id: str = "task-failed",
    section_index: int = 0,
    error_type: ErrorType = ErrorType.TRANSIENT,
    message: str = "провайдер не ответил",
) -> ProcessingTask:
    """Failed chapter task with a typed error (PRD § 16)."""
    return (
        ProcessingTask(
            id=task_id,
            book_id=scenario.book_id,
            task_type=TaskType.PROCESS_TRANSLATION_UNIT,
            scope_id=scenario.section(section_index).id,
        )
        .start()
        .fail(TaskError(type=error_type, message=message, detail="HTTP 503"))
    )


def translated_result(
    block_id: str, *, warnings: Sequence[ResultWarning] = ()
) -> BlockProcessingResult:
    """Completed translation of a block, optionally with warnings (PRD § 15)."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=TRANSLATED_TEXT,
        warnings=tuple(warnings),
    )


def hinted_result(block_id: str) -> BlockProcessingResult:
    """Block with glossary hints instead of translation (PRD § 10.1)."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(
            GlossaryHint(original="truth", translation="истина"),
        ),
    )


def repetition_warning(message: str = "Повтор фрагмента в переводе") -> ResultWarning:
    """Non-blocking warning for a processing result."""
    return ResultWarning(kind=ResultWarningKind.REPETITION, message=message)


def dropped_block_warning(
    section_id: str, *, message: str = "Сценарий на странице отброшен"
) -> BookImportWarning:
    """Import warning associated with a section."""
    return BookImportWarning(
        kind=ImportWarningKind.DROPPED_BLOCK, message=message, section_id=section_id
    )


def section_id_of(book_id: str, order: int) -> str:
    """Section ID of a book by its 0-based order."""
    return f"{book_id}-section-{order + 1}"


def _document(
    *,
    book_id: str,
    title: str,
    chapters: Sequence[Sequence[str]],
    notes_chapters: Sequence[int] = (),
) -> CanonicalDocument:
    """Book composed of chapters defined by their paragraph texts."""
    return CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=title, authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=tuple(
            _section(
                book_id=book_id,
                order=order,
                texts=texts,
                block_type=(
                    BlockType.FOOTNOTE if order in notes_chapters else BlockType.PARAGRAPH
                ),
            )
            for order, texts in enumerate(chapters)
        ),
    )


def _section(
    *,
    book_id: str,
    order: int,
    texts: Sequence[str],
    block_type: BlockType = BlockType.PARAGRAPH,
) -> DocumentSection:
    """Chapter section composed of paragraphs.

    Section ID includes book ID: sections are unique across the repository,
    preventing two books in a test from colliding on section rows.
    """
    section_id = section_id_of(book_id, order)
    return DocumentSection(
        id=section_id,
        document_id=book_id,
        title=f"Chapter {order + 1}",
        order=order,
        source_resource_path=f"{section_id}.xhtml",
        blocks=tuple(
            DocumentBlock.create(
                document_id=book_id,
                section_id=section_id,
                type=block_type,
                order=index,
                text=text,
                source_reference=SourceReference(
                    resource_path=f"{section_id}.xhtml",
                    structural_path=f"/body/p[{index + 1}]",
                ),
                support_status=SupportStatus.SUPPORTED,
            )
            for index, text in enumerate(texts)
        ),
    )


def _analysis(
    section: DocumentSection,
    *,
    treatments: Sequence[BlockTreatment] | None,
    glossary_candidates: Sequence[GlossaryCandidate] = (),
) -> ChapterAnalysis:
    """Full chapter analysis: treatment plan covers all its textual blocks."""
    blocks = section.ordered_blocks
    chosen = tuple(treatments or ()) or (BlockTreatment.TRANSLATE,) * len(blocks)
    return ChapterAnalysis.create(
        id=f"analysis-{section.id}",
        section_id=section.id,
        treatment_plan=TreatmentPlan.of_treatments(
            section.id,
            zip((block.id for block in blocks), chosen, strict=True),
        ),
        translation_instructions="Сохранять иронию рассказчика.",
        chapter_summary="Mr. Bennet visits the new neighbour.",
        expected_block_ids=[block.id for block in blocks],
        glossary_candidates=glossary_candidates,
    )


def _book(*, book_id: str, title: str, status: BookStatus) -> BookRecord:
    """Book record in the specified state."""
    return BookRecord(
        id=book_id,
        metadata=BookMetadata(title=title, authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        status=status,
        source_epub_path=f"/library/{book_id}.epub",
        created_at=CREATED_AT,
    )
