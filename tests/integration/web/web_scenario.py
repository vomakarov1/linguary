"""Состояния книги, которые показывают экраны Web UI.

Экранам нужна книга, дошедшая до определённого места процесса: проанализированная,
обрабатываемая, с неуспешной задачей или с предупреждениями импорта. Модуль кладёт такое
состояние в настоящее хранилище приложения одной функцией, чтобы тест описывал только то,
что проверяет.

Модуль намеренно без префикса `test_`: иначе pytest соберёт его как тестовый.
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
    DocumentBlock,
    DocumentSection,
    ErrorType,
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

#: Фиксированный момент времени: тест не зависит от текущих часов.
CREATED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

#: Абзац оригинала и его перевод.
SOURCE_TEXT = "It is a truth universally acknowledged."
TRANSLATED_TEXT = "Общепризнанная истина гласит следующее."


@dataclass(frozen=True, slots=True)
class WebScenario:
    """Книга в хранилище вместе с её канонической моделью."""

    document: CanonicalDocument
    book_id: str = BOOK_ID
    run_id: str = RUN_ID

    def section(self, index: int = 0) -> DocumentSection:
        """Глава книги по порядку чтения."""
        return self.document.ordered_sections[index]

    def block_ids(self, index: int = 0) -> tuple[str, ...]:
        """Идентификаторы блоков главы в порядке исходного документа."""
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
) -> WebScenario:
    """Кладёт книгу в хранилище приложения вместе с её анализом и предупреждениями.

    `treatments` задаёт режимы блоков главы по её номеру от нуля; без них все блоки
    получают полный перевод. `analysed_chapters` оставляет анализ только у перечисленных
    глав: пустой набор означает книгу, которую ещё не анализировали.
    """
    document = _document(book_id=book_id, title=title, chapters=chapters)
    with application.unit_of_work.transaction() as uow:
        uow.books.add(_book(book_id=book_id, title=title, status=status))
        uow.documents.save(document)
        for index, section in enumerate(document.ordered_sections):
            if analysed_chapters is not None and index not in analysed_chapters:
                continue
            uow.analyses.save(
                book_id, _analysis(section, treatments=(treatments or {}).get(index))
            )
        if import_warnings:
            uow.documents.add_import_warnings(book_id, list(import_warnings))
    return WebScenario(document=document, book_id=book_id)


def start_run(
    application: Application,
    scenario: WebScenario,
    *,
    results: Sequence[BlockProcessingResult] = (),
) -> None:
    """Открывает запуск обработки книги и сохраняет готовые результаты блоков."""
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
    """Кладёт задачу книги прямо в хранилище: экраны показывают её состояние."""
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
    """Выполняющаяся задача главы: её глава показывается как текущая."""
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
    """Неуспешная задача главы с типизированной ошибкой (PRD § 16)."""
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
    """Готовый перевод блока, при необходимости с замечанием (PRD § 15)."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=TRANSLATED_TEXT,
        warnings=tuple(warnings),
    )


def hinted_result(block_id: str) -> BlockProcessingResult:
    """Блок со словарными подсказками вместо перевода (PRD § 10.1)."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(
            GlossaryHint(original="truth", translation="истина"),
        ),
    )


def repetition_warning(message: str = "Повтор фрагмента в переводе") -> ResultWarning:
    """Непреградное замечание к результату обработки."""
    return ResultWarning(kind=ResultWarningKind.REPETITION, message=message)


def dropped_block_warning(
    section_id: str, *, message: str = "Сценарий на странице отброшен"
) -> BookImportWarning:
    """Предупреждение импорта, привязанное к главе."""
    return BookImportWarning(
        kind=ImportWarningKind.DROPPED_BLOCK, message=message, section_id=section_id
    )


def section_id_of(book_id: str, order: int) -> str:
    """Идентификатор главы книги по её порядковому номеру от нуля."""
    return f"{book_id}-section-{order + 1}"


def _document(
    *, book_id: str, title: str, chapters: Sequence[Sequence[str]]
) -> CanonicalDocument:
    """Книга из глав, заданных текстами их абзацев."""
    return CanonicalDocument(
        id=book_id,
        metadata=BookMetadata(title=title, authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=tuple(
            _section(book_id=book_id, order=order, texts=texts)
            for order, texts in enumerate(chapters)
        ),
    )


def _section(
    *, book_id: str, order: int, texts: Sequence[str]
) -> DocumentSection:
    """Глава книги из абзацев.

    Идентификатор главы включает книгу: разделы уникальны по всему хранилищу, и две
    книги одного теста не могут делить одну строку раздела.
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
                type=BlockType.PARAGRAPH,
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
) -> ChapterAnalysis:
    """Полный анализ главы: план обработки покрывает все её текстовые блоки."""
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
    )


def _book(*, book_id: str, title: str, status: BookStatus) -> BookRecord:
    """Запись книги в заданном состоянии."""
    return BookRecord(
        id=book_id,
        metadata=BookMetadata(title=title, authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        status=status,
        source_epub_path=f"/library/{book_id}.epub",
        created_at=CREATED_AT,
    )
