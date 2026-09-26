"""Внешние DTO Web UI: то, что уходит в браузер (API-01).

Ответы описаны отдельно от доменных сущностей: наружу отдаётся ровно то, что нужно
экрану, вместе с русскими названиями состояний и режимов. Так изменение домена не
меняет молча контракт интерфейса, а секреты конфигурации не могут попасть в ответ
случайным реэкспортом поля.

Каждая модель собирается из прикладного результата фабрикой `of_*`: routes только
вызывают сценарий и превращают его результат в ответ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import BaseModel, Field, model_validator

from linguary.adapters.config import AppSettings
from linguary.adapters.validation import EpubCheckAvailability
from linguary.application.book_overview import (
    BookOverview,
    BookProgress,
    BookResultSummary,
    BookWarnings,
    ChapterPlanSummary,
    PlanBlockSummary,
    ProcessingPlanSummary,
    ResultBlockSummary,
    ResultChapterSummary,
    ResultWarningEntry,
    RunningChapter,
    TaskFailure,
)
from linguary.application.glossary_review import GlossaryReview, GlossaryReviewItem
from linguary.application.ports.persistence import BookRecord
from linguary.application.use_cases.approve_glossary import GlossaryDecision
from linguary.application.use_cases.reset_processing import ResetProcessingResult
from linguary.domain import (
    BlockTreatment,
    BookImportWarning,
    ConceptType,
    BookStatus,
    CoverageStatus,
    GlossaryHint,
    LanguageNote,
    ProcessingRun,
    ProcessingTask,
    ResultWarning,
    with_reference_labels,
)
from linguary.web import labels


class GlossaryDecisionRequest(BaseModel):
    """Form-команда решения по одному элементу глоссария.

    `note` — переводческое указание по элементу. Поля нет в форме — указание
    остаётся прежним; пустое поле снимает его (PRD § 11).

    `type` — исправленный тип понятия; как и текст варианта, принимается только
    исправлением.
    """

    action: GlossaryDecision
    text: str | None = None
    note: str | None = None
    type: ConceptType | None = None

    @model_validator(mode="after")
    def validate_text_for_action(self) -> Self:
        """Текст обязателен только для исправления варианта, тип — только с ним."""
        has_text = bool(self.text and self.text.strip())
        if self.action is GlossaryDecision.CORRECT and not has_text:
            raise ValueError("для исправления укажите новый вариант перевода")
        if self.action is not GlossaryDecision.CORRECT and has_text:
            raise ValueError("текст принимается только для действия correct")
        if self.action is not GlossaryDecision.CORRECT and self.type is not None:
            raise ValueError("тип понятия принимается только для действия correct")
        return self


class GlossaryNoteRequest(BaseModel):
    """Form-команда правки переводческого указания по элементу глоссария."""

    note: str = ""


class ResetProcessingRequest(BaseModel):
    """Form-команда сброса с обязательным положительным подтверждением."""

    confirmed: bool

    @model_validator(mode="after")
    def require_confirmation(self) -> Self:
        """Отрицательный флаг не разрешает разрушающее действие."""
        if not self.confirmed:
            raise ValueError("возврат к проверке глоссария нужно подтвердить")
        return self


class GlossaryItemResponse(BaseModel):
    """Один элемент проверки глоссария (PRD § 12, § 21.3).

    `type` — тип понятия с учётом решения пользователя. `other_spellings` — прочие
    написания элемента в книге, объединённые с показанным. `has_decision_conflict` —
    по разным написаниям раньше были приняты разные решения, и действует одно из них.
    """

    concept_id: str
    source_text: str
    type: str
    suggested_translation: str
    translation: str | None = None
    status: str | None = None
    status_label: str | None = None
    confidence: float
    model_flagged_review: bool
    model_doubts: bool
    has_translation_conflict: bool
    requires_review: bool
    is_decided: bool
    note: str = ""
    other_spellings: tuple[str, ...] = ()
    has_decision_conflict: bool = False

    @classmethod
    def of_item(cls, item: GlossaryReviewItem) -> GlossaryItemResponse:
        """Собирает элемент вместе с текущим решением пользователя."""
        decision = item.decision
        variant = decision.variants[0] if decision and decision.variants else None
        translation = variant.text if variant is not None else None
        return cls(
            concept_id=item.concept_id,
            source_text=item.source_text,
            type=item.concept_type.value,
            suggested_translation=item.suggested_translation,
            translation=translation,
            status=decision.status.value if decision is not None else None,
            status_label=(
                labels.glossary_decision_label(
                    decision.status,
                    translation=translation,
                    suggested_translation=item.suggested_translation,
                )
                if decision is not None
                else None
            ),
            confidence=item.confidence,
            model_flagged_review=item.model_flagged_review,
            model_doubts=item.model_doubts,
            has_translation_conflict=item.has_translation_conflict,
            requires_review=item.requires_review,
            is_decided=item.is_decided,
            note=item.note,
            other_spellings=item.other_spellings,
            has_decision_conflict=item.has_decision_conflict,
        )


class GlossaryReviewResponse(BaseModel):
    """Проверка глоссария книги и доступность редактирования."""

    book_id: str
    read_only: bool
    is_complete: bool
    pending_count: int
    items: tuple[GlossaryItemResponse, ...] = ()

    @classmethod
    def of_review(cls, review: GlossaryReview) -> GlossaryReviewResponse:
        """Собирает ответ в прикладном порядке: ожидающие решения уже идут первыми."""
        return cls(
            book_id=review.book_id,
            read_only=not review.is_editable,
            is_complete=review.is_complete,
            pending_count=len(review.awaiting_decision),
            items=tuple(GlossaryItemResponse.of_item(item) for item in review.items),
        )


class ProcessingStartResponse(BaseModel):
    """Открытый запуск обработки и поставленные задачи глав."""

    book_id: str
    run_id: str
    task_ids: tuple[str, ...] = ()

    @classmethod
    def of_start(
        cls, run: ProcessingRun, tasks: Sequence[ProcessingTask]
    ) -> ProcessingStartResponse:
        """Собирает подтверждение асинхронного запуска."""
        return cls(
            book_id=run.book_id,
            run_id=run.id,
            task_ids=tuple(task.id for task in tasks),
        )


class ResetProcessingResponse(BaseModel):
    """Результат подтверждённого возврата к проверке глоссария."""

    book_id: str
    status: str
    removed_run_id: str | None = None
    removed_results: int
    removed_tasks: int

    @classmethod
    def of_result(cls, result: ResetProcessingResult) -> ResetProcessingResponse:
        """Собирает ответ по атомарному результату сброса."""
        return cls(
            book_id=result.book_id,
            status=BookStatus.GLOSSARY_REVIEW.value,
            removed_run_id=result.run_id,
            removed_results=result.removed_results,
            removed_tasks=result.removed_tasks,
        )


class TreatmentCount(BaseModel):
    """Сколько блоков получило один режим обработки (PRD § 21.4)."""

    treatment: BlockTreatment
    label: str
    count: int

    @classmethod
    def of_counts(
        cls, counts: Mapping[BlockTreatment, int]
    ) -> tuple[TreatmentCount, ...]:
        """Счётчики режимов в постоянном порядке: экран не должен переставлять строки."""
        return tuple(
            cls(
                treatment=treatment,
                label=labels.treatment_label(treatment),
                count=counts[treatment],
            )
            for treatment in BlockTreatment
            if treatment in counts
        )


class TaskFailureResponse(BaseModel):
    """Неуспешная задача книги вместе с доступным действием (PRD § 16, § 21.4)."""

    task_id: str
    task_type: str
    task_type_label: str
    scope_id: str
    error_type: str
    error_type_label: str
    message: str
    detail: str | None = None
    attempts: int
    is_retryable: bool

    @classmethod
    def of_failure(cls, failure: TaskFailure) -> TaskFailureResponse:
        """Собирает ответ по прикладному описанию неуспешной задачи."""
        return cls(
            task_id=failure.task_id,
            task_type=failure.task_type.value,
            task_type_label=labels.task_type_label(failure.task_type),
            scope_id=failure.scope_id,
            error_type=failure.error_type.value,
            error_type_label=labels.error_type_label(failure.error_type),
            message=failure.message,
            detail=failure.detail,
            attempts=failure.attempts,
            is_retryable=failure.is_retryable,
        )


class TaskResponse(BaseModel):
    """Текущее состояние фоновой задачи после команды пользователя."""

    id: str
    book_id: str
    task_type: str
    task_type_label: str
    scope_id: str
    status: str
    status_label: str
    attempts: int
    is_retryable: bool

    @classmethod
    def of_task(cls, task: ProcessingTask) -> TaskResponse:
        """Собирает безопасный фрагмент задачи без технической детали ошибки."""
        return cls(
            id=task.id,
            book_id=task.book_id,
            task_type=task.task_type.value,
            task_type_label=labels.task_type_label(task.task_type),
            scope_id=task.scope_id,
            status=task.status.value,
            status_label=labels.task_status_label(task.status),
            attempts=task.attempts,
            is_retryable=task.is_retryable,
        )


class RunningChapterResponse(BaseModel):
    """Глава, задача которой выполняется прямо сейчас."""

    chapter_number: int
    title: str

    @classmethod
    def of_chapter(cls, chapter: RunningChapter) -> RunningChapterResponse:
        """Собирает ответ по выполняющейся главе."""
        return cls(chapter_number=chapter.chapter_number, title=chapter.title)


class BookProgressResponse(BaseModel):
    """Прогресс обработки книги (PRD § 21.4).

    `current_chapter` и `current_chapter_title` — первая из `running_chapters`: при
    параллельной обработке глав выполняется несколько. `remaining_seconds` — грубая
    оценка оставшегося времени идущего этапа, `remaining_time_label` — она же словами;
    оба пусты, когда этап не идёт.
    """

    percent_complete: int
    total_sections: int
    analyzed_sections: int
    planned_blocks: int
    processed_blocks: int
    treatment_counts: tuple[TreatmentCount, ...] = ()
    current_chapter: int | None = None
    current_chapter_title: str | None = None
    running_chapters: tuple[RunningChapterResponse, ...] = ()
    remaining_seconds: int | None = None
    remaining_time_label: str | None = None
    pending_tasks: int
    running_tasks: int
    failures: tuple[TaskFailureResponse, ...] = ()

    @classmethod
    def of_progress(cls, progress: BookProgress) -> BookProgressResponse:
        """Собирает ответ по прикладному прогрессу книги."""
        return cls(
            percent_complete=progress.percent_complete,
            total_sections=progress.total_sections,
            analyzed_sections=progress.analyzed_sections,
            planned_blocks=progress.planned_blocks,
            processed_blocks=progress.processed_blocks,
            treatment_counts=TreatmentCount.of_counts(progress.treatment_counts),
            current_chapter=progress.current_chapter,
            current_chapter_title=progress.current_chapter_title,
            running_chapters=tuple(
                RunningChapterResponse.of_chapter(chapter)
                for chapter in progress.running_chapters
            ),
            remaining_seconds=(
                progress.remaining_time.seconds
                if progress.remaining_time is not None
                else None
            ),
            remaining_time_label=(
                labels.remaining_time_label(progress.remaining_time.seconds)
                if progress.remaining_time is not None
                else None
            ),
            pending_tasks=progress.pending_tasks,
            running_tasks=progress.running_tasks,
            failures=tuple(
                TaskFailureResponse.of_failure(failure)
                for failure in progress.failures
            ),
        )


class BookSummaryResponse(BaseModel):
    """Строка списка книг: название, состояние и языки (PRD § 21.1)."""

    id: str
    title: str
    authors: tuple[str, ...] = ()
    status: str
    status_label: str
    source_language: str
    target_language: str
    created_at: str

    @classmethod
    def of_book(cls, book: BookRecord) -> BookSummaryResponse:
        """Собирает строку списка по записи книги."""
        return cls(
            id=book.id,
            title=book.metadata.title,
            authors=book.metadata.authors,
            status=book.status.value,
            status_label=labels.book_status_label(book.status),
            source_language=book.source_language,
            target_language=book.target_language,
            created_at=book.created_at.isoformat(),
        )


class BookListResponse(BaseModel):
    """Список книг (PRD § 21.1)."""

    books: tuple[BookSummaryResponse, ...] = ()

    @classmethod
    def of_books(cls, books: Sequence[BookRecord]) -> BookListResponse:
        """Собирает список книг для экрана."""
        return cls(books=tuple(BookSummaryResponse.of_book(book) for book in books))


class BookDetailResponse(BaseModel):
    """Карточка книги: состояние, прогресс, счётчики режимов и ошибки (API-01)."""

    book: BookSummaryResponse
    progress: BookProgressResponse
    export_validation_status: str | None = None
    warning_count: int = 0

    @classmethod
    def of_overview(cls, overview: BookOverview) -> BookDetailResponse:
        """Собирает карточку по прикладному описанию книги."""
        return cls(
            book=BookSummaryResponse.of_book(overview.book),
            progress=BookProgressResponse.of_progress(overview.progress),
            export_validation_status=(
                overview.book.export_validation_status.value
                if overview.book.export_validation_status is not None
                else None
            ),
            warning_count=overview.warning_count,
        )


class ChapterPlanResponse(BaseModel):
    """Сводка плана обработки одной главы (UX notes)."""

    section_id: str
    chapter_number: int
    title: str
    is_notes: bool = False
    coverage_status: str
    is_complete: bool
    textual_blocks: int
    treatment_counts: tuple[TreatmentCount, ...] = ()
    blocks: tuple["PlanBlockResponse", ...] = ()

    @classmethod
    def of_chapter(cls, chapter: ChapterPlanSummary) -> ChapterPlanResponse:
        """Собирает сводку главы."""
        return cls(
            section_id=chapter.section_id,
            chapter_number=chapter.chapter_number,
            title=chapter.title,
            is_notes=chapter.is_notes,
            coverage_status=chapter.coverage_status.value,
            is_complete=chapter.coverage_status is CoverageStatus.COMPLETE,
            textual_blocks=chapter.textual_blocks,
            treatment_counts=TreatmentCount.of_counts(chapter.treatment_counts),
            blocks=tuple(PlanBlockResponse.of_block(block) for block in chapter.blocks),
        )


class PlanBlockResponse(BaseModel):
    """Блок главы с режимом, выбранным анализом."""

    block_id: str
    block_type: str
    text: str
    treatment: str | None = None
    treatment_label: str | None = None

    @classmethod
    def of_block(cls, block: PlanBlockSummary) -> PlanBlockResponse:
        """Собирает строку плана по одному исходному блоку."""
        return cls(
            block_id=block.block.id,
            block_type=block.block.type.value,
            text=block.block.text,
            treatment=block.treatment.value if block.treatment is not None else None,
            treatment_label=(
                labels.treatment_label(block.treatment)
                if block.treatment is not None
                else None
            ),
        )


class ProcessingPlanResponse(BaseModel):
    """Диагностическая сводка плана обработки книги (PRD § 21, API-01)."""

    chapters: tuple[ChapterPlanResponse, ...] = ()
    treatment_counts: tuple[TreatmentCount, ...] = ()
    analyzed_sections: int
    total_sections: int
    is_complete: bool

    @classmethod
    def of_plan(cls, plan: ProcessingPlanSummary) -> ProcessingPlanResponse:
        """Собирает сводку плана книги."""
        return cls(
            chapters=tuple(
                ChapterPlanResponse.of_chapter(chapter) for chapter in plan.chapters
            ),
            treatment_counts=TreatmentCount.of_counts(plan.treatment_counts),
            analyzed_sections=plan.analyzed_sections,
            total_sections=plan.total_sections,
            is_complete=plan.is_complete,
        )


class ImportWarningResponse(BaseModel):
    """Предупреждение импорта книги (PRD § 21.6)."""

    kind: str
    kind_label: str
    message: str
    section_id: str | None = None
    block_id: str | None = None

    @classmethod
    def of_warning(cls, warning: BookImportWarning) -> ImportWarningResponse:
        """Собирает предупреждение импорта."""
        return cls(
            kind=warning.kind.value,
            kind_label=labels.import_warning_label(warning.kind),
            message=warning.message,
            section_id=warning.section_id,
            block_id=warning.block_id,
        )


class ResultWarningResponse(BaseModel):
    """Замечание к результату обработки вместе с его местом в книге (PRD § 15)."""

    kind: str
    kind_label: str
    message: str
    block_id: str
    section_id: str | None = None
    chapter_number: int | None = None

    @classmethod
    def of_entry(cls, entry: ResultWarningEntry) -> ResultWarningResponse:
        """Собирает замечание результата."""
        return cls.of_warning(
            entry.warning,
            block_id=entry.block_id,
            section_id=entry.section_id,
            chapter_number=entry.chapter_number,
        )

    @classmethod
    def of_warning(
        cls,
        warning: ResultWarning,
        *,
        block_id: str,
        section_id: str | None,
        chapter_number: int | None,
    ) -> ResultWarningResponse:
        """Собирает замечание уже известного блока результата."""
        return cls(
            kind=warning.kind.value,
            kind_label=labels.result_warning_label(warning.kind),
            message=warning.message,
            block_id=block_id,
            section_id=section_id,
            chapter_number=chapter_number,
        )


class BookWarningsResponse(BaseModel):
    """Предупреждения импорта и результата книги (API-01)."""

    import_warnings: tuple[ImportWarningResponse, ...] = ()
    result_warnings: tuple[ResultWarningResponse, ...] = ()
    total: int

    @classmethod
    def of_warnings(cls, warnings: BookWarnings) -> BookWarningsResponse:
        """Собирает предупреждения книги."""
        return cls(
            import_warnings=tuple(
                ImportWarningResponse.of_warning(warning)
                for warning in warnings.import_warnings
            ),
            result_warnings=tuple(
                ResultWarningResponse.of_entry(entry)
                for entry in warnings.result_warnings
            ),
            total=warnings.total,
        )


class GlossaryHintResponse(BaseModel):
    """Короткая словарная подсказка внутри блока."""

    original: str
    translation: str
    note: str | None = None

    @classmethod
    def of_hint(cls, hint: GlossaryHint) -> GlossaryHintResponse:
        return cls(
            original=hint.original,
            translation=hint.translation,
            note=hint.note,
        )


class LanguageNoteResponse(BaseModel):
    """Учебная заметка, привязанная к результату блока."""

    id: str
    original_excerpt: str
    translated_excerpt: str | None = None
    note: str

    @classmethod
    def of_note(cls, note: LanguageNote) -> LanguageNoteResponse:
        return cls(
            id=note.id,
            original_excerpt=note.original_excerpt,
            translated_excerpt=note.translated_excerpt,
            note=note.note,
        )


class ResultBlockResponse(BaseModel):
    """Оригинал и помощь модели для одного стабильного блока книги."""

    block_id: str
    section_id: str
    chapter_number: int
    block_type: str
    treatment: str | None = None
    treatment_label: str | None = None
    original_text: str
    translated_text: str | None = None
    glossary_hints: tuple[GlossaryHintResponse, ...] = ()
    language_notes: tuple[LanguageNoteResponse, ...] = ()
    warnings: tuple[ResultWarningResponse, ...] = ()
    regeneration_task: TaskResponse | None = None

    @classmethod
    def of_block(
        cls,
        summary: ResultBlockSummary,
        *,
        regeneration_task: ProcessingTask | None = None,
    ) -> ResultBlockResponse:
        """Собирает блок результата или целевой фрагмент перегенерации."""
        result = summary.result
        treatment = summary.treatment
        return cls(
            block_id=summary.block.id,
            section_id=summary.block.section_id,
            chapter_number=summary.chapter_number,
            block_type=summary.block.type.value,
            treatment=treatment.value if treatment is not None else None,
            treatment_label=(
                labels.treatment_label(treatment) if treatment is not None else None
            ),
            original_text=summary.block.text,
            translated_text=(
                with_reference_labels(result.translated_text, summary.block)
                if result is not None and result.translated_text is not None
                else None
            ),
            glossary_hints=(
                tuple(GlossaryHintResponse.of_hint(hint) for hint in result.glossary_hints)
                if result is not None
                else ()
            ),
            language_notes=(
                tuple(LanguageNoteResponse.of_note(note) for note in result.language_notes)
                if result is not None
                else ()
            ),
            warnings=(
                tuple(
                    ResultWarningResponse.of_warning(
                        warning,
                        block_id=summary.block.id,
                        section_id=summary.block.section_id,
                        chapter_number=summary.chapter_number,
                    )
                    for warning in result.warnings
                )
                if result is not None
                else ()
            ),
            regeneration_task=(
                TaskResponse.of_task(regeneration_task)
                if regeneration_task is not None
                else None
            ),
        )


class ResultChapterResponse(BaseModel):
    """Одна глава результата в порядке чтения."""

    section_id: str
    chapter_number: int
    title: str
    is_notes: bool = False
    blocks: tuple[ResultBlockResponse, ...] = ()

    @classmethod
    def of_chapter(cls, chapter: ResultChapterSummary) -> ResultChapterResponse:
        return cls(
            section_id=chapter.section_id,
            chapter_number=chapter.chapter_number,
            title=chapter.title,
            is_notes=chapter.is_notes,
            blocks=tuple(ResultBlockResponse.of_block(block) for block in chapter.blocks),
        )


class BookResultResponse(BaseModel):
    """Просмотр результата по блокам с предупреждениями и итогом EPUBCheck."""

    book_id: str
    status: str
    export_validation_status: str | None = None
    export_validation_status_label: str | None = None
    epubcheck_hint: str | None = None
    warnings: BookWarningsResponse
    chapters: tuple[ResultChapterResponse, ...] = ()

    @classmethod
    def of_result(cls, result: BookResultSummary) -> BookResultResponse:
        validation = result.book.export_validation_status
        return cls(
            book_id=result.book.id,
            status=result.book.status.value,
            export_validation_status=(validation.value if validation is not None else None),
            export_validation_status_label=(
                labels.export_validation_status_label(validation)
                if validation is not None
                else None
            ),
            epubcheck_hint=(
                labels.EPUBCHECK_HINT
                if validation is not None and validation.value == "not_validated"
                else None
            ),
            warnings=BookWarningsResponse.of_warnings(result.warnings),
            chapters=tuple(
                ResultChapterResponse.of_chapter(chapter)
                for chapter in result.chapters
            ),
        )


class LLMConfigResponse(BaseModel):
    """Состояние конфигурации LLM (PRD § 21.6, § 23).

    Ключ провайдера в ответ не попадает — только признак того, что он задан.
    """

    is_ready: bool
    provider: str
    model: str | None = None
    has_api_key: bool
    base_url: str
    max_attempts: int
    hint: str | None = None

    @classmethod
    def of_settings(cls, settings: AppSettings) -> LLMConfigResponse:
        """Собирает состояние LLM по локальной конфигурации."""
        llm = settings.llm
        return cls(
            is_ready=llm.is_ready,
            provider=llm.provider,
            model=llm.model or None,
            has_api_key=bool(llm.api_key and llm.api_key.strip()),
            base_url=llm.base_url,
            max_attempts=llm.max_attempts,
            hint=None if llm.is_ready else labels.LLM_NOT_CONFIGURED_HINT,
        )


class EpubCheckConfigResponse(BaseModel):
    """Доступность внешней проверки EPUB (PRD § 20, § 21.6)."""

    available: bool
    reason: str | None = None
    java_path: str | None = None
    jar_path: str | None = None
    hint: str | None = None

    @classmethod
    def of_availability(
        cls, availability: EpubCheckAvailability
    ) -> EpubCheckConfigResponse:
        """Собирает состояние внешней проверки."""
        return cls(
            available=availability.available,
            reason=availability.reason,
            java_path=availability.java_path,
            jar_path=availability.jar_path,
            hint=None if availability.available else labels.EPUBCHECK_HINT,
        )


class StorageConfigResponse(BaseModel):
    """Где приложение держит данные (PRD § 22, UX notes)."""

    data_root: str
    database_path: str
    books_root: str


class ServerConfigResponse(BaseModel):
    """Адрес локального Web UI (PRD § 4.1, § 21.6)."""

    base_url: str
    host: str
    port: int
    is_local_only: bool
    warning: str | None = None


class SystemConfigResponse(BaseModel):
    """Экран конфигурации и окружения целиком (API-01, PRD § 21.6)."""

    llm: LLMConfigResponse
    epubcheck: EpubCheckConfigResponse
    storage: StorageConfigResponse
    server: ServerConfigResponse
    external_llm_warning: str = Field(default=labels.EXTERNAL_LLM_WARNING)


class ErrorResponse(BaseModel):
    """Отказ приложения в виде, пригодном для показа пользователю."""

    detail: str


__all__ = [
    "BookDetailResponse",
    "BookListResponse",
    "BookProgressResponse",
    "BookResultResponse",
    "BookSummaryResponse",
    "BookWarningsResponse",
    "ChapterPlanResponse",
    "EpubCheckConfigResponse",
    "ErrorResponse",
    "GlossaryDecisionRequest",
    "GlossaryHintResponse",
    "GlossaryItemResponse",
    "GlossaryNoteRequest",
    "GlossaryReviewResponse",
    "ImportWarningResponse",
    "LLMConfigResponse",
    "LanguageNoteResponse",
    "PlanBlockResponse",
    "ProcessingPlanResponse",
    "ProcessingStartResponse",
    "ResetProcessingRequest",
    "ResetProcessingResponse",
    "ResultBlockResponse",
    "ResultChapterResponse",
    "ResultWarningResponse",
    "ServerConfigResponse",
    "StorageConfigResponse",
    "SystemConfigResponse",
    "TaskFailureResponse",
    "TaskResponse",
    "TreatmentCount",
]
