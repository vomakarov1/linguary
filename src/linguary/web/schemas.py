"""External Web UI DTOs: data sent to the browser (API-01).

Responses are described separately from domain entities: only what the screen needs
is exposed externally, along with Russian labels for statuses and modes. This ensures domain changes
do not silently break interface contracts, and configuration secrets cannot leak into responses
via accidental field re-exports.

Each model is assembled from application results by an `of_*` factory: routes only
invoke use cases and transform results into responses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linguary.adapters.config import AppSettings
from linguary.adapters.config.settings import DEFAULT_TIMEOUT_SECONDS, proxy_secrets
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
    """Form command for deciding on a single glossary item.

    `note` — translator note for the item. If omitted from form, the note
    remains unchanged; an empty field clears it (PRD § 11).

    `type` — corrected concept type; like variant text, accepted only with correction.
    """

    action: GlossaryDecision
    text: str | None = None
    note: str | None = None
    type: ConceptType | None = None

    @model_validator(mode="after")
    def validate_text_for_action(self) -> Self:
        """Text is required only for variant correction; type is accepted only with text."""
        has_text = bool(self.text and self.text.strip())
        if self.action is GlossaryDecision.CORRECT and not has_text:
            raise ValueError("для исправления укажите новый вариант перевода")
        if self.action is not GlossaryDecision.CORRECT and has_text:
            raise ValueError("текст принимается только для действия correct")
        if self.action is not GlossaryDecision.CORRECT and self.type is not None:
            raise ValueError("тип понятия принимается только для действия correct")
        return self


class GlossaryNoteRequest(BaseModel):
    """Form command for editing translator note on a glossary item."""

    note: str = ""


class ResetProcessingRequest(BaseModel):
    """Form command for reset requiring explicit positive confirmation."""

    confirmed: bool

    @model_validator(mode="after")
    def require_confirmation(self) -> Self:
        """Negative flag disallows destructive action."""
        if not self.confirmed:
            raise ValueError("возврат к проверке глоссария нужно подтвердить")
        return self


class GlossaryItemResponse(BaseModel):
    """Single glossary review item (PRD § 12, § 21.3).

    `type` — concept type reflecting user decision. `other_spellings` — other
    item spellings in the book merged with this one. `has_decision_conflict` —
    different decisions were previously made for different spellings, and one currently applies.
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
        """Assembles item together with current user decision."""
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
    """Book glossary review state and editing availability."""

    book_id: str
    read_only: bool
    is_complete: bool
    pending_count: int
    items: tuple[GlossaryItemResponse, ...] = ()

    @classmethod
    def of_review(cls, review: GlossaryReview) -> GlossaryReviewResponse:
        """Assembles response in application order: pending decisions appear first."""
        return cls(
            book_id=review.book_id,
            read_only=not review.is_editable,
            is_complete=review.is_complete,
            pending_count=len(review.awaiting_decision),
            items=tuple(GlossaryItemResponse.of_item(item) for item in review.items),
        )


class ProcessingStartResponse(BaseModel):
    """Open processing run and queued chapter tasks."""

    book_id: str
    run_id: str
    task_ids: tuple[str, ...] = ()

    @classmethod
    def of_start(
        cls, run: ProcessingRun, tasks: Sequence[ProcessingTask]
    ) -> ProcessingStartResponse:
        """Assembles confirmation of asynchronous processing start."""
        return cls(
            book_id=run.book_id,
            run_id=run.id,
            task_ids=tuple(task.id for task in tasks),
        )


class ResetProcessingResponse(BaseModel):
    """Result of confirmed return to glossary review."""

    book_id: str
    status: str
    removed_run_id: str | None = None
    removed_results: int
    removed_tasks: int

    @classmethod
    def of_result(cls, result: ResetProcessingResult) -> ResetProcessingResponse:
        """Assembles response from atomic reset result."""
        return cls(
            book_id=result.book_id,
            status=BookStatus.GLOSSARY_REVIEW.value,
            removed_run_id=result.run_id,
            removed_results=result.removed_results,
            removed_tasks=result.removed_tasks,
        )


class TreatmentCount(BaseModel):
    """Count of blocks assigned a specific treatment mode (PRD § 21.4)."""

    treatment: BlockTreatment
    label: str
    count: int

    @classmethod
    def of_counts(
        cls, counts: Mapping[BlockTreatment, int]
    ) -> tuple[TreatmentCount, ...]:
        """Mode counts in consistent order: screen should not reorder rows."""
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
    """Failed book task along with available user action (PRD § 16, § 21.4)."""

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
        """Assembles response from application failure description."""
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
    """Current status of background task after user command."""

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
        """Assembles safe task fragment without technical error details."""
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
    """Chapter whose task is currently running."""

    chapter_number: int
    title: str

    @classmethod
    def of_chapter(cls, chapter: RunningChapter) -> RunningChapterResponse:
        """Assembles response for currently running chapter."""
        return cls(chapter_number=chapter.chapter_number, title=chapter.title)


class BookProgressResponse(BaseModel):
    """Book processing progress (PRD § 21.4).

    `current_chapter` and `current_chapter_title` — first of `running_chapters`:
    with parallel processing multiple chapters run at once. `remaining_seconds` —
    rough estimate of remaining time for active stage, `remaining_time_label` — same in words;
    both are empty when no stage is running.
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
        """Assembles response from application book progress."""
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
    """Book list row: title, status, and languages (PRD § 21.1)."""

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
        """Assembles list row from book record."""
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
    """Book list (PRD § 21.1)."""

    books: tuple[BookSummaryResponse, ...] = ()

    @classmethod
    def of_books(cls, books: Sequence[BookRecord]) -> BookListResponse:
        """Assembles book list for screen."""
        return cls(books=tuple(BookSummaryResponse.of_book(book) for book in books))


class BookDetailResponse(BaseModel):
    """Book card: status, progress, mode counters, and errors (API-01)."""

    book: BookSummaryResponse
    progress: BookProgressResponse
    export_validation_status: str | None = None
    warning_count: int = 0
    conditions: dict = Field(default_factory=dict)
    reproducibility: str = 'unknown'
    recovery: dict = Field(default_factory=dict)
    quality_review: dict = Field(default_factory=lambda: {'state': 'not-requested'})
    epub_ready: bool = False
    ordinary_cost: dict = Field(default_factory=dict)

    @classmethod
    def of_overview(cls, overview: BookOverview) -> BookDetailResponse:
        """Assembles book card from application overview."""
        return cls(
            book=BookSummaryResponse.of_book(overview.book),
            progress=BookProgressResponse.of_progress(overview.progress),
            export_validation_status=(
                overview.book.export_validation_status.value
                if overview.book.export_validation_status is not None
                else None
            ),
            warning_count=overview.warning_count,
            conditions=dict(overview.conditions), reproducibility=overview.reproducibility,
            recovery=dict(overview.recovery), quality_review=dict(overview.quality_review),
            epub_ready=overview.epub_ready, ordinary_cost=dict(overview.ordinary_cost),
        )


class ChapterPlanResponse(BaseModel):
    """Chapter processing plan summary (UX notes)."""

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
        """Assembles chapter summary."""
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
    """Chapter block with treatment mode selected by analysis."""

    block_id: str
    block_type: str
    text: str
    treatment: str | None = None
    treatment_label: str | None = None

    @classmethod
    def of_block(cls, block: PlanBlockSummary) -> PlanBlockResponse:
        """Assembles plan row from single source block."""
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
    """Diagnostic processing plan summary for book (PRD § 21, API-01)."""

    chapters: tuple[ChapterPlanResponse, ...] = ()
    treatment_counts: tuple[TreatmentCount, ...] = ()
    analyzed_sections: int
    total_sections: int
    is_complete: bool

    @classmethod
    def of_plan(cls, plan: ProcessingPlanSummary) -> ProcessingPlanResponse:
        """Assembles book plan summary."""
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
    """Book import warning (PRD § 21.6)."""

    kind: str
    kind_label: str
    message: str
    section_id: str | None = None
    block_id: str | None = None

    @classmethod
    def of_warning(cls, warning: BookImportWarning) -> ImportWarningResponse:
        """Assembles import warning."""
        return cls(
            kind=warning.kind.value,
            kind_label=labels.import_warning_label(warning.kind),
            message=warning.message,
            section_id=warning.section_id,
            block_id=warning.block_id,
        )


class ResultWarningResponse(BaseModel):
    """Processing result warning with its book location (PRD § 15)."""

    kind: str
    kind_label: str
    message: str
    block_id: str
    section_id: str | None = None
    chapter_number: int | None = None

    @classmethod
    def of_entry(cls, entry: ResultWarningEntry) -> ResultWarningResponse:
        """Assembles result warning."""
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
        """Assembles warning for known result block."""
        return cls(
            kind=warning.kind.value,
            kind_label=labels.result_warning_label(warning.kind),
            message=warning.message,
            block_id=block_id,
            section_id=section_id,
            chapter_number=chapter_number,
        )


class BookWarningsResponse(BaseModel):
    """Book import and result warnings (API-01)."""

    import_warnings: tuple[ImportWarningResponse, ...] = ()
    result_warnings: tuple[ResultWarningResponse, ...] = ()
    total: int

    @classmethod
    def of_warnings(cls, warnings: BookWarnings) -> BookWarningsResponse:
        """Assembles book warnings."""
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
    """Brief glossary hint within a block."""

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
    """Educational language note attached to block result."""

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
    """Original text and model assistance for a single stable book block."""

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
        """Assembles result block or target fragment for regeneration."""
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
    """Single result chapter in reading order."""

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
    """Block-by-block result view with warnings and EPUBCheck outcome."""

    book_id: str
    status: str
    export_validation_status: str | None = None
    export_validation_status_label: str | None = None
    epubcheck_hint: str | None = None
    warnings: BookWarningsResponse
    chapters: tuple[ResultChapterResponse, ...] = ()

    conditions: dict = Field(default_factory=dict)
    reproducibility: str = 'unknown'
    recovery: dict = Field(default_factory=dict)
    quality_review: dict = Field(default_factory=lambda: {'state': 'not-requested'})
    epub_ready: bool = False
    ordinary_cost: dict = Field(default_factory=dict)

    @classmethod
    def of_result(cls, result: BookResultSummary) -> BookResultResponse:
        validation = result.book.export_validation_status
        return cls(
            book_id=result.book.id,
            conditions=dict(result.conditions), reproducibility=result.reproducibility,
            recovery=dict(result.recovery), quality_review=dict(result.quality_review),
            epub_ready=result.epub_ready, ordinary_cost=dict(result.ordinary_cost),            status=result.book.status.value,
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
    """LLM configuration status (PRD § 21.6, § 23).

    Provider key is omitted from response — only a flag indicating it is set.
    """

    is_ready: bool
    provider: str
    model: str | None = None
    analysis_model: str | None = None
    processing_model: str | None = None
    temperature: float | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    proxy_mode: Literal["auto", "direct", "custom"] = "auto"
    has_proxy_url: bool = False
    has_api_key: bool
    base_url: str
    max_attempts: int
    requests_per_minute: int | None = None
    hint: str | None = None

    @classmethod
    def of_settings(cls, settings: AppSettings) -> LLMConfigResponse:
        """Assembles LLM state from local configuration."""
        llm = settings.llm
        def public_text(value: str) -> str:
            key = (llm.api_key or "").strip()
            return "" if (key and key in value) or any(secret in value for secret in proxy_secrets(llm.proxy_url)) else value
        return cls(
            is_ready=llm.is_ready,
            provider=public_text(llm.provider),
            model=public_text(llm.model) or None,
            analysis_model=public_text(llm.analysis_model) or None,
            processing_model=public_text(llm.processing_model) or None,
            temperature=llm.temperature,
            timeout_seconds=llm.timeout_seconds,
            proxy_mode=llm.proxy_mode,
            has_proxy_url=bool(llm.proxy_url),
            has_api_key=bool(llm.api_key and llm.api_key.strip()),
            base_url=public_text(llm.base_url),
            max_attempts=llm.max_attempts,
            requests_per_minute=llm.requests_per_minute,
            hint=None if llm.is_ready else labels.LLM_NOT_CONFIGURED_HINT,
        )


class EpubCheckConfigResponse(BaseModel):
    """External EPUBCheck availability (PRD § 20, § 21.6)."""

    available: bool
    reason: str | None = None
    java_path: str | None = None
    jar_path: str | None = None
    hint: str | None = None

    @classmethod
    def of_availability(
        cls, availability: EpubCheckAvailability
    ) -> EpubCheckConfigResponse:
        """Assembles external validator state."""
        return cls(
            available=availability.available,
            reason=availability.reason,
            java_path=availability.java_path,
            jar_path=availability.jar_path,
            hint=None if availability.available else labels.EPUBCHECK_HINT,
        )


class StorageConfigResponse(BaseModel):
    """Where application stores data (PRD § 22, UX notes)."""

    data_root: str
    database_path: str
    books_root: str


class ServerConfigResponse(BaseModel):
    """Local Web UI server address (PRD § 4.1, § 21.6)."""

    base_url: str
    host: str
    port: int
    is_local_only: bool
    warning: str | None = None


class WorkerConfigResponse(BaseModel):
    concurrency: int


class ReadingConfigResponse(BaseModel):
    """Reader level settings (PRD § 30)."""

    reader_level: str
    supported_levels: list[str] = Field(default_factory=lambda: ["a2", "b1", "b2", "c1"])
    default_level: str = "b2"


class SystemConfigResponse(BaseModel):
    """Full configuration and environment screen (API-01, PRD § 21.6)."""

    llm: LLMConfigResponse
    epubcheck: EpubCheckConfigResponse
    storage: StorageConfigResponse
    server: ServerConfigResponse
    worker: WorkerConfigResponse
    revision: int
    reading: ReadingConfigResponse = Field(
        default_factory=lambda: ReadingConfigResponse(reader_level="b2")
    )
    external_llm_warning: str = Field(default=labels.EXTERNAL_LLM_WARNING)


type SettingsField = Literal[
    "provider", "model", "api_key", "base_url", "requests_per_minute",
    "concurrency", "expected_revision", "form", "proxy_mode", "proxy_url",
    "analysis_model", "processing_model", "temperature", "timeout_seconds",
    "reader_level",
]


class SettingsErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal[
        "invalid_settings", "revision_conflict", "settings_not_applied",
        "settings_closing", "unsupported_media_type",
    ]
    message: str
    field_errors: dict[SettingsField, str] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafeSettingsFormValues:
    """Dedicated form projection; key field is intentionally omitted."""

    provider: str = field(repr=False)
    model: str = field(repr=False)
    base_url: str = field(repr=False)
    requests_per_minute: int | None
    concurrency: int | None
    expected_revision: int | None
    proxy_mode: str = "auto"
    analysis_model: str = field(default="", repr=False)
    processing_model: str = field(default="", repr=False)
    temperature: float | None = None
    timeout_seconds: float | None = None
    reader_level: str = "b2"


class ErrorResponse(BaseModel):
    """Application failure formatted for display to user."""

    detail: str


class LLMTestConnectionRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    analysis_model: str | None = None
    processing_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    proxy_mode: Literal["auto", "direct", "custom"] | None = None
    proxy_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _sanitize_empty_proxy_mode(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mode = data.get("proxy_mode")
            if isinstance(mode, str) and not mode.strip():
                data = dict(data)
                data["proxy_mode"] = None
        return data


class LLMTestConnectionResponse(BaseModel):
    ok: bool
    message: str
    error: str | None = None


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
    "LLMTestConnectionRequest",
    "LLMTestConnectionResponse",
    "LanguageNoteResponse",
    "PlanBlockResponse",
    "ProcessingPlanResponse",
    "ProcessingStartResponse",
    "ReadingConfigResponse",
    "ResetProcessingRequest",
    "ResetProcessingResponse",
    "ResultBlockResponse",
    "ResultChapterResponse",
    "ResultWarningResponse",
    "ServerConfigResponse",
    "StorageConfigResponse",
    "SystemConfigResponse",
    "WorkerConfigResponse",
    "SettingsErrorResponse",
    "SafeSettingsFormValues",
    "TaskFailureResponse",
    "TaskResponse",
    "TreatmentCount",
]


class StartQualityReviewRequest(BaseModel):
    """Explicit consent parameters; no implicit monetary limit or scope."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    request_key: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    route_id: str = Field(min_length=1, max_length=256)
    budget_microusd: int = Field(gt=0, le=9223372036854775807, strict=True)
    currency: Literal['USD'] = 'USD'
    scope: Literal['all'] | list[str]
    expected_manifest_sha: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def valid_scope(self):
        if isinstance(self.scope, list) and (not self.scope or any(not b.strip() for b in self.scope)
                or len(set(self.scope)) != len(self.scope)):
            raise ValueError('invalid_scope')
        return self


class ExtendRecoveryLimitRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    additional_calls: int = Field(gt=0, le=9223372036854775807, strict=True)
    expected_generation: int = Field(ge=0, strict=True)
    confirmed: bool = Field(strict=True)

    @model_validator(mode='after')
    def explicit_confirmation(self):
        if self.confirmed is not True:
            raise ValueError('confirmation_required')
        return self


class QualityReviewStateResponse(BaseModel):
    review_id: str | None = None
    state: Literal['not-requested', 'running', 'completed-without-findings', 'completed-with-findings',
        'partial', 'limit', 'unknown', 'stale']
    manifest_sha: str | None = None
    coverage: dict = Field(default_factory=lambda: dict(checked=[], missing=[]))
    budget: int | None = None
    spent: int = 0
    reserved: int = 0
    currency: Literal['USD'] = 'USD'
    uncertainty: str = 'probabilistic'
    findings: list[dict] = Field(default_factory=list)
    reason: str | None = None
    semantic_correctness: str = 'not_evaluated'
    available_routes: list[dict] = Field(default_factory=list)
    current_manifest_sha: str | None = None
