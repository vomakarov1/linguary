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

from pydantic import BaseModel, Field

from linguary.adapters.config import AppSettings
from linguary.adapters.validation import EpubCheckAvailability
from linguary.application.book_overview import (
    BookOverview,
    BookProgress,
    BookWarnings,
    ChapterPlanSummary,
    ProcessingPlanSummary,
    ResultWarningEntry,
    TaskFailure,
)
from linguary.application.ports.persistence import BookRecord
from linguary.domain import BlockTreatment, BookImportWarning, CoverageStatus
from linguary.web import labels


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


class BookProgressResponse(BaseModel):
    """Прогресс обработки книги (PRD § 21.4)."""

    percent_complete: int
    total_sections: int
    analyzed_sections: int
    planned_blocks: int
    processed_blocks: int
    treatment_counts: tuple[TreatmentCount, ...] = ()
    current_chapter: int | None = None
    current_chapter_title: str | None = None
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
    coverage_status: str
    is_complete: bool
    textual_blocks: int
    treatment_counts: tuple[TreatmentCount, ...] = ()

    @classmethod
    def of_chapter(cls, chapter: ChapterPlanSummary) -> ChapterPlanResponse:
        """Собирает сводку главы."""
        return cls(
            section_id=chapter.section_id,
            chapter_number=chapter.chapter_number,
            title=chapter.title,
            coverage_status=chapter.coverage_status.value,
            is_complete=chapter.coverage_status is CoverageStatus.COMPLETE,
            textual_blocks=chapter.textual_blocks,
            treatment_counts=TreatmentCount.of_counts(chapter.treatment_counts),
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
        return cls(
            kind=entry.warning.kind.value,
            kind_label=labels.result_warning_label(entry.warning.kind),
            message=entry.warning.message,
            block_id=entry.block_id,
            section_id=entry.section_id,
            chapter_number=entry.chapter_number,
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
    "BookSummaryResponse",
    "BookWarningsResponse",
    "ChapterPlanResponse",
    "EpubCheckConfigResponse",
    "ErrorResponse",
    "ImportWarningResponse",
    "LLMConfigResponse",
    "ProcessingPlanResponse",
    "ResultWarningResponse",
    "ServerConfigResponse",
    "StorageConfigResponse",
    "SystemConfigResponse",
    "TaskFailureResponse",
    "TreatmentCount",
]
