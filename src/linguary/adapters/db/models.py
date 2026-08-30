"""ORM-модели SQLite-схемы MVP (SDD SQL-01).

Модели намеренно отделены от доменных сущностей. Домен состоит из `frozen=True, slots=True`
dataclass-ов: у них нет ни изменяемых атрибутов, ни `__dict__`, поэтому отобразить их на
таблицы напрямую нельзя. Преобразование в обе стороны выполняют функции `mapping.py`.

Перечисления домена — `StrEnum`, их значения совпадают с внешним представлением PRD, поэтому
в базе они хранятся строками как есть; обратное преобразование делает тот же `mapping.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

#: Соглашение об именах ключей и индексов. Без него SQLite не даёт изменить или удалить
#: безымянный constraint в batch-режиме Alembic: у миграции нет имени, за которое взяться.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: Условие частичного уникального индекса «один активный запуск обработки на книгу».
ACTIVE_RUN_CONDITION = "status = 'active'"


class UtcDateTime(TypeDecorator[datetime]):
    """Момент времени, хранимый в UTC.

    SQLite не хранит смещение часового пояса, поэтому значение приводится к UTC при
    записи и возвращается с явным UTC при чтении. Момент без часового пояса считается
    заданным в UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class DecimalAsText(TypeDecorator[Decimal]):
    """Величина с фиксированной точностью, хранимая текстом.

    Числовые колонки SQLite возвращают `float`, из-за чего стоимость вызова теряет
    точность. Текстовое представление сохраняет её без потерь.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> str | None:
        return None if value is None else str(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> Decimal | None:
        return None if value is None else Decimal(value)


class Base(DeclarativeBase):
    """Общий базовый класс ORM-моделей схемы."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class BookRow(Base):
    """Книга: пользовательское состояние и связь с файлами на диске."""

    __tablename__ = "books"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    authors_json: Mapped[list[str]] = mapped_column(JSON)
    identifier: Mapped[str | None] = mapped_column(String)
    publisher: Mapped[str | None] = mapped_column(String)
    source_language: Mapped[str] = mapped_column(String)
    target_language: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True)
    source_epub_path: Mapped[str] = mapped_column(String)
    export_validation_status: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)


class SectionRow(Base):
    """Глава книги: один XHTML-документ порядка чтения."""

    __tablename__ = "sections"
    __table_args__ = (UniqueConstraint("book_id", "order"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String)
    order: Mapped[int] = mapped_column()
    source_resource_path: Mapped[str] = mapped_column(String)


class DocumentBlockRow(Base):
    """Стабильный блок документа: единица текста, результата и повторной генерации."""

    __tablename__ = "document_blocks"
    __table_args__ = (
        UniqueConstraint("section_id", "order"),
        Index("ix_document_blocks_book_id_section_id", "book_id", "section_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    section_id: Mapped[str] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String)
    order: Mapped[int] = mapped_column()
    text: Mapped[str] = mapped_column(Text)
    inline_annotations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    attributes_json: Mapped[dict[str, str]] = mapped_column(JSON)
    #: Ссылка отложенная: блоки одного раздела пишутся пакетом, и родитель не обязан
    #: попасть в базу раньше вложенного блока.
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "document_blocks.id",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        )
    )
    source_reference_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    support_status: Mapped[str] = mapped_column(String)


class ResourceRow(Base):
    """Ресурс книги: изображение, шрифт или стиль, нужный экспортёру."""

    __tablename__ = "resources"
    __table_args__ = (UniqueConstraint("book_id", "source_path"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    source_path: Mapped[str] = mapped_column(String)
    media_type: Mapped[str] = mapped_column(String)
    stored_path: Mapped[str | None] = mapped_column(String)
    support_status: Mapped[str] = mapped_column(String)


class ImportWarningRow(Base):
    """Непреградное замечание импорта, привязанное к книге, разделу или блоку."""

    __tablename__ = "import_warnings"

    id: Mapped[int] = mapped_column(primary_key=True)
    book_id: Mapped[str] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[str | None] = mapped_column(String)
    block_id: Mapped[str | None] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)


class TermConceptRow(Base):
    """Понятие глоссария книги."""

    __tablename__ = "term_concepts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String)
    canonical_source_text: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    note: Mapped[str] = mapped_column(Text, default="", server_default="")


class TermVariantRow(Base):
    """Языковой вариант перевода понятия глоссария."""

    __tablename__ = "term_variants"

    id: Mapped[int] = mapped_column(primary_key=True)
    concept_id: Mapped[str] = mapped_column(
        ForeignKey("term_concepts.id", ondelete="CASCADE"), index=True
    )
    language: Mapped[str] = mapped_column(String, index=True)
    text: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)


class ChapterAnalysisRow(Base):
    """Результат анализа одной главы: инструкции, сводка и `Before You Read`."""

    __tablename__ = "chapter_analyses"
    __table_args__ = (UniqueConstraint("book_id", "section_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    section_id: Mapped[str] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    coverage_status: Mapped[str] = mapped_column(String)
    translation_instructions: Mapped[str] = mapped_column(Text)
    chapter_summary: Mapped[str] = mapped_column(Text)
    before_you_read_json: Mapped[list[dict[str, str]]] = mapped_column(JSON)


class TreatmentPlanItemRow(Base):
    """Режим обработки одного блока в плане главы."""

    __tablename__ = "treatment_plan_items"
    __table_args__ = (UniqueConstraint("analysis_id", "block_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("chapter_analyses.id", ondelete="CASCADE")
    )
    block_id: Mapped[str] = mapped_column(String)
    treatment: Mapped[str] = mapped_column(String)
    position: Mapped[int] = mapped_column()


class GlossaryCandidateRow(Base):
    """Кандидат глоссария, найденный при анализе главы."""

    __tablename__ = "glossary_candidates"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("chapter_analyses.id", ondelete="CASCADE"), index=True
    )
    source_text: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)
    suggested_translation: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column()
    model_flagged_review: Mapped[bool] = mapped_column()
    requires_review: Mapped[bool] = mapped_column(index=True)
    note: Mapped[str] = mapped_column(Text, default="", server_default="")


class ProcessingRunRow(Base):
    """Запуск обработки книги со снимком глоссария и настроек LLM."""

    __tablename__ = "processing_runs"
    __table_args__ = (
        Index(
            "uq_processing_runs_active_book",
            "book_id",
            unique=True,
            sqlite_where=text(ACTIVE_RUN_CONDITION),
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String)
    glossary_snapshot_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    llm_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)


class BlockResultRow(Base):
    """Результат обработки одного блока в рамках запуска."""

    __tablename__ = "block_results"
    __table_args__ = (UniqueConstraint("run_id", "block_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="CASCADE")
    )
    block_id: Mapped[str] = mapped_column(
        ForeignKey("document_blocks.id", ondelete="CASCADE")
    )
    treatment: Mapped[str] = mapped_column(String)
    translated_text: Mapped[str | None] = mapped_column(Text)
    glossary_hints_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)


class LanguageNoteRow(Base):
    """Языковой комментарий, привязанный ровно к одному блоку."""

    __tablename__ = "language_notes"
    __table_args__ = (
        UniqueConstraint("run_id", "note_id"),
        Index("ix_language_notes_run_id_block_id", "run_id", "block_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="CASCADE")
    )
    block_id: Mapped[str] = mapped_column(
        ForeignKey("document_blocks.id", ondelete="CASCADE")
    )
    note_id: Mapped[str] = mapped_column(String)
    original_excerpt: Mapped[str] = mapped_column(Text)
    translated_excerpt: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str] = mapped_column(Text)


class ResultWarningRow(Base):
    """Непреградное замечание к результату обработки блока."""

    __tablename__ = "result_warnings"
    __table_args__ = (Index("ix_result_warnings_run_id_block_id", "run_id", "block_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="CASCADE")
    )
    block_id: Mapped[str] = mapped_column(
        ForeignKey("document_blocks.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)


class ProcessingTaskRow(Base):
    """Локальная возобновляемая задача обработки книги.

    `scope_id` — пустая строка, а не NULL, для задач уровня книги: в SQLite `NULL != NULL`,
    и на NULL уникальный индекс не различил бы две задачи импорта одной книги.
    """

    __tablename__ = "processing_tasks"
    __table_args__ = (UniqueConstraint("book_id", "task_type", "scope_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    task_type: Mapped[str] = mapped_column(String)
    scope_id: Mapped[str] = mapped_column(String, default="", server_default="")
    status: Mapped[str] = mapped_column(String, index=True)
    attempts: Mapped[int] = mapped_column()
    error_type: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)


class LLMCallLogRow(Base):
    """Запись технического журнала об одном вызове модели.

    Полных текстов запроса и ответа в схеме нет: они содержат текст книги (PRD § 24).
    """

    __tablename__ = "llm_call_logs"
    __table_args__ = (
        Index("ix_llm_call_logs_book_id_stage", "book_id", "stage"),
        Index("ix_llm_call_logs_task_id", "task_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    task_id: Mapped[str | None] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime)
    duration_ms: Mapped[int] = mapped_column()
    attempt: Mapped[int] = mapped_column()
    success: Mapped[bool] = mapped_column()
    error_type: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    cost: Mapped[Decimal | None] = mapped_column(DecimalAsText)


__all__ = [
    "ACTIVE_RUN_CONDITION",
    "NAMING_CONVENTION",
    "Base",
    "BlockResultRow",
    "BookRow",
    "ChapterAnalysisRow",
    "DecimalAsText",
    "DocumentBlockRow",
    "GlossaryCandidateRow",
    "ImportWarningRow",
    "LLMCallLogRow",
    "LanguageNoteRow",
    "ProcessingRunRow",
    "ProcessingTaskRow",
    "ResourceRow",
    "ResultWarningRow",
    "SectionRow",
    "TermConceptRow",
    "TermVariantRow",
    "TreatmentPlanItemRow",
    "UtcDateTime",
]
