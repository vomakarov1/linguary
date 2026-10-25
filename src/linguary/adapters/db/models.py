"""ORM models for the SQLite MVP schema (SDD SQL-01).

Models are intentionally separated from domain entities. The domain consists of `frozen=True, slots=True`
dataclasses: they have neither mutable attributes nor `__dict__`, and therefore cannot be mapped to
tables directly. Bidirectional mapping is handled by functions in `mapping.py`.

Domain enumerations are `StrEnum` instances whose values match the external PRD representation,
so they are stored as raw strings in the database; `mapping.py` handles the reverse transformation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Integer,
    CheckConstraint,
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
from sqlalchemy.exc import DontWrapMixin

from linguary.application.ports.persistence import PersistenceError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

#: Naming convention for keys and constraints. Without it, SQLite cannot alter or drop
#: an unnamed constraint in Alembic batch mode because the migration has no identifier to target.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: Condition for the partial unique index enforcing "one active processing run per book".
ACTIVE_RUN_CONDITION = "status = 'active'"


class ContentJSONError(PersistenceError, DontWrapMixin):
    """Error for new JSON fields without embedding SQL parameters into the message."""


class ContentObjectJSON(TypeDecorator[dict[str, Any]]):
    """SQL NULL for legacy rows; non-null JSON must be a dict/object."""

    impl = JSON
    cache_ok = True

    def __init__(self) -> None:
        super().__init__(none_as_null=True)

    def process_bind_param(self, value, dialect):
        if value is not None and type(value) is not dict:
            raise ContentJSONError("некорректный объект содержательного плана или свидетельств")
        return value

    def result_processor(self, dialect, coltype):
        processor = super().result_processor(dialect, coltype)

        def process(raw):
            if raw is None:
                return None
            try:
                value = processor(raw)
            except (ValueError, TypeError):
                raise ContentJSONError("повреждённый JSON содержательного плана или свидетельств") from None
            if type(value) is not dict:
                raise ContentJSONError("некорректный объект содержательного плана или свидетельств")
            return value

        return process


class UtcDateTime(TypeDecorator[datetime]):
    """Timestamp stored in UTC.

    SQLite does not store timezone offsets, so values are converted to UTC on write
    and returned with an explicit UTC timezone on read. Naive datetimes are assumed
    to be in UTC.
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
    """Fixed-precision numeric value stored as text.

    SQLite numeric columns return `float`, causing call cost calculations to lose
    precision. Textual representation preserves precision without loss.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> str | None:
        return None if value is None else str(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> Decimal | None:
        return None if value is None else Decimal(value)


class Base(DeclarativeBase):
    """Common base class for schema ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class BookRow(Base):
    """Book: user-facing state and association with files on disk."""

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
    """Book section / chapter: a single XHTML document in spine order."""

    __tablename__ = "sections"
    __table_args__ = (UniqueConstraint("book_id", "order"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String)
    order: Mapped[int] = mapped_column()
    source_resource_path: Mapped[str] = mapped_column(String)


class DocumentBlockRow(Base):
    """Stable document block: unit of text, result, and regeneration."""

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
    #: Deferred foreign key: blocks of a section are written in batch, and parents
    #: are not guaranteed to be inserted before child blocks.
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
    """Book resource: image, font, or stylesheet required for export."""

    __tablename__ = "resources"
    __table_args__ = (UniqueConstraint("book_id", "source_path"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    source_path: Mapped[str] = mapped_column(String)
    media_type: Mapped[str] = mapped_column(String)
    stored_path: Mapped[str | None] = mapped_column(String)
    support_status: Mapped[str] = mapped_column(String)


class ImportWarningRow(Base):
    """Non-blocking import warning associated with a book, section, or block."""

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
    """Glossary term concept of a book."""

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
    """Language translation variant for a glossary concept."""

    __tablename__ = "term_variants"

    id: Mapped[int] = mapped_column(primary_key=True)
    concept_id: Mapped[str] = mapped_column(
        ForeignKey("term_concepts.id", ondelete="CASCADE"), index=True
    )
    language: Mapped[str] = mapped_column(String, index=True)
    text: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)


class ChapterAnalysisRow(Base):
    """Analysis result for a chapter: instructions, summary, and `Before You Read`."""

    __tablename__ = "chapter_analyses"
    __table_args__ = (UniqueConstraint("book_id", "section_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    section_id: Mapped[str] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    coverage_status: Mapped[str] = mapped_column(String)
    translation_instructions: Mapped[str] = mapped_column(Text)
    chapter_summary: Mapped[str] = mapped_column(Text)
    before_you_read_json: Mapped[list[dict[str, str]]] = mapped_column(JSON)
    content_plan_json: Mapped[dict[str, Any] | None] = mapped_column(ContentObjectJSON(), nullable=True)


class TreatmentPlanItemRow(Base):
    """Treatment mode for a single block in a chapter treatment plan."""

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
    """Glossary candidate extracted during chapter analysis."""

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
    """Processing run of a book with glossary and LLM settings snapshots."""

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
    """Processing result of a single block within a run."""

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
    content_evidence_json: Mapped[dict[str, Any] | None] = mapped_column(ContentObjectJSON(), nullable=True)


class LanguageNoteRow(Base):
    """Language note linked to exactly one block."""

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
    """Non-blocking warning for a block processing result."""

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
    """Local resumable book processing task.

    `scope_id` is an empty string rather than NULL for book-level tasks: in SQLite `NULL != NULL`,
    and a unique index on NULL values would not distinguish duplicate import tasks for the same book.
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
    """Technical log entry for a single model call.

    Completion response text is stored separately only when debugging is enabled.
    """

    __tablename__ = "llm_call_logs"
    __table_args__ = (
        Index("ix_llm_call_logs_book_id_stage", "book_id", "stage"),
        Index("ix_llm_call_logs_task_id", "task_id"),
        Index("uq_llm_call_logs_attempt_id", "attempt_id", unique=True),
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
    response_text: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    cost: Mapped[Decimal | None] = mapped_column(DecimalAsText)
    attempt_id: Mapped[str | None] = mapped_column(String)
    diagnostic_meta_json: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))


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


class SystemSettingsRow(Base):
    """Singleton row containing application settings overrides."""

    __tablename__ = "system_settings"
    __table_args__ = (
        CheckConstraint("id = 1", name="singleton"),
        CheckConstraint("revision > 0", name="revision_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    revision: Mapped[int]
    overrides_json: Mapped[dict[str, object] | None] = mapped_column(JSON(none_as_null=True))
    api_key: Mapped[str | None] = mapped_column(Text)
    proxy_url: Mapped[str | None] = mapped_column(Text)


class BookConditionsRow(Base):
    """Persisted book conditions."""

    __tablename__ = 'book_conditions'
    __table_args__ = (UniqueConstraint('book_id', 'revision'), CheckConstraint("revision >= 0", name="revision_nonnegative"), CheckConstraint("schema_version >= 0", name="schema_version_nonnegative"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    book_id: Mapped[str] = mapped_column(String, ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reproducibility: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

class WorkBindingRow(Base):
    """Persisted work bindings."""

    __tablename__ = 'work_bindings'
    entity_kind: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    entity_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    book_id: Mapped[str] = mapped_column(String, ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    conditions_id: Mapped[str] = mapped_column(String, ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    attempt_id: Mapped[str | None] = mapped_column(String, ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    artifact_revision_id: Mapped[str | None] = mapped_column(String, ForeignKey('artifact_revisions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

class ArtifactRevisionRow(Base):
    """Persisted artifact revisions."""

    __tablename__ = 'artifact_revisions'
    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    book_id: Mapped[str] = mapped_column(String, ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    conditions_id: Mapped[str | None] = mapped_column(String, ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str] = mapped_column(String, nullable=False)
    body_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    attempt_id: Mapped[str | None] = mapped_column(String, ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

class RecoveryRootRow(Base):
    """Persisted recovery roots."""

    __tablename__ = 'recovery_roots'
    __table_args__ = (CheckConstraint("generation >= 0", name="generation_nonnegative"), CheckConstraint("calls_used >= 0", name="calls_used_nonnegative"), CheckConstraint("extra_calls >= 0", name="extra_calls_nonnegative"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    book_id: Mapped[str] = mapped_column(String, ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    conditions_id: Mapped[str] = mapped_column(String, ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    root_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    calls_used: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)

class RecoveryScopeRow(Base):
    """Persisted recovery scopes."""

    __tablename__ = 'recovery_scopes'
    __table_args__ = (CheckConstraint("generation >= 0", name="generation_nonnegative"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    root_id: Mapped[str] = mapped_column(String, ForeignKey('recovery_roots.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String, ForeignKey('recovery_scopes.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    ordered_blocks_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    accepted_attempt_id: Mapped[str | None] = mapped_column(String, ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED", use_alter=True), nullable=True)

class LLMAttemptRow(Base):
    """Persisted llm attempts."""

    __tablename__ = 'llm_attempts'
    __table_args__ = (UniqueConstraint('root_id', 'ordinal'), UniqueConstraint('review_id', 'ordinal'), CheckConstraint("generation >= 0", name="generation_nonnegative"), CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    book_id: Mapped[str] = mapped_column(String, ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    conditions_id: Mapped[str | None] = mapped_column(String, ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    root_id: Mapped[str | None] = mapped_column(String, ForeignKey('recovery_roots.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    scope_id: Mapped[str | None] = mapped_column(String, ForeignKey('recovery_scopes.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    review_id: Mapped[str | None] = mapped_column(String, ForeignKey('quality_reviews.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    context_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    outcome_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

class QualityReviewRow(Base):
    """Persisted quality reviews."""

    __tablename__ = 'quality_reviews'
    __table_args__ = (UniqueConstraint('book_id', 'request_key'), CheckConstraint("revision >= 0", name="revision_nonnegative"), CheckConstraint("budget_limit >= 0", name="budget_limit_nonnegative"), CheckConstraint("spent >= 0", name="spent_nonnegative"), CheckConstraint("reserved >= 0", name="reserved_nonnegative"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    book_id: Mapped[str] = mapped_column(String, ForeignKey('books.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    conditions_id: Mapped[str | None] = mapped_column(String, ForeignKey('book_conditions.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=True)
    request_key: Mapped[str] = mapped_column(String, nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    findings_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    budget_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    spent: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved: Mapped[int] = mapped_column(Integer, nullable=False)

class QualityReservationRow(Base):
    """Persisted quality reservations."""

    __tablename__ = 'quality_reservations'
    __table_args__ = (CheckConstraint("estimate >= 0", name="estimate_nonnegative"), CheckConstraint("actual >= 0", name="actual_nonnegative"),)
    attempt_id: Mapped[str] = mapped_column(String, ForeignKey('llm_attempts.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), primary_key=True, nullable=False)
    review_id: Mapped[str] = mapped_column(String, ForeignKey('quality_reviews.id', ondelete="CASCADE", deferrable=True, initially="DEFERRED"), nullable=False)
    estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    actual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=False)
    price_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
