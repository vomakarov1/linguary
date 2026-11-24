"""SQLAlchemy ORM models for optional book illustration preparation (SQL-01)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from linguary.adapters.db.base import Base, UtcDateTime


class IllustrationSettingsRow(Base):
    """Immutable per-book illustration settings revision."""

    __tablename__ = "illustration_settings"
    __table_args__ = (
        UniqueConstraint("book_id", "revision"),
        CheckConstraint("revision > 0", name="revision_positive"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(
        String, ForeignKey("books.id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    settings_sha256: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class IllustrationGenerationRow(Base):
    """Tracked generation of illustration preparation for a book."""

    __tablename__ = "illustration_generations"
    __table_args__ = (
        UniqueConstraint("book_id", "generation_number"),
        CheckConstraint("generation_number > 0", name="generation_number_positive"),
        CheckConstraint(
            "state IN ('waiting_source','queued','running','ready','failed','skipped','revoked')",
            name="state_valid",
        ),
        CheckConstraint("calls_reserved >= 0", name="calls_reserved_nonnegative"),
        CheckConstraint("calls_sent >= 0", name="calls_sent_nonnegative"),
        Index(
            "uq_illustration_generations_active_request",
            "book_id",
            "request_key",
            unique=True,
            sqlite_where=text("state != 'revoked'"),
        ),
        Index(
            "ix_illustration_generation_history",
            "book_id",
            "generation_number",
            "created_at",
        ),
        Index("ix_illustration_generation_settings", "settings_id"),
        Index("ix_illustration_generation_registry", "book_id", "registry_sha256"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    book_id: Mapped[str] = mapped_column(
        String, ForeignKey("books.id", ondelete="CASCADE"), nullable=False
    )
    generation_number: Mapped[int] = mapped_column(Integer, nullable=False)
    settings_id: Mapped[str] = mapped_column(
        String, ForeignKey("illustration_settings.id", ondelete="CASCADE"), nullable=False
    )
    request_key: Mapped[str] = mapped_column(String, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String, nullable=False)
    source_authority_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    requested_sections_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    manifest_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    manifest_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    registry_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String, nullable=True)
    calls_reserved: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    calls_sent: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class IllustrationHeadRow(Base):
    """Pointer to current illustration settings revision and active generation for a book."""

    __tablename__ = "illustration_heads"
    __table_args__ = (
        Index("ix_illustration_heads_settings", "settings_id"),
        Index("ix_illustration_heads_active_generation", "active_generation_id"),
    )

    book_id: Mapped[str] = mapped_column(
        String, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True
    )
    settings_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("illustration_settings.id", ondelete="SET NULL"), nullable=True
    )
    active_generation_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("illustration_generations.id", ondelete="SET NULL"), nullable=True
    )


class IllustrationScopeRow(Base):
    """Individual work unit within an illustration generation."""

    __tablename__ = "illustration_scopes"
    __table_args__ = (
        UniqueConstraint("generation_id", "stage", "scope_key"),
        CheckConstraint(
            "stage IN ('extract','alias','design','fact_scope','candidates','selection','prompt','chapter')",
            name="stage_valid",
        ),
        CheckConstraint(
            "state IN ('pending','running','accepted','failed','skipped','split')",
            name="state_valid",
        ),
        Index(
            "ix_illustration_scopes_generation",
            "generation_id",
            "stage",
            "state",
            "section_id",
        ),
        Index("ix_illustration_scopes_parent", "parent_scope_id"),
        Index("ix_illustration_scopes_reuse", "stage", "input_sha256", "state"),
        Index("ix_illustration_scopes_task", "task_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    generation_id: Mapped[str] = mapped_column(
        String, ForeignKey("illustration_generations.id", ondelete="CASCADE"), nullable=False
    )
    parent_scope_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("illustration_scopes.id", ondelete="CASCADE"), nullable=True
    )
    stage: Mapped[str] = mapped_column(String, nullable=False)
    scope_key: Mapped[str] = mapped_column(String, nullable=False)
    section_id: Mapped[str | None] = mapped_column(String, nullable=True)
    section_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    input_sha256: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String, nullable=True)
    attempt_id: Mapped[str | None] = mapped_column(String, nullable=True)
    output_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class IllustrationAttemptRow(Base):
    """Physical LLM call attempt for an illustration scope."""

    __tablename__ = "illustration_attempts"
    __table_args__ = (
        UniqueConstraint("scope_id", "ordinal"),
        CheckConstraint("ordinal > 0", name="ordinal_positive"),
        CheckConstraint(
            "state IN ('reserved','sent','accepted','rejected','failed','interrupted','stale','cancelled')",
            name="state_valid",
        ),
        Index(
            "uq_illustration_attempts_active_scope",
            "scope_id",
            unique=True,
            sqlite_where=text("state IN ('reserved', 'sent')"),
        ),
        Index("ix_illustration_attempt_scope", "scope_id", "state"),
        Index("ix_illustration_attempt_state", "state"),
        Index("ix_illustration_attempt_task", "task_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    scope_id: Mapped[str] = mapped_column(
        String, ForeignKey("illustration_scopes.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    request_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    request_meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    outcome_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class IllustrationRegistryItemRow(Base):
    """Canonical entity, fact, alias decision, or design item for a generation."""

    __tablename__ = "illustration_registry_items"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('entity','fact','alias_decision','design')",
            name="kind_valid",
        ),
        CheckConstraint("is_back_applicable IN (0, 1)", name="is_back_applicable_bool"),
        Index(
            "ix_illustration_registry_entity",
            "generation_id",
            "entity_id",
            "kind",
            "position_key",
        ),
        Index(
            "ix_illustration_registry_back",
            "generation_id",
            "entity_id",
            "is_back_applicable",
        ),
    )

    generation_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("illustration_generations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    kind: Mapped[str] = mapped_column(String, primary_key=True)
    item_key: Mapped[str] = mapped_column(String, primary_key=True)
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    position_key: Mapped[str] = mapped_column(String, nullable=False)
    is_back_applicable: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    payload_sha256: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


__all__ = [
    "IllustrationAttemptRow",
    "IllustrationGenerationRow",
    "IllustrationHeadRow",
    "IllustrationRegistryItemRow",
    "IllustrationScopeRow",
    "IllustrationSettingsRow",
]
