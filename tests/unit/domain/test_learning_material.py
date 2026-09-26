"""Учебные материалы главы и журнал вызовов LLM (PRD § 19, § 24)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from linguary.domain import (
    BeforeYouReadEntry,
    DomainError,
    ErrorType,
    LanguageNote,
    LearningMaterial,
    LLMCallLog,
    LLMCallLogError,
    LLMCallStage,
    TaskError,
    TokenUsage,
    build_language_note_id,
)

SECTION_ID = "section-1"


def make_note(ordinal: int, block_id: str) -> LanguageNote:
    """Заметка с идентификатором из собственного пространства имён."""
    return LanguageNote(
        id=build_language_note_id(chapter_number=3, ordinal=ordinal),
        block_id=block_id,
        original_excerpt="the clocks were striking thirteen",
        note="Часы бьют тринадцать: сигнал перевёрнутого мира.",
    )


def test_material_collects_notes_of_one_block() -> None:
    """Заметки показываются рядом с блоком, к которому относятся."""
    material = LearningMaterial.for_chapter(
        section_id=SECTION_ID,
        before_you_read=[
            BeforeYouReadEntry(
                expression="to strike thirteen",
                explanation="Часы не бьют тринадцать: сигнал, что мир изменился.",
            )
        ],
        chapter_summary="Winston starts a diary.",
        language_notes=[make_note(1, "block-1"), make_note(2, "block-2")],
    )
    assert tuple(note.id for note in material.notes_for("block-1")) == ("ln-ch03-001",)


def test_material_rejects_duplicate_note_ids() -> None:
    """Идентификаторы заметок главы уникальны."""
    with pytest.raises(DomainError):
        LearningMaterial(
            section_id=SECTION_ID,
            language_notes=(make_note(1, "block-1"), make_note(1, "block-2")),
        )


def test_note_requires_block() -> None:
    """Групповых заметок нет: каждая привязана ровно к одному блоку."""
    with pytest.raises(DomainError):
        LanguageNote(
            id="ln-ch03-001",
            block_id="",
            original_excerpt="the clocks were striking thirteen",
            note="Часы бьют тринадцать.",
        )


def test_before_you_read_entry_requires_explanation() -> None:
    """Выражение без объяснения учебным материалом не является."""
    with pytest.raises(DomainError):
        BeforeYouReadEntry(expression="to strike thirteen", explanation="  ")


def make_log(*, success: bool = True, error: TaskError | None = None) -> LLMCallLog:
    """Запись журнала об одном вызове модели."""
    return LLMCallLog(
        id="call-1",
        book_id="book-1",
        stage=LLMCallStage.CHAPTER_ANALYSIS,
        provider="openai",
        model="gpt-test",
        started_at=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
        duration_ms=1200,
        attempt=1,
        success=success,
        error=error,
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=250),
        cost=Decimal("0.0125"),
    )


def test_call_log_keeps_diagnostics_without_payload() -> None:
    """Журнал хранит диагностику вызова, но не текст запроса и ответа."""
    log = make_log()
    assert log.usage is not None
    assert log.usage.total_tokens == 1250
    assert not any(field in LLMCallLog.__slots__ for field in ("request", "response"))


def test_failed_call_requires_error() -> None:
    """Неуспешный вызов без ошибки в журнал не пишется."""
    with pytest.raises(LLMCallLogError):
        make_log(success=False)


def test_successful_call_has_no_error() -> None:
    """Успешный вызов ошибку не содержит."""
    with pytest.raises(LLMCallLogError):
        make_log(
            success=True,
            error=TaskError(type=ErrorType.TRANSIENT, message="Провайдер вернул 429"),
        )


def test_token_usage_without_provider_data() -> None:
    """Провайдер может не сообщить объём токенов."""
    assert TokenUsage().total_tokens is None
