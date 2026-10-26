"""Chapter learning materials and LLM call log (PRD § 19, § 24)."""

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
    """A note with an identifier from its own namespace."""
    return LanguageNote(
        id=build_language_note_id(chapter_number=3, ordinal=ordinal),
        block_id=block_id,
        original_excerpt="the clocks were striking thirteen",
        note="Часы бьют тринадцать: сигнал перевёрнутого мира.",
    )


def test_material_collects_notes_of_one_block() -> None:
    """Notes are shown alongside the block they refer to."""
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
    """Chapter note identifiers are unique."""
    with pytest.raises(DomainError):
        LearningMaterial(
            section_id=SECTION_ID,
            language_notes=(make_note(1, "block-1"), make_note(1, "block-2")),
        )


def test_note_requires_block() -> None:
    """There are no grouped notes: each is attached strictly to a single block."""
    with pytest.raises(DomainError):
        LanguageNote(
            id="ln-ch03-001",
            block_id="",
            original_excerpt="the clocks were striking thirteen",
            note="Часы бьют тринадцать.",
        )


def test_before_you_read_entry_requires_explanation() -> None:
    """An expression without an explanation is not a learning material."""
    with pytest.raises(DomainError):
        BeforeYouReadEntry(expression="to strike thirteen", explanation="  ")


def make_log(*, success: bool = True, error: TaskError | None = None) -> LLMCallLog:
    """A log entry for a single model call."""
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
    """The log stores call diagnostics, but not request or response payloads."""
    log = make_log()
    assert log.usage is not None
    assert log.usage.total_tokens == 1250
    assert not any(field in LLMCallLog.__slots__ for field in ("request", "response"))


def test_failed_call_requires_error() -> None:
    """An unsuccessful call without an error is not recorded in the log."""
    with pytest.raises(LLMCallLogError):
        make_log(success=False)


def test_successful_call_has_no_error() -> None:
    """A successful call contains no error."""
    with pytest.raises(LLMCallLogError):
        make_log(
            success=True,
            error=TaskError(type=ErrorType.TRANSIENT, message="Провайдер вернул 429"),
        )


def test_token_usage_without_provider_data() -> None:
    """The provider may omit token counts."""
    assert TokenUsage().total_tokens is None


@pytest.mark.parametrize('kind,text', [('lexical_hint', 'only'), ('sentence_rephrase', 'She saw only the storm.'), ('russian_explanation', 'Она видела только бурю.'), ('unresolved', None)])
def test_fragment_assistance_preserves_exact_anchors(kind, text):
    from linguary.domain.learning import FragmentAssistance
    from linguary.domain.content_plan import SourceAnchor
    anchor = SourceAnchor('block', 5, 9, 'Echo')
    help = FragmentAssistance('d', ('p',), kind, text, (anchor,))
    assert help.anchors == (anchor,)
    assert help.resolved == (kind != 'unresolved')


@pytest.mark.parametrize('kind,text', [('unknown', 'x'), ('unresolved', 'x'), ('lexical_hint', None), ('sentence_rephrase', ' ')])
def test_fragment_assistance_rejects_invalid_body(kind, text):
    from linguary.domain.learning import FragmentAssistance
    from linguary.domain.content_plan import SourceAnchor
    with pytest.raises(DomainError):
        FragmentAssistance('d', ('p',), kind, text, (SourceAnchor('block', 0, 4, 'Echo'),))
