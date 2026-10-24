"""Block processing result structure by its treatment (PRD § 14-15)."""

from __future__ import annotations

import pytest

from linguary.domain import (
    BlockProcessingResult,
    BlockResultError,
    BlockTreatment,
    GlossaryHint,
    LanguageNote,
    NonTextualBlockError,
    ResultWarning,
    ResultWarningKind,
)

BLOCK_ID = "block-1"
HINT = GlossaryHint(original="bright cold day", translation="ясный морозный день")


def test_translate_requires_full_translation() -> None:
    """For 'TRANSLATE', a full translation is required."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(block_id=BLOCK_ID, treatment=BlockTreatment.TRANSLATE)


def test_translate_accepts_translation_with_hints() -> None:
    """For 'TRANSLATE', glossary hints are permitted alongside the translation."""
    result = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Был холодный ясный апрельский день.",
        glossary_hints=(HINT,),
    )
    assert result.translated_text


def test_glossary_only_requires_hints() -> None:
    """For 'GLOSSARY_ONLY', glossary hints are required."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(block_id=BLOCK_ID, treatment=BlockTreatment.GLOSSARY_ONLY)


def test_glossary_only_forbids_full_translation() -> None:
    """For 'GLOSSARY_ONLY', there must be no full translation."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            translated_text="Был холодный ясный апрельский день.",
            glossary_hints=(HINT,),
        )


def test_read_as_is_forbids_translation_and_hints() -> None:
    """For 'READ_AS_IS', there is neither translation nor glossary hints."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.READ_AS_IS,
            glossary_hints=(HINT,),
        )


def test_read_as_is_allows_language_note() -> None:
    """Language Notes are allowed for any textual treatment."""
    result = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.READ_AS_IS,
        language_notes=(
            LanguageNote(
                id="ln-ch03-001",
                block_id=BLOCK_ID,
                original_excerpt="the clocks were striking thirteen",
                note="Часы бьют тринадцать: сигнал перевёрнутого мира.",
            ),
        ),
    )
    assert result.translated_text is None
    assert result.language_notes[0].block_id == BLOCK_ID


@pytest.mark.parametrize(
    "treatment",
    [
        pytest.param(BlockTreatment.PRESERVE_ONLY, id="preserve-only"),
        pytest.param(BlockTreatment.DROP_WITH_WARNING, id="drop-with-warning"),
    ],
)
def test_non_textual_block_has_no_result(treatment: BlockTreatment) -> None:
    """A block that does not go to the LLM has no processing result."""
    with pytest.raises(NonTextualBlockError):
        BlockProcessingResult(block_id=BLOCK_ID, treatment=treatment)


def test_note_belongs_to_its_own_block() -> None:
    """A note is attached strictly to the block whose result it belongs to."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Был холодный ясный апрельский день.",
            language_notes=(
                LanguageNote(
                    id="ln-ch03-001",
                    block_id="block-2",
                    original_excerpt="the clocks were striking thirteen",
                    note="Часы бьют тринадцать: сигнал перевёрнутого мира.",
                ),
            ),
        )


def test_warnings_do_not_block_result() -> None:
    """A warning is saved alongside the result and does not cancel processing."""
    result = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Был холодный ясный апрельский день.",
        warnings=(
            ResultWarning(
                kind=ResultWarningKind.REPETITION, message="Повтор n-граммы в переводе"
            ),
        ),
    )
    assert result.warnings[0].kind is ResultWarningKind.REPETITION


def test_hint_requires_both_sides() -> None:
    """A glossary hint is meaningless without either the original or the translation."""
    with pytest.raises(BlockResultError):
        GlossaryHint(original="chin", translation="   ")
