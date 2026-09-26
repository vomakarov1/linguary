"""Состав результата обработки блока по его режиму (PRD § 14-15)."""

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
    """Для `TRANSLATE` полный перевод обязателен."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(block_id=BLOCK_ID, treatment=BlockTreatment.TRANSLATE)


def test_translate_accepts_translation_with_hints() -> None:
    """Для `TRANSLATE` словарные подсказки допустимы вместе с переводом."""
    result = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Был холодный ясный апрельский день.",
        glossary_hints=(HINT,),
    )
    assert result.translated_text


def test_glossary_only_requires_hints() -> None:
    """Для `GLOSSARY_ONLY` словарные подсказки обязательны."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(block_id=BLOCK_ID, treatment=BlockTreatment.GLOSSARY_ONLY)


def test_glossary_only_forbids_full_translation() -> None:
    """Для `GLOSSARY_ONLY` полного перевода быть не должно."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            translated_text="Был холодный ясный апрельский день.",
            glossary_hints=(HINT,),
        )


def test_read_as_is_forbids_translation_and_hints() -> None:
    """Для `READ_AS_IS` нет ни перевода, ни словарных подсказок."""
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.READ_AS_IS,
            glossary_hints=(HINT,),
        )


def test_read_as_is_allows_language_note() -> None:
    """`Language Notes` возможны при любом текстовом режиме."""
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
    """Блок, не попадающий в LLM, результата обработки не имеет."""
    with pytest.raises(NonTextualBlockError):
        BlockProcessingResult(block_id=BLOCK_ID, treatment=treatment)


def test_note_belongs_to_its_own_block() -> None:
    """Заметка привязана ровно к тому блоку, чей это результат."""
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
    """Предупреждение сохраняется вместе с результатом и обработку не отменяет."""
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
    """Словарная подсказка бессмысленна без оригинала или перевода."""
    with pytest.raises(BlockResultError):
        GlossaryHint(original="chin", translation="   ")
