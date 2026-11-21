"""Unit tests for the Russian dialogue punctuation validator (P2.T1)."""

from __future__ import annotations

import pytest
from processing_fakes import (
    make_block,
    make_block_result,
    make_response,
    make_unit,
)

from linguary.application.llm_dto import BlockResultResponse, SegmentTranslationResponse
from linguary.application.validators.dialogue import dialogue_violation
from linguary.domain import BlockTreatment, BlockType


def _check(
    translated_text: str,
    *,
    source_text: str = '"Hello," he said.',
    target_language: str = "ru",
    block_type: BlockType = BlockType.PARAGRAPH,
    pass_unit: bool = True,
) -> str | None:
    block = make_block(order=0, text=source_text, type=block_type)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text=translated_text),)
    )
    return dialogue_violation(
        payload,
        target_language=target_language,
        unit=unit if pass_unit else None,
    )


def test_dialogue_validator_rejects_hybrids_and_narration_in_quotes_but_allows_thoughts_and_poems() -> None:
    """Core P2.T1 contract: catches dialogue defects while allowing thoughts, poems, prose dashes, and non-ru."""
    # (a) Hybrids — « and «—
    assert _check("— «Привет», — сказал он.") is not None
    assert _check("«— Привет», — сказал он.") is not None

    # (b) Odd / unmatched guillemets « / »
    assert _check("«Привет, Гарри.") is not None
    assert _check("Привет, Гарри».") is not None

    # (c) Author narration / spoken verb inside «…»
    assert _check("«Привет, — сказал он, — как дела?»") is not None
    assert _check("«Иди сюда! — крикнул Хагрид.»") is not None

    # (d) Stray — between sentences of a single unbroken speech without author narration
    assert (
        _check(
            "— Я не знаю. — Может быть, завтра.",
            source_text='"I do not know. Maybe tomorrow."',
        )
        is not None
    )

    # (e) Paragraph-initial «…» with explicit spoken speech verb
    assert _check("«Привет», — сказал он.") is not None
    assert _check("«Куда ты идёшь?» — спросил Гарри.") is not None
    assert _check("«Стой!» — крикнул Хагрид.") is not None
    assert _check("«Тише…» — прошептала Гермиона.") is not None

    # Negative cases that MUST be accepted:
    # Internal thoughts in guillemets
    assert _check("«Какая странная сова», — подумал он.", source_text="'What a strange owl,' he thought.") is None
    assert _check("«Что же делать?» — подумал Гарри.", source_text="'What should I do?' Harry wondered.") is None
    assert _check("«Он прав», — решил Гарри.", source_text="'He is right,' Harry decided.") is None

    # Poems are skipped even if they contain unusual punctuation
    assert (
        _check(
            "— «Шляпа поёт», — сказал он.",
            source_text="Oh, you may not think I am pretty,",
            block_type=BlockType.POEM,
        )
        is None
    )

    # Ordinary explanatory prose dash
    assert _check("Он вошёл — и замер.", source_text="He walked in — and froze.") is None

    # Valid Russian dash dialogue
    assert _check("— Привет, — сказал он.") is None
    assert _check("— Привет, — сказал он. — Как дела?", source_text='"Hello," he said. "How are you?"') is None

    # Non-Russian target language is skipped
    assert _check("— «Hello», — he said.", target_language="de") is None


@pytest.mark.parametrize(
    "translated_text",
    [
        pytest.param("— «Иди сюда!»", id="leading-em-dash-guillemet"),
        pytest.param("– «Иди сюда!»", id="leading-en-dash-guillemet"),
        pytest.param("«— Иди сюда!»", id="guillemet-em-dash"),
        pytest.param("« — Иди сюда!»", id="guillemet-space-em-dash"),
        pytest.param("Он ответил: — «Иди сюда!»", id="colon-dash-guillemet"),
        pytest.param("— Да, — сказал он, — «но завтра».", id="comma-dash-guillemet"),
    ],
)
def test_dialogue_validator_catches_hybrid_variants(translated_text: str) -> None:
    assert _check(translated_text) is not None


def test_dialogue_validator_allows_explanatory_dash_before_quoted_title() -> None:
    """An explanatory prose dash before a quoted book/sign title is not a dialogue hybrid."""
    assert (
        _check(
            "Он открыл первую книгу — «Историю магии».",
            source_text="He opened the first book — A History of Magic.",
        )
        is None
    )


def test_dialogue_validator_allows_embedded_short_quote_in_narration() -> None:
    """Short inline quote inside narration (not starting the paragraph with «) is valid."""
    assert (
        _check(
            "Он тихо сказал: «Привет».",
            source_text='He quietly said, "Hello."',
        )
        is None
    )


def test_dialogue_validator_allows_unclosed_quote_matching_multi_paragraph_source() -> None:
    """Multi-paragraph direct speech left open in source is not falsely rejected as unmatched."""
    assert (
        _check(
            "«Иди туда. И не оглядывайся.",
            source_text='"Go there. And do not look back.',
        )
        is None
    )
    assert (
        _check(
            "«Иди туда, не оглядывайся и жди у ворот мистера Беннета.",
            source_text="‘Go there, don’t look back, and wait by Mr Bennet’s gate.",
        )
        is None
    )
    # Without unit/source, unclosed quote is still caught
    assert _check("«Иди туда. И не оглядывайся.", pass_unit=False) is not None


def test_dialogue_validator_stray_dash_checks_single_unbroken_speech_without_false_positives() -> None:
    """Stray — between sentences of one speaker is caught, while author attribution and prose dashes pass."""
    assert (
        _check(
            "— Подожди здесь. — Я скоро вернусь.",
            source_text="‘Wait here. I’ll be right back.’",
        )
        is not None
    )
    assert (
        _check(
            "— Подожди здесь! — Я скоро вернусь.",
            source_text='"Wait here! I will be right back."',
        )
        is not None
    )
    # Valid explanatory dash inside a single speech (no sentence break before dash)
    assert (
        _check(
            "— Подожди — что ты делаешь?",
            source_text='"Wait — what are you doing?"',
        )
        is None
    )
    # Valid author attribution between sentences
    assert (
        _check(
            "— Подожди здесь, — сказал он. — Я скоро вернусь.",
            source_text='"Wait here," he said. "I will be right back."',
        )
        is None
    )
    # Two separate speeches in original block
    assert (
        _check(
            "— Привет! — Привет!",
            source_text='"Hi!" "Hi!"',
        )
        is None
    )


def test_dialogue_validator_checks_segmented_translations() -> None:
    """Segmented block translations are joined and checked just like whole-block translations."""
    block = make_block(order=0, text='"Hello," he said. "How are you?"')
    unit = make_unit((block,))
    payload = make_response(
        (
            BlockResultResponse(
                source_block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=1, translated_text="— «Привет», — сказал он."),
                    SegmentTranslationResponse(index=2, translated_text="Как дела?"),
                ),
            ),
        )
    )
    assert dialogue_violation(payload, target_language="ru", unit=unit) is not None
