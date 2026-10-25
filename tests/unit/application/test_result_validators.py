"""Blocking validations and warnings for processing results (P3.T6, ALG-04, PRD § 15)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest
from processing_fakes import (
    SOURCE_TEXT,
    TARGET_LANGUAGE,
    TRANSLATED_TEXT,
    make_block,
    make_block_result,
    make_blocks,
    make_hint,
    make_note,
    make_response,
    make_unit,
    make_validator,
    note_ids,
    translated_response,
)

from linguary.adapters.config import CheckSettings
from linguary.application.llm_dto import (
    BlockResultResponse,
    GlossaryHintResponse,
    TranslationResultResponse,
)
from linguary.application.retry_policy import (
    ExpectedBlock,
    ResultExpectation,
    ResultValidationError,
)
from linguary.application.validators.tails import normalized_tail
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    InlineAnnotation,
    InlineAnnotationType,
    ResultWarningKind,
    TranslationUnit,
)

#: Full translation longer than the language check threshold, remaining in the source language.
ENGLISH_TRANSLATION = (
    "It is a truth universally acknowledged that a single man in possession "
    "of a good fortune must be in want of a wife."
)


def validate(
    payload: TranslationResultResponse,
    *,
    unit: TranslationUnit,
    truncated: bool = False,
    checks: CheckSettings | None = None,
) -> tuple[BlockProcessingResult, ...]:
    """Validates the model response against application thresholds."""
    return make_validator(checks).validate(
        payload,
        unit=unit,
        target_language=TARGET_LANGUAGE,
        note_id_factory=note_ids(),
        truncated=truncated,
    )


def rejection(
    payload: TranslationResultResponse,
    *,
    unit: TranslationUnit,
    truncated: bool = False,
    checks: CheckSettings | None = None,
) -> ResultValidationError:
    """Rejection of the unit returned by validation."""
    with pytest.raises(ResultValidationError) as rejected:
        validate(payload, unit=unit, truncated=truncated, checks=checks)
    return rejected.value


def with_fields(block: BlockResultResponse, **fields: object) -> BlockResultResponse:
    """A model response that would pass adapter schema validation but violates structural rules.

    The DTO checks block structure against its self-declared treatment, so such a response
    cannot be constructed normally. Validating structure against the plan is the last line of
    ALG-04, and it must be tested against a response that bypassed DTO rules.
    """
    for name, value in fields.items():
        object.__setattr__(block, name, value)
    return block


def test_valid_result_is_accepted() -> None:
    """A valid model response is converted into unit block processing results."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)

    results = validate(translated_response(unit), unit=unit)

    assert tuple(result.block_id for result in results) == unit.block_ids
    assert all(result.translated_text == TRANSLATED_TEXT for result in results)


def test_translation_with_extra_paragraphs_rejected() -> None:
    block = make_block(order=0)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Первый.\n\nВторой."),)
    )

    error = rejection(payload, unit=unit)

    assert f"перевод блока {block.id} содержит 2 абзацев, а оригинал — 1" in str(error)
    assert "не включай соседние" in str(error)


@pytest.mark.parametrize("source", ["First.", "First.\n\nSecond."])
def test_extra_lines_retry_message_requires_full_own_text(source: str) -> None:
    block = make_block(order=0, text=source)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Первый.\nВторой.\nТретий."),)
    )

    message = str(rejection(payload, unit=unit))
    source_count = sum(bool(line.strip()) for line in source.splitlines())
    assert f"содержит 3 абзацев, а оригинал — {source_count}" in message
    assert f"сохрани исходное число непустых строк ({source_count})" in message
    assert "весь текст своего блока" in message
    assert "не включай соседние" in message
    assert "объедини собственную реплику или авторское окончание" in message
    assert "не удаляй их" in message


def test_real_neighbour_heading_remains_rejected() -> None:
    """Synthetic case 31d97195: an extraneous heading cannot be accepted or merged."""
    block = make_block(order=0, text='"We shall return."')
    unit = make_unit((block,))
    own_translation = "«Мы вернёмся»."
    payload = make_response(
        (make_block_result(block, translated_text=own_translation + "\nГлава следующая."),)
    )

    assert "содержит 2 абзацев, а оригинал — 1" in str(rejection(payload, unit=unit))
    corrected = make_response((make_block_result(block, translated_text=own_translation),))
    assert validate(corrected, unit=unit)[0].translated_text == own_translation


def test_translation_with_equal_paragraphs_accepted() -> None:
    block = make_block(order=0)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Перевод.\n\n"),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Перевод.\n\n"


def test_multiline_source_allows_same_line_count() -> None:
    block = make_block(order=0, text="First.\n\nSecond.")
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Первый.\n\nВторой."),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Первый.\n\nВторой."


def test_blocks_without_translation_skipped() -> None:
    block = make_block(order=0)
    unit = make_unit((block,), treatments=(BlockTreatment.GLOSSARY_ONLY,))
    payload = make_response(
        (
            make_block_result(
                block,
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(make_hint(),),
            ),
        )
    )

    assert validate(payload, unit=unit)[0].translated_text is None


def test_colon_dash_tail_becomes_colon() -> None:
    block = make_block(order=0, text="He said:")
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Он сказал: —  "),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Он сказал:"


def test_four_dots_tail_becomes_ellipsis() -> None:
    block = make_block(order=0)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Он замолчал....  "),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Он замолчал…"


def test_inner_dash_untouched() -> None:
    block = make_block(order=0)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Он сказал: — я вернусь."),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Он сказал: — я вернусь."


def test_normalized_result_passes_truncation_check() -> None:
    block = make_block(order=0)
    unit = make_unit((block,))
    payload = make_response((make_block_result(block, translated_text="Он сказал:—"),))

    assert validate(payload, unit=unit)[0].translated_text == "Он сказал:"


@pytest.mark.parametrize("text", ["«Он замолчал....»", "«Он сказал:—»"])
def test_tail_before_closing_quote_untouched(text: str) -> None:
    assert normalized_tail(text) == text


def test_result_takes_treatment_from_plan() -> None:
    """The treatment of the saved result is taken from the plan, not from the model response."""
    blocks = make_blocks(2)
    unit = make_unit(
        blocks, treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)
    )

    results = validate(translated_response(unit), unit=unit)

    assert tuple(result.treatment for result in results) == (
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
    )
    assert results[1].glossary_hints[0].translation == "заповедный"


def test_language_notes_get_identifiers_from_factory() -> None:
    """Response language notes are attached to their blocks and receive scenario identifiers."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0]), make_block_result(blocks[1])),
        language_notes=(make_note(blocks[1]), make_note(blocks[1])),
    )

    results = validate(payload, unit=unit)

    assert results[0].language_notes == ()
    assert tuple(note.id for note in results[1].language_notes) == (
        "ln-ch01-001",
        "ln-ch01-002",
    )


def test_missing_block_discards_the_whole_unit() -> None:
    """A missing block discards the entire unit from being saved (PRD § 17)."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    payload = make_response((make_block_result(blocks[0]),))

    assert "нет результата" in str(rejection(payload, unit=unit))


def test_unknown_block_discards_the_whole_unit() -> None:
    """A result for a block outside the unit rejects the response completely."""
    blocks = make_blocks(2)
    unit = make_unit(blocks[:1])
    payload = make_response(
        (make_block_result(blocks[0]), make_block_result(blocks[1]))
    )

    assert "вне группы" in str(rejection(payload, unit=unit))


def test_repeated_block_is_rejected() -> None:
    """A duplicated block is a duplicate rather than reordering: the error message says so."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    payload = make_response(
        (
            make_block_result(blocks[0]),
            make_block_result(blocks[0]),
            make_block_result(blocks[1]),
        )
    )

    assert "дважды" in str(rejection(payload, unit=unit))


def test_reordered_blocks_are_rejected() -> None:
    """The order of blocks in the response must match the order of blocks in the unit."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[1]), make_block_result(blocks[0]))
    )

    assert "порядок блоков в ответе" in str(rejection(payload, unit=unit))


def test_note_for_block_outside_the_unit_is_rejected() -> None:
    """A note for an extraneous block is an unknown identifier, just in another field."""
    blocks = make_blocks(2)
    unit = make_unit(blocks[:1])
    payload = make_response(
        (make_block_result(blocks[0]),), language_notes=(make_note(blocks[1]),)
    )

    assert "заметки привязаны к блокам вне группы" in str(rejection(payload, unit=unit))


def test_treatment_mismatch_discards_the_whole_unit() -> None:
    """A treatment in the model response does not override the plan (PRD § 14)."""
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.GLOSSARY_ONLY,))
    payload = make_response((make_block_result(blocks[0]),))

    assert "не совпадает с планом" in str(rejection(payload, unit=unit))


def test_translate_without_translation_is_rejected() -> None:
    """A `TRANSLATE` block without full translation is rejected (PRD § 15)."""
    blocks = make_blocks(1)
    unit = make_unit(blocks)
    payload = make_response(
        (with_fields(make_block_result(blocks[0]), translated_text=None),)
    )

    assert "нет полного перевода" in str(rejection(payload, unit=unit))


def test_read_as_is_with_glossary_hints_is_rejected() -> None:
    """A `READ_AS_IS` block receives neither translation nor glossary hints (PRD § 10.1)."""
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.READ_AS_IS,))
    payload = make_response(
        (
            with_fields(
                make_block_result(
                    blocks[0],
                    treatment=BlockTreatment.READ_AS_IS,
                    translated_text=None,
                ),
                glossary_hints=(make_hint(),),
            ),
        )
    )

    assert "словарные подсказки недопустимы" in str(rejection(payload, unit=unit))


def test_glossary_only_without_hints_is_rejected() -> None:
    """A `GLOSSARY_ONLY` block without glossary hints is rejected."""
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.GLOSSARY_ONLY,))
    payload = make_response(
        (
            with_fields(
                make_block_result(
                    blocks[0],
                    treatment=BlockTreatment.GLOSSARY_ONLY,
                    translated_text=None,
                    glossary_hints=(make_hint(),),
                ),
                glossary_hints=(),
            ),
        )
    )

    assert "нет словарных подсказок" in str(rejection(payload, unit=unit))


def test_provider_token_limit_stop_is_rejected() -> None:
    """A response truncated by output token limit is rejected even if structurally complete."""
    blocks = make_blocks(1)
    unit = make_unit(blocks)

    rejected = rejection(translated_response(unit), unit=unit, truncated=True)

    assert "по лимиту выходных токенов" in str(rejected)


def test_translation_without_terminal_punctuation_is_rejected() -> None:
    """Translation of a finished sentence must end with terminal punctuation."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (
            make_block_result(
                blocks[0], translated_text="Общепризнанная истина гласит"
            ),
        )
    )

    assert "терминальной пунктуацией" in str(rejection(payload, unit=unit))


def test_heading_without_terminal_punctuation_is_accepted() -> None:
    """Heading translation does not have to end with a period: the source has none either."""
    blocks = (make_block(order=0, text="Chapter One"),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text="Глава первая"),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Глава первая"


def test_unbalanced_quote_is_rejected() -> None:
    """An unclosed quotation mark in translation is a sign of truncated generation."""
    blocks = (make_block(order=0, text='He said: "Go there."'),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text="Он сказал: «Иди туда."),)
    )

    assert "не закрыты кавычки" in str(rejection(payload, unit=unit))


def test_nested_russian_quotes_are_accepted() -> None:
    """Nested Russian quotation marks («... „...“») are balanced and accepted."""
    blocks = (make_block(order=0, text='“He said, ‘Welcome!’ to everyone.”'),)
    unit = make_unit(blocks)
    payload = make_response(
        (
            make_block_result(
                blocks[0], translated_text="«Он сказал: „Добро пожаловать!“ всем.»"
            ),
        )
    )

    assert validate(payload, unit=unit)


def test_quote_left_open_in_the_source_is_accepted() -> None:
    """Multi-paragraph direct speech opens quotes on each paragraph — not a truncation."""
    blocks = (make_block(order=0, text='"Go there. And do not look back.'),)
    unit = make_unit(blocks)
    payload = make_response(
        (
            make_block_result(
                blocks[0], translated_text="«Иди туда. И не оглядывайся."
            ),
        )
    )

    assert validate(payload, unit=unit)


def test_translation_in_the_wrong_language_is_rejected() -> None:
    """Full translation in the source language is rejected (PRD § 15)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=ENGLISH_TRANSLATION),)
    )

    assert "не на языке ru" in str(rejection(payload, unit=unit))


def test_short_translation_is_not_checked_for_language() -> None:
    """Short strings bypass language checking: proper nouns are legitimate translations."""
    blocks = (make_block(order=0, text="Mr. Bennet."),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text="Mr. Bennet."),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Mr. Bennet."


def test_glossary_hints_are_not_checked_for_language() -> None:
    """A glossary hint legitimately consists of a source language word (PRD § 15)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks, treatments=(BlockTreatment.GLOSSARY_ONLY,))
    payload = make_response(
        (
            make_block_result(
                blocks[0],
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(
                    GlossaryHintResponse(
                        original="entailed",
                        translation="заповедный",
                        note=f"{ENGLISH_TRANSLATION} {ENGLISH_TRANSLATION}",
                    ),
                ),
            ),
        )
    )

    assert validate(payload, unit=unit)[0].glossary_hints[0].original == "entailed"


def test_service_message_is_rejected() -> None:
    """A model refusal instead of translation does not enter the book."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (
            make_block_result(
                blocks[0],
                translated_text="Извините, но я не могу выполнить этот запрос.",
            ),
        )
    )

    assert "служебное сообщение модели" in str(rejection(payload, unit=unit))


def test_repetition_is_stored_as_warning() -> None:
    """An obvious repetition is recorded as a warning and does not halt processing."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    repeated = "и всё повторялось снова и снова "
    payload = make_response(
        (make_block_result(blocks[0], translated_text=f"{repeated * 3}конец."),)
    )

    results = validate(payload, unit=unit)

    assert results[0].warnings[0].kind is ResultWarningKind.REPETITION
    assert results[0].translated_text is not None


def test_result_without_repetition_has_no_warnings() -> None:
    """A normal translation produces no warnings."""
    blocks = make_blocks(1)
    unit = make_unit(blocks)

    assert validate(translated_response(unit), unit=unit)[0].warnings == ()


def test_rejection_carries_expected_blocks_and_treatments() -> None:
    """Rejection carries everything needed to build a retry prompt (ALG-05)."""
    blocks = make_blocks(2)
    unit = make_unit(
        blocks, treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY)
    )
    payload = make_response((make_block_result(blocks[0]),))

    expectation = rejection(payload, unit=unit).expectation

    assert expectation is not None
    assert expectation.target_language == TARGET_LANGUAGE
    assert expectation.block_ids == unit.block_ids
    assert tuple(block.treatment for block in expectation.blocks) == (
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
    )


def test_scenario_expectation_replaces_the_default() -> None:
    """A scenario addressing blocks by aliases lists those same aliases on retry."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    aliases = ResultExpectation(
        target_language=TARGET_LANGUAGE,
        blocks=(ExpectedBlock(block_id="b1"), ExpectedBlock(block_id="b2")),
    )

    with pytest.raises(ResultValidationError) as rejected:
        make_validator().validate(
            make_response((make_block_result(blocks[0]),)),
            unit=unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
            expectation=aliases,
        )

    assert rejected.value.expectation is aliases


@pytest.mark.parametrize(
    ("min_length", "expected"),
    [
        pytest.param(200, True, id="below-threshold"),
        pytest.param(10, False, id="above-threshold"),
    ],
)
def test_language_threshold_comes_from_settings(
    min_length: int, expected: bool
) -> None:
    """Minimum language check length is a setting, not a hardcoded validator constant."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=ENGLISH_TRANSLATION),)
    )
    checks = CheckSettings(language_check_min_length=min_length)

    assert _accepted(payload, unit=unit, checks=checks) is expected


def _accepted(
    payload: TranslationResultResponse,
    *,
    unit: TranslationUnit,
    checks: CheckSettings,
) -> bool:
    """Whether validation accepted the model response."""
    try:
        validate(payload, unit=unit, checks=checks)
    except ResultValidationError:
        return False
    return True


def test_results_keep_the_order_of_the_unit() -> None:
    """Results are returned in unit block order: storage expects this ordering."""
    blocks = make_blocks(3)
    unit = make_unit(blocks)

    results: Sequence[BlockProcessingResult] = validate(
        translated_response(unit), unit=unit
    )

    assert [result.block_id for result in results] == list(unit.block_ids)


def test_footnote_markers_stay_only_in_translation_and_only_known_ones() -> None:
    """The translation retains markers for existing footnote links; hints contain none (QA-08)."""
    text = "Nobody liked her face.1 It was a pity."
    block = replace(
        make_block(order=0, text=text),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=text.index("1"),
                end=text.index("1") + 1,
                target="section-1.xhtml#fn1",
            ),
        ),
    )
    unit = make_unit((block,))
    payload = make_response(
        (
            make_block_result(
                block,
                translated_text="Никому не нравилось её лицо,[^1] и жаль [^7].",
                glossary_hints=(make_hint(original="her face[^1]"),),
            ),
        ),
        language_notes=(make_note(block, original_excerpt="her face.[^1]"),),
    )

    (result,) = validate(payload, unit=unit)

    assert result.translated_text == "Никому не нравилось её лицо,[^1] и жаль."
    assert [hint.original for hint in result.glossary_hints] == ["her face"]
    assert [note.original_excerpt for note in result.language_notes] == ["her face."]


@pytest.mark.parametrize(
    "translated_text",
    [
        pytest.param("И стала ждать вместе с Selenой Роу.", id="latin-stem"),
        pytest.param("Рен посмотрела на Селенy.", id="latin-homoglyph"),
    ],
)
def test_word_mixing_latin_and_cyrillic_is_rejected(translated_text: str) -> None:
    """A word mixing Latin and Cyrillic scripts rejects the unit (H-1)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=translated_text),)
    )

    error = rejection(payload, unit=unit)

    assert "смеси латиницы и кириллицы" in str(error)
    assert error.expectation is not None


def test_mixed_word_in_language_note_is_rejected() -> None:
    """A note is written in the target language and validated just like translation."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0]),),
        language_notes=(make_note(blocks[0], note="Обращение к Selenой звучит сухо."),),
    )

    assert "смеси латиницы и кириллицы" in str(rejection(payload, unit=unit))


@pytest.mark.parametrize(
    "translated_text",
    [
        pytest.param("Рен Колдер пришла в Duskmere к Selene Row.", id="latin-names"),
        pytest.param("Это была настоящая Wonka-фабрика и mp3-плеер.", id="hyphen"),
        pytest.param("Мистер O'Брайен сказал: «Don't».", id="apostrophe"),
        pytest.param(
            "Весил он 5kg, как сказано на www.example.com[^1].", id="units-url"
        ),
        pytest.param("Кафе́ «Café» и Straße.", id="diacritics"),
        pytest.param(
            "Слово восходит к праславянскому vьlkodlakъ.", id="slavic-yers"
        ),
    ],
)
def test_latin_words_in_russian_text_are_allowed(translated_text: str) -> None:
    """Entirely Latin words and hyphenated compounds are not violations (H-1)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=translated_text),)
    )

    results = validate(payload, unit=unit)

    assert results[0].translated_text is not None


def test_retry_joins_own_authorial_tail_without_dropping_it() -> None:
    """Synthetic case dcfb5251: authorial tail remains preserved after joining."""
    block = make_block(order=0, text='"We shall return." He left.')
    unit = make_unit((block,))
    speech = "«Мы вернёмся»."
    tail = "Он ушёл."
    split = make_response((make_block_result(block, translated_text=f"{speech}\n{tail}"),))

    assert "не удаляй их" in str(rejection(split, unit=unit))
    joined = f"{speech} {tail}"
    corrected = make_response((make_block_result(block, translated_text=joined),))
    assert validate(corrected, unit=unit)[0].translated_text == joined
