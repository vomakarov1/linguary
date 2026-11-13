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
    SegmentTranslationResponse,
    TranslationResultResponse,
)
from linguary.application.retry_policy import (
    ExpectedBlock,
    ResultExpectation,
    ResultValidationError,
)
from linguary.application.validators import expectation_for_unit
from linguary.application.validators.tails import normalized_tail
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    InlineAnnotation,
    InlineAnnotationType,
    ResultWarningKind,
    TranslationUnit,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.treatment import strongest_treatment

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
    assert tuple(block.segment_treatments for block in expectation.blocks) == ((), ())


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


def test_dialogue_line_breaks_accepted_with_speech_markers() -> None:
    """Translation of a block with direct speech markers split into dialogue dashes is accepted (AC-009)."""
    from linguary.application.validators.paragraphs import paragraph_violation

    source = '"Hello," he said. "How are you?"'
    translation = "— Привет, — сказал он.\n— Как дела?"
    violation = paragraph_violation("block-1", source_text=source, translated_text=translation)
    assert violation is None

    block = make_block(order=0, text=source)
    payload = make_response((make_block_result(block, translated_text=translation),))
    results = validate(payload, unit=make_unit((block,)))
    assert results[0].translated_text == translation


def test_dialogue_line_breaks_rejected_without_speech_markers() -> None:
    """Extra lines are rejected if the source text lacks direct speech markers (AC-009)."""
    from linguary.application.validators.paragraphs import paragraph_violation

    source = "Hello he said how are you."
    translation = "— Привет, — сказал он.\n— Как дела?"
    violation = paragraph_violation("block-1", source_text=source, translated_text=translation)
    assert violation is not None
    assert violation.rule_code == "extra_paragraphs"


def test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash() -> None:
    """Extra lines are rejected if any extra line does not begin with a dialogue dash (AC-009)."""
    from linguary.application.validators.paragraphs import paragraph_violation

    source = '"Hello," he said. "How are you?"'
    translation = "— Привет, — сказал он.\nАвторский текст без тире."
    violation = paragraph_violation("block-1", source_text=source, translated_text=translation)
    assert violation is not None
    assert violation.rule_code == "extra_paragraphs"


def test_reader_text_policy_preserves_dialogue_newlines() -> None:
    """ReaderTextPolicy preserves newlines in translation (AC-009)."""
    from linguary.application.reader_text import ReaderTextPolicy

    text = "— Привет, — сказал он.\n— Как дела?"
    prepared = ReaderTextPolicy().prepare("translation", text)
    assert prepared == text
    assert "\n" in prepared


def test_resolve_response_evidence_algorithmically_binds_when_content_evidence_is_empty() -> None:
    """When LLM returns empty content_evidence, notes and hints are bound algorithmically (AC-008)."""
    from linguary.application.llm_dto import (
        GlossaryHintResponse,
        LanguageNoteResponse,
        TranslationResultResponse,
    )
    from linguary.application.validators.content_plan import resolve_response_evidence
    from linguary.domain import (
        BlockProcessingResult,
        BlockTreatment,
        GlossaryHint,
        LanguageNote,
    )
    from linguary.domain.content_plan import (
        ContentPlan,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )

    d1 = Difficulty(
        "d1",
        (SourceAnchor("b1", 0, 7, "grammar"),),
        "b1",
        "grammar",
        "critical",
        "explanation",
        (RequiredPoint("p1", "Explain grammar point"),),
    )
    d2 = Difficulty(
        "d2",
        (SourceAnchor("b1", 8, 12, "hint"),),
        "b1",
        "wordplay",
        "critical",
        "lexical_hint",
        (RequiredPoint("p2", "Explain hint point"),),
    )
    p = ContentPlan(version=2, profile="content-plan-v2", difficulties=(d1, d2))
    payload = TranslationResultResponse(
        content_evidence=(),  # Empty from LLM
    )
    res = BlockProcessingResult(
        "b1",
        BlockTreatment.TRANSLATE,
        translated_text="Перевод предложения.",
        language_notes=(LanguageNote("n1", "b1", "grammar", "Объяснение грамматической конструкции.", None),),
        glossary_hints=(GlossaryHint("hint", "подсказка", "Заметка к подсказке."),),
    )
    evidence = resolve_response_evidence(p, payload, (res,), (), owner_block_ids=("b1",))
    assert len(evidence.evidence) == 2
    by_diff = {e.difficulty_id: e for e in evidence.evidence}
    assert by_diff["d1"].point_id == "p1"
    assert by_diff["d1"].artifact_kind == "language_note.note"
    assert by_diff["d1"].body_excerpt == "Объяснение грамматической конструкции."
    assert by_diff["d2"].point_id == "p2"
    assert by_diff["d2"].artifact_kind == "glossary_hint.translation"
    assert by_diff["d2"].body_excerpt == "подсказка"


def test_grammar_difficulty_accepts_multiword_construction_in_longer_anchor_and_rejects_single_word_hint() -> None:
    """Multi-word grammar construction inside a full-sentence v5 anchor passes validation while 1-word hint does not."""
    from linguary.application.validators.content_plan import _quote_matches_anchor

    anchor = "Had the queen of Sheba lived in the flat across the airshaft"
    assert _quote_matches_anchor("Had the queen of Sheba lived", anchor, "grammar") is True
    assert _quote_matches_anchor("flat", anchor, "grammar") is False


def test_strict_json_schema_omits_start_end_from_pronunciation_anchor_only() -> None:
    from linguary.adapters.llm.json_schema import strict_json_schema
    from linguary.adapters.llm.schemas import (
        DifficultySchema,
        PronunciationCandidateSchema,
    )

    p_schema = strict_json_schema(PronunciationCandidateSchema)
    p_anchor_props = p_schema["$defs"]["PronunciationAnchorSchema"]["properties"]
    assert "quote" in p_anchor_props
    assert "block_id" in p_anchor_props
    assert "start" not in p_anchor_props
    assert "end" not in p_anchor_props

    diff_schema = strict_json_schema(DifficultySchema)
    src_anchor_props = diff_schema["$defs"]["SourceAnchorSchema"]["properties"]
    assert "quote" in src_anchor_props
    assert "block_id" in src_anchor_props
    assert "start" in src_anchor_props
    assert "end" in src_anchor_props


@pytest.mark.asyncio
async def test_verify_candidates_resolves_word_span_from_quote_without_offsets() -> None:
    from unittest.mock import AsyncMock
    from linguary.application.llm_dto import (
        PronunciationAnchorResponse,
        PronunciationCandidateResponse,
        PronunciationContextResponse,
    )
    from linguary.application.ports.pronunciation import (
        PronunciationEvidence,
        PronunciationSource,
        SourceEntry,
        SourceLookupResult,
    )
    from linguary.application.validators.pronunciation import (
        resolve_pronunciation_anchor,
        verify_candidates,
    )

    texts = {"b1": "A delightful zorp awaits."}
    anchor = PronunciationAnchorResponse(block_id="b1", quote="zorp")
    resolved = resolve_pronunciation_anchor(anchor, texts)
    assert resolved.block_id == "b1"
    assert resolved.start == 13
    assert resolved.end == 17
    assert resolved.quote == "zorp"
    resolved.validate(texts)

    candidate = PronunciationCandidateResponse(
        anchor=anchor,
        form="zorp",
        context=PronunciationContextResponse(
            meaning_in_context=None,
            part_of_speech=None,
            grammatical_form=None,
            is_proper_name=False,
        ),
        selection_reason="unpredictable_spelling",
    )

    source = AsyncMock(spec=PronunciationSource)
    source.configured = True
    sha = "a" * 40
    entry = SourceEntry(
        form="zorp",
        ipa="zɔːp",
        variant="uk",
        sense=None,
        grammatical_form=None,
        evidence=PronunciationEvidence(
            source_id="ipa-dict",
            source_version=f"ipa-dict@{sha}:parser-1",
            entry_id="en_UK.txt:1:zorp",
            sense_id=None,
            part_of_speech=None,
            dictionary_form="zorp",
            dictionary_ipa="zɔːp",
            attribution="IPA-Dict",
            kind="lexicon_form",
            license_id="GPL-3.0",
            source_uri=f"https://raw.githubusercontent.com/open-dict-data/ipa-dict/{sha}/data/en_UK.txt",
        ),
    )
    source.lookup.return_value = SourceLookupResult.found((entry,))

    selection = await verify_candidates(
        (candidate,),
        source=source,
        texts=texts,
        variant="uk",
        scope_id="run-1",
    )
    assert len(selection.accepted) == 1
    accepted = selection.accepted[0]
    assert accepted.form == "zorp"
    assert accepted.anchor.start == 13
    assert accepted.anchor.end == 17
    assert accepted.anchor.quote == "zorp"
    assert accepted.ipa == "zɔːp"
    assert len(selection.skipped) == 0


def test_result_validator_normalizes_tail_once_per_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.validators.result as result_module

    calls: list[str] = []
    original = result_module.normalized_tail

    def counting_normalized_tail(text: str) -> str:
        calls.append(text)
        return original(text)

    monkeypatch.setattr(result_module, "normalized_tail", counting_normalized_tail)

    block = make_block(order=0, text="He said:")
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text="Он сказал: —  "),)
    )

    results = validate(payload, unit=unit)

    assert results[0].translated_text == "Он сказал:"
    assert len(calls) == 1


def test_coverage_violation_detects_duplicates_in_linear_time() -> None:
    import inspect
    from linguary.application.validators.coverage import coverage_violation

    source = inspect.getsource(coverage_violation)
    assert "Counter(" in source
    assert "returned.count(" not in source

    blocks = make_blocks(3)
    unit = make_unit(blocks)
    payload = make_response(
        (
            make_block_result(blocks[1]),
            make_block_result(blocks[1]),
            make_block_result(blocks[0]),
            make_block_result(blocks[0]),
            make_block_result(blocks[2]),
        )
    )

    violation = coverage_violation(payload, unit=unit)
    assert violation == (
        f"результат для блоков возвращён дважды: {[blocks[1].id, blocks[0].id]}"
    )


def test_validator_attaches_structured_violation_with_rule_code_and_block_id() -> None:
    blocks = make_blocks(2)
    unit = make_unit(blocks)

    cases = [
        (
            make_response(
                (
                    make_block_result(blocks[0]),
                    make_block_result(
                        blocks[1],
                        treatment=BlockTreatment.READ_AS_IS,
                        translated_text=None,
                    ),
                )
            ),
            "treatment_mismatch",
            "Режим блока не совпадает с планом обработки",
        ),
        (
            make_response(
                (
                    make_block_result(blocks[0]),
                    with_fields(make_block_result(blocks[1]), translated_text=None),
                )
            ),
            "content_violation",
            "Содержимое блока не прошло структурную проверку",
        ),
        (
            make_response(
                (
                    make_block_result(blocks[0]),
                    make_block_result(blocks[1], translated_text="Первый.\n\nВторой."),
                )
            ),
            "paragraph_count_mismatch",
            "Нарушено число строк или абзацев перевода",
        ),
        (
            make_response(
                (
                    make_block_result(blocks[0]),
                    make_block_result(blocks[1], translated_text=ENGLISH_TRANSLATION),
                )
            ),
            "wrong_target_language",
            "Перевод выполнен не на целевом языке",
        ),
        (
            make_response(
                (
                    make_block_result(blocks[0]),
                    make_block_result(
                        blocks[1],
                        translated_text="Извините, но я не могу выполнить этот запрос.",
                    ),
                )
            ),
            "service_message_leak",
            "В ответе обнаружено служебное сообщение модели",
        ),
        (
            make_response(
                (
                    make_block_result(blocks[0]),
                    make_block_result(
                        blocks[1],
                        translated_text="И стала ждать вместе с Selenой Роу.",
                    ),
                )
            ),
            "mixed_script_leak",
            "В тексте обнаружено смешение алфавитов внутри слова",
        ),
    ]

    for payload, expected_rule_code, expected_safe_message in cases:
        error = rejection(payload, unit=unit)
        assert error.violation is not None
        assert error.violation.rule_code == expected_rule_code
        assert error.violation.stage == "translation_unit"
        assert error.violation.scope == "local"
        assert error.violation.block_ids == (blocks[1].id,)
        assert error.violation.safe_message == expected_safe_message


def test_truncation_allows_british_unclosed_opening_quote_and_rejects_broken_closed_quote() -> None:
    """British multi-paragraph dialogue opens single quote ‘ without closing ’ until the final paragraph."""
    from linguary.application.validators.truncation import _unbalanced, truncation_diagnostic

    unclosed_source = "‘Go there, don’t look back, and wait by Mr Bennet’s gate."
    open_translation = "«Иди туда, не оглядывайся и жди у ворот мистера Беннета."

    assert _unbalanced(open_translation, source=unclosed_source) is None
    block_open = make_block(order=0, text=unclosed_source)
    unit_open = make_unit((block_open,))
    payload_open = make_response(
        (make_block_result(block_open, translated_text=open_translation),)
    )
    assert validate(payload_open, unit=unit_open)[0].translated_text == open_translation
    assert truncation_diagnostic(payload_open, unit=unit_open, truncated=False) is None

    # Second quote in same paragraph left open for next paragraph (2 opening ‘ > 1 closing ’).
    multi_open_source = "‘Go there,’ he said, ‘and don’t look back."
    multi_open_translation = "«Иди туда, — сказал он, — и не оглядывайся."
    assert _unbalanced(multi_open_translation, source=multi_open_source) is None
    block_multi = make_block(order=0, text=multi_open_source)
    unit_multi = make_unit((block_multi,))
    payload_multi = make_response(
        (make_block_result(block_multi, translated_text=multi_open_translation),)
    )
    assert validate(payload_multi, unit=unit_multi)[0].translated_text == multi_open_translation

    # Leading whitespace before British opening quote is still allowed, and balance 0 (dash dialogue) remains allowed.
    assert _unbalanced("«Подожди здесь.", source="  ‘Wait here.") is None
    assert _unbalanced("— Иди туда, не оглядывайся.", source=unclosed_source) is None

    # Closed British single-quoted dialogue whose translation drops its closing quote MUST be rejected.
    closed_source = "‘Go there, don’t look back, and wait by Mr Bennet’s gate.’"
    assert _unbalanced(open_translation, source=closed_source) == "кавычки"
    block_closed = make_block(order=0, text=closed_source)
    unit_closed = make_unit((block_closed,))
    payload_broken_closed = make_response(
        (make_block_result(block_closed, translated_text=open_translation),)
    )
    err_closed = rejection(payload_broken_closed, unit=unit_closed)
    assert "не закрыты кавычки" in str(err_closed)
    assert err_closed.violation is not None
    assert err_closed.violation.rule_code == "quote_balance"

    # British logical punctuation closed quote (‘...’, he said.) dropping closing quote is also rejected.
    logical_closed_source = "‘Go there’, he said."
    logical_broken_translation = "«Иди туда, сказал он."
    assert _unbalanced(logical_broken_translation, source=logical_closed_source) == "кавычки"

    # Non-dialogue-initial ‘ or balance != +1 in translation remain rejected.
    assert (
        _unbalanced(
            "Он прошептал: «Иди туда и не оглядывайся.",
            source="He whispered, ‘Go there and don’t look back.",
        )
        == "кавычки"
    )
    assert _unbalanced("«Иди «туда.", source=unclosed_source) == "кавычки"
    assert _unbalanced("Иди туда.»", source=unclosed_source) == "кавычки"


def _segmented(
    parts: tuple[str, ...],
    treatments: tuple[BlockTreatment, ...],
    *,
    type: BlockType = BlockType.PARAGRAPH,
    annotations: tuple[InlineAnnotation, ...] = (),
) -> tuple[TranslationUnit, DocumentBlock]:
    """A multi-segment block and the unit `of_blocks` builds from its plan."""
    text = "".join(parts)
    offset = 0
    segments: list[BlockSegment] = []
    for index, part in enumerate(parts):
        segments.append(BlockSegment(offset, offset + len(part), index))
        offset += len(part)
    block = replace(
        make_block(order=0, text=text, type=type),
        segments=tuple(segments),
        inline_annotations=annotations,
    )
    plan = TreatmentPlan(
        section_id=block.section_id,
        items=(
            TreatmentPlanItem(
                block_id=block.id,
                treatment=strongest_treatment(treatments),
                segment_treatments=treatments,
                sentence_indexes=tuple(range(len(treatments))),
            ),
        ),
    )
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=(block,), plan=plan)
    return unit, block


def _segmented_response(
    block: DocumentBlock,
    translations: tuple[tuple[int, str], ...],
    *,
    treatment: BlockTreatment,
    hints: tuple[GlossaryHintResponse, ...] = (),
) -> TranslationResultResponse:
    """Segment response with `translated_text` left empty."""
    return make_response(
        (
            BlockResultResponse(
                source_block_id=block.id,
                treatment=treatment,
                translated_text=None,
                glossary_hints=hints,
                segments=tuple(
                    SegmentTranslationResponse(index=index, translated_text=text)
                    for index, text in translations
                ),
            ),
        )
    )


def test_missing_translate_segment_rejected() -> None:
    """Each TRANSLATE span needs a translation. A missing index is not optional."""
    unit, block = _segmented(
        ("First sentence.", " Second sentence."),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        ((1, "Первое предложение."),),
        treatment=BlockTreatment.TRANSLATE,
    )

    message = str(rejection(payload, unit=unit))

    assert "сегмента 2" in message
    assert "TRANSLATE" in message
    assert unit.items[0].segment_treatments == ()
    expectation = expectation_for_unit(unit, target_language=TARGET_LANGUAGE)
    assert expectation.blocks[0].segment_treatments == (
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    )


def test_translation_for_glossary_segment_rejected() -> None:
    """A translation is allowed only on a TRANSLATE span."""
    unit, block = _segmented(
        ("Rare word.", " Clear sentence."),
        (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        ((1, "Редкое слово."), (2, "Ясное предложение.")),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(),),
    )

    message = str(rejection(payload, unit=unit))

    assert "сегмента 1" in message
    assert "GLOSSARY_ONLY" in message
    assert "недопустим" in message


def test_duplicate_segment_index_rejected() -> None:
    """One index names one span. A repeated index is rejected."""
    unit, block = _segmented(
        ("First sentence.", " Second sentence."),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        ((1, "Первое предложение."), (1, "Снова первое.")),
        treatment=BlockTreatment.TRANSLATE,
    )

    assert "указан дважды" in str(rejection(payload, unit=unit))


def test_segment_index_outside_range_rejected() -> None:
    """Indexes are 1-based positions in `segment_spans` and cannot run past the end."""
    unit, block = _segmented(
        ("First sentence.", " Second sentence."),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        (
            (1, "Первое предложение."),
            (2, "Второе предложение."),
            (3, "Лишний перевод."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )

    message = str(rejection(payload, unit=unit))

    assert "вне диапазона" in message
    assert "1..2" in message


def test_segments_rejected_for_single_span_block() -> None:
    """Fewer than two stored segments keep the block translation. `segments` is rejected."""
    block = make_block(order=0)
    unit = make_unit((block,))
    payload = make_response(
        (
            BlockResultResponse(
                source_block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(SegmentTranslationResponse(index=1, translated_text="Перевод."),),
            ),
        )
    )

    assert "segments" in str(rejection(payload, unit=unit))


def test_mixed_block_result_built_from_segments() -> None:
    """A mixed block stores one mode and one translation slot per span, joined for the block."""
    unit, block = _segmented(
        ("Rare word.", " Clear sentence."),
        (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        ((2, "Ясное предложение."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(),),
    )

    (result,) = validate(payload, unit=unit)

    assert result.treatment is BlockTreatment.TRANSLATE
    assert result.segment_treatments == (
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
    )
    assert result.segment_translations == (None, "Ясное предложение.")
    assert result.translated_text == "Ясное предложение."
    assert result.glossary_hints[0].translation == "заповедный"
    assert expectation_for_unit(unit, target_language=TARGET_LANGUAGE).blocks[0].segment_treatments == (
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
    )


def test_footnote_markers_isolated_to_segment_range() -> None:
    """A `[^N]` marker is kept only in the segment whose span contains that reference."""
    left = "See note.1"
    right = " Then go."
    unit, block = _segmented(
        (left, right),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=left.index("1"),
                end=left.index("1") + 1,
                target="section-1.xhtml#fn1",
            ),
        ),
    )
    allowed = _segmented_response(
        block,
        ((2, "Потом иди."), (1, "Смотри.[^1][^7]")),
        treatment=BlockTreatment.TRANSLATE,
    )

    (result,) = validate(allowed, unit=unit)

    assert unit.items[0].segment_treatments == ()
    assert result.segment_treatments == (
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    )
    assert result.segment_translations == ("Смотри.[^1]", "Потом иди.")
    assert result.translated_text == "Смотри.[^1] Потом иди."

    foreign = _segmented_response(
        block,
        ((1, "Смотри."), (2, "Потом [^1] иди.")),
        treatment=BlockTreatment.TRANSLATE,
    )
    message = str(rejection(foreign, unit=unit))
    assert "[^1]" in message
    assert "сегмента 2" in message
    assert "другому сегменту" in message


def test_poem_stanza_and_line_count_match() -> None:
    """A poem keeps every line from `split('\\n')`, including a blank stanza line."""
    source = "Twinkle, twinkle.\n\nLittle star."
    block = make_block(order=0, text=source, type=BlockType.POEM)
    unit = make_unit((block,))
    matched = "Мерцай, мерцай.\n\nЗвёздочка."

    assert validate(
        make_response((make_block_result(block, translated_text=matched),)),
        unit=unit,
    )[0].translated_text == matched

    for translated in ("Мерцай, мерцай.\nЗвёздочка.", "Звёздочка."):
        message = str(
            rejection(
                make_response((make_block_result(block, translated_text=translated),)),
                unit=unit,
            )
        )
        assert "строк" in message
        assert "включая пустые" in message
        assert str(len(source.split("\n"))) in message
        assert str(len(translated.split("\n"))) in message


def test_truncation_validator_checks_only_translated_segments() -> None:
    """Quote pairing and terminal punctuation run on TRANSLATE slices, not the mixed block."""
    unit, block = _segmented(
        ('He said "hello', " She left."),
        (BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
    )
    accepted = _segmented_response(
        block,
        ((2, "Она ушла."),),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert validate(accepted, unit=unit)[0].segment_translations == (None, "Она ушла.")

    unfinished = _segmented_response(
        block,
        ((2, "Она ушла"),),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert "терминальной пунктуацией" in str(rejection(unfinished, unit=unit))

    open_quote = _segmented_response(
        block,
        ((2, "Она сказала «пока."),),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert "не закрыты кавычки" in str(rejection(open_quote, unit=unit))


def test_language_check_uses_joined_segment_translations() -> None:
    """Language is checked once on the joined translation, not on each short slice."""
    first = "It is a truth universally acknowledged in this town."
    second = "A single man in possession of fortune wants a wife."
    assert len(first) <= 80
    assert len(second) <= 80
    assert len(f"{first} {second}") > 80
    unit, block = _segmented(
        ("First sentence here.", " Second sentence here."),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        ((1, first), (2, second)),
        treatment=BlockTreatment.TRANSLATE,
    )

    message = str(rejection(payload, unit=unit))

    assert f"перевод блока {block.id} не на языке {TARGET_LANGUAGE}" in message


def test_service_message_in_segment_translation_rejected() -> None:
    """A service message inside one segment still rejects the block."""
    unit, block = _segmented(
        ("Do this now.", " And that too."),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        (
            (1, "Извините, но я не могу выполнить этот запрос."),
            (2, "И это тоже."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )

    assert "не могу выполнить" in str(rejection(payload, unit=unit))


def test_segment_translation_tail_is_normalized() -> None:
    """A false truncation tail is normalized on the segment before it is stored."""
    unit, block = _segmented(
        ("He said:", " Done."),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    payload = _segmented_response(
        block,
        ((1, "Он сказал:—"), (2, "Готово.")),
        treatment=BlockTreatment.TRANSLATE,
    )

    (result,) = validate(payload, unit=unit)

    assert result.segment_translations == ("Он сказал:", "Готово.")
    assert result.translated_text == "Он сказал: Готово."

