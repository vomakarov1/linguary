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
    validation_policy: str = "v1",
) -> tuple[BlockProcessingResult, ...]:
    """Validates the model response against application thresholds."""
    return make_validator(checks).validate(
        payload,
        unit=unit,
        target_language=TARGET_LANGUAGE,
        note_id_factory=note_ids(),
        truncated=truncated,
        validation_policy=validation_policy,
    )


def rejection(
    payload: TranslationResultResponse,
    *,
    unit: TranslationUnit,
    truncated: bool = False,
    checks: CheckSettings | None = None,
    validation_policy: str = "v1",
) -> ResultValidationError:
    """Rejection of the unit returned by validation."""
    with pytest.raises(ResultValidationError) as rejected:
        validate(
            payload,
            unit=unit,
            truncated=truncated,
            checks=checks,
            validation_policy=validation_policy,
        )
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
    """Dialogue newlines stay intact; the policy no longer joins ``\\n— `` inline."""
    from linguary.application.reader_text import ReaderTextPolicy

    policy = ReaderTextPolicy()
    single = "— Привет, — сказал он.\n— Как дела?"
    assert policy.prepare("translation", single) == single
    kept = "— Привет, — сказал он.\n\n— Как дела?"
    assert policy.prepare("translation", kept) == kept
    assert "\n" in policy.prepare("translation", kept)


def test_resolve_response_evidence_algorithmically_binds_when_content_evidence_is_empty() -> None:
    """When LLM returns empty content_evidence, notes and hints are bound algorithmically (AC-008)."""
    from linguary.application.llm_dto import (
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
        hints=(make_hint(original="Rare word"),),
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


def test_mixed_block_rejects_hints_targeting_only_translate_segment() -> None:
    """In a mixed TRANSLATE + GLOSSARY_ONLY block, hints must cover at least one GLOSSARY_ONLY span."""
    unit, block = _segmented(
        ("The entailed estate stood alone.", " The gloomy sky darkened."),
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
    )
    payload = _segmented_response(
        block,
        ((1, "Заповедное имение стояло особняком."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="the entailed estate"),),
    )

    message = str(rejection(payload, unit=unit))

    assert (
        f"для блока {block.id} ни одна словарная подсказка не относится к сегменту GLOSSARY_ONLY"
        in message
    )


def test_mixed_block_accepts_hint_matching_glossary_only_segment_including_inflection() -> None:
    """In a mixed block, hints matching a GLOSSARY_ONLY span (directly or by stem, e.g. to wink <-> winked) pass."""
    unit, block = _segmented(
        ("Clear sentence.", " He winked at her."),
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
    )
    payload = _segmented_response(
        block,
        ((1, "Ясное предложение."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="to wink", translation="подмигнуть"),),
    )

    (result,) = validate(payload, unit=unit)
    assert result.glossary_hints[0].original == "to wink"

    unit_hurry, block_hurry = _segmented(
        ("The entailed estate stood alone.", " She hurried away."),
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
    )
    payload_hurry = _segmented_response(
        block_hurry,
        ((1, "Заповедное имение стояло особняком."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(
            make_hint(original="entailed"),
            make_hint(original="to hurry", translation="спешить"),
        ),
    )

    (result_hurry,) = validate(payload_hurry, unit=unit_hurry)
    assert len(result_hurry.glossary_hints) == 2


def test_mixed_block_hint_matching_handles_irregular_forms_and_rejects_false_positives() -> None:
    from linguary.application.validators.treatment import _hint_matches_glossary_texts

    # Irregular/suppletive verb and noun forms match.
    assert _hint_matches_glossary_texts("to fling", ("He flung the door open.",))
    assert _hint_matches_glossary_texts("to catch", ("She caught the ball.",))
    assert _hint_matches_glossary_texts("to tear", ("He tore the letter.",))
    assert _hint_matches_glossary_texts("to die", ("The fire died down.",))
    assert _hint_matches_glossary_texts("knife", ("Sharp knives lay on the table.",))
    assert _hint_matches_glossary_texts("mouse", ("Two mice ran away.",))

    # No right-boundary-free substring match or derivational -ly confusion.
    assert not _hint_matches_glossary_texts("act", ("He actually knew the truth.",))
    assert not _hint_matches_glossary_texts("hard", ("She could hardly breathe.",))

    # Common high-frequency stopword ("took") in TRANSLATE hint must not match GLOSSARY_ONLY span.
    unit, block = _segmented(
        ("He took the money from the drawer.", " She took a breath."),
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
    )
    payload = _segmented_response(
        block,
        ((1, "Он взял деньги из ящика."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="took the money", translation="взял деньги"),),
    )
    message = str(rejection(payload, unit=unit))
    assert (
        f"для блока {block.id} ни одна словарная подсказка не относится к сегменту GLOSSARY_ONLY"
        in message
    )

    # Mixed block with irregular GLOSSARY_ONLY hint passes validation.
    unit_fling, block_fling = _segmented(
        ("Clear sentence.", " He flung the coat aside."),
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
    )
    payload_fling = _segmented_response(
        block_fling,
        ((1, "Ясное предложение."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="to fling", translation="швырнуть"),),
    )
    (result_fling,) = validate(payload_fling, unit=unit_fling)
    assert result_fling.glossary_hints[0].original == "to fling"


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


def test_note_language_leak_catches_unmarked_english_and_cyrillic_in_quotes() -> None:
    """Asymmetric NOTE leak check catches Cyrillic in EN quotes >= 2 words and unmarked EN words."""
    from linguary.application.validators.language import (
        note_language_leak_violation as leak_from_language,
    )
    from linguary.application.validators.mixed_script import note_language_leak_violation

    assert leak_from_language is note_language_leak_violation

    block = make_block(order=0, text="He was strongly reminded of Dudley.")

    # 1. Cyrillic inside EN quotes of >= 2 words (guillemets, double quotes, italics)
    for leaky_note in (
        "Оборот «was strongly reminded of Дадли» передаёт сильное сходство.",
        'Выражение "was strongly reminded of Дадли" показывает сходство.',
        "Фраза *was strongly reminded of Дадли* означает сильное напоминание.",
        "Фраза «went to Дадли» смешивает английский оборот с русским именем.",
    ):
        payload = make_response(
            (make_block_result(block),),
            language_notes=(make_note(block, note=leaky_note),),
        )
        assert note_language_leak_violation(payload, target_language="ru") is not None

    # Also caught inside glossary_hint.note
    hint_payload = make_response(
        (
            make_block_result(
                block,
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(
                    make_hint(
                        original="reminded of",
                        translation="напомнил о",
                        note="Фраза «was strongly reminded of Дадли» усиливает сравнение.",
                    ),
                ),
            ),
        )
    )
    assert note_language_leak_violation(hint_payload, target_language="ru") is not None

    # 2. Unmarked frequent EN leak words between Russian words
    for leak_word in (
        "very",
        "strong",
        "really",
        "just",
        "means",
        "shows",
        "because",
        "when",
        "here",
        "also",
        "only",
        "always",
        "never",
        "something",
        "someone",
    ):
        payload = make_response(
            (make_block_result(block),),
            language_notes=(make_note(block, note=f"Это слово {leak_word} усиливает оттенок."),),
        )
        assert note_language_leak_violation(payload, target_language="ru") is not None

    # 3. Clean notes: properly quoted EN, italics, whitelisted grammar/linguary terms, non-ru
    for clean_note in (
        "Оборот «was strongly reminded of Dudley» означает «сильно напомнил о Дадли».",
        "Слово «really» и наречие *very* здесь усиливают значение.",
        "Здесь используется Past Perfect для действия до другого события в Past Simple.",
        "Сравни Present Simple, Present Continuous, Past Continuous, Future Simple и Present Perfect.",
        "После Modal ставится Infinitive без частицы to; в UK и US это уровень A2–B1 (см. IPA).",
        "Окончания -ing, -ed и -s меняют форму глагола на уровнях A1, B2, C1 и C2.",
        "Конструкция «used to + инфинитив» описывает привычку в прошлом.",
        "Конструкция «to + инфинитив» выражает цель действия.",
    ):
        payload = make_response(
            (make_block_result(block),),
            language_notes=(make_note(block, note=clean_note),),
        )
        assert note_language_leak_violation(payload, target_language="ru") is None

    # Non-Russian target language is skipped
    non_ru_payload = make_response(
        (make_block_result(block),),
        language_notes=(make_note(block, note="Это слово really важно."),),
    )
    assert note_language_leak_violation(non_ru_payload, target_language="de") is None


def test_note_language_leak_scopes_notes_to_payload_blocks_for_per_block_validation() -> None:
    """Per-block calls `replace(payload, blocks=(block,))` only inspect notes belonging to that block."""
    from linguary.application.llm_dto import FragmentAssistanceResponse
    from linguary.application.validators.mixed_script import note_language_leak_violation

    b1, b2 = make_blocks(2)
    b1_result = make_block_result(b1)
    b2_result = make_block_result(b2)
    full_payload = TranslationResultResponse(
        blocks=(b1_result, b2_result),
        language_notes=(
            make_note(b1, note="Чистая русская заметка без просачиваний."),
            make_note(b2, note="Это слово really усиливает оттенок."),
        ),
        fragment_assistances=(
            FragmentAssistanceResponse(
                difficulty_id="diff-1",
                point_ids=("p1",),
                kind="russian_explanation",
                text="Это пояснение really просачивается.",
            ),
        ),
    )

    # Checking only b1 must not blame b1 for b2's leaky note or fragment_assistances
    single_b1 = replace(full_payload, blocks=(b1_result,))
    assert note_language_leak_violation(single_b1, target_language="ru") is None

    # Checking only b2 catches b2's leaky note and attributes it to b2
    single_b2 = replace(full_payload, blocks=(b2_result,))
    violation_b2 = note_language_leak_violation(single_b2, target_language="ru")
    assert violation_b2 is not None
    assert f"примечание блока {b2.id}" in violation_b2

    # Standalone call with empty blocks checks fragment_assistances without mislabeling difficulty_id as block ID
    assist_only = TranslationResultResponse(
        blocks=(),
        fragment_assistances=full_payload.fragment_assistances,
    )
    violation_assist = note_language_leak_violation(assist_only, target_language="ru")
    assert violation_assist is not None
    assert "помощь к трудности diff-1" in violation_assist
    assert "примечание блока diff-1" not in violation_assist


def test_quality_warnings_catch_dozen_preposition_initials_and_indeclinable_name() -> None:
    """Non-blocking quality warnings catch dozen/десяток, что-бы, со+sibilant+vowel, initials, indeclinable name, and glossary drift."""
    from linguary.application.validators.language import translation_quality_warnings
    from linguary.domain import ConceptType, GlossarySnapshotEntry, TermStatus

    glossary = (
        GlossarySnapshotEntry(
            source_text="Harry Potter",
            translation="Гарри Поттер",
            type=ConceptType.PERSON,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="Hogwarts",
            translation="Хогвартс",
            type=ConceptType.PLACE,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="Privet Drive",
            translation="Тисовая улица",
            type=ConceptType.PLACE,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="Diagon Alley",
            translation="Косой переулок",
            type=ConceptType.PLACE,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="You-Know-Who",
            translation="Сам-Знаешь-Кто",
            type=ConceptType.PERSON,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="Muggle",
            translation="магл",
            type=ConceptType.FICTIONAL_CONCEPT,
            status=TermStatus.APPROVED,
        ),
    )

    def warnings_for(source: str, translated: str) -> tuple[str, ...]:
        block = make_block(order=0, text=source)
        unit = make_unit((block,))
        payload = make_response((make_block_result(block, translated_text=translated),))
        return translation_quality_warnings(
            payload,
            unit=unit,
            target_language="ru",
            mandatory_glossary=glossary,
        )

    # 1. dozen <-> десяток
    assert any("dozen" in w or "десяток" in w for w in warnings_for("He bought two dozen owls.", "Он купил два десятка сов."))
    assert warnings_for("He bought two dozen owls.", "Он купил две дюжины сов.") == ()

    # 2. что-бы
    assert any("что-бы" in w for w in warnings_for("He came to help.", "Он пришёл, что-бы помочь."))
    assert warnings_for("He came to help.", "Он пришёл, чтобы помочь.") == ()

    # 3. со + [зсжш][гласная] (positive) vs valid consonant clusters (negative)
    for bad in ("Он шёл со золотой монетой.", "Она стояла со садом позади.", "Он говорил со женой.", "Мальчик бежал со шапкой в руках."):
        assert any("со " in w for w in warnings_for("He walked slowly.", bad)), bad
    for good in ("Он сделал это со зла.", "Он ответил со знанием дела.", "Он взял книгу со стола.", "Он пришёл со своей сестрой."):
        assert warnings_for("He walked slowly.", good) == (), good

    # 4. Calque initial Х. Поттер when canonical name starts with Г*
    assert any("Х. Поттер" in w for w in warnings_for("Mr H. Potter", "Мистер Х. Поттер"))
    assert any("Х. Поттер" in w for w in warnings_for("To Mr H. Potter", "Мистеру Х. Поттеру"))
    assert warnings_for("Mr H. Potter", "Мистер Г. Поттер") == ()

    # 5. Indeclinable Сам-Знаешь-Кто after prepositions
    for bad in (
        "Он боялся говорить о Сам-Знаешь-Кто.",
        "Они сбежали от Сам-Знаешь-Кто.",
        "Он пошёл к Сам-Знаешь-Кто.",
        "Она спорила с Сам-Знаешь-Кто.",
    ):
        assert any("Сам-Знаешь-Кто" in w for w in warnings_for("They feared him.", bad)), bad
    assert warnings_for("You-Know-Who arrived.", "Сам-Знаешь-Кто прибыл.") == ()
    assert warnings_for("They spoke of You-Know-Who.", "Они говорили о Сам-Знаешь-Ком.") == ()

    # 6. Mandatory glossary mismatch (when source_text is present in original) vs valid declensions
    assert any("Хогвартс" in w for w in warnings_for("He went to Hogwarts.", "Он поехал в Школу магии."))
    assert any("магл" in w for w in warnings_for("He was a Muggle.", "Он был магглом."))
    assert warnings_for("He was a Muggle.", "Он был маглом.") == ()
    assert warnings_for("He went to Hogwarts.", "Он поехал в Хогвартсе и Хогвартс.") == ()
    assert warnings_for("He lived on Privet Drive.", "Он жил на Тисовой улице.") == ()
    assert warnings_for("They walked down Diagon Alley.", "Они шли по Косому переулку.") == ()
    # Term not in original must not trigger warning
    assert warnings_for("He walked into the room.", "Он вошёл в комнату.") == ()


def test_result_validator_wires_dialogue_note_leak_glossary_warnings_and_spelling_checker() -> None:
    """TranslationResultValidator wires dialogue, NOTE leak, spelling checker, and QUALITY warnings (P2.T3)."""
    from linguary.adapters.config import AppSettings
    from linguary.adapters.validation.english_spelling import IpaDictSpellingChecker
    from linguary.application.llm_dto import FragmentAssistanceResponse
    from linguary.application.validators import TranslationResultValidator
    from linguary.composition.container import _validator as build_container_validator
    from linguary.domain import ConceptType, GlossarySnapshotEntry, TermStatus
    from linguary.web.labels import RESULT_WARNING_LABELS, result_warning_label

    # 1. Domain enum & web label coverage
    assert ResultWarningKind.QUALITY == "quality"
    assert set(RESULT_WARNING_LABELS) == set(ResultWarningKind)
    assert result_warning_label(ResultWarningKind.QUALITY).strip()

    spelling_checker = IpaDictSpellingChecker()
    validator = TranslationResultValidator(
        target_language_min_share=0.5,
        language_check_min_length=80,
        repetition_ngram_size=3,
        repetition_min_repeats=3,
        spelling_checker=spelling_checker,
    )

    # 2. Blocking dialogue_violation (hybrid dash + guillemets)
    dlg_block = make_block(order=0, text='"Hello," he said.')
    dlg_unit = make_unit((dlg_block,))
    dlg_payload = make_response(
        (make_block_result(dlg_block, translated_text="— «Привет», — сказал он."),)
    )
    with pytest.raises(ResultValidationError) as dlg_err:
        validator.validate(
            dlg_payload,
            unit=dlg_unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
        )
    assert dlg_err.value.violation is not None
    assert dlg_err.value.violation.rule_code == "dialogue_punctuation"

    # 3. Blocking note_language_leak_violation (block note and unit-level fragment_assistances)
    note_block = make_block(order=0, text="He was strongly reminded of Dudley.")
    note_unit = make_unit((note_block,))
    leaky_note_payload = make_response(
        (make_block_result(note_block, translated_text="Он сильно напомнил ему Дадли."),),
        language_notes=(make_note(note_block, note="Это слово really усиливает оттенок."),),
    )
    with pytest.raises(ResultValidationError) as note_err:
        validator.validate(
            leaky_note_payload,
            unit=note_unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
        )
    assert note_err.value.violation is not None
    assert note_err.value.violation.rule_code == "note_language_leak"

    leaky_assist_payload = TranslationResultResponse(
        blocks=(make_block_result(note_block, translated_text="Он сильно напомнил ему Дадли."),),
        fragment_assistances=(
            FragmentAssistanceResponse(
                difficulty_id="diff-1",
                point_ids=("p1",),
                kind="russian_explanation",
                text="Это пояснение really просачивается.",
            ),
        ),
    )
    with pytest.raises(ResultValidationError) as assist_err:
        validator.validate(
            leaky_assist_payload,
            unit=note_unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
        )
    assert assist_err.value.violation is not None
    assert assist_err.value.violation.rule_code == "note_language_leak"
    assert assist_err.value.violation.scope == "unit"

    # 4. Blocking spelling check on `to <word>` lemmas (excluding verbatim words in block original)
    expel_block = make_block(order=0, text="We could all have been killed — or worse, expelled.")
    expel_unit = make_unit((expel_block,), treatments=(BlockTreatment.GLOSSARY_ONLY,))
    bad_lemma_payload = make_response(
        (
            make_block_result(
                expel_block,
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(make_hint(original="to expell", translation="исключать"),),
            ),
        )
    )
    with pytest.raises(ResultValidationError) as spell_err:
        validator.validate(
            bad_lemma_payload,
            unit=expel_unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
        )
    assert spell_err.value.violation is not None
    assert spell_err.value.violation.rule_code == "glossary_hint_spelling"

    # Known EN-GB/EN-US lemma passes cleanly with no warnings
    good_lemma_payload = make_response(
        (
            make_block_result(
                expel_block,
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(
                    make_hint(original="to expel", translation="исключать"),
                    make_hint(original="to colour", translation="окрашивать"),
                ),
            ),
        )
    )
    good_results = validator.validate(
        good_lemma_payload,
        unit=expel_unit,
        target_language=TARGET_LANGUAGE,
        note_id_factory=note_ids(),
    )
    assert good_results[0].warnings == ()

    # Fictional word appearing verbatim in original text is excluded from both blocking and warnings
    verbatim_block = make_block(order=0, text="He loved to quidditch with a Muggle.")
    verbatim_unit = make_unit((verbatim_block,), treatments=(BlockTreatment.GLOSSARY_ONLY,))
    verbatim_payload = make_response(
        (
            make_block_result(
                verbatim_block,
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(
                    make_hint(original="to quidditch", translation="играть в квиддич"),
                    make_hint(original="Muggle", translation="магл"),
                ),
            ),
        )
    )
    verbatim_results = validator.validate(
        verbatim_payload,
        unit=verbatim_unit,
        target_language=TARGET_LANGUAGE,
        note_id_factory=note_ids(),
    )
    assert verbatim_results[0].warnings == ()

    # 5. Non-blocking unknown English word in glossary_hints + translation_quality_warnings -> QUALITY
    glossary = (
        GlossarySnapshotEntry(
            source_text="Hogwarts",
            translation="Хогвартс",
            type=ConceptType.PLACE,
            status=TermStatus.APPROVED,
        ),
    )
    warn_block = make_block(order=0, text="He bought two dozen mysterious owls at Hogwarts.")
    warn_unit = make_unit((warn_block,))
    warn_payload = make_response(
        (
            make_block_result(
                warn_block,
                translated_text="Он купил два десятка загадочных сов в Школе магии.",
                glossary_hints=(
                    make_hint(original="misterious", translation="загадочный"),
                ),
            ),
        )
    )
    warn_results = validator.validate(
        warn_payload,
        unit=warn_unit,
        target_language=TARGET_LANGUAGE,
        note_id_factory=note_ids(),
        mandatory_glossary=glossary,
    )
    quality_messages = [
        warning.message
        for warning in warn_results[0].warnings
        if warning.kind is ResultWarningKind.QUALITY
    ]
    assert len(quality_messages) >= 3
    assert any("десяток" in msg or "dozen" in msg for msg in quality_messages)
    assert any("Хогвартс" in msg for msg in quality_messages)
    assert any("misterious" in msg for msg in quality_messages)

    # 6. Composition container injects IpaDictSpellingChecker into _validator(...)
    container_validator = build_container_validator(AppSettings())
    with pytest.raises(ResultValidationError):
        container_validator.validate(
            bad_lemma_payload,
            unit=expel_unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
        )

    # 7. block_processing.TranslationCall and correct_chapter.CorrectChapter pass mandatory_glossary
    import asyncio
    from types import SimpleNamespace
    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.correction import CorrectionInputs
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        ChapterCorrectionResponse,
        CorrectedBlockResponse,
        CorrectionSnapshotV2,
    )
    from linguary.application.ports.llm import LLMResponse
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.use_cases.correct_chapter import CorrectChapter
    from linguary.domain import (
        BookMetadata,
        CanonicalDocument,
        ChapterAnalysis,
        DocumentSection,
        GlossarySnapshot,
    )

    glossary_snapshot = GlossarySnapshot(entries=glossary)
    section = DocumentSection(
        id=warn_block.section_id,
        document_id="book-1",
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=(warn_block,),
    )
    document = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )
    plan = TreatmentPlan.of_treatments(section.id, ((warn_block.id, BlockTreatment.TRANSLATE),))
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(warn_block.id,),
    )
    context = ContextBuilder().for_translation_unit(
        document=document,
        section=section,
        unit=warn_unit,
        analysis=analysis,
        glossary=glossary_snapshot,
    )
    call_payload = make_response(
        (
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Он поехал в Школу магии.",
            ),
        )
    )

    class _FakeLLM:
        async def generate(self, request):
            return LLMResponse(
                payload=call_payload,
                provider="fake",
                model="gpt-test",
                truncated=False,
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=None,
        llm_client=_FakeLLM(),
        retry_policy=RetryPolicy(max_attempts=1),
        validator=validator,
        sleep=None,
    )
    call_result = asyncio.run(
        call.request(
            context,
            unit=warn_unit,
            target_language=TARGET_LANGUAGE,
            note_ids=note_ids(),
            glossary=glossary_snapshot,
        )
    )
    assert any(
        w.kind is ResultWarningKind.QUALITY and "Хогвартс" in w.message
        for w in call_result.results[0].warnings
    )

    previous_result = BlockProcessingResult(
        block_id=warn_block.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Он поехал в Хогвартс.",
    )
    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 500,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    corr_inputs = CorrectionInputs(
        document=document,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=glossary_snapshot),
        conditions=SimpleNamespace(id="cond-1", payload={"schema_version": 1, "versions": {"validators": 1}}),
        snapshot=snapshot,
        blocks=(warn_block,),
        results={warn_block.id: previous_result},
        bindings={warn_block.id: None},
    )
    fake_tx = SimpleNamespace(work=SimpleNamespace(list_correction_candidates=lambda *a, **kw: ()))
    corrector = CorrectChapter(
        unit_of_work=None,
        llm_client=None,
        snapshot=snapshot,
        validator=validator,
    )
    corrected, = corrector._candidates(
        fake_tx,
        corr_inputs,
        ChapterCorrectionResponse(
            (
                CorrectedBlockResponse(
                    block_id=warn_block.id,
                    translated_text="Он поехал в Школу магии.",
                    glossary_hints=(),
                    language_notes=(),
                ),
            )
        ),
    )
    assert any(
        w.kind is ResultWarningKind.QUALITY and "Хогвартс" in w.message
        for w in corrected.warnings
    )


def test_truncated_true_precedes_coverage_and_structure_violations() -> None:
    """Provider truncation outranks a missing block, a wrong mode, and empty text."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    present = make_block_result(blocks[0])
    payloads = (
        make_response((present,)),
        make_response((
            present,
            make_block_result(
                blocks[1],
                treatment=BlockTreatment.READ_AS_IS,
                translated_text=None,
            ),
        )),
        make_response((
            present,
            with_fields(make_block_result(blocks[1]), translated_text=""),
        )),
    )
    for payload in payloads:
        rejected = rejection(payload, unit=unit, truncated=True)
        assert rejected.violation is not None
        assert rejected.violation.rule_code == "provider_truncated"


def test_translation_policy_v3_and_v2_legacy_coexist() -> None:
    """Reliability v4 is current (v3 validation); v3 (v2), legacy, and context v2 stay executable."""
    from linguary.application.translation_policy import (
        CURRENT_TRANSLATION_POLICY,
        LEGACY_TRANSLATION_POLICY,
        TRANSLATION_POLICIES,
        TRANSLATION_RELIABILITY_POLICY_V3,
        require_policy,
    )

    assert CURRENT_TRANSLATION_POLICY == "translation-reliability-v4"
    assert set(TRANSLATION_POLICIES) == {
        LEGACY_TRANSLATION_POLICY,
        "translation-context-v2",
        TRANSLATION_RELIABILITY_POLICY_V3,
        CURRENT_TRANSLATION_POLICY,
    }
    legacy = require_policy(LEGACY_TRANSLATION_POLICY)
    context_v2 = require_policy("translation-context-v2")
    reliability_v3 = require_policy(TRANSLATION_RELIABILITY_POLICY_V3)
    current = require_policy(CURRENT_TRANSLATION_POLICY)
    assert legacy.validation_policy == "v1"
    assert context_v2.validation_policy == "v1"
    assert reliability_v3.validation_policy == "v2"
    assert current.validation_policy == "v3"
    assert (legacy.version, context_v2.version, reliability_v3.version, current.version) == (
        "translation-context-legacy-v1",
        "translation-context-v2",
        "translation-reliability-v3",
        "translation-reliability-v4",
    )
    assert legacy.max_neighbour_chars is None
    assert legacy.capacity_retries_without_neighbours == 0
    assert context_v2.max_neighbour_chars == current.max_neighbour_chars == 8000
    assert context_v2.neighbours_per_side == current.neighbours_per_side == 2
    assert (
        context_v2.capacity_retries_without_neighbours
        == current.capacity_retries_without_neighbours
        == 1
    )


def test_candidate_correction_policy_helper_covers_v2_and_v3() -> None:
    """Candidate correction accepts chapter-correction v2 and v3, not legacy v1."""
    from linguary.domain.book_conditions import WorkStateError
    from linguary.domain.correction import (
        CORRECTION_POLICY_VERSION,
        CORRECTION_POLICY_VERSION_V2,
        CORRECTION_POLICY_VERSION_V3,
        correction_policy,
        is_candidate_correction_policy,
    )

    assert CORRECTION_POLICY_VERSION_V3 == "chapter-correction-v3"
    assert is_candidate_correction_policy(CORRECTION_POLICY_VERSION_V2) is True
    assert is_candidate_correction_policy("chapter-correction-v3") is True
    assert is_candidate_correction_policy(CORRECTION_POLICY_VERSION) is False
    assert is_candidate_correction_policy(None) is False
    assert is_candidate_correction_policy("") is False
    for version in (
        CORRECTION_POLICY_VERSION,
        CORRECTION_POLICY_VERSION_V2,
        CORRECTION_POLICY_VERSION_V3,
    ):
        accepted = correction_policy({"correction": {"required": True, "policy_version": version}})
        assert accepted is not None
        assert accepted["policy_version"] == version
    with pytest.raises(WorkStateError):
        correction_policy({"correction": {"required": True, "policy_version": "chapter-correction-v9"}})


_THREE_LINE_QUOTE = (
    "The first line of the quote.\n"
    "The second line of the quote.\n"
    "The third line of the quote."
)
_THREE_LINE_QUOTE_FULL = (
    "Первая строка цитаты.\nВторая строка цитаты.\nТретья строка цитаты."
)
_THREE_LINE_QUOTE_FIRST = "Первая строка цитаты."


def test_strict_paragraphs_rejects_first_line_only_three_line_quote() -> None:
    """A three-line quote translated as only its first line is rejected under v2."""
    block = make_block(order=0, text=_THREE_LINE_QUOTE, type=BlockType.QUOTE)
    unit = make_unit((block,))
    dropped = make_response(
        (make_block_result(block, translated_text=_THREE_LINE_QUOTE_FIRST),)
    )

    error = rejection(dropped, unit=unit, validation_policy="v2")

    assert error.violation is not None
    assert error.violation.rule_code == "paragraph_count_mismatch"
    assert error.violation.block_ids == (block.id,)
    message = str(error)
    assert f"перевод блока {block.id} содержит 1 абзацев, а оригинал — 3" in message
    assert "непустых строк (3)" in message
    assert "не пропускай строки" in message
    kept = make_response(
        (make_block_result(block, translated_text=_THREE_LINE_QUOTE_FULL),)
    )
    assert (
        validate(kept, unit=unit, validation_policy="v2")[0].translated_text
        == _THREE_LINE_QUOTE_FULL
    )


def test_strict_paragraphs_preserves_dialogue_split_and_partial_segments() -> None:
    """v2 still allows a dash split and checks only the TRANSLATE slices."""
    source = '"Hello," he said. "How are you?"'
    translation = "— Привет, — сказал он.\n— Как дела?"
    block = make_block(order=0, text=source)
    unit = make_unit((block,))
    split = make_response((make_block_result(block, translated_text=translation),))
    assert validate(split, unit=unit, validation_policy="v2")[0].translated_text == translation

    unit_seg, block_seg = _segmented(
        (
            "Leave this sentence alone.",
            " Translate the first line.\nTranslate the second line.",
        ),
        (BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
    )
    kept_segment = "Переведи первую строку.\nПереведи вторую строку."
    kept = _segmented_response(
        block_seg,
        ((2, kept_segment),),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert validate(kept, unit=unit_seg, validation_policy="v2")[0].segment_translations == (
        None,
        kept_segment,
    )

    dropped = _segmented_response(
        block_seg,
        ((2, "Переведи первую строку."),),
        treatment=BlockTreatment.TRANSLATE,
    )
    message = str(rejection(dropped, unit=unit_seg, validation_policy="v2"))
    assert "сегмента 2" in message
    assert "содержит 1 абзацев, а оригинал — 2" in message
    assert "не пропускай строки" in message
    assert (
        validate(dropped, unit=unit_seg, validation_policy="v1")[0].segment_translations[1]
        == "Переведи первую строку."
    )


def test_strict_poem_rejects_blank_line_substitution() -> None:
    """"L1\\nL2\\nL3" -> "С1\\n\\nС3" is rejected when validation_policy is v2."""
    source = "L1\nL2\nL3"
    substituted = "С1\n\nС3"
    block = make_block(order=0, text=source, type=BlockType.POEM)
    unit = make_unit((block,))
    payload = make_response((make_block_result(block, translated_text=substituted),))

    message = str(rejection(payload, unit=unit, validation_policy="v2"))

    assert "непустых строк" in message
    assert "содержит 2 непустых строк, а оригинал — 3" in message
    assert validate(payload, unit=unit, validation_policy="v1")[0].translated_text == substituted

    faithful = "С1\nС2\nС3"
    kept = make_response((make_block_result(block, translated_text=faithful),))
    assert validate(kept, unit=unit, validation_policy="v2")[0].translated_text == faithful

    stanza_source = "L1\n\nL3"
    stanza_translation = "С1\n\nС3"
    stanza = make_block(order=1, text=stanza_source, type=BlockType.POEM)
    stanza_unit = make_unit((stanza,))
    stanza_payload = make_response(
        (make_block_result(stanza, translated_text=stanza_translation),)
    )
    assert (
        validate(stanza_payload, unit=stanza_unit, validation_policy="v2")[0].translated_text
        == stanza_translation
    )


def test_legacy_v1_policy_allows_historical_line_count_decrease() -> None:
    """Historical v1 still accepts a translation that drops non-empty lines."""
    block = make_block(order=0, text=_THREE_LINE_QUOTE, type=BlockType.QUOTE)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text=_THREE_LINE_QUOTE_FIRST),)
    )

    assert validate(payload, unit=unit)[0].translated_text == _THREE_LINE_QUOTE_FIRST
    assert (
        validate(payload, unit=unit, validation_policy="v1")[0].translated_text
        == _THREE_LINE_QUOTE_FIRST
    )


def test_truncation_ignores_trailing_footnote_markers() -> None:
    """A trailing footnote marker does not hide terminal punctuation ("He left.[^1]")."""
    from linguary.application.validators.truncation import _ends_with_punctuation

    assert _ends_with_punctuation("He left.[^1]") is True
    assert _ends_with_punctuation("He left.[^1][^2]") is True
    assert _ends_with_punctuation("Он ушёл.[^1]") is True
    assert _ends_with_punctuation("Он ушёл[^1]") is False

    source = "See note 1. He left."
    marker_at = source.index("1")
    block = replace(
        make_block(order=0, text=source),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=marker_at,
                end=marker_at + 1,
                target="section-1.xhtml#fn1",
            ),
        ),
    )
    unit = make_unit((block,))
    translation = "Он ушёл.[^1]"
    payload = make_response((make_block_result(block, translated_text=translation),))
    assert validate(payload, unit=unit)[0].translated_text == translation

    unfinished = make_response((make_block_result(block, translated_text="Он ушёл[^1]"),))
    assert "терминальной пунктуацией" in str(rejection(unfinished, unit=unit))


def test_validation_policy_v3_preserves_all_v2_checks_and_rejects_unknown_policy() -> None:
    """validation_policy='v3' enforces strict_line_preservation, strict_fields, and fragment_assistances; unknown policy raises ValueError."""
    from linguary.application.correction_dto import unchanged_reader_texts
    from linguary.application.llm_dto import FragmentAssistanceResponse

    # 1. strict_line_preservation
    quote_block = make_block(order=0, text=_THREE_LINE_QUOTE, type=BlockType.QUOTE)
    quote_unit = make_unit((quote_block,))
    dropped = make_response(
        (make_block_result(quote_block, translated_text=_THREE_LINE_QUOTE_FIRST),)
    )
    line_err = rejection(dropped, unit=quote_unit, validation_policy="v3")
    assert line_err.violation is not None
    assert line_err.violation.rule_code == "paragraph_count_mismatch"

    # 2. strict_fields (changed segment with service message leak)
    part1 = "First line of English prose."
    part2 = " Second line of English prose."
    seg_block = replace(
        make_block(order=0, text=part1 + part2),
        segments=(
            BlockSegment(0, len(part1), 0),
            BlockSegment(len(part1), len(part1) + len(part2), 1),
        ),
    )
    seg_plan = TreatmentPlan(
        section_id=seg_block.section_id,
        items=(
            TreatmentPlanItem(
                block_id=seg_block.id,
                treatment=BlockTreatment.TRANSLATE,
                segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
                sentence_indexes=(0, 1),
            ),
        ),
    )
    seg_unit = TranslationUnit.of_blocks(id="u-v3-seg", blocks=(seg_block,), plan=seg_plan)
    old_seg = "As an AI language model, я перевёл первую строку."
    previous = BlockProcessingResult(
        block_id=seg_block.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=f"{old_seg} Старый перевод второй строки.",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        segment_translations=(old_seg, "Старый перевод второй строки."),
    )
    excluded = {seg_block.id: unchanged_reader_texts(previous)}
    dirty_seg = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                seg_block.id,
                BlockTreatment.TRANSLATE,
                segments=(
                    SegmentTranslationResponse(1, old_seg),
                    SegmentTranslationResponse(2, "Извините, но я не могу выполнить этот запрос."),
                ),
            ),
        ),
    )
    with pytest.raises(ResultValidationError) as seg_err:
        make_validator().validate(
            dirty_seg,
            unit=seg_unit,
            target_language=TARGET_LANGUAGE,
            note_id_factory=note_ids(),
            validation_policy="v3",
            excluded_service_texts=excluded,
        )
    assert seg_err.value.violation is not None
    assert seg_err.value.violation.rule_code == "service_message_leak"

    # 3. fragment_assistances service message check
    read_block = make_block(order=0, text="A short line.")
    read_plan = TreatmentPlan.of_treatments(
        read_block.section_id, ((read_block.id, BlockTreatment.READ_AS_IS),)
    )
    read_unit = TranslationUnit.of_blocks(id="u-v3-read", blocks=(read_block,), plan=read_plan)
    assist_payload = TranslationResultResponse(
        blocks=(BlockResultResponse(read_block.id, BlockTreatment.READ_AS_IS),),
        fragment_assistances=(
            FragmentAssistanceResponse(
                "d1",
                ("p1",),
                "russian_explanation",
                "Извините, но я не могу помочь с этим фрагментом.",
            ),
        ),
    )
    assist_err = rejection(assist_payload, unit=read_unit, validation_policy="v3")
    assert assist_err.violation is not None
    assert assist_err.violation.rule_code == "service_message_leak"

    # 4. Fail closed on unknown policy
    clean_payload = make_response(
        (make_block_result(quote_block, translated_text=_THREE_LINE_QUOTE_FULL),)
    )
    with pytest.raises(ValueError):
        validate(clean_payload, unit=quote_unit, validation_policy="v99")


def test_strict_v3_rejects_bare_glossary_hints_on_translate_and_checks_mixed_phrasal_verbs() -> None:
    """Under validation_policy='v3', bare glossary_hints without note are forbidden on TRANSLATE and non-GLOSSARY_ONLY segments."""
    from linguary.application.validators.treatment import (
        _hint_matches_glossary_texts,
        content_violation,
    )

    # 1. Pure whole-block TRANSLATE: bare hint rejected under v3, accepted under v2, hint with note accepted under v3.
    block = make_block(order=0, text="The entailed estate stood alone.")
    unit = make_unit((block,))
    for bare_note in (None, "", "   "):
        bare_payload = make_response(
            (
                make_block_result(
                    block,
                    glossary_hints=(make_hint(original="entailed", note=bare_note),),
                ),
            )
        )
        assert validate(bare_payload, unit=unit, validation_policy="v2")[0].glossary_hints[0].original == "entailed"
        err = rejection(bare_payload, unit=unit, validation_policy="v3")
        assert err.violation is not None
        assert err.violation.rule_code == "content_violation"
        assert content_violation(bare_payload, unit=unit, strict_v3=True) is not None
        assert content_violation(bare_payload, unit=unit, strict_v3=False) is None

    noted_payload = make_response(
        (
            make_block_result(
                block,
                glossary_hints=(
                    make_hint(
                        original="entailed",
                        note="Юридический термин: передаётся по майорату.",
                    ),
                ),
            ),
        )
    )
    assert validate(noted_payload, unit=unit, validation_policy="v3")[0].glossary_hints[0].note

    # 2. Pure segmented (TRANSLATE, TRANSLATE) and mixed (TRANSLATE, READ_AS_IS): bare hints rejected under v3.
    for modes in (
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
    ):
        seg_unit, seg_block = _segmented(
            ("The entailed estate stood alone.", " A clear sentence."),
            modes,
        )
        seg_translations = (
            ((1, "Заповедное имение стояло особняком."), (2, "Ясное предложение."))
            if modes == (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE)
            else ((1, "Заповедное имение стояло особняком."),)
        )
        seg_bare = _segmented_response(
            seg_block,
            seg_translations,
            treatment=BlockTreatment.TRANSLATE,
            hints=(make_hint(original="entailed", note=None),),
        )
        assert validate(seg_bare, unit=seg_unit, validation_policy="v2")[0].glossary_hints[0].original == "entailed"
        seg_err = rejection(seg_bare, unit=seg_unit, validation_policy="v3")
        assert seg_err.violation is not None
        assert seg_err.violation.rule_code == "content_violation"

        seg_noted = _segmented_response(
            seg_block,
            seg_translations,
            treatment=BlockTreatment.TRANSLATE,
            hints=(
                make_hint(
                    original="entailed",
                    note="Юридический термин о праве наследования.",
                ),
            ),
        )
        assert validate(seg_noted, unit=seg_unit, validation_policy="v3")[0].glossary_hints[0].note

    # 3. Light-verb phrasal verbs via _infinitive_end in _hint_matches_glossary_texts:
    #    "to go off" <-> "It went off." (all words in _HINT_STOPWORDS!) and "to make out" <-> "He made it out."
    assert _hint_matches_glossary_texts("to go off", ("It went off.",)) is True
    assert _hint_matches_glossary_texts("to make out", ("He made it out in the fog.",)) is True
    assert _hint_matches_glossary_texts("to go off", ("She went home.",)) is False

    # 4. Mixed (TRANSLATE, GLOSSARY_ONLY): bare "to go off" matching "went off" in GLOSSARY_ONLY passes under v3,
    #    while an extra bare hint from the TRANSLATE segment is rejected under v3.
    mixed_unit, mixed_block = _segmented(
        ("The entailed estate stood alone.", " It went off."),
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
    )
    mixed_phrasal_ok = _segmented_response(
        mixed_block,
        ((1, "Заповедное имение стояло особняком."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="to go off", translation="взорваться / сработать"),),
    )
    assert validate(mixed_phrasal_ok, unit=mixed_unit, validation_policy="v3")[0].glossary_hints[0].original == "to go off"

    mixed_extra_bare = _segmented_response(
        mixed_block,
        ((1, "Заповедное имение стояло особняком."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(
            make_hint(original="to go off", translation="взорваться / сработать"),
            make_hint(original="entailed", note=None),
        ),
    )
    assert len(validate(mixed_extra_bare, unit=mixed_unit, validation_policy="v2")[0].glossary_hints) == 2
    mixed_err = rejection(mixed_extra_bare, unit=mixed_unit, validation_policy="v3")
    assert mixed_err.violation is not None
    assert mixed_err.violation.rule_code == "content_violation"

    # Extra hint from TRANSLATE WITH non-empty note alongside bare GLOSSARY_ONLY hint passes under v3.
    mixed_extra_noted = _segmented_response(
        mixed_block,
        ((1, "Заповедное имение стояло особняком."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(
            make_hint(original="to go off", translation="взорваться / сработать"),
            make_hint(original="entailed", note="Ограниченное право наследования."),
        ),
    )
    assert len(validate(mixed_extra_noted, unit=mixed_unit, validation_policy="v3")[0].glossary_hints) == 2

    # 5. Mixed (GLOSSARY_ONLY, READ_AS_IS): bare hint from READ_AS_IS is rejected under v3.
    gr_unit, gr_block = _segmented(
        ("It went off.", " She smiled."),
        (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS),
    )
    gr_bad = _segmented_response(
        gr_block,
        (),
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(
            make_hint(original="to go off", translation="сработать"),
            make_hint(original="smiled", translation="улыбнулась", note=None),
        ),
    )
    gr_err = rejection(gr_bad, unit=gr_unit, validation_policy="v3")
    assert gr_err.violation is not None
    assert gr_err.violation.rule_code == "content_violation"


def test_resolve_response_evidence_binds_collocation_with_source_texts() -> None:
    """resolve_response_evidence(..., source_texts=...) binds `to wrinkle one’s nose` to anchor `wrinkled`."""
    from linguary.application.llm_dto import TranslationResultResponse
    from linguary.application.validators.content_plan import (
        resolve_response_evidence,
        validate_content_coverage,
    )
    from linguary.domain import GlossaryHint
    from linguary.domain.content_plan import (
        ContentPlan,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )

    source_text = "Aunt Petunia wrinkled her nose at the smell."
    start = source_text.index("wrinkled")
    end = start + len("wrinkled")
    plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                "d-wrinkle",
                (SourceAnchor("b1", start, end, "wrinkled"),),
                "b1",
                "vocabulary",
                "useful",
                "lexical_hint",
                (RequiredPoint("p1", "Дать контекстную подсказку для collocation."),),
            ),
        ),
    )
    res = BlockProcessingResult(
        "b1",
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("to wrinkle one’s nose", "сморщить нос"),),
    )
    payload = TranslationResultResponse(content_evidence=())

    # Without source_texts, single-verb anchor "wrinkled" cannot verify "one's nose" in block text.
    unbound = resolve_response_evidence(plan, payload, (res,), (), owner_block_ids=("b1",))
    assert unbound.evidence == ()

    # With source_texts, _hint_quote_matches_anchor matches verb + _infinitive_end(_ONE_SLOT) in source_block_text.
    bound = resolve_response_evidence(
        plan,
        payload,
        (res,),
        (),
        owner_block_ids=("b1",),
        source_texts={"b1": source_text},
    )
    assert len(bound.evidence) == 1
    assert bound.evidence[0].difficulty_id == "d-wrinkle"
    assert bound.evidence[0].artifact_kind == "glossary_hint.translation"
    assert bound.evidence[0].body_excerpt == "сморщить нос"

    checked = validate_content_coverage(
        plan,
        (res,),
        (),
        bound,
        owner_block_ids=("b1",),
        source_texts={"b1": source_text},
        validators_version=4,
    )
    assert len(checked) == 1

    # End-to-end wiring: TranslationCall (block_processing.py) and CorrectChapter._candidates
    # thread source_texts into resolve_response_evidence.
    import asyncio
    from types import SimpleNamespace

    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.correction import CorrectionInputs
    from linguary.application.correction_dto import (
        CORRECTION_VERSIONS_V2,
        ChapterCorrectionResponse,
        CorrectedBlockResponse,
        CorrectionSnapshotV2,
    )
    from linguary.application.ports.llm import LLMResponse
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.use_cases.correct_chapter import CorrectChapter
    from linguary.domain import (
        BookMetadata,
        CanonicalDocument,
        ChapterAnalysis,
        DocumentSection,
        GlossarySnapshot,
    )

    doc_block = make_block(order=0, text=source_text)
    doc_section = DocumentSection(
        id=doc_block.section_id,
        document_id="book-colloc",
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=(doc_block,),
    )
    doc = CanonicalDocument(
        id="book-colloc",
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(doc_section,),
    )
    colloc_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                "d-wrinkle",
                (SourceAnchor(doc_block.id, start, end, "wrinkled"),),
                doc_block.id,
                "vocabulary",
                "useful",
                "lexical_hint",
                (RequiredPoint("p1", "Дать контекстную подсказку для collocation."),),
            ),
        ),
    )
    t_plan = TreatmentPlan.of_treatments(
        doc_section.id, ((doc_block.id, BlockTreatment.GLOSSARY_ONLY),)
    )
    doc_analysis = ChapterAnalysis.create(
        id="analysis-colloc",
        section_id=doc_section.id,
        treatment_plan=t_plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(doc_block.id,),
        content_plan=colloc_plan,
    )
    doc_unit = TranslationUnit.of_blocks(id="unit-colloc", blocks=(doc_block,), plan=t_plan)
    call_ctx = ContextBuilder(content_plan_profile="content-plan-v3").for_translation_unit(
        document=doc,
        section=doc_section,
        unit=doc_unit,
        analysis=doc_analysis,
        glossary=GlossarySnapshot(entries=()),
    )
    llm_payload = make_response(
        (
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.GLOSSARY_ONLY,
                translated_text=None,
                glossary_hints=(
                    make_hint(original="to wrinkle one’s nose", translation="сморщить нос"),
                ),
            ),
        )
    )

    class _CollocLLM:
        async def generate(self, request):
            return LLMResponse(
                payload=llm_payload,
                provider="fake",
                model="gpt-test",
                truncated=False,
                call_context=request.call_context,
            )

    call = TranslationCall(
        unit_of_work=None,
        llm_client=_CollocLLM(),
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
        sleep=None,
    )
    call_res = asyncio.run(
        call.request(
            call_ctx,
            unit=doc_unit,
            target_language=TARGET_LANGUAGE,
            note_ids=note_ids(),
            glossary=GlossarySnapshot(entries=()),
        )
    )
    assert len(call_res.results[0].content_evidence.evidence) == 1
    assert call_res.results[0].content_evidence.evidence[0].body_excerpt == "сморщить нос"

    snapshot = CorrectionSnapshotV2(
        model="gpt-test",
        route={"provider": "openai_compatible", "base_url": "https://example.test/v1", "proxy": "", "route_id": "r1"},
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 500,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Propose.",
        review_prompt="Review.",
        versions=CORRECTION_VERSIONS_V2,
    )
    prev_res = BlockProcessingResult(
        block_id=doc_block.id,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("to wrinkle one’s nose", "наморщить нос"),),
    )
    corr_inputs = CorrectionInputs(
        document=doc,
        section=doc_section,
        analysis=doc_analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot(entries=())),
        conditions=SimpleNamespace(
            id="cond-1",
            payload={"schema_version": 1, "versions": {"validators": 3}, "reader_level": "b2"},
        ),
        snapshot=snapshot,
        blocks=(doc_block,),
        results={doc_block.id: prev_res},
        bindings={doc_block.id: None},
    )
    fake_tx = SimpleNamespace(work=SimpleNamespace(list_correction_candidates=lambda *a, **kw: ()))
    corrector = CorrectChapter(
        unit_of_work=None,
        llm_client=None,
        snapshot=snapshot,
        validator=make_validator(),
    )
    corrected, = corrector._candidates(
        fake_tx,
        corr_inputs,
        ChapterCorrectionResponse(
            (
                CorrectedBlockResponse(
                    block_id=doc_block.id,
                    translated_text=None,
                    glossary_hints=(
                        make_hint(original="to wrinkle one’s nose", translation="сморщить нос"),
                    ),
                    language_notes=(),
                ),
            )
        ),
    )
    assert corrected.content_evidence is not None
    assert len(corrected.content_evidence.evidence) == 1
    assert corrected.content_evidence.evidence[0].body_excerpt == "сморщить нос"




def test_truncation_allows_british_dialect_apostrophes_and_unclosed_quotes() -> None:
    """British dialect elisions (`killin’ ’em`, `An’ ’em`, `-in’`, `o’`, etc.) preserve open-quote parity."""
    from linguary.application.validators.truncation import _unbalanced, truncation_diagnostic

    unclosed_dialect = "‘An’ he was killin’ ’em, an’ I couldn’ stop ’im."
    open_translation = "«Ну вот, он их убивал, а я не мог его остановить."
    assert _unbalanced(open_translation, source=unclosed_dialect) is None

    block_open = make_block(order=0, text=unclosed_dialect)
    unit_open = make_unit((block_open,))
    payload_open = make_response(
        (make_block_result(block_open, translated_text=open_translation),)
    )
    assert validate(payload_open, unit=unit_open)[0].translated_text == open_translation
    assert truncation_diagnostic(payload_open, unit=unit_open, truncated=False) is None

    many_dialect = (
        "‘An’ ’em, o’ course, don’ know las’ night ol’ Hagrid jus’ mus’ be tellin’ ’em "
        "by the boys’ gate."
    )
    assert _unbalanced("«Да и они, конечно, не знают…", source=many_dialect) is None

    closed_dialect = "‘An’ he was killin’ ’em.’"
    assert _unbalanced("«Ну вот, он их убивал.", source=closed_dialect) == "кавычки"
    block_closed = make_block(order=0, text=closed_dialect)
    unit_closed = make_unit((block_closed,))
    broken_closed = make_response(
        (make_block_result(block_closed, translated_text="«Ну вот, он их убивал."),)
    )
    err = rejection(broken_closed, unit=unit_closed)
    assert err.violation is not None
    assert err.violation.rule_code == "quote_balance"


def test_validation_policy_v3_enforces_strict_dialogue_checks_preserving_v2() -> None:
    """validation_policy='v3' passes normalized_payload and strict_v3=True to dialogue_violation without breaking v2."""
    # 1. Segmented [T, G, T] inside same unclosed quote after author attribution
    parts_tgt = ("‘First,’ he said, ‘second part.", " Easy middle.", " Third part.’")
    unit_tgt, block_tgt = _segmented(
        parts_tgt,
        (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
    )
    bad_tgt = _segmented_response(
        block_tgt,
        ((1, "— Первая, — сказал он, вторая часть."), (3, "— Третья часть.")),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="Easy", translation="лёгкая"),),
    )
    assert validate(bad_tgt, unit=unit_tgt, validation_policy="v2")
    err_tgt = rejection(bad_tgt, unit=unit_tgt, validation_policy="v3")
    assert err_tgt.violation is not None
    assert err_tgt.violation.rule_code == "dialogue_punctuation"

    good_tgt = _segmented_response(
        block_tgt,
        ((1, "— Первая, — сказал он, вторая часть."), (3, "Третья часть.")),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="Easy", translation="лёгкая"),),
    )
    assert validate(good_tgt, unit=unit_tgt, validation_policy="v3")

    # 2. Non-leading inline dash when source has direct speech quotes
    inline_block = make_block(order=0, text='Harry walked in. "Hello! How are you?"')
    inline_unit = make_unit((inline_block,))
    inline_bad = make_response(
        (make_block_result(inline_block, translated_text="Гарри вошёл. — Привет! Как дела?"),)
    )
    assert validate(inline_bad, unit=inline_unit, validation_policy="v2")
    err_inline = rejection(inline_bad, unit=inline_unit, validation_policy="v3")
    assert err_inline.violation is not None
    assert err_inline.violation.rule_code == "dialogue_punctuation"

    # 3. Unclosed British dialect monologue passes under v3
    dialect_block = make_block(
        order=0, text="‘An’ he was killin’ ’em, an’ I couldn’ stop ’im."
    )
    dialect_unit = make_unit((dialect_block,))
    dialect_payload = make_response(
        (
            make_block_result(
                dialect_block,
                translated_text="«Ну вот, он их убивал, а я не мог его остановить.",
            ),
        )
    )
    assert validate(dialect_payload, unit=dialect_unit, validation_policy="v3")


def test_suggested_glossary_warnings_calque_person_restriction_and_v3_union_wiring() -> None:
    """P2.T5B: v3 checks union (is_user_decision | always_requires_review), differentiates suggested copy, and limits single_surnames to PERSON."""
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace
    from uuid import uuid4

    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.ports.llm import LLMResponse
    from linguary.application.translation_policy import (
        CURRENT_TRANSLATION_POLICY,
        TRANSLATION_RELIABILITY_POLICY_V3,
    )
    from linguary.application.validators.language import translation_quality_warnings
    from linguary.domain import (
        BookMetadata,
        CanonicalDocument,
        ChapterAnalysis,
        ConceptType,
        DocumentSection,
        GlossarySnapshot,
        GlossarySnapshotEntry,
        LLMCallStage,
        TermStatus,
    )
    from linguary.domain.llm_log import CallContext

    entries = (
        GlossarySnapshotEntry(
            source_text="Harry",
            translation="Гарри",
            type=ConceptType.PERSON,
            status=TermStatus.SUGGESTED,
        ),
        GlossarySnapshotEntry(
            source_text="Potter",
            translation="Поттер",
            type=ConceptType.PERSON,
            status=TermStatus.SUGGESTED,
        ),
        GlossarySnapshotEntry(
            source_text="Hogwarts",
            translation="Хогвартс",
            type=ConceptType.PLACE,
            status=TermStatus.SUGGESTED,
        ),
        GlossarySnapshotEntry(
            source_text="London",
            translation="Лондон",
            type=ConceptType.PLACE,
            status=TermStatus.SUGGESTED,
        ),
        GlossarySnapshotEntry(
            source_text="Wand",
            translation="палочка",
            type=ConceptType.TERM,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="Owl",
            translation="сова",
            type=ConceptType.TERM,
            status=TermStatus.SUGGESTED,
        ),
    )

    # 1. _glossary_mismatch_warnings differentiates mandatory vs suggested, and _calque_initial_warnings
    #    restricts single_surnames to ConceptType.PERSON.
    block = make_block(
        order=0,
        text="Mr H. Potter took his Wand and Owl from H. London to Hogwarts.",
    )
    unit = make_unit((block,))
    payload = make_response(
        (
            make_block_result(
                block,
                translated_text="Мистер Х. Поттер взял свой жезл и филина из Х. Лондона в Школу магии.",
            ),
        )
    )
    warnings = translation_quality_warnings(
        payload,
        unit=unit,
        target_language="ru",
        mandatory_glossary=entries,
    )
    assert any(
        "расхождение с обязательным глоссарием" in w and "'Wand'" in w
        for w in warnings
    )
    assert any(
        "расхождение с предложенным глоссарием" in w and "'Hogwarts'" in w
        for w in warnings
    )
    assert not any(
        "расхождение с обязательным глоссарием" in w and "'Hogwarts'" in w
        for w in warnings
    )
    assert any("Х. Поттер" in w for w in warnings)
    assert not any("Х. Лондон" in w for w in warnings)

    # 2. block_processing.TranslationCall passes union under v3 and only mandatory_entries under v2.
    snapshot = GlossarySnapshot(entries=entries)
    section = DocumentSection(
        id=block.section_id,
        document_id="book-v3",
        title="Chapter 1",
        order=0,
        source_resource_path="ch1.xhtml",
        blocks=(block,),
    )
    document = CanonicalDocument(
        id="book-v3",
        metadata=BookMetadata(title="Book", authors=("Author",)),
        source_language="en",
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )
    plan = TreatmentPlan.of_treatments(section.id, ((block.id, BlockTreatment.TRANSLATE),))
    analysis = ChapterAnalysis.create(
        id="analysis-v3",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(block.id,),
    )
    context = ContextBuilder().for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=snapshot,
    )
    call_payload = make_response(
        (
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Мистер Х. Поттер взял свой жезл и филина в Школу магии.",
            ),
        )
    )

    def run_with_policy(policy_version: str) -> tuple[str, ...]:
        cond_id = str(uuid4())
        root_id = str(uuid4())
        rec_attempt_id = str(uuid4())
        conditions = SimpleNamespace(
            id=cond_id,
            payload={"models": {"processing": "gpt-test"}, "versions": {"validators": 3}},
        )
        root = SimpleNamespace(
            id=root_id,
            conditions_id=cond_id,
            policy_version=policy_version,
            generation=1,
        )
        rec_ctx = CallContext(
            conditions_id=cond_id,
            stage=LLMCallStage.TRANSLATION_UNIT.value,
            scope_id="scope-1",
            ordered_block_ids=(block.id,),
            attempt_id=rec_attempt_id,
            generation=0,
            assigned_model="gpt-test",
            prompt_hash="a" * 64,
            root_id=root_id,
            policy_version=policy_version,
        )
        fake_tx = SimpleNamespace(
            runs=SimpleNamespace(active_for_book=lambda _bid: None),
            work=SimpleNamespace(
                get_binding=lambda _kind, _key: None,
                get_conditions=lambda cid: conditions if cid == cond_id else None,
                get_root=lambda rid: root if rid == root_id else None,
                begin_recovery_attempt=lambda *a, **kw: rec_ctx,
                owns_recovery_attempt=lambda aid: aid == rec_attempt_id,
            ),
        )
        fake_uow = SimpleNamespace(
            transaction=lambda: nullcontext(fake_tx),
            _diagnostics_service=SimpleNamespace(
                ensure_available=lambda: None,
                fail=lambda: None,
            ),
        )

        class _FakeLLM:
            async def generate(self, request):
                return LLMResponse(
                    payload=call_payload,
                    provider="fake",
                    model="gpt-test",
                    truncated=False,
                    call_context=request.call_context,
                )

        call = TranslationCall(
            unit_of_work=fake_uow,  # type: ignore[arg-type]
            llm_client=_FakeLLM(),  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_attempts=1),
            validator=make_validator(),
            sleep=lambda _: None,  # type: ignore[arg-type]
        )
        res = asyncio.run(
            call.request(
                context,
                unit=unit,
                target_language=TARGET_LANGUAGE,
                note_ids=note_ids(),
                recovery=(root_id, "scope-1"),
                glossary=snapshot,
            )
        )
        return tuple(w.message for w in res.results[0].warnings)

    v3_warnings = run_with_policy(CURRENT_TRANSLATION_POLICY)
    assert any("расхождение с обязательным глоссарием" in m and "'Wand'" in m for m in v3_warnings)
    assert any("расхождение с предложенным глоссарием" in m and "'Hogwarts'" in m for m in v3_warnings)
    assert any("Х. Поттер" in m for m in v3_warnings)
    assert not any("'Owl'" in m for m in v3_warnings)

    v2_warnings = run_with_policy(TRANSLATION_RELIABILITY_POLICY_V3)
    assert any("расхождение с обязательным глоссарием" in m and "'Wand'" in m for m in v2_warnings)
    assert not any("'Hogwarts'" in m for m in v2_warnings)
    assert not any("Х. Поттер" in m for m in v2_warnings)


def test_truncation_supports_british_and_american_multi_segment_direct_speech() -> None:
    """WP-1.1: British/American multi-segment quotes pass when balanced, reject when left unclosed."""
    # 1. b18 pattern: mid-segment British opening after `: `, continued across segments, closed at end.
    unit_b18, block_b18 = _segmented(
        (
            "He stood up.",
            " He looked around.",
            " He cleared his throat.",
            " Then he said: ‘First sentence.",
            " Middle sentence stays inside the speech.",
            " Last sentence closes it.’",
        ),
        (
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
            BlockTreatment.TRANSLATE,
            BlockTreatment.TRANSLATE,
        ),
    )
    ok_b18 = _segmented_response(
        block_b18,
        (
            (4, "Затем он сказал: «Первое предложение."),
            (5, "Среднее предложение остаётся внутри речи."),
            (6, "Последнее предложение её закрывает»."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert validate(ok_b18, unit=unit_b18)

    # 2. b25 pattern: multiple British speech turns spanning segments.
    from linguary.application.validators.truncation import truncation_diagnostic

    unit_b25, block_b25 = _segmented(
        (
            "He cried: ‘Wait a minute. Do you hear that?",
            " Something is moving outside.’",
            " She whispered: ‘Stay back, it is close.",
            " Do not open the door.’",
        ),
        (
            BlockTreatment.TRANSLATE,
            BlockTreatment.TRANSLATE,
            BlockTreatment.TRANSLATE,
            BlockTreatment.TRANSLATE,
        ),
    )
    ok_b25 = _segmented_response(
        block_b25,
        (
            (1, "Он крикнул: «Подожди минутку. Ты слышишь это?"),
            (2, "Снаружи что-то движется»."),
            (3, "Она прошептала: «Отойди назад, оно совсем близко."),
            (4, "Не открывай дверь»."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert truncation_diagnostic(ok_b25, unit=unit_b25, truncated=False) is None
    assert validate(ok_b25, unit=unit_b25)

    # 2b. Mixed segmented block where mid-line British `‘` opens in TRANSLATE and closes in READ_AS_IS.
    unit_mixed_brit, block_mixed_brit = _segmented(
        ("Then he said: ‘First sentence.", " Easy closing.’"),
        (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
    )
    ok_mixed_brit = _segmented_response(
        block_mixed_brit,
        ((1, "Затем он сказал: «Первое предложение."),),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert truncation_diagnostic(ok_mixed_brit, unit=unit_mixed_brit, truncated=False) is None

    # 3. American straight quotes across two TRANSLATE segments.
    unit_us, block_us = _segmented(
        ('He whispered: "Listen carefully.', ' We leave at dawn."'),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    ok_us = _segmented_response(
        block_us,
        (
            (1, "Он прошептал: «Слушай внимательно."),
            (2, "Мы уходим на рассвете»."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert validate(ok_us, unit=unit_us)

    # 4. Unclosed guillemet across TRANSLATE segments when source quotes are closed -> quote_balance.
    for unit_closed, block_closed in ((unit_us, block_us),):
        unclosed = _segmented_response(
            block_closed,
            (
                (1, "Он прошептал: «Слушай внимательно."),
                (2, "Мы уходим на рассвете."),
            ),
            treatment=BlockTreatment.TRANSLATE,
        )
        err = rejection(unclosed, unit=unit_closed)
        assert err.violation is not None
        assert err.violation.rule_code == "quote_balance"

    unit_brit_closed, block_brit_closed = _segmented(
        ("He said: ‘Listen carefully.", " We leave at dawn.’"),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    unclosed_brit = _segmented_response(
        block_brit_closed,
        (
            (1, "Он сказал: «Слушай внимательно."),
            (2, "Мы уходим на рассвете."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )
    err_brit = rejection(unclosed_brit, unit=unit_brit_closed)
    assert err_brit.violation is not None
    assert err_brit.violation.rule_code == "quote_balance"


def test_note_language_leak_allows_verbatim_multiword_source_quotes_only() -> None:
    """WP-1.2: Verbatim >=2-word source quotes pass; single leak words and foreign phrases still fail."""
    from linguary.application.llm_dto import FragmentAssistanceResponse
    from linguary.application.validators.mixed_script import note_language_leak_violation

    block = make_block(
        order=0,
        text="How very wrong he was, and it was a very, very strange day; he didn’t know.",
    )
    unit = make_unit((block,))
    source_texts = {block.id: block.text}

    # 1. Verbatim multi-word quote from original_excerpt / block source passes.
    clean_payload = make_response(
        (
            make_block_result(
                block,
                translated_text="Как же сильно он ошибался, и это был очень странный день; он не знал.",
            ),
        ),
        language_notes=(
            make_note(
                block,
                original_excerpt="How very wrong he was.",
                note="Выражение How very wrong подчёркивает, насколько сильно он ошибался.",
            ),
            make_note(
                block,
                original_excerpt="very, very strange",
                note="Повтор very, very strange усиливает странность происходящего.",
            ),
            make_note(
                block,
                original_excerpt="he didn’t know",
                note="Оборот he didn't know передаёт незнание персонажа.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            clean_payload, target_language="ru", source_texts=source_texts
        )
        is None
    )
    assert validate(clean_payload, unit=unit)

    # 2. Fragment assistances (empty blocks branch) also receive source_texts.
    assist_ok = TranslationResultResponse(
        blocks=(),
        fragment_assistances=(
            FragmentAssistanceResponse(
                difficulty_id="diff-1",
                point_ids=("p1",),
                kind="russian_explanation",
                text="Фраза How very wrong здесь усиливает удивление.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            assist_ok, target_language="ru", source_texts=source_texts
        )
        is None
    )
    assist_leak = TranslationResultResponse(
        blocks=(),
        fragment_assistances=(
            FragmentAssistanceResponse(
                difficulty_id="diff-1",
                point_ids=("p1",),
                kind="russian_explanation",
                text="Это very важное замечание.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            assist_leak, target_language="ru", source_texts=source_texts
        )
        is not None
    )

    # 3. Single unmarked leak word between Cyrillic words is still rejected even if present in source.
    single_word_payload = make_response(
        (make_block_result(block),),
        language_notes=(
            make_note(
                block,
                original_excerpt="How very wrong he was.",
                note="Это very важный оборот в данном предложении.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            single_word_payload, target_language="ru", source_texts=source_texts
        )
        is not None
    )
    err_single = rejection(single_word_payload, unit=unit)
    assert err_single.violation is not None
    assert err_single.violation.rule_code == "note_language_leak"

    # 4. Multi-word English phrase absent from source/original_excerpt is still rejected.
    absent_phrase_payload = make_response(
        (make_block_result(block),),
        language_notes=(
            make_note(
                block,
                original_excerpt="How very wrong he was.",
                note="Он был very strong man в этой сцене.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            absent_phrase_payload, target_language="ru", source_texts=source_texts
        )
        is not None
    )

    # 5. Word boundary check: `very wrong` must not match inside `every wrongful`.
    boundary_block = make_block(order=0, text="Every wrongful deed was remembered.")
    boundary_payload = make_response(
        (make_block_result(boundary_block),),
        language_notes=(
            make_note(
                boundary_block,
                original_excerpt="Every wrongful deed",
                note="Слова very wrong здесь не относятся к тексту.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            boundary_payload,
            target_language="ru",
            source_texts={boundary_block.id: boundary_block.text},
        )
        is not None
    )

    # 6. Cyrillic inside English quotes is still strictly rejected even with source_texts.
    cyr_in_quote_payload = make_response(
        (make_block_result(block),),
        language_notes=(
            make_note(
                block,
                original_excerpt="How very wrong he was.",
                note="Фраза «How very wrong он был» смешивает два языка.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            cyr_in_quote_payload, target_language="ru", source_texts=source_texts
        )
        is not None
    )


def test_validate_note_sources_translated_excerpt_error_gives_repair_guidance() -> None:
    """WP-2: Missing translated_excerpt error keeps strict check and guides verbatim copy / ellipsis."""
    from linguary.application.validators.content_plan import validate_note_sources
    from linguary.domain import BlockProcessingResult, LanguageNote

    result = BlockProcessingResult(
        block_id="b1",
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Он долго смотрел на дорогу и наконец повернул домой.",
        language_notes=(
            LanguageNote(
                id="ln-1",
                block_id="b1",
                original_excerpt="looked at the road",
                note="Здесь описывается внимательный взгляд.",
                translated_excerpt="поглядел на тропу",
            ),
        ),
    )
    with pytest.raises(ResultValidationError) as exc_info:
        validate_note_sources(
            (result,),
            {"b1": "He looked at the road and finally turned home."},
            check_translated_excerpt=True,
        )
    assert exc_info.value.violation is not None
    assert exc_info.value.violation.rule_code == "note_translated_excerpt_missing"
    msg = str(exc_info.value)
    assert "translated_excerpt" in msg
    assert "translated_text" in msg
    assert "..." in msg


def test_truncation_rejects_midline_unclosed_british_quote_in_translation_and_allows_quoted_read_as_is() -> None:
    """Review B3 & B4: reject unclosed mid-line ‘ in translation; allow [T, R(quote), T] runs."""
    from linguary.application.validators.truncation import _unbalanced, truncation_violation

    # B3: mid-line unclosed British opening quote introduced in translation MUST be rejected.
    assert _unbalanced("Он сказал: ‘Привет.", source="He said, ‘Hello.’") == "кавычки"

    # B4: two TRANSLATE slices each opening `"` separated by a quote-bearing READ_AS_IS slice.
    unit, block = _segmented(
        ('He said: "First.', ' Yes!"', ' She added: "Second.'),
        (
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
    )
    payload = _segmented_response(
        block,
        (
            (1, "Он сказал: «Первое."),
            (3, "Она добавила: «Второе."),
        ),
        treatment=BlockTreatment.TRANSLATE,
    )
    assert truncation_violation(payload, unit=unit, truncated=False) is None


def test_note_language_leak_apostrophe_boundary_prevents_partial_contraction_match() -> None:
    """Review L3: `s very` must not match inside `he's very` across an apostrophe."""
    from linguary.application.validators.mixed_script import note_language_leak_violation

    block = make_block(order=0, text="He's very bold indeed.")
    payload = make_response(
        (make_block_result(block),),
        language_notes=(
            make_note(
                block,
                original_excerpt="He's very bold",
                note="Здесь фрагмент s very не является целой фразой.",
            ),
        ),
    )
    assert (
        note_language_leak_violation(
            payload, target_language="ru", source_texts={block.id: block.text}
        )
        is not None
    )


def test_single_segment_block_with_segments_gives_actionable_violation() -> None:
    """WP2.3: returning segments for a non-segmented block tells the model to move translation to translated_text."""
    from linguary.application.validators.treatment import content_violation

    block = make_block(order=0, text="Single paragraph.")
    unit = make_unit((block,))
    payload = make_response(
        (
            BlockResultResponse(
                source_block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                glossary_hints=(),
                segments=(SegmentTranslationResponse(1, "Один абзац."),),
            ),
        )
    )
    violation = content_violation(payload, unit=unit)
    assert violation == (
        f"для блока {block.id} без нескольких сегментов "
        "список segments недопустим: перенеси полный перевод в translated_text "
        "и верни segments: []"
    )


def test_mixed_block_glossary_only_violations_name_exact_segment_indices() -> None:
    """WP2: segmented GLOSSARY_ONLY validator messages include exact 1-based segment indices."""
    unit, block = _segmented(
        ("First sentence.", " Second sentence.", " Third sentence."),
        (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
    )

    # 1. Missing hints altogether -> names segment 3.
    missing_hints = _segmented_response(
        block,
        ((1, "Первое предложение."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(),
    )
    msg_missing = str(rejection(missing_hints, unit=unit))
    assert f"для блока {block.id} обязательны словарные подсказки" in msg_missing
    assert "GLOSSARY_ONLY: 3" in msg_missing

    # 2. Hint only from TRANSLATE segment -> names segment 3.
    wrong_seg = _segmented_response(
        block,
        ((1, "Первое предложение."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(make_hint(original="First"),),
    )
    msg_wrong = str(rejection(wrong_seg, unit=unit))
    assert f"для блока {block.id} ни одна словарная подсказка не относится к сегменту GLOSSARY_ONLY" in msg_wrong
    assert "(3)" in msg_wrong

    # 3. strict_v3 extra bare hint from TRANSLATE segment alongside valid GLOSSARY_ONLY hint -> names segment 3.
    extra_bare = _segmented_response(
        block,
        ((1, "Первое предложение."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(
            make_hint(original="Third", translation="третий", note=""),
            make_hint(original="First", translation="первый", note=""),
        ),
    )
    msg_extra = str(rejection(extra_bare, unit=unit, validation_policy="v3"))
    assert f"для блока {block.id} словарная подсказка «First» без note не относится к сегменту GLOSSARY_ONLY" in msg_extra
    assert "(3)" in msg_extra

    # 4. Multiple GLOSSARY_ONLY segments -> lists all indices (2, 4).
    unit_multi, block_multi = _segmented(
        ("One.", " Two.", " Three.", " Four."),
        (
            BlockTreatment.TRANSLATE,
            BlockTreatment.GLOSSARY_ONLY,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.GLOSSARY_ONLY,
        ),
    )
    missing_multi = _segmented_response(
        block_multi,
        ((1, "Один."),),
        treatment=BlockTreatment.TRANSLATE,
        hints=(),
    )
    assert "GLOSSARY_ONLY: 2, 4" in str(rejection(missing_multi, unit=unit_multi))


def test_result_validator_v3_allows_embedded_quote_in_narrative_and_rejects_line_initial() -> None:
    """WP4: v3 result validation accepts single- and two-line embedded narrative quotes while rejecting line-initial guillemets."""
    b35_single_source = (
        "“We were on a mountain road…” Only this wasn’t a mountain road, "
        "and the trees weren’t even gray. “A dead forest,” Dad had said, pointing. “It happens.”"
    )
    single_block = make_block(order=0, text=b35_single_source)
    single_unit = make_unit((single_block,))

    for valid_single in (
        "— Мы ехали по горной дороге… — сказала она. Но это была не горная дорога, "
        "и деревья даже не были серыми. «Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»",
        "«Мы ехали по горной дороге…» Но это была не горная дорога, "
        "и деревья даже не были серыми. «Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»",
    ):
        payload = make_response(
            (make_block_result(single_block, translated_text=valid_single),)
        )
        results = validate(payload, unit=single_unit, validation_policy="v3")
        assert results[0].translated_text == valid_single

    b35_two_line_source = (
        "“We were on a mountain road…”\n"
        "Only this wasn’t a mountain road, and the trees weren’t even gray. "
        "“A dead forest,” Dad had said, pointing. “It happens.”"
    )
    two_line_block = make_block(order=0, text=b35_two_line_source)
    two_line_unit = make_unit((two_line_block,))
    valid_two_line = (
        "— Мы ехали по горной дороге…\n"
        "Но это была не горная дорога, и деревья даже не были серыми. "
        "«Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»"
    )
    two_line_results = validate(
        make_response((make_block_result(two_line_block, translated_text=valid_two_line),)),
        unit=two_line_unit,
        validation_policy="v3",
    )
    assert two_line_results[0].translated_text == valid_two_line

    # Line-initial spoken dialogue inside `«…»` is still rejected under v3.
    bad_initial = make_response(
        (make_block_result(single_block, translated_text="«Привет, — сказал он.»"),)
    )
    err_initial = rejection(bad_initial, unit=single_unit, validation_policy="v3")
    assert err_initial.violation is not None
    assert err_initial.violation.rule_code == "dialogue_punctuation"

    # Non-embedded inline dash after period is still rejected under v3.
    bad_dash = make_response(
        (
            make_block_result(
                single_block,
                translated_text="Но это была не горная дорога. — Так бывает.",
            ),
        )
    )
    err_dash = rejection(bad_dash, unit=single_unit, validation_policy="v3")
    assert err_dash.violation is not None
    assert err_dash.violation.rule_code == "dialogue_punctuation"


def test_post_speech_narrative_paragraph_allows_1_line_and_3_line_and_rejects_absorbed_neighbor() -> None:
    """WP4: post-speech authorial sentences pass in 1-line dash or 3-line layout; absorbed neighbors fail."""
    from linguary.application.validators.paragraphs import (
        _count_post_speech_narrative_sentences,
        paragraph_violation,
    )

    ollivander_source = (
        "‘Well, now – Mr Potter. Let me see.’ "
        "He pulled a long tape measure with silver markings out of his pocket. "
        "‘Which is your wand arm?’"
    )
    assert _count_post_speech_narrative_sentences(ollivander_source) == 1
    assert _count_post_speech_narrative_sentences("‘Yes.’ ‘No.’") == 0
    assert _count_post_speech_narrative_sentences('‘He said, “Wait.” Next.’') == 0
    assert _count_post_speech_narrative_sentences('"Hello," he said. "How are you?"') == 0
    assert _count_post_speech_narrative_sentences("‘Hello,’ said Mr. Ollivander. ‘Wait.’") == 0
    assert _count_post_speech_narrative_sentences("‘Hello,’ he said. Then he ran.") == 1
    assert _count_post_speech_narrative_sentences("‘Hello!’ he cried. Then he ran.") == 1
    assert _count_post_speech_narrative_sentences("‘Hello,’ Harry said.") == 0
    assert _count_post_speech_narrative_sentences("‘Hello,’ Harry said. Then he ran.") == 1

    one_line = (
        "— Что ж, мистер Поттер. Посмотрим. "
        "— Он вынул из кармана длинную рулетку с серебряными делениями. "
        "— Какой рукой вы держите палочку?"
    )
    three_line = (
        "— Что ж, мистер Поттер. Посмотрим.\n"
        "Он вынул из кармана длинную рулетку с серебряными делениями.\n"
        "— Какой рукой вы держите палочку?"
    )
    absorbed_neighbor = (
        "— Что ж, мистер Поттер. Посмотрим. "
        "— Он вынул из кармана длинную рулетку с серебряными делениями. "
        "— Какой рукой вы держите палочку?\n"
        "Гарри вытянул правую руку."
    )
    lowercase_extra = (
        "— Что ж, мистер Поттер. Посмотрим.\n"
        "он вынул из кармана рулетку.\n"
        "— Какой рукой вы держите палочку?"
    )
    too_many_non_dash = (
        "— Что ж, мистер Поттер. Посмотрим.\n"
        "Он вынул из кармана рулетку.\n"
        "Гарри замер."
    )

    block = make_block(order=0, text=ollivander_source)
    unit = make_unit((block,))
    assert paragraph_violation("b1", source_text=ollivander_source, translated_text=one_line) is None
    assert paragraph_violation("b1", source_text=ollivander_source, translated_text=three_line) is None
    assert paragraph_violation(
        "b1",
        source_text="‘Hello,’ he said. Then he ran.",
        translated_text="— Привет, — сказал он.\nПотом он убежал.",
    ) is None
    with pytest.raises(ResultValidationError, match="самостоятельное авторское предложение"):
        validate(make_response((make_block_result(block, translated_text=one_line),)), unit=unit, validation_policy="v3")
    assert validate(make_response((make_block_result(block, translated_text=three_line),)), unit=unit, validation_policy="v3")[0].translated_text == three_line

    assert paragraph_violation("b1", source_text=ollivander_source, translated_text=absorbed_neighbor) is not None
    assert paragraph_violation("b1", source_text=ollivander_source, translated_text=lowercase_extra) is not None
    assert paragraph_violation("b1", source_text=ollivander_source, translated_text=too_many_non_dash) is not None
    assert paragraph_violation("b1", source_text="‘Yes.’ ‘No.’", translated_text="— Да.\nАвторский текст.\n— Нет.") is not None

    # Mid-line lowercase attribution + resumed speech on line 1 must NOT block valid 3-line layout.
    mid_attr_source = (
        "‘Ah, yes,’ said Mr Ollivander. ‘I remember every wand I’ve ever sold, Mr Potter.’ "
        "He pulled a long tape measure from his pocket. "
        "‘Which is your wand arm?’"
    )
    mid_attr_three_line = (
        "— А, да, — сказал мистер Олливандер. — Я помню каждую палочку, которую когда-либо продал, мистер Поттер.\n"
        "Он вытащил из кармана длинную измерительную ленту.\n"
        "— Какой рукой вы держите палочку?"
    )
    mid_attr_absorbed = (
        "— А, да, — сказал мистер Олливандер. — Я помню каждую палочку. "
        "— Он вытащил из кармана длинную измерительную ленту. "
        "— Какой рукой вы держите палочку?\n"
        "Гарри вытянул правую руку."
    )
    assert paragraph_violation("b1", source_text=mid_attr_source, translated_text=mid_attr_three_line) is None
    assert paragraph_violation("b1", source_text=mid_attr_source, translated_text=mid_attr_absorbed) is not None


