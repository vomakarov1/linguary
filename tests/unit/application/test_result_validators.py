"""Блокирующие проверки и предупреждения результата обработки (P3.T6, ALG-04, PRD § 15)."""

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
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    InlineAnnotation,
    InlineAnnotationType,
    ResultWarningKind,
    TranslationUnit,
)

#: Полный перевод длиннее порога проверки языка, оставшийся на языке оригинала.
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
    """Проверка ответа модели с порогами приложения."""
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
    """Отбраковка группы, которую вернула проверка."""
    with pytest.raises(ResultValidationError) as rejected:
        validate(payload, unit=unit, truncated=truncated, checks=checks)
    return rejected.value


def with_fields(block: BlockResultResponse, **fields: object) -> BlockResultResponse:
    """Ответ модели, который прошёл бы схему адаптера, но нарушает правило состава.

    DTO проверяет состав блока относительно названного им же режима, поэтому такой ответ
    нельзя собрать конструктором. Проверка состава относительно плана — последний рубеж
    ALG-04, и проверять его надо именно на ответе, который правила DTO обошёл.
    """
    for name, value in fields.items():
        object.__setattr__(block, name, value)
    return block


def test_valid_result_is_accepted() -> None:
    """Корректный ответ модели превращается в результаты блоков группы."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)

    results = validate(translated_response(unit), unit=unit)

    assert tuple(result.block_id for result in results) == unit.block_ids
    assert all(result.translated_text == TRANSLATED_TEXT for result in results)


def test_result_takes_treatment_from_plan() -> None:
    """Режим сохранённого результата берётся из плана, а не из ответа модели."""
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
    """Заметки ответа привязываются к своим блокам и получают идентификаторы сценария."""
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
    """Пропущенный блок отменяет сохранение всей группы (PRD § 17)."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    payload = make_response((make_block_result(blocks[0]),))

    assert "нет результата" in str(rejection(payload, unit=unit))


def test_unknown_block_discards_the_whole_unit() -> None:
    """Результат для блока вне группы отбраковывает ответ целиком."""
    blocks = make_blocks(2)
    unit = make_unit(blocks[:1])
    payload = make_response(
        (make_block_result(blocks[0]), make_block_result(blocks[1]))
    )

    assert "вне группы" in str(rejection(payload, unit=unit))


def test_repeated_block_is_rejected() -> None:
    """Дважды возвращённый блок — не порядок, а повтор: сообщение называет его прямо."""
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
    """Порядок блоков ответа обязан совпадать с порядком блоков группы."""
    blocks = make_blocks(2)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[1]), make_block_result(blocks[0]))
    )

    assert "порядок блоков в ответе" in str(rejection(payload, unit=unit))


def test_note_for_block_outside_the_unit_is_rejected() -> None:
    """Заметка о чужом блоке — тот же неизвестный идентификатор, только в другом поле."""
    blocks = make_blocks(2)
    unit = make_unit(blocks[:1])
    payload = make_response(
        (make_block_result(blocks[0]),), language_notes=(make_note(blocks[1]),)
    )

    assert "заметки привязаны к блокам вне группы" in str(rejection(payload, unit=unit))


def test_treatment_mismatch_discards_the_whole_unit() -> None:
    """Режим из ответа модели план не переопределяет (PRD § 14)."""
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.GLOSSARY_ONLY,))
    payload = make_response((make_block_result(blocks[0]),))

    assert "не совпадает с планом" in str(rejection(payload, unit=unit))


def test_translate_without_translation_is_rejected() -> None:
    """Блок `TRANSLATE` без полного перевода не сохраняется (PRD § 15)."""
    blocks = make_blocks(1)
    unit = make_unit(blocks)
    payload = make_response(
        (with_fields(make_block_result(blocks[0]), translated_text=None),)
    )

    assert "нет полного перевода" in str(rejection(payload, unit=unit))


def test_read_as_is_with_glossary_hints_is_rejected() -> None:
    """Блок `READ_AS_IS` не получает ни перевода, ни словарных подсказок (PRD § 10.1)."""
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
    """Блок `GLOSSARY_ONLY` без словарных подсказок не сохраняется."""
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
    """Ответ, обрезанный пределом вывода, отбраковывается даже при полном составе."""
    blocks = make_blocks(1)
    unit = make_unit(blocks)

    rejected = rejection(translated_response(unit), unit=unit, truncated=True)

    assert "по лимиту выходных токенов" in str(rejected)


def test_translation_without_terminal_punctuation_is_rejected() -> None:
    """Перевод законченного предложения обязан заканчиваться знаком конца предложения."""
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
    """Перевод заголовка не обязан заканчиваться точкой: её нет и в оригинале."""
    blocks = (make_block(order=0, text="Chapter One"),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text="Глава первая"),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Глава первая"


def test_unbalanced_quote_is_rejected() -> None:
    """Незакрытая кавычка в переводе — признак оборванной генерации."""
    blocks = (make_block(order=0, text='He said: "Go there."'),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text="Он сказал: «Иди туда."),)
    )

    assert "не закрыты кавычки" in str(rejection(payload, unit=unit))


def test_quote_left_open_in_the_source_is_accepted() -> None:
    """Многоабзацная прямая речь открывает кавычку в каждом абзаце — это не обрыв."""
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
    """Полный перевод на языке оригинала не сохраняется (PRD § 15)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=ENGLISH_TRANSLATION),)
    )

    assert "не на языке ru" in str(rejection(payload, unit=unit))


def test_short_translation_is_not_checked_for_language() -> None:
    """Короткая строка проверку языка не проходит: имя собственное — законный перевод."""
    blocks = (make_block(order=0, text="Mr. Bennet."),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text="Mr. Bennet."),)
    )

    assert validate(payload, unit=unit)[0].translated_text == "Mr. Bennet."


def test_glossary_hints_are_not_checked_for_language() -> None:
    """Словарная подсказка законно состоит из слова исходного языка (PRD § 15)."""
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
    """Отказ модели вместо перевода в книгу не попадает."""
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
    """Явный повтор сохраняется предупреждением и обработку не останавливает."""
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
    """Обычный перевод предупреждений не получает."""
    blocks = make_blocks(1)
    unit = make_unit(blocks)

    assert validate(translated_response(unit), unit=unit)[0].warnings == ()


def test_rejection_carries_expected_blocks_and_treatments() -> None:
    """Отбраковка несёт всё, из чего строится уточняющая инструкция повтора (ALG-05)."""
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
    """Сценарий, адресовавший блоки псевдонимами, перечисляет в повторе их же."""
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
    """Минимальная длина проверки языка — настройка, а не константа проверки."""
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
    """Приняла ли проверка ответ модели."""
    try:
        validate(payload, unit=unit, checks=checks)
    except ResultValidationError:
        return False
    return True


def test_results_keep_the_order_of_the_unit() -> None:
    """Результаты возвращаются в порядке блоков группы: его ждёт сохранение."""
    blocks = make_blocks(3)
    unit = make_unit(blocks)

    results: Sequence[BlockProcessingResult] = validate(
        translated_response(unit), unit=unit
    )

    assert [result.block_id for result in results] == list(unit.block_ids)


def test_footnote_markers_stay_only_in_translation_and_only_known_ones() -> None:
    """В переводе остаются метки существующих ссылок на сноски, в помощи меток нет (QA-08)."""
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
    """Слово из смеси латиницы и кириллицы отбраковывает группу (Н-1)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=translated_text),)
    )

    error = rejection(payload, unit=unit)

    assert "смеси латиницы и кириллицы" in str(error)
    assert error.expectation is not None


def test_mixed_word_in_language_note_is_rejected() -> None:
    """Заметка пишется на целевом языке и проверяется так же, как перевод."""
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
    """Целиком латинские слова и дефисные соединения — не нарушение (Н-1)."""
    blocks = (make_block(order=0, text=SOURCE_TEXT),)
    unit = make_unit(blocks)
    payload = make_response(
        (make_block_result(blocks[0], translated_text=translated_text),)
    )

    results = validate(payload, unit=unit)

    assert results[0].translated_text is not None
