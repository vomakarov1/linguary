"""Блокирующие проверки результата обработки и повторы вызова (P3.T6, ALG-04, ALG-05).

Каждая проверка отбраковывает всю группу целиком: в хранилище не остаётся ни переводов,
ни словарных подсказок, ни заметок этого ответа (PRD § 17). Ответы задаются текстом
структурированного ответа, поэтому негодный ответ модели воспроизводится так же, как
он пришёл бы от настоящего провайдера.
"""

from __future__ import annotations

import pytest
from processing_scenario import (
    BOOK_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    block_reply,
    make_process_chapter,
    note_reply,
    store_processing_book,
    translated_reply,
    translation_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.application.ports.llm import LLMTransportError
from linguary.application.retry_policy import ResultValidationError
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.domain import BlockTreatment

#: Английский текст длиннее порога проверки языка: перевод на целевой язык не сделан.
ENGLISH_TRANSLATION = (
    "It is a truth universally acknowledged that a single man in possession of "
    "a good fortune must be in want of a wife, whatever his own feelings may be."
)

#: Перевод, оборванный на полуслове: терминальной пунктуации нет, хотя в оригинале она есть.
UNFINISHED_TRANSLATION = (
    "Общепризнанная истина гласит, что мужчина, располагающий средствами, непременно"
)

#: Отказ модели вместо перевода: формально корректный ответ, который в книгу не годится.
SERVICE_MESSAGE = (
    "Извините, но я не могу выполнить эту просьбу: как языковая модель я не перевожу "
    "художественные тексты целиком."
)


def command() -> ProcessChapterCommand:
    """Команда обработки единственной главы книги."""
    return ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1")


def assert_nothing_saved(unit_of_work: SqlAlchemyUnitOfWork, run_id: str) -> None:
    """Отбракованная группа не оставляет в хранилище ничего (PRD § 17)."""
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(run_id) == ()


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        pytest.param(
            translation_reply([block_reply("b1")]),
            "нет результата для блоков",
            id="missing-block",
        ),
        pytest.param(
            translation_reply(
                [block_reply("b1"), block_reply("b2"), block_reply("b3")]
            ),
            "вне запроса",
            id="unknown-block",
        ),
        pytest.param(
            translation_reply([block_reply("b2"), block_reply("b1")]),
            "порядок блоков",
            id="wrong-order",
        ),
        pytest.param(
            translation_reply(
                [
                    block_reply("b1", translated_text=ENGLISH_TRANSLATION),
                    block_reply("b2"),
                ]
            ),
            "не на языке ru",
            id="wrong-language",
        ),
        pytest.param(
            translation_reply(
                [
                    block_reply("b1", translated_text=UNFINISHED_TRANSLATION),
                    block_reply("b2"),
                ]
            ),
            "терминальной пунктуацией",
            id="unfinished-text",
        ),
        pytest.param(
            translation_reply(
                [block_reply("b1", translated_text=SERVICE_MESSAGE), block_reply("b2")]
            ),
            "служебное сообщение",
            id="service-message",
        ),
        pytest.param(
            translated_reply(truncated=True),
            "лимиту выходных токенов",
            id="truncated-by-provider",
        ),
        pytest.param(
            translated_reply(notes=(note_reply("b3"),)),
            "вне запроса",
            id="note-of-foreign-block",
        ),
    ],
)
async def test_blocking_check_discards_the_whole_unit(
    unit_of_work: SqlAlchemyUnitOfWork, reply, reason: str
) -> None:
    """Любая блокирующая проверка отменяет сохранение всей группы (PRD § 15)."""
    scenario = store_processing_book(unit_of_work)
    process, _ = make_process_chapter(
        unit_of_work, FakeLLMClient.always(reply), max_attempts=1
    )

    with pytest.raises(ResultValidationError, match=reason):
        await process(command())

    assert_nothing_saved(unit_of_work, scenario.run_id)


async def test_treatment_of_the_answer_must_match_the_stored_plan(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Режим из плана — источник истины; расхождение отбраковывает группу (PRD § 14)."""
    scenario = store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT,),),
        treatments={"section-1": (BlockTreatment.GLOSSARY_ONLY,)},
    )
    process, _ = make_process_chapter(
        unit_of_work,
        FakeLLMClient.always(translation_reply([block_reply("b1")])),
        max_attempts=1,
    )

    with pytest.raises(ResultValidationError, match="не совпадает с планом"):
        await process(command())

    assert_nothing_saved(unit_of_work, scenario.run_id)


async def test_repair_instruction_names_blocks_as_the_model_saw_them(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повтор дописывает уточнение с псевдонимами и режимами блоков (ALG-05)."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.replying(
        [translation_reply([block_reply("b1")]), translated_reply()]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert (result.attempts, result.saved_blocks) == (2, 2)
    repair = client.last_request.instructions
    assert "- b1: режим обработки TRANSLATE" in repair
    assert "- b2: режим обработки TRANSLATE" in repair
    assert "Целевой язык результата — ru." in repair
    assert not any(block_id in repair for block_id in scenario.block_ids())


async def test_transport_failure_retries_with_backoff_and_saves_nothing(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Сбой до транзакции исчерпывает попытки и не оставляет частичного результата."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.always(LLMTransportError("провайдер недоступен"))
    process, delays = make_process_chapter(unit_of_work, client, max_attempts=3)

    with pytest.raises(LLMTransportError):
        await process(command())

    assert client.call_count == 3
    assert delays == [1.0, 2.0]
    assert_nothing_saved(unit_of_work, scenario.run_id)


async def test_transient_failure_of_one_unit_keeps_the_saved_one(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Группы сохраняются независимо: неудача второй не отменяет первую (PRD § 17)."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    client = FakeLLMClient.replying(
        [translated_reply(), LLMTransportError("провайдер недоступен")]
    )
    process, _ = make_process_chapter(
        unit_of_work,
        client,
        max_attempts=1,
        max_unit_source_chars=len(SOURCE_TEXT) * 2,
    )

    with pytest.raises(LLMTransportError):
        await process(command())

    with unit_of_work.transaction() as uow:
        results = uow.results.list_for_run(scenario.run_id)
    assert [item.block_id for item in results] == list(scenario.block_ids()[:2])
    assert all(item.translated_text == TRANSLATED_TEXT for item in results)
