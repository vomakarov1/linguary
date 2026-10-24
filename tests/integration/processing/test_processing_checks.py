"""Blocking checks for processing results and call retries (P3.T6, ALG-04, ALG-05).

Each check discards the entire block group: no translations, glossary hints,
or language notes from that reply remain in storage (PRD § 17). Responses are defined
by structured response text, so defective model replies are reproduced exactly as
they would arrive from a real provider.
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
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.ports.llm import LLMResponseFormatError, LLMTransportError
from linguary.application.retry_policy import ResultValidationError
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.domain import BlockTreatment

#: English text longer than language threshold: translation into target language was not performed.
ENGLISH_TRANSLATION = (
    "It is a truth universally acknowledged that a single man in possession of "
    "a good fortune must be in want of a wife, whatever his own feelings may be."
)

#: Translation truncated mid-sentence: missing terminal punctuation present in source.
UNFINISHED_TRANSLATION = (
    "Общепризнанная истина гласит, что мужчина, располагающий средствами, непременно"
)

#: Response truncated by output limit: unclosed JSON, with truncation reported by provider.
TRUNCATED_CONTENT = '{"blocks": [{"source_block_id": "b1", "treatment": "TRANSL'

#: Model refusal instead of translation: formally valid response unfit for the book.
SERVICE_MESSAGE = (
    "Извините, но я не могу выполнить эту просьбу: как языковая модель я не перевожу "
    "художественные тексты целиком."
)


def command() -> ProcessChapterCommand:
    """Command for processing the single book chapter."""
    return ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1")


def assert_nothing_saved(unit_of_work: SqlAlchemyUnitOfWork, run_id: str) -> None:
    """Discarded group leaves nothing in storage (PRD § 17)."""
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
    """Any blocking check cancels saving of the entire group (PRD § 15)."""
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
    """Planned treatment is source of truth; discrepancy rejects the group (PRD § 14)."""
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
    """Retry appends clarification with aliases and treatments of blocks (ALG-05)."""
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
    """Pre-transaction failure exhausts retry attempts and leaves no partial results."""
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
    """Groups are stored independently: failure of second does not discard first (PRD § 17)."""
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


async def test_output_limit_truncation_is_not_retried_by_the_same_request(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Output limit truncation is not retried: no second billed call is made."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.always(FakeReply(content=TRUNCATED_CONTENT, truncated=True))
    process, delays = make_process_chapter(unit_of_work, client, max_attempts=3)

    with pytest.raises(LLMResponseFormatError) as failure:
        await process(command())

    assert failure.value.truncated is True
    assert client.call_count == 1
    assert delays == []
    assert_nothing_saved(unit_of_work, scenario.run_id)


async def test_unparsable_answer_without_truncation_is_still_retried(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Format error without truncation is still retried: it is transient."""
    store_processing_book(unit_of_work)
    client = FakeLLMClient.replying(
        [FakeReply(content=TRUNCATED_CONTENT), translated_reply()]
    )
    process, _ = make_process_chapter(unit_of_work, client, max_attempts=3)

    result = await process(command())

    assert (result.attempts, result.saved_blocks) == (2, 2)
    assert client.call_count == 2
