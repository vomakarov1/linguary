"""Обработка главы: группы блоков, сохранение результата и учебные заметки (P3.T6, ALG-04).

Тесты идут на настоящем SQLite и `FakeLLMClient`: сетевых вызовов в наборе нет, а всё,
что касается транзакции и замены результатов, проверяется на том хранилище, которое
получит пользователь.
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
from linguary.adapters.llm import FakeLLMClient, FakeScenarioError
from linguary.application.use_cases.process_chapter import (
    ProcessChapterCommand,
    ProcessChapterError,
)
from linguary.domain import BlockProcessingResult, BlockTreatment, ResultWarningKind

#: Абзац с намеренным повтором одного оборота: проверка повторов даёт предупреждение.
REPEATED_TRANSLATION = (
    "Он шёл и шёл вперёд, он шёл и шёл вперёд, он шёл и шёл вперёд, "
    "он шёл и шёл вперёд, и не видел конца."
)


def command(section_id: str = "section-1") -> ProcessChapterCommand:
    """Команда обработки главы книги."""
    return ProcessChapterCommand(book_id=BOOK_ID, section_id=section_id)


def stored_results(
    unit_of_work: SqlAlchemyUnitOfWork, run_id: str
) -> tuple[BlockProcessingResult, ...]:
    """Сохранённые результаты запуска в порядке блоков документа."""
    with unit_of_work.transaction() as uow:
        return uow.results.list_for_run(run_id)


async def test_chapter_results_are_saved_for_every_block(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Успешная обработка сохраняет результат каждого блока главы (PRD § 14)."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.replying([translated_reply()])
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert (result.units, result.processed_units, result.saved_blocks) == (1, 1, 2)
    assert not result.discarded
    assert client.call_count == 1
    results = stored_results(unit_of_work, scenario.run_id)
    assert [item.block_id for item in results] == list(scenario.block_ids())
    assert all(item.translated_text == TRANSLATED_TEXT for item in results)
    assert all(item.treatment is BlockTreatment.TRANSLATE for item in results)


async def test_language_notes_are_saved_with_chapter_numbered_ids(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Заметки живут в своём пространстве идентификаторов `ln-chNN-NNN` (PRD § 19.3)."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.replying(
        [translated_reply(notes=(note_reply("b1"), note_reply("b2")))]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    results = stored_results(unit_of_work, scenario.run_id)
    assert [note.id for item in results for note in item.language_notes] == [
        "ln-ch01-001",
        "ln-ch01-002",
    ]
    assert results[0].language_notes[0].block_id == scenario.block_ids()[0]


async def test_glossary_only_block_gets_hints_without_translation(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Режим из плана определяет состав результата блока (PRD § 10.1, § 15)."""
    scenario = store_processing_book(
        unit_of_work,
        treatments={
            "section-1": (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS)
        },
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.GLOSSARY_ONLY,
                        hints=(("entailed", "заповедный"),),
                    ),
                    block_reply("b2", treatment=BlockTreatment.READ_AS_IS),
                ]
            )
        ]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    hinted, plain = stored_results(unit_of_work, scenario.run_id)
    assert hinted.translated_text is None
    assert [hint.original for hint in hinted.glossary_hints] == ["entailed"]
    assert plain.translated_text is None and plain.glossary_hints == ()


async def test_hints_and_notes_repeating_before_you_read_are_not_saved(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Выражение из «Before You Read» не повторяется ни подсказкой, ни заметкой (QA-03)."""
    scenario = store_processing_book(
        unit_of_work,
        treatments={
            "section-1": (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE)
        },
        before_you_read=(("heaved and groaned", "вздымался и стонал"),),
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.GLOSSARY_ONLY,
                        hints=(
                            ("Heaved and  groaned,", "вздымалось и стонало"),
                            ("entailed", "заповедный"),
                        ),
                    ),
                    block_reply("b2"),
                ],
                notes=(
                    note_reply("b1", original_excerpt="heaved and groaned"),
                    note_reply("b2", original_excerpt="the house heaved and groaned"),
                    note_reply("b2"),
                ),
            )
        ]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    hinted, translated = stored_results(unit_of_work, scenario.run_id)
    assert [hint.original for hint in hinted.glossary_hints] == ["entailed"]
    assert hinted.language_notes == ()
    assert [note.original_excerpt for note in translated.language_notes] == [
        "a truth universally acknowledged"
    ]


async def test_glossary_only_block_keeps_its_only_hint_even_if_explained(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Обязательная подсказка блока GLOSSARY_ONLY не отсеивается до пустого списка."""
    scenario = store_processing_book(
        unit_of_work,
        treatments={
            "section-1": (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS)
        },
        before_you_read=(("entailed", "заповедный"),),
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [
                    block_reply(
                        "b1",
                        treatment=BlockTreatment.GLOSSARY_ONLY,
                        hints=(("entailed", "заповедный"),),
                    ),
                    block_reply("b2", treatment=BlockTreatment.READ_AS_IS),
                ]
            )
        ]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    await process(command())

    hinted, _ = stored_results(unit_of_work, scenario.run_id)
    assert [hint.original for hint in hinted.glossary_hints] == ["entailed"]


async def test_chapter_without_textual_blocks_does_not_call_the_model(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Глава-иллюстрация обрабатывается без обращения к модели (PRD § 10.1)."""
    scenario = store_processing_book(unit_of_work, chapters=((None,),))
    client = FakeLLMClient()
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert (result.units, result.processed_units, result.attempts) == (0, 0, 0)
    assert client.call_count == 0
    assert stored_results(unit_of_work, scenario.run_id) == ()


async def test_long_chapter_is_processed_by_several_units(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Глава, не помещающаяся в один запрос, обрабатывается группами (PRD § 10)."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    client = FakeLLMClient.always(translated_reply())
    process, _ = make_process_chapter(
        unit_of_work, client, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )

    result = await process(command())

    assert (result.units, result.processed_units, result.saved_blocks) == (2, 2, 4)
    assert client.call_count == 2
    assert len(stored_results(unit_of_work, scenario.run_id)) == 4


async def test_saved_units_are_not_requested_again(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повтор задачи не платит за уже сохранённые группы (PRD § 18.1)."""
    scenario = store_processing_book(unit_of_work)
    first = FakeLLMClient.replying([translated_reply()])
    process, _ = make_process_chapter(unit_of_work, first)
    await process(command())

    second = FakeLLMClient()
    repeated, _ = make_process_chapter(unit_of_work, second)
    result = await repeated(command())

    assert (result.units, result.processed_units) == (1, 0)
    assert second.call_count == 0
    assert len(stored_results(unit_of_work, scenario.run_id)) == 2


async def test_partially_saved_chapter_continues_without_note_id_collision(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повтор недоделанной главы продолжает нумерацию заметок, а не начинает её заново."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    failing = FakeLLMClient.replying(
        [translated_reply(notes=(note_reply("b1"), note_reply("b2")))]
    )
    process, _ = make_process_chapter(
        unit_of_work, failing, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )
    with pytest.raises(FakeScenarioError):
        await process(command())

    continued = FakeLLMClient.always(translated_reply(notes=(note_reply("b1"),)))
    resumed, _ = make_process_chapter(
        unit_of_work, continued, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )
    result = await resumed(command())

    assert (result.units, result.processed_units) == (2, 1)
    note_ids = [
        note.id
        for item in stored_results(unit_of_work, scenario.run_id)
        for note in item.language_notes
    ]
    assert note_ids == ["ln-ch01-001", "ln-ch01-002", "ln-ch01-003"]


async def test_repetition_warning_is_stored_and_does_not_block_saving(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повторы в переводе остаются предупреждением и результат сохраняют (PRD § 15)."""
    scenario = store_processing_book(unit_of_work, chapters=((SOURCE_TEXT,),))
    client = FakeLLMClient.replying(
        [translation_reply([block_reply("b1", translated_text=REPEATED_TRANSLATION)])]
    )
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert result.saved_blocks == 1
    stored = stored_results(unit_of_work, scenario.run_id)[0]
    assert stored.translated_text == REPEATED_TRANSLATION
    assert [warning.kind for warning in stored.warnings] == [
        ResultWarningKind.REPETITION
    ]


async def test_unanalysed_chapter_is_not_processed(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Глава без сохранённого анализа обрабатываться не может (PRD § 14)."""
    store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT,), (SOURCE_TEXT,)),
        analysed_sections=("section-1",),
    )
    client = FakeLLMClient()
    process, _ = make_process_chapter(unit_of_work, client)

    with pytest.raises(ProcessChapterError, match="не проанализирована"):
        await process(command("section-2"))

    assert client.call_count == 0
