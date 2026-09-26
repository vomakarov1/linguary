"""Повторная генерация обработки одного блока (P3.T7, PRD § 13, § 19.3, § 21.5).

Проверяется главное обещание действия: заменяется результат выбранного блока и только
его. Соседние блоки модель видит как контекст сцены, но не переводит, а их сохранённые
результаты остаются нетронутыми.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    TRANSLATED_TEXT,
    InterruptingLLMClient,
    block_reply,
    make_process_chapter,
    make_regenerate_block,
    note_reply,
    store_book_with_rejected_variant,
    store_processing_book,
    translated_reply,
    translation_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.application.context_builder import ContextBuilder
from linguary.application.retry_policy import ResultValidationError
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.application.use_cases.regenerate_block import (
    RegenerateBlockCommand,
    RegenerateBlockError,
)
from linguary.domain import BlockProcessingResult, BlockTreatment

#: Другой перевод того же абзаца: по нему видно, что результат блока действительно заменён.
REGENERATED_TEXT = "Всем известно: одинокий богач непременно ищет себе жену."

#: Ответ не на целевом языке и длиннее порога проверки языка: такой результат отбраковывается.
UNTRANSLATED_TEXT = (
    "This paragraph came back in English instead of Russian, so the language check "
    "must reject the whole answer."
)

#: Глава из пяти различимых абзацев: по тексту видно, какие соседи попали в запрос.
DISTINCT_CHAPTER = (
    "The first paragraph opens the scene.",
    "The second paragraph introduces the guest.",
    "The third paragraph carries the conversation.",
    "The fourth paragraph turns the mood.",
    "The fifth paragraph closes the visit.",
)


def stored_results(
    unit_of_work: SqlAlchemyUnitOfWork, run_id: str = RUN_ID
) -> tuple[BlockProcessingResult, ...]:
    """Сохранённые результаты запуска в порядке блоков документа."""
    with unit_of_work.transaction() as uow:
        return uow.results.list_for_run(run_id)


def _delete_run(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Удаляет активный запуск книги: так выглядит возврат к проверке глоссария."""
    with unit_of_work.transaction() as uow:
        uow.runs.delete(RUN_ID)


async def process_chapter(
    unit_of_work: SqlAlchemyUnitOfWork,
    *,
    aliases: tuple[str, ...],
    notes: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Обрабатывает первую главу целиком: у блоков появляются результаты и заметки."""
    process, _ = make_process_chapter(
        unit_of_work, FakeLLMClient.always(translated_reply(aliases, notes=notes))
    )
    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1"))


async def test_regeneration_replaces_only_the_target_block(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повторная генерация меняет выбранный блок и не трогает соседние (PRD § 21.5)."""
    scenario = store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    await process_chapter(unit_of_work, aliases=("b1", "b2", "b3"))
    client = FakeLLMClient.replying(
        [translation_reply([block_reply("b1", translated_text=REGENERATED_TEXT)])]
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    result = await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[1])
    )

    assert (result.saved, result.attempts, result.run_id) == (True, 1, RUN_ID)
    assert [item.translated_text for item in stored_results(unit_of_work)] == [
        TRANSLATED_TEXT,
        REGENERATED_TEXT,
        TRANSLATED_TEXT,
    ]


async def test_regeneration_replaces_notes_of_the_target_block(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Заметки заменяются вместе с блоком и не сталкиваются с чужими (PRD § 19.3)."""
    scenario = store_processing_book(unit_of_work)
    await process_chapter(
        unit_of_work, aliases=("b1", "b2"), notes=(note_reply("b1"), note_reply("b2"))
    )
    client = FakeLLMClient.replying(
        [
            translation_reply(
                [block_reply("b1", translated_text=REGENERATED_TEXT)],
                notes=(note_reply("b1", note="Заново объяснённый оборот."),),
            )
        ]
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[0])
    )

    regenerated, untouched = stored_results(unit_of_work)
    assert [note.id for note in regenerated.language_notes] == ["ln-ch01-003"]
    assert regenerated.language_notes[0].note == "Заново объяснённый оборот."
    assert [note.id for note in untouched.language_notes] == ["ln-ch01-002"]


async def test_request_carries_neighbour_blocks_without_aliases(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Соседние абзацы идут в запрос контекстом сцены, а обрабатывается один блок."""
    scenario = store_processing_book(unit_of_work, chapters=(DISTINCT_CHAPTER,))
    client = FakeLLMClient.always(
        translation_reply([block_reply("b1", translated_text=REGENERATED_TEXT)])
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[2])
    )

    text = client.last_request.input_text
    assert "Соседние абзацы главы" in text
    assert text.index("До блока:") < text.index("После блока:")
    assert all(paragraph in text for paragraph in DISTINCT_CHAPTER)
    assert f"[b1] режим {BlockTreatment.TRANSLATE}" in text
    assert "[b2]" not in text


async def test_neighbour_context_is_limited_by_the_builder(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Соседей ровно столько, сколько разрешает `ContextBuilder`, и только своей главы."""
    scenario = store_processing_book(unit_of_work, chapters=(DISTINCT_CHAPTER,))
    client = FakeLLMClient.always(
        translation_reply([block_reply("b1", translated_text=REGENERATED_TEXT)])
    )
    regenerate, _ = make_regenerate_block(
        unit_of_work, client, context_builder=ContextBuilder(regeneration_context_blocks=1)
    )

    await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[0])
    )

    text = client.last_request.input_text
    assert "До блока:" not in text
    assert DISTINCT_CHAPTER[1] in text
    assert DISTINCT_CHAPTER[2] not in text


async def test_result_for_a_neighbour_block_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Модель, вернувшая заодно соседний блок, результата не сохраняет (ALG-04)."""
    scenario = store_processing_book(unit_of_work)
    await process_chapter(unit_of_work, aliases=("b1", "b2"))
    client = FakeLLMClient.always(
        translation_reply(
            [
                block_reply("b1", translated_text=REGENERATED_TEXT),
                block_reply("b2", translated_text=REGENERATED_TEXT),
            ]
        )
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client, max_attempts=1)

    with pytest.raises(ResultValidationError, match="вне запроса"):
        await regenerate(
            RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[0])
        )

    assert [item.translated_text for item in stored_results(unit_of_work)] == [
        TRANSLATED_TEXT,
        TRANSLATED_TEXT,
    ]


async def test_repeated_rejection_asks_the_model_again_with_the_alias(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повтор получает уточнение с тем именем блока, которое видела модель (ALG-05)."""
    scenario = store_processing_book(unit_of_work)
    client = FakeLLMClient.replying(
        [
            translation_reply([block_reply("b1", translated_text=UNTRANSLATED_TEXT)]),
            translation_reply([block_reply("b1", translated_text=REGENERATED_TEXT)]),
        ]
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    result = await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[0])
    )

    assert (result.saved, result.attempts) == (True, 2)
    repair = client.last_request.instructions
    assert "b1: режим обработки TRANSLATE" in repair
    assert scenario.block_ids()[0] not in repair


async def test_late_result_of_a_cancelled_run_is_discarded(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Запуск исчез, пока ответ был в пути, — результат не сохраняется (ALG-04)."""
    scenario = store_processing_book(unit_of_work)
    client = InterruptingLLMClient(
        inner=FakeLLMClient.always(
            translation_reply([block_reply("b1", translated_text=REGENERATED_TEXT)])
        ),
        interrupt=lambda: _delete_run(unit_of_work),
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    result = await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[0])
    )

    assert not result.saved
    assert stored_results(unit_of_work) == ()


async def test_block_outside_the_book_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Блок, которого нет в книге, повторной генерации не получает."""
    store_processing_book(unit_of_work)
    regenerate, _ = make_regenerate_block(unit_of_work, FakeLLMClient())

    with pytest.raises(RegenerateBlockError, match="нет в книге"):
        await regenerate(
            RegenerateBlockCommand(book_id=BOOK_ID, block_id="block-404")
        )


async def test_non_textual_block_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Иллюстрация результата обработки не имеет: генерировать заново нечего (PRD § 10.1)."""
    scenario = store_processing_book(unit_of_work, chapters=((SOURCE_TEXT, None),))
    client = FakeLLMClient()
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    with pytest.raises(RegenerateBlockError, match="не является текстом"):
        await regenerate(
            RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[1])
        )

    assert client.call_count == 0


async def test_block_outside_the_treatment_plan_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Блок, который план не отдаёт модели, обрабатывать нельзя (PRD § 14)."""
    scenario = store_processing_book(
        unit_of_work,
        chapters=((SOURCE_TEXT, SOURCE_TEXT),),
        treatments={
            "section-1": (BlockTreatment.TRANSLATE, BlockTreatment.PRESERVE_ONLY)
        },
    )
    regenerate, _ = make_regenerate_block(unit_of_work, FakeLLMClient())

    with pytest.raises(RegenerateBlockError, match="не входит в план"):
        await regenerate(
            RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[1])
        )


async def test_regeneration_without_an_active_run_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Без активного запуска заменять нечего: результаты хранит запуск (PRD § 22)."""
    scenario = store_processing_book(unit_of_work)
    with unit_of_work.transaction() as uow:
        uow.runs.delete(RUN_ID)
    regenerate, _ = make_regenerate_block(unit_of_work, FakeLLMClient())

    with pytest.raises(RegenerateBlockError, match="активного запуска"):
        await regenerate(
            RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids()[0])
        )


async def test_rejected_variant_from_other_chapter_does_not_reach_regeneration(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Повторная генерация сверяет указания с кандидатами всей книги, как обработка (Д-4)."""
    scenario = store_book_with_rejected_variant(unit_of_work)
    client = FakeLLMClient.always(
        translation_reply([block_reply("b1", translated_text=REGENERATED_TEXT)])
    )
    regenerate, _ = make_regenerate_block(unit_of_work, client)

    await regenerate(
        RegenerateBlockCommand(book_id=BOOK_ID, block_id=scenario.block_ids(1)[0])
    )

    text = client.last_request.input_text
    assert "Даскмир" not in text
    assert "(Множественное число, склоняется.)" in text
