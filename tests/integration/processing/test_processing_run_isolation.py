"""Поздний результат обработки и активный запуск книги (P3.T6, ALG-04).

Вызов модели идёт вне транзакции, поэтому между отправкой запроса и сохранением ответа
пользователь успевает вернуться к проверке глоссария: запуск с его результатами удаляется
(P3.T7). Транзакция сохранения обязана заметить это и отбросить пришедший результат —
иначе в книге появились бы переводы из отменённого запуска.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    SOURCE_TEXT,
    InterruptingLLMClient,
    make_process_chapter,
    store_processing_book,
    translated_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.domain import (
    BookEvent,
    ConceptType,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    TermStatus,
    apply_book_event,
)


def command() -> ProcessChapterCommand:
    """Команда обработки первой главы книги."""
    return ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1")


def reset_processing(unit_of_work: SqlAlchemyUnitOfWork) -> Callable[[], None]:
    """Возврат к проверке глоссария: запуск с результатами удалён (PRD § 11, P3.T7)."""

    def reset() -> None:
        with unit_of_work.transaction() as uow:
            uow.runs.delete(RUN_ID)
            book = uow.books.get(BOOK_ID)
            assert book is not None
            uow.books.save(
                replace(
                    book,
                    status=apply_book_event(
                        book.status, BookEvent.PROCESSING_RESET_CONFIRMED
                    ),
                )
            )

    return reset


def start_new_run(unit_of_work: SqlAlchemyUnitOfWork) -> Callable[[], None]:
    """Сброс обработки и новый запуск: у книги снова есть активный запуск, но другой."""
    reset = reset_processing(unit_of_work)

    def restart() -> None:
        reset()
        with unit_of_work.transaction() as uow:
            uow.runs.add(
                ProcessingRun(
                    id="run-2",
                    book_id=BOOK_ID,
                    status=ProcessingRunStatus.ACTIVE,
                    glossary_snapshot=GlossarySnapshot(
                        entries=(
                            GlossarySnapshotEntry(
                                source_text="Mr. Bennet",
                                translation="мистер Беннет",
                                type=ConceptType.PERSON,
                                status=TermStatus.LOCKED,
                            ),
                        )
                    ),
                    llm_settings_snapshot=LLMSettingsSnapshot(
                        provider="fake", model="fake-model"
                    ),
                )
            )

    return restart


async def test_result_of_a_cancelled_run_is_discarded(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Запуска больше нет — результат не сохраняется, обработка главы останавливается."""
    store_processing_book(
        unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT, SOURCE_TEXT),)
    )
    client = InterruptingLLMClient(
        inner=FakeLLMClient.always(translated_reply()),
        interrupt=reset_processing(unit_of_work),
    )
    process, _ = make_process_chapter(
        unit_of_work, client, max_unit_source_chars=len(SOURCE_TEXT) * 2
    )

    result = await process(command())

    assert result.discarded
    assert (result.units, result.processed_units, result.saved_blocks) == (2, 1, 0)
    assert client.call_count == 1
    with unit_of_work.transaction() as uow:
        assert uow.runs.get(RUN_ID) is None
        assert uow.runs.active_for_book(BOOK_ID) is None


async def test_result_of_a_previous_run_does_not_reach_the_new_one(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Ответ старого запуска не попадает в результаты нового запуска той же книги."""
    store_processing_book(unit_of_work)
    client = InterruptingLLMClient(
        inner=FakeLLMClient.always(translated_reply()),
        interrupt=start_new_run(unit_of_work),
    )
    process, _ = make_process_chapter(unit_of_work, client)

    result = await process(command())

    assert result.discarded and result.saved_blocks == 0
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run("run-2") == ()
