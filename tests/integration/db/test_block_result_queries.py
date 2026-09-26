"""Чтение результатов запуска: раскладка заметок и замечаний и счётчик обработанных.

Заметки и замечания читаются на весь запуск сразу, поэтому их принадлежность блоку —
поведение репозитория, а не запроса к одной строке. Тесты закрепляют, что каждый блок
получает свои записи в прежнем порядке, а число обработанных блоков считается отдельно
от сборки самих результатов.
"""

from __future__ import annotations

import pytest
from persistence_factories import (
    BLOCK_ID,
    BOOK_ID,
    RUN_ID,
    SECOND_BLOCK_ID,
    make_book,
    make_document,
    make_run,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    LanguageNote,
    ResultWarning,
    ResultWarningKind,
)


def annotated_result(block_id: str, *, marker: str) -> BlockProcessingResult:
    """Перевод блока с двумя заметками и двумя замечаниями, помеченными своим блоком."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=f"Перевод блока {marker}.",
        language_notes=tuple(
            LanguageNote(
                id=f"ln-{marker}-{index}",
                block_id=block_id,
                original_excerpt=f"excerpt {marker} {index}",
                translated_excerpt=f"фрагмент {marker} {index}",
                note=f"Заметка {marker} {index}.",
            )
            for index in (1, 2)
        ),
        warnings=tuple(
            ResultWarning(
                kind=ResultWarningKind.REPETITION,
                message=f"Замечание {marker} {index}.",
            )
            for index in (1, 2)
        ),
    )


@pytest.fixture
def stored_results(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> tuple[BlockProcessingResult, ...]:
    """Запуск с результатами двух блоков: у каждого свои заметки и замечания."""
    results = (
        annotated_result(BLOCK_ID, marker="first"),
        annotated_result(SECOND_BLOCK_ID, marker="second"),
    )
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, list(results))
    return results


def test_results_of_a_run_keep_their_own_notes_and_warnings(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_results: tuple[BlockProcessingResult, ...],
) -> None:
    """Список результатов запуска отдаёт каждому блоку его заметки и замечания."""
    with unit_of_work.transaction() as uow:
        listed = uow.results.list_for_run(RUN_ID)
        single = uow.results.get(RUN_ID, SECOND_BLOCK_ID)

    assert listed == stored_results
    assert single == stored_results[1]


def test_count_for_run_matches_the_number_of_stored_results(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_results: tuple[BlockProcessingResult, ...],
) -> None:
    """Счётчик обработанных блоков совпадает с числом сохранённых результатов."""
    with unit_of_work.transaction() as uow:
        assert uow.results.count_for_run(RUN_ID) == len(stored_results)


def test_count_for_run_without_results_is_zero(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Запуск, у которого нет ни одного результата, даёт ноль, а не ошибку."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.runs.add(make_run())

    with unit_of_work.transaction() as uow:
        assert uow.results.count_for_run(RUN_ID) == 0
