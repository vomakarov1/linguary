"""Reading run results: layout of notes and warnings and the count of processed blocks.

Notes and warnings are loaded for the entire run at once, so their block affiliation
is repository behavior rather than a single-row query. The tests ensure that each block
receives its records in their original order, and the count of processed blocks is computed
separately from assembling the results themselves.
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
    """Block translation with two notes and two warnings tagged with their block."""
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
    """Run with results for two blocks: each has its own notes and warnings."""
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
    """Run result listing assigns each block its own notes and warnings."""
    with unit_of_work.transaction() as uow:
        listed = uow.results.list_for_run(RUN_ID)
        single = uow.results.get(RUN_ID, SECOND_BLOCK_ID)

    assert listed == stored_results
    assert single == stored_results[1]


def test_count_for_run_matches_the_number_of_stored_results(
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_results: tuple[BlockProcessingResult, ...],
) -> None:
    """Processed block count matches the number of stored results."""
    with unit_of_work.transaction() as uow:
        assert uow.results.count_for_run(RUN_ID) == len(stored_results)


def test_count_for_run_without_results_is_zero(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Run without any results yields zero rather than an error."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.runs.add(make_run())

    with unit_of_work.transaction() as uow:
        assert uow.results.count_for_run(RUN_ID) == 0


def test_block_replacement_keeps_evidence_atomic(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence
    evidence = ContentEvidence('a' * 64, (HelpEvidence(
        'difficulty-1', 'meaning', 'language_note.note', SECOND_BLOCK_ID,
        'key-1', stored_results[1].language_notes[0].note,
    ),))
    changed = replace(stored_results[1], translated_text='Новый перевод.', content_evidence=evidence)
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [changed])
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == (stored_results[0], changed)
        assert uow.results.get(RUN_ID, SECOND_BLOCK_ID) == changed
    with pytest.raises(RuntimeError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [replace(changed, content_evidence=())])
        raise RuntimeError('отмена замены')
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == (stored_results[0], changed)


def test_group_serialization_failure_rolls_back_all_rows(unit_of_work, stored_results):
    from dataclasses import replace
    from linguary.application.ports.persistence import PersistenceError
    from linguary.domain.content_plan import ContentEvidence
    changed = replace(stored_results[0], translated_text='Новый перевод.', content_evidence=ContentEvidence('a' * 64))
    invalid = replace(stored_results[1], content_evidence={'private-book-text': 'invalid'})
    with pytest.raises(PersistenceError), unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [changed, invalid])
    with unit_of_work.transaction() as uow:
        assert uow.results.list_for_run(RUN_ID) == stored_results
