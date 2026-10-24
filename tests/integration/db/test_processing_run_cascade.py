"""Deleting a processing run: what disappears with it and what remains.

Returning to glossary review physically deletes results of the current run (PRD § 22),
but import, book blocks, and chapter analyses are preserved: they do not need recomputation.
"""

from __future__ import annotations

import pytest
from persistence_factories import (
    BLOCK_ID,
    BOOK_ID,
    RUN_ID,
    SECOND_BLOCK_ID,
    SECTION_ID,
    make_analysis,
    make_book,
    make_document,
    make_glossary_only_result,
    make_run,
    make_translated_result,
)
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from linguary.adapters.db.models import (
    Base,
    BlockResultRow,
    DocumentBlockRow,
    LanguageNoteRow,
    ResultWarningRow,
)
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork


@pytest.fixture
def processed_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Book with import, chapter analysis, processing run, and its results."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, make_analysis())
        uow.runs.add(make_run())
        uow.results.replace_results(
            RUN_ID, [make_translated_result(), make_glossary_only_result()]
        )


def count(engine: Engine, model: type[Base]) -> int:
    """Row count for a model table."""
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def test_stored_results_are_readable_before_deletion(
    unit_of_work: SqlAlchemyUnitOfWork, processed_book: None
) -> None:
    """Stored results are readable in original document block order."""
    with unit_of_work.transaction() as uow:
        results = uow.results.list_for_run(RUN_ID)

    assert [result.block_id for result in results] == [BLOCK_ID, SECOND_BLOCK_ID]
    assert results[0].language_notes[0].id == "ln-ch01-001"
    assert results[0].warnings[0].message == "Повтор слова «день» в абзаце."


def test_deleting_run_removes_results_notes_and_warnings(
    unit_of_work: SqlAlchemyUnitOfWork, processed_book: None, engine: Engine
) -> None:
    """Deleting a run cascades to block results, notes, and warnings."""
    assert count(engine, BlockResultRow) == 2
    assert count(engine, LanguageNoteRow) == 1
    assert count(engine, ResultWarningRow) == 1

    with unit_of_work.transaction() as uow:
        uow.runs.delete(RUN_ID)

    assert count(engine, BlockResultRow) == 0
    assert count(engine, LanguageNoteRow) == 0
    assert count(engine, ResultWarningRow) == 0


def test_deleting_run_keeps_blocks_and_chapter_analysis(
    unit_of_work: SqlAlchemyUnitOfWork, processed_book: None, engine: Engine
) -> None:
    """Import and chapter analysis survive the deletion of a processing run."""
    with unit_of_work.transaction() as uow:
        uow.runs.delete(RUN_ID)

    assert count(engine, DocumentBlockRow) == 2
    with unit_of_work.transaction() as uow:
        assert uow.runs.get(RUN_ID) is None
        document = uow.documents.get(BOOK_ID)
        analysis = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)

    assert document is not None
    assert len(document.section(SECTION_ID).blocks) == 2
    assert analysis is not None
    assert analysis.chapter_summary.startswith("Уинстон возвращается домой")


def test_replacing_results_does_not_duplicate_notes(
    unit_of_work: SqlAlchemyUnitOfWork, processed_book: None, engine: Engine
) -> None:
    """Re-saving a block result replaces the previous one instead of adding a second."""
    with unit_of_work.transaction() as uow:
        uow.results.replace_results(RUN_ID, [make_translated_result()])

    assert count(engine, BlockResultRow) == 2
    assert count(engine, LanguageNoteRow) == 1
    assert count(engine, ResultWarningRow) == 1
