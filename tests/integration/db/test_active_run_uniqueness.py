"""Частичный уникальный индекс: у книги не более одного активного запуска обработки."""

from __future__ import annotations

import pytest
from persistence_factories import BOOK_ID, make_book, make_run
from sqlalchemy.exc import IntegrityError

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import ProcessingRunStatus


@pytest.fixture
def stored_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Книга, к которой привязываются запуски обработки теста."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())


def test_second_active_run_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Второй активный запуск той же книги отвергается базой."""
    with unit_of_work.transaction() as uow:
        uow.runs.add(make_run(run_id="run-1"))

    with pytest.raises(IntegrityError), unit_of_work.transaction() as uow:
        uow.runs.add(make_run(run_id="run-2"))


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(ProcessingRunStatus.COMPLETED, id="completed"),
        pytest.param(ProcessingRunStatus.CANCELLED, id="cancelled"),
    ],
)
def test_inactive_runs_do_not_block_new_active_run(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None, status: ProcessingRunStatus
) -> None:
    """Индекс частичный: завершённые и отменённые запуски новому не мешают."""
    with unit_of_work.transaction() as uow:
        uow.runs.add(make_run(run_id="run-old", status=status))
        uow.runs.add(make_run(run_id="run-new"))

    with unit_of_work.transaction() as uow:
        active = uow.runs.active_for_book(BOOK_ID)

    assert active is not None
    assert active.id == "run-new"


def test_completing_active_run_frees_the_slot(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """После завершения запуска книга снова может получить активный."""
    with unit_of_work.transaction() as uow:
        uow.runs.add(make_run(run_id="run-1"))

    with unit_of_work.transaction() as uow:
        uow.runs.save(
            make_run(run_id="run-1", status=ProcessingRunStatus.COMPLETED)
        )
        uow.runs.add(make_run(run_id="run-2"))

    with unit_of_work.transaction() as uow:
        active = uow.runs.active_for_book(BOOK_ID)

    assert active is not None
    assert active.id == "run-2"


def test_book_without_runs_has_no_active_run(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """У книги без запусков активного запуска нет."""
    with unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(BOOK_ID) is None
