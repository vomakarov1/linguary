"""Partial unique index: a book has at most one active processing run."""

from __future__ import annotations

import pytest
from persistence_factories import BOOK_ID, make_book, make_run
from sqlalchemy.exc import IntegrityError

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import ProcessingRunStatus


@pytest.fixture
def stored_book(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    """Book to which test processing runs are attached."""
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())


def test_second_active_run_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Second active run for the same book is rejected by the database."""
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
    """Partial index: completed and cancelled runs do not block a new active run."""
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
    """Completing an active run frees the slot for a new one."""
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
    """A book without runs has no active run."""
    with unit_of_work.transaction() as uow:
        assert uow.runs.active_for_book(BOOK_ID) is None
