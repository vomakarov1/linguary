"""Non-destructive book queue pause and resume use cases."""

from __future__ import annotations

from dataclasses import dataclass

from linguary.application.book_overview import BookNotFoundError
from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import BookStatus, TaskStatus
from linguary.domain.book_conditions import WorkStateError


class PauseBookError(ApplicationError):
    """Book queue pause or resume cannot be applied in the current state."""


@dataclass(frozen=True, slots=True)
class PauseBookCommand:
    """Request to pause claiming new pending tasks for a book."""

    book_id: str


@dataclass(frozen=True, slots=True)
class ResumeBookCommand:
    """Request to resume claiming pending tasks for a paused book."""

    book_id: str


@dataclass(frozen=True, slots=True)
class PauseBookResult:
    """Effective pause state of the book queue after the command."""

    book_id: str
    pause_requested: bool
    is_paused: bool
    running_tasks: int = 0
    pending_tasks: int = 0


class PauseBook:
    """Marks a book queue as paused in `work_bindings` while preserving `PENDING` tasks."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: PauseBookCommand | str) -> PauseBookResult:
        book_id = command if isinstance(command, str) else command.book_id
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise BookNotFoundError(f"книги {book_id} нет в хранилище")
            if book.status not in {BookStatus.ANALYZING, BookStatus.PROCESSING}:
                raise PauseBookError("книгу в текущем статусе нельзя поставить на паузу")
            conditions = transaction.work.latest_conditions(book.id)
            if conditions is None:
                raise PauseBookError("условия обработки книги ещё не зафиксированы")
            try:
                transaction.work.set_book_paused(book.id, conditions.id, True)
            except WorkStateError as error:
                raise PauseBookError(str(error)) from error
            tasks = transaction.tasks.list_for_book(book.id)
        running_tasks = sum(1 for task in tasks if task.status is TaskStatus.RUNNING)
        pending_tasks = sum(1 for task in tasks if task.status is TaskStatus.PENDING)
        return PauseBookResult(
            book_id=book_id,
            pause_requested=True,
            is_paused=running_tasks == 0,
            running_tasks=running_tasks,
            pending_tasks=pending_tasks,
        )


class ResumeBook:
    """Clears the book queue pause binding so workers can claim preserved `PENDING` tasks."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: ResumeBookCommand | str) -> PauseBookResult:
        book_id = command if isinstance(command, str) else command.book_id
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None:
                raise BookNotFoundError(f"книги {book_id} нет в хранилище")
            conditions = transaction.work.latest_conditions(book.id)
            try:
                transaction.work.set_book_paused(
                    book.id,
                    None if conditions is None else conditions.id,
                    False,
                )
            except WorkStateError as error:
                raise PauseBookError(str(error)) from error
            tasks = transaction.tasks.list_for_book(book.id)
        running_tasks = sum(1 for task in tasks if task.status is TaskStatus.RUNNING)
        pending_tasks = sum(1 for task in tasks if task.status is TaskStatus.PENDING)
        return PauseBookResult(
            book_id=book_id,
            pause_requested=False,
            is_paused=False,
            running_tasks=running_tasks,
            pending_tasks=pending_tasks,
        )


__all__ = [
    "PauseBook",
    "PauseBookCommand",
    "PauseBookError",
    "PauseBookResult",
    "ResumeBook",
    "ResumeBookCommand",
]
