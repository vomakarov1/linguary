"""Registration of an uploaded book: the book row appears before its import.

The user sees the book in the list immediately after uploading the file, in the `IMPORTING`
status (PRD § 18), while EPUB parsing is executed by a background task. Therefore, the book
record is created by a dedicated short use case rather than by the import itself: otherwise
the book list would remain empty throughout parsing, and a failed import would leave no trace.

The book title is not yet known before file parsing — it is taken from the uploaded file name.
A successful import replaces metadata with the real values extracted from the EPUB.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import BookEvent, BookMetadata, apply_book_event


class RegisterBookError(ApplicationError):
    """An uploaded book cannot be registered."""


@dataclass(frozen=True, slots=True)
class RegisterBookCommand:
    """Registration command: where the file was stored and what languages to read it with."""

    book_id: str
    source_epub_path: str
    title: str
    source_language: str
    target_language: str

    def __post_init__(self) -> None:
        if not self.book_id.strip():
            raise RegisterBookError("идентификатор книги пуст")
        if not self.source_epub_path.strip():
            raise RegisterBookError("путь к загруженному файлу книги пуст")
        if not self.title.strip():
            raise RegisterBookError("название книги пусто")
        if not self.source_language.strip():
            raise RegisterBookError("исходный язык книги не выбран")
        if not self.target_language.strip():
            raise RegisterBookError("целевой язык книги не выбран")


class RegisterBook:
    """Registers a book in the `IMPORTING` status (ALG-06)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._clock = clock or _utc_now

    def __call__(self, command: RegisterBookCommand) -> BookRecord:
        """Creates a book record; re-registering the same identifier is forbidden."""
        record = BookRecord(
            id=command.book_id,
            metadata=BookMetadata(title=command.title),
            source_language=command.source_language,
            target_language=command.target_language,
            status=apply_book_event(None, BookEvent.FILE_UPLOADED),
            source_epub_path=command.source_epub_path,
            created_at=self._clock(),
        )
        with self._unit_of_work.transaction() as transaction:
            if transaction.books.get(command.book_id) is not None:
                raise RegisterBookError(
                    f"книга {command.book_id} уже зарегистрирована"
                )
            transaction.books.add(record)
        return record


def _utc_now() -> datetime:
    """Current UTC timestamp: persistence operates only with timezone-aware datetimes."""
    return datetime.now(UTC)


__all__ = [
    "RegisterBook",
    "RegisterBookCommand",
    "RegisterBookError",
]
