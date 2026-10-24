"""Book import from EPUB use case.

The use case reads the book via the `EpubReader` port, stores its resources via the
`EpubResourceStore` port, and saves the canonical model, book record, and import warnings
in a single short transaction (ADR § 3: file reading and resource writing run outside
the transaction). The use case is agnostic of the EPUB container format, filesystem, and SQLite.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from linguary.application.errors import ApplicationError
from linguary.application.gutenberg import preserve_gutenberg_boilerplate
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubReader,
    EpubReadRequest,
    EpubResourceStore,
    StoreResourcesRequest,
)
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import (
    BookEvent,
    BookImportWarning,
    BookStatus,
    CanonicalDocument,
    ErrorType,
    TaskType,
    apply_book_event,
    status_after_task_failure,
)


class ImportBookError(ApplicationError):
    """Book import is not possible in its current state."""


@dataclass(frozen=True, slots=True)
class ImportBookCommand:
    """Import command: which book and languages to read."""

    book_id: str
    source_epub_path: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class ImportBookResult:
    """Import result: saved book model and warnings for the user."""

    book: BookRecord
    document: CanonicalDocument
    warnings: tuple[BookImportWarning, ...]


class ImportBook:
    """Imports an EPUB and transitions the book to the `IMPORTED` status."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        reader: EpubReader,
        resource_store: EpubResourceStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._reader = reader
        self._resource_store = resource_store
        self._clock = clock or _utc_now

    def __call__(self, command: ImportBookCommand) -> ImportBookResult:
        """Reads EPUB and saves the entire book.

        A book that cannot be imported (`EpubImportFatalError`) transitions to `FAILED`:
        retrying the same task will not change anything, and the card should reflect
        the outcome rather than an endless import (ALG-06). The error is re-raised
        afterwards — the runner marks the task failed.
        """
        try:
            imported = self._reader.read(
                EpubReadRequest(
                    book_id=command.book_id,
                    source_path=command.source_epub_path,
                    target_language=command.target_language,
                    source_language=command.source_language,
                )
            )
            document, gutenberg_warnings = preserve_gutenberg_boilerplate(
                imported.document
            )
            warnings = (*imported.warnings, *gutenberg_warnings)
            stored_resource_paths = self._store_resources(command, document)
        except EpubImportFatalError:
            self._mark_failed(command.book_id)
            raise
        with self._unit_of_work.transaction() as transaction:
            existing = transaction.books.get(command.book_id)
            record = BookRecord(
                id=command.book_id,
                metadata=document.metadata,
                source_language=document.source_language,
                target_language=document.target_language,
                status=_status_after_import(existing),
                source_epub_path=command.source_epub_path,
                created_at=existing.created_at if existing else self._clock(),
                export_validation_status=(
                    existing.export_validation_status if existing else None
                ),
            )
            if existing is None:
                transaction.books.add(record)
            else:
                transaction.books.save(record)
            transaction.documents.save(
                document, stored_resource_paths=stored_resource_paths
            )
            if warnings:
                transaction.documents.add_import_warnings(
                    command.book_id, warnings
                )
        return ImportBookResult(
            book=record, document=document, warnings=warnings
        )

    def _mark_failed(self, book_id: str) -> None:
        """Transitions an importing book to `FAILED` after a permanent import error."""
        with self._unit_of_work.transaction() as transaction:
            book = transaction.books.get(book_id)
            if book is None or book.status is not BookStatus.IMPORTING:
                return
            transaction.books.save(
                replace(
                    book,
                    status=status_after_task_failure(
                        book.status,
                        task_type=TaskType.IMPORT_BOOK,
                        error_type=ErrorType.PERMANENT,
                    ),
                )
            )

    def _store_resources(
        self, command: ImportBookCommand, document: CanonicalDocument
    ) -> Mapping[str, str]:
        """Stores book resources if a resource store is configured.

        Without a resource store, the book is imported solely as a document model:
        diagnostic parsing and import without subsequent export do not require resource copies.
        """
        if self._resource_store is None:
            return {}
        return self._resource_store.store(
            StoreResourcesRequest(
                book_id=command.book_id,
                source_path=command.source_epub_path,
                document=document,
            )
        ).paths


def _status_after_import(existing: BookRecord | None) -> BookStatus:
    """Book status after successful import (ALG-06).

    If the book does not exist yet, import creates it: file upload and successful import
    are applied sequentially. Retrying an import task on an already imported book is
    permitted and preserves its status; importing a book that has progressed further
    in the workflow is rejected.
    """
    if existing is None:
        return apply_book_event(
            apply_book_event(None, BookEvent.FILE_UPLOADED),
            BookEvent.IMPORT_SUCCEEDED,
        )
    if existing.status is BookStatus.IMPORTED:
        return BookStatus.IMPORTED
    if existing.status is not BookStatus.IMPORTING:
        raise ImportBookError(
            f"книгу в состоянии {existing.status} импортировать повторно нельзя"
        )
    return apply_book_event(existing.status, BookEvent.IMPORT_SUCCEEDED)


def _utc_now() -> datetime:
    """Current UTC timestamp: persistence operates only with timezone-aware datetimes."""
    return datetime.now(UTC)
    return datetime.now(UTC)


__all__ = [
    "ImportBook",
    "ImportBookCommand",
    "ImportBookError",
    "ImportBookResult",
]
