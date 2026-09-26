"""Сценарий импорта книги из EPUB.

Сценарий читает книгу через порт `EpubReader`, сохраняет её ресурсы через порт
`EpubResourceStore` и одной короткой транзакцией сохраняет каноническую модель, запись
книги и предупреждения импорта (ADR § 3: чтение файла и запись ресурсов выполняются
вне транзакции). О формате EPUB, файловой системе и SQLite сценарий не знает.
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
    """Импорт книги невозможен в её текущем состоянии."""


@dataclass(frozen=True, slots=True)
class ImportBookCommand:
    """Команда импорта: какую книгу и с какими языками читать."""

    book_id: str
    source_epub_path: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class ImportBookResult:
    """Результат импорта: сохранённая модель книги и замечания для пользователя."""

    book: BookRecord
    document: CanonicalDocument
    warnings: tuple[BookImportWarning, ...]


class ImportBook:
    """Импортирует EPUB и переводит книгу в состояние `IMPORTED`."""

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
        """Читает EPUB и сохраняет книгу целиком.

        Книга, которую импортировать нельзя (`EpubImportFatalError`), переходит в `FAILED`:
        повтор той же задачи ничего не изменит, и карточка должна показать итог, а не
        бесконечный импорт (ALG-06). Ошибка после этого поднимается дальше — задачу
        неуспешной делает runner.
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
        """Переводит импортируемую книгу в `FAILED` после постоянной ошибки импорта."""
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
        """Сохраняет ресурсы книги, если хранилище подключено.

        Без хранилища книга импортируется одной моделью документа: диагностический разбор
        книги и импорт без последующего экспорта копий ресурсов не требуют.
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
    """Состояние книги после успешного импорта (ALG-06).

    Книги ещё нет — её создаёт сам импорт: загрузка файла и успешный импорт применяются
    подряд. Повторный запуск задачи импорта уже импортированной книги допустим и оставляет
    состояние прежним; импортировать книгу, которая ушла дальше по процессу, нельзя.
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
    """Текущий момент в UTC: хранилище работает только с осведомлённым временем."""
    return datetime.now(UTC)


__all__ = [
    "ImportBook",
    "ImportBookCommand",
    "ImportBookError",
    "ImportBookResult",
]
