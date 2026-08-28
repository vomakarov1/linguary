"""Сценарий импорта книги из EPUB.

Сценарий читает книгу через порт `EpubReader` и одной короткой транзакцией сохраняет
каноническую модель, запись книги и предупреждения импорта (ADR § 3: чтение файла
выполняется вне транзакции). О формате EPUB, файловой системе и SQLite сценарий не знает.

Копии ресурсов на диск сохраняет отдельный шаг импорта, поэтому отображение
«исходный путь ресурса → сохранённая копия» здесь пустое.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from linguary.application.errors import ApplicationError
from linguary.application.ports.epub import EpubReader, EpubReadRequest
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import (
    BookEvent,
    BookImportWarning,
    BookStatus,
    CanonicalDocument,
    apply_book_event,
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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._reader = reader
        self._clock = clock or _utc_now

    def __call__(self, command: ImportBookCommand) -> ImportBookResult:
        """Читает EPUB и сохраняет книгу целиком."""
        imported = self._reader.read(
            EpubReadRequest(
                book_id=command.book_id,
                source_path=command.source_epub_path,
                target_language=command.target_language,
                source_language=command.source_language,
            )
        )
        document = imported.document
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
            transaction.documents.save(document, stored_resource_paths={})
            if imported.warnings:
                transaction.documents.add_import_warnings(
                    command.book_id, imported.warnings
                )
        return ImportBookResult(
            book=record, document=document, warnings=imported.warnings
        )


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
