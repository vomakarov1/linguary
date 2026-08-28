"""Порт чтения EPUB.

Прикладной слой не знает ни о контейнере EPUB, ни о XHTML, ни о файловой системе:
сценарий импорта получает готовую каноническую модель документа и предупреждения импорта.
Конкретная реализация живёт в `linguary.adapters.epub` (ADR § 3: логика EPUB находится
только во внешних компонентах импорта и экспорта).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from linguary.domain import BookImportWarning, CanonicalDocument, DomainError


class EpubImportFatalError(DomainError):
    """EPUB импортировать нельзя (`SupportStatus.FATAL`, PRD § 6).

    Причина попадает в сообщение и показывается пользователю: нечитаемый архив,
    отсутствующий spine или порядок чтения, XHTML, который не разбирается до блоков.
    Ошибка доменная, потому что нарушено само условие существования книги в модели,
    а не контракт инфраструктуры.
    """


@dataclass(frozen=True, slots=True)
class EpubReadRequest:
    """Запрос на чтение EPUB.

    `book_id` нужен читателю до разбора: идентификатор блока считается от `document_id`,
    а он совпадает с идентификатором книги (PRD § 8). `source_language` не задан, если
    язык книги берётся из метаданных EPUB.
    """

    book_id: str
    source_path: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class ImportedBook:
    """Результат чтения EPUB: каноническая модель и непреградные замечания."""

    document: CanonicalDocument
    warnings: tuple[BookImportWarning, ...] = ()


class EpubReader(Protocol):
    """Чтение EPUB в каноническую модель документа."""

    def read(self, request: EpubReadRequest) -> ImportedBook:
        """Читает книгу и строит каноническую модель.

        Реализация обязана поднимать `EpubImportFatalError` там, где по матрице поддержки
        наступает `FATAL`, и складывать всё остальное в `ImportedBook.warnings`.
        """


__all__ = [
    "EpubImportFatalError",
    "EpubReadRequest",
    "EpubReader",
    "ImportedBook",
]
