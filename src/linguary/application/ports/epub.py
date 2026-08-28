"""Порт чтения EPUB.

Прикладной слой не знает ни о контейнере EPUB, ни о XHTML, ни о файловой системе:
сценарий импорта получает готовую каноническую модель документа и предупреждения импорта.
Конкретная реализация живёт в `linguary.adapters.epub` (ADR § 3: логика EPUB находится
только во внешних компонентах импорта и экспорта).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class StoreResourcesRequest:
    """Запрос на сохранение ресурсов книги в локальное хранилище.

    Каноническая модель нужна целиком: по ней хранилище знает и состав ресурсов вместе
    с их статусом поддержки, и то, какие из них обязательны для книги.
    """

    book_id: str
    source_path: str
    document: CanonicalDocument


@dataclass(frozen=True, slots=True)
class StoredResources:
    """Результат сохранения: копия каждого ресурса по его исходному пути в книге."""

    paths: Mapping[str, str] = field(default_factory=dict)


class EpubResourceStore(Protocol):
    """Локальное хранилище ресурсов книги (PRD § 22).

    Сценарий импорта знает только то, что ресурсы книги где-то сохранены и что копию
    каждого можно найти по исходному пути ресурса; каталог хранилища, распаковка архива
    и защита от выхода за его пределы остаются ответственностью адаптера.
    """

    def store(self, request: StoreResourcesRequest) -> StoredResources:
        """Сохраняет ресурсы книги и отдаёт пути их копий.

        Реализация обязана поднимать `EpubImportFatalError` там, где по матрице поддержки
        наступает `FATAL`: потерян ресурс книги, путь ведёт за пределы хранилища книги,
        архив превышает допустимые лимиты распаковки.
        """


__all__ = [
    "EpubImportFatalError",
    "EpubReadRequest",
    "EpubReader",
    "EpubResourceStore",
    "ImportedBook",
    "StoreResourcesRequest",
    "StoredResources",
]
