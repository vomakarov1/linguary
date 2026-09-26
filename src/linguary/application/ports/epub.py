"""Порты чтения и записи EPUB.

Прикладной слой не знает ни о контейнере EPUB, ни о XHTML, ни о файловой системе:
сценарий импорта получает готовую каноническую модель документа и предупреждения импорта,
а сценарий экспорта отдаёт модель вместе с результатами обработки и получает путь
к собранному файлу. Конкретная реализация живёт в `linguary.adapters.epub` (ADR § 3:
логика EPUB находится только во внешних компонентах импорта и экспорта).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from linguary.domain import (
    BlockProcessingResult,
    BookImportWarning,
    CanonicalDocument,
    DomainError,
    LearningMaterial,
)


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


class EpubExportFatalError(DomainError):
    """Собрать корректный EPUB нельзя (`SupportStatus.FATAL`, PRD § 6, § 20).

    Ошибка возникает, когда потерян обязательный ресурс книги или когда собранный файл
    не прошёл встроенные технические проверки экспортёра. Экспортная задача в этом случае
    получает `failed`, а недособранный артефакт не остаётся на диске.
    """


class ExportWarningKind(StrEnum):
    """Виды непреградных замечаний экспорта, которые пользователь видит в интерфейсе."""

    UNRESOLVED_LINK = "unresolved_link"
    MISSING_RESOURCE = "missing_resource"
    OPAQUE_FRAGMENT_LOST = "opaque_fragment_lost"


@dataclass(frozen=True, slots=True)
class ExportWarning:
    """Замечание сборки: что именно в итоговой книге отличается от исходной."""

    kind: ExportWarningKind
    message: str
    section_id: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise EpubExportFatalError("текст предупреждения экспорта пуст")


@dataclass(frozen=True, slots=True)
class EpubWriteRequest:
    """Запрос на сборку итогового EPUB.

    Состав главы определяется тем, что передал сценарий: без `block_results`
    и `learning_materials` получается книга режима `structure_only` — только оригинал
    (PRD § 20). `stored_resource_paths` сопоставляет пути ресурсов внутри книги их копиям
    в локальном хранилище, а `source_epub_path` нужен для переноса конструкций
    `PRESERVE_OPAQUE` исходной разметкой.
    """

    document: CanonicalDocument
    target_path: str
    stored_resource_paths: Mapping[str, str] = field(default_factory=dict)
    source_epub_path: str | None = None
    block_results: Mapping[str, BlockProcessingResult] = field(default_factory=dict)
    learning_materials: Mapping[str, LearningMaterial] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_path.strip():
            raise EpubExportFatalError("путь к собираемому файлу книги пуст")


@dataclass(frozen=True, slots=True)
class ExportedBook:
    """Результат сборки: путь к готовому файлу и замечания сборки."""

    path: str
    warnings: tuple[ExportWarning, ...] = ()


class EpubWriter(Protocol):
    """Сборка итогового EPUB из канонической модели и результатов обработки."""

    def write(self, request: EpubWriteRequest) -> ExportedBook:
        """Собирает книгу и выполняет встроенные технические проверки (PRD § 20).

        Реализация обязана поднимать `EpubExportFatalError`, если книгу нельзя собрать
        или собранный файл не прошёл встроенные проверки; всё, что не мешает книге быть
        корректной, складывается в `ExportedBook.warnings`.
        """


__all__ = [
    "EpubExportFatalError",
    "EpubImportFatalError",
    "EpubReadRequest",
    "EpubReader",
    "EpubResourceStore",
    "EpubWriteRequest",
    "EpubWriter",
    "ExportWarning",
    "ExportWarningKind",
    "ExportedBook",
    "ImportedBook",
    "StoreResourcesRequest",
    "StoredResources",
]
