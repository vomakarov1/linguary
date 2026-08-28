"""Адаптеры формата EPUB: импорт книги в каноническую модель документа.

Публичный вход адаптера — `EbookLibEpubReader`, реализация порта
`linguary.application.ports.epub.EpubReader`, и `FilesystemEpubResourceStore`, реализация
порта `linguary.application.ports.epub.EpubResourceStore`. Прикладной слой видит только
порты и о EbookLib, lxml и файлах книги ничего не знает (ADR § 3).
"""

from linguary.adapters.epub.reader import EbookLibEpubReader
from linguary.adapters.epub.resources import (
    ExtractionLimits,
    FilesystemEpubResourceStore,
)

__all__ = [
    "EbookLibEpubReader",
    "ExtractionLimits",
    "FilesystemEpubResourceStore",
]
