"""Адаптеры формата EPUB: импорт книги в каноническую модель документа и её экспорт.

Публичный вход адаптера — `EbookLibEpubReader`, `FilesystemEpubResourceStore`
и `EpubExportWriter`, реализации портов `EpubReader`, `EpubResourceStore` и `EpubWriter`
из `linguary.application.ports.epub`. Прикладной слой видит только порты и о EbookLib,
lxml и файлах книги ничего не знает (ADR § 3).

`verify_exported_epub` вынесена в публичный API адаптера отдельно от писателя: те же
встроенные технические проверки нужны диагностике через CLI и проверке чужого файла,
а не только собственной сборке.
"""

from linguary.adapters.epub.reader import EbookLibEpubReader
from linguary.adapters.epub.resources import (
    ExtractionLimits,
    FilesystemEpubResourceStore,
)
from linguary.adapters.epub.writer import (
    EpubExportWriter,
    ExportCheckFailure,
    ExportCheckKind,
    verify_exported_epub,
)

__all__ = [
    "EbookLibEpubReader",
    "EpubExportWriter",
    "ExportCheckFailure",
    "ExportCheckKind",
    "ExtractionLimits",
    "FilesystemEpubResourceStore",
    "verify_exported_epub",
]
