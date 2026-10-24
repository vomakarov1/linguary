"""EPUB format adapters: importing books into the canonical document model and exporting.

The adapter's public entry points are `EbookLibEpubReader`, `FilesystemEpubResourceStore`,
and `EpubExportWriter`, which implement the `EpubReader`, `EpubResourceStore`, and `EpubWriter` ports
from `linguary.application.ports.epub`. The application layer only sees the ports and has no knowledge
of EbookLib, lxml, or raw book files (ADR § 3).

`verify_exported_epub` is exposed in the public adapter API separately from the writer: the same
built-in technical checks are needed for CLI diagnostics and third-party file inspection,
not just our own exports.
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
