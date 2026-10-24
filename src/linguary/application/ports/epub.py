"""Ports for reading and writing EPUB.

The application layer knows nothing about the EPUB container, XHTML, or the file system:
the import use case receives a ready canonical document model and import warnings,
while the export use case provides the model along with processing results and receives
the path to the assembled file. The concrete implementation lives in `linguary.adapters.epub`
(ADR § 3: EPUB logic belongs only in external import and export components).
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
    """EPUB cannot be imported (`SupportStatus.FATAL`, PRD § 6).

    The reason is included in the message and shown to the user: unreadable archive,
    missing spine or reading order, or XHTML that cannot be parsed into blocks.
    The error is a domain error because the precondition for the book's existence
    in the model is violated, not an infrastructure contract.
    """


@dataclass(frozen=True, slots=True)
class EpubReadRequest:
    """Request to read an EPUB.

    `book_id` is required by the reader before parsing: block identity is computed
    from `document_id`, which matches the book id (PRD § 8). `source_language` is
    optional if the language is inferred from EPUB metadata.
    """

    book_id: str
    source_path: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class ImportedBook:
    """Result of reading an EPUB: canonical model and non-blocking warnings."""

    document: CanonicalDocument
    warnings: tuple[BookImportWarning, ...] = ()


class EpubReader(Protocol):
    """Reads EPUB into a canonical document model."""

    def read(self, request: EpubReadRequest) -> ImportedBook:
        """Reads a book and builds a canonical model.

        The implementation must raise `EpubImportFatalError` wherever the support matrix
        specifies `FATAL`, and collect everything else into `ImportedBook.warnings`.
        """


@dataclass(frozen=True, slots=True)
class StoreResourcesRequest:
    """Request to store book resources in local storage.

    The canonical model is required in its entirety: from it, the storage learns
    the resource inventory along with their support status, and which ones are mandatory.
    """

    book_id: str
    source_path: str
    document: CanonicalDocument


@dataclass(frozen=True, slots=True)
class StoredResources:
    """Result of storing resources: copy of each resource by its original path in the book."""

    paths: Mapping[str, str] = field(default_factory=dict)


class EpubResourceStore(Protocol):
    """Local storage of book resources (PRD § 22).

    The import use case only knows that book resources are preserved somewhere and that
    a copy of each can be retrieved by its original resource path; storage directory,
    archive extraction, and path traversal protection remain the adapter's responsibility.
    """

    def store(self, request: StoreResourcesRequest) -> StoredResources:
        """Stores book resources and returns paths to their copies.

        The implementation must raise `EpubImportFatalError` when the support matrix
        specifies `FATAL`: missing mandatory resource, path traversal out of the book's
        storage directory, or archive exceeding extraction limits.
        """


class EpubExportFatalError(DomainError):
    """A valid EPUB cannot be built (`SupportStatus.FATAL`, PRD § 6, § 20).

    Occurs when a required book resource is missing or when the assembled file fails
    the exporter's built-in technical checks. In this case, the export task fails,
    and incomplete artifacts are not left on disk.
    """


class ExportWarningKind(StrEnum):
    """Kinds of non-blocking export warnings presented to the user in the UI."""

    UNRESOLVED_LINK = "unresolved_link"
    MISSING_RESOURCE = "missing_resource"
    OPAQUE_FRAGMENT_LOST = "opaque_fragment_lost"


@dataclass(frozen=True, slots=True)
class ExportWarning:
    """Build warning: what exactly in the resulting book differs from the original."""

    kind: ExportWarningKind
    message: str
    section_id: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise EpubExportFatalError("текст предупреждения экспорта пуст")


@dataclass(frozen=True, slots=True)
class EpubWriteRequest:
    """Request to build the final EPUB.

    Section composition is determined by what the use case passes: omitting `block_results`
    and `learning_materials` produces a `structure_only` book — original text only
    (PRD § 20). `stored_resource_paths` maps resource paths inside the book to their copies
    in local storage, and `source_epub_path` is needed to carry over `PRESERVE_OPAQUE`
    markup from the source.
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
    """Result of building: path to the assembled file and build warnings."""

    path: str
    warnings: tuple[ExportWarning, ...] = ()


class EpubWriter(Protocol):
    """Assembles final EPUB from canonical model and processing results."""

    def write(self, request: EpubWriteRequest) -> ExportedBook:
        """Assembles the book and runs built-in technical checks (PRD § 20).

        The implementation must raise `EpubExportFatalError` if the book cannot be assembled
        or fails built-in checks; anything that does not prevent the book from being valid
        is collected in `ExportedBook.warnings`.
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
