"""Persisting book resources to local storage (PRD § 22).

Implementation of port `linguary.application.ports.epub.EpubResourceStore`. The canonical model
identifies a resource by its path inside the EPUB, whereas the exporter requires an actual file,
so import extracts everything transferred to the resulting book from the archive and stores it
under the book directory.

The archive here is untrusted input. The resource path from the manifest and the archive entry name
may point anywhere, so both are normalized and writing is allowed only within the book directory.
Extracted content size and file count are bounded: exceeding limits is `FATAL`, the book is not
imported, and no copies of its resources remain in storage.
"""

from __future__ import annotations

import posixpath
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

from linguary.adapters.epub.inline import local_name
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubResourceStore,
    StoredResources,
    StoreResourcesRequest,
)
from linguary.domain import CanonicalDocument, SupportStatus

#: Support statuses where the resource is transferred to the resulting book and therefore saved.
_STORED_STATUSES = frozenset({SupportStatus.SUPPORTED, SupportStatus.PRESERVE_OPAQUE})

#: Book subdirectory containing resource copies.
_RESOURCES_DIRECTORY = "resources"


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Limits for unpacking a book archive.

    Defaults are sized for an illustrated fiction book and are well above any book
    in the sample set; they exist not to save disk space, but to prevent an archive
    expanding into tens of gigabytes from occupying the user's machine.

    `max_document_bytes` bounds a single archive entry that import loads into memory:
    a fiction chapter is well under tens of megabytes, and there is no reason to read
    an entry of such size even within the overall budget.
    """

    max_entries: int = 10_000
    max_total_bytes: int = 256 * 1024 * 1024
    max_document_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            self.max_entries <= 0
            or self.max_total_bytes <= 0
            or self.max_document_bytes <= 0
        ):
            raise ValueError("лимиты распаковки должны быть положительными")


class FilesystemEpubResourceStore(EpubResourceStore):
    """Storage for book resources in a local filesystem directory.

    Book resources reside in `<storage_root>/<book_id>/resources/<path_inside_epub>`.
    Reimporting a book starts the directory afresh: resource copies are derived from the archive,
    and there is no reason to keep files from previous imports in storage.
    """

    def __init__(self, *, root: Path, limits: ExtractionLimits | None = None) -> None:
        self._root = root
        self._limits = limits or ExtractionLimits()

    def store(self, request: StoreResourcesRequest) -> StoredResources:
        """Stores book resources; support matrix violation results in `FATAL`."""
        book_directory = self._book_directory(request.book_id)
        with _ResourceArchive(request.source_path, limits=self._limits) as archive:
            planned = _plan(
                document=request.document,
                archive=archive,
                book_directory=book_directory,
                book_id=request.book_id,
            )
            _reset_directory(book_directory)
            paths = archive.extract(planned, cleanup=book_directory)
        return StoredResources(paths=paths)

    def _book_directory(self, book_id: str) -> Path:
        """Book resource directory inside storage root."""
        directory = _contained_path(
            self._root, book_id, message=f"идентификатор книги {book_id!r}"
        )
        return directory / _RESOURCES_DIRECTORY


@dataclass(frozen=True, slots=True)
class _PlannedResource:
    """Resource, located archive entry, and the destination path where it will be saved."""

    source_path: str
    archive_name: str
    target: Path


class _ResourceArchive:
    """Book archive: looks up entries by resource path and extracts them into storage."""

    def __init__(self, source_path: str, *, limits: ExtractionLimits) -> None:
        self._limits = limits
        try:
            self._archive = zipfile.ZipFile(source_path)
        except (OSError, zipfile.BadZipFile) as error:
            raise EpubImportFatalError(unreadable_epub_message(error)) from error
        try:
            self._check_limits()
            self._directory = posixpath.dirname(package_path(self._archive))
        except (OSError, KeyError, zipfile.BadZipFile, etree.XMLSyntaxError) as error:
            self._archive.close()
            raise EpubImportFatalError(unreadable_epub_message(error)) from error
        except EpubImportFatalError:
            self._archive.close()
            raise
        self._names = frozenset(
            posixpath.normpath(name) for name in self._archive.namelist()
        )

    def __enter__(self) -> _ResourceArchive:
        return self

    def __exit__(self, *exception: object) -> None:
        self._archive.close()

    def locate(self, resource_path: str) -> str | None:
        """Archive entry name for a book resource, or `None` if absent.

        Resource path originates from the manifest and cannot be trusted: `../` inside it
        navigates outside the package directory or even the archive itself, making such a path `FATAL`.
        """
        name = self._archive_name(resource_path)
        decoded = unquote(name)
        for candidate in (name, decoded) if decoded != name else (name,):
            if candidate in self._names:
                return candidate
        return None

    def extract(
        self, planned: Sequence[_PlannedResource], *, cleanup: Path
    ) -> Mapping[str, str]:
        """Writes planned resources and returns paths to their copies.

        Actual size is accumulated during writing: archive headers declare file sizes,
        but an archive cannot be relied upon to report truthful values. Exceeding limits
        leaves the book unimported, so already written copies are removed.
        """
        paths: dict[str, str] = {}
        written = 0
        for item in planned:
            payload = self._read(item.archive_name)
            written += len(payload)
            if written > self._limits.max_total_bytes:
                shutil.rmtree(cleanup, ignore_errors=True)
                raise EpubImportFatalError(
                    f"Распакованный объём книги превышает допустимые "
                    f"{format_size(self._limits.max_total_bytes)}"
                )
            item.target.parent.mkdir(parents=True, exist_ok=True)
            item.target.write_bytes(payload)
            paths[item.source_path] = str(item.target)
        return paths

    def _archive_name(self, resource_path: str) -> str:
        """Archive entry name for resource path; traversing beyond archive root is `FATAL`."""
        joined = posixpath.join(self._directory, resource_path)
        normalized = posixpath.normpath(joined)
        if normalized.startswith(("..", "/")):
            raise EpubImportFatalError(
                f"ресурс {resource_path} ведёт за пределы архива книги"
            )
        return normalized

    def _read(self, name: str) -> bytes:
        """Archive entry content; an unreadable entry is `FATAL`."""
        try:
            return self._archive.read(name)
        except (KeyError, OSError, zipfile.BadZipFile) as error:
            raise EpubImportFatalError(
                f"ресурс {name} книги нельзя прочитать: {failure_reason(error)}"
            ) from error

    def _check_limits(self) -> None:
        """Validates file count and extracted size declared by the archive."""
        check_extraction_limits(self._archive, limits=self._limits)


def check_extraction_limits(
    archive: zipfile.ZipFile, *, limits: ExtractionLimits
) -> None:
    """Checks archive-declared file count and extracted volume against limits.

    Validation is performed against archive headers before reading any entries:
    an archive expanding into tens of gigabytes must not occupy machine memory.
    Declared sizes cannot be trusted, so the reader also tracks accumulated bytes itself.
    """
    entries = archive.infolist()
    if len(entries) > limits.max_entries:
        raise EpubImportFatalError(
            f"Книга содержит {len(entries)} файлов при допустимых {limits.max_entries}"
        )
    declared = sum(entry.file_size for entry in entries)
    if declared > limits.max_total_bytes:
        raise EpubImportFatalError(
            f"Распакованный объём книги {format_size(declared)} (заявлено в архиве) "
            f"превышает допустимые {format_size(limits.max_total_bytes)}"
        )


def unreadable_epub_message(error: Exception) -> str:
    """User-facing message for an unreadable book archive.

    The message is displayed in the UI, so data directory file paths are omitted:
    filesystem errors report only the reason, while the full path remains in the exception chain.
    """
    return f"EPUB нельзя прочитать: {failure_reason(error)}"


def failure_reason(error: Exception) -> str:
    """Failure reason without the file path.

    `OSError` includes the file name in text, which is an internal data directory path;
    its reason is taken from `strerror`. Other archive errors contain no paths.
    """
    if isinstance(error, OSError):
        return error.strerror or type(error).__name__
    return str(error).strip() or type(error).__name__


def format_size(size: int) -> str:
    """Size in bytes formatted in user-facing units: bytes, KB, or MB."""
    for unit, name in ((1024 * 1024, "МБ"), (1024, "КБ")):
        if size >= unit:
            value = f"{size / unit:.1f}".rstrip("0").rstrip(".").replace(".", ",")
            return f"{value} {name}"
    return f"{size} байт"


def package_path(archive: zipfile.ZipFile) -> str:
    """Path to the OPF package file inside archive via `META-INF/container.xml`.

    The container comes from an untrusted archive and is parsed with the same hardened
    parser as the rest of the book XML: external entities are not resolved, DTD is not
    loaded, and no network access occurs.
    """
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    container = etree.fromstring(archive.read("META-INF/container.xml"), parser=parser)
    for element in container.iter():
        if local_name(element) == "rootfile" and element.get("full-path"):
            return element.get("full-path", "")
    raise EpubImportFatalError("в EPUB не объявлен файл пакета OPF")


def _plan(
    *,
    document: CanonicalDocument,
    archive: _ResourceArchive,
    book_directory: Path,
    book_id: str,
) -> tuple[_PlannedResource, ...]:
    """What and where will be written.

    The plan is constructed entirely before the first write: a path leading outside
    the book directory must halt import before any file appears in storage.

    A manifest resource missing from the archive is also `FATAL` (PRD § 6). Books rarely
    reach this check because EbookLib reads manifest contents on opening and rejects
    unreadable archives earlier, but storage integrity does not depend on which library opened the archive.
    """
    planned: list[_PlannedResource] = []
    for resource in document.resources:
        if resource.support_status not in _STORED_STATUSES:
            continue
        target = _contained_path(
            book_directory,
            resource.source_path,
            message=f"ресурс {resource.source_path} книги {book_id}",
        )
        archive_name = archive.locate(resource.source_path)
        if archive_name is None:
            raise EpubImportFatalError(
                f"ресурс {resource.source_path} объявлен в манифесте книги {book_id}, "
                f"но отсутствует в архиве"
            )
        planned.append(
            _PlannedResource(
                source_path=resource.source_path,
                archive_name=archive_name,
                target=target,
            )
        )
    return tuple(planned)


def _reset_directory(directory: Path) -> None:
    """Clears book resource directory before a new import."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def _contained_path(base: Path, relative: str, *, message: str) -> Path:
    """Path inside `base`; traversing outside is `FATAL` (zip-slip protection)."""
    normalized = relative.replace("\\", "/").strip()
    parts = tuple(part for part in normalized.split("/") if part and part != ".")
    candidate = Path(base, *parts)
    base_resolved = base.resolve()
    if (
        not parts
        or normalized.startswith("/")
        or ".." in parts
        or base_resolved not in candidate.resolve().parents
    ):
        raise EpubImportFatalError(
            f"{message} ведёт за пределы каталога хранилища книги: {relative!r}"
        )
    return candidate


__all__ = [
    "ExtractionLimits",
    "FilesystemEpubResourceStore",
    "check_extraction_limits",
    "failure_reason",
    "format_size",
    "package_path",
    "unreadable_epub_message",
]

