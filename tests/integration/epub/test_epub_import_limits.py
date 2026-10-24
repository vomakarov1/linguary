"""Archive extraction limits during import: untrusted books do not overwhelm the host (PRD § 6).

An EPUB file is untrusted input: it may originate from a third party and declare hundreds
of gigabytes of uncompressed content while remaining tiny on disk. This test verifies
that `ExtractionLimits` are enforced before entries are read, that individual reading-order
documents are capped by their respective limit, and that regular books continue to import.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from epub_builder import build_epub, xhtml_document

from linguary.adapters.epub import EbookLibEpubReader, ExtractionLimits
from linguary.adapters.epub.reader import _ArchiveContents
from linguary.adapters.epub.resources import _ResourceArchive
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubReadRequest,
    ImportedBook,
)

ReadFixture = Callable[..., ImportedBook]

#: Declared byte size of the compression bomb: takes only a few kilobytes on disk.
BOMB_BYTES = 32 * 1024 * 1024

#: Container without OPF package file declared.
CONTAINER_WITHOUT_ROOTFILE = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
    b'version="1.0"><rootfiles/></container>\n'
)


def test_archive_declaring_more_than_the_limit_is_rejected(tmp_path: Path) -> None:
    """An archive declaring volume exceeding the limit is rejected with a limit message."""
    source = build_epub(
        tmp_path / "big.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
    )

    with pytest.raises(EpubImportFatalError, match="заявлено"):
        _read(source, ExtractionLimits(max_total_bytes=512))


def test_archive_with_too_many_entries_is_rejected(tmp_path: Path) -> None:
    """An archive with more entries than allowed is rejected before reading entries."""
    source = build_epub(
        tmp_path / "many.epub",
        documents={
            "chapter1.xhtml": xhtml_document("<p>First.</p>"),
            "chapter2.xhtml": xhtml_document("<p>Second.</p>"),
        },
    )

    with pytest.raises(EpubImportFatalError, match="файлов"):
        _read(source, ExtractionLimits(max_entries=2))


def test_compressed_bomb_is_rejected_before_it_is_read(tmp_path: Path) -> None:
    """A small archive expanding into tens of megabytes is rejected prior to reading.

    The rejection triggers during declared volume validation over the zip central directory:
    no entries are decompressed, sparing system memory.
    """
    source = build_epub(
        tmp_path / "bomb.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        extra_files={"bomb.bin": b"\0" * BOMB_BYTES},
        compression=zipfile.ZIP_DEFLATED,
    )

    assert source.stat().st_size < BOMB_BYTES // 100

    with pytest.raises(EpubImportFatalError, match="заявлено"):
        _read(source, ExtractionLimits(max_total_bytes=1024 * 1024))


def test_limit_rejection_names_sizes_and_hides_the_file_path(tmp_path: Path) -> None:
    """Limit rejection message is user-friendly and hides internal filesystem paths (QA-02).

    The message is shown in the book card while the file lives in application storage:
    instead of paths and byte counts, the user sees human-readable MB quantities.
    """
    source = build_epub(
        tmp_path / "bomb.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        extra_files={"bomb.bin": b"\0" * BOMB_BYTES},
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(EpubImportFatalError) as error:
        _read(source, ExtractionLimits(max_total_bytes=1024 * 1024))

    message = str(error.value)
    assert message.startswith("Распакованный объём книги 32 МБ")
    assert message.endswith("превышает допустимые 1 МБ")
    assert str(tmp_path) not in message


def test_missing_file_rejection_hides_the_file_path(tmp_path: Path) -> None:
    """Filesystem error states the cause without exposing internal data directories (QA-02)."""
    missing = tmp_path / "books" / "book-1" / "source.epub"

    with pytest.raises(EpubImportFatalError) as error:
        _read(missing, ExtractionLimits())

    assert str(error.value).startswith("EPUB нельзя прочитать")
    assert str(tmp_path) not in str(error.value)


def test_spine_document_over_its_limit_is_rejected(tmp_path: Path) -> None:
    """A reading-order document exceeding its size limit is rejected separately."""
    source = build_epub(
        tmp_path / "long_chapter.epub",
        documents={"chapter1.xhtml": xhtml_document(f"<p>{'a' * 200_000}</p>")},
    )

    with pytest.raises(EpubImportFatalError, match="заявлен как"):
        _read(
            source,
            ExtractionLimits(
                max_document_bytes=1024, max_total_bytes=8 * 1024 * 1024
            ),
        )


def test_book_within_the_limits_is_imported(tmp_path: Path) -> None:
    """A book within extraction limits imports as expected."""
    source = build_epub(
        tmp_path / "normal.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
    )

    document = _read(source, ExtractionLimits(max_total_bytes=1024 * 1024)).document

    assert [section.source_resource_path for section in document.ordered_sections] == [
        "chapter1.xhtml"
    ]


def test_book_of_the_fixture_set_is_imported_with_default_limits(
    read_fixture: ReadFixture,
) -> None:
    """Default limits do not affect valid fixture books (PRD § 28.1)."""
    document = read_fixture("synthetic_story").document

    assert len(document.ordered_sections) == 3


@pytest.mark.parametrize(
    "archive_class",
    [
        pytest.param(_ArchiveContents, id="reading-order"),
        pytest.param(_ResourceArchive, id="resources"),
    ],
)
@pytest.mark.parametrize(
    ("build_options", "limits"),
    [
        pytest.param(
            {"include_container": False}, ExtractionLimits(), id="no-container"
        ),
        pytest.param(
            {
                "include_container": False,
                "archive_files": {"META-INF/container.xml": b"<container"},
            },
            ExtractionLimits(),
            id="broken-container",
        ),
        pytest.param(
            {
                "include_container": False,
                "archive_files": {
                    "META-INF/container.xml": CONTAINER_WITHOUT_ROOTFILE
                },
            },
            ExtractionLimits(),
            id="no-rootfile",
        ),
        pytest.param({}, ExtractionLimits(max_entries=1), id="over-limit"),
    ],
)
def test_rejected_archive_is_closed(
    archive_class: Callable[..., Any],
    build_options: Mapping[str, Any],
    limits: ExtractionLimits,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An archive rejected on open is closed immediately without waiting for GC."""
    source = build_epub(
        tmp_path / "rejected.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        **build_options,
    )
    opened: list[zipfile.ZipFile] = []
    open_archive = zipfile.ZipFile

    def recording_zip_file(*args: Any, **kwargs: Any) -> zipfile.ZipFile:
        archive = open_archive(*args, **kwargs)
        opened.append(archive)
        return archive

    monkeypatch.setattr(zipfile, "ZipFile", recording_zip_file)

    with pytest.raises(EpubImportFatalError):
        archive_class(str(source), limits=limits)

    assert len(opened) == 1
    assert opened[0].fp is None


def _read(source: Path, limits: ExtractionLimits) -> ImportedBook:
    """Reads a book using a reader configured with the specified limits."""
    return EbookLibEpubReader(limits=limits).read(
        EpubReadRequest(
            book_id="book-1", source_path=str(source), target_language="ru"
        )
    )
