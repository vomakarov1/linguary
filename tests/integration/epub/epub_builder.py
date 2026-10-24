"""Building small EPUB files directly in tests.

The fixture suite in `tests/fixtures/epub` is intentionally untouched: `FATAL` cases
and targeted markup checks require broken or extremely narrow books that are more
convenient to build in a test's temporary directory than to store in the repository.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Package directory inside the archive: manifest and documents are relative to it.
PACKAGE_DIRECTORY = "OEBPS"

_CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_MEDIA_TYPES = {
    ".xhtml": "application/xhtml+xml",
    ".htm": "application/xhtml+xml",
    ".css": "text/css",
    ".js": "application/javascript",
    ".png": "image/png",
    ".jpg": "image/jpeg",
}


def build_epub(
    path: Path,
    *,
    documents: Mapping[str, str],
    spine: Sequence[str] | None = None,
    toc: Sequence[tuple[str, str]] | None = None,
    title: str = "Test Book",
    language: str = "en",
    extra_files: Mapping[str, bytes] | None = None,
    archive_files: Mapping[str, bytes] | None = None,
    include_container: bool = True,
    include_package: bool = True,
    omit_from_archive: Sequence[str] = (),
    omit_from_manifest: Sequence[str] = (),
    compression: int = zipfile.ZIP_STORED,
) -> Path:
    """Builds an EPUB3 from prepared XHTML documents and returns its path.

    `omit_from_archive` declares a file in manifest but does not place it in the archive:
    creates a book with a missing spine document or missing resource.
    `omit_from_manifest`, inversely, places the file in archive and keeps it in spine,
    but omits manifest declaration: produces a book with a broken manifest.
    `archive_files` places files in archive at given path without package directory prefix
    or manifest entry: creates an out-of-package resource.
    `compression` compresses archive entries: creates a book that is small on disk
    but huge when decompressed.
    """
    order = list(spine) if spine is not None else list(documents)
    with zipfile.ZipFile(path, "w", compression) as archive:
        archive.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        if include_container:
            archive.writestr("META-INF/container.xml", _CONTAINER_XML)
        if include_package:
            archive.writestr(
                f"{PACKAGE_DIRECTORY}/content.opf",
                _package(
                    documents=documents,
                    order=order,
                    toc=toc,
                    title=title,
                    language=language,
                    extra_files=extra_files or {},
                    omit_from_manifest=omit_from_manifest,
                ),
            )
        if toc is not None:
            archive.writestr(f"{PACKAGE_DIRECTORY}/nav.xhtml", _navigation(toc))
        for href, content in documents.items():
            if href in omit_from_archive:
                continue
            archive.writestr(f"{PACKAGE_DIRECTORY}/{href}", content)
        for href, payload in (extra_files or {}).items():
            if href in omit_from_archive:
                continue
            archive.writestr(f"{PACKAGE_DIRECTORY}/{href}", payload)
        for name, payload in (archive_files or {}).items():
            archive.writestr(name, payload)
    return path


def xhtml_document(body: str, *, title: str = "Chapter") -> str:
    """Minimal XHTML document with the given body."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops" lang="en">\n'
        f"<head><title>{title}</title></head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )


def _package(
    *,
    documents: Mapping[str, str],
    order: Sequence[str],
    toc: Sequence[tuple[str, str]] | None,
    title: str,
    language: str,
    extra_files: Mapping[str, bytes],
    omit_from_manifest: Sequence[str] = (),
) -> str:
    """OPF package file with manifest and spine."""
    items = [
        f'<item id="{_item_id(href)}" href="{href}" '
        f'media-type="{_media_type(href)}"/>'
        for href in (*documents, *extra_files)
        if href not in omit_from_manifest
    ]
    if toc is not None:
        items.insert(
            0,
            '<item id="nav" href="nav.xhtml" '
            'media-type="application/xhtml+xml" properties="nav"/>',
        )
    itemrefs = [f'<itemref idref="{_item_id(href)}"/>' for href in order]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="book-id">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="book-id">urn:uuid:test-book</dc:identifier>\n'
        f"    <dc:title>{title}</dc:title>\n"
        f"    <dc:language>{language}</dc:language>\n"
        "  </metadata>\n"
        f"  <manifest>{''.join(items)}</manifest>\n"
        f"  <spine>{''.join(itemrefs)}</spine>\n"
        "</package>\n"
    )


def _navigation(toc: Sequence[tuple[str, str]]) -> str:
    """EPUB3 navigation document with the listed TOC entries."""
    entries = "".join(f'<li><a href="{href}">{title}</a></li>' for href, title in toc)
    return xhtml_document(
        f'<nav epub:type="toc" id="toc"><ol>{entries}</ol></nav>', title="Navigation"
    )


def _item_id(href: str) -> str:
    """Manifest item ID derived from its path."""
    return href.replace("/", "-").replace(".", "-")


def _media_type(href: str) -> str:
    """MIME content type derived from file extension."""
    return _MEDIA_TYPES.get(Path(href).suffix, "application/octet-stream")
    return _MEDIA_TYPES.get(Path(href).suffix, "application/octet-stream")
