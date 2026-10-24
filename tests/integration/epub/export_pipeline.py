"""Pass `import -> export` and archive parsing helper for export tests.

Export requires the entire import input: canonical model, stored resource copies,
and the source archive itself (used to carry over markup for `PRESERVE_OPAQUE` structures).
Reassembling this pipeline in every single test is redundant, so it lives here.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from linguary.adapters.epub import (
    EbookLibEpubReader,
    EpubExportWriter,
    FilesystemEpubResourceStore,
)
from linguary.application.ports.epub import (
    EpubReadRequest,
    EpubWriteRequest,
    ExportedBook,
    ImportedBook,
    StoreResourcesRequest,
)
from linguary.domain import BlockProcessingResult, LearningMaterial

_XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"


@dataclass(frozen=True, slots=True)
class ExportRun:
    """Pipeline execution result: what was imported and what was built."""

    imported: ImportedBook
    exported: ExportedBook

    @property
    def path(self) -> Path:
        """Path to the built book file."""
        return Path(self.exported.path)


def run_export(
    source: Path,
    *,
    tmp_path: Path,
    book_id: str = "book-1",
    block_results: Mapping[str, BlockProcessingResult] | None = None,
    learning_materials: Mapping[str, LearningMaterial] | None = None,
    keep_source: bool = True,
    export_source: Path | None = None,
) -> ExportRun:
    """Imports a book and rebuilds it.

    Without `block_results` and `learning_materials` it runs in `structure_only` mode:
    the book is rebuilt without processing results or learning materials.
    `keep_source = False` hides the source archive from the exporter — testing a book
    whose source file is no longer on the machine. `export_source` supplies a different
    archive to the exporter — testing a book substituted between import and export.
    """
    imported = EbookLibEpubReader().read(
        EpubReadRequest(
            book_id=book_id, source_path=str(source), target_language="ru"
        )
    )
    stored = FilesystemEpubResourceStore(root=tmp_path / "storage").store(
        StoreResourcesRequest(
            book_id=book_id, source_path=str(source), document=imported.document
        )
    )
    exported = EpubExportWriter().write(
        EpubWriteRequest(
            document=imported.document,
            target_path=str(tmp_path / f"{book_id}-export.epub"),
            stored_resource_paths=stored.paths,
            source_epub_path=str(export_source or source) if keep_source else None,
            block_results=block_results or {},
            learning_materials=learning_materials or {},
        )
    )
    return ExportRun(imported=imported, exported=exported)


#: Document extensions: Gutenberg releases store chapters in both `.html` and `.htm`.
_DOCUMENT_SUFFIXES = (".xhtml", ".html", ".htm")


def documents_of(path: Path) -> Mapping[str, etree._Element]:
    """Parsed XHTML documents from the archive indexed by their internal names."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return {
            name: etree.fromstring(archive.read(name), parser=parser)
            for name in archive.namelist()
            if name.endswith(_DOCUMENT_SUFFIXES)
        }


def elements(root: etree._Element, tag: str) -> tuple[etree._Element, ...]:
    """All elements in the document matching the given tag name."""
    return tuple(root.iter(f"{{{_XHTML_NAMESPACE}}}{tag}"))


def package_of(path: Path) -> etree._Element:
    """Parsed package file of the built book."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("OEBPS/content.opf"), parser=parser)


def spine_hrefs(path: Path) -> tuple[str, ...]:
    """Spine documents of the built book in spine order."""
    package = package_of(path)
    manifest = {
        element.get("id"): element.get("href")
        for element in package.iter()
        if etree.QName(element).localname == "item"
    }
    return tuple(
        manifest[element.get("idref")] or ""
        for element in package.iter()
        if etree.QName(element).localname == "itemref"
    )
