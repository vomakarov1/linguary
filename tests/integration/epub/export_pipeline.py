"""Проход `импорт -> экспорт` и разбор собранного архива для тестов экспорта.

Экспорту нужен вход импорта целиком: каноническая модель, копии ресурсов в хранилище
и сам исходный архив (из него переносится разметка конструкций `PRESERVE_OPAQUE`).
Собирать этот проход в каждом тесте заново незачем, поэтому он живёт здесь.
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
    """Результат прохода: что импортировали и что собрали."""

    imported: ImportedBook
    exported: ExportedBook

    @property
    def path(self) -> Path:
        """Путь к собранному файлу книги."""
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
    """Импортирует книгу и собирает её заново.

    Без `block_results` и `learning_materials` получается режим `structure_only`:
    книга собирается без результатов обработки и без учебных материалов.
    `keep_source = False` прячет исходный архив от экспортёра — так проверяется книга,
    у которой исходного файла на машине уже нет. `export_source` отдаёт экспортёру другой
    архив вместо исходного — так проверяется книга, подменённая между импортом и экспортом.
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


#: Расширения документов книги: издания Gutenberg держат главы и в `.html`, и в `.htm`.
_DOCUMENT_SUFFIXES = (".xhtml", ".html", ".htm")


def documents_of(path: Path) -> Mapping[str, etree._Element]:
    """Разобранные XHTML-документы архива по их именам внутри него."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return {
            name: etree.fromstring(archive.read(name), parser=parser)
            for name in archive.namelist()
            if name.endswith(_DOCUMENT_SUFFIXES)
        }


def elements(root: etree._Element, tag: str) -> tuple[etree._Element, ...]:
    """Все элементы документа с указанным именем тега."""
    return tuple(root.iter(f"{{{_XHTML_NAMESPACE}}}{tag}"))


def package_of(path: Path) -> etree._Element:
    """Разобранный файл пакета собранной книги."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("OEBPS/content.opf"), parser=parser)


def spine_hrefs(path: Path) -> tuple[str, ...]:
    """Документы порядка чтения собранной книги в его порядке."""
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
