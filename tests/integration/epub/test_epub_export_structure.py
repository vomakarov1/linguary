"""Экспорт EPUB: цикл `import -> export` без LLM на наборе книг (P2.T5)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from conftest import FIXTURE_DIRECTORY
from epub_builder import build_epub, xhtml_document
from export_pipeline import documents_of, elements, package_of, run_export, spine_hrefs

from linguary.adapters.epub import EbookLibEpubReader, verify_exported_epub
from linguary.application.ports.epub import EpubReadRequest, ExportWarningKind
from linguary.domain import BlockType

#: Набор тестовых EPUB (PRD § 28.1): каждая книга обязана пережить экспорт.
FIXTURES = (
    "synthetic_story",
    "beowulf",
    "great_gatsby",
    "dracula",
    "alice_rackham",
)


def anchors_of(root) -> set[str]:
    """Идентификаторы, объявленные в документе."""
    return {
        identifier
        for element in root.iter()
        if (identifier := element.get("id")) is not None
    }


@pytest.mark.parametrize("name", FIXTURES)
def test_structure_only_export_passes_built_in_checks(
    name: str, tmp_path: Path
) -> None:
    """Каждая книга набора собирается заново и проходит встроенные проверки."""
    run = run_export(FIXTURE_DIRECTORY / f"{name}.epub", tmp_path=tmp_path)

    assert run.path.is_file()
    assert verify_exported_epub(run.path, expected_spine=spine_hrefs(run.path)) == ()


@pytest.mark.parametrize("name", FIXTURES)
def test_exported_book_imports_back_with_the_same_chapters(
    name: str, tmp_path: Path
) -> None:
    """Собранная книга снова импортируется и сохраняет главы в том же порядке."""
    run = run_export(FIXTURE_DIRECTORY / f"{name}.epub", tmp_path=tmp_path)

    again = EbookLibEpubReader().read(
        EpubReadRequest(
            book_id="reimported", source_path=str(run.path), target_language="ru"
        )
    )

    assert [section.title for section in again.document.ordered_sections] == [
        section.title for section in run.imported.document.ordered_sections
    ]
    assert any(block.text for block in again.document.blocks())


@pytest.mark.parametrize("name", FIXTURES)
def test_chapter_order_matches_the_source_book(name: str, tmp_path: Path) -> None:
    """Порядок глав в собранной книге совпадает с порядком чтения исходной."""
    run = run_export(FIXTURE_DIRECTORY / f"{name}.epub", tmp_path=tmp_path)

    assert spine_hrefs(run.path) == tuple(
        section.source_resource_path
        for section in run.imported.document.ordered_sections
    )


def test_images_survive_export_including_those_inside_tables(tmp_path: Path) -> None:
    """Изображения переносятся все, включая лежащие внутри таблиц.

    Импортёр разбирает таблицу единым блоком `PRESERVE_OPAQUE` и внутрь неё не спускается,
    поэтому отдельного блока `Image` у части иллюстраций нет. Экспортёр переносит такой
    блок исходной разметкой, и книга не теряет ни одной иллюстрации.
    """
    run = run_export(FIXTURE_DIRECTORY / "alice_rackham.epub", tmp_path=tmp_path)

    image_blocks = [
        block for block in run.imported.document.blocks() if block.type is BlockType.IMAGE
    ]
    exported = [
        element
        for root in documents_of(run.path).values()
        for element in elements(root, "img")
    ]
    with zipfile.ZipFile(run.path) as archive:
        names = set(archive.namelist())

    assert len(image_blocks) == 28
    assert len(exported) == 41
    assert all(
        f"OEBPS/{element.get('src')}" in names for element in exported
    )


def test_internal_links_and_footnotes_resolve_after_export(tmp_path: Path) -> None:
    """Внутренние ссылки и ссылки на сноски в собранной книге разрешимы."""
    run = run_export(FIXTURE_DIRECTORY / "beowulf.epub", tmp_path=tmp_path)
    documents = documents_of(run.path)
    anchors = {name: anchors_of(root) for name, root in documents.items()}

    links = [
        (name, href)
        for name, root in documents.items()
        for element in elements(root, "a")
        if (href := element.get("href", "")).startswith("#")
        or (href and "#" in href and not href.startswith("http"))
    ]
    unresolved = [
        (name, href)
        for name, href in links
        for target in (_resolve(name, href),)
        if target[1] not in anchors.get(target[0], set())
    ]

    assert len(links) > 100
    assert unresolved == []


def test_metadata_of_the_book_is_preserved(tmp_path: Path) -> None:
    """Название, автор и язык книги переносятся в итоговый файл."""
    run = run_export(FIXTURE_DIRECTORY / "great_gatsby.epub", tmp_path=tmp_path)
    metadata = {
        element.tag.rpartition("}")[2]: element.text
        for element in package_of(run.path).iter()
        if element.tag.rpartition("}")[2]
        in ("title", "creator", "language", "identifier")
    }
    source = run.imported.document.metadata

    assert metadata["title"] == source.title
    assert metadata["creator"] == source.authors[0]
    assert metadata["language"] == run.imported.document.source_language
    assert metadata["identifier"] == source.identifier


def test_export_without_the_source_file_keeps_the_book_valid(tmp_path: Path) -> None:
    """Пропавший исходный EPUB экспорт не ломает: книга собирается с предупреждением.

    Предупреждения сводятся по виду и главе: в книге с иллюстрациями Gutenberg
    конструкций `PRESERVE_OPAQUE` десятки, и строка на каждую превратила бы список
    замечаний в шум.
    """
    run = run_export(
        FIXTURE_DIRECTORY / "alice_rackham.epub", tmp_path=tmp_path, keep_source=False
    )
    warnings = run.exported.warnings

    assert verify_exported_epub(run.path, expected_spine=spine_hrefs(run.path)) == ()
    assert {warning.kind for warning in warnings} == {
        ExportWarningKind.OPAQUE_FRAGMENT_LOST
    }
    assert len(warnings) <= len(run.imported.document.sections)
    assert any("Всего таких случаев" in warning.message for warning in warnings)


def test_source_without_package_file_does_not_stop_export(tmp_path: Path) -> None:
    """Исходный EPUB, потерявший объявление пакета после импорта, экспорт не обрывает.

    Разметка `PRESERVE_OPAQUE` из такого архива не достаётся, и книга собирается так же,
    как без исходного файла: блок переносится текстом с предупреждением.
    """
    source = build_epub(
        tmp_path / "book.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<p>Text.</p><table><tr><td>Cell</td></tr></table>"
            )
        },
    )
    replaced = build_epub(
        tmp_path / "replaced.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        include_container=False,
        archive_files={
            "META-INF/container.xml": (
                b'<?xml version="1.0" encoding="UTF-8"?>\n'
                b'<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                b'version="1.0"><rootfiles/></container>\n'
            )
        },
    )

    run = run_export(source, tmp_path=tmp_path, export_source=replaced)

    assert verify_exported_epub(run.path, expected_spine=spine_hrefs(run.path)) == ()
    assert {warning.kind for warning in run.exported.warnings} == {
        ExportWarningKind.OPAQUE_FRAGMENT_LOST
    }


def test_htm_chapter_is_declared_as_xhtml(tmp_path: Path) -> None:
    """Глава с расширением `.htm` объявляется в манифесте как XHTML-документ."""
    source = build_epub(
        tmp_path / "htm_chapter.epub",
        documents={"chapter01.htm": xhtml_document("<p>Text.</p>")},
    )

    run = run_export(source, tmp_path=tmp_path)

    media_types = {
        item.get("href"): item.get("media-type")
        for item in package_of(run.path).iter("{http://www.idpf.org/2007/opf}item")
    }
    assert media_types["chapter01.htm"] == "application/xhtml+xml"
    assert verify_exported_epub(run.path, expected_spine=spine_hrefs(run.path)) == ()


def _resolve(name: str, href: str) -> tuple[str, str]:
    """Ссылка как пара «документ архива, якорь»."""
    path, _, fragment = href.partition("#")
    if not path:
        return name, fragment
    return f"OEBPS/{path}", fragment
