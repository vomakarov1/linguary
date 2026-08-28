"""Сборка небольших EPUB прямо в тесте.

Набор `tests/fixtures/epub` намеренно не трогается: случаи `FATAL` и точечные проверки
разметки требуют испорченных или предельно узких книг, которые удобнее собрать во временной
директории теста, чем держать в репозитории.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Каталог пакета внутри архива: манифест и документы лежат относительно него.
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
) -> Path:
    """Собирает EPUB3 из готовых XHTML-документов и отдаёт путь к нему.

    `omit_from_archive` объявляет файл в манифесте, но не кладёт его в архив: так
    получается книга с потерянным документом порядка чтения или потерянным ресурсом.
    `archive_files` кладёт файл в архив по указанному пути, не добавляя каталог пакета
    и не объявляя его в манифесте: так получается ресурс, лежащий вне каталога книги.
    """
    order = list(spine) if spine is not None else list(documents)
    with zipfile.ZipFile(path, "w") as archive:
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
    """Минимальный XHTML-документ с заданным телом."""
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
) -> str:
    """Файл пакета OPF с манифестом и порядком чтения."""
    items = [
        f'<item id="{_item_id(href)}" href="{href}" '
        f'media-type="{_media_type(href)}"/>'
        for href in (*documents, *extra_files)
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
    """Документ навигации EPUB3 с перечисленными пунктами оглавления."""
    entries = "".join(f'<li><a href="{href}">{title}</a></li>' for href, title in toc)
    return xhtml_document(
        f'<nav epub:type="toc" id="toc"><ol>{entries}</ol></nav>', title="Navigation"
    )


def _item_id(href: str) -> str:
    """Идентификатор элемента манифеста, выведенный из его пути."""
    return href.replace("/", "-").replace(".", "-")


def _media_type(href: str) -> str:
    """Тип содержимого по расширению файла."""
    return _MEDIA_TYPES.get(Path(href).suffix, "application/octet-stream")
