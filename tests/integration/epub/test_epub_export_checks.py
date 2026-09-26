"""Экспорт EPUB: встроенные технические проверки собранного файла (P2.T5).

Проверки натравливаются на намеренно испорченные книги: собственная сборка экспортёра
корректна по построению, и без битых файлов эти проверки никогда не показали бы, что они
вообще срабатывают.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document

from linguary.adapters.epub import ExportCheckKind, verify_exported_epub

#: Минимальная корректная книга: две главы, оглавление и одна внутренняя ссылка.
_DOCUMENTS: Mapping[str, str] = {
    "chapter1.xhtml": xhtml_document(
        '<h1 id="start">Chapter One</h1><p>The road ran east.</p>'
        '<p><a href="chapter2.xhtml#second">Onward</a></p>'
        '<p><img src="images/marker.png" alt="marker"/></p>'
    ),
    "chapter2.xhtml": xhtml_document(
        '<h1 id="second">Chapter Two</h1><p>The town appeared at dusk.</p>'
    ),
}

_TOC = (("chapter1.xhtml", "Chapter One"), ("chapter2.xhtml", "Chapter Two"))

#: Крошечный, но настоящий PNG: проверкам важно наличие файла, а не его содержимое.
_IMAGE = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6360000002000100ffff03000006"
    "00057fa8a2b40000000049454e44ae426082"
)

_SPINE = ("chapter1.xhtml", "chapter2.xhtml")


def build_book(path: Path, **overrides: object) -> Path:
    """Собирает книгу набора с точечной поломкой из `overrides`."""
    arguments: dict[str, object] = {
        "documents": dict(_DOCUMENTS),
        "toc": list(_TOC),
        "extra_files": {"images/marker.png": _IMAGE, "styles.css": b"p { margin: 0; }"},
    }
    arguments.update(overrides)
    return build_epub(path, **arguments)  # type: ignore[arg-type]


def kinds(path: Path, *, expected_spine: Sequence[str] = _SPINE) -> set[ExportCheckKind]:
    """Виды несошедшихся проверок для книги."""
    return {
        failure.kind
        for failure in verify_exported_epub(path, expected_spine=expected_spine)
    }


def test_correct_book_passes_every_built_in_check(tmp_path: Path) -> None:
    """Корректная книга не даёт ни одной несошедшейся проверки."""
    book = build_book(tmp_path / "correct.epub")

    assert verify_exported_epub(book, expected_spine=_SPINE) == ()


def test_broken_manifest_is_detected(tmp_path: Path) -> None:
    """Глава есть в порядке чтения, но не объявлена в манифесте."""
    book = build_book(
        tmp_path / "manifest.epub", omit_from_manifest=["chapter2.xhtml"]
    )

    assert ExportCheckKind.SPINE in kinds(book)


def test_lost_resource_is_detected(tmp_path: Path) -> None:
    """Ресурс объявлен в манифесте, но потерян в архиве."""
    book = build_book(tmp_path / "resource.epub", omit_from_archive=["styles.css"])

    assert ExportCheckKind.RESOURCE in kinds(book)


def test_image_without_its_file_is_detected(tmp_path: Path) -> None:
    """Глава ссылается на изображение, которого в книге нет вовсе."""
    documents = dict(_DOCUMENTS)
    documents["chapter2.xhtml"] = xhtml_document(
        '<h1 id="second">Chapter Two</h1><p><img src="images/absent.png" alt="lost"/></p>'
    )
    book = build_book(tmp_path / "image.epub", documents=documents)

    assert ExportCheckKind.RESOURCE in kinds(book)


def test_unresolvable_link_is_detected(tmp_path: Path) -> None:
    """Внутренняя ссылка ведёт на несуществующий якорь."""
    documents = dict(_DOCUMENTS)
    documents["chapter1.xhtml"] = xhtml_document(
        '<h1 id="start">Chapter One</h1><p><a href="chapter2.xhtml#nowhere">Onward</a></p>'
    )
    book = build_book(tmp_path / "link.epub", documents=documents)

    assert ExportCheckKind.LINK in kinds(book)


def test_link_to_a_missing_document_is_detected(tmp_path: Path) -> None:
    """Внутренняя ссылка ведёт на документ, которого в книге нет."""
    documents = dict(_DOCUMENTS)
    documents["chapter1.xhtml"] = xhtml_document(
        '<h1 id="start">Chapter One</h1><p><a href="chapter9.xhtml">Onward</a></p>'
    )
    book = build_book(tmp_path / "missing-document.epub", documents=documents)

    assert ExportCheckKind.LINK in kinds(book)


def test_broken_xhtml_is_detected(tmp_path: Path) -> None:
    """Документ книги не разбирается как XHTML."""
    documents = dict(_DOCUMENTS)
    documents["chapter2.xhtml"] = '<html><body><p id="second">no closing tag</body>'
    book = build_book(tmp_path / "xhtml.epub", documents=documents)

    assert ExportCheckKind.XHTML in kinds(book)


def test_duplicate_identifier_is_detected(tmp_path: Path) -> None:
    """Один и тот же идентификатор объявлен в документе дважды."""
    documents = dict(_DOCUMENTS)
    documents["chapter2.xhtml"] = xhtml_document(
        '<h1 id="second">Chapter Two</h1><p id="second">The town appeared.</p>'
    )
    book = build_book(tmp_path / "duplicate.epub", documents=documents)

    assert ExportCheckKind.XHTML in kinds(book)


def test_missing_container_is_detected(tmp_path: Path) -> None:
    """В архиве нет `META-INF/container.xml`."""
    book = build_book(tmp_path / "container.epub", include_container=False)

    assert kinds(book) == {ExportCheckKind.CONTAINER}


def test_missing_navigation_is_detected(tmp_path: Path) -> None:
    """В книге не объявлено оглавление."""
    documents = dict(_DOCUMENTS)
    documents["chapter1.xhtml"] = xhtml_document(
        '<h1 id="start">Chapter One</h1><p>The road ran east.</p>'
        '<p><a href="chapter2.xhtml#second">Onward</a></p>'
    )
    book = build_book(
        tmp_path / "navigation.epub", documents=documents, toc=None, extra_files={}
    )

    assert ExportCheckKind.NAVIGATION in kinds(book)


def test_changed_chapter_order_is_detected(tmp_path: Path) -> None:
    """Порядок глав в книге отличается от ожидаемого."""
    book = build_book(tmp_path / "order.epub", spine=list(reversed(_SPINE)))

    assert ExportCheckKind.CHAPTER_ORDER in kinds(book)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"not an epub at all", id="not-an-archive"),
        pytest.param(b"", id="empty-file"),
    ],
)
def test_file_that_is_not_an_archive_is_detected(
    tmp_path: Path, payload: bytes
) -> None:
    """Файл, который вообще не является архивом EPUB."""
    book = tmp_path / "broken.epub"
    book.write_bytes(payload)

    assert kinds(book) == {ExportCheckKind.CONTAINER}
