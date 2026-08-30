"""Пределы распаковки архива на импорте: недоверенная книга не занимает машину (PRD § 6).

Файл EPUB — недоверенный вход: он может прийти от третьего лица и объявлять сотни
гигабайт содержимого при небольшом размере на диске. Тест закрепляет, что пределы
`ExtractionLimits` применяются до первого чтения записей архива, что отдельный документ
порядка чтения ограничен своим пределом и что обычная книга по-прежнему импортируется.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document

from linguary.adapters.epub import EbookLibEpubReader, ExtractionLimits
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubReadRequest,
    ImportedBook,
)

ReadFixture = Callable[..., ImportedBook]

#: Сколько байт объявляет запись-бомба: на диске она занимает единицы килобайт.
BOMB_BYTES = 32 * 1024 * 1024


def test_archive_declaring_more_than_the_limit_is_rejected(tmp_path: Path) -> None:
    """Архив, заявивший объём больше допустимого, отвергается сообщением о пределе."""
    source = build_epub(
        tmp_path / "big.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
    )

    with pytest.raises(EpubImportFatalError, match="заявлено"):
        _read(source, ExtractionLimits(max_total_bytes=512))


def test_archive_with_too_many_entries_is_rejected(tmp_path: Path) -> None:
    """Архив с числом файлов больше допустимого отвергается до чтения записей."""
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
    """Маленький архив, распаковывающийся в десятки мегабайт, до чтения не доходит.

    Отказ приходит от проверки объявленного объёма, а она идёт по оглавлению архива:
    ни одна запись при этом не читается, и память машины не занимается.
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


def test_spine_document_over_its_limit_is_rejected(tmp_path: Path) -> None:
    """Документ порядка чтения крупнее своего предела отвергается отдельно."""
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
    """Книга в пределах распаковки импортируется как раньше."""
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
    """Значения пределов по умолчанию не задевают книги тестового набора (PRD § 28.1)."""
    document = read_fixture("synthetic_story").document

    assert len(document.ordered_sections) == 3


def _read(source: Path, limits: ExtractionLimits) -> ImportedBook:
    """Читает книгу читателем с заданными пределами распаковки."""
    return EbookLibEpubReader(limits=limits).read(
        EpubReadRequest(
            book_id="book-1", source_path=str(source), target_language="ru"
        )
    )
