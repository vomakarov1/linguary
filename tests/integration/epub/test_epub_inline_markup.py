"""Импорт EPUB: встроенное форматирование, ссылки и сноски (P2.T2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document

from linguary.adapters.epub import EbookLibEpubReader
from linguary.application.ports.epub import EpubReadRequest, ImportedBook
from linguary.domain import (
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    InlineAnnotationType,
)

ReadFixture = Callable[..., ImportedBook]

#: Разметка, которая не имеет смысла без адреса назначения.
_TARGETED = (InlineAnnotationType.LINK, InlineAnnotationType.FOOTNOTE_REFERENCE)


def anchor_index(document: CanonicalDocument) -> dict[str, DocumentBlock]:
    """Отображение «путь ресурса#якорь → блок» для всей книги."""
    index: dict[str, DocumentBlock] = {}
    for block in document.blocks():
        for anchor in block.attributes.get("anchor_ids", "").split():
            index.setdefault(f"{block.source_reference.resource_path}#{anchor}", block)
    return index


def test_block_text_holds_no_markup(read_fixture: ReadFixture) -> None:
    """`DocumentBlock.text` не содержит XHTML: оформление снято в разметку."""
    document = read_fixture("dracula").document

    for block in document.blocks():
        assert "<" not in block.text
        assert "&lt;" not in block.text


@pytest.mark.parametrize(
    ("annotation_type", "expected"),
    [
        pytest.param(InlineAnnotationType.ITALIC, "Wren Calder", id="italic"),
        pytest.param(InlineAnnotationType.BOLD, "Duskmere", id="bold"),
    ],
)
def test_inline_annotation_offsets_match_block_text(
    read_fixture: ReadFixture, annotation_type: InlineAnnotationType, expected: str
) -> None:
    """Смещения разметки указывают ровно на оформленный фрагмент текста блока."""
    document = read_fixture("synthetic_story").document
    block = next(
        block
        for block in document.blocks()
        if block.text.startswith("Wren Calder had walked for three days")
    )

    fragments = [
        block.text[annotation.start : annotation.end]
        for annotation in block.inline_annotations
        if annotation.type is annotation_type
    ]
    assert fragments == [expected]


def test_emphasis_tags_map_to_distinct_annotation_types(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """`i`, `em`, `b` и `strong` дают разные виды встроенной разметки."""
    source = build_epub(
        tmp_path / "markup.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<p>Plain <i>italic</i> and <em>emphasis</em> and "
                "<b>bold</b> and <strong>strong</strong>.</p>"
            )
        },
    )

    document = reader.read(
        EpubReadRequest(
            book_id="book-1", source_path=str(source), target_language="ru"
        )
    ).document
    block = next(iter(document.blocks()))

    assert block.text == "Plain italic and emphasis and bold and strong."
    assert [
        (annotation.type, block.text[annotation.start : annotation.end])
        for annotation in block.inline_annotations
    ] == [
        (InlineAnnotationType.ITALIC, "italic"),
        (InlineAnnotationType.EMPHASIS, "emphasis"),
        (InlineAnnotationType.BOLD, "bold"),
        (InlineAnnotationType.BOLD, "strong"),
    ]


def test_footnote_reference_resolves_to_original_footnote(
    read_fixture: ReadFixture,
) -> None:
    """Ссылка на сноску ведёт к блоку сноски с её оригинальным идентификатором."""
    document = read_fixture("synthetic_story").document
    index = anchor_index(document)

    citation = next(
        block
        for block in document.blocks()
        for annotation in block.inline_annotations
        if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE
    )
    reference = next(
        annotation
        for annotation in citation.inline_annotations
        if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE
    )
    footnote = index[reference.target or ""]

    assert reference.target == "chapter1.xhtml#fn1"
    assert footnote.type is BlockType.FOOTNOTE
    assert footnote.source_reference.element_id == "fn1"


def test_back_reference_from_footnote_returns_to_citation(
    read_fixture: ReadFixture,
) -> None:
    """Обратная ссылка из текста сноски ведёт к блоку, где сноска процитирована."""
    document = read_fixture("synthetic_story").document
    index = anchor_index(document)

    footnote_text = next(
        block
        for block in document.blocks()
        if block.type is BlockType.FOOTNOTE and block.text
    )
    back_reference = next(
        annotation
        for annotation in footnote_text.inline_annotations
        if annotation.type is InlineAnnotationType.LINK
    )

    assert back_reference.target == "chapter1.xhtml#fnref1"
    assert index[back_reference.target].text.startswith("Wren carried very little")


@pytest.mark.parametrize(
    ("name", "expected_style"),
    [
        pytest.param("synthetic_story", "#fn1", id="bare-fragment"),
        pytest.param("beowulf", "#I.FNDEF.1", id="path-with-fragment"),
    ],
)
def test_internal_links_resolve_for_both_href_styles(
    read_fixture: ReadFixture, name: str, expected_style: str
) -> None:
    """Голая ссылка на фрагмент и ссылка с именем файла разрешаются одинаково."""
    document = read_fixture(name).document
    index = anchor_index(document)

    internal = [
        annotation
        for block in document.blocks()
        for annotation in block.inline_annotations
        if annotation.type in _TARGETED
        and "#" in (annotation.target or "")
        and not (annotation.target or "").startswith(("http", "mailto"))
    ]

    assert internal
    assert any((annotation.target or "").endswith(expected_style) for annotation in internal)
    assert not [
        annotation.target for annotation in internal if annotation.target not in index
    ]


def test_original_footnotes_keep_their_identifiers(read_fixture: ReadFixture) -> None:
    """Идентификаторы оригинальных сносок книги сохраняются без изменений."""
    document = read_fixture("beowulf").document

    footnote_ids = {
        block.source_reference.element_id
        for block in document.blocks()
        if block.type is BlockType.FOOTNOTE
    }

    assert "I.FNDEF.1" in footnote_ids
    assert "I.FNDEF.2" in footnote_ids


def test_every_footnote_reference_points_to_a_footnote_block(
    read_fixture: ReadFixture,
) -> None:
    """Книга со сносками восстанавливается без потери связей «ссылка → сноска»."""
    document = read_fixture("beowulf").document
    index = anchor_index(document)

    references = [
        annotation
        for block in document.blocks()
        for annotation in block.inline_annotations
        if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE
    ]

    assert len(references) == 132
    targets = [index.get(annotation.target or "") for annotation in references]
    assert all(target is not None for target in targets)
    assert {target.type for target in targets if target} == {BlockType.FOOTNOTE}


def test_external_links_are_kept_as_written(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Ссылка наружу книги не превращается во внутренний путь."""
    source = build_epub(
        tmp_path / "external.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                '<p>See <a href="https://example.org/page">the source</a>.</p>'
            )
        },
    )

    document = reader.read(
        EpubReadRequest(
            book_id="book-1", source_path=str(source), target_language="ru"
        )
    ).document
    block = next(iter(document.blocks()))

    assert [annotation.target for annotation in block.inline_annotations] == [
        "https://example.org/page"
    ]
