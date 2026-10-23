"""Тесты для интеграции деления составных абзацев в reader.py (P2.T1)."""

from __future__ import annotations

from lxml import etree

from linguary.adapters.epub.reader import _BlockContext, _SectionBuilder
from linguary.domain import BlockType


def _make_builder() -> _SectionBuilder:
    return _SectionBuilder(
        book_id="test_book",
        section_id="sec_1",
        resource_path="chapter1.xhtml",
        footnote_ids=frozenset(),
        manifest_paths=frozenset({"chapter1.xhtml"}),
    )


def _make_long_paragraph_html() -> tuple[etree._Element, str]:
    # Создаём абзац > 800 символов ровно из 3 предложений
    s1 = "This is the first very long sentence with " + "many descriptive words " * 15 + "to fill up character counts nicely. "
    s2 = "Then comes the second equally detailed sentence with " + "extra explanations " * 15 + "showing how splitting operates cleanly. "
    s3 = "Finally the third sentence finishes the paragraph structure with " + "concluding remarks " * 15 + "and proper termination."
    text = s1 + s2 + s3
    assert len(text) >= 800

    html = etree.fromstring(f"<html><body><p id='para1'>{text}</p></body></html>")
    p = html.find(".//p")
    assert p is not None
    return p, text


def test_reader_splits_long_paragraph_into_multiple_blocks() -> None:
    """Длинный составной абзац при импорте делится на несколько блоков."""
    builder = _make_builder()
    context = _BlockContext()
    p, _ = _make_long_paragraph_html()

    builder._emit_leaf(p, context)

    assert len(builder.blocks) == 3
    assert all(block.type is BlockType.PARAGRAPH for block in builder.blocks)


def test_reader_generates_indexed_structural_paths() -> None:
    """Для фрагментов формируются пути вида base_xpath/text()[k] с уникальными block_id."""
    builder = _make_builder()
    context = _BlockContext()
    p, _ = _make_long_paragraph_html()

    builder._emit_leaf(p, context)

    assert len(builder.blocks) == 3
    paths = [block.source_reference.structural_path for block in builder.blocks]
    assert paths == [
        "/html[1]/body[1]/p[1]/text()[1]",
        "/html[1]/body[1]/p[1]/text()[2]",
        "/html[1]/body[1]/p[1]/text()[3]",
    ]

    # Идентификаторы блоков должны быть уникальными и детерминированными
    ids = [block.id for block in builder.blocks]
    assert len(set(ids)) == 3


def test_reader_sets_split_attributes() -> None:
    """Фрагменты получают атрибуты split_group, split_index, split_total."""
    builder = _make_builder()
    context = _BlockContext()
    p, _ = _make_long_paragraph_html()

    builder._emit_leaf(p, context)

    base_path = "/html[1]/body[1]/p[1]"
    assert builder.blocks[0].attributes["split_group"] == base_path
    assert builder.blocks[0].attributes["split_index"] == "1"
    assert builder.blocks[0].attributes["split_total"] == "3"

    assert builder.blocks[1].attributes["split_group"] == base_path
    assert builder.blocks[1].attributes["split_index"] == "2"
    assert builder.blocks[1].attributes["split_total"] == "3"

    assert builder.blocks[2].attributes["split_group"] == base_path
    assert builder.blocks[2].attributes["split_index"] == "3"
    assert builder.blocks[2].attributes["split_total"] == "3"


def test_reader_preserves_sequential_order() -> None:
    """Порядковый номер блоков order возрастает непрерывно."""
    builder = _make_builder()
    context = _BlockContext()

    h1 = etree.fromstring("<html><body><h1>Heading</h1></body></html>").find(".//h1")
    assert h1 is not None
    builder._emit_leaf(h1, context)

    p, _ = _make_long_paragraph_html()
    builder._emit_leaf(p, context)

    p_short = etree.fromstring("<html><body><p>Short paragraph.</p></body></html>").find(".//p")
    assert p_short is not None
    builder._emit_leaf(p_short, context)

    # 1 (h1) + 3 (split p) + 1 (short p) = 5 блоков
    assert len(builder.blocks) == 5
    orders = [block.order for block in builder.blocks]
    assert orders == [0, 1, 2, 3, 4]


def test_reader_preserves_short_paragraphs() -> None:
    """Короткие абзацы не расщепляются и сохраняют базовый structural_path."""
    builder = _make_builder()
    context = _BlockContext()

    p_short = etree.fromstring("<html><body><p>Just a short paragraph.</p></body></html>").find(".//p")
    assert p_short is not None
    builder._emit_leaf(p_short, context)

    assert len(builder.blocks) == 1
    block = builder.blocks[0]
    assert block.source_reference.structural_path == "/html[1]/body[1]/p[1]"
    assert "split_group" not in block.attributes
    assert "split_index" not in block.attributes
    assert "split_total" not in block.attributes
