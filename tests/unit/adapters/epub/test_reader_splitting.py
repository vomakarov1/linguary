"""Tests for compound paragraph splitting integration in reader.py (P2.T1)."""

from __future__ import annotations

from lxml import etree

from linguary.adapters.epub.reader import _BlockContext, _SectionBuilder
from linguary.adapters.epub.splitting import SPLIT_MIN_CHARS
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
    # Create a paragraph > 800 characters consisting of exactly 3 sentences
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
    """A long paragraph is split into balanced blocks during import."""
    builder = _make_builder()
    context = _BlockContext()
    p, text = _make_long_paragraph_html()

    builder._emit_leaf(p, context)

    assert len(builder.blocks) > 1
    assert "".join(block.text for block in builder.blocks) == text
    assert all(block.type is BlockType.PARAGRAPH for block in builder.blocks)


def test_reader_generates_indexed_structural_paths() -> None:
    """Indexed paths like base_xpath/text()[k] are generated for fragments with unique block_id values."""
    builder = _make_builder()
    context = _BlockContext()
    p, text = _make_long_paragraph_html()

    builder._emit_leaf(p, context)

    assert len(builder.blocks) > 1
    paths = [block.source_reference.structural_path for block in builder.blocks]
    assert paths == [
        f"/html[1]/body[1]/p[1]/text()[{index}]"
        for index in range(1, len(builder.blocks) + 1)
    ]
    assert "".join(block.text for block in builder.blocks) == text

    # Block identifiers must be unique and deterministic
    ids = [block.id for block in builder.blocks]
    assert len(set(ids)) == len(builder.blocks)


def test_reader_sets_split_attributes() -> None:
    """Fragments receive split_group, split_index, and split_total attributes."""
    builder = _make_builder()
    context = _BlockContext()
    p, _ = _make_long_paragraph_html()

    builder._emit_leaf(p, context)

    base_path = "/html[1]/body[1]/p[1]"
    total = str(len(builder.blocks))
    for index, block in enumerate(builder.blocks, start=1):
        assert block.attributes["split_group"] == base_path
        assert block.attributes["split_index"] == str(index)
        assert block.attributes["split_total"] == total


def test_reader_preserves_sequential_order() -> None:
    """Block sequence number 'order' increments monotonically."""
    builder = _make_builder()
    context = _BlockContext()

    h1 = etree.fromstring("<html><body><h1>Heading</h1></body></html>").find(".//h1")
    assert h1 is not None
    builder._emit_leaf(h1, context)

    p, _ = _make_long_paragraph_html()
    builder._emit_leaf(p, context)
    split_count = len(builder.blocks) - 1

    p_short = etree.fromstring("<html><body><p>Short paragraph.</p></body></html>").find(".//p")
    assert p_short is not None
    builder._emit_leaf(p_short, context)

    # Heading, split paragraph fragments, and short paragraph remain in document order.
    assert split_count > 1
    assert len(builder.blocks) == split_count + 2
    orders = [block.order for block in builder.blocks]
    assert orders == list(range(len(builder.blocks)))


def test_reader_preserves_short_paragraphs() -> None:
    """Short paragraphs are not split and preserve the base structural_path."""
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


def test_reader_and_writer_split_and_regroup_long_leaf_blockquotes_preserving_nested_quotes() -> None:
    """Long leaf blockquotes and quote inline runs split at SPLIT_MIN_CHARS while container quotes keep child blocks."""
    assert SPLIT_MIN_CHARS == 600
    _, long_text = _make_long_paragraph_html()

    html = etree.fromstring(
        f"<html><body>"
        f"<blockquote id='leaf-q'>{long_text}</blockquote>"
        f"<blockquote id='run-q'>{long_text}<p>Short attribution.</p></blockquote>"
        f"<blockquote id='nested-q'><p>{long_text}</p><blockquote><p>Inner nested quote.</p></blockquote></blockquote>"
        f"</body></html>"
    )
    body = html.find("body")
    assert body is not None

    builder = _make_builder()
    builder.walk(body, _BlockContext())

    leaf_blocks = [
        block
        for block in builder.blocks
        if block.attributes.get("split_group") == "/html[1]/body[1]/blockquote[1]"
    ]
    assert len(leaf_blocks) > 1
    assert all(block.type is BlockType.QUOTE for block in leaf_blocks)
    assert "".join(block.text for block in leaf_blocks) == long_text
    assert [block.source_reference.structural_path for block in leaf_blocks] == [
        f"/html[1]/body[1]/blockquote[1]/text()[{idx}]"
        for idx in range(1, len(leaf_blocks) + 1)
    ]

    run_blocks = [
        block
        for block in builder.blocks
        if block.attributes.get("split_group") == "/html[1]/body[1]/blockquote[2]/text()[1]"
    ]
    assert len(run_blocks) > 1
    assert all(block.type is BlockType.QUOTE for block in run_blocks)
    assert "".join(block.text for block in run_blocks) == long_text

    attribution_blocks = [
        block
        for block in builder.blocks
        if block.source_reference.structural_path == "/html[1]/body[1]/blockquote[2]/p[1]"
    ]
    assert len(attribution_blocks) == 1
    assert attribution_blocks[0].type is BlockType.QUOTE
    assert "split_group" not in attribution_blocks[0].attributes

    outer_quote_blocks = [
        block
        for block in builder.blocks
        if block.attributes.get("split_group") == "/html[1]/body[1]/blockquote[3]/p[1]"
    ]
    assert len(outer_quote_blocks) > 1
    assert all(block.type is BlockType.QUOTE for block in outer_quote_blocks)

    inner_nested_blocks = [
        block
        for block in builder.blocks
        if block.source_reference.structural_path
        == "/html[1]/body[1]/blockquote[3]/blockquote[1]/p[1]"
    ]
    assert len(inner_nested_blocks) == 1
    assert inner_nested_blocks[0].type is BlockType.QUOTE
    assert inner_nested_blocks[0].text == "Inner nested quote."
    assert "split_group" not in inner_nested_blocks[0].attributes

