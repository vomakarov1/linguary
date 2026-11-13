"""Tests for importing a paragraph or quote as one block with segments."""

from __future__ import annotations

from lxml import etree

from linguary.adapters.epub.reader import _BlockContext, _SectionBuilder
from linguary.adapters.epub.splitting import segment_text
from linguary.domain import BlockSegment, BlockType, InlineAnnotationType
from linguary.domain.identifiers import build_block_id


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


def test_reader_preserves_sequential_order() -> None:
    """Block sequence number 'order' increments monotonically across one document."""
    _, long_text = _make_long_paragraph_html()
    html = etree.fromstring(
        "<html><body>"
        "<h1>Heading</h1>"
        f"<p id='para1'>{long_text}</p>"
        "<p>Short paragraph.</p>"
        "</body></html>"
    )
    body = html.find("body")
    assert body is not None
    builder = _make_builder()
    builder.walk(body, _BlockContext())

    assert [block.type for block in builder.blocks] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.PARAGRAPH,
    ]
    assert builder.blocks[1].text == long_text
    assert builder.blocks[2].text == "Short paragraph."
    assert [block.order for block in builder.blocks] == list(range(len(builder.blocks)))
    assert len({block.id for block in builder.blocks}) == len(builder.blocks)


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
    assert block.segments == ()
    assert block.segment_spans == (BlockSegment(0, len(block.text), 0),)
    assert "split_group" not in block.attributes
    assert "split_index" not in block.attributes
    assert "split_total" not in block.attributes


def test_reader_keeps_long_quotes_whole_and_preserves_nested_quotes() -> None:
    """A long quote is one block; nested quotes stay separate child blocks."""
    _, long_text = _make_long_paragraph_html()
    expected = segment_text(long_text)
    assert len(expected) > 1

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

    leaf = next(
        block
        for block in builder.blocks
        if block.source_reference.structural_path == "/html[1]/body[1]/blockquote[1]"
    )
    assert leaf.type is BlockType.QUOTE
    assert leaf.text == long_text
    assert leaf.segments == expected
    assert leaf.source_reference.element_id == "leaf-q"

    run = next(
        block
        for block in builder.blocks
        if block.source_reference.structural_path
        == "/html[1]/body[1]/blockquote[2]/text()[1]"
    )
    assert run.type is BlockType.QUOTE
    assert run.text == long_text
    assert run.segments == expected

    attribution_blocks = [
        block
        for block in builder.blocks
        if block.source_reference.structural_path == "/html[1]/body[1]/blockquote[2]/p[1]"
    ]
    assert len(attribution_blocks) == 1
    assert attribution_blocks[0].type is BlockType.QUOTE
    assert attribution_blocks[0].text == "Short attribution."
    assert attribution_blocks[0].segments == ()
    assert "split_group" not in attribution_blocks[0].attributes

    outer_quote = next(
        block
        for block in builder.blocks
        if block.source_reference.structural_path == "/html[1]/body[1]/blockquote[3]/p[1]"
    )
    assert outer_quote.type is BlockType.QUOTE
    assert outer_quote.text == long_text
    assert outer_quote.segments == expected

    inner_nested_blocks = [
        block
        for block in builder.blocks
        if block.source_reference.structural_path
        == "/html[1]/body[1]/blockquote[3]/blockquote[1]/p[1]"
    ]
    assert len(inner_nested_blocks) == 1
    assert inner_nested_blocks[0].type is BlockType.QUOTE
    assert inner_nested_blocks[0].text == "Inner nested quote."
    assert inner_nested_blocks[0].segments == ()
    assert "split_group" not in inner_nested_blocks[0].attributes


def _split_attribute_names() -> frozenset[str]:
    return frozenset({"split_group", "split_index", "split_total"})


def test_long_paragraph_is_single_block_with_segments() -> None:
    """A long paragraph or quote is one block, and its segments are the clause spans."""
    builder = _make_builder()
    context = _BlockContext()
    paragraph, text = _make_long_paragraph_html()
    builder._emit_leaf(paragraph, context)

    assert len(builder.blocks) == 1
    block = builder.blocks[0]
    assert block.type is BlockType.PARAGRAPH
    assert block.text == text
    expected = segment_text(text)
    assert len(expected) > 1
    assert block.segments == expected
    assert block.segment_spans == expected
    assert "".join(block.text[span.start : span.end] for span in block.segment_spans) == text
    assert {span.sentence_index for span in block.segments} == {0, 1, 2}

    quote_html = etree.fromstring(
        f"<html><body><blockquote>{text}</blockquote></body></html>"
    )
    quote = quote_html.find(".//blockquote")
    assert quote is not None
    quote_builder = _make_builder()
    quote_builder._emit_leaf(quote, context)
    assert len(quote_builder.blocks) == 1
    quote_block = quote_builder.blocks[0]
    assert quote_block.type is BlockType.QUOTE
    assert quote_block.text == text
    assert quote_block.segments == expected

    left = "alpha " * 50
    right = "beta " * 60
    link = "KEEP; THIS"
    footnote_html = etree.fromstring(
        '<html xmlns:epub="http://www.idpf.org/2007/ops"><body><p>'
        + left
        + '<a epub:type="noteref" href="#n1">'
        + link
        + "</a>"
        + right
        + "</p></body></html>"
    )
    footnote_paragraph = footnote_html.find(".//p")
    assert footnote_paragraph is not None
    footnote_builder = _make_builder()
    footnote_builder._emit_leaf(footnote_paragraph, context)
    assert len(footnote_builder.blocks) == 1
    footnoted = footnote_builder.blocks[0]
    footnote = next(
        annotation
        for annotation in footnoted.inline_annotations
        if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE
    )
    assert footnoted.text[footnote.start : footnote.end] == link
    protected = ((footnote.start, footnote.end),)
    unprotected = segment_text(footnoted.text)
    assert any(
        footnote.start < span.end < footnote.end for span in unprotected[:-1]
    )
    assert footnoted.segments == segment_text(footnoted.text, protected=protected)
    assert not any(
        footnote.start < span.end < footnote.end for span in footnoted.segment_spans[:-1]
    )


def test_block_id_from_element_path() -> None:
    """The block id comes from the element path, not from a split text()[k] suffix."""
    builder = _make_builder()
    paragraph, text = _make_long_paragraph_html()
    builder._emit_leaf(paragraph, _BlockContext())

    assert len(builder.blocks) == 1
    block = builder.blocks[0]
    path = "/html[1]/body[1]/p[1]"
    assert block.text == text
    assert block.source_reference.structural_path == path
    assert "/text()" not in block.source_reference.structural_path
    assert block.source_reference.element_id == "para1"
    assert block.id == build_block_id(
        document_id="test_book",
        source_resource_path="chapter1.xhtml",
        structural_path=path,
    )

    _, long_text = _make_long_paragraph_html()
    run_html = etree.fromstring(
        f"<html><body><div>{long_text}<p>After.</p></div></body></html>"
    )
    run_body = run_html.find("body")
    assert run_body is not None
    run_builder = _make_builder()
    run_builder.walk(run_body, _BlockContext())
    run_block = next(block for block in run_builder.blocks if block.text == long_text)
    run_path = "/html[1]/body[1]/div[1]/text()[1]"
    assert run_block.source_reference.structural_path == run_path
    assert not run_block.source_reference.structural_path.endswith("/text()[1]/text()[1]")
    assert run_block.id == build_block_id(
        document_id="test_book",
        source_resource_path="chapter1.xhtml",
        structural_path=run_path,
    )
    assert len([block for block in run_builder.blocks if long_text in block.text]) == 1


def test_poem_heading_list_have_no_segments() -> None:
    """A long heading, poem, or list item stays one block and stores no segments."""
    _, long_text = _make_long_paragraph_html()
    assert len(long_text) >= 600
    html = etree.fromstring(
        "<html><body>"
        f"<h1>{long_text}</h1>"
        f"<p class='poem'>{long_text}</p>"
        f"<ul><li>{long_text}</li></ul>"
        "</body></html>"
    )
    body = html.find("body")
    assert body is not None
    builder = _make_builder()
    builder.walk(body, _BlockContext())

    assert [block.type for block in builder.blocks] == [
        BlockType.HEADING,
        BlockType.POEM,
        BlockType.LIST_ITEM,
    ]
    assert [block.text for block in builder.blocks] == [long_text, long_text, long_text]
    assert all(block.segments == () for block in builder.blocks)
    assert all(
        block.segment_spans == (BlockSegment(0, len(long_text), 0),)
        for block in builder.blocks
    )


def test_no_split_attributes() -> None:
    """Imported blocks do not carry split_group, split_index, or split_total."""
    _, long_text = _make_long_paragraph_html()
    html = etree.fromstring(
        "<html><body>"
        f"<p id='para1'>{long_text}</p>"
        f"<blockquote id='quote1'>{long_text}</blockquote>"
        "<blockquote id='empty-q'></blockquote>"
        "</body></html>"
    )
    body = html.find("body")
    assert body is not None
    builder = _make_builder()
    builder.walk(body, _BlockContext())

    assert len(builder.blocks) == 3
    forbidden = _split_attribute_names()
    assert all(forbidden.isdisjoint(block.attributes) for block in builder.blocks)
    empty_quote = builder.blocks[2]
    assert empty_quote.type is BlockType.QUOTE
    assert empty_quote.text == ""
    assert empty_quote.segments == ()
    assert empty_quote.segment_spans == (BlockSegment(0, 0, 0),)


def test_anchors_kept_on_single_block() -> None:
    """Element, pending, and inline anchors stay on the one paragraph block."""
    _, long_text = _make_long_paragraph_html()
    html = etree.fromstring(
        "<html><body><div id='wrap'>"
        f"<p id='para1'>{long_text}<span id='mid-mark'>kept</span></p>"
        "</div></body></html>"
    )
    body = html.find("body")
    assert body is not None
    builder = _make_builder()
    builder.walk(body, _BlockContext())

    assert len(builder.blocks) == 1
    block = builder.blocks[0]
    assert block.type is BlockType.PARAGRAPH
    assert block.text == long_text + "kept"
    assert block.source_reference.element_id == "para1"
    assert block.attributes["anchor_ids"] == "para1 wrap mid-mark"
    assert _split_attribute_names().isdisjoint(block.attributes)

