"""Tests for split paragraph continuation styling in writer.py (P3.T1)."""

from __future__ import annotations

from lxml import etree

from linguary.adapters.epub.writer import (
    _STYLESHEET,
    _block_attributes,
    _render_chapters,
    _OpaqueSource,
    _ExportWarnings,
)
from linguary.application.ports.epub import EpubWriteRequest
from linguary.domain import (
    BlockTreatment,
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    TreatmentPlan,
)


def _make_block(
    order: int,
    text: str,
    *,
    split_group: str | None = None,
    split_index: str | None = None,
    split_total: str | None = None,
) -> DocumentBlock:
    attributes: dict[str, str] = {"tag": "p"}
    if split_group:
        attributes["split_group"] = split_group
    if split_index:
        attributes["split_index"] = split_index
    if split_total:
        attributes["split_total"] = split_total

    structural_path = (
        f"{split_group}/text()[{split_index}]"
        if split_group and split_index
        else f"/body/p[{order + 1}]"
    )

    return DocumentBlock.create(
        document_id="book_1",
        section_id="sec_1",
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path="sec_1.xhtml",
            structural_path=structural_path,
        ),
        attributes=attributes,
    )


def test_writer_adds_split_continuation_class_to_subsequent_fragments() -> None:
    """Fragments with split_index > 1 receive the 'linguary-split-continuation' class."""
    b2 = _make_block(1, "Second fragment", split_group="/body/p[1]", split_index="2", split_total="3")
    attrs2 = _block_attributes(b2, tag="p")
    assert "class" in attrs2
    assert "linguary-split-continuation" in attrs2["class"]

    b3 = _make_block(2, "Third fragment", split_group="/body/p[1]", split_index="3", split_total="3")
    attrs3 = _block_attributes(b3, tag="p")
    assert "class" in attrs3
    assert "linguary-split-continuation" in attrs3["class"]

    # Safe conversion: whitespace around the index is handled properly
    b_spaced = _make_block(3, "Spaced fragment", split_group="/body/p[1]", split_index=" 2 ", split_total="3")
    assert "linguary-split-continuation" in _block_attributes(b_spaced, tag="p")["class"]

    # Invalid index does not raise and does not add the class
    b_invalid = _make_block(4, "Invalid index", split_group="/body/p[1]", split_index="invalid", split_total="3")
    assert "linguary-split-continuation" not in _block_attributes(b_invalid, tag="p").get("class", "")


def test_writer_preserves_default_class_for_first_fragment() -> None:
    """Continuation class is not added to the first fragment (split_index = 1) or regular blocks."""
    b1 = _make_block(0, "First fragment", split_group="/body/p[1]", split_index="1", split_total="3")
    attrs1 = _block_attributes(b1, tag="p")
    assert "linguary-split-continuation" not in attrs1.get("class", "")

    b_normal = _make_block(5, "Normal paragraph")
    attrs_normal = _block_attributes(b_normal, tag="p")
    assert "linguary-split-continuation" not in attrs_normal.get("class", "")


def test_writer_stylesheet_defines_split_continuation_style() -> None:
    """The stylesheet _STYLESHEET contains selector .linguary-split-continuation resetting indents."""
    assert ".linguary-split-continuation" in _STYLESHEET
    assert "text-indent: 0" in _STYLESHEET
    assert "margin-top:" in _STYLESHEET


def test_writer_renders_valid_xhtml_structure() -> None:
    """Markup of paragraph continuations produces valid XHTML without nesting <p> inside <p>."""
    b1 = _make_block(0, "Opening sentence.", split_group="/body/p[1]", split_index="1", split_total="2")
    b2 = _make_block(1, "Continuation sentence.", split_group="/body/p[1]", split_index="2", split_total="2")

    attrs1 = _block_attributes(b1, tag="p")
    attrs2 = _block_attributes(b2, tag="p")

    # Form XHTML fragment
    p1 = etree.Element("p", attrs1)
    p1.text = b1.text
    p2 = etree.Element("p", attrs2)
    p2.text = b2.text

    body = etree.Element("body")
    body.append(p1)
    body.append(p2)

    serialized = etree.tostring(body, encoding="utf-8").decode("utf-8")
    assert '<p class="linguary-split-continuation">' in serialized
    assert p1.find(".//p") is None
    assert p2.find(".//p") is None
