"""Tests for split paragraph continuation styling in writer.py (P3.T1)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

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
    BlockProcessingResult,
    BookMetadata,
    InlineAnnotation,
    InlineAnnotationType,
    BlockTreatment,
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    TreatmentPlan,
)
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.learning import FragmentAssistance


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


_XHTML = "{http://www.w3.org/1999/xhtml}"
_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"


def render_split_blocks(blocks, results=None):
    section = DocumentSection(
        id="sec_1", document_id="book_1", title="Chapter One", order=0,
        source_resource_path="sec_1.xhtml", blocks=tuple(blocks),
    )
    document = CanonicalDocument(
        id="book_1", metadata=BookMetadata(title="Story"),
        source_language="en", target_language="ru", sections=(section,),
    )
    request = EpubWriteRequest(document=document, target_path="unused.epub", block_results=results or {})
    with _OpaqueSource(None) as opaque:
        return _render_chapters(request, opaque=opaque, warnings=_ExportWarnings())[0].root


def test_split_group_renders_one_exact_original_with_all_inline_links():
    first = _make_block(0, "First bold ", split_group="g", split_index="1")
    first = replace(first, inline_annotations=(InlineAnnotation(InlineAnnotationType.BOLD, 6, 10),))
    second = _make_block(1, "linked end.", split_group="g", split_index="2")
    second = replace(second, inline_annotations=(
        InlineAnnotation(InlineAnnotationType.LINK, 0, 6, "https://example.com"),
    ))
    root = render_split_blocks((first, second))
    paragraphs = list(root.iter(f"{_XHTML}p"))
    assert len(paragraphs) == 1
    assert "".join(paragraphs[0].itertext()) == first.text + second.text
    assert next(root.iter(f"{_XHTML}b")).text == "bold"
    link = next(root.iter(f"{_XHTML}a"))
    assert link.get("href") == "https://example.com"
    assert link.text == "linked"


def test_nonadjacent_split_groups_remain_separate():
    blocks = (
        _make_block(0, "First", split_group="g", split_index="1"),
        _make_block(1, "Between"),
        _make_block(2, "Last", split_group="g", split_index="2"),
    )
    root = render_split_blocks(blocks)
    assert ["".join(p.itertext()) for p in root.iter(f"{_XHTML}p")] == [b.text for b in blocks]


def test_fragment_help_links_all_precise_anchors_and_preserves_unresolved():
    first = _make_block(0, "Repeated bold ", split_group="g", split_index="1")
    first = replace(first, inline_annotations=(InlineAnnotation(InlineAnnotationType.BOLD, 9, 13),))
    second = _make_block(1, "Repeated linked.", split_group="g", split_index="2")
    second = replace(second, inline_annotations=(
        InlineAnnotation(InlineAnnotationType.LINK, 9, 15, "https://example.com"),
    ))
    anchors = (SourceAnchor(first.id, 0, 8, "Repeated"), SourceAnchor(second.id, 0, 8, "Repeated"))
    helps = (
        FragmentAssistance("difficulty", ("p1",), "sentence_rephrase", "A simpler sentence.", anchors),
        FragmentAssistance("missing", ("p2",), "unresolved", None,
                           (SourceAnchor(second.id, 9, 15, "linked"),)),
    )
    results = {
        first.id: BlockProcessingResult(first.id, BlockTreatment.READ_AS_IS, fragment_assistances=helps[:1]),
        second.id: BlockProcessingResult(second.id, BlockTreatment.READ_AS_IS, fragment_assistances=helps[1:]),
    }
    root = render_split_blocks((first, second), results)
    original = next(root.iter(f"{_XHTML}p"))
    assert "".join(original.itertext()) == first.text + second.text
    container = original.getnext()
    assert container.get("class") == "linguary-fragment-help"
    assert len(root.xpath('//*[@class="linguary-fragment-help"]')) == 1
    notes = [node for node in container if node.get(_EPUB_TYPE) == "footnote"]
    assert len(notes) == 2
    assert "Помощь пока не подготовлена." in "".join(notes[1].itertext())
    references = [node for node in root.iter(f"{_XHTML}a") if node.get(_EPUB_TYPE) == "noteref"]
    assert len(references) == 3
    assert [node.get("aria-label") for node in references] == ["Repeated", "Repeated", "linked"]
    assert [node.get("data-source-block") for node in references] == [first.id, second.id, second.id]
    ids = [node.get("id") for node in root.iter() if node.get("id")]
    assert len(ids) == len(set(ids))
    for node in root.iter(f"{_XHTML}a"):
        href = node.get("href", "")
        if href.startswith("#"):
            assert href[1:] in ids
        assert not list(node.iterdescendants(f"{_XHTML}a"))


@pytest.mark.parametrize("original", ["line\nx", "line\nword", "line\n\nx", "line\nx\n\nword"])
def test_fragment_help_offsets_include_source_line_breaks(original):
    block = _make_block(0, original)
    quote = original.rsplit("\n", 1)[-1]
    start = len(original) - len(quote)
    anchor = SourceAnchor(block.id, start, len(original), quote)
    assistance = FragmentAssistance("difficulty", ("point",), "lexical_hint", "A word.", (anchor,))
    result = BlockProcessingResult(block.id, BlockTreatment.READ_AS_IS,
                                   fragment_assistances=(assistance,))
    root = render_split_blocks((block,), {block.id: result})
    paragraph = next(root.iter(f"{_XHTML}p"))
    references = [a for a in paragraph.iter(f"{_XHTML}a") if a.get(_EPUB_TYPE) == "noteref"]
    assert len(references) == 1
    reference = references[0]
    assert reference.get("data-source-start") == str(start)
    assert reference.tail == quote
    assert reference.getprevious().tag == f"{_XHTML}br"
    assert reference.getprevious().tail == ""
    assert len(list(paragraph.iter(f"{_XHTML}br"))) == original.count("\n")
    source = deepcopy(paragraph)
    source_reference = next(a for a in source.iter(f"{_XHTML}a") if a.get(_EPUB_TYPE) == "noteref")
    source_reference.getprevious().tail = source_reference.tail
    source.remove(source_reference)
    rendered = etree.tostring(source, encoding="unicode")
    assert rendered.endswith(original.replace("\n", "<br/>") + "</p>")
    ids = {node.get("id") for node in root.iter() if node.get("id")}
    footnote = next(node for node in root.iter() if node.get("id") == reference.get("href")[1:])
    assert footnote.get(_EPUB_TYPE) == "footnote"
    assert any(a.get("href") == f"#{reference.get('id')}" for a in footnote.iter(f"{_XHTML}a"))
    assert all(a.get("href")[1:] in ids for a in footnote.iter(f"{_XHTML}a")
               if a.get("href", "").startswith("#"))
