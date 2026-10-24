"""Invariants of canonical document, sections, and blocks (PRD § 6-7)."""

from __future__ import annotations

import pytest
from domain_factories import (
    DOCUMENT_ID,
    RESOURCE_PATH,
    SECTION_ID,
    make_block,
    make_blocks,
    make_document,
    make_section,
)

from linguary.domain import (
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentStructureError,
    InlineAnnotation,
    InlineAnnotationType,
    SourceReference,
    SupportStatus,
    UnknownBlockError,
)


def test_document_returns_blocks_in_reading_order() -> None:
    """Book blocks are listed in reading order, not creation order."""
    blocks = make_blocks(3)
    section = make_section(reversed(blocks))
    document = make_document([section])
    assert tuple(block.order for block in document.blocks()) == (0, 1, 2)


def test_document_finds_block_by_id() -> None:
    """A book block is accessible by its stable identifier."""
    blocks = make_blocks(2)
    document = make_document([make_section(blocks)])
    assert document.block(blocks[1].id) is blocks[1]


def test_document_rejects_unknown_block_id() -> None:
    """Accessing a non-existent block is rejected rather than returning an empty result."""
    document = make_document([make_section(make_blocks(1))])
    with pytest.raises(UnknownBlockError):
        document.block("missing-block")


def test_document_rejects_duplicate_section_order() -> None:
    """Two chapters cannot share the same position in reading order."""
    with pytest.raises(DocumentStructureError):
        make_document(
            [
                make_section(section_id="section-1", order=0),
                make_section(section_id="section-2", order=0),
            ]
        )


def test_document_rejects_same_source_and_target_language() -> None:
    """A book is translated from one language to another."""
    with pytest.raises(DocumentStructureError):
        CanonicalDocument(
            id=DOCUMENT_ID,
            metadata=BookMetadata(title="Nineteen Eighty-Four"),
            source_language="en",
            target_language="en",
        )


def test_section_rejects_block_from_other_section() -> None:
    """A section does not accept a block from another chapter."""
    foreign = make_block(section_id="section-2")
    with pytest.raises(DocumentStructureError):
        make_section([foreign])


def test_section_rejects_duplicate_block_order() -> None:
    """Block sequence numbers within a chapter are unique."""
    first = make_block(order=0, structural_path="/body/p[1]")
    second = make_block(order=0, structural_path="/body/p[2]")
    with pytest.raises(DocumentStructureError):
        make_section([first, second])


def test_section_rejects_missing_parent_block() -> None:
    """Nesting references a block within the same chapter."""
    child = DocumentBlock(
        id="child",
        section_id=SECTION_ID,
        type=BlockType.LIST_ITEM,
        order=0,
        text="первый пункт",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/ul[1]/li[1]"
        ),
        parent_id="missing-parent",
    )
    with pytest.raises(DocumentStructureError):
        make_section([child])


def test_section_accepts_nested_block() -> None:
    """A list item references a parent block within the same chapter."""
    parent = make_block(order=0, type=BlockType.OTHER)
    child = DocumentBlock(
        id="child",
        section_id=SECTION_ID,
        type=BlockType.LIST_ITEM,
        order=1,
        text="первый пункт",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/ul[1]/li[1]"
        ),
        parent_id=parent.id,
    )
    assert make_section([parent, child]).block(child.id).parent_id == parent.id


def test_block_rejects_annotation_outside_text() -> None:
    """Markup cannot extend beyond the block text boundaries."""
    with pytest.raises(DocumentStructureError):
        DocumentBlock(
            id="block-1",
            section_id=SECTION_ID,
            type=BlockType.PARAGRAPH,
            order=0,
            text="короткий",
            source_reference=SourceReference(
                resource_path=RESOURCE_PATH, structural_path="/body/p[1]"
            ),
            inline_annotations=(
                InlineAnnotation(type=InlineAnnotationType.ITALIC, start=0, end=99),
            ),
        )


def test_block_keeps_markup_separate_from_text() -> None:
    """Block text is stored without XHTML, while markup is stored as offsets within this text."""
    block = DocumentBlock(
        id="block-1",
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=0,
        text="Winston Smith",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(type=InlineAnnotationType.ITALIC, start=0, end=7),
        ),
    )
    assert "<" not in block.text
    assert block.inline_annotations[0].end == 7


def test_link_annotation_requires_target() -> None:
    """A link or footnote reference is meaningless without a destination target."""
    with pytest.raises(DocumentStructureError):
        InlineAnnotation(type=InlineAnnotationType.LINK, start=0, end=3)


def test_annotation_rejects_empty_range() -> None:
    """Markup covers at least one character."""
    with pytest.raises(DocumentStructureError):
        InlineAnnotation(type=InlineAnnotationType.BOLD, start=4, end=4)


def test_block_rejects_fatal_support_status() -> None:
    """A construct with status FATAL does not become a block: import is impossible."""
    with pytest.raises(DocumentStructureError):
        DocumentBlock(
            id="block-1",
            section_id=SECTION_ID,
            type=BlockType.OTHER,
            order=0,
            text="",
            source_reference=SourceReference(
                resource_path=RESOURCE_PATH, structural_path="/body/div[1]"
            ),
            support_status=SupportStatus.FATAL,
        )


@pytest.mark.parametrize(
    ("types", "texts", "expected"),
    [
        pytest.param(
            (BlockType.HEADING, BlockType.FOOTNOTE, BlockType.FOOTNOTE),
            ("Notes", "1 A plain face.", "2 A storm."),
            True,
            id="heading-and-footnotes",
        ),
        pytest.param(
            (BlockType.FOOTNOTE, BlockType.PAGE_BREAK, BlockType.OTHER),
            ("1 A plain face.", "", ""),
            True,
            id="footnotes-with-empty-service-blocks",
        ),
        pytest.param(
            (BlockType.PARAGRAPH, BlockType.FOOTNOTE),
            ("Her face.", "1 A plain face."),
            False,
            id="chapter-with-footnote",
        ),
        pytest.param(
            (BlockType.IMAGE, BlockType.FOOTNOTE),
            ("", "1 A plain face."),
            False,
            id="illustration-with-footnote",
        ),
        pytest.param(
            (BlockType.HEADING,), ("Notes",), False, id="heading-only"
        ),
    ],
)
def test_section_of_only_footnotes_is_a_notes_page(
    types: tuple[BlockType, ...], texts: tuple[str, ...], expected: bool
) -> None:
    """A section consisting solely of footnotes is a notes page, not a chapter (QA-04)."""
    blocks = [
        make_block(order=index, type=type_, text=text)
        for index, (type_, text) in enumerate(zip(types, texts, strict=True))
    ]

    assert make_section(blocks).is_notes_only is expected


def test_nested_paragraph_of_a_footnote_does_not_make_it_a_chapter() -> None:
    """Paragraphs inside a footnote are part of the footnote, not chapter text."""
    note = make_block(order=0, type=BlockType.FOOTNOTE, text="")
    inner = DocumentBlock(
        id="inner",
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=1,
        text="A plain face.",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/aside[1]/p[1]"
        ),
        parent_id=note.id,
    )

    assert make_section([note, inner]).is_notes_only
