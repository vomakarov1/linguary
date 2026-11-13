"""Sentence segments are markup inside one document block, not a split of it."""

from __future__ import annotations

import pytest
from domain_factories import DOCUMENT_ID, RESOURCE_PATH, SECTION_ID

from linguary.domain import (
    BlockSegment,
    BlockType,
    DocumentBlock,
    DocumentStructureError,
    SourceReference,
)

TEXT = "Alpha. Beta."


def _block(text: str, segments: tuple[BlockSegment, ...] = ()) -> DocumentBlock:
    """A paragraph that stores only the segments under test."""
    return DocumentBlock.create(
        document_id=DOCUMENT_ID,
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH,
            structural_path="/body/section[1]/p[1]",
        ),
        segments=segments,
    )


def test_block_without_segments_has_single_span() -> None:
    """No stored segments means one span over the whole text of the same block."""
    block = _block(TEXT)
    assert block.segments == ()
    assert block.segment_spans == (BlockSegment(0, len(TEXT), 0),)

    empty = _block("")
    assert empty.segments == ()
    assert empty.segment_spans == (BlockSegment(0, 0, 0),)


@pytest.mark.parametrize(
    "segments",
    [
        pytest.param((BlockSegment(0, 4), BlockSegment(5, 12)), id="gap"),
        pytest.param((BlockSegment(0, 7), BlockSegment(6, 12)), id="overlap"),
        pytest.param((BlockSegment(1, 12),), id="late-start"),
        pytest.param((BlockSegment(0, 11),), id="short"),
        pytest.param((BlockSegment(0, 13),), id="past-end"),
        pytest.param((BlockSegment(6, 12), BlockSegment(0, 6)), id="out-of-order"),
    ],
)
def test_segments_must_cover_text_contiguously(segments: tuple[BlockSegment, ...]) -> None:
    """Stored segments are a half-open cover of [0, len(text)) without gaps or overlap."""
    with pytest.raises(DocumentStructureError):
        _block(TEXT, segments)


def test_empty_segment_rejected() -> None:
    """A stored segment covers at least one character, and its offsets are ordered."""
    with pytest.raises(DocumentStructureError):
        _block(TEXT, (BlockSegment(0, 0),))
    with pytest.raises(DocumentStructureError):
        _block("", (BlockSegment(0, 0),))
    with pytest.raises(DocumentStructureError):
        _block(
            TEXT,
            (BlockSegment(0, 6), BlockSegment(6, 6), BlockSegment(6, len(TEXT))),
        )
    with pytest.raises(DocumentStructureError):
        BlockSegment(4, 2)
    with pytest.raises(DocumentStructureError):
        BlockSegment(-1, 3)
    with pytest.raises(DocumentStructureError):
        BlockSegment(0, 3, -1)


def test_single_segment_normalized_to_empty() -> None:
    """One covering segment is the default span and is not stored separately."""
    span = BlockSegment(0, len(TEXT), 0)
    block = _block(TEXT, (span,))
    assert block.segments == ()
    assert block.segment_spans == (span,)


def test_sentence_index_preserved_across_spans() -> None:
    """Parts of one sentence keep its index; the next sentence gets another."""
    spans = (
        BlockSegment(0, 6, 0),
        BlockSegment(6, 8, 1),
        BlockSegment(8, len(TEXT), 1),
    )
    block = _block(TEXT, spans)
    assert block.segments == spans
    assert block.segment_spans == spans
    assert [span.sentence_index for span in block.segment_spans] == [0, 1, 1]
