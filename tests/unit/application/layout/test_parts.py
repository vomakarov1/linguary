"""Display parts projected from segment treatments."""

from __future__ import annotations

import pytest

from linguary.application.layout.parts import (
    PART_MAX_CHARS,
    PART_MIN_CHARS,
    PART_TARGET_CHARS,
    LayoutPart,
    compose_parts,
)
from linguary.domain import (
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    SourceReference,
)


def _block(text: str, segments: tuple[BlockSegment, ...] = ()) -> DocumentBlock:
    """A paragraph whose only varying input is text and segments."""
    return DocumentBlock.create(
        document_id="book-1",
        section_id="section-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="chapter.xhtml",
            structural_path="/body/p[1]",
        ),
        segments=segments,
    )


def _spans(length: int, count: int) -> tuple[str, tuple[BlockSegment, ...]]:
    """`count` contiguous spans of `length` characters."""
    text = "a" * (length * count)
    segments = tuple(
        BlockSegment(index * length, (index + 1) * length, index) for index in range(count)
    )
    return text, segments


def test_english_help_block_single_part_without_translation() -> None:
    """English read-as-is help stays one part even when it is longer than the maximum."""
    sentence = "This English note stays as written. "
    assert len(sentence) == 36
    text = sentence * 12
    segments = tuple(BlockSegment(index * 108, (index + 1) * 108, index) for index in range(4))
    block = _block(text, segments)
    snapshot = (block.id, block.text, block.segments)

    parts = compose_parts(block, (BlockTreatment.READ_AS_IS,) * 4)

    assert parts == (
        LayoutPart(
            start=0,
            end=len(text),
            treatment=BlockTreatment.READ_AS_IS,
            segments=(0, 1, 2, 3),
        ),
    )
    assert parts[0].end - parts[0].start > PART_MAX_CHARS
    assert (block.id, block.text, block.segments) == snapshot


def test_untranslated_block_single_part() -> None:
    """No stored segments and no treatment are one part over the whole block."""
    text = "It was a bright cold day in April."
    block = _block(text)
    assert block.segments == ()

    parts = compose_parts(block, (None,))

    assert block.segment_spans == (BlockSegment(0, len(text), 0),)
    assert parts == (LayoutPart(start=0, end=len(text), treatment=None, segments=(0,)),)


def test_treatment_change_is_part_boundary() -> None:
    """A treatment change splits parts even when every side is shorter than the minimum."""
    text = "abcdefghij" * 3
    segments = (
        BlockSegment(0, 10, 0),
        BlockSegment(10, 20, 1),
        BlockSegment(20, 30, 2),
    )
    block = _block(text, segments)

    changed = compose_parts(
        block,
        (
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
        ),
    )
    assert changed == (
        LayoutPart(0, 10, BlockTreatment.TRANSLATE, (0,)),
        LayoutPart(10, 30, BlockTreatment.READ_AS_IS, (1, 2)),
    )
    assert all(part.end - part.start < PART_MIN_CHARS for part in changed)

    unset = compose_parts(block, (None, None, BlockTreatment.GLOSSARY_ONLY))
    assert unset == (
        LayoutPart(0, 20, None, (0, 1)),
        LayoutPart(20, 30, BlockTreatment.GLOSSARY_ONLY, (2,)),
    )

    kept = compose_parts(
        block,
        (
            BlockTreatment.PRESERVE_ONLY,
            BlockTreatment.DROP_WITH_WARNING,
            BlockTreatment.DROP_WITH_WARNING,
        ),
    )
    assert kept == (
        LayoutPart(0, 10, BlockTreatment.PRESERVE_ONLY, (0,)),
        LayoutPart(10, 30, BlockTreatment.DROP_WITH_WARNING, (1, 2)),
    )


def test_translated_run_parts_within_250_350() -> None:
    """A translate run is grouped near 300 characters and stays inside 250..350."""
    assert (PART_MIN_CHARS, PART_TARGET_CHARS, PART_MAX_CHARS) == (250, 300, 350)
    text, segments = _spans(100, 9)
    block = _block(text, segments)
    snapshot = (block.id, block.text, block.segments)

    parts = compose_parts(block, (BlockTreatment.TRANSLATE,) * 9)

    assert parts == (
        LayoutPart(0, 300, BlockTreatment.TRANSLATE, (0, 1, 2)),
        LayoutPart(300, 600, BlockTreatment.TRANSLATE, (3, 4, 5)),
        LayoutPart(600, 900, BlockTreatment.TRANSLATE, (6, 7, 8)),
    )
    assert all(PART_MIN_CHARS <= part.end - part.start <= PART_MAX_CHARS for part in parts)
    assert (block.id, block.text, block.segments) == snapshot


def test_oversized_segment_stays_whole() -> None:
    """One segment above the maximum stays whole; shorter neighbours may still join."""
    text = ("a" * 100) + ("b" * 400) + ("c" * 120) + ("d" * 120)
    segments = (
        BlockSegment(0, 100, 0),
        BlockSegment(100, 500, 1),
        BlockSegment(500, 620, 2),
        BlockSegment(620, 740, 3),
    )
    block = _block(text, segments)

    parts = compose_parts(block, (BlockTreatment.TRANSLATE,) * 4)

    assert parts == (
        LayoutPart(0, 100, BlockTreatment.TRANSLATE, (0,)),
        LayoutPart(100, 500, BlockTreatment.TRANSLATE, (1,)),
        LayoutPart(500, 740, BlockTreatment.TRANSLATE, (2, 3)),
    )
    assert parts[1].end - parts[1].start > PART_MAX_CHARS
    assert parts[2].end - parts[2].start < PART_MIN_CHARS


def test_translate_ties_prefer_fewer_parts_then_earlier_cut() -> None:
    """Equal cost keeps fewer parts; a remaining tie cuts at the earlier boundary."""
    short = _block(
        "abcde" + "0123456789",
        (BlockSegment(0, 5, 0), BlockSegment(5, 15, 1)),
    )
    fewer = compose_parts(
        short,
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        target_chars=10,
        min_chars=1,
        max_chars=20,
    )
    assert fewer == (LayoutPart(0, 15, BlockTreatment.TRANSLATE, (0, 1)),)

    text, segments = _spans(100, 3)
    earlier = compose_parts(
        _block(text, segments),
        (BlockTreatment.TRANSLATE,) * 3,
        target_chars=200,
        min_chars=50,
        max_chars=250,
    )
    assert earlier == (
        LayoutPart(0, 100, BlockTreatment.TRANSLATE, (0,)),
        LayoutPart(100, 300, BlockTreatment.TRANSLATE, (1, 2)),
    )


def test_treatment_count_must_match_segment_spans() -> None:
    """The treatment sequence is one entry per span, including the default span."""
    block = _block("Hello")
    with pytest.raises(ValueError, match="число режимов 0 не совпадает с числом сегментов 1"):
        compose_parts(block, ())
    with pytest.raises(ValueError, match="число режимов 2 не совпадает с числом сегментов 1"):
        compose_parts(block, (BlockTreatment.TRANSLATE, None))
