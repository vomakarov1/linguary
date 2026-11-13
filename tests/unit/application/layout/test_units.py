"""Reading units projected from one block's display parts."""

from __future__ import annotations

import pytest

from linguary.application.layout.notes import LayoutNote
from linguary.application.layout.units import LayoutUnit, NumberedNote, block_units
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    SourceReference,
)
from linguary.domain.treatment import strongest_treatment


def _block(
    text: str,
    segments: tuple[BlockSegment, ...] = (),
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
) -> DocumentBlock:
    """A block whose only varying input is text, segments, and type."""
    return DocumentBlock.create(
        document_id="book-1",
        section_id="section-1",
        type=block_type,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="chapter.xhtml",
            structural_path="/body/p[1]",
        ),
        segments=segments,
    )


def _spans(*lengths: int) -> tuple[str, tuple[BlockSegment, ...]]:
    """Contiguous spans of the given lengths."""
    segments: list[BlockSegment] = []
    offset = 0
    for index, length in enumerate(lengths):
        segments.append(BlockSegment(offset, offset + length, index))
        offset += length
    return "a" * offset, tuple(segments)


def _note(marker_end: int | None, body: str) -> LayoutNote:
    """A note whose only varying input is the marker and the body."""
    return LayoutNote(
        labels=(),
        term=None,
        ipa=None,
        variant=None,
        body=(body,),
        marker_end=marker_end,
        origin="language_note",
        difficulty_kinds=(),
    )


def _speaker_note(name: str) -> NumberedNote:
    """The unnumbered note used when a speaker has no translation."""
    return NumberedNote(
        None,
        LayoutNote(
            labels=(),
            term=None,
            ipa=None,
            variant=None,
            body=(f"Говорит {name}",),
            marker_end=None,
            origin="speaker",
            difficulty_kinds=(),
        ),
    )


def _result(
    block: DocumentBlock,
    treatments: tuple[BlockTreatment, ...],
    translations: tuple[str | None, ...],
) -> BlockProcessingResult:
    """A result whose block translation is the join of its segment translations."""
    pieces = [text for text in translations if text and text.strip()]
    translated = " ".join(pieces) if pieces else None
    return BlockProcessingResult(
        block.id,
        strongest_treatment(treatments),
        translated_text=translated,
        segment_treatments=treatments,
        segment_translations=translations,
    )


def test_split_block_marks_continuation_and_end() -> None:
    """One part ends the paragraph. Later parts continue, and only the last ends it."""
    text, segments = _spans(10, 10, 10, 10)
    block = _block(text, segments)
    snapshot = (block.id, block.text, block.segments)

    units = block_units(
        block,
        None,
        (
            BlockTreatment.TRANSLATE,
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.GLOSSARY_ONLY,
        ),
        (),
        {},
    )

    assert units == (
        LayoutUnit(block.id, 0, 20, None, None, (), False, False, False),
        LayoutUnit(block.id, 20, 30, None, None, (), True, False, False),
        LayoutUnit(block.id, 30, 40, None, None, (), True, True, False),
    )
    assert (block.id, block.text, block.segments) == snapshot

    whole = _block("Hello.")
    assert block_units(whole, None, (None,), (), {}) == (
        LayoutUnit(whole.id, 0, len(whole.text), None, None, (), False, True, False),
    )
    with pytest.raises(ValueError, match="число режимов 0 не совпадает с числом сегментов 1"):
        block_units(whole, None, (), (), {})


def test_translation_only_under_translated_part() -> None:
    """A part joins its own translated segments and does not copy the block translation."""
    text, segments = _spans(10, 10, 10)
    block = _block(text, segments)
    treatments = (
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
    )
    result = _result(block, treatments, ("Альфа.", None, "Гамма."))
    assert result.translated_text == "Альфа. Гамма."

    split = block_units(block, result, treatments, (), {})

    assert [(unit.start, unit.end, unit.translation) for unit in split] == [
        (0, 10, "Альфа."),
        (10, 20, None),
        (20, 30, "Гамма."),
    ]

    joined_text, joined_segments = _spans(20, 20, 20)
    joined_block = _block(joined_text, joined_segments)
    joined_treatments = (
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
    )
    joined = _result(joined_block, joined_treatments, ("Раз.", "Два.", None))
    assert joined.translated_text == "Раз. Два."
    assert [
        (unit.start, unit.end, unit.translation)
        for unit in block_units(joined_block, joined, joined_treatments, (), {})
    ] == [
        (0, 40, "Раз. Два."),
        (40, 60, None),
    ]

    legacy_block = _block("Legacy sentence.")
    legacy = BlockProcessingResult(
        legacy_block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Целиком.",
    )
    legacy_units = block_units(legacy_block, legacy, (BlockTreatment.TRANSLATE,), (), {})
    assert legacy.translated_text == "Целиком."
    assert legacy.segment_translations == ()
    assert legacy_units[0].translation is None


def test_note_in_marker_part_numbering_restarts() -> None:
    """The first marked note of this block is 1, ordered by marker, even on a later part."""
    text, segments = _spans(10, 10, 10)
    block = _block(text, segments)
    treatments = (
        BlockTreatment.READ_AS_IS,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
    )
    loose = _note(None, "loose")
    third = _note(28, "third")
    second = _note(18, "second")
    first = _note(12, "first")

    units = block_units(block, None, treatments, (loose, third, second, first), {})

    assert [unit.notes for unit in units] == [
        (),
        (NumberedNote(1, first), NumberedNote(2, second)),
        (NumberedNote(3, third), NumberedNote(None, loose)),
    ]

    boundary = _note(10, "boundary")
    inside = _note(11, "inside")
    placed = block_units(block, None, treatments, (inside, boundary), {})
    assert [unit.notes for unit in placed] == [
        (NumberedNote(1, boundary),),
        (NumberedNote(2, inside),),
        (),
    ]


def test_inferred_speaker_prefixes_matching_dialogue_line() -> None:
    """The unit containing the line start keeps the name off the translation."""
    text, segments = _spans(200, 200, 200)
    block = _block(text, segments)
    treatments = (
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    )
    result = _result(block, treatments, ("Первая.", "Вторая.", "Третья."))

    units = block_units(
        block,
        result,
        treatments,
        (),
        {(50, 500): "Мышь", (200, 250): "Алиса"},
    )

    assert units == (
        LayoutUnit(block.id, 0, 200, "Первая.", "Мышь", (), False, False, False),
        LayoutUnit(block.id, 200, 400, "Вторая.", "Алиса", (), True, False, False),
        LayoutUnit(block.id, 400, 600, "Третья.", None, (), True, True, False),
    )


def test_speaker_without_translation_unnumbered_note() -> None:
    """No translation adds «Говорит …» unnumbered. A translated line does not."""
    text, segments = _spans(10, 10)
    block = _block(text, segments)
    treatments = (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS)
    result = _result(block, treatments, ("Реплика.", None))
    loose = _note(None, "loose")
    marked = _note(4, "marked")

    units = block_units(
        block,
        result,
        treatments,
        (loose, marked),
        {(2, 9): "Алиса", (12, 19): "Шляпник"},
    )

    assert units == (
        LayoutUnit(
            block.id,
            0,
            10,
            "Реплика.",
            "Алиса",
            (NumberedNote(1, marked),),
            False,
            False,
            False,
        ),
        LayoutUnit(
            block.id,
            10,
            20,
            None,
            "Шляпник",
            (NumberedNote(None, loose), _speaker_note("Шляпник")),
            True,
            True,
            False,
        ),
    )


def test_poem_unit_keeps_lines() -> None:
    """Every unit of a poem keeps lines. The same text as a paragraph does not."""
    text, segments = _spans(10, 10)
    treatments = (BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE)
    poem = _block(text, segments, block_type=BlockType.POEM)
    assert poem.type is BlockType.POEM
    result = _result(poem, treatments, (None, "Строка."))

    units = block_units(poem, result, treatments, (), {})

    assert [(unit.lines, unit.translation) for unit in units] == [
        (True, None),
        (True, "Строка."),
    ]
    paragraph = _block(text, segments, block_type=BlockType.PARAGRAPH)
    plain = block_units(
        paragraph,
        _result(paragraph, treatments, (None, "Строка.")),
        treatments,
        (),
        {},
    )
    assert [unit.lines for unit in plain] == [False, False]
