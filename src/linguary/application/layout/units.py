"""Reading units projected from display parts of one block.

A unit is one part plus the translation of its translated segments, the
speaker of a dialogue line that starts inside it, and the notes that belong
to that part. Nothing is persisted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from linguary.application.layout.notes import LayoutNote
from linguary.application.layout.parts import LayoutPart, compose_parts
from linguary.domain import BlockProcessingResult, BlockTreatment, BlockType, DocumentBlock


@dataclass(frozen=True, slots=True)
class NumberedNote:
    """One note and its number inside this block.

    `number` is none when the note has no marker. A speaker line shown because
    the unit has no translation is also unnumbered.
    """

    number: int | None
    note: LayoutNote


@dataclass(frozen=True, slots=True)
class LayoutUnit:
    """One display part of a block and the reading layers under it."""

    block_id: str
    start: int
    end: int
    translation: str | None
    speaker: str | None
    notes: tuple[NumberedNote, ...]
    continues: bool
    ends_paragraph: bool
    lines: bool


def block_units(
    block: DocumentBlock,
    result: BlockProcessingResult | None,
    treatments: Sequence[BlockTreatment | None],
    notes: Sequence[LayoutNote],
    speakers: Mapping[tuple[int, int], str],
) -> tuple[LayoutUnit, ...]:
    """Project `block` into reading units.

    `treatments` has one entry per `block.segment_spans`; a different length
    raises `ValueError`. Parts come from `compose_parts`. The only unit does
    not continue and ends the paragraph. Each later unit continues, and only
    the last of them ends the paragraph.

    A note with `marker_end` is numbered from 1 in increasing marker order
    inside this call and placed on the part where `start < marker_end <= end`.
    A note without a marker stays on the last part and is not numbered. A
    marker that sits on no part stays on the last part and keeps its number.

    A part translation joins the non-empty translations of that part's
    `TRANSLATE` segments, the same way `rebuild_translation` joins them. No
    such text, including when `result` is none, leaves the translation empty.
    `translated_text` is not copied onto a part.

    `speakers` maps a dialogue line `(line_start, line_end)` to a name. The
    unit containing `line_start` (`start <= line_start < end`) records the
    earliest such name and does not write it into the translation. When that
    unit has no translation, an unnumbered note `Говорит {name}` with origin
    `speaker` is appended. `lines` is true only for `BlockType.POEM`, on
    every unit of that block.
    """
    chosen = tuple(treatments)
    spans = block.segment_spans
    if len(chosen) != len(spans):
        raise ValueError(
            f"число режимов {len(chosen)} не совпадает с числом сегментов {len(spans)}"
        )
    parts = compose_parts(block, chosen)
    note_buckets = _notes_by_part(parts, notes)
    speaker_names = _speakers_by_part(parts, speakers)
    poem = block.type is BlockType.POEM
    last = len(parts) - 1
    units: list[LayoutUnit] = []
    for index, part in enumerate(parts):
        translation = _part_translation(part, result, chosen)
        speaker = speaker_names[index]
        part_notes = list(note_buckets[index])
        if speaker is not None and translation is None:
            part_notes.append(_speaker_note(speaker))
        units.append(
            LayoutUnit(
                block_id=block.id,
                start=part.start,
                end=part.end,
                translation=translation,
                speaker=speaker,
                notes=tuple(part_notes),
                continues=index > 0,
                ends_paragraph=index == last,
                lines=poem,
            )
        )
    return tuple(units)


def _part_translation(
    part: LayoutPart,
    result: BlockProcessingResult | None,
    treatments: tuple[BlockTreatment | None, ...],
) -> str | None:
    """Join translations of this part's `TRANSLATE` segments. None when absent."""
    if result is None:
        return None
    texts = result.segment_translations
    pieces: list[str] = []
    for index in part.segments:
        if treatments[index] != BlockTreatment.TRANSLATE:
            continue
        if index >= len(texts):
            continue
        text = texts[index]
        if text and text.strip():
            pieces.append(text)
    if not pieces:
        return None
    return " ".join(pieces)


def _notes_by_part(
    parts: Sequence[LayoutPart],
    notes: Sequence[LayoutNote],
) -> list[list[NumberedNote]]:
    """Number marked notes from 1 and bucket them by the owning part."""
    buckets: list[list[NumberedNote]] = [[] for _ in parts]
    if not parts:
        return buckets
    for item in _numbered(notes):
        buckets[_part_index(parts, item.note.marker_end)].append(item)
    for bucket in buckets:
        bucket.sort(key=_note_order)
    return buckets


def _numbered(notes: Sequence[LayoutNote]) -> tuple[NumberedNote, ...]:
    """Assign 1..n by increasing marker. Notes without a marker stay unnumbered."""
    marked = sorted(
        (index for index, note in enumerate(notes) if note.marker_end is not None),
        key=lambda index: (notes[index].marker_end, index),
    )
    numbers = {index: number for number, index in enumerate(marked, start=1)}
    return tuple(NumberedNote(numbers.get(index), note) for index, note in enumerate(notes))


def _part_index(parts: Sequence[LayoutPart], marker_end: int | None) -> int:
    """Part that ends at `marker_end`. No marker, or no match, is the last part."""
    if marker_end is None:
        return len(parts) - 1
    for index, part in enumerate(parts):
        if part.start < marker_end <= part.end:
            return index
    return len(parts) - 1


def _note_order(item: NumberedNote) -> tuple[bool, int]:
    """Marked notes by offset, then unmarked notes. Equal keys keep source order."""
    marker = item.note.marker_end
    if marker is None:
        return (True, 0)
    return (False, marker)


def _speakers_by_part(
    parts: Sequence[LayoutPart],
    speakers: Mapping[tuple[int, int], str],
) -> list[str | None]:
    """Name of the earliest line that starts inside each part.

    `line_end` does not choose the part. It only orders two lines that start
    at the same offset. A later line in the same part does not replace it.
    """
    chosen: list[str | None] = [None] * len(parts)
    ranked = sorted(speakers.items(), key=lambda item: (item[0][0], item[0][1]))
    for (line_start, _line_end), name in ranked:
        for index, part in enumerate(parts):
            if part.start <= line_start < part.end:
                if chosen[index] is None:
                    chosen[index] = name
                break
    return chosen


def _speaker_note(name: str) -> NumberedNote:
    """Unnumbered note used when a speaker's unit has no translation."""
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
