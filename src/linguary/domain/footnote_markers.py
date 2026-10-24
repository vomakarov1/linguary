"""Book footnote references within block translations (PRD § 7, § 20).

Translation is stored as plain text without inline markup. An original footnote reference
placed mid-paragraph (`…her face.<a epub:type="noteref">1</a>`) cannot be directly
represented in such text: the model sees its number as a plain digit and carries it
over as a bare number that links nowhere.

Therefore, footnote references are passed to the model as `[^N]` markers, where `N` is
the 1-based ordinal of the footnote reference in the source block, and the model preserves
the marker in the translation. The marker references the markup of the source block
rather than duplicating it: the footnote target and visible reference text are derived
from the original, ensuring that the link in the exported book points to the same destination.
"""

from __future__ import annotations

import re

from linguary.domain.document import DocumentBlock, InlineAnnotation, InlineAnnotationType

#: Footnote reference marker in translated text. Leading space belongs to the marker:
#: stripping the marker avoids leaving a dangling space before punctuation.
_MARKER = re.compile(r" ?\[\^(\d+)\]")


def footnote_marker(ordinal: int) -> str:
    """Marker for the `ordinal`-th footnote reference of a block."""
    return f"[^{ordinal}]"


def footnote_references(block: DocumentBlock) -> tuple[InlineAnnotation, ...]:
    """Block footnote references in text order: `[^1]` is the first among them."""
    return tuple(
        sorted(
            (
                annotation
                for annotation in block.inline_annotations
                if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE
            ),
            key=lambda annotation: annotation.start,
        )
    )


def translation_segments(
    text: str, block: DocumentBlock
) -> tuple[str | InlineAnnotation, ...]:
    """Translation parsed into text chunks and source block footnote references.

    A marker lacking a corresponding reference in the block is dropped: a hallucinated
    reference would lead nowhere.
    """
    references = footnote_references(block)
    segments: list[str | InlineAnnotation] = []
    position = 0
    for match in _MARKER.finditer(text):
        segments.append(text[position : match.start()])
        reference = _reference(references, match.group(1))
        if reference is not None:
            if match.group(0).startswith(" "):
                segments.append(" ")
            segments.append(reference)
        position = match.end()
    segments.append(text[position:])
    return tuple(segment for segment in segments if segment != "")


def known_markers_only(text: str, block: DocumentBlock) -> str:
    """Translation retaining only markers of existing block references."""
    references = footnote_references(block)
    return _MARKER.sub(
        lambda match: match.group(0)
        if _reference(references, match.group(1)) is not None
        else "",
        text,
    )


def without_footnote_markers(text: str) -> str:
    """Text without footnote markers: for fields where references are not needed."""
    return _MARKER.sub("", text)


def with_reference_labels(text: str, block: DocumentBlock) -> str:
    """Translation with markers replaced by visible reference text, matching source block text."""
    return "".join(
        segment
        if isinstance(segment, str)
        else block.text[segment.start : segment.end]
        for segment in translation_segments(text, block)
    )


def _reference(
    references: tuple[InlineAnnotation, ...], ordinal: str
) -> InlineAnnotation | None:
    """Footnote reference by ordinal from marker or `None` if absent."""
    index = int(ordinal) - 1
    return references[index] if 0 <= index < len(references) else None


__all__ = [
    "footnote_marker",
    "footnote_references",
    "known_markers_only",
    "translation_segments",
    "with_reference_labels",
    "without_footnote_markers",
]
