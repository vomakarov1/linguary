"""Reading units inside an EPUB chapter.

One unit is a slice of the source block, its translation, and its notes.
The writer passes units from ``request.layout`` and does not build them here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from lxml import etree

from linguary.adapters.epub.inline import local_name
from linguary.adapters.epub.writer import (
    _EPUB_NAMESPACE,
    _SectionState,
    _anchors,
    _append_range,
    _append_text,
    _element,
    _nestable,
    _sub,
    _target_attributes,
    _unique_id,
)
from linguary.application.layout.notes import LayoutNote
from linguary.application.layout.units import LayoutUnit, NumberedNote
from linguary.domain import (
    BlockType,
    DocumentBlock,
    InlineAnnotation,
    build_language_note_id,
    translation_segments,
)

_REFERENCE_COLON = "Языковая справка:"
_REFERENCE = "Языковая справка"
_UNPREPARED = "Помощь пока не подготовлена."
_UNPREPARED_BARE = "Помощь пока не подготовлена"
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def source_only_unit(block: DocumentBlock) -> LayoutUnit:
    """One source span of the whole block, with no translation and no notes."""
    return LayoutUnit(
        block_id=block.id,
        start=0,
        end=len(block.text),
        translation=None,
        speaker=None,
        notes=(),
        continues=False,
        ends_paragraph=True,
        lines=block.type is BlockType.POEM,
    )


def render_unit(
    host: etree._Element,
    block: DocumentBlock,
    unit: LayoutUnit,
    *,
    state: _SectionState,
    source_language: str,
    target_language: str,
) -> None:
    """Append one reading unit to ``host``."""
    classes = ["lg-unit"]
    if unit.continues:
        classes.append("lg-cont")
    if unit.ends_paragraph:
        classes.append("lg-end")
    reading = _sub(host, "div", {"class": " ".join(classes)})
    heading = block.type is BlockType.HEADING
    if not heading and not unit.continues:
        _place_anchors(reading, block, state)
    source_attributes = {"lang": source_language}
    if not heading:
        source_attributes["class"] = "lg-src"
    source = _sub(reading, _source_tag(block), source_attributes)
    _append_replaced_content(source, block, unit)
    _append_range(
        source,
        block.text,
        _clipped(block.inline_annotations, unit.start, unit.end),
        start=unit.start,
        end=unit.end,
        state=state,
    )
    prepared = _prepare_notes(unit, state)
    _insert_markers(source, unit, prepared)
    if heading and not unit.continues:
        _place_anchors(source, block, state)
    if unit.translation and unit.translation.strip():
        _append_translation(
            reading, block, unit, state=state, target_language=target_language
        )
    if prepared:
        _append_notes(
            reading,
            prepared,
            source_language=source_language,
            target_language=target_language,
        )


def _source_tag(block: DocumentBlock) -> str:
    """Source element. A heading keeps h1–h6, and an unknown level is h2."""
    if block.type is not BlockType.HEADING:
        return "p"
    tag = block.attributes.get("tag", "")
    return tag if tag in _HEADING_TAGS else "h2"


def _place_anchors(
    reading: etree._Element, block: DocumentBlock, state: _SectionState
) -> None:
    """First unused anchor becomes this element's id. The rest are empty spans."""
    pending = [
        anchor
        for anchor in _anchors(block)
        if anchor and anchor not in state.used_ids
    ]
    if not pending:
        return
    if reading.get("id") is None:
        reading.set("id", pending[0])
        state.used_ids.add(pending[0])
        pending = pending[1:]
    for anchor in pending:
        _sub(reading, "span", {"id": anchor})
        state.used_ids.add(anchor)


def _append_replaced_content(
    source: etree._Element, block: DocumentBlock, unit: LayoutUnit
) -> None:
    """Images and page breaks travel with the first unit of the block."""
    if unit.continues:
        return
    if block.type is BlockType.IMAGE:
        _sub(
            source,
            "img",
            {
                "src": block.attributes.get("src", ""),
                "alt": block.attributes.get("alt", ""),
            },
        )
    elif block.type is BlockType.PAGE_BREAK:
        _sub(source, "span", {f"{{{_EPUB_NAMESPACE}}}type": "pagebreak"})


def _clipped(
    annotations: Sequence[InlineAnnotation], start: int, end: int
) -> tuple[InlineAnnotation, ...]:
    """Annotations overlapping ``[start, end)``, clipped to that slice."""
    kept: list[InlineAnnotation] = []
    for annotation in annotations:
        left = max(annotation.start, start)
        right = min(annotation.end, end)
        if right <= left:
            continue
        if left == annotation.start and right == annotation.end:
            kept.append(annotation)
        else:
            kept.append(replace(annotation, start=left, end=right))
    return _nestable(tuple(kept))


def _prepare_notes(
    unit: LayoutUnit, state: _SectionState
) -> tuple[tuple[NumberedNote, tuple[str, ...], str], ...]:
    """Visible notes in unit order, each with a document id."""
    prepared: list[tuple[NumberedNote, tuple[str, ...], str]] = []
    for numbered in unit.notes:
        body = _visible_body(numbered.note)
        if not (numbered.note.labels or numbered.note.term or numbered.note.ipa or body):
            continue
        state.notes += 1
        if state.number >= 1:
            candidate = build_language_note_id(
                chapter_number=state.number, ordinal=state.notes
            )
        else:
            candidate = f"lg-n{state.notes}"
        prepared.append(
            (numbered, body, _unique_id(candidate, state.used_ids))
        )
    return tuple(prepared)


def _visible_body(note: LayoutNote) -> tuple[str, ...]:
    """Note text without a leading reference prefix or an unprepared sentence."""
    pieces: list[str] = []
    for part in note.body:
        text = part.strip()
        while text.startswith(_REFERENCE_COLON) or text.startswith(_REFERENCE):
            if text.startswith(_REFERENCE_COLON):
                text = text[len(_REFERENCE_COLON) :].strip()
            else:
                text = text[len(_REFERENCE) :].strip()
        if text in {_UNPREPARED, _UNPREPARED_BARE}:
            continue
        if text:
            pieces.append(text)
    return tuple(pieces)


def _append_translation(
    reading: etree._Element,
    block: DocumentBlock,
    unit: LayoutUnit,
    *,
    state: _SectionState,
    target_language: str,
) -> None:
    """Translation paragraph. The speaker name is its own span."""
    paragraph = _sub(
        reading, "p", {"class": "lg-tr", "lang": target_language}
    )
    if unit.speaker:
        speaker = _sub(paragraph, "span", {"class": "lg-speaker"})
        speaker.text = unit.speaker
        speaker.tail = " "
    assert unit.translation is not None
    for segment in translation_segments(unit.translation, block):
        if isinstance(segment, str):
            _append_text(paragraph, segment)
            continue
        link = _sub(paragraph, "a", _target_attributes(segment, state))
        link.text = block.text[segment.start : segment.end]


def _append_notes(
    reading: etree._Element,
    prepared: Sequence[tuple[NumberedNote, tuple[str, ...], str]],
    *,
    source_language: str,
    target_language: str,
) -> None:
    """One list: number, kind labels, term, pronunciation, then the text."""
    listing = _sub(
        reading, "ol", {"class": "lg-notes", "lang": target_language}
    )
    for numbered, body, note_id in prepared:
        item = _sub(listing, "li", {"id": note_id})
        note = numbered.note
        if numbered.number is not None:
            number = _sub(item, "span", {"class": "lg-num"})
            number.text = str(numbered.number)
            number.tail = " "
        for label in note.labels:
            kind = _sub(item, "span", {"class": "lg-kind"})
            kind.text = label
            kind.tail = " "
        if note.term:
            term = _sub(item, "span", {"class": "lg-term", "lang": source_language})
            term.text = note.term
            term.tail = " "
        if note.ipa:
            pronunciation = _sub(item, "span", {"class": "lg-ipa"})
            pronunciation.text = _slashed(note.ipa)
            pronunciation.tail = (
                f" ({note.variant.upper()})" if note.variant else " "
            )
        for piece in body:
            _append_text(_sub(item, "p"), piece)


def _slashed(ipa: str) -> str:
    """IPA wrapped in slashes once."""
    text = ipa.strip()
    if len(text) >= 2 and text.startswith("/") and text.endswith("/"):
        return text
    return f"/{text.strip('/')}/"


def _insert_markers(
    source: etree._Element,
    unit: LayoutUnit,
    prepared: Sequence[tuple[NumberedNote, tuple[str, ...], str]],
) -> None:
    """Digit links after each numbered marker that falls inside this slice."""
    markers: list[tuple[int, int, int, str]] = []
    for index, (numbered, _, note_id) in enumerate(prepared):
        number = numbered.number
        if number is None or numbered.note.marker_end is None:
            continue
        offset = numbered.note.marker_end
        if unit.start < offset <= unit.end:
            markers.append((offset, index, number, note_id))
    for offset, _, number, note_id in sorted(markers, key=lambda item: (-item[0], item[1])):
        _insert_marker(source, offset - unit.start, number, note_id)


def _insert_marker(
    source: etree._Element, position: int, number: int | None, note_id: str
) -> None:
    """Place ``a.lg-ref`` at ``position`` without nesting it inside a link."""
    if number is None:
        return
    link = _element("a", {"class": "lg-ref", "href": f"#{note_id}"})
    _sub(link, "sup").text = str(number)
    slots, length = _text_slots(source)
    if position >= length:
        source.append(link)
        return
    for node, is_tail, lower, upper in slots:
        if position != lower and not lower < position <= upper:
            continue
        anchor = _enclosing_anchor(node, is_tail, source)
        if anchor is not None:
            link.tail = anchor.tail
            anchor.tail = None
            anchor.addnext(link)
            return
        _split_insert(node, is_tail, position - lower, link)
        return
    source.append(link)


def _split_insert(
    node: etree._Element, is_tail: bool, offset: int, link: etree._Element
) -> None:
    """Insert ``link`` after ``offset`` characters of this text slot."""
    value = (node.tail if is_tail else node.text) or ""
    head, rest = value[:offset], value[offset:]
    link.tail = rest or None
    if is_tail:
        node.tail = head or None
        node.addnext(link)
        return
    node.text = head or None
    node.insert(0, link)


def _enclosing_anchor(
    node: etree._Element, is_tail: bool, root: etree._Element
) -> etree._Element | None:
    """Outermost link that contains this slot, if the slot is inside one."""
    cursor = node.getparent() if is_tail else node
    found: etree._Element | None = None
    while cursor is not None and cursor is not root:
        if local_name(cursor) == "a":
            found = cursor
        cursor = cursor.getparent()
    return found


def _text_slots(
    element: etree._Element,
) -> tuple[list[tuple[etree._Element, bool, int, int]], int]:
    """Text ranges in source order. A line break counts as one character.

    Marker digits are not part of the source, so an existing ``a.lg-ref`` is skipped.
    """
    slots: list[tuple[etree._Element, bool, int, int]] = []
    position = 0

    def collect(node: etree._Element) -> None:
        nonlocal position
        if node is not element and _is_marker(node):
            return
        if node.text:
            slots.append((node, False, position, position + len(node.text)))
            position += len(node.text)
        for child in list(node):
            if _is_marker(child):
                if child.tail:
                    slots.append((child, True, position, position + len(child.tail)))
                    position += len(child.tail)
                continue
            if local_name(child) == "br":
                position += 1
            else:
                collect(child)
            if child.tail:
                slots.append((child, True, position, position + len(child.tail)))
                position += len(child.tail)

    collect(element)
    return slots, position


def _is_marker(node: etree._Element) -> bool:
    return "lg-ref" in (node.get("class") or "").split()
