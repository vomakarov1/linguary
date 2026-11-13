"""Review list for one chapter: earlier hints brought back on a schedule.

A hint qualifies when its content chapter is 1, 3, or 7 before the current
one and the expression has a single translation in the book. Nothing is
stored and no model is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from linguary.application.explained_expressions import _same_word, expression_words
from linguary.application.layout.who_is_who import _content_chapter_numbers
from linguary.domain import BlockProcessingResult, CanonicalDocument, DocumentSection

REVIEW_DISTANCES = (1, 3, 7)
REVIEW_LIMITS: dict[str, int] = {"a2": 8, "b1": 6, "b2": 4, "c1": 3}


@dataclass(frozen=True, slots=True)
class ReviewEntry:
    """One expression to repeat, from the earliest scheduled chapter."""

    expression: str
    translation: str
    first_chapter: int


@dataclass(frozen=True, slots=True)
class _Hint:
    """One glossary hint in reading order."""

    chapter: int | None
    appearance: int
    original: str
    translation: str
    words: tuple[str, ...]


def chapter_review(
    document: CanonicalDocument,
    results: Mapping[str, BlockProcessingResult],
    *,
    section_id: str,
    reader_level: str | None,
) -> tuple[ReviewEntry, ...]:
    """Scheduled hints for this chapter, capped by `reader_level`.

    Nearer distances come first. Inside a chapter, appearance order is kept.
    The same expression scheduled from more than one distance is one entry,
    and `first_chapter` is the earliest of those chapters. An expression
    already present as a hint or language note in this chapter is excluded.
    An unknown or missing reader level returns an empty tuple.
    """
    if not isinstance(reader_level, str):
        return ()
    limit = REVIEW_LIMITS.get(reader_level.strip().casefold())
    if limit is None:
        return ()
    section = document.section(section_id)
    current = _content_chapter_numbers(document).get(section_id)
    if current is None:
        return ()
    hints = _hints(document, results)
    noted = _noted_words(section, results)
    chosen: list[tuple[int, int, ReviewEntry]] = []
    for indexes in _groups(hints):
        translations = {_translation_key(hints[index].translation) for index in indexes}
        if len(translations) != 1:
            continue
        scheduled = [hints[index] for index in indexes if _scheduled(hints[index], current)]
        if not scheduled or _already_noted(indexes, hints, noted):
            continue
        earliest = min(scheduled, key=lambda item: (item.chapter, item.appearance))
        nearest = min(scheduled, key=lambda item: (current - _chapter(item), item.appearance))
        chosen.append((
            current - _chapter(nearest),
            nearest.appearance,
            ReviewEntry(earliest.original, earliest.translation, _chapter(earliest)),
        ))
    chosen.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in chosen[:limit])


def _chapter(hint: _Hint) -> int:
    """Content chapter of a hint that was already checked to have one."""
    if hint.chapter is None:
        raise ValueError("у подсказки нет номера содержательной главы")
    return hint.chapter


def _hints(
    document: CanonicalDocument,
    results: Mapping[str, BlockProcessingResult],
) -> tuple[_Hint, ...]:
    """Glossary hints in reading order. Non-content chapters have no number."""
    numbers = _content_chapter_numbers(document)
    found: list[_Hint] = []
    for section in document.ordered_sections:
        chapter = numbers.get(section.id)
        appearance = 0
        for block in section.ordered_blocks:
            result = results.get(block.id)
            if result is None:
                continue
            for hint in result.glossary_hints:
                words = expression_words(hint.original)
                if not words:
                    continue
                found.append(_Hint(chapter, appearance, hint.original, hint.translation, words))
                if chapter is not None:
                    appearance += 1
    return tuple(found)


def _noted_words(
    section: DocumentSection,
    results: Mapping[str, BlockProcessingResult],
) -> tuple[tuple[str, ...], ...]:
    """Word forms of hints and language-note excerpts in this section."""
    found: list[tuple[str, ...]] = []
    for block in section.ordered_blocks:
        result = results.get(block.id)
        if result is None:
            continue
        texts = [hint.original for hint in result.glossary_hints]
        texts.extend(note.original_excerpt for note in result.language_notes)
        for text in texts:
            words = expression_words(text)
            if words:
                found.append(words)
    return tuple(found)


def _scheduled(hint: _Hint, current: int) -> bool:
    """Whether this hint's chapter is 1, 3, or 7 content chapters back."""
    return hint.chapter is not None and (current - hint.chapter) in REVIEW_DISTANCES


def _already_noted(
    indexes: Sequence[int],
    hints: Sequence[_Hint],
    noted: Sequence[tuple[str, ...]],
) -> bool:
    """Whether any form of this expression is already noted in the chapter."""
    return any(
        _same_words(hints[index].words, words)
        for index in indexes
        for words in noted
    )


def _groups(hints: Sequence[_Hint]) -> tuple[tuple[int, ...], ...]:
    """Indexes of hints that are the same expression, in first-seen order."""
    parent = list(range(len(hints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(hints)):
        for right in range(left + 1, len(hints)):
            if _same_words(hints[left].words, hints[right].words):
                parent[find(right)] = find(left)
    grouped: dict[int, list[int]] = {}
    for index in range(len(hints)):
        grouped.setdefault(find(index), []).append(index)
    return tuple(tuple(indexes) for indexes in grouped.values())


def _same_words(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Whether two expressions match word for word, including inflections.

    The stem rule lives in `explained_expressions` and is not copied here.
    """
    return bool(left) and len(left) == len(right) and all(
        _same_word(first, second) for first, second in zip(left, right, strict=True)
    )


def _translation_key(text: str) -> tuple[str, ...]:
    """Normalized translation. Punctuation and case do not make a second sense."""
    words = expression_words(text)
    if words:
        return words
    return (text.casefold().strip(),)
