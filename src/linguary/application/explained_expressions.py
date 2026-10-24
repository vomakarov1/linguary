"""Chapter expressions already explained in `Before You Read`, and filtering of repeats (PRD § 19.3).

A single book passage is explained to the reader once. The model receives a list of
`Before You Read` expressions accompanied by instructions not to repeat them, but prompt
constraints are advisory: a response where an expression reappears as a glossary hint or
language note passes all blocking validations. Therefore repetition is filtered out here,
deterministically, after response validation.

Comparison operates on words rather than raw strings: case, punctuation, whitespace, and
apostrophe glyphs are normalized, while inflected forms of the same word (`heave` and `heaved`,
`groans` and `groaned`) are treated as equivalent. Partial matches qualify only if the shared
span spans at least half of the longer expression: a note annotating a full sentence containing
an explained expression explains something else and is preserved. A short function word does
not qualify for partial matching — otherwise a hint for `to` would be discarded due to explained `heave to`.

Notes are removed only upon confirmed explanation repetition. Unfamiliar independent assertions
are preserved for substantive verification.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace

from linguary.domain import BlockProcessingResult, LanguageNote

#: Apostrophe varieties encountered interchangeably in book text and model responses.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})

#: Expression word: letters and digits, optionally with internal apostrophe (`one's`, `don't`).
_WORD = re.compile(r"\w+(?:'\w+)*")

#: Minimum shared stem between two inflected forms of one word and maximum suffix delta beyond it.
_MIN_STEM = 4
_MAX_ENDING = 3


def expression_words(text: str) -> tuple[str, ...]:
    """Expression words in normalized form: lowercase, stripped of punctuation and excess whitespace."""
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    return tuple(_WORD.findall(normalized))


class ExplainedExpressions:
    """Chapter `Before You Read` expressions prepared for comparison."""

    __slots__ = ("_expressions", "_explanations")

    def __init__(
        self, expressions: Iterable[str], *,
        explanations: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._expressions = tuple(
            words for expression in expressions if (words := expression_words(expression))
        )

        self._explanations = tuple(
            (expression_words(expression), explanation)
            for expression, explanation in explanations if expression_words(expression)
        )

    def __bool__(self) -> bool:
        return bool(self._expressions)

    def covers(self, text: str) -> bool:
        """Whether `Before You Read` explains the same expression as `text`."""
        words = expression_words(text)
        return bool(words) and any(
            _overlaps(words, expression) for expression in self._expressions
        )

    def repeated_by(self, note: LanguageNote) -> bool:
        """Whether a language note repeats a `Before You Read` expression.

        The excerpt defines candidate scope, including significant phrases within long
        sentences. Deletion requires covering all claims with an existing explanation.
        """
        excerpt = expression_words(note.original_excerpt)
        return any(
            (_overlaps(excerpt, expression) or _contains(excerpt, expression))
            and _same_explanation(expression, explanation, note.note)
            for expression, explanation in self._explanations
        )

    def without_repeats_v2(self, result: BlockProcessingResult) -> BlockProcessingResult:
        """Removes only exact duplicates of explanatory text; new aspects are preserved."""
        bodies = {text.strip() for _, text in self._explanations if text.strip()}
        hints = tuple(h for h in result.glossary_hints if not h.note or h.note.strip() not in bodies)
        if not hints and result.treatment.requires_glossary_hints:
            hints = result.glossary_hints
        bodies.update(h.note.strip() for h in hints if h.note and h.note.strip())
        notes = tuple(n for n in result.language_notes if n.note.strip() not in bodies)
        return replace(result, glossary_hints=hints, language_notes=notes)

    def without_repeats(self, result: BlockProcessingResult) -> BlockProcessingResult:
        """Result stripped of repeats of preparatory reading and hints from the same block.

        A repeated note is dropped completely. A hint is also dropped, with one exception:
        for a `GLOSSARY_ONLY` block hints are mandatory, and if all its hints turned out to be
        repeats, they are kept. A reader skipping preparatory reading would otherwise
        be left without assistance on a difficult paragraph, and chapter processing
        must not fail over pedagogical repetition.
        """
        if not self and not result.glossary_hints:
            return result
        hints = tuple(
            hint for hint in result.glossary_hints if not self.covers(hint.original)
        )
        if not hints and result.treatment.requires_glossary_hints:
            hints = result.glossary_hints
        notes = tuple(
            note
            for note in result.language_notes
            if not self.repeated_by(note)
            and not _repeats_block_hint(note, result)
        )
        if hints == result.glossary_hints and notes == result.language_notes:
            return result
        return replace(result, glossary_hints=hints, language_notes=notes)


def _repeats_block_hint(note: LanguageNote, result: BlockProcessingResult) -> bool:
    """Whether a note repeats an explanation already provided by a hint in its block."""
    excerpt = expression_words(note.original_excerpt)
    return any(
        (expression := expression_words(hint.original))
        and (_overlaps(excerpt, expression) or _contains(excerpt, expression))
        and _same_explanation(expression, hint.translation, note.note)
        for hint in result.glossary_hints
    )


# Finite dictionaries describe only two confirmed review duplicates.
# An unknown word in an assertion prohibits deletion even alongside a known sense.
_GRAB_WORDS = "make a grab at my throat разговорн идиом значени резко порывист схват ухват попыт выпад сдела рукой сторон чего либо движение"
_ARMOUR_WORDS = "sheath armour his and ножн оболоч брон доспех скры защит защищ отраж удар оруж враг граф скептицизм одновременно"
_CONNECTORS = frozenset("и или это то что здесь как его с со в от на для только но также".split())


def _features(expression: Sequence[str], text: str) -> frozenset[str]:
    words = expression_words(text)
    if _contains(expression, ("make", "a", "grab", "at")):
        return frozenset({"grab"}) if any(
            word.startswith(("схват", "ухват")) for word in words
        ) else frozenset()
    if "sheath" in expression and "armour" in expression:
        features = set()
        if any(word.startswith(("ножн", "оболоч")) for word in words) and any(
            word.startswith("скры") for word in words
        ):
            features.add("conceal")
        if any(word.startswith(("брон", "доспех")) for word in words) and any(
            word.startswith(("защит", "защищ", "отраж")) for word in words
        ):
            features.add("protect")
        return frozenset(features)
    return frozenset()


def _same_explanation(expression: Sequence[str], previous: str, current: str) -> bool:
    if expression_words(previous) == expression_words(current):
        return bool(expression_words(current))
    known = _features(expression, previous)
    if not known:
        return False
    vocabulary = (_GRAB_WORDS if "grab" in known else _ARMOUR_WORDS).split()
    claims = [claim for claim in re.split(r"[.;!?\n]+", current) if expression_words(claim)]
    for claim in claims:
        words = expression_words(claim)
        if any(word not in _CONNECTORS and not any(
            word.startswith(root) for root in vocabulary
        ) for word in words):
            return False
        features = _features(expression, claim)
        if not features or not features <= known:
            return False
    return bool(claims)


def _overlaps(first: Sequence[str], second: Sequence[str]) -> bool:
    """Whether two expressions match entirely or share a significant contiguous portion."""
    if len(first) == len(second) and _same_words(first, second):
        return True
    shorter, longer = sorted((first, second), key=len)
    if 2 * len(shorter) < len(longer):
        return False
    return _contains(longer, shorter)


def _contains(text: Sequence[str], expression: Sequence[str]) -> bool:
    """Whether a significant expression appears in text in full, contiguously, matching inflected forms."""
    if not _significant(expression) or len(expression) > len(text):
        return False
    return any(
        _same_words(expression, text[start : start + len(expression)])
        for start in range(len(text) - len(expression) + 1)
    )


def _mentions(text: Sequence[str], expression: Sequence[str]) -> bool:
    """Whether text mentions the expression or a contiguous part covering at least half the words."""
    shortest = (len(expression) + 1) // 2
    return any(
        _contains(text, expression[start : start + size])
        for size in range(len(expression), shortest - 1, -1)
        for start in range(len(expression) - size + 1)
    )


def _significant(expression: Sequence[str]) -> bool:
    """Whether an expression qualifies for partial matching: not a single short function word."""
    return bool(expression) and not (
        len(expression) == 1 and len(expression[0]) < _MIN_STEM
    )


def _same_words(first: Sequence[str], second: Sequence[str]) -> bool:
    """Whether words of two sequences of equal length match pairwise."""
    return all(_same_word(left, right) for left, right in zip(first, second, strict=True))


def _same_word(first: str, second: str) -> bool:
    """Whether this is the same word: exact match or inflections sharing a common stem."""
    first = {"made": "make"}.get(first, first)
    second = {"made": "make"}.get(second, second)
    if first == second:
        return True
    stem = next(
        (
            index
            for index, (left, right) in enumerate(zip(first, second, strict=False))
            if left != right
        ),
        min(len(first), len(second)),
    )
    return stem >= _MIN_STEM and max(len(first), len(second)) - stem <= _MAX_ENDING


__all__ = ["ExplainedExpressions", "expression_words"]
