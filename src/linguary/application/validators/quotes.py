"""Shared British single-quote, dialect-apostrophe, and direct-speech quote helpers.

Used by `dialogue` and `truncation` validators so multi-paragraph British single
quotes (`‘...`), adjacent dialect elisions (`killin’ ’em`, `An’ ’em`), word-final
dialect/possessive apostrophes, outer unclosed quote boundaries, and direct-speech
quote detection follow a single source of truth.
"""

from __future__ import annotations

import re

#: Fixed dialect words whose word-final `’` represents dropped letters rather than a closing quote.
_DIALECT_FINAL_WORDS = frozenset({"an", "o", "don", "las", "ol", "jus", "mus"})

#: Capitalized forms of fixed dialect words that still count as apostrophes before punctuation.
_DIALECT_CAPITALIZED_BEFORE_PUNCT = frozenset({"An", "Las", "Ol", "Jus", "Mus"})

#: Known negative/auxiliary contractions whose final `t` is elided (`couldn’`, `didn’`, ...).
_ELIDED_AUXILIARIES = frozenset(
    {
        "ain",
        "aren",
        "couldn",
        "didn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "shouldn",
        "wasn",
        "weren",
        "wouldn",
    }
)

#: Standard English `-in` words (length >= 4, not `-ain`) that are not `-ing -> -in’` dialect forms.
_NON_DIALECT_IN_WORDS = frozenset(
    {
        "akin",
        "aspirin",
        "assassin",
        "basin",
        "begin",
        "berlin",
        "bulletin",
        "cabin",
        "chin",
        "coffin",
        "coin",
        "cousin",
        "dolphin",
        "dublin",
        "gelatin",
        "goblin",
        "gremlin",
        "grin",
        "groin",
        "herein",
        "heroin",
        "insulin",
        "javelin",
        "join",
        "keratin",
        "kremlin",
        "latin",
        "loin",
        "margin",
        "matin",
        "melanin",
        "merlin",
        "muffin",
        "napkin",
        "origin",
        "paladin",
        "paraffin",
        "penicillin",
        "penguin",
        "protein",
        "puffin",
        "pumpkin",
        "raisin",
        "resin",
        "robin",
        "ruin",
        "satin",
        "shin",
        "skin",
        "spin",
        "therein",
        "thin",
        "toxin",
        "twin",
        "urchin",
        "vermin",
        "violin",
        "virgin",
        "vitamin",
        "wherein",
        "within",
    }
)

#: Word-initial straight-apostrophe elisions that must not act as quote openers when unpaired.
_STRAIGHT_ELISION_FOLLOW = re.compile(
    r"^(?:em|im|er|un|ave|ome|ouse|bout|cause|course|til|till|fraid|spect|n|s)(?!\w)"
)

_FOLLOWS_LOWER_WORD_OR_ELISION = re.compile(r"\s+(?:[a-z]|['’][a-z])")
_FOLLOWS_WORD_OR_ELISION = re.compile(r"\s+(?:\w|['’]\w)")

_DIRECT_OPEN_MARKS = ("‘", "“", '"', "«", "„")
_TRAILING_QUOTE_CONTINUATORS = frozenset(",;:!?…—–-.”»\"')]} \t\n\r")


def _is_word_char(character: str) -> bool:
    return character.isalnum() or character == "_"


def _word_before(text: str, pos: int) -> str:
    start = pos
    while start > 0 and _is_word_char(text[start - 1]):
        start -= 1
    return text[start:pos]


def _is_dialect_stem(word: str, *, before_punct: bool) -> bool:
    if not word:
        return False
    lower = word.lower()
    if lower in _DIALECT_FINAL_WORDS:
        if before_punct and word != lower and word not in _DIALECT_CAPITALIZED_BEFORE_PUNCT:
            return False
        return True
    return (
        word.islower()
        and len(lower) >= 4
        and lower.endswith("in")
        and not lower.endswith("ain")
        and lower not in _NON_DIALECT_IN_WORDS
    )


def _can_be_word_final_apostrophe(
    word: str, *, allow_plural_s: bool, before_punct: bool
) -> bool:
    if not word:
        return False
    lower = word.lower()
    if allow_plural_s and lower.endswith("s"):
        return True
    return lower in _ELIDED_AUXILIARIES or _is_dialect_stem(word, before_punct=before_punct)


def _is_word_final_apostrophe(text: str, idx: int) -> bool:
    """Whether a word-final `’` or `'` at `idx` (not followed by `\\w`) is an apostrophe."""
    if idx == 0 or not _is_word_char(text[idx - 1]):
        return False
    word = _word_before(text, idx)
    rest = text[idx + 1 :]
    if _FOLLOWS_LOWER_WORD_OR_ELISION.match(rest) is not None:
        return _can_be_word_final_apostrophe(word, allow_plural_s=True, before_punct=False)
    if _FOLLOWS_WORD_OR_ELISION.match(rest) is not None:
        return _can_be_word_final_apostrophe(word, allow_plural_s=False, before_punct=False)
    if not rest or rest[0] in ",;:.!?…—–-":
        return _can_be_word_final_apostrophe(word, allow_plural_s=False, before_punct=True)
    return False


def _is_local_british_apostrophe(text: str, idx: int) -> bool:
    """Whether `text[idx] == '’'` is an apostrophe based on local context alone."""
    if idx + 1 < len(text) and _is_word_char(text[idx + 1]):
        return True
    return _is_word_final_apostrophe(text, idx)


def _british_single_quote_events(text: str) -> list[tuple[str, int]]:
    """Return sorted `('open'|'close', index)` events for British single quotes `‘` / `’`."""
    events: list[tuple[str, int]] = []
    open_positions = [idx for idx, ch in enumerate(text) if ch == "‘"]
    boundaries = [*open_positions, len(text)]
    seg_start = 0
    for seg_end in boundaries:
        close_candidates = [idx for idx in range(seg_start, seg_end) if text[idx] == "’"]
        seg_closes: list[int] = []
        has_closer_right = False
        for idx in reversed(close_candidates):
            if _is_local_british_apostrophe(text, idx):
                continue
            if (
                has_closer_right
                and idx > 0
                and _is_word_char(text[idx - 1])
                and (idx + 1 == len(text) or text[idx + 1] in _TRAILING_QUOTE_CONTINUATORS)
                and _can_be_word_final_apostrophe(
                    _word_before(text, idx), allow_plural_s=True, before_punct=False
                )
            ):
                continue
            seg_closes.append(idx)
            has_closer_right = True
        for idx in reversed(seg_closes):
            events.append(("close", idx))
        if seg_end < len(text):
            events.append(("open", seg_end))
        seg_start = seg_end + 1
    return events


def british_quote_counts(text: str) -> tuple[int, int]:
    """Return `(opening_count, closing_count)` of real British single quotes in `text`."""
    opens = 0
    closes = 0
    for kind, _idx in _british_single_quote_events(text):
        if kind == "open":
            opens += 1
        else:
            closes += 1
    return opens, closes


def british_ends_with_closing_quote(text: str) -> bool:
    """Whether `text` ends with a real British closing single quote `’` (not an apostrophe)."""
    if not text.endswith("’"):
        return False
    events = _british_single_quote_events(text)
    return bool(events) and events[-1] == ("close", len(text) - 1)


def has_unclosed_british_quote(text: str) -> bool:
    """Whether `text` opens British single-quoted dialogue left unclosed at paragraph end."""
    if not text.lstrip().startswith("‘"):
        return False
    opens, closes = british_quote_counts(text)
    return opens > closes


def _matched_straight_single_quotes(text: str) -> list[tuple[int, int]]:
    """Return `(open_idx, close_idx)` pairs for paired straight `'...'` quotes, ignoring apostrophes."""
    if "'" not in text:
        return []
    open_candidates: list[int] = []
    pairs: list[tuple[int, int]] = []
    length = len(text)
    for idx, ch in enumerate(text):
        if ch != "'":
            continue
        prev_is_word = idx > 0 and _is_word_char(text[idx - 1])
        next_is_word = idx + 1 < length and _is_word_char(text[idx + 1])
        if prev_is_word and next_is_word:
            continue
        if not prev_is_word and idx + 1 < length and not text[idx + 1].isspace():
            if next_is_word and _STRAIGHT_ELISION_FOLLOW.match(text[idx + 1 :]) is not None:
                continue
            open_candidates.append(idx)
            continue
        if next_is_word:
            continue
        if _is_word_final_apostrophe(text, idx):
            continue
        if open_candidates:
            pairs.append((open_candidates.pop(), idx))
    return pairs


def has_direct_speech_quotes(text: str) -> bool:
    """Whether `text` contains direct-speech quotes (`‘`, `“`, `\"`, or paired `'...'`), not bare apostrophes."""
    if any(mark in text for mark in _DIRECT_OPEN_MARKS):
        return True
    return bool(_matched_straight_single_quotes(text))


def _outer_unclosed_open(text: str, pos: int) -> tuple[str, int] | None:
    """Identifier `(quote_mark, open_offset)` of the outermost quote still open at `pos`, or `None`."""
    if pos <= 0 or not text:
        return None
    limit = min(pos, len(text))
    tagged: list[tuple[int, str, str]] = []
    for kind, idx in _british_single_quote_events(text):
        if idx < limit:
            tagged.append((idx, kind, "‘"))
    for idx, ch in enumerate(text[:limit]):
        if ch == "“":
            tagged.append((idx, "open", "“"))
        elif ch == "”":
            tagged.append((idx, "close", "“"))
        elif ch == "«":
            tagged.append((idx, "open", "«"))
        elif ch == "»":
            tagged.append((idx, "close", "«"))
        elif ch == '"':
            tagged.append((idx, "toggle", '"'))
    for open_idx, close_idx in _matched_straight_single_quotes(text):
        if open_idx < limit:
            tagged.append((open_idx, "open", "'"))
        if close_idx < limit:
            tagged.append((close_idx, "close", "'"))
    tagged.sort(key=lambda item: item[0])

    stack: list[tuple[str, int]] = []
    for idx, action, mark in tagged:
        if action == "toggle":
            action = "close" if any(item[0] == mark for item in stack) else "open"
        if action == "open":
            stack.append((mark, idx))
        else:
            for stack_idx in range(len(stack) - 1, -1, -1):
                if stack[stack_idx][0] == mark:
                    del stack[stack_idx:]
                    break
    return stack[0] if stack else None


outer_unclosed_open = _outer_unclosed_open

__all__ = [
    "_outer_unclosed_open",
    "british_ends_with_closing_quote",
    "british_quote_counts",
    "has_direct_speech_quotes",
    "has_unclosed_british_quote",
    "outer_unclosed_open",
]
