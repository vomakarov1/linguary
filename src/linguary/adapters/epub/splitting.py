"""Sentence and dialogue tokenization for splitting long book paragraphs.

Performs intelligent division of long compound paragraphs (PRD/ADR-paragraph-splitting)
into balanced reading fragments. Sentence and dialogue boundaries are preferred without
splitting abbreviations, initials, quotes, or ellipses; long sentences fall back to word
boundaries.

`segment_text` marks clause-aware spans inside one block and does not divide the block.
`split_inline_content` remains until P3.T2 as the import adapter: it still projects
`split_paragraph` fragments, so a long paragraph stays several blocks.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence
from math import ceil

from linguary.adapters.epub.inline import InlineContent
from linguary.domain import BlockSegment, InlineAnnotation, InlineAnnotationType

SPLIT_MIN_CHARS: int = 600
SPLIT_MIN_SENTENCES: int = 2
SPLIT_TARGET_CHARS: int = 300
SPLIT_MIN_PART_CHARS: int = 250
SPLIT_MAX_PART_CHARS: int = 350

#: Text shorter than this is one segment. Longer text is split into sentences.
SEGMENT_MIN_BLOCK_CHARS: int = 600
#: A sentence longer than this is split at the first clause level that fits.
SEGMENT_MAX_CHARS: int = 350

#: Honorific titles and salutations where a trailing period is followed by a capitalized name,
#: rather than the start of a new sentence.
_TITLES = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "rev",
        "st",
        "gen",
        "col",
        "capt",
        "lt",
        "sgt",
        "maj",
        "hon",
        "jr",
        "sr",
        "esq",
        "mme",
        "mlle",
        "messrs",
        "pres",
        "gov",
        "sen",
        "rep",
        "sir",
        "lord",
        "lady",
    }
)

#: Common abbreviations where a trailing period does not terminate a sentence.
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "vs",
        "v",
        "viz",
        "al",
        "ca",
        "cf",
        "ibid",
        "op",
        "cit",
        "vol",
        "vols",
        "no",
        "nos",
        "pp",
        "p",
        "ch",
        "chap",
        "sec",
        "ed",
        "eds",
        "fig",
        "figs",
        "co",
        "corp",
        "inc",
        "ltd",
        "dept",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "u.s",
        "u.k",
        "e.u",
        "u.s.a",
    }
)

#: Speech attribution verbs linking a dialogue line to the speaker.
_ATTRIBUTION_VERBS = frozenset(
    {
        "said",
        "says",
        "spoke",
        "speaks",
        "replied",
        "replies",
        "answered",
        "answers",
        "asked",
        "asks",
        "inquired",
        "inquires",
        "cried",
        "cries",
        "shouted",
        "shouts",
        "yelled",
        "yells",
        "whispered",
        "whispers",
        "murmured",
        "murmurs",
        "muttered",
        "mutters",
        "exclaimed",
        "exclaims",
        "gasped",
        "gasps",
        "added",
        "adds",
        "continued",
        "continues",
        "remarked",
        "remarks",
        "observed",
        "observes",
        "repeated",
        "repeats",
        "returned",
        "returns",
        "thought",
        "thinks",
        "called",
        "calls",
        "began",
        "begins",
    }
)

_CLOSING_PUNCTUATION = "\"'\u201d\u2019\u00bb\u203a"

_CANDIDATE_PATTERN = re.compile(
    r"""
    (?P<term>\.\.\.+|…|(?:\.\s+){2,}\.|[.!?])
    (?P<close>(?:["'\u201d\u2019\u00bb\u203a)\]*†‡0-9]|\[[0-9a-zA-Z]+\])*)
    (?P<space>\s+)
    """,
    re.VERBOSE,
)


_MARKER_PATTERN = re.compile(
    r"(?:^|[\s\"\'\u201c\u2018\u00ab(\[])([0-9A-Za-z]+(?:\.[0-9A-Za-z]+)+)$"
)
_WORD_PATTERN = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$")
_INITIAL_PATTERN = re.compile(r"(?:^|[\s\"\'\u201c(])([A-Z])$")
_ATTRIBUTION_PATTERN = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([a-z]+)"
)

#: Comma, whitespace, then a coordinating or subordinating conjunction as a whole word.
_CLAUSE_CONJUNCTION_PATTERN = re.compile(
    r",\s+(?:although|because|though|while|where|which|when|and|but|nor|yet|who|so|or|if)\b",
    re.IGNORECASE,
)
_CLAUSE_LEVELS: tuple[str, ...] = (";", ":", "dash", "conjunction")
_DASH_MARKS: tuple[str, ...] = ("—", "–", " - ")


def _find_split_points(text: str) -> list[int]:
    """Finds indices of sentence and dialogue boundaries in text."""
    split_points: list[int] = []

    for match in _CANDIDATE_PATTERN.finditer(text):
        term = match.group("term")
        close = match.group("close")
        next_start = match.end("space")

        if next_start >= len(text):
            continue

        next_char = text[next_start]

        # Handle dialogue turns introduced with a dash
        if next_char in "—–-":
            after_dash = re.match(r"[—–-]+\s*([^\s])", text[next_start:])
            if after_dash:
                char_after_dash = after_dash.group(1)
                # If lowercase author words follow dash (e.g. "— said Alice") — do not split
                if char_after_dash.islower():
                    continue
            else:
                continue
        elif next_char.islower():
            # Lowercase letter after punctuation mark (author words: "Hello!" said Alice)
            continue
        elif not (
            next_char.isupper()
            or next_char.isdigit()
            or next_char in "\"'\u201c\u2018\u00ab\u2039(["
        ):
            continue

        # Decimal fractions (e.g. 3.14) are not sentence ends
        if term == "." and match.start("term") > 0 and text[match.start("term") - 1].isdigit():
            if close and close[0].isdigit():
                continue

        # Protect abbreviations and titles on single dot
        if term == ".":
            dot_pos = match.start("term")
            prefix_start = max(0, dot_pos - 64)
            prefix = text[prefix_start:dot_pos]

            # Protect composite numbered list/item markers (e.g. 1.C., 1.F.2.)
            marker_match = _MARKER_PATTERN.search(prefix)
            if marker_match:
                marker = marker_match.group(1)
                has_digit = any(c.isdigit() for c in marker)
                has_alpha = any(c.isalpha() for c in marker)
                is_at_start = prefix_start == 0 and marker_match.start(1) == 0
                if (
                    (has_digit and has_alpha)
                    or marker.count(".") >= 2
                    or is_at_start
                ):
                    continue

            word_match = _WORD_PATTERN.search(prefix)
            if word_match:
                word = word_match.group(1).lower()
                if word in _TITLES or word in _ABBREVIATIONS:
                    continue

            # Protect initials (H. G. Wells, A. Smith)
            initial_match = _INITIAL_PATTERN.search(prefix)
            if initial_match and re.match(r"[A-Z]", text[next_start:]):
                continue

        # Protect author attribution with a proper noun at start ("Stop!" Alice said.)
        if close and any(c in _CLOSING_PUNCTUATION for c in close):
            attr_match = _ATTRIBUTION_PATTERN.match(
                text[next_start : next_start + 64]
            )
            if attr_match:
                verb = attr_match.group(2).lower()
                if verb in _ATTRIBUTION_VERBS:
                    continue

        split_points.append(next_start)

    return split_points


def split_sentences(
    text: str,
    *,
    min_chars: int = SPLIT_MIN_CHARS,
    min_sentences: int = SPLIT_MIN_SENTENCES,
) -> tuple[tuple[int, int], ...]:
    """Splits text into half-open intervals [start, end) along sentence boundaries.

    If text length is below `min_chars` or the number of detected sentences is below
    `min_sentences`, the text is returned as a monolithic slice `((0, len(text)),)`.
    Concatenating all slices character-for-character preserves the original text without loss.
    """
    if len(text) < min_chars:
        return ((0, len(text)),)

    split_points = _find_split_points(text)
    if len(split_points) + 1 < min_sentences:
        return ((0, len(text)),)

    boundaries = [0, *split_points, len(text)]
    return tuple((boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1))


def _split_span_by_words(
    text: str,
    start: int,
    end: int,
    *,
    target_chars: int,
) -> list[tuple[int, int]]:
    span_len = end - start
    if span_len <= 1:
        return [(start, end)]

    max_part = (
        SPLIT_MAX_PART_CHARS
        if target_chars == SPLIT_TARGET_CHARS
        else max(1, round(target_chars * SPLIT_MAX_PART_CHARS / SPLIT_TARGET_CHARS))
    )
    min_part = (
        SPLIT_MIN_PART_CHARS
        if target_chars == SPLIT_TARGET_CHARS
        else max(1, round(target_chars * SPLIT_MIN_PART_CHARS / SPLIT_TARGET_CHARS))
    )
    min_count = max(2, ceil(span_len / max_part))
    max_count = max(2, span_len // min_part)
    counts = (
        range(min_count, max_count + 1)
        if min_count <= max_count
        else {min_count, max_count}
    )
    part_count = min(
        counts,
        key=lambda count: (abs(span_len / count - target_chars), -count),
    )
    part_count = min(part_count, span_len)
    if part_count <= 1:
        return [(start, end)]

    word_boundaries = [
        start + match.end()
        for match in re.finditer(r"\s+", text[start:end])
        if 0 < match.end() < span_len
    ]

    if len(word_boundaries) >= part_count - 1:
        chosen = [start]
        prev_idx = -1
        total_wb = len(word_boundaries)
        for part_index in range(1, part_count):
            remaining_cuts = part_count - 1 - part_index
            hi = total_wb - remaining_cuts
            lo = prev_idx + 1
            target_boundary = start + round(span_len * part_index / part_count)
            pos = bisect_left(word_boundaries, target_boundary, lo, hi)
            if pos == lo:
                best_idx = lo
            elif pos == hi:
                best_idx = hi - 1
            elif (
                abs(word_boundaries[pos] - target_boundary)
                < abs(word_boundaries[pos - 1] - target_boundary)
            ):
                best_idx = pos
            else:
                best_idx = pos - 1
            chosen.append(word_boundaries[best_idx])
            prev_idx = best_idx
        chosen.append(end)
        return [(chosen[index], chosen[index + 1]) for index in range(part_count)]

    candidates = sorted(set(word_boundaries) | set(range(start + 1, end)))
    chosen = [start]
    for part_index in range(1, part_count):
        remaining_parts = part_count - part_index
        latest_boundary = end - remaining_parts
        available = [
            boundary
            for boundary in candidates
            if chosen[-1] < boundary <= latest_boundary
        ]
        target_boundary = start + round(span_len * part_index / part_count)
        if not available:
            chosen.append(min(max(chosen[-1] + 1, target_boundary), latest_boundary))
        else:
            chosen.append(min(available, key=lambda item: abs(item - target_boundary)))
    chosen.append(end)
    return [(chosen[index], chosen[index + 1]) for index in range(part_count)]


def _group_atoms_dp(
    atoms: list[tuple[int, int]],
    *,
    target_chars: int,
    max_sentence_part: int,
) -> tuple[tuple[int, int], ...]:
    atom_count = len(atoms)
    if atom_count <= 1:
        return tuple(atoms)

    dp: list[int] = [0] * (atom_count + 1)
    next_idx: list[int] = [atom_count] * (atom_count + 1)

    for i in range(atom_count - 1, -1, -1):
        best_cost: int | None = None
        best_j = i + 1
        start_pos = atoms[i][0]
        for j in range(i + 1, atom_count + 1):
            group_len = atoms[j - 1][1] - start_pos
            if j > i + 1 and group_len > max_sentence_part:
                break
            if i == 0 and j == atom_count:
                continue
            cost = (group_len - target_chars) ** 2 + dp[j]
            if best_cost is None or cost < best_cost or (cost == best_cost and j > best_j):
                best_cost = cost
                best_j = j
        assert best_cost is not None
        dp[i] = best_cost
        next_idx[i] = best_j

    result: list[tuple[int, int]] = []
    cur = 0
    while cur < atom_count:
        nxt = next_idx[cur]
        result.append((atoms[cur][0], atoms[nxt - 1][1]))
        cur = nxt
    return tuple(result)


def split_paragraph(
    text: str,
    *,
    min_chars: int = SPLIT_MIN_CHARS,
    target_chars: int = SPLIT_TARGET_CHARS,
    min_sentences: int = SPLIT_MIN_SENTENCES,
) -> tuple[tuple[int, int], ...]:
    """Divides long paragraphs into balanced pieces near 300 characters.

    Whole sentences up to `max(SPLIT_MAX_PART_CHARS, min_chars - 50)` are kept intact and grouped
    around `target_chars`. Oversized single sentences fall back to word boundaries, with a
    character boundary as a final fallback for unspaced text. The returned intervals preserve
    every source character exactly.
    """
    if min_chars < 1 or target_chars < 1 or min_sentences < 1:
        raise ValueError("параметры разбиения абзаца должны быть положительными")
    if len(text) <= min_chars:
        return ((0, len(text)),)

    max_sentence_part = max(SPLIT_MAX_PART_CHARS, min_chars - 50)
    sentence_points = _find_split_points(text)
    if len(sentence_points) + 1 < min_sentences:
        return tuple(_split_span_by_words(text, 0, len(text), target_chars=target_chars))

    boundaries = [0, *sentence_points, len(text)]
    atoms: list[tuple[int, int]] = []
    for index in range(len(boundaries) - 1):
        sent_start = boundaries[index]
        sent_end = boundaries[index + 1]
        if sent_end - sent_start <= max_sentence_part:
            atoms.append((sent_start, sent_end))
        else:
            atoms.extend(
                _split_span_by_words(
                    text,
                    sent_start,
                    sent_end,
                    target_chars=target_chars,
                )
            )

    return _group_atoms_dp(
        atoms,
        target_chars=target_chars,
        max_sentence_part=max_sentence_part,
    )


def _normalize_protected(
    protected: Sequence[tuple[int, int]],
    length: int,
) -> tuple[tuple[int, int], ...]:
    """Clamps protected ranges to `text` and merges overlaps."""
    ranges: list[tuple[int, int]] = []
    for start, end in protected:
        if start < 0:
            start = 0
        if end > length:
            end = length
        if end <= start:
            continue
        ranges.append((start, end))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _move_past_protected(point: int, protected: Sequence[tuple[int, int]]) -> int:
    """Moves a cut that lands inside a protected range to that range's end."""
    changed = True
    while changed:
        changed = False
        for start, end in protected:
            if start < point < end:
                point = end
                changed = True
    return point


def _boundaries_respecting_protected(
    start: int,
    end: int,
    points: Sequence[int],
    protected: Sequence[tuple[int, int]],
) -> list[int]:
    """Shifts a cut inside a protected range to that range's end and drops empty cuts."""
    boundaries = [start]
    for point in points:
        if point <= boundaries[-1] or point >= end:
            continue
        shifted = _move_past_protected(point, protected)
        if boundaries[-1] < shifted < end:
            boundaries.append(shifted)
    boundaries.append(end)
    return boundaries


def _after_mark(text: str, mark_end: int, span_end: int) -> int:
    point = mark_end
    while point < span_end and text[point].isspace():
        point += 1
    return point


def _marked_points(text: str, start: int, end: int, mark: str) -> list[int]:
    points: list[int] = []
    cursor = start
    while cursor < end:
        found = text.find(mark, cursor, end)
        if found < 0:
            break
        point = _after_mark(text, found + len(mark), end)
        if start < point < end:
            points.append(point)
        cursor = found + len(mark)
    return points


def _dash_points(text: str, start: int, end: int) -> list[int]:
    points: list[int] = []
    cursor = start
    while cursor < end:
        found: tuple[int, int] | None = None
        for mark in _DASH_MARKS:
            mark_at = text.find(mark, cursor, end)
            if mark_at < 0:
                continue
            candidate = (mark_at, mark_at + len(mark))
            if found is None or candidate[0] < found[0]:
                found = candidate
        if found is None:
            break
        _, mark_end = found
        point = _after_mark(text, mark_end, end)
        if start < point < end:
            points.append(point)
        cursor = mark_end
    return points


def _conjunction_points(text: str, start: int, end: int) -> list[int]:
    points: list[int] = []
    for match in _CLAUSE_CONJUNCTION_PATTERN.finditer(text, start, end):
        point = match.start() + 1
        while point < end and text[point].isspace():
            point += 1
        if start < point < end:
            points.append(point)
    return points


def _clause_points(text: str, start: int, end: int, level: str) -> list[int]:
    if level in {";", ":"}:
        return _marked_points(text, start, end, level)
    if level == "dash":
        return _dash_points(text, start, end)
    return _conjunction_points(text, start, end)


def _word_split_target(max_segment_chars: int) -> int:
    """Target length whose existing word-split cap is `max_segment_chars`."""
    if max_segment_chars == SPLIT_MAX_PART_CHARS:
        return SPLIT_TARGET_CHARS
    return max(1, round(max_segment_chars * SPLIT_TARGET_CHARS / SPLIT_MAX_PART_CHARS))


def _split_sentence(
    text: str,
    start: int,
    end: int,
    sentence_index: int,
    protected: Sequence[tuple[int, int]],
    max_segment_chars: int,
) -> list[BlockSegment]:
    """Splits one sentence at the first clause level whose parts all fit."""
    if end - start <= max_segment_chars:
        return [BlockSegment(start, end, sentence_index)]

    for level in _CLAUSE_LEVELS:
        boundaries = _boundaries_respecting_protected(
            start,
            end,
            _clause_points(text, start, end, level),
            protected,
        )
        if len(boundaries) < 3:
            continue
        spans = list(zip(boundaries, boundaries[1:]))
        if all(span_end - span_start <= max_segment_chars for span_start, span_end in spans):
            return [
                BlockSegment(span_start, span_end, sentence_index)
                for span_start, span_end in spans
            ]

    word_spans = _split_span_by_words(
        text,
        start,
        end,
        target_chars=_word_split_target(max_segment_chars),
    )
    interior = [span_end for _, span_end in word_spans[:-1]]
    boundaries = _boundaries_respecting_protected(start, end, interior, protected)
    return [
        BlockSegment(span_start, span_end, sentence_index)
        for span_start, span_end in zip(boundaries, boundaries[1:])
    ]


def segment_text(
    text: str,
    *,
    protected: Sequence[tuple[int, int]] = (),
    min_block_chars: int = SEGMENT_MIN_BLOCK_CHARS,
    max_segment_chars: int = SEGMENT_MAX_CHARS,
) -> tuple[BlockSegment, ...]:
    """Marks clause-aware spans covering `text` inside one block.

    Text shorter than `min_block_chars` is one segment with sentence index 0.
    Longer text is split into sentences. A sentence longer than `max_segment_chars`
    is split at the first level that keeps every part within the limit: `;`,
    `:`, a dash (`—`, `–`, ` - `), a comma before a conjunction, then a word
    boundary. Parts of one sentence share its `sentence_index`. Protected ranges
    are never cut. A single covering span is returned as a segment and is not
    collapsed to an empty tuple.
    """
    if len(text) < min_block_chars:
        return (BlockSegment(0, len(text), 0),)

    protected_ranges = _normalize_protected(protected, len(text))
    sentence_boundaries = _boundaries_respecting_protected(
        0,
        len(text),
        _find_split_points(text),
        protected_ranges,
    )
    segments: list[BlockSegment] = []
    for sentence_index, (start, end) in enumerate(
        zip(sentence_boundaries, sentence_boundaries[1:])
    ):
        segments.extend(
            _split_sentence(
                text,
                start,
                end,
                sentence_index,
                protected_ranges,
                max_segment_chars,
            )
        )
    return tuple(segments)


def split_inline_content(
    content: InlineContent,
    *,
    min_chars: int = SPLIT_MIN_CHARS,
    min_sentences: int = SPLIT_MIN_SENTENCES,
) -> tuple[InlineContent, ...]:
    """Projects inline markup and anchors onto balanced paragraph fragments.

    Temporary import adapter until P3.T2. Boundaries still come from
    `split_paragraph`, so a long paragraph remains several blocks. Clause-aware
    spans for one unsplit block are `segment_text` and are not applied here.

    Markup offsets are recalculated via `[start - offset, end - offset)`,
    spanning annotations are clipped at fragment boundaries, footnote references are never cut
    across, and anchors of the original paragraph attach to the first fragment. Sentence breaks
    are preferred; a long sentence can fall back to a word boundary to stay near the target size.
    If splitting does not occur, a single-element tuple of the original `content` object is
    returned.
    """
    slices = split_paragraph(
        content.text,
        min_chars=min_chars,
        min_sentences=min_sentences,
    )
    if len(slices) <= 1:
        return (content,)

    # Ensure slice boundaries do not bisect footnote references
    boundaries = [s[0] for s in slices] + [slices[-1][1]]
    adjusted_boundaries = [0]
    for b in boundaries[1:-1]:
        changed = True
        while changed:
            changed = False
            for ann in content.annotations:
                if ann.type is InlineAnnotationType.FOOTNOTE_REFERENCE and ann.start < b < ann.end:
                    new_b = max(b, ann.end)
                    if new_b != b:
                        b = new_b
                        changed = True
        if b > adjusted_boundaries[-1] and b < boundaries[-1]:
            adjusted_boundaries.append(b)
    adjusted_boundaries.append(boundaries[-1])

    if len(adjusted_boundaries) <= 2:
        return (content,)

    fragments: list[InlineContent] = []
    for i in range(len(adjusted_boundaries) - 1):
        slice_start = adjusted_boundaries[i]
        slice_end = adjusted_boundaries[i + 1]
        offset = slice_start
        frag_text = content.text[slice_start:slice_end]

        frag_annotations: list[InlineAnnotation] = []
        for ann in content.annotations:
            clipped_start = max(ann.start, slice_start)
            clipped_end = min(ann.end, slice_end)
            if clipped_end > clipped_start:
                frag_annotations.append(
                    InlineAnnotation(
                        type=ann.type,
                        start=clipped_start - offset,
                        end=clipped_end - offset,
                        target=ann.target,
                    )
                )

        frag_anchors = content.anchor_ids if i == 0 else ()
        fragments.append(
            InlineContent(
                text=frag_text,
                annotations=tuple(frag_annotations),
                anchor_ids=frag_anchors,
            )
        )

    return tuple(fragments)

