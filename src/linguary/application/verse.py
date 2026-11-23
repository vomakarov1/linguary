"""Pure verse stanza detection helpers for prose and explicit poem blocks."""

from __future__ import annotations

import re
from collections.abc import Sequence

from linguary.domain import BlockType, DocumentBlock, InlineAnnotationType

_VERSE_TERMINAL_PUNCT_RE = re.compile(r"""[.!?…:][”"’'»)\]]*\s*$""")
_VERSE_CLOSURE_PUNCT_RE = re.compile(r"""[.!?…][”"’'»)\]]*\s*$""")
_VERSE_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-•*–—]|\d+[.)])\s+")
_VERSE_CARDINAL_START_RE = re.compile(
    r"^\s*(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_VERSE_OPENING_QUOTES = '‘“"\'«'
_VERSE_CLOSING_QUOTES = '’”"\'»'
_VERSE_WORD_RE = re.compile(
    r"[A-Za-z\u00C0-\u024FА-Яа-яЁё]+(?:['’-][A-Za-z\u00C0-\u024FА-Яа-яЁё]+)*"
)
_SINGLE_QUOTE_APOSTROPHE = (
    r"’(?=\w)"
    r"|(?<=\w)’(?=(?:\s+\w|[,;:])(?:[^‘’]|’(?=\w))*’(?!\w))"
)
_DIALOGUE_QUOTES = re.compile(
    rf"‘(?:[^‘’]|{_SINGLE_QUOTE_APOSTROPHE})*’(?!\w)|“[^“”]*”|\"[^\"“”]*\"|«[^«»]*»"
)


def _strip_leading_quotes(text: str) -> str:
    return text.lstrip().lstrip(_VERSE_OPENING_QUOTES).lstrip()


def _starts_with_upper_verse(text: str) -> bool:
    stripped = _strip_leading_quotes(text)
    for ch in stripped:
        if ch.isalpha():
            return ch.isupper()
    return False


def _is_closed_dialogue_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in _VERSE_OPENING_QUOTES and stripped[-1] in _VERSE_CLOSING_QUOTES:
        return True
    quotes = list(_DIALOGUE_QUOTES.finditer(stripped))
    if quotes and quotes[-1].end() < len(stripped):
        tail = stripped[quotes[-1].end():].strip()
        if tail and any(ch.isalpha() for ch in tail):
            return True
    return False


def _is_full_block_italic(block: DocumentBlock) -> bool:
    text = block.text
    first = next((i for i, ch in enumerate(text) if not ch.isspace()), None)
    if first is None:
        return False
    last = next((i for i in range(len(text) - 1, -1, -1) if not text[i].isspace()), first) + 1
    return any(
        ann.type is InlineAnnotationType.ITALIC and ann.start <= first and ann.end >= last
        for ann in block.inline_annotations
    )


def _is_verse_paragraph_candidate(block: DocumentBlock) -> bool:
    if block.type is not BlockType.PARAGRAPH:
        return False
    stripped = block.text.strip()
    if not stripped or len(stripped) > 100:
        return False
    if _VERSE_LIST_PREFIX_RE.match(stripped):
        return False
    if _is_closed_dialogue_line(stripped):
        return False
    words = _VERSE_WORD_RE.findall(stripped)
    if not (2 <= len(words) <= 15):
        return False
    internal = stripped[:-1]
    breaks = len(re.findall(r"[.!?…]\s+[A-ZА-ЯЁ]", internal))
    return breaks <= 1


def _is_valid_implicit_verse_run(run: Sequence[DocumentBlock]) -> bool:
    if len(run) < 3:
        return False
    non_initial_alpha = 0
    non_initial_lower = 0
    for block in run:
        tokens = _VERSE_WORD_RE.findall(block.text)
        for token in tokens[1:]:
            if len(token) > 1:
                non_initial_alpha += 1
                if token[0].islower():
                    non_initial_lower += 1
    if non_initial_alpha == 0 or non_initial_lower * 2 <= non_initial_alpha:
        return False
    final_closed = _VERSE_CLOSURE_PUNCT_RE.search(run[-1].text) is not None
    all_italic = all(_is_full_block_italic(block) for block in run)
    if not final_closed and not all_italic:
        return False
    cardinal_starts = sum(1 for block in run if _VERSE_CARDINAL_START_RE.match(block.text))
    if cardinal_starts * 2 >= len(run):
        return False
    if not final_closed and sum(1 for block in run if "(" in block.text and ")" in block.text) * 2 >= len(run):
        return False
    non_final = len(run) - 1
    enjambed_caps = 0
    for index in range(non_final):
        cur_text = run[index].text.strip()
        next_text = run[index + 1].text.strip()
        if (
            _VERSE_TERMINAL_PUNCT_RE.search(cur_text) is None
            and _starts_with_upper_verse(next_text)
        ):
            enjambed_caps += 1
    return enjambed_caps >= 2 and enjambed_caps * 2 >= non_final


def _is_enjambed_pair(first_text: str, second_text: str) -> bool:
    return (
        _VERSE_TERMINAL_PUNCT_RE.search(first_text.strip()) is None
        and _starts_with_upper_verse(second_text.strip())
    )


def _extract_implicit_verse_subruns(
    strip: Sequence[DocumentBlock],
) -> list[tuple[DocumentBlock, ...]]:
    """Trim leading/trailing prose paragraphs around enjambed verse lines inside a strip."""
    if len(strip) < 3:
        return []
    enjambed_indices = [
        i for i in range(len(strip) - 1)
        if _is_enjambed_pair(strip[i].text, strip[i + 1].text)
    ]
    if len(enjambed_indices) < 2:
        return []
    groups: list[tuple[int, int]] = []
    cur_start = enjambed_indices[0]
    cur_end = enjambed_indices[0] + 1
    for idx in enjambed_indices[1:]:
        if idx <= cur_end + 1:
            cur_end = idx + 1
        else:
            groups.append((cur_start, cur_end))
            cur_start = idx
            cur_end = idx + 1
    groups.append((cur_start, cur_end))
    result: list[tuple[DocumentBlock, ...]] = []
    for start_idx, end_idx in groups:
        candidate = tuple(strip[start_idx:end_idx + 1])
        if _is_valid_implicit_verse_run(candidate):
            result.append(candidate)
    return result


def detect_verse_stanza_runs(
    blocks: Sequence[DocumentBlock],
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Detect contiguous multi-block verse stanzas (explicit POEM or enjambed PARAGRAPH)."""
    ordered = tuple(sorted(blocks, key=lambda block: block.order))
    runs: list[tuple[DocumentBlock, ...]] = []
    n = len(ordered)
    index = 0
    while index < n:
        block = ordered[index]
        if block.type is BlockType.POEM:
            end = index + 1
            while (
                end < n
                and ordered[end].type is BlockType.POEM
                and ordered[end].order == ordered[end - 1].order + 1
            ):
                end += 1
            if end - index >= 2:
                runs.append(tuple(ordered[index:end]))
            index = end
            continue
        if _is_verse_paragraph_candidate(block):
            italic_flag = _is_full_block_italic(block)
            end = index + 1
            while (
                end < n
                and ordered[end].order == ordered[end - 1].order + 1
                and _is_verse_paragraph_candidate(ordered[end])
                and _is_full_block_italic(ordered[end]) is italic_flag
            ):
                end += 1
            runs.extend(_extract_implicit_verse_subruns(ordered[index:end]))
            index = end
            continue
        index += 1
    return tuple(runs)


__all__ = ["detect_verse_stanza_runs"]
