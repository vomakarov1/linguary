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
_VERSE_QUOTE_PAIRS = {"‘": "’", "“": "”"}
_VERSE_DASH_END_RE = re.compile(r"[—–-]\s*$")
_VERSE_SPEECH_VERB_RE = re.compile(
    r"\b(?:said|asked|replied|answered|whispered|shouted|muttered|"
    r"murmured|mumbled|stammered|yelled|bellowed|exclaimed|retorted|snapped|growled)\b",
    re.IGNORECASE,
)
_VERSE_TAIL_MAX_LEN = 80
_VERSE_TAIL_MAX_BLOCKS = 4
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


def _has_speech_verb(text: str) -> bool:
    return _VERSE_SPEECH_VERB_RE.search(text) is not None


def _is_single_quote_apostrophe(text: str, idx: int) -> bool:
    if text[idx] != "’":
        return False
    if idx + 1 < len(text) and text[idx + 1].isalnum():
        return True
    if idx > 0 and text[idx - 1].isalnum():
        tail = text[idx + 1 :]
        if not tail.strip():
            word_match = re.search(r"[A-Za-z]+$", text[:idx])
            if (
                word_match is not None
                and len(word_match.group(0)) >= 3
                and word_match.group(0).casefold().endswith("in")
            ):
                return True
            return False
        if tail[0] in ".!?…,:;)]}":
            return False
        return True
    return False


def _advance_quote_stack(text: str, stack: list[str]) -> None:
    for idx, ch in enumerate(text):
        if ch in _VERSE_QUOTE_PAIRS:
            stack.append(ch)
        elif stack and ch == _VERSE_QUOTE_PAIRS[stack[-1]]:
            if ch == "’" and _is_single_quote_apostrophe(text, idx):
                continue
            stack.pop()


def _unclosed_quote_opener(blocks: Sequence[DocumentBlock]) -> str | None:
    stack: list[str] = []
    for block in blocks:
        _advance_quote_stack(block.text.strip(), stack)
    return stack[0] if stack else None


def _has_lowercase_prose_ratio(blocks: Sequence[DocumentBlock]) -> bool:
    non_initial_alpha = 0
    non_initial_lower = 0
    for block in blocks:
        tokens = _VERSE_WORD_RE.findall(block.text)
        for token in tokens[1:]:
            if len(token) > 1:
                non_initial_alpha += 1
                if token[0].islower():
                    non_initial_lower += 1
    return non_initial_alpha > 0 and non_initial_lower * 2 > non_initial_alpha


def _is_valid_implicit_verse_run(
    run: Sequence[DocumentBlock],
    *,
    core_len: int | None = None,
) -> bool:
    core = run if core_len is None else run[:core_len]
    if len(core) < 3:
        return False
    if not _has_lowercase_prose_ratio(run):
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
    non_final = len(core) - 1
    enjambed_caps = 0
    for index in range(non_final):
        cur_text = core[index].text.strip()
        next_text = core[index + 1].text.strip()
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
        elif idx <= cur_end + 3 and all(
            _is_verse_paragraph_candidate(strip[k]) and not _has_speech_verb(strip[k].text)
            for k in range(cur_end + 1, idx)
        ):
            cur_end = idx + 1
        else:
            groups.append((cur_start, cur_end))
            cur_start = idx
            cur_end = idx + 1
    groups.append((cur_start, cur_end))
    result: list[tuple[DocumentBlock, ...]] = []
    last_emitted_end = -1
    for start_idx, end_idx in groups:
        if start_idx <= last_emitted_end:
            continue
        core = tuple(strip[start_idx:end_idx + 1])
        extended_candidate: tuple[DocumentBlock, ...] | None = None
        extended_end = end_idx
        stack: list[str] = []
        for block in core:
            _advance_quote_stack(block.text.strip(), stack)
        if stack:
            tail_limit = min(len(strip), end_idx + 1 + _VERSE_TAIL_MAX_BLOCKS)
            for tail_idx in range(end_idx + 1, tail_limit):
                tail_block = strip[tail_idx]
                tail_text = tail_block.text.strip()
                if (
                    len(tail_text) > _VERSE_TAIL_MAX_LEN
                    or not _is_verse_paragraph_candidate(tail_block)
                    or _has_speech_verb(tail_text)
                ):
                    break
                _advance_quote_stack(tail_text, stack)
                if not stack:
                    candidate = tuple(strip[start_idx:tail_idx + 1])
                    if _is_valid_implicit_verse_run(candidate, core_len=len(core)):
                        extended_candidate = candidate
                        extended_end = tail_idx
                    break
        if extended_candidate is not None:
            result.append(extended_candidate)
            last_emitted_end = extended_end
        elif _is_valid_implicit_verse_run(core):
            result.append(core)
            last_emitted_end = end_idx
    return result


def _detect_implicit_couplets(
    ordered: Sequence[DocumentBlock],
    covered_ids: frozenset[str],
) -> list[tuple[DocumentBlock, ...]]:
    """Detect strict 2-line quoted verse couplets outside longer stanzas."""
    couplets: list[tuple[DocumentBlock, ...]] = []
    n = len(ordered)
    index = 0
    while index + 1 < n:
        first = ordered[index]
        second = ordered[index + 1]
        if (
            first.id in covered_ids
            or second.id in covered_ids
            or second.order != first.order + 1
            or not _is_verse_paragraph_candidate(first)
            or not _is_verse_paragraph_candidate(second)
            or _is_full_block_italic(first) is not _is_full_block_italic(second)
        ):
            index += 1
            continue
        t1 = first.text.strip()
        t2 = second.text.strip()
        if (
            len(t1) > 100
            or len(t2) > 100
            or _has_speech_verb(t1)
            or _has_speech_verb(t2)
        ):
            index += 1
            continue
        opener = t1[0]
        if (
            opener not in _VERSE_QUOTE_PAIRS
            or _unclosed_quote_opener((first,)) != opener
            or _VERSE_TERMINAL_PUNCT_RE.search(t1) is not None
            or _VERSE_DASH_END_RE.search(t1) is not None
        ):
            index += 1
            continue
        expected_closer = _VERSE_QUOTE_PAIRS[opener]
        if (
            t2[0] in _VERSE_OPENING_QUOTES
            or not _starts_with_upper_verse(t2)
            or not t2.endswith(expected_closer)
            or _VERSE_CLOSURE_PUNCT_RE.search(t2) is None
            or _unclosed_quote_opener((first, second)) is not None
            or not _has_lowercase_prose_ratio((first, second))
        ):
            index += 1
            continue
        couplets.append((first, second))
        index += 2
    return couplets


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
    covered_ids = frozenset(block.id for run in runs for block in run)
    couplets = _detect_implicit_couplets(ordered, covered_ids)
    if couplets:
        runs.extend(couplets)
        runs.sort(key=lambda run: run[0].order)
    return tuple(runs)


__all__ = ["detect_verse_stanza_runs"]
