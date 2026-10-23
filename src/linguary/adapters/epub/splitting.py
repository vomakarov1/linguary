"""Токенизация предложений и диалогов для расщепления длинных абзацев книги.

Выполняет интеллектуальное деление длинных составных абзацев (PRD/ADR-paragraph-splitting)
на самостоятельные смысловые фрагменты по границам предложений и реплик диалогов
без разрыва сокращений, инициалов, кавычек и многоточий.
"""

from __future__ import annotations

import re

from linguary.adapters.epub.inline import InlineContent
from linguary.domain import InlineAnnotation, InlineAnnotationType

SPLIT_MIN_CHARS: int = 800
SPLIT_MIN_SENTENCES: int = 2

#: Почётные титулы и обращения, после точки которых с заглавной буквы следует имя,
#: а не начало нового предложения.
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

#: Общепринятые сокращения, точка в которых не завершает предложение.
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

#: Глаголы атрибуции речи, связывающие реплику с автором.
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
    (?P<close>(?:["'\u201d\u2019\u00bb\u203a)\]*†‡]|[0-9]+|\[[0-9a-zA-Z]+\])*)
    (?P<space>\s+)
    """,
    re.VERBOSE,
)


def _find_split_points(text: str) -> list[int]:
    """Находит индексы границ предложений и реплик диалога в тексте."""
    split_points: list[int] = []

    for match in _CANDIDATE_PATTERN.finditer(text):
        term = match.group("term")
        close = match.group("close")
        next_start = match.end("space")

        if next_start >= len(text):
            continue

        next_char = text[next_start]

        # Обработка реплик через тире
        if next_char in "—–-":
            after_dash = re.match(r"[—–-]+\s*([^\s])", text[next_start:])
            if after_dash:
                char_after_dash = after_dash.group(1)
                # Если после тире строчные слова автора (напр. "— said Alice") — не делим
                if char_after_dash.islower():
                    continue
            else:
                continue
        elif next_char.islower():
            # Строчная буква после знака препинания (слова автора: "Hello!" said Alice)
            continue
        elif not (
            next_char.isupper()
            or next_char.isdigit()
            or next_char in "\"'\u201c\u2018\u00ab\u2039(["
        ):
            continue

        # Десятичные дроби (например, 3.14) не являются концом предложения
        if term == "." and match.start("term") > 0 and text[match.start("term") - 1].isdigit():
            if close and close[0].isdigit():
                continue

        # Защита сокращений и титулов при одиночной точке
        if term == ".":
            dot_pos = match.start("term")

            # Защита составных нумерованных маркеров списков и пунктов (напр. 1.C., 1.F.2.)
            marker_match = re.search(
                r"(?:^|[\s\"\'\u201c\u2018\u00ab(\[])([0-9A-Za-z]+(?:\.[0-9A-Za-z]+)+)$",
                text[:dot_pos],
            )
            if marker_match:
                marker = marker_match.group(1)
                has_digit = any(c.isdigit() for c in marker)
                has_alpha = any(c.isalpha() for c in marker)
                if (
                    (has_digit and has_alpha)
                    or marker.count(".") >= 2
                    or marker_match.start(1) == 0
                ):
                    continue

            word_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", text[:dot_pos])
            if word_match:
                word = word_match.group(1).lower()
                if word in _TITLES or word in _ABBREVIATIONS:
                    continue

            # Защита инициалов (H. G. Wells, A. Smith)
            initial_match = re.search(r"(?:^|[\s\"\'\u201c(])([A-Z])$", text[:dot_pos])
            if initial_match and re.match(r"[A-Z]", text[next_start:]):
                continue

        # Защита слов автора с именем собственным в начале ("Stop!" Alice said.)
        if close and any(c in _CLOSING_PUNCTUATION for c in close):
            attr_match = re.match(
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([a-z]+)", text[next_start:]
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
    """Разбивает текст на полуинтервалы [start, end) по границам предложений.

    Если длина текста меньше `min_chars` или количество найденных предложений меньше
    `min_sentences`, текст возвращается монолитным срезом `((0, len(text)),)`.
    Конкатенация всех срезов посимвольно сохраняет исходный текст без потерь.
    """
    if len(text) < min_chars:
        return ((0, len(text)),)

    split_points = _find_split_points(text)
    if len(split_points) + 1 < min_sentences:
        return ((0, len(text)),)

    boundaries = [0, *split_points, len(text)]
    return tuple((boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1))


def split_inline_content(
    content: InlineContent,
    *,
    min_chars: int = SPLIT_MIN_CHARS,
    min_sentences: int = SPLIT_MIN_SENTENCES,
) -> tuple[InlineContent, ...]:
    """Проецирует встроенную разметку и якоря на фрагменты расщеплённого абзаца.

    Смещения разметки пересчитываются по формуле `[start - offset, end - offset)`,
    сквозные аннотации подрезаются на границах фрагментов, ссылки на сноски не
    рассекаются, а якоря исходного абзаца закрепляются за первым фрагментом.
    Если деление не состоялось, возвращается кортеж из одного исходного объекта `content`.
    """
    slices = split_sentences(content.text, min_chars=min_chars, min_sentences=min_sentences)
    if len(slices) <= 1:
        return (content,)

    # Проверяем, чтобы границы срезов не рассекали ссылки на сноски
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

