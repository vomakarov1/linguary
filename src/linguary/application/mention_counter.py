"""Подсчёт упоминаний понятия в исходном тексте книги."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from linguary.application.context_builder import textual_blocks
from linguary.domain import CanonicalDocument, glossary_key


def mention_counter(document: CanonicalDocument) -> Callable[[Sequence[str]], int]:
    """Считает вхождения написаний по границам слов без двойного счёта перекрытий."""
    texts = tuple(
        glossary_key(block.text)
        for section in document.ordered_sections
        for block in textual_blocks(section)
    )

    def count(spellings: Sequence[str]) -> int:
        normalized = {glossary_key(spelling) for spelling in spellings}
        patterns = tuple(
            re.compile(r"(?<!\w)" + re.escape(spelling) + r"(?!\w)")
            for spelling in normalized if spelling
        )
        total = 0
        for text in texts:
            spans = sorted(
                (match.start(), match.end())
                for pattern in patterns for match in pattern.finditer(text)
            )
            end = -1
            for start, stop in spans:
                if start >= end:
                    total += 1
                    end = stop
        return total

    return count
