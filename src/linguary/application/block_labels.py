"""Удаление служебных псевдонимов блоков из текста анализа."""

from __future__ import annotations

import re

_LABEL = r"b\d{1,4}"
_LIST = rf"{_LABEL}(?:\s*[,;–—-]\s*{_LABEL})*"
_PARENTHESISED = re.compile(rf"\(\s*{_LIST}\s*\)", re.IGNORECASE)
_WITH_WORD = re.compile(rf"\bблок(?:и|а|у|е)?\s+{_LABEL}\b", re.IGNORECASE)
_BARE = re.compile(rf"\b{_LABEL}\b", re.IGNORECASE)


def without_block_labels(text: str) -> str:
    """Убирает временные номера блоков и лишние пробелы из текста модели."""
    text = _PARENTHESISED.sub("", text)
    text = _WITH_WORD.sub("", text)
    text = _BARE.sub("", text)
    text = re.sub(r"\(\s*[,;–—-]*\s*\)", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
