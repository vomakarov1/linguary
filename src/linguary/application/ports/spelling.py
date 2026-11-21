"""Port for checking English word spelling against a local lexicon."""

from __future__ import annotations

from typing import Protocol


class EnglishSpellingChecker(Protocol):
    """Reports whether a single English word token is a known spelling."""

    def is_known_word(self, word: str) -> bool: ...
