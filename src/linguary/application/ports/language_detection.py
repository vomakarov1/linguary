"""Port for checking text against its known source language."""

from __future__ import annotations

from typing import Protocol


class LanguageDetector(Protocol):
    """Returns a text's probability for a language, or `None` when unsupported/too short."""

    def probability_for_language(self, text: str, language_code: str) -> float | None: ...
