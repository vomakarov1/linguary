"""English spelling adapter backed by the cached bundled ipa-dict lexicons."""

from __future__ import annotations

from collections.abc import Iterable

from linguary.adapters.pronunciation.source import known_english_words
from linguary.application.ports.spelling import EnglishSpellingChecker

_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "‛": "'",
        "\u2010": "-",
        "\u2011": "-",
    }
)
_OUTER_PUNCTUATION = " \t\r\n.,!?;:()[]{}<>\"«»„“”—–-…/*~_|+=\\"
_CONTRACTION_SUFFIXES = ("'s", "n't", "'re", "'ve", "'ll", "'d", "'m")
_IRREGULAR_CONTRACTIONS = {
    "can't": "can",
    "won't": "will",
    "shan't": "shall",
}


def _normalize(word: str) -> str:
    if not isinstance(word, str):
        return ""
    return word.translate(_PUNCTUATION_TRANSLATION).casefold().strip(_OUTER_PUNCTUATION)


class IpaDictSpellingChecker(EnglishSpellingChecker):
    """Checks English words against bundled EN-GB and EN-US ipa-dict wordlists."""

    def __init__(self, words: Iterable[str] | None = None) -> None:
        if words is None:
            self._words = known_english_words()
        else:
            self._words = frozenset(
                item.translate(_PUNCTUATION_TRANSLATION).casefold()
                for item in words
                if isinstance(item, str) and item
            )

    def is_known_word(self, word: str) -> bool:
        token = _normalize(word)
        if not token or token in _CONTRACTION_SUFFIXES:
            return False
        if self._matches(token):
            return True
        if token.startswith("'") and token.endswith("'"):
            unquoted = token.strip("'").strip(_OUTER_PUNCTUATION)
            if unquoted and unquoted not in _CONTRACTION_SUFFIXES and self._matches(unquoted):
                return True
        if token.endswith("'") and len(token) > 2:
            possessive_base = token.rstrip("'")
            if possessive_base and self._matches(possessive_base):
                return True
        return False

    def _matches(self, token: str, *, allow_suffix: bool = True) -> bool:
        if not token or not any(ch.isalpha() for ch in token):
            return False
        if token in self._words:
            return True
        irregular_base = _IRREGULAR_CONTRACTIONS.get(token)
        if irregular_base is not None and irregular_base in self._words:
            return True
        if allow_suffix:
            for suffix in _CONTRACTION_SUFFIXES:
                if token.endswith(suffix):
                    base = token[: -len(suffix)]
                    if base and self._matches(base, allow_suffix=False):
                        return True
        if "-" in token:
            parts = token.split("-")
            if len(parts) > 1 and all(part and self._matches(part) for part in parts):
                return True
        return False
