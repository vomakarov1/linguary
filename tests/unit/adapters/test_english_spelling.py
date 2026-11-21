"""Tests for the cached ipa-dict lexicon loader and English spelling adapter."""

from __future__ import annotations

import pytest

from linguary.adapters.pronunciation.source import known_english_words
from linguary.adapters.validation.english_spelling import IpaDictSpellingChecker
from linguary.application.ports.spelling import EnglishSpellingChecker


def test_ipa_dict_spelling_checker_accepts_en_gb_hyphenated_and_rejects_expell() -> None:
    checker: EnglishSpellingChecker = IpaDictSpellingChecker()

    # EN-GB and EN-US spellings, including uppercase / mixed case
    for word in (
        "colour",
        "Colour",
        "colourful",
        "flavour",
        "realise",
        "travelling",
        "color",
        "realize",
        "traveling",
        "expel",
        "I",
    ):
        assert checker.is_known_word(word), f"expected known word: {word!r}"

    # Hyphenated words (both directly in lexicon and composed of known parts)
    for word in ("well-known", "self-esteem", "half-giant", "father-in-law's"):
        assert checker.is_known_word(word), f"expected known hyphenated word: {word!r}"

    # Contractions and possessives with straight and curly apostrophes
    for word in (
        "wizard's",
        "wizard’s",
        "harry’s",
        "don't",
        "don’t",
        "couldn't",
        "couldn’t",
        "daren't",
        "daren’t",
        "shan't",
        "shan’t",
        "wizards'",
        "wizards’",
    ):
        assert checker.is_known_word(word), f"expected known contraction/possessive: {word!r}"

    # Surrounding punctuation and quotes
    for word in ("(colour),", "“realise”!", "‘travelling’?", "«expel»", "'colour'", "don’t!"):
        assert checker.is_known_word(word), f"expected known word with punctuation: {word!r}"

    # Rejected misspellings and invalid tokens
    for word in (
        "expell",
        "Expell",
        "(expell)",
        "expell's",
        "expell’s",
        "expelln't",
        "self-expell",
        "well--known",
        "colour's's",
        "",
        "   ",
        "...",
        "---",
        "'s",
        "n't",
        "12345",
        "привет",
    ):
        assert not checker.is_known_word(word), f"expected rejected word: {word!r}"


def test_known_english_words_is_cached_and_combines_uk_and_us() -> None:
    first = known_english_words()
    second = known_english_words()

    assert first is second
    assert isinstance(first, frozenset)
    assert known_english_words.cache_info().maxsize == 1
    assert known_english_words.cache_info().hits >= 1
    assert "colourful" in first  # EN-GB only in ipa-dict
    assert "color" in first  # EN-US only in ipa-dict
    assert "i" in first  # casefolded from uppercase 'I' in en_UK.txt
    assert "expell" not in first


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("colour", True),
        ("colour’s", True),
        ("daren’t", True),
        ("can’t", True),
        ("self-colour", True),
        ("expell", False),
    ],
)
def test_ipa_dict_spelling_checker_supports_injected_lexicon(
    token: str,
    expected: bool,
) -> None:
    checker = IpaDictSpellingChecker(words=frozenset({"colour", "dare", "can", "self"}))
    assert checker.is_known_word(token) is expected
