"""Grouping key for spelling variants of a single glossary concept (QA-11, PRD § 11)."""

from __future__ import annotations

import pytest

from linguary.domain import glossary_key, translations_conflict


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("Mr Wonka", "Mr. Wonka", id="dot-after-mr"),
        pytest.param("Mrs. Bucket", "Mrs Bucket", id="dot-after-mrs"),
        pytest.param("Ms. Salt", "Ms Salt", id="dot-after-ms"),
        pytest.param("Dr. Foster", "Dr Foster", id="dot-after-dr"),
        pytest.param("St. Paul", "St Paul", id="dot-after-st"),
        pytest.param("Mr.Wonka", "Mr Wonka", id="dot-without-space"),
        pytest.param("WONKA'S FACTORY", "Wonka's factory", id="letter-case"),
        pytest.param("Wonka’s factory", "Wonka's factory", id="typographic-apostrophe"),
        pytest.param("“Whipple-Scrumptious”", '"Whipple-Scrumptious"', id="typographic-quotes"),
        pytest.param("Fudge  Mallow Delight", "Fudge Mallow Delight", id="whitespace"),
        pytest.param(" Charlie ", "Charlie", id="outer-whitespace"),
        pytest.param("ﬁzzy lifting", "fizzy lifting", id="nfkc-ligature"),
    ],
)
def test_obvious_spellings_share_the_key(first: str, second: str) -> None:
    """Case, typography, whitespace, and abbreviation dots do not change the concept key."""
    assert glossary_key(first) == glossary_key(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("Wonka's factory", "Wonka factory", id="possessive-is-kept"),
        pytest.param("Charlie", "Charlie Bucket", id="short-and-full-name"),
        pytest.param("Mr Wonka", "Mr Willy Wonka", id="name-with-first-name"),
        pytest.param("August.", "August", id="st-inside-a-word"),
        pytest.param("Mrs. Bucket", "Mr. Bucket", id="different-titles"),
    ],
)
def test_different_spellings_keep_different_keys(first: str, second: str) -> None:
    """The key does not merge spellings that are not obvious variants."""
    assert glossary_key(first) != glossary_key(second)


def test_typographic_variants_of_translation_are_not_a_conflict() -> None:
    """Different typography of the same translation does not create a conflict."""
    assert not translations_conflict(["фабрика Вонки", "Фабрика  Вонки"])
    assert not translations_conflict(["«Вонка»", "“Вонка”"])
    assert translations_conflict(["Чарли", "Чарли Баккет"])
