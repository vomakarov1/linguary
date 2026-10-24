"""Processing result language validation (PRD § 15, ALG-04).

Language compliance is determined by the share of target alphabet letters among all letters
in the text. The check applies only to full translations longer than a configurable minimum:
a short string may legitimately consist of a proper name, and a glossary hint consists of
the source language word it was written for.

The alphabet is defined only for the MVP target language — Russian (PRD § 1). For other
target languages, the share of "target alphabet characters" is undefined and validation
is skipped: this is safer than counting a foreign alphabet's share and rejecting a valid translation.
"""

from __future__ import annotations

from linguary.application.llm_dto import TranslationResultResponse

#: Target languages whose alphabet is known to the application. Key is the book language code without region.
_TARGET_ALPHABETS: dict[str, tuple[str, str]] = {"ru": ("Ѐ", "ӿ")}


def language_violation(
    payload: TranslationResultResponse,
    *,
    target_language: str,
    min_share: float,
    min_length: int,
) -> str | None:
    """Violation description of translation language mismatch with target language, or `None`."""
    alphabet = _TARGET_ALPHABETS.get(_language_key(target_language))
    if alphabet is None:
        return None
    for block in payload.blocks:
        text = block.translated_text
        if text is None or len(text) <= min_length:
            continue
        share = _alphabet_share(text, alphabet=alphabet)
        if share is not None and share < min_share:
            return (
                f"перевод блока {block.source_block_id} не на языке {target_language}: "
                f"доля целевого алфавита {share:.2f} ниже порога {min_share:.2f}"
            )
    return None


def _language_key(target_language: str) -> str:
    """Book language code normalized without region or case: `ru-RU` and `RU` both represent Russian."""
    return target_language.strip().lower().split("-")[0]


def _alphabet_share(text: str, *, alphabet: tuple[str, str]) -> float | None:
    """Share of target alphabet characters among text letters; `None` if text contains no letters."""
    first, last = alphabet
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return None
    target = sum(1 for character in letters if first <= character <= last)
    return target / len(letters)


__all__ = ["language_violation"]
