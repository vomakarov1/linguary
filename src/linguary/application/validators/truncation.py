"""Validation of truncated generation by formal indicators (PRD § 15, ALG-04).

There are three indicators: the provider reported truncation due to token limit, the translation
does not end with terminal punctuation, or a pair of quotes or brackets is left unclosed. The first
is reported by the provider itself (`LLMResponse.truncated`) — checking it is mandatory in processing:
the response might be complete in terms of block structure but cut short in text, and coverage checks
(which cover this gap in chapter analysis) would detect nothing here.

The two textual indicators mirror the original text, otherwise they catch normal writing rather
than truncation. A heading, caption, or list item legitimately ends without a period, and multi-paragraph
dialogue conventionally opens quotes on each paragraph while closing them only on the last. Therefore
terminal punctuation is required in translation only where the source block ends with it, and an
unclosed pair is considered truncation only when the original lacked such an imbalance.
"""

from __future__ import annotations

from collections.abc import Mapping

from linguary.application.llm_dto import TranslationResultResponse
from linguary.domain import TranslationUnit

#: Punctuation marks that conclude a completed sentence.
_TERMINAL_PUNCTUATION = frozenset(".!?…:;")

#: Trailing characters that can appear after terminal punctuation.
_TRAILING_CLOSERS = "»\"”’'*)]}"

#: Brackets whose imbalance indicates truncated text.
_BRACKETS: tuple[tuple[str, str], ...] = (("(", ")"), ("[", "]"), ("{", "}"))

#: Symmetric quotation mark: pairing is determined by the parity of occurrence count.
_SYMMETRIC_QUOTE = '"'


def truncation_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit, truncated: bool
) -> str | None:
    """Violation description of truncated generation, or `None`.

    `truncated` is a signal from the provider: output was cut off by token limit
    rather than finished naturally by the model.
    """
    if truncated:
        return "провайдер сообщил об обрыве ответа по лимиту выходных токенов"
    sources = {item.block.id: item.block.text for item in unit.items}
    for block in payload.blocks:
        text = block.translated_text
        if text is None:
            continue
        violation = _text_violation(
            text, source=sources[block.source_block_id], block_id=block.source_block_id
        )
        if violation is not None:
            return violation
    return None


def _text_violation(text: str, *, source: str, block_id: str) -> str | None:
    """Truncation indicator for a single translation relative to its original."""
    if _ends_with_punctuation(source) and not _ends_with_punctuation(text):
        return f"перевод блока {block_id} не заканчивается терминальной пунктуацией"
    unbalanced = _unbalanced(text, source=source)
    if unbalanced is not None:
        return f"в переводе блока {block_id} не закрыты {unbalanced}"
    return None


def _ends_with_punctuation(text: str) -> bool:
    """Whether the text ends with a sentence-terminal punctuation mark."""
    stripped = text.rstrip().rstrip(_TRAILING_CLOSERS).rstrip()
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCTUATION


def _unbalanced(text: str, *, source: str) -> str | None:
    """Type of paired punctuation whose imbalance was introduced in the translation."""
    original = _balances(source)
    for name, balance in _balances(text).items():
        if balance != 0 and balance != original[name]:
            return name
    return None


def _balances(text: str) -> Mapping[str, int]:
    """Imbalance for each type of paired punctuation: number of open marks left unclosed."""
    balances = {
        f"скобки {opening}{closing}": text.count(opening) - text.count(closing)
        for opening, closing in _BRACKETS
    }
    # « and „ are always openings. » and ” are always closings.
    # “ is closing when paired with „ (Russian/German „...“), and opening when paired with ” (English “...”).
    n_low = text.count("„")
    n_high_open = text.count("“")
    n_high_close = text.count("”")
    paired_low = min(n_low, n_high_open)
    openings = text.count("«") + n_low + (n_high_open - paired_low)
    closings = text.count("»") + n_high_close + paired_low
    balances["кавычки"] = (openings - closings) + text.count(_SYMMETRIC_QUOTE) % 2
    return balances


__all__ = ["truncation_violation"]
