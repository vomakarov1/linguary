"""Detection of conspicuous repetitions in translation (PRD § 15, ALG-04).

A looping model repeats the same phrase consecutively, visible via repeating n-grams.
This check remains a warning: literary text repeats phrases intentionally — refrains,
anaphora, character speech habits — and blocking would cause futile retries instead
of producing a result.
"""

from __future__ import annotations

import re

#: Translated word: sequence of letters or digits. Punctuation and quotes are ignored —
#: the same phrase in different surroundings remains the same phrase.
_WORD = re.compile(r"\w+", re.UNICODE)


def repetition_warning(text: str, *, ngram_size: int, min_repeats: int) -> str | None:
    """Description of the most frequent repetition in text, or `None` if no repetitions found."""
    words = _WORD.findall(text.lower())
    if len(words) < ngram_size:
        return None
    counts: dict[tuple[str, ...], int] = {}
    for start in range(len(words) - ngram_size + 1):
        ngram = tuple(words[start : start + ngram_size])
        counts[ngram] = counts.get(ngram, 0) + 1
    ngram, repeats = max(counts.items(), key=lambda item: item[1])
    if repeats < min_repeats:
        return None
    return (
        f"оборот «{' '.join(ngram)}» повторяется в переводе, "
        f"число вхождений: {repeats}"
    )


__all__ = ["repetition_warning"]
