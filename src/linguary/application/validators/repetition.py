"""Обнаружение явных повторов в переводе (PRD § 15, ALG-04).

Зациклившаяся модель повторяет один и тот же оборот подряд, и это видно по повторяющимся
n-граммам. Проверка остаётся предупреждением: художественный текст повторяет обороты
намеренно — рефрен, анафора, речевая привычка персонажа, — и блокировка приводила бы
к бесполезным повторным запросам вместо результата.
"""

from __future__ import annotations

import re

#: Слово перевода: последовательность букв или цифр. Пунктуация и кавычки в счёт не идут —
#: один и тот же оборот в разном окружении остаётся тем же оборотом.
_WORD = re.compile(r"\w+", re.UNICODE)


def repetition_warning(text: str, *, ngram_size: int, min_repeats: int) -> str | None:
    """Описание самого частого повтора в тексте либо `None`, если повторов нет."""
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
