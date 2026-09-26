"""Проверка языка результата обработки (PRD § 15, ALG-04).

Соответствие языку определяется долей букв целевого алфавита среди всех букв текста.
Проверка применяется только к полным переводам длиннее настраиваемого минимума: короткая
строка законно состоит из имени собственного, а словарная подсказка — из слова исходного
языка, ради которого она и написана.

Алфавит известен только для целевого языка MVP — русского (PRD § 1). Для другого целевого
языка доля «символов целевого алфавита» не определена, и проверка не выполняется: это
честнее, чем считать долю чужого алфавита и отбраковывать верный перевод.
"""

from __future__ import annotations

from linguary.application.llm_dto import TranslationResultResponse

#: Целевые языки, алфавит которых приложение знает. Ключ — код языка книги без региона.
_TARGET_ALPHABETS: dict[str, tuple[str, str]] = {"ru": ("Ѐ", "ӿ")}


def language_violation(
    payload: TranslationResultResponse,
    *,
    target_language: str,
    min_share: float,
    min_length: int,
) -> str | None:
    """Описание несоответствия языка перевода целевому либо `None`."""
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
    """Код языка книги без региона и регистра: `ru-RU` и `RU` — один и тот же русский."""
    return target_language.strip().lower().split("-")[0]


def _alphabet_share(text: str, *, alphabet: tuple[str, str]) -> float | None:
    """Доля букв целевого алфавита среди букв текста; `None`, если букв нет."""
    first, last = alphabet
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return None
    target = sum(1 for character in letters if first <= character <= last)
    return target / len(letters)


__all__ = ["language_violation"]
