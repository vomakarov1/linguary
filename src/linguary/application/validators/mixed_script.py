"""Проверка на слова из смеси латиницы и кириллицы (PRD § 15, ALG-04).

Модель иногда склеивает исходное написание имени с русским окончанием — «Selenой Роу»
вместо «Селеной Роу». Такое слово не принадлежит ни одному языку и читателю ничего
не даёт, поэтому результат с ним отбраковывается и уходит на повтор с уточнением.

Словом здесь считается непрерывная последовательность букв: цифра, дефис, апостроф,
пробел и знаки препинания слово разделяют. Поэтому латинские имена и термины в русском
тексте («Wren», «Duskmere»), метки сносок `[^1]`, адреса, числа с единицами («5kg»)
и дефисные соединения двух слов («Wonka-фабрика», «mp3-плеер») нарушением не считаются:
в них каждое слово целиком написано одним алфавитом. Отбраковывается только слово,
в котором буквы двух алфавитов стоят вплотную.

Проверяются все тексты результата, которые модель пишет на целевом языке: перевод,
перевод и пояснение словарной подсказки, заметка и её перевод отрывка. Исходные
отрывки (`original`, `original_excerpt`) — цитаты оригинала, их не проверяем.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse

#: Слово: непрерывная последовательность букв без цифр и знаков.
_LETTERS = re.compile(r"[^\W\d_]+")


def mixed_script_violation(payload: TranslationResultResponse) -> str | None:
    """Описание первого слова из смеси латиницы и кириллицы либо `None`."""
    for block in payload.blocks:
        for text in _texts_of(payload, block=block):
            word = mixed_script_word(text)
            if word is not None:
                return (
                    f"результат блока {block.source_block_id} содержит слово из смеси "
                    f"латиницы и кириллицы: {word!r}. Каждое слово пиши одним "
                    f"алфавитом: имя из глоссария — в его варианте перевода с русским "
                    f"окончанием, исходное написание — без русского окончания"
                )
    return None


def mixed_script_word(text: str) -> str | None:
    """Первое слово текста, в котором стоят и латинские, и кириллические буквы."""
    for match in _LETTERS.finditer(text):
        word = match.group()
        if any(_is_latin(letter) for letter in word) and any(
            _is_cyrillic(letter) for letter in word
        ):
            return word
    return None


def _is_latin(letter: str) -> bool:
    """Латинская ли буква, включая буквы с диакритикой (é, ß)."""
    return letter.isascii() or "À" <= letter <= "ɏ"


def _is_cyrillic(letter: str) -> bool:
    """Кириллическая ли буква."""
    return "Ѐ" <= letter <= "ӿ"


def _texts_of(
    payload: TranslationResultResponse, *, block: BlockResultResponse
) -> Iterator[str]:
    """Тексты блока результата, которые модель пишет на целевом языке."""
    if block.translated_text is not None:
        yield block.translated_text
    for hint in block.glossary_hints:
        yield hint.translation
        if hint.note is not None:
            yield hint.note
    for note in payload.notes_for(block.source_block_id):
        yield note.note
        if note.translated_excerpt is not None:
            yield note.translated_excerpt


__all__ = ["mixed_script_violation", "mixed_script_word"]
