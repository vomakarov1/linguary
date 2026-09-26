"""Выражения главы, уже объяснённые в `Before You Read`, и отсев их повторов (PRD § 19.3).

Одно место книги объясняется читателю один раз. Модель получает список выражений
`Before You Read` вместе с запретом их повторять, но запрет в промпте — только просьба:
ответ, где выражение всё же вернулось словарной подсказкой или языковой заметкой, проходит
все блокирующие проверки. Поэтому повтор отсеивается здесь, детерминированно, уже после
проверки ответа.

Сравнение идёт по словам, а не по строкам: регистр, пунктуация, пробелы и вид апострофа
не различаются, а словоформы одного слова (`heave` и `heaved`, `groans` и `groaned`)
считаются одним словом. Частичное совпадение засчитывается, только если общий фрагмент
занимает не меньше половины более длинного выражения: заметка к целому предложению,
в котором встретилось объяснённое выражение, объясняет уже другое и остаётся. Короткое
служебное слово частичным совпадением не считается вовсе — иначе подсказка к `to`
пропала бы из-за объяснённого `heave to`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace

from linguary.domain import BlockProcessingResult

#: Разновидности апострофа, которые в тексте книги и в ответе модели встречаются вперемешку.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})

#: Слово выражения: буквы и цифры, возможно со внутренним апострофом (`one's`, `don't`).
_WORD = re.compile(r"\w+(?:'\w+)*")

#: Минимальная общая основа двух словоформ одного слова и наибольшее окончание сверх неё.
_MIN_STEM = 4
_MAX_ENDING = 3


def expression_words(text: str) -> tuple[str, ...]:
    """Слова выражения в нормализованном виде: без регистра, пунктуации и лишних пробелов."""
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    return tuple(_WORD.findall(normalized))


class ExplainedExpressions:
    """Выражения `Before You Read` одной главы, подготовленные к сравнению."""

    __slots__ = ("_expressions",)

    def __init__(self, expressions: Iterable[str]) -> None:
        self._expressions = tuple(
            words for expression in expressions if (words := expression_words(expression))
        )

    def __bool__(self) -> bool:
        return bool(self._expressions)

    def covers(self, text: str) -> bool:
        """Объясняет ли `Before You Read` то же выражение, что и `text`."""
        words = expression_words(text)
        return bool(words) and any(
            _overlaps(words, expression) for expression in self._expressions
        )

    def without_repeats(self, result: BlockProcessingResult) -> BlockProcessingResult:
        """Результат блока без подсказок и заметок, повторяющих `Before You Read`.

        Повторная заметка отбрасывается целиком. Подсказка — тоже, кроме одного случая:
        блоку `GLOSSARY_ONLY` подсказки обязательны, и если повторами оказались все его
        подсказки, они остаются. Читатель, пропустивший подготовку к чтению, иначе
        остался бы у трудного абзаца без помощи, а обработка главы из-за учебного
        повтора не должна срываться.
        """
        if not self:
            return result
        hints = tuple(
            hint for hint in result.glossary_hints if not self.covers(hint.original)
        )
        if not hints and result.treatment.requires_glossary_hints:
            hints = result.glossary_hints
        notes = tuple(
            note
            for note in result.language_notes
            if not self.covers(note.original_excerpt)
        )
        if hints == result.glossary_hints and notes == result.language_notes:
            return result
        return replace(result, glossary_hints=hints, language_notes=notes)


def _overlaps(first: Sequence[str], second: Sequence[str]) -> bool:
    """Совпадают ли два выражения целиком или значимой непрерывной частью."""
    if len(first) == len(second) and _same_words(first, second):
        return True
    shorter, longer = sorted((first, second), key=len)
    if 2 * len(shorter) < len(longer):
        return False
    if len(shorter) == 1 and len(shorter[0]) < _MIN_STEM:
        return False
    return any(
        _same_words(shorter, longer[start : start + len(shorter)])
        for start in range(len(longer) - len(shorter) + 1)
    )


def _same_words(first: Sequence[str], second: Sequence[str]) -> bool:
    """Попарно совпадают ли слова двух последовательностей одной длины."""
    return all(_same_word(left, right) for left, right in zip(first, second, strict=True))


def _same_word(first: str, second: str) -> bool:
    """Одно ли это слово: совпадение или словоформы с общей основой."""
    if first == second:
        return True
    stem = next(
        (
            index
            for index, (left, right) in enumerate(zip(first, second, strict=False))
            if left != right
        ),
        min(len(first), len(second)),
    )
    return stem >= _MIN_STEM and max(len(first), len(second)) - stem <= _MAX_ENDING


__all__ = ["ExplainedExpressions", "expression_words"]
