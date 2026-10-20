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

Заметки удаляются только при подтверждённом повторе объяснения. Неизвестные
самостоятельные утверждения сохраняются для содержательной проверки.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace

from linguary.domain import BlockProcessingResult, LanguageNote

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

    __slots__ = ("_expressions", "_explanations")

    def __init__(
        self, expressions: Iterable[str], *,
        explanations: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._expressions = tuple(
            words for expression in expressions if (words := expression_words(expression))
        )

        self._explanations = tuple(
            (expression_words(expression), explanation)
            for expression, explanation in explanations if expression_words(expression)
        )

    def __bool__(self) -> bool:
        return bool(self._expressions)

    def covers(self, text: str) -> bool:
        """Объясняет ли `Before You Read` то же выражение, что и `text`."""
        words = expression_words(text)
        return bool(words) and any(
            _overlaps(words, expression) for expression in self._expressions
        )

    def repeated_by(self, note: LanguageNote) -> bool:
        """Повторяет ли языковая заметка выражение `Before You Read`.

        Отрывок определяет кандидата, включая значимую фразу внутри длинного
        предложения. Удаление требует покрытия всех утверждений объяснением.
        """
        excerpt = expression_words(note.original_excerpt)
        return any(
            (_overlaps(excerpt, expression) or _contains(excerpt, expression))
            and _same_explanation(expression, explanation, note.note)
            for expression, explanation in self._explanations
        )

    def without_repeats_v2(self, result: BlockProcessingResult) -> BlockProcessingResult:
        """Удаляет лишь точные повторы объясняющего текста; новые аспекты остаются."""
        bodies = {text.strip() for _, text in self._explanations if text.strip()}
        hints = tuple(h for h in result.glossary_hints if not h.note or h.note.strip() not in bodies)
        if not hints and result.treatment.requires_glossary_hints:
            hints = result.glossary_hints
        bodies.update(h.note.strip() for h in hints if h.note and h.note.strip())
        notes = tuple(n for n in result.language_notes if n.note.strip() not in bodies)
        return replace(result, glossary_hints=hints, language_notes=notes)

    def without_repeats(self, result: BlockProcessingResult) -> BlockProcessingResult:
        """Результат без повторов подготовки к чтению и подсказок того же блока.

        Повторная заметка отбрасывается целиком. Подсказка — тоже, кроме одного случая:
        блоку `GLOSSARY_ONLY` подсказки обязательны, и если повторами оказались все его
        подсказки, они остаются. Читатель, пропустивший подготовку к чтению, иначе
        остался бы у трудного абзаца без помощи, а обработка главы из-за учебного
        повтора не должна срываться.
        """
        if not self and not result.glossary_hints:
            return result
        hints = tuple(
            hint for hint in result.glossary_hints if not self.covers(hint.original)
        )
        if not hints and result.treatment.requires_glossary_hints:
            hints = result.glossary_hints
        notes = tuple(
            note
            for note in result.language_notes
            if not self.repeated_by(note)
            and not _repeats_block_hint(note, result)
        )
        if hints == result.glossary_hints and notes == result.language_notes:
            return result
        return replace(result, glossary_hints=hints, language_notes=notes)


def _repeats_block_hint(note: LanguageNote, result: BlockProcessingResult) -> bool:
    """Повторяет ли заметка уже данное объяснение подсказки своего блока."""
    excerpt = expression_words(note.original_excerpt)
    return any(
        (expression := expression_words(hint.original))
        and (_overlaps(excerpt, expression) or _contains(excerpt, expression))
        and _same_explanation(expression, hint.translation, note.note)
        for hint in result.glossary_hints
    )


# Конечные словари описывают только два повтора, подтверждённых рецензией.
# Неизвестное слово в утверждении запрещает удаление даже рядом с известным значением.
_GRAB_WORDS = "make a grab at my throat разговорн идиом значени резко порывист схват ухват попыт выпад сдела рукой сторон чего либо движение"
_ARMOUR_WORDS = "sheath armour his and ножн оболоч брон доспех скры защит защищ отраж удар оруж враг граф скептицизм одновременно"
_CONNECTORS = frozenset("и или это то что здесь как его с со в от на для только но также".split())


def _features(expression: Sequence[str], text: str) -> frozenset[str]:
    words = expression_words(text)
    if _contains(expression, ("make", "a", "grab", "at")):
        return frozenset({"grab"}) if any(
            word.startswith(("схват", "ухват")) for word in words
        ) else frozenset()
    if "sheath" in expression and "armour" in expression:
        features = set()
        if any(word.startswith(("ножн", "оболоч")) for word in words) and any(
            word.startswith("скры") for word in words
        ):
            features.add("conceal")
        if any(word.startswith(("брон", "доспех")) for word in words) and any(
            word.startswith(("защит", "защищ", "отраж")) for word in words
        ):
            features.add("protect")
        return frozenset(features)
    return frozenset()


def _same_explanation(expression: Sequence[str], previous: str, current: str) -> bool:
    if expression_words(previous) == expression_words(current):
        return bool(expression_words(current))
    known = _features(expression, previous)
    if not known:
        return False
    vocabulary = (_GRAB_WORDS if "grab" in known else _ARMOUR_WORDS).split()
    claims = [claim for claim in re.split(r"[.;!?\n]+", current) if expression_words(claim)]
    for claim in claims:
        words = expression_words(claim)
        if any(word not in _CONNECTORS and not any(
            word.startswith(root) for root in vocabulary
        ) for word in words):
            return False
        features = _features(expression, claim)
        if not features or not features <= known:
            return False
    return bool(claims)


def _overlaps(first: Sequence[str], second: Sequence[str]) -> bool:
    """Совпадают ли два выражения целиком или значимой непрерывной частью."""
    if len(first) == len(second) and _same_words(first, second):
        return True
    shorter, longer = sorted((first, second), key=len)
    if 2 * len(shorter) < len(longer):
        return False
    return _contains(longer, shorter)


def _contains(text: Sequence[str], expression: Sequence[str]) -> bool:
    """Стоит ли значимое выражение в тексте целиком, подряд, с точностью до словоформ."""
    if not _significant(expression) or len(expression) > len(text):
        return False
    return any(
        _same_words(expression, text[start : start + len(expression)])
        for start in range(len(text) - len(expression) + 1)
    )


def _mentions(text: Sequence[str], expression: Sequence[str]) -> bool:
    """Называет ли текст выражение или его непрерывную часть не меньше половины слов."""
    shortest = (len(expression) + 1) // 2
    return any(
        _contains(text, expression[start : start + size])
        for size in range(len(expression), shortest - 1, -1)
        for start in range(len(expression) - size + 1)
    )


def _significant(expression: Sequence[str]) -> bool:
    """Годится ли выражение для частичного совпадения: не одно короткое служебное слово."""
    return bool(expression) and not (
        len(expression) == 1 and len(expression[0]) < _MIN_STEM
    )


def _same_words(first: Sequence[str], second: Sequence[str]) -> bool:
    """Попарно совпадают ли слова двух последовательностей одной длины."""
    return all(_same_word(left, right) for left, right in zip(first, second, strict=True))


def _same_word(first: str, second: str) -> bool:
    """Одно ли это слово: совпадение или словоформы с общей основой."""
    first = {"made": "make"}.get(first, first)
    second = {"made": "make"}.get(second, second)
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
