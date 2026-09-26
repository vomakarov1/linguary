"""Указания переводчику главы, согласованные с решениями глоссария (PRD § 11, § 13).

Указания главы пишет модель на анализе — до того, как пользователь проверил глоссарий.
Промпт анализа запрещает назначать в них переводы имён и терминов, но модель запрет
нарушает: пересказывает свой вариант из кандидатов («Duskmere переводить по глоссарию —
„Даскмир“») и выбирает род персонажа. Пользователь потом исправляет решение в глоссарии,
а указание главы остаётся прежним и спорит с ним в каждом запросе обработки. Промпт
ставит решение глоссария выше указаний главы, но это тоже только просьба. Поэтому
указание, которое противоречит решению пользователя, в контекст обработки не попадает.

Противоречие определяется детерминированно и только там, где это надёжно:

- указание упоминает понятие решения: исходное написание (ключ `glossary_key`),
  вариант пользователя, вариант модели из анализа этой главы, у персонажа — ещё и
  отдельные части имени (указания обычно говорят «Рен», а не «Рен Колдер»);
- и называет в кавычках другой вариант перевода: вариант модели, отвергнутый
  пользователем, или слово, начинающееся так же, как вариант пользователя, но не
  являющееся его грамматической формой («Даскмир» при решении «Даскмер»);
- или называет грамматический род, отличный от рода в указании решения. Род
  сравнивается, только если решение его называет: без этого противоречить нечему.

Остальное — голос рассказчика, речь персонажа, регистр — остаётся, даже если упоминает
понятие. Указание, спорящее с решением иначе (перевод без кавычек, пересказ своими
словами), здесь не узнаётся: его перекрывает приоритет решения в промпте.

Указания делятся на пункты по строкам: модель пишет их нумерованным списком. Строка без
маркера списка продолжает предыдущий пункт. Сплошной абзац без списка делится на
предложения, чтобы противоречие не уносило с собой весь текст указаний.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from linguary.domain import (
    ConceptType,
    GlossaryCandidate,
    GlossarySnapshotEntry,
    glossary_key,
)

#: Маркер пункта списка в начале строки: `1.`, `2)`, `-`, `•`, `*`.
_ITEM_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-•*–—])\s")

#: Граница предложения в сплошном абзаце указаний.
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")

#: Фрагмент в кавычках: ёлочки, лапки, английские и прямые двойные кавычки.
_QUOTED = re.compile(r"«([^»]+)»|„([^“”]+)[“”]|“([^”]+)”|\"([^\"]+)\"")

#: Слово: буквы и цифры, возможно с внутренним апострофом или дефисом.
_WORD = re.compile(r"\w+(?:['’-]\w+)*")

#: Грамматический род в указании: «женский род», «в мужском роде».
_GENDER = re.compile(r"\b(мужск|женск|средн)\w*\s+род\w*", re.IGNORECASE)

#: Обращения, которые частью имени не считаются.
_TITLES = frozenset(
    {
        *("mr", "mrs", "ms", "dr", "st", "sir", "lady", "lord", "miss"),
        *("мистер", "миссис", "мисс", "доктор", "сэр", "леди", "лорд"),
    }
)

#: Наименьшая длина части имени, по которой указание связывается с персонажем.
_MIN_NAME_PART = 3

#: Наибольшее окончание словоформы сверх общей основы и наименьшая общая основа
#: двух слов, по которой кавычки считаются вариантом перевода того же понятия.
_MAX_ENDING = 3
_MIN_SHARED_PREFIX = 3


@dataclass(frozen=True, slots=True)
class _Decision:
    """Решение пользователя в том виде, в каком с ним сравниваются указания."""

    mentions: tuple[tuple[str, ...], ...]
    translation: tuple[str, ...]
    source_key: str
    rejected: frozenset[str]
    rejected_words: tuple[tuple[str, ...], ...]
    genders: frozenset[str]


def instructions_respecting_glossary(
    instructions: str,
    *,
    decisions: Sequence[GlossarySnapshotEntry],
    chapter_candidates: Iterable[GlossaryCandidate] = (),
) -> str:
    """Указания главы без пунктов, противоречащих решениям пользователя.

    `decisions` — обязательные решения снимка запуска, `chapter_candidates` — кандидаты
    анализа этой главы: их варианты перевода указания главы и пересказывают.
    """
    if not decisions or not instructions.strip():
        return instructions.strip()
    prepared = _prepare(decisions, chapter_candidates)
    kept = [item for item in _items(instructions) if not _contradicts(item, prepared)]
    return "\n".join(kept).strip()


def note_respecting_glossary(
    entry: GlossarySnapshotEntry,
    *,
    decisions: Sequence[GlossarySnapshotEntry],
    chapter_candidates: Iterable[GlossaryCandidate] = (),
) -> str:
    """Указание элемента глоссария без фрагментов, спорящих с решениями других понятий.

    Указание элемента, принятого через «Принять все», написала модель на анализе, и оно
    пересказывает её варианты других понятий: «пост при въезде в Даскмир», хотя
    решение пользователя по Duskmere — «Даскмер». Такой фрагмент уходил бы в блок
    обязательных решений рядом с самим решением и спорил с ним.

    Хранилище не различает указание, введённое пользователем, и указание модели,
    принятое без правки, поэтому отсеивается только противоречие решению пользователя
    по другому понятию: собственное решение элемента его указание вправе обсуждать.
    Кроме правил указаний главы здесь противоречием считается и упоминание без кавычек
    отвергнутого варианта модели (`chapter_candidates`) — в указании элемента другое
    понятие называют вариантом перевода, а не исходным написанием.
    """
    note = entry.note.strip()
    key = glossary_key(entry.source_text)
    others = [
        decision
        for decision in decisions
        if decision.status.is_user_decision
        and glossary_key(decision.source_text) != key
    ]
    if not others or not note:
        return note
    prepared = _prepare(others, chapter_candidates)
    items = _items(note)
    kept = [
        item
        for item in items
        if not _contradicts(item, prepared) and not _names_rejected(item, prepared)
    ]
    if len(kept) == len(items):
        return note
    return " ".join(line.strip() for item in kept for line in item.splitlines()).strip()


def _names_rejected(item: str, decisions: Sequence[_Decision]) -> bool:
    """Называет ли фрагмент, хотя бы без кавычек, вариант, отвергнутый пользователем."""
    words = _words(item)
    return any(
        _contains(words, rejected)
        for decision in decisions
        for rejected in decision.rejected_words
    )


def _prepare(
    decisions: Sequence[GlossarySnapshotEntry],
    chapter_candidates: Iterable[GlossaryCandidate],
) -> tuple[_Decision, ...]:
    """Решения с формами упоминания, отвергнутыми вариантами и родом."""
    suggested: dict[str, set[str]] = {}
    for candidate in chapter_candidates:
        suggested.setdefault(glossary_key(candidate.source_text), set()).add(
            candidate.suggested_translation
        )
    prepared = []
    for entry in decisions:
        key = glossary_key(entry.source_text)
        translation_key = glossary_key(entry.translation)
        rejected = {
            glossary_key(text)
            for text in suggested.get(key, ())
            if glossary_key(text) != translation_key
        }
        forms = [entry.source_text, entry.translation, *suggested.get(key, ())]
        mentions = {_words(form) for form in forms}
        if entry.type is ConceptType.PERSON:
            mentions.update(
                (part,)
                for form in forms
                for part, original in zip(_words(form), _WORD.findall(form), strict=True)
                if original[:1].isupper()
                and len(part) >= _MIN_NAME_PART
                and part not in _TITLES
            )
        prepared.append(
            _Decision(
                mentions=tuple(mention for mention in mentions if mention),
                translation=_words(entry.translation),
                source_key=key,
                rejected=frozenset(rejected),
                rejected_words=tuple(
                    words
                    for text in suggested.get(key, ())
                    if glossary_key(text) in rejected
                    and (words := _words(text))
                    and not _is_form_of(words, _words(entry.translation))
                ),
                genders=_genders(entry.note),
            )
        )
    return tuple(prepared)


def _items(instructions: str) -> list[str]:
    """Пункты указаний: строки списка вместе с продолжениями либо предложения абзаца."""
    items: list[str] = []
    marked = False
    for line in instructions.strip().splitlines():
        if not line.strip():
            continue
        if _ITEM_MARKER.match(line):
            marked = True
            items.append(line.rstrip())
        elif items and marked:
            items[-1] = f"{items[-1]}\n{line.rstrip()}"
        else:
            items.append(line.rstrip())
    if marked or len(items) != 1:
        return items
    return [sentence for sentence in _SENTENCE_END.split(items[0]) if sentence.strip()]


def _contradicts(item: str, decisions: Sequence[_Decision]) -> bool:
    """Противоречит ли пункт указаний хотя бы одному решению пользователя."""
    words = _words(item)
    genders = _genders(item)
    quoted = tuple(
        next(group for group in match.groups() if group is not None)
        for match in _QUOTED.finditer(item)
    )
    for decision in decisions:
        if not any(_contains(words, mention) for mention in decision.mentions):
            continue
        if decision.genders and genders and not genders <= decision.genders:
            return True
        if any(_other_translation(text, decision) for text in quoted):
            return True
    return False


def _other_translation(quoted: str, decision: _Decision) -> bool:
    """Называет ли фрагмент в кавычках вариант перевода, отличный от решения."""
    key = glossary_key(quoted)
    if key == decision.source_key:
        return False
    if key in decision.rejected:
        return True
    words = _words(quoted)
    if not words or _is_form_of(words, decision.translation):
        return False
    return any(
        _shared_prefix(word, other) >= _MIN_SHARED_PREFIX and not _same_word(word, other)
        for word in words
        for other in decision.translation
    )


def _is_form_of(words: Sequence[str], translation: Sequence[str]) -> bool:
    """Грамматическая ли это форма варианта пользователя или его части."""
    return all(any(_same_word(word, other) for other in translation) for word in words)


def _contains(words: Sequence[str], mention: Sequence[str]) -> bool:
    """Стоит ли упоминание в тексте подряд, с точностью до словоформ."""
    size = len(mention)
    return any(
        all(
            _same_word(word, expected)
            for word, expected in zip(words[start : start + size], mention, strict=True)
        )
        for start in range(len(words) - size + 1)
    )


def _same_word(word: str, expected: str) -> bool:
    """Словоформа ли `word` слова `expected`: общая основа и короткое окончание.

    Основа — всё слово, кроме последней буквы: у русского имени она меняется
    («Селена» — «Селены»), а у английского совпадает целиком.
    """
    if word == expected:
        return True
    stem = max(len(expected) - 1, _MIN_NAME_PART)
    return (
        _shared_prefix(word, expected) >= stem
        and abs(len(word) - len(expected)) <= _MAX_ENDING
    )


def _shared_prefix(first: str, second: str) -> int:
    """Длина общего начала двух слов."""
    length = 0
    for left, right in zip(first, second, strict=False):
        if left != right:
            break
        length += 1
    return length


def _words(text: str) -> tuple[str, ...]:
    """Слова текста без регистра."""
    return tuple(word.casefold().replace("’", "'") for word in _WORD.findall(text))


def _genders(text: str) -> frozenset[str]:
    """Грамматические рода, которые называет текст."""
    return frozenset(match.group(1).casefold() for match in _GENDER.finditer(text))


__all__ = ["instructions_respecting_glossary", "note_respecting_glossary"]
