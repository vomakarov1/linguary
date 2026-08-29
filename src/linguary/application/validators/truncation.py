"""Проверка оборванной генерации по формальным признакам (PRD § 15, ALG-04).

Признаков три: провайдер сообщил об обрыве по лимиту токенов, перевод не заканчивается
терминальной пунктуацией, в переводе не закрыта пара кавычек или скобок. Первый признак
сообщает сам провайдер (`LLMResponse.truncated`) — в обработке его проверка обязательна:
ответ может быть полным по составу блоков и обрезанным по тексту, и проверка покрытия,
которая закрывает эту дыру в анализе главы, здесь ничего не заметит.

Два текстовых признака зеркалят оригинал, иначе они ловят не обрыв, а нормальный текст.
Заголовок, подпись и элемент списка законно заканчиваются без точки, а многоабзацная
прямая речь по литературной конвенции открывает кавычку в каждом абзаце и закрывает её
только в последнем. Поэтому терминальная пунктуация требуется от перевода лишь там, где
ею заканчивается исходный блок, а незакрытая пара считается обрывом лишь тогда, когда
в оригинале такого дисбаланса не было.
"""

from __future__ import annotations

from collections.abc import Mapping

from linguary.application.llm_dto import TranslationResultResponse
from linguary.domain import TranslationUnit

#: Знаки, которыми заканчивается дописанное предложение.
_TERMINAL_PUNCTUATION = frozenset(".!?…:;")

#: Закрывающие знаки, которые могут стоять после терминальной пунктуации.
_TRAILING_CLOSERS = "»\"”’'*)]}"

#: Скобки, дисбаланс которых означает недописанный текст.
_BRACKETS: tuple[tuple[str, str], ...] = (("(", ")"), ("[", "]"), ("{", "}"))

#: Кавычки считаются одним видом пары независимо от начертания: оригинал и перевод
#: оформляют прямую речь по правилам своих языков, и `"Go"` становится «Иди».
_QUOTE_OPENINGS = "«“„"
_QUOTE_CLOSINGS = "»”"

#: Симметричная кавычка: пара определяется чётностью числа вхождений.
_SYMMETRIC_QUOTE = '"'


def truncation_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit, truncated: bool
) -> str | None:
    """Описание признака оборванной генерации либо `None`.

    `truncated` — признак от провайдера: ответ обрезан пределом вывода, а не закончен
    моделью по смыслу.
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
    """Признак обрыва одного перевода относительно его оригинала."""
    if _ends_with_punctuation(source) and not _ends_with_punctuation(text):
        return f"перевод блока {block_id} не заканчивается терминальной пунктуацией"
    unbalanced = _unbalanced(text, source=source)
    if unbalanced is not None:
        return f"в переводе блока {block_id} не закрыты {unbalanced}"
    return None


def _ends_with_punctuation(text: str) -> bool:
    """Заканчивается ли текст знаком конца предложения."""
    stripped = text.rstrip().rstrip(_TRAILING_CLOSERS).rstrip()
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCTUATION


def _unbalanced(text: str, *, source: str) -> str | None:
    """Вид парных знаков, дисбаланс которых появился именно в переводе."""
    original = _balances(source)
    for name, balance in _balances(text).items():
        if balance != 0 and balance != original[name]:
            return name
    return None


def _balances(text: str) -> Mapping[str, int]:
    """Дисбаланс каждого вида парных знаков: сколько открытых осталось незакрытыми."""
    balances = {
        f"скобки {opening}{closing}": text.count(opening) - text.count(closing)
        for opening, closing in _BRACKETS
    }
    balances["кавычки"] = (
        sum(text.count(mark) for mark in _QUOTE_OPENINGS)
        - sum(text.count(mark) for mark in _QUOTE_CLOSINGS)
        + text.count(_SYMMETRIC_QUOTE) % 2
    )
    return balances


__all__ = ["truncation_violation"]
