"""Проверка на служебные сообщения модели в результате обработки (PRD § 15, ALG-04).

Модель иногда возвращает вместо перевода отказ, извинение или комментарий о собственной
работе. Формально ответ при этом корректен: блок на месте, режим совпадает, текст на
целевом языке — и без этой проверки такой «перевод» попал бы в книгу.

Перечень маркеров намеренно узкий. Художественный текст может содержать почти любую
фразу, поэтому маркером становится только оборот, который принадлежит модели, а не
рассказчику: разговор о себе как о языковой модели, отказ выполнить запрос, разметка
ответа кодовым блоком.
"""

from __future__ import annotations

from collections.abc import Iterator

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse

#: Обороты, которыми модель говорит о себе и о запросе вместо того, чтобы переводить.
SERVICE_MESSAGE_MARKERS: tuple[str, ...] = (
    "как языковая модель",
    "как большая языковая модель",
    "я языковая модель",
    "as an ai language model",
    "as a large language model",
    "я не могу выполнить",
    "не могу выполнить этот запрос",
    "извините, но я не могу",
    "i cannot fulfill",
    "i can't fulfill",
    "i'm sorry, but i cannot",
    "```",
)


def service_message_violation(payload: TranslationResultResponse) -> str | None:
    """Описание найденного служебного сообщения модели либо `None`."""
    for block in payload.blocks:
        for text in _texts_of(payload, block=block):
            marker = _marker_in(text)
            if marker is not None:
                return (
                    f"результат блока {block.source_block_id} содержит служебное "
                    f"сообщение модели: {marker!r}"
                )
    return None


def _texts_of(
    payload: TranslationResultResponse, *, block: BlockResultResponse
) -> Iterator[str]:
    """Тексты одного блока результата, которые пишет модель."""
    if block.translated_text is not None:
        yield block.translated_text
    for hint in block.glossary_hints:
        yield hint.translation
        if hint.note is not None:
            yield hint.note
    for note in payload.notes_for(block.source_block_id):
        yield note.note


def _marker_in(text: str) -> str | None:
    """Первый маркер служебного сообщения, найденный в тексте."""
    lowered = text.lower()
    return next(
        (marker for marker in SERVICE_MESSAGE_MARKERS if marker in lowered), None
    )


__all__ = ["SERVICE_MESSAGE_MARKERS", "service_message_violation"]
