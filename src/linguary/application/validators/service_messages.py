"""Validation against model service messages in processing results (PRD § 15, ALG-04).

The model occasionally returns a refusal, apology, or meta-commentary about its own operation
instead of a translation. Formally the response remains valid: block present, mode matches, text
in target language — and without this check such "translation" would end up in the book.

The list of markers is intentionally narrow. Literary text may contain almost any phrase,
so only phrases characteristic of the model rather than the narrator are considered markers:
referring to itself as a language model, refusing to fulfill the request, or wrapping responses
in code fences.
"""

from __future__ import annotations

from collections.abc import Iterator

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse

#: Markers with which the model talks about itself and the request instead of translating.
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
    """Violation description of detected model service message, or `None`."""
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
    """Texts of a single result block written by the model."""
    if block.translated_text is not None:
        yield block.translated_text
    for hint in block.glossary_hints:
        yield hint.translation
        if hint.note is not None:
            yield hint.note
    for note in payload.notes_for(block.source_block_id):
        yield note.note


def _marker_in(text: str) -> str | None:
    """First service message marker found in the text."""
    lowered = text.lower()
    return next(
        (marker for marker in SERVICE_MESSAGE_MARKERS if marker in lowered), None
    )


__all__ = ["SERVICE_MESSAGE_MARKERS", "service_message_violation"]
