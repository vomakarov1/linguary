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

from collections.abc import Iterator, Mapping, Sequence

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

#: Typographic apostrophes the model uses instead of the straight marker spelling.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "‛": "'"})


def service_message_violation(
    payload: TranslationResultResponse,
    *,
    strict_fields: bool = False,
    protected_quotes: Sequence[str] = (),
    excluded_texts: Mapping[str, frozenset[str]] | None = None,
    include_assistances: bool = True,
) -> str | None:
    """Violation description of detected model service message, or `None`."""
    for block in payload.blocks:
        excluded = _excluded_for(block.source_block_id, excluded_texts)
        for text in _texts_of(payload, block=block, strict_fields=strict_fields):
            if text in excluded:
                continue
            marker = _marker_in(_mask_protected(text, protected_quotes))
            if marker is not None:
                return (
                    f"результат блока {block.source_block_id} содержит служебное "
                    f"сообщение модели: {marker!r}"
                )
    if strict_fields and include_assistances:
        carried = _carried_texts(excluded_texts)
        for assistance in payload.fragment_assistances:
            text = assistance.text
            if not isinstance(text, str) or not text.strip() or text in carried:
                continue
            marker = _marker_in(_mask_protected(text, protected_quotes))
            if marker is not None:
                return (
                    f"помощь к трудности {assistance.difficulty_id} содержит служебное "
                    f"сообщение модели: {marker!r}"
                )
    return None


def _excluded_for(
    block_id: str, excluded_texts: Mapping[str, frozenset[str]] | None
) -> frozenset[str]:
    if not excluded_texts:
        return frozenset()
    return excluded_texts.get(block_id, frozenset())


def _carried_texts(
    excluded_texts: Mapping[str, frozenset[str]] | None,
) -> frozenset[str]:
    """Assistance has no block id, so any carried reader string can excuse it."""
    if not excluded_texts:
        return frozenset()
    return frozenset(text for texts in excluded_texts.values() for text in texts)


def _texts_of(
    payload: TranslationResultResponse,
    *,
    block: BlockResultResponse,
    strict_fields: bool,
) -> Iterator[str]:
    """Texts of a single result block written by the model."""
    if block.translated_text is not None:
        yield block.translated_text
    if strict_fields:
        for segment in block.segments:
            yield segment.translated_text
    for hint in block.glossary_hints:
        yield hint.translation
        if hint.note is not None:
            yield hint.note
    for note in payload.notes_for(block.source_block_id):
        yield note.note
        if strict_fields and note.translated_excerpt is not None:
            yield note.translated_excerpt


def _mask_protected(text: str, protected_quotes: Sequence[str]) -> str:
    """Hide verbatim source quotes before marker search, longest match first."""
    quotes = sorted(
        {quote for quote in protected_quotes if quote}, key=len, reverse=True
    )
    masked = text
    for quote in quotes:
        masked = masked.replace(quote, " ")
    return masked


def _marker_in(text: str) -> str | None:
    """First service message marker found in the text."""
    lowered = text.translate(_APOSTROPHES).lower()
    return next(
        (marker for marker in SERVICE_MESSAGE_MARKERS if marker in lowered), None
    )


__all__ = ["SERVICE_MESSAGE_MARKERS", "service_message_violation"]
