"""Проверка, что перевод не вобрал абзацы соседних блоков."""

from linguary.application.llm_dto import TranslationResultResponse
from linguary.domain import TranslationUnit


def paragraph_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Причина превышения числа абзацев перевода либо `None`."""
    sources = {item.block.id: item.block.text for item in unit.items}
    for block in payload.blocks:
        if block.translated_text is None:
            continue
        translated_count = _paragraph_count(block.translated_text)
        source_count = _paragraph_count(sources[block.source_block_id])
        if translated_count > source_count:
            return (
                f"перевод блока {block.source_block_id} содержит {translated_count} "
                f"абзацев, а оригинал — {source_count}: переводи только текст этого "
                "блока, не включай соседние"
            )
    return None


def _paragraph_count(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


__all__ = ["paragraph_violation"]
