"""Проверка ответа обработки на соответствие сохранённому плану главы (ALG-04).

План — единственный источник истины о режиме блока: поле `treatment` в ответе модели его
подтверждает, но не переопределяет (PRD § 14). Отсюда две проверки: режим ответа совпадает
с планом, а состав результата отвечает режиму плана.

Вторая проверка сформулирована относительно плана намеренно. DTO `BlockResultResponse`
проверяет состав относительно того режима, который назвала сама модель, — этого мало:
согласованный сам с собой ответ может быть согласован не с тем режимом. После проверки
режима оба утверждения совпадают, и проверка состава остаётся явным утверждением ALG-04:
она сработает, если правила DTO когда-нибудь разойдутся с планом.
"""

from __future__ import annotations

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.domain import BlockTreatment, TranslationUnit


def treatment_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Описание расхождения режима ответа с планом либо `None`."""
    for block in payload.blocks:
        planned = unit.treatment_for(block.source_block_id)
        if block.treatment is not planned:
            return (
                f"режим блока {block.source_block_id} в ответе модели ({block.treatment}) "
                f"не совпадает с планом ({planned})"
            )
    return None


def content_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Описание несоответствия состава результата режиму плана либо `None`."""
    for block in payload.blocks:
        violation = _block_content_violation(
            block, planned=unit.treatment_for(block.source_block_id)
        )
        if violation is not None:
            return violation
    return None


def _block_content_violation(
    block: BlockResultResponse, *, planned: BlockTreatment
) -> str | None:
    """Состав результата одного блока относительно его режима из плана (PRD § 15)."""
    has_translation = bool(block.translated_text and block.translated_text.strip())
    if planned.requires_translation and not has_translation:
        return f"для блока {block.source_block_id} с режимом {planned} нет полного перевода"
    if not planned.allows_translation and block.translated_text is not None:
        return (
            f"для блока {block.source_block_id} с режимом {planned} "
            "полный перевод недопустим"
        )
    if planned.requires_glossary_hints and not block.glossary_hints:
        return (
            f"для блока {block.source_block_id} с режимом {planned} "
            "нет словарных подсказок"
        )
    if not planned.allows_glossary_hints and block.glossary_hints:
        return (
            f"для блока {block.source_block_id} с режимом {planned} "
            "словарные подсказки недопустимы"
        )
    return None


__all__ = ["content_violation", "treatment_violation"]
