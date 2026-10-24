"""Validation of processing response against the saved chapter plan (ALG-04).

The plan is the single source of truth for a block's treatment mode: the `treatment` field
in the model response confirms it, but does not override it (PRD § 14). This yields two checks:
the response treatment matches the plan, and the result content matches the plan's treatment.

The second check is formulated relative to the plan intentionally. The `BlockResultResponse` DTO
validates content relative to the mode declared by the model itself — which is insufficient:
a self-consistent response might be consistent with the wrong treatment mode. Once the mode check
passes, both assertions align, and content validation remains an explicit ALG-04 invariant:
it triggers if DTO rules ever diverge from the plan.
"""

from __future__ import annotations

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.domain import BlockTreatment, TranslationUnit


def treatment_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Violation description of mismatch between response treatment and plan, or `None`."""
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
    """Violation description of result content mismatching the planned treatment, or `None`."""
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
    """Result content of a single block relative to its planned treatment mode (PRD § 15)."""
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
