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
from linguary.domain import BlockTreatment, TranslationUnit, TranslationUnitItem


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
    items = {item.block.id: item for item in unit.items}
    for block in payload.blocks:
        item = items.get(block.source_block_id)
        if item is None:
            continue
        violation = _block_content_violation(block, item=item)
        if violation is not None:
            return violation
    return None


def planned_segment_modes(item: TranslationUnitItem) -> tuple[BlockTreatment, ...]:
    """One mode per stored span. A one-span block keeps `()`.

    An empty `segment_treatments` on a multi-segment block means every span uses
    the block mode. Callers that store a result repeat that mode and do not
    collapse the tuple.
    """
    if len(item.block.segments) < 2:
        return ()
    if item.segment_treatments:
        return item.segment_treatments
    return (item.treatment,) * len(item.block.segments)


def joined_segment_translation(block: BlockResultResponse) -> str | None:
    """Join of segment translations in index order, or the whole-block translation.

    Matches `BlockProcessingResult.rebuild_translation`: non-empty pieces separated
    by one space. Empty `segments` keeps `translated_text`.
    """
    if not block.segments:
        return block.translated_text
    parts = [
        segment.translated_text
        for segment in sorted(block.segments, key=lambda item: item.index)
        if segment.translated_text and segment.translated_text.strip()
    ]
    if not parts:
        return None
    return " ".join(parts)


def _block_content_violation(
    block: BlockResultResponse, *, item: TranslationUnitItem
) -> str | None:
    """Result content of a single block relative to its planned treatment mode (PRD § 15)."""
    modes = planned_segment_modes(item)
    if not modes:
        if block.segments:
            return (
                f"для блока {block.source_block_id} без нескольких сегментов "
                "список segments недопустим"
            )
        return _legacy_content(block, planned=item.treatment)
    if block.translated_text is not None:
        return (
            f"для блока {block.source_block_id} с сегментами "
            "полный перевод блока недопустим"
        )
    spans = item.block.segment_spans
    if len(modes) != len(spans):
        return (
            f"у блока {block.source_block_id} {len(modes)} режимов сегментов "
            f"на {len(spans)} сегментов"
        )
    seen: set[int] = set()
    for segment in block.segments:
        index = segment.index
        if index in seen:
            return f"сегмент {index} блока {block.source_block_id} указан дважды"
        seen.add(index)
        if index > len(modes):
            return (
                f"сегмент {index} блока {block.source_block_id} "
                f"вне диапазона 1..{len(modes)}"
            )
        mode = modes[index - 1]
        if not mode.requires_translation:
            return (
                f"для сегмента {index} блока {block.source_block_id} "
                f"с режимом {mode} перевод недопустим"
            )
    for index, mode in enumerate(modes, start=1):
        if mode.requires_translation and index not in seen:
            return (
                f"для сегмента {index} блока {block.source_block_id} "
                "с режимом TRANSLATE нет перевода"
            )
    if any(mode is BlockTreatment.GLOSSARY_ONLY for mode in modes):
        if not block.glossary_hints:
            return f"для блока {block.source_block_id} обязательны словарные подсказки"
    elif (
        all(mode is BlockTreatment.READ_AS_IS for mode in modes)
        and block.glossary_hints
    ):
        return (
            f"для блока {block.source_block_id} с режимом {BlockTreatment.READ_AS_IS} "
            "словарные подсказки недопустимы"
        )
    return None


def _legacy_content(
    block: BlockResultResponse, *, planned: BlockTreatment
) -> str | None:
    """Whole-block content check used when the block has a single span."""
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


__all__ = [
    "content_violation",
    "joined_segment_translation",
    "planned_segment_modes",
    "treatment_violation",
]
