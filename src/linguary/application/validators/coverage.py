"""Processing response coverage validation: unit blocks, ordering, and foreign blocks (ALG-04).

The validator operates on the response DTO rather than domain results: domain
`TranslationUnit.validate_results` knows the same rules, but can only be applied to
already assembled `BlockProcessingResult` objects, and assembling a result for an unknown
block is impossible — its treatment mode cannot be determined. Therefore response coverage
is checked here prior to assembly, and the clarification instruction sent to the model on retry
is formulated here as well.
"""

from __future__ import annotations

from linguary.application.llm_dto import TranslationResultResponse
from linguary.domain import TranslationUnit


def coverage_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Violation description of response coverage, or `None` if coverage is valid.

    Language notes are verified alongside blocks: a note referencing a block outside
    the group is the same unknown identifier, just in a different response field (PRD § 19.3).
    """
    expected = unit.block_ids
    returned = payload.block_ids
    known = set(expected)
    unknown = sorted({block_id for block_id in returned if block_id not in known})
    if unknown:
        return f"ответ содержит результат для блоков вне группы: {unknown}"
    missing = sorted(known - set(returned))
    if missing:
        return f"в ответе нет результата для блоков: {missing}"
    repeated = sorted({block_id for block_id in returned if returned.count(block_id) > 1})
    if repeated:
        return f"результат для блоков возвращён дважды: {repeated}"
    if returned != expected:
        return "порядок блоков в ответе не совпадает с порядком блоков группы"
    stray_notes = sorted(
        {
            note.source_block_id
            for note in payload.language_notes
            if note.source_block_id not in known
        }
    )
    if stray_notes:
        return f"языковые заметки привязаны к блокам вне группы: {stray_notes}"
    return None


__all__ = ["coverage_violation"]
