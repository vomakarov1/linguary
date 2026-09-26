"""Проверка состава ответа обработки: блоки группы, их порядок и чужие блоки (ALG-04).

Проверка работает над DTO ответа, а не над доменными результатами: доменный
`TranslationUnit.validate_results` те же правила знает, но применить их можно только к уже
собранным `BlockProcessingResult`, а собрать результат для неизвестного блока нечем —
режим для него взять неоткуда. Поэтому состав ответа проверяется здесь, до сборки, и здесь
же формулируется причина, которая уйдёт модели уточняющей инструкцией повтора.
"""

from __future__ import annotations

from linguary.application.llm_dto import TranslationResultResponse
from linguary.domain import TranslationUnit


def coverage_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Описание нарушения состава ответа либо `None`, если состав верен.

    Языковые заметки проверяются вместе с блоками: заметка о блоке вне группы — тот же
    неизвестный идентификатор, только в другом поле ответа (PRD § 19.3).
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
