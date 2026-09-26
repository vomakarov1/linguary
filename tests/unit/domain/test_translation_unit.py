"""Вычислительная группа блоков и проверка её результата (PRD § 10, § 15)."""

from __future__ import annotations

import pytest
from domain_factories import SECTION_ID, make_blocks, make_plan

from linguary.domain import (
    BlockOrderError,
    BlockProcessingResult,
    BlockTreatment,
    GlossaryHint,
    MissingBlockResultError,
    NonTextualBlockError,
    TranslationUnit,
    TranslationUnitError,
    TreatmentMismatchError,
    TreatmentPlan,
    UnknownBlockError,
)


def make_unit(count: int = 3) -> tuple[TranslationUnit, tuple[str, ...]]:
    """Группа из нескольких блоков режима `TRANSLATE`."""
    blocks = make_blocks(count)
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=blocks, plan=make_plan(blocks))
    return unit, tuple(block.id for block in blocks)


def translated(block_id: str) -> BlockProcessingResult:
    """Результат полного перевода блока."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Был холодный ясный апрельский день.",
    )


@pytest.mark.parametrize(
    "treatment",
    [
        pytest.param(BlockTreatment.PRESERVE_ONLY, id="preserve-only"),
        pytest.param(BlockTreatment.DROP_WITH_WARNING, id="drop-with-warning"),
    ],
)
def test_unit_rejects_non_textual_block(treatment: BlockTreatment) -> None:
    """Нетекстовый блок не попадает в группу, а не отфильтровывается молча."""
    blocks = make_blocks(1)
    with pytest.raises(NonTextualBlockError):
        TranslationUnit.of_blocks(
            id="unit-1", blocks=blocks, plan=make_plan(blocks, treatment=treatment)
        )


def test_unit_takes_treatment_from_plan() -> None:
    """Режимы блоков группы берутся из сохранённого плана главы."""
    blocks = make_blocks(2)
    plan = TreatmentPlan.of_treatments(
        SECTION_ID,
        (
            (blocks[0].id, BlockTreatment.TRANSLATE),
            (blocks[1].id, BlockTreatment.READ_AS_IS),
        ),
    )
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=blocks, plan=plan)
    assert unit.treatment_for(blocks[1].id) is BlockTreatment.READ_AS_IS


def test_unit_rejects_block_missing_from_plan() -> None:
    """Блок без сохранённого режима в группу не попадает."""
    blocks = make_blocks(2)
    with pytest.raises(UnknownBlockError):
        TranslationUnit.of_blocks(
            id="unit-1", blocks=blocks, plan=make_plan(blocks[:1])
        )


def test_unit_rejects_blocks_out_of_order() -> None:
    """Группа состоит из последовательных блоков в порядке документа."""
    blocks = make_blocks(2)
    with pytest.raises(BlockOrderError):
        TranslationUnit.of_blocks(
            id="unit-1", blocks=tuple(reversed(blocks)), plan=make_plan(blocks)
        )


def test_unit_rejects_empty_group() -> None:
    """Пустая группа блоков модели не отправляется."""
    with pytest.raises(TranslationUnitError):
        TranslationUnit.of_blocks(id="unit-1", blocks=(), plan=make_plan(()))


def test_unit_accepts_complete_result() -> None:
    """Полный и упорядоченный результат группы проходит проверку."""
    unit, ids = make_unit()
    unit.validate_results([translated(block_id) for block_id in ids])


def test_unit_rejects_missing_block_result() -> None:
    """Отсутствие результата хотя бы для одного блока отменяет всю группу."""
    unit, ids = make_unit()
    with pytest.raises(MissingBlockResultError):
        unit.validate_results([translated(block_id) for block_id in ids[:-1]])


def test_unit_rejects_unknown_block_result() -> None:
    """Результат для чужого блока отменяет всю группу."""
    unit, ids = make_unit()
    results = [translated(block_id) for block_id in ids]
    results.append(translated("foreign-block"))
    with pytest.raises(UnknownBlockError):
        unit.validate_results(results)


def test_unit_rejects_reordered_result() -> None:
    """Порядок блоков в ответе совпадает с порядком исходного документа."""
    unit, ids = make_unit()
    results = [translated(block_id) for block_id in reversed(ids)]
    with pytest.raises(BlockOrderError):
        unit.validate_results(results)


def test_unit_rejects_treatment_that_contradicts_plan() -> None:
    """Режим в ответе модели подтверждает план, а не переопределяет его."""
    unit, ids = make_unit(2)
    results = [
        translated(ids[0]),
        BlockProcessingResult(
            block_id=ids[1],
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint(original="chin", translation="подбородок"),),
        ),
    ]
    with pytest.raises(TreatmentMismatchError):
        unit.validate_results(results)
