"""Computational block group and validation of its results (PRD § 10, § 15)."""

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
    """A group of multiple blocks with 'TRANSLATE' treatment."""
    blocks = make_blocks(count)
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=blocks, plan=make_plan(blocks))
    return unit, tuple(block.id for block in blocks)


def translated(block_id: str) -> BlockProcessingResult:
    """A full translation result for a block."""
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
    """A non-textual block cannot be added to a group and is not silently filtered out."""
    blocks = make_blocks(1)
    with pytest.raises(NonTextualBlockError):
        TranslationUnit.of_blocks(
            id="unit-1", blocks=blocks, plan=make_plan(blocks, treatment=treatment)
        )


def test_unit_takes_treatment_from_plan() -> None:
    """Block treatments for the group are taken from the stored chapter plan."""
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
    """A block without a stored treatment cannot be added to the group."""
    blocks = make_blocks(2)
    with pytest.raises(UnknownBlockError):
        TranslationUnit.of_blocks(
            id="unit-1", blocks=blocks, plan=make_plan(blocks[:1])
        )


def test_unit_rejects_blocks_out_of_order() -> None:
    """A group consists of consecutive blocks in document order."""
    blocks = make_blocks(2)
    with pytest.raises(BlockOrderError):
        TranslationUnit.of_blocks(
            id="unit-1", blocks=tuple(reversed(blocks)), plan=make_plan(blocks)
        )


def test_unit_rejects_empty_group() -> None:
    """An empty block group is not submitted to the model."""
    with pytest.raises(TranslationUnitError):
        TranslationUnit.of_blocks(id="unit-1", blocks=(), plan=make_plan(()))


def test_unit_accepts_complete_result() -> None:
    """A complete and ordered group result passes validation."""
    unit, ids = make_unit()
    unit.validate_results([translated(block_id) for block_id in ids])


def test_unit_rejects_missing_block_result() -> None:
    """A missing result for even a single block invalidates the entire group."""
    unit, ids = make_unit()
    with pytest.raises(MissingBlockResultError):
        unit.validate_results([translated(block_id) for block_id in ids[:-1]])


def test_unit_rejects_unknown_block_result() -> None:
    """A result for an unexpected block invalidates the entire group."""
    unit, ids = make_unit()
    results = [translated(block_id) for block_id in ids]
    results.append(translated("foreign-block"))
    with pytest.raises(UnknownBlockError):
        unit.validate_results(results)


def test_unit_rejects_reordered_result() -> None:
    """The order of blocks in the response must match the original document order."""
    unit, ids = make_unit()
    results = [translated(block_id) for block_id in reversed(ids)]
    with pytest.raises(BlockOrderError):
        unit.validate_results(results)


def test_unit_rejects_treatment_that_contradicts_plan() -> None:
    """Treatment in the model response confirms the plan rather than overriding it."""
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
