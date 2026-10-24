"""Chapter treatment plan and block treatments (PRD § 10.1, § 14)."""

from __future__ import annotations

import pytest
from domain_factories import SECTION_ID, make_blocks, make_plan

from linguary.domain import (
    AnalysisIncompleteError,
    BlockTreatment,
    ChapterAnalysis,
    CoverageStatus,
    TreatmentPlan,
    TreatmentPlanError,
    UnknownBlockError,
)


@pytest.mark.parametrize(
    ("treatment", "is_textual"),
    [
        pytest.param(BlockTreatment.TRANSLATE, True, id="translate"),
        pytest.param(BlockTreatment.GLOSSARY_ONLY, True, id="glossary-only"),
        pytest.param(BlockTreatment.READ_AS_IS, True, id="read-as-is"),
        pytest.param(BlockTreatment.PRESERVE_ONLY, False, id="preserve-only"),
        pytest.param(BlockTreatment.DROP_WITH_WARNING, False, id="drop-with-warning"),
    ],
)
def test_only_textual_treatments_reach_the_model(
    treatment: BlockTreatment, is_textual: bool
) -> None:
    """'PRESERVE_ONLY' and 'DROP_WITH_WARNING' are not passed to the LLM."""
    assert treatment.is_textual is is_textual


@pytest.mark.parametrize(
    ("treatment", "requires_translation", "requires_hints", "allows_hints"),
    [
        pytest.param(BlockTreatment.TRANSLATE, True, False, True, id="translate"),
        pytest.param(BlockTreatment.GLOSSARY_ONLY, False, True, True, id="glossary-only"),
        pytest.param(BlockTreatment.READ_AS_IS, False, False, False, id="read-as-is"),
    ],
)
def test_treatment_defines_expected_result(
    treatment: BlockTreatment,
    requires_translation: bool,
    requires_hints: bool,
    allows_hints: bool,
) -> None:
    """Treatment specifies result structure: translation and glossary hints."""
    assert treatment.requires_translation is requires_translation
    assert treatment.requires_glossary_hints is requires_hints
    assert treatment.allows_glossary_hints is allows_hints


def test_plan_returns_treatment_of_known_block() -> None:
    """The plan is the source of truth for block treatment."""
    blocks = make_blocks(2)
    plan = make_plan(blocks, treatment=BlockTreatment.GLOSSARY_ONLY)
    assert plan.treatment_for(blocks[0].id) is BlockTreatment.GLOSSARY_ONLY


def test_plan_rejects_unknown_block() -> None:
    """Treatment for a block missing from the plan is not inferred."""
    plan = make_plan(make_blocks(1))
    with pytest.raises(UnknownBlockError):
        plan.treatment_for("missing-block")


def test_plan_rejects_duplicate_block() -> None:
    """A block appears in the plan exactly once."""
    blocks = make_blocks(1)
    with pytest.raises(TreatmentPlanError):
        TreatmentPlan.of_treatments(
            SECTION_ID,
            (
                (blocks[0].id, BlockTreatment.TRANSLATE),
                (blocks[0].id, BlockTreatment.READ_AS_IS),
            ),
        )


def test_plan_selects_blocks_by_treatment() -> None:
    """The plan indicates which blocks do not go to the LLM."""
    blocks = make_blocks(3)
    plan = TreatmentPlan.of_treatments(
        SECTION_ID,
        (
            (blocks[0].id, BlockTreatment.TRANSLATE),
            (blocks[1].id, BlockTreatment.PRESERVE_ONLY),
            (blocks[2].id, BlockTreatment.DROP_WITH_WARNING),
        ),
    )
    assert plan.textual_block_ids == (blocks[0].id,)
    assert plan.block_ids_with(
        BlockTreatment.PRESERVE_ONLY, BlockTreatment.DROP_WITH_WARNING
    ) == (blocks[1].id, blocks[2].id)


@pytest.mark.parametrize(
    ("covered", "expected"),
    [
        pytest.param(3, CoverageStatus.COMPLETE, id="all-blocks-covered"),
        pytest.param(2, CoverageStatus.INCOMPLETE, id="block-missing"),
    ],
)
def test_plan_coverage(covered: int, expected: CoverageStatus) -> None:
    """Incomplete block coverage marks chapter analysis as incomplete."""
    blocks = make_blocks(3)
    plan = make_plan(blocks[:covered])
    assert plan.coverage_for(block.id for block in blocks) is expected


def test_incomplete_analysis_blocks_processing() -> None:
    """A chapter with incomplete analysis cannot be processed."""
    blocks = make_blocks(3)
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=SECTION_ID,
        treatment_plan=make_plan(blocks[:1]),
        translation_instructions="Сохранять сухой канцелярский тон Министерства правды.",
        chapter_summary="Winston starts a diary.",
        expected_block_ids=[block.id for block in blocks],
    )
    assert analysis.coverage_status is CoverageStatus.INCOMPLETE
    with pytest.raises(AnalysisIncompleteError):
        analysis.ensure_ready_for_processing()


def test_complete_analysis_allows_processing() -> None:
    """Full block coverage enables chapter processing."""
    blocks = make_blocks(2)
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=SECTION_ID,
        treatment_plan=make_plan(blocks),
        translation_instructions="Сохранять сухой канцелярский тон Министерства правды.",
        chapter_summary="Winston starts a diary.",
        expected_block_ids=[block.id for block in blocks],
    )
    assert analysis.is_complete
    analysis.ensure_ready_for_processing()
