"""Per-segment treatment and translation inside one block."""

from __future__ import annotations

import dataclasses

import pytest
from domain_factories import DOCUMENT_ID, RESOURCE_PATH, SECTION_ID

from linguary.domain import (
    BlockProcessingResult,
    BlockResultError,
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    GlossaryHint,
    SourceReference,
    TranslationUnit,
    TreatmentMismatchError,
    TreatmentPlan,
    TreatmentPlanError,
    TreatmentPlanItem,
    UnknownBlockError,
)
from linguary.domain.treatment import strongest_treatment

BLOCK_ID = "block-1"
HINT = GlossaryHint(original="bright", translation="ясный")
TEXT = "Alpha. Beta."


def test_strongest_treatment_order() -> None:
    """TRANSLATE outranks GLOSSARY_ONLY, which outranks READ_AS_IS."""
    assert strongest_treatment((BlockTreatment.READ_AS_IS,)) is BlockTreatment.READ_AS_IS
    assert (
        strongest_treatment((BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS))
        is BlockTreatment.GLOSSARY_ONLY
    )
    assert (
        strongest_treatment(
            (
                BlockTreatment.READ_AS_IS,
                BlockTreatment.TRANSLATE,
                BlockTreatment.GLOSSARY_ONLY,
            )
        )
        is BlockTreatment.TRANSLATE
    )


def test_mixed_plan_item_requires_strongest_block_treatment() -> None:
    """A block keeps one mode, and that mode is the strongest segment mode."""
    with pytest.raises(TreatmentPlanError):
        TreatmentPlanItem(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            segment_treatments=(BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
            sentence_indexes=(0, 1),
        )
    with pytest.raises(TreatmentPlanError):
        TreatmentPlanItem(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
            sentence_indexes=(0,),
        )

    item = TreatmentPlanItem(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        segment_treatments=(BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
        sentence_indexes=(0, 1),
    )
    assert item.treatment is BlockTreatment.TRANSLATE
    assert item.segment_treatments == (
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
    )
    assert [field.name for field in dataclasses.fields(item)] == [
        "block_id",
        "treatment",
        "segment_treatments",
    ]

    plan = TreatmentPlan(section_id=SECTION_ID, items=(item,))
    assert plan.segment_treatments_for(BLOCK_ID, 2) == item.segment_treatments
    with pytest.raises(TreatmentPlanError):
        plan.segment_treatments_for(BLOCK_ID, 3)
    with pytest.raises(UnknownBlockError):
        plan.segment_treatments_for("missing-block", 2)


def test_uniform_segment_treatments_collapse() -> None:
    """Identical segment modes are stored only as the block mode."""
    item = TreatmentPlanItem(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        segment_treatments=(BlockTreatment.GLOSSARY_ONLY, BlockTreatment.GLOSSARY_ONLY),
        sentence_indexes=(0, 0),
    )
    assert item.segment_treatments == ()
    assert item.treatment is BlockTreatment.GLOSSARY_ONLY
    across_sentences = TreatmentPlanItem(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.READ_AS_IS,
        segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        sentence_indexes=(0, 1),
    )
    assert across_sentences.segment_treatments == ()
    with pytest.raises(TreatmentPlanError):
        TreatmentPlanItem(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
            sentence_indexes=(0, 1),
        )

    plan = TreatmentPlan(section_id=SECTION_ID, items=(item,))
    assert plan.segment_treatments_for(BLOCK_ID, 2) == (
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
    )

    with pytest.raises(TreatmentPlanError):
        TreatmentPlanItem(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.READ_AS_IS,
            segment_treatments=(BlockTreatment.READ_AS_IS,),
            sentence_indexes=(0,),
        )


def test_preserve_only_not_allowed_in_segments() -> None:
    """Non-textual modes are not a treatment of a segment inside a block."""
    for treatment in (BlockTreatment.PRESERVE_ONLY, BlockTreatment.DROP_WITH_WARNING):
        with pytest.raises(TreatmentPlanError):
            TreatmentPlanItem(
                block_id=BLOCK_ID,
                treatment=BlockTreatment.TRANSLATE,
                segment_treatments=(BlockTreatment.TRANSLATE, treatment),
                sentence_indexes=(0, 1),
            )


def test_different_treatments_in_same_sentence_rejected() -> None:
    """Spans of one sentence share a mode; another sentence may differ."""
    with pytest.raises(TreatmentPlanError):
        TreatmentPlanItem(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            segment_treatments=(
                BlockTreatment.READ_AS_IS,
                BlockTreatment.TRANSLATE,
                BlockTreatment.TRANSLATE,
            ),
            sentence_indexes=(0, 0, 1),
        )

    item = TreatmentPlanItem(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        segment_treatments=(
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
        sentence_indexes=(0, 0, 1),
    )
    assert item.segment_treatments == (
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
    )


def test_result_requires_translation_only_for_translate_segments() -> None:
    """A segment translation exists exactly where the segment mode is TRANSLATE."""
    segments = (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE)
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Ясный день.",
            glossary_hints=(HINT,),
            segment_treatments=segments,
            segment_translations=(None, None),
        )
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Ясный день.",
            glossary_hints=(HINT,),
            segment_treatments=segments,
            segment_translations=("лишнее", "Ясный день."),
        )
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Ясный день.",
            segment_treatments=segments,
            segment_translations=(None, "Ясный день."),
        )
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Ясный день.",
            glossary_hints=(HINT,),
            segment_treatments=segments,
            segment_translations=(None,),
        )
    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.READ_AS_IS,
            glossary_hints=(HINT,),
            segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
            segment_translations=(None, None),
        )

    result = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Ясный день.",
        glossary_hints=(HINT,),
        segment_treatments=segments,
        segment_translations=(None, "Ясный день."),
    )
    assert result.segment_translations == (None, "Ясный день.")


def test_result_translated_text_is_join_of_segments() -> None:
    """The block translation joins non-empty segment translations with a space."""
    result = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Альфа. Гамма.",
        segment_treatments=(
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
        segment_translations=("Альфа.", None, "Гамма."),
    )
    assert result.rebuild_translation() == "Альфа. Гамма."
    assert result.translated_text == result.rebuild_translation()

    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="не та склейка",
            segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
            segment_translations=("Альфа.", None),
        )

    glossary = BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(HINT,),
        segment_treatments=(BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS),
        segment_translations=(None, None),
    )
    assert glossary.translated_text is None
    assert glossary.rebuild_translation() is None

    with pytest.raises(BlockResultError):
        BlockProcessingResult(
            block_id=BLOCK_ID,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            translated_text="нельзя",
            glossary_hints=(HINT,),
            segment_treatments=(BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS),
            segment_translations=(None, None),
        )


def test_unit_rejects_segment_treatment_mismatch() -> None:
    """A unit result must repeat the segment modes taken from the plan."""
    block = DocumentBlock.create(
        document_id=DOCUMENT_ID,
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=0,
        text=TEXT,
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH,
            structural_path="/body/section[1]/p[1]",
        ),
        segments=(
            BlockSegment(0, 7, 0),
            BlockSegment(7, len(TEXT), 1),
        ),
    )
    planned = (BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE)
    plan = TreatmentPlan(
        section_id=SECTION_ID,
        items=(
            TreatmentPlanItem(
                block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                segment_treatments=planned,
                sentence_indexes=(0, 1),
            ),
        ),
    )
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=(block,), plan=plan)
    assert unit.items[0].segment_treatments == planned

    unit.validate_results(
        [
            BlockProcessingResult(
                block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Бета.",
                segment_treatments=planned,
                segment_translations=(None, "Бета."),
            )
        ]
    )
    with pytest.raises(TreatmentMismatchError):
        unit.validate_results(
            [
                BlockProcessingResult(
                    block_id=block.id,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text="Альфа. Бета.",
                    segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
                    segment_translations=("Альфа.", "Бета."),
                )
            ]
        )
    with pytest.raises(TreatmentMismatchError):
        unit.validate_results(
            [
                BlockProcessingResult(
                    block_id=block.id,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text="Бета.",
                )
            ]
        )
