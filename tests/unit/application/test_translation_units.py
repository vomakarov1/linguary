"""Building `TranslationUnit`s from chapter blocks and treatment plan (P3.T6, PRD § 10)."""

from __future__ import annotations

import pytest
from analysis_fakes import sequential_ids
from processing_fakes import SECTION_ID, make_block, make_blocks, make_plan

from linguary.application.translation_units import (
    DEFAULT_MAX_UNIT_SOURCE_CHARS,
    plan_translation_units,
)
from linguary.domain import (
    BlockTreatment,
    BlockType,
    DocumentBlock,
    SupportStatus,
    TranslationUnit,
    TranslationUnitError,
    TreatmentPlan,
    UnknownBlockError,
)


def plan_units(
    blocks: tuple[DocumentBlock, ...],
    *,
    plan: TreatmentPlan | None = None,
    max_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
) -> tuple[TranslationUnit, ...]:
    """Block groups with predictable identifiers."""
    return plan_translation_units(
        blocks,
        plan if plan is not None else make_plan(blocks),
        id_factory=sequential_ids(),
        max_source_chars=max_source_chars,
    )


def sizes(units: tuple[TranslationUnit, ...]) -> tuple[int, ...]:
    """Source text character count for each group."""
    return tuple(sum(len(block.text) for block in unit.blocks) for unit in units)


def test_short_chapter_becomes_single_unit() -> None:
    """A chapter that fits within the size limit is sent to the model in a single request."""
    blocks = make_blocks(5)

    units = plan_units(blocks)

    assert len(units) == 1
    assert units[0].block_ids == tuple(block.id for block in blocks)
    assert units[0].section_id == SECTION_ID


def test_unit_keeps_treatment_from_plan() -> None:
    """Block treatment in the unit is taken from the stored plan rather than reassigned."""
    blocks = make_blocks(3)
    plan = make_plan(
        blocks,
        treatments=(
            BlockTreatment.TRANSLATE,
            BlockTreatment.GLOSSARY_ONLY,
            BlockTreatment.READ_AS_IS,
        ),
    )

    units = plan_units(blocks, plan=plan)

    assert tuple(item.treatment for item in units[0].items) == (
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
    )


def test_long_chapter_splits_by_source_size() -> None:
    """A chapter longer than the limit is split into units, none of which exceed the limit."""
    blocks = make_blocks(10, text="a" * 100)

    units = plan_units(blocks, max_source_chars=250)

    assert sizes(units) == (200, 200, 200, 200, 200)
    assert sum(len(unit.items) for unit in units) == len(blocks)


def test_units_are_evenly_sized_instead_of_leaving_a_tail() -> None:
    """The chapter is split evenly rather than by max chunks: chunking would yield (400, 100)."""
    blocks = make_blocks(5, text="a" * 100)

    units = plan_units(blocks, max_source_chars=400)

    assert sizes(units) == (300, 200)


def test_unit_size_counts_characters_not_blocks() -> None:
    """The limit is measured by source text volume: a long paragraph goes alone, short ones together."""
    blocks = (
        make_block(order=0, text="a" * 200),
        make_block(order=1, text="b" * 20),
        make_block(order=2, text="c" * 20),
    )

    units = plan_units(blocks, max_source_chars=200)

    assert tuple(unit.block_ids for unit in units) == (
        (blocks[0].id,),
        (blocks[1].id, blocks[2].id),
    )


def test_block_longer_than_limit_becomes_its_own_unit() -> None:
    """A block longer than the limit is not split: it is sent to the model as its own unit."""
    blocks = (
        make_block(order=0, text="a" * 50),
        make_block(order=1, text="b" * 500),
        make_block(order=2, text="c" * 50),
    )

    units = plan_units(blocks, max_source_chars=100)

    assert tuple(unit.block_ids for unit in units) == (
        (blocks[0].id,),
        (blocks[1].id,),
        (blocks[2].id,),
    )


def test_list_items_stay_in_one_unit() -> None:
    """A list is not split across requests: otherwise the model would see it without context."""
    blocks = (
        make_block(order=0, text="a" * 50),
        *(
            make_block(order=index, text="b" * 30, type=BlockType.LIST_ITEM)
            for index in range(1, 4)
        ),
    )

    units = plan_units(blocks, max_source_chars=100)

    assert tuple(unit.block_ids for unit in units) == (
        (blocks[0].id,),
        tuple(block.id for block in blocks[1:]),
    )


def test_blocks_of_one_parent_stay_in_one_unit() -> None:
    """Blocks belonging to the same compound parent are not split across different requests."""
    parent = make_block(order=0, text="a" * 50)
    blocks = (
        parent,
        *(
            make_block(order=index, text="b" * 30, parent_id=parent.id)
            for index in range(1, 4)
        ),
    )

    units = plan_units(blocks, max_source_chars=100)

    assert tuple(unit.block_ids for unit in units) == (
        (parent.id,),
        tuple(block.id for block in blocks[1:]),
    )


def test_cohesive_run_longer_than_limit_splits_into_blocks() -> None:
    """A cohesive run longer than the limit is split into blocks rather than exceeding the limit."""
    blocks = tuple(
        make_block(order=index, text="b" * 80, type=BlockType.POEM) for index in range(4)
    )

    units = plan_units(blocks, max_source_chars=100)

    assert len(units) == 4
    assert max(sizes(units)) == 80


def test_non_textual_blocks_do_not_reach_the_model() -> None:
    """Images and preserved opaque constructs do not enter the unit (PRD § 10.1)."""
    blocks = (
        make_block(order=0),
        make_block(order=1, text="", type=BlockType.IMAGE),
        make_block(order=2, support_status=SupportStatus.PRESERVE_OPAQUE),
        make_block(order=3),
    )
    plan = make_plan((blocks[0], blocks[3]))

    units = plan_units(blocks, plan=plan)

    assert units[0].block_ids == (blocks[0].id, blocks[3].id)


def test_non_textual_treatment_is_skipped() -> None:
    """A block assigned a non-textual treatment by the plan is not sent to the model."""
    blocks = make_blocks(2)
    plan = TreatmentPlan.of_treatments(
        SECTION_ID,
        (
            (blocks[0].id, BlockTreatment.TRANSLATE),
            (blocks[1].id, BlockTreatment.PRESERVE_ONLY),
        ),
    )

    units = plan_units(blocks, plan=plan)

    assert units[0].block_ids == (blocks[0].id,)


def test_block_missing_from_plan_is_rejected() -> None:
    """A text block without a treatment in the plan is rejected: chapter analysis is incomplete."""
    blocks = make_blocks(2)
    plan = make_plan(blocks[:1])

    with pytest.raises(UnknownBlockError, match="отсутствует в плане"):
        plan_units(blocks, plan=plan)


def test_chapter_without_textual_blocks_gives_no_units() -> None:
    """A chapter with only an illustration produces no units for the model."""
    blocks = (make_block(order=0, text="", type=BlockType.IMAGE),)

    assert plan_units(blocks, plan=TreatmentPlan(section_id=SECTION_ID)) == ()


def test_non_positive_limit_is_rejected() -> None:
    """The unit size limit must be positive."""
    with pytest.raises(TranslationUnitError, match="положительным"):
        plan_units(make_blocks(1), max_source_chars=0)


def test_units_get_identifiers_from_factory() -> None:
    """Unit identifiers are issued by the caller factory, not the module itself."""
    blocks = make_blocks(4, text="a" * 100)

    units = plan_units(blocks, max_source_chars=200)

    assert tuple(unit.id for unit in units) == ("id-1", "id-2")
