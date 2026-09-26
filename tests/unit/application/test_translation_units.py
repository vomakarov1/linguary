"""Формирование `TranslationUnit` из блоков главы и её плана (P3.T6, PRD § 10)."""

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
    """Группы блоков с предсказуемыми идентификаторами."""
    return plan_translation_units(
        blocks,
        plan if plan is not None else make_plan(blocks),
        id_factory=sequential_ids(),
        max_source_chars=max_source_chars,
    )


def sizes(units: tuple[TranslationUnit, ...]) -> tuple[int, ...]:
    """Объём исходного текста каждой группы в символах."""
    return tuple(sum(len(block.text) for block in unit.blocks) for unit in units)


def test_short_chapter_becomes_single_unit() -> None:
    """Глава, помещающаяся в предел размера, уходит в модель одним запросом."""
    blocks = make_blocks(5)

    units = plan_units(blocks)

    assert len(units) == 1
    assert units[0].block_ids == tuple(block.id for block in blocks)
    assert units[0].section_id == SECTION_ID


def test_unit_keeps_treatment_from_plan() -> None:
    """Режим блока в группе берётся из сохранённого плана, а не назначается заново."""
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
    """Глава длиннее предела делится на группы, ни одна из которых предел не превышает."""
    blocks = make_blocks(10, text="a" * 100)

    units = plan_units(blocks, max_source_chars=250)

    assert sizes(units) == (200, 200, 200, 200, 200)
    assert sum(len(unit.items) for unit in units) == len(blocks)


def test_units_are_evenly_sized_instead_of_leaving_a_tail() -> None:
    """Глава делится ровными долями, а не кусками предела: нарезка дала бы (400, 100)."""
    blocks = make_blocks(5, text="a" * 100)

    units = plan_units(blocks, max_source_chars=400)

    assert sizes(units) == (300, 200)


def test_unit_size_counts_characters_not_blocks() -> None:
    """Предел меряется объёмом оригинала: длинный абзац уходит один, короткие — вместе."""
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
    """Блок длиннее предела не делится: он уходит в модель отдельным запросом."""
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
    """Список не разрывается между запросами: модель увидела бы его без начала."""
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
    """Блоки одного составного целого не расходятся по разным запросам."""
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
    """Связный кусок длиннее предела разбирается на блоки, а не тащит группу за предел."""
    blocks = tuple(
        make_block(order=index, text="b" * 80, type=BlockType.POEM) for index in range(4)
    )

    units = plan_units(blocks, max_source_chars=100)

    assert len(units) == 4
    assert max(sizes(units)) == 80


def test_non_textual_blocks_do_not_reach_the_model() -> None:
    """Изображение и перенесённая как есть конструкция в группу не попадают (PRD § 10.1)."""
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
    """Блок, которому план назначил нетекстовый режим, модели не передаётся."""
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
    """Текстовый блок без режима в плане обработку не начинает: анализ главы неполон."""
    blocks = make_blocks(2)
    plan = make_plan(blocks[:1])

    with pytest.raises(UnknownBlockError, match="отсутствует в плане"):
        plan_units(blocks, plan=plan)


def test_chapter_without_textual_blocks_gives_no_units() -> None:
    """Глава из одной иллюстрации не даёт ни одного запроса к модели."""
    blocks = (make_block(order=0, text="", type=BlockType.IMAGE),)

    assert plan_units(blocks, plan=TreatmentPlan(section_id=SECTION_ID)) == ()


def test_non_positive_limit_is_rejected() -> None:
    """Предел размера группы должен быть положительным."""
    with pytest.raises(TranslationUnitError, match="положительным"):
        plan_units(make_blocks(1), max_source_chars=0)


def test_units_get_identifiers_from_factory() -> None:
    """Идентификатор группы выдаёт сценарий, а не сам модуль."""
    blocks = make_blocks(4, text="a" * 100)

    units = plan_units(blocks, max_source_chars=200)

    assert tuple(unit.id for unit in units) == ("id-1", "id-2")
