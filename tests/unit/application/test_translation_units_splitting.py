"""Тесты защиты фрагментов расщеплённого абзаца от разрыва в translation_units.py (P2.T2)."""

from __future__ import annotations

from analysis_fakes import sequential_ids
from processing_fakes import make_plan

from linguary.application.translation_units import (
    _is_cohesive_with,
    plan_translation_units,
)
from linguary.domain import BlockType, DocumentBlock, SourceReference


def _make_block(
    order: int,
    text: str,
    *,
    split_group: str | None = None,
    split_index: str | None = None,
    split_total: str | None = None,
) -> DocumentBlock:
    attributes: dict[str, str] = {}
    if split_group:
        attributes["split_group"] = split_group
    if split_index:
        attributes["split_index"] = split_index
    if split_total:
        attributes["split_total"] = split_total

    structural_path = (
        f"{split_group}/text()[{split_index}]"
        if split_group and split_index
        else f"/body/p[{order + 1}]"
    )

    return DocumentBlock.create(
        document_id="book_1",
        section_id="sec_1",
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path="sec_1.xhtml",
            structural_path=structural_path,
        ),
        attributes=attributes,
    )


def test_is_cohesive_with_matches_split_group() -> None:
    """Блоки с одинаковым атрибутом split_group признаются связными."""
    b1 = _make_block(0, "Fragment 1", split_group="/body/p[1]", split_index="1", split_total="2")
    b2 = _make_block(1, "Fragment 2", split_group="/body/p[1]", split_index="2", split_total="2")
    b3 = _make_block(2, "Fragment 3", split_group="/body/p[2]", split_index="1", split_total="2")
    b4 = _make_block(3, "Regular paragraph")

    # Одинаковый split_group -> связные
    assert _is_cohesive_with(b1, b2) is True

    # Разные split_group -> несвязные
    assert _is_cohesive_with(b2, b3) is False

    # Один со split_group, другой без -> несвязные
    assert _is_cohesive_with(b3, b4) is False

    # Обычные абзацы Paragraph без split_group -> несвязные
    b5 = _make_block(4, "Another regular paragraph")
    assert _is_cohesive_with(b4, b5) is False


def test_plan_translation_units_keeps_split_group_together() -> None:
    """Фрагменты одного расщеплённого абзаца не разрываются между TranslationUnit."""
    # Обычный блок (400 символов) + 2 фрагмента расщеплённого абзаца по 400 символов
    b0 = _make_block(0, "A" * 400)
    b1 = _make_block(1, "B" * 400, split_group="/body/p[2]", split_index="1", split_total="2")
    b2 = _make_block(2, "C" * 400, split_group="/body/p[2]", split_index="2", split_total="2")

    blocks = (b0, b1, b2)
    plan = make_plan(blocks)

    # При лимите 1000 символов все три блока (1200) не помещаются в один запрос.
    # Фрагменты b1 и b2 связны, поэтому b0 уходит в первый юнит, а b1+b2 целиком — во второй.
    units = plan_translation_units(
        blocks,
        plan,
        id_factory=sequential_ids(),
        max_source_chars=1000,
    )

    assert len(units) == 2
    assert units[0].block_ids == (b0.id,)
    assert units[1].block_ids == (b1.id, b2.id)


def test_plan_translation_units_handles_adjacent_different_split_groups() -> None:
    """Два соседних расщеплённых абзаца образуют самостоятельные связные группы."""
    # Группа A (два фрагмента по 400 символов = 800)
    a1 = _make_block(0, "A" * 400, split_group="/body/p[1]", split_index="1", split_total="2")
    a2 = _make_block(1, "B" * 400, split_group="/body/p[1]", split_index="2", split_total="2")

    # Группа B (два фрагмента по 400 символов = 800)
    b1 = _make_block(2, "C" * 400, split_group="/body/p[2]", split_index="1", split_total="2")
    b2 = _make_block(3, "D" * 400, split_group="/body/p[2]", split_index="2", split_total="2")

    blocks = (a1, a2, b1, b2)
    plan = make_plan(blocks)

    # При лимите 1000 символов обе группы (1600) не влезут вместе,
    # но каждая группа по отдельности (800 <= 1000) помещается в свой TranslationUnit.
    units = plan_translation_units(
        blocks,
        plan,
        id_factory=sequential_ids(),
        max_source_chars=1000,
    )

    assert len(units) == 2
    assert units[0].block_ids == (a1.id, a2.id)
    assert units[1].block_ids == (b1.id, b2.id)
