"""Состав контекста анализа главы (P3.T3, PRD § 13)."""

from __future__ import annotations

import pytest
from analysis_fakes import (
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    make_analysis_block,
    make_analysis_candidate,
    make_analysis_document,
    make_analysis_section,
)

from linguary.application.context_builder import (
    ContextBuildError,
    ContextBuilder,
    textual_blocks,
)
from linguary.domain import BlockTreatment, BlockType, ConceptType, SupportStatus

NEIGHBOUR_TEXT = "The neighbouring chapter must never reach the model."


def build_context(
    *,
    texts: tuple[str, ...] = ("It is a truth universally acknowledged.",),
    candidates: tuple[object, ...] = (),
    max_accumulated_candidates: int = 200,
):
    """Контекст анализа первой главы книги из двух глав."""
    section = make_analysis_section(
        section_id="section-1", title="Chapter 1", order=0, texts=texts
    )
    neighbour = make_analysis_section(
        section_id="section-2", title="Chapter 2", order=1, texts=(NEIGHBOUR_TEXT,)
    )
    document = make_analysis_document((section, neighbour))
    builder = ContextBuilder(max_accumulated_candidates=max_accumulated_candidates)
    return builder.for_chapter_analysis(
        document=document,
        section=section,
        accumulated_candidates=candidates,  # type: ignore[arg-type]  # фабрика двойников
    )


def test_analysis_context_contains_chapter_blocks() -> None:
    """Текст каждого блока главы попадает в запрос под своим псевдонимом."""
    context = build_context(texts=("First paragraph.", "Second paragraph."))

    input_text = context.request.input_text

    assert "[b1]\nFirst paragraph." in input_text
    assert "[b2]\nSecond paragraph." in input_text


def test_analysis_context_excludes_neighbour_chapters() -> None:
    """Текст соседних глав в контекст анализа не передаётся (PRD § 13)."""
    context = build_context()

    assert NEIGHBOUR_TEXT not in context.request.input_text
    assert NEIGHBOUR_TEXT not in context.request.instructions


def test_analysis_context_names_book_chapter_and_languages() -> None:
    """Запрос называет книгу, главу и языки издания."""
    input_text = build_context().request.input_text

    assert "Pride and Prejudice" in input_text
    assert "Jane Austen" in input_text
    assert "Chapter 1" in input_text
    assert f"Язык оригинала: {SOURCE_LANGUAGE}" in input_text
    assert f"Язык перевода: {TARGET_LANGUAGE}" in input_text


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(
            make_analysis_block(
                section_id="section-1",
                order=1,
                text="",
                type=BlockType.IMAGE,
            ),
            id="image",
        ),
        pytest.param(
            make_analysis_block(
                section_id="section-1",
                order=1,
                text="| a | b |",
                type=BlockType.TABLE,
                support_status=SupportStatus.PRESERVE_OPAQUE,
            ),
            id="preserve-opaque-table",
        ),
        pytest.param(
            make_analysis_block(section_id="section-1", order=1, text="   "),
            id="blank-text",
        ),
    ],
)
def test_non_textual_blocks_stay_out_of_context(block) -> None:
    """`PRESERVE_ONLY` и нетекстовые блоки модели не передаются (PRD § 10.1)."""
    textual = make_analysis_block(
        section_id="section-1", order=0, text="A textual paragraph."
    )
    section = make_analysis_section(section_id="section-1", blocks=(textual, block))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert textual_blocks(section) == (textual,)
    assert context.aliases == ("b1",)
    assert context.block_id_for("b1") == textual.id
    assert block.id not in context.block_ids_by_alias.values()


def test_preserved_block_text_stays_out_of_the_request() -> None:
    """Текст конструкции, перенесённой как есть, модель не видит (ADR § 3)."""
    textual = make_analysis_block(
        section_id="section-1", order=0, text="A textual paragraph."
    )
    preserved = make_analysis_block(
        section_id="section-1",
        order=1,
        text="| column | column |",
        type=BlockType.TABLE,
        support_status=SupportStatus.PRESERVE_OPAQUE,
    )
    section = make_analysis_section(section_id="section-1", blocks=(textual, preserved))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert preserved.text not in context.request.input_text


def test_accumulated_candidates_reach_the_request() -> None:
    """Кандидаты предыдущих глав передаются как принятые решения книги."""
    context = build_context(
        candidates=(
            make_analysis_candidate(
                source_text="Longbourn", suggested_translation="Лонгборн"
            ),
        )
    )

    assert "Longbourn" in context.request.input_text
    assert "Лонгборн" in context.request.input_text


def test_empty_accumulated_candidates_are_stated_explicitly() -> None:
    """Отсутствие принятых решений описано словами, а не пустотой в запросе."""
    assert "Предыдущие главы" in build_context().request.input_text


def test_accumulated_candidates_are_deduplicated_and_limited() -> None:
    """Повторы схлопываются, а список обрезается настроенным пределом."""
    context = build_context(
        candidates=(
            make_analysis_candidate(source_text="Longbourn", analysis_id="analysis-1"),
            make_analysis_candidate(source_text="Longbourn", analysis_id="analysis-2"),
            make_analysis_candidate(source_text="Netherfield"),
        ),
        max_accumulated_candidates=1,
    )

    decisions = context.request.input_text
    assert decisions.count("Longbourn") == 1
    assert "Netherfield" not in decisions


def test_aliases_map_back_to_stable_block_ids() -> None:
    """Псевдоним запроса возвращается к стабильному идентификатору блока."""
    section = make_analysis_section(texts=("First.", "Second."))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert context.aliases == ("b1", "b2")
    assert [context.block_id_for(alias) for alias in context.aliases] == [
        block.id for block in section.ordered_blocks
    ]


def test_expectation_describes_aliases_and_target_language() -> None:
    """Ожидаемый результат перечисляет то, что модель видела: псевдонимы блоков."""
    context = build_context(texts=("First.", "Second."))

    expectation = context.expectation

    assert expectation.block_ids == ("b1", "b2")
    assert expectation.target_language == TARGET_LANGUAGE
    assert all(block.treatment is None for block in expectation.blocks)


def test_unknown_alias_is_rejected() -> None:
    """Псевдоним, которого в запросе не было, к блоку не приводится."""
    context = build_context()

    with pytest.raises(ContextBuildError):
        context.block_id_for("b99")


def test_instructions_describe_every_textual_treatment() -> None:
    """Инструкция называет все режимы обработки и оба языка издания (PRD § 10.1)."""
    instructions = build_context().request.instructions

    for treatment in (
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
    ):
        assert treatment.value in instructions
    assert BlockTreatment.PRESERVE_ONLY.value not in instructions
    assert f"учит язык {SOURCE_LANGUAGE}" in instructions
    assert f"владеет языком {TARGET_LANGUAGE}" in instructions


def test_instructions_name_every_glossary_concept_type() -> None:
    """Инструкция называет каждый тип понятия глоссария (PRD § 12)."""
    instructions = build_context().request.instructions

    for concept_type in ConceptType:
        assert concept_type.value in instructions


def test_instructions_require_chapter_summary_in_source_language() -> None:
    """Инструкция требует сводку главы и материал `Before You Read` (PRD § 19)."""
    instructions = build_context().request.instructions

    assert "chapter_summary" in instructions
    assert "before_you_read" in instructions
    assert "translation_instructions" in instructions
    assert "glossary_candidates" in instructions


def test_chapter_without_textual_blocks_has_no_context() -> None:
    """Главу без текста анализировать нечем: контекст не собирается."""
    section = make_analysis_section(
        blocks=(
            make_analysis_block(
                section_id="section-1", order=0, text="", type=BlockType.IMAGE
            ),
        )
    )
    document = make_analysis_document((section,))

    with pytest.raises(ContextBuildError):
        ContextBuilder().for_chapter_analysis(document=document, section=section)


def test_negative_candidate_limit_is_rejected() -> None:
    """Отрицательный предел числа накопленных решений — ошибка настройки."""
    with pytest.raises(ContextBuildError):
        ContextBuilder(max_accumulated_candidates=-1)
