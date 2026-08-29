"""Резервный сценарий длинной главы: дробление и слияние фрагментов (P3.T4, ALG-03)."""

from __future__ import annotations

import pytest
from analysis_fakes import CONFIDENCE_THRESHOLD, make_analysis_block, sequential_ids

from linguary.application.analysis_merge import (
    AnalysisMergeError,
    FragmentAnalysis,
    merge_chapter_analysis,
    plan_fragments,
)
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
    TreatmentPlanItemResponse,
)
from linguary.domain import (
    AnalysisIncompleteError,
    BlockTreatment,
    ChapterAnalysis,
    ConceptType,
    CoverageStatus,
    DocumentBlock,
    TreatmentPlanItem,
)

SECTION_ID = "section-1"


def chapter_blocks(count: int) -> tuple[DocumentBlock, ...]:
    """Текстовые блоки главы в порядке исходного документа."""
    return tuple(
        make_analysis_block(
            section_id=SECTION_ID, order=index, text=f"Paragraph {index}."
        )
        for index in range(count)
    )


def block_ids(count: int) -> tuple[str, ...]:
    """Стабильные идентификаторы блоков главы без построения самих блоков."""
    return tuple(f"block-{index}" for index in range(1, count + 1))


def make_candidate(
    *,
    source_text: str = "Mr. Bennet",
    type: ConceptType = ConceptType.TERM,
    suggested_translation: str = "мистер Беннет",
    confidence: float = 0.95,
    model_flagged_review: bool = False,
) -> GlossaryCandidateResponse:
    """Кандидат глоссария, предложенный одним фрагментом."""
    return GlossaryCandidateResponse(
        source_text=source_text,
        type=type,
        suggested_translation=suggested_translation,
        confidence=confidence,
        model_flagged_review=model_flagged_review,
    )


def make_fragment(
    order: int,
    *,
    blocks: tuple[str, ...],
    treatment: BlockTreatment = BlockTreatment.TRANSLATE,
    chapter_summary: str = "Mr. Bennet visits the new neighbour.",
    translation_instructions: str = "Сохранять иронию рассказчика.",
    candidates: tuple[GlossaryCandidateResponse, ...] = (),
    before_you_read: tuple[BeforeYouReadEntryResponse, ...] = (),
) -> FragmentAnalysis:
    """Разобранный ответ одного фрагмента главы."""
    return FragmentAnalysis(
        order=order,
        translation_instructions=translation_instructions,
        chapter_summary=chapter_summary,
        treatments=tuple(
            TreatmentPlanItem(block_id=block_id, treatment=treatment)
            for block_id in blocks
        ),
        glossary_candidates=candidates,
        before_you_read=before_you_read,
    )


def merge(
    *fragments: FragmentAnalysis, expected: tuple[str, ...]
) -> ChapterAnalysis:
    """Слияние фрагментов главы с предсказуемыми идентификаторами кандидатов."""
    return merge_chapter_analysis(
        fragments,
        id="analysis-1",
        section_id=SECTION_ID,
        expected_block_ids=expected,
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        id_factory=sequential_ids(),
    )


def test_long_chapter_is_split_into_equal_fragments() -> None:
    """Глава сверх предела делится на равные последовательные фрагменты."""
    blocks = chapter_blocks(9)

    fragments = plan_fragments(blocks, max_blocks=4)

    assert [fragment.order for fragment in fragments] == [0, 1, 2]
    assert [len(fragment.blocks) for fragment in fragments] == [3, 3, 3]
    assert [
        block.id for fragment in fragments for block in fragment.blocks
    ] == [block.id for block in blocks]


def test_chapter_within_limit_stays_one_fragment() -> None:
    """Глава, помещающаяся в один запрос, на фрагменты не делится."""
    blocks = chapter_blocks(4)

    fragments = plan_fragments(blocks, max_blocks=4)

    assert len(fragments) == 1
    assert fragments[0].block_ids == tuple(block.id for block in blocks)


def test_chapter_without_text_has_no_fragments() -> None:
    """Главе без текстовых блоков анализировать нечего."""
    assert plan_fragments((), max_blocks=4) == ()


def test_fragment_response_is_mapped_to_chapter_block_ids() -> None:
    """План фрагмента возвращается от псевдонимов запроса к блокам главы."""
    payload = ChapterAnalysisResponse(
        translation_instructions="Сохранять иронию рассказчика.",
        chapter_summary="Mr. Bennet visits the new neighbour.",
        treatment_plan=(
            TreatmentPlanItemResponse(
                source_block_id="b2",
                treatment=BlockTreatment.READ_AS_IS,
                reason="простая фраза",
            ),
            TreatmentPlanItemResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                reason="сюжетный абзац",
            ),
        ),
    )

    fragment = FragmentAnalysis.of_response(
        payload, order=1, block_ids_by_alias={"b1": "block-3", "b2": "block-4"}
    )

    assert fragment.treatments == (
        TreatmentPlanItem(block_id="block-3", treatment=BlockTreatment.TRANSLATE),
        TreatmentPlanItem(block_id="block-4", treatment=BlockTreatment.READ_AS_IS),
    )


def test_fragment_response_with_foreign_block_is_rejected() -> None:
    """Фрагмент, назвавший режим для чужого блока, разобрать нельзя."""
    payload = ChapterAnalysisResponse(
        translation_instructions="Сохранять иронию рассказчика.",
        chapter_summary="Mr. Bennet visits the new neighbour.",
        treatment_plan=(
            TreatmentPlanItemResponse(
                source_block_id="b9",
                treatment=BlockTreatment.TRANSLATE,
                reason="сюжетный абзац",
            ),
        ),
    )

    with pytest.raises(AnalysisMergeError):
        FragmentAnalysis.of_response(
            payload, order=0, block_ids_by_alias={"b1": "block-1"}
        )


def test_complete_analysis_covers_every_block_of_the_chapter() -> None:
    """Фрагменты, покрывшие все блоки главы, дают полный анализ."""
    expected = block_ids(4)

    analysis = merge(
        make_fragment(0, blocks=expected[:2]),
        make_fragment(1, blocks=expected[2:], treatment=BlockTreatment.READ_AS_IS),
        expected=expected,
    )

    assert analysis.coverage_status is CoverageStatus.COMPLETE
    assert analysis.treatment_plan.block_ids == expected
    assert analysis.treatment_plan.treatment_for("block-4") is BlockTreatment.READ_AS_IS
    analysis.ensure_ready_for_processing()


def test_missing_fragment_leaves_analysis_incomplete() -> None:
    """Невыполненный фрагмент оставляет главу непокрытой и блокирует обработку."""
    expected = block_ids(4)

    analysis = merge(make_fragment(0, blocks=expected[:2]), expected=expected)

    assert analysis.coverage_status is CoverageStatus.INCOMPLETE
    assert analysis.treatment_plan.block_ids == expected[:2]
    with pytest.raises(AnalysisIncompleteError):
        analysis.ensure_ready_for_processing()


def test_conflicting_treatments_leave_the_block_without_plan() -> None:
    """Разные режимы одного блока решения не дают: блок выпадает из плана."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(0, blocks=expected, treatment=BlockTreatment.TRANSLATE),
        make_fragment(1, blocks=expected[1:], treatment=BlockTreatment.READ_AS_IS),
        expected=expected,
    )

    assert analysis.treatment_plan.block_ids == expected[:1]
    assert analysis.coverage_status is CoverageStatus.INCOMPLETE


def test_repeated_treatment_of_two_fragments_is_not_a_conflict() -> None:
    """Фрагменты, назвавшие один и тот же режим, друг другу не противоречат."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(0, blocks=expected),
        make_fragment(1, blocks=expected[1:]),
        expected=expected,
    )

    assert analysis.coverage_status is CoverageStatus.COMPLETE
    assert analysis.treatment_plan.block_ids == expected


def test_conflicting_translations_require_review() -> None:
    """Разные варианты перевода одного термина требуют решения пользователя (PRD § 12)."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            candidates=(make_candidate(suggested_translation="мистер Беннет"),),
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            candidates=(make_candidate(suggested_translation="господин Беннет"),),
        ),
        expected=expected,
    )

    candidates = analysis.glossary_candidates
    assert len(candidates) == 1
    assert candidates[0].source_text == "Mr. Bennet"
    assert candidates[0].requires_review is True
    assert analysis.candidates_requiring_review == candidates


def test_agreed_candidate_does_not_require_review() -> None:
    """Совпавший у фрагментов уверенный термин внимания не требует."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            candidates=(make_candidate(source_text="entail", suggested_translation="майорат"),),
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            candidates=(make_candidate(source_text="entail", suggested_translation="майорат"),),
        ),
        expected=expected,
    )

    assert analysis.glossary_candidates[0].requires_review is False


def test_review_signals_of_every_fragment_are_kept() -> None:
    """Сигналы проверки не теряются: побеждает самый уверенный вариант, но не его сигналы."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            candidates=(
                make_candidate(
                    source_text="Netherfield",
                    type=ConceptType.PLACE,
                    suggested_translation="Незерфилд",
                    confidence=0.5,
                    model_flagged_review=True,
                ),
            ),
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            candidates=(
                make_candidate(
                    source_text="Netherfield",
                    type=ConceptType.TERM,
                    suggested_translation="Незерфилд",
                    confidence=0.99,
                ),
            ),
        ),
        expected=expected,
    )

    candidate = analysis.glossary_candidates[0]
    assert candidate.suggested_translation == "Незерфилд"
    assert candidate.confidence == 0.99
    assert candidate.type is ConceptType.PLACE
    assert candidate.model_flagged_review is True
    assert candidate.requires_review is True


def test_chapter_summary_joins_fragment_summaries_in_reading_order() -> None:
    """Сводка разбитой главы собирается из сводок фрагментов в порядке чтения (ALG-03)."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(1, blocks=expected[1:], chapter_summary="Bingley takes the house."),
        make_fragment(0, blocks=expected[:1], chapter_summary="Mr. Bennet stays silent."),
        expected=expected,
    )

    assert analysis.chapter_summary == "Mr. Bennet stays silent. Bingley takes the house."


def test_merged_instructions_keep_each_directive_once() -> None:
    """Одинаковые указания фрагментов не повторяются в инструкциях главы."""
    expected = block_ids(3)

    analysis = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions="Сохранять иронию."),
        make_fragment(1, blocks=expected[1:2], translation_instructions="Сохранять иронию."),
        make_fragment(2, blocks=expected[2:], translation_instructions="Речь Коллинза — витиевата."),
        expected=expected,
    )

    assert analysis.translation_instructions == (
        "Сохранять иронию.\n\nРечь Коллинза — витиевата."
    )


def test_before_you_read_keeps_each_expression_once() -> None:
    """Выражение, встреченное несколькими фрагментами, попадает в материал один раз."""
    expected = block_ids(2)
    entry = BeforeYouReadEntryResponse(
        expression="entailed estate", explanation="имение с ограничением наследования"
    )

    analysis = merge(
        make_fragment(0, blocks=expected[:1], before_you_read=(entry,)),
        make_fragment(1, blocks=expected[1:], before_you_read=(entry,)),
        expected=expected,
    )

    assert [item.expression for item in analysis.before_you_read] == ["entailed estate"]


def test_fragments_with_repeated_order_are_rejected() -> None:
    """Два фрагмента с одним номером собрать в главу нельзя: их порядок неизвестен."""
    expected = block_ids(2)

    with pytest.raises(AnalysisMergeError):
        merge(
            make_fragment(0, blocks=expected[:1]),
            make_fragment(0, blocks=expected[1:]),
            expected=expected,
        )
