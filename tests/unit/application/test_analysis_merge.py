"""Fallback scenario for a long chapter: chunking and merging fragments (P3.T4, ALG-03)."""

from __future__ import annotations

import pytest
from analysis_fakes import CONFIDENCE_THRESHOLD, make_analysis_block, sequential_ids

from linguary.application.analysis_merge import (
    AnalysisMergeError,
    FragmentAnalysis,
    merge_chapter_analysis,
    plan_fragments,
    fragment_blocks_for,
    before_you_read_entries,
    sanitize_before_you_read_explanation,
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
    """Chapter text blocks in original document order."""
    return tuple(
        make_analysis_block(
            section_id=SECTION_ID, order=index, text=f"Paragraph {index}."
        )
        for index in range(count)
    )


def block_ids(count: int) -> tuple[str, ...]:
    """Stable chapter block identifiers without constructing the blocks themselves."""
    return tuple(f"block-{index}" for index in range(1, count + 1))


def make_candidate(
    *,
    source_text: str = "Mr. Bennet",
    type: ConceptType = ConceptType.TERM,
    suggested_translation: str = "мистер Беннет",
    confidence: float = 0.95,
    model_flagged_review: bool = False,
) -> GlossaryCandidateResponse:
    """Glossary candidate proposed by a single fragment."""
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
    """Parsed response of a single chapter fragment."""
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
    """Merge chapter fragments with predictable candidate identifiers."""
    return merge_chapter_analysis(
        fragments,
        id="analysis-1",
        section_id=SECTION_ID,
        expected_block_ids=expected,
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        id_factory=sequential_ids(),
    )


def test_long_chapter_is_split_into_equal_fragments() -> None:
    """A chapter exceeding the limit is split into equal consecutive fragments."""
    blocks = chapter_blocks(9)

    fragments = plan_fragments(blocks, max_blocks=4)

    assert [fragment.order for fragment in fragments] == [0, 1, 2]
    assert [len(fragment.blocks) for fragment in fragments] == [3, 3, 3]
    assert [
        block.id for fragment in fragments for block in fragment.blocks
    ] == [block.id for block in blocks]


def test_chapter_within_limit_stays_one_fragment() -> None:
    """A chapter fitting into a single request is not split into fragments."""
    blocks = chapter_blocks(4)

    fragments = plan_fragments(blocks, max_blocks=4)

    assert len(fragments) == 1
    assert fragments[0].block_ids == tuple(block.id for block in blocks)


def test_chapter_without_text_has_no_fragments() -> None:
    """A chapter without text blocks has nothing to analyze."""
    assert plan_fragments((), max_blocks=4) == ()


def test_fragment_response_is_mapped_to_chapter_block_ids() -> None:
    """Fragment plan maps request aliases back to chapter blocks."""
    payload = ChapterAnalysisResponse(
        translation_instructions="Сохранять иронию рассказчика.",
        chapter_summary="Mr. Bennet visits the new neighbour.",
        treatment_plan=(
            TreatmentPlanItemResponse(
                source_block_id="b2",
                treatment=BlockTreatment.READ_AS_IS,
            ),
            TreatmentPlanItemResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
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
    """A fragment specifying treatment for a foreign block cannot be parsed."""
    payload = ChapterAnalysisResponse(
        translation_instructions="Сохранять иронию рассказчика.",
        chapter_summary="Mr. Bennet visits the new neighbour.",
        treatment_plan=(
            TreatmentPlanItemResponse(
                source_block_id="b9",
                treatment=BlockTreatment.TRANSLATE,
            ),
        ),
    )

    with pytest.raises(AnalysisMergeError):
        FragmentAnalysis.of_response(
            payload, order=0, block_ids_by_alias={"b1": "block-1"}
        )


def test_complete_analysis_covers_every_block_of_the_chapter() -> None:
    """Fragments covering all chapter blocks produce a complete analysis."""
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
    """A missing fragment leaves the chapter incomplete and blocks processing."""
    expected = block_ids(4)

    analysis = merge(make_fragment(0, blocks=expected[:2]), expected=expected)

    assert analysis.coverage_status is CoverageStatus.INCOMPLETE
    assert analysis.treatment_plan.block_ids == expected[:2]
    with pytest.raises(AnalysisIncompleteError):
        analysis.ensure_ready_for_processing()


def test_conflicting_treatments_leave_the_block_without_plan() -> None:
    """Conflicting treatments yield no decision: the block is omitted from the plan."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(0, blocks=expected, treatment=BlockTreatment.TRANSLATE),
        make_fragment(1, blocks=expected[1:], treatment=BlockTreatment.READ_AS_IS),
        expected=expected,
    )

    assert analysis.treatment_plan.block_ids == expected[:1]
    assert analysis.coverage_status is CoverageStatus.INCOMPLETE


def test_repeated_treatment_of_two_fragments_is_not_a_conflict() -> None:
    """Fragments proposing the same treatment do not conflict with each other."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(0, blocks=expected),
        make_fragment(1, blocks=expected[1:]),
        expected=expected,
    )

    assert analysis.coverage_status is CoverageStatus.COMPLETE
    assert analysis.treatment_plan.block_ids == expected


def test_conflicting_translations_require_review() -> None:
    """Different translation variants for a term require user review (PRD § 12)."""
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
    """A confident term agreed upon across fragments requires no review."""
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
    """Review signals from every fragment are preserved: most confident wins, keeping all signals."""
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


def test_obvious_spellings_of_one_name_become_one_candidate() -> None:
    """Casing and a trailing dot in `Mr.` do not turn one name into different candidates (QA-11).

    The candidate is displayed with the most frequent spelling, while the translation
    variant comes from the most confident fragment.
    """
    expected = block_ids(3)

    analysis = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            candidates=(
                make_candidate(
                    source_text="Mr Wonka",
                    type=ConceptType.PERSON,
                    suggested_translation="мистер Вонка",
                    confidence=0.99,
                ),
            ),
        ),
        make_fragment(
            1,
            blocks=expected[1:2],
            candidates=(
                make_candidate(
                    source_text="Mr. Wonka",
                    type=ConceptType.PERSON,
                    suggested_translation="мистер Вонка",
                    confidence=0.9,
                ),
            ),
        ),
        make_fragment(
            2,
            blocks=expected[2:],
            candidates=(
                make_candidate(
                    source_text="MR. WONKA",
                    type=ConceptType.PERSON,
                    suggested_translation="Мистер  Вонка",
                    confidence=0.8,
                ),
                make_candidate(
                    source_text="Mr. Wonka",
                    type=ConceptType.PERSON,
                    suggested_translation="мистер Вонка",
                    confidence=0.7,
                ),
            ),
        ),
        expected=expected,
    )

    candidates = analysis.glossary_candidates
    assert len(candidates) == 1
    assert candidates[0].source_text == "Mr. Wonka"
    assert candidates[0].suggested_translation == "мистер Вонка"
    assert candidates[0].confidence == 0.99


def test_possessive_form_stays_a_separate_candidate() -> None:
    """Possessive form is not considered an obvious variant and is not merged."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            candidates=(make_candidate(source_text="Wonka's factory"),),
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            candidates=(make_candidate(source_text="Wonka factory"),),
        ),
        expected=expected,
    )

    assert {candidate.source_text for candidate in analysis.glossary_candidates} == {
        "Wonka's factory",
        "Wonka factory",
    }


def test_merged_summary_is_last_fragment_summary() -> None:
    """The last fragment returns the summary for the entire available chapter."""
    expected = block_ids(2)

    analysis = merge(
        make_fragment(1, blocks=expected[1:], chapter_summary="Bingley takes the house."),
        make_fragment(0, blocks=expected[:1], chapter_summary="Mr. Bennet stays silent."),
        expected=expected,
    )

    assert analysis.chapter_summary == "Bingley takes the house."


def test_merged_instructions_keep_each_directive_once() -> None:
    """Identical fragment instructions are not repeated in chapter instructions."""
    expected = block_ids(3)

    analysis = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions="Сохранять иронию."),
        make_fragment(1, blocks=expected[1:2], translation_instructions="Сохранять иронию."),
        make_fragment(2, blocks=expected[2:], translation_instructions="Речь Коллинза — витиевата."),
        expected=expected,
    )

    assert analysis.translation_instructions == (
        "1. Сохранять иронию.\n2. Речь Коллинза — витиевата."
    )


def test_merged_instructions_single_list_without_repeats() -> None:
    expected = block_ids(2)
    analysis = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions="- Сохранять иронию.\n• Передать акцент."),
        make_fragment(1, blocks=expected[1:], translation_instructions="1. сохранять иронию\n2. Не сглаживать речь."),
        expected=expected,
    )
    assert analysis.translation_instructions == "1. Сохранять иронию.\n2. Передать акцент.\n3. Не сглаживать речь."


def test_merged_instructions_have_no_block_labels() -> None:
    expected = block_ids(1)
    analysis = merge(make_fragment(0, blocks=expected, translation_instructions="- Сохранить паузу в блоке b3."), expected=expected)
    assert "b3" not in analysis.translation_instructions


def test_before_you_read_deduplicated_by_normalized_expression() -> None:
    entries = (
        BeforeYouReadEntryResponse(expression="  Entailed  estate! ", explanation="первое"),
        BeforeYouReadEntryResponse(expression="entailed estate", explanation="второе"),
    )
    assert len(before_you_read_entries(entries)) == 1
    assert before_you_read_entries(entries)[0].explanation == "первое"


def test_before_you_read_capped_at_ten() -> None:
    entries = tuple(BeforeYouReadEntryResponse(expression=f"phrase {n}", explanation="значение") for n in range(12))
    assert len(before_you_read_entries(entries)) == 10


def test_fragment_blocks_for_32000_is_160() -> None:
    assert fragment_blocks_for(32000) == 160
    assert len(plan_fragments(chapter_blocks(106), max_blocks=fragment_blocks_for(32000))) == 1


def test_fragment_blocks_for_8000_is_40() -> None:
    assert fragment_blocks_for(8000) == 40


def test_fragment_blocks_without_limit_is_default() -> None:
    assert fragment_blocks_for(None) == 40


def test_before_you_read_keeps_each_expression_once() -> None:
    """An expression encountered by multiple fragments is included in the material once."""
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
    """Two fragments with the same order cannot be merged: their order is unknown."""
    expected = block_ids(2)

    with pytest.raises(AnalysisMergeError):
        merge(
            make_fragment(0, blocks=expected[:1]),
            make_fragment(0, blocks=expected[1:]),
            expected=expected,
        )


def test_fragment_aliases_do_not_collide():
    from linguary.application.llm_dto import ContentPlanResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint
    difficulty = Difficulty("d1", (SourceAnchor("b1", 0, 4, "Echo"),), "b1", "wordplay", "critical", "explanation", (RequiredPoint("p1", "Explain the two meanings"),))
    payload = ChapterAnalysisResponse("Keep voice", "A voice echoes", content_plan=ContentPlanResponse((difficulty,)))
    first = FragmentAnalysis.of_response(payload, order=0, block_ids_by_alias={"b1": "stable1"})
    second = FragmentAnalysis.of_response(payload, order=1, block_ids_by_alias={"b1": "stable2"})
    assert first.content_plan.difficulties[0].id != second.content_plan.difficulties[0].id
    assert first.content_plan.difficulties[0].owner_block_id == "stable1"


def test_conflicting_duplicate_difficulty_rejected():
    from linguary.domain.content_plan import ContentPlan, Difficulty, SourceAnchor, RequiredPoint
    from dataclasses import replace
    d = Difficulty("x", (SourceAnchor("stable", 0, 4, "Echo"),), "stable", "wordplay", "critical", "explanation", (RequiredPoint("p", "Explain"),))
    fragments = [FragmentAnalysis(0, "i", "s", content_plan=ContentPlan(difficulties=(d,))), FragmentAnalysis(1, "i", "s", content_plan=ContentPlan(difficulties=(replace(d, priority="useful"),)))]
    with pytest.raises(AnalysisMergeError):
        merge_chapter_analysis(fragments, id="a", section_id="s", expected_block_ids=(), glossary_confidence_threshold=.8, id_factory=lambda: "id")


def test_merge_preserves_all_required_points():
    from linguary.domain.content_plan import ContentPlan, Difficulty, SourceAnchor, RequiredPoint
    points = (RequiredPoint("p1", "First meaning"), RequiredPoint("p2", "Second meaning"))
    d = Difficulty("stable", (SourceAnchor("block-1", 0, 4, "Echo"),), "block-1", "wordplay", "critical", "explanation", points)
    analysis = merge_chapter_analysis([FragmentAnalysis(0, "i", "s", treatments=(TreatmentPlanItem("block-1", BlockTreatment.READ_AS_IS),), content_plan=ContentPlan(difficulties=(d,)))], id="a", section_id="s", expected_block_ids=("block-1", "block-2"), glossary_confidence_threshold=.8, id_factory=lambda: "id")
    assert analysis.content_plan.difficulties[0].required_points == points
    assert not analysis.is_complete


def test_anchor_and_byr_validation_uses_visible_real_bodies():
    from dataclasses import replace
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint
    from linguary.domain.errors import DomainError
    d = Difficulty("d", (SourceAnchor("b1", 0, 4, "Echo"),), "b1", "wordplay", "critical", "explanation", (RequiredPoint("p", "Explain both meanings"),))
    e = HelpEvidenceResponse("d", "p", "before_you_read.explanation", "b1", 0, "two meanings")
    payload = ChapterAnalysisResponse("i", "s", treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),), before_you_read=(BeforeYouReadEntryResponse("Echo", "This has two meanings."),), content_plan=ContentPlanResponse((d,), (e,)))
    validate_content_plan(payload, {"b1": "stable"}, {"stable": "Echo"})
    with pytest.raises(DomainError):
        validate_content_plan(payload, {"b1": "stable"}, {"stable": "Other"})
    with pytest.raises(AnalysisMergeError):
        validate_content_plan(payload, {"different": "stable"}, {"stable": "Echo"})
    for invalid in (replace(e, artifact_index=9), replace(e, body_excerpt="invented"), replace(e, point_id="other")):
        with pytest.raises((AnalysisMergeError, DomainError)):
            validate_content_plan(replace(payload, content_plan=ContentPlanResponse((d,), (invalid,))), {"b1": "stable"}, {"stable": "Echo"})


def test_duplicate_anchor_merges_all_aspects():
    from dataclasses import replace
    from linguary.domain.content_plan import ContentPlan, Difficulty, SourceAnchor, RequiredPoint
    d = Difficulty("first", (SourceAnchor("block-1", 0, 4, "Echo"),), "block-1", "wordplay", "critical", "explanation", (RequiredPoint("p1", "First meaning"),))
    other = replace(d, id="second", required_points=(RequiredPoint("p2", "Second meaning"),))
    fragments = [FragmentAnalysis(i, "i", "s", content_plan=ContentPlan(difficulties=(item,))) for i, item in enumerate((d, other))]
    analysis = merge_chapter_analysis(fragments, id="a", section_id="s", expected_block_ids=(), glossary_confidence_threshold=.8, id_factory=lambda: "id")
    assert len(analysis.content_plan.difficulties) == 1
    merged = analysis.content_plan.difficulties[0]
    assert {p.id for p in merged.required_points} == {"p1", "p2"}
    assert merged.id == merged.canonical_id


@pytest.mark.parametrize("quoted", ['«Echo»', '“Echo”', '„Echo“', '「Echo」', '"Echo"', "'Echo'", ' \u00a0« Echo »!\n'])
@pytest.mark.parametrize("expression", ["Echo", "A different expression"])
def test_quoted_byr_cannot_cover_lexical_help(quoted, expression):
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint

    difficulty = Difficulty(
        "d", (SourceAnchor("b1", 0, 4, "Echo"),), "b1",
        "vocabulary", "useful", "lexical_hint", (RequiredPoint("p", "Explain the word"),),
    )
    evidence = HelpEvidenceResponse("d", "p", "before_you_read.explanation", "b1", 0, quoted)
    payload = ChapterAnalysisResponse(
        "instructions", "summary",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        before_you_read=(BeforeYouReadEntryResponse(expression, quoted),),
        content_plan=ContentPlanResponse((difficulty,), (evidence,)),
    )
    with pytest.raises(AnalysisMergeError):
        validate_content_plan(payload, {"b1": "stable"}, {"stable": "Echo"})


def test_byr_quote_with_additional_explanation_is_not_rejected_as_quote():
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint

    body = '«Echo» refers to a reflected sound.'
    difficulty = Difficulty(
        "d", (SourceAnchor("b1", 0, 4, "Echo"),), "b1",
        "vocabulary", "useful", "lexical_hint", (RequiredPoint("p", "Explain the word"),),
    )
    payload = ChapterAnalysisResponse(
        "instructions", "summary",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        before_you_read=(BeforeYouReadEntryResponse("Echo", body),),
        content_plan=ContentPlanResponse((difficulty,), (
            HelpEvidenceResponse("d", "p", "before_you_read.explanation", "b1", 0, body),
        )),
    )
    validate_content_plan(payload, {"b1": "stable"}, {"stable": "Echo"})


def test_analysis_plan_respects_configured_note_budget():
    from dataclasses import replace
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint
    d = Difficulty('d', (SourceAnchor('b1', 0, 4, 'Echo'),), 'b1', 'wordplay', 'critical', 'explanation', (RequiredPoint('p', 'Explain'),))
    payload = ChapterAnalysisResponse('i', 's', treatment_plan=(TreatmentPlanItemResponse('b1', BlockTreatment.READ_AS_IS),), content_plan=ContentPlanResponse((d,)))
    with pytest.raises(AnalysisMergeError):
        validate_content_plan(payload, {'b1': 'stable'}, {'stable': 'Echo'}, language_notes_per_chapter=0, note_limits_by_block={'stable': 0})
    e = HelpEvidenceResponse('d', 'p', 'before_you_read.explanation', 'b1', 0, 'Two meanings')
    covered = replace(payload, before_you_read=(BeforeYouReadEntryResponse('Echo', 'Two meanings are contrasted.'),), content_plan=ContentPlanResponse((d,), (e,)))
    validate_content_plan(covered, {'b1': 'stable'}, {'stable': 'Echo'}, language_notes_per_chapter=0, note_limits_by_block={'stable': 0})
    with pytest.raises(AnalysisMergeError):
        validate_content_plan(covered, {'b1': 'stable'}, {'stable': 'Echo'}, before_you_read_limit=0)


def test_find_quote_span_and_anchor_alignment_with_typography():
    from linguary.application.analysis_merge import content_plan_from_response, find_quote_span, validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse
    from linguary.domain.content_plan import Difficulty, RequiredPoint, SourceAnchor

    text = "One dollar. Pennies saved until one’s cheeks burned with imputation of parsimony."
    # 1. Direct span resolution
    res = find_quote_span(text, "one's cheeks burned", estimated_start=15)
    assert res is not None
    start, end, actual = res
    assert actual == "one’s cheeks burned"
    assert text[start:end] == actual

    # Contractions resolution (e.g. she is -> she's)
    text_contractions = "She said she’s such a capital one for catching mice."
    res_contractions = find_quote_span(text_contractions, "she is such a capital one for catching mice")
    assert res_contractions is not None
    assert res_contractions[2] == "she’s such a capital one for catching mice"

    # 2. Inexact bounds from model are resolved when texts are provided
    d = Difficulty(
        "d1",
        (SourceAnchor("b1", 99, 118, "one's cheeks burned"),),
        "b1",
        "vocabulary",
        "critical",
        "explanation",
        (RequiredPoint("p1", "Explain"),),
    )
    payload = ChapterAnalysisResponse(
        "i",
        "s",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),),
        content_plan=ContentPlanResponse((d,)),
    )
    plan = content_plan_from_response(payload, {"b1": "stable"}, {"stable": text})
    assert plan is not None
    resolved_anchor = plan.difficulties[0].anchors[0]
    assert resolved_anchor.start == start
    assert resolved_anchor.end == end
    assert resolved_anchor.quote == "one’s cheeks burned"
    # Validation passes without error
    validate_content_plan(payload, {"b1": "stable"}, {"stable": text})


def test_byr_body_excerpt_with_outer_quotes_aligned_to_explanation():
    from linguary.application.analysis_merge import content_plan_from_response, validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse
    from linguary.domain.content_plan import Difficulty, RequiredPoint, SourceAnchor

    text = "She got it on the sly."
    d = Difficulty(
        "d1",
        (SourceAnchor("b1", 14, 21, "the sly"),),
        "b1",
        "vocabulary",
        "useful",
        "explanation",
        (RequiredPoint("p1", "Explain"),),
    )
    # The explanation does not have outer quotes, but the LLM wrapped body_excerpt in «...»
    payload = ChapterAnalysisResponse(
        "i",
        "s",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),),
        before_you_read=(BeforeYouReadEntryResponse("on the sly", "тайком, украдкой, стараясь не привлекать внимания"),),
        content_plan=ContentPlanResponse(
            (d,),
            (HelpEvidenceResponse("d1", "p1", "before_you_read.explanation", "b1", 0, "«тайком, украдкой, стараясь не привлекать внимания»"),),
        ),
    )
    plan = content_plan_from_response(payload, {"b1": "stable"}, {"stable": text})
    assert plan is not None
    assert plan.before_you_read_evidence[0].body_excerpt == "тайком, украдкой, стараясь не привлекать внимания"
    validate_content_plan(payload, {"b1": "stable"}, {"stable": text})


def test_sanitize_before_you_read_explanation():
    """Sanitizer strips meta-instructions to the translator from Before You Read cards."""
    # 1. Instruction at the end of the sentence ("Render as...")
    text1 = (
        "Намеренно неправильная сравнительная степень (вместо more and more curious): "
        "Алиса так удивлена, что забыла грамматику. Передавай как забавное искажение."
    )
    assert sanitize_before_you_read_explanation(text1) == (
        "Намеренно неправильная сравнительная степень (вместо more and more curious): "
        "Алиса так удивлена, что забыла грамматику."
    )

    # 2. Instruction after a semicolon
    text2 = "Устойчивое выражение; переводи как «не в духе»."
    assert sanitize_before_you_read_explanation(text2) == "Устойчивое выражение."

    # 3. Instruction in a separate sentence with "In the translation..."
    text3 = "Разговорный оборот. В переводе передавай как «чудом спасся»."
    assert sanitize_before_you_read_explanation(text3) == "Разговорный оборот."

    # 4. "Instruction to translator:" prefix
    text4 = "Идиома. Указание переводчику: сохраняй иронию оригинала."
    assert sanitize_before_you_read_explanation(text4) == "Идиома."

    # 5. The entire phrase consists of an instruction "Render as..."
    text5 = "Передавай как «из ряда вон выходящее»."
    assert sanitize_before_you_read_explanation(text5) == "«из ряда вон выходящее»."

    # 6. Instruction in parentheses
    text6 = "Сленговое выражение (в переводе передавай разговорно). Встречается у подростков."
    assert sanitize_before_you_read_explanation(text6) == "Сленговое выражение. Встречается у подростков."

    # 7. Normal explanatory text without instructions remains unchanged
    text7 = "Необычный, странный; в тексте out-of-the-way things — «из ряда вон выходящие вещи»."
    assert sanitize_before_you_read_explanation(text7) == text7


def test_before_you_read_entries_sanitizes_explanation():
    """The before_you_read_entries function automatically cleans up explanations."""
    entries = (
        BeforeYouReadEntryResponse(
            expression="Curiouser and curiouser",
            explanation="Искажение грамматики. Передавай как забавную ошибку.",
        ),
    )
    result = before_you_read_entries(entries)
    assert len(result) == 1
    assert result[0].explanation == "Искажение грамматики."



