"""Fallback scenario for a long chapter: chunking and merging fragments (P3.T4, ALG-03)."""

from __future__ import annotations
from dataclasses import replace

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
from linguary.domain.name_translation import NameTranslationAssessment, NameTranslationPolicy, NameTranslationStrategy

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


def test_fragment_low_confidence_survives() -> None:
    policy = NameTranslationPolicy(1, NameTranslationStrategy.SOUND)
    assessment = NameTranslationAssessment("sound", True, (), "Однозначное имя.")
    fragments = (
        make_fragment(0, blocks=("block-1",), candidates=(
            make_candidate(confidence=0.95, type=ConceptType.PERSON),)),
        make_fragment(1, blocks=("block-2",), candidates=(
            make_candidate(confidence=0.3, type=ConceptType.PERSON),)),
    )
    fragments = tuple(replace(fragment, glossary_candidates=tuple(
        replace(candidate, name_translation_assessment=assessment)
        for candidate in fragment.glossary_candidates)) for fragment in fragments)
    result = merge_chapter_analysis(fragments, id="analysis-1", section_id=SECTION_ID,
        expected_block_ids=("block-1", "block-2"),
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD, id_factory=sequential_ids(),
        name_translation_policy=policy, conditions_id="conditions-1")
    evidence = result.glossary_candidates[0].name_translation_evidence
    assert evidence is not None
    assert [item.confidence for item in evidence.observations] == [0.95, 0.3]


def test_old_merge_unchanged() -> None:
    result = merge(make_fragment(0, blocks=("block-1",), candidates=(make_candidate(),)),
                   expected=("block-1",))
    assert result.glossary_candidates[0].name_translation_evidence is None


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


def test_before_you_read_deduplicates_stems_and_phrase_extensions_by_first_word() -> None:
    entries = (
        BeforeYouReadEntryResponse(expression="Snapped", explanation="первая карточка"),
        BeforeYouReadEntryResponse(expression="snapping", explanation="однокоренное слово"),
        BeforeYouReadEntryResponse(expression="snapped at", explanation="отдельное фразовое выражение"),
        BeforeYouReadEntryResponse(expression="Make a grab", explanation="идиома"),
        BeforeYouReadEntryResponse(expression="making a grab at", explanation="расширение"),
        BeforeYouReadEntryResponse(expression="Started", explanation="другая карточка"),
    )

    result = before_you_read_entries(entries)

    assert [entry.expression for entry in result] == [
        "Snapped",
        "snapped at",
        "Make a grab",
        "Started",
    ]
    assert result[0].explanation == "первая карточка"
    assert result[2].explanation == "идиома"


def test_content_plan_evidence_moves_to_the_first_byr_when_explanation_matches() -> None:
    from linguary.application.analysis_merge import content_plan_from_response, validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse, TreatmentPlanItemResponse
    from linguary.domain.content_plan import Difficulty, RequiredPoint, SourceAnchor, artifact_key

    explanation = "Она сделала резкое движение рукой."
    payload = ChapterAnalysisResponse(
        "instructions",
        "summary",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        before_you_read=(
            BeforeYouReadEntryResponse("made a grab", explanation),
            BeforeYouReadEntryResponse("made a grab at", explanation),
        ),
        content_plan=ContentPlanResponse(
            (Difficulty(
                "d",
                (SourceAnchor("b1", 4, 18, "made a grab at"),),
                "b1",
                "idiom",
                "critical",
                "explanation",
                (RequiredPoint("p", "Объяснить значение выражения"),),
            ),),
            (HelpEvidenceResponse(
                "d", "p", "before_you_read.explanation", "b1", 1, explanation,
            ),),
        ),
    )

    plan = content_plan_from_response(
        payload, {"b1": "stable"}, {"stable": "She made a grab at him."}
    )
    validate_content_plan(
        payload, {"b1": "stable"}, {"stable": "She made a grab at him."}
    )

    evidence = plan.before_you_read_evidence[0]
    assert evidence.artifact_key == artifact_key(
        "before_you_read.explanation",
        "stable",
        {"expression": "made a grab", "explanation": explanation},
    )
    assert evidence.body_excerpt == explanation


def test_content_plan_drops_duplicate_byr_evidence_not_supported_by_first_card() -> None:
    from linguary.application.analysis_merge import content_plan_from_response, validate_content_plan
    from linguary.application.llm_dto import ContentPlanResponse, HelpEvidenceResponse, TreatmentPlanItemResponse
    from linguary.domain.content_plan import Difficulty, RequiredPoint, SourceAnchor

    payload = ChapterAnalysisResponse(
        "instructions",
        "summary",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        before_you_read=(
            BeforeYouReadEntryResponse("made a grab", "Она сделала резкое движение."),
            BeforeYouReadEntryResponse("made a grab at", "Она резко попыталась схватить его."),
        ),
        content_plan=ContentPlanResponse(
            (Difficulty(
                "d",
                (SourceAnchor("b1", 4, 18, "made a grab at"),),
                "b1",
                "idiom",
                "critical",
                "explanation",
                (RequiredPoint("p", "Объяснить значение выражения"),),
            ),),
            (HelpEvidenceResponse(
                "d", "p", "before_you_read.explanation", "b1", 1,
                "Она резко попыталась схватить его.",
            ),),
        ),
    )

    plan = content_plan_from_response(
        payload, {"b1": "stable"}, {"stable": "She made a grab at him."}
    )
    validate_content_plan(
        payload, {"b1": "stable"}, {"stable": "She made a grab at him."}
    )

    assert plan.before_you_read_evidence == ()


def test_before_you_read_capped_at_ten() -> None:
    first_words = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                   "golf", "hotel", "india", "juliet", "kilo", "lima")
    entries = tuple(
        BeforeYouReadEntryResponse(expression=f"{word} phrase", explanation="значение")
        for word in first_words
    )
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


def test_content_plan_without_byr_evidence_binds_algorithmically() -> None:
    """When LLM returns no before_you_read_evidence, it is bound algorithmically by matching anchors (AC-008)."""
    from linguary.application.analysis_merge import content_plan_from_response
    from linguary.application.llm_dto import ContentPlanResponse
    from linguary.domain.content_plan import Difficulty, RequiredPoint, SourceAnchor

    text = "She snapped at him angrily."
    d = Difficulty(
        "d1",
        (SourceAnchor("b1", 4, 14, "snapped at"),),
        "b1",
        "idiom",
        "critical",
        "explanation",
        (RequiredPoint("p1", "Explain the idiom"),),
    )
    payload = ChapterAnalysisResponse(
        "instructions",
        "summary",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),),
        before_you_read=(BeforeYouReadEntryResponse("snapped at", "Ответил собеседнику сердито и резко."),),
        content_plan=ContentPlanResponse(
            (d,),
            before_you_read_evidence=(),  # Empty from LLM
        ),
    )
    plan = content_plan_from_response(payload, {"b1": "block-1"}, {"block-1": text})
    assert plan is not None
    assert len(plan.before_you_read_evidence) == 1
    ev = plan.before_you_read_evidence[0]
    assert ev.difficulty_id == plan.difficulties[0].id
    assert ev.point_id == "p1"
    assert ev.artifact_kind == "before_you_read.explanation"
    assert ev.block_id == "block-1"
    assert ev.body_excerpt == "Ответил собеседнику сердито и резко."


def test_content_plan_schemas_do_not_contain_evidence_in_strict_json_schema() -> None:
    """Evidence fields are excluded from JSON schemas sent to the provider (AC-008)."""
    from linguary.adapters.llm.json_schema import strict_json_schema
    from linguary.adapters.llm.schemas import (
        ContentPlanSchema,
        TranslationResultV2Schema,
        TranslationResultV3Schema,
    )

    plan_schema = strict_json_schema(ContentPlanSchema)
    assert "before_you_read_evidence" not in plan_schema.get("properties", {})

    t_v2_schema = strict_json_schema(TranslationResultV2Schema)
    assert "content_evidence" not in t_v2_schema.get("properties", {})

    t_v3_schema = strict_json_schema(TranslationResultV3Schema)
    assert "content_evidence" not in t_v3_schema.get("properties", {})


def test_deduplicate_book_before_you_read_removes_later_duplicates_and_updates_artifact_keys() -> None:
    from analysis_fakes import make_analysis_document, make_analysis_section
    from linguary.application.analysis_merge import deduplicate_book_before_you_read
    from linguary.domain import BeforeYouReadEntry, BlockTreatment, TreatmentPlan
    from linguary.domain.content_plan import ContentPlan, Difficulty, HelpEvidence, RequiredPoint, SourceAnchor

    s1 = make_analysis_section(section_id="sec-1", order=0, texts=("First text.",))
    b1 = s1.blocks[0]
    a1 = ChapterAnalysis.create(
        id="a1",
        section_id="sec-1",
        treatment_plan=TreatmentPlan.of_treatments(s1.id, ((b1.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id,),
        before_you_read=(BeforeYouReadEntry("make a grab at", "резко схватить"),),
    )

    s2 = make_analysis_section(section_id="sec-2", order=1, texts=("Second text.",))
    b2 = s2.blocks[0]
    card_dup = BeforeYouReadEntry("make a grab", "схватить")
    card_novel = BeforeYouReadEntry("iron curtain", "железный занавес")
    d = Difficulty("d1", (SourceAnchor(b2.id, 0, 5, "start"),), b2.id, "idiom", "critical", "explanation", (RequiredPoint("p1", "desc"),))
    ev = HelpEvidence("d1", "p1", "before_you_read.explanation", b2.id, "byr:1", "железный занавес")
    plan2 = ContentPlan(version=3, profile="content-plan-v3", difficulties=(d,), before_you_read_evidence=(ev,))
    a2 = ChapterAnalysis.create(
        id="a2",
        section_id="sec-2",
        treatment_plan=TreatmentPlan.of_treatments(s2.id, ((b2.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b2.id,),
        before_you_read=(card_dup, card_novel),
        content_plan=plan2,
    )

    doc = make_analysis_document((s1, s2))
    res1, res2 = deduplicate_book_before_you_read(doc, (a1, a2))

    assert res1.before_you_read == a1.before_you_read
    assert len(res2.before_you_read) == 1
    assert res2.before_you_read[0].expression == "iron curtain"
    assert len(res2.content_plan.before_you_read_evidence) == 1
    assert res2.content_plan.before_you_read_evidence[0].artifact_key == "byr:0"


def test_deduplicate_book_before_you_read_preserves_card_required_by_chapter_plan() -> None:
    from analysis_fakes import make_analysis_document, make_analysis_section
    from linguary.application.analysis_merge import deduplicate_book_before_you_read
    from linguary.domain import BeforeYouReadEntry, BlockTreatment, TreatmentPlan
    from linguary.domain.content_plan import ContentPlan, Difficulty, HelpEvidence, RequiredPoint, SourceAnchor

    s1 = make_analysis_section(section_id="sec-1", order=0, texts=("First text.",))
    b1 = s1.blocks[0]
    a1 = ChapterAnalysis.create(
        id="a1",
        section_id="sec-1",
        treatment_plan=TreatmentPlan.of_treatments(s1.id, ((b1.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id,),
        before_you_read=(BeforeYouReadEntry("make a grab at", "резко схватить"),),
    )

    s2 = make_analysis_section(section_id="sec-2", order=1, texts=("Second text.",))
    b2 = s2.blocks[0]
    card_dup = BeforeYouReadEntry("make a grab", "схватить")
    d = Difficulty("d1", (SourceAnchor(b2.id, 0, 5, "start"),), b2.id, "idiom", "critical", "explanation", (RequiredPoint("p1", "desc"),))
    ev = HelpEvidence("d1", "p1", "before_you_read.explanation", b2.id, "byr:0", "схватить")
    plan2 = ContentPlan(version=3, profile="content-plan-v3", difficulties=(d,), before_you_read_evidence=(ev,))
    a2 = ChapterAnalysis.create(
        id="a2",
        section_id="sec-2",
        treatment_plan=TreatmentPlan.of_treatments(s2.id, ((b2.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b2.id,),
        before_you_read=(card_dup,),
        content_plan=plan2,
    )

    doc = make_analysis_document((s1, s2))
    res1, res2 = deduplicate_book_before_you_read(doc, (a1, a2))

    assert len(res2.before_you_read) == 1
    assert res2.before_you_read[0].expression == "make a grab"
    assert len(res2.content_plan.before_you_read_evidence) == 1
    assert res2.content_plan.before_you_read_evidence[0].artifact_key == "byr:0"


def test_before_you_read_keeps_distinct_phrasal_verbs_with_same_verb() -> None:
    from analysis_fakes import make_analysis_document, make_analysis_section
    from linguary.application.analysis_merge import (
        _before_you_read_key,
        before_you_read_entries,
        deduplicate_book_before_you_read,
    )
    from linguary.domain import BeforeYouReadEntry, BlockTreatment, TreatmentPlan

    entries = (
        BeforeYouReadEntryResponse("take off", "взлететь или снять одежду"),
        BeforeYouReadEntryResponse("take place", "происходить, иметь место"),
        BeforeYouReadEntryResponse("taken off", "дубль первого выражения"),
    )
    selected = before_you_read_entries(entries)
    assert [e.expression for e in selected] == ["take off", "take place"]

    assert _before_you_read_key("president") != _before_you_read_key("present")
    assert _before_you_read_key("take off") != _before_you_read_key("take place")

    s1 = make_analysis_section(section_id="sec-1", order=0, texts=("First text.",))
    s2 = make_analysis_section(section_id="sec-2", order=1, texts=("Second text.",))
    b1, b2 = s1.blocks[0], s2.blocks[0]
    a1 = ChapterAnalysis.create(
        id="a1",
        section_id="sec-1",
        treatment_plan=TreatmentPlan.of_treatments(s1.id, ((b1.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b1.id,),
        before_you_read=(
            BeforeYouReadEntry("take off", "взлететь"),
            BeforeYouReadEntry("president", "президент"),
        ),
    )
    a2 = ChapterAnalysis.create(
        id="a2",
        section_id="sec-2",
        treatment_plan=TreatmentPlan.of_treatments(s2.id, ((b2.id, BlockTreatment.TRANSLATE),)),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(b2.id,),
        before_you_read=(
            BeforeYouReadEntry("take place", "происходить"),
            BeforeYouReadEntry("present", "подарок"),
            BeforeYouReadEntry("taken off", "взлететь снова"),
        ),
    )
    doc = make_analysis_document((s1, s2))
    _, res2 = deduplicate_book_before_you_read(doc, (a1, a2))
    assert [e.expression for e in res2.before_you_read] == ["take place", "present"]


def test_resolve_quoted_anchor_resolves_repeated_quote_in_block() -> None:
    from linguary.application.analysis_merge import _resolve_quoted_anchor, content_plan_from_response
    from linguary.application.llm_dto import (
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    block_text = "Wait for him. Wait until dark."
    aliases = {"b1": "block-1"}
    texts = {"block-1": block_text}
    visible = {"b1": block_text}

    anchor = _resolve_quoted_anchor("b1", "Wait", aliases, texts, visible)
    assert anchor is not None
    assert anchor.block_id == "block-1"
    assert anchor.start == 0
    assert anchor.end == 4
    assert anchor.quote == "Wait"

    payload = ChapterAnalysisResponse(
        translation_instructions="Сохраняй смысл.",
        chapter_summary="Сводка.",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                QuotedDifficultyResponse(
                    id="d1",
                    anchors=(QuotedSourceAnchorResponse(block_id="b1", quote="Wait"),),
                    owner_block_id="b1",
                    kind="idiom",
                    priority="critical",
                    help_kind="explanation",
                    required_points=(RequiredPoint("p1", "Explain wait"),),
                ),
            ),
        ),
    )
    plan = content_plan_from_response(payload, aliases, texts, visible_texts_by_alias=visible)
    assert plan is not None
    assert len(plan.difficulties) == 1
    assert plan.difficulties[0].anchors[0].quote == "Wait"


def test_validate_content_plan_attaches_local_canonical_violation_and_parses_composite_gender_notes() -> None:
    from linguary.adapters.validation import Py3LangidDetector
    from linguary.application.analysis_merge import (
        content_plan_from_response,
        glossary_context_constraints,
        validate_content_plan,
        validate_fact_glossary,
    )
    from linguary.application.llm_dto import (
        QuotedContentPlanResponse,
        QuotedContextFactResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.application.reader_text import ReaderTextError, validate_a2_summary
    from linguary.domain import TermConcept, TermStatus, TermVariant
    from linguary.domain.content_plan import ContextFact, RequiredPoint, SourceAnchor
    from linguary.domain.llm_log import CallContext

    aliases = {"b1": "block-uuid-1", "b2": "block-uuid-2"}
    texts = {
        "block-uuid-1": "First paragraph.",
        "block-uuid-2": "Mr Wonka smiled politely.",
    }
    visible = {
        "b1": "First paragraph.",
        "b2": "Mr Wonka smiled politely.",
    }
    call_context = CallContext(
        conditions_id="00000000-0000-0000-0000-000000000001",
        stage="chapter_analysis",
        scope_id="section-1",
        ordered_block_ids=("block-uuid-1", "block-uuid-2"),
        attempt_id="00000000-0000-0000-0000-000000000002",
        generation=0,
        assigned_model="gpt-test",
        prompt_hash="a" * 64,
    )

    unresolved_payload = ChapterAnalysisResponse(
        translation_instructions="Сохраняй смысл.",
        chapter_summary="Сводка.",
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
        ),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                QuotedDifficultyResponse(
                    id="d1",
                    anchors=(
                        QuotedSourceAnchorResponse(block_id="b2", quote="missing phrase"),
                    ),
                    owner_block_id="b2",
                    kind="idiom",
                    priority="critical",
                    help_kind="explanation",
                    required_points=(RequiredPoint("p1", "Explain phrase"),),
                ),
            ),
        ),
    )

    assert (
        content_plan_from_response(
            unresolved_payload, aliases, texts, visible_texts_by_alias=visible
        )
        is None
    )

    with pytest.raises(AnalysisMergeError) as unresolved_exc:
        validate_content_plan(
            unresolved_payload, aliases, texts, visible_texts_by_alias=visible
        )
    unresolved_err = unresolved_exc.value
    assert str(unresolved_err).startswith("цитата не разрешается однозначно в тексте блока")
    assert "b2" in str(unresolved_err)
    assert "«missing phrase»" in str(unresolved_err)
    assert "—" in str(unresolved_err)
    assert "block-uuid-2" not in str(unresolved_err)
    assert unresolved_err.violation is not None
    assert unresolved_err.violation.stage == "chapter_analysis"
    assert unresolved_err.violation.scope == "local"
    assert unresolved_err.violation.block_ids == ("block-uuid-2",)
    verified_unresolved = unresolved_err.violation.verified(call_context)
    assert verified_unresolved.scope == "local"
    assert verified_unresolved.block_ids == ("block-uuid-2",)

    lexical_payload = ChapterAnalysisResponse(
        translation_instructions="Сохраняй смысл.",
        chapter_summary="Сводка.",
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
        ),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                QuotedDifficultyResponse(
                    id="d1",
                    anchors=(
                        QuotedSourceAnchorResponse(block_id="b2", quote="politely"),
                    ),
                    owner_block_id="b2",
                    kind="vocabulary",
                    priority="useful",
                    help_kind="lexical_hint",
                    required_points=(RequiredPoint("p1", "Explain word"),),
                ),
            ),
            context_facts=(
                QuotedContextFactResponse(
                    id="cf-1",
                    kind="gender",
                    anchors=(QuotedSourceAnchorResponse(block_id="b2", quote="Mr Wonka"),),
                    subject_anchor=QuotedSourceAnchorResponse(block_id="b2", quote="Mr Wonka"),
                    value="female",
                    certainty="inferred",
                ),
            ),
        ),
    )

    with pytest.raises(AnalysisMergeError) as lexical_exc:
        validate_content_plan(
            lexical_payload, aliases, texts, visible_texts_by_alias=visible
        )
    lexical_err = lexical_exc.value
    assert str(lexical_err).startswith(
        "лексическая помощь требует GLOSSARY_ONLY или подтверждённый BYR"
    )
    assert "b2" in str(lexical_err)
    assert "block-uuid-2" not in str(lexical_err)
    assert lexical_err.violation is not None
    assert lexical_err.violation.verified(call_context).scope == "local"
    assert lexical_err.violation.verified(call_context).block_ids == ("block-uuid-2",)

    concepts = (
        TermConcept(
            id="c1",
            book_id="book-1",
            type=ConceptType.PERSON,
            canonical_source_text="Mara",
            status=TermStatus.APPROVED,
            note="имя собств., женский род, не склоняется",
            variants=(TermVariant("c1", "ru", "Мара", TermStatus.APPROVED),),
        ),
        TermConcept(
            id="c2",
            book_id="book-1",
            type=ConceptType.PERSON,
            canonical_source_text="MR. WONKA",
            status=TermStatus.LOCKED,
            note="титул и фамилия, мужского рода",
            variants=(TermVariant("c2", "ru", "мистер Вонка", TermStatus.LOCKED),),
        ),
        TermConcept(
            id="c3",
            book_id="book-1",
            type=ConceptType.TERM,
            canonical_source_text="It-Entity",
            status=TermStatus.APPROVED,
            note="сущ., средний род",
            variants=(TermVariant("c3", "ru", "Существо", TermStatus.APPROVED),),
        ),
        TermConcept(
            id="c4",
            book_id="book-1",
            type=ConceptType.PERSON,
            canonical_source_text="SuggestedHero",
            status=TermStatus.SUGGESTED,
            note="female",
            variants=(TermVariant("c4", "ru", "Героиня", TermStatus.SUGGESTED),),
        ),
    )
    constraints = glossary_context_constraints(concepts)
    assert constraints == {
        "mara": {"gender": "female"},
        "mr wonka": {"gender": "male"},
        "it-entity": {"gender": "neuter"},
    }

    valid_plan_payload = replace(
        lexical_payload,
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.GLOSSARY_ONLY),
        ),
    )
    resolved_plan = validate_content_plan(
        valid_plan_payload, aliases, texts, visible_texts_by_alias=visible
    )
    assert resolved_plan is not None

    with pytest.raises(AnalysisMergeError) as fact_exc:
        validate_fact_glossary(
            resolved_plan,
            {"MR. WONKA": {"gender": "male"}},
            aliases_by_block_id={"block-uuid-2": "b2"},
        )
    fact_err = fact_exc.value
    assert str(fact_err).startswith(
        "контекстная атрибуция противоречит обязательному решению глоссария"
    )
    assert "b2" in str(fact_err)
    assert "block-uuid-2" not in str(fact_err)
    assert fact_err.violation is not None
    assert fact_err.violation.verified(call_context).scope == "local"
    assert fact_err.violation.verified(call_context).block_ids == ("block-uuid-2",)

    matching_russian_gender_plan = replace(
        resolved_plan,
        context_facts=(
            ContextFact(
                id="cf-ok",
                kind="gender",
                anchors=(SourceAnchor("block-uuid-2", 0, 8, "Mr Wonka"),),
                subject_anchor=SourceAnchor("block-uuid-2", 0, 8, "Mr Wonka"),
                value="мужской род",
                certainty="explicit",
            ),
        ),
    )
    validate_fact_glossary(matching_russian_gender_plan, {"MR. WONKA": {"gender": "male"}})

    detector = Py3LangidDetector()
    with pytest.raises(ReaderTextError) as a2_exc:
        validate_a2_summary(
            "Mr Wonka smiled politely.\n\nStill in English instead of Russian.",
            source_texts=("Mr Wonka smiled politely.",),
            source_language="en",
            language_detector=detector,
        )
    a2_err = a2_exc.value
    assert a2_err.reason_code == "a2_summary_requires_parallel_russian"
    assert a2_err.violation is not None
    assert a2_err.violation.rule_code == "a2_summary_requires_parallel_russian"
    assert "абзац 2" in str(a2_err)
    assert "EN\\nRU" in str(a2_err)


def test_find_quote_span_prefers_exact_case_and_falls_back_to_case_insensitive_typography() -> None:
    from linguary.application.analysis_merge import find_quote_span

    exact_text = "WAIT here, then Wait there."
    assert find_quote_span(exact_text, "Wait", estimated_start=0) == (16, 20, "Wait")

    typo_case_text = "ONE’S cheeks burned, while one’s hands trembled."
    assert find_quote_span(typo_case_text, "one's", estimated_start=0) == (27, 32, "one’s")

    ci_text = "One’s cheeks burned. Later, ONE’S hands trembled."
    assert find_quote_span(ci_text, "one's", estimated_start=0) == (0, 5, "One’s")
    assert find_quote_span(ci_text, "one's", estimated_start=25) == (28, 33, "ONE’S")

    wrapped_text = "Don’t make a grab at it!"
    assert find_quote_span(wrapped_text, "“—don't make a grab at—”") == (
        0,
        20,
        "Don’t make a grab at",
    )
    assert find_quote_span(wrapped_text, "‘–don't make a grab at–’") == (
        0,
        20,
        "Don’t make a grab at",
    )


def test_same_byr_prefix_preserves_multiword_idioms_and_keeps_make_a_grab_overlap() -> None:
    from linguary.application.analysis_merge import _same_byr_prefix

    # Single-word items match single-word items only
    assert _same_byr_prefix(("snapped",), ("snapping",)) is True
    assert _same_byr_prefix(("snapped",), ("snapped", "at")) is False
    assert _same_byr_prefix(("dead",), ("dead", "end")) is False

    # Length ratio guard: 2 * len(shorter) >= len(longer)
    assert _same_byr_prefix(("break", "the", "ice"), ("breaking", "the", "ice", "now")) is True
    assert (
        _same_byr_prefix(
            ("break", "the", "ice"),
            ("break", "the", "ice", "at", "the", "formal", "dinner"),
        )
        is False
    )

    # At least 2 non-function words in shorter
    assert _same_byr_prefix(("in", "the"), ("in", "the", "dark")) is False
    assert _same_byr_prefix(("on", "fire"), ("on", "fire", "with")) is False
    assert _same_byr_prefix(("make", "a"), ("make", "a", "grab")) is False

    # Full lexical verbs (make, take, give, keep, turn) are NOT treated as stop-words
    assert _same_byr_prefix(("make", "a", "grab"), ("make", "a", "grab", "at")) is True
    assert _same_byr_prefix(("take", "heart"), ("taking", "heart", "from")) is True
    assert _same_byr_prefix(("give", "way"), ("giving", "way", "to")) is True
    assert _same_byr_prefix(("keep", "watch"), ("keeping", "watch", "over")) is True
    assert _same_byr_prefix(("turn", "tail"), ("turning", "tail", "quickly")) is True

