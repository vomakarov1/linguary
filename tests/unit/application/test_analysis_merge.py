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
    BlockType,
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
    """A chapter over the block cap balances blocks across the minimum fragments."""
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


def test_merged_instructions_deduplicates_semantic_rephrasings_preserving_polarity_and_names() -> None:
    # 1. Semantic rephrasing with >= 3 shared significant stems folds; asymmetric containment replaces in-place.
    expected = block_ids(3)
    folded = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            translation_instructions="Сохранять иронию рассказчика в диалогах.",
        ),
        make_fragment(
            1,
            blocks=expected[1:2],
            translation_instructions="Передавать живой тон писем без упрощения.",
        ),
        make_fragment(
            2,
            blocks=expected[2:],
            translation_instructions=(
                "- В диалогах сохраняйте иронию рассказчика.\n"
                "- Сохранять лёгкую иронию рассказчика в диалогах без потери интонации."
            ),
        ),
        expected=expected,
    )
    assert folded.translation_instructions == (
        "1. Сохранять лёгкую иронию рассказчика в диалогах без потери интонации.\n"
        "2. Передавать живой тон писем без упрощения."
    )

    # 2. Guards forbid merging on register (ты/вы), negation (не), quoted strings, and mid-sentence proper nouns.
    guarded = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            translation_instructions="\n\n".join([
                "Обращаться к спутнику на ты в диалогах главы.",
                "Обращаться к спутнику на вы в диалогах главы.",
                "Передавать обращение к тебе уважительным тоном в письмах.",
                "Передавать обращение к вам уважительным тоном в письмах.",
                "Не сглаживать грубую разговорную речь персонажей.",
                "Сглаживать грубую разговорную речь персонажей.",
                "Передавать восклицание «Oh my» естественной русской репликой.",
                "Передавать восклицание «Good grief» естественной русской репликой.",
                "Передавать ироничный тон реплик Ника в диалогах.",
                "Передавать ироничный тон реплик Элли в диалогах.",
                "Передавать речь героя в доме Nick's и Allie's мягко.",
            ]),
        ),
        make_fragment(
            1,
            blocks=expected[1:2],
            translation_instructions="\n\n".join([
                'Передавать восклицание "oh my" естественной русской репликой.',
                "Передавать восклицание 'Oh my' естественной русской репликой.",
                "В диалогах передавать ироничный тон реплик Нику.",
                "Мягко передавать речь героя в доме Nick's и Allie's.",
            ]),
        ),
        make_fragment(
            2,
            blocks=expected[2:],
            translation_instructions=(
                "Сохранять иронию: Реплики звучат сухо.\n\n"
                "Сохранять иронию — Реплики звучат сухо."
            ),
        ),
        expected=expected,
    )
    lines = guarded.translation_instructions.splitlines()
    assert len(lines) == 12
    assert "Обращаться к спутнику на ты в диалогах главы." in guarded.translation_instructions
    assert "Обращаться к спутнику на вы в диалогах главы." in guarded.translation_instructions
    assert "Передавать обращение к тебе уважительным тоном в письмах." in guarded.translation_instructions
    assert "Передавать обращение к вам уважительным тоном в письмах." in guarded.translation_instructions
    assert "Не сглаживать грубую разговорную речь персонажей." in guarded.translation_instructions
    assert "Сглаживать грубую разговорную речь персонажей." in guarded.translation_instructions
    assert "Передавать восклицание «Oh my» естественной русской репликой." in guarded.translation_instructions
    assert "Передавать восклицание «Good grief» естественной русской репликой." in guarded.translation_instructions
    assert "Передавать ироничный тон реплик Ника в диалогах." in guarded.translation_instructions
    assert "Передавать ироничный тон реплик Элли в диалогах." in guarded.translation_instructions
    assert "Передавать речь героя в доме Nick's и Allie's мягко." in guarded.translation_instructions
    assert "Сохранять иронию: Реплики звучат сухо." in guarded.translation_instructions
    assert "Сохранять иронию — Реплики звучат сухо." not in guarded.translation_instructions

    # 3. Short instructions with < 3 significant stems are not collapsed.
    short = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions="Keep instruction one."),
        make_fragment(1, blocks=expected[1:2], translation_instructions="Keep instruction two."),
        expected=expected[:2],
    )
    assert short.translation_instructions == "1. Keep instruction one.\n2. Keep instruction two."


def test_merged_instructions_lowercase_russian_abbreviations_do_not_block_dedup_while_initials_and_honorifics_do() -> None:
    expected = block_ids(2)

    # Lowercase Russian abbreviations ("и т. д.", "и т. п.") end a sentence/clause:
    # the next capitalized word ("Передавай...") must not be misclassified as a proper noun.
    ru_abbrev = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            translation_instructions=(
                "Сохранять детали быта, титулы и т. д. "
                "Передавай живую разговорную речь героев."
            ),
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            translation_instructions=(
                "Сохранять детали быта, титулы и т. д. "
                "В диалогах передавай живую разговорную речь героев."
            ),
        ),
        expected=expected,
    )
    assert ru_abbrev.translation_instructions == (
        "1. Сохранять детали быта, титулы и т. д. "
        "В диалогах передавай живую разговорную речь героев."
    )

    ru_tp = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            translation_instructions=(
                "Учитывать бытовые детали, прозвища и т. п. "
                "Передавай живой тон диалогов."
            ),
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            translation_instructions=(
                "Учитывать бытовые детали, прозвища и т. п. "
                "В диалогах передавай живой тон."
            ),
        ),
        expected=expected,
    )
    assert len(ru_tp.translation_instructions.splitlines()) == 1

    # Real uppercase initials ("J. Smith") and honorifics ("Mr. Smith", "mr. Smith")
    # must still prevent sentence-start reset so distinct names are not merged.
    for pair in (
        (
            "Сохранять сдержанный тон реплик в сценах. J. Smith говорит кратко в диалогах.",
            "Сохранять сдержанный тон реплик в сценах. J. Brown говорит кратко в диалогах.",
        ),
        (
            "Сохранять сдержанный тон реплик в сценах. Mr. Smith говорит кратко в диалогах.",
            "Сохранять сдержанный тон реплик в сценах. Mr. Brown говорит кратко в диалогах.",
        ),
        (
            "Сохранять сдержанный тон реплик в сценах. mr. Smith говорит кратко в диалогах.",
            "Сохранять сдержанный тон реплик в сценах. mr. Brown говорит кратко в диалогах.",
        ),
    ):
        preserved = merge(
            make_fragment(0, blocks=expected[:1], translation_instructions=pair[0]),
            make_fragment(1, blocks=expected[1:], translation_instructions=pair[1]),
            expected=expected,
        )
        assert len(preserved.translation_instructions.splitlines()) == 2


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
    assert fragment_blocks_for(None) == 160


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

    from linguary.domain.content_plan import SourceAnchor

    anchor = _resolve_quoted_anchor("b1", "Wait", aliases, texts, visible)
    assert not isinstance(anchor, SourceAnchor)

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
    assert plan.difficulties == ()


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
            "Mr Wonka smiled politely.\n\nМистер Вонка вежливо улыбнулся.",
            source_texts=("Mr Wonka smiled politely.",),
            source_language="en",
            language_detector=detector,
        )
    a2_err = a2_exc.value
    assert a2_err.reason_code == "a2_summary_requires_single_source_paragraph"
    assert a2_err.violation is not None
    assert a2_err.violation.rule_code == "a2_summary_requires_single_source_paragraph"
    assert "получено абзацев — 2" in str(a2_err)


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


def _text_blocks(*texts: str) -> tuple[DocumentBlock, ...]:
    return tuple(
        make_analysis_block(section_id=SECTION_ID, order=index, text=text)
        for index, text in enumerate(texts)
    )


def test_fragment_splits_on_char_budget_and_keeps_block_whole() -> None:
    """A fragment stays within block, char, and weight budgets, without cutting a block."""
    from linguary.application.analysis_merge import (
        DEFAULT_MAX_FRAGMENT_BLOCKS,
        DEFAULT_MAX_FRAGMENT_CHARS,
    )

    assert DEFAULT_MAX_FRAGMENT_BLOCKS == 160
    assert DEFAULT_MAX_FRAGMENT_CHARS == 24_000
    piece = "a" * 12_000
    fragments = plan_fragments(_text_blocks(piece, piece, piece))
    assert [len(fragment.blocks) for fragment in fragments] == [2, 1]
    assert [fragment.blocks[0].text for fragment in fragments] == [piece, piece]

    huge = "b" * (DEFAULT_MAX_FRAGMENT_CHARS + 1)
    fragments = plan_fragments(_text_blocks(huge, "tail"))
    assert [len(fragment.blocks) for fragment in fragments] == [1, 1]
    assert fragments[0].blocks[0].text == huge
    assert fragments[1].blocks[0].text == "tail"

    assert len(plan_fragments(chapter_blocks(41))) == 1

    many = chapter_blocks(DEFAULT_MAX_FRAGMENT_BLOCKS + 1)
    fragments = plan_fragments(many)
    assert [len(fragment.blocks) for fragment in fragments] == [81, 80]
    assert [block.id for fragment in fragments for block in fragment.blocks] == [
        block.id for block in many
    ]


def _quote_payload(*difficulties, facts=()):
    from linguary.application.llm_dto import QuotedContentPlanResponse

    return ChapterAnalysisResponse(
        translation_instructions="Сохраняй смысл.",
        chapter_summary="Сводка.",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),),
        content_plan=QuotedContentPlanResponse(difficulties=difficulties, context_facts=facts),
    )


def _difficulty(quote, *, start=None, end=None, difficulty_id="d1", owner="b1", point="Explain"):
    from linguary.application.llm_dto import QuotedDifficultyResponse, QuotedSourceAnchorResponse
    from linguary.domain.content_plan import RequiredPoint

    return QuotedDifficultyResponse(
        id=difficulty_id,
        anchors=(QuotedSourceAnchorResponse(owner, quote, start, end),),
        owner_block_id=owner,
        kind="idiom",
        priority="critical",
        help_kind="explanation",
        required_points=(RequiredPoint("p1", point),),
    )


def _fact(quote, *, start=None, end=None, fact_id="f1"):
    from linguary.application.llm_dto import QuotedContextFactResponse, QuotedSourceAnchorResponse

    anchor = QuotedSourceAnchorResponse("b1", quote, start, end)
    return QuotedContextFactResponse(fact_id, "gender", (anchor,), anchor, "male", "explicit")


def test_unique_quote_binds_without_offset() -> None:
    """A quote that occurs once is bound by its text, with no start or end."""
    from linguary.application.analysis_merge import content_plan_from_response

    text = "Mr Wonka smiled politely."
    aliases = {"b1": "block-1"}
    texts = {"block-1": text}
    visible = {"b1": text}
    plan = content_plan_from_response(
        _quote_payload(_difficulty("politely")),
        aliases,
        texts,
        visible_texts_by_alias=visible,
    )
    assert plan is not None
    anchor = plan.difficulties[0].anchors[0]
    start = text.index("politely")
    assert (anchor.block_id, anchor.start, anchor.end, anchor.quote) == (
        "block-1",
        start,
        start + len("politely"),
        "politely",
    )


def test_repeated_quote_with_exact_offset_accepted() -> None:
    """A repeated quote binds only the occurrence named by start and end."""
    from linguary.adapters.llm.schemas import QuotedSourceAnchorSchema
    from linguary.application.analysis_merge import content_plan_from_response
    from linguary.application.analysis_pass_cache import decode_pass, encode_pass
    from linguary.application.llm_dto import (
        NarrativeAnalysisResponse,
        QuotedContextFactResponse,
        QuotedSourceAnchorResponse,
    )

    text = "Wait for him. Wait until dark."
    second = text.index("Wait", 1)
    aliases = {"b1": "block-1"}
    texts = {"block-1": text}
    visible = {"b1": text}
    plan = content_plan_from_response(
        _quote_payload(_difficulty("Wait", start=second, end=second + len("Wait"))),
        aliases,
        texts,
        visible_texts_by_alias=visible,
    )
    assert plan is not None
    anchor = plan.difficulties[0].anchors[0]
    assert (anchor.start, anchor.end, anchor.quote) == (second, second + 4, "Wait")
    assert anchor.start != 0

    parsed = QuotedSourceAnchorSchema.model_validate(
        {"block_id": "b1", "quote": "Wait", "start": second, "end": second + 4, "note": "лишнее"}
    )
    assert (parsed.start, parsed.end, parsed.quote) == (second, second + 4, "Wait")
    omitted = QuotedSourceAnchorSchema.model_validate({"block_id": "b1", "quote": "Wait"})
    assert omitted.start is None and omitted.end is None

    repeated = QuotedSourceAnchorResponse("b1", "Wait", second, second + 4)
    unique = QuotedSourceAnchorResponse("b1", "dark")
    encoded = encode_pass(NarrativeAnalysisResponse(
        "Keep.", "Summary.",
        context_facts=(QuotedContextFactResponse("f", "gender", (repeated,), unique, "male", "explicit"),),
    ))
    stored = encoded["response"]["context_facts"][0]
    assert stored["anchors"][0]["start"] == second
    assert stored["anchors"][0]["end"] == second + 4
    assert "start" not in stored["subject_anchor"]
    assert "end" not in stored["subject_anchor"]
    restored = decode_pass(encoded, narrative=True)
    assert restored.context_facts[0].anchors[0].start == second
    assert restored.context_facts[0].subject_anchor.start is None


def test_repeated_quote_without_offset_drops_anchor_not_plan() -> None:
    """A missing or inexact offset removes that anchor; the rest of the plan stays."""
    from linguary.application.analysis_merge import content_plan_from_response, validate_content_plan
    from linguary.application.llm_dto import QuotedDifficultyResponse, QuotedSourceAnchorResponse
    from linguary.domain.content_plan import RequiredPoint

    text = "Wait for him. Wait until dark."
    aliases = {"b1": "block-1"}
    texts = {"block-1": text}
    visible = {"b1": text}
    kept = _difficulty("dark", difficulty_id="kept", point="Explain dark")
    mixed = QuotedDifficultyResponse(
        id="mixed",
        anchors=(
            QuotedSourceAnchorResponse("b1", "Wait"),
            QuotedSourceAnchorResponse("b1", "until"),
        ),
        owner_block_id="b1",
        kind="vocabulary",
        priority="useful",
        help_kind="explanation",
        required_points=(RequiredPoint("p1", "Explain the ending"),),
    )
    payload = _quote_payload(
        _difficulty("Wait", difficulty_id="repeated"),
        _difficulty("Wait", start=1, end=5, difficulty_id="near", point="Nearby offset"),
        kept,
        mixed,
        facts=(
            _fact("Wait", fact_id="bad-subject"),
            _fact("dark", fact_id="good-subject"),
        ),
    )
    plan = content_plan_from_response(payload, aliases, texts, visible_texts_by_alias=visible)
    assert plan is not None
    quotes = {anchor.quote for difficulty in plan.difficulties for anchor in difficulty.anchors}
    assert quotes == {"dark", "until"}
    assert len(plan.difficulties) == 2
    assert {fact.subject_anchor.quote for fact in plan.context_facts} == {"dark"}
    validated = validate_content_plan(payload, aliases, texts, visible_texts_by_alias=visible)
    assert validated is not None
    assert {anchor.quote for difficulty in validated.difficulties for anchor in difficulty.anchors} == {"dark", "until"}


def _footnote_block(text: str, call: str, *, at: int | None = None):
    """Paragraph whose footnote call is `call`, so the model does not see those characters."""
    from linguary.domain import InlineAnnotation, InlineAnnotationType

    start = text.index(call) if at is None else at
    return replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=text),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=start,
                end=start + len(call),
                target="section-1.xhtml#fn1",
            ),
        ),
    )


def test_repeated_quote_offset_binds_visible_span_then_source_coordinates() -> None:
    """Model start/end address the visible text; the anchor stores the source span."""
    from linguary.application.analysis_merge import content_plan_from_response
    from linguary.application.context_builder import _analysis_visible_origins

    block = _footnote_block("Wait 1 Wait until dark.", "1")
    visible, origins = _analysis_visible_origins(block)
    quote = "Wait"
    visible_start = visible.index(quote, visible.index(quote) + 1)
    visible_end = visible_start + len(quote)
    source_start, source_end = origins[visible_start], origins[visible_end - 1] + 1
    assert visible != block.text
    assert (visible_start, visible_end) != (source_start, source_end)
    assert block.text[source_start:source_end] == quote
    assert visible[visible_start:visible_end] == quote

    aliases = {"b1": block.id}
    texts = {block.id: block.text}
    shown = {"b1": visible}
    plan = content_plan_from_response(
        _quote_payload(_difficulty(quote, start=visible_start, end=visible_end)),
        aliases,
        texts,
        visible_texts_by_alias=shown,
        blocks=(block,),
    )
    assert plan is not None
    anchor = plan.difficulties[0].anchors[0]
    assert (anchor.start, anchor.end, anchor.quote) == (source_start, source_end, quote)

    again = content_plan_from_response(
        _quote_payload(_difficulty(quote, start=anchor.start, end=anchor.end)),
        aliases,
        texts,
        visible_texts_by_alias=shown,
        blocks=(block,),
    )
    assert again is not None
    assert (again.difficulties[0].anchors[0].start, again.difficulties[0].anchors[0].end) == (
        source_start,
        source_end,
    )

    # The second Holmes is the footnote call, so the word is unique in the visible text.
    named_text = "Holmes saw Holmes there."
    named = _footnote_block(named_text, "Holmes", at=named_text.rindex("Holmes"))
    named_visible, named_origins = _analysis_visible_origins(named)
    assert named_visible.count("Holmes") == 1
    assert named.text.count("Holmes") == 2
    body = named_visible.index("Holmes")
    plan = content_plan_from_response(
        _quote_payload(_difficulty("Holmes", difficulty_id="name", point="Who Holmes is")),
        {"b1": named.id},
        {named.id: named.text},
        visible_texts_by_alias={"b1": named_visible},
        blocks=(named,),
    )
    assert plan is not None
    kept = plan.difficulties[0].anchors[0]
    assert (kept.start, kept.end, kept.quote) == (named_origins[body], named_origins[body] + len("Holmes"), "Holmes")


def test_repeated_visual_quote_without_offset_keeps_the_plan() -> None:
    """A repeated visual quote with a missing or wrong offset drops that fact, not the plan."""
    from linguary.application.analysis_merge import content_plan_from_response
    from linguary.application.llm_dto import (
        QuotedSceneResponse,
        QuotedSceneVisualFactResponse,
        QuotedSourceAnchorResponse,
    )

    text = "Wait for him. Wait until dark."
    aliases = {"b1": "block-1"}
    texts = {"block-1": text}
    visible = {"b1": text}
    kept = _difficulty("dark", difficulty_id="kept", point="Explain dark")

    def plan_for(anchor: QuotedSourceAnchorResponse):
        scene = QuotedSceneResponse(
            "b1",
            "b1",
            "He waits.",
            visual_facts=(
                QuotedSceneVisualFactResponse(anchor, "He", "Waiting"),
                QuotedSceneVisualFactResponse(
                    QuotedSourceAnchorResponse("b1", "dark"), "Sky", "Dark",
                ),
            ),
        )
        payload = _quote_payload(kept)
        return content_plan_from_response(
            replace(payload, content_plan=replace(payload.content_plan, scenes=(scene,))),
            aliases,
            texts,
            visible_texts_by_alias=visible,
        )

    plan = plan_for(QuotedSourceAnchorResponse("b1", "Wait"))
    assert plan is not None
    assert {item.anchors[0].quote for item in plan.difficulties} == {"dark"}
    assert len(plan.scenes) == 1
    assert [fact.anchor.quote for fact in plan.scenes[0].visual_facts] == ["dark"]

    shifted = plan_for(QuotedSourceAnchorResponse("b1", "Wait", 1, 5))
    assert shifted is not None
    assert {item.anchors[0].quote for item in shifted.difficulties} == {"dark"}
    assert [fact.anchor.quote for fact in shifted.scenes[0].visual_facts] == ["dark"]


def test_russian_gender_adjective_matches_glossary_code() -> None:
    from types import SimpleNamespace

    from linguary.application.analysis_merge import _extract_gender, validate_fact_glossary

    assert _extract_gender("женский") == "female"
    assert _extract_gender("мужской") == "male"
    assert _extract_gender("средний") == "neuter"
    assert _extract_gender("Женский род; та же героиня") == "female"
    assert _extract_gender("женский пол") == "female"
    assert _extract_gender("мужского рода") == "male"
    assert _extract_gender("женский (она)") == "female"
    assert _extract_gender("город без пометы") is None
    assert _extract_gender("человек среднего роста") is None
    assert _extract_gender("средний возраст") is None
    assert _extract_gender("средняя школа") is None
    assert _extract_gender("мужской монастырь") is None

    def plan(value: str):
        anchor = SimpleNamespace(quote="Mrs Dursley", block_id="b1")
        fact = SimpleNamespace(
            kind="gender", value=value, certainty="explicit", subject_anchor=anchor
        )
        return SimpleNamespace(version=3, context_facts=(fact,))

    validate_fact_glossary(plan("женский"), {"Mrs Dursley": {"gender": "female"}})
    with pytest.raises(AnalysisMergeError) as conflict:
        validate_fact_glossary(plan("мужской"), {"Mrs Dursley": {"gender": "female"}})
    assert conflict.value.violation.rule_code == "glossary_fact_conflict"


def test_everlost_ch2_dialogue_chapter_fits_single_fragment() -> None:
    """A 121-block dialogue chapter (~15K chars) fits into one default fragment."""
    from linguary.application.analysis_merge import (
        DEFAULT_FRAGMENT_BLOCK_OVERHEAD,
        DEFAULT_MAX_FRAGMENT_BLOCKS,
        DEFAULT_MAX_FRAGMENT_WEIGHT,
    )

    assert DEFAULT_MAX_FRAGMENT_BLOCKS == 160
    assert DEFAULT_FRAGMENT_BLOCK_OVERHEAD == 60
    assert DEFAULT_MAX_FRAGMENT_WEIGHT == 26_000

    # 121 blocks * 124 chars = 15,004 chars; weight = 15,004 + 121 * 60 = 22,264 <= 26,000.
    blocks = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=index + (1 if index >= 40 else 0) + (1 if index >= 85 else 0),
            type=BlockType.QUOTE if index in (59, 60) else BlockType.PARAGRAPH,
            text=("«Wake up!» " * 12)[:124],
        )
        for index in range(121)
    )
    fragments = plan_fragments(blocks)
    assert len(fragments) == 1
    assert fragments[0].block_ids == tuple(block.id for block in blocks)


def test_plan_fragments_balances_long_chapter_without_single_block_tail() -> None:
    """Exceeding block, char, or weight limits balances fragments without 1-block tails."""
    from linguary.application.analysis_merge import (
        DEFAULT_FRAGMENT_BLOCK_OVERHEAD,
        DEFAULT_MAX_FRAGMENT_WEIGHT,
    )

    # 121 short blocks with explicit max_blocks=40 -> 4 balanced fragments, no 40+40+40+1 tail.
    dialogue = tuple(
        make_analysis_block(section_id=SECTION_ID, order=i, text="x" * 124)
        for i in range(121)
    )
    by_blocks = plan_fragments(dialogue, max_blocks=40)
    sizes = [len(f.blocks) for f in by_blocks]
    assert len(by_blocks) == 4
    assert min(sizes) >= 30
    assert max(sizes) <= 31
    assert sum(sizes) == 121

    # 145 medium blocks: 145 <= 160 blocks and 21,750 <= 24,000 chars, but weight 30,450 > 26,000.
    medium = tuple(
        make_analysis_block(section_id=SECTION_ID, order=i, text="m" * 150)
        for i in range(145)
    )
    by_weight = plan_fragments(medium)
    assert len(by_weight) == 2
    assert [len(f.blocks) for f in by_weight] == [73, 72]
    for frag in by_weight:
        weight = sum(len(b.text) + DEFAULT_FRAGMENT_BLOCK_OVERHEAD for b in frag.blocks)
        assert weight <= DEFAULT_MAX_FRAGMENT_WEIGHT

    # Oversized block in the middle stays isolated as its own 1-block fragment; sides stay balanced.
    oversized = make_analysis_block(section_id=SECTION_ID, order=50, text="H" * 25_000)
    with_huge = (
        tuple(make_analysis_block(section_id=SECTION_ID, order=i, text="a" * 50) for i in range(50))
        + (oversized,)
        + tuple(make_analysis_block(section_id=SECTION_ID, order=51 + i, text="b" * 50) for i in range(50))
    )
    around_huge = plan_fragments(with_huge, max_blocks=40)
    assert [len(f.blocks) for f in around_huge] == [25, 25, 1, 25, 25]
    assert around_huge[2].blocks == (oversized,)

    # Validation of new parameters and proportional max_weight default.
    with pytest.raises(AnalysisMergeError):
        plan_fragments(dialogue[:2], block_overhead=-1)
    with pytest.raises(AnalysisMergeError):
        plan_fragments(dialogue[:2], max_weight=0)
    scaled = plan_fragments(
        tuple(make_analysis_block(section_id=SECTION_ID, order=i, text="w" * 100) for i in range(20)),
        max_chars=1_000,
    )
    # effective_max_weight = 1_000 + 2_000 = 3_000; 20 * 100 = 2_000 > 1_000 -> 2 fragments [10, 10].
    assert [len(f.blocks) for f in scaled] == [10, 10]


def test_plan_fragments_keeps_multiparagraph_quote_and_respects_explicit_limit() -> None:
    """Soft rules keep consecutive QUOTE blocks and headings intact without violating limits."""
    # 1a. Consecutive QUOTE blocks across naive midpoint (4, 5 of 10) stay together within max_blocks=6.
    quote_blocks = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=i,
            type=BlockType.QUOTE if i in (4, 5) else BlockType.PARAGRAPH,
            text=f"Block {i} content.",
        )
        for i in range(10)
    )
    quote_frags = plan_fragments(quote_blocks, max_blocks=6)
    assert [len(f.blocks) for f in quote_frags] == [6, 4]
    assert all(len(f.blocks) <= 6 for f in quote_frags)
    assert len({frag.order for frag in quote_frags for b in frag.blocks if b.type is BlockType.QUOTE}) == 1

    # 1b. Consecutive QUOTE blocks across greedy boundary (3, 4 of 9 with max_blocks=4) stay together.
    quote_9 = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=i,
            type=BlockType.QUOTE if i in (3, 4) else BlockType.PARAGRAPH,
            text=f"Block {i} content.",
        )
        for i in range(9)
    )
    quote_9_frags = plan_fragments(quote_9, max_blocks=4)
    assert [len(f.blocks) for f in quote_9_frags] == [3, 3, 3]
    assert len({frag.order for frag in quote_9_frags for b in frag.blocks if b.type is BlockType.QUOTE}) == 1

    # 2. Avoid dangling trailing BlockType.HEADING (both on naive midpoint 4 of 10 and greedy boundary 3 of 9).
    heading_blocks = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=i,
            type=BlockType.HEADING if i == 4 else BlockType.PARAGRAPH,
            text=f"Heading {i}" if i == 4 else f"Body {i}.",
        )
        for i in range(10)
    )
    heading_frags = plan_fragments(heading_blocks, max_blocks=6)
    assert len(heading_frags) == 2
    assert heading_frags[0].blocks[-1].type is not BlockType.HEADING
    assert all(len(f.blocks) <= 6 for f in heading_frags)

    heading_9 = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=i,
            type=BlockType.HEADING if i == 3 else BlockType.PARAGRAPH,
            text=f"Heading {i}" if i == 3 else f"Body {i}.",
        )
        for i in range(9)
    )
    heading_9_frags = plan_fragments(heading_9, max_blocks=4)
    assert [len(f.blocks) for f in heading_9_frags] == [3, 3, 3]
    assert all(frag.blocks[-1].type is not BlockType.HEADING for frag in heading_9_frags)

    # 3. Prefer splitting on a block.order gap (scene separator) near the balanced midpoint.
    gapped_10 = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=i + (2 if i >= 4 else 0),
            text=f"Scene block {i}.",
        )
        for i in range(10)
    )
    gapped_frags = plan_fragments(gapped_10, max_blocks=6)
    assert [len(f.blocks) for f in gapped_frags] == [4, 6]
    assert gapped_frags[0].blocks[-1].order == 3
    assert gapped_frags[1].blocks[0].order == 6

    # 4. Explicit max_blocks keeps unconditional priority and does not inflate fragment count.
    tight_quote = tuple(
        make_analysis_block(
            section_id=SECTION_ID,
            order=i,
            type=BlockType.QUOTE if i in (2, 3, 4) else BlockType.PARAGRAPH,
            text=f"Block {i}.",
        )
        for i in range(7)
    )
    tight_frags = plan_fragments(tight_quote, max_blocks=4)
    assert [len(f.blocks) for f in tight_frags] == [4, 3]
    assert all(len(f.blocks) <= 4 for f in tight_frags)


def test_select_staged_a2_summary_keeps_full_chapter_over_short_tail_and_handles_zero_overlap() -> None:
    """A2 summary selector maximizes narrative coverage with script check and predictable fallback."""
    from linguary.application.analysis_merge import select_staged_a2_summary

    narrative_summary = (
        "Nick and Allie wake up in Everlost, meet Lief in the forest, "
        "and walk toward the Twin Towers where Mary lives."
    )

    # 1. Full summary in parts[0] beats short tail in parts[1].
    full_first = make_fragment(
        0,
        blocks=("b1",),
        chapter_summary=(
            "Nick and Allie woke up in Everlost, met Lief in the forest, "
            "and walked to the Twin Towers where Mary lived."
        ),
    )
    short_tail = make_fragment(
        1,
        blocks=("b2",),
        chapter_summary="They walked into the towers and saw Mary.",
    )
    assert select_staged_a2_summary((full_first, short_tail), narrative_summary=narrative_summary) == full_first.chapter_summary

    # 2. Full summary in later fragment beats incomplete earlier summary.
    incomplete_first = make_fragment(
        0,
        blocks=("b1",),
        chapter_summary="Nick and Allie woke up in Everlost.",
    )
    assert select_staged_a2_summary((incomplete_first, full_first), narrative_summary=narrative_summary) == full_first.chapter_summary

    # 3. Wrong script is rejected in favor of matching script with overlap.
    russian_summary = make_fragment(
        0,
        blocks=("b1",),
        chapter_summary="Ник и Элли просыпаются в Эверлосте, встречают Лифа в лесу и идут к башням Мэри.",
    )
    assert select_staged_a2_summary((russian_summary, short_tail), narrative_summary=narrative_summary) == short_tail.chapter_summary

    # 4. Zero overlap falls back to the fullest non-empty candidate (and skips empty ones).
    empty_frag = make_fragment(0, blocks=("b1",), chapter_summary="   ")
    short_no_overlap = make_fragment(1, blocks=("b2",), chapter_summary="Two kids travel far.")
    full_no_overlap = make_fragment(
        2,
        blocks=("b3",),
        chapter_summary="Two lost children explore a strange shadowy world together and reach a tall city building.",
    )
    assert (
        select_staged_a2_summary(
            (empty_frag, short_no_overlap, full_no_overlap),
            narrative_summary="Completely different tokens without shared stems.",
        )
        == full_no_overlap.chapter_summary
    )

    # 5. All candidates mismatch script -> still falls back to fullest non-empty candidate.
    assert (
        select_staged_a2_summary(
            (short_no_overlap, full_no_overlap),
            narrative_summary="Совершенно другой русский текст без совпадений алфавита.",
        )
        == full_no_overlap.chapter_summary
    )

    # 6. Equal coverage or equal fallback fullness breaks ties in favor of parts[0].
    tie_a = make_fragment(0, blocks=("b1",), chapter_summary="Nick and Allie reach Everlost.")
    tie_b = make_fragment(1, blocks=("b2",), chapter_summary="Nick and Allie arrive at Everlost.")
    assert select_staged_a2_summary((tie_a, tie_b), narrative_summary=narrative_summary) == tie_a.chapter_summary

    fallback_tie_a = make_fragment(0, blocks=("b1",), chapter_summary="Alpha beta gamma.")
    fallback_tie_b = make_fragment(1, blocks=("b2",), chapter_summary="Theta kappa zeta.")
    assert (
        select_staged_a2_summary(
            (fallback_tie_a, fallback_tie_b),
            narrative_summary="No overlap words here.",
        )
        == fallback_tie_a.chapter_summary
    )
    assert select_staged_a2_summary((), narrative_summary=narrative_summary) == ""


def test_lenient_quote_pattern_collapses_dash_spaces_and_preserves_intra_word_hyphen() -> None:
    from linguary.application.analysis_merge import _lenient_quote_pattern, find_quote_span

    assert find_quote_span("Wait—he paused.", "Wait — he paused.") == (0, 15, "Wait—he paused.")
    assert find_quote_span("Wait — he paused.", "Wait—he paused.") == (0, 17, "Wait — he paused.")
    assert find_quote_span("Wait–he paused.", "Wait - he paused.") == (0, 15, "Wait–he paused.")
    assert find_quote_span("Wait - he paused.", "Wait–he paused.") == (0, 17, "Wait - he paused.")
    assert find_quote_span("Alpha—-Beta", "Alpha — - Beta") == (0, 11, "Alpha—-Beta")
    assert find_quote_span("Alpha--Beta", "Alpha - - Beta") == (0, 11, "Alpha--Beta")

    intra_word_pattern = _lenient_quote_pattern("dead-spots")
    assert r"\s*" not in intra_word_pattern
    assert find_quote_span("Avoid dead - spots there.", "dead-spots") is None
    assert find_quote_span("Avoid dead — spots there.", "dead-spots") is None
    assert find_quote_span("Avoid dead-spots there.", "dead - spots") == (6, 16, "dead-spots")


def test_validate_content_plan_collects_all_unresolved_quotes_and_global_unknown_alias() -> None:
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import (
        QuotedContentPlanResponse,
        QuotedContextFactResponse,
        QuotedSourceAnchorResponse,
    )

    aliases = {"b1": "block-1", "b2": "block-2"}
    texts = {"block-1": "Holmes stayed.", "block-2": "Watson left."}
    visible = {"b1": "Holmes stayed.", "b2": "Watson left."}

    multi_payload = ChapterAnalysisResponse(
        translation_instructions="Keep.",
        chapter_summary="Summary.",
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.TRANSLATE),
        ),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                _difficulty("missing first", difficulty_id="d1", owner="b1"),
                _difficulty("missing unknown", difficulty_id="d2", owner="b9"),
            ),
            context_facts=(
                QuotedContextFactResponse(
                    "f1",
                    "speaker",
                    (QuotedSourceAnchorResponse("b2", "missing fact"),),
                    QuotedSourceAnchorResponse("b2", "missing fact"),
                    "Watson",
                    "explicit",
                ),
            ),
        ),
    )
    with pytest.raises(AnalysisMergeError) as multi_exc:
        validate_content_plan(multi_payload, aliases, texts, visible_texts_by_alias=visible)
    assert multi_exc.value.violation is not None
    assert multi_exc.value.violation.rule_code == "unresolved_quote"
    assert multi_exc.value.violation.scope == "local"
    assert multi_exc.value.violation.block_ids == ("block-1", "block-2")
    msg = str(multi_exc.value)
    assert "missing first" in msg
    assert "missing unknown" in msg
    assert "b9" in msg
    assert "missing fact" in msg

    unknown_only_payload = ChapterAnalysisResponse(
        translation_instructions="Keep.",
        chapter_summary="Summary.",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),),
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(_difficulty("missing unknown", difficulty_id="d1", owner="b9"),),
        ),
    )
    with pytest.raises(AnalysisMergeError) as unknown_exc:
        validate_content_plan(unknown_only_payload, aliases, texts, visible_texts_by_alias=visible)
    assert unknown_exc.value.violation is not None
    assert unknown_exc.value.violation.rule_code == "unresolved_quote"
    assert unknown_exc.value.violation.stage == "chapter_analysis"
    assert unknown_exc.value.violation.scope == "global"
    assert unknown_exc.value.violation.block_ids == ()


def test_validate_content_plan_hints_parent_block_and_targets_local_repair_for_segment_owner() -> None:
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import QuotedContentPlanResponse, QuotedSourceAnchorResponse
    from linguary.application.retry_policy import (
        ExpectedBlock,
        ResultExpectation,
        ResultValidationError,
        RetryPolicy,
    )

    aliases = {f"b{i}": f"block-{i}" for i in range(1, 122)}
    texts = {f"block-{i}": f"Block {i} text. Nick paused." for i in range(1, 122)}
    visible = {alias: texts[block_id] for alias, block_id in aliases.items()}
    treatment_plan = tuple(
        TreatmentPlanItemResponse(f"b{i}", BlockTreatment.TRANSLATE) for i in range(1, 122)
    )

    for bad_owner in ("b120.1", "B120.1"):
        payload = ChapterAnalysisResponse(
            translation_instructions="Keep.",
            chapter_summary="Summary.",
            treatment_plan=treatment_plan,
            content_plan=QuotedContentPlanResponse(
                version=3,
                profile="content-plan-v3",
                difficulties=(
                    _difficulty("Nick paused.", difficulty_id="d1", owner=bad_owner),
                ),
            ),
        )
        with pytest.raises(AnalysisMergeError) as exc:
            validate_content_plan(payload, aliases, texts, visible_texts_by_alias=visible)

        violation = exc.value.violation
        assert violation is not None
        assert violation.rule_code == "unresolved_quote"
        assert violation.scope == "local"
        assert violation.block_ids == ("block-120",)
        msg = str(exc.value)
        assert f"неизвестный блок-владелец трудности: используй идентификатор целого блока b120, а не сегмента {bad_owner}" in msg

    # Verify owner_block_id="b120.1" with valid whole-block anchor "b120" still attaches block-120 locally.
    owner_only_payload = ChapterAnalysisResponse(
        translation_instructions="Keep.",
        chapter_summary="Summary.",
        treatment_plan=treatment_plan,
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                replace(
                    _difficulty("Nick paused.", difficulty_id="d1", owner="b120"),
                    owner_block_id="b120.1",
                    anchors=(QuotedSourceAnchorResponse("b120", "Nick paused."),),
                ),
            ),
        ),
    )
    with pytest.raises(AnalysisMergeError) as owner_exc:
        validate_content_plan(owner_only_payload, aliases, texts, visible_texts_by_alias=visible)
    assert owner_exc.value.violation is not None
    assert owner_exc.value.violation.scope == "local"
    assert owner_exc.value.violation.block_ids == ("block-120",)

    # Verify RetryPolicy builds a targeted local repair (- b120) instead of b1..b121.
    aliased_violation = replace(owner_exc.value.violation, block_ids=("b120",))
    err = ResultValidationError(
        str(owner_exc.value),
        expectation=ResultExpectation(
            target_language="ru",
            blocks=tuple(ExpectedBlock(block_id=f"b{i}") for i in range(1, 122)),
        ),
        violation=aliased_violation,
    )
    decision = RetryPolicy(max_attempts=3).decide(err, attempts=1, raw_content="{}")
    assert decision.repair_instruction is not None
    assert "- b120" in decision.repair_instruction
    assert "b1..b121" not in decision.repair_instruction
    assert "- b1\n" not in decision.repair_instruction
    assert "- b121" not in decision.repair_instruction


def test_validate_content_plan_hints_parent_block_for_segment_anchor_and_keeps_unknown_global() -> None:
    from linguary.application.analysis_merge import _diagnose_quoted_anchor, validate_content_plan
    from linguary.application.llm_dto import QuotedContentPlanResponse, QuotedSourceAnchorResponse

    aliases = {f"b{i}": f"block-{i}" for i in range(1, 122)}
    texts = {f"block-{i}": f"Block {i} text. Nick paused. Allie waited." for i in range(1, 122)}
    visible = {alias: texts[block_id] for alias, block_id in aliases.items()}
    aliases_by_id = {block_id: alias for alias, block_id in aliases.items()}
    treatment_plan = tuple(
        TreatmentPlanItemResponse(f"b{i}", BlockTreatment.TRANSLATE) for i in range(1, 122)
    )

    assert _diagnose_quoted_anchor("b120.2", "Allie waited.", aliases, texts, visible, aliases_by_id) == (
        "b120.2",
        "Allie waited.",
        "указана метка сегмента b120.2 — используй идентификатор целого блока b120",
        ("block-120",),
    )
    assert _diagnose_quoted_anchor("B120.1", "Nick paused.", aliases, texts, visible, aliases_by_id) == (
        "B120.1",
        "Nick paused.",
        "указана метка сегмента B120.1 — используй идентификатор целого блока b120",
        ("block-120",),
    )
    assert _diagnose_quoted_anchor("b999.1", "Nick paused.", aliases, texts, visible, aliases_by_id) == (
        "b999.1",
        "Nick paused.",
        "неизвестный блок",
        (),
    )

    bad_anchor_payload = ChapterAnalysisResponse(
        translation_instructions="Keep.",
        chapter_summary="Summary.",
        treatment_plan=treatment_plan,
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                replace(
                    _difficulty("Nick paused.", difficulty_id="d1", owner="b120"),
                    anchors=(
                        QuotedSourceAnchorResponse("b120", "Nick paused."),
                        QuotedSourceAnchorResponse("b120.2", "Allie waited."),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(AnalysisMergeError) as exc:
        validate_content_plan(bad_anchor_payload, aliases, texts, visible_texts_by_alias=visible)

    assert exc.value.violation is not None
    assert exc.value.violation.rule_code == "unresolved_quote"
    assert exc.value.violation.scope == "local"
    assert exc.value.violation.block_ids == ("block-120",)
    assert "указана метка сегмента b120.2 — используй идентификатор целого блока b120" in str(exc.value)

    unknown_parent_payload = ChapterAnalysisResponse(
        translation_instructions="Keep.",
        chapter_summary="Summary.",
        treatment_plan=treatment_plan,
        content_plan=QuotedContentPlanResponse(
            version=3,
            profile="content-plan-v3",
            difficulties=(
                _difficulty("Nick paused.", difficulty_id="d1", owner="b999.1"),
            ),
        ),
    )
    with pytest.raises(AnalysisMergeError) as unknown_exc:
        validate_content_plan(unknown_parent_payload, aliases, texts, visible_texts_by_alias=visible)

    assert unknown_exc.value.violation is not None
    assert unknown_exc.value.violation.rule_code == "unresolved_quote"
    assert unknown_exc.value.violation.scope == "global"
    assert unknown_exc.value.violation.block_ids == ()
    assert "используй идентификатор целого блока" not in str(unknown_exc.value)


def test_plan_item_from_response_wraps_all_segment_errors_as_local_violation() -> None:
    from linguary.application.analysis_merge import _plan_item_from_response
    from linguary.application.llm_dto import SegmentTreatmentResponse
    from linguary.domain import BlockSegment

    single_block = make_analysis_block(
        section_id=SECTION_ID, order=0, text="Single sentence."
    )
    two_seg_text = "First sentence. Second sentence."
    split_at = len("First sentence. ")
    two_seg_block = replace(
        make_analysis_block(section_id=SECTION_ID, order=1, text=two_seg_text),
        segments=(
            BlockSegment(0, split_at, 0),
            BlockSegment(split_at, len(two_seg_text), 1),
        ),
    )
    same_sentence_block = replace(
        make_analysis_block(section_id=SECTION_ID, order=2, text=two_seg_text),
        segments=(
            BlockSegment(0, split_at, 0),
            BlockSegment(split_at, len(two_seg_text), 0),
        ),
    )

    cases = [
        # 1. block is None
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                ),
            ),
            "block-missing",
            None,
        ),
        # 2. single-segment block
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),),
            ),
            single_block.id,
            single_block,
        ),
        # 3. out-of-range segment index
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(3, BlockTreatment.READ_AS_IS),
                ),
            ),
            two_seg_block.id,
            two_seg_block,
        ),
        # 4. duplicate segment index
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
                ),
            ),
            two_seg_block.id,
            two_seg_block,
        ),
        # 5. incomplete segment coverage
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),),
            ),
            two_seg_block.id,
            two_seg_block,
        ),
        # 6a. TreatmentPlanError: block treatment weaker than strongest segment
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.GLOSSARY_ONLY,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(2, BlockTreatment.GLOSSARY_ONLY),
                ),
            ),
            two_seg_block.id,
            two_seg_block,
        ),
        # 6b. TreatmentPlanError: different modes within the same sentence_index
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                ),
            ),
            same_sentence_block.id,
            same_sentence_block,
        ),
        # 6c. TreatmentPlanError: block treatment stronger than strongest segment
        (
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
                    SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                ),
            ),
            two_seg_block.id,
            two_seg_block,
        ),
    ]

    for item, block_id, block in cases:
        with pytest.raises(AnalysisMergeError) as exc:
            _plan_item_from_response(item, block_id=block_id, block=block)
        assert exc.value.violation is not None
        assert exc.value.violation.rule_code == "invalid_segment_treatment"
        assert exc.value.violation.scope == "local"
        assert exc.value.violation.block_ids == (block_id,)


def test_validate_content_plan_resolves_lexical_hint_and_notes_by_overlapping_segment_treatments() -> None:
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import (
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
        SegmentTreatmentResponse,
    )
    from linguary.domain import BlockSegment
    from linguary.domain.content_plan import RequiredPoint

    s1 = "He would have gone home. "
    s2 = "She muttered quietly. "
    s3 = "They waited outside."
    text = s1 + s2 + s3
    b = replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=text),
        segments=(
            BlockSegment(0, len(s1), 0),
            BlockSegment(len(s1), len(s1) + len(s2), 1),
            BlockSegment(len(s1) + len(s2), len(text), 2),
        ),
    )
    aliases = {"b1": b.id}
    texts = {b.id: text}
    visible = {"b1": text}
    point = (RequiredPoint("p1", "Объяснить значение."),)
    mixed_item = TreatmentPlanItemResponse(
        "b1",
        BlockTreatment.TRANSLATE,
        (
            SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
            SegmentTreatmentResponse(2, BlockTreatment.GLOSSARY_ONLY),
            SegmentTreatmentResponse(3, BlockTreatment.READ_AS_IS),
        ),
    )

    def make_payload(difficulties, item=mixed_item) -> ChapterAnalysisResponse:
        return ChapterAnalysisResponse(
            translation_instructions="Keep voice.",
            chapter_summary="Summary.",
            treatment_plan=(item,),
            content_plan=QuotedContentPlanResponse(
                version=3,
                profile="content-plan-v3",
                difficulties=difficulties,
            ),
        )

    # 1. lexical_hint on GLOSSARY_ONLY segment succeeds even when block treatment is TRANSLATE.
    hint_on_glossary = QuotedDifficultyResponse(
        "d-hint-ok",
        (QuotedSourceAnchorResponse("b1", "muttered"),),
        "b1",
        "vocabulary",
        "useful",
        "lexical_hint",
        point,
    )
    plan = validate_content_plan(
        make_payload((hint_on_glossary,)),
        aliases,
        texts,
        visible_texts_by_alias=visible,
        blocks=(b,),
        language_notes_per_chapter=0,
    )
    assert plan is not None
    assert plan.difficulties[0].help_kind == "lexical_hint"

    # 2. lexical_hint on TRANSLATE segment or spanning TRANSLATE + GLOSSARY_ONLY is rejected.
    for bad_quote in ("would have gone", "home. She muttered"):
        hint_bad = QuotedDifficultyResponse(
            "d-hint-bad",
            (QuotedSourceAnchorResponse("b1", bad_quote),),
            "b1",
            "vocabulary",
            "useful",
            "lexical_hint",
            point,
        )
        with pytest.raises(AnalysisMergeError) as hint_exc:
            validate_content_plan(
                make_payload((hint_bad,)),
                aliases,
                texts,
                visible_texts_by_alias=visible,
                blocks=(b,),
            )
        assert hint_exc.value.violation is not None
        assert hint_exc.value.violation.rule_code == "lexical_help_channel"
        assert hint_exc.value.violation.block_ids == (b.id,)

    # 3. explanation on TRANSLATE segment only does NOT count in note_owners (fits budget 0).
    expl_on_translate = QuotedDifficultyResponse(
        "d-expl-tr",
        (QuotedSourceAnchorResponse("b1", "would have gone"),),
        "b1",
        "grammar",
        "critical",
        "explanation",
        point,
    )
    assert validate_content_plan(
        make_payload((expl_on_translate,)),
        aliases,
        texts,
        visible_texts_by_alias=visible,
        blocks=(b,),
        language_notes_per_chapter=0,
        note_limits_by_block={b.id: 0},
    ) is not None

    # 4. explanation overlapping READ_AS_IS segment DOES count in note_owners -> exceeds budget 0.
    expl_on_read_as_is = QuotedDifficultyResponse(
        "d-expl-rai",
        (QuotedSourceAnchorResponse("b1", "waited outside"),),
        "b1",
        "vocabulary",
        "useful",
        "explanation",
        point,
    )
    with pytest.raises(AnalysisMergeError) as note_exc:
        validate_content_plan(
            make_payload((expl_on_read_as_is,)),
            aliases,
            texts,
            visible_texts_by_alias=visible,
            blocks=(b,),
            language_notes_per_chapter=0,
            note_limits_by_block={b.id: 0},
        )
    assert note_exc.value.violation is not None
    assert note_exc.value.violation.rule_code == "note_chapter_budget"
    assert note_exc.value.violation.block_ids == (b.id,)

    # 5. validate_content_plan validates segments upfront when blocks are passed.
    bad_segments_item = TreatmentPlanItemResponse(
        "b1",
        BlockTreatment.GLOSSARY_ONLY,
        (
            SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
            SegmentTreatmentResponse(2, BlockTreatment.GLOSSARY_ONLY),
            SegmentTreatmentResponse(3, BlockTreatment.READ_AS_IS),
        ),
    )
    with pytest.raises(AnalysisMergeError) as seg_exc:
        validate_content_plan(
            make_payload((), item=bad_segments_item),
            aliases,
            texts,
            visible_texts_by_alias=visible,
            blocks=(b,),
        )
    assert seg_exc.value.violation is not None
    assert seg_exc.value.violation.rule_code == "invalid_segment_treatment"
    assert seg_exc.value.violation.scope == "local"
    assert seg_exc.value.violation.block_ids == (b.id,)


def test_normalize_a2_lifts_stoplist_in_read_as_is_segments_and_unbacked_glossary_only() -> None:
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
        SegmentTreatmentResponse,
    )
    from linguary.domain import BlockSegment
    from linguary.domain.content_plan import RequiredPoint

    point = (RequiredPoint("p1", "Объяснить конструкцию."),)

    # 1. Segmented READ_AS_IS block: simple seg stays READ_AS_IS, single reporting verb -> GLOSSARY_ONLY,
    #    stoplist -> TRANSLATE; also single vocab seg with vocab difficulty -> GLOSSARY_ONLY.
    seg_s1 = "He sat down. "
    seg_s2 = "It winked. "
    seg_s3 = "What did you do that for? "
    seg_s4 = "She felt nervous."
    seg_text = seg_s1 + seg_s2 + seg_s3 + seg_s4
    seg_block = replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=seg_text),
        segments=(
            BlockSegment(0, len(seg_s1), 0),
            BlockSegment(len(seg_s1), len(seg_s1) + len(seg_s2), 1),
            BlockSegment(len(seg_s1) + len(seg_s2), len(seg_s1) + len(seg_s2) + len(seg_s3), 2),
            BlockSegment(len(seg_s1) + len(seg_s2) + len(seg_s3), len(seg_text), 3),
        ),
    )

    # 2. Whole READ_AS_IS blocks covering each A2 stoplist detector + negative controls.
    stoplist_read_as_is = (
        "He has been here.",              # Present Perfect (been)
        "We have seen it before.",        # Present Perfect (participle)
        "She would stay.",                # would
        "I know what you mean.",          # embedded what-clause
        "Oh, great!",                     # sarcasm
        "I don't got time.",              # nonstandard don't got
        "He ain't leaving.",              # nonstandard ain't
        "Her face went red.",             # state-change idiom went red
        "It made him uneasy.",            # state-change idiom made ... uneasy
        "She felt uneasy.",               # uneasy in _SHORT_B1_WORDS (no vocab diff -> TRANSLATE)
    )
    plain_read_as_is = (
        "I have a cat.",                  # possessive have -> stays READ_AS_IS
        "We have to go now.",             # have to -> stays READ_AS_IS
        "What did you say?",              # direct Wh-question -> stays READ_AS_IS
        "‘A what?’",                      # short question -> stays READ_AS_IS
    )
    # 3. GLOSSARY_ONLY blocks with nonlexical stoplist: unbacked -> TRANSLATE, backed -> stays GLOSSARY_ONLY.
    glossary_unbacked = (
        "He has gone home.",              # Present Perfect without grammar diff
        "Yeah, right.",                   # sarcasm without idiom/speech_style diff
        "He ain't here.",                 # nonstandard without speech_style diff
        "It made Nick uneasy.",           # state-change idiom with only vocabulary diff
    )
    glossary_backed = (
        "What is it for?",                # backed by grammar difficulty
        "I don't got it.",                # backed by speech_style difficulty
        "Her face went red.",             # backed by idiom difficulty
    )

    all_blocks = (
        seg_block,
        *(
            make_analysis_block(section_id=SECTION_ID, order=i + 1, text=t)
            for i, t in enumerate(stoplist_read_as_is)
        ),
        *(
            make_analysis_block(section_id=SECTION_ID, order=20 + i, text=t)
            for i, t in enumerate(plain_read_as_is)
        ),
        *(
            make_analysis_block(section_id=SECTION_ID, order=40 + i, text=t)
            for i, t in enumerate(glossary_unbacked)
        ),
        *(
            make_analysis_block(section_id=SECTION_ID, order=60 + i, text=t)
            for i, t in enumerate(glossary_backed)
        ),
    )
    aliases = {f"b{i + 1}": b.id for i, b in enumerate(all_blocks)}

    n_stop = len(stoplist_read_as_is)
    n_plain = len(plain_read_as_is)
    n_unbacked = len(glossary_unbacked)
    n_backed = len(glossary_backed)

    plan_items = [
        TreatmentPlanItemResponse(
            "b1",
            BlockTreatment.READ_AS_IS,
            (
                SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
                SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                SegmentTreatmentResponse(3, BlockTreatment.READ_AS_IS),
                SegmentTreatmentResponse(4, BlockTreatment.READ_AS_IS),
            ),
        )
    ]
    for i in range(n_stop + n_plain):
        plan_items.append(TreatmentPlanItemResponse(f"b{2 + i}", BlockTreatment.READ_AS_IS))
    for i in range(n_unbacked + n_backed):
        plan_items.append(
            TreatmentPlanItemResponse(
                f"b{2 + n_stop + n_plain + i}", BlockTreatment.GLOSSARY_ONLY
            )
        )

    unbacked_start = 2 + n_stop + n_plain
    backed_start = unbacked_start + n_unbacked
    difficulties = (
        # Vocabulary difficulty on seg 4 ("nervous") -> seg 4 lifts to GLOSSARY_ONLY
        QuotedDifficultyResponse(
            "d-nervous",
            (QuotedSourceAnchorResponse("b1", "nervous"),),
            "b1",
            "vocabulary",
            "useful",
            "lexical_hint",
            point,
        ),
        # Only vocabulary difficulty on "It made Nick uneasy." -> still unbacked for idiom -> TRANSLATE
        QuotedDifficultyResponse(
            "d-unbacked-vocab",
            (QuotedSourceAnchorResponse(f"b{unbacked_start + 3}", "uneasy"),),
            f"b{unbacked_start + 3}",
            "vocabulary",
            "useful",
            "lexical_hint",
            point,
        ),
        # Backed GLOSSARY_ONLY difficulties (grammar, speech_style, idiom)
        QuotedDifficultyResponse(
            "d-backed-gram",
            (QuotedSourceAnchorResponse(f"b{backed_start}", "What is it for?"),),
            f"b{backed_start}",
            "grammar",
            "critical",
            "explanation",
            point,
        ),
        QuotedDifficultyResponse(
            "d-backed-speech",
            (QuotedSourceAnchorResponse(f"b{backed_start + 1}", "don't got"),),
            f"b{backed_start + 1}",
            "speech_style",
            "useful",
            "explanation",
            point,
        ),
        QuotedDifficultyResponse(
            "d-backed-idiom",
            (QuotedSourceAnchorResponse(f"b{backed_start + 2}", "went red"),),
            f"b{backed_start + 2}",
            "idiom",
            "useful",
            "explanation",
            point,
        ),
    )

    normalized = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=tuple(plan_items),
            difficulties=difficulties,
        ),
        all_blocks,
        aliases,
        reader_level="a2",
        english_help=False,
    )

    # Segmented READ_AS_IS block: (READ_AS_IS, GLOSSARY_ONLY, TRANSLATE, GLOSSARY_ONLY)
    assert normalized.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert [s.treatment for s in normalized.treatment_plan[0].segments] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
    ]
    # Stoplist READ_AS_IS blocks -> all TRANSLATE
    assert [
        item.treatment for item in normalized.treatment_plan[1 : 1 + n_stop]
    ] == [BlockTreatment.TRANSLATE] * n_stop
    # Negative controls -> stay READ_AS_IS
    assert [
        item.treatment
        for item in normalized.treatment_plan[1 + n_stop : 1 + n_stop + n_plain]
    ] == [BlockTreatment.READ_AS_IS] * n_plain
    # Unbacked GLOSSARY_ONLY -> all TRANSLATE
    assert [
        item.treatment
        for item in normalized.treatment_plan[
            1 + n_stop + n_plain : 1 + n_stop + n_plain + n_unbacked
        ]
    ] == [BlockTreatment.TRANSLATE] * n_unbacked
    # Backed GLOSSARY_ONLY -> stay GLOSSARY_ONLY
    assert [
        item.treatment
        for item in normalized.treatment_plan[1 + n_stop + n_plain + n_unbacked :]
    ] == [BlockTreatment.GLOSSARY_ONLY] * n_backed
    # Unbacked vocab hint on promoted whole GLOSSARY_ONLY block became explanation
    help_by_id = {d.id: d.help_kind for d in normalized.difficulties}
    assert help_by_id["d-unbacked-vocab"] == "explanation"


def test_normalize_a2_preserves_lexical_hint_on_untouched_glossary_segment_and_guards_demotion() -> None:
    from linguary.application.analysis_merge import (
        normalize_a2_dialogue_response,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
        SegmentTreatmentResponse,
    )
    from linguary.domain import BlockSegment
    from linguary.domain.content_plan import RequiredPoint

    point = (RequiredPoint("p1", "Объяснить слово."),)

    # Case 1: Already-TRANSLATE block (TRANSLATE, GLOSSARY_ONLY, READ_AS_IS -> seg 3 lifted to TRANSLATE).
    # lexical_hint on untouched seg 2 MUST stay lexical_hint; hint on promoted seg 3 becomes explanation.
    s1 = "He turned away. "
    s2 = "She felt nervous. "
    s3 = "He has gone home."
    text1 = s1 + s2 + s3
    b1 = replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=text1),
        segments=(
            BlockSegment(0, len(s1), 0),
            BlockSegment(len(s1), len(s1) + len(s2), 1),
            BlockSegment(len(s1) + len(s2), len(text1), 2),
        ),
    )

    # Case 2: GLOSSARY_ONLY block (GLOSSARY_ONLY, READ_AS_IS -> seg 2 lifted only to GLOSSARY_ONLY).
    # lexical_hint on untouched seg 1 MUST stay lexical_hint.
    t1 = "She felt nervous. "
    t2 = "It winked."
    text2 = t1 + t2
    b2 = replace(
        make_analysis_block(section_id=SECTION_ID, order=1, text=text2),
        segments=(
            BlockSegment(0, len(t1), 0),
            BlockSegment(len(t1), len(text2), 1),
        ),
    )

    # Case 3: Demotion guard — short TRANSLATE replies hitting stoplist stay TRANSLATE; plain reply demotes.
    guarded_texts = (
        "He has gone.",
        "She would go.",
        "What is it for?",
        "I know what it is.",
        "Oh, great!",
        "He ain't here.",
        "Her face went red.",
        "She felt uneasy.",
        "He was nervous.",
    )
    guarded_blocks = tuple(
        make_analysis_block(section_id=SECTION_ID, order=10 + i, text=t)
        for i, t in enumerate(guarded_texts)
    )
    plain_reply = make_analysis_block(
        section_id=SECTION_ID, order=30, text="‘Yes,’ he said."
    )

    blocks = (b1, b2, *guarded_blocks, plain_reply)
    aliases = {f"b{i + 1}": b.id for i, b in enumerate(blocks)}

    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(2, BlockTreatment.GLOSSARY_ONLY),
                    SegmentTreatmentResponse(3, BlockTreatment.GLOSSARY_ONLY),
                ),
            ),
            TreatmentPlanItemResponse(
                "b2",
                BlockTreatment.GLOSSARY_ONLY,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.GLOSSARY_ONLY),
                    SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                ),
            ),
            *(
                TreatmentPlanItemResponse(f"b{3 + i}", BlockTreatment.TRANSLATE)
                for i in range(len(guarded_blocks))
            ),
            TreatmentPlanItemResponse(
                f"b{3 + len(guarded_blocks)}", BlockTreatment.TRANSLATE
            ),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                "d-b1-untouched",
                (QuotedSourceAnchorResponse("b1", "nervous"),),
                "b1",
                "vocabulary",
                "useful",
                "lexical_hint",
                point,
            ),
            QuotedDifficultyResponse(
                "d-b1-promoted",
                (QuotedSourceAnchorResponse("b1", "gone"),),
                "b1",
                "vocabulary",
                "useful",
                "lexical_hint",
                point,
            ),
            QuotedDifficultyResponse(
                "d-b2-untouched",
                (QuotedSourceAnchorResponse("b2", "nervous"),),
                "b2",
                "vocabulary",
                "useful",
                "lexical_hint",
                point,
            ),
        ),
    )

    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level="a2", english_help=False
    )

    # b1: seg 3 lifted to TRANSLATE; untouched seg 2 keeps lexical_hint, promoted seg 3 becomes explanation
    assert [s.treatment for s in normalized.treatment_plan[0].segments] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
    ]
    # b2: seg 2 lifted only to GLOSSARY_ONLY; untouched seg 1 keeps lexical_hint
    assert normalized.treatment_plan[1].treatment is BlockTreatment.GLOSSARY_ONLY
    assert [s.treatment for s in normalized.treatment_plan[1].segments] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.GLOSSARY_ONLY,
    ]
    help_by_id = {d.id: d.help_kind for d in normalized.difficulties}
    assert help_by_id == {
        "d-b1-untouched": "lexical_hint",
        "d-b1-promoted": "explanation",
        "d-b2-untouched": "lexical_hint",
        "auto-vocab-b2-1": "lexical_hint",
    }

    # Demotion guard: all guarded short replies keep TRANSLATE; plain reply demotes to READ_AS_IS
    assert [
        item.treatment
        for item in normalized.treatment_plan[2 : 2 + len(guarded_blocks)]
    ] == [BlockTreatment.TRANSLATE] * len(guarded_blocks)
    assert normalized.treatment_plan[-1].treatment is BlockTreatment.READ_AS_IS

    # Combined validation with P5.T1 accepts the preserved lexical_hint on b1 seg 2.
    combined = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=normalized.treatment_plan,
        content_plan=QuotedContentPlanResponse(difficulties=normalized.difficulties),
    )
    texts = {b.id: b.text for b in blocks}
    visible = {alias: texts[bid] for alias, bid in aliases.items()}
    validated = validate_content_plan(
        combined, aliases, texts, visible_texts_by_alias=visible, blocks=blocks
    )
    assert validated is not None


def test_normalize_a2_unsegmented_multisentence_read_as_is_checks_vocab_per_sentence() -> None:
    """Unsegmented multi-sentence READ_AS_IS must not let a vocab difficulty in sentence 2 mask an unbacked B1 word in sentence 1."""
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain import BlockSegment
    from linguary.domain.content_plan import RequiredPoint

    point = (RequiredPoint("p1", "Объяснить слово."),)

    s1 = "She felt nervous. "
    s2 = "He looked gloomy."
    text = s1 + s2
    b_unbacked_s1 = replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=text),
        segments=(
            BlockSegment(0, len(s1), 0),
            BlockSegment(len(s1), len(text), 1),
        ),
    )
    b_backed_s1 = replace(
        make_analysis_block(section_id=SECTION_ID, order=1, text=text),
        segments=(
            BlockSegment(0, len(s1), 0),
            BlockSegment(len(s1), len(text), 1),
        ),
    )

    blocks = (b_unbacked_s1, b_backed_s1)
    aliases = {"b1": b_unbacked_s1.id, "b2": b_backed_s1.id}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
        ),
        difficulties=(
            # b1: vocabulary difficulty is ONLY on sentence 2 ("gloomy"), leaving sentence 1 ("nervous") unbacked.
            QuotedDifficultyResponse(
                "d-b1-s2",
                (QuotedSourceAnchorResponse("b1", "gloomy"),),
                "b1",
                "vocabulary",
                "useful",
                "lexical_hint",
                point,
            ),
            # b2: vocabulary difficulties cover BOTH sentence 1 ("nervous") and sentence 2 ("gloomy").
            QuotedDifficultyResponse(
                "d-b2-s1",
                (QuotedSourceAnchorResponse("b2", "nervous"),),
                "b2",
                "vocabulary",
                "useful",
                "lexical_hint",
                point,
            ),
            QuotedDifficultyResponse(
                "d-b2-s2",
                (QuotedSourceAnchorResponse("b2", "gloomy"),),
                "b2",
                "vocabulary",
                "useful",
                "lexical_hint",
                point,
            ),
        ),
    )

    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level="a2", english_help=False
    )
    assert normalized.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert normalized.treatment_plan[1].treatment is BlockTreatment.GLOSSARY_ONLY


def test_normalize_a2_present_perfect_catches_inverted_proper_nouns_and_safe_used() -> None:
    """_A2_PRESENT_PERFECT_RE catches inverted proper-noun questions and safe `used` while avoiding `have used cars`."""
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import PedagogicalAnalysisResponse

    positive_texts = (
        "Has Harry seen the owl?",
        "Have Nick and Allie been here?",
        "Hasn't Mary arrived yet?",
        "I have used it.",
        "She has used this wand.",
        "We've used the key.",
        "He's used my map.",
        "Have you used a wand?",
        "Has Harry used it?",
        "They have used her boat.",
    )
    negative_texts = (
        "They have used cars.",
        "We have used books on the shelf.",
        "I have something hidden.",
        "He's used to the dark.",
    )
    all_texts = (*positive_texts, *negative_texts)
    blocks = tuple(
        make_analysis_block(section_id=SECTION_ID, order=i, text=t)
        for i, t in enumerate(all_texts)
    )
    aliases = {f"b{i + 1}": b.id for i, b in enumerate(blocks)}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(f"b{i + 1}", BlockTreatment.READ_AS_IS)
            for i in range(len(blocks))
        ),
    )

    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level="a2", english_help=False
    )
    n_pos = len(positive_texts)
    assert [
        (all_texts[i], normalized.treatment_plan[i].treatment)
        for i in range(n_pos)
    ] == [(t, BlockTreatment.TRANSLATE) for t in positive_texts]
    assert [
        (all_texts[n_pos + i], normalized.treatment_plan[n_pos + i].treatment)
        for i in range(len(negative_texts))
    ] == [(t, BlockTreatment.READ_AS_IS) for t in negative_texts]


def test_kept_candidate_description_supports_all_five_concept_types() -> None:
    from linguary.application.analysis_merge import kept_candidate_description

    expected = block_ids(2)
    for concept_type in (
        ConceptType.PERSON,
        ConceptType.PLACE,
        ConceptType.ORGANIZATION,
        ConceptType.FICTIONAL_CONCEPT,
        ConceptType.TERM,
    ):
        low = GlossaryCandidateResponse(
            source_text=f"Entity-{concept_type.value}",
            type=concept_type,
            suggested_translation="Перевод",
            confidence=0.6,
            model_flagged_review=False,
            description="Раннее короткое описание.",
        )
        high = GlossaryCandidateResponse(
            source_text=f"Entity-{concept_type.value}",
            type=concept_type,
            suggested_translation="Перевод",
            confidence=0.95,
            model_flagged_review=False,
            description="Точное описание понятия.",
        )
        empty_higher = GlossaryCandidateResponse(
            source_text=f"Entity-{concept_type.value}",
            type=concept_type,
            suggested_translation="Перевод",
            confidence=0.99,
            model_flagged_review=False,
            description="   ",
        )
        assert kept_candidate_description(concept_type, (low, high, empty_higher)) == "Точное описание понятия."

        merged = merge(
            make_fragment(0, blocks=expected[:1], candidates=(low,)),
            make_fragment(1, blocks=expected[1:], candidates=(high,)),
            expected=expected,
        )
        assert merged.glossary_candidates[0].description == "Точное описание понятия."


def test_merged_instructions_splits_long_solid_paragraphs_guards_quotes_abbrevs_and_filters_fragment_artifacts() -> None:
    expected = block_ids(2)

    # 1. Solid paragraph with >= 3 sentences splits into numbered points and deduplicates across fragments.
    solid_f0 = (
        "Сохранять разговорный ритм диалогов без канцелярита. "
        "Передавать реплики «Wait. Stop.» и \"No. Never.\" без сглаживания. "
        "Учитывать сокращения т. д., т. п., т. е., др., ед. ч., мн. ч., см., стр., 1998 г., "
        "а также обращения Mr. Smith, Mrs. Brown, Ms. Green, Dr. House, St. Peter и инициалы J. K. Rowling."
    )
    solid_f1 = (
        "Сохранять разговорный ритм диалогов без канцелярита. "
        "Передавать реплики «Wait. Stop.» и \"No. Never.\" без сглаживания. "
        "Держать напряжённый темп в описаниях."
    )
    merged_solid = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions=solid_f0),
        make_fragment(1, blocks=expected[1:], translation_instructions=solid_f1),
        expected=expected,
    )
    solid_lines = merged_solid.translation_instructions.splitlines()
    assert solid_lines == [
        "1. Сохранять разговорный ритм диалогов без канцелярита.",
        "2. Передавать реплики «Wait. Stop.» и \"No. Never.\" без сглаживания.",
        (
            "3. Учитывать сокращения т. д., т. п., т. е., др., ед. ч., мн. ч., см., стр., 1998 г., "
            "а также обращения Mr. Smith, Mrs. Brown, Ms. Green, Dr. House, St. Peter и инициалы J. K. Rowling."
        ),
        "4. Держать напряжённый темп в описаниях.",
    ]

    # 2. Curly quotes “...” are also protected from splitting.
    curly_f0 = (
        "Сохранять иронию рассказчика в тексте. "
        "Передавать фразу “Stop. Look around.” целиком в репликах. "
        "Избегать лишних пояснений в скобках."
    )
    merged_curly = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions=curly_f0),
        make_fragment(1, blocks=expected[1:], translation_instructions="Избегать лишних пояснений в скобках."),
        expected=expected,
    )
    assert merged_curly.translation_instructions.splitlines() == [
        "1. Сохранять иронию рассказчика в тексте.",
        "2. Передавать фразу “Stop. Look around.” целиком в репликах.",
        "3. Избегать лишних пояснений в скобках.",
    ]

    # 3. Solid 2-sentence paragraph splits when one sentence has ты/вы register rule -> register deduplicates cleanly.
    reg_f0 = "Ник и Элли обращаются друг к другу на «ты». Сохранять живой темп повествования."
    reg_f1 = "Ник и Элли обращаются друг к другу на «ты». Не упрощать метафоры в описаниях."
    merged_reg = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions=reg_f0),
        make_fragment(1, blocks=expected[1:], translation_instructions=reg_f1),
        expected=expected,
    )
    assert merged_reg.translation_instructions.splitlines() == [
        "1. Ник и Элли обращаются друг к другу на «ты».",
        "2. Сохранять живой темп повествования.",
        "3. Не упрощать метафоры в описаниях.",
    ]

    # 4. Fragment boundary artifacts are split out and filtered when len(fragments) > 1,
    #    but preserved when len(fragments) == 1.
    art_f0 = (
        "Сохранять тревожную интонацию. "
        "Незнакомца оставлять безымянным до конца фрагмента."
    )
    art_f1 = (
        "Держать короткие рубленые фразы. "
        "Переводить без подстановки имени из других глав. "
        "Герой в этом фрагменте остаётся без имени."
    )
    merged_multi = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions=art_f0),
        make_fragment(1, blocks=expected[1:], translation_instructions=art_f1),
        expected=expected,
    )
    assert merged_multi.translation_instructions.splitlines() == [
        "1. Сохранять тревожную интонацию.",
        "2. Держать короткие рубленые фразы.",
    ]

    single_expected = block_ids(1)
    merged_single = merge(
        make_fragment(0, blocks=single_expected, translation_instructions=art_f0),
        expected=single_expected,
    )
    assert "до конца фрагмента" in merged_single.translation_instructions


def test_normalize_a2_it_is_time_and_a2_core_vocab_promotion_and_demotion_guard() -> None:
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    point = (RequiredPoint("p1", "Объяснить слово."),)

    texts_and_expected = [
        # _A2_IT_IS_TIME_RE positives -> TRANSLATE
        ("It's about time you woke up.", BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        ("It’s high time we left.", BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        ("It's time he knew.", BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        # _A2_IT_IS_TIME_RE negatives -> stay READ_AS_IS
        ("It's time to go.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        ("It's time for bed.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        ("It's about time to leave.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        # a2_core_vocab: 1..3 uncovered non-core words -> GLOSSARY_ONLY (empty difficulties allowed)
        ("We could be the only survivors?", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("I bumped into you.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("No rips, no scratches.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("Major whoa.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        # a2_core_vocab: > 3 uncovered non-core words -> TRANSLATE
        (
            "The freckled wizard stumbled across the jagged cobblestone.",
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
        # a2_core_vocab: covered non-core word stays READ_AS_IS
        ("We could be the only survivors?", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        # a2_core_vocab: known_names skipped -> stays READ_AS_IS
        ("Skinjacker waited here.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        # a2_core_vocab: non-Latin script skipped -> stays READ_AS_IS
        ("Незнакомое редкое слово.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        # Demotion guard: non-core words block demotion to READ_AS_IS
        ("I bumped into you.", BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        ("We could be the only survivors?", BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        # Demotion allowed for core A2 + HP exceptions + known_names
        ("Nearly.", BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
        ("“Shoo!” said Mr Dursley loudly.", BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
        ("“No,” she said sharply. “Why?”", BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
        ("1 telescope", BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
        ("‘Come here,’ Skinjacker said.", BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
    ]

    blocks = tuple(
        make_analysis_block(section_id=SECTION_ID, order=i, text=text)
        for i, (text, _, _) in enumerate(texts_and_expected)
    )
    aliases = {f"b{i + 1}": block.id for i, block in enumerate(blocks)}
    covered_alias = "b12"
    payload = PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(f"b{i + 1}", initial_mode)
            for i, (_, initial_mode, _) in enumerate(texts_and_expected)
        ),
        difficulties=(
            QuotedDifficultyResponse(
                "d-covered-survivor",
                (QuotedSourceAnchorResponse(covered_alias, "survivors"),),
                covered_alias,
                "vocabulary",
                "useful",
                "explanation",
                point,
            ),
        ),
    )

    normalized = normalize_a2_dialogue_response(
        payload,
        blocks,
        aliases,
        reader_level="a2",
        english_help=False,
        known_names=("Skinjacker", "Mr Dursley"),
    )

    actual = [
        (texts_and_expected[i][0], normalized.treatment_plan[i].treatment)
        for i in range(len(texts_and_expected))
    ]
    expected = [(text, target_mode) for text, _, target_mode in texts_and_expected]
    assert actual == expected

    # Without known_names, "Skinjacker waited here." is promoted to GLOSSARY_ONLY.
    single_block = make_analysis_block(section_id=SECTION_ID, order=99, text="Skinjacker waited here.")
    without_known = normalize_a2_dialogue_response(
        PedagogicalAnalysisResponse(
            treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        ),
        (single_block,),
        {"b1": single_block.id},
        reader_level="a2",
        english_help=False,
    )
    assert without_known.treatment_plan[0].treatment is BlockTreatment.GLOSSARY_ONLY


def test_dialogue_quotes_and_a2_normalization_accept_straight_quotes_with_typographic_apostrophe() -> None:
    from linguary.application.analysis_merge import _DIALOGUE_QUOTES, normalize_a2_dialogue_response
    from linguary.application.llm_dto import PedagogicalAnalysisResponse

    assert [m.group(0) for m in _DIALOGUE_QUOTES.finditer('"Don’t move!" said Harry.')] == ['"Don’t move!"']
    assert [m.group(0) for m in _DIALOGUE_QUOTES.finditer('"I’m ready."')] == ['"I’m ready."']

    blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text='"Don’t move!" said Harry.'),
        make_analysis_block(section_id=SECTION_ID, order=1, text='"I’m ready, if you come now."'),
        make_analysis_block(section_id=SECTION_ID, order=2, text='"I’m ready."'),
    )
    aliases = {"b1": blocks[0].id, "b2": blocks[1].id, "b3": blocks[2].id}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b3", BlockTreatment.TRANSLATE),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload,
        blocks,
        aliases,
        reader_level="a2",
        english_help=False,
        known_names=("Harry",),
    )
    assert [item.treatment for item in normalized.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    ]


def test_split_instruction_sentences_splits_after_terminal_punctuation_before_closing_quote() -> None:
    from linguary.application.analysis_merge import _split_instruction_sentences

    assert _split_instruction_sentences("Передавать «Wait.» Обращение на «вы» сохранять.") == (
        "Передавать «Wait.»",
        "Обращение на «вы» сохранять.",
    )
    assert _split_instruction_sentences("Передавать “Wait!” Сохранять короткий ритм.") == (
        "Передавать “Wait!”",
        "Сохранять короткий ритм.",
    )
    assert _split_instruction_sentences('Передавать "Wait?" Сохранять короткий ритм.') == (
        'Передавать "Wait?"',
        "Сохранять короткий ритм.",
    )
    assert _split_instruction_sentences("Передавать «Wait.» без сглаживания.") == (
        "Передавать «Wait.» без сглаживания.",
    )

    expected = block_ids(2)
    merged = merge(
        make_fragment(
            0,
            blocks=expected[:1],
            translation_instructions="Передавать «Wait.» Обращение на «вы» сохранять.",
        ),
        make_fragment(
            1,
            blocks=expected[1:],
            translation_instructions="Обращение на «вы» сохранять.",
        ),
        expected=expected,
    )
    assert merged.translation_instructions.splitlines() == [
        "1. Передавать «Wait.»",
        "2. Обращение на «вы» сохранять.",
    ]


def test_prune_forward_difficulty_references_renumbers_standard_point_ids_and_remaps_evidence() -> None:
    from linguary.application.analysis_merge import (
        _prune_forward_difficulty_references,
        normalize_a2_dialogue_response,
    )
    from linguary.application.llm_dto import (
        HelpEvidenceResponse,
        PedagogicalAnalysisResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="Nick stared at the silent road."),
        make_analysis_block(
            section_id=SECTION_ID,
            order=1,
            text="Later Allie found a hidden golden compass near the tall tower.",
        ),
    )
    aliases = {"b1": blocks[0].id, "b2": blocks[1].id}

    # 1. Direct _prune_forward_difficulty_references: p1 pruned -> p2 renumbered to p1.
    diff_std = QuotedDifficultyResponse(
        "d1",
        (QuotedSourceAnchorResponse("b1", "silent road"),),
        "b1",
        "vocabulary",
        "useful",
        "explanation",
        (
            RequiredPoint("p1", 'Позже встречается фраза "hidden golden compass" про путь.'),
            RequiredPoint("p2", "В этом блоке silent road означает тихую дорогу."),
        ),
    )
    pruned_std = _prune_forward_difficulty_references(diff_std, blocks, aliases)
    assert [(pt.id, pt.description) for pt in pruned_std.required_points] == [
        ("p1", "В этом блоке silent road означает тихую дорогу."),
    ]

    # 2. Multiple surviving standard points: p1(fwd), p2(ok), p3(fwd), p4(ok) -> p1, p2.
    diff_multi = QuotedDifficultyResponse(
        "d2",
        (QuotedSourceAnchorResponse("b1", "silent road"),),
        "b1",
        "vocabulary",
        "useful",
        "explanation",
        (
            RequiredPoint("p1", 'Спойлер "hidden golden compass" из следующего блока.'),
            RequiredPoint("p2", "Первое локальное пояснение."),
            RequiredPoint("p3", 'Ещё один спойлер "tall tower" из следующего блока.'),
            RequiredPoint("p4", "Второе локальное пояснение."),
        ),
    )
    pruned_multi = _prune_forward_difficulty_references(diff_multi, blocks, aliases)
    assert [(pt.id, pt.description) for pt in pruned_multi.required_points] == [
        ("p1", "Первое локальное пояснение."),
        ("p2", "Второе локальное пояснение."),
    ]

    # 3. Non-standard point IDs are NOT renumbered.
    diff_custom = QuotedDifficultyResponse(
        "d3",
        (QuotedSourceAnchorResponse("b1", "silent road"),),
        "b1",
        "vocabulary",
        "useful",
        "explanation",
        (
            RequiredPoint("pt-forward", 'Спойлер "hidden golden compass" из следующего блока.'),
            RequiredPoint("pt-keep", "Локальное пояснение без спойлера."),
        ),
    )
    pruned_custom = _prune_forward_difficulty_references(diff_custom, blocks, aliases)
    assert [pt.id for pt in pruned_custom.required_points] == ["pt-keep"]

    # 4. normalize_a2_dialogue_response drops evidence for pruned p1 and remaps kept p2 -> p1.
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.TRANSLATE),
        ),
        difficulties=(diff_std,),
        before_you_read_evidence=(
            HelpEvidenceResponse("d1", "p1", "before_you_read.explanation", "b1", 0, "Спойлерный пункт."),
            HelpEvidenceResponse("d1", "p2", "before_you_read.explanation", "b1", 0, "Локальный пункт."),
        ),
    )
    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level="a2", english_help=False,
    )
    assert [pt.id for pt in normalized.difficulties[0].required_points] == ["p1"]
    assert [(ev.difficulty_id, ev.point_id, ev.body_excerpt) for ev in normalized.before_you_read_evidence] == [
        ("d1", "p1", "Локальный пункт."),
    ]


def test_merged_instructions_filters_extended_fragment_boundary_artifacts() -> None:
    expected = block_ids(2)
    f0 = (
        "Сохранять тревожный ритм. "
        "В начале фрагмента мальчик ещё не назван. "
        "В конце фрагмента герои уходят. "
        "В этом фрагменте не раскрывать место действия. "
        "В текущем фрагменте герои держат дистанцию. "
        "До конца данного фрагмента сохранять интригу. "
        "В этом фрагменте письма сохранять высокий стиль."
    )
    f1 = (
        "Передавать речь без придуманного имени. "
        "Незнакомка остаётся неназванной. "
        "Мальчик остается без имени. "
        "Герой остаётся неназванным до конца фрагмента. "
        "Девочка остаётся неназванной до сцены в лесу. "
        "Мальчик остается без имени до знакомства с Элли."
    )
    merged_multi = merge(
        make_fragment(0, blocks=expected[:1], translation_instructions=f0),
        make_fragment(1, blocks=expected[1:], translation_instructions=f1),
        expected=expected,
    )
    assert merged_multi.translation_instructions.splitlines() == [
        "1. Сохранять тревожный ритм.",
        "2. В этом фрагменте письма сохранять высокий стиль.",
        "3. Девочка остаётся неназванной до сцены в лесу.",
        "4. Мальчик остается без имени до знакомства с Элли.",
    ]

    # Single fragment preserves all instructions intact.
    single_expected = block_ids(1)
    merged_single = merge(
        make_fragment(0, blocks=single_expected, translation_instructions=f0),
        expected=single_expected,
    )
    assert "В начале фрагмента" in merged_single.translation_instructions
    assert "в этом фрагменте" in merged_single.translation_instructions.lower()


def test_normalize_a2_promotes_b1_phrasal_verbs_and_quoted_subject_inversion() -> None:
    from linguary.application.analysis_merge import normalize_a2_dialogue_response
    from linguary.application.llm_dto import (
        PedagogicalAnalysisResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    point = (RequiredPoint("p1", "Объяснить фразовый глагол."),)
    cases = [
        # B1 phrasal verbs across inflections (base, past, 3sg, -ing) -> GLOSSARY_ONLY
        ("“Calm down,” he said.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("He calmed down.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("It turned out fine.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("She figured out the way.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("The car broke down.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("We ran out of water.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("He ends up alone.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("Put up with it.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        ("She is holding on.", BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY),
        # Quoted subject inversion (curly, straight double, curly/straight single, NP subject) -> TRANSLATE
        ("“Here,” was all he said.", BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        ('"Wait", was all she asked.', BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        ("‘No,’ was all Nick replied.", BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        ("'Here,' was all the boy said.", BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        # Literal A1 spatial motion and hyphenated noun stay READ_AS_IS
        ("Run out of the house!", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        ("Come across the street!", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        ("Bring up the bag!", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        ("She has make-up.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        # Negative controls stay READ_AS_IS
        ("“Here,” he said.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
        ("That was all he said.", BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
    ]
    blocks = tuple(
        make_analysis_block(section_id=SECTION_ID, order=idx, text=text)
        for idx, (text, _, _) in enumerate(cases)
    )
    aliases = {f"b{idx + 1}": b.id for idx, b in enumerate(blocks)}
    plan_items = tuple(
        TreatmentPlanItemResponse(f"b{idx + 1}", initial)
        for idx, (_, initial, _) in enumerate(cases)
    )
    payload = PedagogicalAnalysisResponse(
        treatment_plan=plan_items,
        difficulties=(),
    )
    normalized = normalize_a2_dialogue_response(
        payload,
        blocks,
        aliases,
        reader_level="a2",
        english_help=False,
        known_names=("Nick",),
    )
    actual = [item.treatment for item in normalized.treatment_plan]
    expected = [exp for _, _, exp in cases]
    assert actual == expected


def test_validate_content_plan_rejects_duplicate_difficulty_anchor_in_same_block() -> None:
    from linguary.application.analysis_merge import validate_content_plan
    from linguary.application.llm_dto import (
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="Nick could barely calm down."),
        make_analysis_block(section_id=SECTION_ID, order=1, text="Allie watched him calm down."),
    )
    aliases = {"b1": blocks[0].id, "b2": blocks[1].id}
    texts = {b.id: b.text for b in blocks}
    visible = {alias: texts[bid] for alias, bid in aliases.items()}
    plan_items = (
        TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
        TreatmentPlanItemResponse("b2", BlockTreatment.TRANSLATE),
    )

    dup_payload = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=plan_items,
        content_plan=QuotedContentPlanResponse(
            difficulties=(
                QuotedDifficultyResponse(
                    "d1",
                    (QuotedSourceAnchorResponse("b1", "calm down"),),
                    "b1",
                    "vocabulary",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Лексическое значение."),),
                ),
                QuotedDifficultyResponse(
                    "d2",
                    (QuotedSourceAnchorResponse("b1", "calm down"),),
                    "b1",
                    "idiom",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Фразовый глагол."),),
                ),
            ),
        ),
    )
    with pytest.raises(AnalysisMergeError) as exc:
        validate_content_plan(
            dup_payload,
            aliases,
            texts,
            visible_texts_by_alias=visible,
            blocks=blocks,
        )
    assert exc.value.violation is not None
    assert exc.value.violation.rule_code == "duplicate_difficulty_anchor"
    assert exc.value.violation.block_ids == (blocks[0].id,)

    # Mixed start=None vs explicit start/end for the same occurrence also collides.
    mixed_dup_payload = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=plan_items,
        content_plan=QuotedContentPlanResponse(
            difficulties=(
                QuotedDifficultyResponse(
                    "d1",
                    (QuotedSourceAnchorResponse("b1", "calm down", start=None, end=None),),
                    "b1",
                    "vocabulary",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Лексическое значение."),),
                ),
                QuotedDifficultyResponse(
                    "d2",
                    (QuotedSourceAnchorResponse("b1", "calm down", start=18, end=27),),
                    "b1",
                    "idiom",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Фразовый глагол."),),
                ),
            ),
        ),
    )
    with pytest.raises(AnalysisMergeError) as mixed_exc:
        validate_content_plan(
            mixed_dup_payload,
            aliases,
            texts,
            visible_texts_by_alias=visible,
            blocks=blocks,
        )
    assert mixed_exc.value.violation is not None
    assert mixed_exc.value.violation.rule_code == "duplicate_difficulty_anchor"

    # Overlapping anchors of different length AND same quote in different blocks pass validation.
    ok_payload = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=plan_items,
        content_plan=QuotedContentPlanResponse(
            difficulties=(
                QuotedDifficultyResponse(
                    "d1",
                    (QuotedSourceAnchorResponse("b1", "barely"),),
                    "b1",
                    "vocabulary",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Значение barely."),),
                ),
                QuotedDifficultyResponse(
                    "d2",
                    (QuotedSourceAnchorResponse("b1", "barely calm down"),),
                    "b1",
                    "grammar",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Оборот целиком."),),
                ),
                QuotedDifficultyResponse(
                    "d3",
                    (QuotedSourceAnchorResponse("b2", "calm down"),),
                    "b2",
                    "idiom",
                    "useful",
                    "explanation",
                    (RequiredPoint("p1", "Фразовый глагол во втором блоке."),),
                ),
            ),
        ),
    )
    validated = validate_content_plan(
        ok_payload,
        aliases,
        texts,
        visible_texts_by_alias=visible,
        blocks=blocks,
    )
    assert validated is not None
    assert len(validated.difficulties) == 3








def test_normalize_difficulty_kinds_and_non_lexical_validation() -> None:
    from linguary.application.analysis_merge import (
        AnalysisMergeError,
        normalize_a2_dialogue_response,
        normalize_difficulty_kinds,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        PedagogicalAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain.content_plan import RequiredPoint

    blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="\u2018Where\u2019s the cannon?\u2019"),
        make_analysis_block(section_id=SECTION_ID, order=1, text="\u2018Get up!\u2019 she demanded."),
        make_analysis_block(section_id=SECTION_ID, order=2, text="Stalagmite\u2019s got an \u201cm\u201d in it."),
    )
    aliases = {"b1": blocks[0].id, "b2": blocks[1].id, "b3": blocks[2].id}
    texts = {b.id: b.text for b in blocks}
    visible = {a: texts[bid] for a, bid in aliases.items()}

    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse("b2", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b3", BlockTreatment.TRANSLATE),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                "d-cannon",
                (QuotedSourceAnchorResponse("b1", "cannon"),),
                "b1",
                "wordplay",
                "useful",
                "lexical_hint",
                (RequiredPoint("p1", "cannon \u2014 \u043f\u0443\u0448\u043a\u0430."),),
            ),
            QuotedDifficultyResponse(
                "d-demanded",
                (QuotedSourceAnchorResponse("b2", "demanded"),),
                "b2",
                "idiom",
                "useful",
                "explanation",
                (RequiredPoint("p1", "\u041f\u043e\u0442\u0440\u0435\u0431\u043e\u0432\u0430\u043b\u0430."),),
            ),
            QuotedDifficultyResponse(
                "d-pun",
                (QuotedSourceAnchorResponse("b3", "Stalagmite"),),
                "b3",
                "wordplay",
                "useful",
                "explanation",
                (RequiredPoint("p1", "\u041a\u0430\u043b\u0430\u043c\u0431\u0443\u0440 \u0441 \u0431\u0443\u043a\u0432\u043e\u0439 m."),),
            ),
        ),
    )

    normalized = normalize_a2_dialogue_response(
        payload, blocks, aliases, reader_level="a2", english_help=False
    )
    by_id = {d.id: d for d in normalized.difficulties}
    assert by_id["d-cannon"].kind == "vocabulary"
    assert by_id["d-cannon"].help_kind == "lexical_hint"
    assert normalized.treatment_plan[0].treatment is BlockTreatment.GLOSSARY_ONLY
    assert by_id["d-demanded"].kind == "vocabulary"
    assert by_id["d-pun"].kind == "wordplay"

    # Genuine pun description on lexical_hint is NOT silently turned into vocabulary,
    # so validate_content_plan rejects it with non_lexical_kind_requires_explanation.
    bad_pun = PedagogicalAnalysisResponse(
        treatment_plan=(TreatmentPlanItemResponse("b3", BlockTreatment.GLOSSARY_ONLY),),
        difficulties=(
            QuotedDifficultyResponse(
                "d-bad-pun",
                (QuotedSourceAnchorResponse("b3", "Stalagmite"),),
                "b3",
                "wordplay",
                "useful",
                "lexical_hint",
                (RequiredPoint("p1", "\u041a\u0430\u043b\u0430\u043c\u0431\u0443\u0440 \u0441 \u0431\u0443\u043a\u0432\u043e\u0439 m."),),
            ),
        ),
    )
    still_pun = normalize_difficulty_kinds(bad_pun, (blocks[2],), {"b3": blocks[2].id})
    assert still_pun.difficulties[0].kind == "wordplay"
    combined = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=still_pun.treatment_plan,
        content_plan=QuotedContentPlanResponse(difficulties=still_pun.difficulties),
    )
    with pytest.raises(AnalysisMergeError) as exc:
        validate_content_plan(
            combined,
            {"b3": blocks[2].id},
            {blocks[2].id: blocks[2].text},
            visible_texts_by_alias={"b3": blocks[2].text},
            blocks=(blocks[2],),
        )
    assert exc.value.violation.rule_code == "non_lexical_kind_requires_explanation"


def test_auto_vocab_materialization_supports_content_plan_v3_and_mixed_segments() -> None:
    from linguary.application.analysis_merge import (
        normalize_a2_dialogue_response,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        ContentPlanResponse,
        PedagogicalAnalysisResponse,
        QuotedContentPlanResponse,
        SegmentTreatmentResponse,
    )
    from linguary.domain import BlockSegment

    # 1. Mixed block initially READ_AS_IS: seg1 -> GLOSSARY_ONLY, seg2 -> TRANSLATE.
    s1 = "It winked. "
    s2 = "He has been here."
    mixed_block = replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=s1 + s2),
        segments=(
            BlockSegment(0, len(s1), 0),
            BlockSegment(len(s1), len(s1) + len(s2), 1),
        ),
    )
    aliases = {"b1": mixed_block.id}
    ped_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.READ_AS_IS,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
                    SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                ),
            ),
        ),
    )
    norm_ped = normalize_a2_dialogue_response(
        ped_payload, (mixed_block,), aliases, reader_level="a2", english_help=False
    )
    assert [seg.treatment for seg in norm_ped.treatment_plan[0].segments] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.TRANSLATE,
    ]
    assert len(norm_ped.difficulties) == 1
    assert norm_ped.difficulties[0].kind == "vocabulary"
    assert norm_ped.difficulties[0].help_kind == "lexical_hint"
    assert norm_ped.difficulties[0].anchors[0].quote == "winked"
    validated = validate_content_plan(
        ChapterAnalysisResponse(
            translation_instructions="Keep voice.",
            chapter_summary="Summary.",
            treatment_plan=norm_ped.treatment_plan,
            content_plan=QuotedContentPlanResponse(difficulties=norm_ped.difficulties),
        ),
        aliases,
        {mixed_block.id: mixed_block.text},
        visible_texts_by_alias={"b1": mixed_block.text},
        blocks=(mixed_block,),
    )
    assert validated is not None and len(validated.difficulties) == 1

    # 2. ContentPlanResponse (v3 domain Difficulty/SourceAnchor) uses alias b1 and validates cleanly.
    plain_block = make_analysis_block(section_id=SECTION_ID, order=1, text="It winked.")
    v3_payload = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        content_plan=ContentPlanResponse(version=3, profile="content-plan-v3"),
    )
    norm_v3 = normalize_a2_dialogue_response(
        v3_payload, (plain_block,), {"b1": plain_block.id}, reader_level="a2", english_help=False
    )
    assert norm_v3.content_plan is not None and len(norm_v3.content_plan.difficulties) == 1
    assert norm_v3.content_plan.difficulties[0].owner_block_id == "b1"
    validated_v3 = validate_content_plan(
        norm_v3,
        {"b1": plain_block.id},
        {plain_block.id: plain_block.text},
        visible_texts_by_alias={"b1": plain_block.text},
        blocks=(plain_block,),
    )
    assert validated_v3 is not None and validated_v3.difficulties[0].owner_block_id == plain_block.id

    # 3. content_plan=None does not fail.
    none_payload = ChapterAnalysisResponse(
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),),
        content_plan=None,
    )
    norm_none = normalize_a2_dialogue_response(
        none_payload, (plain_block,), {"b1": plain_block.id}, reader_level="a2", english_help=False
    )
    assert norm_none.treatment_plan[0].treatment is BlockTreatment.GLOSSARY_ONLY
    assert norm_none.content_plan is None


def test_unify_verse_stanza_treatments_and_poem_fragment_boundary_guard() -> None:
    from linguary.application.analysis_merge import (
        normalize_pedagogical_response,
        plan_fragments,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        PedagogicalAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
    )
    from linguary.domain import InlineAnnotation, InlineAnnotationType
    from linguary.domain.content_plan import RequiredPoint

    # 1. Implicit enjambed verse stanza in <p><i>...</i></p> (mixed TRANSLATE / READ_AS_IS -> all TRANSLATE).
    verse_lines = (
        "Enter, stranger, but take heed",
        "Of what awaits the sin of greed,",
        "For those who take, but do not earn,",
        "Must pay most dearly in their turn.",
    )
    verse_blocks = tuple(
        replace(
            make_analysis_block(section_id=SECTION_ID, order=i, text=line),
            inline_annotations=(InlineAnnotation(InlineAnnotationType.ITALIC, 0, len(line)),),
        )
        for i, line in enumerate(verse_lines)
    )
    aliases = {f"b{i + 1}": b.id for i, b in enumerate(verse_blocks)}
    payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b3", BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse("b4", BlockTreatment.READ_AS_IS),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                "d-earn",
                (QuotedSourceAnchorResponse("b3", "earn"),),
                "b3",
                "vocabulary",
                "useful",
                "lexical_hint",
                (RequiredPoint("p1", "earn \u2014 \u0437\u0430\u0440\u0430\u0431\u0430\u0442\u044b\u0432\u0430\u0442\u044c."),),
            ),
        ),
    )
    # Works for both A2 and B2 (switching promoted lexical_hint -> explanation and not synthesizing auto-vocab).
    for level in ("a2", "b2"):
        norm = normalize_pedagogical_response(
            payload, verse_blocks, aliases, reader_level=level, english_help=False
        )
        assert [item.treatment for item in norm.treatment_plan] == [BlockTreatment.TRANSLATE] * 4
        assert len(norm.difficulties) == 1
        assert norm.difficulties[0].help_kind == "explanation"
        validated = validate_content_plan(
            ChapterAnalysisResponse(
                translation_instructions="Keep voice.",
                chapter_summary="Summary.",
                treatment_plan=norm.treatment_plan,
                content_plan=QuotedContentPlanResponse(difficulties=norm.difficulties),
            ),
            aliases,
            {b.id: b.text for b in verse_blocks},
            visible_texts_by_alias={a: verse_blocks[i].text for i, a in enumerate(aliases)},
            blocks=verse_blocks,
        )
        assert validated is not None

    # All-light verse stanza stays untouched.
    light_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.GLOSSARY_ONLY),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b3", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b4", BlockTreatment.READ_AS_IS),
        ),
    )
    light_norm = normalize_pedagogical_response(
        light_payload, verse_blocks, aliases, reader_level="b2", english_help=False
    )
    assert [item.treatment for item in light_norm.treatment_plan] == [
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
    ]

    # 2. Negative controls: unnumbered item list and interrupted closed dialogue are NOT unified.
    list_blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="Three sets of plain work robes (black)"),
        make_analysis_block(section_id=SECTION_ID, order=1, text="One plain pointed hat (black) for day wear"),
        make_analysis_block(section_id=SECTION_ID, order=2, text="One winter cloak (black, silver fastenings)"),
    )
    list_aliases = {f"b{i + 1}": b.id for i, b in enumerate(list_blocks)}
    list_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b3", BlockTreatment.TRANSLATE),
        ),
    )
    list_norm = normalize_pedagogical_response(
        list_payload, list_blocks, list_aliases, reader_level="b2", english_help=False
    )
    assert [item.treatment for item in list_norm.treatment_plan] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
    ]

    dialogue_blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="\u2018Wait \u2014\u2019"),
        make_analysis_block(section_id=SECTION_ID, order=1, text="\u2018Don\u2019t move \u2014\u2019"),
        make_analysis_block(section_id=SECTION_ID, order=2, text="\u2018I told you \u2014\u2019"),
        make_analysis_block(section_id=SECTION_ID, order=3, text="\u2018Shut up!\u2019"),
    )
    dlg_aliases = {f"b{i + 1}": b.id for i, b in enumerate(dialogue_blocks)}
    dlg_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b2", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b3", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b4", BlockTreatment.READ_AS_IS),
        ),
    )
    dlg_norm = normalize_pedagogical_response(
        dlg_payload, dialogue_blocks, dlg_aliases, reader_level="b2", english_help=False
    )
    assert [item.treatment for item in dlg_norm.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
    ]

    # 3. plan_fragments avoids cutting contiguous BlockType.POEM blocks when another balanced cut exists.
    mixed_poem_blocks = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="Intro paragraph one."),
        make_analysis_block(section_id=SECTION_ID, order=1, text="Intro paragraph two."),
        replace(make_analysis_block(section_id=SECTION_ID, order=2, text="Line one of verse"), type=BlockType.POEM),
        replace(make_analysis_block(section_id=SECTION_ID, order=3, text="Line two of verse"), type=BlockType.POEM),
        make_analysis_block(section_id=SECTION_ID, order=4, text="Outro paragraph."),
    )
    frags = plan_fragments(mixed_poem_blocks, max_blocks=3, max_chars=10_000)
    assert len(frags) == 2
    poem_frag_indices = [
        idx for idx, frag in enumerate(frags)
        if any(b.type is BlockType.POEM for b in frag.blocks)
    ]
    assert len(poem_frag_indices) == 1


def test_verse_unification_clears_stale_demotion_trims_adjacent_prose_and_drops_auto_vocab() -> None:
    from linguary.application.analysis_merge import (
        normalize_difficulty_kinds,
        normalize_pedagogical_response,
        unify_chapter_analysis_verse_stanzas,
        validate_content_plan,
    )
    from linguary.application.llm_dto import (
        ChapterAnalysisResponse,
        PedagogicalAnalysisResponse,
        QuotedContentPlanResponse,
        QuotedDifficultyResponse,
        QuotedSourceAnchorResponse,
        SegmentTreatmentResponse,
    )
    from linguary.domain import (
        BlockSegment,
        ChapterAnalysis,
        CoverageStatus,
        TreatmentPlan,
        TreatmentPlanItem,
    )
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, SourceAnchor

    # 1. Blocker 1 regression: short TRANSLATE verse line demoted on step 2, then lifted back to TRANSLATE
    #    on step 3 must keep help_kind="explanation" and pass validate_content_plan without lexical_help_channel.
    stanza_lines = (
        "Enter, stranger, but take heed",
        "Of the sin of greed,",
        "For those who take, but do not earn,",
        "Must pay most dearly in their turn.",
    )
    stanza_blocks = tuple(
        make_analysis_block(section_id=SECTION_ID, order=i, text=line)
        for i, line in enumerate(stanza_lines)
    )
    stanza_aliases = {f"b{i + 1}": b.id for i, b in enumerate(stanza_blocks)}
    stanza_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b2", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b3", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b4", BlockTreatment.READ_AS_IS),
        ),
        difficulties=(
            QuotedDifficultyResponse(
                "d-greed",
                (QuotedSourceAnchorResponse("b2", "greed"),),
                "b2",
                "vocabulary",
                "useful",
                "explanation",
                (RequiredPoint("p1", "greed \u2014 \u0436\u0430\u0434\u043d\u043e\u0441\u0442\u044c."),),
            ),
        ),
    )
    stanza_norm = normalize_pedagogical_response(
        stanza_payload, stanza_blocks, stanza_aliases, reader_level="a2", english_help=False
    )
    assert [item.treatment for item in stanza_norm.treatment_plan] == [BlockTreatment.TRANSLATE] * 4
    assert [d.help_kind for d in stanza_norm.difficulties] == ["explanation"]
    validated = validate_content_plan(
        ChapterAnalysisResponse(
            translation_instructions="Keep voice.",
            chapter_summary="Summary.",
            treatment_plan=stanza_norm.treatment_plan,
            content_plan=QuotedContentPlanResponse(difficulties=stanza_norm.difficulties),
        ),
        stanza_aliases,
        {b.id: b.text for b in stanza_blocks},
        visible_texts_by_alias={f"b{i + 1}": b.text for i, b in enumerate(stanza_blocks)},
        blocks=stanza_blocks,
    )
    assert validated is not None

    # 2. Blocker 2 regression: short prose paragraphs before and after a non-italic verse stanza
    #    are neither absorbed into TRANSLATE nor allowed to hide the stanza.
    prose_and_verse = (
        make_analysis_block(section_id=SECTION_ID, order=0, text="They stopped near the doors."),
        make_analysis_block(section_id=SECTION_ID, order=1, text="Words were carved on them:"),
        make_analysis_block(section_id=SECTION_ID, order=2, text="Enter, stranger, but take heed"),
        make_analysis_block(section_id=SECTION_ID, order=3, text="Of what awaits the sin of greed,"),
        make_analysis_block(section_id=SECTION_ID, order=4, text="For those who take, but do not earn,"),
        make_analysis_block(section_id=SECTION_ID, order=5, text="Must pay most dearly in their turn."),
        make_analysis_block(section_id=SECTION_ID, order=6, text="Harry looked at the doors."),
        make_analysis_block(section_id=SECTION_ID, order=7, text="Hagrid stood beside him."),
        make_analysis_block(section_id=SECTION_ID, order=8, text="The hall was quiet."),
    )
    pv_aliases = {f"b{i + 1}": b.id for i, b in enumerate(prose_and_verse)}
    pv_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b2", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b3", BlockTreatment.TRANSLATE),
            TreatmentPlanItemResponse("b4", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b5", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b6", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b7", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b8", BlockTreatment.READ_AS_IS),
            TreatmentPlanItemResponse("b9", BlockTreatment.READ_AS_IS),
        ),
    )
    pv_norm = normalize_pedagogical_response(
        pv_payload, prose_and_verse, pv_aliases, reader_level="b2", english_help=False
    )
    assert [item.treatment for item in pv_norm.treatment_plan] == [
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
        BlockTreatment.READ_AS_IS,
    ]

    # 3. Single BlockType.POEM block with mixed segments unifies to TRANSLATE.
    poem_text = "Enter, stranger, take heed. Pay dearly in your turn."
    cut = poem_text.index("Pay")
    single_poem = replace(
        make_analysis_block(section_id=SECTION_ID, order=0, text=poem_text),
        type=BlockType.POEM,
        segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(poem_text), 1)),
    )
    single_poem_payload = PedagogicalAnalysisResponse(
        treatment_plan=(
            TreatmentPlanItemResponse(
                "b1",
                BlockTreatment.TRANSLATE,
                (
                    SegmentTreatmentResponse(1, BlockTreatment.TRANSLATE),
                    SegmentTreatmentResponse(2, BlockTreatment.READ_AS_IS),
                ),
            ),
        ),
    )
    sp_norm = normalize_pedagogical_response(
        single_poem_payload, (single_poem,), {"b1": single_poem.id}, reader_level="b2", english_help=False
    )
    assert [seg.treatment for seg in sp_norm.treatment_plan[0].segments] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    ]

    # 4. Cross-fragment unify_chapter_analysis_verse_stanzas drops promoted auto-vocab instead of
    #    converting it into an explanation on TRANSLATE.
    p1 = replace(make_analysis_block(section_id=SECTION_ID, order=0, text="Line one of verse"), type=BlockType.POEM)
    p2 = replace(make_analysis_block(section_id=SECTION_ID, order=1, text="Line two with heed."), type=BlockType.POEM)
    heed_start = p2.text.index("heed")
    auto_diff = Difficulty(
        id="canonical-hash-id",
        anchors=(SourceAnchor(p2.id, heed_start, heed_start + 4, "heed"),),
        owner_block_id=p2.id,
        kind="vocabulary",
        priority="useful",
        help_kind="lexical_hint",
        required_points=(RequiredPoint("p1", "Дать краткую контекстную словарную подсказку для «heed»."),),
    )
    merged_analysis = ChapterAnalysis(
        id="a1",
        section_id=SECTION_ID,
        coverage_status=CoverageStatus.COMPLETE,
        translation_instructions="Keep voice.",
        chapter_summary="Summary.",
        treatment_plan=TreatmentPlan(
            section_id=SECTION_ID,
            items=(
                TreatmentPlanItem(p1.id, BlockTreatment.TRANSLATE),
                TreatmentPlanItem(p2.id, BlockTreatment.GLOSSARY_ONLY),
            ),
        ),
        content_plan=ContentPlan(version=3, profile="content-plan-v3", difficulties=(auto_diff,)),
    )
    unified_analysis = unify_chapter_analysis_verse_stanzas(merged_analysis, (p1, p2), english_help=False)
    assert [item.treatment for item in unified_analysis.treatment_plan.items] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.TRANSLATE,
    ]
    assert unified_analysis.content_plan is not None
    assert unified_analysis.content_plan.difficulties == ()

    # 5. Russian oblique cases ("игру слов", "двойной смысл") protect genuine wordplay from reclassification.
    oblique_block = make_analysis_block(section_id=SECTION_ID, order=0, text="A clever pun.")
    for desc in ("Объяснить игру слов.", "Раскрыть двойной смысл выражения."):
        oblique_payload = PedagogicalAnalysisResponse(
            treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.GLOSSARY_ONLY),),
            difficulties=(
                QuotedDifficultyResponse(
                    "d-oblique",
                    (QuotedSourceAnchorResponse("b1", "pun"),),
                    "b1",
                    "wordplay",
                    "useful",
                    "lexical_hint",
                    (RequiredPoint("p1", desc),),
                ),
            ),
        )
        kept = normalize_difficulty_kinds(oblique_payload, (oblique_block,), {"b1": oblique_block.id})
        assert kept.difficulties[0].kind == "wordplay"
