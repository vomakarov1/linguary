"""Chapter analysis scenario: request, response validation, and persistence (P3.T3, PRD § 14)."""

from __future__ import annotations

import pytest
from analysis_fakes import (
    BOOK_ID,
    CONFIDENCE_THRESHOLD,
    TARGET_LANGUAGE,
    FakeUnitOfWork,
    analysis_reply,
    make_analysis_block,
    make_analysis_candidate,
    make_analysis_document,
    make_analysis_section,
    make_analyze_chapter,
    make_chapter_analysis,
)

from linguary.adapters.llm import FakeLLMClient
from linguary.application.ports.llm import LLMTransportError
from linguary.application.use_cases.analyze_chapter import (
    AnalyzeChapterCommand,
    AnalyzeChapterError,
)
from linguary.domain import (
    BlockTreatment,
    BlockType,
    ConceptType,
    CoverageStatus,
    TermConcept,
    TermStatus,
    TermVariant,
    TokenUsage,
)


def two_block_book() -> FakeUnitOfWork:
    """A book with one chapter of two paragraphs."""
    section = make_analysis_section(texts=("First paragraph.", "Second paragraph."))
    return FakeUnitOfWork(document=make_analysis_document((section,)))


def command() -> AnalyzeChapterCommand:
    """Command to analyze the single chapter of the book."""
    return AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-1")


async def test_saved_analysis_keeps_chapter_summary_and_materials() -> None:
    """Saved analysis preserves chapter summary, instructions, and `Before You Read`."""
    unit_of_work = two_block_book()
    scenario, _ = make_analyze_chapter(unit_of_work, FakeLLMClient.always(analysis_reply()))

    result = await scenario(command())

    stored = unit_of_work.analyses.get_for_section(BOOK_ID, "section-1")
    assert stored == result.analysis
    assert stored is not None
    assert stored.chapter_summary == "Mr. Bennet visits the new neighbour."
    assert stored.translation_instructions == "Сохранять иронию рассказчика."
    assert stored.before_you_read[0].expression == "entailed estate"


async def test_saved_plan_covers_chapter_blocks_in_document_order() -> None:
    """The processing plan covers chapter blocks and stores their stable identifiers."""
    unit_of_work = two_block_book()
    section = unit_of_work.documents.get(BOOK_ID).section("section-1")  # type: ignore[union-attr]
    scenario, _ = make_analyze_chapter(
        unit_of_work,
        FakeLLMClient.always(analysis_reply(aliases=("b2", "b1"))),
    )

    analysis = (await scenario(command())).analysis

    assert analysis.coverage_status is CoverageStatus.COMPLETE
    assert analysis.treatment_plan.block_ids == tuple(
        block.id for block in section.ordered_blocks
    )
    assert all(
        analysis.treatment_plan.treatment_for(block.id) is BlockTreatment.TRANSLATE
        for block in section.ordered_blocks
    )


async def test_requires_review_is_computed_by_application() -> None:
    """`requires_review` is computed by the application, not the model (DTO-01, PRD § 12)."""
    unit_of_work = two_block_book()
    scenario, _ = make_analyze_chapter(
        unit_of_work,
        FakeLLMClient.always(
            analysis_reply(
                candidates=[
                    {
                        "source_text": "Mr. Bennet",
                        "type": "person",
                        "suggested_translation": "мистер Беннет",
                        "confidence": 1.0,
                        "model_flagged_review": False,
                        "note": None,
                    },
                    {
                        "source_text": "entail",
                        "type": "term",
                        "suggested_translation": "майорат",
                        "confidence": 0.95,
                        "model_flagged_review": False,
                        "note": None,
                    },
                ]
            )
        ),
    )

    candidates = (await scenario(command())).analysis.glossary_candidates

    by_text = {candidate.source_text: candidate for candidate in candidates}
    assert by_text["Mr. Bennet"].type is ConceptType.PERSON
    assert by_text["Mr. Bennet"].requires_review is True
    assert by_text["entail"].requires_review is False
    assert all(candidate.analysis_id == "id-1" for candidate in candidates)


async def test_low_confidence_candidate_requires_review() -> None:
    """Confidence below the configured threshold marks the candidate as requiring attention."""
    unit_of_work = two_block_book()
    scenario, _ = make_analyze_chapter(
        unit_of_work,
        FakeLLMClient.always(
            analysis_reply(
                candidates=[
                    {
                        "source_text": "entail",
                        "type": "term",
                        "suggested_translation": "майорат",
                        "confidence": CONFIDENCE_THRESHOLD - 0.1,
                        "model_flagged_review": False,
                        "note": None,
                    }
                ]
            )
        ),
    )

    candidate = (await scenario(command())).analysis.glossary_candidates[0]

    assert candidate.requires_review is True


async def test_request_carries_previous_chapter_candidates_only() -> None:
    """The chapter request includes decisions from previous chapters, but not their text (PRD § 13)."""
    first = make_analysis_section(
        section_id="section-1", title="Chapter 1", order=0, texts=("First chapter text.",)
    )
    second = make_analysis_section(
        section_id="section-2", title="Chapter 2", order=1, texts=("Second chapter text.",)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((first, second)))
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-1"))
    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2"))

    second_request = client.requests[1].input_text
    assert "Mr. Bennet" in second_request
    assert "First chapter text." not in second_request
    assert "Second chapter text." in second_request


async def test_repeated_analysis_skips_candidates_of_following_chapters() -> None:
    """Repeated analysis of a middle chapter does not see decisions of subsequent chapters."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text.",),
        )
        for number in (1, 2, 3)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    for section_id, analysis_id, source_text in (
        ("section-1", "analysis-1", "Mr. Bennet"),
        ("section-3", "analysis-3", "Mr. Darcy"),
    ):
        unit_of_work.analyses.save(
            BOOK_ID,
            make_chapter_analysis(
                id=analysis_id,
                section_id=section_id,
                candidates=(
                    make_analysis_candidate(
                        analysis_id=analysis_id, source_text=source_text
                    ),
                ),
            ),
        )
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2"))

    request = client.requests[0].input_text
    assert "Mr. Bennet" in request
    assert "Mr. Darcy" not in request


@pytest.mark.parametrize(
    ("status", "expected", "absent"),
    [
        pytest.param(
            TermStatus.APPROVED,
            "- Duskmere [place] -> Даскмер (несклоняемое)",
            "Даскмир",
            id="corrected",
        ),
        pytest.param(
            TermStatus.LOCKED,
            "- Duskmere [place] -> Даскмер (несклоняемое)",
            "Даскмир",
            id="locked",
        ),
        pytest.param(
            TermStatus.SUGGESTED,
            "- Duskmere [place] -> Даскмир",
            "Даскмер",
            id="skipped",
        ),
    ],
)
async def test_user_decision_outranks_model_candidate_in_next_chapter(
    status: TermStatus, expected: str, absent: str
) -> None:
    """Next chapter analysis sees user decisions rather than model candidates (D-1).

    A skipped item is not considered a decision: the chapter gets the model candidate.
    """
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text.",),
        )
        for number in (1, 2)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            candidates=(
                make_analysis_candidate(
                    source_text="Duskmere",
                    suggested_translation="Даскмир",
                    type=ConceptType.PLACE,
                ),
            ),
        ),
    )
    unit_of_work.glossary.save(
        TermConcept(
            id="concept-1",
            book_id=BOOK_ID,
            type=ConceptType.PLACE,
            canonical_source_text="DUSKMERE",
            status=status,
            note="несклоняемое",
            variants=(
                TermVariant(
                    concept_id="concept-1",
                    language=TARGET_LANGUAGE,
                    text="Даскмер",
                    status=status,
                ),
            ),
        )
    )
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2"))

    request = client.requests[0].input_text
    assert expected in request
    assert absent not in request


def _decision(
    concept_id: str, source_text: str, translation: str, status: TermStatus
) -> TermConcept:
    """User decision on a book glossary item."""
    return TermConcept(
        id=concept_id,
        book_id=BOOK_ID,
        type=ConceptType.PERSON,
        canonical_source_text=source_text,
        status=status,
        variants=(
            TermVariant(
                concept_id=concept_id,
                language=TARGET_LANGUAGE,
                text=translation,
                status=status,
            ),
        ),
    )


async def test_repeated_analysis_sees_user_decisions_of_whole_book() -> None:
    """Repeated chapter analysis sees user decisions across the whole book (N-3).

    Decisions for concepts in this and subsequent chapters enter the analysis just like decisions
    from previous chapters; candidates of subsequent chapters without decisions and skipped items
    do not: the model variant of a subsequent chapter contributes nothing to this chapter.
    """
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text.",),
        )
        for number in (1, 2, 3)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    for number, candidates in (
        (1, (("Mr. Bennet", "мистер Беннет"),)),
        (2, (("Selene Row", "Селин Роу"), ("gatehouse", "привратный домик"))),
        (3, (("Mr. Darcy", "мистер Дарси"), ("Lady Grey", "леди Грэй"))),
    ):
        analysis_id = f"analysis-{number}"
        unit_of_work.analyses.save(
            BOOK_ID,
            make_chapter_analysis(
                id=analysis_id,
                section_id=f"section-{number}",
                candidates=tuple(
                    make_analysis_candidate(
                        analysis_id=analysis_id,
                        source_text=source_text,
                        suggested_translation=translation,
                    )
                    for source_text, translation in candidates
                ),
            ),
        )
    for concept in (
        _decision("concept-1", "Selene Row", "Селена Роу", TermStatus.LOCKED),
        _decision("concept-2", "gatehouse", "дом привратника", TermStatus.APPROVED),
        _decision("concept-3", "Lady Grey", "леди Грей", TermStatus.APPROVED),
        _decision("concept-4", "Mr. Darcy", "мистер Дарси", TermStatus.SUGGESTED),
    ):
        unit_of_work.glossary.save(concept)
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2"))

    request = client.requests[0].input_text
    assert "Mr. Bennet" in request
    assert "Селена Роу" in request
    assert "дом привратника" in request
    assert "леди Грей" in request
    assert "Селин Роу" not in request
    assert "привратный домик" not in request
    assert "Mr. Darcy" not in request
    assert request.index("Селена Роу") < request.index("Mr. Bennet")


async def test_glossary_note_in_analysis_context_drops_rejected_variant() -> None:
    """A glossary item note in the analysis context drops a rejected variant.

    Third live run: in the context of chapter 2 analysis, the note for `the marsh gate`
    referred to the rejected form ("Daskmir"), even though the user decision for Duskmere was "Daskmer".
    The decision for another concept and its inflections are preserved.
    """
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text.",),
        )
        for number in (1, 2)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            candidates=(
                make_analysis_candidate(
                    source_text="Duskmere",
                    suggested_translation="Даскмир",
                    type=ConceptType.PLACE,
                ),
                make_analysis_candidate(
                    source_text="the marsh gate",
                    suggested_translation="болотные ворота",
                    type=ConceptType.PLACE,
                ),
                make_analysis_candidate(
                    source_text="boardwalk",
                    suggested_translation="деревянный настил",
                    type=ConceptType.TERM,
                    note="Настил ведёт к Даскмиру. Держать одно слово.",
                ),
            ),
        ),
    )
    unit_of_work.glossary.save(
        _decision("concept-1", "Duskmere", "Даскмер", TermStatus.APPROVED)
    )
    unit_of_work.glossary.save(
        TermConcept(
            id="concept-2",
            book_id=BOOK_ID,
            type=ConceptType.PLACE,
            canonical_source_text="the marsh gate",
            status=TermStatus.APPROVED,
            note=(
                "Сторожевой пост при въезде в Даскмир. Множественное число, "
                "склоняется. Стоит у стен Даскмера."
            ),
            variants=(
                TermVariant(
                    concept_id="concept-2",
                    language=TARGET_LANGUAGE,
                    text="болотные ворота",
                    status=TermStatus.APPROVED,
                ),
            ),
        )
    )
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2"))

    request = client.requests[0].input_text
    assert "Даскмир" not in request
    assert (
        "- the marsh gate [place] -> болотные ворота "
        "(Множественное число, склоняется. Стоит у стен Даскмера.)"
    ) in request
    assert "- boardwalk [term] -> деревянный настил (Держать одно слово.)" in request
    assert "- Duskmere [person] -> Даскмер" in request


async def test_chapter_without_text_is_analysed_without_the_model() -> None:
    """A chapter without text blocks does not call the model, but receives a complete analysis."""
    section = make_analysis_section(
        blocks=(
            make_analysis_block(
                section_id="section-1", order=0, text="", type=BlockType.IMAGE
            ),
        )
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((section,)))
    client = FakeLLMClient.always(analysis_reply())
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert client.call_count == 0
    assert result.attempts == 0
    assert result.analysis.coverage_status is CoverageStatus.COMPLETE
    assert result.analysis.treatment_plan.items == ()
    assert unit_of_work.analyses.get_for_section(BOOK_ID, "section-1") is not None


async def test_missing_block_in_plan_is_retried_with_repair_instruction() -> None:
    """An incomplete plan is rejected and retried with a clarifying repair instruction (ALG-05)."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (analysis_reply(aliases=("b1",)), analysis_reply(aliases=("b1", "b2")))
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.attempts == 2
    assert client.call_count == 2
    repaired = client.requests[1].instructions
    assert repaired.startswith(client.requests[0].instructions)
    assert "b2" in repaired
    assert result.analysis.coverage_status is CoverageStatus.COMPLETE


async def test_a2_analysis_retries_when_summary_does_not_match_source_and_translation_languages() -> None:
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (
            analysis_reply(chapter_summary="История продолжается.\n\nГерои уходят."),
            analysis_reply(chapter_summary="The story continues.\n\nИстория продолжается."),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client, reader_level="a2")

    result = await scenario(command())

    assert result.attempts == 2
    assert client.call_count == 2
    assert result.analysis.chapter_summary == "The story continues.\n\nИстория продолжается."


async def test_a2_analysis_retries_when_first_summary_paragraph_is_in_another_latin_language() -> None:
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (
            analysis_reply(chapter_summary="La historia continúa.\n\nИстория продолжается."),
            analysis_reply(chapter_summary="The story continues.\n\nИстория продолжается."),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client, reader_level="a2")

    result = await scenario(command())

    assert result.attempts == 2
    assert client.call_count == 2
    assert result.analysis.chapter_summary == "The story continues.\n\nИстория продолжается."


async def test_unknown_block_in_plan_is_rejected() -> None:
    """A plan with an unknown block is rejected and triggers a retry."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (
            analysis_reply(aliases=("b1", "b2", "b3")),
            analysis_reply(aliases=("b1", "b2")),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.attempts == 2
    assert "b3" in client.requests[1].instructions


async def test_duplicated_block_in_plan_is_rejected() -> None:
    """Duplicate treatment for a block is an invalid plan and is also retried."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (
            analysis_reply(aliases=("b1", "b1", "b2")),
            analysis_reply(aliases=("b1", "b2")),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    assert (await scenario(command())).attempts == 2


async def test_mixed_script_glossary_variant_is_rejected() -> None:
    """A glossary variant with mixed scripts is rejected and retried (ALG-04)."""
    unit_of_work = two_block_book()
    mixed = {
        "source_text": "PGLAF",
        "type": "organization",
        "suggested_translation": "PГЛАФ",
        "confidence": 0.9,
        "model_flagged_review": False,
        "note": None,
    }
    client = FakeLLMClient.replying(
        (
            analysis_reply(candidates=[mixed]),
            analysis_reply(candidates=[{**mixed, "suggested_translation": "PGLAF"}]),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.attempts == 2
    assert "PГЛАФ" in client.requests[1].instructions


async def test_transient_error_is_retried_after_backoff() -> None:
    """A transient error is retried, respecting the policy backoff delay (ALG-05)."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (LLMTransportError("429 Too Many Requests"), analysis_reply())
    )
    scenario, delays = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.attempts == 2
    assert delays == [1.0]


async def test_exhausted_attempts_leave_no_saved_analysis() -> None:
    """When attempts are exhausted, the scenario raises an error without saving an analysis."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.always(LLMTransportError("провайдер недоступен"))
    scenario, _ = make_analyze_chapter(unit_of_work, client, max_attempts=2)

    with pytest.raises(LLMTransportError):
        await scenario(command())

    assert client.call_count == 2
    assert unit_of_work.analyses.get_for_section(BOOK_ID, "section-1") is None


async def test_result_reports_token_usage_of_successful_call() -> None:
    """The result reports call token usage for cost estimation (PRD § 29.1)."""
    unit_of_work = two_block_book()
    usage = TokenUsage(prompt_tokens=1200, completion_tokens=300)
    scenario, _ = make_analyze_chapter(
        unit_of_work, FakeLLMClient.always(analysis_reply(usage=usage))
    )

    assert (await scenario(command())).usage == usage


async def test_unknown_book_is_rejected() -> None:
    """Analyzing a book not present in storage is rejected."""
    scenario, _ = make_analyze_chapter(FakeUnitOfWork(), FakeLLMClient.always(analysis_reply()))

    with pytest.raises(AnalyzeChapterError):
        await scenario(command())


async def test_unknown_section_is_rejected() -> None:
    """Analyzing a chapter not present in the book is rejected."""
    scenario, _ = make_analyze_chapter(two_block_book(), FakeLLMClient.always(analysis_reply()))

    with pytest.raises(AnalyzeChapterError):
        await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-9"))
