"""Chapter analysis scenario: request, response validation, and persistence (P3.T3, PRD § 14)."""

from __future__ import annotations

import pytest
from analysis_fakes import (
    BOOK_ID,
    CONFIDENCE_THRESHOLD,
    TARGET_LANGUAGE,
    FakeTaskQueue,
    FakeUnitOfWork,
    analysis_reply,
    make_analysis_block,
    make_analysis_candidate,
    make_analysis_document,
    make_analysis_section,
    make_analyze_chapter,
    make_book_record,
    make_chapter_analysis,
)

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.context_builder import ContextBuilder
from linguary.application.ports.llm import LLMResponseFormatError, LLMTransportError
from types import SimpleNamespace

from linguary.application.use_cases.analyze_chapter import (
    AnalyzeChapterCommand,
    AnalyzeChapterError,
    ChapterAnalysisSequence,
)
from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookStatus,
    ConceptType,
    CoverageStatus,
    ProcessingTask,
    TaskStatus,
    TaskType,
    TermConcept,
    TermStatus,
    TermVariant,
    TokenUsage,
)
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy


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
        section_id="section-2", title="Chapter 2", order=1, texts=("Second chapter text with Mr. Bennet.",)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((first, second)))
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-1"))
    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2"))

    second_request = client.requests[1].input_text
    assert "Mr. Bennet" in second_request
    assert "First chapter text." not in second_request
    assert "Second chapter text with Mr. Bennet." in second_request


async def test_repeated_analysis_skips_candidates_of_following_chapters() -> None:
    """Repeated analysis of a middle chapter does not see decisions of subsequent chapters."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text with Mr. Bennet, Selene Row, gatehouse, and Lady Grey.",),
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
            texts=(f"Chapter {number} text with Duskmere, the marsh gate, and boardwalk.",),
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


async def test_first_request_after_reanalysis_contains_saved_decisions() -> None:
    section = make_analysis_section(texts=("First paragraph with Duskmere.", "Second paragraph with Ravenwood."))
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((section,)))
    unit_of_work.glossary.save(_decision('approved-1', 'Duskmere', 'Даскмер', TermStatus.APPROVED))
    unit_of_work.glossary.save(_decision('suggested-1', 'Ravenwood', 'Рейвенвуд', TermStatus.SUGGESTED))
    client = FakeLLMClient.always(analysis_reply())
    scenario, _ = make_analyze_chapter(unit_of_work, client)
    scenario._context_builder = ContextBuilder(name_translation_policy=NameTranslationPolicy(1, NameTranslationStrategy.MEANING))

    await scenario(command())

    request = client.requests[0]
    assert request.name_translation_policy_version == 1
    assert 'name_translation_assessment' in request.instructions
    assert 'Даскмер' in request.input_text
    assert 'Принятые решения пользователя' in request.input_text
    assert 'Рейвенвуд' in request.input_text
    assert 'Необязательные подсказки' in request.input_text
    assert request.input_text.index('Даскмер') < request.input_text.index('Рейвенвуд')


async def test_suggested_and_model_candidates_are_not_accepted() -> None:
    sections = (
        make_analysis_section(section_id='section-1', order=0, texts=('First chapter.',)),
        make_analysis_section(section_id='section-2', order=1, texts=('Second chapter with Ravenwood and Stonebridge.',)),
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(BOOK_ID, make_chapter_analysis(section_id='section-1', candidates=(
        make_analysis_candidate(source_text='Ravenwood', suggested_translation='Рейвенвуд'),
    )))
    unit_of_work.glossary.save(_decision('approved-1', 'Ravenwood', 'Воронолесье', TermStatus.APPROVED))
    unit_of_work.glossary.save(_decision('suggested-1', 'Stonebridge', 'Стоунбридж', TermStatus.SUGGESTED))
    client = FakeLLMClient.always(analysis_reply(aliases=('b1',)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)
    scenario._context_builder = ContextBuilder(name_translation_policy=NameTranslationPolicy(1, NameTranslationStrategy.MEANING))

    await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id='section-2'))

    request = client.requests[0].input_text
    accepted, optional = request.split('Необязательные подсказки', 1)
    assert 'Воронолесье' in accepted
    assert 'Рейвенвуд' not in request
    assert 'Стоунбридж' in optional


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
            texts=(f"Chapter {number} text with Mr. Bennet, Selene Row, gatehouse, and Lady Grey.",),
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
            texts=(f"Chapter {number} text with Duskmere, the marsh gate, and boardwalk.",),
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
    assert client.requests[1].instructions == client.requests[0].instructions
    assert any("b2" in instr for instr in client.requests[1].repair_instructions)
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
    assert client.requests[1].instructions == client.requests[0].instructions
    assert any("b3" in instr for instr in client.requests[1].repair_instructions)


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
    assert client.requests[1].instructions == client.requests[0].instructions
    assert any("PГЛАФ" in instr for instr in client.requests[1].repair_instructions)


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


async def test_front_matter_section_analyzed_locally_without_llm() -> None:
    """Front matter sections receive local analysis with READ_AS_IS without invoking LLM (P1.T1, AC-003, AC-004)."""
    front_sec = make_analysis_section(
        section_id="section-title",
        title="Title Page",
        order=0,
        texts=("Pride and Prejudice", "Jane Austen"),
    )
    story_sec = make_analysis_section(
        section_id="section-ch1",
        title="Chapter 1",
        order=1,
        texts=("It is a truth universally acknowledged...",),
    )
    doc = make_analysis_document((front_sec, story_sec))
    unit_of_work = FakeUnitOfWork(document=doc)
    client = FakeLLMClient.always(analysis_reply())
    scenario, _ = make_analyze_chapter(
        unit_of_work, client,
    )
    # Inject context_builder with content-plan-v3
    scenario._context_builder = ContextBuilder(content_plan_profile="content-plan-v3")

    result = await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-title"))

    assert client.call_count == 0
    assert result.analysis is not None
    assert result.analysis.chapter_summary == ""
    assert result.analysis.before_you_read == ()
    assert result.analysis.glossary_candidates == ()
    assert result.analysis.translation_instructions == ""
    for item in result.analysis.treatment_plan.items:
        assert item.treatment is BlockTreatment.READ_AS_IS
    assert result.analysis.content_plan is not None
    assert result.analysis.content_plan.profile == "content-plan-v3"
    assert result.analysis.content_plan.difficulties == ()


async def test_chapter_analysis_sequence_enqueues_all_chapters_in_parallel() -> None:
    """Starting sequence queues tasks for all unanalyzed chapters at once (P1.T2, AC-001)."""
    front_sec = make_analysis_section(section_id="section-title", title="Title Page", order=0, texts=("Title",))
    ch1 = make_analysis_section(section_id="section-1", title="Chapter 1", order=1, texts=("Ch 1 text",))
    ch2 = make_analysis_section(section_id="section-2", title="Chapter 2", order=2, texts=("Ch 2 text",))
    doc = make_analysis_document((front_sec, ch1, ch2))
    unit_of_work = FakeUnitOfWork(document=doc, book=make_book_record(BookStatus.IMPORTED))
    queue = FakeTaskQueue()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
    )

    first_task = await sequence.start(BOOK_ID)

    assert first_task is not None
    assert first_task.scope_id == "section-title"
    assert len(queue.tasks) == 3
    assert [t.scope_id for t in queue.tasks] == ["section-title", "section-1", "section-2"]
    book = unit_of_work.books.get(BOOK_ID)
    assert book is not None and book.status is BookStatus.ANALYZING


async def test_chapter_analysis_sequence_auto_approve_glossary_success() -> None:
    """Clean glossary candidates are auto-approved, transitioning book to READY_TO_PROCESS (P1.T2)."""
    ch1 = make_analysis_section(section_id="section-1", title="Chapter 1", order=0, texts=("Ch 1",))
    ch2 = make_analysis_section(section_id="section-2", title="Chapter 2", order=1, texts=("Ch 2",))
    doc = make_analysis_document((ch1, ch2))
    unit_of_work = FakeUnitOfWork(document=doc, book=make_book_record(BookStatus.ANALYZING))
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": True,
                "glossary_confidence_threshold": 0.7,
                "glossary_review_min_mentions": 1,
                "glossary_review_min_chapters": 1,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue = FakeTaskQueue()
    reply = analysis_reply(
        aliases=("b1",),
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Даскмер",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    client = FakeLLMClient.always(reply)
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
    )

    tasks = await sequence._enqueue_all(BOOK_ID)
    assert len(tasks) == 2
    await sequence.run_task(tasks[0].start())
    assert unit_of_work.books.get(BOOK_ID).status is BookStatus.ANALYZING

    await sequence.run_task(tasks[1].start())
    assert unit_of_work.books.get(BOOK_ID).status is BookStatus.READY_TO_PROCESS
    glossary_items = unit_of_work.glossary.list_for_book(BOOK_ID)
    assert len(glossary_items) == 1
    assert glossary_items[0].canonical_source_text == "Duskmere"
    assert glossary_items[0].status == TermStatus.APPROVED


async def test_chapter_analysis_sequence_auto_approve_glossary_skips_conflicts() -> None:
    """Conflicting candidates flagged for review remain unapproved in GLOSSARY_REVIEW (P1.T2, P2.T1)."""
    ch1 = make_analysis_section(section_id="section-1", title="Chapter 1", order=0, texts=("Ch 1",))
    ch2 = make_analysis_section(section_id="section-2", title="Chapter 2", order=1, texts=("Ch 2",))
    doc = make_analysis_document((ch1, ch2))
    unit_of_work = FakeUnitOfWork(document=doc, book=make_book_record(BookStatus.ANALYZING))
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": True,
                "glossary_confidence_threshold": 0.7,
                "glossary_review_min_mentions": 1,
                "glossary_review_min_chapters": 1,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue = FakeTaskQueue()
    reply1 = analysis_reply(
        aliases=("b1",),
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Даскмер",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    reply2 = analysis_reply(
        aliases=("b1",),
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Сумеречье",
                "confidence": 0.9,
                "model_flagged_review": True,
                "note": None,
            }
        ],
    )
    client = FakeLLMClient.replying((reply1, reply2))
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
    )

    tasks = await sequence._enqueue_all(BOOK_ID)
    await sequence.run_task(tasks[0].start())
    await sequence.run_task(tasks[1].start())

    assert unit_of_work.books.get(BOOK_ID).status is BookStatus.GLOSSARY_REVIEW
    glossary_items = unit_of_work.glossary.list_for_book(BOOK_ID)
    assert len(glossary_items) == 0


async def test_chapter_analysis_sequence_auto_approve_disabled_leaves_in_glossary_review() -> None:
    """When auto_approve_glossary is disabled, book remains in GLOSSARY_REVIEW (P1.T2)."""
    ch1 = make_analysis_section(section_id="section-1", title="Chapter 1", order=0, texts=("Ch 1",))
    doc = make_analysis_document((ch1,))
    unit_of_work = FakeUnitOfWork(document=doc, book=make_book_record(BookStatus.ANALYZING))
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": False,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue = FakeTaskQueue()
    reply = analysis_reply(
        aliases=("b1",),
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Даскмер",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    client = FakeLLMClient.always(reply)
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
    )

    tasks = await sequence._enqueue_all(BOOK_ID)
    await sequence.run_task(tasks[0].start())

    assert unit_of_work.books.get(BOOK_ID).status is BookStatus.GLOSSARY_REVIEW
    assert len(unit_of_work.glossary.list_for_book(BOOK_ID)) == 0


async def test_truncated_chapter_analysis_splits_blocks_dynamically() -> None:
    """When analysis response is truncated, blocks are dynamically split into fragments (AC-005)."""
    section = make_analysis_section(
        texts=("Paragraph 1.", "Paragraph 2.", "Paragraph 3.", "Paragraph 4.")
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((section,)))
    client = FakeLLMClient.replying(
        (
            FakeReply(content="truncated json...", truncated=True),
            analysis_reply(aliases=("b1", "b2")),
            FakeReply(content="truncated json fragment...", truncated=True),
            analysis_reply(aliases=("b1",)),
            analysis_reply(aliases=("b1",)),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.analysis.coverage_status is CoverageStatus.COMPLETE
    assert len(result.analysis.treatment_plan.block_ids) == 4
    stored = unit_of_work.analyses.get_for_section(BOOK_ID, "section-1")
    assert stored is not None
    assert stored.coverage_status is CoverageStatus.COMPLETE
    assert len(stored.treatment_plan.block_ids) == 4


async def test_truncated_single_block_analysis_raises_give_up() -> None:
    """When a single-block fragment is truncated, splitting is impossible and an error is raised (AC-005)."""
    section = make_analysis_section(texts=("Single paragraph.",))
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((section,)))
    client = FakeLLMClient.always(FakeReply(content="truncated json...", truncated=True))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    with pytest.raises(LLMResponseFormatError) as exc_info:
        await scenario(command())

    assert exc_info.value.truncated is True


async def test_parallel_batch_analysis_ignores_unfinished_peer_chapter_candidates() -> None:
    """Parallel batch analysis ignores unconfirmed model candidates when peer chapter tasks are active, keeping user decisions."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text with Mr. Bennet and Duskmere.",),
        )
        for number in (1, 2, 3)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            id="analysis-1",
            section_id="section-1",
            candidates=(
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Mr. Bennet",
                    suggested_translation="мистер Беннет",
                ),
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Duskmere",
                    suggested_translation="Даскмир",
                    type=ConceptType.PLACE,
                ),
            ),
        ),
    )
    unit_of_work.glossary.save(
        _decision("concept-1", "Duskmere", "Даскмер", TermStatus.APPROVED)
    )
    tasks = (
        ProcessingTask(
            id="task-1",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-1",
            status=TaskStatus.SUCCEEDED,
            attempts=1,
        ),
        ProcessingTask(
            id="task-2",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-2",
            status=TaskStatus.RUNNING,
            attempts=1,
        ),
        ProcessingTask(
            id="task-3",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-3",
            status=TaskStatus.RUNNING,
            attempts=1,
        ),
    )
    unit_of_work.tasks = SimpleNamespace(list_for_book=lambda _: tasks)  # type: ignore[attr-defined]
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(
        AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2", task_id="task-2")
    )

    request_text = client.requests[0].input_text
    assert "мистер Беннет" not in request_text
    assert "Даскмер" in request_text


async def test_sequential_analysis_retains_previous_chapter_candidates() -> None:
    """When no other ANALYZE_CHAPTER tasks are pending or running, previous chapter candidates are retained."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text with Mr. Bennet.",),
        )
        for number in (1, 2)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            id="analysis-1",
            section_id="section-1",
            candidates=(
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Mr. Bennet",
                    suggested_translation="мистер Беннет",
                ),
            ),
        ),
    )
    tasks = (
        ProcessingTask(
            id="task-1",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-1",
            status=TaskStatus.SUCCEEDED,
            attempts=1,
        ),
        ProcessingTask(
            id="task-2",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-2",
            status=TaskStatus.RUNNING,
            attempts=1,
        ),
    )
    unit_of_work.tasks = SimpleNamespace(list_for_book=lambda _: tasks)  # type: ignore[attr-defined]
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    await scenario(
        AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2", task_id="task-2")
    )

    request_text = client.requests[0].input_text
    assert "мистер Беннет" in request_text


async def test_auto_approve_glossary_computes_mention_counter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-approving glossary computes mention_counter and loads analyses once across both review passes (P1.T2)."""
    import linguary.application.mention_counter as mention_counter_module

    original_mention_counter = mention_counter_module.mention_counter
    mention_counter_calls = 0

    def counting_mention_counter(document):
        nonlocal mention_counter_calls
        mention_counter_calls += 1
        return original_mention_counter(document)

    monkeypatch.setattr(
        mention_counter_module, "mention_counter", counting_mention_counter
    )

    ch1 = make_analysis_section(
        section_id="section-1", title="Chapter 1", order=0, texts=("Ch 1 Duskmere",)
    )
    ch2 = make_analysis_section(
        section_id="section-2", title="Chapter 2", order=1, texts=("Ch 2 Duskmere",)
    )
    doc = make_analysis_document((ch1, ch2))
    unit_of_work = FakeUnitOfWork(
        document=doc, book=make_book_record(BookStatus.ANALYZING)
    )
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "checks": {
                "auto_approve_glossary": True,
                "glossary_confidence_threshold": 0.7,
                "glossary_review_min_mentions": 1,
                "glossary_review_min_chapters": 1,
            }
        },
    )
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)  # type: ignore[attr-defined]
    queue = FakeTaskQueue()
    reply = analysis_reply(
        aliases=("b1",),
        candidates=[
            {
                "source_text": "Duskmere",
                "type": "place",
                "suggested_translation": "Даскмер",
                "confidence": 0.9,
                "model_flagged_review": False,
                "note": None,
            }
        ],
    )
    client = FakeLLMClient.always(reply)
    analyze_chapter, _ = make_analyze_chapter(unit_of_work, client)
    sequence = ChapterAnalysisSequence(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        analyze_chapter=analyze_chapter,
    )

    tasks = await sequence._enqueue_all(BOOK_ID)
    await sequence.run_task(tasks[0].start())

    list_for_book_calls = 0
    original_list_for_book = type(unit_of_work.analyses).list_for_book

    def counting_list_for_book(self, book_id: str):
        nonlocal list_for_book_calls
        list_for_book_calls += 1
        return original_list_for_book(self, book_id)

    monkeypatch.setattr(
        type(unit_of_work.analyses), "list_for_book", counting_list_for_book
    )

    await sequence.run_task(tasks[1].start())

    assert unit_of_work.books.get(BOOK_ID).status is BookStatus.READY_TO_PROCESS
    assert mention_counter_calls == 1
    # 1 call inside AnalyzeChapter(command) and 1 shared call in _try_complete_analysis
    # reused across deduplicate_book_before_you_read and auto_approve.
    assert list_for_book_calls == 2


async def test_process_chapter_reuses_source_texts_and_defers_reconcile_until_completion() -> None:
    """ProcessChapter caches source_texts and pronunciation_help_enabled in _load and passes reconcile=False per unit (P1.T2)."""
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.application.use_cases.process_chapter import (
        ProcessChapter,
        ProcessChapterCommand,
        ProcessChapterResult,
    )
    from linguary.domain import GlossarySnapshot, TranslationUnit, TreatmentPlan
    from processing_fakes import make_validator

    ch1 = make_analysis_section(
        section_id="section-1",
        title="Chapter 1",
        order=0,
        texts=("Read as is paragraph.", "Paragraph to translate."),
    )
    ch2 = make_analysis_section(
        section_id="section-2",
        title="Chapter 2",
        order=1,
        texts=("Another chapter paragraph.",),
    )
    doc = make_analysis_document((ch1, ch2))
    b1, b2 = ch1.ordered_blocks

    plan = TreatmentPlan.of_treatments(
        ch1.id,
        (
            (b1.id, BlockTreatment.READ_AS_IS),
            (b2.id, BlockTreatment.TRANSLATE),
        ),
    )
    analysis = make_chapter_analysis(id="analysis-1", section_id=ch1.id)
    analysis = SimpleNamespace(
        id=analysis.id,
        section_id=ch1.id,
        treatment_plan=plan,
        content_plan=None,
        scenes=(),
        translation_instructions="",
        chapter_summary="",
        before_you_read=(),
        glossary_candidates=(),
        ensure_ready_for_processing=lambda: None,
    )

    unit_of_work = FakeUnitOfWork(
        document=doc, book=make_book_record(BookStatus.PROCESSING)
    )
    unit_of_work.analyses.analyses[(BOOK_ID, ch1.id)] = analysis  # type: ignore[assignment]
    run = SimpleNamespace(
        id="run-1",
        book_id=BOOK_ID,
        glossary_snapshot=GlossarySnapshot(),
        is_active=True,
    )
    unit_of_work.runs.runs["run-1"] = run  # type: ignore[assignment]
    unit_of_work.results = SimpleNamespace(  # type: ignore[attr-defined]
        list_for_section=lambda _run_id, _sec_id: (),
    )

    latest_conditions_calls = 0
    conditions = SimpleNamespace(
        id="cond-1",
        payload={
            "pronunciation_help_enabled": False,
            "models": {"processing": "gpt-test"},
        },
    )
    root = SimpleNamespace(
        id="root-1",
        conditions_id="cond-1",
        policy_version=CURRENT_TRANSLATION_POLICY,
        manifest={"ordered_blocks": (b1.id, b2.id)},
    )

    class FakeWork:
        def latest_conditions(self, book_id: str):
            nonlocal latest_conditions_calls
            latest_conditions_calls += 1
            return conditions

        def get_root(self, root_id: str):
            return root

        def get_conditions(self, conditions_id: str):
            return conditions

        def begin_recovery_attempt(self, root_id: str, scope_id: str, **kwargs):
            return SimpleNamespace(
                attempt_id="att-1",
                root_id=root_id,
                scope_id=scope_id,
                conditions_id="cond-1",
                policy_version=root.policy_version,
                prompt_hash="hash",
            )

        def transition_attempt(self, attempt_id: str, state: str, **kwargs) -> None:
            return None

        def get_binding(self, kind: str, entity_id: str):
            return None

    unit_of_work.work = FakeWork()  # type: ignore[attr-defined]

    process_chapter = ProcessChapter(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        llm_client=SimpleNamespace(),  # type: ignore[arg-type]
        retry_policy= SimpleNamespace(),  # type: ignore[arg-type]
        validator=make_validator(),
    )

    work = process_chapter._load(
        ProcessChapterCommand(book_id=BOOK_ID, section_id=ch1.id, task_id="task-1")
    )

    assert latest_conditions_calls == 1
    assert work.pronunciation_help_enabled is False
    assert dict(work.source_texts) == {
        b1.id: b1.text,
        b2.id: b2.text,
    }

    save_calls: list[dict[str, object]] = []
    request_source_texts: list[object] = []

    async def fake_request(context, **kwargs):
        request_source_texts.append(kwargs.get("source_texts"))
        return SimpleNamespace(
            results=(
                SimpleNamespace(block_id=b2.id),
            ),
            attempts=1,
            usage=None,
        )

    def fake_save(**kwargs) -> bool:
        save_calls.append(kwargs)
        return True

    process_chapter._call.request = fake_request  # type: ignore[method-assign]
    process_chapter._call.save = fake_save  # type: ignore[method-assign]

    unit_read_as_is = TranslationUnit.of_blocks(
        id="scope-1", blocks=(b1,), plan=plan
    )
    unit_translate = TranslationUnit.of_blocks(
        id="scope-2", blocks=(b2,), plan=plan
    )

    initial = ProcessChapterResult(section_id=ch1.id, run_id="run-1", units=2)
    after_first = await process_chapter._process_unit(
        unit_read_as_is,
        work=work,
        note_ids=lambda: "ln-ch01-001",
        result=initial,
        recovery=("root-1", "scope-1"),
    )
    after_second = await process_chapter._process_unit(
        unit_translate,
        work=work,
        note_ids=lambda: "ln-ch01-002",
        result=after_first,
        recovery=("root-1", "scope-2"),
    )

    # _process_unit must not call latest_conditions on each unit.
    assert latest_conditions_calls == 1
    assert after_second.processed_units == 2
    assert len(save_calls) == 2
    assert save_calls[0]["reconcile"] is False
    assert save_calls[1]["reconcile"] is False
    assert len(request_source_texts) == 1
    assert request_source_texts[0] is work.source_texts


async def test_sequential_analysis_accumulates_candidates_when_pending_tasks_exist() -> None:
    """At concurrency == 1, queued PENDING peer tasks do not disable candidate accumulation (P2.T1)."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text with Mr. Bennet and Duskmere.",),
        )
        for number in (1, 2, 3)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            id="analysis-1",
            section_id="section-1",
            candidates=(
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Mr. Bennet",
                    suggested_translation="мистер Беннет",
                ),
            ),
        ),
    )
    tasks = (
        ProcessingTask(
            id="task-1",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-1",
            status=TaskStatus.SUCCEEDED,
            attempts=1,
        ),
        ProcessingTask(
            id="task-2",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-2",
            status=TaskStatus.RUNNING,
            attempts=1,
        ),
        ProcessingTask(
            id="task-3",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-3",
            status=TaskStatus.PENDING,
            attempts=0,
        ),
    )
    unit_of_work.tasks = SimpleNamespace(list_for_book=lambda _: tasks)  # type: ignore[attr-defined]
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)
    scenario._max_concurrency = 1

    await scenario(
        AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-2", task_id="task-2")
    )

    request_text = client.requests[0].input_text
    assert "мистер Беннет" in request_text


async def test_parallel_analysis_keeps_last_chapter_in_parallel_batch_when_concurrency_gt_1() -> None:
    """When max_concurrency > 1, even the last chapter runs in parallel_batch mode without unconfirmed peer candidates (P2.T1)."""
    sections = tuple(
        make_analysis_section(
            section_id=f"section-{number}",
            title=f"Chapter {number}",
            order=number - 1,
            texts=(f"Chapter {number} text with Mr. Bennet and Duskmere.",),
        )
        for number in (1, 2, 3)
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document(sections))
    unit_of_work.analyses.save(
        BOOK_ID,
        make_chapter_analysis(
            id="analysis-1",
            section_id="section-1",
            candidates=(
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Mr. Bennet",
                    suggested_translation="мистер Беннет",
                ),
                make_analysis_candidate(
                    analysis_id="analysis-1",
                    source_text="Duskmere",
                    suggested_translation="Даскмир",
                    type=ConceptType.PLACE,
                ),
            ),
        ),
    )
    unit_of_work.glossary.save(
        _decision("concept-1", "Duskmere", "Даскмер", TermStatus.APPROVED)
    )
    tasks = (
        ProcessingTask(
            id="task-1",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-1",
            status=TaskStatus.SUCCEEDED,
            attempts=1,
        ),
        ProcessingTask(
            id="task-2",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-2",
            status=TaskStatus.SUCCEEDED,
            attempts=1,
        ),
        ProcessingTask(
            id="task-3",
            book_id=BOOK_ID,
            task_type=TaskType.ANALYZE_CHAPTER,
            scope_id="section-3",
            status=TaskStatus.RUNNING,
            attempts=1,
        ),
    )
    unit_of_work.tasks = SimpleNamespace(list_for_book=lambda _: tasks)  # type: ignore[attr-defined]
    client = FakeLLMClient.always(analysis_reply(aliases=("b1",)))
    scenario, _ = make_analyze_chapter(unit_of_work, client)
    scenario._max_concurrency = 2

    await scenario(
        AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-3", task_id="task-3")
    )

    request_text = client.requests[0].input_text
    assert "мистер Беннет" not in request_text
    assert "Даскмер" in request_text


async def test_analysis_post_validation_errors_use_block_aliases_and_expectation() -> None:
    """Post-validation errors in AnalyzeChapter replace internal block UUIDs with b1..bN aliases and attach expectation (P2.T1)."""
    unit_of_work = two_block_book()
    section = unit_of_work.documents.get(BOOK_ID).section("section-1")  # type: ignore[union-attr]
    block_1, block_2 = section.ordered_blocks
    client = FakeLLMClient.replying(
        (
            analysis_reply(
                aliases=("b1",),
                chapter_summary=f"Ошибочная сводка с внутренним ID {block_1.id}.",
            ),
            analysis_reply(
                aliases=("b1", "b2"),
                chapter_summary="The story continues.\n\nИстория продолжается.",
            ),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.attempts == 2
    repair_text = "\n".join(client.requests[1].repair_instructions)
    assert "['b2']" in repair_text
    assert block_1.id not in repair_text
    assert block_2.id not in repair_text
    assert "- b1" in repair_text
    assert "- b2" in repair_text
    assert "Целевой язык результата — ru." in repair_text


async def test_request_analysis_reuses_validated_content_plan_without_triple_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_content_plan returns ContentPlan and AnalyzeChapter reuses it in whole-chapter and fragment paths without triple parsing (P2.T1)."""
    import linguary.application.analysis_merge as analysis_merge_mod
    import linguary.application.use_cases.analyze_chapter as analyze_chapter_mod

    original_parse = analysis_merge_mod.content_plan_from_response
    parse_calls = 0

    def counting_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(
        analysis_merge_mod, "content_plan_from_response", counting_parse
    )
    monkeypatch.setattr(
        analyze_chapter_mod, "content_plan_from_response", counting_parse
    )

    content_plan_payload = {
        "version": 3,
        "profile": "content-plan-v3",
        "difficulties": [],
        "before_you_read_evidence": [],
        "context_facts": [],
    }

    # 1. Whole-chapter analysis (1 LLM call -> 1 content_plan_from_response call)
    unit_of_work = two_block_book()
    reply_whole = analysis_reply(
        aliases=("b1", "b2"),
        before_you_read=[],
        content_plan=content_plan_payload,
    )
    client_whole = FakeLLMClient.always(reply_whole)
    scenario_whole, _ = make_analyze_chapter(unit_of_work, client_whole)
    scenario_whole._context_builder = ContextBuilder(
        content_plan_profile="content-plan-v3"
    )

    result_whole = await scenario_whole(command())
    assert result_whole.analysis.content_plan is not None
    assert result_whole.analysis.content_plan.version == 3
    assert parse_calls == 1

    # 2. Fragment-based analysis (2 fragments -> 2 content_plan_from_response calls)
    parse_calls = 0
    unit_of_work_frag = two_block_book()
    reply_frag = analysis_reply(
        aliases=("b1",),
        before_you_read=[],
        content_plan=content_plan_payload,
    )
    client_frag = FakeLLMClient.always(reply_frag)
    scenario_frag, _ = make_analyze_chapter(
        unit_of_work_frag, client_frag, max_fragment_blocks=1
    )
    scenario_frag._context_builder = ContextBuilder(
        content_plan_profile="content-plan-v3"
    )

    result_frag = await scenario_frag(command())
    assert result_frag.analysis.content_plan is not None
    assert result_frag.analysis.content_plan.version == 3
    assert parse_calls == 2


async def test_analyze_chapter_preserves_attempts_and_usage_on_truncated_fallback_and_split() -> None:
    """Truncated whole-chapter fallback and recursive fragment split preserve attempts and token usage, and _to_aliases replaces IDs in a single pass (P3.T1)."""
    from linguary.application.use_cases.analyze_chapter import _to_aliases

    section = make_analysis_section(
        texts=("Paragraph 1.", "Paragraph 2.", "Paragraph 3.", "Paragraph 4.")
    )
    unit_of_work = FakeUnitOfWork(document=make_analysis_document((section,)))
    client = FakeLLMClient.replying(
        (
            LLMResponseFormatError(
                "truncated whole chapter",
                truncated=True,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            ),
            analysis_reply(
                aliases=("b1", "b2"),
                usage=TokenUsage(prompt_tokens=60, completion_tokens=30),
            ),
            LLMResponseFormatError(
                "truncated second fragment",
                truncated=True,
                usage=TokenUsage(prompt_tokens=70, completion_tokens=40),
            ),
            analysis_reply(
                aliases=("b1",),
                usage=TokenUsage(prompt_tokens=40, completion_tokens=20),
            ),
            analysis_reply(
                aliases=("b1",),
                usage=TokenUsage(prompt_tokens=45, completion_tokens=25),
            ),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert client.call_count == 5
    assert result.attempts == 5
    assert result.usage == TokenUsage(prompt_tokens=315, completion_tokens=165)

    fake_context = SimpleNamespace(
        block_ids_by_alias={"b1": "block-1", "b2": "block-10"}
    )
    aliased = _to_aliases(
        fake_context,  # type: ignore[arg-type]
        "не покрывает блоки: ['block-10', 'block-1']",
    )
    assert aliased == "не покрывает блоки: ['b2', 'b1']"







