"""Fallback scenario for a long chapter in the chapter analysis workflow (P3.T4, PRD § 14, ALG-03)."""

from __future__ import annotations

from typing import Any

import pytest
from analysis_fakes import (
    BOOK_ID,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    FakeUnitOfWork,
    analysis_reply,
    make_analysis_document,
    make_analysis_section,
    make_analyze_chapter,
)

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.context_builder import (
    chapter_analysis_instructions,
    chapter_fragment_instructions,
)
from linguary.application.ports.llm import LLMTransportError
from linguary.application.use_cases.analyze_chapter import AnalyzeChapterCommand
from linguary.domain import CoverageStatus, TokenUsage

#: A chapter with four paragraphs: with a two-block limit it splits into two fragments.
CHAPTER_TEXTS = (
    "Paragraph one.",
    "Paragraph two.",
    "Paragraph three.",
    "Paragraph four.",
)


def four_block_book() -> FakeUnitOfWork:
    """A book with one chapter of four paragraphs."""
    section = make_analysis_section(texts=CHAPTER_TEXTS)
    return FakeUnitOfWork(document=make_analysis_document((section,)))


def command() -> AnalyzeChapterCommand:
    """Command to analyze the single chapter of the book."""
    return AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-1")


def fragment_reply(**overrides: Any) -> FakeReply:
    """Model response for a two-block fragment."""
    return analysis_reply(aliases=("b1", "b2"), **overrides)


async def test_chapter_within_the_limit_is_analysed_by_one_request() -> None:
    """A chapter within the splitting threshold is sent to the model in a single request."""
    unit_of_work = four_block_book()
    client = FakeLLMClient.always(analysis_reply(aliases=("b1", "b2", "b3", "b4")))
    scenario, _ = make_analyze_chapter(unit_of_work, client, max_fragment_blocks=4)

    result = await scenario(command())

    assert client.call_count == 1
    assert client.requests[0].instructions == chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    )
    assert result.analysis.coverage_status is CoverageStatus.COMPLETE


async def test_long_chapter_is_analysed_by_fragments_and_merged() -> None:
    """A long chapter is sent to the model in fragments and merged into a single analysis."""
    unit_of_work = four_block_book()
    section = unit_of_work.documents.get(BOOK_ID).section("section-1")  # type: ignore[union-attr]
    client = FakeLLMClient.replying(
        (
            fragment_reply(chapter_summary="Mr. Bennet stays silent."),
            fragment_reply(chapter_summary="Bingley takes the house."),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client, max_fragment_blocks=2)

    analysis = (await scenario(command())).analysis

    assert client.call_count == 2
    assert analysis.coverage_status is CoverageStatus.COMPLETE
    assert analysis.treatment_plan.block_ids == tuple(
        block.id for block in section.ordered_blocks
    )
    assert analysis.chapter_summary == "Bingley takes the house."
    assert unit_of_work.analyses.get_for_section(BOOK_ID, "section-1") == analysis


async def test_second_fragment_request_contains_previous_findings() -> None:
    client = FakeLLMClient.replying((
        fragment_reply(chapter_summary="First event.", translation_instructions="Сохранять иронию."),
        fragment_reply(chapter_summary="First event. Second event."),
    ))
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=2)
    await scenario(command())
    assert "Уже найдено в предыдущих частях главы" in client.requests[1].input_text
    assert "First event." in client.requests[1].input_text
    assert "Сохранять иронию." in client.requests[1].input_text


async def test_first_fragment_request_has_no_findings() -> None:
    client = FakeLLMClient.always(fragment_reply())
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=2)
    await scenario(command())
    assert "Уже найдено в предыдущих частях главы" not in client.requests[0].input_text


async def test_whole_chapter_analysis_strips_block_labels() -> None:
    client = FakeLLMClient.always(analysis_reply(
        aliases=("b1", "b2", "b3", "b4"),
        chapter_summary="Событие в блоке b3.",
        translation_instructions="Сохранить паузу в блоке b4.",
    ))
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=4)
    analysis = (await scenario(command())).analysis
    assert "b3" not in analysis.chapter_summary
    assert "b4" not in analysis.translation_instructions


async def test_whole_chapter_before_you_read_capped() -> None:
    first_words = (
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima",
    )
    entries = [{"expression": f"{word} phrase", "explanation": "значение"} for word in first_words]
    client = FakeLLMClient.always(analysis_reply(
        aliases=("b1", "b2", "b3", "b4"), before_you_read=entries,
    ))
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=4)
    assert len((await scenario(command())).analysis.before_you_read) == 10


async def test_fragment_request_carries_only_its_own_blocks() -> None:
    """A fragment sees only its own blocks and not blocks from neighbouring fragments (PRD § 13)."""
    client = FakeLLMClient.always(fragment_reply())
    scenario, _ = make_analyze_chapter(
        four_block_book(), client, max_fragment_blocks=2
    )

    await scenario(command())

    first, second = (request.input_text for request in client.requests)
    assert "Paragraph one." in first
    assert "Paragraph two." in first
    assert "Paragraph three." not in first
    assert "Paragraph four." not in first
    assert "Paragraph three." in second
    assert "Paragraph four." in second
    assert "Paragraph one." not in second


async def test_fragment_request_uses_the_fragment_instruction() -> None:
    """The fragment is informed that it sees part of a chapter and its position is given."""
    client = FakeLLMClient.always(fragment_reply())
    scenario, _ = make_analyze_chapter(
        four_block_book(), client, max_fragment_blocks=2
    )

    await scenario(command())

    instructions = client.requests[0].instructions
    assert instructions == chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    )
    assert instructions != chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    )
    assert "Фрагмент главы: 1-й из 2" in client.requests[0].input_text
    assert "Фрагмент главы: 2-й из 2" in client.requests[1].input_text


async def test_candidates_of_every_fragment_reach_the_chapter_analysis() -> None:
    """Glossary candidates from every fragment reach the chapter analysis (ALG-03)."""
    client = FakeLLMClient.replying(
        (
            fragment_reply(
                candidates=[
                    {
                        "source_text": "Mr. Bennet",
                        "type": "person",
                        "suggested_translation": "мистер Беннет",
                        "confidence": 0.9,
                        "model_flagged_review": False,
                        "note": None,
                    }
                ]
            ),
            fragment_reply(
                candidates=[
                    {
                        "source_text": "Netherfield",
                        "type": "place",
                        "suggested_translation": "Незерфилд",
                        "confidence": 0.9,
                        "model_flagged_review": False,
                        "note": None,
                    }
                ]
            ),
        )
    )
    scenario, _ = make_analyze_chapter(
        four_block_book(), client, max_fragment_blocks=2
    )

    analysis = (await scenario(command())).analysis

    assert {candidate.source_text for candidate in analysis.glossary_candidates} == {
        "Mr. Bennet",
        "Netherfield",
    }


async def test_attempts_and_token_usage_are_summed_over_fragments() -> None:
    """Attempts and token usage across fragments are summed into the chapter total (PRD § 29.1)."""
    client = FakeLLMClient.replying(
        (
            LLMTransportError("429 Too Many Requests"),
            fragment_reply(usage=TokenUsage(prompt_tokens=1000, completion_tokens=400)),
            fragment_reply(usage=TokenUsage(prompt_tokens=900, completion_tokens=300)),
        )
    )
    scenario, delays = make_analyze_chapter(
        four_block_book(), client, max_fragment_blocks=2
    )

    result = await scenario(command())

    assert result.attempts == 3
    assert result.usage == TokenUsage(prompt_tokens=1900, completion_tokens=700)
    assert delays == [1.0]


async def test_failed_fragment_leaves_the_chapter_unanalysed() -> None:
    """Failure of a single fragment does not silently produce an incomplete chapter analysis (ALG-03)."""
    unit_of_work = four_block_book()
    client = FakeLLMClient.replying(
        (
            fragment_reply(),
            LLMTransportError("провайдер недоступен"),
            LLMTransportError("провайдер недоступен"),
        )
    )
    scenario, _ = make_analyze_chapter(
        unit_of_work, client, max_attempts=2, max_fragment_blocks=2
    )

    with pytest.raises(LLMTransportError):
        await scenario(command())

    assert client.call_count == 3
    assert unit_of_work.analyses.get_for_section(BOOK_ID, "section-1") is None


@pytest.mark.parametrize("max_blocks", [2, 4])
@pytest.mark.parametrize("bad_anchor", ["unknown", "mismatch"])
async def test_unknown_or_mismatched_anchor_retries_analysis(max_blocks, bad_anchor):
    from dataclasses import replace
    from linguary.application.llm_dto import ContentPlanResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint

    class CandidateClient:
        def __init__(self):
            self.base = FakeLLMClient.always(analysis_reply(aliases=tuple(f"b{i}" for i in range(1, max_blocks + 1))))
            self.calls = 0

        async def generate(self, request):
            self.calls += 1
            response = await self.base.generate(request)
            alias = "invisible" if self.calls == 1 and bad_anchor == "unknown" else "b1"
            quote = "Invented!" if self.calls == 1 and bad_anchor == "mismatch" else "Paragraph"
            d = Difficulty("model-id", (SourceAnchor(alias, 0, len(quote), quote),), alias, "grammar", "useful", "explanation", (RequiredPoint("p", "Explain the construction"),))
            return replace(response, payload=replace(response.payload, content_plan=ContentPlanResponse((d,))))

    client = CandidateClient()
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=max_blocks)
    result = await scenario(command())
    assert result.attempts == (3 if max_blocks == 2 else 2)
    assert len(result.analysis.content_plan.difficulties) == (2 if max_blocks == 2 else 1)
    assert all(d.owner_block_id != "b1" for d in result.analysis.content_plan.difficulties)


async def test_legacy_analysis_has_no_content_validation():
    client = FakeLLMClient.always(analysis_reply(aliases=("b1", "b2", "b3", "b4")))
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=4)
    assert (await scenario(command())).analysis.content_plan is None


@pytest.mark.parametrize("max_blocks", [2, 4])
@pytest.mark.parametrize("quoted", ['«Paragraph»', '"Paragraph"'])
async def test_quote_only_byr_retries_whole_and_fragment_analysis(max_blocks, quoted):
    from dataclasses import replace
    from linguary.application.llm_dto import (
        BeforeYouReadEntryResponse, ContentPlanResponse, HelpEvidenceResponse,
        TreatmentPlanItemResponse,
    )
    from linguary.domain import BlockTreatment
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint

    class CandidateClient:
        def __init__(self):
            self.base = FakeLLMClient.always(analysis_reply(
                aliases=tuple(f"b{i}" for i in range(1, max_blocks + 1)),
            ))
            self.calls = 0

        async def generate(self, request):
            self.calls += 1
            response = await self.base.generate(request)
            body = quoted if self.calls == 1 else '«Paragraph» means a section of written text.'
            difficulty = Difficulty(
                "d", (SourceAnchor("b1", 0, 9, "Paragraph"),), "b1",
                "vocabulary", "useful", "lexical_hint",
                (RequiredPoint("p", "Explain the word"),),
            )
            payload = replace(
                response.payload,
                treatment_plan=tuple(
                    TreatmentPlanItemResponse(f"b{i}", BlockTreatment.READ_AS_IS)
                    for i in range(1, max_blocks + 1)
                ),
                before_you_read=(BeforeYouReadEntryResponse("Paragraph", body),),
                content_plan=ContentPlanResponse((difficulty,), (
                    HelpEvidenceResponse("d", "p", "before_you_read.explanation", "b1", 0, body),
                )),
            )
            return replace(response, payload=payload)

    client = CandidateClient()
    unit_of_work = four_block_book()
    scenario, _ = make_analyze_chapter(unit_of_work, client, max_fragment_blocks=max_blocks)
    result = await scenario(command())
    assert result.attempts == (3 if max_blocks == 2 else 2)
    assert result.analysis.content_plan is not None
    assert result.analysis.before_you_read[0].explanation != quoted
    assert unit_of_work.analyses.get_for_section(BOOK_ID, "section-1") == result.analysis


@pytest.mark.parametrize('max_blocks', [2, 4])
async def test_candidate_note_budget_retries_before_save(max_blocks):
    import json
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.use_cases.analyze_chapter import AnalyzeChapter
    from linguary.application.retry_policy import RetryPolicy
    from analysis_fakes import CONFIDENCE_THRESHOLD, sequential_ids
    data = json.loads(analysis_reply(aliases=tuple(f'b{i}' for i in range(1, max_blocks + 1))).content)
    data['before_you_read'] = []
    data['treatment_plan'] = [dict(source_block_id=f'b{i}', treatment='READ_AS_IS') for i in range(1, max_blocks + 1)]
    difficulty = dict(id='d', anchors=[dict(block_id='b1', start=0, end=9, quote='Paragraph')], owner_block_id='b1', kind='grammar', priority='critical', help_kind='explanation', required_points=[dict(id='p', description='Explain the construction')])
    data['content_plan'] = dict(version=2, profile='content-plan-v2', difficulties=[difficulty], before_you_read_evidence=[])
    bad = FakeReply.of_payload(data)
    data['content_plan']['difficulties'] = []
    good = FakeReply.of_payload(data)
    client = FakeLLMClient.replying([bad, good, good] if max_blocks == 2 else [bad, good])
    work = four_block_book()
    async def sleep(_):
        pass
    scenario = AnalyzeChapter(unit_of_work=work, llm_client=client, retry_policy=RetryPolicy(max_attempts=2), glossary_confidence_threshold=CONFIDENCE_THRESHOLD, max_fragment_blocks=max_blocks, id_factory=sequential_ids(), sleep=sleep, context_builder=ContextBuilder(content_plan_profile='content-plan-v2', language_notes_per_chapter=0))
    result = await scenario(command())
    assert result.analysis.content_plan.difficulties == ()
    assert result.attempts == (3 if max_blocks == 2 else 2)
    assert all(r.contract_version == 2 for r in client.requests)
    assert [r.attempt for r in client.requests[:2]] == [1, 2]
    assert work.analyses.get_for_section(BOOK_ID, 'section-1') == result.analysis


async def test_fragment_byr_budgets_preserve_all_bound_evidence():
    import json
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.use_cases.analyze_chapter import AnalyzeChapter
    from linguary.application.retry_policy import RetryPolicy
    from analysis_fakes import CONFIDENCE_THRESHOLD, sequential_ids
    first_words = iter((
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima",
        "mango", "nectar", "orange", "papaya",
    ))
    def reply(label, count):
        data = json.loads(analysis_reply(aliases=('b1', 'b2')).content)
        data['before_you_read'] = [dict(expression=f'{next(first_words)} {label} expression {i}', explanation=f'{label} explanation for aspect {i}.') for i in range(count)]
        points = [dict(id=f'p{i}', description=f'Explain aspect {i}') for i in range(count)]
        data['content_plan'] = dict(version=2, profile='content-plan-v2', difficulties=[dict(id='d', anchors=[dict(block_id='b1', start=0, end=9, quote='Paragraph')], owner_block_id='b1', kind='semantic_relation', priority='critical', help_kind='explanation', required_points=points)], before_you_read_evidence=[dict(difficulty_id='d', point_id=f'p{i}', artifact_kind='before_you_read.explanation', source_block_id='b1', artifact_index=i, body_excerpt=f'{label} explanation for aspect {i}.') for i in range(count)])
        return FakeReply.of_payload(data)
    client = FakeLLMClient.replying([reply('First', 6), reply('First', 5), reply('Second', 5)])
    work = four_block_book()
    async def sleep(_):
        pass
    scenario = AnalyzeChapter(unit_of_work=work, llm_client=client, retry_policy=RetryPolicy(max_attempts=2), glossary_confidence_threshold=CONFIDENCE_THRESHOLD, max_fragment_blocks=2, id_factory=sequential_ids(), sleep=sleep, context_builder=ContextBuilder(content_plan_profile='content-plan-v2', language_notes_per_chapter=0))
    result = await scenario(command())
    assert result.attempts == 3
    assert len(result.analysis.before_you_read) == 10
    assert len(result.analysis.content_plan.before_you_read_evidence) == 10
    assert len(result.analysis.content_plan.difficulties) == 2
    assert work.analyses.get_for_section(BOOK_ID, 'section-1') == result.analysis


async def test_real_group_note_budget_retries_before_save():
    import json
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.use_cases.analyze_chapter import AnalyzeChapter
    from linguary.application.retry_policy import RetryPolicy
    from analysis_fakes import CONFIDENCE_THRESHOLD, sequential_ids
    section = make_analysis_section(texts=('Echo', 'Echo', 'Echo', 'Long text. ' * 1000))
    work = FakeUnitOfWork(document=make_analysis_document((section,)))
    data = json.loads(analysis_reply(aliases=('b1', 'b2', 'b3', 'b4')).content)
    data['before_you_read'] = []
    data['treatment_plan'] = [dict(source_block_id=f'b{i}', treatment='READ_AS_IS') for i in range(1, 5)]
    data['content_plan'] = dict(version=2, profile='content-plan-v2', difficulties=[dict(id=f'd{i}', anchors=[dict(block_id=f'b{i}', start=0, end=4, quote='Echo')], owner_block_id=f'b{i}', kind='wordplay', priority='critical', help_kind='explanation', required_points=[dict(id='p', description='Explain the relationship')]) for i in range(1, 4)], before_you_read_evidence=[])
    bad = FakeReply.of_payload(data)
    data['content_plan']['difficulties'] = data['content_plan']['difficulties'][:1]
    client = FakeLLMClient.replying([bad, FakeReply.of_payload(data)])
    async def sleep(_):
        pass
    scenario = AnalyzeChapter(unit_of_work=work, llm_client=client, retry_policy=RetryPolicy(max_attempts=2), glossary_confidence_threshold=CONFIDENCE_THRESHOLD, max_fragment_blocks=4, id_factory=sequential_ids(), sleep=sleep, context_builder=ContextBuilder(content_plan_profile='content-plan-v2'))
    result = await scenario(command())
    assert result.attempts == 2
    assert len(result.analysis.content_plan.difficulties) == 1
    assert work.analyses.get_for_section(BOOK_ID, 'section-1') == result.analysis
    assert 'реальную группу' in client.requests[1].instructions
