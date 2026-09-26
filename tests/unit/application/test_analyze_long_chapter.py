"""Резервный сценарий длинной главы в сценарии анализа (P3.T4, PRD § 14, ALG-03)."""

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

#: Глава из четырёх абзацев: при пределе в два блока она делится на два фрагмента.
CHAPTER_TEXTS = (
    "Paragraph one.",
    "Paragraph two.",
    "Paragraph three.",
    "Paragraph four.",
)


def four_block_book() -> FakeUnitOfWork:
    """Книга из одной главы в четыре абзаца."""
    section = make_analysis_section(texts=CHAPTER_TEXTS)
    return FakeUnitOfWork(document=make_analysis_document((section,)))


def command() -> AnalyzeChapterCommand:
    """Команда анализа единственной главы книги."""
    return AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-1")


def fragment_reply(**overrides: Any) -> FakeReply:
    """Ответ модели по фрагменту из двух блоков."""
    return analysis_reply(aliases=("b1", "b2"), **overrides)


async def test_chapter_within_the_limit_is_analysed_by_one_request() -> None:
    """Глава в пределах порога дробления уходит в модель одним запросом."""
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
    """Длинная глава уходит в модель по фрагментам и собирается в один анализ."""
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
    entries = [{"expression": f"phrase {n}", "explanation": "значение"} for n in range(12)]
    client = FakeLLMClient.always(analysis_reply(
        aliases=("b1", "b2", "b3", "b4"), before_you_read=entries,
    ))
    scenario, _ = make_analyze_chapter(four_block_book(), client, max_fragment_blocks=4)
    assert len((await scenario(command())).analysis.before_you_read) == 10


async def test_fragment_request_carries_only_its_own_blocks() -> None:
    """Фрагмент видит свои блоки и не видит блоков соседних фрагментов (PRD § 13)."""
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
    """Фрагменту говорят, что он видит часть главы, и называют его место в ней."""
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
    """Кандидаты глоссария всех фрагментов попадают в анализ главы (ALG-03)."""
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
    """Попытки и расход токенов фрагментов складываются в итог главы (PRD § 29.1)."""
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
    """Неудача одного фрагмента не даёт молча неполный анализ главы (ALG-03)."""
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
