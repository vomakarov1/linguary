"""Сценарий анализа главы: запрос, проверка ответа и сохранение (P3.T3, PRD § 14)."""

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
    """Книга из одной главы с двумя абзацами."""
    section = make_analysis_section(texts=("First paragraph.", "Second paragraph."))
    return FakeUnitOfWork(document=make_analysis_document((section,)))


def command() -> AnalyzeChapterCommand:
    """Команда анализа единственной главы книги."""
    return AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-1")


async def test_saved_analysis_keeps_chapter_summary_and_materials() -> None:
    """Сохранённый анализ несёт сводку главы, инструкции и `Before You Read`."""
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
    """План обработки покрывает блоки главы и хранит их стабильные идентификаторы."""
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
    """`requires_review` вычисляет приложение, а не модель (DTO-01, PRD § 12)."""
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
    """Уверенность ниже настроенного порога делает кандидата требующим внимания."""
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
    """В запрос главы попадают решения предыдущих глав, но не их текст (PRD § 13)."""
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
    """Повторный анализ главы в середине книги не видит решений следующих глав."""
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
    """Анализ следующей главы видит решение пользователя, а не вариант модели (Д-1).

    Пропущенный элемент решением не считается: главе идёт вариант модели.
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
    """Решение пользователя по элементу глоссария книги."""
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
    """Повторный анализ главы видит решения пользователя по всей книге (Н-3).

    Решения по понятиям этой и следующих глав приходят в анализ так же, как решения
    предыдущих глав; кандидаты следующих глав без решения и пропущенные элементы —
    нет: вариант модели следующей главы этой главе ничего не даёт.
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
    """Указание элемента глоссария в контексте анализа не несёт отвергнутый вариант.

    Третий живой прогон: в контексте анализа главы 2 указание `the marsh gate`
    называло «Даскмир», хотя решение пользователя по Duskmere — «Даскмер».
    Решение по другому понятию и его формы при этом остаются.
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
    """Глава без текстовых блоков не даёт вызова модели, но получает полный анализ."""
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
    """Неполный план отбраковывается и повторяется с уточняющей инструкцией (ALG-05)."""
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


async def test_unknown_block_in_plan_is_rejected() -> None:
    """План с чужим блоком не сохраняется и вызывает повтор."""
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
    """Два режима на один блок — негодный план: он тоже повторяется."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (
            analysis_reply(aliases=("b1", "b1", "b2")),
            analysis_reply(aliases=("b1", "b2")),
        )
    )
    scenario, _ = make_analyze_chapter(unit_of_work, client)

    assert (await scenario(command())).attempts == 2


async def test_transient_error_is_retried_after_backoff() -> None:
    """Временная ошибка повторяется, и цикл выдерживает паузу политики (ALG-05)."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.replying(
        (LLMTransportError("429 Too Many Requests"), analysis_reply())
    )
    scenario, delays = make_analyze_chapter(unit_of_work, client)

    result = await scenario(command())

    assert result.attempts == 2
    assert delays == [1.0]


async def test_exhausted_attempts_leave_no_saved_analysis() -> None:
    """Исчерпав попытки, сценарий поднимает ошибку и анализа не сохраняет."""
    unit_of_work = two_block_book()
    client = FakeLLMClient.always(LLMTransportError("провайдер недоступен"))
    scenario, _ = make_analyze_chapter(unit_of_work, client, max_attempts=2)

    with pytest.raises(LLMTransportError):
        await scenario(command())

    assert client.call_count == 2
    assert unit_of_work.analyses.get_for_section(BOOK_ID, "section-1") is None


async def test_result_reports_token_usage_of_successful_call() -> None:
    """Результат несёт объём токенов вызова для ориентиров стоимости (PRD § 29.1)."""
    unit_of_work = two_block_book()
    usage = TokenUsage(prompt_tokens=1200, completion_tokens=300)
    scenario, _ = make_analyze_chapter(
        unit_of_work, FakeLLMClient.always(analysis_reply(usage=usage))
    )

    assert (await scenario(command())).usage == usage


async def test_unknown_book_is_rejected() -> None:
    """Анализ книги, которой нет в хранилище, невозможен."""
    scenario, _ = make_analyze_chapter(FakeUnitOfWork(), FakeLLMClient.always(analysis_reply()))

    with pytest.raises(AnalyzeChapterError):
        await scenario(command())


async def test_unknown_section_is_rejected() -> None:
    """Анализ главы, которой нет в книге, невозможен."""
    scenario, _ = make_analyze_chapter(two_block_book(), FakeLLMClient.always(analysis_reply()))

    with pytest.raises(AnalyzeChapterError):
        await scenario(AnalyzeChapterCommand(book_id=BOOK_ID, section_id="section-9"))
