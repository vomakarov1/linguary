"""Разбор ответа модели и его преобразование во внутренние DTO прикладного слоя."""

from __future__ import annotations

import pytest
from llm_payload_factories import analysis_payload, as_content, block_payload, translation_payload

from linguary.adapters.llm.payloads import parse_payload
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import LLMResponseFormatError, LLMResponseKind
from linguary.domain import BlockTreatment, ConceptType


def test_chapter_analysis_is_mapped_to_dto() -> None:
    """Ответ анализа главы превращается в DTO прикладного слоя целиком."""
    payload = parse_payload(
        LLMResponseKind.CHAPTER_ANALYSIS, as_content(analysis_payload())
    )

    assert isinstance(payload, ChapterAnalysisResponse)
    assert payload.chapter_summary == "Mr. Bennet visits the new neighbour."
    assert payload.translation_instructions == "Сохранять иронию рассказчика."
    assert payload.planned_block_ids == ("block-1", "block-2")
    assert payload.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert payload.glossary_candidates[0].type is ConceptType.PERSON
    assert payload.glossary_candidates[0].model_flagged_review is False
    assert payload.before_you_read[0].expression == "entailed estate"


def test_translation_result_is_mapped_to_dto() -> None:
    """Ответ обработки группы блоков превращается в DTO прикладного слоя целиком."""
    payload = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT, as_content(translation_payload())
    )

    assert isinstance(payload, TranslationResultResponse)
    assert payload.block_ids == ("block-1", "block-2")
    assert payload.blocks[0].treatment is BlockTreatment.TRANSLATE
    assert payload.blocks[1].treatment is BlockTreatment.GLOSSARY_ONLY
    assert payload.blocks[1].translated_text is None
    assert payload.blocks[1].glossary_hints[0].original == "entailed"
    assert payload.notes_for("block-1")[0].note == "Ироническая формула зачина."
    assert payload.notes_for("block-2") == ()


def test_read_as_is_block_is_mapped_without_content() -> None:
    """Блок `READ_AS_IS` попадает в DTO без перевода и подсказок."""
    payload = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT,
        as_content(
            translation_payload(
                blocks=[
                    block_payload(
                        treatment="READ_AS_IS", translated_text=None, glossary_hints=[]
                    )
                ],
                language_notes=[],
            )
        ),
    )

    assert isinstance(payload, TranslationResultResponse)
    assert payload.blocks[0].translated_text is None
    assert payload.blocks[0].glossary_hints == ()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("сейчас переведу, минутку", id="not-json"),
        pytest.param('{"blocks": [{"source_block_id": "block-1"', id="truncated-json"),
        pytest.param("{}", id="missing-required-fields"),
    ],
)
def test_invalid_response_gives_format_error(content: str) -> None:
    """Неразбираемый ответ приводится к ошибке формата, а не к ошибке чужой библиотеки."""
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, content)


def test_truncated_response_is_reported_as_truncated() -> None:
    """Обрыв по лимиту токенов виден в ошибке разбора и назван причиной."""
    with pytest.raises(LLMResponseFormatError) as error:
        parse_payload(
            LLMResponseKind.TRANSLATION_RESULT, '{"blocks": [{"source', truncated=True
        )

    assert error.value.truncated is True
    assert "лимиту токенов" in str(error.value)


def test_format_error_does_not_expose_full_response() -> None:
    """Текст ошибки не выносит наружу весь ответ: в нём может быть текст книги."""
    content = "не json, " + "текст книги " * 200

    with pytest.raises(LLMResponseFormatError) as error:
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, content)

    assert len(str(error.value)) < len(content)


def test_response_of_wrong_kind_is_rejected() -> None:
    """Ответ анализа на запрос обработки не принимается за результат обработки."""
    with pytest.raises(LLMResponseFormatError):
        parse_payload(
            LLMResponseKind.TRANSLATION_RESULT, as_content(analysis_payload())
        )


def test_v2_round_trip_plan_and_evidence():
    data = analysis_payload()
    data['content_plan'] = {'version': 2, 'profile': 'content-plan-v2', 'difficulties': [], 'before_you_read_evidence': []}
    result = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), contract_version=2)
    assert result.content_plan.version == 2
    data = translation_payload()
    data['content_evidence'] = []
    assert parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=2).content_evidence == ()


@pytest.mark.parametrize('kind,data', [(LLMResponseKind.CHAPTER_ANALYSIS, analysis_payload()), (LLMResponseKind.TRANSLATION_RESULT, translation_payload())])
def test_v2_missing_fields_rejected(kind, data):
    with pytest.raises(LLMResponseFormatError):
        parse_payload(kind, as_content(data), contract_version=2)


def test_unknown_contract_version_rejected():
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, '{}', contract_version=3)
