"""Model response parsing and mapping to internal application DTOs."""

from __future__ import annotations

import pytest
from llm_payload_factories import analysis_payload, as_content, block_payload, translation_payload

from linguary.adapters.llm.payloads import parse_payload
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import LLMResponseFormatError, LLMResponseKind
from linguary.domain import BlockTreatment, ConceptType
from linguary.domain.name_translation import NameTranslationMethod


def test_missing_or_null_assessment_falls_back() -> None:
    data = analysis_payload()
    for value in ("missing", None):
        if value is not None:
            data["glossary_candidates"][0]["name_translation_assessment"] = value
        else:
            data["glossary_candidates"][0]["name_translation_assessment"] = None
        if value == "missing":
            data["glossary_candidates"][0].pop("name_translation_assessment", None)
        result = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)
        assert result.glossary_candidates[0].name_translation_assessment is None


def test_malformed_assessment_rejected() -> None:
    data = analysis_payload()
    data["glossary_candidates"][0]["name_translation_assessment"] = {
        "applied_method": "sound", "eligible_without_review": "true", "review_reasons": [], "rationale": "Звучание"
    }
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)
    data["glossary_candidates"][0]["name_translation_assessment"]["eligible_without_review"] = True
    result = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)
    assert result.glossary_candidates[0].name_translation_assessment.applied_method is NameTranslationMethod.SOUND


def test_chapter_analysis_is_mapped_to_dto() -> None:
    """Chapter analysis response maps completely to application DTO."""
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
    """Block group processing response maps completely to application DTO."""
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
    """A 'READ_AS_IS' block is mapped into DTO without translation or hints."""
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
    """An unparseable response yields a format error rather than a third-party library error."""
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, content)


def test_truncated_response_is_reported_as_truncated() -> None:
    """Token limit truncation is reflected in parse error as the cause."""
    with pytest.raises(LLMResponseFormatError) as error:
        parse_payload(
            LLMResponseKind.TRANSLATION_RESULT, '{"blocks": [{"source', truncated=True
        )

    assert error.value.truncated is True
    assert "лимиту токенов" in str(error.value)


def test_format_error_does_not_expose_full_response() -> None:
    """Error message does not leak the entire response which may contain book text."""
    content = "не json, " + "текст книги " * 200

    with pytest.raises(LLMResponseFormatError) as error:
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, content)

    assert len(str(error.value)) < len(content)


def test_response_of_wrong_kind_is_rejected() -> None:
    """An analysis response to a translation request is rejected as invalid."""
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


@pytest.mark.parametrize('kind,text', [('lexical_hint', 'only'), ('sentence_rephrase', 'She saw only the storm.'), ('russian_explanation', 'Она видела только бурю.'), ('unresolved', None)])
def test_v4_maps_typed_fragment_assistance(kind, text):
    data = translation_payload()
    data.update(content_evidence=[], fragment_assistances=[dict(difficulty_id='d', point_ids=['p'], kind=kind, text=text)])
    payload = parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=4)
    assert payload.fragment_assistances[0].point_ids == ('p',)
    assert payload.fragment_assistances[0].text == text
    for version in (1, 2, 3):
        with pytest.raises(LLMResponseFormatError):
            parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=version)


@pytest.mark.parametrize('item', [dict(kind='unknown', text='x', point_ids=['p']), dict(kind='unresolved', text='x', point_ids=['p']), dict(kind='lexical_hint', text='', point_ids=['p']), dict(kind='lexical_hint', text='only', point_ids=['p', 'p'])])
def test_v4_rejects_invalid_fragment_assistance(item):
    data = translation_payload()
    data.update(content_evidence=[], fragment_assistances=[dict(difficulty_id='d', **item)])
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=4)
