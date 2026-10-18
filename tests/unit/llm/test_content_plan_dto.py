"""Отдельные DTO кандидата сохраняют прежние значения по умолчанию."""
import pytest
from linguary.application.llm_dto import ChapterAnalysisResponse, ContentPlanResponse, HelpEvidenceResponse, LLMPayloadError, TranslationResultResponse


def test_empty_candidate_plan_is_valid():
    assert ChapterAnalysisResponse("instructions", "summary", content_plan=ContentPlanResponse()).content_plan.difficulties == ()


def test_evidence_requires_point_and_nonempty_body():
    for point, body in (("", "body"), ("p", " ")):
        with pytest.raises(LLMPayloadError):
            HelpEvidenceResponse("d", point, "before_you_read.explanation", "b1", 0, body)
    with pytest.raises(TypeError):
        HelpEvidenceResponse(fulfilled=True)


def test_legacy_dto_defaults_unchanged():
    assert ChapterAnalysisResponse("instructions", "summary").content_plan is None
    assert TranslationResultResponse().content_evidence == ()
