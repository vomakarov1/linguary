import math

import pytest

from linguary.application.llm_dto import GlossaryCandidateResponse, LLMPayloadError
from linguary.domain import ConceptType
from linguary.domain.glossary import GlossaryCandidate, GlossaryError
from linguary.domain.name_translation import (
    NameTranslationAssessment,
    NameTranslationEvidence,
    NameTranslationObservation,
    NameTranslationPolicy,
    NameTranslationStrategy,
)


def test_only_two_strategies() -> None:
    assert {item.value for item in NameTranslationStrategy} == {"sound", "meaning"}
    with pytest.raises(ValueError):
        NameTranslationStrategy("mixed")


def test_missing_assessment_is_not_positive() -> None:
    response = GlossaryCandidateResponse("Ravenclaw", ConceptType.PERSON, "Ровенкло", 0.9, False)
    candidate = GlossaryCandidate("c", "a", "Ravenclaw", ConceptType.PERSON, "Ровенкло", 0.9, False, True)
    assert response.name_translation_assessment is None
    assert candidate.name_translation_evidence is None
    observation = NameTranslationObservation("Ровенкло", 0.9, False, None)
    assert not observation.supports(NameTranslationStrategy.SOUND)


def test_inconsistent_assessment_cannot_exempt() -> None:
    assessment = NameTranslationAssessment("meaning", True, ("wordplay",), "Есть игра слов")
    observation = NameTranslationObservation("Когтевран", 0.9, False, assessment)
    evidence = NameTranslationEvidence(1, "conditions-1", 1, NameTranslationStrategy.SOUND, (observation,))
    assert not observation.supports(NameTranslationStrategy.SOUND)
    assert not evidence.supports(NameTranslationPolicy(1, NameTranslationStrategy.SOUND), "conditions-1")


def test_meaning_sound_requires_explicit_opaque_name() -> None:
    implicit = NameTranslationAssessment("sound", True, (), "Имя непрозрачно")
    explicit = NameTranslationAssessment("sound", True, (), "Имя непрозрачно", opaque_name=True)
    assert not implicit.supports(NameTranslationStrategy.MEANING)
    assert explicit.supports(NameTranslationStrategy.MEANING)
    assert implicit.supports(NameTranslationStrategy.SOUND)
    with pytest.raises(ValueError):
        NameTranslationAssessment("sound", True, (), "Имя непрозрачно", opaque_name="true")


def test_unknown_or_foreign_evidence_cannot_exempt() -> None:
    observation = NameTranslationObservation(
        "Ровенкло", 0.99, False, NameTranslationAssessment("sound", True, (), "Передано по звучанию")
    )
    policy = NameTranslationPolicy(1, NameTranslationStrategy.SOUND)
    evidence = NameTranslationEvidence(2, "conditions-1", 1, NameTranslationStrategy.SOUND, (observation,))
    assert not evidence.supports(policy, "conditions-1")
    assert not NameTranslationEvidence(1, "conditions-1", 1, NameTranslationStrategy.SOUND, (observation,)).supports(policy, "other-conditions")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_confidence_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        NameTranslationObservation("Имя", value, False, None)
    with pytest.raises(LLMPayloadError):
        GlossaryCandidateResponse("Name", ConceptType.PERSON, "Имя", value, False)
    with pytest.raises(GlossaryError):
        GlossaryCandidate("c", "a", "Name", ConceptType.PERSON, "Имя", value, False, True)
