"""Policy and provenance for model suggestions about names and titles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class NameTranslationStrategy(StrEnum):
    SOUND = "sound"
    MEANING = "meaning"


class NameTranslationMethod(StrEnum):
    SOUND = "sound"
    MEANING = "meaning"
    MIXED = "mixed"
    CREATIVE = "creative"
    UNCERTAIN = "uncertain"


class NameTranslationReviewReason(StrEnum):
    AMBIGUOUS = "ambiguous"
    WORDPLAY = "wordplay"
    ANAGRAM = "anagram"
    UNCERTAIN_ETYMOLOGY = "uncertain_etymology"
    CREATIVE_ADAPTATION = "creative_adaptation"
    PERSONAL_NAME_SUBSTITUTION = "personal_name_substitution"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class NameTranslationPolicy:
    version: int
    strategy: NameTranslationStrategy

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported name translation policy version")
        object.__setattr__(self, "strategy", NameTranslationStrategy(self.strategy))


@dataclass(frozen=True, slots=True)
class NameTranslationAssessment:
    applied_method: NameTranslationMethod
    eligible_without_review: bool
    review_reasons: tuple[NameTranslationReviewReason, ...]
    rationale: str
    opaque_name: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied_method", NameTranslationMethod(self.applied_method))
        if type(self.eligible_without_review) is not bool:
            raise ValueError("eligibility must be a boolean")
        if type(self.opaque_name) is not bool:
            raise ValueError("opaque name flag must be a boolean")
        if not isinstance(self.review_reasons, (tuple, list)):
            raise ValueError("review reasons must be a sequence")
        object.__setattr__(self, "review_reasons", tuple(NameTranslationReviewReason(reason) for reason in self.review_reasons))
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("assessment rationale is empty")
        if not self.eligible_without_review and not self.review_reasons:
            raise ValueError("negative assessment requires a reason")

    def supports(self, strategy: NameTranslationStrategy) -> bool:
        if not self.eligible_without_review or self.review_reasons:
            return False
        if strategy == NameTranslationStrategy.SOUND:
            return self.applied_method == NameTranslationMethod.SOUND
        return self.applied_method in (NameTranslationMethod.MEANING, NameTranslationMethod.MIXED) or (
            self.applied_method == NameTranslationMethod.SOUND and self.opaque_name
        )


@dataclass(frozen=True, slots=True)
class NameTranslationObservation:
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    assessment: NameTranslationAssessment | None

    def __post_init__(self) -> None:
        if not isinstance(self.suggested_translation, str) or not self.suggested_translation.strip():
            raise ValueError("suggested translation is empty")
        if type(self.confidence) not in (int, float) or not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and within 0..1")
        if type(self.model_flagged_review) is not bool:
            raise ValueError("model review flag must be a boolean")
        if self.assessment is not None and not isinstance(self.assessment, NameTranslationAssessment):
            raise ValueError("invalid name translation assessment")

    def supports(self, strategy: NameTranslationStrategy) -> bool:
        return not self.model_flagged_review and self.assessment is not None and self.assessment.supports(strategy)


@dataclass(frozen=True, slots=True)
class NameTranslationEvidence:
    version: int
    conditions_id: str
    policy_version: int
    strategy: NameTranslationStrategy
    observations: tuple[NameTranslationObservation, ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 1:
            raise ValueError("invalid evidence version")
        if not isinstance(self.conditions_id, str) or not self.conditions_id.strip():
            raise ValueError("conditions ID is empty")
        if type(self.policy_version) is not int or self.policy_version < 1:
            raise ValueError("invalid policy version")
        object.__setattr__(self, "strategy", NameTranslationStrategy(self.strategy))
        if not isinstance(self.observations, (tuple, list)) or not self.observations or any(
            not isinstance(item, NameTranslationObservation) for item in self.observations
        ):
            raise ValueError("evidence requires observations")
        object.__setattr__(self, "observations", tuple(self.observations))

    def supports(self, policy: NameTranslationPolicy, conditions_id: str) -> bool:
        return (
            self.version == 1
            and self.conditions_id == conditions_id
            and self.policy_version == policy.version
            and self.strategy == policy.strategy
            and all(item.supports(policy.strategy) for item in self.observations)
        )
