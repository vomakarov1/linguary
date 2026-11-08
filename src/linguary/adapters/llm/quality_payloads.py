"""Strict version-one optional quality review response."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from linguary.application.quality_context import QualityReviewResponse


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class ReviewAnchor(ReviewModel):
    block_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)


class ReviewFinding(ReviewModel):
    block_ids: list[str] = Field(min_length=1)
    artifact_kind: Literal['original', 'analysis', 'result']
    artifact_sha: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    severity: Literal['critical', 'major', 'minor']
    issue: str = Field(min_length=1)
    evidence_anchors: list[ReviewAnchor] = Field(min_length=1)
    suggestion: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


class QualityReviewSchema(ReviewModel):
    findings: list[ReviewFinding]
    checked_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_artifact_sets(self) -> 'QualityReviewSchema':
        has_checked = 'checked_artifacts' in self.model_fields_set
        has_missing = 'missing_artifacts' in self.model_fields_set
        if has_checked != has_missing:
            raise ValueError('checked_artifacts and missing_artifacts must both be provided or both omitted')
        return self

    def to_dto(self):
        return QualityReviewResponse(tuple(item.model_dump() for item in self.findings),
            tuple(self.checked_artifacts), tuple(self.missing_artifacts))
