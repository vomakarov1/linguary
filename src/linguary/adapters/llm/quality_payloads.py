"""Strict version-one optional quality review response."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from linguary.application.quality_context import QualityReviewResponse


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class ReviewAnchor(ReviewModel):
    block_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class ReviewFinding(ReviewModel):
    block_ids: list[str] = Field(min_length=1)
    artifact_kind: Literal['original', 'analysis', 'result']
    artifact_sha: str = Field(pattern=r'^[0-9a-f]{64}$')
    severity: Literal['critical', 'major', 'minor']
    issue: str = Field(min_length=1)
    evidence_anchors: list[ReviewAnchor] = Field(min_length=1)
    suggestion: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


class QualityReviewSchema(ReviewModel):
    findings: list[ReviewFinding]
    checked_artifacts: list[str]
    missing_artifacts: list[str]

    def to_dto(self):
        return QualityReviewResponse(tuple(item.model_dump() for item in self.findings),
            tuple(self.checked_artifacts), tuple(self.missing_artifacts))
