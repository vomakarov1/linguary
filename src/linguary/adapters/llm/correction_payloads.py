"""Strict provider schema for complete chapter correction responses."""

from typing import Literal

from pydantic import ConfigDict, Field

from linguary.adapters.llm.schemas import _ResponseModel, GlossaryHintSchema, LanguageNoteSchema
from linguary.application.correction_dto import (
    ChapterCorrectionResponse, CorrectedBlockResponse, UnchangedBlockResponse,
    ChapterCorrectionV2Response, ProposedBlockResponse, CorrectionIssue,
    CurrentQuoteEvidence, CurrentAbsenceEvidence, CorrectionVerdict, ChapterCorrectionReviewResponse,
)
from linguary.application.llm_dto import GlossaryHintResponse, LanguageNoteResponse


class CorrectionModel(_ResponseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class UnchangedBlockSchema(CorrectionModel):
    block_id: str = Field(min_length=1)
    status: Literal['unchanged']


class CorrectedBlockSchema(CorrectionModel):
    block_id: str = Field(min_length=1)
    status: Literal['corrected']
    translated_text: str | None
    glossary_hints: list[GlossaryHintSchema]
    language_notes: list[LanguageNoteSchema]


class ChapterCorrectionSchema(CorrectionModel):
    blocks: list[UnchangedBlockSchema | CorrectedBlockSchema] = Field(min_length=1)

    def to_dto(self) -> ChapterCorrectionResponse:
        blocks = []
        for block in self.blocks:
            if isinstance(block, UnchangedBlockSchema):
                blocks.append(UnchangedBlockResponse(block.block_id))
            else:
                blocks.append(CorrectedBlockResponse(
                    block.block_id, block.translated_text,
                    tuple(GlossaryHintResponse(**hint.model_dump()) for hint in block.glossary_hints),
                    tuple(LanguageNoteResponse(**note.model_dump()) for note in block.language_notes),
                ))
        return ChapterCorrectionResponse(tuple(blocks))


class CurrentQuoteSchema(CorrectionModel):
    kind: Literal['quote']
    field_path: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class CurrentAbsenceSchema(CorrectionModel):
    kind: Literal['absence']
    field_path: str = Field(min_length=1)
    observed: Literal['empty_list', 'empty_text', 'meaning_missing'] | None
    missing_meaning: str = Field(min_length=1)


class CorrectionIssueSchema(CorrectionModel):
    field_path: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    current_evidence: CurrentQuoteSchema | CurrentAbsenceSchema
    problem: str = Field(min_length=1)
    minimal_fix_reason: str = Field(min_length=1)

    def to_dto(self):
        payload = self.current_evidence.model_dump(exclude={'kind'})
        evidence = (CurrentQuoteEvidence(**payload) if isinstance(self.current_evidence, CurrentQuoteSchema)
                    else CurrentAbsenceEvidence(**payload))
        return CorrectionIssue(self.field_path, self.source_quote, evidence, self.problem, self.minimal_fix_reason)


class ProposedBlockSchema(CorrectionModel):
    block_id: str = Field(min_length=1)
    status: Literal['proposed']
    translated_text: str | None
    glossary_hints: list[GlossaryHintSchema]
    language_notes: list[LanguageNoteSchema]
    issues: list[CorrectionIssueSchema] = Field(min_length=1)


class ChapterCorrectionV2Schema(CorrectionModel):
    blocks: list[UnchangedBlockSchema | ProposedBlockSchema] = Field(min_length=1)

    def to_dto(self):
        return ChapterCorrectionV2Response(tuple(
            UnchangedBlockResponse(b.block_id) if isinstance(b, UnchangedBlockSchema) else
            ProposedBlockResponse(b.block_id, b.translated_text,
                tuple(GlossaryHintResponse(**h.model_dump()) for h in b.glossary_hints),
                tuple(LanguageNoteResponse(**n.model_dump()) for n in b.language_notes),
                tuple(i.to_dto() for i in b.issues)) for b in self.blocks))


class CorrectionVerdictSchema(CorrectionModel):
    candidate_id: str = Field(min_length=1)
    proposal_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    verdict: Literal['confirmed', 'needs_human']
    error_supported: bool
    evidence_valid: bool
    minimal: bool
    no_semantic_loss: bool
    reason: str = Field(min_length=1)


class ChapterCorrectionReviewSchema(CorrectionModel):
    candidates: list[CorrectionVerdictSchema] = Field(min_length=1)

    def to_dto(self):
        return ChapterCorrectionReviewResponse(tuple(CorrectionVerdict(**v.model_dump()) for v in self.candidates))
