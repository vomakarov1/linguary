"""Strict provider schema for complete chapter correction responses."""

from typing import Literal

from pydantic import ConfigDict, Field

from linguary.adapters.llm.schemas import _ResponseModel, GlossaryHintSchema, LanguageNoteSchema
from linguary.application.correction_dto import (
    ChapterCorrectionResponse, CorrectedBlockResponse, UnchangedBlockResponse,
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
