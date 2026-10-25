"""Model response parsing and conversion into internal application DTOs.

Defines the boundary for Pydantic: raw provider response text arrives on the outside,
standard `linguary.application.llm_dto` dataclasses leave on the inside (ADR § 3).
Any parsing failure is translated into `LLMResponseFormatError` so that retry policies
do not inspect third-party library exceptions (PRD § 16).

Truncation due to token limits is communicated by the provider separately from content.
Truncated JSON fails validation entirely, so the truncation flag is propagated into
the format error: otherwise the cause would appear as random syntax corruption.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ValidationError
from linguary.adapters.llm.quality_payloads import QualityReviewSchema
from linguary.domain.llm_log import Violation

from linguary.adapters.llm.schemas import (
    ChapterAnalysisV2Schema,
    ChapterAnalysisV3Schema,
    TranslationResultV3Schema,
    TranslationResultV2Schema,
    BeforeYouReadEntrySchema,
    BlockResultSchema,
    ChapterAnalysisSchema,
    GlossaryCandidateSchema,
    GlossaryHintSchema,
    LanguageNoteSchema,
    TranslationResultSchema,
    TreatmentPlanItemSchema,
)
from linguary.application.llm_dto import (
    ContentPlanResponse,
    HelpEvidenceResponse,
    BeforeYouReadEntryResponse,
    BlockResultResponse,
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
    GlossaryHintResponse,
    LanguageNoteResponse,
    LLMPayloadError,
    TranslationResultResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.llm import (
    LLMPayload,
    LLMResponseFormatError,
    LLMResponseKind,
)
from linguary.domain import BlockTreatment
from linguary.domain.content_plan import ContextFact, Difficulty, SourceAnchor, RequiredPoint
from linguary.domain.errors import DomainError

#: Response schema for each request kind.
SCHEMAS: Mapping[LLMResponseKind, type[BaseModel]] = {
    LLMResponseKind.QUALITY_REVIEW: QualityReviewSchema,
    LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisSchema,
    LLMResponseKind.TRANSLATION_RESULT: TranslationResultSchema,
}

def schema_for(kind: LLMResponseKind, *, contract_version: int = 1) -> type[BaseModel]:
    """Structured response schema for request kind."""
    if kind == LLMResponseKind.QUALITY_REVIEW:
        if contract_version != 1:
            raise LLMResponseFormatError("неизвестная версия вычитки")
        return QualityReviewSchema
    if type(contract_version) is not int or contract_version not in (1, 2, 3):
        raise LLMResponseFormatError("неизвестная версия контракта ответа")
    if contract_version == 3:
        return {LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisV3Schema,
                LLMResponseKind.TRANSLATION_RESULT: TranslationResultV3Schema}[kind]
    if contract_version == 2:
        return {LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisV2Schema,
                LLMResponseKind.TRANSLATION_RESULT: TranslationResultV2Schema}[kind]
    return SCHEMAS[kind]


def parse_payload(
    kind: LLMResponseKind, content: str, *, contract_version: int = 1, truncated: bool = False
) -> LLMPayload:
    """Parses model response text and returns application layer DTO.

    `truncated` indicates provider token limit truncation; included in the parsing
    error when a truncated response cannot be validated.
    """
    schema = schema_for(kind, contract_version=contract_version)
    try:
        validated = schema.model_validate_json(content)
    except ValidationError as error:
        raise LLMResponseFormatError(
            _format_error(kind, error=error, truncated=truncated),
            truncated=truncated,
            raw_content=content,
            violation=Violation('provider_truncated' if truncated else ('response_parse' if any(item['type'] == 'json_invalid' for item in error.errors(include_input=False, include_context=False)) else 'response_schema'), kind.value,
                                safe_message='Ответ не соответствует формату'),
        ) from None
    try:
        return _to_dto(validated)
    except (LLMPayloadError, DomainError) as error:
        raise LLMResponseFormatError(
            f"ответ модели ({kind.value}) не соответствует внутреннему контракту: {type(error).__name__}",
            truncated=truncated,
            raw_content=content,
            violation=Violation('response_domain', kind.value, safe_message='Ответ не соответствует внутреннему контракту'),
        ) from None


def _to_dto(validated: BaseModel) -> LLMPayload:
    """Converts validated schema object into an application layer DTO."""
    if isinstance(validated, QualityReviewSchema):
        return validated.to_dto()
    if isinstance(validated, ChapterAnalysisSchema):
        return _chapter_analysis(validated)
    if isinstance(validated, TranslationResultSchema):
        return _translation_result(validated)
    raise LLMPayloadError(f"нет преобразования для схемы {type(validated).__name__}")


def _chapter_analysis(schema: ChapterAnalysisSchema) -> ChapterAnalysisResponse:
    """Chapter analysis DTO."""
    return ChapterAnalysisResponse(
        content_plan=_content_plan(schema.content_plan) if isinstance(schema, (ChapterAnalysisV2Schema, ChapterAnalysisV3Schema)) else None,
        translation_instructions=schema.translation_instructions,
        chapter_summary=schema.chapter_summary,
        glossary_candidates=tuple(
            _glossary_candidate(item) for item in schema.glossary_candidates
        ),
        treatment_plan=tuple(_plan_item(item) for item in schema.treatment_plan),
        before_you_read=tuple(_before_you_read(item) for item in schema.before_you_read),
    )


def _glossary_candidate(schema: GlossaryCandidateSchema) -> GlossaryCandidateResponse:
    """Glossary candidate DTO."""
    return GlossaryCandidateResponse(
        source_text=schema.source_text,
        type=schema.type,
        suggested_translation=schema.suggested_translation,
        confidence=schema.confidence,
        model_flagged_review=schema.model_flagged_review,
        note=(schema.note or "").strip(),
    )


def _plan_item(schema: TreatmentPlanItemSchema) -> TreatmentPlanItemResponse:
    """Treatment plan item DTO."""
    return TreatmentPlanItemResponse(
        source_block_id=schema.source_block_id,
        treatment=BlockTreatment(schema.treatment),
    )


def _before_you_read(schema: BeforeYouReadEntrySchema) -> BeforeYouReadEntryResponse:
    """Before You Read material DTO."""
    return BeforeYouReadEntryResponse(
        expression=schema.expression, explanation=schema.explanation
    )


def _translation_result(schema: TranslationResultSchema) -> TranslationResultResponse:
    """Block group translation result DTO."""
    return TranslationResultResponse(
        content_evidence=tuple(HelpEvidenceResponse(**e.model_dump()) for e in schema.content_evidence) if isinstance(schema, (TranslationResultV2Schema, TranslationResultV3Schema)) else (),
        blocks=tuple(_block_result(item) for item in schema.blocks),
        language_notes=tuple(_language_note(item) for item in schema.language_notes),
    )


def _block_result(schema: BlockResultSchema) -> BlockResultResponse:
    """Single block result DTO."""
    return BlockResultResponse(
        source_block_id=schema.source_block_id,
        treatment=BlockTreatment(schema.treatment),
        translated_text=schema.translated_text or None,
        glossary_hints=tuple(_glossary_hint(item) for item in schema.glossary_hints),
    )


def _glossary_hint(schema: GlossaryHintSchema) -> GlossaryHintResponse:
    """Glossary hint DTO."""
    return GlossaryHintResponse(
        original=schema.original, translation=schema.translation, note=schema.note or None
    )


def _language_note(schema: LanguageNoteSchema) -> LanguageNoteResponse:
    """Language note DTO."""
    return LanguageNoteResponse(
        source_block_id=schema.source_block_id,
        original_excerpt=schema.original_excerpt,
        note=schema.note,
        translated_excerpt=schema.translated_excerpt or None,
    )


def _format_error(
    kind: LLMResponseKind, *, error: ValidationError, truncated: bool
) -> str:
    """Parsing error message explaining schema validation failures.

    Token limit truncation is explicitly stated: otherwise it appears as random syntax
    corruption, whereas resolution requires a larger token budget rather than identical retries.
    """
    violations = ", ".join(
        sorted({str(item["type"]) for item in error.errors(include_input=False, include_context=False)})
    )
    reason = "обрыв ответа по лимиту токенов" if truncated else "ответ не оборван"
    return (
        f"ответ модели ({kind.value}) не соответствует схеме: "
        f"{error.error_count()} нарушений ({violations}); {reason}"
    )


__all__ = ["SCHEMAS", "parse_payload", "schema_for"]


def _content_plan(schema) -> ContentPlanResponse:
    difficulties = []
    for d in schema.difficulties:
        anchors = []
        for a in d.anchors:
            start = a.start
            end = a.end if schema.version == 3 or a.end - a.start == len(a.quote) else a.start + len(a.quote)
            anchors.append(SourceAnchor(block_id=a.block_id, start=start, end=end, quote=a.quote))
        difficulties.append(
            Difficulty(
                id=d.id,
                owner_block_id=d.owner_block_id,
                kind=d.kind,
                priority=d.priority,
                help_kind=d.help_kind,
                anchors=tuple(anchors),
                required_points=tuple(RequiredPoint(**p.model_dump()) for p in d.required_points),
            )
        )
    return ContentPlanResponse(
        version=schema.version,
        profile=schema.profile,
        difficulties=tuple(difficulties),
        before_you_read_evidence=tuple(HelpEvidenceResponse(**e.model_dump()) for e in schema.before_you_read_evidence),
        context_facts=tuple(ContextFact(id=fact.id, kind=fact.kind,
            anchors=tuple(SourceAnchor(**anchor.model_dump()) for anchor in fact.anchors),
            subject_anchor=SourceAnchor(**fact.subject_anchor.model_dump()), value=fact.value, certainty=fact.certainty)
            for fact in getattr(schema, 'context_facts', ())),
    )
