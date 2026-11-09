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
from dataclasses import replace
import json

from pydantic import BaseModel, ValidationError
from linguary.adapters.llm.correction_payloads import (
    ChapterCorrectionSchema, ChapterCorrectionV2Schema, ChapterCorrectionReviewSchema,
)
from linguary.adapters.llm.quality_payloads import QualityReviewSchema
from linguary.domain.llm_log import Violation

from linguary.adapters.llm.schemas import (
    HelpEvidenceSchema,
    PronunciationCandidateSchema,
    TranslationWithPronunciationSchema,
    TranslationV2WithPronunciationSchema,
    TranslationV3WithPronunciationSchema,
    TranslationV4WithPronunciationSchema,
    ChapterAnalysisV2Schema,
    ChapterAnalysisV3Schema,
    ChapterAnalysisV5Schema,
    ChapterAnalysisWithNameAssessmentSchema,
    ChapterAnalysisV2WithNameAssessmentSchema,
    ChapterAnalysisV3WithNameAssessmentSchema,
    ChapterAnalysisV5WithNameAssessmentSchema,
    QuotedContentPlanSchema,
    GlossaryCandidateWithNameAssessmentSchema,
    TranslationResultV3Schema,
    TranslationResultV4Schema,
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
    PronunciationAnchorResponse,
    PronunciationCandidateResponse,
    PronunciationContextResponse,
    ContentPlanResponse,
    QuotedContentPlanResponse,
    QuotedSourceAnchorResponse,
    QuotedDifficultyResponse,
    QuotedContextFactResponse,
    FragmentAssistanceResponse,
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
from linguary.domain.name_translation import NameTranslationAssessment

#: Response schema for each request kind.
SCHEMAS: Mapping[LLMResponseKind, type[BaseModel]] = {
    LLMResponseKind.CHAPTER_CORRECTION: ChapterCorrectionSchema,
    LLMResponseKind.CHAPTER_CORRECTION_REVIEW: ChapterCorrectionReviewSchema,
    LLMResponseKind.QUALITY_REVIEW: QualityReviewSchema,
    LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisSchema,
    LLMResponseKind.TRANSLATION_RESULT: TranslationResultSchema,
}

def schema_for(
    kind: LLMResponseKind, *, contract_version: int = 1,
    name_translation_policy_version: int | None = None,
    pronunciation_candidates_version: int | None = None,
) -> type[BaseModel]:
    """Structured response schema for request kind."""
    if kind in {LLMResponseKind.CHAPTER_CORRECTION, LLMResponseKind.CHAPTER_CORRECTION_REVIEW}:
        if (type(contract_version) is not int or contract_version not in {1, 2}
                or name_translation_policy_version is not None or pronunciation_candidates_version is not None):
            raise LLMResponseFormatError('неизвестная версия контракта исправления')
        if kind == LLMResponseKind.CHAPTER_CORRECTION_REVIEW:
            if contract_version != 1:
                raise LLMResponseFormatError('неизвестная версия проверки кандидатов')
            return ChapterCorrectionReviewSchema
        return ChapterCorrectionSchema if contract_version == 1 else ChapterCorrectionV2Schema
    if pronunciation_candidates_version is not None:
        if (type(pronunciation_candidates_version) is not int or pronunciation_candidates_version != 1
                or kind != LLMResponseKind.TRANSLATION_RESULT):
            raise LLMResponseFormatError('неизвестный режим кандидатов произношения')
        schema_for(kind, contract_version=contract_version,
                   name_translation_policy_version=name_translation_policy_version)
        return {1: TranslationWithPronunciationSchema, 2: TranslationV2WithPronunciationSchema,
                3: TranslationV3WithPronunciationSchema, 4: TranslationV4WithPronunciationSchema}[contract_version]
    if name_translation_policy_version is not None and (
        type(name_translation_policy_version) is not int or name_translation_policy_version != 1
    ):
        raise LLMResponseFormatError("неизвестная версия политики передачи имён")
    if kind == LLMResponseKind.QUALITY_REVIEW:
        if contract_version != 1 or name_translation_policy_version is not None:
            raise LLMResponseFormatError("неизвестная версия вычитки")
        return QualityReviewSchema
    if type(contract_version) is not int or contract_version not in (1, 2, 3, 4, 5):
        raise LLMResponseFormatError("неизвестная версия контракта ответа")
    if contract_version == 5:
        if kind != LLMResponseKind.CHAPTER_ANALYSIS:
            raise LLMResponseFormatError('версия 5 доступна только для анализа главы')
        return (ChapterAnalysisV5WithNameAssessmentSchema
                if name_translation_policy_version == 1 else ChapterAnalysisV5Schema)
    if contract_version == 4:
        if kind != LLMResponseKind.TRANSLATION_RESULT:
            raise LLMResponseFormatError('версия 4 доступна только для обработки')
        return TranslationResultV4Schema
    if contract_version == 3:
        if kind == LLMResponseKind.CHAPTER_ANALYSIS and name_translation_policy_version == 1:
            return ChapterAnalysisV3WithNameAssessmentSchema
        return {LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisV3Schema,
                LLMResponseKind.TRANSLATION_RESULT: TranslationResultV3Schema}[kind]
    if contract_version == 2:
        if kind == LLMResponseKind.CHAPTER_ANALYSIS and name_translation_policy_version == 1:
            return ChapterAnalysisV2WithNameAssessmentSchema
        return {LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisV2Schema,
                LLMResponseKind.TRANSLATION_RESULT: TranslationResultV2Schema}[kind]
    if kind == LLMResponseKind.CHAPTER_ANALYSIS and name_translation_policy_version == 1:
        return ChapterAnalysisWithNameAssessmentSchema
    return SCHEMAS[kind]


def parse_payload(
    kind: LLMResponseKind, content: str, *, contract_version: int = 1,
    name_translation_policy_version: int | None = None, truncated: bool = False,
    pronunciation_candidates_version: int | None = None,
) -> LLMPayload:
    """Parses model response text and returns application layer DTO.

    `truncated` indicates provider token limit truncation; included in the parsing
    error when a truncated response cannot be validated.
    """
    schema = schema_for(kind, contract_version=contract_version, name_translation_policy_version=name_translation_policy_version,
                        pronunciation_candidates_version=pronunciation_candidates_version)
    if kind in {LLMResponseKind.CHAPTER_CORRECTION, LLMResponseKind.CHAPTER_CORRECTION_REVIEW} and truncated:
        raise LLMResponseFormatError(
            'ответ исправления оборван по лимиту токенов', truncated=True, raw_content=content,
            violation=Violation('provider_truncated', kind.value, safe_message='Ответ исправления оборван'),
        )
    try:
        raw_object = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        reason = "обрыв ответа по лимиту токенов" if truncated else "ответ не оборван"
        raise LLMResponseFormatError(
            f"ответ модели ({kind.value}) не соответствует схеме: 1 нарушений (json_invalid); {reason}",
            truncated=truncated,
            raw_content=content,
            violation=Violation(
                'provider_truncated' if truncated else 'response_parse',
                kind.value,
                safe_message='Ответ не соответствует формату',
            ),
        ) from None
    candidate_items = ()
    if pronunciation_candidates_version == 1:
        if isinstance(raw_object, dict) and isinstance(raw_object.get('pronunciation_candidates'), list):
            candidate_items = raw_object.pop('pronunciation_candidates')
            schema = schema_for(kind, contract_version=contract_version,
                                name_translation_policy_version=name_translation_policy_version)
    if kind == LLMResponseKind.CHAPTER_ANALYSIS and name_translation_policy_version == 1:
        if isinstance(raw_object, dict) and isinstance(raw_object.get("glossary_candidates"), list):
            for candidate in raw_object["glossary_candidates"]:
                if isinstance(candidate, dict):
                    candidate.setdefault("name_translation_assessment", None)
    byr_evidence_items = ()
    content_evidence_items = ()
    if isinstance(raw_object, dict):
        if kind == LLMResponseKind.TRANSLATION_RESULT and "content_evidence" in raw_object:
            raw_ev = raw_object.pop("content_evidence")
            if isinstance(raw_ev, list):
                content_evidence_items = raw_ev
        elif kind == LLMResponseKind.CHAPTER_ANALYSIS and isinstance(raw_object.get("content_plan"), dict):
            if "before_you_read_evidence" in raw_object["content_plan"]:
                raw_byr = raw_object["content_plan"].pop("before_you_read_evidence")
                if isinstance(raw_byr, list):
                    byr_evidence_items = raw_byr
    try:
        validated = schema.model_validate(raw_object)
    except ValidationError as error:
        raise LLMResponseFormatError(
            _format_error(kind, error=error, truncated=truncated),
            truncated=truncated,
            raw_content=content,
            violation=Violation('provider_truncated' if truncated else ('response_parse' if any(item['type'] == 'json_invalid' for item in error.errors(include_input=False, include_context=False)) else 'response_schema'), kind.value,
                                safe_message='Ответ не соответствует формату'),
        ) from None
    try:
        result = _to_dto(validated)
    except (LLMPayloadError, DomainError, ValueError) as error:
        raise LLMResponseFormatError(
            f"ответ модели ({kind.value}) не соответствует внутреннему контракту: {type(error).__name__}",
            truncated=truncated,
            raw_content=content,
            violation=Violation('response_domain', kind.value, safe_message='Ответ не соответствует внутреннему контракту'),
        ) from None
    if byr_evidence_items and getattr(result, "content_plan", None) is not None:
        evs = []
        for item in byr_evidence_items:
            try:
                evs.append(HelpEvidenceResponse(**HelpEvidenceSchema.model_validate(item).model_dump()))
            except ValidationError as error:
                raise LLMResponseFormatError(
                    _format_error(kind, error=error, truncated=truncated),
                    truncated=truncated,
                    raw_content=content,
                    violation=Violation('response_schema', kind.value, safe_message='Ответ не соответствует формату'),
                ) from None
        result = replace(result, content_plan=replace(result.content_plan, before_you_read_evidence=tuple(evs)))
    if content_evidence_items and isinstance(result, TranslationResultResponse):
        evs = []
        for item in content_evidence_items:
            try:
                evs.append(HelpEvidenceResponse(**HelpEvidenceSchema.model_validate(item).model_dump()))
            except ValidationError as error:
                raise LLMResponseFormatError(
                    _format_error(kind, error=error, truncated=truncated),
                    truncated=truncated,
                    raw_content=content,
                    violation=Violation('response_schema', kind.value, safe_message='Ответ не соответствует формату'),
                ) from None
        result = replace(result, content_evidence=tuple(evs))
    if pronunciation_candidates_version == 1:
        candidates = []
        for item in candidate_items:
            try:
                candidates.append(_pronunciation_candidate(PronunciationCandidateSchema.model_validate(item)))
            except (ValidationError, LLMPayloadError, DomainError, ValueError):
                continue
        result = replace(result, pronunciation_candidates=tuple(candidates))
    return result


def _pronunciation_candidate(item: PronunciationCandidateSchema) -> PronunciationCandidateResponse:
    form = item.form if item.form is not None else item.anchor.quote
    return PronunciationCandidateResponse(
        anchor=PronunciationAnchorResponse(**item.anchor.model_dump()), form=form,
        context=PronunciationContextResponse(**item.context.model_dump()), selection_reason=item.selection_reason,
    )


def _to_dto(validated: BaseModel) -> LLMPayload:
    """Converts validated schema object into an application layer DTO."""
    if isinstance(validated, (ChapterCorrectionSchema, ChapterCorrectionV2Schema, ChapterCorrectionReviewSchema)):
        return validated.to_dto()
    if isinstance(validated, QualityReviewSchema):
        return validated.to_dto()
    if isinstance(validated, ChapterAnalysisSchema):
        return _chapter_analysis(validated)
    if isinstance(validated, TranslationResultSchema):
        return _translation_result(validated)
    raise LLMPayloadError(f"нет преобразования для схемы {type(validated).__name__}")


def _chapter_analysis(schema: ChapterAnalysisSchema) -> ChapterAnalysisResponse:
    """Chapter analysis DTO."""
    if isinstance(getattr(schema, 'content_plan', None), QuotedContentPlanSchema):
        content_plan = _quoted_content_plan(schema.content_plan)
    elif isinstance(schema, (ChapterAnalysisV2Schema, ChapterAnalysisV3Schema)):
        content_plan = _content_plan(schema.content_plan)
    else:
        content_plan = None
    return ChapterAnalysisResponse(
        content_plan=content_plan,
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
        name_translation_assessment=(
            NameTranslationAssessment(
                applied_method=schema.name_translation_assessment.applied_method,
                eligible_without_review=schema.name_translation_assessment.eligible_without_review,
                review_reasons=tuple(schema.name_translation_assessment.review_reasons),
                rationale=schema.name_translation_assessment.rationale,
                opaque_name=schema.name_translation_assessment.opaque_name,
            )
            if isinstance(schema, GlossaryCandidateWithNameAssessmentSchema)
            and schema.name_translation_assessment is not None else None
        ),
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
        pronunciation_candidates=tuple(_pronunciation_candidate(item)
            for item in getattr(schema, 'pronunciation_candidates', ())),
        content_evidence=tuple(HelpEvidenceResponse(**e.model_dump()) for e in getattr(schema, 'content_evidence', ())),
        fragment_assistances=tuple(FragmentAssistanceResponse(**item.model_dump()) for item in schema.fragment_assistances) if isinstance(schema, TranslationResultV4Schema) else (),
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
        before_you_read_evidence=tuple(HelpEvidenceResponse(**e.model_dump()) for e in getattr(schema, 'before_you_read_evidence', ())),
        context_facts=tuple(ContextFact(id=fact.id, kind=fact.kind,
            anchors=tuple(SourceAnchor(**anchor.model_dump()) for anchor in fact.anchors),
            subject_anchor=SourceAnchor(**fact.subject_anchor.model_dump()), value=fact.value, certainty=fact.certainty)
            for fact in getattr(schema, 'context_facts', ())),
    )


def _quoted_content_plan(schema: QuotedContentPlanSchema) -> QuotedContentPlanResponse:
    """Preserve v5 quotes without inventing source coordinates at the adapter boundary."""
    return QuotedContentPlanResponse(
        version=schema.version,
        profile=schema.profile,
        difficulties=tuple(QuotedDifficultyResponse(
            id=difficulty.id,
            anchors=tuple(QuotedSourceAnchorResponse(**anchor.model_dump())
                          for anchor in difficulty.anchors),
            owner_block_id=difficulty.owner_block_id,
            kind=difficulty.kind,
            priority=difficulty.priority,
            help_kind=difficulty.help_kind,
            required_points=tuple(RequiredPoint(**point.model_dump())
                                  for point in difficulty.required_points),
        ) for difficulty in schema.difficulties),
        before_you_read_evidence=tuple(HelpEvidenceResponse(**item.model_dump())
                                       for item in getattr(schema, 'before_you_read_evidence', ())),
        context_facts=tuple(QuotedContextFactResponse(
            id=item.id.strip() or f"cf-{index}",
            kind=item.kind,
            anchors=tuple(QuotedSourceAnchorResponse(**anchor.model_dump())
                          for anchor in item.anchors),
            subject_anchor=QuotedSourceAnchorResponse(**item.subject_anchor.model_dump()),
            value=item.value,
            certainty=item.certainty,
        ) for index, item in enumerate(schema.context_facts, start=1)),
    )
