"""Разбор ответа модели и преобразование его во внутренние DTO прикладного слоя.

Здесь проходит граница Pydantic: снаружи — текст ответа провайдера, внутри — обычные
датаклассы `linguary.application.llm_dto` (ADR § 3). Любая неудача разбора приводится
к `LLMResponseFormatError`, чтобы политика повторов не разбирала исключения чужой
библиотеки (PRD § 16).

Обрыв генерации по лимиту токенов провайдер сообщает отдельно от содержимого. Оборванный
JSON не разбирается вовсе, поэтому признак обрыва переносится в ошибку разбора: иначе
причина выглядела бы как случайная поломка формата.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ValidationError

from linguary.adapters.llm.schemas import (
    ChapterAnalysisV2Schema,
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
from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint
from linguary.domain.errors import DomainError

#: Схема ответа для каждого вида запроса.
SCHEMAS: Mapping[LLMResponseKind, type[BaseModel]] = {
    LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisSchema,
    LLMResponseKind.TRANSLATION_RESULT: TranslationResultSchema,
}

def schema_for(kind: LLMResponseKind, *, contract_version: int = 1) -> type[BaseModel]:
    """Схема структурированного ответа для вида запроса."""
    if type(contract_version) is not int or contract_version not in (1, 2):
        raise LLMResponseFormatError("неизвестная версия контракта ответа")
    if contract_version == 2:
        return {LLMResponseKind.CHAPTER_ANALYSIS: ChapterAnalysisV2Schema,
                LLMResponseKind.TRANSLATION_RESULT: TranslationResultV2Schema}[kind]
    return SCHEMAS[kind]


def parse_payload(
    kind: LLMResponseKind, content: str, *, contract_version: int = 1, truncated: bool = False
) -> LLMPayload:
    """Разбирает текст ответа модели и отдаёт внутренний DTO.

    `truncated` — сообщение провайдера об обрыве по лимиту токенов; оно попадает
    в ошибку разбора, когда оборванный ответ не удаётся разобрать.
    """
    schema = schema_for(kind, contract_version=contract_version)
    try:
        validated = schema.model_validate_json(content)
    except ValidationError as error:
        raise LLMResponseFormatError(
            _format_error(kind, error=error, truncated=truncated),
            truncated=truncated,
            raw_content=content,
        ) from error
    try:
        return _to_dto(validated)
    except (LLMPayloadError, DomainError) as error:
        raise LLMResponseFormatError(
            f"ответ модели ({kind.value}) не соответствует внутреннему контракту: "
            f"{type(error).__name__}",
            truncated=truncated,
            raw_content=content,
        ) from error


def _to_dto(validated: BaseModel) -> LLMPayload:
    """Преобразует разобранный ответ в DTO прикладного слоя."""
    if isinstance(validated, ChapterAnalysisSchema):
        return _chapter_analysis(validated)
    if isinstance(validated, TranslationResultSchema):
        return _translation_result(validated)
    raise LLMPayloadError(f"нет преобразования для схемы {type(validated).__name__}")


def _chapter_analysis(schema: ChapterAnalysisSchema) -> ChapterAnalysisResponse:
    """DTO анализа главы."""
    return ChapterAnalysisResponse(
        content_plan=_content_plan(schema.content_plan) if isinstance(schema, ChapterAnalysisV2Schema) else None,
        translation_instructions=schema.translation_instructions,
        chapter_summary=schema.chapter_summary,
        glossary_candidates=tuple(
            _glossary_candidate(item) for item in schema.glossary_candidates
        ),
        treatment_plan=tuple(_plan_item(item) for item in schema.treatment_plan),
        before_you_read=tuple(_before_you_read(item) for item in schema.before_you_read),
    )


def _glossary_candidate(schema: GlossaryCandidateSchema) -> GlossaryCandidateResponse:
    """DTO кандидата глоссария."""
    return GlossaryCandidateResponse(
        source_text=schema.source_text,
        type=schema.type,
        suggested_translation=schema.suggested_translation,
        confidence=schema.confidence,
        model_flagged_review=schema.model_flagged_review,
        note=(schema.note or "").strip(),
    )


def _plan_item(schema: TreatmentPlanItemSchema) -> TreatmentPlanItemResponse:
    """DTO элемента плана обработки."""
    return TreatmentPlanItemResponse(
        source_block_id=schema.source_block_id,
        treatment=BlockTreatment(schema.treatment),
    )


def _before_you_read(schema: BeforeYouReadEntrySchema) -> BeforeYouReadEntryResponse:
    """DTO материала Before You Read."""
    return BeforeYouReadEntryResponse(
        expression=schema.expression, explanation=schema.explanation
    )


def _translation_result(schema: TranslationResultSchema) -> TranslationResultResponse:
    """DTO результата обработки группы блоков."""
    return TranslationResultResponse(
        content_evidence=tuple(HelpEvidenceResponse(**e.model_dump()) for e in schema.content_evidence) if isinstance(schema, TranslationResultV2Schema) else (),
        blocks=tuple(_block_result(item) for item in schema.blocks),
        language_notes=tuple(_language_note(item) for item in schema.language_notes),
    )


def _block_result(schema: BlockResultSchema) -> BlockResultResponse:
    """DTO результата одного блока."""
    return BlockResultResponse(
        source_block_id=schema.source_block_id,
        treatment=BlockTreatment(schema.treatment),
        translated_text=schema.translated_text or None,
        glossary_hints=tuple(_glossary_hint(item) for item in schema.glossary_hints),
    )


def _glossary_hint(schema: GlossaryHintSchema) -> GlossaryHintResponse:
    """DTO словарной подсказки."""
    return GlossaryHintResponse(
        original=schema.original, translation=schema.translation, note=schema.note or None
    )


def _language_note(schema: LanguageNoteSchema) -> LanguageNoteResponse:
    """DTO языковой заметки."""
    return LanguageNoteResponse(
        source_block_id=schema.source_block_id,
        original_excerpt=schema.original_excerpt,
        note=schema.note,
        translated_excerpt=schema.translated_excerpt or None,
    )


def _format_error(
    kind: LLMResponseKind, *, error: ValidationError, truncated: bool
) -> str:
    """Сообщение об ошибке разбора: чем именно ответ не подошёл.

    Обрыв по лимиту токенов называется прямо: иначе он выглядит как случайная поломка
    формата, хотя лечится другим лимитом, а не повтором того же запроса.
    """
    violations = ", ".join(
        sorted({str(item["type"]) for item in error.errors(include_input=False)})
    )
    reason = "обрыв ответа по лимиту токенов" if truncated else "ответ не оборван"
    return (
        f"ответ модели ({kind.value}) не соответствует схеме: "
        f"{error.error_count()} нарушений ({violations}); {reason}"
    )


__all__ = ["SCHEMAS", "parse_payload", "schema_for"]


def _content_plan(schema) -> ContentPlanResponse:
    return ContentPlanResponse(
        version=schema.version, profile=schema.profile,
        difficulties=tuple(Difficulty(
            id=d.id, owner_block_id=d.owner_block_id, kind=d.kind,
            priority=d.priority, help_kind=d.help_kind,
            anchors=tuple(SourceAnchor(**a.model_dump()) for a in d.anchors),
            required_points=tuple(RequiredPoint(**p.model_dump()) for p in d.required_points),
        ) for d in schema.difficulties),
        before_you_read_evidence=tuple(HelpEvidenceResponse(**e.model_dump()) for e in schema.before_you_read_evidence),
    )
