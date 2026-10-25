"""Pydantic schemas for structured model responses (DTO-02, ADR § 3).

The schemas describe the external boundary: this is the only place where the LLM response JSON
format is known. The provider receives them as strict structured output JSON Schema, and parsing
the response yields objects ready for conversion. The application layer never sees these classes.

Composition rules visible within the response itself are validated here: mandatory translation
for `TRANSLATE`, prohibition of translation and hints where the treatment disallows them, confidence
range. Everything requiring knowledge of the input — block coverage, block ordering, matching
treatment against the stored plan, result language — is validated by the application layer (PRD § 15).

Extra fields are forbidden: strict structured output does not produce them, and in a free-form
response they indicate that the model did not adhere to the schema.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from linguary.domain import BlockTreatment, ConceptType

#: Processing modes that the model selects and confirms. Non-textual modes
#: (`PRESERVE_ONLY`, `DROP_WITH_WARNING`) are assigned by the importer based on the support
#: matrix, and such blocks are not sent to the model, so they are not in the schema.
type TextualTreatment = Literal["TRANSLATE", "GLOSSARY_ONLY", "READ_AS_IS"]


class _ResponseModel(BaseModel):
    """Common configuration for model response schemas."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GlossaryCandidateSchema(_ResponseModel):
    """Кандидат глоссария, найденный моделью в главе.

    `note` — переводческое указание по решению, которое одним вариантом перевода
    не выражается: род персонажа в целевом языке, склонение названия, регистр речи.
    """

    source_text: str = Field(min_length=1)
    type: ConceptType
    suggested_translation: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    model_flagged_review: bool
    note: str | None


class TreatmentPlanItemSchema(_ResponseModel):
    """Предложенный моделью режим обработки одного блока.

    Обоснования режима схема не просит: оно нигде не сохраняется, а объём вывода
    определяет длительность ответа анализа.
    """

    source_block_id: str = Field(min_length=1)
    treatment: TextualTreatment


class BeforeYouReadEntrySchema(_ResponseModel):
    """Выражение главы вместе с объяснением (PRD § 19.1)."""

    expression: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ChapterAnalysisSchema(_ResponseModel):
    """Ответ анализа главы или её фрагмента (PRD § 11, § 14, § 19)."""

    glossary_candidates: list[GlossaryCandidateSchema]
    translation_instructions: str = Field(min_length=1)
    treatment_plan: list[TreatmentPlanItemSchema]
    before_you_read: list[BeforeYouReadEntrySchema]
    chapter_summary: str = Field(min_length=1)


class GlossaryHintSchema(_ResponseModel):
    """Перевод слова или короткого выражения внутри блока."""

    original: str = Field(min_length=1)
    translation: str = Field(min_length=1)
    note: str | None


class BlockResultSchema(_ResponseModel):
    """Результат обработки одного входного текстового блока."""

    source_block_id: str = Field(min_length=1)
    treatment: TextualTreatment
    translated_text: str | None
    glossary_hints: list[GlossaryHintSchema]

    @model_validator(mode="after")
    def _check_treatment_composition(self) -> BlockResultSchema:
        """The result composition must correspond to the declared treatment (PRD § 15)."""
        treatment = BlockTreatment(self.treatment)
        has_translation = bool(self.translated_text and self.translated_text.strip())
        if treatment.requires_translation and not has_translation:
            raise ValueError(
                f"для блока {self.source_block_id} с режимом {treatment} "
                "обязателен полный перевод"
            )
        if not treatment.allows_translation and has_translation:
            raise ValueError(
                f"для блока {self.source_block_id} с режимом {treatment} "
                "полный перевод недопустим"
            )
        if treatment.requires_glossary_hints and not self.glossary_hints:
            raise ValueError(
                f"для блока {self.source_block_id} с режимом {treatment} "
                "обязательны словарные подсказки"
            )
        if not treatment.allows_glossary_hints and self.glossary_hints:
            raise ValueError(
                f"для блока {self.source_block_id} с режимом {treatment} "
                "словарные подсказки недопустимы"
            )
        return self


class LanguageNoteSchema(_ResponseModel):
    """Языковой комментарий к одному блоку (PRD § 19.3)."""

    source_block_id: str = Field(min_length=1)
    original_excerpt: str = Field(min_length=1)
    translated_excerpt: str | None
    note: str = Field(min_length=1)


class TranslationResultSchema(_ResponseModel):
    """Ответ обработки группы блоков (PRD § 10, § 14)."""

    blocks: list[BlockResultSchema] = Field(min_length=1)
    language_notes: list[LanguageNoteSchema]


__all__ = [
    "BeforeYouReadEntrySchema",
    "BlockResultSchema",
    "ChapterAnalysisSchema",
    "GlossaryCandidateSchema",
    "GlossaryHintSchema",
    "LanguageNoteSchema",
    "TextualTreatment",
    "TranslationResultSchema",
    "TreatmentPlanItemSchema",
]


class SourceAnchorSchema(_ResponseModel):
    block_id: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    quote: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)]


class RequiredPointSchema(_ResponseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DifficultySchema(_ResponseModel):
    id: str = Field(min_length=1)
    anchors: list[SourceAnchorSchema] = Field(min_length=1)
    owner_block_id: str = Field(min_length=1)
    kind: Literal['wordplay', 'semantic_relation', 'grammar', 'cultural_reference', 'idiom', 'vocabulary']
    priority: Literal['critical', 'useful']
    help_kind: Literal['explanation', 'lexical_hint']
    required_points: list[RequiredPointSchema] = Field(min_length=1)


class HelpEvidenceSchema(_ResponseModel):
    difficulty_id: str = Field(min_length=1)
    point_id: str = Field(min_length=1)
    artifact_kind: Literal['before_you_read.explanation', 'language_note.note', 'glossary_hint.note', 'glossary_hint.translation']
    source_block_id: str = Field(min_length=1)
    artifact_index: int = Field(ge=0, strict=True)
    body_excerpt: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)]


class ContentPlanSchema(_ResponseModel):
    version: Literal[2]
    profile: Literal['content-plan-v2']
    difficulties: list[DifficultySchema]
    before_you_read_evidence: list[HelpEvidenceSchema]


class ChapterAnalysisV2Schema(ChapterAnalysisSchema):
    content_plan: ContentPlanSchema


class TranslationResultV2Schema(TranslationResultSchema):
    content_evidence: list[HelpEvidenceSchema]


class ContextFactSchema(_ResponseModel):
    id: str = Field(min_length=1)
    kind: Literal['speaker', 'gender', 'pronoun_referent']
    anchors: list[SourceAnchorSchema] = Field(min_length=1)
    subject_anchor: SourceAnchorSchema
    value: str | None = None
    certainty: Literal['explicit', 'inferred', 'unknown']

    @model_validator(mode='after')
    def _check_certainty(self):
        if (self.certainty == 'unknown') != (self.value is None):
            raise ValueError('неопределённость и значение атрибуции не согласованы')
        return self


class ContentPlanV3Schema(ContentPlanSchema):
    version: Literal[3]
    profile: Literal['content-plan-v3']
    context_facts: list[ContextFactSchema]


class ChapterAnalysisV3Schema(ChapterAnalysisSchema):
    content_plan: ContentPlanV3Schema


class TranslationResultV3Schema(TranslationResultSchema):
    content_evidence: list[HelpEvidenceSchema]
