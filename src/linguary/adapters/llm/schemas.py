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

from collections.abc import Mapping
import copy
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from linguary.domain import BlockTreatment, ConceptType
from linguary.domain.name_translation import NameTranslationMethod, NameTranslationReviewReason

#: Processing modes that the model selects and confirms. Non-textual modes
#: (`PRESERVE_ONLY`, `DROP_WITH_WARNING`) are assigned by the importer based on the support
#: matrix, and such blocks are not sent to the model, so they are not in the schema.
type TextualTreatment = Literal["TRANSLATE", "GLOSSARY_ONLY", "READ_AS_IS"]


class _ResponseModel(BaseModel):
    """Common configuration for model response schemas."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# Поля учебного контракта остаются в модели, но не входят в замороженную строгую схему v1.
_V1_ANALYSIS_FIELDS = {
    "TreatmentPlanItemSchema": frozenset({"segments"}),
    "GlossaryCandidateSchema": frozenset({"description"}),
}


def _referenced_defs(schema: dict[str, Any]) -> set[str]:
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return set()
    seen: set[str] = set()
    stack: list[Any] = [{key: value for key, value in schema.items() if key != "$defs"}]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name not in seen:
                    seen.add(name)
                    if name in defs:
                        stack.append(defs[name])
            stack.extend(value for key, value in node.items() if key != "$ref")
        elif isinstance(node, list):
            stack.extend(node)
    return seen


def _omit_wire_fields(
    schema: dict[str, Any], omitted: dict[str, frozenset[str]]
) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema
    for name, fields in omitted.items():
        node = defs.get(name)
        if not isinstance(node, dict):
            continue
        properties = node.get("properties")
        if isinstance(properties, dict):
            for field in fields:
                properties.pop(field, None)
        required = node.get("required")
        if isinstance(required, list):
            node["required"] = [item for item in required if item not in fields]
    used = _referenced_defs(schema)
    for name in [name for name in defs if name not in used]:
        defs.pop(name)
    return schema


def _v1_analysis_wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _omit_wire_fields(schema, _V1_ANALYSIS_FIELDS)


# Переводы сегментов принимаются моделью, но не входят в замороженную строгую схему v1.
_V1_TRANSLATION_FIELDS = {
    "BlockResultSchema": frozenset({"segments"}),
}


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
    description: str = ""


class SegmentTreatmentSchema(_ResponseModel):
    """Режим одного сегмента. Индекс начинается с 1."""

    index: int = Field(ge=1, strict=True)
    treatment: TextualTreatment


class TreatmentPlanItemSchema(_ResponseModel):
    """Предложенный моделью режим обработки одного блока.

    Обоснования режима схема не просит: оно нигде не сохраняется, а объём вывода
    определяет длительность ответа анализа.
    """

    source_block_id: str = Field(min_length=1)
    treatment: TextualTreatment
    segments: list[SegmentTreatmentSchema] = Field(default_factory=list)


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

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        if cls is not ChapterAnalysisSchema:
            return schema
        return _v1_analysis_wire_schema(schema)


class GlossaryHintSchema(_ResponseModel):
    """Перевод слова или короткого выражения внутри блока."""

    original: str = Field(min_length=1)
    translation: str = Field(min_length=1)
    note: str | None


class SegmentTranslationSchema(_ResponseModel):
    """Перевод одного сегмента. Индекс начинается с 1."""

    index: int = Field(
        ge=1,
        strict=True,
        description=(
            "1-based номер сегмента M из метки [bN.M] в исходном блоке "
            "(только для сегментов TRANSLATE; не с 0 и без перенумерации пропущенных сегментов)"
        ),
    )
    translated_text: str | None

    @field_validator("translated_text")
    @classmethod
    def _reject_blank_translation(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("пустой перевод сегмента")
        return value


def _is_valid_empty_segment(item: object) -> bool:
    """Whether `item` is a well-formed empty/null segment safe to drop before validation."""
    if isinstance(item, SegmentTranslationSchema):
        return item.translated_text is None
    if not isinstance(item, dict) or set(item) != {"index", "translated_text"}:
        return False
    index = item["index"]
    translated_text = item["translated_text"]
    if type(index) is not int or index < 1:
        return False
    return translated_text is None or (
        isinstance(translated_text, str) and not translated_text.strip()
    )


class BlockResultSchema(_ResponseModel):
    """Результат обработки одного входного текстового блока."""

    source_block_id: str = Field(min_length=1)
    treatment: TextualTreatment
    translated_text: str | None
    glossary_hints: list[GlossaryHintSchema] = Field(
        description=(
            "Словарные подсказки к блоку: обязателен непустой список (минимум 1 подсказка) "
            "для каждого блока в режиме GLOSSARY_ONLY и для смешанного блока с сегментом GLOSSARY_ONLY; "
            "для READ_AS_IS всегда []"
        ),
    )
    # Пустой список сохраняет проверку состава по режиму блока.
    segments: list[SegmentTranslationSchema] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_valid_empty_segments(cls, data: object) -> object:
        """Drop valid empty/null segments and exact empty block translation on a dict copy."""
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        if copied.get("translated_text") == "":
            copied["translated_text"] = None
        segments = copied.get("segments")
        if isinstance(segments, list):
            copied["segments"] = [
                item for item in segments if not _is_valid_empty_segment(item)
            ]
        return copied

    @model_validator(mode="after")
    def _check_treatment_composition(self) -> BlockResultSchema:
        """The result composition must correspond to the declared treatment (PRD § 15)."""
        if self.segments:
            if self.translated_text is not None:
                raise ValueError(
                    "нельзя одновременно заполнять translated_text и segments: "
                    "для блока без меток сегментов верни перевод в translated_text и segments: []; "
                    "для блока с метками сегментов верни translated_text: null и заполни segments"
                )
            return self
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

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        if cls is not TranslationResultSchema:
            return schema
        wire = _omit_wire_fields(schema, _V1_TRANSLATION_FIELDS)
        defs = wire.get("$defs")
        if isinstance(defs, dict):
            block_def = defs.get("BlockResultSchema")
            if isinstance(block_def, dict):
                props = block_def.get("properties")
                if isinstance(props, dict) and isinstance(props.get("glossary_hints"), dict):
                    props["glossary_hints"].pop("description", None)
        return wire


__all__ = [
    "BeforeYouReadEntrySchema",
    "BlockResultSchema",
    "ChapterAnalysisSchema",
    "GlossaryCandidateSchema",
    "GlossaryHintSchema",
    "LanguageNoteSchema",
    "NarrativeAnalysisSchema",
    "NarrativeAnalysisWithNameAssessmentSchema",
    "PedagogicalAnalysisA2Schema",
    "PedagogicalAnalysisSchema",
    "QuotedSceneSchema",
    "QuotedSceneVisualFactSchema",
    "SegmentTranslationSchema",
    "SegmentTreatmentSchema",
    "TextualTreatment",
    "TranslationResultSchema",
    "TreatmentPlanItemSchema",
]


class SourceAnchorSchema(_ResponseModel):
    block_id: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    quote: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)]


class PronunciationAnchorSchema(_ResponseModel):
    block_id: str = Field(min_length=1)
    quote: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)]
    start: int | None = Field(default=None, ge=0, strict=True)
    end: int | None = Field(default=None, gt=0, strict=True)


class PronunciationContextSchema(_ResponseModel):
    meaning_in_context: Annotated[str, StringConstraints(min_length=1)] | None
    part_of_speech: Annotated[str, StringConstraints(min_length=1)] | None
    grammatical_form: Annotated[str, StringConstraints(min_length=1)] | None
    is_proper_name: bool = Field(strict=True)


class PronunciationCandidateSchema(_ResponseModel):
    anchor: PronunciationAnchorSchema
    form: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)] | None = None
    context: PronunciationContextSchema
    selection_reason: Literal['unpredictable_spelling', 'unpredictable_stress', 'context_dependent']

    @model_validator(mode="after")
    def _check_form_matches_quote(self) -> PronunciationCandidateSchema:
        if self.form is not None and self.form != self.anchor.quote:
            raise ValueError("форма кандидата должна совпадать с цитатой якоря")
        return self


class _PronunciationCandidatesSchema(_ResponseModel):
    pronunciation_candidates: list[PronunciationCandidateSchema]


class RequiredPointSchema(_ResponseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DifficultySchema(_ResponseModel):
    id: str = Field(min_length=1)
    anchors: list[SourceAnchorSchema] = Field(min_length=1)
    owner_block_id: str = Field(min_length=1)
    kind: Literal[
        'vocabulary', 'grammar', 'idiom', 'cultural_reference',
        'speech_style', 'false_friend', 'semantic_relation', 'wordplay', 'parody',
    ]
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
    before_you_read_evidence: list[HelpEvidenceSchema] = Field(default_factory=list)


class ChapterAnalysisV2Schema(ChapterAnalysisSchema):
    content_plan: ContentPlanSchema


class TranslationResultV2Schema(TranslationResultSchema):
    content_evidence: list[HelpEvidenceSchema] = Field(default_factory=list)


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
    version: Literal[3]  # ty: ignore[incompatible-attribute-override]
    profile: Literal['content-plan-v3']  # ty: ignore[incompatible-attribute-override]
    context_facts: list[ContextFactSchema]


class ChapterAnalysisV3Schema(ChapterAnalysisSchema):
    content_plan: ContentPlanV3Schema


class QuotedSourceAnchorSchema(_ResponseModel):
    model_config = ConfigDict(extra="ignore")
    block_id: str = Field(min_length=1)
    quote: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)]
    start: int | None = Field(default=None, strict=True)
    end: int | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def _check_quote(self):
        if not self.quote.strip():
            raise ValueError('цитата не должна быть пустой')
        return self


class QuotedDifficultySchema(DifficultySchema):
    anchors: list[QuotedSourceAnchorSchema] = Field(min_length=1)  # ty: ignore[incompatible-attribute-override]


class QuotedContextFactSchema(ContextFactSchema):
    id: str = Field(default="")
    anchors: list[QuotedSourceAnchorSchema] = Field(min_length=1)  # ty: ignore[incompatible-attribute-override]
    subject_anchor: QuotedSourceAnchorSchema  # ty: ignore[incompatible-attribute-override]

    @model_validator(mode="before")
    @classmethod
    def _fallback_inferred_anchors(cls, data: object) -> object:
        if (
            isinstance(data, dict)
            and isinstance(certainty := data.get("certainty"), str)
            and certainty.strip() == "inferred"
            and isinstance(data.get("anchors"), list)
            and len(data["anchors"]) == 0
            and isinstance(data.get("subject_anchor"), dict)
        ):
            return {**data, "anchors": [dict(data["subject_anchor"])]}
        return data


class QuotedContentPlanSchema(ContentPlanV3Schema):
    version: Literal[3] = 3
    profile: Literal['content-plan-v3'] = 'content-plan-v3'
    difficulties: list[QuotedDifficultySchema]  # ty: ignore[incompatible-attribute-override]
    context_facts: list[QuotedContextFactSchema]  # ty: ignore[incompatible-attribute-override]


class ChapterAnalysisV5Schema(ChapterAnalysisSchema):
    content_plan: QuotedContentPlanSchema


class TranslationResultV3Schema(TranslationResultSchema):
    content_evidence: list[HelpEvidenceSchema] = Field(default_factory=list)


class FragmentAssistanceSchema(_ResponseModel):
    difficulty_id: str = Field(min_length=1)
    point_ids: list[Annotated[str, StringConstraints(min_length=1)]]
    kind: Literal['lexical_hint', 'sentence_rephrase', 'russian_explanation', 'unresolved']
    text: str | None

    @model_validator(mode='after')
    def _check_body(self):
        from linguary.domain.learning import validate_assistance_body
        from linguary.domain.errors import DomainError
        try:
            validate_assistance_body(self.kind, self.text)
        except DomainError as error:
            raise ValueError(str(error)) from error
        if len(set(self.point_ids)) != len(self.point_ids):
            raise ValueError('повтор смыслового пункта помощи')
        return self


class TranslationResultV4Schema(TranslationResultV3Schema):
    fragment_assistances: list[FragmentAssistanceSchema]


class NameTranslationAssessmentSchema(_ResponseModel):
    applied_method: NameTranslationMethod
    eligible_without_review: bool = Field(strict=True)
    review_reasons: list[NameTranslationReviewReason]
    rationale: str = Field(min_length=1)
    opaque_name: bool = Field(strict=True)

    @model_validator(mode="after")
    def _deduplicate_review_reasons(self) -> NameTranslationAssessmentSchema:
        self.review_reasons = list(dict.fromkeys(self.review_reasons))
        return self

    @model_validator(mode="after")
    def _check_negative_reason(self) -> NameTranslationAssessmentSchema:
        if not self.eligible_without_review and not self.review_reasons:
            raise ValueError("negative assessment requires a reason")
        return self


class GlossaryCandidateWithNameAssessmentSchema(GlossaryCandidateSchema):
    name_translation_assessment: NameTranslationAssessmentSchema | None


class ChapterAnalysisWithNameAssessmentSchema(ChapterAnalysisSchema):
    glossary_candidates: list[GlossaryCandidateWithNameAssessmentSchema]  # ty: ignore[incompatible-attribute-override]


class ChapterAnalysisV2WithNameAssessmentSchema(ChapterAnalysisV2Schema):
    glossary_candidates: list[GlossaryCandidateWithNameAssessmentSchema]  # ty: ignore[incompatible-attribute-override]


class ChapterAnalysisV3WithNameAssessmentSchema(ChapterAnalysisV3Schema):
    glossary_candidates: list[GlossaryCandidateWithNameAssessmentSchema]  # ty: ignore[incompatible-attribute-override]


class ChapterAnalysisV5WithNameAssessmentSchema(ChapterAnalysisV5Schema):
    glossary_candidates: list[GlossaryCandidateWithNameAssessmentSchema]  # ty: ignore[incompatible-attribute-override]


class TranslationWithPronunciationSchema(_PronunciationCandidatesSchema, TranslationResultSchema):
    pass


class TranslationV2WithPronunciationSchema(_PronunciationCandidatesSchema, TranslationResultV2Schema):
    pass


class TranslationV3WithPronunciationSchema(_PronunciationCandidatesSchema, TranslationResultV3Schema):
    pass


class TranslationV4WithPronunciationSchema(_PronunciationCandidatesSchema, TranslationResultV4Schema):
    pass


class QuotedSceneVisualFactSchema(_ResponseModel):
    anchor: QuotedSourceAnchorSchema
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class QuotedSceneSchema(_ResponseModel):
    start_block_id: str = Field(min_length=1)
    end_block_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    location: str | None = None
    participants: list[str] = Field(default_factory=list)
    visual_facts: list[QuotedSceneVisualFactSchema] = Field(default_factory=list)
    visual_assumptions: list[str] = Field(default_factory=list)
    continues_previous_scene: bool = False


class NarrativeAnalysisSchema(_ResponseModel):
    translation_instructions: str = Field(min_length=1)
    chapter_summary: str = Field(min_length=1)
    glossary_candidates: list[GlossaryCandidateSchema]
    context_facts: list[QuotedContextFactSchema]
    scenes: list[QuotedSceneSchema]


class NarrativeAnalysisWithNameAssessmentSchema(NarrativeAnalysisSchema):
    glossary_candidates: list[GlossaryCandidateWithNameAssessmentSchema]  # ty: ignore[incompatible-attribute-override]


class PedagogicalAnalysisSchema(_ResponseModel):
    treatment_plan: list[TreatmentPlanItemSchema]
    before_you_read: list[BeforeYouReadEntrySchema]
    difficulties: list[QuotedDifficultySchema]
    before_you_read_evidence: list[HelpEvidenceSchema] = Field(default_factory=list)


class PedagogicalAnalysisA2Schema(PedagogicalAnalysisSchema):
    reader_chapter_summary: str = Field(min_length=1)
