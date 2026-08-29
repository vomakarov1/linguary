"""Pydantic-схемы структурированных ответов модели (DTO-02, ADR § 3).

Схемы описывают внешнюю границу: это единственное место, где известен JSON-формат ответа
LLM. Провайдер получает их как JSON Schema строгого структурированного вывода, а разбор
ответа даёт готовые к преобразованию объекты. Прикладной слой этих классов не видит.

Правила состава, которые видны в самом ответе, проверяются здесь: обязательный перевод
для `TRANSLATE`, запрет перевода и подсказок там, где режим их не допускает, диапазон
уверенности. Всё, что требует знания входа — покрытие блоков, их порядок, совпадение
режима с сохранённым планом, язык результата, — проверяет прикладной слой (PRD § 15).

Лишние поля запрещены: строгий структурированный вывод их не порождает, а в свободном
ответе они означают, что модель отвечала не по схеме.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linguary.domain import BlockTreatment, ConceptType

#: Режимы обработки, которые модель выбирает и подтверждает. Нетекстовые режимы
#: (`PRESERVE_ONLY`, `DROP_WITH_WARNING`) назначает импортёр по матрице поддержки,
#: и такие блоки модели не передаются, поэтому в схеме их нет.
type TextualTreatment = Literal["TRANSLATE", "GLOSSARY_ONLY", "READ_AS_IS"]


class _ResponseModel(BaseModel):
    """Общая настройка схем ответа модели."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GlossaryCandidateSchema(_ResponseModel):
    """Кандидат глоссария, найденный моделью в главе."""

    source_text: str = Field(min_length=1)
    type: ConceptType
    suggested_translation: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    model_flagged_review: bool


class TreatmentPlanItemSchema(_ResponseModel):
    """Предложенный моделью режим обработки одного блока и его обоснование."""

    source_block_id: str = Field(min_length=1)
    treatment: TextualTreatment
    reason: str


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
        """Состав результата обязан соответствовать заявленному режиму (PRD § 15)."""
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
