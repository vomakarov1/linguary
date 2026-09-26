"""Внутренние DTO ответов LLM (DTO-02).

Это форма ответа модели после того, как адаптер разобрал и проверил его по Pydantic-схеме
на своей границе. Прикладной слой работает только с этими датаклассами и о Pydantic,
JSON и провайдере не знает (ADR § 3).

DTO не повторяют доменные правила: проверки полноты группы, порядка блоков, совпадения
режима с сохранённым планом, языка и оборванной генерации выполняются уже над этими DTO
прикладными валидаторами (PRD § 15, ALG-04). Здесь остаётся только то, без чего DTO
бессмысленны как носители: непустые идентификаторы и тексты, диапазон уверенности.
"""

from __future__ import annotations

from dataclasses import dataclass

from linguary.application.errors import ApplicationError
from linguary.domain import BlockTreatment, ConceptType


class LLMPayloadError(ApplicationError):
    """Ответ модели нельзя представить внутренним DTO.

    Ошибка границы: адаптер уже проверил ответ по схеме, поэтому её появление означает
    расхождение схемы адаптера с этим контрактом, а не обычный некорректный ответ модели.
    """


@dataclass(frozen=True, slots=True)
class GlossaryCandidateResponse:
    """Кандидат глоссария из анализа главы (PRD § 11).

    `model_flagged_review` — собственная пометка модели, входной сигнал. Итоговое
    `requires_review` вычисляет приложение по DTO-01 (`domain.compute_requires_review`).

    `note` — переводческое указание, которое одним вариантом перевода не выражается:
    род персонажа в целевом языке, склонение названия, регистр речи (PRD § 11).
    """

    source_text: str
    type: ConceptType
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source_text.strip():
            raise LLMPayloadError("исходный текст кандидата глоссария пуст")
        if not self.suggested_translation.strip():
            raise LLMPayloadError(
                f"предложенный перевод кандидата {self.source_text!r} пуст"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise LLMPayloadError(
                f"уверенность модели вне диапазона 0..1: {self.confidence}"
            )


@dataclass(frozen=True, slots=True)
class TreatmentPlanItemResponse:
    """Предложенный моделью режим обработки одного блока (PRD § 10.1).

    План становится источником истины только после сохранения: при обработке поле
    `treatment` в ответе модели этот план подтверждает, но не переопределяет.
    """

    source_block_id: str
    treatment: BlockTreatment

    def __post_init__(self) -> None:
        if not self.source_block_id.strip():
            raise LLMPayloadError("элемент плана обработки не привязан к блоку")
        if not self.treatment.is_textual:
            raise LLMPayloadError(
                f"режим {self.treatment} блока {self.source_block_id} не выбирается моделью: "
                "нетекстовые блоки классифицирует импортёр"
            )


@dataclass(frozen=True, slots=True)
class BeforeYouReadEntryResponse:
    """Выражение главы, для которого обычного словаря может быть мало (PRD § 19.1)."""

    expression: str
    explanation: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise LLMPayloadError("выражение материала Before You Read пусто")
        if not self.explanation.strip():
            raise LLMPayloadError(f"объяснение выражения {self.expression!r} пусто")


@dataclass(frozen=True, slots=True)
class ChapterAnalysisResponse:
    """Ответ модели на анализ одной главы или одного её фрагмента (PRD § 14).

    `chapter_summary` описывает события этой главы и пишется на языке оригинала:
    экспортёр показывает её читателю перед следующей главой (PRD § 19.2).
    """

    translation_instructions: str
    chapter_summary: str
    glossary_candidates: tuple[GlossaryCandidateResponse, ...] = ()
    treatment_plan: tuple[TreatmentPlanItemResponse, ...] = ()
    before_you_read: tuple[BeforeYouReadEntryResponse, ...] = ()

    def __post_init__(self) -> None:
        if not self.translation_instructions.strip():
            raise LLMPayloadError("инструкции перевода главы пусты")
        if not self.chapter_summary.strip():
            raise LLMPayloadError("сводка событий главы пуста")

    @property
    def planned_block_ids(self) -> tuple[str, ...]:
        """Блоки, покрытые планом, в порядке ответа модели.

        Полнота покрытия относительно текстовых блоков главы проверяется прикладным
        слоем при сохранении анализа: сам ответ о составе главы не знает.
        """
        return tuple(item.source_block_id for item in self.treatment_plan)


@dataclass(frozen=True, slots=True)
class GlossaryHintResponse:
    """Перевод отдельного слова или короткого выражения внутри блока (PRD § 10.1)."""

    original: str
    translation: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise LLMPayloadError("исходное выражение словарной подсказки пусто")
        if not self.translation.strip():
            raise LLMPayloadError(f"перевод подсказки {self.original!r} пуст")


@dataclass(frozen=True, slots=True)
class BlockResultResponse:
    """Результат обработки одного входного текстового блока.

    Состав определяется режимом: полный перевод обязателен и допустим только для
    `TRANSLATE`, словарные подсказки обязательны для `GLOSSARY_ONLY` и запрещены
    для `READ_AS_IS` (PRD § 15).
    """

    source_block_id: str
    treatment: BlockTreatment
    translated_text: str | None = None
    glossary_hints: tuple[GlossaryHintResponse, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_block_id.strip():
            raise LLMPayloadError("результат обработки не привязан к блоку")
        if not self.treatment.is_textual:
            raise LLMPayloadError(
                f"блок {self.source_block_id} с режимом {self.treatment} модели не передаётся"
            )
        has_translation = bool(self.translated_text and self.translated_text.strip())
        if self.treatment.requires_translation and not has_translation:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} обязателен полный перевод"
            )
        if not self.treatment.allows_translation and self.translated_text is not None:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} с режимом {self.treatment} "
                "полный перевод недопустим"
            )
        if self.treatment.requires_glossary_hints and not self.glossary_hints:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} обязательны словарные подсказки"
            )
        if not self.treatment.allows_glossary_hints and self.glossary_hints:
            raise LLMPayloadError(
                f"для блока {self.source_block_id} с режимом {self.treatment} "
                "словарные подсказки недопустимы"
            )


@dataclass(frozen=True, slots=True)
class LanguageNoteResponse:
    """Языковой комментарий, привязанный ровно к одному блоку (PRD § 19.3)."""

    source_block_id: str
    original_excerpt: str
    note: str
    translated_excerpt: str | None = None

    def __post_init__(self) -> None:
        if not self.source_block_id.strip():
            raise LLMPayloadError("языковая заметка не привязана к блоку")
        if not self.original_excerpt.strip():
            raise LLMPayloadError(
                f"фрагмент оригинала заметки блока {self.source_block_id} пуст"
            )
        if not self.note.strip():
            raise LLMPayloadError(f"текст заметки блока {self.source_block_id} пуст")


@dataclass(frozen=True, slots=True)
class TranslationResultResponse:
    """Ответ модели на обработку одной группы блоков (PRD § 10, § 14).

    Соответствие состава и порядка блоков входной группе проверяется прикладным слоем:
    ответ сам по себе не знает, какая группа его запросила.
    """

    blocks: tuple[BlockResultResponse, ...] = ()
    language_notes: tuple[LanguageNoteResponse, ...] = ()

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Блоки результата в порядке ответа модели."""
        return tuple(block.source_block_id for block in self.blocks)

    def notes_for(self, block_id: str) -> tuple[LanguageNoteResponse, ...]:
        """Заметки ответа, относящиеся к указанному блоку."""
        return tuple(note for note in self.language_notes if note.source_block_id == block_id)


__all__ = [
    "BeforeYouReadEntryResponse",
    "BlockResultResponse",
    "ChapterAnalysisResponse",
    "GlossaryCandidateResponse",
    "GlossaryHintResponse",
    "LLMPayloadError",
    "LanguageNoteResponse",
    "TranslationResultResponse",
    "TreatmentPlanItemResponse",
]
