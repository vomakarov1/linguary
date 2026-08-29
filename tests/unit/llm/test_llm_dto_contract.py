"""Внутренние DTO ответов LLM: что они гарантируют прикладному слою (DTO-02)."""

from __future__ import annotations

import pytest

from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import (
    BlockResultResponse,
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
    GlossaryHintResponse,
    LanguageNoteResponse,
    LLMPayloadError,
    TranslationResultResponse,
    TreatmentPlanItemResponse,
)
from linguary.domain import BlockTreatment, ConceptType


def hint() -> GlossaryHintResponse:
    """Словарная подсказка для блока."""
    return GlossaryHintResponse(original="entailed", translation="заповедный")


def test_payload_error_is_an_application_error() -> None:
    """Ошибка DTO остаётся ошибкой прикладного слоя и не тянет за собой чужие типы."""
    assert issubclass(LLMPayloadError, ApplicationError)


def test_translate_block_requires_translation() -> None:
    """Для режима `TRANSLATE` полный перевод обязателен."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(
            source_block_id="block-1", treatment=BlockTreatment.TRANSLATE
        )


def test_read_as_is_block_rejects_glossary_hints() -> None:
    """Блок `READ_AS_IS` не получает словарных подсказок."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(
            source_block_id="block-1",
            treatment=BlockTreatment.READ_AS_IS,
            glossary_hints=(hint(),),
        )


def test_glossary_only_block_rejects_full_translation() -> None:
    """Полный перевод допустим только для `TRANSLATE`."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(
            source_block_id="block-1",
            treatment=BlockTreatment.GLOSSARY_ONLY,
            translated_text="Полный перевод",
            glossary_hints=(hint(),),
        )


@pytest.mark.parametrize(
    "treatment",
    [
        pytest.param(BlockTreatment.PRESERVE_ONLY, id="preserve-only"),
        pytest.param(BlockTreatment.DROP_WITH_WARNING, id="drop-with-warning"),
    ],
)
def test_non_textual_treatment_has_no_model_result(treatment: BlockTreatment) -> None:
    """Нетекстовые блоки модели не передаются, и результата от неё не имеют."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(source_block_id="block-1", treatment=treatment)
    with pytest.raises(LLMPayloadError):
        TreatmentPlanItemResponse(
            source_block_id="block-1", treatment=treatment, reason="служебный блок"
        )


def test_glossary_candidate_confidence_is_bounded() -> None:
    """Уверенность кандидата глоссария лежит в диапазоне 0..1."""
    with pytest.raises(LLMPayloadError):
        GlossaryCandidateResponse(
            source_text="Mr. Bennet",
            type=ConceptType.PERSON,
            suggested_translation="мистер Беннет",
            confidence=1.4,
            model_flagged_review=False,
        )


def test_chapter_analysis_requires_summary_and_instructions() -> None:
    """Анализ без сводки главы и инструкций перевода бессмыслен (PRD § 14, § 19.2)."""
    with pytest.raises(LLMPayloadError):
        ChapterAnalysisResponse(translation_instructions="Переводи", chapter_summary="  ")
    with pytest.raises(LLMPayloadError):
        ChapterAnalysisResponse(translation_instructions="", chapter_summary="События главы")


def test_language_note_belongs_to_one_block() -> None:
    """Языковая заметка привязана ровно к одному блоку (PRD § 19.3)."""
    result = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="block-1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод",
            ),
        ),
        language_notes=(
            LanguageNoteResponse(
                source_block_id="block-1",
                original_excerpt="excerpt",
                note="объяснение",
            ),
        ),
    )

    assert result.notes_for("block-1")[0].note == "объяснение"
    assert result.notes_for("block-2") == ()


def test_note_without_block_is_rejected() -> None:
    """Заметка без блока не привязывается к результату."""
    with pytest.raises(LLMPayloadError):
        LanguageNoteResponse(source_block_id="  ", original_excerpt="excerpt", note="текст")
