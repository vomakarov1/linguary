"""Internal LLM response DTOs: guarantees provided to the application layer (DTO-02)."""

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
    """Glossary hint for a block."""
    return GlossaryHintResponse(original="entailed", translation="заповедный")


def test_payload_error_is_an_application_error() -> None:
    """DTO error remains an application layer error and does not introduce foreign types."""
    assert issubclass(LLMPayloadError, ApplicationError)


def test_translate_block_requires_translation() -> None:
    """For 'TRANSLATE' treatment, a full translation is required."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(
            source_block_id="block-1", treatment=BlockTreatment.TRANSLATE
        )


def test_read_as_is_block_rejects_glossary_hints() -> None:
    """A 'READ_AS_IS' block does not receive glossary hints."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(
            source_block_id="block-1",
            treatment=BlockTreatment.READ_AS_IS,
            glossary_hints=(hint(),),
        )


def test_glossary_only_block_rejects_full_translation() -> None:
    """Full translation is allowed only for 'TRANSLATE'."""
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
    """Non-textual blocks are not passed to the model and have no result from it."""
    with pytest.raises(LLMPayloadError):
        BlockResultResponse(source_block_id="block-1", treatment=treatment)
    with pytest.raises(LLMPayloadError):
        TreatmentPlanItemResponse(source_block_id="block-1", treatment=treatment)


def test_glossary_candidate_confidence_is_bounded() -> None:
    """Glossary candidate confidence must be in the 0..1 range."""
    with pytest.raises(LLMPayloadError):
        GlossaryCandidateResponse(
            source_text="Mr. Bennet",
            type=ConceptType.PERSON,
            suggested_translation="мистер Беннет",
            confidence=1.4,
            model_flagged_review=False,
        )


def test_chapter_analysis_requires_summary_and_instructions() -> None:
    """Analysis without chapter summary and translation instructions is meaningless (PRD § 14, § 19.2)."""
    with pytest.raises(LLMPayloadError):
        ChapterAnalysisResponse(translation_instructions="Переводи", chapter_summary="  ")
    with pytest.raises(LLMPayloadError):
        ChapterAnalysisResponse(translation_instructions="", chapter_summary="События главы")


def test_language_note_belongs_to_one_block() -> None:
    """A language note is bound to exactly one block (PRD § 19.3)."""
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
    """A note without a block cannot be attached to a result."""
    with pytest.raises(LLMPayloadError):
        LanguageNoteResponse(source_block_id="  ", original_excerpt="excerpt", note="текст")
