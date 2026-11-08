"""Yo/ye normalization in glossary keys and majority frequency selection in glossary review (P3.T1)."""

from __future__ import annotations

from linguary.application.glossary_review import build_glossary_review
from linguary.domain import (
    ChapterAnalysis,
    ConceptType,
    GlossaryCandidate,
    TreatmentPlan,
    glossary_key,
    translations_conflict,
)
from linguary.domain.glossary import _comparable_translation

CONFIDENCE_THRESHOLD = 0.8


def _chapter(
    index: int,
    *,
    source_text: str = "Dark Lord",
    suggested_translation: str,
    confidence: float,
    concept_type: ConceptType = ConceptType.TERM,
) -> ChapterAnalysis:
    analysis_id = f"analysis-{index}"
    section_id = f"section-{index}"
    candidate = GlossaryCandidate.create(
        id=f"candidate-{index}",
        analysis_id=analysis_id,
        source_text=source_text,
        type=concept_type,
        suggested_translation=suggested_translation,
        confidence=confidence,
        model_flagged_review=False,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    return ChapterAnalysis.create(
        id=analysis_id,
        section_id=section_id,
        treatment_plan=TreatmentPlan(section_id=section_id),
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(),
        glossary_candidates=(candidate,),
    )


def test_glossary_key_and_comparable_translation_normalize_yo_to_ye() -> None:
    """Cyrillic 'ё' and 'е' normalize to the same glossary key and do not cause false translation conflicts."""
    assert glossary_key("Тёмный лорд") == glossary_key("Темный лорд")
    assert glossary_key("Ёжик") == glossary_key("ежик")
    assert _comparable_translation("Тёмный лорд") == _comparable_translation("Темный лорд")
    assert not translations_conflict(["Тёмный лорд", "Темный лорд"])

    review = build_glossary_review(
        book_id="book-1",
        analyses=(
            _chapter(1, suggested_translation="Тёмный лорд", confidence=0.90),
            _chapter(2, suggested_translation="Темный лорд", confidence=0.95),
        ),
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    assert len(review.items) == 1
    assert review.items[0].has_translation_conflict is False
    assert review.items[0].requires_review is False


def test_build_glossary_review_prefers_majority_translation_across_chapters() -> None:
    """`suggested_translation` selects the variant supported by most chapters rather than a single high-confidence outlier."""
    review = build_glossary_review(
        book_id="book-1",
        analyses=(
            _chapter(1, suggested_translation="Повелитель Тьмы", confidence=0.98),
            _chapter(2, suggested_translation="Тёмный лорд", confidence=0.85),
            _chapter(3, suggested_translation="Темный лорд", confidence=0.89),
            _chapter(4, suggested_translation="Тёмный лорд", confidence=0.86),
        ),
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    assert len(review.items) == 1
    item = review.items[0]
    assert item.suggested_translation == "Темный лорд"
    assert item.has_translation_conflict is True

    tied_review = build_glossary_review(
        book_id="book-1",
        analyses=(
            _chapter(1, suggested_translation="Повелитель Тьмы", confidence=0.95),
            _chapter(2, suggested_translation="Тёмный лорд", confidence=0.90),
        ),
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    assert tied_review.items[0].suggested_translation == "Повелитель Тьмы"
