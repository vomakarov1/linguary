"""Computation of 'requires_review' and glossary invariants (PRD § 11-12)."""

from __future__ import annotations

import pytest

from linguary.domain import (
    ConceptType,
    GlossaryCandidate,
    GlossaryError,
    TermConcept,
    TermStatus,
    TermVariant,
    compute_requires_review,
    translations_conflict,
)

THRESHOLD = 0.7


@pytest.mark.parametrize(
    "concept_type",
    [
        pytest.param(ConceptType.PERSON, id="person"),
        pytest.param(ConceptType.PLACE, id="place"),
        pytest.param(ConceptType.ORGANIZATION, id="organization"),
        pytest.param(ConceptType.FICTIONAL_CONCEPT, id="fictional-concept"),
    ],
)
def test_named_entity_always_requires_review(concept_type: ConceptType) -> None:
    """A person, place, organization, or fictional concept always requires review."""
    assert compute_requires_review(
        type=concept_type,
        confidence=1.0,
        model_flagged_review=False,
        confidence_threshold=THRESHOLD,
    )


def test_type_exemption_only_removes_type_reason() -> None:
    assert not compute_requires_review(type=ConceptType.PERSON, confidence=THRESHOLD,
        model_flagged_review=False, confidence_threshold=THRESHOLD,
        exempt_type_review=True)
    assert compute_requires_review(type=ConceptType.PERSON, confidence=THRESHOLD - 0.01,
        model_flagged_review=False, confidence_threshold=THRESHOLD,
        exempt_type_review=True)
    assert compute_requires_review(type=ConceptType.PERSON, confidence=THRESHOLD,
        model_flagged_review=True, confidence_threshold=THRESHOLD,
        exempt_type_review=True)


def test_confident_term_does_not_require_review() -> None:
    """A standard term with high confidence remains an optional hint."""
    assert not compute_requires_review(
        type=ConceptType.TERM,
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=THRESHOLD,
    )


@pytest.mark.parametrize(
    ("concept_type", "flagged", "confidence", "conflict", "frequent", "expected"),
    [
        (ConceptType.PERSON, False, 0.9, False, False, False),
        (ConceptType.PERSON, False, 0.9, False, True, True),
        (ConceptType.TERM, False, 0.6, False, False, True),
        (ConceptType.TERM, False, 0.9, True, False, True),
        (ConceptType.TERM, True, 0.9, False, False, False),
    ],
)
def test_frequency_review_rule(
    concept_type: ConceptType, flagged: bool, confidence: float,
    conflict: bool, frequent: bool, expected: bool,
) -> None:
    assert compute_requires_review(
        type=concept_type, confidence=confidence,
        model_flagged_review=flagged, confidence_threshold=THRESHOLD,
        has_translation_conflict=conflict, is_frequent=frequent,
    ) is expected


@pytest.mark.parametrize(
    ("confidence", "model_flagged_review", "conflict", "expected"),
    [
        pytest.param(0.69, False, False, True, id="confidence-below-threshold"),
        pytest.param(THRESHOLD, False, False, False, id="confidence-at-threshold"),
        pytest.param(0.95, True, False, True, id="model-flagged"),
        pytest.param(0.95, False, True, True, id="merge-conflict"),
    ],
)
def test_term_review_rules(
    confidence: float, model_flagged_review: bool, conflict: bool, expected: bool
) -> None:
    """Each condition of PRD § 12 independently triggers review for a term."""
    assert (
        compute_requires_review(
            type=ConceptType.TERM,
            confidence=confidence,
            model_flagged_review=model_flagged_review,
            confidence_threshold=THRESHOLD,
            has_translation_conflict=conflict,
        )
        is expected
    )


def test_candidate_factory_computes_review_flag() -> None:
    """The final review flag is computed by the application, not the model."""
    candidate = GlossaryCandidate.create(
        id="cand-1",
        analysis_id="analysis-1",
        source_text="Winston Smith",
        type=ConceptType.PERSON,
        suggested_translation="Уинстон Смит",
        confidence=0.99,
        model_flagged_review=False,
        confidence_threshold=THRESHOLD,
    )
    assert candidate.model_flagged_review is False
    assert candidate.requires_review is True


def test_candidate_review_flag_is_immutable() -> None:
    """After saving, the review requirement flag cannot be modified."""
    candidate = GlossaryCandidate.create(
        id="cand-1",
        analysis_id="analysis-1",
        source_text="doublethink",
        type=ConceptType.TERM,
        suggested_translation="двоемыслие",
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=THRESHOLD,
    )
    with pytest.raises(AttributeError):
        candidate.requires_review = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "confidence",
    [pytest.param(-0.1, id="below-zero"), pytest.param(1.1, id="above-one")],
)
def test_candidate_rejects_confidence_out_of_range(confidence: float) -> None:
    """Model confidence must be in the 0..1 range."""
    with pytest.raises(GlossaryError):
        GlossaryCandidate(
            id="cand-1",
            analysis_id="analysis-1",
            source_text="Airstrip One",
            type=ConceptType.PLACE,
            suggested_translation="Взлётная полоса I",
            confidence=confidence,
            model_flagged_review=False,
            requires_review=True,
        )


def test_review_rules_reject_threshold_out_of_range() -> None:
    """Confidence threshold is a setting in the 0..1 range."""
    with pytest.raises(GlossaryError):
        compute_requires_review(
            type=ConceptType.TERM,
            confidence=0.5,
            model_flagged_review=False,
            confidence_threshold=1.5,
        )


@pytest.mark.parametrize(
    ("translations", "expected"),
    [
        pytest.param(("Уинстон", "Уинстон "), False, id="same-translation"),
        pytest.param(("Уинстон", "Винстон"), True, id="different-translations"),
    ],
)
def test_merge_conflict_detection(translations: tuple[str, ...], expected: bool) -> None:
    """A conflict arises only when translation variants genuinely differ."""
    assert translations_conflict(translations) is expected


def test_locked_concept_requires_mandatory_variant() -> None:
    """A locked concept must have an approved or locked translation variant."""
    with pytest.raises(GlossaryError):
        TermConcept(
            id="concept-1",
            book_id="book-1",
            type=ConceptType.PERSON,
            canonical_source_text="Winston Smith",
            status=TermStatus.LOCKED,
            variants=(
                TermVariant(
                    concept_id="concept-1",
                    language="ru",
                    text="Уинстон Смит",
                    status=TermStatus.APPROVED,
                ),
            ),
        )


def test_concept_prefers_strongest_variant() -> None:
    """For a given language, the variant with the strongest status is selected."""
    concept = TermConcept(
        id="concept-1",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Winston Smith",
        status=TermStatus.LOCKED,
        variants=(
            TermVariant(
                concept_id="concept-1",
                language="ru",
                text="Винстон Смит",
                status=TermStatus.SUGGESTED,
            ),
            TermVariant(
                concept_id="concept-1",
                language="ru",
                text="Уинстон Смит",
                status=TermStatus.LOCKED,
            ),
        ),
    )
    variant = concept.variant_for("ru")
    assert variant is not None
    assert variant.text == "Уинстон Смит"


def test_concept_rejects_variant_of_other_concept() -> None:
    """A variant belongs to its own concept."""
    with pytest.raises(GlossaryError):
        TermConcept(
            id="concept-1",
            book_id="book-1",
            type=ConceptType.TERM,
            canonical_source_text="doublethink",
            variants=(
                TermVariant(concept_id="concept-2", language="ru", text="двоемыслие"),
            ),
        )
