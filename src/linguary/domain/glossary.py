"""Book glossary: concepts, language variants, and candidates from chapter analyses.

The glossary is the sole source of mandatory translation decisions (PRD § 11).
A candidate originates from chapter analysis with the model's own `model_flagged_review` flag,
while the final `requires_review` flag is computed by the application per PRD § 12 rules.
"""

from __future__ import annotations

import re
import unicodedata
from math import isfinite
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from linguary.domain.errors import GlossaryError
from linguary.domain.name_translation import NameTranslationEvidence


class ConceptType(StrEnum):
    """Тип понятия глоссария (PRD § 12)."""

    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    FICTIONAL_CONCEPT = "fictional_concept"
    TERM = "term"

    @property
    def always_requires_review(self) -> bool:
        """Types requiring review when frequently mentioned."""
        return self is not ConceptType.TERM


class TermStatus(StrEnum):
    """Status of a translation decision for a concept and its variant (PRD § 11)."""

    SUGGESTED = "suggested"
    APPROVED = "approved"
    LOCKED = "locked"

    @property
    def is_mandatory(self) -> bool:
        """`locked` — decision is locked: the concept must have a locked variant."""
        return self is TermStatus.LOCKED

    @property
    def is_user_decision(self) -> bool:
        """User decision that the model must respect (PRD § 11-12).

        Approved, corrected, and locked variants are chosen by the user, and the model
        is not permitted to replace them with its own—neither during processing nor during subsequent
        chapter analysis. `suggested` is a model variant: an item that required no review or was skipped
        by the user as non-essential for consistency. It remains a suggestion.
        """
        return self is not TermStatus.SUGGESTED


@dataclass(frozen=True, slots=True)
class TermVariant:
    """Language variant of a single concept."""

    concept_id: str
    language: str
    text: str
    status: TermStatus = TermStatus.SUGGESTED

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise GlossaryError("вариант термина не привязан к понятию")
        if not self.language.strip():
            raise GlossaryError("язык варианта термина не задан")
        if not self.text.strip():
            raise GlossaryError("текст варианта термина пуст")


@dataclass(frozen=True, slots=True)
class TermConcept:
    """Glossary concept grouping variants of the same meaning.

    `note` is translation guidance for this concept passed to the model alongside
    the translation variant: a single translation term is insufficient where a book-level
    decision affects grammatical agreement in text (PRD § 11).
    """

    id: str
    book_id: str
    type: ConceptType
    canonical_source_text: str
    status: TermStatus = TermStatus.SUGGESTED
    variants: tuple[TermVariant, ...] = ()
    note: str = ""


    def __post_init__(self) -> None:
        if not self.id.strip():
            raise GlossaryError("идентификатор понятия глоссария пуст")
        if not self.book_id.strip():
            raise GlossaryError("понятие глоссария не привязано к книге")
        if not self.canonical_source_text.strip():
            raise GlossaryError("исходный текст понятия глоссария пуст")
        for variant in self.variants:
            if variant.concept_id != self.id:
                raise GlossaryError(
                    f"вариант {variant.text!r} принадлежит другому понятию: {variant.concept_id}"
                )
        if self.status.is_mandatory and not any(
            variant.status.is_mandatory for variant in self.variants
        ):
            raise GlossaryError(
                f"понятие {self.id} зафиксировано, но обязательного варианта перевода нет"
            )

    def variant_for(self, language: str) -> TermVariant | None:
        """Translation variant for the specified language with the highest priority status."""
        candidates = [variant for variant in self.variants if variant.language == language]
        if not candidates:
            return None
        priority = {TermStatus.LOCKED: 0, TermStatus.APPROVED: 1, TermStatus.SUGGESTED: 2}
        return min(candidates, key=lambda variant: priority[variant.status])


@dataclass(frozen=True, slots=True)
class GlossaryCandidate:
    """Glossary candidate from chapter analysis.

    `model_flagged_review` is the model's own flag, serving as an input signal. The final
    `requires_review` is computed by the application and cannot be edited by the model once persisted.

    `note` is translation guidance for this decision that cannot be expressed by a single
    translation variant: character grammatical gender, title declension, speech register.
    Without it, a book-level decision splinters into separate chapter decisions (PRD § 11):
    the name appears consistently from the glossary, but words agreeing with it diverge across chapters.
    """

    id: str
    analysis_id: str
    source_text: str
    type: ConceptType
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    requires_review: bool
    note: str = ""
    name_translation_evidence: NameTranslationEvidence | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise GlossaryError("идентификатор кандидата глоссария пуст")
        if not self.analysis_id.strip():
            raise GlossaryError("кандидат глоссария не привязан к анализу главы")
        if not self.source_text.strip():
            raise GlossaryError("исходный текст кандидата глоссария пуст")
        if not self.suggested_translation.strip():
            raise GlossaryError(f"предложенный перевод кандидата {self.source_text!r} пуст")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise GlossaryError(
                f"уверенность модели вне диапазона 0..1: {self.confidence}"
            )

    @classmethod
    def create(
        cls,
        *,
        id: str,
        analysis_id: str,
        source_text: str,
        type: ConceptType,
        suggested_translation: str,
        confidence: float,
        model_flagged_review: bool,
        confidence_threshold: float,
        has_translation_conflict: bool = False,
        note: str = "",
    ) -> GlossaryCandidate:
        """Creates a candidate with computed `requires_review` flag."""
        return cls(
            id=id,
            analysis_id=analysis_id,
            source_text=source_text,
            type=type,
            suggested_translation=suggested_translation,
            confidence=confidence,
            model_flagged_review=model_flagged_review,
            note=note,
            requires_review=compute_requires_review(
                type=type,
                confidence=confidence,
                model_flagged_review=model_flagged_review,
                confidence_threshold=confidence_threshold,
                has_translation_conflict=has_translation_conflict,
            ),
        )


def compute_requires_review(
    *,
    type: ConceptType,
    confidence: float,
    model_flagged_review: bool,
    confidence_threshold: float,
    has_translation_conflict: bool = False,
    is_frequent: bool = True,
) -> bool:
    """Whether the candidate requires user attention (PRD § 12).

    Frequent concept type or model flag requires review; low confidence and translation
    conflicts require review regardless of frequency. By default, a concept is considered
    frequent so that single-chapter behavior remains unchanged.
    """
    if not isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise GlossaryError(
            f"порог уверенности вне диапазона 0..1: {confidence_threshold}"
        )
    return (
        (is_frequent and (type.always_requires_review or model_flagged_review))
        or confidence < confidence_threshold
        or has_translation_conflict
    )


#: Typographic apostrophes and quotation marks straightened in keys.
_STRAIGHT_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u02bc": "'",
        "\u2032": "'",
        "`": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2033": '"',
    }
)

#: Dot following abbreviated titles: `Mr. Wonka` and `Mr Wonka` represent the same name.
_TITLE_DOT = re.compile(r"\b(mr|mrs|ms|dr|st)\.\s*")

_WHITESPACE = re.compile(r"\s+")


def glossary_key(text: str) -> str:
    """Key used to group spellings of the same concept into a single element (PRD § 11).

    Only unambiguous variants are merged: case, Unicode compatibility forms, typographic
    apostrophes and quotes, extra whitespace, and dots after `Mr`, `Mrs`, `Ms`, `Dr`, `St`.
    Possessive forms and name fragments do not merge into the same key: `Wonka's factory`
    and `Wonka factory`, `Charlie` and `Charlie Bucket` remain distinct elements because
    ambiguous cases are never merged automatically.
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_STRAIGHT_QUOTES)
    normalized = _WHITESPACE.sub(" ", normalized).strip().casefold()
    return _WHITESPACE.sub(" ", _TITLE_DOT.sub(r"\1 ", normalized)).strip()


def spellings_by_preference(mentions: Iterable[tuple[str, float]]) -> tuple[str, ...]:
    """Distinct spellings of a concept, ordered starting with the primary display form.

    `mentions` are (spelling, model confidence) pairs for each occurrence. The most frequent
    spelling comes first, breaking ties by highest confidence, then by earliest occurrence:
    this ensures the review screen displays the form most prevalent in the book.
    """
    counts: dict[str, int] = {}
    confidence: dict[str, float] = {}
    for text, mention_confidence in mentions:
        spelling = text.strip()
        counts[spelling] = counts.get(spelling, 0) + 1
        confidence[spelling] = max(confidence.get(spelling, 0.0), mention_confidence)
    order = {spelling: index for index, spelling in enumerate(counts)}
    return tuple(
        sorted(
            counts,
            key=lambda spelling: (-counts[spelling], -confidence[spelling], order[spelling]),
        )
    )


def translations_conflict(translations: Iterable[str]) -> bool:
    """Whether analysis chunks produce conflicting translation variants for the same concept.

    Variants are compared using the same normalization key as source spellings: translation
    case and typography alone do not constitute distinct decisions.
    """
    return len({glossary_key(translation) for translation in translations}) > 1
