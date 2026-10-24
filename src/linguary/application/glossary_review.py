"""Book glossary: assembly from chapter candidates and selection of items requiring review (PRD § 11-12).

Candidates arrive from analyses of individual chapters, so a single source text appears in the book
as many times as chapters mentioned it — with differing types, confidences, and suggested translations.
This module aggregates these mentions into one review item per source text, recomputes the `requires_review`
flag using domain rules (DTO-01), and attaches any user decision already made.

Source texts are compared via `glossary_key`: obvious spelling variants (`Mr Wonka` and `Mr. Wonka`,
`WONKA'S FACTORY` and `Wonka's factory`) form a single item represented by the most frequent spelling,
with all spellings preserved on the item. Decisions are resolved using the same key. If differing
decisions were previously recorded for different spellings of the same item, they are not silently
merged: the active decision takes effect, and the item is marked with a decision conflict until the user
re-confirms a choice.

The user decision is owned by `TermConcept`: a concept exists strictly for source texts the user has
decided on. An item requiring review that lacks a concept remains unprocessed and blocks processing
(PRD § 12). An item that does not require review does not establish a concept and lives in the glossary
as an optional hint.

Filtering criteria are not re-decided here: they are owned by the domain (`compute_requires_review`).
The module is responsible for the inputs to these criteria at the book level and for display ordering.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from linguary.application.chapter_instructions import note_respecting_glossary
from linguary.application.errors import ApplicationError
from linguary.domain import (
    BookStatus,
    ChapterAnalysis,
    ConceptType,
    GlossaryCandidate,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    TermConcept,
    TermStatus,
    compute_requires_review,
    glossary_key,
    spellings_by_preference,
    translations_conflict,
)

#: Type ordering for review display (PRD § 12, § 21.3). Characters come first:
#: their translation choices affect the entire book, and obvious terms must not
#: obscure decisions the review was created for.
REVIEW_TYPE_ORDER: tuple[ConceptType, ...] = (
    ConceptType.PERSON,
    ConceptType.PLACE,
    ConceptType.ORGANIZATION,
    ConceptType.FICTIONAL_CONCEPT,
    ConceptType.TERM,
)


class GlossaryReviewError(ApplicationError):
    """Book glossary cannot be read or modified in its current state."""


class GlossaryReviewIncompleteError(GlossaryReviewError):
    """Unprocessed items requiring review remain (PRD § 12)."""


class GlossaryNotFoundError(GlossaryReviewError):
    """Book or glossary item was not found."""


@dataclass(frozen=True, slots=True)
class GlossaryReviewItem:
    """A single source text in the book glossary alongside the user's decision.

    Fields before `requires_review` describe what chapters reported; `decision`
    describes what the user chose.

    `suggested_note` is the translation directive proposed by chapters; the user decision
    carries its own directive in `decision.note`.

    `spellings` contains all spellings of the item across the book, with displayed
    `source_text` first. `superseded_decisions` contains previous decisions for other spellings
    of the same item: `decision` governs, while superseded decisions are pruned upon the user's
    next action.
    """

    concept_id: str
    source_text: str
    type: ConceptType
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    has_translation_conflict: bool
    requires_review: bool
    is_frequent: bool = True
    decision: TermConcept | None = None
    suggested_note: str = ""
    spellings: tuple[str, ...] = ()
    superseded_decisions: tuple[TermConcept, ...] = ()

    @property
    def model_doubts(self) -> bool:
        """Model asked to review an optional hint."""
        return self.model_flagged_review and not self.requires_review

    @property
    def other_spellings(self) -> tuple[str, ...]:
        """Item spellings other than the displayed primary spelling."""
        return tuple(
            spelling for spelling in self.spellings if spelling != self.source_text
        )

    @property
    def concept_type(self) -> ConceptType:
        """Concept type taking user decision into account: user may override model type."""
        return self.decision.type if self.decision is not None else self.type

    @property
    def has_decision_conflict(self) -> bool:
        """Whether prior decisions made on differing spellings of the item conflict."""
        if self.decision is None:
            return False
        chosen = _decision_signature(self.decision)
        return any(
            _decision_signature(other) != chosen for other in self.superseded_decisions
        )

    @property
    def note(self) -> str:
        """Active translation note for this item.

        User decision takes precedence over chapter suggestions: an empty note in the decision
        means no note is desired, not a fallback to the suggested note.
        """
        if self.decision is not None:
            return self.decision.note
        return self.suggested_note

    @property
    def is_decided(self) -> bool:
        """Whether the user has recorded a decision for this item."""
        return self.decision is not None

    @property
    def awaits_decision(self) -> bool:
        """Whether the item requires a review decision that has not yet been recorded."""
        return self.requires_review and self.decision is None


@dataclass(frozen=True, slots=True)
class GlossaryReview:
    """Book glossary structured for user review."""

    book_id: str
    book_status: BookStatus | None = None
    items: tuple[GlossaryReviewItem, ...] = ()
    is_editable: bool = True

    @property
    def requiring_review(self) -> tuple[GlossaryReviewItem, ...]:
        """Items on which the user is required to make a decision (PRD § 12)."""
        return tuple(item for item in self.items if item.requires_review)

    @property
    def awaiting_decision(self) -> tuple[GlossaryReviewItem, ...]:
        """Items requiring review that lack a decision."""
        return tuple(item for item in self.items if item.awaits_decision)

    @property
    def is_complete(self) -> bool:
        """Whether glossary review is complete."""
        return not self.awaiting_decision

    def item_for(self, source_text: str) -> GlossaryReviewItem | None:
        """Review item for any of its spellings, or `None` if not present in the book."""
        key = glossary_key(source_text)
        return next(
            (item for item in self.items if glossary_key(item.source_text) == key), None
        )

    def item_for_concept(self, concept_id: str) -> GlossaryReviewItem | None:
        """Review item by external concept ID."""
        return next(
            (item for item in self.items if item.concept_id == concept_id), None
        )

    def ensure_complete(self) -> None:
        """Verifies that no unprocessed items requiring review remain."""
        pending = self.awaiting_decision
        if pending:
            raise GlossaryReviewIncompleteError(
                f"проверка глоссария книги {self.book_id} не завершена: "
                f"элементов без решения — {len(pending)}, среди них "
                f"{', '.join(repr(item.source_text) for item in pending[:3])}"
            )


def build_glossary_review(
    *,
    book_id: str,
    analyses: Iterable[ChapterAnalysis],
    confidence_threshold: float,
    decisions: Iterable[TermConcept] = (),
    count_mentions: Callable[[Sequence[str]], int] | None = None,
    min_mentions: int = 5,
    min_chapters: int = 3,
) -> GlossaryReview:
    """Assembles book glossary from candidates across all analyzed chapters (PRD § 11-12).

    A decision whose source text no longer appears in any chapter is excluded: the book
    glossary reflects what was found in its text, not an archive of stale decisions.
    """
    grouped: dict[str, list[GlossaryCandidate]] = {}
    for analysis in analyses:
        for candidate in analysis.glossary_candidates:
            grouped.setdefault(glossary_key(candidate.source_text), []).append(candidate)
    decided: dict[str, list[TermConcept]] = {}
    for concept in decisions:
        decided.setdefault(glossary_key(concept.canonical_source_text), []).append(concept)
    items = (
        _review_item(
            group,
            decisions=decided.get(key, ()),
            confidence_threshold=confidence_threshold,
            count_mentions=count_mentions,
            min_mentions=min_mentions,
            min_chapters=min_chapters,
        )
        for key, group in grouped.items()
    )
    return GlossaryReview(book_id=book_id, items=tuple(sorted(items, key=_display_order)))


def glossary_snapshot(review: GlossaryReview, *, language: str) -> GlossarySnapshot:
    """Glossary snapshot used for executing a processing run (PRD § 11, § 13).

    Decisions are conveyed to the model by status: user decisions as `approved` (accepted
    or edited) and `locked` — mandatory translation directives; skipped items (`suggested`) —
    optional hints. Items without a decision enter the snapshot as hints with the model's
    suggested translation: they did not require review.
    """
    return GlossarySnapshot(
        entries=tuple(
            entry
            for item in review.items
            if (entry := _snapshot_entry(item, language=language)) is not None
        )
    )


def candidates_with_decisions(
    candidates: Iterable[GlossaryCandidate],
    decisions: Iterable[TermConcept],
    *,
    language: str,
    decided_only: Iterable[GlossaryCandidate] = (),
) -> tuple[GlossaryCandidate, ...]:
    """Candidates from prior chapters where user decisions override model suggestions.

    Subsequent chapter analysis repeats accumulated book decisions verbatim, so user
    choices are required: otherwise the chapter receives and repeats model suggestions
    the user already corrected (PRD § 11). Decisions match candidates by `glossary_key`
    and override translation, type, and note; skipped items are not user decisions and retain
    model suggestions. Decided candidates come first so context limits do not prune them.

    `decided_only` denotes candidates included only if accompanied by a user decision:
    during re-analysis, these are candidates from the current and subsequent chapters. User
    decisions govern the whole book, whereas subsequent chapter model suggestions offer
    nothing to the current chapter.

    Each candidate's note is sanitized of phrases mentioning model suggestions rejected
    by the user on other concepts: 'post at the entrance to Duskmere' when 'Duskmere' was
    chosen over 'Duskmeer'. Rejected suggestions are detected across all provided book
    candidates — identical to the processing context rule (`note_respecting_glossary`).
    """
    decided: dict[str, list[TermConcept]] = {}
    for concept in decisions:
        decided.setdefault(glossary_key(concept.canonical_source_text), []).append(concept)
    with_decision: list[GlossaryCandidate] = []
    without_decision: list[GlossaryCandidate] = []
    user_decisions: dict[str, GlossarySnapshotEntry] = {}
    marked = [
        *((candidate, False) for candidate in candidates),
        *((candidate, True) for candidate in decided_only),
    ]
    for candidate, needs_decision in marked:
        decision, _ = _chosen_decision(
            decided.get(glossary_key(candidate.source_text), ()),
            shown=candidate.source_text.strip(),
        )
        variant = decision.variant_for(language) if decision is not None else None
        if decision is None or variant is None or not decision.status.is_user_decision:
            if not needs_decision:
                without_decision.append(candidate)
            continue
        decided_candidate = replace(
            candidate,
            type=decision.type,
            suggested_translation=variant.text,
            note=decision.note,
        )
        with_decision.append(decided_candidate)
        user_decisions.setdefault(
            glossary_key(candidate.source_text),
            GlossarySnapshotEntry(
                source_text=decided_candidate.source_text,
                translation=variant.text,
                type=decision.type,
                status=decision.status,
                note=decision.note,
            ),
        )
    book_candidates = [candidate for candidate, _ in marked]
    return tuple(
        replace(
            candidate,
            note=note_respecting_glossary(
                candidate,
                decisions=tuple(user_decisions.values()),
                chapter_candidates=book_candidates,
            ),
        )
        for candidate in (*with_decision, *without_decision)
    )


def _review_item(
    group: Sequence[GlossaryCandidate],
    *,
    decisions: Sequence[TermConcept],
    confidence_threshold: float,
    count_mentions: Callable[[Sequence[str]], int] | None,
    min_mentions: int,
    min_chapters: int,
) -> GlossaryReviewItem:
    """Review item constructed from mentions of a source text across chapters.

    The suggested translation is taken from the highest-confidence chapter, and review
    signals are aggregated: model flag from any chapter, type, and translation conflicts
    are evaluated alongside mention frequency. Stored candidates from multiple chapters
    are unified here.
    """
    leading = max(group, key=lambda candidate: candidate.confidence)
    spellings = spellings_by_preference(
        (candidate.source_text, candidate.confidence) for candidate in group
    )
    is_frequent = count_mentions is None or (
        count_mentions(spellings) >= min_mentions
        or len({candidate.analysis_id for candidate in group}) >= min_chapters
    )
    decision, superseded = _chosen_decision(decisions, shown=spellings[0])
    concept_type = next(
        (
            candidate.type
            for candidate in group
            if candidate.type.always_requires_review
        ),
        leading.type,
    )
    model_flagged = any(candidate.model_flagged_review for candidate in group)
    conflict = translations_conflict(
        candidate.suggested_translation for candidate in group
    )
    suggested_note = next(
        (
            candidate.note
            for candidate in sorted(
                group, key=lambda candidate: candidate.confidence, reverse=True
            )
            if candidate.note.strip()
        ),
        "",
    )
    return GlossaryReviewItem(
        concept_id=decision.id if decision is not None else leading.id,
        source_text=spellings[0],
        type=concept_type,
        suggested_translation=leading.suggested_translation,
        confidence=leading.confidence,
        model_flagged_review=model_flagged,
        has_translation_conflict=conflict,
        is_frequent=is_frequent,
        requires_review=compute_requires_review(
            type=concept_type,
            confidence=leading.confidence,
            model_flagged_review=model_flagged,
            confidence_threshold=confidence_threshold,
            has_translation_conflict=conflict,
            is_frequent=is_frequent,
        ),
        decision=decision,
        suggested_note=suggested_note,
        spellings=spellings,
        superseded_decisions=superseded,
    )


#: Priority order for resolving active decisions among multiple spellings of an item:
#: locked > approved > suggested.
_DECISION_PRIORITY: dict[TermStatus, int] = {
    TermStatus.LOCKED: 0,
    TermStatus.APPROVED: 1,
    TermStatus.SUGGESTED: 2,
}


def _chosen_decision(
    decisions: Sequence[TermConcept], *, shown: str
) -> tuple[TermConcept | None, tuple[TermConcept, ...]]:
    """Active item decision and superseded decisions from other spellings.

    Each item has a single decision; multiple decisions per key linger from older
    glossaries before spellings were unified. Decisions do not record timestamps,
    so 'latest' cannot be determined: strongest status wins, then matching the displayed
    spelling, then first stored. Divergence from other decisions is exposed
    via `has_decision_conflict`.
    """
    if not decisions:
        return None, ()
    ordered = sorted(
        enumerate(decisions),
        key=lambda pair: (
            _DECISION_PRIORITY[pair[1].status],
            pair[1].canonical_source_text.strip() != shown,
            pair[0],
        ),
    )
    chosen = ordered[0][1]
    return chosen, tuple(concept for _, concept in ordered[1:])


def _decision_signature(concept: TermConcept) -> tuple[object, ...]:
    """Attributes distinguishing decisions for the user: status, type, note, translation."""
    return (
        concept.status,
        concept.type,
        concept.note.strip(),
        tuple(
            sorted(
                (variant.language, variant.text.strip(), variant.status)
                for variant in concept.variants
            )
        ),
    )


def _display_order(item: GlossaryReviewItem) -> tuple[bool, bool, bool, int, str]:
    """Review display order (PRD § 12).

    Items awaiting decision come first, then remaining items requiring review, then
    optional hints; grouped internally by concept type and source text.
    """
    return (
        not item.awaits_decision,
        not item.requires_review,
        not item.model_doubts,
        REVIEW_TYPE_ORDER.index(item.concept_type),
        item.source_text.casefold(),
    )


def _snapshot_entry(
    item: GlossaryReviewItem, *, language: str
) -> GlossarySnapshotEntry | None:
    """Snapshot entry for a single item, or `None` if there is nothing to enforce on the model."""
    if item.decision is None:
        return GlossarySnapshotEntry(
            source_text=item.source_text,
            translation=item.suggested_translation,
            type=item.type,
            status=TermStatus.SUGGESTED,
            note=item.note,
        )
    variant = item.decision.variant_for(language)
    if variant is None:
        return None
    return GlossarySnapshotEntry(
        source_text=item.decision.canonical_source_text,
        translation=variant.text,
        type=item.decision.type,
        status=item.decision.status,
        note=item.note,
    )


__all__ = [
    "REVIEW_TYPE_ORDER",
    "GlossaryReview",
    "GlossaryReviewError",
    "GlossaryReviewIncompleteError",
    "GlossaryNotFoundError",
    "GlossaryReviewItem",
    "build_glossary_review",
    "candidates_with_decisions",
    "glossary_snapshot",
]
