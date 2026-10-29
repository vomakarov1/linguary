"""User glossary decisions and processing start (P3.T5, PRD § 11-12, § 21.3)."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace

import pytest
from analysis_fakes import (
    BOOK_ID,
    CONFIDENCE_THRESHOLD,
    FakeUnitOfWork,
    make_analysis_candidate,
    make_chapter_analysis,
    make_analysis_document,
    make_analysis_section,
    make_glossary_book,
    sequential_ids,
)

from linguary.application.glossary_review import (
    GlossaryNotFoundError,
    GlossaryReviewError,
    GlossaryReviewIncompleteError,
)
from linguary.application.use_cases.approve_glossary import (
    ApproveGlossary,
    GlossaryDecision,
    GlossaryDecisionCommand,
    GlossaryNoteCommand,
    StartProcessing,
    StartProcessingCommand,
)
from linguary.adapters.config.settings import CheckSettings, SettingsError
from linguary.domain import (
    BookStatus,
    BookStatusTransitionError,
    ConceptType,
    GlossaryCandidate,
    LLMSettingsSnapshot,
    TermConcept,
    TermStatus,
    TermVariant,
)
from linguary.domain.name_translation import (
    NameTranslationAssessment, NameTranslationEvidence, NameTranslationObservation,
    NameTranslationStrategy,
)

LLM_SETTINGS = LLMSettingsSnapshot(provider="openai_compatible", model="gpt-test")

#: High-confidence translated term: hint, does not require a decision.
HINT = "entailed estate"


def hint_candidate() -> GlossaryCandidate:
    """Candidate that does not require review and does not prevent starting processing."""
    return make_analysis_candidate(
        source_text=HINT,
        type=ConceptType.TERM,
        suggested_translation="заповедное имение",
        confidence=0.95,
    )


def glossary_book(
    *candidates: GlossaryCandidate,
    status: BookStatus = BookStatus.GLOSSARY_REVIEW,
) -> FakeUnitOfWork:
    """Book with a single analyzed chapter and its glossary candidates."""
    unit_of_work = make_glossary_book(
        (make_chapter_analysis(candidates=candidates),), status=status
    )
    unit_of_work.documents.save(make_analysis_document((
        make_analysis_section(texts=(" ".join(
            candidate.source_text for candidate in candidates for _ in range(5)
        ),)),
    )))
    return unit_of_work


def approve(unit_of_work: FakeUnitOfWork) -> ApproveGlossary:
    """Glossary decisions use case on top of storage double."""
    return ApproveGlossary(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # storage port double
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        id_factory=sequential_ids(),
    )


def start(unit_of_work: FakeUnitOfWork) -> StartProcessing:
    """Start processing use case on top of storage double."""
    return StartProcessing(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # storage port double
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        llm_settings_snapshot=LLM_SETTINGS,
        id_factory=sequential_ids(),
    )


def command(
    decision: GlossaryDecision,
    *,
    source_text: str = "Mr. Bennet",
    translation: str | None = None,
) -> GlossaryDecisionCommand:
    """User decision for a single book glossary item."""
    return GlossaryDecisionCommand(
        book_id=BOOK_ID,
        source_text=source_text,
        decision=decision,
        translation=translation,
    )


def started_book() -> FakeUnitOfWork:
    """Book whose processing is already started with a frozen glossary."""
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())
    approve(unit_of_work)(command(GlossaryDecision.ACCEPT))
    start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))
    return unit_of_work


def test_start_allowed_when_only_rare_names_undecided() -> None:
    unit_of_work = glossary_book(make_analysis_candidate())
    unit_of_work.documents.save(make_analysis_document((
        make_analysis_section(texts=("Mr. Bennet appeared once.",)),
    )))
    assert start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))


def test_display_and_start_share_policy() -> None:
    candidate = make_analysis_candidate(source_text="Alice", suggested_translation="Элис")
    candidate = replace(candidate, name_translation_evidence=NameTranslationEvidence(
        1, "conditions-1", 1, NameTranslationStrategy.SOUND,
        (NameTranslationObservation("Элис", 0.9, False,
            NameTranslationAssessment("sound", True, (), "Однозначное имя.")),)))
    unit_of_work = glossary_book(candidate)
    conditions = SimpleNamespace(id="conditions-1", payload={
        "name_translation_policy": {"version": 1, "strategy": "sound"},
        "checks": {"glossary_confidence_threshold": CONFIDENCE_THRESHOLD,
                   "glossary_review_min_mentions": 5, "glossary_review_min_chapters": 3},
    })
    unit_of_work.work = SimpleNamespace(latest_conditions=lambda _: conditions)

    assert not approve(unit_of_work).review(BOOK_ID).items[0].requires_review
    assert start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))


def test_thresholds_from_settings_applied() -> None:
    settings = CheckSettings(glossary_review_min_mentions=1)
    unit_of_work = glossary_book(make_analysis_candidate())
    unit_of_work.documents.save(make_analysis_document((
        make_analysis_section(texts=("Mr. Bennet appeared once.",)),
    )))
    service = ApproveGlossary(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        glossary_review_min_mentions=settings.glossary_review_min_mentions,
        glossary_review_min_chapters=settings.glossary_review_min_chapters,
    )
    assert service.review(BOOK_ID).items[0].requires_review
    with pytest.raises(SettingsError):
        CheckSettings(glossary_review_min_chapters=0)


def test_accept_all_touches_only_items_requiring_review() -> None:
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())
    concepts = approve(unit_of_work).accept_all_awaiting(BOOK_ID)
    assert [concept.canonical_source_text for concept in concepts] == ["Mr. Bennet"]


def test_accept_stores_approved_decision() -> None:
    """Acceptance saves the model variant as an approved decision."""
    unit_of_work = glossary_book(make_analysis_candidate())

    concept = approve(unit_of_work)(command(GlossaryDecision.ACCEPT))

    variant = concept.variant_for("ru")
    assert concept.status is TermStatus.APPROVED
    assert variant is not None
    assert variant.text == "мистер Беннет"
    assert unit_of_work.glossary.list_for_book(BOOK_ID) == (concept,)
    assert approve(unit_of_work).review(BOOK_ID).is_complete


def test_accept_all_awaiting_decides_every_pending_item() -> None:
    """Bulk acceptance resolves all pending items in a single call."""
    unit_of_work = glossary_book(
        make_analysis_candidate(),
        make_analysis_candidate(
            source_text="Longbourn",
            type=ConceptType.PLACE,
            suggested_translation="Лонгборн",
        ),
        hint_candidate(),
    )

    concepts = approve(unit_of_work).accept_all_awaiting(BOOK_ID)

    translations = {
        concept.canonical_source_text: variant.text
        for concept in concepts
        if (variant := concept.variant_for("ru")) is not None
    }
    review = approve(unit_of_work).review(BOOK_ID)
    hint = review.item_for(HINT)
    book = unit_of_work.books.get(BOOK_ID)
    assert translations == {"Mr. Bennet": "мистер Беннет", "Longbourn": "Лонгборн"}
    assert all(concept.status is TermStatus.APPROVED for concept in concepts)
    assert review.is_complete
    assert hint is not None
    assert not hint.is_decided
    assert book is not None
    assert book.status is BookStatus.READY_TO_PROCESS


def test_accept_all_awaiting_is_rejected_after_processing_started() -> None:
    """Bulk acceptance follows the same restriction as item-level decisions."""
    unit_of_work = glossary_book(
        make_analysis_candidate(), status=BookStatus.PROCESSING
    )

    with pytest.raises(GlossaryReviewError, match="только"):
        approve(unit_of_work).accept_all_awaiting(BOOK_ID)

    assert unit_of_work.glossary.list_for_book(BOOK_ID) == ()


def test_correct_stores_user_translation() -> None:
    """Correction stores the user's translation instead of the one proposed by the model."""
    unit_of_work = glossary_book(make_analysis_candidate())

    concept = approve(unit_of_work)(
        command(GlossaryDecision.CORRECT, translation="мистер Беннетт")
    )

    variant = concept.variant_for("ru")
    assert concept.status is TermStatus.APPROVED
    assert variant is not None
    assert variant.text == "мистер Беннетт"


@pytest.mark.parametrize(
    "translation",
    [pytest.param(None, id="no-translation"), pytest.param("   ", id="blank")],
)
def test_correction_without_translation_is_rejected(translation: str | None) -> None:
    """A correction without a translation variant is not considered a valid decision."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError):
        approve(unit_of_work)(
            command(GlossaryDecision.CORRECT, translation=translation)
        )

    assert unit_of_work.glossary.list_for_book(BOOK_ID) == ()


@pytest.mark.parametrize(
    "decision",
    [
        pytest.param(GlossaryDecision.ACCEPT, id="accept"),
        pytest.param(GlossaryDecision.LOCK, id="lock"),
        pytest.param(GlossaryDecision.SKIP, id="skip"),
    ],
)
def test_decision_other_than_correction_rejects_translation(
    decision: GlossaryDecision,
) -> None:
    """Only correction accepts a translation variant; other actions change status."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError):
        approve(unit_of_work)(command(decision, translation="мистер Беннетт"))

    assert unit_of_work.glossary.list_for_book(BOOK_ID) == ()


def test_lock_keeps_corrected_translation() -> None:
    """Locking preserves the corrected variant and overwrites the previous decision."""
    unit_of_work = glossary_book(make_analysis_candidate())
    scenario = approve(unit_of_work)
    corrected = scenario(command(GlossaryDecision.CORRECT, translation="мистер Беннетт"))

    locked = scenario(command(GlossaryDecision.LOCK))

    variant = locked.variant_for("ru")
    assert locked.id == corrected.id
    assert locked.status is TermStatus.LOCKED
    assert variant is not None
    assert variant.text == "мистер Беннетт"
    assert unit_of_work.glossary.list_for_book(BOOK_ID) == (locked,)


def test_skip_leaves_item_as_optional_hint() -> None:
    """An item skipped as non-essential remains an optional hint."""
    unit_of_work = glossary_book(make_analysis_candidate())

    concept = approve(unit_of_work)(command(GlossaryDecision.SKIP))

    review = approve(unit_of_work).review(BOOK_ID)
    assert concept.status is TermStatus.SUGGESTED
    assert review.items[0].is_decided
    assert review.is_complete


def test_decision_on_unknown_item_is_rejected() -> None:
    """A decision can only be made on a glossary item of this book."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError, match="Winston Smith"):
        approve(unit_of_work)(
            command(GlossaryDecision.ACCEPT, source_text="Winston Smith")
        )


def test_review_of_unknown_book_is_rejected() -> None:
    """Glossary can only be read for an existing book."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError):
        approve(unit_of_work).review("book-404")


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(BookStatus.PROCESSING, id="processing"),
        pytest.param(BookStatus.READY_TO_EXPORT, id="ready-to-export"),
        pytest.param(BookStatus.COMPLETED, id="completed"),
    ],
)
def test_glossary_is_read_only_after_processing_started(status: BookStatus) -> None:
    """After processing starts, glossary decisions cannot be modified (PRD § 11, § 21.3)."""
    unit_of_work = glossary_book(make_analysis_candidate(), status=status)

    with pytest.raises(GlossaryReviewError, match="только"):
        approve(unit_of_work)(command(GlossaryDecision.ACCEPT))

    assert unit_of_work.glossary.list_for_book(BOOK_ID) == ()


def test_decisions_are_frozen_after_start() -> None:
    """Running processing locks the glossary against edits, and the snapshot remains unchanged."""
    unit_of_work = started_book()

    with pytest.raises(GlossaryReviewError):
        approve(unit_of_work)(
            command(GlossaryDecision.CORRECT, translation="мистер Беннетт")
        )

    run = unit_of_work.runs.active_for_book(BOOK_ID)
    assert run is not None
    entries = {entry.source_text: entry for entry in run.glossary_snapshot.entries}
    assert entries["Mr. Bennet"].translation == "мистер Беннет"


def test_processing_does_not_start_while_decisions_are_pending() -> None:
    """Processing does not start while there are unresolved items requiring review."""
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())

    with pytest.raises(GlossaryReviewIncompleteError, match="Mr. Bennet"):
        start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    book = unit_of_work.books.get(BOOK_ID)
    assert book is not None
    assert book.status is BookStatus.GLOSSARY_REVIEW
    assert unit_of_work.runs.active_for_book(BOOK_ID) is None


def test_start_freezes_glossary_snapshot() -> None:
    """Starting processing freezes decisions into a run snapshot (PRD § 11)."""
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())
    approve(unit_of_work)(
        command(GlossaryDecision.CORRECT, translation="мистер Беннетт")
    )
    approve(unit_of_work)(command(GlossaryDecision.LOCK))

    run = start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    entries = {entry.source_text: entry for entry in run.glossary_snapshot.entries}
    assert run.is_active
    assert unit_of_work.runs.active_for_book(BOOK_ID) == run
    assert run.llm_settings_snapshot == LLM_SETTINGS
    assert entries["Mr. Bennet"].translation == "мистер Беннетт"
    assert entries[HINT].status is TermStatus.SUGGESTED
    assert tuple(
        entry.source_text for entry in run.glossary_snapshot.mandatory_entries
    ) == ("Mr. Bennet",)


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(BookStatus.GLOSSARY_REVIEW, id="glossary-review"),
        pytest.param(BookStatus.READY_TO_PROCESS, id="ready-to-process"),
    ],
)
def test_start_moves_book_to_processing(status: BookStatus) -> None:
    """Starting processing transitions the book to `PROCESSING` (ALG-06)."""
    unit_of_work = glossary_book(hint_candidate(), status=status)

    start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    book = unit_of_work.books.get(BOOK_ID)
    assert book is not None
    assert book.status is BookStatus.PROCESSING


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(BookStatus.IMPORTED, id="imported"),
        pytest.param(BookStatus.ANALYZING, id="analyzing"),
    ],
)
def test_processing_does_not_start_outside_review(status: BookStatus) -> None:
    """Processing does not start from a state prior to glossary review."""
    unit_of_work = glossary_book(hint_candidate(), status=status)

    with pytest.raises(BookStatusTransitionError):
        start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    assert unit_of_work.runs.active_for_book(BOOK_ID) is None


def test_second_active_run_is_rejected() -> None:
    """A book cannot have two active processing runs (ADR § 3)."""
    unit_of_work = started_book()

    with pytest.raises(GlossaryReviewError, match="уже идёт обработка"):
        start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    assert len(unit_of_work.runs.runs) == 1


def test_suggested_note_of_the_chapter_reaches_the_review() -> None:
    """A note suggested by chapter analysis is visible in glossary review."""
    unit_of_work = glossary_book(make_analysis_candidate(note="женский род"))

    item = approve(unit_of_work).review(BOOK_ID).item_for("Mr. Bennet")

    assert item is not None
    assert item.note == "женский род"


def test_decision_keeps_the_note_of_the_chapter() -> None:
    """A decision without its own note preserves what the chapter suggested."""
    unit_of_work = glossary_book(make_analysis_candidate(note="женский род"))

    concept = approve(unit_of_work)(command(GlossaryDecision.ACCEPT))

    assert concept.note == "женский род"


def test_note_of_the_user_replaces_the_note_of_the_chapter() -> None:
    """The user's note takes precedence over the chapter's suggestion, including an empty note."""
    unit_of_work = glossary_book(make_analysis_candidate(note="женский род"))

    corrected = approve(unit_of_work).set_note(
        GlossaryNoteCommand(
            book_id=BOOK_ID, concept_id="candidate-analysis-1-Mr. Bennet", note="мужской род"
        )
    )
    cleared = approve(unit_of_work).set_note(
        GlossaryNoteCommand(book_id=BOOK_ID, concept_id=corrected.id, note="  ")
    )

    assert corrected.note == "мужской род"
    assert cleared.note == ""


def test_note_edit_keeps_the_corrected_translation() -> None:
    """Editing a note does not restore the model variant over the user's correction.

    For this reason the note is edited as a separate action: character gender becomes known
    later than name translation is accepted, and the corrected variant must not be lost.
    """
    unit_of_work = glossary_book(make_analysis_candidate())
    decided = approve(unit_of_work)(
        command(GlossaryDecision.CORRECT, translation="мистер Беннетт")
    )

    concept = approve(unit_of_work).set_note(
        GlossaryNoteCommand(book_id=BOOK_ID, concept_id=decided.id, note="мужской род")
    )

    variant = concept.variant_for("ru")
    assert variant is not None
    assert variant.text == "мистер Беннетт"
    assert concept.status is TermStatus.APPROVED
    assert concept.note == "мужской род"


def test_note_edit_of_unknown_concept_is_rejected() -> None:
    """A note for a concept not present in the book is rejected."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryNotFoundError):
        approve(unit_of_work).set_note(
            GlossaryNoteCommand(book_id=BOOK_ID, concept_id="missing", note="женский род")
        )


def test_note_is_frozen_in_the_snapshot_of_the_run() -> None:
    """The launch snapshot carries the note together with the translation variant (PRD § 11)."""
    unit_of_work = glossary_book(make_analysis_candidate(note="женский род"), hint_candidate())
    approve(unit_of_work)(command(GlossaryDecision.LOCK))

    run = start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    entries = {entry.source_text: entry for entry in run.glossary_snapshot.entries}
    assert entries["Mr. Bennet"].note == "женский род"


def test_decision_can_be_changed_after_implicit_acceptance_by_note() -> None:
    """A decision accepted implicitly along with a note can be corrected before processing starts (QA-07)."""
    unit_of_work = glossary_book(make_analysis_candidate())
    implicit = approve(unit_of_work).set_note(
        GlossaryNoteCommand(
            book_id=BOOK_ID,
            concept_id="candidate-analysis-1-Mr. Bennet",
            note="мужской род",
        )
    )

    corrected = approve(unit_of_work)(
        command(GlossaryDecision.CORRECT, translation="мистер Беннетт")
    )
    returned = approve(unit_of_work)(command(GlossaryDecision.ACCEPT))

    assert corrected.id == implicit.id
    corrected_variant = corrected.variant_for("ru")
    assert corrected_variant is not None
    assert corrected_variant.text == "мистер Беннетт"
    assert corrected.note == "мужской род"
    returned_variant = returned.variant_for("ru")
    assert returned_variant is not None
    assert returned_variant.text == "мистер Беннет"


def test_correction_changes_the_concept_type() -> None:
    """Correction changes the concept type chosen mistakenly by the model (QA-11, T2).

    The type is retained in subsequent decisions and included in the launch snapshot.
    """
    unit_of_work = glossary_book(
        make_analysis_candidate(
            source_text="Charlotte Russe", suggested_translation="Шарлотта Русс"
        )
    )
    corrected = approve(unit_of_work)(
        GlossaryDecisionCommand(
            book_id=BOOK_ID,
            source_text="Charlotte Russe",
            decision=GlossaryDecision.CORRECT,
            translation="шарлотка по-русски",
            type=ConceptType.TERM,
        )
    )
    locked = approve(unit_of_work)(
        command(GlossaryDecision.LOCK, source_text="Charlotte Russe")
    )

    run = start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    assert corrected.type is ConceptType.TERM
    assert locked.type is ConceptType.TERM
    (entry,) = run.glossary_snapshot.entries
    assert entry.type is ConceptType.TERM
    assert entry.translation == "шарлотка по-русски"


def test_type_is_accepted_only_with_correction() -> None:
    """Concept type is only accepted with a correction, just like a translation variant."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError):
        approve(unit_of_work)(
            GlossaryDecisionCommand(
                book_id=BOOK_ID,
                source_text="Mr. Bennet",
                decision=GlossaryDecision.ACCEPT,
                type=ConceptType.TERM,
            )
        )


def test_skip_does_not_pass_the_unconfirmed_note_of_the_model() -> None:
    """Skipping does not propagate a note that was not confirmed by the user.

    Otherwise "feminine gender", mistakenly suggested for a dessert, would reach
    the processing model with the skipped item (QA-11).
    """
    unit_of_work = glossary_book(
        make_analysis_candidate(
            source_text="Charlotte Russe",
            suggested_translation="Шарлотта Русс",
            note="женский род",
        )
    )

    skipped = approve(unit_of_work)(
        command(GlossaryDecision.SKIP, source_text="Charlotte Russe")
    )
    run = start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    assert skipped.note == ""
    (entry,) = run.glossary_snapshot.entries
    assert entry.note == ""


def test_skip_keeps_the_note_saved_by_the_user() -> None:
    """A note saved directly by the user is preserved upon skipping."""
    unit_of_work = glossary_book(make_analysis_candidate(note="женский род"))
    approve(unit_of_work).set_note(
        GlossaryNoteCommand(
            book_id=BOOK_ID,
            concept_id="candidate-analysis-1-Mr. Bennet",
            note="мужской род",
        )
    )

    skipped = approve(unit_of_work)(command(GlossaryDecision.SKIP))
    typed = approve(unit_of_work)(
        GlossaryDecisionCommand(
            book_id=BOOK_ID,
            source_text="Mr. Bennet",
            decision=GlossaryDecision.SKIP,
            note="обращаться на «вы»",
        )
    )

    assert skipped.note == "мужской род"
    assert typed.note == "обращаться на «вы»"


def test_decision_replaces_diverging_decisions_of_other_spellings() -> None:
    """A new decision on an item removes previous decisions on its other spellings."""
    unit_of_work = glossary_book(
        make_analysis_candidate(source_text="Mr Bennet"),
        make_analysis_candidate(source_text="Mr. Bennet"),
    )
    unit_of_work.glossary.save(
        TermConcept(
            id="legacy-1",
            book_id=BOOK_ID,
            type=ConceptType.PERSON,
            canonical_source_text="Mr Bennet",
            status=TermStatus.APPROVED,
            variants=(
                TermVariant(
                    concept_id="legacy-1",
                    language="ru",
                    text="господин Беннет",
                    status=TermStatus.APPROVED,
                ),
            ),
        )
    )
    unit_of_work.glossary.save(
        TermConcept(
            id="legacy-2",
            book_id=BOOK_ID,
            type=ConceptType.PERSON,
            canonical_source_text="Mr. Bennet",
            status=TermStatus.APPROVED,
            variants=(
                TermVariant(
                    concept_id="legacy-2",
                    language="ru",
                    text="мистер Беннет",
                    status=TermStatus.APPROVED,
                ),
            ),
        )
    )
    before = approve(unit_of_work).review(BOOK_ID).item_for("Mr. Bennet")
    assert before is not None
    assert before.has_decision_conflict

    concept = approve(unit_of_work)(command(GlossaryDecision.LOCK))

    after = approve(unit_of_work).review(BOOK_ID).item_for("Mr Bennet")
    assert after is not None
    assert not after.has_decision_conflict
    assert after.superseded_decisions == ()
    assert [stored.id for stored in unit_of_work.glossary.list_for_book(BOOK_ID)] == [
        concept.id
    ]
