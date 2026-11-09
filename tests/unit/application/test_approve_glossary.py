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
from linguary.application.correction_dto import CorrectionSnapshot, CORRECTION_VERSIONS
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
CORRECTION_SETTINGS = CorrectionSnapshot(
    model=LLM_SETTINGS.model,
    route={"provider": LLM_SETTINGS.provider, "base_url": "https://example.test/v1",
           "proxy": "", "route_id": "test-route"},
    parameters={"temperature": LLM_SETTINGS.temperature,
                "max_output_tokens": LLM_SETTINGS.max_output_tokens,
                "timeout_seconds": 30, "max_attempts": 3,
                "max_unit_source_chars": 1000,
                "input_price_per_mtok": 0, "output_price_per_mtok": 0},
    prompt="Check the chapter translation.",
    versions=CORRECTION_VERSIONS,
)


def test_product_processing_requires_correction_snapshot():
    from linguary.application.book_conditions import ConditionsError
    with pytest.raises(ConditionsError, match='correction_policy_unavailable'):
        StartProcessing(unit_of_work=None, glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
            llm_settings_snapshot=LLM_SETTINGS, book_conditions=object())


def test_legacy_processing_without_conditions_service_keeps_compatibility():
    start = StartProcessing(unit_of_work=None, glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        llm_settings_snapshot=LLM_SETTINGS)
    assert start._correction_snapshot is None


class ProcessingWork:
    """Conditions and run bindings needed by the processing start use case."""

    def __init__(self, conditions: SimpleNamespace) -> None:
        self.conditions = conditions
        self.bindings: dict[tuple[str, str], dict] = {}

    def latest_conditions(self, book_id: str) -> SimpleNamespace:
        return self.conditions

    def bind(self, scope: str, scope_id: str, **binding) -> None:
        self.bindings[scope, scope_id] = binding

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
    work = getattr(unit_of_work, "work", None)
    conditions = None if work is None else work.latest_conditions(BOOK_ID)
    payload = {
        "route": {"provider": LLM_SETTINGS.provider},
        "models": {"processing": LLM_SETTINGS.model},
        "llm_parameters": {"temperature": LLM_SETTINGS.temperature,
                           "max_output_tokens": LLM_SETTINGS.max_output_tokens},
    }
    if conditions is not None:
        payload.update(conditions.payload)
    conditions = SimpleNamespace(
        id="conditions-1" if conditions is None else conditions.id,
        payload=payload,
    )
    unit_of_work.work = ProcessingWork(conditions)
    return StartProcessing(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # storage port double
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        llm_settings_snapshot=LLM_SETTINGS,
        correction_snapshot=CORRECTION_SETTINGS,
        book_conditions=SimpleNamespace(require=lambda *args, **kwargs: conditions),
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


def test_approve_glossary_decision_and_accept_all_compute_mention_counter_once_per_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.use_cases.approve_glossary as ag_module
    from analysis_fakes import FakeChapterAnalysisRepository
    from linguary.application.mention_counter import mention_counter as real_mention_counter

    counter_calls = 0

    def counting_mention_counter(document):
        nonlocal counter_calls
        counter_calls += 1
        return real_mention_counter(document)

    monkeypatch.setattr(ag_module, "mention_counter", counting_mention_counter)

    analyses_calls = 0
    orig_list_for_book = FakeChapterAnalysisRepository.list_for_book

    def tracked_list_for_book(self, book_id: str):
        nonlocal analyses_calls
        analyses_calls += 1
        return orig_list_for_book(self, book_id)

    monkeypatch.setattr(FakeChapterAnalysisRepository, "list_for_book", tracked_list_for_book)

    uow1 = glossary_book(make_analysis_candidate())
    analyses_calls = 0
    counter_calls = 0

    approve(uow1)(command(GlossaryDecision.ACCEPT))
    assert counter_calls == 1
    assert analyses_calls == 1
    assert uow1.books.get(BOOK_ID).status is BookStatus.READY_TO_PROCESS

    uow2 = glossary_book(
        make_analysis_candidate(),
        make_analysis_candidate(
            source_text="Longbourn",
            type=ConceptType.PLACE,
            suggested_translation="Лонгборн",
        ),
    )
    analyses_calls = 0
    counter_calls = 0

    approve(uow2).accept_all_awaiting(BOOK_ID)
    assert counter_calls == 1
    assert analyses_calls == 1
    assert uow2.books.get(BOOK_ID).status is BookStatus.READY_TO_PROCESS


@pytest.mark.asyncio
async def test_translation_call_reuses_supplied_source_texts_without_reloading_document() -> None:
    from contextlib import nullcontext
    from processing_fakes import (
        SOURCE_TEXT, make_block, make_plan, make_unit, make_validator, note_ids,
    )
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    from linguary.application.block_processing import TranslationCall
    from linguary.application.context_builder import ContextBuilder
    from linguary.application.retry_policy import RetryPolicy
    from linguary.domain import (
        BlockTreatment, ChapterAnalysis, GlossarySnapshot,
    )
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, SourceAnchor

    block = make_block(order=0, text=SOURCE_TEXT)
    section = make_analysis_section(blocks=(block,))
    doc = make_analysis_document((section,))
    plan = make_plan((block,), treatments=(BlockTreatment.READ_AS_IS,))
    unit = make_unit((block,), treatments=(BlockTreatment.READ_AS_IS,))
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                id="d1",
                anchors=(SourceAnchor(block_id=block.id, start=0, end=2, quote="It"),),
                owner_block_id=block.id,
                kind="grammar",
                priority="useful",
                help_kind="lexical_hint",
                required_points=(RequiredPoint(id="p1", description="grammar point"),),
            ),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(block.id,),
        content_plan=content_plan,
    )
    builder = ContextBuilder(
        reader_level="a2",
        content_plan_profile="content-plan-v3",
        prompt_templates={"english_help": "1", "english_help_processing": "English help prompt."},
    )
    ctx = builder.for_translation_unit(
        document=doc,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=GlossarySnapshot(),
    )

    def forbidden_doc_get(book_id: str):
        raise AssertionError("documents.get must not be called when source_texts is supplied")

    fake_tx = SimpleNamespace(
        runs=SimpleNamespace(active_for_book=lambda book_id: None),
        work=SimpleNamespace(
            get_binding=lambda kind, key: None,
            get_conditions=lambda cid: SimpleNamespace(
                id=cid,
                payload={"versions": {"validators": 4}, "reader_level": "a2"},
            ),
        ),
        documents=SimpleNamespace(get=forbidden_doc_get),
        llm_calls=SimpleNamespace(),
    )
    fake_uow = SimpleNamespace(transaction=lambda: nullcontext(fake_tx))

    reply_payload = {
        "blocks": [
            {
                "source_block_id": "b1",
                "treatment": "READ_AS_IS",
                "translated_text": None,
                "glossary_hints": [],
            }
        ],
        "language_notes": [],
        "fragment_assistances": [
            {
                "difficulty_id": "d1",
                "point_ids": ["p1"],
                "kind": "lexical_hint",
                "text": "a known fact",
            }
        ],
    }
    client = FakeLLMClient.always(FakeReply.of_payload(reply_payload))
    call = TranslationCall(
        unit_of_work=fake_uow,  # type: ignore[arg-type]
        llm_client=client,
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
        sleep=lambda _: None,  # type: ignore[arg-type]
    )
    res = await call.request(
        ctx,
        unit=unit,
        target_language="ru",
        note_ids=note_ids(),
        source_texts={block.id: block.text},
    )
    assert len(res.results) == 1
    assert res.results[0].fragment_assistances[0].text == "a known fact"

    call_with_init_texts = TranslationCall(
        unit_of_work=fake_uow,  # type: ignore[arg-type]
        llm_client=client,
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
        sleep=lambda _: None,  # type: ignore[arg-type]
        source_texts={block.id: block.text},
    )
    res2 = await call_with_init_texts.request(
        ctx,
        unit=unit,
        target_language="ru",
        note_ids=note_ids(),
    )
    assert len(res2.results) == 1


def test_approve_glossary_single_decision_does_not_recount_book_mentions_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.use_cases.approve_glossary as ag_module
    from linguary.application.mention_counter import mention_counter as real_mention_counter

    mention_queries: list[tuple[str, ...]] = []
    load_review_calls = 0

    def spy_mention_counter(document):
        inner = real_mention_counter(document)

        def counted(spellings):
            key = tuple(spellings)
            mention_queries.append(key)
            return inner(spellings)

        return counted

    orig_load_review = ag_module._load_review

    def spy_load_review(*args, **kwargs):
        nonlocal load_review_calls
        load_review_calls += 1
        return orig_load_review(*args, **kwargs)

    monkeypatch.setattr(ag_module, "mention_counter", spy_mention_counter)
    monkeypatch.setattr(ag_module, "_load_review", spy_load_review)

    uow = glossary_book(make_analysis_candidate())
    approve(uow)(command(GlossaryDecision.ACCEPT))

    assert load_review_calls == 1
    assert len(mention_queries) == 1
    assert uow.books.get(BOOK_ID).status is BookStatus.READY_TO_PROCESS

    mention_queries.clear()
    load_review_calls = 0
    uow2 = glossary_book(
        make_analysis_candidate(),
        make_analysis_candidate(
            source_text="Longbourn",
            type=ConceptType.PLACE,
            suggested_translation="Лонгборн",
        ),
    )
    approve(uow2).accept_all_awaiting(BOOK_ID)

    assert load_review_calls == 1
    assert len(mention_queries) == 2
    assert uow2.books.get(BOOK_ID).status is BookStatus.READY_TO_PROCESS


@pytest.mark.asyncio
async def test_finalize_attempt_runs_bounded_body_when_redacted_count_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from contextlib import nullcontext
    from datetime import UTC, datetime
    import linguary.adapters.pronunciation.source as pron_source
    from linguary.application.diagnostics import Diagnostics
    from linguary.application.task_runner import TaskRunner

    started_at = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    stored_bodies: dict[str, str | None] = {}
    stored_meta: dict[str, dict] = {}
    record = {
        "call_id": "call-1",
        "attempt_id": None,
        "book_id": BOOK_ID,
        "started_at": started_at,
        "metadata": {
            "schema_version": 1,
            "decision": "unknown",
            "body_meta": {
                "body_status": "available",
                "redacted": False,
                "truncated": False,
                "original_bytes": 42,
                "stored_bytes": 42,
            },
        },
        "body": None,
    }
    tx = SimpleNamespace(
        llm_calls=SimpleNamespace(
            diagnostic_record=lambda **kwargs: record,
            update_diagnostics=lambda cid, *, metadata: stored_meta.__setitem__(cid, metadata),
            store_diagnostic_body=lambda cid, body: stored_bodies.__setitem__(cid, body),
            enforce_diagnostics_policy=lambda *args, **kwargs: None,
        ),
        work=SimpleNamespace(),
    )
    uow = SimpleNamespace(transaction=lambda: nullcontext(tx))
    diagnostics = Diagnostics(uow, clock=lambda: started_at)

    raw_with_secret = "Authorization: Bearer sk-live-secret-12345"
    meta = diagnostics.finalize_attempt(
        None,
        "rejected",
        body=raw_with_secret,
        call_id="call-1",
        body_meta=record["metadata"]["body_meta"],
    )

    assert stored_bodies["call-1"] == "Authorization: Bearer [REDACTED]"
    assert meta["body_meta"]["redacted"] is True

    assert diagnostics.needs_periodic_purge(started_at) is True
    assert diagnostics.periodic_purge(started_at) is True
    assert diagnostics.needs_periodic_purge(started_at) is False

    to_thread_calls = 0
    orig_to_thread = asyncio.to_thread

    async def spy_to_thread(func, *args, **kwargs):
        nonlocal to_thread_calls
        to_thread_calls += 1
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    def maintain():
        diagnostics.ensure_available()

    maintain.should_run = diagnostics.needs_periodic_purge
    queue = SimpleNamespace(
        claim_next_pending=lambda: asyncio.sleep(0, result=None),
        list_running=lambda: asyncio.sleep(0, result=()),
    )
    runner = TaskRunner(queue=queue, handlers={}, maintenance=maintain)
    assert await runner.run_next() is None
    assert to_thread_calls == 0

    pron_source._load_index.cache_clear()
    source = pron_source.IpaDictPronunciationSource()
    first_lookup = await source.lookup("cat", "uk")
    assert first_lookup.status == "found"
    assert to_thread_calls == 1

    second_lookup = await source.lookup("dog", "uk")
    assert second_lookup.status == "found"
    assert to_thread_calls == 1


def test_approve_glossary_reuses_or_loads_document_and_passes_to_build_glossary_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.use_cases.approve_glossary as ag_module

    passed_documents: list[object] = []
    orig_build_review = ag_module.build_glossary_review

    def spy_build_review(*args, **kwargs):
        passed_documents.append(kwargs.get("document"))
        return orig_build_review(*args, **kwargs)

    monkeypatch.setattr(ag_module, "build_glossary_review", spy_build_review)

    def wrap_documents_repo(uow: FakeUnitOfWork) -> list[str]:
        calls: list[str] = []
        backing = uow._documents

        class CountingDocuments:
            def get(self, book_id: str):
                calls.append(book_id)
                return backing.get(book_id)

        uow._documents = CountingDocuments()  # type: ignore[assignment]
        return calls

    # 1. review() loads document when document=None and passes it to build_glossary_review
    uow1 = glossary_book(make_analysis_candidate())
    expected_doc1 = uow1.documents.get(BOOK_ID)
    doc_calls1 = wrap_documents_repo(uow1)
    passed_documents.clear()

    approve(uow1).review(BOOK_ID)
    assert doc_calls1 == [BOOK_ID]
    assert passed_documents == [expected_doc1]

    # 2. __call__() reuses preloaded document (1 read per transaction) and passes it to build_glossary_review
    doc_calls1.clear()
    passed_documents.clear()

    approve(uow1)(command(GlossaryDecision.ACCEPT))
    assert doc_calls1 == [BOOK_ID]
    assert passed_documents == [expected_doc1]

    # 3. accept_all_awaiting() reuses preloaded document (1 read per transaction) and passes it to build_glossary_review
    uow2 = glossary_book(
        make_analysis_candidate(),
        make_analysis_candidate(
            source_text="Longbourn",
            type=ConceptType.PLACE,
            suggested_translation="Лонгборн",
        ),
    )
    expected_doc2 = uow2.documents.get(BOOK_ID)
    doc_calls2 = wrap_documents_repo(uow2)
    passed_documents.clear()

    approve(uow2).accept_all_awaiting(BOOK_ID)
    assert doc_calls2 == [BOOK_ID]
    assert passed_documents == [expected_doc2]




