"""User glossary review and book processing start (PRD § 11-12, § 21.3).

`ApproveGlossary` displays the book glossary and manages four user actions per
item: accept suggested translation, correct it, lock as mandatory,
and skip the item as not requiring translation consistency. Each action persists
a `TermConcept` — the accepted decision; they differ in concept status and variant status.
While the glossary is being edited, decisions can be re-applied — both to items
already decided and to suggestions that did not require attention (PRD § 11, § 21.3).

`StartProcessing` completes the review: it rejects if unresolved items requiring attention
remain (PRD § 12), and otherwise freezes decisions into a snapshot
in `ProcessingRun`. From that point on, the book glossary is read-only — returning
it to editable status is only possible by rewinding to glossary review (P3.T7), which deletes
processing results.

Block processing itself is outside this use case: this opens the run with its snapshots,
while `TranslationUnit` processes in P3.T6.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from uuid import uuid4

from linguary.application.glossary_review import (
    GlossaryNotFoundError,
    GlossaryReview,
    GlossaryReviewError,
    GlossaryReviewItem,
    build_glossary_review,
    glossary_snapshot,
)
from linguary.application.mention_counter import mention_counter
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import (
    BookEvent,
    BookStatus,
    ConceptType,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    TermConcept,
    TermStatus,
    TermVariant,
    apply_book_event,
)

#: Factory for IDs of persisted glossary records and processing runs.
type IdFactory = Callable[[], str]

#: Book statuses where glossary decisions remain editable (PRD § 11).
EDITABLE_STATUSES: frozenset[BookStatus] = frozenset(
    {BookStatus.GLOSSARY_REVIEW, BookStatus.READY_TO_PROCESS}
)


class GlossaryDecision(StrEnum):
    """User action for a glossary item (PRD § 12)."""

    ACCEPT = "accept"
    CORRECT = "correct"
    LOCK = "lock"
    SKIP = "skip"


#: Status of the concept and its variant after each action. `skip` leaves the item
#: as an optional hint: user decided consistency is not required,
#: and the model-suggested variant remains in the glossary without binding force (PRD § 12).
DECISION_STATUSES: dict[GlossaryDecision, TermStatus] = {
    GlossaryDecision.ACCEPT: TermStatus.APPROVED,
    GlossaryDecision.CORRECT: TermStatus.APPROVED,
    GlossaryDecision.LOCK: TermStatus.LOCKED,
    GlossaryDecision.SKIP: TermStatus.SUGGESTED,
}


@dataclass(frozen=True, slots=True)
class GlossaryDecisionCommand:
    """User decision for a single book glossary item.

    `translation` is only used for corrections: other actions work with an already known
    variant — suggested by the model or preserved from a previous decision. `concept_id`
    links the HTTP command to the displayed item and preserves its ID upon the first decision;
    direct use case invocations can continue working by source text.

    `note` is a translation guideline for the item. `None` means "leave existing unchanged",
    and a provided string replaces it completely, including an empty string: the user can
    clear a guideline just as they can set one (PRD § 11). The exception is skipping an item
    without an existing decision: the model-suggested guideline was not confirmed by the user,
    and skipping does not save it.

    `type` is the concept type corrected by the user. Like `translation`, it is only
    accepted during correction: a model classification mistake ("dessert — character")
    is corrected in the same action as the translation variant.
    """

    book_id: str
    source_text: str
    decision: GlossaryDecision
    translation: str | None = None
    concept_id: str | None = None
    note: str | None = None
    type: ConceptType | None = None


@dataclass(frozen=True, slots=True)
class GlossaryNoteCommand:
    """Editing translation note for a single glossary item (PRD § 11)."""

    book_id: str
    concept_id: str
    note: str


@dataclass(frozen=True, slots=True)
class StartProcessingCommand:
    """Command to start processing a book."""

    book_id: str


class ApproveGlossary:
    """Presents book glossary and saves user decisions (PRD § 12)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        glossary_confidence_threshold: float,
        glossary_review_min_mentions: int = 5,
        glossary_review_min_chapters: int = 3,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._glossary_review_min_mentions = glossary_review_min_mentions
        self._glossary_review_min_chapters = glossary_review_min_chapters
        self._id_factory = id_factory or _new_id

    def review(self, book_id: str) -> GlossaryReview:
        """Book glossary along with already accepted decisions."""
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, book_id)
            return replace(
                _load_review(
                    transaction,
                    book_id,
                    confidence_threshold=self._glossary_confidence_threshold,
                    min_mentions=self._glossary_review_min_mentions,
                    min_chapters=self._glossary_review_min_chapters,
                ),
                book_status=book.status,
                is_editable=book.status in EDITABLE_STATUSES,
            )

    def __call__(self, command: GlossaryDecisionCommand) -> TermConcept:
        """Saves user decision for a single glossary item."""
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            _ensure_editable(book)
            review = _load_review(
                transaction,
                command.book_id,
                confidence_threshold=self._glossary_confidence_threshold,
                min_mentions=self._glossary_review_min_mentions,
                min_chapters=self._glossary_review_min_chapters,
            )
            item = review.item_for(command.source_text)
            if item is None:
                raise GlossaryNotFoundError(
                    f"в глоссарии книги {command.book_id} нет элемента "
                    f"{command.source_text!r}"
                )
            if (
                command.concept_id is not None
                and command.concept_id != item.concept_id
            ):
                raise GlossaryNotFoundError(
                    f"в глоссарии книги {command.book_id} нет понятия "
                    f"{command.concept_id!r}"
                )
            concept = self._decide(item, command, book=book)
            transaction.glossary.save(concept)
            _drop_superseded(transaction, item)
            _complete_review_if_ready(
                transaction,
                book,
                confidence_threshold=self._glossary_confidence_threshold,
                min_mentions=self._glossary_review_min_mentions,
                min_chapters=self._glossary_review_min_chapters,
            )
        return concept

    def accept_all_awaiting(self, book_id: str) -> tuple[TermConcept, ...]:
        """Accepts suggested translations for all items awaiting decision (PRD § 12).

        Batch action runs in a single transaction and over a single assembled glossary:
        per-item invocation would reassemble the whole book glossary for each item.
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, book_id)
            _ensure_editable(book)
            review = _load_review(
                transaction,
                book_id,
                confidence_threshold=self._glossary_confidence_threshold,
                min_mentions=self._glossary_review_min_mentions,
                min_chapters=self._glossary_review_min_chapters,
            )
            concepts = tuple(
                self._decide(
                    item,
                    GlossaryDecisionCommand(
                        book_id=book_id,
                        source_text=item.source_text,
                        concept_id=item.concept_id,
                        decision=GlossaryDecision.ACCEPT,
                    ),
                    book=book,
                )
                for item in review.awaiting_decision
            )
            for concept in concepts:
                transaction.glossary.save(concept)
            _complete_review_if_ready(
                transaction,
                book,
                confidence_threshold=self._glossary_confidence_threshold,
                min_mentions=self._glossary_review_min_mentions,
                min_chapters=self._glossary_review_min_chapters,
            )
        return concepts

    def set_note(self, command: GlossaryNoteCommand) -> TermConcept:
        """Updates translation note for an item without modifying the decision itself.

        Notes are edited separately from decisions because they outlive them: a character's
        gender may become apparent only after several chapters, while their name's translation
        has already been approved. This cannot be done via normal decision — re-applying `accept`
        would revert to model-suggested variant and overwrite user correction.
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            _ensure_editable(book)
            review = _load_review(
                transaction,
                command.book_id,
                confidence_threshold=self._glossary_confidence_threshold,
                min_mentions=self._glossary_review_min_mentions,
                min_chapters=self._glossary_review_min_chapters,
            )
            item = review.item_for_concept(command.concept_id)
            if item is None:
                raise GlossaryNotFoundError(
                    f"в глоссарии книги {command.book_id} нет понятия "
                    f"{command.concept_id!r}"
                )
            concept = self._with_note(item, command.note.strip(), book=book)
            transaction.glossary.save(concept)
        return concept

    def _with_note(
        self, item: GlossaryReviewItem, note: str, *, book: BookRecord
    ) -> TermConcept:
        """Glossary concept with new note.

        There may not yet be an explicit decision for the item: adding a note to a model-suggested
        variant constitutes a user decision, so the item becomes approved with the suggested variant.
        """
        if item.decision is not None:
            return replace(item.decision, note=note)
        concept_id = item.concept_id
        return TermConcept(
            id=concept_id,
            book_id=book.id,
            type=item.type,
            canonical_source_text=item.source_text,
            status=TermStatus.APPROVED,
            note=note,
            variants=(
                TermVariant(
                    concept_id=concept_id,
                    language=book.target_language,
                    text=item.suggested_translation,
                    status=TermStatus.APPROVED,
                ),
            ),
        )

    def complete_if_ready(self, book_id: str) -> None:
        """Completes glossary review if no mandatory decisions remain."""
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, book_id)
            _ensure_editable(book)
            _complete_review_if_ready(
                transaction,
                book,
                confidence_threshold=self._glossary_confidence_threshold,
                min_mentions=self._glossary_review_min_mentions,
                min_chapters=self._glossary_review_min_chapters,
            )

    def _decide(
        self,
        item: GlossaryReviewItem,
        command: GlossaryDecisionCommand,
        *,
        book: BookRecord,
    ) -> TermConcept:
        """Glossary concept resulting from user decision.

        A repeated decision on the same item rewrites the concept completely:
        concept ID is preserved, while status and translation variant are replaced.
        Concept type is only changed by correction; other decisions preserve previous type.
        """
        status = DECISION_STATUSES[command.decision]
        concept_id = item.decision.id if item.decision is not None else (
            command.concept_id or self._id_factory()
        )
        return TermConcept(
            id=concept_id,
            book_id=book.id,
            type=_concept_type(item, command),
            canonical_source_text=item.source_text,
            status=status,
            note=_decision_note(item, command),
            variants=(
                TermVariant(
                    concept_id=concept_id,
                    language=book.target_language,
                    text=_translation(item, command, language=book.target_language),
                    status=status,
                ),
            ),
        )


class StartProcessing:
    """Freezes glossary decisions and initiates book processing run (PRD § 11-12)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        glossary_confidence_threshold: float,
        llm_settings_snapshot: LLMSettingsSnapshot,
        glossary_review_min_mentions: int = 5,
        glossary_review_min_chapters: int = 3,
        id_factory: IdFactory | None = None,
        book_conditions=None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._book_conditions = book_conditions
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._glossary_review_min_mentions = glossary_review_min_mentions
        self._glossary_review_min_chapters = glossary_review_min_chapters
        self._llm_settings_snapshot = llm_settings_snapshot
        self._id_factory = id_factory or _new_id

    def __call__(self, command: StartProcessingCommand) -> ProcessingRun:
        """Creates an active processing run with snapshots of glossary and LLM settings.

        Rejection can originate from three places: unhandled items requiring attention
        (PRD § 12), an already active run for the book (ADR § 3), and book states from which
        processing cannot be launched (ALG-06).
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            conditions = None if self._book_conditions is None else self._book_conditions.require(book.id, transaction=transaction)
            snapshot = self._llm_settings_snapshot
            if conditions is not None:
                payload = conditions.payload
                snapshot = LLMSettingsSnapshot(provider=payload['route']['provider'], model=payload['models']['processing'],
                    temperature=payload['llm_parameters']['temperature'], max_output_tokens=payload['llm_parameters']['max_output_tokens'])
            if transaction.runs.active_for_book(book.id) is not None:
                raise GlossaryReviewError(f"у книги {book.id} уже идёт обработка")
            pinned_checks = conditions.payload['checks'] if conditions is not None else {}
            review = _load_review(
                transaction,
                book.id,
                confidence_threshold=pinned_checks.get('glossary_confidence_threshold', self._glossary_confidence_threshold),
                min_mentions=pinned_checks.get('glossary_review_min_mentions', self._glossary_review_min_mentions),
                min_chapters=pinned_checks.get('glossary_review_min_chapters', self._glossary_review_min_chapters),
            )
            review.ensure_complete()
            started = _status_after_start(book.status)
            run = ProcessingRun(
                id=self._id_factory(),
                book_id=book.id,
                status=ProcessingRunStatus.ACTIVE,
                glossary_snapshot=glossary_snapshot(
                    review, language=book.target_language
                ),
                llm_settings_snapshot=snapshot,
            )
            transaction.runs.add(run)
            if conditions is not None:
                from dataclasses import asdict
                from linguary.application.book_conditions import payload_digest
                glossary = asdict(run.glossary_snapshot)
                transaction.work.bind('run', run.id, book_id=book.id, conditions_id=conditions.id,
                    payload={'schema_version': 1, 'glossary_snapshot': glossary, 'glossary_sha256': payload_digest(glossary)})
            transaction.books.save(replace(book, status=started))
        return run


def _load_review(
    transaction: UnitOfWork, book_id: str, *, confidence_threshold: float,
    min_mentions: int = 5, min_chapters: int = 3,
) -> GlossaryReview:
    """Book glossary assembled in open transaction."""
    return build_glossary_review(
        book_id=book_id,
        analyses=transaction.analyses.list_for_book(book_id),
        confidence_threshold=confidence_threshold,
        decisions=transaction.glossary.list_for_book(book_id),
        count_mentions=mention_counter(document) if (document := transaction.documents.get(book_id)) else None,
        min_mentions=min_mentions,
        min_chapters=min_chapters,
    )


def _book(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Book record in open transaction."""
    book = transaction.books.get(book_id)
    if book is None:
        raise GlossaryNotFoundError(f"книги {book_id} нет в хранилище")
    return book


def _complete_review_if_ready(
    transaction: UnitOfWork,
    book: BookRecord,
    *,
    confidence_threshold: float,
    min_mentions: int = 5,
    min_chapters: int = 3,
) -> None:
    """Transitions book to ready to process after closing all mandatory decisions."""
    if book.status is not BookStatus.GLOSSARY_REVIEW:
        return
    review = _load_review(
        transaction,
        book.id,
        confidence_threshold=confidence_threshold,
        min_mentions=min_mentions,
        min_chapters=min_chapters,
    )
    if review.is_complete:
        transaction.books.save(
            replace(
                book,
                status=apply_book_event(
                    book.status, BookEvent.GLOSSARY_REVIEW_COMPLETED
                ),
            )
        )


def _ensure_editable(book: BookRecord) -> None:
    """Checks that the book glossary is still editable (PRD § 11, § 21.3).

    After processing starts, decisions are frozen in the run snapshot, and the review screen
    is read-only: returning glossary to editable state requires rewinding to glossary review,
    which deletes processing results.
    """
    if book.status not in EDITABLE_STATUSES:
        raise GlossaryReviewError(
            f"глоссарий книги {book.id} в состоянии {book.status} доступен только "
            f"для чтения: чтобы изменить решения, вернитесь к проверке глоссария"
        )


def _drop_superseded(transaction: UnitOfWork, item: GlossaryReviewItem) -> None:
    """Removes previous decisions for other spellings of the item after a new decision.

    Until user makes a new decision, diverging decisions are shown as conflicts;
    the new decision applies to the whole item, and previous decisions no longer apply.
    """
    for concept in item.superseded_decisions:
        transaction.glossary.delete(concept.id)


def _concept_type(
    item: GlossaryReviewItem, command: GlossaryDecisionCommand
) -> ConceptType:
    """Concept type after decision: user corrected or previous."""
    if command.type is not None and command.decision is not GlossaryDecision.CORRECT:
        raise GlossaryReviewError(
            f"действие {command.decision} по элементу {item.source_text!r} типа понятия "
            f"не принимает: тип меняется исправлением"
        )
    return command.type or item.concept_type


def _decision_note(item: GlossaryReviewItem, command: GlossaryDecisionCommand) -> str:
    """Translation note after user decision.

    An explicitly provided note replaces the current one. Without one, the decision preserves
    the previous decision's note, or for the first decision — the note suggested by chapters:
    approving an item with a shown note implies agreeing with it. Skipping an item
    without a decision does not preserve the suggested note: user did not confirm it,
    and it should not pass to processing.
    """
    if command.note is not None:
        return command.note.strip()
    if item.decision is not None:
        return item.decision.note
    if command.decision is GlossaryDecision.SKIP:
        return ""
    return item.suggested_note


def _translation(
    item: GlossaryReviewItem, command: GlossaryDecisionCommand, *, language: str
) -> str:
    """Translation variant text after user decision.

    `correct` uses submitted text, `accept` uses model suggestion, and `lock` and `skip`
    preserve text of previous decision if one existed: lock and skip change the item's
    status rather than its translation.
    """
    corrected = (command.translation or "").strip()
    if command.decision is GlossaryDecision.CORRECT:
        if not corrected:
            raise GlossaryReviewError(
                f"исправление элемента {item.source_text!r} не содержит варианта перевода"
            )
        return corrected
    if corrected:
        raise GlossaryReviewError(
            f"действие {command.decision} по элементу {item.source_text!r} варианта "
            f"перевода не принимает: исправление — отдельное действие"
        )
    if command.decision is GlossaryDecision.ACCEPT:
        return item.suggested_translation
    previous = item.decision.variant_for(language) if item.decision is not None else None
    return previous.text if previous is not None else item.suggested_translation


def _status_after_start(status: BookStatus) -> BookStatus:
    """Book status after starting processing (ALG-06).

    Completing glossary review and starting the run are two transition table rows,
    but a single user action. A book in a state where processing cannot start
    is rejected by the domain (409 in Web UI).
    """
    if status is BookStatus.GLOSSARY_REVIEW:
        status = apply_book_event(status, BookEvent.GLOSSARY_REVIEW_COMPLETED)
    return apply_book_event(status, BookEvent.PROCESSING_RUN_STARTED)


def _new_id() -> str:
    """ID for new glossary concept or processing run."""
    return str(uuid4())


__all__ = [
    "DECISION_STATUSES",
    "EDITABLE_STATUSES",
    "ApproveGlossary",
    "GlossaryDecision",
    "GlossaryDecisionCommand",
    "GlossaryNoteCommand",
    "IdFactory",
    "StartProcessing",
    "StartProcessingCommand",
]
