"""Решения пользователя по глоссарию и старт обработки (P3.T5, PRD § 11-12, § 21.3)."""

from __future__ import annotations

import pytest
from analysis_fakes import (
    BOOK_ID,
    CONFIDENCE_THRESHOLD,
    FakeUnitOfWork,
    make_analysis_candidate,
    make_chapter_analysis,
    make_glossary_book,
    sequential_ids,
)

from linguary.application.glossary_review import (
    GlossaryReviewError,
    GlossaryReviewIncompleteError,
)
from linguary.application.use_cases.approve_glossary import (
    ApproveGlossary,
    GlossaryDecision,
    GlossaryDecisionCommand,
    StartProcessing,
    StartProcessingCommand,
)
from linguary.domain import (
    BookStatus,
    BookStatusTransitionError,
    ConceptType,
    GlossaryCandidate,
    LLMSettingsSnapshot,
    TermStatus,
)

LLM_SETTINGS = LLMSettingsSnapshot(provider="openai_compatible", model="gpt-test")

#: Уверенно переведённый термин: подсказка, решения не требует.
HINT = "entailed estate"


def hint_candidate() -> GlossaryCandidate:
    """Кандидат, который проверки не требует и старту обработки не мешает."""
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
    """Книга с одной проанализированной главой и её кандидатами глоссария."""
    return make_glossary_book(
        (make_chapter_analysis(candidates=candidates),), status=status
    )


def approve(unit_of_work: FakeUnitOfWork) -> ApproveGlossary:
    """Сценарий решений по глоссарию поверх двойника хранилища."""
    return ApproveGlossary(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # двойник порта хранения
        glossary_confidence_threshold=CONFIDENCE_THRESHOLD,
        id_factory=sequential_ids(),
    )


def start(unit_of_work: FakeUnitOfWork) -> StartProcessing:
    """Сценарий старта обработки поверх двойника хранилища."""
    return StartProcessing(
        unit_of_work=unit_of_work,  # type: ignore[arg-type]  # двойник порта хранения
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
    """Решение пользователя по одному элементу глоссария книги."""
    return GlossaryDecisionCommand(
        book_id=BOOK_ID,
        source_text=source_text,
        decision=decision,
        translation=translation,
    )


def started_book() -> FakeUnitOfWork:
    """Книга, обработка которой уже запущена с зафиксированным глоссарием."""
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())
    approve(unit_of_work)(command(GlossaryDecision.ACCEPT))
    start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))
    return unit_of_work


def test_accept_stores_approved_decision() -> None:
    """Принятие сохраняет вариант модели как утверждённое решение."""
    unit_of_work = glossary_book(make_analysis_candidate())

    concept = approve(unit_of_work)(command(GlossaryDecision.ACCEPT))

    variant = concept.variant_for("ru")
    assert concept.status is TermStatus.APPROVED
    assert variant is not None
    assert variant.text == "мистер Беннет"
    assert unit_of_work.glossary.list_for_book(BOOK_ID) == (concept,)
    assert approve(unit_of_work).review(BOOK_ID).is_complete


def test_correct_stores_user_translation() -> None:
    """Исправление сохраняет вариант пользователя вместо предложенного моделью."""
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
    """Исправление без варианта перевода решением не считается."""
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
    """Вариант перевода принимает только исправление: остальные действия меняют статус."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError):
        approve(unit_of_work)(command(decision, translation="мистер Беннетт"))

    assert unit_of_work.glossary.list_for_book(BOOK_ID) == ()


def test_lock_keeps_corrected_translation() -> None:
    """Фиксация сохраняет исправленный вариант и переписывает прежнее решение."""
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
    """Пропущенный как неважный элемент остаётся необязательной подсказкой."""
    unit_of_work = glossary_book(make_analysis_candidate())

    concept = approve(unit_of_work)(command(GlossaryDecision.SKIP))

    review = approve(unit_of_work).review(BOOK_ID)
    assert concept.status is TermStatus.SUGGESTED
    assert review.items[0].is_decided
    assert review.is_complete


def test_decision_on_unknown_item_is_rejected() -> None:
    """Решение принимается только по элементу глоссария этой книги."""
    unit_of_work = glossary_book(make_analysis_candidate())

    with pytest.raises(GlossaryReviewError, match="Winston Smith"):
        approve(unit_of_work)(
            command(GlossaryDecision.ACCEPT, source_text="Winston Smith")
        )


def test_review_of_unknown_book_is_rejected() -> None:
    """Глоссарий читается только у существующей книги."""
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
    """После старта обработки решения глоссария не меняются (PRD § 11, § 21.3)."""
    unit_of_work = glossary_book(make_analysis_candidate(), status=status)

    with pytest.raises(GlossaryReviewError, match="только"):
        approve(unit_of_work)(command(GlossaryDecision.ACCEPT))

    assert unit_of_work.glossary.list_for_book(BOOK_ID) == ()


def test_decisions_are_frozen_after_start() -> None:
    """Запущенная обработка закрывает глоссарий для правки, а снимок остаётся прежним."""
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
    """Обработка не запускается, пока среди требующих внимания есть необработанные."""
    unit_of_work = glossary_book(make_analysis_candidate(), hint_candidate())

    with pytest.raises(GlossaryReviewIncompleteError, match="Mr. Bennet"):
        start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    book = unit_of_work.books.get(BOOK_ID)
    assert book is not None
    assert book.status is BookStatus.GLOSSARY_REVIEW
    assert unit_of_work.runs.active_for_book(BOOK_ID) is None


def test_start_freezes_glossary_snapshot() -> None:
    """Старт обработки замораживает решения снимком запуска (PRD § 11)."""
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
    """Старт обработки переводит книгу в `PROCESSING` (ALG-06)."""
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
    """Из состояния, не дошедшего до проверки глоссария, обработка не запускается."""
    unit_of_work = glossary_book(hint_candidate(), status=status)

    with pytest.raises(BookStatusTransitionError):
        start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    assert unit_of_work.runs.active_for_book(BOOK_ID) is None


def test_second_active_run_is_rejected() -> None:
    """У книги не бывает двух активных запусков обработки (ADR § 3)."""
    unit_of_work = started_book()

    with pytest.raises(GlossaryReviewError, match="уже идёт обработка"):
        start(unit_of_work)(StartProcessingCommand(book_id=BOOK_ID))

    assert len(unit_of_work.runs.runs) == 1
