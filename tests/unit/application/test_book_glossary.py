"""Глоссарий книги: сборка из кандидатов глав и отбор требующих внимания (P3.T5, PRD § 11-12)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from analysis_fakes import (
    BOOK_ID,
    CONFIDENCE_THRESHOLD,
    make_analysis_candidate,
    make_chapter_analysis,
)

from linguary.application.glossary_review import (
    GlossaryReview,
    GlossaryReviewIncompleteError,
    build_glossary_review,
    glossary_snapshot,
)
from linguary.domain import (
    ChapterAnalysis,
    ConceptType,
    GlossaryCandidate,
    TermConcept,
    TermStatus,
    TermVariant,
)

TARGET_LANGUAGE = "ru"


def chapter_one(*candidates: GlossaryCandidate) -> ChapterAnalysis:
    """Анализ первой главы книги с её кандидатами глоссария."""
    return make_chapter_analysis(
        id="analysis-1", section_id="section-1", candidates=candidates
    )


def chapter_two(*candidates: GlossaryCandidate) -> ChapterAnalysis:
    """Анализ второй главы книги с её кандидатами глоссария."""
    return make_chapter_analysis(
        id="analysis-2", section_id="section-2", candidates=candidates
    )


def decision(
    source_text: str,
    *,
    text: str = "мистер Беннет",
    status: TermStatus = TermStatus.APPROVED,
    concept_type: ConceptType = ConceptType.PERSON,
    language: str = TARGET_LANGUAGE,
    concept_id: str = "concept-1",
) -> TermConcept:
    """Сохранённое решение пользователя по элементу глоссария."""
    return TermConcept(
        id=concept_id,
        book_id=BOOK_ID,
        type=concept_type,
        canonical_source_text=source_text,
        status=status,
        variants=(
            TermVariant(
                concept_id=concept_id, language=language, text=text, status=status
            ),
        ),
    )


def review(
    *analyses: ChapterAnalysis, decisions: Sequence[TermConcept] = ()
) -> GlossaryReview:
    """Глоссарий книги, собранный из переданных анализов глав."""
    return build_glossary_review(
        book_id=BOOK_ID,
        analyses=analyses,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        decisions=decisions,
    )


def test_review_counts_chapters_by_analyses() -> None:
    first = make_analysis_candidate(source_text="Dracula")
    second = make_analysis_candidate(source_text="Dracula")
    book = build_glossary_review(
        book_id=BOOK_ID,
        analyses=(chapter_one(first, second),),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        count_mentions=lambda _: 0,
        min_chapters=2,
    )
    assert not book.items[0].is_frequent


def test_review_uses_mentions_for_frequency() -> None:
    candidate = make_analysis_candidate(source_text="Dracula")
    rare = build_glossary_review(
        book_id=BOOK_ID, analyses=(chapter_one(candidate),),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        count_mentions=lambda _: 4, min_mentions=5,
    )
    frequent = build_glossary_review(
        book_id=BOOK_ID, analyses=(chapter_one(candidate),),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        count_mentions=lambda _: 5, min_mentions=5,
    )
    assert not rare.items[0].requires_review
    assert frequent.items[0].requires_review


def test_model_doubts_sorted_first_among_suggestions() -> None:
    book = build_glossary_review(
        book_id=BOOK_ID,
        analyses=(chapter_one(
            make_analysis_candidate(source_text="Alpha", type=ConceptType.TERM),
            make_analysis_candidate(source_text="Zeta", type=ConceptType.TERM,
                                    model_flagged_review=True),
        ),),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        count_mentions=lambda _: 1,
    )
    assert [item.source_text for item in book.items] == ["Zeta", "Alpha"]
    assert book.items[0].model_doubts


@pytest.mark.parametrize(
    "concept_type",
    [
        pytest.param(ConceptType.PERSON, id="person"),
        pytest.param(ConceptType.PLACE, id="place"),
        pytest.param(ConceptType.ORGANIZATION, id="organization"),
        pytest.param(ConceptType.FICTIONAL_CONCEPT, id="fictional-concept"),
    ],
)
def test_named_entity_requires_review_at_full_confidence(
    concept_type: ConceptType,
) -> None:
    """Имя, место, организация и вымышленное понятие требуют решения при любой уверенности."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                source_text="Bistritz",
                type=concept_type,
                suggested_translation="Бистрица",
                confidence=1.0,
            )
        )
    )

    item = book.items[0]
    assert item.requires_review
    assert item.awaits_decision
    assert book.is_complete is False


def test_confident_term_stays_optional_hint() -> None:
    """Уверенно переведённый термин решения не требует и остаётся подсказкой."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                source_text="paprika hendl",
                type=ConceptType.TERM,
                suggested_translation="паприкаш",
                confidence=0.95,
            )
        )
    )

    item = book.items[0]
    assert item.requires_review is False
    assert item.awaits_decision is False
    assert book.is_complete


def test_term_below_confidence_threshold_requires_review() -> None:
    """Уверенность ниже настроенного порога переводит термин в требующие внимания."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                source_text="vrolok",
                type=ConceptType.TERM,
                suggested_translation="вролок",
                confidence=CONFIDENCE_THRESHOLD - 0.1,
            )
        )
    )

    assert book.items[0].requires_review


def test_model_flag_from_any_chapter_requires_review() -> None:
    """Пометка модели хотя бы в одной главе делает общий элемент требующим внимания."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                analysis_id="analysis-1",
                source_text="vrolok",
                type=ConceptType.TERM,
                suggested_translation="вролок",
                confidence=0.95,
            )
        ),
        chapter_two(
            make_analysis_candidate(
                analysis_id="analysis-2",
                source_text="vrolok",
                type=ConceptType.TERM,
                suggested_translation="вролок",
                confidence=0.9,
                model_flagged_review=True,
            )
        ),
    )

    assert len(book.items) == 1
    item = book.items[0]
    assert item.model_flagged_review
    assert item.requires_review


def test_conflicting_translations_between_chapters_require_review() -> None:
    """Разные варианты перевода одного текста в разных главах требуют решения."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                analysis_id="analysis-1",
                source_text="Mittel Land",
                type=ConceptType.TERM,
                suggested_translation="Миттельланд",
                confidence=0.99,
            )
        ),
        chapter_two(
            make_analysis_candidate(
                analysis_id="analysis-2",
                source_text="Mittel Land",
                type=ConceptType.TERM,
                suggested_translation="Средний край",
                confidence=0.9,
            )
        ),
    )

    item = book.items[0]
    assert item.has_translation_conflict
    assert item.requires_review
    assert item.suggested_translation == "Миттельланд"


def test_type_requiring_review_wins_over_more_confident_chapter() -> None:
    """Тип, требующий решения всегда, побеждает тип уверенной главы."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                analysis_id="analysis-1",
                source_text="Golden Krone",
                type=ConceptType.TERM,
                suggested_translation="Золотая корона",
                confidence=0.99,
            )
        ),
        chapter_two(
            make_analysis_candidate(
                analysis_id="analysis-2",
                source_text="Golden Krone",
                type=ConceptType.ORGANIZATION,
                suggested_translation="Золотая корона",
                confidence=0.5,
            )
        ),
    )

    item = book.items[0]
    assert item.type is ConceptType.ORGANIZATION
    assert item.requires_review


def test_review_shows_pending_items_first() -> None:
    """Сначала показывается то, что ждёт решения, затем решённое и подсказки."""
    book = review(
        chapter_one(
            make_analysis_candidate(
                source_text="entailed estate",
                type=ConceptType.TERM,
                suggested_translation="заповедное имение",
                confidence=0.95,
            ),
            make_analysis_candidate(source_text="Mr. Bennet"),
            make_analysis_candidate(
                source_text="Netherfield",
                type=ConceptType.PLACE,
                suggested_translation="Незерфилд",
            ),
        ),
        decisions=(decision("Netherfield", text="Незерфилд"),),
    )

    assert tuple(item.source_text for item in book.items) == (
        "Mr. Bennet",
        "Netherfield",
        "entailed estate",
    )


def test_decision_completes_review_of_item() -> None:
    """Принятое решение снимает элемент с проверки, не меняя его признака внимания."""
    book = review(
        chapter_one(make_analysis_candidate(source_text="Mr. Bennet")),
        decisions=(decision("Mr. Bennet"),),
    )

    item = book.items[0]
    assert item.requires_review
    assert item.is_decided
    assert item.awaits_decision is False
    assert book.is_complete
    book.ensure_complete()


def test_pending_items_block_processing() -> None:
    """Необработанные требующие внимания элементы не дают запустить обработку."""
    book = review(chapter_one(make_analysis_candidate(source_text="Mr. Bennet")))

    with pytest.raises(GlossaryReviewIncompleteError, match="Mr. Bennet"):
        book.ensure_complete()


def test_decision_without_candidates_is_not_shown() -> None:
    """Решение по тексту, которого в книге больше нет, в проверку не входит."""
    book = review(chapter_one(), decisions=(decision("Mr. Bennet"),))

    assert book.items == ()


def test_snapshot_carries_status_of_each_decision() -> None:
    """Снимок передаёт модели зафиксированные решения как обязательные, прочие — как подсказки."""
    book = review(
        chapter_one(
            make_analysis_candidate(source_text="Mr. Bennet"),
            make_analysis_candidate(
                source_text="entailed estate",
                type=ConceptType.TERM,
                suggested_translation="заповедное имение",
                confidence=0.95,
            ),
        ),
        decisions=(
            decision("Mr. Bennet", text="мистер Беннетт", status=TermStatus.LOCKED),
        ),
    )

    snapshot = glossary_snapshot(book, language=TARGET_LANGUAGE)

    entries = {entry.source_text: entry for entry in snapshot.entries}
    assert entries["Mr. Bennet"].translation == "мистер Беннетт"
    assert entries["Mr. Bennet"].status is TermStatus.LOCKED
    assert entries["entailed estate"].status is TermStatus.SUGGESTED
    assert tuple(entry.source_text for entry in snapshot.mandatory_entries) == (
        "Mr. Bennet",
    )


def test_snapshot_skips_decision_without_target_language_variant() -> None:
    """Решение без варианта на целевом языке модели ничего не навязывает."""
    book = review(
        chapter_one(make_analysis_candidate(source_text="Mr. Bennet")),
        decisions=(decision("Mr. Bennet", language="de", text="Herr Bennet"),),
    )

    assert glossary_snapshot(book, language=TARGET_LANGUAGE).entries == ()


def chapter_three(*candidates: GlossaryCandidate) -> ChapterAnalysis:
    """Анализ третьей главы книги с её кандидатами глоссария."""
    return make_chapter_analysis(
        id="analysis-3", section_id="section-3", candidates=candidates
    )


def wonka(spelling: str, *, analysis_id: str, confidence: float = 0.9) -> GlossaryCandidate:
    """Упоминание мистера Вонки в одной главе под заданным написанием."""
    return make_analysis_candidate(
        source_text=spelling,
        suggested_translation="мистер Вонка",
        analysis_id=analysis_id,
        confidence=confidence,
    )


def test_obvious_spellings_of_one_name_are_one_item() -> None:
    """Написания с точкой и без, в разном регистре — один элемент проверки (QA-11).

    Элемент показан самым частым написанием, остальные написания видны в нём.
    """
    book = review(
        chapter_one(wonka("Mr Wonka", analysis_id="analysis-1", confidence=0.99)),
        chapter_two(
            wonka("Mr. Wonka", analysis_id="analysis-2"),
            make_analysis_candidate(
                source_text="Wonka's factory",
                type=ConceptType.PLACE,
                suggested_translation="фабрика Вонки",
                analysis_id="analysis-2",
                confidence=0.95,
            ),
        ),
        chapter_three(
            wonka("Mr. Wonka", analysis_id="analysis-3"),
            make_analysis_candidate(
                source_text="WONKA’S FACTORY",
                type=ConceptType.PLACE,
                suggested_translation="Фабрика Вонки",
                analysis_id="analysis-3",
            ),
        ),
    )

    assert [item.source_text for item in book.items] == ["Mr. Wonka", "Wonka's factory"]
    person = book.items[0]
    assert person.spellings == ("Mr. Wonka", "Mr Wonka")
    assert person.other_spellings == ("Mr Wonka",)
    assert book.item_for("MR WONKA") is person
    assert len(book.awaiting_decision) == 2
    assert not book.items[1].has_translation_conflict


def test_decision_made_under_another_spelling_is_found() -> None:
    """Решение, принятое по одному написанию, действует для всего элемента."""
    book = review(
        chapter_one(wonka("Mr Wonka", analysis_id="analysis-1")),
        chapter_two(wonka("Mr. Wonka", analysis_id="analysis-2")),
        chapter_three(wonka("Mr. Wonka", analysis_id="analysis-3")),
        decisions=(decision("Mr Wonka", text="мистер Вонка"),),
    )

    (item,) = book.items
    assert item.decision is not None
    assert item.decision.canonical_source_text == "Mr Wonka"
    assert book.is_complete
    assert not item.has_decision_conflict


def test_diverging_decisions_of_spellings_are_not_merged_silently() -> None:
    """Разные прежние решения по написаниям одного элемента помечаются конфликтом.

    Действует зафиксированное решение, прочие видны как вытесненные и в снимок
    запуска не попадают: снимок получает ровно одно решение на элемент.
    """
    book = review(
        chapter_one(wonka("Mr Wonka", analysis_id="analysis-1")),
        chapter_two(wonka("Mr. Wonka", analysis_id="analysis-2")),
        decisions=(
            decision("Mr. Wonka", text="мистер Вонка", concept_id="concept-1"),
            decision(
                "Mr Wonka",
                text="господин Вонка",
                status=TermStatus.LOCKED,
                concept_id="concept-2",
            ),
        ),
    )

    (item,) = book.items
    assert item.decision is not None
    assert item.decision.id == "concept-2"
    assert item.has_decision_conflict
    assert [concept.id for concept in item.superseded_decisions] == ["concept-1"]
    snapshot = glossary_snapshot(book, language=TARGET_LANGUAGE)
    assert [(entry.source_text, entry.translation) for entry in snapshot.entries] == [
        ("Mr Wonka", "господин Вонка")
    ]


def test_matching_decisions_of_spellings_are_not_a_conflict() -> None:
    """Одинаковые решения по разным написаниям конфликтом не считаются."""
    book = review(
        chapter_one(wonka("Mr Wonka", analysis_id="analysis-1")),
        chapter_two(wonka("Mr. Wonka", analysis_id="analysis-2")),
        chapter_three(wonka("Mr. Wonka", analysis_id="analysis-3")),
        decisions=(
            decision("Mr Wonka", text="мистер Вонка", concept_id="concept-1"),
            decision("Mr. Wonka", text="мистер Вонка", concept_id="concept-2"),
        ),
    )

    (item,) = book.items
    assert not item.has_decision_conflict
    assert item.decision is not None
    assert item.decision.canonical_source_text == "Mr. Wonka"
