"""Глоссарий книги: сборка из кандидатов глав и отбор требующих внимания (PRD § 11-12).

Кандидаты приходят из анализа отдельных глав, поэтому один исходный текст встречается
в книге столько раз, сколько глав его упомянуло, — с разным типом, уверенностью и
предложенным переводом. Модуль собирает из этих упоминаний по одному элементу проверки
на исходный текст, пересчитывает признак `requires_review` доменным правилом (DTO-01)
и присоединяет к элементу уже принятое пользователем решение.

Исходные тексты сравниваются ключом `glossary_key`: очевидные варианты написания
(`Mr Wonka` и `Mr. Wonka`, `WONKA'S FACTORY` и `Wonka's factory`) — один элемент,
показанный самым частым написанием, а остальные написания видны в самом элементе.
Решения ищутся тем же ключом. Если по разным написаниям одного элемента раньше были
приняты разные решения, они молча не сливаются: действует зафиксированное, а элемент
помечается конфликтом решений, пока пользователь не примет по нему решение заново.

Решением пользователя владеет `TermConcept`: понятие существует ровно у тех исходных
текстов, по которым пользователь высказался. Элемент, требующий внимания и не получивший
понятия, остаётся необработанным и не даёт запустить обработку (PRD § 12). Элемент,
внимания не требующий, понятия не заводит и живёт в глоссарии необязательной подсказкой.

Условия отбора здесь не решаются заново: их держит домен (`compute_requires_review`).
Модуль отвечает за то, из чего эти условия считаются на уровне книги, и за порядок показа.
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

#: Порядок типов при показе проверки (PRD § 12, § 21.3). Действующие лица идут первыми:
#: их переводческие решения отражаются на всей книге, и очевидные термины не должны
#: заслонять решения, ради которых проверка и существует.
REVIEW_TYPE_ORDER: tuple[ConceptType, ...] = (
    ConceptType.PERSON,
    ConceptType.PLACE,
    ConceptType.ORGANIZATION,
    ConceptType.FICTIONAL_CONCEPT,
    ConceptType.TERM,
)


class GlossaryReviewError(ApplicationError):
    """Глоссарий книги нельзя прочитать или изменить в текущем состоянии."""


class GlossaryReviewIncompleteError(GlossaryReviewError):
    """Среди требующих внимания элементов остались необработанные (PRD § 12)."""


class GlossaryNotFoundError(GlossaryReviewError):
    """Книга или элемент её глоссария не найден."""


@dataclass(frozen=True, slots=True)
class GlossaryReviewItem:
    """Один исходный текст глоссария книги вместе с решением пользователя.

    Поля до `requires_review` описывают то, что о тексте сказали главы; `decision` —
    то, что о нём сказал пользователь.

    `suggested_note` — переводческое указание, предложенное главами; решение
    пользователя несёт своё указание в `decision.note`.

    `spellings` — все написания элемента в книге, первым идёт показанное
    `source_text`. `superseded_decisions` — прежние решения по другим написаниям того
    же элемента: действует `decision`, а они удаляются при следующем решении
    пользователя.
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
        """Модель просила проверить необязательную подсказку."""
        return self.model_flagged_review and not self.requires_review

    @property
    def other_spellings(self) -> tuple[str, ...]:
        """Написания элемента, кроме показанного."""
        return tuple(
            spelling for spelling in self.spellings if spelling != self.source_text
        )

    @property
    def concept_type(self) -> ConceptType:
        """Тип понятия с учётом решения: пользователь вправе исправить тип модели."""
        return self.decision.type if self.decision is not None else self.type

    @property
    def has_decision_conflict(self) -> bool:
        """Расходятся ли решения, принятые раньше по разным написаниям элемента."""
        if self.decision is None:
            return False
        chosen = _decision_signature(self.decision)
        return any(
            _decision_signature(other) != chosen for other in self.superseded_decisions
        )

    @property
    def note(self) -> str:
        """Переводческое указание, действующее для этого элемента.

        Решение пользователя старше предложения глав: пустое указание в решении
        означает, что указания не нужно, а не что нужно вернуться к предложенному.
        """
        if self.decision is not None:
            return self.decision.note
        return self.suggested_note

    @property
    def is_decided(self) -> bool:
        """Высказался ли пользователь по этому элементу."""
        return self.decision is not None

    @property
    def awaits_decision(self) -> bool:
        """Требует ли элемент решения, которого пользователь ещё не принял."""
        return self.requires_review and self.decision is None


@dataclass(frozen=True, slots=True)
class GlossaryReview:
    """Глоссарий книги в том виде, в каком его проверяет пользователь."""

    book_id: str
    book_status: BookStatus | None = None
    items: tuple[GlossaryReviewItem, ...] = ()
    is_editable: bool = True

    @property
    def requiring_review(self) -> tuple[GlossaryReviewItem, ...]:
        """Элементы, по которым пользователь обязан принять решение (PRD § 12)."""
        return tuple(item for item in self.items if item.requires_review)

    @property
    def awaiting_decision(self) -> tuple[GlossaryReviewItem, ...]:
        """Требующие внимания элементы, решения по которым ещё нет."""
        return tuple(item for item in self.items if item.awaits_decision)

    @property
    def is_complete(self) -> bool:
        """Завершена ли проверка глоссария."""
        return not self.awaiting_decision

    def item_for(self, source_text: str) -> GlossaryReviewItem | None:
        """Элемент проверки по любому его написанию либо `None`, если такого в книге нет."""
        key = glossary_key(source_text)
        return next(
            (item for item in self.items if glossary_key(item.source_text) == key), None
        )

    def item_for_concept(self, concept_id: str) -> GlossaryReviewItem | None:
        """Элемент проверки по внешнему идентификатору понятия."""
        return next(
            (item for item in self.items if item.concept_id == concept_id), None
        )

    def ensure_complete(self) -> None:
        """Проверяет, что необработанных требующих внимания элементов не осталось."""
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
    """Собирает глоссарий книги из кандидатов всех проанализированных глав (PRD § 11-12).

    Решение, исходного текста которого больше нет ни в одной главе, в проверку не входит:
    глоссарий книги — это то, что нашли в её тексте, а не архив прежних решений.
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
    """Снимок глоссария, с которым выполняется запуск обработки (PRD § 11, § 13).

    Решение уходит модели своим статусом: решение пользователя — `approved` (принято
    или исправлено) и `locked` — обязательным переводческим указанием, пропущенный
    элемент (`suggested`) — необязательной подсказкой. Элемент без решения попадает
    в снимок подсказкой с вариантом модели: внимания он не требовал.
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
    """Кандидаты предыдущих глав, в которых решение пользователя заменило вариант модели.

    Анализ следующей главы повторяет накопленные решения книги дословно, поэтому
    ему нужен вариант пользователя: иначе глава получает и повторяет в своих указаниях
    вариант модели, который пользователь уже исправил (PRD § 11). Решение сопоставляется
    с кандидатом ключом `glossary_key` и заменяет вариант перевода, тип и указание;
    пропущенный элемент решением не считается, и у него остаётся вариант модели.
    Кандидаты с решением идут первыми: ограничение контекста их не отрезает.

    `decided_only` — кандидаты, которые нужны только вместе с решением пользователя:
    при повторном анализе это кандидаты самой главы и следующих глав. Решение
    пользователя действует на всю книгу, а вариант модели из следующей главы этой
    главе ничего не даёт, поэтому кандидат без решения отсюда не попадает.

    Указание каждого кандидата очищается от фрагментов, называющих вариант модели,
    который пользователь отверг по другому понятию: «пост при въезде в Даскмир» при
    решении «Даскмер». Отвергнутые варианты известны из всех переданных кандидатов
    книги — тем же правилом, что и в контексте обработки (`note_respecting_glossary`).
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
    """Элемент проверки из упоминаний одного исходного текста в разных главах.

    Вариант перевода берётся у самой уверенной главы, а сигналы проверки объединяются:
    пометка модели хотя бы от одной главы, тип и расхождение вариантов перевода
    учитываются вместе с частотой упоминаний. Здесь объединяются сохранённые
    кандидаты разных глав.
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


#: Порядок, в котором выбирается действующее решение среди решений по разным
#: написаниям одного элемента: зафиксированное пользователем сильнее принятого,
#: принятое — пропущенного.
_DECISION_PRIORITY: dict[TermStatus, int] = {
    TermStatus.LOCKED: 0,
    TermStatus.APPROVED: 1,
    TermStatus.SUGGESTED: 2,
}


def _chosen_decision(
    decisions: Sequence[TermConcept], *, shown: str
) -> tuple[TermConcept | None, tuple[TermConcept, ...]]:
    """Действующее решение элемента и прежние решения по другим его написаниям.

    Решение по каждому элементу одно; несколько решений у одного ключа остаются
    от глоссария, в котором написания ещё не объединялись. Времени принятия решение
    не хранит, поэтому «последнее» определить нельзя: побеждает самый сильный статус,
    затем решение по показанному написанию, затем первое сохранённое. Расхождение
    с остальными решениями не прячется — его показывает `has_decision_conflict`.
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
    """То, чем решения различаются для пользователя: статус, тип, указание, перевод."""
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
    """Порядок показа проверки (PRD § 12).

    Сначала то, что ждёт решения, затем остальные требующие внимания элементы, затем
    необязательные подсказки; внутри группы — по типу понятия и исходному тексту.
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
    """Запись снимка для одного элемента либо `None`, если навязывать модели нечего."""
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
