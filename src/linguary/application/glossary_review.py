"""Глоссарий книги: сборка из кандидатов глав и отбор требующих внимания (PRD § 11-12).

Кандидаты приходят из анализа отдельных глав, поэтому один исходный текст встречается
в книге столько раз, сколько глав его упомянуло, — с разным типом, уверенностью и
предложенным переводом. Модуль собирает из этих упоминаний по одному элементу проверки
на исходный текст, пересчитывает признак `requires_review` доменным правилом (DTO-01)
и присоединяет к элементу уже принятое пользователем решение.

Решением пользователя владеет `TermConcept`: понятие существует ровно у тех исходных
текстов, по которым пользователь высказался. Элемент, требующий внимания и не получивший
понятия, остаётся необработанным и не даёт запустить обработку (PRD § 12). Элемент,
внимания не требующий, понятия не заводит и живёт в глоссарии необязательной подсказкой.

Условия отбора здесь не решаются заново: их держит домен (`compute_requires_review`).
Модуль отвечает за то, из чего эти условия считаются на уровне книги, и за порядок показа.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from linguary.application.errors import ApplicationError
from linguary.domain import (
    ChapterAnalysis,
    ConceptType,
    GlossaryCandidate,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    TermConcept,
    TermStatus,
    compute_requires_review,
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


@dataclass(frozen=True, slots=True)
class GlossaryReviewItem:
    """Один исходный текст глоссария книги вместе с решением пользователя.

    Поля до `requires_review` описывают то, что о тексте сказали главы; `decision` —
    то, что о нём сказал пользователь.
    """

    source_text: str
    type: ConceptType
    suggested_translation: str
    confidence: float
    model_flagged_review: bool
    has_translation_conflict: bool
    requires_review: bool
    decision: TermConcept | None = None

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
    items: tuple[GlossaryReviewItem, ...] = ()

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
        """Элемент проверки по исходному тексту либо `None`, если такого в книге нет."""
        return next(
            (item for item in self.items if item.source_text == source_text), None
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
) -> GlossaryReview:
    """Собирает глоссарий книги из кандидатов всех проанализированных глав (PRD § 11-12).

    Решение, исходного текста которого больше нет ни в одной главе, в проверку не входит:
    глоссарий книги — это то, что нашли в её тексте, а не архив прежних решений.
    """
    grouped: dict[str, list[GlossaryCandidate]] = {}
    for analysis in analyses:
        for candidate in analysis.glossary_candidates:
            grouped.setdefault(candidate.source_text, []).append(candidate)
    decided = {concept.canonical_source_text: concept for concept in decisions}
    items = (
        _review_item(
            group,
            decision=decided.get(source_text),
            confidence_threshold=confidence_threshold,
        )
        for source_text, group in grouped.items()
    )
    return GlossaryReview(book_id=book_id, items=tuple(sorted(items, key=_display_order)))


def glossary_snapshot(review: GlossaryReview, *, language: str) -> GlossarySnapshot:
    """Снимок глоссария, с которым выполняется запуск обработки (PRD § 11, § 13).

    Решение уходит модели своим статусом: `locked` — обязательным переводческим указанием,
    `approved` и `suggested` — необязательной подсказкой. Элемент без решения попадает
    в снимок подсказкой с вариантом модели: внимания он не требовал.
    """
    return GlossarySnapshot(
        entries=tuple(
            entry
            for item in review.items
            if (entry := _snapshot_entry(item, language=language)) is not None
        )
    )


def _review_item(
    group: Sequence[GlossaryCandidate],
    *,
    decision: TermConcept | None,
    confidence_threshold: float,
) -> GlossaryReviewItem:
    """Элемент проверки из упоминаний одного исходного текста в разных главах.

    Вариант перевода берётся у самой уверенной главы, а сигналы проверки объединяются:
    пометка модели хотя бы от одной главы, тип, требующий решения всегда, и расхождение
    вариантов перевода делают элемент требующим внимания. Правило повторяет слияние
    фрагментов одной главы, но работает на другом уровне: там объединяются ответы модели
    об одной главе, здесь — сохранённые кандидаты разных глав.
    """
    leading = max(group, key=lambda candidate: candidate.confidence)
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
    return GlossaryReviewItem(
        source_text=leading.source_text,
        type=concept_type,
        suggested_translation=leading.suggested_translation,
        confidence=leading.confidence,
        model_flagged_review=model_flagged,
        has_translation_conflict=conflict,
        requires_review=compute_requires_review(
            type=concept_type,
            confidence=leading.confidence,
            model_flagged_review=model_flagged,
            confidence_threshold=confidence_threshold,
            has_translation_conflict=conflict,
        ),
        decision=decision,
    )


def _display_order(item: GlossaryReviewItem) -> tuple[bool, bool, int, str]:
    """Порядок показа проверки (PRD § 12).

    Сначала то, что ждёт решения, затем остальные требующие внимания элементы, затем
    необязательные подсказки; внутри группы — по типу понятия и исходному тексту.
    """
    return (
        not item.awaits_decision,
        not item.requires_review,
        REVIEW_TYPE_ORDER.index(item.type),
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
        )
    variant = item.decision.variant_for(language)
    if variant is None:
        return None
    return GlossarySnapshotEntry(
        source_text=item.decision.canonical_source_text,
        translation=variant.text,
        type=item.decision.type,
        status=item.decision.status,
    )


__all__ = [
    "REVIEW_TYPE_ORDER",
    "GlossaryReview",
    "GlossaryReviewError",
    "GlossaryReviewIncompleteError",
    "GlossaryReviewItem",
    "build_glossary_review",
    "glossary_snapshot",
]
