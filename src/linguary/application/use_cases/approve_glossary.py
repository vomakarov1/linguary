"""Проверка глоссария пользователем и старт обработки книги (PRD § 11-12, § 21.3).

`ApproveGlossary` показывает глоссарий книги и ведёт четыре действия пользователя по
элементу, требующему внимания: принять предложенный вариант, исправить его, зафиксировать
как обязательный и пропустить элемент как неважный для единообразия перевода. Каждое
действие сохраняет `TermConcept` — принятое решение; различаются они статусом понятия
и его варианта.

`StartProcessing` завершает проверку: он отказывает, пока среди требующих внимания
остались необработанные элементы (PRD § 12), а иначе замораживает решения снимком
в `ProcessingRun`. С этого момента глоссарий книги доступен только для чтения — вернуть
его в работу можно лишь возвратом к проверке глоссария (P3.T7), который удаляет
результаты обработки.

Сама обработка блоков в этот сценарий не входит: здесь открывается запуск с его снимками,
а `TranslationUnit` обрабатывает P3.T6.
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
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import (
    BookEvent,
    BookStatus,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    TermConcept,
    TermStatus,
    TermVariant,
    apply_book_event,
)

#: Фабрика идентификаторов сохраняемых записей глоссария и запусков обработки.
type IdFactory = Callable[[], str]

#: Состояния книги, в которых решения глоссария ещё редактируются (PRD § 11).
EDITABLE_STATUSES: frozenset[BookStatus] = frozenset(
    {BookStatus.GLOSSARY_REVIEW, BookStatus.READY_TO_PROCESS}
)


class GlossaryDecision(StrEnum):
    """Действие пользователя по элементу, требующему внимания (PRD § 12)."""

    ACCEPT = "accept"
    CORRECT = "correct"
    LOCK = "lock"
    SKIP = "skip"


#: Статус понятия и его варианта после каждого действия. `skip` оставляет элемент
#: необязательной подсказкой: пользователь решил, что единообразия он не требует,
#: и найденный моделью вариант остаётся в глоссарии без обязательной силы (PRD § 12).
DECISION_STATUSES: dict[GlossaryDecision, TermStatus] = {
    GlossaryDecision.ACCEPT: TermStatus.APPROVED,
    GlossaryDecision.CORRECT: TermStatus.APPROVED,
    GlossaryDecision.LOCK: TermStatus.LOCKED,
    GlossaryDecision.SKIP: TermStatus.SUGGESTED,
}


@dataclass(frozen=True, slots=True)
class GlossaryDecisionCommand:
    """Решение пользователя по одному элементу глоссария книги.

    `translation` несёт только исправление: остальные действия работают с уже известным
    вариантом — предложенным моделью или сохранённым прежним решением. `concept_id`
    связывает HTTP-команду с показанным элементом и сохраняет его идентификатор при
    первом решении; прямые вызовы сценария могут по-прежнему работать по исходному тексту.

    `note` — переводческое указание по элементу. `None` означает «оставить действующее»,
    а переданная строка заменяет его целиком, в том числе пустая: снять указание
    пользователь вправе так же, как и задать (PRD § 11).
    """

    book_id: str
    source_text: str
    decision: GlossaryDecision
    translation: str | None = None
    concept_id: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class GlossaryNoteCommand:
    """Правка переводческого указания по одному элементу глоссария (PRD § 11)."""

    book_id: str
    concept_id: str
    note: str


@dataclass(frozen=True, slots=True)
class StartProcessingCommand:
    """Команда старта обработки книги."""

    book_id: str


class ApproveGlossary:
    """Показывает глоссарий книги и сохраняет решения пользователя (PRD § 12)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        glossary_confidence_threshold: float,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._id_factory = id_factory or _new_id

    def review(self, book_id: str) -> GlossaryReview:
        """Глоссарий книги вместе с уже принятыми решениями."""
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, book_id)
            return replace(
                _load_review(
                    transaction,
                    book_id,
                    confidence_threshold=self._glossary_confidence_threshold,
                ),
                book_status=book.status,
                is_editable=book.status in EDITABLE_STATUSES,
            )

    def __call__(self, command: GlossaryDecisionCommand) -> TermConcept:
        """Сохраняет решение пользователя по одному элементу глоссария."""
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            _ensure_editable(book)
            review = _load_review(
                transaction,
                command.book_id,
                confidence_threshold=self._glossary_confidence_threshold,
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
            _complete_review_if_ready(
                transaction,
                book,
                confidence_threshold=self._glossary_confidence_threshold,
            )
        return concept

    def accept_all_awaiting(self, book_id: str) -> tuple[TermConcept, ...]:
        """Принимает предложенный вариант по всем элементам, ждущим решения (PRD § 12).

        Массовое действие идёт одной транзакцией и по одному собранному глоссарию:
        поэлементный вызов пересобирал бы глоссарий всей книги на каждый элемент.
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, book_id)
            _ensure_editable(book)
            review = _load_review(
                transaction,
                book_id,
                confidence_threshold=self._glossary_confidence_threshold,
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
            )
        return concepts

    def set_note(self, command: GlossaryNoteCommand) -> TermConcept:
        """Меняет переводческое указание по элементу, не трогая само решение.

        Указание правится отдельно от решения, потому что живёт дольше него: род
        персонажа виден только после нескольких глав, а перевод имени к этому моменту
        уже принят. Через обычное решение это не сделать — повторное `accept` вернуло бы
        вариант, предложенный моделью, и стёрло исправление пользователя.
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            _ensure_editable(book)
            review = _load_review(
                transaction,
                command.book_id,
                confidence_threshold=self._glossary_confidence_threshold,
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
        """Понятие глоссария с новым указанием.

        Решения по элементу может ещё не быть: указание к предложенному моделью
        варианту — это уже решение пользователя, поэтому элемент становится принятым
        с тем же вариантом, который был предложен.
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
        """Завершает проверку глоссария, если обязательных решений больше нет."""
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, book_id)
            _ensure_editable(book)
            _complete_review_if_ready(
                transaction,
                book,
                confidence_threshold=self._glossary_confidence_threshold,
            )

    def _decide(
        self,
        item: GlossaryReviewItem,
        command: GlossaryDecisionCommand,
        *,
        book: BookRecord,
    ) -> TermConcept:
        """Понятие глоссария, каким его делает решение пользователя.

        Повторное решение по тому же элементу переписывает прежнее понятие целиком:
        идентификатор сохраняется, статус и вариант перевода заменяются.
        """
        status = DECISION_STATUSES[command.decision]
        concept_id = item.decision.id if item.decision is not None else (
            command.concept_id or self._id_factory()
        )
        return TermConcept(
            id=concept_id,
            book_id=book.id,
            type=item.type,
            canonical_source_text=item.source_text,
            status=status,
            note=item.note if command.note is None else command.note.strip(),
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
    """Замораживает решения глоссария и открывает запуск обработки книги (PRD § 11-12)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        glossary_confidence_threshold: float,
        llm_settings_snapshot: LLMSettingsSnapshot,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._glossary_confidence_threshold = glossary_confidence_threshold
        self._llm_settings_snapshot = llm_settings_snapshot
        self._id_factory = id_factory or _new_id

    def __call__(self, command: StartProcessingCommand) -> ProcessingRun:
        """Создаёт активный запуск обработки со снимками глоссария и настроек LLM.

        Отказ приходит из трёх мест: непринятые решения по требующим внимания элементам
        (PRD § 12), уже идущий запуск книги (ADR § 3) и состояние книги, из которого
        обработка не запускается (ALG-06).
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            if transaction.runs.active_for_book(book.id) is not None:
                raise GlossaryReviewError(f"у книги {book.id} уже идёт обработка")
            review = _load_review(
                transaction,
                book.id,
                confidence_threshold=self._glossary_confidence_threshold,
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
                llm_settings_snapshot=self._llm_settings_snapshot,
            )
            transaction.runs.add(run)
            transaction.books.save(replace(book, status=started))
        return run


def _load_review(
    transaction: UnitOfWork, book_id: str, *, confidence_threshold: float
) -> GlossaryReview:
    """Глоссарий книги, собранный в открытой транзакции."""
    return build_glossary_review(
        book_id=book_id,
        analyses=transaction.analyses.list_for_book(book_id),
        confidence_threshold=confidence_threshold,
        decisions=transaction.glossary.list_for_book(book_id),
    )


def _book(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Запись книги в открытой транзакции."""
    book = transaction.books.get(book_id)
    if book is None:
        raise GlossaryNotFoundError(f"книги {book_id} нет в хранилище")
    return book


def _complete_review_if_ready(
    transaction: UnitOfWork,
    book: BookRecord,
    *,
    confidence_threshold: float,
) -> None:
    """Переводит книгу к запуску обработки после закрытия всех обязательных решений."""
    if book.status is not BookStatus.GLOSSARY_REVIEW:
        return
    review = _load_review(
        transaction,
        book.id,
        confidence_threshold=confidence_threshold,
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
    """Проверяет, что глоссарий книги ещё редактируется (PRD § 11, § 21.3).

    После старта обработки решения заморожены снимком запуска, и экран проверки доступен
    только для чтения: вернуть глоссарий в работу можно возвратом к проверке глоссария,
    который удаляет результаты обработки.
    """
    if book.status not in EDITABLE_STATUSES:
        raise GlossaryReviewError(
            f"глоссарий книги {book.id} в состоянии {book.status} доступен только "
            f"для чтения: чтобы изменить решения, вернитесь к проверке глоссария"
        )


def _translation(
    item: GlossaryReviewItem, command: GlossaryDecisionCommand, *, language: str
) -> str:
    """Текст варианта перевода после решения пользователя.

    `correct` берёт присланный текст, `accept` — предложенный моделью, а `lock` и `skip`
    сохраняют текст прежнего решения, если оно было: фиксация и пропуск меняют статус
    элемента, а не его перевод.
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
    """Состояние книги после старта обработки (ALG-06).

    Завершение проверки глоссария и старт запуска — две строки таблицы переходов, а для
    пользователя это одно действие. Книга, из состояния которой обработка не запускается,
    получает отказ домена (в Web UI — 409).
    """
    if status is BookStatus.GLOSSARY_REVIEW:
        status = apply_book_event(status, BookEvent.GLOSSARY_REVIEW_COMPLETED)
    return apply_book_event(status, BookEvent.PROCESSING_RUN_STARTED)


def _new_id() -> str:
    """Идентификатор нового понятия глоссария или запуска обработки."""
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
