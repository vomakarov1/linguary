"""Общие шаги обработки блоков книги: вызов модели и запись результата (ALG-04).

Обработка главы группами блоков (`use_cases/process_chapter.py`) и повторная генерация
одного блока (`use_cases/regenerate_block.py`) различаются составом контекста и тем, что
считают единицей работы. Всё остальное у них общее и живёт здесь: цикл вызова модели
с повторами по политике, блокирующие проверки ответа, короткая транзакция сохранения
с проверкой активного запуска, нумерация учебных заметок и завершение обработки книги.

Транзакция намеренно не охватывает вызов модели (ADR § 3): пока ответ в пути, пользователь
успевает вернуться к проверке глоссария, поэтому сохранение сверяет активный запуск книги
с тем, с которым работа начиналась, и при расхождении не пишет ничего.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, replace

from linguary.application.context_builder import TranslationContext
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import ExplainedExpressions
from linguary.application.llm_rejections import record_rejection
from linguary.application.ports.llm import LLMClient
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BlockProcessingResult,
    BookEvent,
    BookStatus,
    CanonicalDocument,
    DocumentSection,
    TaskStatus,
    TaskType,
    TokenUsage,
    TranslationUnit,
    apply_book_event,
    build_language_note_id,
)

#: Пауза между попытками вызова модели: политика повторов её вычисляет, выдерживает
#: сценарий — прикладной слой циклом событий не владеет (ALG-05).
type Sleep = Callable[[float], Awaitable[None]]


class BlockProcessingError(ApplicationError):
    """Блоки книги нельзя обработать в её текущем состоянии."""


@dataclass(frozen=True, slots=True)
class TranslationCallResult:
    """Проверенный результат одного вызова вместе с его ценой.

    `attempts` считает обращения к модели, а не запуски задачи: одна группа блоков может
    стоить нескольких попыток, а провайдер считает расход по каждой из них (PRD § 24).
    """

    results: tuple[BlockProcessingResult, ...]
    attempts: int = 0
    usage: TokenUsage | None = None


class TranslationCall:
    """Вызов модели по готовому контексту и атомарная запись результата (ALG-04)."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        llm_client: LLMClient,
        retry_policy: RetryPolicy,
        validator: TranslationResultValidator,
        sleep: Sleep,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._llm_client = llm_client
        self._retry_policy = retry_policy
        self._validator = validator
        self._sleep = sleep

    async def request(
        self,
        context: TranslationContext,
        *,
        unit: TranslationUnit,
        target_language: str,
        note_ids: Callable[[], str],
    ) -> TranslationCallResult:
        """Вызывает модель и проверяет ответ, повторяя по политике (ALG-04, ALG-05).

        Проверки идут внутри цикла: отбраковка ответа — обычный повод для повтора
        с уточняющей инструкцией, а не отказ задачи. Исчерпав попытки, вызов поднимает
        исходную ошибку: типизировать её и перевести задачу в `failed` — работа runner-а.

        Из проверенного результата отсеиваются подсказки и заметки, которые повторяют
        `Before You Read` главы (PRD § 19.3): повтор не повод отбраковывать ответ.
        """
        request = context.request
        explained = ExplainedExpressions(context.explained_expressions)
        attempts = 0
        while True:
            attempts += 1
            response = None
            try:
                response = await self._llm_client.generate(request)
                results = self._validator.validate(
                    context.to_block_ids(response.payload),
                    unit=unit,
                    target_language=target_language,
                    note_id_factory=note_ids,
                    truncated=response.truncated,
                    expectation=context.expectation,
                )
            except Exception as error:
                if response is not None:
                    record_rejection(self._unit_of_work, response, error)
                decision = self._retry_policy.decide(
                    _worded_for_the_model(error, context), attempts=attempts
                )
                if not decision.should_retry:
                    raise
                if decision.backoff_seconds > 0:
                    await self._sleep(decision.backoff_seconds)
                request = decision.next_request(request)
                continue
            return TranslationCallResult(
                results=tuple(explained.without_repeats(result) for result in results),
                attempts=attempts,
                usage=response.usage,
            )

    def save(
        self, *, book_id: str, run_id: str, results: Sequence[BlockProcessingResult]
    ) -> bool:
        """Атомарно заменяет результаты перечисленных блоков (ALG-04, PRD § 17).

        Возвращает `False`, если сохранять уже некуда: активного запуска у книги нет
        или это другой запуск. Так отбрасывается поздний результат вызова, начатого
        до возврата к проверке глоссария.
        """
        with self._unit_of_work.transaction() as transaction:
            run = transaction.runs.active_for_book(book_id)
            if run is None or run.id != run_id:
                return False
            transaction.results.replace_results(run_id, results)
        return True


def note_id_factory(*, chapter_number: int, used: Collection[str]) -> Callable[[], str]:
    """Фабрика идентификаторов учебных заметок главы вида `ln-ch03-001` (PRD § 19.3).

    Идентификаторы уникальны в пределах запуска, поэтому нумерация продолжается за уже
    сохранёнными заметками этой главы: ни повтор частично обработанной главы, ни повторная
    генерация отдельного блока не должны столкнуться с чужими заметками. Пропуски
    в нумерации допустимы — идентификатор обязан быть уникальным, а не подряд идущим.
    """
    ordinal = 0

    def next_id() -> str:
        nonlocal ordinal
        while True:
            ordinal += 1
            note_id = build_language_note_id(
                chapter_number=chapter_number, ordinal=ordinal
            )
            if note_id not in used:
                return note_id

    return next_id


def chapter_number(document: CanonicalDocument, section: DocumentSection) -> int:
    """Номер главы в порядке чтения: он входит в идентификаторы заметок (PRD § 19.3)."""
    sections = document.ordered_sections
    return next(
        (
            number
            for number, other in enumerate(sections, start=1)
            if other.id == section.id
        ),
        section.order + 1,
    )


def complete_processing(
    unit_of_work: UnitOfWork,
    *,
    book_id: str,
    finished_task_id: str | None,
    task_types: Collection[TaskType],
) -> None:
    """Переводит книгу в `READY_TO_EXPORT`, когда её работа закончена (ALG-06).

    Задача, которая выполняется прямо сейчас, ещё числится `running`: runner фиксирует
    её итог уже после обработчика. Поэтому её успех записывается здесь же, в одной
    транзакции с проверкой соседей: иначе при параллельной обработке (PRD § 25) две
    главы, закончившие почти одновременно, видят друг друга ещё выполняющимися, и книга
    навсегда остаётся в `PROCESSING`. Повторная запись того же итога runner-ом ничего
    не меняет.
    Книга, ушедшая из `PROCESSING`, состояние не меняет: пока вызов был в пути,
    пользователь мог вернуться к проверке глоссария.
    """
    with unit_of_work.transaction() as transaction:
        _record_success(transaction, finished_task_id)
        book = book_record(transaction, book_id)
        if book.status is not BookStatus.PROCESSING:
            return
        unfinished = tuple(
            task
            for task in transaction.tasks.list_for_book(book_id)
            if task.task_type in task_types
            and task.id != finished_task_id
            and task.status is not TaskStatus.SUCCEEDED
        )
        if unfinished:
            return
        transaction.books.save(
            replace(
                book,
                status=apply_book_event(book.status, BookEvent.PROCESSING_COMPLETED),
            )
        )


def _record_success(transaction: UnitOfWork, task_id: str | None) -> None:
    """Отмечает выполняющуюся задачу успешной, если она ещё существует и выполняется.

    Задачи отменённого запуска удаляются вместе с ним, поэтому отсутствие строки — не ошибка.
    """
    if task_id is None:
        return
    task = transaction.tasks.get(task_id)
    if task is not None and task.status is TaskStatus.RUNNING:
        transaction.tasks.save(task.succeed())


def book_record(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Запись книги в открытой транзакции."""
    book = transaction.books.get(book_id)
    if book is None:
        raise BlockProcessingError(f"книги {book_id} нет в хранилище")
    return book


def total_usage(usages: Iterable[TokenUsage | None]) -> TokenUsage | None:
    """Суммарный расход токенов: провайдер считает каждый вызов отдельно.

    Провайдер, не сообщающий объём токенов, оставляет `None`: ноль означал бы бесплатный
    вызов, а не отсутствие данных (PRD § 24).
    """
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        prompt_tokens=_total_tokens(usage.prompt_tokens for usage in reported),
        completion_tokens=_total_tokens(usage.completion_tokens for usage in reported),
    )


def _total_tokens(values: Iterable[int | None]) -> int | None:
    """Сумма тех значений, которые провайдер сообщил."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _worded_for_the_model(error: Exception, context: TranslationContext) -> Exception:
    """Та же ошибка, но названная именами блоков, которые видела модель (ALG-05).

    Причина отклонения уходит в уточняющую инструкцию повтора, а проверки формулируют
    её стабильными идентификаторами блоков. Перечень ожидаемых блоков остаётся тот,
    который заполнил валидатор. Наружу — в журнал и в состояние задачи — уходит исходная
    ошибка с настоящими идентификаторами.
    """
    if not isinstance(error, ResultValidationError):
        return error
    return ResultValidationError(
        context.to_aliases(str(error)), expectation=error.expectation
    )


__all__ = [
    "BlockProcessingError",
    "Sleep",
    "TranslationCall",
    "TranslationCallResult",
    "book_record",
    "chapter_number",
    "complete_processing",
    "note_id_factory",
    "total_usage",
]
