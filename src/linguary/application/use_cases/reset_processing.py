"""Возврат к проверке глоссария: сброс результатов обработки книги (PRD § 11, § 21.5, § 22).

Пользователь выполняет этот сброс, когда увидел неудачное переводческое решение, влияющее
на всю книгу: точечные ошибки отдельных блоков исправляет повторная генерация. Сброс
удаляет запуск обработки вместе с его результатами — переводами, словарными подсказками,
`Language Notes` и замечаниями, — а импорт, стабильные `DocumentBlock` и анализ глав
сохраняет: повторный анализ самый дорогой этап и заново из-за правки одного термина
выполняться не должен.

Всё делается одной транзакцией: запуск, его результаты, задачи обработки и повторной
генерации и состояние книги меняются вместе. Вызов модели, начатый до сброса, транзакцией
не охвачен, и его поздний результат отбрасывает уже проверка активного запуска в транзакции
сохранения (ALG-04): запуска, с которым начиналась глава, больше нет.

Отдельного «размораживания» глоссария сброс не требует: правка решений разрешена в
`GLOSSARY_REVIEW` и `READY_TO_PROCESS`, и книга попадает в первое из них (ALG-06).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import BookRecord, UnitOfWork
from linguary.domain import BookEvent, TaskType, apply_book_event

#: Задачи, которые сброс снимает: они принадлежат отменённому запуску и после сброса
#: выполняться не должны (PRD § 21.5).
RESET_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK}
)


class ResetProcessingError(ApplicationError):
    """Обработку книги нельзя сбросить."""


@dataclass(frozen=True, slots=True)
class ResetProcessingCommand:
    """Команда возврата к проверке глоссария.

    Подтверждение пользователя — забота интерфейса: он показывает, что именно будет
    удалено (PRD § 11), и вызывает сценарий только после явного согласия.
    """

    book_id: str


@dataclass(frozen=True, slots=True)
class ResetProcessingResult:
    """Что сброс удалил: пользователю показывается цена подтверждённого действия.

    `run_id` пуст, если активного запуска у книги уже не было: сброс тогда только
    возвращает её к проверке глоссария.
    """

    book_id: str
    run_id: str | None = None
    removed_results: int = 0
    removed_tasks: int = 0


class ResetProcessing:
    """Удаляет результаты обработки книги и возвращает её в `GLOSSARY_REVIEW`."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: ResetProcessingCommand) -> ResetProcessingResult:
        """Сбрасывает обработку книги одной транзакцией (PRD § 17, § 21.5).

        Книга, из состояния которой возврат не выполняется, получает отказ домена
        (в Web UI — 409): сбрасывать нечего, пока обработка не начиналась.
        """
        with self._unit_of_work.transaction() as transaction:
            book = _book(transaction, command.book_id)
            status = apply_book_event(book.status, BookEvent.PROCESSING_RESET_CONFIRMED)
            run = transaction.runs.active_for_book(book.id)
            removed_results = (
                transaction.results.count_for_run(run.id) if run is not None else 0
            )
            removed_tasks = _remove_tasks(transaction, book.id)
            if run is not None:
                transaction.runs.delete(run.id)
            transaction.books.save(replace(book, status=status))
        return ResetProcessingResult(
            book_id=book.id,
            run_id=run.id if run is not None else None,
            removed_results=removed_results,
            removed_tasks=removed_tasks,
        )


def _remove_tasks(transaction: UnitOfWork, book_id: str) -> int:
    """Снимает задачи обработки и повторной генерации книги (PRD § 18.1).

    Удаляются задачи в любом состоянии, а не только запланированные и выполняющиеся:
    идемпотентный ключ задачи живёт, пока живёт сама задача, и завершённая задача
    отменённого запуска не дала бы поставить такую же задачу следующему запуску —
    книгу нельзя было бы обработать во второй раз. Задачи импорта, анализа глав и
    экспорта сброс не трогает: их результаты книга сохраняет.
    """
    removed = tuple(
        task
        for task in transaction.tasks.list_for_book(book_id)
        if task.task_type in RESET_TASK_TYPES
    )
    for task in removed:
        transaction.tasks.delete(task.id)
    return len(removed)


def _book(transaction: UnitOfWork, book_id: str) -> BookRecord:
    """Запись книги в открытой транзакции."""
    book = transaction.books.get(book_id)
    if book is None:
        raise ResetProcessingError(f"книги {book_id} нет в хранилище")
    return book


__all__ = [
    "RESET_TASK_TYPES",
    "ResetProcessing",
    "ResetProcessingCommand",
    "ResetProcessingError",
    "ResetProcessingResult",
]
