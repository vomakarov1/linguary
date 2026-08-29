"""Сценарий повтора неуспешной задачи (PRD § 16, § 18).

Пользователь возвращает задачу `failed` в очередь: она снова становится `pending`, счётчик
попыток сохраняется, ошибка снимается. Повторить можно только то, что имеет смысл повторять:
успешно завершённая задача повторно не выполняется, а задача, упавшая с `PermanentError`
(нечитаемый EPUB, неподдержанный вход), требует другого файла, а не новой попытки.

Дублирования результата повтор не создаёт: сценарий задачи идемпотентен по своему ключу
`book_id + task_type + scope_id`, а результат `TranslationUnit` сохраняется одной
транзакцией целиком (PRD § 17).
"""

from __future__ import annotations

from dataclasses import dataclass

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import ProcessingTask, TaskStatus


class RetryTaskError(ApplicationError):
    """Задачу нельзя запустить повторно в её текущем состоянии."""


@dataclass(frozen=True, slots=True)
class RetryTaskCommand:
    """Команда повтора: какую задачу вернуть в очередь."""

    task_id: str


class RetryTask:
    """Возвращает неуспешную задачу в очередь: `failed -> pending`."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: RetryTaskCommand) -> ProcessingTask:
        """Переводит задачу в `pending` и возвращает её новое состояние."""
        with self._unit_of_work.transaction() as transaction:
            task = transaction.tasks.get(command.task_id)
            if task is None:
                raise RetryTaskError(f"задачи {command.task_id} нет в хранилище")
            _ensure_retryable(task)
            restarted = task.retry()
            transaction.tasks.save(restarted)
        return restarted


def _ensure_retryable(task: ProcessingTask) -> None:
    """Проверяет, что задачу вообще имеет смысл повторять."""
    if task.status is TaskStatus.SUCCEEDED:
        raise RetryTaskError(
            f"задача {task.id} уже выполнена успешно и повторно не выполняется"
        )
    if task.status is not TaskStatus.FAILED:
        raise RetryTaskError(
            f"задача {task.id} в состоянии {task.status}: повторяют только неуспешные задачи"
        )
    if not task.is_retryable:
        raise RetryTaskError(
            f"задача {task.id} упала с невосстановимой ошибкой: нужен другой исходный файл, "
            f"а не повторная попытка"
        )


__all__ = [
    "RetryTask",
    "RetryTaskCommand",
    "RetryTaskError",
]
