"""Локальный runner задач: захват, исполнение и фиксация итога (ALG-02).

Runner владеет машиной состояний одной задачи и ничем больше. Он берёт очередную задачу
из очереди, находит обработчик её типа, выполняет его и записывает результат — `succeeded`
либо `failed` с типизированной ошибкой. Планирование, параллелизм и SQLite лежат за портом
`TaskQueue` и в `linguary.adapters.runtime`: прикладной слой о них не знает (ADR § 3).

Переходы состояния берутся из домена (`ProcessingTask.start/succeed/fail/interrupt`), а не
переписываются здесь: атомарность захвата обеспечивает адаптер, но правило перехода остаётся
одно на всё приложение.

Runner асинхронный: в фазах 3-4 его запускают FastAPI и CLI, а обработка `TranslationUnit`
ждёт HTTP-ответов LLM, поэтому настраиваемый параллелизм (PRD § 25) — это ожидание
ввода-вывода. Синхронные сценарии подключаются обёрткой `to_async_handler` из адаптера.

Классификация ошибок обработчика — отдельная политика (PRD § 16), и здесь она не решается:
умолчание берётся из `linguary.application.retry_policy`, а сборка приложения может
подменить его через `classify_error`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from linguary.application.retry_policy import classify_error as classify_task_error
from linguary.domain import (
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskIdentity,
    TaskType,
)

#: Обработчик одного типа задач. Команду сценария он собирает сам: runner знает о задаче
#: только её идемпотентный ключ и не умеет строить вход `ImportBook` или `ExportBook`.
type TaskHandler = Callable[[ProcessingTask], Awaitable[None]]

#: Классификатор неожиданной ошибки обработчика в типизированную ошибку задачи.
type TaskErrorClassifier = Callable[[Exception], TaskError]


class TaskQueue(Protocol):
    """Очередь локальных задач.

    Постановка задачи идемпотентна по ключу `book_id + task_type + scope_id`, а захват
    атомарен: две попытки не могут взять одну задачу (ALG-02).
    """

    async def enqueue(self, identity: TaskIdentity) -> ProcessingTask:
        """Ставит задачу в очередь.

        Задача с таким ключом уже есть — возвращается она сама в своём текущем состоянии:
        успешно завершённая задача повторно не выполняется, а неуспешную пользователь
        перезапускает явно (PRD § 18).
        """

    async def claim_next_pending(self) -> ProcessingTask | None:
        """Захватывает очередную задачу `pending`, переводя её в `running`.

        Возвращает захваченную задачу либо `None`, если брать нечего.
        """

    async def save(self, task: ProcessingTask) -> None:
        """Сохраняет новое состояние задачи.

        Задачи, удалённой из очереди, сохранение не воскрешает: возврат к проверке
        глоссария снимает задачи отменённого запуска вместе с выполняющейся, и её итог
        фиксировать уже некуда (PRD § 21.5).
        """

    async def list_running(self) -> tuple[ProcessingTask, ...]:
        """Задачи в состоянии `running` в порядке их создания."""


class TaskRunner:
    """Исполняет задачи очереди по одной, соблюдая машину состояний ALG-02."""

    def __init__(
        self,
        *,
        queue: TaskQueue,
        handlers: Mapping[TaskType, TaskHandler],
        classify_error: TaskErrorClassifier | None = None,
    ) -> None:
        self._queue = queue
        self._handlers = dict(handlers)
        self._classify_error = classify_error or classify_task_error

    async def recover_interrupted(self) -> tuple[ProcessingTask, ...]:
        """Переводит задачи `running` в `failed` при старте приложения (ALG-02).

        Приложение, завершившееся во время работы задачи, оставляет её в `running`, где
        её уже никто не исполняет. Такая задача помечается прерванной, после чего
        пользователь запускает её повторно (PRD § 17-18).
        """
        interrupted = tuple(task.interrupt() for task in await self._queue.list_running())
        for task in interrupted:
            await self._queue.save(task)
        return interrupted

    async def run_next(self) -> ProcessingTask | None:
        """Выполняет одну задачу очереди и возвращает её завершённое состояние.

        `None` означает, что задач `pending` в очереди нет. Отмена корутины наружу не
        перехватывается: задача остаётся в `running` и будет помечена прерванной при
        следующем старте приложения — это честнее, чем объявить её неуспешной по причине,
        которой не было.
        """
        task = await self._queue.claim_next_pending()
        if task is None:
            return None
        return await self._finish(task)

    async def _finish(self, task: ProcessingTask) -> ProcessingTask:
        """Выполняет захваченную задачу и фиксирует её итог."""
        handler = self._handlers.get(task.task_type)
        if handler is None:
            return await self._save(task.fail(_unregistered_handler_error(task)))
        try:
            await handler(task)
        except Exception as error:  # любая ошибка сценария даёт неуспешную задачу
            return await self._save(task.fail(self._classify_error(error)))
        return await self._save(task.succeed())

    async def _save(self, task: ProcessingTask) -> ProcessingTask:
        """Сохраняет завершённое состояние задачи."""
        await self._queue.save(task)
        return task


def _unregistered_handler_error(task: ProcessingTask) -> TaskError:
    """Ошибка задачи, для типа которой не зарегистрирован обработчик.

    Повторять такую задачу нечем: приложение собрано без обработчика этого типа, и до
    его подключения результат не изменится.
    """
    return TaskError(
        type=ErrorType.PERMANENT,
        message=f"обработчик задач типа {task.task_type} не зарегистрирован",
        detail=f"task_type={task.task_type.value}",
    )


__all__ = [
    "TaskErrorClassifier",
    "TaskHandler",
    "TaskQueue",
    "TaskRunner",
]
