"""Машина состояний задачи в runner-е: исполнение, восстановление и повтор (P2.T7)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from task_factories import make_identity

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.runtime import SqliteTaskQueue
from linguary.application.task_runner import TaskHandler, TaskRunner
from linguary.application.use_cases.retry_task import (
    RetryTask,
    RetryTaskCommand,
    RetryTaskError,
)
from linguary.domain import (
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskStatus,
    TaskType,
)


def make_runner(
    task_queue: SqliteTaskQueue,
    handlers: Mapping[TaskType, TaskHandler] | None = None,
) -> TaskRunner:
    """Runner с указанным набором обработчиков."""
    return TaskRunner(queue=task_queue, handlers=handlers or {})


def recording_handler(executed: list[str]) -> TaskHandler:
    """Обработчик-двойник, отмечающий, какие задачи он выполнил."""

    async def handle(task: ProcessingTask) -> None:
        executed.append(task.id)

    return handle


def failing_handler(error: Exception) -> TaskHandler:
    """Обработчик-двойник, который всегда падает."""

    async def handle(task: ProcessingTask) -> None:
        raise error

    return handle


async def test_successful_task_becomes_succeeded(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Успешно выполненная задача сохраняется как `succeeded` без ошибки."""
    executed: list[str] = []
    task = await task_queue.enqueue(make_identity())
    runner = make_runner(
        task_queue, {TaskType.ANALYZE_CHAPTER: recording_handler(executed)}
    )

    finished = await runner.run_next()

    assert executed == [task.id]
    assert finished is not None
    assert (finished.status, finished.last_error) == (TaskStatus.SUCCEEDED, None)
    with unit_of_work.transaction() as uow:
        assert uow.tasks.get(task.id) == finished


async def test_run_next_returns_none_on_empty_queue(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Пустая очередь не заставляет runner ничего выполнять."""
    assert await make_runner(task_queue).run_next() is None


async def test_failed_handler_marks_task_failed_with_typed_error(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Ошибка сценария даёт `failed` с типом, сообщением и технической деталью."""
    await task_queue.enqueue(make_identity())
    runner = make_runner(
        task_queue,
        {TaskType.ANALYZE_CHAPTER: failing_handler(RuntimeError("провайдер недоступен"))},
    )

    finished = await runner.run_next()

    assert finished is not None
    assert finished.status is TaskStatus.FAILED
    assert finished.last_error == TaskError(
        type=ErrorType.TRANSIENT,
        message="провайдер недоступен",
        detail="builtins.RuntimeError",
    )
    assert finished.is_retryable


async def test_error_classifier_defines_error_type(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Разбор ошибок подключается снаружи: runner собственной политики не изобретает."""
    await task_queue.enqueue(make_identity())
    runner = TaskRunner(
        queue=task_queue,
        handlers={TaskType.ANALYZE_CHAPTER: failing_handler(ValueError("не тот язык"))},
        classify_error=lambda error: TaskError(
            type=ErrorType.MODEL_OUTPUT, message=str(error)
        ),
    )

    finished = await runner.run_next()

    assert finished is not None
    assert finished.last_error is not None
    assert finished.last_error.type is ErrorType.MODEL_OUTPUT


async def test_unregistered_task_type_fails_without_execution(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Тип без обработчика не выполняется втихую: задача падает с постоянной ошибкой.

    Обработчики анализа главы, обработки `TranslationUnit` и повторной генерации появятся
    в фазе 3; до этого runner честно сообщает, что выполнять задачу нечем.
    """
    await task_queue.enqueue(make_identity(task_type=TaskType.PROCESS_TRANSLATION_UNIT))
    runner = make_runner(task_queue)

    finished = await runner.run_next()

    assert finished is not None
    assert finished.status is TaskStatus.FAILED
    assert finished.last_error is not None
    assert finished.last_error.type is ErrorType.PERMANENT
    assert "не зарегистрирован" in finished.last_error.message
    assert not finished.is_retryable


async def test_interrupted_tasks_fail_on_application_start(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Задача, оставшаяся `running` после падения приложения, помечается прерванной."""
    await task_queue.enqueue(make_identity())
    claimed = await task_queue.claim_next_pending()
    assert claimed is not None

    interrupted = await make_runner(task_queue).recover_interrupted()

    assert [task.id for task in interrupted] == [claimed.id]
    with unit_of_work.transaction() as uow:
        stored = uow.tasks.get(claimed.id)
    assert stored is not None
    assert stored.status is TaskStatus.FAILED
    assert stored.last_error is not None
    assert stored.last_error.detail == "Interrupted"
    assert stored.is_retryable


async def test_recovery_leaves_finished_tasks_alone(
    task_queue: SqliteTaskQueue, stored_book: None
) -> None:
    """Восстановление трогает только незавершённые задачи."""
    executed: list[str] = []
    await task_queue.enqueue(make_identity(scope_id="section-1"))
    pending = await task_queue.enqueue(make_identity(scope_id="section-2"))
    runner = make_runner(
        task_queue, {TaskType.ANALYZE_CHAPTER: recording_handler(executed)}
    )
    succeeded = await runner.run_next()

    assert await runner.recover_interrupted() == ()

    assert succeeded is not None
    claimed = await task_queue.claim_next_pending()
    assert claimed is not None
    assert claimed.id == pending.id


async def test_retry_returns_failed_task_to_the_queue(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Повтор возвращает неуспешную задачу в очередь, сохраняя счётчик попыток."""
    task = await task_queue.enqueue(make_identity())
    runner = make_runner(
        task_queue,
        {TaskType.ANALYZE_CHAPTER: failing_handler(RuntimeError("таймаут"))},
    )
    await runner.run_next()

    restarted = RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))

    assert (restarted.status, restarted.attempts, restarted.last_error) == (
        TaskStatus.PENDING,
        1,
        None,
    )
    claimed = await task_queue.claim_next_pending()
    assert claimed is not None
    assert claimed.id == task.id


async def test_retried_task_is_executed_once_more(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """После повтора задача выполняется заново и доходит до `succeeded`."""
    executed: list[str] = []
    task = await task_queue.enqueue(make_identity())
    failing = make_runner(
        task_queue,
        {TaskType.ANALYZE_CHAPTER: failing_handler(RuntimeError("таймаут"))},
    )
    await failing.run_next()

    RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))
    finished = await make_runner(
        task_queue, {TaskType.ANALYZE_CHAPTER: recording_handler(executed)}
    ).run_next()

    assert executed == [task.id]
    assert finished is not None
    assert (finished.status, finished.attempts) == (TaskStatus.SUCCEEDED, 2)


async def test_succeeded_task_is_not_retried(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
) -> None:
    """Успешно завершённую задачу повторить нельзя (PRD § 18)."""
    executed: list[str] = []
    task = await task_queue.enqueue(make_identity())
    await make_runner(
        task_queue, {TaskType.ANALYZE_CHAPTER: recording_handler(executed)}
    ).run_next()

    with pytest.raises(RetryTaskError, match="уже выполнена успешно"):
        RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        pytest.param(
            lambda task: task,
            "повторяют только неуспешные задачи",
            id="pending",
        ),
        pytest.param(
            lambda task: task.start(),
            "повторяют только неуспешные задачи",
            id="running",
        ),
        pytest.param(
            lambda task: task.start().fail(
                TaskError(type=ErrorType.PERMANENT, message="EPUB нечитаем")
            ),
            "невосстановимой ошибкой",
            id="permanent-error",
        ),
    ],
)
async def test_retry_refuses_tasks_it_must_not_restart(
    task_queue: SqliteTaskQueue,
    unit_of_work: SqlAlchemyUnitOfWork,
    stored_book: None,
    prepare: Callable[[ProcessingTask], ProcessingTask],
    message: str,
) -> None:
    """Повтор доступен только для неуспешных задач с восстановимой ошибкой (PRD § 16)."""
    task = await task_queue.enqueue(make_identity())
    await task_queue.save(prepare(task))

    with pytest.raises(RetryTaskError, match=message):
        RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id=task.id))


async def test_retry_of_unknown_task_is_rejected(
    unit_of_work: SqlAlchemyUnitOfWork, stored_book: None
) -> None:
    """Повтор несуществующей задачи — ошибка сценария, а не тихая пустышка."""
    with pytest.raises(RetryTaskError, match="нет в хранилище"):
        RetryTask(unit_of_work=unit_of_work)(RetryTaskCommand(task_id="task-404"))
