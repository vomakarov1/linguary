"""Retry failed task use case (PRD § 16, § 18).

The user returns a `failed` task to the queue: it transitions back to `pending`, the attempt
counter is preserved, and error details are cleared. Only tasks that make sense to retry can be
retried: a successfully completed task is not re-executed, and a task failed with a `PermanentError`
(unreadable EPUB, unsupported input) requires a different file rather than a retry attempt.
An LLM configuration error (`ConfigurationError`) is retryable: after fixing the API key or model,
the same book can resume processing.

Retrying does not duplicate results: the task use case is idempotent by its
`book_id + task_type + scope_id` key, and `TranslationUnit` results are saved in a single transaction
as a whole (PRD § 17).

An ordinary retry of a chapter translation or block regeneration does not clear an unknown
outcome of a sent request and does not resume a root stopped on the call limit. Those need
an explicit restart or a limit extension.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import ProcessingTask, TaskStatus, TaskType


class RetryTaskError(ApplicationError):
    """Task cannot be retried in its current state."""


class TaskNotFoundError(RetryTaskError):
    """Task with this ID was not found in persistence."""


@dataclass(frozen=True, slots=True)
class RetryTaskCommand:
    """Retry command: which task to return to the queue."""

    task_id: str


class RetryTask:
    """Returns a failed task to the queue: `failed -> pending`."""

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __call__(self, command: RetryTaskCommand) -> ProcessingTask:
        """Transitions the task to `pending` and returns its updated state."""
        with self._unit_of_work.transaction() as transaction:
            task = transaction.tasks.get(command.task_id)
            if task is None:
                raise TaskNotFoundError(f"задачи {command.task_id} нет в хранилище")
            _ensure_retryable(task)
            allowed, reason = is_task_retryable_for_recovery(transaction, task)
            if not allowed:
                raise RetryTaskError(
                    reason or f"задача {task.id} не может быть повторена"
                )
            restarted = task.retry()
            transaction.tasks.save(restarted)
        return restarted


def _ensure_retryable(task: ProcessingTask) -> None:
    """Ensures that retrying the task makes sense."""
    if task.status is TaskStatus.SUPERSEDED:
        raise RetryTaskError(f"задача {task.id} вытеснена новой операцией и не может быть повторена")
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


_RECOVERY_TASKS = frozenset({
    TaskType.PROCESS_TRANSLATION_UNIT,
    TaskType.REGENERATE_BLOCK,
})
_OPEN_ROOT_STATUSES = frozenset({"pending", "running"})
_UNKNOWN_REASONS = frozenset({"unknown_outcome", "interrupted"})


def is_task_retryable_for_recovery(transaction, task) -> tuple[bool, str | None]:
    """Whether ordinary retry may resume this task's bound recovery roots.

    Translation and regeneration tasks with bound roots need an active run, current
    book conditions, and at least one root that is not superseded, still
    `pending` or `running`, free of `attempt.state == 'unknown'`, and inside its
    call budget. Other tasks, and translation tasks without bound roots, keep the
    existing retry rules. A refusal names an explicit restart or a limit extension.
    """
    if task.task_type not in _RECOVERY_TASKS:
        return True, None
    work = getattr(transaction, "work", None)
    if work is None:
        return True, None
    binding = work.get_binding("task", task.id)
    root_ids = () if binding is None else _bound_root_ids(binding)
    if not root_ids:
        return True, None
    runs = getattr(transaction, "runs", None)
    active = None if runs is None else runs.active_for_book(task.book_id)
    conditions = work.latest_conditions(task.book_id)
    if active is None or conditions is None or binding.get("conditions_id") != conditions.id:
        return False, _refusal(task.id, restart=True, extend=False, unknown=False, stale=True)
    restart = False
    extend = False
    unknown = False
    for root_id in root_ids:
        root = work.get_root(root_id)
        if root is None or root.book_id != task.book_id or root.superseded_by is not None:
            restart = True
            continue
        attempts = work.list_root_attempts(root.id)
        if _root_is_retryable(root, attempts, conditions_id=conditions.id, run_id=active.id):
            return True, None
        has_unknown = _has_unknown_outcome(root, attempts)
        if has_unknown:
            unknown = True
            restart = True
        if root.reason == "limit" or (
            not _budget_remaining(root) and root.status in _OPEN_ROOT_STATUSES
        ):
            extend = True
        if not has_unknown and root.reason != "limit":
            restart = True
    if not restart and not extend:
        restart = True
    return False, _refusal(task.id, restart=restart, extend=extend, unknown=unknown, stale=False)


def _bound_root_ids(binding) -> tuple[str, ...]:
    payload = binding.get("payload_json") or {}
    refs = payload.get("operation_refs") if isinstance(payload, Mapping) else None
    if not refs:
        return ()
    return tuple(
        ref["root_id"]
        for ref in refs
        if isinstance(ref, Mapping)
        and isinstance(ref.get("root_id"), str)
        and ref["root_id"]
    )


def _root_is_retryable(root, attempts, *, conditions_id, run_id) -> bool:
    return (
        root.conditions_id == conditions_id
        and root.manifest.get("active_run_id") == run_id
        and root.status in _OPEN_ROOT_STATUSES
        and not any(attempt.state == "unknown" for attempt in attempts)
        and _budget_remaining(root)
    )


def _has_unknown_outcome(root, attempts) -> bool:
    return root.reason in _UNKNOWN_REASONS or any(
        attempt.state == "unknown" for attempt in attempts
    )


def _budget_remaining(root) -> bool:
    policy = root.manifest.get("policy")
    if not isinstance(policy, Mapping):
        return False
    max_calls = policy.get("max_calls")
    if type(max_calls) is not int or max_calls < 0:
        return False
    if type(root.calls_used) is not int or type(root.extra_calls) is not int:
        return False
    return root.calls_used < max_calls + root.extra_calls


def _refusal(task_id: str, *, restart: bool, extend: bool, unknown: bool, stale: bool) -> str:
    if stale:
        return (
            f"задача {task_id} не может быть повторена: нет активного запуска "
            f"или условия книги неактуальны; нужен явный перезапуск"
        )
    if unknown and extend:
        return (
            f"задача {task_id} не может быть повторена: неизвестный исход требует "
            f"явного перезапуска, а остановка по пределу — продления лимита"
        )
    if restart and extend:
        return (
            f"задача {task_id} не может быть повторена: нужен явный перезапуск "
            f"или продление лимита"
        )
    if extend:
        return (
            f"задача {task_id} не может быть повторена: достигнут предел обращений, "
            f"нужно продление лимита"
        )
    if unknown:
        return (
            f"задача {task_id} не может быть повторена: неизвестный исход отправленного "
            f"запроса не снимается обычным повтором, нужен явный перезапуск"
        )
    return (
        f"задача {task_id} не может быть повторена: обычный повтор недоступен, "
        f"нужен явный перезапуск"
    )


__all__ = [
    "RetryTask",
    "RetryTaskCommand",
    "RetryTaskError",
    "TaskNotFoundError",
    "is_task_retryable_for_recovery",
]
