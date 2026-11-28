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

An ordinary retry of a chapter translation, block regeneration, or chapter correction does
not clear an unknown outcome of a sent request and does not resume a root stopped on the
call limit. Translation and regeneration need an explicit restart or a limit extension.
Chapter correction may send again only after an explicit duplicate-billing acknowledgement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from linguary.application.errors import ApplicationError
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import AnalysisPassKind, ProcessingTask, TaskStatus, TaskType
from linguary.domain.book_conditions import thaw


class RetryTaskError(ApplicationError):
    """Task cannot be retried in its current state."""


class TaskNotFoundError(RetryTaskError):
    """Task with this ID was not found in persistence."""


@dataclass(frozen=True, slots=True)
class RetryTaskCommand:
    """Retry command: which task to return to the queue."""

    task_id: str
    acknowledge_unknown_outcome: bool = False
    clear_cache: bool = False


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
            if (
                command.acknowledge_unknown_outcome
                and task.task_type is TaskType.CHAPTER_CORRECTION
            ):
                acknowledge = getattr(transaction.work, "acknowledge_unknown_attempts", None)
                if callable(acknowledge):
                    acknowledge(task.id)
            allowed, reason = is_task_retryable_for_recovery(transaction, task)
            if not allowed:
                raise RetryTaskError(
                    reason or f"задача {task.id} не может быть повторена"
                )
            if (
                command.clear_cache
                and task.task_type is TaskType.ANALYZE_CHAPTER
                and task.scope_id
            ):
                analyses = getattr(transaction, "analyses", None)
                delete_fn = getattr(analyses, "delete_fragment_passes_for_section", None)
                if callable(delete_fn):
                    for pass_kind in (AnalysisPassKind.NARRATIVE, AnalysisPassKind.PEDAGOGICAL):
                        delete_fn(task.book_id, task.scope_id, pass_kind)
            restarted = task.retry()
            transaction.tasks.save(restarted)
        return restarted


_ILLUSTRATION_TASKS = frozenset(
    {
        TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
    }
)


def _ensure_retryable(task: ProcessingTask) -> None:
    """Ensures that retrying the task makes sense."""
    if task.task_type in _ILLUSTRATION_TASKS:
        raise RetryTaskError(
            "задачи подготовки иллюстраций повторяются через команду illustrations retry"
        )
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


def evaluate_task_retryable(
    task,
    *,
    active_run,
    conditions,
    binding,
    roots_by_id: Mapping[str, Any],
    scopes_by_root: Mapping[str, Sequence[Any]] | None = None,
    attempts_by_root: Mapping[str, Sequence[Any]] | None = None,
    task_attempts: Sequence[Any] | None = None,
) -> tuple[bool, str | None, bool]:
    """Pure recovery-guard check returning ``(allowed, reason, requires_unknown_ack)``."""
    if task.task_type in _ILLUSTRATION_TASKS:
        return (
            False,
            "задачи подготовки иллюстраций повторяются через команду illustrations retry",
            False,
        )
    if task.task_type is TaskType.CHAPTER_CORRECTION:
        if task_attempts is None:
            return True, None, False
        if any(
            item.state == "unknown" and not _acknowledged_duplicate_billing(item)
            for item in task_attempts
        ):
            return (
                False,
                _refusal(task.id, restart=True, extend=False, unknown=True, stale=False),
                True,
            )
        return True, None, False
    if task.task_type not in _RECOVERY_TASKS:
        return True, None, False
    root_ids = () if binding is None else _bound_root_ids(binding)
    if not root_ids:
        return True, None, False
    if active_run is None or conditions is None or binding.get("conditions_id") != conditions.id:
        return (
            False,
            _refusal(task.id, restart=True, extend=False, unknown=False, stale=True),
            False,
        )
    restart = False
    extend = False
    unknown = False
    all_complete = True
    for root_id in root_ids:
        root = roots_by_id.get(root_id)
        if root is None or root.book_id != task.book_id or getattr(root, "superseded_by", None) is not None:
            all_complete = False
            restart = True
            continue
        attempts = () if attempts_by_root is None else attempts_by_root.get(root.id, ())
        scopes = None if scopes_by_root is None else scopes_by_root.get(root.id)
        if _root_is_retryable(
            root, attempts, scopes=scopes, conditions_id=conditions.id, run_id=active_run.id
        ):
            return True, None, False
        if not (
            root.conditions_id == conditions.id
            and root.manifest.get("active_run_id") == active_run.id
            and root.status == "complete"
            and not any(attempt.state in {"unknown", "sent"} for attempt in attempts)
        ):
            all_complete = False
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
    if all_complete:
        return True, None, False
    if not restart and not extend:
        restart = True
    return (
        False,
        _refusal(task.id, restart=restart, extend=extend, unknown=unknown, stale=False),
        False,
    )


def is_task_retryable_for_recovery(transaction, task) -> tuple[bool, str | None]:
    """Whether ordinary retry may resume this task's bound recovery roots."""
    if task.task_type in _ILLUSTRATION_TASKS:
        return (
            False,
            "задачи подготовки иллюстраций повторяются через команду illustrations retry",
        )
    if task.task_type is TaskType.CHAPTER_CORRECTION:
        work = getattr(transaction, "work", None)
        if work is None:
            return True, None
        listed = getattr(work, "list_task_attempts", None)
        task_attempts = listed(task.id) if callable(listed) else None
        allowed, reason, _ = evaluate_task_retryable(
            task,
            active_run=None,
            conditions=None,
            binding=None,
            roots_by_id={},
            task_attempts=task_attempts,
        )
        return allowed, reason
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
    roots_by_id = {}
    attempts_by_root = {}
    list_scopes = getattr(work, "list_scopes", None)
    scopes_by_root = {} if callable(list_scopes) else None
    if active is not None and conditions is not None and binding.get("conditions_id") == conditions.id:
        for root_id in root_ids:
            root = work.get_root(root_id)
            if root is not None:
                roots_by_id[root_id] = root
                attempts_by_root[root.id] = work.list_root_attempts(root.id)
                if scopes_by_root is not None and callable(list_scopes):
                    scopes_by_root[root.id] = list_scopes(root.id)
    allowed, reason, _ = evaluate_task_retryable(
        task,
        active_run=active,
        conditions=conditions,
        binding=binding,
        roots_by_id=roots_by_id,
        scopes_by_root=scopes_by_root,
        attempts_by_root=attempts_by_root,
    )
    return allowed, reason


def _acknowledged_duplicate_billing(attempt) -> bool:
    outcome = thaw(getattr(attempt, "outcome", None) or {})
    return outcome.get("acknowledged_duplicate_billing_risk") is True


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


def _root_is_retryable(root, attempts, *, scopes=None, conditions_id, run_id) -> bool:
    if not (
        root.conditions_id == conditions_id
        and root.manifest.get("active_run_id") == run_id
        and root.status in _OPEN_ROOT_STATUSES
        and getattr(root, "reason", None) not in _UNKNOWN_REASONS
        and not any(attempt.state in {"unknown", "sent"} for attempt in attempts)
        and _budget_remaining(root)
    ):
        return False
    if scopes is not None:
        policy = root.manifest.get("policy")
        max_scope_calls = policy.get("max_scope_calls") if isinstance(policy, Mapping) else None
        if type(max_scope_calls) is not int or max_scope_calls < 0:
            return False
        if any(scope.status == "claimed" for scope in scopes):
            return False
        scope_counts = {
            scope.id: sum(
                attempt.scope_id == scope.id and attempt.state != "cancelled"
                for attempt in attempts
            )
            for scope in scopes
        }
        if not any(
            scope.status == "pending" and scope_counts[scope.id] < max_scope_calls
            for scope in scopes
        ):
            return False
    return True


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
    "evaluate_task_retryable",
    "is_task_retryable_for_recovery",
]
