"""Application service and transactional helpers for optional book illustrations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from linguary.application.illustration_view import (
    IllustrationRegistryView,
    IllustrationView,
    build_illustration_registry_view,
    build_illustration_view,
)
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.domain.illustrations import (
    ILLUSTRATION_POLICY_VERSION,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScope,
    IllustrationSettings,
    IllustrationSettingsRevision,
)

ILLUSTRATION_TASK_TYPES = frozenset(
    {
        TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
    }
)


def _parse_activation_suffix(operation_id: str, generation_id: str) -> int:
    prefix = f"{generation_id}:a"
    if not operation_id.startswith(prefix):
        return 0
    suffix = operation_id[len(prefix) :]
    return int(suffix) if suffix.isdigit() else 0


def dispatch_scope_task(
    tx: UnitOfWork,
    scope: IllustrationScope,
    *,
    generation_id: str,
    task_type: TaskType,
    id_factory: Callable[[], str],
    book_id: str | None = None,
) -> ProcessingTask:
    """Dispatch or re-activate a `ProcessingTask` for an illustration scope (`:a{n}` protocol).

    - If `scope.task_id` already points to a `PENDING` or `RUNNING` task for this scope,
      returns it idempotently without superseding.
    - Otherwise supersedes any previous non-terminal/failed task for the scope via
      `task.supersede()` (clearing `last_error=None`) and enqueues a fresh `PENDING` task
      with `operation_id=f"{generation_id}:a{next_activation}"` and `policy_version='illustration-v1'`.
    """
    existing: ProcessingTask | None = None
    if scope.task_id is not None:
        existing = tx.tasks.get(scope.task_id)
        if existing is not None and existing.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            return existing

    resolved_book_id = book_id
    if resolved_book_id is None and existing is not None:
        resolved_book_id = existing.book_id
    if resolved_book_id is None:
        for gen in tx.illustrations.list_active_generations():
            if gen.id == generation_id:
                resolved_book_id = gen.book_id
                break
    if resolved_book_id is None:
        raise ValueError(f"не удалось определить book_id для поколения {generation_id}")

    prior_tasks = [
        t
        for t in tx.tasks.list_for_book(resolved_book_id)
        if t.task_type is task_type and t.scope_id == scope.id
    ]
    max_activation = max(
        (_parse_activation_suffix(t.operation_id, generation_id) for t in prior_tasks),
        default=0,
    )
    next_activation = max_activation + 1

    if existing is not None and existing.status is not TaskStatus.SUPERSEDED:
        tx.tasks.save(existing.supersede())

    new_task = ProcessingTask(
        id=id_factory(),
        book_id=resolved_book_id,
        task_type=task_type,
        scope_id=scope.id,
        status=TaskStatus.PENDING,
        operation_id=f"{generation_id}:a{next_activation}",
        policy_version=ILLUSTRATION_POLICY_VERSION,
    )
    tx.tasks.add(new_task)
    tx.illustrations.set_scope_task(scope.id, new_task.id)
    return new_task


def revoke_active_in_transaction(
    tx: UnitOfWork,
    book_id: str,
    *,
    reason_code: str,
) -> str | None:
    """Revoke the book's active illustration generation and supersede all its tasks."""
    illustrations_repo = getattr(tx, "illustrations", None)
    if illustrations_repo is None:
        return None
    head = illustrations_repo.get_head(book_id)
    active_id = head.active_generation_id
    if active_id is None:
        return None
    illustrations_repo.revoke_generation(book_id, active_id, reason_code=reason_code)
    prefix = f"{active_id}:"
    for task in tx.tasks.list_for_book(book_id):
        if task.task_type not in ILLUSTRATION_TASK_TYPES:
            continue
        if task.operation_id != active_id and not task.operation_id.startswith(prefix):
            continue
        if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED}:
            tx.tasks.save(task.supersede())
    return active_id


class IllustrationService:
    """Facade for reading illustration state/registry and mutating settings/commands."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        *,
        provider_ready_fn: Callable[[], bool] | None = None,
        default_model_fn: Callable[[], str | None] | None = None,
        source_snapshot_fn: Callable[[Any, str], Any] | None = None,
        is_book_paused_fn: Callable[[Any, str], bool] | None = None,
    ) -> None:
        self._uow = unit_of_work
        self._provider_ready_fn = provider_ready_fn or (lambda: False)
        self._default_model_fn = default_model_fn or (lambda: None)
        self._source_snapshot_fn = source_snapshot_fn
        self._is_book_paused_fn = is_book_paused_fn

    def save_settings(
        self,
        book_id: str,
        *,
        expected_revision: int,
        settings: IllustrationSettings,
    ) -> IllustrationSettingsRevision:
        """Persist book illustration settings without enqueueing tasks or calling any LLM."""
        with self._uow.transaction() as tx:
            before = tx.illustrations.get_head(book_id)
            saved = tx.illustrations.save_settings(
                book_id,
                expected_revision=expected_revision,
                settings=settings,
            )
            if saved.id != before.settings_id:
                revoke_active_in_transaction(
                    tx,
                    book_id,
                    reason_code=IllustrationReasonCode.SETTINGS_CHANGED.value,
                )
            return saved

    def get(
        self,
        book_id: str,
        *,
        offset: int = 0,
        limit: int = 10,
        section_id: str | None = None,
        generation_id: str | None = None,
    ) -> IllustrationView:
        """Read paginated `DTO-01` state without side effects or provider API key."""
        return build_illustration_view(
            self._uow,
            book_id,
            offset=offset,
            limit=limit,
            section_id=section_id,
            generation_id=generation_id,
            provider_ready=self._provider_ready_fn(),
            default_model=self._default_model_fn(),
            source_snapshot_fn=self._source_snapshot_fn,
            is_book_paused_fn=self._is_book_paused_fn,
        )

    def get_registry(
        self,
        book_id: str,
        *,
        generation_id: str | None = None,
        entity_id: str | None = None,
        kind: IllustrationRegistryKind | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> IllustrationRegistryView:
        """Read paginated read-only `DTO-02` registry without side effects."""
        return build_illustration_registry_view(
            self._uow,
            book_id,
            generation_id=generation_id,
            entity_id=entity_id,
            kind=kind,
            offset=offset,
            limit=limit,
            source_snapshot_fn=self._source_snapshot_fn,
        )


__all__ = [
    "ILLUSTRATION_TASK_TYPES",
    "IllustrationService",
    "dispatch_scope_task",
    "revoke_active_in_transaction",
]
