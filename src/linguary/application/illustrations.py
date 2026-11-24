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
from linguary.domain import TaskStatus, TaskType
from linguary.domain.illustrations import (
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationSettings,
    IllustrationSettingsRevision,
)

ILLUSTRATION_TASK_TYPES = frozenset(
    {
        TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
    }
)


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
    "revoke_active_in_transaction",
]
