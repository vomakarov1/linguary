"""Coordinator linking accepted NARRATIVE passes to optional illustration generations (`ALG-01`)."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from linguary.application.illustration_source import IllustrationSource
from linguary.application.illustrations import (
    dispatch_scope_task,
    revoke_active_in_transaction,
)
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import TaskStatus, TaskType
from linguary.domain.illustrations import (
    IllustrationGenerationState,
    IllustrationReasonCode,
    IllustrationScopeState,
)


def _default_id() -> str:
    return str(uuid4())


class IllustrationCoordinator:
    """Seals manifests, detects source drift, and dispatches ready illustration scopes."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        *,
        source: IllustrationSource,
        id_factory: Callable[[], str] | None = None,
        notify_work_available: Callable[[], None] | None = None,
        plan_reference_scopes: Callable[..., bool] | None = None,
        plan_chapter_scopes: Callable[..., bool] | None = None,
    ) -> None:
        self._uow = unit_of_work
        self._source = source
        self._id_factory = id_factory or _default_id
        self._notify_work_available = notify_work_available
        self._plan_reference_scopes = plan_reference_scopes
        self._plan_chapter_scopes = plan_chapter_scopes

    def bind_planners(
        self,
        *,
        plan_reference_scopes: Callable[..., bool] | None = None,
        plan_chapter_scopes: Callable[..., bool] | None = None,
    ) -> None:
        """Attach reference/chapter planners after `RuntimeSettings` construction."""
        if plan_reference_scopes is not None:
            self._plan_reference_scopes = plan_reference_scopes
        if plan_chapter_scopes is not None:
            self._plan_chapter_scopes = plan_chapter_scopes

    def reconcile_in_transaction(
        self,
        tx: UnitOfWork,
        book_id: str,
        *,
        precomputed_source_digest: str | None = None,
    ) -> bool:
        """Reconcile a book's active illustration generation inside an open transaction.

        Returns `True` if work was dispatched (caller should invoke `notify_work_available`
        after commit).
        """
        illustrations_repo = getattr(tx, "illustrations", None)
        if illustrations_repo is None:
            return False
        head = illustrations_repo.get_head(book_id)
        if head.active_generation_id is None:
            return False
        gen = illustrations_repo.get_generation(book_id, head.active_generation_id)
        if gen is None or gen.state is IllustrationGenerationState.REVOKED:
            return False

        snapshot = self._source.snapshot(
            tx, book_id, precomputed_source_digest=precomputed_source_digest
        )

        # If a manifest was already sealed and source/manifest drifted -> revoke active generation.
        if gen.manifest_sha256 is not None:
            if (
                not snapshot.compatible
                or (snapshot.source_sha256 is not None and snapshot.source_sha256 != gen.source_sha256)
                or snapshot.manifest_sha256 != gen.manifest_sha256
            ):
                revoke_active_in_transaction(
                    tx, book_id, reason_code=IllustrationReasonCode.SOURCE_CHANGED.value
                )
                return False

        if gen.state is IllustrationGenerationState.WAITING_SOURCE:
            if not snapshot.compatible:
                illustrations_repo.update_generation_state(
                    book_id,
                    gen.id,
                    state=IllustrationGenerationState.WAITING_SOURCE,
                    reason_code=snapshot.reason_code,
                )
                return False
            if snapshot.ready and snapshot.reason_code == IllustrationReasonCode.NO_CONTENT.value:
                if snapshot.manifest is not None and snapshot.manifest_sha256 is not None:
                    illustrations_repo.seal_manifest(
                        book_id,
                        gen.id,
                        manifest=snapshot.manifest,
                        manifest_sha256=snapshot.manifest_sha256,
                    )
                illustrations_repo.update_generation_state(
                    book_id,
                    gen.id,
                    state=IllustrationGenerationState.SKIPPED,
                    reason_code=IllustrationReasonCode.NO_CONTENT.value,
                )
                return False
            if not snapshot.ready or snapshot.manifest is None or snapshot.manifest_sha256 is None:
                illustrations_repo.update_generation_state(
                    book_id,
                    gen.id,
                    state=IllustrationGenerationState.WAITING_SOURCE,
                    reason_code=snapshot.reason_code,
                )
                return False

            gen = illustrations_repo.seal_manifest(
                book_id,
                gen.id,
                manifest=snapshot.manifest,
                manifest_sha256=snapshot.manifest_sha256,
            )
            gen = illustrations_repo.update_generation_state(
                book_id,
                gen.id,
                state=IllustrationGenerationState.QUEUED,
                reason_code=None,
            )

        enqueued = False
        if (
            gen.manifest_sha256 is not None
            and gen.registry_sha256 is None
            and self._plan_reference_scopes is not None
            and gen.state in {IllustrationGenerationState.QUEUED, IllustrationGenerationState.RUNNING}
        ):
            if self._plan_reference_scopes(tx, gen, snapshot, id_factory=self._id_factory):
                enqueued = True
            refreshed_gen = illustrations_repo.get_generation(book_id, gen.id)
            if refreshed_gen is not None:
                gen = refreshed_gen

        if (
            gen.manifest_sha256 is not None
            and gen.registry_sha256 is not None
            and self._plan_chapter_scopes is not None
            and gen.state in {IllustrationGenerationState.QUEUED, IllustrationGenerationState.RUNNING}
        ):
            if self._plan_chapter_scopes(tx, gen, snapshot, id_factory=self._id_factory):
                enqueued = True

        return enqueued

    def reconcile(self, book_id: str) -> None:
        """Reconcile a single book in its own short transaction and notify after commit."""
        with self._uow.transaction() as tx:
            enqueued = self.reconcile_in_transaction(tx, book_id)
        if enqueued and self._notify_work_available is not None:
            self._notify_work_available()

    def on_narrative_ready(self, book_id: str, section_id: str) -> None:
        """Callback invoked by `AnalyzeChapter` after full NARRATIVE acceptance."""
        del section_id
        self.reconcile(book_id)

    def reconcile_active(self) -> None:
        """Reconcile all active generations safely (safe when `alone=False`)."""
        with self._uow.transaction() as tx:
            illustrations_repo = getattr(tx, "illustrations", None)
            if illustrations_repo is None:
                return
            active = illustrations_repo.list_active_generations()
            enqueued_any = False
            for gen in active:
                if self.reconcile_in_transaction(tx, gen.book_id):
                    enqueued_any = True
        if enqueued_any and self._notify_work_available is not None:
            self._notify_work_available()

    def recover_interrupted(self) -> None:
        """Exclusive startup crash recovery (`alone=True`)."""
        enqueued_any = False
        with self._uow.transaction() as tx:
            illustrations_repo = getattr(tx, "illustrations", None)
            if illustrations_repo is None:
                return
            illustrations_repo.recover_interrupted_attempts()
            for gen in illustrations_repo.list_active_generations():
                scopes = illustrations_repo.list_scopes(gen.id)
                for scope in scopes:
                    if scope.task_id is None:
                        continue
                    task = tx.tasks.get(scope.task_id)
                    if task is None or task.status is not TaskStatus.FAILED:
                        continue
                    if scope.state is IllustrationScopeState.PENDING:
                        task_type = (
                            TaskType.PREPARE_ILLUSTRATION_REFERENCE
                            if scope.stage.value in {"extract", "alias", "design"}
                            else TaskType.PREPARE_CHAPTER_ILLUSTRATIONS
                        )
                        dispatch_scope_task(
                            tx,
                            scope,
                            generation_id=gen.id,
                            task_type=task_type,
                            id_factory=self._id_factory,
                            book_id=gen.book_id,
                        )
                        enqueued_any = True
                    elif scope.state in {
                        IllustrationScopeState.ACCEPTED,
                        IllustrationScopeState.SKIPPED,
                        IllustrationScopeState.SPLIT,
                    }:
                        tx.tasks.save(task.supersede())
                refreshed_scopes = illustrations_repo.list_scopes(gen.id)
                active_scopes = [
                    s for s in refreshed_scopes if s.state is not IllustrationScopeState.SPLIT
                ]
                has_active_work = any(
                    s.state in {IllustrationScopeState.PENDING, IllustrationScopeState.RUNNING}
                    for s in active_scopes
                )
                failed_scope = next(
                    (s for s in active_scopes if s.state is IllustrationScopeState.FAILED),
                    None,
                )
                if not has_active_work and failed_scope is not None:
                    illustrations_repo.update_generation_state(
                        gen.book_id,
                        gen.id,
                        state=IllustrationGenerationState.FAILED,
                        reason_code=failed_scope.reason_code
                        or IllustrationReasonCode.INTERRUPTED.value,
                    )
                    continue
                if self.reconcile_in_transaction(tx, gen.book_id):
                    enqueued_any = True
        if enqueued_any and self._notify_work_available is not None:
            self._notify_work_available()


__all__ = ["IllustrationCoordinator"]
