"""Application service and transactional helpers for optional book illustrations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from linguary.application.illustration_view import (
    IllustrationRegistryView,
    IllustrationView,
    build_illustration_registry_view,
    build_illustration_view,
)
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import ProcessingTask, TaskStatus, TaskType
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    ILLUSTRATION_POLICY_VERSION,
    IllustrationConflictError,
    IllustrationGenerationState,
    IllustrationLimitError,
    IllustrationNotFoundError,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScope,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationSettingsRevision,
    IllustrationValidationError,
    executable_scopes,
)

ILLUSTRATION_TASK_TYPES = frozenset(
    {
        TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
    }
)
_REFERENCE_STAGE_SET = frozenset(
    {
        IllustrationScopeStage.EXTRACT,
        IllustrationScopeStage.ALIAS,
        IllustrationScopeStage.DESIGN,
    }
)
_LIMIT_EXHAUSTED_REASONS = frozenset(
    {
        IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value,
        IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED.value,
    }
)


def _default_id() -> str:
    return str(uuid4())


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

    list_for_scope = getattr(tx.tasks, "list_for_scope", None)
    if callable(list_for_scope):
        prior_tasks = list(list_for_scope(resolved_book_id, task_type, scope.id))
    else:
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


def has_unfinished_illustration_work(tx: UnitOfWork, book_id: str) -> bool:
    """Return True if the book has pending/running illustration tasks or an active generation."""
    for task in tx.tasks.list_for_book(book_id):
        if task.task_type in ILLUSTRATION_TASK_TYPES and task.status in {
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
        }:
            return True
    illustrations_repo = getattr(tx, "illustrations", None)
    if illustrations_repo is None:
        return False
    head = illustrations_repo.get_head(book_id)
    if head.active_generation_id is None:
        return False
    gen = illustrations_repo.get_generation(book_id, head.active_generation_id)
    if gen is None:
        return False
    return gen.state in {
        IllustrationGenerationState.WAITING_SOURCE,
        IllustrationGenerationState.QUEUED,
        IllustrationGenerationState.RUNNING,
    }


def finalize_generation_state_in_tx(
    tx: UnitOfWork,
    book_id: str,
    generation_id: str,
    *,
    fallback_reason: str | None = None,
) -> bool:
    """Transition a non-revoked generation out of `RUNNING`/`QUEUED` when no executable scopes remain.

    Returns `True` if the generation transitioned to `FAILED` or was already in `FAILED`.
    """
    gen = tx.illustrations.get_generation(book_id, generation_id)
    if gen is None or gen.state in {
        IllustrationGenerationState.REVOKED,
        IllustrationGenerationState.READY,
        IllustrationGenerationState.SKIPPED,
    }:
        return False
    scopes = tx.illustrations.list_scopes(generation_id)
    exec_scopes = executable_scopes(scopes)
    has_active = any(
        s.state in {IllustrationScopeState.PENDING, IllustrationScopeState.RUNNING}
        for s in exec_scopes
    )
    if has_active:
        return False
    failed_scope = next(
        (s for s in exec_scopes if s.state is IllustrationScopeState.FAILED),
        None,
    ) or next(
        (s for s in scopes if s.state is IllustrationScopeState.FAILED),
        None,
    )
    if failed_scope is not None:
        tx.illustrations.update_generation_state(
            book_id,
            generation_id,
            state=IllustrationGenerationState.FAILED,
            reason_code=failed_scope.reason_code or fallback_reason,
        )
        return True
    return gen.state is IllustrationGenerationState.FAILED


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
        reconcile_in_tx_fn: Callable[[Any, str], bool] | None = None,
        notify_work_available: Callable[[], None] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._uow = unit_of_work
        self._provider_ready_fn = provider_ready_fn or (lambda: False)
        self._default_model_fn = default_model_fn or (lambda: None)
        self._source_snapshot_fn = source_snapshot_fn
        self._is_book_paused_fn = is_book_paused_fn
        self._reconcile_in_tx_fn = reconcile_in_tx_fn
        self._notify_work_available = notify_work_available
        self._id_factory = id_factory or _default_id

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

    async def prepare(
        self,
        book_id: str,
        *,
        expected_revision: int,
        expected_generation_id: str | None = None,
        section_id: str | None = None,
        force_new_generation: bool = False,
    ) -> IllustrationView:
        """Explicitly start or expand optional illustration preparation (`API-03`)."""
        if type(expected_revision) is not int or expected_revision < 0:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "expected_revision должен быть неотрицательным целым числом",
            )
        if type(force_new_generation) is not bool:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "force_new_generation должен быть логическим значением",
            )

        enqueued = False
        with self._uow.transaction() as tx:
            book = tx.books.get(book_id)
            if book is None:
                raise IllustrationNotFoundError(
                    IllustrationReasonCode.BOOK_NOT_FOUND, f"книга {book_id} не найдена"
                )
            head = tx.illustrations.get_head(book_id)
            if head.revision_number != expected_revision:
                raise IllustrationConflictError(
                    IllustrationReasonCode.REVISION_CONFLICT,
                    f"ожидалась ревизия {expected_revision}, актуальна {head.revision_number}",
                )
            if not head.effective_settings.enabled or head.settings_id is None:
                raise IllustrationConflictError(
                    IllustrationReasonCode.CHANNEL_DISABLED,
                    "канал иллюстраций выключен для этой книги",
                )
            if not self._provider_ready_fn():
                raise IllustrationConflictError(
                    IllustrationReasonCode.ROUTE_UNAVAILABLE,
                    "маршрут провайдера LLM не готов",
                )

            if expected_generation_id is not None:
                exp_gen = tx.illustrations.get_generation(book_id, expected_generation_id)
                if exp_gen is None:
                    raise IllustrationNotFoundError(
                        IllustrationReasonCode.GENERATION_NOT_FOUND,
                        f"поколение {expected_generation_id} не найдено",
                    )
                if head.active_generation_id != expected_generation_id:
                    raise IllustrationConflictError(
                        IllustrationReasonCode.GENERATION_CONFLICT,
                        "указанное поколение не является активным",
                    )

            document = tx.documents.get(book_id)
            if section_id is not None:
                if document is None or not any(
                    sec.id == section_id for sec in document.ordered_sections
                ):
                    raise IllustrationNotFoundError(
                        IllustrationReasonCode.SECTION_NOT_FOUND,
                        f"глава {section_id} не найдена",
                    )

            if self._source_snapshot_fn is None:
                raise IllustrationConflictError(
                    IllustrationReasonCode.SOURCE_INCOMPATIBLE,
                    "источник анализа иллюстраций не подключён",
                )
            snapshot = self._source_snapshot_fn(tx, book_id)
            if not snapshot.compatible:
                raise IllustrationConflictError(
                    snapshot.reason_code or IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
                    "книга несовместима с каналом иллюстраций",
                )
            if section_id is not None and section_id not in set(snapshot.content_section_ids):
                raise IllustrationConflictError(
                    IllustrationReasonCode.NON_CONTENT_SECTION,
                    "раздел не содержит иллюстрируемого текста",
                )

            active_gen = (
                tx.illustrations.get_generation(book_id, head.active_generation_id)
                if head.active_generation_id is not None
                else None
            )
            if active_gen is not None and expected_generation_id is not None and active_gen.id != expected_generation_id:
                raise IllustrationConflictError(
                    IllustrationReasonCode.GENERATION_CONFLICT,
                    "конфликт активного поколения иллюстраций",
                )

            target_sections = (
                (section_id,) if section_id is not None else tuple(snapshot.content_section_ids)
            )
            source_sha = snapshot.source_sha256 or ""
            base_authority = {
                k: v
                for k, v in (snapshot.source_authority or {}).items()
                if k != "manifest_sha256"
            }
            base_request_key = canonical_sha(
                {
                    "settings_id": head.settings_id,
                    "source_digest": source_sha,
                    "source_authority": base_authority,
                }
            )

            need_new_generation = active_gen is None
            request_key = base_request_key

            if active_gen is not None:
                is_stale = (
                    (snapshot.source_sha256 is not None and snapshot.source_sha256 != active_gen.source_sha256)
                    or (
                        active_gen.manifest_sha256 is not None
                        and snapshot.manifest_sha256 != active_gen.manifest_sha256
                    )
                )
                if is_stale:
                    if expected_generation_id is not None and expected_generation_id != active_gen.id:
                        raise IllustrationConflictError(
                            IllustrationReasonCode.GENERATION_CONFLICT,
                            "активное поколение устарело; подтвердите его идентификатор",
                        )
                    revoke_active_in_transaction(
                        tx, book_id, reason_code=IllustrationReasonCode.SOURCE_CHANGED.value
                    )
                    need_new_generation = True
                elif force_new_generation:
                    scopes = tx.illustrations.list_scopes(active_gen.id)
                    limit_reason: str | None = None
                    if active_gen.reason_code in _LIMIT_EXHAUSTED_REASONS:
                        limit_reason = active_gen.reason_code
                    else:
                        for sc in scopes:
                            if (
                                sc.state is IllustrationScopeState.FAILED
                                and sc.reason_code in _LIMIT_EXHAUSTED_REASONS
                            ):
                                limit_reason = sc.reason_code
                                break
                    if limit_reason is None:
                        raise IllustrationConflictError(
                            IllustrationReasonCode.GENERATION_CONFLICT,
                            "force_new_generation разрешён только после исчерпания лимита вызовов или областей",
                        )
                    revoke_active_in_transaction(tx, book_id, reason_code=limit_reason)
                    need_new_generation = True
                else:
                    # Active compatible generation exists and not force_new_generation.
                    gen, added = tx.illustrations.union_requested_sections(
                        book_id, active_gen.id, target_sections
                    )
                    if added:
                        if gen.state in {
                            IllustrationGenerationState.READY,
                            IllustrationGenerationState.FAILED,
                        }:
                            scopes_for_add = tx.illustrations.list_scopes(gen.id)
                            ref_failed = any(
                                s.stage in _REFERENCE_STAGE_SET
                                and s.state is IllustrationScopeState.FAILED
                                for s in scopes_for_add
                            )
                            if gen.state is IllustrationGenerationState.READY or not ref_failed:
                                tx.illustrations.update_generation_state(
                                    book_id,
                                    gen.id,
                                    state=IllustrationGenerationState.QUEUED,
                                    reason_code=None,
                                )
                        if self._reconcile_in_tx_fn is not None and self._reconcile_in_tx_fn(
                            tx, book_id
                        ):
                            enqueued = True
                    else:
                        scopes = tx.illustrations.list_scopes(active_gen.id)
                        has_failed = any(
                            s.state is IllustrationScopeState.FAILED
                            and (
                                section_id is None
                                or s.section_id is None
                                or s.section_id == section_id
                                or s.stage in _REFERENCE_STAGE_SET
                            )
                            for s in scopes
                        )
                        if has_failed or (
                            section_id is None
                            and active_gen.state is IllustrationGenerationState.FAILED
                        ):
                            raise IllustrationConflictError(
                                IllustrationReasonCode.GENERATION_CONFLICT,
                                "в поколении есть неуспешные области; используйте retry",
                            )
                        if self._reconcile_in_tx_fn is not None and self._reconcile_in_tx_fn(
                            tx, book_id
                        ):
                            enqueued = True
            elif force_new_generation:
                raise IllustrationConflictError(
                    IllustrationReasonCode.GENERATION_CONFLICT,
                    "force_new_generation разрешён только для исчерпанного активного поколения",
                )

            if need_new_generation:
                gen, _created = tx.illustrations.create_or_get_generation(
                    book_id,
                    settings_id=head.settings_id,
                    request_key=request_key,
                    source_sha256=source_sha,
                    source_authority=snapshot.source_authority or base_authority,
                    requested_sections=target_sections,
                )
                if snapshot.ready and snapshot.reason_code == IllustrationReasonCode.NO_CONTENT.value:
                    if snapshot.manifest is not None and snapshot.manifest_sha256 is not None:
                        tx.illustrations.seal_manifest(
                            book_id,
                            gen.id,
                            manifest=snapshot.manifest,
                            manifest_sha256=snapshot.manifest_sha256,
                        )
                    tx.illustrations.update_generation_state(
                        book_id,
                        gen.id,
                        state=IllustrationGenerationState.SKIPPED,
                        reason_code=IllustrationReasonCode.NO_CONTENT.value,
                    )
                elif self._reconcile_in_tx_fn is not None:
                    if self._reconcile_in_tx_fn(tx, book_id):
                        enqueued = True

        if enqueued and self._notify_work_available is not None:
            self._notify_work_available()
        return self.get(book_id, section_id=section_id)

    async def retry(
        self,
        book_id: str,
        *,
        expected_revision: int,
        expected_generation_id: str,
        section_id: str | None = None,
    ) -> IllustrationView:
        """Retry only retryable failed scopes of the active generation (`API-03`)."""
        if type(expected_revision) is not int or expected_revision < 0:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "expected_revision должен быть неотрицательным целым числом",
            )
        if not isinstance(expected_generation_id, str) or not expected_generation_id.strip():
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "для retry обязателен expected_generation_id",
            )

        enqueued = False
        revoked_for_source_change = False
        with self._uow.transaction() as tx:
            book = tx.books.get(book_id)
            if book is None:
                raise IllustrationNotFoundError(
                    IllustrationReasonCode.BOOK_NOT_FOUND, f"книга {book_id} не найдена"
                )
            head = tx.illustrations.get_head(book_id)
            if head.revision_number != expected_revision:
                raise IllustrationConflictError(
                    IllustrationReasonCode.REVISION_CONFLICT,
                    f"ожидалась ревизия {expected_revision}, актуальна {head.revision_number}",
                )
            if not head.effective_settings.enabled or head.settings_id is None:
                raise IllustrationConflictError(
                    IllustrationReasonCode.CHANNEL_DISABLED,
                    "канал иллюстраций выключен для этой книги",
                )
            if not self._provider_ready_fn():
                raise IllustrationConflictError(
                    IllustrationReasonCode.ROUTE_UNAVAILABLE,
                    "маршрут провайдера LLM не готов",
                )

            gen = tx.illustrations.get_generation(book_id, expected_generation_id)
            if gen is None:
                raise IllustrationNotFoundError(
                    IllustrationReasonCode.GENERATION_NOT_FOUND,
                    f"поколение {expected_generation_id} не найдено",
                )
            if head.active_generation_id != gen.id or gen.state is IllustrationGenerationState.REVOKED:
                raise IllustrationConflictError(
                    IllustrationReasonCode.GENERATION_CONFLICT,
                    "повторять можно только актуальное активное поколение",
                )

            if section_id is not None:
                document = tx.documents.get(book_id)
                if document is None or not any(
                    sec.id == section_id for sec in document.ordered_sections
                ):
                    raise IllustrationNotFoundError(
                        IllustrationReasonCode.SECTION_NOT_FOUND,
                        f"глава {section_id} не найдена",
                    )

            if self._source_snapshot_fn is not None:
                snapshot = self._source_snapshot_fn(tx, book_id)
                if (
                    not snapshot.compatible
                    or (snapshot.source_sha256 is not None and snapshot.source_sha256 != gen.source_sha256)
                    or (
                        gen.manifest_sha256 is not None
                        and snapshot.manifest_sha256 != gen.manifest_sha256
                    )
                ):
                    revoke_active_in_transaction(
                        tx, book_id, reason_code=IllustrationReasonCode.SOURCE_CHANGED.value
                    )
                    revoked_for_source_change = True

            if not revoked_for_source_change:
                policy = head.effective_settings.policy
                failed_scopes = tx.illustrations.list_scopes(
                    gen.id, state=IllustrationScopeState.FAILED, section_id=section_id
                )
                if (
                    gen.calls_reserved + gen.calls_sent >= policy.max_calls_per_generation
                    or any(
                        s.reason_code == IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value
                        for s in failed_scopes
                    )
                ):
                    raise IllustrationLimitError(
                        IllustrationReasonCode.CALL_LIMIT_EXHAUSTED,
                        "исчерпан лимит вызовов поколения; создайте новое поколение с force_new_generation",
                    )

                requeued = tx.illustrations.requeue_failed_scopes(
                    book_id, gen.id, section_id=section_id
                )
                if not requeued:
                    raise IllustrationConflictError(
                        IllustrationReasonCode.NOTHING_TO_RETRY,
                        "нет перезапускаемых неуспешных областей",
                    )
                for scope in requeued:
                    task_type = (
                        TaskType.PREPARE_ILLUSTRATION_REFERENCE
                        if scope.stage in _REFERENCE_STAGE_SET
                        else TaskType.PREPARE_CHAPTER_ILLUSTRATIONS
                    )
                    dispatch_scope_task(
                        tx,
                        scope,
                        generation_id=gen.id,
                        task_type=task_type,
                        id_factory=self._id_factory,
                        book_id=book_id,
                    )
                    enqueued = True
                tx.illustrations.update_generation_state(
                    book_id,
                    gen.id,
                    state=IllustrationGenerationState.QUEUED,
                    reason_code=None,
                )

        if revoked_for_source_change:
            raise IllustrationConflictError(
                IllustrationReasonCode.SOURCE_CHANGED,
                "исходные данные книги изменились",
            )
        if enqueued and self._notify_work_available is not None:
            self._notify_work_available()
        return self.get(book_id, section_id=section_id)

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
    "finalize_generation_state_in_tx",
    "has_unfinished_illustration_work",
    "revoke_active_in_transaction",
]
