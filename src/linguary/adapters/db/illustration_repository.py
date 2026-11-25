"""SQLAlchemy repository for optional book illustration preparation (SQL-01)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from linguary.adapters.db.illustration_models import (
    IllustrationAttemptRow,
    IllustrationGenerationRow,
    IllustrationHeadRow,
    IllustrationRegistryItemRow,
    IllustrationScopeRow,
    IllustrationSettingsRow,
)
from linguary.adapters.db.models import AnalysisPassResultRow, BookRow, SectionRow
from linguary.domain.analysis import AnalysisPassResult
from linguary.domain.book_conditions import thaw
from linguary.domain.illustrations import (
    ChapterHeaderRow,
    IllustrationAttempt,
    IllustrationAttemptState,
    IllustrationConflictError,
    IllustrationGeneration,
    IllustrationGenerationState,
    IllustrationGenerationSummary,
    IllustrationHead,
    IllustrationLimitError,
    IllustrationNotFoundError,
    IllustrationOwnershipError,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScope,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationSettingsRevision,
    IllustrationValidationError,
    RecoveredIllustrationAttempt,
    RegistryItem,
    derive_chapter_state,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _to_settings_revision(row: IllustrationSettingsRow) -> IllustrationSettingsRevision:
    return IllustrationSettingsRevision(
        id=row.id,
        book_id=row.book_id,
        revision=row.revision,
        settings_sha256=row.settings_sha256,
        settings=IllustrationSettings.from_payload(row.payload_json),
        created_at=row.created_at,
    )


def _to_generation(row: IllustrationGenerationRow) -> IllustrationGeneration:
    return IllustrationGeneration(
        id=row.id,
        book_id=row.book_id,
        generation_number=row.generation_number,
        settings_id=row.settings_id,
        request_key=row.request_key,
        source_sha256=row.source_sha256,
        source_authority=dict(row.source_authority_json),
        requested_sections=tuple(row.requested_sections_json),
        manifest=dict(row.manifest_json) if row.manifest_json is not None else None,
        manifest_sha256=row.manifest_sha256,
        registry_sha256=row.registry_sha256,
        state=IllustrationGenerationState(row.state),
        reason_code=row.reason_code,
        calls_reserved=row.calls_reserved,
        calls_sent=row.calls_sent,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_scope(row: IllustrationScopeRow) -> IllustrationScope:
    return IllustrationScope(
        id=row.id,
        generation_id=row.generation_id,
        parent_scope_id=row.parent_scope_id,
        stage=IllustrationScopeStage(row.stage),
        scope_key=row.scope_key,
        section_id=row.section_id,
        section_order=row.section_order,
        section_title=row.section_title,
        task_id=row.task_id,
        input_sha256=row.input_sha256,
        state=IllustrationScopeState(row.state),
        reason_code=row.reason_code,
        attempt_id=row.attempt_id,
        output_sha256=row.output_sha256,
        payload=dict(row.payload_json) if row.payload_json is not None else None,
        summary=dict(row.summary_json),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_attempt(row: IllustrationAttemptRow) -> IllustrationAttempt:
    return IllustrationAttempt(
        id=row.id,
        scope_id=row.scope_id,
        task_id=row.task_id,
        ordinal=row.ordinal,
        state=IllustrationAttemptState(row.state),
        request_sha256=row.request_sha256,
        request_meta=dict(row.request_meta_json),
        outcome=dict(row.outcome_json) if row.outcome_json is not None else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_registry_item(row: IllustrationRegistryItemRow) -> RegistryItem:
    return RegistryItem(
        generation_id=row.generation_id,
        kind=IllustrationRegistryKind(row.kind),
        item_key=row.item_key,
        entity_id=row.entity_id,
        position_key=row.position_key,
        is_back_applicable=bool(row.is_back_applicable),
        payload_sha256=row.payload_sha256,
        payload=dict(row.payload_json),
    )


def _chapter_header_state(scope_state: IllustrationScopeState) -> str:
    if scope_state is IllustrationScopeState.ACCEPTED:
        return "ready"
    if scope_state is IllustrationScopeState.PENDING:
        return "queued"
    if scope_state is IllustrationScopeState.SPLIT:
        return "running"
    return scope_state.value


class SqlAlchemyIllustrationRepository:
    """SQLite-backed implementation of `IllustrationRepository`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _require_book(self, book_id: str) -> BookRow:
        row = self._session.get(BookRow, book_id)
        if row is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.BOOK_NOT_FOUND,
                f"книга {book_id} не найдена",
            )
        return row

    def _require_generation(self, book_id: str, generation_id: str) -> IllustrationGenerationRow:
        row = self._session.scalars(
            select(IllustrationGenerationRow).where(
                IllustrationGenerationRow.id == generation_id,
                IllustrationGenerationRow.book_id == book_id,
            )
        ).one_or_none()
        if row is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.GENERATION_NOT_FOUND,
                f"поколение иллюстраций {generation_id} не найдено",
            )
        return row

    def _get_or_create_head(self, book_id: str) -> IllustrationHeadRow:
        row = self._session.get(IllustrationHeadRow, book_id)
        if row is None:
            row = IllustrationHeadRow(book_id=book_id, settings_id=None, active_generation_id=None)
            self._session.add(row)
            self._session.flush()
        return row

    def _generation_max_calls(self, generation: IllustrationGenerationRow) -> int:
        settings_row = self._session.get(IllustrationSettingsRow, generation.settings_id)
        if settings_row is None:
            return 2048
        return IllustrationSettings.from_payload(settings_row.payload_json).policy.max_calls_per_generation

    def save_settings(
        self,
        book_id: str,
        *,
        expected_revision: int,
        settings: IllustrationSettings,
    ) -> IllustrationSettingsRevision:
        self._require_book(book_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "ожидаемая ревизия должна быть неотрицательным целым числом",
            )

        head = self._session.get(IllustrationHeadRow, book_id)
        current_row: IllustrationSettingsRow | None = None
        if head is not None and head.settings_id is not None:
            current_row = self._session.get(IllustrationSettingsRow, head.settings_id)
        if current_row is None:
            current_row = self._session.scalars(
                select(IllustrationSettingsRow)
                .where(IllustrationSettingsRow.book_id == book_id)
                .order_by(IllustrationSettingsRow.revision.desc())
                .limit(1)
            ).one_or_none()

        current_revision = 0 if current_row is None else current_row.revision
        payload = settings.to_payload()
        settings_sha = settings.sha256()

        if current_row is not None and current_row.settings_sha256 == settings_sha:
            if expected_revision in (current_revision, current_revision - 1):
                if head is None or head.settings_id != current_row.id:
                    head_row = self._get_or_create_head(book_id)
                    head_row.settings_id = current_row.id
                    self._session.flush()
                return _to_settings_revision(current_row)

        if expected_revision != current_revision:
            raise IllustrationConflictError(
                IllustrationReasonCode.REVISION_CONFLICT,
                f"конфликт ревизии настроек иллюстраций: ожидалась {expected_revision}, актуальна {current_revision}",
            )

        new_row = IllustrationSettingsRow(
            id=str(uuid4()),
            book_id=book_id,
            revision=current_revision + 1,
            settings_sha256=settings_sha,
            payload_json=payload,
            created_at=_now(),
        )
        self._session.add(new_row)
        self._session.flush()
        head_row = self._get_or_create_head(book_id)
        head_row.settings_id = new_row.id
        self._session.flush()
        return _to_settings_revision(new_row)

    def get_settings(
        self,
        book_id: str,
        settings_id: str,
    ) -> IllustrationSettingsRevision | None:
        row = self._session.scalars(
            select(IllustrationSettingsRow).where(
                IllustrationSettingsRow.id == settings_id,
                IllustrationSettingsRow.book_id == book_id,
            )
        ).one_or_none()
        return None if row is None else _to_settings_revision(row)

    def get_head(self, book_id: str) -> IllustrationHead:
        head = self._session.get(IllustrationHeadRow, book_id)
        if head is None:
            return IllustrationHead(book_id=book_id)
        settings_rev: IllustrationSettingsRevision | None = None
        if head.settings_id is not None:
            settings_row = self._session.scalars(
                select(IllustrationSettingsRow).where(
                    IllustrationSettingsRow.id == head.settings_id,
                    IllustrationSettingsRow.book_id == book_id,
                )
            ).one_or_none()
            if settings_row is not None:
                settings_rev = _to_settings_revision(settings_row)
        return IllustrationHead(
            book_id=book_id,
            settings_id=head.settings_id,
            active_generation_id=head.active_generation_id,
            settings_revision=settings_rev,
        )

    def create_or_get_generation(
        self,
        book_id: str,
        *,
        settings_id: str,
        request_key: str,
        source_sha256: str,
        source_authority: Mapping[str, Any],
        requested_sections: Sequence[str],
    ) -> tuple[IllustrationGeneration, bool]:
        self._require_book(book_id)
        settings_row = self._session.scalars(
            select(IllustrationSettingsRow).where(
                IllustrationSettingsRow.id == settings_id,
                IllustrationSettingsRow.book_id == book_id,
            )
        ).one_or_none()
        if settings_row is None:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                f"ревизия настроек {settings_id} не принадлежит книге {book_id}",
            )

        existing = self._session.scalars(
            select(IllustrationGenerationRow).where(
                IllustrationGenerationRow.book_id == book_id,
                IllustrationGenerationRow.request_key == request_key,
                IllustrationGenerationRow.state != IllustrationGenerationState.REVOKED.value,
            )
        ).one_or_none()
        if existing is not None:
            head_row = self._get_or_create_head(book_id)
            if head_row.active_generation_id != existing.id:
                head_row.active_generation_id = existing.id
                self._session.flush()
            return _to_generation(existing), False

        max_num = self._session.scalar(
            select(func.coalesce(func.max(IllustrationGenerationRow.generation_number), 0)).where(
                IllustrationGenerationRow.book_id == book_id
            )
        )
        next_num = int(max_num or 0) + 1
        now = _now()
        dedup_sections = list(dict.fromkeys(requested_sections))
        row = IllustrationGenerationRow(
            id=str(uuid4()),
            book_id=book_id,
            generation_number=next_num,
            settings_id=settings_id,
            request_key=request_key,
            source_sha256=source_sha256,
            source_authority_json=thaw(source_authority),
            requested_sections_json=dedup_sections,
            manifest_json=None,
            manifest_sha256=None,
            registry_sha256=None,
            state=IllustrationGenerationState.WAITING_SOURCE.value,
            reason_code=None,
            calls_reserved=0,
            calls_sent=0,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        try:
            self._session.flush()
        except IntegrityError as error:
            raise IllustrationConflictError(
                IllustrationReasonCode.GENERATION_CONFLICT,
                "конфликт создания активного поколения иллюстраций",
            ) from error
        head_row = self._get_or_create_head(book_id)
        head_row.settings_id = settings_id
        head_row.active_generation_id = row.id
        self._session.flush()
        return _to_generation(row), True

    def get_generation(
        self,
        book_id: str,
        generation_id: str,
    ) -> IllustrationGeneration | None:
        row = self._session.scalars(
            select(IllustrationGenerationRow).where(
                IllustrationGenerationRow.id == generation_id,
                IllustrationGenerationRow.book_id == book_id,
            )
        ).one_or_none()
        return None if row is None else _to_generation(row)

    def list_active_generations(self) -> tuple[IllustrationGeneration, ...]:
        rows = self._session.scalars(
            select(IllustrationGenerationRow)
            .join(
                IllustrationHeadRow,
                IllustrationHeadRow.active_generation_id == IllustrationGenerationRow.id,
            )
            .where(IllustrationGenerationRow.state != IllustrationGenerationState.REVOKED.value)
            .order_by(IllustrationGenerationRow.created_at, IllustrationGenerationRow.id)
        ).all()
        return tuple(_to_generation(row) for row in rows)

    def list_history(
        self,
        book_id: str,
        *,
        limit: int = 10,
    ) -> tuple[IllustrationGenerationSummary, ...]:
        rows = self._session.execute(
            select(IllustrationGenerationRow, IllustrationSettingsRow.revision)
            .join(
                IllustrationSettingsRow,
                IllustrationSettingsRow.id == IllustrationGenerationRow.settings_id,
            )
            .where(IllustrationGenerationRow.book_id == book_id)
            .order_by(
                IllustrationGenerationRow.generation_number.desc(),
                IllustrationGenerationRow.created_at.desc(),
            )
            .limit(limit)
        ).all()
        return tuple(
            IllustrationGenerationSummary(
                generation_id=gen.id,
                generation_number=gen.generation_number,
                state=IllustrationGenerationState(gen.state),
                settings_revision=rev,
                source_sha256=gen.source_sha256,
                created_at=gen.created_at,
            )
            for gen, rev in rows
        )

    def update_generation_state(
        self,
        book_id: str,
        generation_id: str,
        *,
        state: IllustrationGenerationState,
        reason_code: str | None = None,
    ) -> IllustrationGeneration:
        row = self._require_generation(book_id, generation_id)
        row.state = IllustrationGenerationState(state).value
        row.reason_code = reason_code
        row.updated_at = _now()
        self._session.flush()
        return _to_generation(row)

    def seal_manifest(
        self,
        book_id: str,
        generation_id: str,
        *,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
    ) -> IllustrationGeneration:
        row = self._require_generation(book_id, generation_id)
        if row.manifest_sha256 is not None and row.manifest_sha256 != manifest_sha256:
            raise IllustrationConflictError(
                IllustrationReasonCode.SOURCE_CHANGED,
                "манифест поколения уже зафиксирован с другой контрольной суммой",
            )
        row.manifest_json = thaw(manifest)
        row.manifest_sha256 = manifest_sha256
        authority = dict(row.source_authority_json)
        authority["manifest_sha256"] = manifest_sha256
        row.source_authority_json = authority
        row.updated_at = _now()
        self._session.flush()
        return _to_generation(row)

    def seal_registry(
        self,
        book_id: str,
        generation_id: str,
        *,
        registry_sha256: str,
    ) -> IllustrationGeneration:
        row = self._require_generation(book_id, generation_id)
        row.registry_sha256 = registry_sha256
        row.updated_at = _now()
        self._session.flush()
        return _to_generation(row)

    def union_requested_sections(
        self,
        book_id: str,
        generation_id: str,
        section_ids: Sequence[str],
    ) -> tuple[IllustrationGeneration, bool]:
        row = self._require_generation(book_id, generation_id)
        existing = list(row.requested_sections_json)
        existing_set = set(existing)
        added = [sid for sid in dict.fromkeys(section_ids) if sid not in existing_set]
        if not added:
            return _to_generation(row), False
        row.requested_sections_json = existing + added
        row.updated_at = _now()
        self._session.flush()
        return _to_generation(row), True

    def revoke_generation(
        self,
        book_id: str,
        generation_id: str,
        *,
        reason_code: str,
    ) -> bool:
        row = self._require_generation(book_id, generation_id)
        if row.state == IllustrationGenerationState.REVOKED.value:
            head = self._session.get(IllustrationHeadRow, book_id)
            if head is not None and head.active_generation_id == generation_id:
                head.active_generation_id = None
                self._session.flush()
            return False
        row.state = IllustrationGenerationState.REVOKED.value
        row.reason_code = reason_code
        row.updated_at = _now()
        running_scopes = self._session.scalars(
            select(IllustrationScopeRow).where(
                IllustrationScopeRow.generation_id == generation_id,
                IllustrationScopeRow.state == IllustrationScopeState.RUNNING.value,
            )
        ).all()
        for scope_row in running_scopes:
            scope_row.state = IllustrationScopeState.FAILED.value
            scope_row.reason_code = reason_code
            scope_row.updated_at = row.updated_at
        head = self._session.get(IllustrationHeadRow, book_id)
        if head is not None and head.active_generation_id == generation_id:
            head.active_generation_id = None
        self._session.flush()
        return True

    def upsert_scopes(
        self,
        generation_id: str,
        scopes: Sequence[IllustrationScopeSpec],
    ) -> tuple[IllustrationScope, ...]:
        gen = self._session.get(IllustrationGenerationRow, generation_id)
        if gen is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.GENERATION_NOT_FOUND,
                f"поколение иллюстраций {generation_id} не найдено",
            )
        results: list[IllustrationScope] = []
        now = _now()
        for spec in scopes:
            existing = self._session.scalars(
                select(IllustrationScopeRow).where(
                    IllustrationScopeRow.generation_id == generation_id,
                    IllustrationScopeRow.stage == spec.stage.value,
                    IllustrationScopeRow.scope_key == spec.scope_key,
                )
            ).one_or_none()
            if existing is not None:
                if existing.input_sha256 != spec.input_sha256:
                    raise IllustrationConflictError(
                        IllustrationReasonCode.GENERATION_CONFLICT,
                        f"область {spec.stage.value}:{spec.scope_key} уже существует с другим input_sha256",
                    )
                results.append(_to_scope(existing))
                continue
            row = IllustrationScopeRow(
                id=str(uuid4()),
                generation_id=generation_id,
                parent_scope_id=spec.parent_scope_id,
                stage=spec.stage.value,
                scope_key=spec.scope_key,
                section_id=spec.section_id,
                section_order=spec.section_order,
                section_title=spec.section_title,
                task_id=spec.task_id,
                input_sha256=spec.input_sha256,
                state=spec.state.value,
                reason_code=spec.reason_code,
                attempt_id=spec.attempt_id,
                output_sha256=spec.output_sha256,
                payload_json=thaw(spec.payload) if spec.payload is not None else None,
                summary_json=thaw(spec.summary),
                created_at=now,
                updated_at=now,
            )
            self._session.add(row)
            self._session.flush()
            results.append(_to_scope(row))
        return tuple(results)

    def get_scope(self, scope_id: str) -> IllustrationScope | None:
        row = self._session.get(IllustrationScopeRow, scope_id)
        return None if row is None else _to_scope(row)

    def set_scope_task(self, scope_id: str, task_id: str | None) -> IllustrationScope:
        row = self._session.get(IllustrationScopeRow, scope_id)
        if row is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.GENERATION_NOT_FOUND,
                f"область {scope_id} не найдена",
            )
        row.task_id = task_id
        row.updated_at = _now()
        self._session.flush()
        return _to_scope(row)

    def list_scopes(
        self,
        generation_id: str,
        *,
        stage: IllustrationScopeStage | None = None,
        state: IllustrationScopeState | None = None,
        section_id: str | None = None,
    ) -> tuple[IllustrationScope, ...]:
        statement = select(IllustrationScopeRow).where(
            IllustrationScopeRow.generation_id == generation_id
        )
        if stage is not None:
            statement = statement.where(IllustrationScopeRow.stage == stage.value)
        if state is not None:
            statement = statement.where(IllustrationScopeRow.state == state.value)
        if section_id is not None:
            statement = statement.where(IllustrationScopeRow.section_id == section_id)
        rows = self._session.scalars(
            statement.order_by(
                IllustrationScopeRow.stage,
                func.coalesce(IllustrationScopeRow.section_order, -1),
                IllustrationScopeRow.scope_key,
                IllustrationScopeRow.created_at,
            )
        ).all()
        return tuple(_to_scope(row) for row in rows)

    def list_scope_summaries(
        self,
        generation_id: str,
        *,
        stage: IllustrationScopeStage | None = None,
        state: IllustrationScopeState | None = None,
        section_id: str | None = None,
    ) -> tuple[IllustrationScope, ...]:
        """List scope metadata without loading `payload_json`."""
        statement = select(
            IllustrationScopeRow.id,
            IllustrationScopeRow.generation_id,
            IllustrationScopeRow.parent_scope_id,
            IllustrationScopeRow.stage,
            IllustrationScopeRow.scope_key,
            IllustrationScopeRow.section_id,
            IllustrationScopeRow.section_order,
            IllustrationScopeRow.section_title,
            IllustrationScopeRow.task_id,
            IllustrationScopeRow.input_sha256,
            IllustrationScopeRow.state,
            IllustrationScopeRow.reason_code,
            IllustrationScopeRow.attempt_id,
            IllustrationScopeRow.output_sha256,
            IllustrationScopeRow.summary_json,
            IllustrationScopeRow.created_at,
            IllustrationScopeRow.updated_at,
        ).where(IllustrationScopeRow.generation_id == generation_id)
        if stage is not None:
            statement = statement.where(IllustrationScopeRow.stage == stage.value)
        if state is not None:
            statement = statement.where(IllustrationScopeRow.state == state.value)
        if section_id is not None:
            statement = statement.where(IllustrationScopeRow.section_id == section_id)
        rows = self._session.execute(
            statement.order_by(
                IllustrationScopeRow.stage,
                func.coalesce(IllustrationScopeRow.section_order, -1),
                IllustrationScopeRow.scope_key,
                IllustrationScopeRow.created_at,
            )
        ).all()
        return tuple(
            IllustrationScope(
                id=row.id,
                generation_id=row.generation_id,
                parent_scope_id=row.parent_scope_id,
                stage=IllustrationScopeStage(row.stage),
                scope_key=row.scope_key,
                section_id=row.section_id,
                section_order=row.section_order,
                section_title=row.section_title,
                task_id=row.task_id,
                input_sha256=row.input_sha256,
                state=IllustrationScopeState(row.state),
                reason_code=row.reason_code,
                attempt_id=row.attempt_id,
                output_sha256=row.output_sha256,
                payload=None,
                summary=dict(row.summary_json) if isinstance(row.summary_json, Mapping) else {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        )

    def _resolve_pending_chapter_headers(
        self,
        generation_id: str,
        page: Sequence[ChapterHeaderRow],
    ) -> tuple[ChapterHeaderRow, ...]:
        pending_sids = [
            h.section_id
            for h in page
            if h.scope_id is not None and h.state == "queued"
        ]
        if not pending_sids:
            return tuple(page)
        child_rows = self._session.execute(
            select(
                IllustrationScopeRow.section_id,
                IllustrationScopeRow.state,
                IllustrationScopeRow.reason_code,
            )
            .where(
                IllustrationScopeRow.generation_id == generation_id,
                IllustrationScopeRow.stage != IllustrationScopeStage.CHAPTER.value,
                IllustrationScopeRow.state != IllustrationScopeState.SPLIT.value,
                IllustrationScopeRow.section_id.in_(pending_sids),
            )
            .order_by(
                IllustrationScopeRow.stage,
                IllustrationScopeRow.scope_key,
            )
        ).all()
        children_by_sid: dict[str, list[tuple[IllustrationScopeState, str | None]]] = defaultdict(list)
        for row in child_rows:
            if row.section_id is not None:
                children_by_sid[row.section_id].append(
                    (IllustrationScopeState(row.state), row.reason_code)
                )
        resolved: list[ChapterHeaderRow] = []
        for header in page:
            if header.scope_id is not None and header.state == "queued":
                new_state, new_reason = derive_chapter_state(
                    IllustrationScopeState.PENDING,
                    header.reason_code,
                    children_by_sid.get(header.section_id, ()),
                )
                resolved.append(
                    ChapterHeaderRow(
                        section_id=header.section_id,
                        order=header.order,
                        title=header.title,
                        state=new_state,
                        reason_code=new_reason,
                        moment_count=header.moment_count,
                        scope_id=header.scope_id,
                    )
                )
            else:
                resolved.append(header)
        return tuple(resolved)

    def get_chapter_header_page(
        self,
        book_id: str,
        generation_id: str | None,
        *,
        offset: int,
        limit: int,
    ) -> tuple[tuple[ChapterHeaderRow, ...], int]:
        self._require_book(book_id)
        if generation_id is not None:
            gen = self._require_generation(book_id, generation_id)
            scope_rows = self._session.execute(
                select(
                    IllustrationScopeRow.id,
                    IllustrationScopeRow.section_id,
                    IllustrationScopeRow.section_order,
                    IllustrationScopeRow.section_title,
                    IllustrationScopeRow.state,
                    IllustrationScopeRow.reason_code,
                    IllustrationScopeRow.summary_json,
                    IllustrationScopeRow.scope_key,
                )
                .where(
                    IllustrationScopeRow.generation_id == gen.id,
                    IllustrationScopeRow.stage == IllustrationScopeStage.CHAPTER.value,
                )
                .order_by(
                    func.coalesce(IllustrationScopeRow.section_order, 0),
                    IllustrationScopeRow.scope_key,
                )
            ).all()
            headers_by_sid: dict[str, ChapterHeaderRow] = {}
            ordered_scope_headers: list[ChapterHeaderRow] = []
            for row in scope_rows:
                scope_state = IllustrationScopeState(row.state)
                summary = row.summary_json if isinstance(row.summary_json, Mapping) else {}
                raw_count = summary.get("moment_count")
                moment_count = (
                    int(raw_count)
                    if scope_state is IllustrationScopeState.ACCEPTED
                    and type(raw_count) is int
                    else None
                )
                sid = row.section_id or row.scope_key
                header = ChapterHeaderRow(
                    section_id=sid,
                    order=0 if row.section_order is None else row.section_order,
                    title=row.section_title or "",
                    state=_chapter_header_state(scope_state),
                    reason_code=row.reason_code,
                    moment_count=moment_count,
                    scope_id=row.id,
                )
                headers_by_sid[sid] = header
                ordered_scope_headers.append(header)

            manifest_sections = (
                gen.manifest_json.get("sections")
                if isinstance(gen.manifest_json, Mapping)
                else None
            )
            if isinstance(manifest_sections, list) and manifest_sections:
                merged: list[ChapterHeaderRow] = []
                seen: set[str] = set()
                sorted_entries = sorted(
                    (e for e in manifest_sections if isinstance(e, Mapping) and e.get("section_id")),
                    key=lambda e: (int(e.get("order", 0)), str(e.get("section_id", ""))),
                )
                for entry in sorted_entries:
                    sid = str(entry["section_id"])
                    seen.add(sid)
                    existing = headers_by_sid.get(sid)
                    if existing is not None:
                        merged.append(existing)
                    elif bool(entry.get("skipped")):
                        merged.append(
                            ChapterHeaderRow(
                                section_id=sid,
                                order=int(entry.get("order", 0)),
                                title=str(entry.get("title") or ""),
                                state="skipped",
                                reason_code=(
                                    str(entry["skip_reason"])
                                    if entry.get("skip_reason") is not None
                                    else None
                                ),
                                moment_count=None,
                                scope_id=None,
                            )
                        )
                    else:
                        merged.append(
                            ChapterHeaderRow(
                                section_id=sid,
                                order=int(entry.get("order", 0)),
                                title=str(entry.get("title") or ""),
                                state="not_started",
                                reason_code=None,
                                moment_count=None,
                                scope_id=None,
                            )
                        )
                for header in ordered_scope_headers:
                    if header.section_id not in seen:
                        merged.append(header)
                return (
                    self._resolve_pending_chapter_headers(gen.id, merged[offset : offset + limit]),
                    len(merged),
                )

            if ordered_scope_headers:
                live_sections = self._session.execute(
                    select(SectionRow.id, SectionRow.order, SectionRow.title)
                    .where(SectionRow.book_id == book_id)
                    .order_by(SectionRow.order, SectionRow.id)
                ).all()
                live_ids = {row.id for row in live_sections}
                if live_sections and set(headers_by_sid).issubset(live_ids):
                    merged_live: list[ChapterHeaderRow] = []
                    for row in live_sections:
                        existing = headers_by_sid.get(row.id)
                        if existing is not None:
                            merged_live.append(existing)
                        else:
                            merged_live.append(
                                ChapterHeaderRow(
                                    section_id=row.id,
                                    order=row.order,
                                    title=row.title,
                                    state="not_started",
                                    reason_code=None,
                                    moment_count=None,
                                    scope_id=None,
                                )
                            )
                    return (
                        self._resolve_pending_chapter_headers(
                            gen.id, merged_live[offset : offset + limit]
                        ),
                        len(merged_live),
                    )
                return (
                    self._resolve_pending_chapter_headers(
                        gen.id, ordered_scope_headers[offset : offset + limit]
                    ),
                    len(ordered_scope_headers),
                )

        section_total = int(
            self._session.scalar(
                select(func.count(SectionRow.id)).where(SectionRow.book_id == book_id)
            )
            or 0
        )
        section_rows = self._session.execute(
            select(SectionRow.id, SectionRow.order, SectionRow.title)
            .where(SectionRow.book_id == book_id)
            .order_by(SectionRow.order, SectionRow.id)
            .offset(offset)
            .limit(limit)
        ).all()
        return (
            tuple(
                ChapterHeaderRow(
                    section_id=row.id,
                    order=row.order,
                    title=row.title,
                    state="not_started",
                    reason_code=None,
                    moment_count=None,
                    scope_id=None,
                )
                for row in section_rows
            ),
            section_total,
        )

    def find_reusable_scope(
        self,
        book_id: str,
        *,
        stage: IllustrationScopeStage,
        input_sha256: str,
    ) -> IllustrationScope | None:
        row = self._session.scalars(
            select(IllustrationScopeRow)
            .join(
                IllustrationGenerationRow,
                IllustrationGenerationRow.id == IllustrationScopeRow.generation_id,
            )
            .where(
                IllustrationGenerationRow.book_id == book_id,
                IllustrationScopeRow.stage == stage.value,
                IllustrationScopeRow.input_sha256 == input_sha256,
                IllustrationScopeRow.state == IllustrationScopeState.ACCEPTED.value,
            )
            .order_by(IllustrationScopeRow.updated_at.desc(), IllustrationScopeRow.id.desc())
            .limit(1)
        ).one_or_none()
        return None if row is None else _to_scope(row)

    def find_reusable_registry(
        self,
        book_id: str,
        *,
        registry_sha256: str,
    ) -> str | None:
        return self._session.scalar(
            select(IllustrationGenerationRow.id)
            .where(
                IllustrationGenerationRow.book_id == book_id,
                IllustrationGenerationRow.registry_sha256 == registry_sha256,
            )
            .order_by(IllustrationGenerationRow.generation_number.desc())
            .limit(1)
        )

    def requeue_failed_scopes(
        self,
        book_id: str,
        generation_id: str,
        *,
        section_id: str | None = None,
    ) -> tuple[IllustrationScope, ...]:
        gen = self._require_generation(book_id, generation_id)
        if gen.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationConflictError(
                IllustrationReasonCode.GENERATION_CONFLICT,
                "отозванное поколение нельзя перезапустить",
            )
        statement = select(IllustrationScopeRow).where(
            IllustrationScopeRow.generation_id == generation_id,
            IllustrationScopeRow.stage != IllustrationScopeStage.CHAPTER.value,
            IllustrationScopeRow.state == IllustrationScopeState.FAILED.value,
        )
        if section_id is not None:
            statement = statement.where(IllustrationScopeRow.section_id == section_id)
        rows = self._session.scalars(
            statement.order_by(IllustrationScopeRow.stage, IllustrationScopeRow.scope_key)
        ).all()
        now = _now()
        requeued: list[IllustrationScope] = []
        for row in rows:
            scope = _to_scope(row)
            if not scope.is_retryable_failure:
                continue
            row.state = IllustrationScopeState.PENDING.value
            row.reason_code = None
            row.attempt_id = None
            row.updated_at = now
            requeued.append(_to_scope(row))
        if requeued:
            gen.updated_at = now
            self._session.flush()
        return tuple(requeued)

    def _create_reserved_attempt(
        self,
        scope_row: IllustrationScopeRow,
        gen_row: IllustrationGenerationRow,
        *,
        task_id: str,
        max_calls: int,
    ) -> IllustrationAttempt:
        activation_attempts = int(
            self._session.scalar(
                select(func.count(IllustrationAttemptRow.id)).where(
                    IllustrationAttemptRow.scope_id == scope_row.id,
                    IllustrationAttemptRow.task_id == task_id,
                )
            )
            or 0
        )
        if activation_attempts >= max_calls:
            raise IllustrationLimitError(
                IllustrationReasonCode.CALL_LIMIT_EXHAUSTED,
                "исчерпан лимит попыток в текущей активации области",
            )
        gen_max_calls = self._generation_max_calls(gen_row)
        if gen_row.calls_reserved + gen_row.calls_sent >= gen_max_calls:
            raise IllustrationLimitError(
                IllustrationReasonCode.CALL_LIMIT_EXHAUSTED,
                "исчерпан лимит вызовов для поколения иллюстраций",
            )

        max_ordinal = int(
            self._session.scalar(
                select(func.coalesce(func.max(IllustrationAttemptRow.ordinal), 0)).where(
                    IllustrationAttemptRow.scope_id == scope_row.id
                )
            )
            or 0
        )
        now = _now()
        attempt_row = IllustrationAttemptRow(
            id=str(uuid4()),
            scope_id=scope_row.id,
            task_id=task_id,
            ordinal=max_ordinal + 1,
            state=IllustrationAttemptState.RESERVED.value,
            request_sha256=None,
            request_meta_json={},
            outcome_json=None,
            created_at=now,
            updated_at=now,
        )
        self._session.add(attempt_row)
        scope_row.state = IllustrationScopeState.RUNNING.value
        scope_row.task_id = task_id
        scope_row.attempt_id = attempt_row.id
        scope_row.reason_code = None
        scope_row.updated_at = now
        gen_row.calls_reserved += 1
        gen_row.updated_at = now
        try:
            self._session.flush()
        except IntegrityError as error:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_row.id} уже имеет активную попытку",
            ) from error
        return _to_attempt(attempt_row)

    def claim_scope(
        self,
        scope_id: str,
        *,
        task_id: str,
        expected_generation_id: str,
        max_calls: int,
    ) -> IllustrationAttempt:
        scope_row = self._session.get(IllustrationScopeRow, scope_id)
        if scope_row is None or scope_row.generation_id != expected_generation_id:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "область не найдена или относится к другому поколению",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None or gen_row.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение отозвано",
            )
        if scope_row.state != IllustrationScopeState.PENDING.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_id} нельзя захватить в состоянии {scope_row.state}",
            )
        return self._create_reserved_attempt(
            scope_row,
            gen_row,
            task_id=task_id,
            max_calls=max_calls,
        )

    def begin_next_attempt(
        self,
        scope_id: str,
        *,
        task_id: str,
        expected_generation_id: str,
        max_calls: int,
    ) -> IllustrationAttempt:
        scope_row = self._session.get(IllustrationScopeRow, scope_id)
        if scope_row is None or scope_row.generation_id != expected_generation_id:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "область не найдена или относится к другому поколению",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None or gen_row.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение отозвано",
            )
        if scope_row.state != IllustrationScopeState.RUNNING.value or scope_row.task_id != task_id:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "область не принадлежит текущей активации задачи",
            )
        return self._create_reserved_attempt(
            scope_row,
            gen_row,
            task_id=task_id,
            max_calls=max_calls,
        )

    def mark_attempt_sent(
        self,
        attempt_id: str,
        *,
        request_sha256: str,
        request_meta: Mapping[str, Any],
    ) -> IllustrationAttempt:
        attempt_row = self._session.get(IllustrationAttemptRow, attempt_id)
        if attempt_row is None or attempt_row.state != IllustrationAttemptState.RESERVED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "попытка не найдена или не находится в состоянии reserved",
            )
        scope_row = self._session.get(IllustrationScopeRow, attempt_row.scope_id)
        if scope_row is None or scope_row.attempt_id != attempt_id or scope_row.state != IllustrationScopeState.RUNNING.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "владение областью потеряно до отправки запроса",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None or gen_row.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение отозвано до отправки запроса",
            )
        if gen_row.calls_reserved <= 0:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "отсутствует зарезервированный вызов",
            )
        now = _now()
        attempt_row.state = IllustrationAttemptState.SENT.value
        attempt_row.request_sha256 = request_sha256
        attempt_row.request_meta_json = thaw(request_meta)
        attempt_row.updated_at = now
        gen_row.calls_reserved -= 1
        gen_row.calls_sent += 1
        gen_row.updated_at = now
        self._session.flush()
        return _to_attempt(attempt_row)

    def release_reserved_call(
        self,
        attempt_id: str,
        *,
        reason_code: str,
    ) -> None:
        attempt_row = self._session.get(IllustrationAttemptRow, attempt_id)
        if attempt_row is None or attempt_row.state != IllustrationAttemptState.RESERVED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "резерв попытки отсутствует или уже завершён",
            )
        scope_row = self._session.get(IllustrationScopeRow, attempt_row.scope_id)
        if scope_row is None:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "область попытки не найдена",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение попытки не найдено",
            )
        now = _now()
        attempt_row.state = IllustrationAttemptState.CANCELLED.value
        attempt_row.outcome_json = {"reason_code": reason_code}
        attempt_row.updated_at = now
        if gen_row.calls_reserved > 0:
            gen_row.calls_reserved -= 1
            gen_row.updated_at = now
        if scope_row.attempt_id == attempt_id and scope_row.state == IllustrationScopeState.RUNNING.value:
            scope_row.state = IllustrationScopeState.PENDING.value
            scope_row.attempt_id = None
            scope_row.updated_at = now
        self._session.flush()

    def record_attempt_outcome(
        self,
        attempt_id: str,
        *,
        state: IllustrationAttemptState,
        outcome: Mapping[str, Any] | None,
    ) -> IllustrationAttempt:
        attempt_row = self._session.get(IllustrationAttemptRow, attempt_id)
        if attempt_row is None:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"попытка {attempt_id} не найдена",
            )
        target = IllustrationAttemptState(state)
        if attempt_row.state == IllustrationAttemptState.RESERVED.value and target not in {
            IllustrationAttemptState.RESERVED,
            IllustrationAttemptState.SENT,
        }:
            scope_row = self._session.get(IllustrationScopeRow, attempt_row.scope_id)
            if scope_row is not None:
                gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
                if gen_row is not None and gen_row.calls_reserved > 0:
                    gen_row.calls_reserved -= 1
                    gen_row.updated_at = _now()
        attempt_row.state = target.value
        attempt_row.outcome_json = thaw(outcome) if outcome is not None else None
        attempt_row.updated_at = _now()
        self._session.flush()
        return _to_attempt(attempt_row)

    def split_scope(
        self,
        scope_id: str,
        *,
        attempt_id: str | None,
        children: Sequence[IllustrationScopeSpec],
        max_scopes: int,
    ) -> tuple[IllustrationScope, ...]:
        if not children:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "разбиение области требует хотя бы одного дочернего элемента",
            )
        scope_row = self._session.get(IllustrationScopeRow, scope_id)
        if scope_row is None:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_id} не найдена",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None or gen_row.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение отозвано",
            )
        if scope_row.state not in {
            IllustrationScopeState.PENDING.value,
            IllustrationScopeState.RUNNING.value,
        }:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_id} нельзя разбить в состоянии {scope_row.state}",
            )
        current_total = int(
            self._session.scalar(
                select(func.count(IllustrationScopeRow.id)).where(
                    IllustrationScopeRow.generation_id == scope_row.generation_id
                )
            )
            or 0
        )
        if current_total + len(children) > max_scopes:
            raise IllustrationLimitError(
                IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED,
                "превышен лимит областей поколения иллюстраций",
            )
        now = _now()
        if attempt_id is not None:
            attempt_row = self._session.get(IllustrationAttemptRow, attempt_id)
            if (
                attempt_row is None
                or attempt_row.scope_id != scope_id
                or scope_row.attempt_id != attempt_id
            ):
                raise IllustrationOwnershipError(
                    IllustrationReasonCode.STALE_ATTEMPT,
                    "попытка не владеет разбиваемой областью",
                )
            if attempt_row.state == IllustrationAttemptState.RESERVED.value:
                self.release_reserved_call(attempt_id, reason_code=IllustrationReasonCode.CONTEXT_LIMIT.value)
            elif attempt_row.state == IllustrationAttemptState.SENT.value:
                attempt_row.state = IllustrationAttemptState.REJECTED.value
                attempt_row.outcome_json = {"status": "split"}
                attempt_row.updated_at = now

        scope_row.state = IllustrationScopeState.SPLIT.value
        scope_row.reason_code = None
        scope_row.updated_at = now
        self._session.flush()

        child_specs = [
            IllustrationScopeSpec(
                stage=child.stage,
                scope_key=child.scope_key,
                input_sha256=child.input_sha256,
                section_id=child.section_id if child.section_id is not None else scope_row.section_id,
                section_order=(
                    child.section_order if child.section_order is not None else scope_row.section_order
                ),
                section_title=(
                    child.section_title if child.section_title is not None else scope_row.section_title
                ),
                parent_scope_id=scope_id,
                summary=child.summary,
                state=child.state,
                reason_code=child.reason_code,
                payload=child.payload,
                output_sha256=child.output_sha256,
                task_id=child.task_id,
                attempt_id=child.attempt_id,
            )
            for child in children
        ]
        return self.upsert_scopes(scope_row.generation_id, child_specs)

    def accept_scope(
        self,
        scope_id: str,
        *,
        attempt_id: str | None = None,
        expected_input_sha256: str,
        payload: Mapping[str, Any],
        output_sha256: str,
        summary: Mapping[str, Any],
        pass_mirror: AnalysisPassResult | None = None,
    ) -> None:
        scope_row = self._session.get(IllustrationScopeRow, scope_id)
        if scope_row is None:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_id} не найдена",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None or gen_row.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение отозвано",
            )
        if scope_row.input_sha256 != expected_input_sha256:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "контрольная сумма входа области изменилась",
            )
        now = _now()
        if attempt_id is not None:
            attempt_row = self._session.get(IllustrationAttemptRow, attempt_id)
            if (
                attempt_row is None
                or attempt_row.scope_id != scope_id
                or attempt_row.state != IllustrationAttemptState.SENT.value
                or scope_row.attempt_id != attempt_id
                or scope_row.state != IllustrationScopeState.RUNNING.value
            ):
                raise IllustrationOwnershipError(
                    IllustrationReasonCode.STALE_ATTEMPT,
                    "владение попыткой потеряно перед принятием области",
                )
            attempt_row.state = IllustrationAttemptState.ACCEPTED.value
            attempt_row.outcome_json = {"status": "accepted"}
            attempt_row.updated_at = now

        scope_row.state = IllustrationScopeState.ACCEPTED.value
        scope_row.reason_code = None
        scope_row.attempt_id = attempt_id
        scope_row.output_sha256 = output_sha256
        scope_row.payload_json = thaw(payload)
        scope_row.summary_json = thaw(summary)
        scope_row.updated_at = now
        gen_row.updated_at = now

        if pass_mirror is not None:
            self._upsert_analysis_pass_mirror(pass_mirror, now=now)

        self._session.flush()

    def _upsert_analysis_pass_mirror(self, result: AnalysisPassResult, *, now: datetime) -> None:
        existing = self._session.scalars(
            select(AnalysisPassResultRow).where(
                AnalysisPassResultRow.book_id == result.book_id,
                AnalysisPassResultRow.section_id == result.section_id,
                AnalysisPassResultRow.pass_kind == result.pass_kind.value,
                AnalysisPassResultRow.scope_key == result.scope_key,
            )
        ).one_or_none()
        if existing is None:
            self._session.add(
                AnalysisPassResultRow(
                    id=result.id,
                    book_id=result.book_id,
                    section_id=result.section_id,
                    pass_kind=result.pass_kind.value,
                    scope_key=result.scope_key,
                    input_sha256=result.input_sha256,
                    output_sha256=result.output_sha256,
                    conditions_id=result.conditions_id,
                    attempt_id=result.attempt_id,
                    payload_json=thaw(result.payload),
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            existing.input_sha256 = result.input_sha256
            existing.output_sha256 = result.output_sha256
            existing.conditions_id = result.conditions_id
            existing.attempt_id = result.attempt_id
            existing.payload_json = thaw(result.payload)
            existing.updated_at = now

    def fail_scope(
        self,
        scope_id: str,
        *,
        attempt_id: str | None = None,
        reason_code: str,
        outcome: Mapping[str, Any] | None = None,
    ) -> None:
        scope_row = self._session.get(IllustrationScopeRow, scope_id)
        if scope_row is None:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_id} не найдена",
            )
        gen_row = self._session.get(IllustrationGenerationRow, scope_row.generation_id)
        if gen_row is None or gen_row.state == IllustrationGenerationState.REVOKED.value:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                "поколение отозвано",
            )
        if scope_row.state in {
            IllustrationScopeState.ACCEPTED.value,
            IllustrationScopeState.SKIPPED.value,
            IllustrationScopeState.SPLIT.value,
        }:
            raise IllustrationOwnershipError(
                IllustrationReasonCode.STALE_ATTEMPT,
                f"область {scope_id} уже завершена в состоянии {scope_row.state}",
            )
        now = _now()
        if attempt_id is not None:
            attempt_row = self._session.get(IllustrationAttemptRow, attempt_id)
            if (
                attempt_row is None
                or attempt_row.scope_id != scope_id
                or (scope_row.attempt_id is not None and scope_row.attempt_id != attempt_id)
            ):
                raise IllustrationOwnershipError(
                    IllustrationReasonCode.STALE_ATTEMPT,
                    "попытка не владеет областью",
                )
            if attempt_row.state == IllustrationAttemptState.RESERVED.value:
                attempt_row.state = IllustrationAttemptState.CANCELLED.value
                if gen_row is not None and gen_row.calls_reserved > 0:
                    gen_row.calls_reserved -= 1
            elif attempt_row.state == IllustrationAttemptState.SENT.value:
                attempt_row.state = IllustrationAttemptState.FAILED.value
            if outcome is not None:
                attempt_row.outcome_json = thaw(outcome)
            elif attempt_row.outcome_json is None:
                attempt_row.outcome_json = {"reason_code": reason_code}
            attempt_row.updated_at = now
        scope_row.state = IllustrationScopeState.FAILED.value
        scope_row.reason_code = reason_code
        scope_row.updated_at = now
        if gen_row is not None:
            gen_row.updated_at = now
        self._session.flush()

    def recover_interrupted_attempts(self) -> tuple[RecoveredIllustrationAttempt, ...]:
        rows = self._session.execute(
            select(IllustrationAttemptRow, IllustrationScopeRow, IllustrationGenerationRow)
            .join(IllustrationScopeRow, IllustrationScopeRow.id == IllustrationAttemptRow.scope_id)
            .join(
                IllustrationGenerationRow,
                IllustrationGenerationRow.id == IllustrationScopeRow.generation_id,
            )
            .where(
                IllustrationAttemptRow.state.in_(
                    (
                        IllustrationAttemptState.RESERVED.value,
                        IllustrationAttemptState.SENT.value,
                    )
                )
            )
            .order_by(IllustrationAttemptRow.created_at, IllustrationAttemptRow.id)
        ).all()
        now = _now()
        recovered: list[RecoveredIllustrationAttempt] = []
        for attempt_row, scope_row, gen_row in rows:
            prev_state = IllustrationAttemptState(attempt_row.state)
            if prev_state is IllustrationAttemptState.RESERVED:
                new_state = IllustrationAttemptState.CANCELLED
                attempt_row.state = new_state.value
                attempt_row.outcome_json = {"reason_code": IllustrationReasonCode.INTERRUPTED.value}
                attempt_row.updated_at = now
                if gen_row.calls_reserved > 0:
                    gen_row.calls_reserved -= 1
                    gen_row.updated_at = now
                if scope_row.state == IllustrationScopeState.RUNNING.value:
                    scope_row.state = IllustrationScopeState.PENDING.value
                    scope_row.attempt_id = None
                    scope_row.updated_at = now
            else:
                new_state = IllustrationAttemptState.INTERRUPTED
                attempt_row.state = new_state.value
                attempt_row.outcome_json = {"reason_code": IllustrationReasonCode.INTERRUPTED.value}
                attempt_row.updated_at = now
                if scope_row.state == IllustrationScopeState.RUNNING.value:
                    scope_row.state = IllustrationScopeState.FAILED.value
                    scope_row.reason_code = IllustrationReasonCode.INTERRUPTED.value
                    scope_row.updated_at = now
                gen_row.updated_at = now
            recovered.append(
                RecoveredIllustrationAttempt(
                    attempt_id=attempt_row.id,
                    scope_id=scope_row.id,
                    generation_id=gen_row.id,
                    book_id=gen_row.book_id,
                    task_id=attempt_row.task_id,
                    previous_state=prev_state,
                    new_state=new_state,
                )
            )
        if recovered:
            self._session.flush()
        return tuple(recovered)

    def list_generation_attempt_ids(
        self,
        book_id: str,
        generation_id: str,
    ) -> tuple[tuple[str, str], ...]:
        self._require_generation(book_id, generation_id)
        rows = self._session.execute(
            select(IllustrationAttemptRow.id, IllustrationScopeRow.stage)
            .join(IllustrationScopeRow, IllustrationScopeRow.id == IllustrationAttemptRow.scope_id)
            .where(IllustrationScopeRow.generation_id == generation_id)
            .order_by(IllustrationAttemptRow.created_at, IllustrationAttemptRow.id)
        ).all()
        return tuple((attempt_id, stage) for attempt_id, stage in rows)

    def upsert_registry_items(
        self,
        generation_id: str,
        items: Sequence[RegistryItem],
    ) -> None:
        if not items:
            return
        gen = self._session.get(IllustrationGenerationRow, generation_id)
        if gen is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.GENERATION_NOT_FOUND,
                f"поколение иллюстраций {generation_id} не найдено",
            )
        for item in items:
            if item.generation_id != generation_id:
                raise IllustrationValidationError(
                    IllustrationReasonCode.INVALID_REQUEST,
                    "элемент справочника относится к другому поколению",
                )
            existing = self._session.get(
                IllustrationRegistryItemRow,
                (generation_id, item.kind.value, item.item_key),
            )
            if existing is None:
                self._session.add(
                    IllustrationRegistryItemRow(
                        generation_id=generation_id,
                        kind=item.kind.value,
                        item_key=item.item_key,
                        entity_id=item.entity_id,
                        position_key=item.position_key,
                        is_back_applicable=1 if item.is_back_applicable else 0,
                        payload_sha256=item.payload_sha256,
                        payload_json=thaw(item.payload),
                    )
                )
            else:
                existing.entity_id = item.entity_id
                existing.position_key = item.position_key
                existing.is_back_applicable = 1 if item.is_back_applicable else 0
                existing.payload_sha256 = item.payload_sha256
                existing.payload_json = thaw(item.payload)
        self._session.flush()

    def query_registry_items(
        self,
        generation_id: str,
        *,
        entity_id: str | None = None,
        kind: IllustrationRegistryKind | None = None,
        max_position_key: str | None = None,
        include_back_applicable: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[tuple[RegistryItem, ...], int]:
        conditions = [IllustrationRegistryItemRow.generation_id == generation_id]
        if entity_id is not None:
            conditions.append(IllustrationRegistryItemRow.entity_id == entity_id)
        if kind is not None:
            conditions.append(IllustrationRegistryItemRow.kind == IllustrationRegistryKind(kind).value)
        if max_position_key is not None:
            if include_back_applicable:
                conditions.append(
                    or_(
                        IllustrationRegistryItemRow.position_key <= max_position_key,
                        IllustrationRegistryItemRow.is_back_applicable == 1,
                    )
                )
            else:
                conditions.append(IllustrationRegistryItemRow.position_key <= max_position_key)

        total = int(
            self._session.scalar(
                select(func.count())
                .select_from(IllustrationRegistryItemRow)
                .where(*conditions)
            )
            or 0
        )
        rows = self._session.scalars(
            select(IllustrationRegistryItemRow)
            .where(*conditions)
            .order_by(
                IllustrationRegistryItemRow.position_key,
                IllustrationRegistryItemRow.kind,
                IllustrationRegistryItemRow.item_key,
            )
            .offset(offset)
            .limit(limit)
        ).all()
        return tuple(_to_registry_item(row) for row in rows), total


__all__ = ["SqlAlchemyIllustrationRepository"]
