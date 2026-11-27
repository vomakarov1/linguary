"""Persistence port for optional book illustration preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from linguary.application.ports.llm import LLMClient, LLMRequest
from linguary.domain.analysis import AnalysisPassResult
from linguary.domain.llm_log import LLMCallStage
from linguary.domain.illustrations import (
    ChapterHeaderRow,
    IllustrationAttempt,
    IllustrationAttemptState,
    IllustrationGeneration,
    IllustrationGenerationState,
    IllustrationGenerationSummary,
    IllustrationHead,
    IllustrationRegistryKind,
    IllustrationScope,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    IllustrationSettingsRevision,
    RecoveredIllustrationAttempt,
    RegistryItem,
)


class IllustrationRepository(Protocol):
    """Transactional storage contract for illustration settings, generations, scopes, and registry."""

    def save_settings(
        self,
        book_id: str,
        *,
        expected_revision: int,
        settings: IllustrationSettings,
    ) -> IllustrationSettingsRevision: ...

    def get_settings(
        self,
        book_id: str,
        settings_id: str,
    ) -> IllustrationSettingsRevision | None: ...

    def get_head(self, book_id: str) -> IllustrationHead: ...

    def create_or_get_generation(
        self,
        book_id: str,
        *,
        settings_id: str,
        request_key: str,
        source_sha256: str,
        source_authority: Mapping[str, Any],
        requested_sections: Sequence[str],
    ) -> tuple[IllustrationGeneration, bool]: ...

    def get_generation(
        self,
        book_id: str,
        generation_id: str,
    ) -> IllustrationGeneration | None: ...

    def list_active_generations(self) -> tuple[IllustrationGeneration, ...]: ...

    def list_history(
        self,
        book_id: str,
        *,
        limit: int = 10,
    ) -> tuple[IllustrationGenerationSummary, ...]: ...

    def update_generation_state(
        self,
        book_id: str,
        generation_id: str,
        *,
        state: IllustrationGenerationState,
        reason_code: str | None = None,
    ) -> IllustrationGeneration: ...

    def seal_manifest(
        self,
        book_id: str,
        generation_id: str,
        *,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
    ) -> IllustrationGeneration: ...

    def seal_registry(
        self,
        book_id: str,
        generation_id: str,
        *,
        registry_sha256: str,
    ) -> IllustrationGeneration: ...

    def union_requested_sections(
        self,
        book_id: str,
        generation_id: str,
        section_ids: Sequence[str],
    ) -> tuple[IllustrationGeneration, bool]: ...

    def revoke_generation(
        self,
        book_id: str,
        generation_id: str,
        *,
        reason_code: str,
    ) -> bool: ...

    def upsert_scopes(
        self,
        generation_id: str,
        scopes: Sequence[IllustrationScopeSpec],
    ) -> tuple[IllustrationScope, ...]: ...

    def get_scope(self, scope_id: str) -> IllustrationScope | None: ...

    def set_scope_task(self, scope_id: str, task_id: str | None) -> IllustrationScope: ...

    def list_scopes(
        self,
        generation_id: str,
        *,
        stage: IllustrationScopeStage | None = None,
        state: IllustrationScopeState | None = None,
        section_id: str | None = None,
    ) -> tuple[IllustrationScope, ...]: ...

    def list_scope_summaries(
        self,
        generation_id: str,
        *,
        stage: IllustrationScopeStage | None = None,
        state: IllustrationScopeState | None = None,
        section_id: str | None = None,
    ) -> tuple[IllustrationScope, ...]: ...

    def get_chapter_header_page(
        self,
        book_id: str,
        generation_id: str | None,
        *,
        offset: int,
        limit: int,
    ) -> tuple[tuple[ChapterHeaderRow, ...], int]: ...

    def find_reusable_scope(
        self,
        book_id: str,
        *,
        stage: IllustrationScopeStage,
        input_sha256: str,
    ) -> IllustrationScope | None: ...

    def find_reusable_registry(
        self,
        book_id: str,
        *,
        registry_sha256: str,
    ) -> str | None: ...

    def requeue_failed_scopes(
        self,
        book_id: str,
        generation_id: str,
        *,
        section_id: str | None = None,
    ) -> tuple[IllustrationScope, ...]: ...

    def claim_scope(
        self,
        scope_id: str,
        *,
        task_id: str,
        expected_generation_id: str,
        max_calls: int,
    ) -> IllustrationAttempt: ...

    def begin_next_attempt(
        self,
        scope_id: str,
        *,
        task_id: str,
        expected_generation_id: str,
        max_calls: int,
    ) -> IllustrationAttempt: ...

    def mark_attempt_sent(
        self,
        attempt_id: str,
        *,
        request_sha256: str,
        request_meta: Mapping[str, Any],
    ) -> IllustrationAttempt: ...

    def release_reserved_call(
        self,
        attempt_id: str,
        *,
        reason_code: str,
    ) -> None: ...

    def record_attempt_outcome(
        self,
        attempt_id: str,
        *,
        state: IllustrationAttemptState,
        outcome: Mapping[str, Any] | None,
    ) -> IllustrationAttempt: ...

    def split_scope(
        self,
        scope_id: str,
        *,
        attempt_id: str | None,
        children: Sequence[IllustrationScopeSpec],
        max_scopes: int,
    ) -> tuple[IllustrationScope, ...]: ...

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
    ) -> None: ...

    def fail_scope(
        self,
        scope_id: str,
        *,
        attempt_id: str | None = None,
        reason_code: str,
        outcome: Mapping[str, Any] | None = None,
    ) -> None: ...

    def recover_interrupted_attempts(self) -> tuple[RecoveredIllustrationAttempt, ...]: ...

    def list_generation_attempt_ids(
        self,
        book_id: str,
        generation_id: str,
    ) -> tuple[tuple[str, str], ...]: ...

    def upsert_registry_items(
        self,
        generation_id: str,
        items: Sequence[RegistryItem],
    ) -> None: ...

    def query_registry_items(
        self,
        generation_id: str,
        *,
        entity_id: str | None = None,
        kind: IllustrationRegistryKind | None = None,
        kinds: Sequence[IllustrationRegistryKind] | None = None,
        max_position_key: str | None = None,
        include_back_applicable: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[tuple[RegistryItem, ...], int]: ...


@dataclass(frozen=True, slots=True)
class RequestWireSnapshot:
    """Exact UTF-8 JSON wire size and digest of a serialized LLM request."""

    sha256: str
    size_bytes: int


class RequestWireSizer(Protocol):
    """Measures exact serialized wire bytes of an `LLMRequest` without network I/O."""

    def measure(self, request: LLMRequest[Any]) -> RequestWireSnapshot: ...


class IllustrationRuntimePort(Protocol):
    """Runtime port supplying bounded LLM clients and wire sizers for illustration stages."""

    def client_for(
        self, revision: IllustrationSettingsRevision, stage: LLMCallStage
    ) -> LLMClient: ...

    def sizer_for(
        self, revision: IllustrationSettingsRevision, stage: LLMCallStage
    ) -> RequestWireSizer: ...

    def is_provider_ready(self) -> bool: ...

    def default_model(self) -> str | None: ...

    def route_id(self) -> str: ...


__all__ = [
    "IllustrationRepository",
    "IllustrationRuntimePort",
    "RequestWireSizer",
    "RequestWireSnapshot",
]
