"""Domain entities, states, invariants, and errors for optional book illustration prompts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from linguary.domain.book_conditions import freeze, thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.errors import DomainError

ILLUSTRATION_STYLE_MONOCHROME_INK = "monochrome_ink"
ILLUSTRATION_SUPPORTED_STYLES = frozenset({ILLUSTRATION_STYLE_MONOCHROME_INK})
ILLUSTRATION_POLICY_VERSION = "illustration-v1"
ILLUSTRATION_MODEL_MAX_LENGTH = 200
ILLUSTRATION_MIN_PER_CHAPTER = 1
ILLUSTRATION_MAX_PER_CHAPTER = 3
TIMELESS_POSITION_KEY = "000000:000000:00000000"
_MAX_POSITION_ORDER = 999_999
_MAX_POSITION_CHAR = 99_999_999


class IllustrationGenerationState(StrEnum):
    """Persisted lifecycle states of an illustration generation."""

    WAITING_SOURCE = "waiting_source"
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"
    REVOKED = "revoked"


class IllustrationScopeStage(StrEnum):
    """Persisted execution stages of an illustration scope."""

    EXTRACT = "extract"
    ALIAS = "alias"
    DESIGN = "design"
    FACT_SCOPE = "fact_scope"
    CANDIDATES = "candidates"
    SELECTION = "selection"
    PROMPT = "prompt"
    CHAPTER = "chapter"


class IllustrationScopeState(StrEnum):
    """Persisted lifecycle states of an illustration scope."""

    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    FAILED = "failed"
    SKIPPED = "skipped"
    SPLIT = "split"


class IllustrationAttemptState(StrEnum):
    """Persisted states of an individual physical call attempt."""

    RESERVED = "reserved"
    SENT = "sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    STALE = "stale"
    CANCELLED = "cancelled"


class IllustrationRegistryKind(StrEnum):
    """Kinds of entries stored in `illustration_registry_items`."""

    ENTITY = "entity"
    FACT = "fact"
    ALIAS_DECISION = "alias_decision"
    DESIGN = "design"


class IllustrationReasonCode(StrEnum):
    """Closed set of 28 safe illustration reason codes."""

    BOOK_NOT_FOUND = "book_not_found"
    SECTION_NOT_FOUND = "section_not_found"
    GENERATION_NOT_FOUND = "generation_not_found"
    INVALID_REQUEST = "invalid_request"
    REVISION_CONFLICT = "revision_conflict"
    GENERATION_CONFLICT = "generation_conflict"
    NOTHING_TO_RETRY = "nothing_to_retry"
    SOURCE_INCOMPATIBLE = "source_incompatible"
    SOURCE_CHANGED = "source_changed"
    NARRATIVE_MISSING = "narrative_missing"
    NARRATIVE_FAILED = "narrative_failed"
    NON_CONTENT_SECTION = "non_content_section"
    NO_CONTENT = "no_content"
    SETTINGS_CHANGED = "settings_changed"
    CHANNEL_DISABLED = "channel_disabled"
    BOOK_PAUSED = "book_paused"
    ROUTE_UNAVAILABLE = "route_unavailable"
    CONTEXT_LIMIT = "context_limit"
    ATOMIC_INPUT_TOO_LARGE = "atomic_input_too_large"
    SPLIT_DEPTH_EXHAUSTED = "split_depth_exhausted"
    CALL_LIMIT_EXHAUSTED = "call_limit_exhausted"
    SCOPE_LIMIT_EXHAUSTED = "scope_limit_exhausted"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    PROVIDER_TRUNCATED = "provider_truncated"
    PROVIDER_ERROR = "provider_error"
    INTERRUPTED = "interrupted"
    STALE_ATTEMPT = "stale_attempt"
    STORAGE_UNAVAILABLE = "storage_unavailable"


#: Deterministic scope failures that cannot be fixed by retrying the same generation.
DETERMINISTIC_NON_RETRYABLE_REASONS: frozenset[str] = frozenset(
    {
        IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
        IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
        IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED.value,
    }
)


class IllustrationError(DomainError):
    """Typed illustration error carrying a closed safe `reason_code`."""

    def __init__(self, reason_code: str | IllustrationReasonCode, message: str = "") -> None:
        code = reason_code.value if isinstance(reason_code, IllustrationReasonCode) else str(reason_code)
        if code not in {member.value for member in IllustrationReasonCode}:
            raise ValueError(f"неизвестный код причины иллюстраций: {code!r}")
        self.reason_code = code
        super().__init__(message or code)


class IllustrationNotFoundError(IllustrationError):
    """Requested book, section, or illustration generation does not exist."""


class IllustrationConflictError(IllustrationError):
    """State, revision, generation, or precondition conflict."""


class IllustrationValidationError(IllustrationError):
    """Invalid input or illustration domain invariant violation."""


class IllustrationOwnershipError(IllustrationError):
    """Concurrent claim or stale ownership check failed."""


class IllustrationLimitError(IllustrationError):
    """Call, scope, or split depth limit exceeded."""


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def normalize_model_name(value: object) -> str | None:
    """Validate and normalize an optional model identifier."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "имя модели должно быть строкой",
        )
    if _has_control_chars(value):
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "имя модели содержит управляющие символы",
        )
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > ILLUSTRATION_MODEL_MAX_LENGTH:
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            f"имя модели не должно превышать {ILLUSTRATION_MODEL_MAX_LENGTH} символов",
        )
    return stripped


def format_position_key(section_order: int, block_order: int, char_start: int) -> str:
    """Build fixed-width numerically sortable position key (`%06d:%06d:%08d`)."""
    if (
        type(section_order) is not int
        or type(block_order) is not int
        or type(char_start) is not int
    ):
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "координаты позиции должны быть целыми числами",
        )
    if not (0 <= section_order <= _MAX_POSITION_ORDER):
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "номер раздела вне допустимого диапазона позиции",
        )
    if not (0 <= block_order <= _MAX_POSITION_ORDER):
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "номер блока вне допустимого диапазона позиции",
        )
    if not (0 <= char_start <= _MAX_POSITION_CHAR):
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            "смещение символа вне допустимого диапазона позиции",
        )
    return f"{section_order:06d}:{block_order:06d}:{char_start:08d}"


@dataclass(frozen=True, slots=True)
class IllustrationPolicy:
    """Hidden execution policy persisted verbatim in every settings revision."""

    max_request_bytes: int = 24576
    repair_reserve_bytes: int = 1200
    max_completion_tokens: int = 4096
    max_split_depth: int = 8
    max_calls_per_generation: int = 2048
    max_scopes_per_generation: int = 8192
    max_calls_per_scope_activation: int = 3
    max_repair_turns: int = 1
    identity_candidate_page_size: int = 16
    max_candidates_per_scene: int = 2
    selection_fan_in: int = 8
    temperature: float = 0.2
    policy_version: str = ILLUSTRATION_POLICY_VERSION

    def __post_init__(self) -> None:
        positive_ints = (
            self.max_request_bytes,
            self.repair_reserve_bytes,
            self.max_completion_tokens,
            self.max_split_depth,
            self.max_calls_per_generation,
            self.max_scopes_per_generation,
            self.max_calls_per_scope_activation,
            self.identity_candidate_page_size,
            self.max_candidates_per_scene,
            self.selection_fan_in,
        )
        if any(type(v) is not int or v <= 0 for v in positive_ints):
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "параметры политики иллюстраций должны быть положительными целыми числами",
            )
        if type(self.max_repair_turns) is not int or self.max_repair_turns < 0:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "число раундов исправления не может быть отрицательным",
            )
        if self.repair_reserve_bytes >= self.max_request_bytes:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "резерв на исправление должен быть меньше лимита запроса",
            )
        if type(self.temperature) not in (int, float) or not (0.0 <= float(self.temperature) <= 2.0):
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "некорректная температура политики иллюстраций",
            )
        if not self.policy_version.strip():
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "версия политики иллюстраций не может быть пустой",
            )

    @property
    def initial_target_bytes(self) -> int:
        return self.max_request_bytes - self.repair_reserve_bytes

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "max_request_bytes": self.max_request_bytes,
            "repair_reserve_bytes": self.repair_reserve_bytes,
            "max_completion_tokens": self.max_completion_tokens,
            "max_split_depth": self.max_split_depth,
            "max_calls_per_generation": self.max_calls_per_generation,
            "max_scopes_per_generation": self.max_scopes_per_generation,
            "max_calls_per_scope_activation": self.max_calls_per_scope_activation,
            "max_repair_turns": self.max_repair_turns,
            "identity_candidate_page_size": self.identity_candidate_page_size,
            "max_candidates_per_scene": self.max_candidates_per_scene,
            "selection_fan_in": self.selection_fan_in,
            "temperature": self.temperature,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> IllustrationPolicy:
        if not payload:
            return DEFAULT_ILLUSTRATION_POLICY
        return cls(
            policy_version=str(payload.get("policy_version", ILLUSTRATION_POLICY_VERSION)),
            max_request_bytes=payload.get("max_request_bytes", 24576),
            repair_reserve_bytes=payload.get("repair_reserve_bytes", 1200),
            max_completion_tokens=payload.get("max_completion_tokens", 4096),
            max_split_depth=payload.get("max_split_depth", 8),
            max_calls_per_generation=payload.get("max_calls_per_generation", 2048),
            max_scopes_per_generation=payload.get("max_scopes_per_generation", 8192),
            max_calls_per_scope_activation=payload.get("max_calls_per_scope_activation", 3),
            max_repair_turns=payload.get("max_repair_turns", 1),
            identity_candidate_page_size=payload.get("identity_candidate_page_size", 16),
            max_candidates_per_scene=payload.get("max_candidates_per_scene", 2),
            selection_fan_in=payload.get("selection_fan_in", 8),
            temperature=payload.get("temperature", 0.2),
        )


DEFAULT_ILLUSTRATION_POLICY = IllustrationPolicy()


@dataclass(frozen=True, slots=True)
class IllustrationSettings:
    """User-configurable settings for a book's illustration channel."""

    enabled: bool = False
    max_per_chapter: int = ILLUSTRATION_MAX_PER_CHAPTER
    style: str = ILLUSTRATION_STYLE_MONOCHROME_INK
    model_reference: str | None = None
    model_prompts: str | None = None
    policy: IllustrationPolicy = DEFAULT_ILLUSTRATION_POLICY

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "признак включения должен быть логическим значением",
            )
        if (
            type(self.max_per_chapter) is not int
            or not (ILLUSTRATION_MIN_PER_CHAPTER <= self.max_per_chapter <= ILLUSTRATION_MAX_PER_CHAPTER)
        ):
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "лимит иллюстраций на главу должен быть целым числом от 1 до 3",
            )
        if not isinstance(self.style, str) or self.style not in ILLUSTRATION_SUPPORTED_STYLES:
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "неподдерживаемый стиль иллюстраций",
            )
        model_reference = normalize_model_name(self.model_reference)
        model_prompts = normalize_model_name(self.model_prompts)
        object.__setattr__(self, "model_reference", model_reference)
        object.__setattr__(self, "model_prompts", model_prompts)
        if self.enabled and (model_reference is None or model_prompts is None):
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "при включении канала необходимо указать обе модели",
            )
        if not isinstance(self.policy, IllustrationPolicy):
            raise IllustrationValidationError(
                IllustrationReasonCode.INVALID_REQUEST,
                "некорректная политика иллюстраций",
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "max_per_chapter": self.max_per_chapter,
            "style": self.style,
            "model_reference": self.model_reference,
            "model_prompts": self.model_prompts,
            "languages": {"descriptions": "ru", "prompts": "en"},
            "policy": self.policy.to_payload(),
        }

    def sha256(self) -> str:
        return canonical_sha(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IllustrationSettings:
        policy_raw = payload.get("policy")
        policy = (
            IllustrationPolicy.from_payload(policy_raw)
            if isinstance(policy_raw, Mapping)
            else DEFAULT_ILLUSTRATION_POLICY
        )
        return cls(
            enabled=payload.get("enabled", False),
            max_per_chapter=payload.get("max_per_chapter", ILLUSTRATION_MAX_PER_CHAPTER),
            style=payload.get("style", ILLUSTRATION_STYLE_MONOCHROME_INK),
            model_reference=payload.get("model_reference"),
            model_prompts=payload.get("model_prompts"),
            policy=policy,
        )


DEFAULT_ILLUSTRATION_SETTINGS = IllustrationSettings()


@dataclass(frozen=True, slots=True)
class IllustrationSettingsRevision:
    """Immutable saved revision of a book's illustration settings."""

    id: str
    book_id: str
    revision: int
    settings_sha256: str
    settings: IllustrationSettings
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IllustrationHead:
    """Current pointer to active settings revision and active generation for a book."""

    book_id: str
    settings_id: str | None = None
    active_generation_id: str | None = None
    settings_revision: IllustrationSettingsRevision | None = None

    @property
    def revision_number(self) -> int:
        return 0 if self.settings_revision is None else self.settings_revision.revision

    @property
    def effective_settings(self) -> IllustrationSettings:
        if self.settings_revision is None:
            return DEFAULT_ILLUSTRATION_SETTINGS
        return self.settings_revision.settings


@dataclass(frozen=True, slots=True)
class IllustrationGeneration:
    """Persisted generation of illustration preparation for a book."""

    id: str
    book_id: str
    generation_number: int
    settings_id: str
    request_key: str
    source_sha256: str
    source_authority: Mapping[str, Any]
    requested_sections: tuple[str, ...]
    manifest: Mapping[str, Any] | None
    manifest_sha256: str | None
    registry_sha256: str | None
    state: IllustrationGenerationState
    reason_code: str | None
    calls_reserved: int
    calls_sent: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_authority", freeze(self.source_authority))
        object.__setattr__(self, "requested_sections", tuple(self.requested_sections))
        if self.manifest is not None:
            object.__setattr__(self, "manifest", freeze(self.manifest))
        object.__setattr__(self, "state", IllustrationGenerationState(self.state))


@dataclass(frozen=True, slots=True)
class IllustrationGenerationSummary:
    """Summary entry for generation history listing."""

    generation_id: str
    generation_number: int
    state: IllustrationGenerationState
    settings_revision: int
    source_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", IllustrationGenerationState(self.state))


_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class IllustrationScopeSpec:
    """Input descriptor for creating or splitting scopes."""

    stage: IllustrationScopeStage
    scope_key: str
    input_sha256: str
    section_id: str | None = None
    section_order: int | None = None
    section_title: str | None = None
    parent_scope_id: str | None = None
    summary: Mapping[str, Any] = _EMPTY_MAPPING
    state: IllustrationScopeState = IllustrationScopeState.PENDING
    reason_code: str | None = None
    payload: Mapping[str, Any] | None = None
    output_sha256: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", IllustrationScopeStage(self.stage))
        object.__setattr__(self, "state", IllustrationScopeState(self.state))
        object.__setattr__(self, "summary", freeze(self.summary))
        if self.payload is not None:
            object.__setattr__(self, "payload", freeze(self.payload))


@dataclass(frozen=True, slots=True)
class IllustrationScope:
    """Persisted unit of work inside an illustration generation."""

    id: str
    generation_id: str
    parent_scope_id: str | None
    stage: IllustrationScopeStage
    scope_key: str
    section_id: str | None
    section_order: int | None
    section_title: str | None
    task_id: str | None
    input_sha256: str
    state: IllustrationScopeState
    reason_code: str | None
    attempt_id: str | None
    output_sha256: str | None
    payload: Mapping[str, Any] | None
    summary: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", IllustrationScopeStage(self.stage))
        object.__setattr__(self, "state", IllustrationScopeState(self.state))
        if self.payload is not None:
            object.__setattr__(self, "payload", freeze(self.payload))
        object.__setattr__(self, "summary", freeze(self.summary))

    @property
    def is_retryable_failure(self) -> bool:
        return (
            self.state is IllustrationScopeState.FAILED
            and self.reason_code not in DETERMINISTIC_NON_RETRYABLE_REASONS
        )


@dataclass(frozen=True, slots=True)
class IllustrationAttempt:
    """Persisted physical call attempt bound to a single scope."""

    id: str
    scope_id: str
    task_id: str
    ordinal: int
    state: IllustrationAttemptState
    request_sha256: str | None
    request_meta: Mapping[str, Any]
    outcome: Mapping[str, Any] | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", IllustrationAttemptState(self.state))
        object.__setattr__(self, "request_meta", freeze(self.request_meta))
        if self.outcome is not None:
            object.__setattr__(self, "outcome", freeze(self.outcome))


@dataclass(frozen=True, slots=True)
class RecoveredIllustrationAttempt:
    """Summary of an attempt transitioned during startup crash recovery."""

    attempt_id: str
    scope_id: str
    generation_id: str
    book_id: str
    task_id: str
    previous_state: IllustrationAttemptState
    new_state: IllustrationAttemptState


@dataclass(frozen=True, slots=True)
class ChapterHeaderRow:
    """Paginated chapter header returned by repository without loading full payloads."""

    section_id: str
    order: int
    title: str
    state: str
    reason_code: str | None
    moment_count: int | None
    scope_id: str | None = None


@dataclass(frozen=True, slots=True)
class RegistryItem:
    """Entry in the book-wide illustration registry."""

    generation_id: str
    kind: IllustrationRegistryKind
    item_key: str
    entity_id: str
    position_key: str
    is_back_applicable: bool
    payload_sha256: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", IllustrationRegistryKind(self.kind))
        object.__setattr__(self, "is_back_applicable", bool(self.is_back_applicable))
        object.__setattr__(self, "payload", freeze(self.payload))

    def raw_payload(self) -> dict[str, Any]:
        return thaw(self.payload)


__all__ = [
    "DEFAULT_ILLUSTRATION_POLICY",
    "DEFAULT_ILLUSTRATION_SETTINGS",
    "DETERMINISTIC_NON_RETRYABLE_REASONS",
    "ILLUSTRATION_MAX_PER_CHAPTER",
    "ILLUSTRATION_MIN_PER_CHAPTER",
    "ILLUSTRATION_MODEL_MAX_LENGTH",
    "ILLUSTRATION_POLICY_VERSION",
    "ILLUSTRATION_STYLE_MONOCHROME_INK",
    "ILLUSTRATION_SUPPORTED_STYLES",
    "TIMELESS_POSITION_KEY",
    "ChapterHeaderRow",
    "IllustrationAttempt",
    "IllustrationAttemptState",
    "IllustrationConflictError",
    "IllustrationError",
    "IllustrationGeneration",
    "IllustrationGenerationState",
    "IllustrationGenerationSummary",
    "IllustrationHead",
    "IllustrationLimitError",
    "IllustrationNotFoundError",
    "IllustrationOwnershipError",
    "IllustrationPolicy",
    "IllustrationReasonCode",
    "IllustrationRegistryKind",
    "IllustrationScope",
    "IllustrationScopeSpec",
    "IllustrationScopeStage",
    "IllustrationScopeState",
    "IllustrationSettings",
    "IllustrationSettingsRevision",
    "IllustrationValidationError",
    "RecoveredIllustrationAttempt",
    "RegistryItem",
    "format_position_key",
    "normalize_model_name",
]
