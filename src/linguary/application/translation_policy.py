"""Supported translation rules, independent of book conditions and content plans."""

from dataclasses import dataclass
from types import MappingProxyType

from linguary.domain.book_conditions import WorkStateError

LEGACY_TRANSLATION_POLICY = "translation-context-legacy-v1"
TRANSLATION_CONTEXT_POLICY_V2 = "translation-context-v2"
TRANSLATION_RELIABILITY_POLICY_V3 = "translation-reliability-v3"
CURRENT_TRANSLATION_POLICY = "translation-reliability-v4"

_VALIDATION_POLICY_LEVELS = MappingProxyType({"v1": 1, "v2": 2, "v3": 3})


def validation_policy_level(policy: str) -> int:
    """Return numeric strictness level for a validation policy; fail closed on unknown values."""
    if not isinstance(policy, str) or policy not in _VALIDATION_POLICY_LEVELS:
        raise ValueError(f"неизвестная политика валидации: {policy!r}")
    return _VALIDATION_POLICY_LEVELS[policy]


class TranslationPolicyUnavailable(WorkStateError):
    """The operation's pinned rules cannot be executed."""

    reason = "conditions_unavailable"


@dataclass(frozen=True, slots=True)
class TranslationPolicy:
    version: str
    neighbours_per_side: int
    max_neighbour_chars: int | None
    capacity_retries_without_neighbours: int
    validation_policy: str = "v1"


# Legacy execution keeps its existing route-specific context and saved templates.
# Context limits apply equally to chapter translation and block regeneration.
# Strict acceptance is pinned on the operation policy, not on book conditions.
TRANSLATION_POLICIES = MappingProxyType({
    LEGACY_TRANSLATION_POLICY: TranslationPolicy(
        LEGACY_TRANSLATION_POLICY, 2, None, 0, validation_policy="v1",
    ),
    TRANSLATION_CONTEXT_POLICY_V2: TranslationPolicy(
        TRANSLATION_CONTEXT_POLICY_V2, 2, 8000, 1, validation_policy="v1",
    ),
    TRANSLATION_RELIABILITY_POLICY_V3: TranslationPolicy(
        TRANSLATION_RELIABILITY_POLICY_V3, 2, 8000, 1, validation_policy="v2",
    ),
    CURRENT_TRANSLATION_POLICY: TranslationPolicy(
        CURRENT_TRANSLATION_POLICY, 2, 8000, 1, validation_policy="v3",
    ),
})


def require_policy(version: str | None) -> TranslationPolicy:
    """Resolve an explicit version; absent or unsupported versions cannot resume work."""
    if version is None or version not in TRANSLATION_POLICIES:
        raise TranslationPolicyUnavailable("условия политики перевода недоступны")
    return TRANSLATION_POLICIES[version]
