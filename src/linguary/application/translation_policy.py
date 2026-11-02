"""Supported translation rules, independent of book conditions and content plans."""

from dataclasses import dataclass
from types import MappingProxyType

from linguary.domain.book_conditions import WorkStateError

LEGACY_TRANSLATION_POLICY = "translation-context-legacy-v1"
CURRENT_TRANSLATION_POLICY = "translation-context-v2"


class TranslationPolicyUnavailable(WorkStateError):
    """The operation's pinned rules cannot be executed."""

    reason = "conditions_unavailable"


@dataclass(frozen=True, slots=True)
class TranslationPolicy:
    version: str
    neighbours_per_side: int
    max_neighbour_chars: int | None
    capacity_retries_without_neighbours: int


# Legacy execution keeps its existing route-specific context and saved templates.
# Limits in v2 apply equally to chapter translation and block regeneration.
TRANSLATION_POLICIES = MappingProxyType({
    LEGACY_TRANSLATION_POLICY: TranslationPolicy(LEGACY_TRANSLATION_POLICY, 2, None, 0),
    CURRENT_TRANSLATION_POLICY: TranslationPolicy(CURRENT_TRANSLATION_POLICY, 2, 8000, 1),
})


def require_policy(version: str | None) -> TranslationPolicy:
    """Resolve an explicit version; absent or unsupported versions cannot resume work."""
    if version not in TRANSLATION_POLICIES:
        raise TranslationPolicyUnavailable("условия политики перевода недоступны")
    return TRANSLATION_POLICIES[version]
