"""Adaptive output token budget calculation for unit translation calls."""

from __future__ import annotations

#: Share of `max_output_tokens` reserved for reasoning tokens when only `reasoning_effort`
#: is configured (integer percent).
REASONING_EFFORT_RESERVE_PCT: dict[str, int] = {
    "none": 0,
    "minimal": 15,
    "low": 35,
    "medium": 55,
    "high": 75,
}


def recommended_unit_max_completion_tokens(
    source_chars: int,
    *,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    reasoning_max_tokens: int | None,
) -> int:
    """Compute a safe `max_completion_tokens` ceiling for one translation unit."""
    visible = max(2048, int(source_chars * 1.35) + 1024)
    effort = reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) else None
    if effort is not None:
        reserve = REASONING_EFFORT_RESERVE_PCT.get(effort, 0)
        total = (visible * 100 + (99 - reserve)) // (100 - reserve) if reserve > 0 else visible
    elif reasoning_max_tokens is not None:
        total = visible + reasoning_max_tokens
    else:
        total = visible
    if max_output_tokens is not None:
        total = min(max_output_tokens, total)
        if effort is None and reasoning_max_tokens is not None and max_output_tokens > reasoning_max_tokens:
            total = max(total, reasoning_max_tokens + 1)
    return total


__all__ = [
    "REASONING_EFFORT_RESERVE_PCT",
    "recommended_unit_max_completion_tokens",
]
