"""Adaptive output token budget and narrative fragment sizing calculations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Share of `max_output_tokens` reserved for reasoning tokens when only `reasoning_effort`
#: is configured (integer percent).
REASONING_EFFORT_RESERVE_PCT: dict[str, int] = {
    "none": 0,
    "minimal": 15,
    "low": 35,
    "medium": 55,
    "high": 75,
}

#: Minimum reasoning token floor added to visible output budget per reasoning effort level.
REASONING_EFFORT_MIN_TOKENS: dict[str, int] = {
    "none": 0,
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 12288,
}

_DEFAULT_MAX_FRAGMENT_BLOCKS = 160
_TOKENS_PER_ANALYSED_BLOCK = 200
_DEFAULT_ANALYSIS_OUTPUT_BUDGET = 32_768


def _budget_after_reasoning(
    total_output_tokens: int,
    *,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
) -> int:
    if reasoning_max_tokens is not None:
        return max(1, total_output_tokens - reasoning_max_tokens)
    if reasoning_effort is not None:
        effort = reasoning_effort.strip().lower()
        reserve_pct = REASONING_EFFORT_RESERVE_PCT.get(effort, 0)
        return max(1, (total_output_tokens * (100 - reserve_pct)) // 100)
    return total_output_tokens


def effective_output_token_budget_for(
    max_output_tokens: int | None,
    *,
    default_max_output_tokens: int = _DEFAULT_ANALYSIS_OUTPUT_BUDGET,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
) -> int:
    """Visible output token budget using `default_max_output_tokens` when unset."""
    total = max_output_tokens if max_output_tokens is not None else default_max_output_tokens
    return _budget_after_reasoning(
        total,
        reasoning_effort=reasoning_effort,
        reasoning_max_tokens=reasoning_max_tokens,
    )


def narrative_max_fragment_blocks_for(llm_parameters: Mapping[str, Any]) -> int:
    """Pure equivalent of `AnalyzeChapter._narrative_max_fragment_blocks()` from saved conditions."""
    raw_max_fragment = llm_parameters.get("max_fragment_blocks")
    raw_max_output = llm_parameters.get("max_output_tokens")
    raw_effort = llm_parameters.get("reasoning_effort")
    raw_reasoning_max = llm_parameters.get("reasoning_max_tokens")

    max_output = raw_max_output if type(raw_max_output) is int else None
    effort = raw_effort.strip().lower() if isinstance(raw_effort, str) and raw_effort.strip() else None
    reasoning_max = raw_reasoning_max if type(raw_reasoning_max) is int else None

    if type(raw_max_fragment) is int and raw_max_fragment >= 1:
        effective_max_fragment_blocks = raw_max_fragment
    elif max_output is None and (effort is not None or reasoning_max is not None):
        budget_for_frag = effective_output_token_budget_for(
            None,
            default_max_output_tokens=_DEFAULT_ANALYSIS_OUTPUT_BUDGET,
            reasoning_effort=effort,
            reasoning_max_tokens=reasoning_max,
        )
        effective_max_fragment_blocks = max(1, budget_for_frag // _TOKENS_PER_ANALYSED_BLOCK)
    elif max_output is None:
        effective_max_fragment_blocks = _DEFAULT_MAX_FRAGMENT_BLOCKS
    else:
        visible_budget = _budget_after_reasoning(
            max_output,
            reasoning_effort=effort,
            reasoning_max_tokens=reasoning_max,
        )
        effective_max_fragment_blocks = max(1, visible_budget // _TOKENS_PER_ANALYSED_BLOCK)

    output_budget_32k = effective_output_token_budget_for(
        max_output,
        default_max_output_tokens=_DEFAULT_ANALYSIS_OUTPUT_BUDGET,
        reasoning_effort=effort,
        reasoning_max_tokens=reasoning_max,
    )
    return min(effective_max_fragment_blocks, max(20, output_budget_32k // 650))


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
        min_reasoning = REASONING_EFFORT_MIN_TOKENS.get(effort, 0)
        pct_total = (visible * 100 + (99 - reserve)) // (100 - reserve) if reserve > 0 else visible
        total = max(pct_total, visible + min_reasoning)
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
    "REASONING_EFFORT_MIN_TOKENS",
    "REASONING_EFFORT_RESERVE_PCT",
    "effective_output_token_budget_for",
    "narrative_max_fragment_blocks_for",
    "recommended_unit_max_completion_tokens",
]
