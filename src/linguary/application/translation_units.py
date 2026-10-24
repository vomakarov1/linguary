"""Formation of `TranslationUnit`: grouping blocks sent to the model in a single call (PRD § 10, ALG-04).

A unit is a compute boundary rather than a storage unit: the model receives it whole for context,
but returns an individual result for each input block, and block results are persisted (ADR § 3).
The responsibility of this module is to determine unit boundaries.

Unit size is measured in characters of source text rather than block count. Processing exhibits
a different cost profile than chapter analysis: analysis returns a treatment mode and concise rationale,
meaning roughly constant token consumption per block regardless of length, whereas processing
returns full translations proportional to original length. Forty dialogue lines and forty full-page
paragraphs represent requests of vastly different cost; hence `analysis_merge.DEFAULT_MAX_FRAGMENT_BLOCKS`
does not apply here. Processing defines its own limit `DEFAULT_MAX_UNIT_SOURCE_CHARS`, configured
via `LLMSettings.max_unit_source_chars`.

`PRESERVE_ONLY` and `DROP_WITH_WARNING` blocks do not enter units and do not fragment adjacent context:
they remain the concern of importer and exporter (PRD § 10.1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from linguary.application.context_builder import is_textual
from linguary.domain import (
    BlockType,
    DocumentBlock,
    TranslationUnit,
    TranslationUnitError,
    TreatmentPlan,
)

#: Number of source characters sent in a single processing request. PRD § 10 targets
#: 2000-5000 source tokens; 8000 characters is ~2000 English tokens (~4 chars/token),
#: the lower bound of that target. Output fits within typical structured response limits:
#: a full Russian translation of that volume with vocabulary hints, language notes, and
#: JSON wrapping consumes ~5000-6000 tokens.
DEFAULT_MAX_UNIT_SOURCE_CHARS = 8000

#: Block types occurring contiguously as a cohesive whole: list item, poem stanza, quote, table.
#: Splitting a list midway deprives the model of its start, while poetry loses its stanza context.
_COHESIVE_BLOCK_TYPES = frozenset(
    {BlockType.LIST_ITEM, BlockType.POEM, BlockType.QUOTE, BlockType.TABLE}
)


def plan_translation_units(
    blocks: Iterable[DocumentBlock],
    plan: TreatmentPlan,
    *,
    id_factory: Callable[[], str],
    max_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
) -> tuple[TranslationUnit, ...]:
    """Splits chapter blocks into groups for model calls (PRD § 10).

    Accepts chapter blocks in document order; non-textual blocks are filtered out
    (`context_builder.is_textual`), and treatments are retrieved from the saved plan.
    A textual block absent from the plan is not silently skipped: processing starts
    only after full chapter analysis, so an unlisted block raises `UnknownBlockError`.

    A chapter with no textual blocks (cover, illustration page) yields an empty result:
    nothing to process, no model calls.
    """
    if max_source_chars < 1:
        raise TranslationUnitError("предел размера группы блоков должен быть положительным")
    ordered = tuple(
        block
        for block in blocks
        if is_textual(block) and plan.treatment_for(block.id).is_textual
    )
    if not ordered:
        return ()
    return tuple(
        TranslationUnit.of_blocks(id=id_factory(), blocks=group, plan=plan)
        for group in _grouped(ordered, max_source_chars=max_source_chars)
    )


def _grouped(
    blocks: Sequence[DocumentBlock], *, max_source_chars: int
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Group boundaries: cohesive runs intact, sizing divided into balanced chapter portions.

    The chapter is not partitioned into maximal-size chunks, but into `ceil(total / limit)`
    evenly sized portions: a one-paragraph tail would cost an extra call without scene context.
    The limit remains strict — a portion never exceeds it, unless a single block exceeds the
    limit: `DocumentBlock` is indivisible (ADR § 3).
    """
    total = sum(len(block.text) for block in blocks)
    count = max(1, -(-total // max_source_chars))
    groups: list[list[DocumentBlock]] = []
    current: list[DocumentBlock] = []
    current_size = 0
    placed = 0
    for cohesive in _cohesive_runs(blocks, max_source_chars=max_source_chars):
        size = sum(len(block.text) for block in cohesive)
        if current and _closes_group(
            position=placed + current_size,
            size=size,
            current_size=current_size,
            filled=len(groups),
            count=count,
            total=total,
            max_source_chars=max_source_chars,
        ):
            groups.append(current)
            placed += current_size
            current, current_size = [], 0
        current.extend(cohesive)
        current_size += size
    groups.append(current)
    return tuple(tuple(group) for group in groups)


def _closes_group(
    *,
    position: int,
    size: int,
    current_size: int,
    filled: int,
    count: int,
    total: int,
    max_source_chars: int,
) -> bool:
    """Whether the current group should close before the next cohesive run.

    Two rules: strict request size limit and proximity to the balanced chapter partition
    boundary. The second rule prevents the chapter tail from becoming a tiny request — a chunk
    is appended to the group as long as it brings it closer to the partition boundary rather
    than further away.
    """
    if current_size + size > max_source_chars:
        return True
    if filled >= count - 1:
        return False
    boundary = total * (filled + 1) / count
    return abs(position + size - boundary) > abs(position - boundary)


def _cohesive_runs(
    blocks: Sequence[DocumentBlock], *, max_source_chars: int
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Chapter chunks that should not be split across requests (PRD § 10).

    Adjacent blocks sharing a parent container and contiguous blocks of lists, poetry,
    quotes, or tables are considered cohesive. Dialogue and logically linked paragraphs
    lack structural markers: block order within the group holds them together. A chunk
    exceeding the request limit on its own is broken into individual blocks: otherwise
    it would drag the group past the size limit.
    """
    runs: list[list[DocumentBlock]] = []
    for block in blocks:
        if runs and _is_cohesive_with(runs[-1][-1], block):
            runs[-1].append(block)
        else:
            runs.append([block])
    return tuple(
        run
        for group in runs
        for run in _split_oversized(group, max_source_chars=max_source_chars)
    )


def _is_cohesive_with(previous: DocumentBlock, block: DocumentBlock) -> bool:
    """Whether adjacent blocks form a single cohesive unit."""
    prev_split = previous.attributes.get("split_group")
    if prev_split and prev_split == block.attributes.get("split_group"):
        return True
    if previous.parent_id is not None and previous.parent_id == block.parent_id:
        return True
    return previous.type is block.type and block.type in _COHESIVE_BLOCK_TYPES


def _split_oversized(
    run: Sequence[DocumentBlock], *, max_source_chars: int
) -> tuple[tuple[DocumentBlock, ...], ...]:
    """Cohesive run as a whole or, if exceeding the limit, decomposed into individual blocks."""
    if sum(len(block.text) for block in run) <= max_source_chars:
        return (tuple(run),)
    return tuple((block,) for block in run)


__all__ = [
    "DEFAULT_MAX_UNIT_SOURCE_CHARS",
    "plan_translation_units",
]
