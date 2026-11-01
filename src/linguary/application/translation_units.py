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
from math import ceil
from typing import TypeVar

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
PARAGRAPH_FRAGMENT_MIN_CHARS = 250
PARAGRAPH_FRAGMENT_TARGET_CHARS = 300
PARAGRAPH_FRAGMENT_MAX_CHARS = 350

_FragmentT = TypeVar("_FragmentT")

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


def group_paragraph_fragments(
    fragments: Sequence[_FragmentT],
    *,
    text_of: Callable[[_FragmentT], str],
    target_chars: int = PARAGRAPH_FRAGMENT_TARGET_CHARS,
) -> tuple[tuple[_FragmentT, ...], ...]:
    """Balance fragments into readable paragraph-sized groups.

    EPUB import stores long paragraphs as source fragments so they can be processed safely.
    Presentation joins adjacent fragments into groups near `target_chars`, without capping the
    number of groups and while keeping source fragments intact.
    """
    values = tuple(fragments)
    if len(values) <= 1:
        return (values,) if values else ()
    if target_chars < 1:
        raise ValueError("целевой размер группы должен быть положительным")

    lengths = tuple(len(text_of(fragment)) for fragment in values)
    total = sum(lengths)
    if total == 0:
        return (values,)
    min_count = min(
        len(values), max(2, ceil(total / PARAGRAPH_FRAGMENT_MAX_CHARS))
    )
    max_count = min(
        len(values), max(2, total // PARAGRAPH_FRAGMENT_MIN_CHARS)
    )
    counts = (
        range(min_count, max_count + 1)
        if min_count <= max_count
        else {min_count, max_count}
    )
    group_count = min(
        counts, key=lambda count: (abs(total / count - target_chars), -count)
    )
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)

    groups: list[tuple[_FragmentT, ...]] = []
    start = 0
    for group_index in range(group_count - 1):
        remaining_groups = group_count - group_index - 1
        last_end = len(values) - remaining_groups
        target_boundary = total * (group_index + 1) / group_count
        end = min(
            range(start + 1, last_end + 1),
            key=lambda candidate: abs(boundaries[candidate] - target_boundary),
        )
        groups.append(values[start:end])
        start = end
    groups.append(values[start:])
    return tuple(groups)


def paragraph_fragment_label(index: int, total: int, language: str) -> str:
    """Short ordinal label shown in the source language."""
    language = language.lower().replace("_", "-").split("-", 1)[0]
    if language == "ru":
        return f"Часть {index} из {total}"
    if language == "en":
        return f"Part {index} of {total}"
    return f"{index} / {total}"


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
    "group_paragraph_fragments",
    "paragraph_fragment_label",
    "plan_translation_units",
]
