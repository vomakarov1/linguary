"""Display parts composed from segments that already exist on a block.

Composition is a pure projection: it does not import text, does not persist parts,
and does not change block identity. A part boundary follows a treatment change.
A run that is not `TRANSLATE` stays one part. A `TRANSLATE` run is split by one
dynamic-programming pass over segment boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from linguary.domain import BlockSegment, BlockTreatment, DocumentBlock

PART_TARGET_CHARS = 300
PART_MIN_CHARS = 250
PART_MAX_CHARS = 350


@dataclass(frozen=True, slots=True)
class LayoutPart:
    """One display part: contiguous segment indexes sharing a treatment."""

    start: int
    end: int
    treatment: BlockTreatment | None
    segments: tuple[int, ...]


def compose_parts(
    block: DocumentBlock,
    treatments: Sequence[BlockTreatment | None],
    *,
    target_chars: int = PART_TARGET_CHARS,
    min_chars: int = PART_MIN_CHARS,
    max_chars: int = PART_MAX_CHARS,
) -> tuple[LayoutPart, ...]:
    """Project `block.segment_spans` into display parts.

    `treatments` has one entry per span, in that order. `None` matches only `None`.
    Part cost is `(length - target_chars) ** 2`. A part longer than `max_chars` is
    forbidden unless it is exactly one segment. A part shorter than `min_chars` is
    legal; that distance from the target is its cost. Equal totals prefer fewer
    parts, then the earlier cut.
    """
    spans = block.segment_spans
    chosen = tuple(treatments)
    if len(chosen) != len(spans):
        raise ValueError(
            f"число режимов {len(chosen)} не совпадает с числом сегментов {len(spans)}"
        )
    parts: list[LayoutPart] = []
    for start, end, treatment in _runs(chosen):
        if treatment == BlockTreatment.TRANSLATE:
            parts.extend(
                _partition_translate(
                    spans,
                    start,
                    end,
                    target_chars=target_chars,
                    max_chars=max_chars,
                )
            )
        else:
            parts.append(_part(spans, start, end, treatment))
    return tuple(parts)


def _same(left: BlockTreatment | None, right: BlockTreatment | None) -> bool:
    """Whether two treatments continue the same run. `None` matches only `None`."""
    if left is None or right is None:
        return left is right
    return left == right


def _runs(
    treatments: tuple[BlockTreatment | None, ...],
) -> tuple[tuple[int, int, BlockTreatment | None], ...]:
    """Maximal ranges of one treatment, as half-open indexes into `treatments`."""
    if not treatments:
        return ()
    runs: list[tuple[int, int, BlockTreatment | None]] = []
    start = 0
    current = treatments[0]
    for index in range(1, len(treatments)):
        treatment = treatments[index]
        if _same(current, treatment):
            continue
        runs.append((start, index, current))
        start = index
        current = treatment
    runs.append((start, len(treatments), current))
    return tuple(runs)


def _part(
    spans: tuple[BlockSegment, ...],
    start: int,
    end: int,
    treatment: BlockTreatment | None,
) -> LayoutPart:
    """A part covering spans `[start, end)`."""
    return LayoutPart(
        start=spans[start].start,
        end=spans[end - 1].end,
        treatment=treatment,
        segments=tuple(range(start, end)),
    )


def _partition_translate(
    spans: tuple[BlockSegment, ...],
    run_start: int,
    run_end: int,
    *,
    target_chars: int,
    max_chars: int,
) -> tuple[LayoutPart, ...]:
    """Split one `TRANSLATE` run. A part may be shorter than `PART_MIN_CHARS`."""
    count = run_end - run_start
    prefix = [0]
    for index in range(count):
        span = spans[run_start + index]
        prefix.append(prefix[-1] + (span.end - span.start))
    # (cost, part count, cut ends). A smaller tuple wins, so an earlier cut wins a tie.
    best: list[tuple[int, int, tuple[int, ...]] | None] = [None] * (count + 1)
    best[0] = (0, 0, ())
    for end in range(1, count + 1):
        chosen: tuple[int, int, tuple[int, ...]] | None = None
        for start in range(end):
            segment_count = end - start
            length = prefix[end] - prefix[start]
            if segment_count > 1 and length > max_chars:
                continue
            previous = best[start]
            if previous is None:
                continue
            candidate = (
                previous[0] + (length - target_chars) ** 2,
                previous[1] + 1,
                previous[2] + (end,),
            )
            if chosen is None or candidate < chosen:
                chosen = candidate
        if chosen is None:
            raise RuntimeError("не удалось разбить серию перевода на части")
        best[end] = chosen
    winner = best[count]
    if winner is None:
        raise RuntimeError("не удалось разбить серию перевода на части")
    parts: list[LayoutPart] = []
    previous_end = 0
    for end in winner[2]:
        parts.append(
            _part(spans, run_start + previous_end, run_start + end, BlockTreatment.TRANSLATE)
        )
        previous_end = end
    return tuple(parts)
