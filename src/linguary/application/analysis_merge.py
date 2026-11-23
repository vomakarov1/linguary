"""Fallback scenario for long chapters: splitting and merging fragments (ALG-03, PRD § 14).

Single-request chapter analysis is the primary scenario. A chapter that exceeds model limits
or is represented by an oversized XHTML document is split into sequential fragments: each fragment
is analyzed in a separate request and returns glossary candidates, new instructions, and a treatment
plan strictly for its block range. The summary describes the chapter from its beginning to the end
of that fragment. This module owns both boundaries of the fallback scenario — how a chapter is
partitioned into fragments, and how fragment responses are reassembled into a single `ChapterAnalysis`.

Merging makes no decisions on behalf of the domain: translation choice conflicts are passed into
`GlossaryCandidate.create` via `has_translation_conflict`, while final `requires_review` is calculated
by `compute_requires_review` per DTO-01. Coverage completeness is also a domain calculation
by `TreatmentPlan.coverage_for`: the module only gathers the plan and expected chapter block IDs.

Incomplete analysis blocks chapter processing: `ChapterAnalysis.ensure_ready_for_processing`
rejects a chapter with `incomplete` coverage; a missing fragment is resolved by re-running analysis,
never by partial processing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from linguary.domain.content_plan import ContentPlan, ContextFact, Difficulty, HelpEvidence, RequiredPoint, Scene, SceneVisualFact, SourceAnchor, artifact_key
from linguary.domain.llm_log import Violation

from linguary.application.a2_core_vocab import (
    _TITLE_CASE_MINOR_WORDS,
    non_core_a2_word_spans,
    non_core_a2_words,
)
from linguary.application.block_labels import without_block_labels
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import _same_word, expression_words
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    ChapterAnalysisResponse,
    ContentPlanResponse,
    GlossaryCandidateResponse,
    QuotedContentPlanResponse,
    QuotedDifficultyResponse,
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedContextFactResponse,
    QuotedSceneResponse,
    QuotedSceneVisualFactResponse,
    QuotedSourceAnchorResponse,
    SegmentTreatmentResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.language_detection import LanguageDetector
from linguary.application.verse import detect_verse_stanza_runs
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    BlockType,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DomainError,
    GlossaryCandidate,
    SupportStatus,
    TreatmentPlan,
    TreatmentPlanError,
    TreatmentPlanItem,
    glossary_key,
    spellings_by_preference,
    translations_conflict,
    without_footnote_markers,
)
from linguary.domain.treatment import strongest_treatment
from linguary.domain.name_translation import (
    NameTranslationEvidence,
    NameTranslationMethod,
    NameTranslationObservation,
    NameTranslationPolicy,
    NameTranslationReviewReason,
    NameTranslationStrategy,
)

#: Fallback limit if provider does not report output token capacity.
DEFAULT_MAX_FRAGMENT_BLOCKS = 160
#: Source-text cap of one analysis fragment. A single block may exceed it.
DEFAULT_MAX_FRAGMENT_CHARS = 24_000
#: Per-block weight overhead representing structured analysis response cost per block.
DEFAULT_FRAGMENT_BLOCK_OVERHEAD = 60
#: Combined character + block-overhead budget of one analysis fragment.
DEFAULT_MAX_FRAGMENT_WEIGHT = 26_000
#: Upper-bound estimate of response tokens per analyzed block from real runs.
TOKENS_PER_ANALYSED_BLOCK = 200
MAX_BEFORE_YOU_READ = 10
MAX_MERGED_INSTRUCTIONS = 20


def fragment_blocks_for(max_output_tokens: int | None) -> int:
    """Number of analysis blocks allowed by the model's output limit."""
    if max_output_tokens is None:
        return DEFAULT_MAX_FRAGMENT_BLOCKS
    return max(1, max_output_tokens // TOKENS_PER_ANALYSED_BLOCK)


class AnalysisMergeError(ApplicationError):
    """Chapter analysis fragments cannot be parsed or merged."""

    def __init__(self, message: str = "", *, violation: Violation | None = None) -> None:
        super().__init__(message)
        self.violation = violation


def _analysis_error(
    message: str,
    *,
    rule_code: str = "content_plan",
    block_ids: Iterable[str] = (),
) -> AnalysisMergeError:
    local_blocks = tuple(dict.fromkeys(b for b in block_ids if b))
    violation = (
        Violation(
            rule_code,
            "chapter_analysis",
            scope="local",
            block_ids=local_blocks,
            safe_message=message,
        )
        if local_blocks
        else Violation(
            rule_code,
            "chapter_analysis",
            scope="global",
            safe_message=message,
        )
    )
    return AnalysisMergeError(message, violation=violation)


def _preview_quote(quote: str) -> str:
    compact = " ".join(quote.split())
    return compact[:77] + "..." if len(compact) > 80 else compact


@dataclass(frozen=True, slots=True)
class ChapterFragment:
    """Sequential fragment of a long chapter: a subset of its textual blocks.

    `order` defines the fragment's position in reading order.
    """

    order: int
    blocks: tuple[DocumentBlock, ...]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise AnalysisMergeError("порядковый номер фрагмента главы отрицателен")
        if not self.blocks:
            raise AnalysisMergeError("фрагмент главы без текстовых блоков не анализируется")

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Fragment block IDs in the original document order."""
        return tuple(block.id for block in self.blocks)


@dataclass(frozen=True, slots=True)
class FragmentAnalysis:
    """Parsed model response for a single chapter fragment.

    Request aliases are already resolved back to stable block IDs: merge
    operates on chapter blocks rather than per-request numbering.
    """

    order: int
    translation_instructions: str
    chapter_summary: str
    treatments: tuple[TreatmentPlanItem, ...] = ()
    glossary_candidates: tuple[GlossaryCandidateResponse, ...] = ()
    before_you_read: tuple[BeforeYouReadEntryResponse, ...] = ()
    content_plan: ContentPlan | None = None

    def __post_init__(self) -> None:
        if self.order < 0:
            raise AnalysisMergeError("порядковый номер фрагмента главы отрицателен")

    @classmethod
    def of_response(
        cls,
        payload: ChapterAnalysisResponse,
        *,
        order: int,
        block_ids_by_alias: Mapping[str, str],
        texts: Mapping[str, str] | None = None,
        visible_texts_by_alias: Mapping[str, str] | None = None,
        resolved_content_plan: ContentPlan | None = None,
        blocks: Sequence[DocumentBlock] | None = None,
    ) -> FragmentAnalysis:
        """Fragment response with plan mapped to chapter block IDs.

        The plan is assembled in document order: `block_ids_by_alias` enumerates
        fragment blocks as received by the model. A block omitted by the fragment
        is excluded from the plan and will trigger incomplete chapter coverage during merge.
        `blocks` supply `segment_spans` for response `segments`. An empty segment
        list means the block treatment covers the whole block.
        """
        reported = {item.source_block_id for item in payload.treatment_plan}
        unknown = sorted(reported - set(block_ids_by_alias))
        if unknown:
            raise AnalysisMergeError(
                f"фрагмент {order} вернул режимы для чужих блоков: {unknown}"
            )
        return cls(
            order=order,
            translation_instructions=without_block_labels(payload.translation_instructions),
            chapter_summary=payload.chapter_summary if payload.content_plan is not None and payload.content_plan.version == 3 else without_block_labels(payload.chapter_summary),
            treatments=plan_items_from_response(
                payload, block_ids_by_alias=block_ids_by_alias, blocks=blocks,
            ),
            glossary_candidates=payload.glossary_candidates,
            before_you_read=payload.before_you_read,
            content_plan=(
                resolved_content_plan
                if resolved_content_plan is not None or payload.content_plan is None
                else content_plan_from_response(
                    payload, block_ids_by_alias, texts=texts,
                    visible_texts_by_alias=visible_texts_by_alias,
                    blocks=blocks,
                )
            ),
        )


_DESCRIBED_CANDIDATE_TYPES = frozenset({
    ConceptType.PERSON,
    ConceptType.PLACE,
    ConceptType.ORGANIZATION,
    ConceptType.FICTIONAL_CONCEPT,
    ConceptType.TERM,
})


def kept_candidate_description(
    candidate_type: ConceptType, responses: Iterable[GlossaryCandidateResponse]
) -> str:
    """Description is kept for every concept type.

    Among duplicates, the highest-confidence non-empty description wins.
    """
    if candidate_type not in _DESCRIBED_CANDIDATE_TYPES:
        return ""
    described = [
        item for item in responses
        if item.type is candidate_type and item.description.strip()
    ]
    if not described:
        return ""
    return max(described, key=lambda item: item.confidence).description


def plan_items_from_response(
    payload: ChapterAnalysisResponse,
    *,
    block_ids_by_alias: Mapping[str, str],
    blocks: Sequence[DocumentBlock] | None = None,
    require_all: bool = False,
) -> tuple[TreatmentPlanItem, ...]:
    """Plan rows in document order.

    A missing alias is skipped unless `require_all` is set. Empty response
    `segments` mean the block treatment covers every span.
    """
    response_by_alias = {item.source_block_id: item for item in payload.treatment_plan}
    blocks_by_id = {block.id: block for block in blocks} if blocks is not None else {}
    items: list[TreatmentPlanItem] = []
    for alias, block_id in block_ids_by_alias.items():
        response_item = response_by_alias.get(alias)
        if response_item is None:
            if require_all:
                raise KeyError(alias)
            continue
        block = blocks_by_id.get(block_id) if blocks is not None else None
        items.append(_plan_item_from_response(response_item, block_id=block_id, block=block))
    return tuple(items)


def _plan_item_from_response(
    item: TreatmentPlanItemResponse,
    *,
    block_id: str,
    block: DocumentBlock | None,
) -> TreatmentPlanItem:
    """Map 1-based segment indexes onto `block.segment_spans`.

    The block treatment must already be the strongest segment mode. Identical
    modes collapse to an empty `segment_treatments` in the constructor, and
    spans of one sentence must share a mode.
    """
    if not item.segments:
        return TreatmentPlanItem(block_id=block_id, treatment=item.treatment)
    if block is None:
        raise _analysis_error(
            f"режимы сегментов блока {block_id} нельзя собрать без самого блока",
            rule_code="invalid_segment_treatment",
            block_ids=(block_id,),
        )
    spans = block.segment_spans
    if len(spans) < 2:
        raise _analysis_error(
            f"блок {block_id} состоит из одного сегмента, режимы сегментов не задаются",
            rule_code="invalid_segment_treatment",
            block_ids=(block_id,),
        )
    by_index: dict[int, BlockTreatment] = {}
    for segment in item.segments:
        if segment.index < 1 or segment.index > len(spans):
            raise _analysis_error(
                f"индекс сегмента {segment.index} блока {block_id} "
                f"вне диапазона 1..{len(spans)}",
                rule_code="invalid_segment_treatment",
                block_ids=(block_id,),
            )
        if segment.index in by_index:
            raise _analysis_error(
                f"сегмент {segment.index} блока {block_id} указан повторно",
                rule_code="invalid_segment_treatment",
                block_ids=(block_id,),
            )
        by_index[segment.index] = segment.treatment
    if set(by_index) != set(range(1, len(spans) + 1)):
        raise _analysis_error(
            f"режимы сегментов блока {block_id} покрывают не все сегменты",
            rule_code="invalid_segment_treatment",
            block_ids=(block_id,),
        )
    try:
        return TreatmentPlanItem(
            block_id=block_id,
            treatment=item.treatment,
            segment_treatments=tuple(by_index[index] for index in range(1, len(spans) + 1)),
            sentence_indexes=tuple(span.sentence_index for span in spans),
        )
    except TreatmentPlanError as err:
        raise _analysis_error(
            str(err),
            rule_code="invalid_segment_treatment",
            block_ids=(block_id,),
        ) from err


def _is_oversized_block(
    block: DocumentBlock, *, max_chars: int, block_overhead: int, max_weight: int
) -> bool:
    length = len(block.text)
    return length > max_chars or length + block_overhead > max_weight


def _split_around_oversized(
    blocks: Sequence[DocumentBlock],
    *,
    max_chars: int,
    block_overhead: int,
    max_weight: int,
) -> list[tuple[bool, Sequence[DocumentBlock]]]:
    """Separates individually oversized blocks so they never skew neighbour balancing."""
    runs: list[tuple[bool, Sequence[DocumentBlock]]] = []
    normal_start: int | None = None
    for index, block in enumerate(blocks):
        if _is_oversized_block(
            block,
            max_chars=max_chars,
            block_overhead=block_overhead,
            max_weight=max_weight,
        ):
            if normal_start is not None:
                runs.append((False, blocks[normal_start:index]))
                normal_start = None
            runs.append((True, (block,)))
        elif normal_start is None:
            normal_start = index
    if normal_start is not None:
        runs.append((False, blocks[normal_start:]))
    return runs


def _cut_priority_key(
    blocks: Sequence[DocumentBlock],
    *,
    start: int,
    cut: int,
    rem_frags: int,
    prefix_chars: Sequence[int],
    prefix_weights: Sequence[int],
    max_blocks: int,
    max_chars: int,
    max_weight: int,
) -> tuple[float, int, int]:
    """Scores a candidate fragment split: lower balance + boundary cost wins, fuller earlier breaks ties."""
    n = len(blocks)
    frag_b = cut - start
    frag_c = prefix_chars[cut] - prefix_chars[start]
    frag_w = prefix_weights[cut] - prefix_weights[start]
    rem_b = n - start
    rem_c = prefix_chars[n] - prefix_chars[start]
    rem_w = prefix_weights[n] - prefix_weights[start]

    weight_imbalance = abs(frag_w * rem_frags - rem_w) / (rem_frags * max_weight)
    char_imbalance = abs(frag_c * rem_frags - rem_c) / (rem_frags * max_chars)
    block_imbalance = abs(frag_b * rem_frags - rem_b) / (rem_frags * max_blocks)
    balance_cost = max(weight_imbalance, char_imbalance) + block_imbalance

    prev_block = blocks[cut - 1]
    next_block = blocks[cut]
    splits_quote = (
        prev_block.type in {BlockType.QUOTE, BlockType.POEM}
        and next_block.type is prev_block.type
        and next_block.order == prev_block.order + 1
    )
    trailing_heading = prev_block.type is BlockType.HEADING
    has_order_gap = next_block.order - prev_block.order > 1
    tiny_head = frag_b == 1 and 2 * frag_w * rem_frags <= rem_w
    tiny_tail = (
        rem_frags == 2
        and (n - cut) == 1
        and 2 * (rem_w - frag_w) * rem_frags <= rem_w
    )

    boundary_cost = (
        (0.60 if splits_quote else 0.0)
        + (0.60 if trailing_heading else 0.0)
        + (0.20 if (tiny_head or tiny_tail) else 0.0)
        - (0.26 if has_order_gap else 0.0)
    )
    return (round(balance_cost + boundary_cost, 12), -frag_w, -frag_b)


def _plan_normal_run(
    blocks: Sequence[DocumentBlock],
    *,
    max_blocks: int,
    max_chars: int,
    block_overhead: int,
    max_weight: int,
) -> list[tuple[DocumentBlock, ...]]:
    """Partitions non-oversized blocks into the minimum balanced fragments."""
    n = len(blocks)
    prefix_chars = [0] * (n + 1)
    prefix_weights = [0] * (n + 1)
    for index, block in enumerate(blocks):
        length = len(block.text)
        prefix_chars[index + 1] = prefix_chars[index] + length
        prefix_weights[index + 1] = prefix_weights[index] + length + block_overhead

    farthest_end = [0] * n
    right = 0
    for left in range(n):
        if right <= left:
            right = left + 1
        while (
            right < n
            and (right + 1 - left) <= max_blocks
            and (prefix_chars[right + 1] - prefix_chars[left]) <= max_chars
            and (prefix_weights[right + 1] - prefix_weights[left]) <= max_weight
        ):
            right += 1
        farthest_end[left] = right

    min_frags_from = [0] * (n + 1)
    for index in range(n - 1, -1, -1):
        min_frags_from[index] = 1 + min_frags_from[farthest_end[index]]

    if min_frags_from[0] == 1:
        return [tuple(blocks)]

    groups: list[tuple[DocumentBlock, ...]] = []
    start = 0
    rem_frags = min_frags_from[0]
    while rem_frags > 1:
        cut = min(
            (
                candidate
                for candidate in range(start + 1, farthest_end[start] + 1)
                if min_frags_from[candidate] == rem_frags - 1
            ),
            key=lambda candidate: _cut_priority_key(
                blocks,
                start=start,
                cut=candidate,
                rem_frags=rem_frags,
                prefix_chars=prefix_chars,
                prefix_weights=prefix_weights,
                max_blocks=max_blocks,
                max_chars=max_chars,
                max_weight=max_weight,
            ),
        )
        groups.append(tuple(blocks[start:cut]))
        start = cut
        rem_frags -= 1
    groups.append(tuple(blocks[start:]))
    return groups


def plan_fragments(
    blocks: Sequence[DocumentBlock],
    *,
    max_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS,
    max_chars: int = DEFAULT_MAX_FRAGMENT_CHARS,
    block_overhead: int = DEFAULT_FRAGMENT_BLOCK_OVERHEAD,
    max_weight: int | None = None,
) -> tuple[ChapterFragment, ...]:
    """Splits textual blocks into balanced sequential analysis fragments.

    Respects `max_blocks`, `max_chars`, and `effective_max_weight` (source characters
    plus `block_overhead` per block) for every multi-block fragment. A single block
    exceeding character or weight limits is emitted as its own 1-block fragment.
    When splitting is required, distributes blocks across the minimum fragment count
    while keeping consecutive quotes together, avoiding trailing headings, and
    preferring scene breaks (`block.order` gaps).
    """
    effective_max_weight = (
        max_weight
        if max_weight is not None
        else max_chars + (DEFAULT_MAX_FRAGMENT_WEIGHT - DEFAULT_MAX_FRAGMENT_CHARS)
    )
    if max_blocks < 1:
        raise AnalysisMergeError("размер фрагмента главы должен быть положительным")
    if max_chars < 1:
        raise AnalysisMergeError("предел знаков фрагмента главы должен быть положительным")
    if block_overhead < 0:
        raise AnalysisMergeError("накладной вес блока фрагмента не может быть отрицательным")
    if effective_max_weight < 1:
        raise AnalysisMergeError("весовой предел фрагмента главы должен быть положительным")
    if not blocks:
        return ()

    groups: list[tuple[DocumentBlock, ...]] = []
    for is_oversized, run in _split_around_oversized(
        blocks,
        max_chars=max_chars,
        block_overhead=block_overhead,
        max_weight=effective_max_weight,
    ):
        if is_oversized:
            groups.append(tuple(run))
        else:
            groups.extend(
                _plan_normal_run(
                    run,
                    max_blocks=max_blocks,
                    max_chars=max_chars,
                    block_overhead=block_overhead,
                    max_weight=effective_max_weight,
                )
            )
    return tuple(
        ChapterFragment(order=order, blocks=group)
        for order, group in enumerate(groups)
    )


def merge_chapter_analysis(
    fragments: Iterable[FragmentAnalysis],
    *,
    id: str,
    section_id: str,
    expected_block_ids: Iterable[str],
    glossary_confidence_threshold: float,
    id_factory: Callable[[], str],
    name_translation_policy: NameTranslationPolicy | None = None,
    conditions_id: str | None = None,
) -> ChapterAnalysis:
    """Assembles full analysis of a chunked chapter from its fragment responses (ALG-03).

    `expected_block_ids` contains all textual blocks of the chapter in document order:
    coverage is evaluated against them. An unexecuted fragment is simply absent from
    `fragments`, leaving its blocks uncovered — marking the chapter analysis `incomplete`.
    """
    ordered = _ordered_fragments(fragments)
    expected = tuple(expected_block_ids)
    return ChapterAnalysis.create(
        id=id,
        section_id=section_id,
        treatment_plan=_merged_plan(ordered, section_id=section_id, expected=expected),
        translation_instructions=_merged_instructions(ordered),
        chapter_summary=_merged_summary(ordered),
        expected_block_ids=expected,
        glossary_candidates=_merged_candidates(
            ordered,
            analysis_id=id,
            confidence_threshold=glossary_confidence_threshold,
            id_factory=id_factory,
            name_translation_policy=name_translation_policy,
            conditions_id=conditions_id,
        ),
        before_you_read=_merged_before_you_read(ordered),
        content_plan=_merged_content_plan(ordered),
    )


def _ordered_fragments(
    fragments: Iterable[FragmentAnalysis],
) -> tuple[FragmentAnalysis, ...]:
    """Executed fragments sorted in chapter reading order."""
    ordered = tuple(sorted(fragments, key=lambda fragment: fragment.order))
    orders = [fragment.order for fragment in ordered]
    if len(set(orders)) != len(orders):
        raise AnalysisMergeError(
            f"фрагменты главы получили повторяющиеся номера: {sorted(orders)}"
        )
    return ordered


def _merged_plan(
    fragments: Sequence[FragmentAnalysis], *, section_id: str, expected: Sequence[str]
) -> TreatmentPlan:
    """Chapter treatment plan assembled by `block_id` (ALG-03).

    Two fragments that assigned differing treatments to the same block yield no decision:
    the application does not choose between them, the block remains unassigned, and chapter
    coverage is incomplete. Repeating an identical plan row, including segment modes,
    is not a conflict — fragments agree.
    """
    decided: dict[str, TreatmentPlanItem] = {}
    conflicting: set[str] = set()
    for fragment in fragments:
        for item in fragment.treatments:
            previous = decided.get(item.block_id)
            if previous is None:
                decided[item.block_id] = item
            elif previous != item:
                conflicting.add(item.block_id)
    for block_id in conflicting:
        del decided[block_id]
    known = set(expected)
    order = [block_id for block_id in expected if block_id in decided]
    order.extend(block_id for block_id in decided if block_id not in known)
    return TreatmentPlan(
        section_id=section_id,
        items=tuple(decided[block_id] for block_id in order),
    )


_INSTRUCTION_JACCARD_THRESHOLD = 0.70
_INSTRUCTION_CONTAINMENT_THRESHOLD = 0.85
_MIN_INSTRUCTION_SIGNIFICANT_STEMS = 3

_TY_REGISTER_FORMS = frozenset({
    "ты", "тебя", "тебе", "тобой", "тобою",
    "твой", "твоя", "твое", "твои", "твоего", "твоей",
    "твоему", "твоим", "твоем", "твою", "твоих", "твоими",
})
_VY_REGISTER_FORMS = frozenset({
    "вы", "вас", "вам", "вами",
    "ваш", "ваша", "ваше", "ваши", "вашего", "вашей",
    "вашему", "вашим", "вашем", "вашу", "ваших", "вашими",
})

_RU_INSTRUCTION_STOP_WORDS = frozenset({
    "в", "во", "на", "за", "к", "ко", "с", "со", "у", "о", "об", "обо",
    "от", "ото", "до", "из", "изо", "по", "под", "подо", "над", "надо",
    "при", "про", "без", "безо", "для", "через", "между", "перед", "передо",
    "сквозь", "среди", "возле", "около", "вокруг", "после", "кроме", "вместо", "ради",
    "и", "а", "но", "да", "или", "либо", "ни", "не", "бы", "б", "же", "ж",
    "ли", "ль", "то", "что", "чтобы", "чтоб", "как", "когда", "если", "хотя",
    "пока", "потому", "поэтому", "так", "также", "тоже", "ведь", "уже", "уж",
    "еще", "лишь", "только", "даже", "вот", "вон", "тут", "там", "здесь",
    "где", "куда", "откуда", "зачем", "почему", "чем", "словно", "будто",
    "я", "меня", "мне", "мной", "мною",
    "мой", "моя", "мое", "мои", "моего", "моей", "моему", "моим", "моем", "мою", "моих", "моими",
    "мы", "нас", "нам", "нами",
    "наш", "наша", "наше", "наши", "нашего", "нашей", "нашему", "нашим", "нашем", "нашу", "наших", "нашими",
    *_TY_REGISTER_FORMS,
    *_VY_REGISTER_FORMS,
    "он", "она", "оно", "они", "его", "ее", "их", "ему", "ей", "им",
    "него", "нее", "них", "нему", "ней", "ним", "ними", "нем",
    "себя", "себе", "собой", "собою",
    "свой", "своя", "свое", "свои", "своего", "своей", "своему", "своим", "своем", "свою", "своих", "своими",
    "это", "этот", "эта", "эти", "этого", "этой", "этому", "этим", "этом", "эту", "этих", "этими",
    "тот", "та", "те", "того", "той", "тому", "тем", "том", "ту", "тех", "теми",
    "кто", "кого", "кому", "кем", "ком", "чего", "чему",
    "который", "которая", "которое", "которые", "которого", "которой", "которому",
    "которым", "котором", "которую", "которых", "которыми",
    "весь", "вся", "все", "всего", "всей", "всему", "всем", "всю", "всех", "всеми",
    "сам", "сама", "само", "сами", "самого", "самому", "самим", "самом", "саму", "самих", "самими",
    "такой", "такая", "такое", "такие", "такого", "такому", "таким", "таком", "такую", "таких", "такими",
    "один", "одна", "одно", "одни", "одного", "одной", "одному", "одним", "одном", "одну", "одних", "одними",
    "другой", "другая", "другое", "другие", "другого", "другому", "другим", "других",
    "быть", "есть", "был", "была", "было", "были", "будет", "будут", "будем", "будете", "буду", "будешь",
})

_RU_DAVAT_VERB_RE = re.compile(r"^(.+?(?:д|зн|ст)а)(?:вайте|вать|вай|вая|йте|ть|й)$")
_RU_INSTRUCTION_SUFFIXES = tuple(
    sorted(
        {
            "овать", "евать", "овал", "евал", "овала", "евала", "овали", "евали",
            "яйте", "айте", "ейте", "уйте", "юйте", "ите", "йте",
            "яете", "аете", "еете", "яет", "ает", "еет", "иет",
            "яют", "ают", "еют", "иют", "яем", "аем", "еем", "ием",
            "ять", "ать", "еть", "ить", "уть", "ыть", "ть",
            "яй", "ай", "ей", "уй", "юй", "яя", "ая", "ея", "уя", "юя",
            "ыми", "ими", "ого", "его", "ому", "ему", "ых", "их",
            "ую", "юю", "ой", "ый", "ий", "ое", "ее", "ые", "ие", "ым", "им", "ом", "ем",
            "иями", "ями", "ами", "иям", "иях", "ией", "ию", "ия", "ии",
            "ов", "ев", "ам", "ям", "ах", "ях", "ью",
            "а", "я", "о", "е", "у", "ю", "ы", "и", "ь", "й",
        },
        key=lambda suffix: (-len(suffix), suffix),
    )
)

_INSTRUCTION_NEGATION_RE = re.compile(r"(?<!\w)не(?!\w)", re.IGNORECASE)
_INSTRUCTION_QUOTED_RE = re.compile(
    r"«([^»\n]+)»"
    r"|„([^“”\n]+)[“”]"
    r"|“([^”\n]+)”"
    r"|\"([^\"\n]+)\""
    r"|(?<!\w)'((?:[^'\n]|(?<=\w)'(?=\w))+)'(?!\w)"
    r"|(?<!\w)‘((?:[^’\n]|(?<=\w)’(?=\w))+)’(?!\w)"
)
_QUOTE_APOSTROPHE_TRANS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_INSTRUCTION_PROPER_NOUN_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9'’-]*")
_INSTRUCTION_ENGLISH_TERM_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z]+(?:['’-][A-Za-z]+)*"
    r"(?:\.?\s+[A-Za-z]+(?:['’-][A-Za-z]+)*)*"
    r"(?![A-Za-z0-9])"
)
_HONORIFIC_OR_INITIAL_BEFORE_DOT_RE = re.compile(
    r"(?:^|\W)(?:(?i:Mr|Mrs|Ms|Dr|Prof|St)|[A-ZА-ЯЁ])\.$",
)


@dataclass(frozen=True, slots=True)
class _InstructionEntry:
    key: str
    text: str
    stems: frozenset[str]
    register: frozenset[str]
    has_negation: bool
    quotes: frozenset[str]
    proper_nouns: frozenset[str]
    english_terms: frozenset[str] = frozenset()


def _stem_instruction_word(word: str) -> str:
    """Normalizes and stems a Russian or English instruction token."""
    normalized = word.translate(_QUOTE_APOSTROPHE_TRANS).casefold().replace("ё", "е")
    if not any("а" <= ch <= "я" for ch in normalized):
        return _stem_byr_word(normalized)
    if len(normalized) >= 5 and normalized.endswith(("ся", "сь")):
        normalized = normalized[:-2]
    if (
        (davat_match := _RU_DAVAT_VERB_RE.match(normalized)) is not None
        and len(davat_match.group(1)) >= 3
    ):
        return davat_match.group(1)
    for suffix in _RU_INSTRUCTION_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 3:
            return normalized[: -len(suffix)]
    return normalized


def _significant_instruction_stems(text: str) -> frozenset[str]:
    """Significant word stems extracted from a translation instruction item."""
    stems: set[str] = set()
    for raw_word in expression_words(text):
        word = raw_word.replace("ё", "е")
        if word in _RU_INSTRUCTION_STOP_WORDS or word in _BYR_STOP_WORDS:
            continue
        stem = _stem_instruction_word(word)
        if stem in _BYR_STOP_WORDS:
            continue
        stems.add(stem)
    return frozenset(stems)


def _instruction_register(text: str) -> frozenset[str]:
    tags: set[str] = set()
    for raw_word in expression_words(text):
        word = raw_word.replace("ё", "е")
        if word in _TY_REGISTER_FORMS:
            tags.add("ты")
        elif word in _VY_REGISTER_FORMS:
            tags.add("вы")
    return frozenset(tags)


def _instruction_quoted_strings(text: str) -> frozenset[str]:
    quotes: set[str] = set()
    for match in _INSTRUCTION_QUOTED_RE.finditer(text):
        raw_quote = next((group for group in match.groups() if group is not None), "")
        normalized = unicodedata.normalize("NFKC", raw_quote).translate(_QUOTE_APOSTROPHE_TRANS)
        normalized = re.sub(r"\s+", " ", normalized).strip().casefold().replace("ё", "е")
        if normalized:
            quotes.add(normalized)
    return frozenset(quotes)


def _is_sentence_or_clause_start(prefix: str) -> bool:
    stripped = prefix.rstrip()
    if not stripped:
        return True
    if stripped[-1] in '«"“„([':
        return True
    if stripped[-1] in "'‘" and (len(stripped) == 1 or not stripped[-2].isalnum()):
        return True
    core = stripped.rstrip('«"“„([ ')
    if not core:
        return True
    if core[-1] in "!?:;…—–":
        return True
    if core[-1] == "-" and (len(core) == 1 or core[-2].isspace()):
        return True
    if core[-1] == ".":
        return _HONORIFIC_OR_INITIAL_BEFORE_DOT_RE.search(core) is None
    return False


def _instruction_proper_nouns(text: str) -> frozenset[str]:
    """Capitalized non-initial words (excluding ты/вы forms) in normalized stem form."""
    proper_nouns: set[str] = set()
    for match in _INSTRUCTION_PROPER_NOUN_TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2 or not token[0].isupper():
            continue
        if _is_sentence_or_clause_start(text[: match.start()]):
            continue
        folded = token.casefold().replace("ё", "е")
        if folded in _TY_REGISTER_FORMS or folded in _VY_REGISTER_FORMS:
            continue
        proper_nouns.add(_stem_instruction_word(folded))
    return frozenset(proper_nouns)


def _instruction_english_terms(text: str) -> frozenset[str]:
    """Canonical English term keys referenced inside a Cyrillic translation directive."""
    if not _has_cyrillic(text):
        return frozenset()
    terms: set[str] = set()
    for match in _INSTRUCTION_ENGLISH_TERM_RE.finditer(text):
        key = glossary_key(match.group(0))
        if sum(ch.isalpha() for ch in key) >= 2:
            terms.add(key)
    return frozenset(terms)


def _instruction_entry(text: str) -> _InstructionEntry | None:
    cleaned = without_block_labels(text)
    key = re.sub(r"\s+", " ", cleaned).casefold().strip(" \t.,;:!?")
    if not key:
        return None
    return _InstructionEntry(
        key=key,
        text=cleaned,
        stems=_significant_instruction_stems(cleaned),
        register=_instruction_register(cleaned),
        has_negation=bool(_INSTRUCTION_NEGATION_RE.search(cleaned)),
        quotes=_instruction_quoted_strings(cleaned),
        proper_nouns=_instruction_proper_nouns(cleaned),
        english_terms=_instruction_english_terms(cleaned),
    )


def _instructions_semantically_match(
    left: _InstructionEntry, right: _InstructionEntry
) -> bool:
    """Whether two directives repeat the same guidance without conflicting polarity or names."""
    if (
        left.register != right.register
        or left.has_negation != right.has_negation
    ):
        return False
    if left.english_terms and left.english_terms == right.english_terms:
        return True
    if (
        left.quotes != right.quotes
        or left.proper_nouns != right.proper_nouns
    ):
        return False
    min_stems = min(len(left.stems), len(right.stems))
    if min_stems < _MIN_INSTRUCTION_SIGNIFICANT_STEMS:
        return False
    overlap = len(left.stems & right.stems)
    union = len(left.stems | right.stems)
    return (
        overlap / union >= _INSTRUCTION_JACCARD_THRESHOLD
        or overlap / min_stems >= _INSTRUCTION_CONTAINMENT_THRESHOLD
    )


_INSTRUCTION_ABBREV_BEFORE_DOT_RE = re.compile(
    r"(?:^|(?<!\w))(?:"
    r"т\.\s*[пде]"
    r"|(?:ед|мн)\.\s*ч"
    r"|др|см|стр|г"
    r"|mr|mrs|ms|dr|st"
    r")\.$",
    re.IGNORECASE,
)
_INSTRUCTION_INITIAL_BEFORE_DOT_RE = re.compile(r"(?:^|(?<!\w))[A-ZА-ЯЁ]\.$")
_FRAGMENT_BOUNDARY_ARTIFACT_RE = re.compile(
    r"до\s+конца\s+(?:(?:этого|текущего|данного)\s+)?фрагмента"
    r"|без\s+подстановки\s+имени\s+из\s+других\s+глав"
    r"|в\s+(?:начале|конце)\s+фрагмента"
    r"|в\s+(?:этом|текущем)\s+фрагменте(?!\s+(?:письма|книги|текста|дневника|цитаты))"
    r"|без\s+придуманного\s+имени",
    re.IGNORECASE,
)
_FRAGMENT_UNBOUNDED_UNNAMED_RE = re.compile(
    r"оста[её]тся\s+(?:неназванн|без\s+имени)",
    re.IGNORECASE,
)
_EXPLICIT_PLOT_BOUNDARY_RE = re.compile(r"\bдо\s+", re.IGNORECASE)


def _is_fragment_boundary_artifact(text: str) -> bool:
    if _FRAGMENT_BOUNDARY_ARTIFACT_RE.search(text) is not None:
        return True
    return (
        _FRAGMENT_UNBOUNDED_UNNAMED_RE.search(text) is not None
        and _EXPLICIT_PLOT_BOUNDARY_RE.search(text) is None
    )


def _split_instruction_sentences(text: str) -> tuple[str, ...]:
    """Splits a plain paragraph into sentences without cutting inside quotes or abbreviations."""
    stripped = text.strip()
    if not stripped:
        return ()
    track_straight_double = stripped.count('"') % 2 == 0
    guillemet_depth = 0
    curly_double_depth = 0
    in_straight_double = False
    parts: list[str] = []
    start = 0
    n = len(stripped)
    idx = 0
    while idx < n:
        ch = stripped[idx]
        if ch == "«":
            guillemet_depth += 1
            idx += 1
            continue
        if ch == "“":
            curly_double_depth += 1
            idx += 1
            continue
        closed_quote = False
        if ch == "»":
            if guillemet_depth > 0:
                guillemet_depth -= 1
                closed_quote = True
        elif ch == "”":
            if curly_double_depth > 0:
                curly_double_depth -= 1
                closed_quote = True
        elif ch == '"' and track_straight_double:
            was_in_straight = in_straight_double
            in_straight_double = not in_straight_double
            closed_quote = was_in_straight
        if ch in '»”"':
            if (
                closed_quote
                and guillemet_depth == 0
                and curly_double_depth == 0
                and not in_straight_double
            ):
                prev_idx = idx - 1
                while prev_idx >= start and stripped[prev_idx] in '»”"':
                    prev_idx -= 1
                if prev_idx >= start and stripped[prev_idx] in ".!?…":
                    is_single_dot = stripped[prev_idx] == "." and (
                        prev_idx == start or stripped[prev_idx - 1] not in ".!?…"
                    )
                    prefix = stripped[start : prev_idx + 1]
                    if not (
                        is_single_dot
                        and (
                            _INSTRUCTION_ABBREV_BEFORE_DOT_RE.search(prefix) is not None
                            or _INSTRUCTION_INITIAL_BEFORE_DOT_RE.search(prefix) is not None
                        )
                    ):
                        end = idx + 1
                        while end < n and stripped[end] in ")]":
                            end += 1
                        if end < n and stripped[end].isspace():
                            next_idx = end
                            while next_idx < n and stripped[next_idx].isspace():
                                next_idx += 1
                            probe = next_idx
                            while probe < n and stripped[probe] in '«"“([':
                                probe += 1
                            if probe < n and (stripped[probe].isupper() or stripped[probe].isdigit()):
                                sentence = stripped[start:end].strip()
                                if sentence:
                                    parts.append(sentence)
                                start = next_idx
                                idx = next_idx
                                continue
            idx += 1
            continue
        if (
            ch in ".!?…"
            and guillemet_depth == 0
            and curly_double_depth == 0
            and not in_straight_double
        ):
            end = idx + 1
            while end < n and stripped[end] in ".!?…":
                end += 1
            is_single_dot = ch == "." and end == idx + 1
            prefix = stripped[start:end]
            if is_single_dot and (
                _INSTRUCTION_ABBREV_BEFORE_DOT_RE.search(prefix) is not None
                or _INSTRUCTION_INITIAL_BEFORE_DOT_RE.search(prefix) is not None
            ):
                idx = end
                continue
            while end < n and stripped[end] in ")]":
                end += 1
            if end < n and stripped[end].isspace():
                next_idx = end
                while next_idx < n and stripped[next_idx].isspace():
                    next_idx += 1
                probe = next_idx
                while probe < n and stripped[probe] in '«"“([':
                    probe += 1
                if probe < n and (stripped[probe].isupper() or stripped[probe].isdigit()):
                    sentence = stripped[start:end].strip()
                    if sentence:
                        parts.append(sentence)
                    start = next_idx
                    idx = next_idx
                    continue
            idx = end
            continue
        idx += 1
    tail = stripped[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _should_split_instruction_paragraph(sentences: Sequence[str]) -> bool:
    if len(sentences) >= 3:
        return True
    if len(sentences) >= 2 and any(
        bool(_instruction_register(sentence))
        or _is_fragment_boundary_artifact(sentence)
        for sentence in sentences
    ):
        return True
    return False


def _merged_instructions(fragments: Sequence[FragmentAnalysis]) -> str:
    """Chapter translation instructions: fragment directives ordered by reading sequence.

    Exact duplicates and guarded semantic rephrasings are folded; when a later directive
    contains an earlier one and is more complete, it replaces the earlier item in place.
    """
    entries: list[_InstructionEntry] = []
    exact_seen: set[str] = set()
    drop_fragment_artifacts = len(fragments) > 1
    for fragment in fragments:
        for point in _instruction_points(fragment.translation_instructions):
            if drop_fragment_artifacts and _is_fragment_boundary_artifact(point):
                continue
            candidate = _instruction_entry(point)
            if candidate is None or candidate.key in exact_seen:
                continue
            exact_seen.add(candidate.key)
            matched = [
                idx
                for idx, existing in enumerate(entries)
                if _instructions_semantically_match(existing, candidate)
            ]
            if not matched:
                entries.append(candidate)
                continue
            if any(len(entries[idx].stems) >= len(candidate.stems) for idx in matched):
                continue
            entries[matched[0]] = candidate
            for idx in reversed(matched[1:]):
                del entries[idx]
    return "\n".join(
        f"{index}. {entry.text}"
        for index, entry in enumerate(entries[:MAX_MERGED_INSTRUCTIONS], 1)
    )


def _instruction_points(text: str) -> tuple[str, ...]:
    """Parses bulleted items and plain paragraphs with guarded sentence splitting."""
    points: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = paragraph.splitlines()
        if len(lines) > 1 and any(re.match(r"^\s*(?:[-•]|\d+\.)\s+", line) for line in lines):
            points.extend(
                re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", line).strip()
                for line in lines
                if line.strip()
            )
        elif paragraph.strip():
            cleaned = re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", paragraph).strip()
            sentences = _split_instruction_sentences(cleaned)
            if _should_split_instruction_paragraph(sentences):
                points.extend(sentences)
            else:
                points.append(cleaned)
    return tuple(points)


def _merged_summary(fragments: Sequence[FragmentAnalysis]) -> str:
    """The latest non-empty summary covers the chapter up to the end of its fragment."""
    return next(
        (
            summary
            for fragment in reversed(fragments)
            if (summary := fragment.chapter_summary.strip())
        ),
        "",
    )


def dominant_script(text: str) -> str | None:
    """Dominant Unicode script name among alphabetic characters, or None if none."""
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith("CJK UNIFIED IDEOGRAPH"):
            script = "CJK"
        elif name:
            script = name.split(" ", 1)[0]
        else:
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)


def _significant_summary_stems(text: str) -> frozenset[str]:
    return frozenset(
        stem
        for word in expression_words(text)
        if word not in _BYR_STOP_WORDS
        and (stem := _stem_byr_word(word)) not in _BYR_STOP_WORDS
    )


def select_staged_a2_summary(
    parts: Sequence[FragmentAnalysis], *, narrative_summary: str
) -> str:
    """Selects the staged A2 chapter summary best covering `narrative_summary`.

    Splits `narrative_summary` into sentences, drops empty-stem sentences, and filters out
    ubiquitous stems appearing in `>= 50%` of narrative sentences whenever rarer stems exist
    in a sentence. Candidates whose dominant script disagrees with `narrative_summary` (when
    known) are excluded from overlap scoring, and remaining candidates are ranked by
    `(covered_sentence_ratio, covers_last_sentence, total_stem_overlap, -index)`. When no
    candidate achieves positive stem overlap in the matching script, falls back predictably to
    the fullest non-empty candidate, breaking ties in favor of earlier fragments (`parts[0]`).
    """
    candidates = [
        (index, summary)
        for index, part in enumerate(parts)
        if (summary := part.chapter_summary.strip())
    ]
    if not candidates:
        return ""

    narrative_script = dominant_script(narrative_summary)
    sentence_stems = [
        stems
        for sentence in _split_instruction_sentences(narrative_summary)
        if (stems := _significant_summary_stems(sentence))
    ]
    num_sentences = len(sentence_stems)
    stem_doc_freq: dict[str, int] = {}
    for stems in sentence_stems:
        for stem in stems:
            stem_doc_freq[stem] = stem_doc_freq.get(stem, 0) + 1

    sentence_entries: list[tuple[frozenset[str], bool]] = []
    for stems in sentence_stems:
        rare = frozenset(stem for stem in stems if stem_doc_freq[stem] * 2 < num_sentences)
        if rare:
            sentence_entries.append((rare, True))
        else:
            sentence_entries.append((stems, False))

    all_effective_stems = (
        frozenset().union(*(eff for eff, _ in sentence_entries))
        if sentence_entries
        else frozenset()
    )

    best_overlap: tuple[float, bool, int, int, str] | None = None
    if all_effective_stems:
        for index, summary in candidates:
            candidate_script = dominant_script(summary)
            if narrative_script is not None and candidate_script != narrative_script:
                continue
            candidate_stems = _significant_summary_stems(summary)
            total_stem_overlap = len(all_effective_stems & candidate_stems)
            if total_stem_overlap == 0:
                continue
            covered_flags = [
                (match_count >= 1)
                if has_rare
                else (match_count >= 2 or match_count / len(eff) >= 0.25)
                for eff, has_rare in sentence_entries
                for match_count in (len(eff & candidate_stems),)
            ]
            covered_sentence_ratio = sum(covered_flags) / num_sentences
            covers_last_sentence = covered_flags[-1]
            rank = (
                covered_sentence_ratio,
                covers_last_sentence,
                total_stem_overlap,
                -index,
            )
            if best_overlap is None or rank > best_overlap[:4]:
                best_overlap = (*rank, summary)
    if best_overlap is not None:
        return best_overlap[4]

    best_fallback: tuple[bool, int, int, int, str] | None = None
    for index, summary in candidates:
        candidate_script = dominant_script(summary)
        script_matches = narrative_script is not None and candidate_script == narrative_script
        word_count = len(expression_words(summary))
        collapsed_len = len(" ".join(summary.split()))
        rank = (script_matches, word_count, collapsed_len, -index)
        if best_fallback is None or rank > best_fallback[:4]:
            best_fallback = (*rank, summary)
    assert best_fallback is not None
    return best_fallback[4]


def _merged_candidates(
    fragments: Sequence[FragmentAnalysis],
    *,
    analysis_id: str,
    confidence_threshold: float,
    id_factory: Callable[[], str],
    name_translation_policy: NameTranslationPolicy | None = None,
    conditions_id: str | None = None,
) -> tuple[GlossaryCandidate, ...]:
    """Chapter glossary candidates merged by source text (PRD § 11-12).

    Obvious spelling variants of the same text — case, typography, period after `Mr` —
    are grouped via `glossary_key`: otherwise each spelling would become a separate
    mandatory decision, and decisions could diverge.
    """
    groups = _candidate_groups(
        candidate for fragment in fragments for candidate in fragment.glossary_candidates
    )
    return tuple(
        _merged_candidate(
            group,
            id=id_factory(),
            analysis_id=analysis_id,
            confidence_threshold=confidence_threshold,
            name_translation_policy=name_translation_policy,
            conditions_id=conditions_id,
        )
        for group in groups
    )


def _candidate_groups[CandidateT: GlossaryCandidate | GlossaryCandidateResponse](
    candidates: Iterable[CandidateT],
) -> tuple[list[CandidateT], ...]:
    groups: dict[str, list[CandidateT]] = {}
    for candidate in candidates:
        groups.setdefault(glossary_key(candidate.source_text), []).append(candidate)
    return tuple(groups.values())


def _merged_candidate_suggestion(
    group: Sequence[GlossaryCandidate | GlossaryCandidateResponse],
) -> tuple[str, str, str]:
    leading = max(group, key=lambda candidate: candidate.confidence)
    source_text = spellings_by_preference(
        (candidate.source_text, candidate.confidence) for candidate in group
    )[0]
    note = next((candidate.note for candidate in sorted(
        group, key=lambda candidate: candidate.confidence, reverse=True
    ) if candidate.note.strip()), "")
    return source_text, leading.suggested_translation, note


def merged_glossary_suggestion_keys(candidates: Iterable[GlossaryCandidate | GlossaryCandidateResponse]) -> frozenset[tuple[str, str, str]]:
    """Semantic output keys, using the same independent field selection as chapter persistence."""
    keys: set[tuple[str, str, str]] = set()
    for group in _candidate_groups(candidates):
        source, translation, note = _merged_candidate_suggestion(group)
        norm_source = glossary_key(source)
        keys.add((norm_source, translation, note))
        keys.add((norm_source, translation, ""))
    return frozenset(keys)


def _merged_candidate(
    group: Sequence[GlossaryCandidateResponse],
    *,
    id: str,
    analysis_id: str,
    confidence_threshold: float,
    name_translation_policy: NameTranslationPolicy | None = None,
    conditions_id: str | None = None,
) -> GlossaryCandidate:
    """Single chapter candidate merged from variants proposed by fragments.

    The suggested translation is taken from the highest-confidence fragment, and review
    flags are combined: a model flag from any fragment, a type always requiring review,
    or conflicting translation choices make the candidate require review. Agreement across
    fragments does not in itself require review.

    Source text is the most frequent spelling among fragments (`spellings_by_preference`).

    The translation note is taken from the highest-confidence fragment that provided one:
    an empty note must not displace a non-empty one, as fragments see different portions
    of the chapter and characteristics requiring notes may not appear in every fragment.
    A description is kept only for a person or a place, by the same confidence rule.
    """
    leading = max(group, key=lambda candidate: candidate.confidence)
    source_text, suggested_translation, note = _merged_candidate_suggestion(group)
    candidate = GlossaryCandidate.create(
        id=id,
        analysis_id=analysis_id,
        source_text=source_text,
        type=next(
            (
                candidate.type
                for candidate in group
                if candidate.type.always_requires_review
            ),
            leading.type,
        ),
        suggested_translation=suggested_translation,
        confidence=leading.confidence,
        model_flagged_review=any(candidate.model_flagged_review for candidate in group),
        confidence_threshold=confidence_threshold,
        has_translation_conflict=translations_conflict(
            candidate.suggested_translation for candidate in group
        ),
        note=note,
    )
    evidence = name_translation_evidence(
        group, policy=name_translation_policy, conditions_id=conditions_id
    )
    description = kept_candidate_description(candidate.type, group)
    changes = {}
    if evidence is not None:
        changes["name_translation_evidence"] = evidence
    if description:
        changes["description"] = description
    return replace(candidate, **changes) if changes else candidate


def name_translation_evidence(
    responses: Sequence[GlossaryCandidateResponse],
    *,
    policy: NameTranslationPolicy | None,
    conditions_id: str | None,
) -> NameTranslationEvidence | None:
    """Server provenance and every raw model observation for a candidate."""
    if policy is None or conditions_id is None:
        return None
    return NameTranslationEvidence(
        version=1, conditions_id=conditions_id, policy_version=policy.version,
        strategy=policy.strategy,
        observations=tuple(
            NameTranslationObservation(
                suggested_translation=response.suggested_translation,
                confidence=response.confidence,
                model_flagged_review=response.model_flagged_review,
                assessment=response.name_translation_assessment,
            )
            for response in responses
        ),
    )


def _merged_before_you_read(
    fragments: Sequence[FragmentAnalysis],
) -> tuple[BeforeYouReadEntry, ...]:
    """Chapter `Before You Read` materials: deduplicated expressions across fragments (PRD § 19.1)."""
    return before_you_read_entries(
        (entry for fragment in fragments for entry in fragment.before_you_read),
        strict=any(fragment.content_plan is not None and fragment.content_plan.version == 3 for fragment in fragments),
    )


_BYR_DIRECTIVE_PREFIX = re.compile(
    r"^(?:(?:(?:в|при)\s+переводе|для\s+перевода|(?:указание|инструкция)\s+переводчику|переводчику)\s*[:—–,-]?\s*)?"
    r"(?:(?:лучше|следует|рекомендуется|можно|нужно|необходимо|желательно|стоит)\s+)?"
    r"(?:передавай(?:те)?|передавать|передать|переводи(?:те)?|переводить|перевести|сохраняй(?:те)?|сохранять|сохранить|используй(?:те)?|использовать)"
    r"(?:\s+как|\s+в\s+значении|\s+через|\s+буквально|\b)\s*",
    re.IGNORECASE,
)

_BYR_META_PREFIX = re.compile(
    r"^(?:(?:указание|инструкция)\s+переводчику|переводчику)\s*[:—–,-]\s*",
    re.IGNORECASE,
)


def sanitize_before_you_read_explanation(text: str) -> str:
    """Sanitizes Before You Read explanation by stripping translator directives."""
    if not text:
        return ""
    s = text.strip()
    s = re.sub(
        r"\s*\((?:(?:в|при)\s+переводе|передавай|переводи|переводить|передавать)[^)]*\)",
        "",
        s,
        flags=re.IGNORECASE,
    )

    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    cleaned_sentences = []

    for sent in sentences:
        if ";" in sent:
            clauses = [c.strip() for c in sent.split(";") if c.strip()]
            kept_clauses = []
            for c in clauses:
                clean_c = c.rstrip(".!?")
                if _BYR_DIRECTIVE_PREFIX.match(clean_c) or _BYR_META_PREFIX.match(clean_c):
                    continue
                kept_clauses.append(c)
            if kept_clauses:
                reconstructed = "; ".join(kept_clauses)
                if not reconstructed.endswith((".", "!", "?")):
                    reconstructed += "."
                cleaned_sentences.append(reconstructed)
            continue

        clean_sent = sent.rstrip(".!?")
        if _BYR_DIRECTIVE_PREFIX.match(clean_sent) or _BYR_META_PREFIX.match(clean_sent):
            continue
        cleaned_sentences.append(sent)

    if not cleaned_sentences:
        first = sentences[0]
        match = _BYR_DIRECTIVE_PREFIX.match(first) or _BYR_META_PREFIX.match(first)
        if match:
            rest = first[match.end() :].strip()
            if rest:
                rest = rest[0].upper() + rest[1:]
                if not rest.endswith((".", "!", "?")):
                    rest += "."
                return rest
        return text.strip()

    result = " ".join(cleaned_sentences).strip()
    result = re.sub(r"\s+([,;:?.!])", r"\1", result)
    result = re.sub(r"\s+", " ", result)
    return result


_BYR_IRREGULAR_STEMS = {
    "made": "make",
    "making": "make",
    "taken": "take",
    "took": "take",
    "taking": "take",
    "gave": "give",
    "given": "give",
    "giving": "give",
    "kept": "keep",
}


def _stem_byr_word(word: str) -> str:
    word = word.casefold()
    if word in _BYR_IRREGULAR_STEMS:
        return _BYR_IRREGULAR_STEMS[word]
    for suffix in ("'s", "ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            base = word[: -len(suffix)]
            if suffix == "ies":
                return base + "y"
            if (
                suffix in ("ing", "ed")
                and len(base) >= 3
                and base[-1] == base[-2]
                and base[-1] not in "lsz"
            ):
                return base[:-1]
            return base
    return word


_BYR_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those",
        "some", "any", "each", "every", "either", "neither", "no", "such",
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "one", "ones", "oneself",
        "who", "whom", "whose", "which", "what",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "up", "down", "out", "off", "over", "under", "into", "onto", "upon",
        "about", "above", "across", "after", "against", "along", "amid", "among",
        "around", "as", "before", "behind", "below", "beneath", "beside",
        "between", "beyond", "despite", "during", "except", "inside", "near",
        "outside", "per", "since", "than", "through", "throughout", "till",
        "toward", "towards", "until", "via", "within", "without",
        "and", "or", "nor", "but", "so", "yet", "if", "though", "although",
        "because", "while", "when", "where", "whether", "not",
        "be", "am", "is", "are", "was", "were", "been", "being",
        "have", "has", "had", "having", "hav",
        "do", "does", "did",
        "will", "would", "shall", "should", "can", "cannot", "could",
        "may", "might", "must",
        "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't",
        "couldn't", "shouldn't", "mustn't",
    }
)


def _same_byr_prefix(words_a: Sequence[str], words_b: Sequence[str]) -> bool:
    if not words_a or not words_b:
        return False
    if len(words_a) == len(words_b):
        return all(
            _stem_byr_word(a) == _stem_byr_word(b)
            for a, b in zip(words_a, words_b, strict=True)
        )
    shorter, longer = (
        (words_a, words_b) if len(words_a) < len(words_b) else (words_b, words_a)
    )
    if len(shorter) < 2 or 2 * len(shorter) < len(longer):
        return False
    significant = sum(
        1
        for word in shorter
        if word.casefold() not in _BYR_STOP_WORDS
        and _stem_byr_word(word) not in _BYR_STOP_WORDS
    )
    if significant < 2:
        return False
    return all(
        _stem_byr_word(shorter[i]) == _stem_byr_word(longer[i])
        for i in range(len(shorter))
    )


def before_you_read_entries(
    entries: Iterable[BeforeYouReadEntryResponse], *, strict: bool = False,
) -> tuple[BeforeYouReadEntry, ...]:
    """Retains the first entry for each expression stem/prefix, up to ten chapter expressions."""
    selected: list[BeforeYouReadEntry] = []
    for entry in entries:
        words = expression_words(entry.expression)
        if not words or _before_you_read_stem_duplicate(entry, selected) is not None:
            continue
        selected.append(
            BeforeYouReadEntry(
                expression=entry.expression,
                explanation=entry.explanation if strict else sanitize_before_you_read_explanation(entry.explanation),
            )
        )
        if len(selected) >= MAX_BEFORE_YOU_READ:
            break
    return tuple(selected)


def _before_you_read_stem_duplicate(
    entry: BeforeYouReadEntryResponse | BeforeYouReadEntry,
    selected: Sequence[BeforeYouReadEntry],
) -> BeforeYouReadEntry | None:
    words = expression_words(entry.expression)
    if not words:
        return None
    return next(
        (
            candidate
            for candidate in selected
            if (candidate_words := expression_words(candidate.expression))
            and _same_byr_prefix(words, candidate_words)
        ),
        None,
    )


def _align_byr_evidence_excerpt(
    entry: BeforeYouReadEntry | BeforeYouReadEntryResponse, body_excerpt: str
) -> str | None:
    if body_excerpt not in entry.explanation:
        span = find_quote_span(entry.explanation, body_excerpt)
        if span is not None:
            body_excerpt = span[2]
        else:
            stripped = _quotation_key(body_excerpt)
            if stripped and stripped in entry.explanation:
                body_excerpt = stripped
    if (
        body_excerpt not in entry.explanation
        or _quotation_key(body_excerpt) == _quotation_key(entry.expression)
    ):
        return None
    return body_excerpt


def _before_you_read_key(expression: str) -> str:
    words = expression_words(expression)
    if not words:
        return ""
    return " ".join(_stem_byr_word(word) for word in words)


def _card_index_for_evidence(
    evidence: HelpEvidence, cards: Sequence[BeforeYouReadEntry]
) -> int | None:
    if evidence.artifact_key.startswith("byr:"):
        try:
            idx = int(evidence.artifact_key.split(":", 1)[1])
            if 0 <= idx < len(cards):
                return idx
        except ValueError:
            pass
    for idx, card in enumerate(cards):
        expected_key = artifact_key(
            evidence.artifact_kind,
            evidence.block_id,
            {"expression": card.expression, "explanation": card.explanation},
        )
        if evidence.artifact_key == expected_key:
            return idx
    for idx, card in enumerate(cards):
        if (
            evidence.body_excerpt == card.explanation
            or evidence.body_excerpt in card.explanation
        ):
            return idx
    return None


def deduplicate_book_before_you_read(
    document: CanonicalDocument, analyses: Sequence[ChapterAnalysis]
) -> tuple[ChapterAnalysis, ...]:
    """Deduplicates Before You Read cards across chapters in document section order (AC-007).

    Earlier chapters take precedence. In later chapters, duplicate cards by
    _before_you_read_key are removed unless a card provides the only evidence
    covering a required point or difficulty in that chapter's content plan.
    HelpEvidence.artifact_key is updated to reflect new card indices.
    """
    analyses_by_section = {a.section_id: a for a in analyses}
    ordered_section_ids = [
        s.id for s in document.ordered_sections if s.id in analyses_by_section
    ]

    seen_byr_keys: set[str] = set()
    updated_by_section: dict[str, ChapterAnalysis] = {}

    for section_id in ordered_section_ids:
        analysis = analyses_by_section[section_id]
        cards = analysis.before_you_read
        plan = analysis.content_plan

        if not cards:
            updated_by_section[section_id] = analysis
            continue

        required_card_indices: set[int] = set()
        if plan is not None and plan.before_you_read_evidence:
            point_cards: dict[tuple[str, str], set[int]] = {}
            difficulty_cards: dict[str, set[int]] = {}
            for e in plan.before_you_read_evidence:
                c_idx = _card_index_for_evidence(e, cards)
                if c_idx is not None:
                    point_cards.setdefault((e.difficulty_id, e.point_id), set()).add(
                        c_idx
                    )
                    difficulty_cards.setdefault(e.difficulty_id, set()).add(c_idx)

            for c_indices in point_cards.values():
                if len(c_indices) == 1:
                    required_card_indices.update(c_indices)
            for c_indices in difficulty_cards.values():
                if len(c_indices) == 1:
                    required_card_indices.update(c_indices)

        kept_cards: list[tuple[int, BeforeYouReadEntry]] = []
        for i, card in enumerate(cards):
            key = _before_you_read_key(card.expression)
            if not key:
                kept_cards.append((i, card))
                continue
            key_words = key.split()
            is_seen = key in seen_byr_keys or any(
                _same_byr_prefix(key_words, seen_key.split())
                for seen_key in seen_byr_keys
            )
            if is_seen:
                if i in required_card_indices:
                    kept_cards.append((i, card))
                    seen_byr_keys.add(key)
                else:
                    pass
            else:
                kept_cards.append((i, card))
                seen_byr_keys.add(key)

        if len(kept_cards) == len(cards):
            updated_by_section[section_id] = analysis
            continue

        new_cards = tuple(card for _, card in kept_cards)
        old_to_new = {old_i: new_i for new_i, (old_i, _) in enumerate(kept_cards)}

        new_plan = plan
        if plan is not None:
            new_evidences = []
            for e in plan.before_you_read_evidence:
                old_i = _card_index_for_evidence(e, cards)
                if old_i is not None and old_i in old_to_new:
                    new_i = old_to_new[old_i]
                    card = new_cards[new_i]
                    if e.artifact_key.startswith("byr:"):
                        new_key = f"byr:{new_i}"
                    else:
                        new_key = artifact_key(
                            e.artifact_kind,
                            e.block_id,
                            {"expression": card.expression, "explanation": card.explanation},
                        )
                    new_evidences.append(replace(e, artifact_key=new_key))
            new_plan = replace(plan, before_you_read_evidence=tuple(new_evidences))

        updated_by_section[section_id] = replace(
            analysis, before_you_read=new_cards, content_plan=new_plan
        )

    return tuple(updated_by_section.get(a.section_id, a) for a in analyses)


__all__ = [
    "DEFAULT_FRAGMENT_BLOCK_OVERHEAD",
    "DEFAULT_MAX_FRAGMENT_BLOCKS",
    "DEFAULT_MAX_FRAGMENT_CHARS",
    "DEFAULT_MAX_FRAGMENT_WEIGHT",
    "MAX_BEFORE_YOU_READ",
    "MAX_MERGED_INSTRUCTIONS",
    "TOKENS_PER_ANALYSED_BLOCK",
    "AnalysisMergeError",
    "ChapterFragment",
    "FragmentAnalysis",
    "before_you_read_entries",
    "content_plan_from_response",
    "deduplicate_book_before_you_read",
    "dominant_script",
    "find_quote_span",
    "fragment_blocks_for",
    "glossary_context_constraints",
    "merge_chapter_analysis",
    "merged_glossary_suggestion_keys",
    "plan_fragments",
    "sanitize_before_you_read_explanation",
    "select_staged_a2_summary",
    "unify_homogeneous_list_treatments",
    "validate_content_plan",
    "validate_fact_glossary",
]


def _exact_quote_spans(text: str, quote: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    if not text or not quote:
        return spans
    quote_len = len(quote)
    position = 0
    while True:
        found = text.find(quote, position)
        if found < 0:
            break
        spans.append((found, found + quote_len, text[found : found + quote_len]))
        position = found + 1
    return spans


_QUOTE_LENIENT_TOKEN_RE = re.compile(
    r"(?P<dash_sp>\s*(?:[—–]|(?<=\s)-(?=\s))\s*)|(?P<ws>\s+)|(?P<ch>.)",
    re.DOTALL,
)


def _lenient_quote_pattern(quote: str) -> str:
    """Typography-tolerant pattern used when the quote is not an exact substring."""
    pattern_parts: list[str] = []
    for match in _QUOTE_LENIENT_TOKEN_RE.finditer(quote):
        kind = match.lastgroup
        part = match.group(0)
        if kind == "dash_sp":
            pattern_parts.append(r"\s*[-—–]\s*")
        elif kind == "ws":
            pattern_parts.append(r"\s+")
        elif part in ("'", "’", "‘", "`"):
            pattern_parts.append(r"['’‘`]")
        elif part in ('"', "“", "”", "«", "»"):
            pattern_parts.append(r'["“”«»]')
        elif part == "-":
            pattern_parts.append(r"[-—–]")
        else:
            pattern_parts.append(re.escape(part))
    pattern = "".join(pattern_parts)
    pattern = re.sub(r"\b(she|he|it|that|there|what|who)(?:\\s\+is|\['’‘`\]s)\b", r"\1(?:\\s+is|['’‘`]s)", pattern, flags=re.I)
    pattern = re.sub(r"\b(I)(?:\\s\+am|\['’‘`\]m)\b", r"\1(?:\\s+am|['’‘`]m)", pattern, flags=re.I)
    pattern = re.sub(r"\b(you|we|they)(?:\\s\+are|\['’‘`\]re)\b", r"\1(?:\\s+are|['’‘`]re)", pattern, flags=re.I)
    pattern = re.sub(r"\b(do|does|did|is|are|was|were|have|has|had|could|would|should|must)(?:\\s\+not|n\['’‘`\]t)\b", r"\1(?:\\s+not|n['’‘`]t)", pattern, flags=re.I)
    pattern = re.sub(r"\b(?:will\\s\+not|won\['’‘`\]t)\b", r"(?:will\\s+not|won['’‘`]t)", pattern, flags=re.I)
    pattern = re.sub(r"\b(?:can\\s\+not|cannot|can\['’‘`\]t)\b", r"(?:can\\s+not|cannot|can['’‘`]t)", pattern, flags=re.I)
    return pattern


def _quote_spans(text: str, quote: str) -> list[tuple[int, int, str]]:
    """Every occurrence, exact first, then the same lenient fallbacks as before."""
    if not text or not quote:
        return []
    exact = _exact_quote_spans(text, quote)
    if exact:
        return exact
    try:
        pattern = _lenient_quote_pattern(quote)
        matches = [
            (match.start(), match.end(), text[match.start() : match.end()])
            for match in re.finditer(pattern, text)
        ]
        if matches:
            return matches
        folded = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if folded:
            return [
                (match.start(), match.end(), text[match.start() : match.end()])
                for match in folded
            ]
    except re.error:
        pass
    stripped = quote.strip(" \t\r\n.,;:!?\"'«»()[]“”‘’—–")
    if stripped and stripped != quote:
        return _quote_spans(text, stripped)
    return []


def find_quote_span(
    text: str, quote: str, *, estimated_start: int = 0
) -> tuple[int, int, str] | None:
    """Locates quote occurrences in text accounting for typography and whitespace."""
    spans = _quote_spans(text, quote)
    if not spans:
        return None
    return min(spans, key=lambda span: abs(span[0] - estimated_start))


def _quote_matches_anchor(quote: str, anchor_quote: str, difficulty_kind: str) -> bool:
    if not quote or not anchor_quote:
        return False
    if find_quote_span(quote, anchor_quote) is not None:
        return True
    span = find_quote_span(anchor_quote, quote)
    if span is not None:
        if (
            difficulty_kind == "grammar"
            and len(anchor_quote.split()) > 1
            and (len(quote.split()) < 2 or len(span[2].split()) < 2)
        ):
            return False
        return True
    return False


def _substance(text: str) -> str:
    return "".join(re.findall(r"\w+", without_footnote_markers(text).casefold()))


_NARRATIVE_RULE_PRIORITY = (
    "notes_only_scenes",
    "scene_bounds",
    "unresolved_quote",
    "scene_visual_anchor_out_of_bounds",
    "narrative_summary_language",
    "scene_summary_language",
    "translation_instructions_language",
    "name_assessment_conflict",
    "spoiler_name_leak",
    "speaker_subject_anchor",
    "gender_plural_subject",
    "context_fact_conflict",
    "glossary_fact_conflict",
)
_NARRATIVE_LANGUAGE_MIN_PROBABILITY = 0.05
_SCENE_SUMMARY_DETECTOR_MIN_LETTERS = 30
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_SPOILER_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:['’][A-Za-z]+)?")
_VISUAL_SAFE_EXCLUSIONS_RE = re.compile(
    r"\b(?:"
    r"беззвучн\w*|бесшумн\w*|soundless\w*"
    r"|музыкальн\w*\s+(?:магазин\w*|лавк\w*|инструмент\w*|шкатулк\w*|зал\w*|школ\w*|киоск\w*|колонк\w*|автомат\w*)"
    r"|(?:магазин\w*|лавк\w*)\s+музык\w*"
    r"|music(?:al)?\s+(?:shop|store|instrument|box|hall|room|stand|school)s?"
    r")\b",
    re.IGNORECASE,
)
_NON_VISUAL_FACT_RE = re.compile(
    r"\b(?:"
    # Russian sound markers
    r"звук\w*|звуч\w*|шум\w*|голос\w*|шёпот\w*|шепот\w*|крик\w*|вопл\w*|визг\w*"
    r"|грохот\w*|стук\w*|скрип(?:а|ом|е|у|ов|ами|ах|ит|ят|ел\w*|ну\w*|ящ\w*|уч\w*)?"
    r"|звон(?:а|ом|е|у|ит|ят|ил\w*|ел\w*|ящ\w*|кий|кая|кое|кие|кого|кой|кому|ким|ком|кую|ких|кими|ко)?"
    r"|гул(?:а|ом|е|у)?|гудит|гудел\w*"
    r"|свист\w*|шорох\w*|шелест\w*|хруст\w*|щёлкань\w*|щелкань\w*|щелчк\w*|щелчок\w*"
    r"|рычан\w*|рёв\w*|ревёт|ревет|ревел\w*|громк\w*|хрипл\w*|пискляв\w*"
    r"|мелоди\w*|музык\w*|эхо\w*|слышн\w*|слышит\w*|слышать\w*|аудио\w*|интонаци\w*"
    # Russian smell markers
    r"|запах\w*|аромат\w*|вонь\w*|воня\w*|пахн\w*|благоухан\w*|смрад\w*|зловон\w*|душист\w*"
    # Russian taste markers (taste-specific context; avoids visible foods/shops like «магазин сладостей» or «сладкий чай»)
    r"|вкус\w*|привкус\w*"
    r"|сладк\w*\s+(?:вкус\w*|привкус\w*|запах\w*|аромат\w*)|сладковат\w*"
    r"|горьк\w*\s+(?:вкус\w*|привкус\w*|запах\w*|аромат\w*)|горчит\w*|гореч\w*"
    r"|кисл(?:ый|ая|ое|ые|ого|ой|ому|ым|ом|ую|ых|ыми|оват\w*)|солён\w*|солен\w*"
    # English sound/smell/taste markers
    r"|sound(?:s|ed|ing)?|noise(?:s|y)?|noisy|loud(?:ly|er|est)?|voice(?:s)?"
    r"|whisper(?:s|ed|ing)?|scream(?:s|ed|ing)?|shout(?:s|ed|ing)?|echo(?:es|ed|ing)?"
    r"|creak(?:s|ed|ing)?|buzz(?:es|ed|ing)?|hum(?:s|med|ming)?|audible|audibly|melody|music"
    r"|smell(?:s|ed|ing|y)?|scent(?:s|ed)?|odou?r(?:s)?|aroma(?:s)?|stench|stink(?:s|ing)?"
    r"|fragrance(?:s)?|fragrant|reek(?:s|ed|ing)?"
    r"|taste(?:s|d|ing|y)?|flavou?r(?:s|ed)?|bitter(?:ly)?|sour(?:ly)?|salty|savou?ry|sweetness|aftertaste"
    r")\b",
    re.IGNORECASE,
)


def _is_non_visual_fact(detail: str) -> bool:
    """Return True when a scene visual fact describes sound, smell, or taste rather than sight."""
    cleaned = _VISUAL_SAFE_EXCLUSIONS_RE.sub(" ", detail)
    return _NON_VISUAL_FACT_RE.search(cleaned) is not None


def _has_cyrillic(text: str) -> bool:
    return any(
        unicodedata.name(char, "").startswith("CYRILLIC ")
        for char in text
        if char.isalpha()
    )


def _collect_narrative_language_issues(
    payload: NarrativeAnalysisResponse,
    scenes: Sequence[Scene],
    texts: Mapping[str, str],
    ordered_block_ids: Sequence[str],
    aliases_by_block_id: Mapping[str, str],
    *,
    source_language: str | None,
    target_language: str | None,
    language_detector: LanguageDetector | None,
    is_notes_only: bool,
) -> list[tuple[str, str, tuple[str, ...]]]:
    issues: list[tuple[str, str, tuple[str, ...]]] = []
    if source_language is not None:
        summary = payload.chapter_summary.strip()
        if summary:
            source_corpus = " ".join(
                texts.get(bid, "") for bid in ordered_block_ids if bid in texts
            ) or " ".join(texts.values())
            source_script = dominant_script(source_corpus)
            summary_script = dominant_script(summary)
            if source_script is not None and summary_script != source_script:
                issues.append(
                    (
                        "narrative_summary_language",
                        f"пересказ главы должен быть на языке оригинала ({source_language})",
                        (),
                    )
                )
            elif source_script == "LATIN" and _has_cyrillic(summary):
                issues.append(
                    (
                        "narrative_summary_language",
                        f"пересказ главы на языке оригинала ({source_language}) не должен содержать кириллицу",
                        (),
                    )
                )
            elif (
                language_detector is not None
                and not is_notes_only
                and sum(ch.isalpha() for ch in summary) >= _SCENE_SUMMARY_DETECTOR_MIN_LETTERS
            ):
                prob = language_detector.probability_for_language(summary, source_language)
                if prob is not None and prob < _NARRATIVE_LANGUAGE_MIN_PROBABILITY:
                    issues.append(
                        (
                            "narrative_summary_language",
                            f"пересказ главы не распознан как язык оригинала ({source_language})",
                            (),
                        )
                    )

    if target_language is not None and not is_notes_only:
        norm_target = target_language.strip().casefold().replace("_", "-").split("-", 1)[0]
        bad_scene_details: list[str] = []
        bad_scene_blocks: list[str] = []
        for sc in scenes:
            sc_summary = sc.summary.strip()
            if not sc_summary:
                continue
            bad = False
            if norm_target == "ru":
                if not _has_cyrillic(sc_summary):
                    bad = True
                elif (
                    language_detector is not None
                    and sum(ch.isalpha() for ch in sc_summary) >= _SCENE_SUMMARY_DETECTOR_MIN_LETTERS
                ):
                    prob = language_detector.probability_for_language(sc_summary, target_language)
                    if prob is not None and prob < _NARRATIVE_LANGUAGE_MIN_PROBABILITY:
                        latin_tokens = _LATIN_WORD_RE.findall(sc_summary)
                        if latin_tokens and all(token[0].isupper() for token in latin_tokens):
                            stripped_latin = _LATIN_WORD_RE.sub(" ", sc_summary)
                            recheck = language_detector.probability_for_language(
                                stripped_latin, target_language
                            )
                            if recheck is not None and recheck < _NARRATIVE_LANGUAGE_MIN_PROBABILITY:
                                bad = True
                        else:
                            bad = True
            elif (
                language_detector is not None
                and sum(ch.isalpha() for ch in sc_summary) >= _SCENE_SUMMARY_DETECTOR_MIN_LETTERS
            ):
                prob = language_detector.probability_for_language(sc_summary, target_language)
                if prob is not None and prob < _NARRATIVE_LANGUAGE_MIN_PROBABILITY:
                    bad = True
            if bad:
                start_label = aliases_by_block_id.get(sc.start_block_id, sc.start_block_id)
                bad_scene_details.append(f"{start_label}: «{_preview_quote(sc_summary)}»")
                bad_scene_blocks.append(sc.start_block_id)
        if bad_scene_details:
            issues.append(
                (
                    "scene_summary_language",
                    f"описание сцены должно быть на целевом языке ({target_language}) "
                    f"({'; '.join(dict.fromkeys(bad_scene_details))})",
                    tuple(dict.fromkeys(bad_scene_blocks)),
                )
            )
        if norm_target == "ru":
            bad_instruction_details: list[str] = []
            for point in _instruction_points(payload.translation_instructions):
                unquoted = _INSTRUCTION_QUOTED_RE.sub(" ", point)
                has_required_cyrillic = (
                    _has_cyrillic(unquoted)
                    if len(_LATIN_WORD_RE.findall(unquoted)) >= 2
                    else _has_cyrillic(point)
                )
                if not has_required_cyrillic:
                    bad_instruction_details.append(f"«{_preview_quote(point)}»")
            if bad_instruction_details:
                issues.append(
                    (
                        "translation_instructions_language",
                        f"инструкции переводчику должны быть на целевом языке ({target_language}) "
                        f"({'; '.join(dict.fromkeys(bad_instruction_details))})",
                        (),
                    )
                )
    return issues


_ALLOWED_REVIEW_REASONS_HINT = (
    "(допустимые review_reasons: "
    + ", ".join(reason.value for reason in NameTranslationReviewReason)
    + ")"
)
_DESCRIPTIVE_LOWERCASE_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+[a-z]")


def _collect_name_assessment_conflicts(
    candidates: Sequence[GlossaryCandidateResponse],
    *,
    name_translation_policy: NameTranslationPolicy | None = None,
) -> list[str]:
    details: list[str] = []
    for candidate in candidates:
        assessment = candidate.name_translation_assessment
        if assessment is None:
            continue
        method = assessment.applied_method
        reasons = set(assessment.review_reasons)
        if method is NameTranslationMethod.SOUND and (
            reasons
            & {
                NameTranslationReviewReason.CREATIVE_ADAPTATION,
                NameTranslationReviewReason.PERSONAL_NAME_SUBSTITUTION,
            }
        ):
            details.append(
                f"«{candidate.source_text}»: метод sound несовместим с creative_adaptation/personal_name_substitution "
                f"{_ALLOWED_REVIEW_REASONS_HINT}"
            )
        if (
            method is NameTranslationMethod.SOUND
            and assessment.opaque_name
            and (
                reasons
                & {
                    NameTranslationReviewReason.WORDPLAY,
                    NameTranslationReviewReason.ANAGRAM,
                }
            )
        ):
            details.append(
                f"«{candidate.source_text}»: непрозрачное имя с методом sound несовместимо с wordplay/anagram "
                f"{_ALLOWED_REVIEW_REASONS_HINT}"
            )
        if method is NameTranslationMethod.MEANING and assessment.opaque_name:
            details.append(
                f"«{candidate.source_text}»: метод meaning несовместим с opaque_name=True "
                f"{_ALLOWED_REVIEW_REASONS_HINT}"
            )
        if (
            method in {NameTranslationMethod.CREATIVE, NameTranslationMethod.UNCERTAIN}
            and assessment.eligible_without_review
        ):
            details.append(
                f"«{candidate.source_text}»: метод {method.value} не может иметь eligible_without_review=True "
                f"{_ALLOWED_REVIEW_REASONS_HINT}"
            )
        if (
            name_translation_policy is not None
            and name_translation_policy.strategy is NameTranslationStrategy.MEANING
            and method is NameTranslationMethod.SOUND
            and not assessment.opaque_name
            and assessment.eligible_without_review
        ):
            details.append(
                f"«{candidate.source_text}»: при стратегии meaning прозрачное имя (opaque_name=False) с методом sound "
                f"не может иметь eligible_without_review=True {_ALLOWED_REVIEW_REASONS_HINT}"
            )
        if (
            method is NameTranslationMethod.SOUND
            and not assessment.opaque_name
            and _DESCRIPTIVE_LOWERCASE_ARTICLE_RE.match(candidate.source_text.strip()) is not None
        ):
            details.append(
                f"«{candidate.source_text}»: описательная именная группа с opaque_name=False "
                f"должна переводиться по смыслу (meaning), а не методом sound {_ALLOWED_REVIEW_REASONS_HINT}"
            )
    return list(dict.fromkeys(details))


def _strip_possessive_token(token: str) -> str:
    lowered = token.casefold().replace("ё", "е")
    if lowered.endswith(("'s", "’s")) and len(lowered) > 2:
        return lowered[:-2]
    return lowered.rstrip("'’")


def _spoiler_name_stems(name: str) -> frozenset[str]:
    cleaned = re.sub(r"\([^)]*\)", " ", name)
    stems: set[str] = set()
    for raw in _SPOILER_TOKEN_RE.findall(cleaned):
        base = _strip_possessive_token(raw)
        if (
            len(base) < 2
            or base in _ALL_GENDER_TITLES
            or base in _SPEAKER_NAME_STOPWORDS
        ):
            continue
        stems.add(base)
    return frozenset(stems)


def _same_spoiler_name(
    left: str,
    right: str,
    glossary_candidates: Sequence[GlossaryCandidateResponse] = (),
) -> bool:
    key_a = glossary_key(left)
    key_b = glossary_key(right)
    if key_a and key_a == key_b:
        return True
    titles_a = _title_tokens(left)
    titles_b = _title_tokens(right)
    if _titles_conflict(titles_a, titles_b):
        return False
    stems_a = _spoiler_name_stems(left)
    stems_b = _spoiler_name_stems(right)
    if stems_a and stems_b and (stems_a <= stems_b or stems_b <= stems_a):
        return True
    for candidate in glossary_candidates:
        src = candidate.source_text
        tr = candidate.suggested_translation
        if not src or not tr:
            continue
        src_stems = _spoiler_name_stems(src)
        tr_stems = _spoiler_name_stems(tr)
        a_matches_src = bool(stems_a and src_stems and (stems_a <= src_stems or src_stems <= stems_a))
        a_matches_tr = bool(stems_a and tr_stems and (stems_a <= tr_stems or tr_stems <= stems_a))
        b_matches_src = bool(stems_b and src_stems and (stems_b <= src_stems or src_stems <= stems_b))
        b_matches_tr = bool(stems_b and tr_stems and (stems_b <= tr_stems or tr_stems <= stems_b))
        if (a_matches_src and b_matches_tr) or (a_matches_tr and b_matches_src):
            return True
    return False


def _extract_block_spoiler_tokens(
    ordered_block_ids: Sequence[str],
    texts: Mapping[str, str],
) -> tuple[tuple[tuple[str, bool, str | None], ...], ...]:
    result: list[tuple[tuple[str, bool, str | None], ...]] = []
    for bid in ordered_block_ids:
        text = texts.get(bid, "")
        tokens: list[tuple[str, bool, str | None]] = []
        prev: str | None = None
        for match in _SPOILER_TOKEN_RE.finditer(text):
            raw = match.group(0)
            base = _strip_possessive_token(raw)
            tokens.append((base, raw[:1].isupper(), prev))
            prev = base
        result.append(tuple(tokens))
    return tuple(result)


_LEADING_ROLE_ARTICLES = frozenset({"the", "a", "an"})
_ROLE_PREPOSITIONS = frozenset(
    {
        "in",
        "of",
        "with",
        "by",
        "from",
        "on",
        "at",
        "to",
        "for",
        "near",
        "behind",
        "в",
        "из",
        "с",
        "у",
        "на",
        "от",
        "за",
    }
)
_NAME_PREFIX_TITLES = frozenset(
    {
        "mr",
        "mrs",
        "miss",
        "ms",
        "dr",
        "professor",
        "sir",
        "madam",
        "madame",
        "lord",
        "lady",
        "uncle",
        "aunt",
        "мистер",
        "миссис",
        "мисс",
        "профессор",
        "дядя",
        "тётя",
        "тетя",
        "сэр",
        "лорд",
        "мадам",
        "леди",
    }
)
_ANONYMOUS_PARTICIPANT_MODIFIERS = frozenset(
    {
        "unnamed",
        "unknown",
        "unidentified",
        "anonymous",
        "unseen",
        "mysterious",
        "someone",
        "somebody",
        "неназванный",
        "неназванная",
        "неназванное",
        "неназванные",
        "неизвестный",
        "неизвестная",
        "неизвестное",
        "неизвестные",
        "безымянный",
        "безымянная",
        "безымянные",
        "незнакомый",
        "незнакомая",
        "незнакомые",
        "неустановленный",
        "неустановленная",
        "неустановленные",
    }
)
_GENERIC_ROLE_PARTICIPANTS = frozenset(
    {
        "person",
        "people",
        "figure",
        "secretary",
        "crowd",
        "passer",
        "passers",
        "passerby",
        "passersby",
        "customer",
        "customers",
        "shopper",
        "shoppers",
        "whisperer",
        "whisperers",
        "boy",
        "boys",
        "girl",
        "girls",
        "man",
        "men",
        "woman",
        "women",
        "child",
        "children",
        "kid",
        "kids",
        "baby",
        "stranger",
        "strangers",
        "driver",
        "drivers",
        "giant",
        "guard",
        "guards",
        "soldier",
        "officer",
        "clerk",
        "clerks",
        "waiter",
        "waitress",
        "teacher",
        "teachers",
        "doctor",
        "nurse",
        "captain",
        "king",
        "queen",
        "prince",
        "princess",
        "servant",
        "maid",
        "butler",
        "porter",
        "conductor",
        "pilot",
        "sailor",
        "farmer",
        "baker",
        "shopkeeper",
        "merchant",
        "priest",
        "monk",
        "witch",
        "witches",
        "wizard",
        "wizards",
        "ghost",
        "monster",
        "narrator",
        "speaker",
        "voice",
        "visitor",
        "visitors",
        "guest",
        "guests",
        "passenger",
        "passengers",
        "neighbor",
        "neighbors",
        "neighbour",
        "neighbours",
        "friend",
        "friends",
        "mother",
        "father",
        "brother",
        "sister",
        "son",
        "daughter",
        "husband",
        "wife",
        "grandmother",
        "grandfather",
        "aunt",
        "uncle",
        "cousin",
        "boss",
        "chief",
        "leader",
        "student",
        "students",
        "pupil",
        "pupils",
        "lawyer",
        "judge",
        "policeman",
        "detective",
        "thief",
        "bandit",
        "pirate",
        "knight",
        "hunter",
        "keeper",
        "owner",
        "host",
        "bartender",
        "barman",
        "cook",
        "chef",
        "dog",
        "cat",
        "cats",
        "horse",
        "bird",
        "wolf",
        "owl",
        "owls",
        "rat",
        "toad",
        "мальчик",
        "девочка",
        "девушка",
        "юноша",
        "мужчина",
        "женщина",
        "старик",
        "старуха",
        "ребенок",
        "незнакомец",
        "незнакомка",
        "незнакомцы",
        "водитель",
        "великан",
        "волшебник",
        "ведьма",
        "охранник",
        "стражник",
        "солдат",
        "офицер",
        "продавец",
        "официант",
        "учитель",
        "доктор",
        "врач",
        "капитан",
        "король",
        "королева",
        "слуга",
        "повар",
        "хозяин",
        "хозяйка",
        "гость",
        "пассажир",
        "сосед",
        "соседка",
        "друг",
        "подруга",
        "мать",
        "отец",
        "брат",
        "сестра",
        "сын",
        "дочь",
        "рассказчик",
        "человек",
        "люди",
        "фигура",
        "секретарь",
        "прохожий",
        "прохожие",
        "толпа",
        "кот",
        "кошка",
        "сова",
    }
)


def _matches_glossary_name(
    name: str,
    glossary_candidates: Sequence[GlossaryCandidateResponse],
) -> bool:
    key = glossary_key(name)
    if not key:
        return False
    cleaned = re.sub(r"\([^)]*\)", " ", name).strip()
    tokens = _SPOILER_TOKEN_RE.findall(cleaned)
    has_any_upper = any(token[:1].isupper() for token in tokens)
    has_non_leading_upper = any(token[:1].isupper() for token in tokens[1:])
    has_proper_case = has_non_leading_upper if len(tokens) >= 2 else has_any_upper
    last_base = _strip_possessive_token(tokens[-1]) if tokens else ""
    name_stems = _spoiler_name_stems(name)
    generic_tokens = _GENERIC_ROLE_PARTICIPANTS | _ANONYMOUS_PARTICIPANT_MODIFIERS
    for candidate in glossary_candidates:
        for form in (candidate.source_text, candidate.suggested_translation):
            if form and glossary_key(form) == key:
                return True
        if not name_stems:
            continue
        if candidate.type is ConceptType.PERSON:
            for form in (candidate.source_text, candidate.suggested_translation):
                if not form:
                    continue
                form_stems = _spoiler_name_stems(form)
                if not form_stems:
                    continue
                if not (form_stems <= generic_tokens) and form_stems <= name_stems:
                    return True
                if (
                    not (name_stems <= generic_tokens)
                    and (has_proper_case or last_base not in _GENERIC_ROLE_PARTICIPANTS)
                    and name_stems <= form_stems
                ):
                    return True
        else:
            for form in (candidate.source_text, candidate.suggested_translation):
                if not form:
                    continue
                form_stems = _spoiler_name_stems(form)
                if len(form_stems) >= 2 and form_stems == name_stems:
                    return True
    return False


def _role_head_base(bases: Sequence[str]) -> str:
    for idx in range(1, len(bases)):
        if bases[idx] in _ROLE_PREPOSITIONS:
            return bases[idx - 1]
    return bases[-1]


def _is_non_name_role_label(
    name: str,
    *,
    matches_glossary: bool,
) -> bool:
    """True for neutral descriptive/anonymous role labels that cannot be name spoilers."""
    if matches_glossary:
        return False
    cleaned = re.sub(r"\([^)]*\)", " ", name).strip()
    tokens = _SPOILER_TOKEN_RE.findall(cleaned)
    if not tokens:
        return True
    bases = [_strip_possessive_token(token) for token in tokens]
    first_base = bases[0]
    has_any_upper = any(token[:1].isupper() for token in tokens)
    has_non_leading_upper = any(token[:1].isupper() for token in tokens[1:])

    if (
        len(tokens) >= 2
        and first_base in _NAME_PREFIX_TITLES
        and bases[-1] not in _GENERIC_ROLE_PARTICIPANTS
    ):
        return False

    if any(base in _ANONYMOUS_PARTICIPANT_MODIFIERS for base in bases):
        remaining = [
            base
            for base in bases
            if base not in _ANONYMOUS_PARTICIPANT_MODIFIERS
            and base not in _LEADING_ROLE_ARTICLES
            and base not in _ROLE_PREPOSITIONS
        ]
        if not remaining or all(base in _GENERIC_ROLE_PARTICIPANTS for base in remaining):
            return True
        if (
            not has_non_leading_upper
            and _role_head_base(remaining) in _GENERIC_ROLE_PARTICIPANTS
        ):
            return True

    if (
        len(tokens) >= 2
        and first_base in _LEADING_ROLE_ARTICLES
        and not has_non_leading_upper
    ):
        return True

    if not has_any_upper and len(tokens) == 1:
        return (
            first_base in _GENERIC_ROLE_PARTICIPANTS
            or first_base in _ANONYMOUS_PARTICIPANT_MODIFIERS
        )

    if len(tokens) >= 2 and not has_non_leading_upper:
        if tokens[0][:1].isupper() and tokens[0].endswith(("'s", "’s")):
            return False
        if _role_head_base(bases) in _GENERIC_ROLE_PARTICIPANTS:
            return True

    return False


def _prefers_capitalized_name_match(name: str) -> bool:
    cleaned = re.sub(r"\([^)]*\)", " ", name).strip()
    tokens = _SPOILER_TOKEN_RE.findall(cleaned)
    if not tokens:
        return False
    if (
        tokens[0].casefold() in _LEADING_ROLE_ARTICLES
        and not any(token[:1].isupper() for token in tokens[1:])
    ):
        return False
    stems = _spoiler_name_stems(cleaned)
    if not stems:
        return False
    stem_tokens = [
        token for token in tokens if _strip_possessive_token(token) in stems
    ]
    if not stem_tokens or not all(token[:1].isupper() for token in stem_tokens):
        return False
    if (
        len(stems) == 1
        and not _title_tokens(cleaned)
        and next(iter(stems)) in _GENERIC_ROLE_PARTICIPANTS
    ):
        return False
    return True


def _first_name_block_index(
    name: str,
    block_tokens: Sequence[tuple[tuple[str, bool, str | None], ...]],
    glossary_candidates: Sequence[GlossaryCandidateResponse],
) -> int | None:
    matches_glossary = _matches_glossary_name(name, glossary_candidates)
    if _is_non_name_role_label(name, matches_glossary=matches_glossary):
        return None
    name_key = glossary_key(name)
    name_stems = _spoiler_name_stems(name)
    generic_tokens = _GENERIC_ROLE_PARTICIPANTS | _ANONYMOUS_PARTICIPANT_MODIFIERS
    forms: list[str] = [name]
    for candidate in glossary_candidates:
        if not (candidate.source_text and candidate.suggested_translation):
            continue
        if glossary_key(candidate.suggested_translation) == name_key:
            forms.append(candidate.source_text)
        elif (
            candidate.type is ConceptType.PERSON
            and name_stems
            and not (name_stems <= generic_tokens)
            and _same_spoiler_name(name, candidate.suggested_translation)
        ):
            forms.append(candidate.source_text)
    stems: set[str] = set()
    titles: set[str] = set()
    for form in forms:
        stems.update(_spoiler_name_stems(form))
        titles.update(_title_tokens(form))
    non_anonymous_stems = stems - _ANONYMOUS_PARTICIPANT_MODIFIERS
    if non_anonymous_stems:
        stems = non_anonymous_stems
    if not stems:
        return None
    name_titles = frozenset(titles)
    cleaned_name = re.sub(r"\([^)]*\)", " ", name).strip()
    name_tokens = _SPOILER_TOKEN_RE.findall(cleaned_name)
    all_lowercase = bool(name_tokens) and not any(
        token[:1].isupper() for token in name_tokens
    )
    lowercase_non_glossary = all_lowercase and not matches_glossary
    single_capitalized_generic = (
        len(name_tokens) == 1
        and name_tokens[0][:1].isupper()
        and _strip_possessive_token(name_tokens[0]) in _GENERIC_ROLE_PARTICIPANTS
        and not matches_glossary
    )
    prefer_capitalized = (all_lowercase and matches_glossary) or any(
        _prefers_capitalized_name_match(form) for form in forms
    )

    def scan(*, require_capitalized: bool) -> int | None:
        for idx, tokens in enumerate(block_tokens):
            for base, is_cap, prev in tokens:
                if base not in stems:
                    continue
                if require_capitalized and not is_cap:
                    continue
                if (
                    prev is not None
                    and prev in _ALL_GENDER_TITLES
                    and _titles_conflict(name_titles, frozenset({prev}))
                ):
                    continue
                return idx
        return None

    if lowercase_non_glossary:
        return scan(require_capitalized=True)
    if single_capitalized_generic:
        if scan(require_capitalized=True) is None:
            return None
        return scan(require_capitalized=False)
    if prefer_capitalized:
        found = scan(require_capitalized=True)
        if found is not None:
            return found
    return scan(require_capitalized=False)


def _collect_spoiler_name_leaks(
    scenes: Sequence[Scene],
    mapped_facts: Sequence[ContextFact],
    texts: Mapping[str, str],
    ordered_block_ids: Sequence[str],
    block_order: Mapping[str, int],
    known_names: Iterable[str],
    glossary_candidates: Sequence[GlossaryCandidateResponse],
    aliases_by_block_id: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    known = tuple(name for name in known_names if isinstance(name, str) and name.strip())
    block_tokens = _extract_block_spoiler_tokens(ordered_block_ids, texts)
    details: list[str] = []
    block_ids: list[str] = []

    for sc in scenes:
        end_pos = block_order.get(sc.end_block_id)
        if end_pos is None:
            continue
        start_label = aliases_by_block_id.get(sc.start_block_id, sc.start_block_id)
        end_label = aliases_by_block_id.get(sc.end_block_id, sc.end_block_id)
        for participant in sc.participants:
            if not participant.strip():
                continue
            if any(
                _same_spoiler_name(participant, kn, glossary_candidates) for kn in known
            ):
                continue
            first_pos = _first_name_block_index(participant, block_tokens, glossary_candidates)
            if first_pos is not None and first_pos > end_pos:
                details.append(
                    f"сцена {start_label}..{end_label}: участник «{participant}» появляется позже в главе"
                )
                block_ids.append(sc.start_block_id)

    for fact in mapped_facts:
        if fact.certainty != "explicit":
            continue
        if fact.kind not in {"speaker", "pronoun_referent"} or not fact.value:
            continue
        fact_pos = block_order.get(fact.subject_anchor.block_id)
        if fact_pos is None:
            continue
        first_pos = _first_name_block_index(fact.value, block_tokens, glossary_candidates)
        if first_pos is not None and first_pos > fact_pos:
            label = aliases_by_block_id.get(
                fact.subject_anchor.block_id, fact.subject_anchor.block_id
            )
            details.append(
                f"{label}: явная атрибуция «{fact.value}» опережает первое упоминание в главе"
            )
            block_ids.append(fact.subject_anchor.block_id)

    return list(dict.fromkeys(details)), list(dict.fromkeys(block_ids))


def _format_unresolved_anchor_issue(
    block_alias: str,
    quote: str,
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None,
    visible_texts_by_alias: Mapping[str, str] | None,
    aliases_by_block_id: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Diagnose an unresolved quoted anchor, preserving block label and quote text."""
    diag = _diagnose_quoted_anchor(
        block_alias, quote, aliases, texts, visible_texts_by_alias, aliases_by_block_id
    )
    if diag is not None:
        block_label, preview, reason, canonical = diag
        return f"{block_label}: «{preview}» — {reason}", canonical
    block_id = aliases.get(block_alias)
    block_label = aliases_by_block_id.get(block_id, block_alias) if block_id else block_alias
    canonical = (block_id,) if block_id and texts is not None and block_id in texts else ()
    return f"{block_label}: «{_preview_quote(quote)}» — цитата не разрешается в исходном тексте блока", canonical


def _raise_collected_narrative_issues(
    issues: Sequence[tuple[str, str, tuple[str, ...]]],
) -> None:
    """Raise one AnalysisMergeError combining all collected narrative issues."""
    rule_code = min(
        (code for code, _, _ in issues),
        key=lambda code: (
            _NARRATIVE_RULE_PRIORITY.index(code)
            if code in _NARRATIVE_RULE_PRIORITY
            else len(_NARRATIVE_RULE_PRIORITY)
        ),
    )
    message = "; ".join(dict.fromkeys(msg for _, msg, _ in issues))
    block_ids = tuple(dict.fromkeys(bid for _, _, bids in issues for bid in bids))
    raise _analysis_error(message, rule_code=rule_code, block_ids=block_ids)


def _validate_scene_visual_facts(
    visual_facts: Sequence[QuotedSceneVisualFactResponse],
    start_pos: int,
    end_pos: int,
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    visible_texts_by_alias: Mapping[str, str],
    aliases_by_block_id: Mapping[str, str],
    positions: Mapping[str, int],
    *,
    blocks: Sequence[DocumentBlock] | None,
    source_offsets: bool,
    vf_unresolved_details: list[str],
    vf_unresolved_blocks: list[str],
    vf_oob_details: list[str],
    vf_oob_blocks: list[str],
) -> tuple[SceneVisualFact, ...]:
    """Resolve visual facts of a scene with valid bounds, filtering non-visual details."""
    facts: list[SceneVisualFact] = []
    for fact in visual_facts:
        if _is_non_visual_fact(fact.detail):
            continue
        anchor = _resolve_quoted_anchor(
            fact.anchor.block_id,
            fact.anchor.quote,
            aliases,
            texts,
            visible_texts_by_alias,
            start=fact.anchor.start,
            end=fact.anchor.end,
            strict=False,
            blocks=blocks,
            source_offsets=source_offsets,
        )
        if anchor is _AMBIGUOUS:
            continue
        if not isinstance(anchor, SourceAnchor):
            detail, canonical = _format_unresolved_anchor_issue(
                fact.anchor.block_id,
                fact.anchor.quote,
                aliases,
                texts,
                visible_texts_by_alias,
                aliases_by_block_id,
            )
            vf_unresolved_details.append(detail)
            vf_unresolved_blocks.extend(bid for bid in canonical if bid in positions)
            continue
        if anchor.block_id not in positions or not start_pos <= positions[anchor.block_id] <= end_pos:
            fact_label = aliases_by_block_id.get(anchor.block_id, fact.anchor.block_id)
            vf_oob_details.append(f"{fact_label}: «{_preview_quote(anchor.quote)}»")
            if anchor.block_id in positions:
                vf_oob_blocks.append(anchor.block_id)
            continue
        facts.append(SceneVisualFact(anchor, fact.subject, fact.detail))
    return tuple(facts)


def _inspect_scenes(
    scenes: Sequence[QuotedSceneResponse],
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    visible_texts_by_alias: Mapping[str, str],
    *,
    ordered_block_ids: Sequence[str],
    blocks: Sequence[DocumentBlock] | None = None,
    source_offsets: bool = False,
    check_coverage: bool = False,
) -> tuple[tuple[Scene, ...], list[tuple[str, str, tuple[str, ...]]]]:
    """Resolve valid scenes and collect all bounds, coverage, and visual-fact issues."""
    positions = {block_id: index for index, block_id in enumerate(ordered_block_ids)}
    aliases_by_block_id = {block_id: alias for alias, block_id in aliases.items() if block_id}
    resolved: list[Scene] = []
    covered_positions: set[int] = set()
    previous_end = -1
    bounds_details: list[str] = []
    bounds_blocks: list[str] = []
    vf_unresolved_details: list[str] = []
    vf_unresolved_blocks: list[str] = []
    vf_oob_details: list[str] = []
    vf_oob_blocks: list[str] = []

    for order, raw in enumerate(scenes):
        start, end = aliases.get(raw.start_block_id), aliases.get(raw.end_block_id)
        start_ok = (
            start is not None
            and start in positions
            and raw.start_block_id in visible_texts_by_alias
            and start in texts
        )
        end_ok = (
            end is not None
            and end in positions
            and raw.end_block_id in visible_texts_by_alias
            and end in texts
        )
        if not start_ok or not end_ok:
            bad_labels = tuple(
                dict.fromkeys(
                    label
                    for label, ok in ((raw.start_block_id, start_ok), (raw.end_block_id, end_ok))
                    if not ok
                )
            )
            bounds_details.append(f"границы вне главы: {', '.join(bad_labels)}")
            for block_id in (start, end):
                if block_id in positions:
                    bounds_blocks.append(block_id)
            continue

        start_pos, end_pos = positions[start], positions[end]
        start_label = aliases_by_block_id.get(start, raw.start_block_id)
        end_label = aliases_by_block_id.get(end, raw.end_block_id)
        if start_pos > end_pos:
            bounds_details.append(f"перевёрнутые границы: {start_label}..{end_label}")
            bounds_blocks.extend((start, end))
            continue

        scene_range = range(start_pos, end_pos + 1)
        overlaps = [idx for idx in scene_range if idx in covered_positions]
        if overlaps:
            overlap_ids = [ordered_block_ids[idx] for idx in overlaps]
            overlap_labels = tuple(
                dict.fromkeys(aliases_by_block_id.get(bid, bid) for bid in overlap_ids)
            )
            bounds_details.append(f"пересекающиеся блоки: {', '.join(overlap_labels)}")
            bounds_blocks.extend(overlap_ids)
        elif start_pos <= previous_end:
            bounds_details.append(f"нарушен порядок сцен: {start_label}..{end_label}")
            bounds_blocks.extend((start, end))
        covered_positions.update(scene_range)
        previous_end = max(previous_end, end_pos)

        facts = _validate_scene_visual_facts(
            raw.visual_facts,
            start_pos,
            end_pos,
            aliases,
            texts,
            visible_texts_by_alias,
            aliases_by_block_id,
            positions,
            blocks=blocks,
            source_offsets=source_offsets,
            vf_unresolved_details=vf_unresolved_details,
            vf_unresolved_blocks=vf_unresolved_blocks,
            vf_oob_details=vf_oob_details,
            vf_oob_blocks=vf_oob_blocks,
        )

        item = Scene(
            "pending",
            order,
            start,
            end,
            raw.summary,
            raw.location,
            raw.participants,
            facts,
            raw.visual_assumptions,
        )
        resolved.append(replace(item, id=item.canonical_id))

    if check_coverage and ordered_block_ids:
        missing_ids = [
            ordered_block_ids[idx]
            for idx in range(len(ordered_block_ids))
            if idx not in covered_positions
        ]
        if missing_ids:
            missing_labels = [aliases_by_block_id.get(bid, bid) for bid in missing_ids]
            bounds_details.append(f"пропущены блоки: {', '.join(missing_labels)}")
            bounds_blocks.extend(missing_ids)

    issues: list[tuple[str, str, tuple[str, ...]]] = []
    if bounds_details:
        prefix = (
            "карта сцен должна непрерывно покрывать все блоки главы"
            if check_coverage
            else "границы сцен должны принадлежать главе, идти по порядку и не пересекаться"
        )
        unique_bounds = tuple(dict.fromkeys(bounds_details))
        issues.append(
            (
                "scene_bounds",
                f"{prefix} ({'; '.join(unique_bounds)})",
                tuple(dict.fromkeys(bounds_blocks)),
            )
        )
    if vf_unresolved_details:
        unique_vf = tuple(dict.fromkeys(vf_unresolved_details))
        issues.append(
            (
                "unresolved_quote",
                f"цитата визуального факта не разрешается в исходном блоке ({'; '.join(unique_vf)})",
                tuple(dict.fromkeys(vf_unresolved_blocks)),
            )
        )
    if vf_oob_details:
        unique_oob = tuple(dict.fromkeys(vf_oob_details))
        issues.append(
            (
                "scene_visual_anchor_out_of_bounds",
                f"визуальный факт находится вне сцены ({'; '.join(unique_oob)})",
                tuple(dict.fromkeys(vf_oob_blocks)),
            )
        )
    return tuple(resolved), issues


def resolve_scenes(
    scenes: Sequence[QuotedSceneResponse],
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    visible_texts_by_alias: Mapping[str, str],
    *,
    ordered_block_ids: Sequence[str],
    blocks: Sequence[DocumentBlock] | None = None,
    source_offsets: bool = False,
) -> tuple[Scene, ...]:
    """Resolve scene references within the visible chapter or fragment."""
    resolved, issues = _inspect_scenes(
        scenes,
        aliases,
        texts,
        visible_texts_by_alias,
        ordered_block_ids=ordered_block_ids,
        blocks=blocks,
        source_offsets=source_offsets,
        check_coverage=False,
    )
    if issues:
        _raise_collected_narrative_issues(issues)
    return resolved


def _resolve_quoted_fact_entries(
    context_facts: Sequence[QuotedContextFactResponse],
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None,
    visible_texts_by_alias: Mapping[str, str] | None,
    aliases_by_block_id: Mapping[str, str],
    *,
    blocks: Sequence[DocumentBlock] | None = None,
    source_offsets: bool = False,
) -> tuple[
    list[tuple[QuotedContextFactResponse, tuple[SourceAnchor, ...], SourceAnchor]],
    list[str],
    list[str],
]:
    """Resolve quoted context facts in one pass without stopping on broken anchors."""
    resolved_facts: list[tuple[QuotedContextFactResponse, tuple[SourceAnchor, ...], SourceAnchor]] = []
    missing_details: list[str] = []
    missing_blocks: list[str] = []

    def resolve_anchor(anchor: QuotedSourceAnchorResponse):
        resolved = _resolve_quoted_anchor(
            anchor.block_id,
            anchor.quote,
            aliases,
            texts,
            visible_texts_by_alias,
            start=anchor.start,
            end=anchor.end,
            blocks=blocks,
            source_offsets=source_offsets,
        )
        if resolved is _MISSING:
            detail, canonical = _format_unresolved_anchor_issue(
                anchor.block_id,
                anchor.quote,
                aliases,
                texts,
                visible_texts_by_alias,
                aliases_by_block_id,
            )
            missing_details.append(detail)
            missing_blocks.extend(canonical)
        return resolved

    for fact in context_facts:
        subject_anchor = resolve_anchor(fact.subject_anchor)
        fact_missing = subject_anchor is _MISSING
        kept_anchors: list[SourceAnchor] = []
        for anchor in fact.anchors:
            resolved = resolve_anchor(anchor)
            if resolved is _MISSING:
                fact_missing = True
            elif resolved is not _AMBIGUOUS:
                kept_anchors.append(resolved)
        if not fact_missing and isinstance(subject_anchor, SourceAnchor) and kept_anchors:
            resolved_facts.append((fact, tuple(kept_anchors), subject_anchor))
    return (
        resolved_facts,
        list(dict.fromkeys(missing_details)),
        list(dict.fromkeys(missing_blocks)),
    )


def _conflicting_subject_block_ids(facts: Sequence[ContextFact]) -> tuple[str, ...]:
    subjects: dict[tuple[str, SourceAnchor], set[tuple[str | None, str]]] = {}
    for fact in facts:
        subjects.setdefault((fact.kind, fact.subject_anchor), set()).add(
            (fact.value, fact.certainty)
        )
    return tuple(
        dict.fromkeys(
            anchor.block_id
            for (_, anchor), values in subjects.items()
            if len(values) > 1
        )
    )


def _alias_domain_conflict_message(
    message: str,
    block_ids: Sequence[str],
    aliases_by_block_id: Mapping[str, str],
) -> str:
    aliased = message
    for block_id in sorted(set(block_ids), key=len, reverse=True):
        alias = aliases_by_block_id.get(block_id)
        if alias and alias != block_id:
            aliased = aliased.replace(f"@{block_id}(", f"@{alias}(")
    return aliased


def _collect_speaker_subject_anchor_issues(
    context_facts: Sequence[QuotedContextFactResponse],
    resolved_facts: Sequence[
        tuple[QuotedContextFactResponse, tuple[SourceAnchor, ...], SourceAnchor]
    ],
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    aliases_by_block_id: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    resolved_subject_by_fact_id = {
        id(raw_fact): subject_anchor
        for raw_fact, _, subject_anchor in resolved_facts
    }
    details: list[str] = []
    block_ids: list[str] = []
    for fact in context_facts:
        if fact.kind != "speaker":
            continue
        resolved_subject = resolved_subject_by_fact_id.get(id(fact))
        if resolved_subject is not None:
            block_id = resolved_subject.block_id
            text = texts.get(block_id, "")
            subject_span: tuple[int, int] | None = (
                resolved_subject.start,
                resolved_subject.end,
            )
            quote_preview = resolved_subject.quote
        else:
            block_id = aliases.get(fact.subject_anchor.block_id)
            if block_id is None or block_id not in texts:
                continue
            text = texts[block_id]
            quote_preview = fact.subject_anchor.quote
            s_start = fact.subject_anchor.start
            s_end = fact.subject_anchor.end
            if (
                type(s_start) is int
                and type(s_end) is int
                and 0 <= s_start < s_end <= len(text)
            ):
                subject_span = (s_start, s_end)
            else:
                found = _quote_spans(text, fact.subject_anchor.quote)
                subject_span = (found[0][0], found[0][1]) if found else None
        if subject_span is None:
            continue
        dialogue_spans = tuple(
            (match.start(), match.end()) for match in _DIALOGUE_QUOTES.finditer(text)
        )
        if not dialogue_spans:
            continue
        sub_start, sub_end = subject_span
        intersects = any(
            sub_start < d_end and d_start < sub_end
            for d_start, d_end in dialogue_spans
        )
        if not intersects:
            label = aliases_by_block_id.get(block_id, fact.subject_anchor.block_id)
            details.append(f"{label}: «{_preview_quote(quote_preview)}»")
            block_ids.append(block_id)
    return list(dict.fromkeys(details)), list(dict.fromkeys(block_ids))


_GENDER_ROLE_NOUN = (
    r"(?:brother|sister|brothers|sisters|mother|father|mom|mum|dad|parent|parents|"
    r"husband|wife|son|daughter|sons|daughters|aunt|uncle|grandma|grandpa|"
    r"boy|girl|boys|girls|man|woman|men|women|kid|kids|child|children|"
    r"friend|friends|sibling|siblings)"
)
_GENDER_PLURAL_SUBJECT_RE = re.compile(
    rf"\b{_GENDER_ROLE_NOUN}\s+and\s+"
    rf"(?:(?:the|a|an|his|her|their|our|my|your)\s+)?{_GENDER_ROLE_NOUN}\b",
    re.IGNORECASE,
)
_GENDER_CONJOINED_NAMES_RE = re.compile(
    r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$"
)


def _collect_gender_plural_subject_issues(
    context_facts: Sequence[QuotedContextFactResponse],
    resolved_facts: Sequence[
        tuple[QuotedContextFactResponse, tuple[SourceAnchor, ...], SourceAnchor]
    ],
    aliases: Mapping[str, str],
    aliases_by_block_id: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    resolved_subject_by_fact_id = {
        id(raw_fact): subject_anchor
        for raw_fact, _, subject_anchor in resolved_facts
    }
    details: list[str] = []
    block_ids: list[str] = []
    for fact in context_facts:
        if fact.kind != "gender":
            continue
        if (fact.value or "").strip().casefold() not in {"male", "female"}:
            continue
        resolved_subject = resolved_subject_by_fact_id.get(id(fact))
        quote = (
            resolved_subject.quote
            if resolved_subject is not None
            else fact.subject_anchor.quote
        )
        if (
            _GENDER_PLURAL_SUBJECT_RE.search(quote) is None
            and _GENDER_CONJOINED_NAMES_RE.match(quote) is None
        ):
            continue
        block_id = (
            resolved_subject.block_id
            if resolved_subject is not None
            else aliases.get(fact.subject_anchor.block_id)
        )
        label = (
            aliases_by_block_id.get(block_id, fact.subject_anchor.block_id)
            if block_id is not None
            else fact.subject_anchor.block_id
        )
        details.append(f"{label}: «{_preview_quote(quote)}»")
        if block_id is not None:
            block_ids.append(block_id)
    return list(dict.fromkeys(details)), list(dict.fromkeys(block_ids))


def validate_narrative_response(
    payload: NarrativeAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    visible_texts_by_alias: Mapping[str, str],
    *,
    ordered_block_ids: Sequence[str],
    mandatory_facts: Mapping[str, Mapping[str, str]] | None = None,
    is_notes_only: bool = False,
    blocks: Sequence[DocumentBlock] | None = None,
    source_offsets: bool = False,
    source_language: str | None = None,
    target_language: str | None = None,
    language_detector: LanguageDetector | None = None,
    known_names: Iterable[str] = (),
    name_translation_policy: NameTranslationPolicy | None = None,
    enforce_strict_gender_subject: bool = True,
) -> tuple[tuple[Scene, ...], tuple[ContextFact, ...]]:
    """Validate complete scene coverage and source-bound narrative facts."""
    issues: list[tuple[str, str, tuple[str, ...]]] = []
    if is_notes_only and payload.scenes:
        issues.append(
            (
                "notes_only_scenes",
                "служебный раздел не должен содержать сцены",
                tuple(ordered_block_ids),
            )
        )
    scenes, scene_issues = _inspect_scenes(
        payload.scenes,
        aliases,
        texts,
        visible_texts_by_alias,
        ordered_block_ids=ordered_block_ids,
        blocks=blocks,
        source_offsets=source_offsets,
        check_coverage=not is_notes_only,
    )
    issues.extend(scene_issues)

    block_order = {block: index for index, block in enumerate(ordered_block_ids)}
    scoped_aliases = dict(
        sorted(
            ((alias, block) for alias, block in aliases.items() if block in block_order),
            key=lambda item: block_order[item[1]],
        )
    )
    scoped_aliases_by_block_id = {v: k for k, v in scoped_aliases.items()}

    issues.extend(
        _collect_narrative_language_issues(
            payload,
            scenes,
            texts,
            ordered_block_ids,
            scoped_aliases_by_block_id,
            source_language=source_language,
            target_language=target_language,
            language_detector=language_detector,
            is_notes_only=is_notes_only,
        )
    )

    assessment_conflicts = _collect_name_assessment_conflicts(
        payload.glossary_candidates,
        name_translation_policy=name_translation_policy,
    )
    if assessment_conflicts:
        issues.append(
            (
                "name_assessment_conflict",
                f"несогласованная оценка перевода имени ({'; '.join(assessment_conflicts)})",
                (),
            )
        )

    resolved_facts, cf_missing_details, cf_missing_blocks = _resolve_quoted_fact_entries(
        payload.context_facts,
        scoped_aliases,
        texts,
        visible_texts_by_alias,
        scoped_aliases_by_block_id,
        blocks=blocks,
        source_offsets=source_offsets,
    )
    if cf_missing_details:
        issues.append(
            (
                "unresolved_quote",
                f"цитата не разрешается однозначно в тексте блока ({'; '.join(cf_missing_details)})",
                tuple(cf_missing_blocks),
            )
        )

    speaker_anchor_details, speaker_anchor_blocks = _collect_speaker_subject_anchor_issues(
        payload.context_facts,
        resolved_facts,
        scoped_aliases,
        texts,
        scoped_aliases_by_block_id,
    )
    if speaker_anchor_details:
        issues.append(
            (
                "speaker_subject_anchor",
                "для факта speaker опора subject_anchor должна указывать на закавыченную реплику, "
                f"а не на авторскую ремарку ({'; '.join(speaker_anchor_details)})",
                tuple(speaker_anchor_blocks),
            )
        )

    if enforce_strict_gender_subject:
        gender_plural_details, gender_plural_blocks = _collect_gender_plural_subject_issues(
            payload.context_facts,
            resolved_facts,
            scoped_aliases,
            scoped_aliases_by_block_id,
        )
        if gender_plural_details:
            issues.append(
                (
                    "gender_plural_subject",
                    "для kind=gender (male/female) subject_anchor должен указывать на одного персонажа, "
                    f"а не на группу через and ({'; '.join(gender_plural_details)})",
                    tuple(gender_plural_blocks),
                )
            )

    mapped_facts = [
        ContextFact(
            id=fact.id,
            kind=fact.kind,
            anchors=anchors,
            subject_anchor=subject_anchor,
            value=fact.value,
            certainty=fact.certainty,
        )
        for fact, anchors, subject_anchor in resolved_facts
    ]

    spoiler_details, spoiler_blocks = _collect_spoiler_name_leaks(
        scenes,
        mapped_facts,
        texts,
        ordered_block_ids,
        block_order,
        known_names,
        payload.glossary_candidates,
        scoped_aliases_by_block_id,
    )
    if spoiler_details:
        issues.append(
            (
                "spoiler_name_leak",
                f"обнаружен спойлер имени до первого упоминания в главе ({'; '.join(spoiler_details)})",
                tuple(spoiler_blocks),
            )
        )

    facts: list[ContextFact] = []
    for mapped in mapped_facts:
        mapped = _explicit_speaker_if_remark_names(
            mapped,
            texts,
            blocks,
            scenes=scenes,
            glossary_candidates=payload.glossary_candidates,
            context_facts=mapped_facts,
            mandatory_facts=mandatory_facts,
            ordered_block_ids=ordered_block_ids,
        )
        mapped.validate(texts)
        facts.append(replace(mapped, id=mapped.canonical_id))

    plan: ContentPlan | None = None
    try:
        plan = ContentPlan(
            version=3,
            profile="content-plan-v3",
            context_facts=tuple(facts),
            scenes=scenes,
        )
    except DomainError as error:
        if str(error).startswith("конфликт атрибуций одного субъекта"):
            conflict_blocks = _conflicting_subject_block_ids(facts)
            aliased_msg = _alias_domain_conflict_message(
                str(error), conflict_blocks, scoped_aliases_by_block_id
            )
            issues.append(("context_fact_conflict", aliased_msg, conflict_blocks))
        elif issues:
            issues.append(("content_plan", str(error), ()))
        else:
            raise

    glossary_details, glossary_blocks = _collect_glossary_fact_conflicts(
        facts if plan is None else plan.context_facts,
        mandatory_facts or {},
        scoped_aliases_by_block_id,
    )
    if glossary_details:
        issues.append(
            (
                "glossary_fact_conflict",
                "контекстная атрибуция противоречит обязательному решению глоссария "
                f"({'; '.join(glossary_details)})",
                tuple(glossary_blocks),
            )
        )

    if issues:
        _raise_collected_narrative_issues(issues)
    assert plan is not None
    return scenes, plan.context_facts


def _quoted_anchor(anchor: SourceAnchor) -> QuotedSourceAnchorResponse:
    return QuotedSourceAnchorResponse(anchor.block_id, anchor.quote, anchor.start, anchor.end)


def _quoted_scene(scene: Scene) -> QuotedSceneResponse:
    return QuotedSceneResponse(scene.start_block_id, scene.end_block_id, scene.summary,
        scene.location, scene.participants,
        tuple(QuotedSceneVisualFactResponse(_quoted_anchor(f.anchor), f.subject, f.detail)
              for f in scene.visual_facts),
        scene.visual_assumptions)


def _quoted_fact(fact: ContextFact) -> QuotedContextFactResponse:
    return QuotedContextFactResponse(fact.id, fact.kind,
        tuple(_quoted_anchor(a) for a in fact.anchors),
        _quoted_anchor(fact.subject_anchor),
        fact.value, fact.certainty)


def _normalize_location(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _merge_scene_locations(
    prev_location: str | None, first_location: str | None
) -> str | None:
    """Combines scene locations across a merged boundary without duplicating the tail segment."""
    prev_loc = (prev_location or "").strip()
    first_loc = (first_location or "").strip()
    if not prev_loc and not first_loc:
        return None
    if not prev_loc:
        return first_loc
    if not first_loc:
        return prev_loc
    last_segment = prev_loc.split("→")[-1]
    if _normalize_location(last_segment) == _normalize_location(first_loc):
        return prev_loc
    return f"{prev_loc} → {first_loc}"


def _is_scene_separator_block(block: DocumentBlock) -> bool:
    return (
        block.support_status is SupportStatus.PRESERVE_OPAQUE
        or block.type in (BlockType.PAGE_BREAK, BlockType.IMAGE)
    )


def _has_scene_separator_between(
    prev_end_block_id: str,
    next_start_block_id: str,
    section_blocks: Sequence[DocumentBlock],
    block_index_by_id: Mapping[str, int],
) -> bool:
    if not section_blocks:
        return False
    prev_idx = block_index_by_id.get(prev_end_block_id)
    next_idx = block_index_by_id.get(next_start_block_id)
    if prev_idx is None or next_idx is None or prev_idx >= next_idx:
        return False
    return any(
        _is_scene_separator_block(block)
        for block in section_blocks[prev_idx + 1 : next_idx]
    )


def merge_narrative_fragments(
    fragments: Sequence[tuple[NarrativeAnalysisResponse, tuple[Scene, ...], tuple[ContextFact, ...]]],
    *,
    section_blocks: Sequence[DocumentBlock] = (),
) -> tuple[NarrativeAnalysisResponse, tuple[Scene, ...], tuple[ContextFact, ...]]:
    """Merge validated fragments; returned quoted references use stable block IDs.

    Callers resolving this payload must explicitly expose identity aliases for the
    current chapter, together with its visible text. Local b1 aliases are never reused.
    """
    if not fragments:
        raise AnalysisMergeError('нет фрагментов повествовательного анализа')
    scenes: list[Scene] = []
    facts: dict[str, ContextFact] = {}
    candidates = []
    instruction_fragments = []
    previous_had_scene = False
    block_index_by_id = {block.id: idx for idx, block in enumerate(section_blocks)}
    for index, (raw, resolved, context_facts) in enumerate(fragments):
        instruction_fragments.append(FragmentAnalysis(index, raw.translation_instructions, raw.chapter_summary))
        for candidate in raw.glossary_candidates:
            if candidate not in candidates:
                candidates.append(candidate)
        for fact in context_facts:
            facts.setdefault(fact.canonical_id, replace(fact, id=fact.canonical_id))
        current = list(resolved)
        has_separator = (
            previous_had_scene and bool(current)
            and _has_scene_separator_between(
                scenes[-1].end_block_id,
                current[0].start_block_id,
                section_blocks,
                block_index_by_id,
            )
        )
        implicit_continuation = (
            previous_had_scene and current and not has_separator
            and scenes[-1].location is not None and bool(scenes[-1].location.strip())
            and scenes[-1].location == current[0].location
            and bool(scenes[-1].participants)
            and set(scenes[-1].participants) == set(current[0].participants)
        )
        if (previous_had_scene and current and not has_separator
                and ((raw.scenes and raw.scenes[0].continues_previous_scene) or implicit_continuation)):
            prev, first = scenes.pop(), current.pop(0)
            merged = replace(prev, end_block_id=first.end_block_id,
                summary=' '.join(dict.fromkeys((prev.summary, first.summary))),
                location=_merge_scene_locations(prev.location, first.location),
                participants=tuple(dict.fromkeys((*prev.participants, *first.participants))),
                visual_facts=tuple(dict.fromkeys((*prev.visual_facts, *first.visual_facts))),
                visual_assumptions=tuple(dict.fromkeys((*prev.visual_assumptions, *first.visual_assumptions))))
            scenes.append(merged)
        scenes.extend(current)
        previous_had_scene = bool(resolved)
    normalized = []
    for index, item in enumerate(scenes):
        item = replace(item, order=index)
        normalized.append(replace(item, id=item.canonical_id))
    result = NarrativeAnalysisResponse(_merged_instructions(instruction_fragments), fragments[-1][0].chapter_summary,
        tuple(candidates), tuple(_quoted_fact(f) for f in facts.values()), tuple(_quoted_scene(s) for s in normalized))
    return result, tuple(normalized), tuple(facts.values())


def combine_staged_analysis_responses(
    narrative: NarrativeAnalysisResponse,
    pedagogical: PedagogicalAnalysisResponse,
    *,
    resolved_scenes: Sequence[Scene] = (),
    resolved_blocks: Sequence[DocumentBlock] = (),
    mandatory_facts: Mapping[str, Mapping[str, str]] | Sequence[object] | None = (),
) -> ChapterAnalysisResponse:
    """Assemble both pass contracts without changing pedagogical reference aliases."""
    facts = narrative.context_facts
    if resolved_blocks:
        texts_by_id = {block.id: block.text for block in resolved_blocks}
        ordered_block_ids = tuple(block.id for block in resolved_blocks)
        facts = tuple(
            _explicit_speaker_if_remark_names(
                fact,
                texts_by_id,
                resolved_blocks,
                scenes=resolved_scenes,
                glossary_candidates=narrative.glossary_candidates,
                context_facts=facts,
                mandatory_facts=mandatory_facts,
                ordered_block_ids=ordered_block_ids,
            )
            for fact in facts
        )
    return ChapterAnalysisResponse(narrative.translation_instructions,
        pedagogical.reader_chapter_summary or narrative.chapter_summary,
        narrative.glossary_candidates, pedagogical.treatment_plan, pedagogical.before_you_read,
        QuotedContentPlanResponse(difficulties=pedagogical.difficulties,
            before_you_read_evidence=pedagogical.before_you_read_evidence,
            context_facts=facts,
            scenes=tuple(_quoted_scene(s) for s in resolved_scenes) if resolved_scenes else narrative.scenes))


def content_plan_from_response(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None = None,
    *,
    visible_texts_by_alias: Mapping[str, str] | None = None,
    blocks: Sequence[DocumentBlock] | None = None,
    source_offsets: bool = False,
    mandatory_facts: Mapping[str, Mapping[str, str]] | Sequence[object] | None = (),
) -> ContentPlan | None:
    """Replaces local aliases and computes stable application references."""
    if payload.content_plan is None:
        return None
    is_quoted = isinstance(payload.content_plan, QuotedContentPlanResponse)
    is_v3 = payload.content_plan.version == 3
    if is_v3 and texts is None:
        raise AnalysisMergeError('план v3 требует доступный исходный фрагмент')

    resolved_difficulties = []
    resolved_facts = []
    dropped_difficulty_ids: set[str] = set()
    if is_quoted:
        if visible_texts_by_alias is None:
            return None
        for difficulty in payload.content_plan.difficulties:
            owner_block_id = aliases.get(difficulty.owner_block_id)
            if owner_block_id is None:
                return None
            anchors = _bind_quoted_anchors(
                difficulty.anchors, aliases, texts, visible_texts_by_alias,
                blocks=blocks, source_offsets=source_offsets,
            )
            if anchors is _MISSING:
                return None
            if not anchors or owner_block_id not in {item.block_id for item in anchors}:
                dropped_difficulty_ids.add(difficulty.id)
                continue
            resolved_difficulties.append((difficulty, anchors, owner_block_id))
        aliases_by_block_id = {bid: alias for alias, bid in aliases.items() if bid}
        resolved_facts, missing_fact_details, _ = _resolve_quoted_fact_entries(
            payload.content_plan.context_facts,
            aliases,
            texts,
            visible_texts_by_alias,
            aliases_by_block_id,
            blocks=blocks,
            source_offsets=source_offsets,
        )
        if missing_fact_details:
            return None

    difficulties: list[Difficulty] = []
    ids: dict[str, str] = {}
    if is_quoted:
        for difficulty, resolved_anchors, resolved_owner in resolved_difficulties:
            mapped = Difficulty(
                id=difficulty.id,
                anchors=resolved_anchors,
                owner_block_id=resolved_owner,
                kind=difficulty.kind,
                priority=difficulty.priority,
                help_kind=difficulty.help_kind,
                required_points=difficulty.required_points,
            )
            mapped = replace(mapped, id=mapped.canonical_id)
            ids[difficulty.id] = mapped.id
            difficulties.append(mapped)
    else:
        for difficulty in payload.content_plan.difficulties:
            assert isinstance(difficulty, Difficulty)
            try:
                mapped_anchors: list[SourceAnchor] = []
                for a in difficulty.anchors:
                    real_block_id = aliases[a.block_id]
                    if is_v3:
                        mapped_anchor = replace(a, block_id=real_block_id)
                        if texts is not None:
                            mapped_anchor.validate(texts)
                        mapped_anchors.append(mapped_anchor)
                    elif texts is not None and real_block_id in texts:
                        span = find_quote_span(texts[real_block_id], a.quote, estimated_start=a.start)
                        if span is not None:
                            start, end, actual_quote = span
                            mapped_anchors.append(
                                SourceAnchor(block_id=real_block_id, start=start, end=end, quote=actual_quote)
                            )
                        else:
                            mapped_anchors.append(replace(a, block_id=real_block_id))
                    else:
                        mapped_anchors.append(replace(a, block_id=real_block_id))
                mapped = replace(
                    difficulty,
                    anchors=tuple(mapped_anchors),
                    owner_block_id=aliases[difficulty.owner_block_id],
                )
            except KeyError as error:
                raise AnalysisMergeError("привязка к невидимому блоку") from error
            mapped = replace(mapped, id=mapped.canonical_id)
            ids[difficulty.id] = mapped.id
            difficulties.append(mapped)
    evidence = []
    selected = before_you_read_entries(payload.before_you_read, strict=is_v3)
    if not payload.content_plan.before_you_read_evidence:
        seen_points = set()
        for difficulty in difficulties:
            anchor_substances = {_substance(a.quote) for a in difficulty.anchors}
            for anchor in difficulty.anchors:
                quote = without_footnote_markers(anchor.quote).strip() if is_v3 else anchor.quote.strip()
                for entry in selected:
                    expr = without_footnote_markers(entry.expression).strip() if is_v3 else entry.expression.strip()
                    if is_v3:
                        matched = _quote_matches_anchor(
                            expr, quote, difficulty.kind
                        ) and (
                            texts is None
                            or find_quote_span(
                                without_footnote_markers(texts.get(anchor.block_id, "")),
                                expr,
                            )
                            is not None
                        )
                    else:
                        matched = (
                            _same_word(expr, quote)
                            or find_quote_span(quote, expr) is not None
                            or find_quote_span(expr, quote) is not None
                        )
                    forbidden = anchor_substances | {_substance(entry.expression)}
                    if matched and _substance(entry.explanation) and _substance(entry.explanation) not in forbidden:
                        key = artifact_key(
                            "before_you_read.explanation",
                            difficulty.owner_block_id,
                            {"expression": entry.expression, "explanation": entry.explanation},
                        )
                        for point in difficulty.required_points:
                            pair = (difficulty.id, point.id)
                            if pair not in seen_points:
                                seen_points.add(pair)
                                evidence.append(
                                    HelpEvidence(
                                        difficulty.id,
                                        point.id,
                                        "before_you_read.explanation",
                                        difficulty.owner_block_id,
                                        key,
                                        entry.explanation,
                                    )
                                )
    else:
        for item in payload.content_plan.before_you_read_evidence:
            if item.difficulty_id in dropped_difficulty_ids:
                continue
            if item.artifact_kind != "before_you_read.explanation" or item.artifact_index >= len(payload.before_you_read):
                raise AnalysisMergeError("свидетельство анализа не ссылается на существующий BYR")
            entry = payload.before_you_read[item.artifact_index]
            if not any(e.expression == entry.expression and e.explanation == entry.explanation for e in selected):
                if (
                    not is_v3
                    and any(candidate.expression == entry.expression for candidate in selected)
                ):
                    raise AnalysisMergeError("BYR удалён нормализацией или превышает бюджет")
                representative = _before_you_read_stem_duplicate(entry, selected)
                if representative is None:
                    raise AnalysisMergeError("BYR удалён нормализацией или превышает бюджет")
                body_excerpt = _align_byr_evidence_excerpt(representative, item.body_excerpt)
                if body_excerpt is None:
                    continue
                entry = representative
            else:
                body_excerpt = _align_byr_evidence_excerpt(entry, item.body_excerpt)
            if body_excerpt is None:
                raise AnalysisMergeError("отрывок не принадлежит телу объяснения BYR")
            try:
                evidence.append(HelpEvidence(ids[item.difficulty_id], item.point_id, item.artifact_kind, aliases[item.source_block_id], artifact_key(item.artifact_kind, aliases[item.source_block_id], {"expression": entry.expression, "explanation": entry.explanation}), body_excerpt))
            except KeyError as error:
                raise AnalysisMergeError("неизвестная трудность или владелец свидетельства") from error
    ordered_block_ids = tuple(dict.fromkeys(aliases.values()))
    scenes = (
        resolve_scenes(
            payload.content_plan.scenes,
            aliases,
            texts,
            visible_texts_by_alias,
            ordered_block_ids=ordered_block_ids,
            blocks=blocks,
            source_offsets=source_offsets,
        )
        if is_quoted and texts is not None and visible_texts_by_alias is not None
        else ()
    )
    mapped_facts: list[ContextFact] = []
    if is_quoted:
        for fact, resolved_anchors, resolved_subject in resolved_facts:
            mapped_facts.append(
                ContextFact(
                    id=fact.id,
                    kind=fact.kind,
                    anchors=resolved_anchors,
                    subject_anchor=resolved_subject,
                    value=fact.value,
                    certainty=fact.certainty,
                )
            )
    else:
        for fact in payload.content_plan.context_facts:
            assert isinstance(fact, ContextFact)
            try:
                mapped_facts.append(
                    replace(
                        fact,
                        anchors=tuple(
                            replace(anchor, block_id=aliases[anchor.block_id])
                            for anchor in fact.anchors
                        ),
                        subject_anchor=replace(
                            fact.subject_anchor,
                            block_id=aliases[fact.subject_anchor.block_id],
                        ),
                    )
                )
            except KeyError:
                raise AnalysisMergeError('контекстная атрибуция ссылается на невидимый блок') from None
    facts = []
    for mapped in mapped_facts:
        mapped = _explicit_speaker_if_remark_names(
            mapped,
            texts,
            blocks,
            scenes=scenes,
            glossary_candidates=payload.glossary_candidates,
            context_facts=mapped_facts,
            mandatory_facts=mandatory_facts,
            ordered_block_ids=ordered_block_ids,
        )
        mapped.validate(texts)
        facts.append(replace(mapped, id=mapped.canonical_id))
    return ContentPlan(version=payload.content_plan.version, profile=payload.content_plan.profile,
        difficulties=tuple(difficulties), before_you_read_evidence=tuple(evidence), context_facts=tuple(facts),
        scenes=scenes)


_MISSING = object()
_AMBIGUOUS = object()


def _analysis_origins(block_alias, aliases, texts, visible_texts_by_alias, blocks):
    """Source offset of each visible character. Identity when the model saw `block.text`."""
    visible = visible_texts_by_alias[block_alias]
    block_id = aliases.get(block_alias)
    source = texts[block_id] if texts is not None and block_id in texts else None
    if blocks and block_id is not None and source is not None:
        block = next((item for item in blocks if item.id == block_id), None)
        if block is not None and block.text == source:
            from linguary.application.context_builder import _analysis_visible_origins

            computed, origins = _analysis_visible_origins(block)
            if computed == visible and len(origins) == len(visible):
                return origins
    if source is not None and source == visible:
        return tuple(range(len(visible)))
    return None


def _source_span(origins, start: int, end: int, source: str):
    """Map a visible `[start, end)` through character origins. The span must stay contiguous."""
    if origins is None or not (0 <= start < end <= len(origins)):
        return None
    piece = origins[start:end]
    if any(piece[index] + 1 != piece[index + 1] for index in range(len(piece) - 1)):
        return None
    source_start = piece[0]
    source_end = piece[-1] + 1
    if source_end > len(source):
        return None
    return source_start, source_end, source[source_start:source_end]


def _visible_offset_span(visible: str, quote: str, spans, start, end):
    if (
        type(start) is int and type(end) is int
        and 0 <= start < end <= len(visible)
        and visible[start:end] == quote
        and any(span[0] == start and span[1] == end for span in spans)
    ):
        return next(span for span in spans if span[0] == start and span[1] == end)
    return None


def _source_offset_span(source: str, origins, quote: str, spans, start, end):
    """Domain anchors store source coordinates. Accept that image of one visible span."""
    if not (
        origins is not None
        and type(start) is int and type(end) is int
        and 0 <= start < end <= len(source)
        and source[start:end] == quote
    ):
        return None
    matched = None
    for span in spans:
        mapped = _source_span(origins, span[0], span[1], source)
        if mapped is not None and mapped[0] == start and mapped[1] == end and mapped[2] == quote:
            if matched is not None:
                return None
            matched = span
    return matched


def _resolve_quoted_anchor(
    block_alias, quote, aliases, texts, visible_texts_by_alias, *,
    start=None, end=None, strict=False, blocks=None, source_offsets=False,
):
    """Bind on the visible text, then store source coordinates.

    One visible span binds by text. A repeated quote binds at the model's visible
    offset or at the source offset stored on a domain anchor. `source_offsets` accepts
    only that stored image, so a colliding visible index is not chosen again. A quote
    the model did not see rejects the plan. A missing or inexact offset does not bind.
    """
    block_id = aliases.get(block_alias)
    if block_id is None or texts is None or block_id not in texts or block_alias not in visible_texts_by_alias:
        return _MISSING
    visible = visible_texts_by_alias[block_alias]
    source = texts[block_id]
    spans = _exact_quote_spans(visible, quote) if strict else _quote_spans(visible, quote)
    if not spans:
        return _MISSING
    origins = _analysis_origins(block_alias, aliases, texts, visible_texts_by_alias, blocks)
    if len(spans) == 1:
        chosen = spans[0]
    elif source_offsets:
        chosen = _source_offset_span(source, origins, quote, spans, start, end)
    else:
        chosen = _visible_offset_span(visible, quote, spans, start, end)
        if chosen is None:
            chosen = _source_offset_span(source, origins, quote, spans, start, end)
    if chosen is None:
        return _AMBIGUOUS
    mapped = _source_span(origins, chosen[0], chosen[1], source)
    if mapped is None or mapped[2] != chosen[2]:
        return _MISSING if origins is None else _AMBIGUOUS
    source_start, source_end, actual_quote = mapped
    return SourceAnchor(block_id=block_id, start=source_start, end=source_end, quote=actual_quote)


def _bind_quoted_anchors(anchors, aliases, texts, visible_texts_by_alias, *, blocks=None, source_offsets=False):
    """Keep anchors that bind. A quote absent from the visible text rejects the plan."""
    kept = []
    for anchor in anchors:
        resolved = _resolve_quoted_anchor(
            anchor.block_id, anchor.quote, aliases, texts, visible_texts_by_alias,
            start=anchor.start, end=anchor.end, blocks=blocks, source_offsets=source_offsets,
        )
        if resolved is _MISSING:
            return _MISSING
        if resolved is not _AMBIGUOUS:
            kept.append(resolved)
    return tuple(kept)


def _unique_exact_span(text, quote):
    if not text or not quote:
        return None
    first = None
    position = 0
    while True:
        position = text.find(quote, position)
        if position < 0:
            break
        if first is not None:
            return None
        first = position
        position += 1
    if first is None:
        return None
    return first, first + len(quote)


def _merged_content_plan(fragments: Sequence[FragmentAnalysis]) -> ContentPlan | None:
    plans = [f.content_plan for f in fragments if f.content_plan is not None]
    if not plans:
        return None
    if len(plans) != len(fragments) or len({plan.version for plan in plans}) != 1:
        raise AnalysisMergeError("нельзя смешивать версии анализа")
    groups = {}
    for plan in plans:
        for difficulty in plan.difficulties:
            key = (difficulty.anchors, difficulty.kind)
            groups.setdefault(key, []).append(difficulty)
    difficulties = []
    ids = {}
    for group in groups.values():
        first = group[0]
        points = {}
        for difficulty in group:
            if (difficulty.owner_block_id, difficulty.priority, difficulty.help_kind) != (first.owner_block_id, first.priority, first.help_kind):
                raise AnalysisMergeError("конфликт повторяющейся трудности")
            for point in difficulty.required_points:
                if point.id in points and points[point.id] != point:
                    raise AnalysisMergeError("конфликт описаний смыслового аспекта")
                points[point.id] = point
        merged = replace(first, required_points=tuple(points.values()))
        merged = replace(merged, id=merged.canonical_id)
        difficulties.append(merged)
        for difficulty in group:
            previous = ids.setdefault(difficulty.id, merged.id)
            if previous != merged.id:
                raise AnalysisMergeError("идентификатор обозначает разные трудности")
    evidence = []
    for plan in plans:
        for item in plan.before_you_read_evidence:
            mapped = replace(item, difficulty_id=ids[item.difficulty_id])
            if mapped not in evidence:
                evidence.append(mapped)
    selected = _merged_before_you_read(fragments)
    retained_evidence: list[HelpEvidence] = []
    for item in evidence:
        if any(
            artifact_key(
                item.artifact_kind,
                item.block_id,
                {"expression": entry.expression, "explanation": entry.explanation},
            )
            == item.artifact_key
            for entry in selected
        ):
            retained_evidence.append(item)
            continue
        source_entry = next(
            (
                entry
                for fragment in fragments
                for entry in fragment.before_you_read
                if artifact_key(
                    item.artifact_kind,
                    item.block_id,
                    {"expression": entry.expression, "explanation": entry.explanation},
                )
                == item.artifact_key
            ),
            None,
        )
        if source_entry is None:
            raise AnalysisMergeError("слияние удалило BYR со свидетельством")
        representative = _before_you_read_stem_duplicate(source_entry, selected)
        if representative is None:
            raise AnalysisMergeError("слияние удалило BYR со свидетельством")
        body_excerpt = _align_byr_evidence_excerpt(representative, item.body_excerpt)
        if body_excerpt is None:
            body_excerpt = _align_byr_evidence_excerpt(representative, representative.explanation)
        if body_excerpt is None:
            continue
        remapped = replace(
            item,
            artifact_key=artifact_key(
                item.artifact_kind,
                item.block_id,
                {
                    "expression": representative.expression,
                    "explanation": representative.explanation,
                },
            ),
            body_excerpt=body_excerpt,
        )
        if remapped not in retained_evidence:
            retained_evidence.append(remapped)
    facts = {}
    for plan in plans:
        for fact in plan.context_facts:
            previous = facts.setdefault(fact.canonical_id, fact)
            if previous != fact:
                raise AnalysisMergeError('конфликт канонической атрибуции')
    return ContentPlan(version=plans[0].version, profile=plans[0].profile,
        difficulties=tuple(difficulties), before_you_read_evidence=tuple(retained_evidence),
        context_facts=tuple(facts.values()))


_SEGMENT_ALIAS_RE = re.compile(r"^(b\d+)\.\d+$", re.IGNORECASE)


def _segment_parent_alias(
    block_alias: str,
    aliases: Mapping[str, str],
    aliases_by_block_id: Mapping[str, str],
) -> tuple[str, str] | None:
    """Return `(parent_label, parent_block_id)` when `block_alias` is `bN.M` for a known `bN`."""
    match = _SEGMENT_ALIAS_RE.match(block_alias)
    if match is None:
        return None
    raw_parent = match.group(1)
    for candidate in (raw_parent, raw_parent.lower()):
        parent_block_id = aliases.get(candidate)
        if parent_block_id is not None:
            return aliases_by_block_id.get(parent_block_id, candidate), parent_block_id
    return None


def _diagnose_quoted_anchor(
    block_alias: str,
    quote: str,
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None,
    visible_texts_by_alias: Mapping[str, str] | None,
    aliases_by_block_id: Mapping[str, str],
) -> tuple[str, str, str, tuple[str, ...]] | None:
    block_id = aliases.get(block_alias)
    label = aliases_by_block_id.get(block_id, block_alias) if block_id else block_alias
    preview = _preview_quote(quote)
    if block_id is None:
        parent = _segment_parent_alias(block_alias, aliases, aliases_by_block_id)
        if parent is not None:
            parent_label, parent_block_id = parent
            return (
                block_alias,
                preview,
                f"указана метка сегмента {block_alias} — используй идентификатор целого блока {parent_label}",
                (parent_block_id,),
            )
        if visible_texts_by_alias is None:
            return (label, preview, "нет видимого представления блока", ())
        return (block_alias, preview, "неизвестный блок", ())
    if visible_texts_by_alias is None:
        return (label, preview, "нет видимого представления блока", (block_id,))
    if texts is None or block_id not in texts:
        return (label, preview, "нет текста исходного блока", ())
    if block_alias not in visible_texts_by_alias:
        return (label, preview, "блок отсутствует в видимом фрагменте", (block_id,))
    visible_span = find_quote_span(visible_texts_by_alias[block_alias], quote)
    source_span = find_quote_span(texts[block_id], quote)
    if visible_span is None and source_span is None:
        return (label, preview, "цитата не найдена в тексте блока", (block_id,))
    if visible_span is None:
        return (label, preview, "цитата не найдена в видимом тексте блока", (block_id,))
    if source_span is None:
        return (label, preview, "цитата не найдена в исходном тексте блока", (block_id,))
    return None


def _unresolved_quote_error(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None,
    visible_texts_by_alias: Mapping[str, str] | None,
    aliases_by_block_id: Mapping[str, str],
) -> AnalysisMergeError:
    prefix = "цитата не разрешается однозначно в тексте блока"
    plan = payload.content_plan
    details: list[str] = []
    block_ids: list[str] = []
    if isinstance(plan, QuotedContentPlanResponse):
        anchors_to_check: list[QuotedSourceAnchorResponse] = []
        for difficulty in plan.difficulties:
            owner_missing = aliases.get(difficulty.owner_block_id) is None
            sample = difficulty.anchors[0].quote if difficulty.anchors else difficulty.id
            if owner_missing:
                parent = _segment_parent_alias(
                    difficulty.owner_block_id, aliases, aliases_by_block_id
                )
                if parent is not None:
                    parent_label, parent_block_id = parent
                    details.append(
                        f"{difficulty.owner_block_id}: «{_preview_quote(sample)}» — "
                        f"неизвестный блок-владелец трудности: используй идентификатор целого блока "
                        f"{parent_label}, а не сегмента {difficulty.owner_block_id}"
                    )
                    block_ids.append(parent_block_id)
                else:
                    details.append(
                        f"{difficulty.owner_block_id}: «{_preview_quote(sample)}» — неизвестный блок-владелец трудности"
                    )
            for idx, anchor in enumerate(difficulty.anchors):
                if (
                    owner_missing
                    and idx == 0
                    and anchor.block_id == difficulty.owner_block_id
                    and anchor.quote == sample
                ):
                    continue
                anchors_to_check.append(anchor)
        for fact in plan.context_facts:
            anchors_to_check.extend((fact.subject_anchor, *fact.anchors))
        for anchor in anchors_to_check:
            diag = _diagnose_quoted_anchor(
                anchor.block_id,
                anchor.quote,
                aliases,
                texts,
                visible_texts_by_alias,
                aliases_by_block_id,
            )
            if diag is not None:
                block_label, preview, reason, canonical = diag
                details.append(f"{block_label}: «{preview}» — {reason}")
                block_ids.extend(canonical)
    unique_details = tuple(dict.fromkeys(details))
    if unique_details:
        msg = f"{prefix} ({'; '.join(unique_details)})"
        return _analysis_error(msg, rule_code="unresolved_quote", block_ids=block_ids)
    return _analysis_error(prefix, rule_code="unresolved_quote")


def _overlapping_segment_treatments(
    difficulty: Difficulty,
    plan_item: TreatmentPlanItemResponse | None,
    block: DocumentBlock | None,
    fallback_treatment: BlockTreatment | None,
) -> tuple[BlockTreatment, ...]:
    if (
        plan_item is not None
        and plan_item.segments
        and block is not None
        and block.segment_spans
    ):
        by_index = {segment.index: segment.treatment for segment in plan_item.segments}
        owner_anchors = tuple(
            anchor
            for anchor in difficulty.anchors
            if anchor.block_id == difficulty.owner_block_id
        )
        if owner_anchors and len(by_index) == len(block.segment_spans):
            overlapped = tuple(
                by_index[index]
                for index, span in enumerate(block.segment_spans, start=1)
                if any(
                    span.start < anchor.end and anchor.start < span.end
                    for anchor in owner_anchors
                )
            )
            if overlapped:
                return overlapped
    return (fallback_treatment,) if fallback_treatment is not None else ()


def validate_content_plan(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    *,
    before_you_read_limit: int = 10,
    language_notes_per_chapter: int = 8,
    english_help: bool = False,
    byr_enabled: bool = True,
    note_limits_by_block: Mapping[str, int] | None = None,
    note_group_limits: Sequence[tuple[Sequence[str], int]] = (),
    visible_texts_by_alias: Mapping[str, str] | None = None,
    mandatory_facts: Mapping[str, Mapping[str, str]] | None = None,
    blocks: Sequence[DocumentBlock] | None = None,
    source_offsets: bool = False,
) -> ContentPlan | None:
    """Validates source anchors and channel eligibility prior to accepting the response."""
    if blocks is not None:
        plan_items_from_response(payload, block_ids_by_alias=aliases, blocks=blocks)
    aliases_by_block_id = {bid: alias for alias, bid in (aliases or {}).items() if bid}
    blocks_by_id = {block.id: block for block in blocks} if blocks is not None else {}
    plan = content_plan_from_response(
        payload,
        aliases,
        texts=texts,
        visible_texts_by_alias=visible_texts_by_alias,
        blocks=blocks,
        source_offsets=source_offsets,
        mandatory_facts=mandatory_facts,
    )
    if plan is None:
        if payload.content_plan is not None:
            raise _unresolved_quote_error(
                payload, aliases, texts, visible_texts_by_alias, aliases_by_block_id
            )
        if english_help:
            raise AnalysisMergeError("режиму английской помощи нужен содержательный план")
        return None
    if english_help:
        if plan.version != 3:
            raise AnalysisMergeError("режим английской помощи требует план версии 3")
        if payload.before_you_read or plan.before_you_read_evidence:
            raise AnalysisMergeError("режим английской помощи не использует материалы BYR")
    if len(payload.before_you_read) > before_you_read_limit:
        raise AnalysisMergeError("BYR превышает согласованный бюджет фрагмента")
    try:
        plan_items_by_block_id = {
            aliases[i.source_block_id]: i for i in payload.treatment_plan
        }
    except KeyError as error:
        raise AnalysisMergeError("привязка к невидимому блоку") from error
    treatments = {
        block_id: item.treatment for block_id, item in plan_items_by_block_id.items()
    }
    bad_english_blocks = [
        block_id
        for block_id, treatment in treatments.items()
        if treatment is not BlockTreatment.READ_AS_IS
    ]
    if english_help and bad_english_blocks:
        labels = ", ".join(aliases_by_block_id.get(b, b) for b in bad_english_blocks)
        msg = f"режим английской помощи требует чтение блоков в оригинале ({labels})"
        raise _analysis_error(
            msg, rule_code="english_help_treatment", block_ids=bad_english_blocks
        )
    note_owners = set()
    english_help_anchors = set()
    seen_difficulty_anchors: dict[tuple[str, frozenset[tuple[str, int, int, str]]], str] = {}
    for difficulty in plan.difficulties:
        for anchor in difficulty.anchors:
            anchor.validate(texts)
            occurrence = (anchor.block_id, anchor.start, anchor.end)
            if english_help and occurrence in english_help_anchors:
                label = aliases_by_block_id.get(anchor.block_id, anchor.block_id)
                msg = (
                    "несколько трудностей английской помощи ссылаются на одно место "
                    f"({label}, {difficulty.id})"
                )
                raise _analysis_error(
                    msg,
                    rule_code="duplicate_english_help_anchor",
                    block_ids=(anchor.block_id,),
                )
            english_help_anchors.add(occurrence)
        anchor_signature = frozenset(
            (anchor.block_id, anchor.start, anchor.end, anchor.quote.strip())
            for anchor in difficulty.anchors
        )
        if anchor_signature:
            owner_key = (difficulty.owner_block_id, anchor_signature)
            existing_id = seen_difficulty_anchors.get(owner_key)
            if existing_id is not None:
                label = aliases_by_block_id.get(
                    difficulty.owner_block_id, difficulty.owner_block_id
                )
                msg = (
                    "в одном блоке две трудности ссылаются на один и тот же якорь — "
                    "объедини их или уточни якорь каждой трудности "
                    f"({label}: {existing_id}, {difficulty.id})"
                )
                raise _analysis_error(
                    msg,
                    rule_code="duplicate_difficulty_anchor",
                    block_ids=(difficulty.owner_block_id,),
                )
            seen_difficulty_anchors[owner_key] = difficulty.id
        covered = {
            e.point_id
            for e in plan.before_you_read_evidence
            if e.difficulty_id == difficulty.id
        }
        owner_treatment = treatments.get(difficulty.owner_block_id)
        effective_treatments = _overlapping_segment_treatments(
            difficulty,
            plan_items_by_block_id.get(difficulty.owner_block_id),
            blocks_by_id.get(difficulty.owner_block_id),
            owner_treatment,
        )
        fragment_help = (
            english_help
            and plan.version == 3
            and owner_treatment is BlockTreatment.READ_AS_IS
        )
        if (
            plan.version == 3
            and difficulty.kind == "grammar"
            and difficulty.help_kind != "explanation"
        ):
            label = aliases_by_block_id.get(
                difficulty.owner_block_id, difficulty.owner_block_id
            )
            msg = (
                "грамматическая трудность требует объяснения "
                f"({label}, {difficulty.id})"
            )
            raise _analysis_error(
                msg,
                rule_code="grammar_requires_explanation",
                block_ids=(difficulty.owner_block_id,),
            )
        if (
            plan.version == 3
            and difficulty.kind in _EXPLANATION_ONLY_NON_GRAMMAR_KINDS
            and difficulty.help_kind != "explanation"
            and covered != {p.id for p in difficulty.required_points}
        ):
            label = aliases_by_block_id.get(
                difficulty.owner_block_id, difficulty.owner_block_id
            )
            msg = (
                f"трудность вида {difficulty.kind} требует объяснения "
                f"({label}, {difficulty.id})"
            )
            raise _analysis_error(
                msg,
                rule_code="non_lexical_kind_requires_explanation",
                block_ids=(difficulty.owner_block_id,),
            )
        all_glossary = bool(effective_treatments) and all(
            mode is BlockTreatment.GLOSSARY_ONLY for mode in effective_treatments
        )
        if (
            difficulty.help_kind == "lexical_hint"
            and not fragment_help
            and not all_glossary
            and covered != {p.id for p in difficulty.required_points}
        ):
            label = aliases_by_block_id.get(
                difficulty.owner_block_id, difficulty.owner_block_id
            )
            msg = (
                (
                    "лексическая помощь требует GLOSSARY_ONLY или подтверждённый BYR "
                    f"({label}, {difficulty.id})"
                )
                if byr_enabled
                else (
                    "лексическая помощь допустима только в GLOSSARY_ONLY "
                    f"({label}, {difficulty.id})"
                )
            )
            raise _analysis_error(
                msg,
                rule_code="lexical_help_channel",
                block_ids=(difficulty.owner_block_id,),
            )
        if (
            difficulty.help_kind == "explanation"
            and covered != {p.id for p in difficulty.required_points}
            and not fragment_help
            and any(mode is BlockTreatment.READ_AS_IS for mode in effective_treatments)
        ):
            note_owners.add(difficulty.owner_block_id)
        for item in plan.before_you_read_evidence:
            if item.difficulty_id == difficulty.id and any(
                _quotation_key(item.body_excerpt) == _quotation_key(a.quote)
                for a in difficulty.anchors
            ):
                label = aliases_by_block_id.get(
                    difficulty.owner_block_id, difficulty.owner_block_id
                )
                msg = f"цитата оригинала не является объяснением ({label}, {difficulty.id})"
                raise _analysis_error(
                    msg,
                    rule_code="quote_as_explanation",
                    block_ids=(difficulty.owner_block_id,),
                )
    for ids, limit in note_group_limits:
        group_owners = [owner for owner in ids if owner in note_owners]
        if len(group_owners) > limit:
            labels = ", ".join(aliases_by_block_id.get(b, b) for b in group_owners)
            msg = f"обязательные заметки не помещаются в реальную группу обработки ({labels})"
            raise _analysis_error(
                msg, rule_code="note_group_budget", block_ids=group_owners
            )
    over_block_owners = [
        owner
        for owner in aliases.values()
        if owner in note_owners
        and note_limits_by_block is not None
        and note_limits_by_block.get(owner, 0) < 1
    ] or [
        owner
        for owner in sorted(note_owners)
        if note_limits_by_block is not None and note_limits_by_block.get(owner, 0) < 1
    ]
    if len(note_owners) > language_notes_per_chapter or over_block_owners:
        target_owners = (
            over_block_owners
            or [b for b in aliases.values() if b in note_owners]
            or sorted(note_owners)
        )
        labels = ", ".join(aliases_by_block_id.get(b, b) for b in target_owners)
        suffix = f" ({labels})" if labels else ""
        msg = f"содержательный план не помещается в бюджет языковых заметок{suffix}"
        raise _analysis_error(
            msg, rule_code="note_chapter_budget", block_ids=target_owners
        )
    if mandatory_facts is not None:
        validate_fact_glossary(
            plan, mandatory_facts, aliases_by_block_id=aliases_by_block_id
        )
    return plan


def _quotation_key(text: str) -> str:
    """Strips only outer quotation formatting without semantic evaluation."""
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or unicodedata.category(text[start]).startswith("P")):
        start += 1
    while end > start and (text[end - 1].isspace() or unicodedata.category(text[end - 1]).startswith("P")):
        end -= 1
    return text[start:end]


_GENDER_NOTE_PATTERN = re.compile(
    r"\b("
    r"мужск\w*\s+(?:род\w*|пол\w*)"
    r"|женск\w*\s+(?:род\w*|пол\w*)"
    r"|средн\w*\s+род\w*"
    r"|male|female|neuter"
    r"|(?:мужск(?:ой|ого)|женск(?:ий|ого)|средн(?:ий|его))(?=$|[\s,;:.(\[])(?!(?:\s+\w))"
    r")\b",
    re.IGNORECASE,
)


def _extract_gender(note: str | None) -> str | None:
    if not note:
        return None
    match = _GENDER_NOTE_PATTERN.search(note)
    if match is None:
        return None
    label = match.group(1).casefold()
    if label.startswith("мужск") or label == "male":
        return "male"
    if label.startswith("женск") or label == "female":
        return "female"
    if label.startswith("средн") or label == "neuter":
        return "neuter"
    return None


def _collect_glossary_fact_conflicts(
    context_facts: Sequence[ContextFact],
    mandatory_facts: Mapping[str, Mapping[str, str]],
    aliases_by_block_id: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    normalized_mandatory = {
        glossary_key(k): v for k, v in mandatory_facts.items()
    }
    alias_map = aliases_by_block_id or {}
    details: list[str] = []
    block_ids: list[str] = []
    for fact in context_facts:
        expected = normalized_mandatory.get(
            glossary_key(fact.subject_anchor.quote), {}
        ).get(fact.kind)
        if expected is None or fact.certainty == "unknown":
            continue
        if fact.kind == "gender":
            expected_gender = _extract_gender(expected) or expected
            actual_gender = _extract_gender(fact.value) or fact.value
            has_conflict = actual_gender != expected_gender
        else:
            has_conflict = fact.value != expected
        if has_conflict:
            block_id = fact.subject_anchor.block_id
            label = alias_map.get(block_id, block_id)
            details.append(f"{label}: «{_preview_quote(fact.subject_anchor.quote)}»")
            block_ids.append(block_id)
    return list(dict.fromkeys(details)), list(dict.fromkeys(block_ids))


def validate_fact_glossary(
    plan,
    mandatory_facts,
    aliases_by_block_id: Mapping[str, str] | None = None,
):
    """Reject explicit contradictions with structured mandatory glossary decisions."""
    if plan is None or plan.version != 3:
        return
    details, block_ids = _collect_glossary_fact_conflicts(
        plan.context_facts, mandatory_facts, aliases_by_block_id
    )
    if details:
        msg = (
            "контекстная атрибуция противоречит обязательному решению глоссария "
            f"({'; '.join(details)})"
        )
        raise _analysis_error(
            msg,
            rule_code="glossary_fact_conflict",
            block_ids=block_ids,
        )


# ’ inside a word (Don’t, car’s), before a word (’em), or after a word when a
# closing ’ still follows inside the same replica (boys’ room, Dursleys’,) is
# an apostrophe, not the quote closer.
_SINGLE_QUOTE_APOSTROPHE = (
    r"’(?=\w)"
    r"|(?<=\w)’(?=(?:\s+\w|[,;:])(?:[^‘’]|’(?=\w))*’(?!\w))"
)
_DIALOGUE_QUOTES = re.compile(
    rf"‘(?:[^‘’]|{_SINGLE_QUOTE_APOSTROPHE})*’(?!\w)|“[^“”]*”|\"[^\"“”]*\"|«[^«»]*»"
)
_SUBORDINATORS = frozenset({
    "if",
    "when",
    "because",
    "although",
    "though",
    "unless",
    "whether",
    "couldn't",
    "wouldn't",
    "shouldn't",
})
_QUESTION_TAG_RE = re.compile(
    r",\s*(?:weren['’]t|were|isn['’]t|is|aren['’]t|are|wasn['’]t|was|"
    r"don['’]t|do|doesn['’]t|does|didn['’]t|did|haven['’]t|have|hasn['’]t|has|"
    r"hadn['’]t|had|won['’]t|will|wouldn['’]t|would|can['’]t|can|couldn['’]t|could|"
    r"shouldn['’]t|should)\s+(?:there|it|he|she|they|we|you|i)\s*\?",
    re.IGNORECASE,
)
_NARRATIVE_GLOSSARY_WORD_LIMIT = 45
_NARRATIVE_GLOSSARY_SENTENCE_LIMIT = 4
_NARRATIVE_GLOSSARY_DIFFICULTY_LIMIT = 3
_DIALOGUE_GLOSSARY_WORD_LIMIT = 15
_REMARK_DIFFICULTY_KINDS = frozenset({"vocabulary", "false_friend"})
_LIGHT_DIALOGUE_TREATMENTS = frozenset({BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS})
_DIALOGUE_BLOCK_TYPES = frozenset({BlockType.PARAGRAPH, BlockType.QUOTE})
_DEMOTABLE_BLOCK_TYPES = frozenset({BlockType.PARAGRAPH, BlockType.QUOTE, BlockType.LIST_ITEM})
_BASIC_REPORTING = frozenset({
    "said", "asked", "replied", "answered", "told", "cried", "shouted", "called",
})
_NONBASIC_REPORTING = frozenset({
    "demanded", "demanding", "demands",
    "murmured", "murmuring", "murmurs",
    "whispered", "whispering", "whispers",
    "snapped", "snapping", "snaps",
    "grudgingly",
    "muttered", "muttering", "mutters",
    "stammered", "stammering", "stammers",
    "gasped", "gasping", "gasps",
    "bellowed", "bellowing", "bellows",
    "hissed", "hissing", "hisses",
    "growled", "growling", "growls",
    "exclaimed", "exclaiming",
    "sobbed", "sobbing", "sobs",
    "pleaded", "pleading", "pleads",
    "retorted", "retorting",
    "barked", "barking", "barks",
    "croaked", "croaking",
    "spluttered", "spluttering",
    "drawled", "drawling",
    "sneered", "sneering",
    "sighed", "sighing", "sighs",
    "groaned", "groaning", "groans",
    "yelled", "yelling", "yells",
    "shrieked", "shrieking",
    "whimpered", "whimpering",
    "thundered", "thundering",
    "roared", "roaring", "roars",
    "inquired", "inquiring",
    "protested", "protesting",
    "confessed", "confessing",
    "admitted", "admitting",
    "announced", "announcing",
    "declared", "declaring",
    "remarked", "remarking",
    "interrupted", "interrupting",
    "warned", "warning",
    "shouted", "shouting", "shouts",
    "cried", "crying", "cries",
    "mumbled", "mumbling", "mumbles",
    "snarled", "snarling", "snarls",
    "wailed", "wailing",
    "screamed", "screaming", "screams",
    "squeaked", "squeaking",
    "rasped", "rasping",
    "blurted", "blurting",
    "scoffed", "scoffing",
    "faltered", "faltering",
    "grunted", "grunting", "grunts",
    "snorted", "snorting",
    "whined", "whining",
    "boomed", "booming",
    "spat",
    "stupidly", "eagerly", "gingerly", "furiously", "nervously", "casually",
    "sternly", "firmly", "anxiously", "bitterly", "wearily", "sleepily", "coldly",
})
_SHORT_B1_WORD_LIMIT = 10
_SINGLE_B1_REPORTING_VERBS = frozenset({
    "winked", "chuckled", "groaned", "peered", "sniggered", "moaned",
})
_DIALECT_WORDS = frozenset({
    "yeh", "yer", "ter", "summat", "fer", "meself", "mighta", "dunno",
    "shan't", "blimey", "codswallop",
})
_SHORT_B1_WORDS = frozenset({
    "winked", "groaned", "chuckled", "peered", "sniggered", "moaned", "sweetums",
    "cannon", "budge", "lump", "shack", "shivered", "uneasy", "nervous",
    "yeh", "yer", "ter", "summat", "fer", "meself", "mighta", "dunno",
    "shan't", "blimey", "codswallop",
})
_SHORT_B1_PHRASES = (
    ("fancy", "seeing", "you"),
    ("how", "very", "wrong"),
    ("suppose", "so"),
    ("kept", "what", "from"),
)
_A2_B1_PHRASAL_VERB_RE = re.compile(
    r"(?:"
    r"\b(?:calm|calms|calmed|calming)\s+down\b"
    r"|\b(?:figure|figures|figured|figuring)\s+out\b"
    r"|\b(?:work|works|worked|working)\s+out\b"
    r"|\b(?:turn|turns|turned|turning)\s+out\b"
    r"|\b(?:give|gives|gave|given|giving)\s+in\b"
    r"|\b(?:end|ends|ended|ending)\s+up\b"
    r"|\b(?:break|breaks|broke|broken|breaking)\s+down\b"
    r"|\b(?:pass|passes|passed|passing)\s+out\b"
    r"|\b(?:show|shows|showed|shown|showing)\s+up\b"
    r"|\b(?:hang|hangs|hung|hanging)\s+on\b"
    r"|\b(?:hold|holds|held|holding)\s+on\b"
    r"|(?<!-)\b(?:make|makes|made|making)\s+up\b(?!-)"
    r"|\b(?:catch|catches|caught|catching)\s+up\b"
    r"|\b(?:set|sets|setting)\s+off\b"
    r"|\b(?:carry|carries|carried|carrying)\s+on\b"
    r"|\b(?:put|puts|putting)\s+up\s+with\b"
    r"|\b(?:run|runs|ran|running)\s+out\s+of\b"
    r"(?!\s+(?:the|a|an|my|his|her|their|our|your|this|that)\s+"
    r"(?:house|room|door|building|shop|store|school|street|park|yard|garden|"
    r"car|bus|train|hall|kitchen|bedroom|office|forest|woods)\b)"
    r"|\b(?:come|comes|came|coming)\s+across\b"
    r"(?!\s+(?:the|a|an|this|that)\s+"
    r"(?:street|road|room|bridge|river|yard|park|hall|field|floor|path)\b)"
    r")",
    re.IGNORECASE,
)


def _has_b1_phrasal_verb(text: str) -> bool:
    return _A2_B1_PHRASAL_VERB_RE.search(text) is not None
_A2_PRESENT_PERFECT_ADVERBS = (
    r"(?:not|never|ever|already|just|still|really|always|often|only|almost|"
    r"finally|barely|hardly|nearly|recently|yet)"
)
_A2_PRESENT_PERFECT_EXCLUDED_ED = (
    r"(?!hundred\b|naked\b|wicked\b|sacred\b|crooked\b|ragged\b|jagged\b|"
    r"rugged\b|wretched\b|beloved\b|pointed\b|colored\b|coloured\b|"
    r"fried\b|baked\b|boiled\b|roasted\b|sliced\b|dried\b|smoked\b|"
    r"salted\b|pickled\b|canned\b|bottled\b|chilled\b|scented\b|"
    r"padded\b|lined\b|faded\b|cracked\b|rusted\b|stained\b|tangled\b|"
    r"matted\b|curled\b|braided\b|bearded\b|winged\b|horned\b|clawed\b|"
    r"fanged\b|scaled\b|feathered\b|armored\b|armoured\b|jeweled\b|"
    r"beaded\b|carved\b|painted\b|printed\b|gilded\b|silvered\b)"
)
_A2_IRREGULAR_PARTICIPLES = (
    r"been|done|gone|seen|taken|given|made|found|told|heard|known|kept|left|"
    r"lost|met|felt|sent|spent|brought|bought|caught|taught|thought|written|"
    r"spoken|broken|chosen|driven|eaten|fallen|forgotten|forgiven|frozen|"
    r"hidden|ridden|risen|shaken|shown|stolen|sworn|thrown|woken|worn|begun|"
    r"drunk|rung|sung|sunk|swum|become|come|held|paid|laid|stood|understood|"
    r"slept|wept|swept|crept|dealt|meant|built|burnt|learnt|dreamt|lit|sat|"
    r"won|hung|dug|stuck|struck|swung|fed|led|bled|bred|fled|grown|blown|"
    r"flown|drawn|withdrawn|torn|born|borne|beaten|bitten|clung|flung|slid|"
    r"spun|stung|wrung|gotten|said|had"
)
_A2_ACTIVE_CONTRACTION_PARTICIPLES = (
    r"been|gone|done|seen|taken|given|written|spoken|eaten|forgotten|forgiven|"
    r"left|made|told|heard|found|met|known|kept|sent|spent|brought|bought|"
    r"caught|taught|thought|built|paid|stood|understood|won|grown|begun|"
    r"become|come|had|slept|felt|lost|gotten"
)
_A2_SAFE_USED_AFTER = (
    r"used(?=\s+(?:to|it|this|that|these|those|them|him|her|me|us|you|"
    r"my|your|his|its|our|their|the|an?|some|any|all|every|one|up)\b)"
)
_A2_SAFE_USED_AFTER_CONTRACTION = (
    r"used(?=\s+(?:it|this|that|these|those|them|him|her|me|us|you|"
    r"my|your|his|its|our|their|the|an?|some|any|all|every|one|up)\b)"
)
_A2_PRESENT_PERFECT_PARTICIPLE = (
    rf"(?:{_A2_IRREGULAR_PARTICIPLES}|{_A2_SAFE_USED_AFTER}|"
    rf"{_A2_PRESENT_PERFECT_EXCLUDED_ED}(?:died|lied|tied|[a-z]{{3,}}ed))"
)
_A2_INVERTED_PROPER_SUBJECT = (
    r"(?-i:[A-Z][a-z.'’]+(?:\s+(?:and\s+)?[A-Z][a-z.'’]+){0,2})"
)
_A2_PRESENT_PERFECT_RE = re.compile(
    rf"(?:"
    rf"\b(?:have|has|haven['’]t|hasn['’]t|(?:i|you|we|they|who|what|there)['’]ve)"
    rf"\s+(?:{_A2_PRESENT_PERFECT_ADVERBS}\s+){{0,2}}"
    rf"{_A2_PRESENT_PERFECT_PARTICIPLE}\b"
    rf"|\b(?:have|has|haven['’]t|hasn['’]t)\s+"
    rf"(?:i|you|we|they|he|she|it|{_A2_INVERTED_PROPER_SUBJECT})\s+"
    rf"(?:{_A2_PRESENT_PERFECT_ADVERBS}\s+){{0,2}}"
    rf"{_A2_PRESENT_PERFECT_PARTICIPLE}\b"
    rf"|\b(?:he|she|it|that|there|who|what)['’]s\s+"
    rf"(?:{_A2_PRESENT_PERFECT_ADVERBS}\s+){{0,2}}"
    rf"(?:{_A2_ACTIVE_CONTRACTION_PARTICIPLES}|{_A2_SAFE_USED_AFTER_CONTRACTION})\b"
    rf")",
    re.IGNORECASE,
)
_A2_WOULD_RE = re.compile(
    r"(?:"
    r"\b(?:would|wouldn['’]t)\b"
    r"|\b(?:i|you|he|she|it|we|they|who|what|that|there)['’]d\s+"
    r"(?:(?:not|never|always|probably|really|surely|definitely|almost|just|still)\s+)?"
    r"(?:rather|better|like|love|prefer|be|have|go|come|say|tell|know|think|see|"
    r"get|make|take|give|want|need|do|feel|look|seem|try|let|stay|leave|help|"
    r"wait|keep|find|call|ask|hear|work|live|die|run|turn|start|stop|remember|"
    r"forget|believe|understand|agree|care|mind|bet|swear|guess|hate)\b"
    r")",
    re.IGNORECASE,
)
_A2_WH_STRANDED_PREP_RE = re.compile(
    r"\b(?:what|who|whom|which|where)(?:['’]s|['’]re|['’]d|['’]ll)?\b"
    r"[^.!?]{0,80}?"
    r"\b(?:for|about|with|from|to|at|of|into|onto|through|after|by|upon|in|on)\s*\?",
    re.IGNORECASE,
)
_A2_EMBEDDED_WHAT_RE = re.compile(
    r"\b(?:"
    r"is|was|were|be|being|been|am|are"
    r"|(?:that|it|this|here|there)['’]s"
    r"|of|about|for|at|to|with|from|on|in|by|into|after|before|over|than|like"
    r"|know|knew|known|knows|see|saw|seen|sees|hear|heard|hears"
    r"|say|said|says|tell|told|tells|ask|asked|asks|wonder|wondered|wonders"
    r"|remember|remembered|remembers|forget|forgot|forgotten|forgets"
    r"|understand|understood|understands|believe|believed|believes"
    r"|mean|meant|means|want|wanted|wants|need|needed|needs"
    r"|care|cares|mind|minds|matter|matters|do|did|done|does"
    r"|find|found|finds|show|showed|shown|shows|explain|explained|explains"
    r"|guess|guessed|guesses|imagine|imagined|imagines"
    r"|realize|realized|realise|realised|notice|noticed|watch|watched|look|looked"
    r"|think|thought|thinks|feel|felt|feels|keep|kept|keeps"
    r"|lose|lost|loses|get|got|gets|take|took|taken|takes"
    r"|give|gave|given|gives|make|made|makes"
    r"|me|him|her|us|them|you"
    r")\s+what(?:['’]s|['’]d|['’]ll)?\s+(?!a\b|an\b|about\b)[a-z]",
    re.IGNORECASE,
)
_A2_INITIAL_WHAT_CLAUSE_RE = re.compile(
    r"^\s*[‘“\"']?what\s+"
    r"(?:(?:i|you|he|she|it|we|they|this|that|there|i['’]ve|you['’]ve|we['’]ve|he['’]s|she['’]s)\b"
    r"|happened\b|matters\b|comes\b|came\b|follows\b|followed\b|makes\b|made\b|seemed\b|looked\b)",
    re.IGNORECASE,
)
_A2_SARCASM_RE = re.compile(
    r"(?:"
    r"\b(?:yeah,?\s+right"
    r"|oh,?\s+(?:great|sure|really|brilliant|fantastic|terrific|wonderful|splendid|swell|fabulous)"
    r"|just\s+(?:great|wonderful|perfect|what\s+(?:i|we)\s+needed)"
    r"|how\s+(?:wonderful|lovely|helpful|clever|original|thoughtful|charming|convenient)"
    r"|real\s+(?:nice|smart|funny|helpful|smooth|bright|genius)"
    r"|very\s+funny"
    r"|big\s+(?:deal|surprise|help)"
    r"|what\s+a\s+surprise"
    r"|no\s+kidding"
    r"|you\s+don['’]t\s+say"
    r"|fat\s+chance"
    r"|good\s+luck\s+with\s+that"
    r"|like\s+(?:i|he|she|we|they)\s+care"
    r"|gee,?\s+thanks"
    r"|lucky\s+(?:me|us|you)"
    r"|nice\s+going"
    r"|way\s+to\s+go"
    r"|so\s+much\s+for"
    r"|thanks\s+for\s+nothing"
    r"|tell\s+me\s+about\s+it)\b"
    r"|(?:^|[‘“\"'\s])as\s+if\s*[!.]"
    r")",
    re.IGNORECASE,
)
_A2_NONSTANDARD_SPEECH_RE = re.compile(
    r"\b(?:"
    r"ain['’]t"
    r"|(?:don['’]t|doesn['’]t|didn['’]t)\s+got"
    r"|(?:he|she|it)\s+don['’]t"
    r"|(?:you|we|they)\s+was"
    r"|gonna|wanna|gotta|lemme|gimme|outta|innit|shoulda|woulda|coulda"
    r")\b",
    re.IGNORECASE,
)
_A2_STATE_CHANGE_IDIOM_RE = re.compile(
    r"\b(?:"
    r"(?:go|goes|went|going|gone)\s+(?:red|pale|white|purple|pink|green|blue|gray|grey|"
    r"dark|black|blank|numb|silent|quiet|still|stiff|rigid|cold|limp|weak|crazy|mad|"
    r"wild|insane|blind|deaf|wrong|bad|sour|stale|missing|dead)"
    r"|(?:turn|turns|turned|turning)\s+(?:red|pale|white|purple|pink|green|blue|gray|grey|"
    r"crimson|scarlet|cold|sour|nasty|violent|fierce)"
    r"|(?:grow|grows|grew|growing|grown)\s+(?:pale|dark|cold|quiet|silent|still|uneasy|"
    r"nervous|tense|angry|fierce|weak|tired|faint|impatient|suspicious|restless)"
    r"|(?:fall|falls|fell|falling|fallen)\s+(?:silent|quiet|still|asleep|ill|sick|short|flat|apart|prey)"
    r"|(?:make|makes|made|making)\s+(?:me|him|her|us|them|you|it|everyone|everybody|"
    r"anyone|someone|[a-z]+(?:\s+[a-z]+){0,2})\s+(?:uneasy|nervous|anxious|uncomfortable|"
    r"sick|dizzy|furious|mad|angry|crazy|insane|jealous|suspicious|miserable|helpless|"
    r"speechless|breathless|shiver|shudder|flinch|cringe|wince|tremble)"
    r")\b",
    re.IGNORECASE,
)
_A2_IT_IS_TIME_RE = re.compile(
    r"\bit['’]s\s+(?:(?:about|high)\s+)?time\s+(?!to\b|for\b)(?:you|we|i|he|she|they|[a-z]+)\b",
    re.IGNORECASE,
)
_A2_QUOTED_SUBJECT_INVERSION_RE = re.compile(
    r"""[”"’']\s*,?\s*was\s+all\s+(?:(?:the|a|an|this|that|my|his|her)\s+)?[a-z]+\s+"""
    r"""(?:said|asked|replied|answered|told|knew|saw|heard|whispered|muttered|thought)\b""",
    re.IGNORECASE,
)
_A2_NONLEXICAL_BACKING_KINDS = frozenset({"grammar", "idiom", "speech_style"})


@dataclass(frozen=True, slots=True)
class _PromotedBlock:
    owners: frozenset[str]
    had_glossary: bool
    block_text: str = ""
    promoted_ranges: tuple[tuple[int, int], ...] | None = None


def _has_a2_nonlexical_stoplist(text: str) -> bool:
    if (
        _A2_PRESENT_PERFECT_RE.search(text) is not None
        or _A2_WOULD_RE.search(text) is not None
        or _A2_WH_STRANDED_PREP_RE.search(text) is not None
        or _A2_EMBEDDED_WHAT_RE.search(text) is not None
        or _A2_SARCASM_RE.search(text) is not None
        or _A2_NONSTANDARD_SPEECH_RE.search(text) is not None
        or _A2_STATE_CHANGE_IDIOM_RE.search(text) is not None
        or _A2_IT_IS_TIME_RE.search(text) is not None
        or _A2_QUOTED_SUBJECT_INVERSION_RE.search(text) is not None
    ):
        return True
    if "?" not in text and _A2_INITIAL_WHAT_CLAUSE_RE.search(text) is not None:
        return True
    words = expression_words(text)
    return any(word in _DIALECT_WORDS for word in words)
_TRIVIAL_A2_ANCHORS = frozenset({
    ("sounding", "relieved"),
    ("cupboard",),
    ("in", "my", "cupboard"),
    ("kept", "his", "eyes", "shut", "tight"),
    ("a", "giant", "called", "hagrid"),
    ("called", "hagrid"),
    ("was", "going", "to", "a", "school"),
    ("i", "was", "going", "to", "a", "school", "for", "wizards"),
    ("when", "i", "open", "my", "eyes"),
    ("where's",),
    ("stand", "up"),
    ("brazil",),
    ("mr", "h", "potter"),
})
_SPEECH_POINT_QUOTES = re.compile(
    rf'"([^"]+)"|«([^»]+)»|“([^”]+)”|‘((?:[^‘’]|{_SINGLE_QUOTE_APOSTROPHE})+)’(?!\w)'
)
_LEMMA_GLOSS = re.compile(r"(?<![\w'’])([A-Za-z][\w'’]*)\s*=\s*[^,;\n]+")
_STANDARD_SPEECH_GLOSS_WORDS = frozenset({
    "you", "your", "yours", "to", "for", "of", "from", "my", "myself",
    "something", "nothing", "anything", "everything", "last", "want",
    "were", "was", "have", "had", "will", "shall", "not",
})
_SPEAKER_TITLES = frozenset({"mr", "mrs", "miss", "ms", "dr"})
_MALE_TITLES = frozenset({
    "mr", "uncle", "sir", "lord", "king", "prince", "father", "brother",
    "grandfather", "grandpa", "мистер", "дядя", "сэр", "лорд",
})
_FEMALE_TITLES = frozenset({
    "mrs", "miss", "ms", "aunt", "madam", "madame", "lady", "queen", "princess",
    "mother", "sister", "grandmother", "grandma", "миссис", "мисс", "тётя",
    "тетя", "мадам", "леди",
})
_ALL_GENDER_TITLES = _MALE_TITLES | _FEMALE_TITLES | _SPEAKER_TITLES
_SURNAME_TITLES = frozenset({
    "mr", "mrs", "miss", "ms", "dr", "мистер", "миссис", "мисс",
})
_PRONOUN_WORDS = frozenset({
    "he", "she", "it", "they", "him", "her", "his", "hers", "i", "you", "we", "them",
})
_EXTRA_REPORTING_VERBS = frozenset({
    "added", "adding", "began", "beginning", "continued", "continuing",
    "finished", "finishing", "repeated", "repeating", "suggested", "suggesting",
    "explained", "explaining", "agreed", "agreeing",
    "spoke", "speaking", "answered", "answering", "replied", "replying",
    "asked", "asking", "told", "telling", "said", "saying", "called", "calling",
})
_ALL_REPORTING_VERBS = (
    _BASIC_REPORTING | _NONBASIC_REPORTING | _SINGLE_B1_REPORTING_VERBS | _EXTRA_REPORTING_VERBS
)
_CLAUSE_BOUNDARY_WORDS = frozenset({
    "and", "but", "or", "while", "as", "when", "before", "after", "because",
    "until", "although", "though", "if", "since", "so", "that", "who", "which",
})
_SUBORDINATING_CLAUSE_BOUNDARIES = _CLAUSE_BOUNDARY_WORDS - {"and"}
_ADDRESSEE_PREPOSITIONS = frozenset({"to", "at"})
_SPEAKER_NAME_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "mr", "mrs", "miss", "ms", "dr",
    "uncle", "aunt", "sir", "madam", "madame", "lord", "lady", "professor",
})
_REMARK_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st"})
_CHAPTER_NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety", "hundred",
})
_ROMAN_NUMERAL = re.compile(
    r"(?=[ivxlcdm]+$)m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
)
_EXPLANATION_ONLY_NON_GRAMMAR_KINDS = frozenset({"wordplay", "parody", "semantic_relation"})
_AUTO_VOCAB_POINT_PREFIX = "Дать краткую контекстную словарную подсказку для «"
_NON_LEXICAL_KIND_MARKER_RE = re.compile(
    r"(?:каламбур|игр(?:а|ы|е|у|ой|ою)\s+слов|двойн\w*\s+смысл\w*|двусмыслен|"
    r"пароди|обыгрыва|рифм|созвуч\w*|прототип|аллюзи|"
    r"\bwordplay\b|\bpun\b|\bparody\b)",
    re.IGNORECASE,
)
_HOMOGENEOUS_LIST_BLOCK_TYPES = frozenset({BlockType.LIST_ITEM, BlockType.PARAGRAPH})
_NUMBERED_LIST_ITEM_RE = re.compile(r"^\d+[.)]\s+\S")


@dataclass(frozen=True, slots=True)
class _StoplistDecision:
    mode: BlockTreatment
    target_spans: tuple[tuple[int, int], ...] = ()


def _replace_payload_difficulties(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
    difficulties,
    *,
    treatment_plan=None,
    before_you_read_evidence=None,
):
    if isinstance(payload, PedagogicalAnalysisResponse):
        kwargs: dict[str, object] = {"difficulties": tuple(difficulties)}
        if treatment_plan is not None:
            kwargs["treatment_plan"] = tuple(treatment_plan)
        if before_you_read_evidence is not None:
            kwargs["before_you_read_evidence"] = tuple(before_you_read_evidence)
        return replace(payload, **kwargs)
    content_plan = payload.content_plan
    if content_plan is None:
        if treatment_plan is None:
            return payload
        return replace(payload, treatment_plan=tuple(treatment_plan))
    cp_kwargs: dict[str, object] = {"difficulties": tuple(difficulties)}
    if before_you_read_evidence is not None:
        cp_kwargs["before_you_read_evidence"] = tuple(before_you_read_evidence)
    updated_cp = replace(content_plan, **cp_kwargs)
    if treatment_plan is not None:
        return replace(payload, treatment_plan=tuple(treatment_plan), content_plan=updated_cp)
    return replace(payload, content_plan=updated_cp)


def _is_auto_vocab_difficulty(difficulty) -> bool:
    if getattr(difficulty, "id", "").startswith("auto-vocab-"):
        return True
    points = getattr(difficulty, "required_points", ())
    return (
        getattr(difficulty, "kind", None) == "vocabulary"
        and len(points) == 1
        and points[0].description.startswith(_AUTO_VOCAB_POINT_PREFIX)
    )


def _difficulty_on_glossary_only(
    difficulty,
    items_by_owner: Mapping[str, TreatmentPlanItemResponse],
    aliases: Mapping[str, str],
    blocks_by_id: Mapping[str, DocumentBlock],
) -> bool:
    owner = difficulty.owner_block_id
    item = items_by_owner.get(owner) or items_by_owner.get(aliases.get(owner, ""))
    if item is None:
        return False
    if not item.segments:
        return item.treatment is BlockTreatment.GLOSSARY_ONLY
    block = _block_for_plan_item(item.source_block_id, blocks_by_id, aliases)
    if block is None or not block.segment_spans:
        return any(seg.treatment is BlockTreatment.GLOSSARY_ONLY for seg in item.segments)
    spans = block.segment_spans
    owners = frozenset(part for part in (item.source_block_id, block.id, owner) if part)
    glossary_ranges = tuple(
        (spans[seg.index - 1].start, spans[seg.index - 1].end)
        for seg in item.segments
        if seg.treatment is BlockTreatment.GLOSSARY_ONLY and 1 <= seg.index <= len(spans)
    )
    if not glossary_ranges:
        return False
    return _difficulty_hits_ranges(difficulty, owners, block.text, glossary_ranges)


def _is_single_word_anchor(quote: str) -> bool:
    if any(dash in quote for dash in ("-", "–", "—")):
        return False
    return len(expression_words(quote)) == 1


def normalize_difficulty_kinds(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
    blocks: Sequence[DocumentBlock] = (),
    block_ids_by_alias: Mapping[str, str] | None = None,
) -> ChapterAnalysisResponse | PedagogicalAnalysisResponse:
    """Fix provably contradictory difficulty kinds before A2 normalization and validation.

    Reclassifies single-anchor `wordplay`/`parody`/`semantic_relation` with
    `help_kind="lexical_hint"` on `GLOSSARY_ONLY` blocks/segments (when `required_points`
    contains no pun/parody explanation markers) and single-word `idiom` entries
    to `kind="vocabulary"`. Multi-anchor and genuine wordplay/parody explanations
    are preserved untouched.
    """
    difficulties = _response_difficulties(payload)
    if not difficulties:
        return payload
    aliases = block_ids_by_alias or {}
    blocks_by_id = {block.id: block for block in blocks}
    items_by_owner: dict[str, TreatmentPlanItemResponse] = {}
    for item in payload.treatment_plan:
        items_by_owner[item.source_block_id] = item
        resolved_id = aliases.get(item.source_block_id)
        if resolved_id:
            items_by_owner[resolved_id] = item
    updated: list = []
    changed = False
    for difficulty in difficulties:
        new_kind = difficulty.kind
        anchors = getattr(difficulty, "anchors", ())
        if (
            difficulty.kind in _EXPLANATION_ONLY_NON_GRAMMAR_KINDS
            and difficulty.help_kind == "lexical_hint"
            and len(anchors) == 1
            and _difficulty_on_glossary_only(difficulty, items_by_owner, aliases, blocks_by_id)
            and not any(
                _NON_LEXICAL_KIND_MARKER_RE.search(point.description)
                for point in difficulty.required_points
            )
        ):
            new_kind = "vocabulary"
        elif (
            difficulty.kind == "idiom"
            and len(anchors) == 1
            and _is_single_word_anchor(getattr(anchors[0], "quote", ""))
        ):
            new_kind = "vocabulary"
        if new_kind != difficulty.kind:
            updated.append(replace(difficulty, kind=new_kind))
            changed = True
        else:
            updated.append(difficulty)
    if not changed:
        return payload
    return _replace_payload_difficulties(payload, updated)


def _item_has_mixed_translate_segments(item: TreatmentPlanItemResponse) -> bool:
    if not item.segments:
        return False
    modes = {seg.treatment for seg in item.segments}
    return BlockTreatment.TRANSLATE in modes and len(modes) > 1


def unify_verse_stanza_treatments(
    items: Sequence[TreatmentPlanItemResponse],
    blocks: Sequence[DocumentBlock],
    block_ids_by_alias: Mapping[str, str],
    *,
    english_help: bool = False,
) -> tuple[tuple[TreatmentPlanItemResponse, ...], bool]:
    """Keep verse/poem stanzas in one cohesive language mode (never half-translated).

    Leaves all-light (`READ_AS_IS`/`GLOSSARY_ONLY`) stanzas untouched. When any line
    or segment of a stanza requires `TRANSLATE` alongside non-`TRANSLATE` lines,
    elevates the entire stanza to `TRANSLATE`.
    """
    if english_help or not items or not blocks:
        return tuple(items), False
    blocks_by_id = {block.id: block for block in blocks}
    item_index_by_block_id: dict[str, int] = {}
    ordered_blocks: list[DocumentBlock] = []
    for idx, item in enumerate(items):
        block = _block_for_plan_item(item.source_block_id, blocks_by_id, block_ids_by_alias)
        if block is not None:
            item_index_by_block_id[block.id] = idx
            ordered_blocks.append(block)
    ordered_blocks.sort(key=lambda b: b.order)
    updated = list(items)
    changed = False

    # Group 0: single BlockType.POEM block with mixed TRANSLATE / non-TRANSLATE segments.
    for block in ordered_blocks:
        if block.type is BlockType.POEM:
            idx = item_index_by_block_id[block.id]
            if _item_has_mixed_translate_segments(updated[idx]):
                updated[idx] = _force_block_treatment(updated[idx], BlockTreatment.TRANSLATE)
                changed = True

    # Group 1 (contiguous BlockType.POEM) & Group 2 (implicit enjambed PARAGRAPH stanzas).
    for run in detect_verse_stanza_runs(ordered_blocks):
        run_indices = [item_index_by_block_id[b.id] for b in run if b.id in item_index_by_block_id]
        if len(run_indices) < 2:
            continue
        has_translate = any(
            updated[i].treatment is BlockTreatment.TRANSLATE
            or any(seg.treatment is BlockTreatment.TRANSLATE for seg in updated[i].segments)
            for i in run_indices
        )
        has_non_translate = any(
            updated[i].treatment is not BlockTreatment.TRANSLATE
            or any(seg.treatment is not BlockTreatment.TRANSLATE for seg in updated[i].segments)
            for i in run_indices
        )
        if has_translate and has_non_translate:
            for i in run_indices:
                forced = _force_block_treatment(updated[i], BlockTreatment.TRANSLATE)
                if forced is not updated[i]:
                    updated[i] = forced
                    changed = True

    return tuple(updated), changed


def _is_title_by_author_item(text: str) -> bool:
    """Whether `text` matches a `<Title> by <Author>` booklist item rather than narrative prose."""
    if "\n" in text or len(text) > 200:
        return False
    title_part, sep, author_part = text.rpartition(" by ")
    if not sep:
        return False
    author_tokens = author_part.strip().rstrip(".;,").split()
    if not (1 <= len(author_tokens) <= 4):
        return False
    for token in author_tokens:
        core = token.strip(".,;:()[]\"'“”‘’«»")
        if not core or not any(ch.isalpha() for ch in core) or not core[0].isupper():
            return False
    title_tokens = title_part.strip().split()
    if not (1 <= len(title_tokens) <= 16):
        return False
    has_upper_alpha = False
    for idx, token in enumerate(title_tokens):
        core = token.strip(".,:;!?()[]\"'“”‘’«»—–-")
        if not core:
            continue
        if any(ch.isdigit() for ch in core) and not any(ch.isalpha() for ch in core):
            continue
        if core[0].isupper():
            has_upper_alpha = True
            continue
        if idx > 0 and core.casefold() in _TITLE_CASE_MINOR_WORDS:
            continue
        return False
    return has_upper_alpha


def _homogeneous_list_kind(block: DocumentBlock) -> str | None:
    """Classify a block as a numbered (`1.`/`1)`) or `<Title> by <Author>` list item."""
    if block.type not in _HOMOGENEOUS_LIST_BLOCK_TYPES:
        return None
    stripped = block.text.strip()
    if not stripped:
        return None
    if _NUMBERED_LIST_ITEM_RE.match(stripped):
        return "numbered"
    if _is_title_by_author_item(stripped):
        return "title_by_author"
    return None


def unify_homogeneous_list_treatments(
    items: Sequence[TreatmentPlanItemResponse],
    blocks: Sequence[DocumentBlock],
    block_ids_by_alias: Mapping[str, str],
    *,
    english_help: bool = False,
) -> tuple[tuple[TreatmentPlanItemResponse, ...], bool]:
    """Lift a single stray `READ_AS_IS` inside a numbered or `<Title> by <Author>` list.

    Leaves `GLOSSARY_ONLY`, quantitative items (`1 telescope`), and prose untouched.
    """
    if english_help or not items or not blocks:
        return tuple(items), False
    blocks_by_id = {block.id: block for block in blocks}
    item_index_by_block_id: dict[str, int] = {}
    ordered_blocks: list[DocumentBlock] = []
    for idx, item in enumerate(items):
        block = _block_for_plan_item(item.source_block_id, blocks_by_id, block_ids_by_alias)
        if block is not None:
            item_index_by_block_id[block.id] = idx
            ordered_blocks.append(block)
    ordered_blocks.sort(key=lambda b: b.order)
    updated = list(items)
    changed = False

    n = len(ordered_blocks)
    index = 0
    while index < n:
        kind = _homogeneous_list_kind(ordered_blocks[index])
        if kind is None:
            index += 1
            continue
        end = index + 1
        while (
            end < n
            and ordered_blocks[end].order == ordered_blocks[end - 1].order + 1
            and _homogeneous_list_kind(ordered_blocks[end]) == kind
        ):
            end += 1
        if end - index >= 3:
            run_indices = [
                item_index_by_block_id[b.id]
                for b in ordered_blocks[index:end]
                if b.id in item_index_by_block_id
            ]
            translate_count = sum(
                1
                for i in run_indices
                if updated[i].treatment is BlockTreatment.TRANSLATE
                or any(seg.treatment is BlockTreatment.TRANSLATE for seg in updated[i].segments)
            )
            read_as_is_indices = [
                i
                for i in run_indices
                if updated[i].treatment is BlockTreatment.READ_AS_IS
                and all(seg.treatment is BlockTreatment.READ_AS_IS for seg in updated[i].segments)
            ]
            if translate_count >= 2 and len(read_as_is_indices) == 1:
                target_idx = read_as_is_indices[0]
                forced = _force_block_treatment(updated[target_idx], BlockTreatment.TRANSLATE)
                if forced is not updated[target_idx]:
                    updated[target_idx] = forced
                    changed = True
        index = end

    return tuple(updated), changed


def unify_chapter_analysis_verse_stanzas(
    analysis: ChapterAnalysis,
    blocks: Sequence[DocumentBlock],
    *,
    english_help: bool = False,
) -> ChapterAnalysis:
    """Unify verse stanzas across fragment boundaries on a merged `ChapterAnalysis`."""
    if english_help or not blocks or not analysis.treatment_plan.items:
        return analysis
    response_items = tuple(
        TreatmentPlanItemResponse(
            source_block_id=item.block_id,
            treatment=item.treatment,
            segments=tuple(
                SegmentTreatmentResponse(index=idx, treatment=mode)
                for idx, mode in enumerate(item.segment_treatments, start=1)
            ),
        )
        for item in analysis.treatment_plan.items
    )
    identity_aliases = {block.id: block.id for block in blocks}
    unified_items, changed = unify_verse_stanza_treatments(
        response_items, blocks, identity_aliases, english_help=english_help,
    )
    if not changed:
        return analysis
    blocks_by_id = {block.id: block for block in blocks}
    promoted_blocks: list[_PromotedBlock] = []
    new_plan_items: list[TreatmentPlanItem] = []
    for orig, upd in zip(response_items, unified_items, strict=True):
        block = blocks_by_id.get(upd.source_block_id)
        if block is not None and upd is not orig:
            promoted = _compute_promoted_block(orig, upd, block, frozenset((block.id,)))
            if promoted is not None:
                promoted_blocks.append(promoted)
        seg_modes = tuple(seg.treatment for seg in upd.segments)
        new_plan_items.append(
            TreatmentPlanItem(
                block_id=upd.source_block_id,
                treatment=upd.treatment,
                segment_treatments=seg_modes,
            )
        )
    new_treatment_plan = TreatmentPlan(
        section_id=analysis.section_id,
        items=tuple(new_plan_items),
    )
    new_content_plan = analysis.content_plan
    if new_content_plan is not None and promoted_blocks:
        covered_points: dict[str, set[str]] = {}
        for ev in new_content_plan.before_you_read_evidence:
            covered_points.setdefault(ev.difficulty_id, set()).add(ev.point_id)
        updated_diffs = []
        dropped_ids: set[str] = set()
        for difficulty in new_content_plan.difficulties:
            switched = _switch_promoted_hint(difficulty, promoted_blocks, covered_points)
            if switched is None or (
                switched is not difficulty and _is_auto_vocab_difficulty(difficulty)
            ):
                dropped_ids.add(difficulty.id)
                continue
            updated_diffs.append(switched)
        new_difficulties = tuple(updated_diffs)
        if new_difficulties != new_content_plan.difficulties:
            new_evidence = tuple(
                ev for ev in new_content_plan.before_you_read_evidence
                if ev.difficulty_id not in dropped_ids
            )
            new_content_plan = replace(
                new_content_plan,
                difficulties=new_difficulties,
                before_you_read_evidence=new_evidence,
            )
    return replace(
        analysis,
        treatment_plan=new_treatment_plan,
        content_plan=new_content_plan,
    )


def _resolved_anchor_spans_for_block(
    difficulties,
    owners: frozenset[str],
    block_text: str,
) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    for difficulty in difficulties:
        if difficulty.owner_block_id not in owners:
            continue
        for anchor in getattr(difficulty, "anchors", ()):
            if getattr(anchor, "block_id", None) not in owners:
                continue
            start = getattr(anchor, "start", None)
            end = getattr(anchor, "end", None)
            if type(start) is int and type(end) is int and 0 <= start < end <= len(block_text):
                spans.add((start, end))
                continue
            quote = getattr(anchor, "quote", "")
            if quote:
                for q_start, q_end, _ in _quote_spans(block_text, quote):
                    spans.add((q_start, q_end))
    return spans


def _has_difficulty_in_range(
    difficulties,
    owners: frozenset[str],
    block_text: str,
    span_start: int,
    span_end: int,
    *,
    whole_block: bool,
) -> bool:
    ranges = ((span_start, span_end),)
    for difficulty in difficulties:
        if difficulty.owner_block_id not in owners:
            continue
        if whole_block or _difficulty_hits_ranges(difficulty, owners, block_text, ranges):
            return True
    return False


def _uncovered_spans_in_slice(
    text: str,
    block_text: str,
    span_start: int,
    span_end: int,
    difficulties,
    owners: frozenset[str],
    *,
    whole_block: bool,
    known_names: Sequence[str] = (),
) -> tuple[tuple[int, int], ...]:
    spans_out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for match in _A2_B1_PHRASAL_VERB_RE.finditer(text):
        phrase = match.group(0)
        if not _uncovered_a2_words(
            (phrase.casefold(),),
            difficulties,
            owners,
            block_text,
            span_start,
            span_end,
            whole_block=whole_block,
        ):
            continue
        span = (span_start + match.start(), span_start + match.end())
        if span not in seen:
            seen.add(span)
            spans_out.append(span)
            if len(spans_out) >= 3:
                return tuple(spans_out)
    if dominant_script(text) == "LATIN":
        word_spans = non_core_a2_word_spans(text, known_names=known_names)
        if word_spans:
            uncovered = set(
                _uncovered_a2_words(
                    tuple(w for w, _, _ in word_spans),
                    difficulties,
                    owners,
                    block_text,
                    span_start,
                    span_end,
                    whole_block=whole_block,
                )
            )
            for word, local_s, local_e in word_spans:
                if word not in uncovered:
                    continue
                span = (span_start + local_s, span_start + local_e)
                if span not in seen and not any(
                    span[0] < ex[1] and ex[0] < span[1] for ex in seen
                ):
                    seen.add(span)
                    spans_out.append(span)
                    if len(spans_out) >= 3:
                        break
    return tuple(spans_out)


def _materialize_a2_glossary_difficulties(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
    plan_items: Sequence[TreatmentPlanItemResponse],
    difficulties: Sequence,
    blocks_by_id: Mapping[str, DocumentBlock],
    block_ids_by_alias: Mapping[str, str],
    stoplist_spans_by_block: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    known_names: Sequence[str] = (),
) -> tuple:
    """Append canonical `vocabulary`/`lexical_hint` difficulties for final GLOSSARY_ONLY spans."""
    is_quoted = isinstance(payload, PedagogicalAnalysisResponse) or isinstance(
        getattr(payload, "content_plan", None), QuotedContentPlanResponse
    )
    is_domain_plan = isinstance(payload, ChapterAnalysisResponse) and isinstance(
        payload.content_plan, ContentPlanResponse
    )
    if not is_quoted and not is_domain_plan:
        return tuple(difficulties)
    from linguary.application.context_builder import _analysis_visible_origins

    existing_ids = {d.id for d in difficulties}
    appended: list = []
    for item in plan_items:
        block = _block_for_plan_item(item.source_block_id, blocks_by_id, block_ids_by_alias)
        if block is None:
            continue
        owners = frozenset(part for part in (item.source_block_id, block.id) if part)
        # Determine final GLOSSARY_ONLY slices of this block.
        glossary_slices: list[tuple[int, int, bool]] = []
        if item.segments:
            spans = block.segment_spans
            sentence_bounds = _sentence_bounds_by_index(block)
            sentence_modes: dict[int, list[BlockTreatment]] = {}
            for seg in item.segments:
                if 1 <= seg.index <= len(spans):
                    sentence_modes.setdefault(spans[seg.index - 1].sentence_index, []).append(
                        seg.treatment
                    )
            for s_idx, modes in sentence_modes.items():
                if strongest_treatment(modes) is BlockTreatment.GLOSSARY_ONLY:
                    s_start, s_end = sentence_bounds[s_idx]
                    glossary_slices.append((s_start, s_end, len(sentence_modes) == 1))
        elif item.treatment is BlockTreatment.GLOSSARY_ONLY:
            glossary_slices.append((0, len(block.text), True))
        if not glossary_slices:
            continue

        candidate_spans: list[tuple[int, int]] = []
        for s_start, s_end in stoplist_spans_by_block.get(block.id, ()):
            if any(g_start <= s_start and s_end <= g_end for g_start, g_end, _ in glossary_slices):
                candidate_spans.append((s_start, s_end))
        for g_start, g_end, whole_block in glossary_slices:
            if any(g_start <= s and e <= g_end for s, e in candidate_spans):
                continue
            if not _has_difficulty_in_range(
                difficulties, owners, block.text, g_start, g_end, whole_block=whole_block
            ):
                for span in _uncovered_spans_in_slice(
                    block.text[g_start:g_end],
                    block.text,
                    g_start,
                    g_end,
                    difficulties,
                    owners,
                    whole_block=whole_block,
                    known_names=known_names,
                ):
                    if span not in candidate_spans:
                        candidate_spans.append(span)
        if not candidate_spans:
            continue

        visible_text, origins = _analysis_visible_origins(block)
        origin_set = set(origins)
        occupied = _resolved_anchor_spans_for_block(difficulties, owners, block.text)
        # Deduplicate and cap at 3 new synthesized hints per block.
        added_for_block = 0
        for s_start, s_end in candidate_spans:
            if added_for_block >= 3:
                break
            if not (0 <= s_start < s_end <= len(block.text)):
                continue
            if (s_start, s_end) in occupied:
                continue
            if any(pos not in origin_set for pos in range(s_start, s_end)):
                continue
            quote = block.text[s_start:s_end]
            if not quote.strip():
                continue
            visible_matches = _quote_spans(visible_text, quote)
            source_matches = _quote_spans(block.text, quote)
            if not visible_matches:
                continue
            point = (
                RequiredPoint(
                    "p1",
                    f"Дать краткую контекстную словарную подсказку для «{quote}».",
                ),
            )
            seq = 1
            while f"auto-vocab-{item.source_block_id}-{seq}" in existing_ids:
                seq += 1
            diff_id = f"auto-vocab-{item.source_block_id}-{seq}"
            if is_quoted:
                if len(visible_matches) == 1 and len(source_matches) == 1:
                    anchor = QuotedSourceAnchorResponse(item.source_block_id, quote)
                elif visible_text == block.text:
                    anchor = QuotedSourceAnchorResponse(
                        item.source_block_id, quote, start=s_start, end=s_end
                    )
                else:
                    # Repeated quote in a block whose visible text differs from source text:
                    # skip rather than emit an offset that could conflict across source_offsets modes.
                    continue
                appended.append(
                    QuotedDifficultyResponse(
                        id=diff_id,
                        anchors=(anchor,),
                        owner_block_id=item.source_block_id,
                        kind="vocabulary",
                        priority="useful",
                        help_kind="lexical_hint",
                        required_points=point,
                    )
                )
            else:
                anchor = SourceAnchor(
                    block_id=item.source_block_id,
                    start=s_start,
                    end=s_end,
                    quote=quote,
                )
                appended.append(
                    Difficulty(
                        id=diff_id,
                        anchors=(anchor,),
                        owner_block_id=item.source_block_id,
                        kind="vocabulary",
                        priority="useful",
                        help_kind="lexical_hint",
                        required_points=point,
                    )
                )
            existing_ids.add(diff_id)
            occupied.add((s_start, s_end))
            added_for_block += 1
    if not appended:
        return tuple(difficulties)
    return (*difficulties, *appended)


def normalize_pedagogical_response(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
    blocks: Sequence[DocumentBlock],
    block_ids_by_alias: Mapping[str, str],
    *,
    reader_level: str | None = None,
    english_help: bool = False,
    known_names: Sequence[str] = (),
) -> ChapterAnalysisResponse | PedagogicalAnalysisResponse:
    """Unified 5-step normalization pipeline for pedagogical/chapter analysis responses.

    1. `normalize_difficulty_kinds` (all reader levels)
    2. A2 treatment promotion/demotion (only `reader_level == "a2"` and `not english_help`)
    3. `unify_verse_stanza_treatments` (all reader levels except `english_help`)
    4. Switch `help_kind` on promoted/demoted model difficulties + A2 forward pruning
    5. Materialize `auto-vocab` difficulties strictly on final `GLOSSARY_ONLY` spans (A2)
    """
    is_a2 = (reader_level or "").strip().lower() == "a2" and not english_help

    # Step 1: Fix provably contradictory difficulty kinds before A2 rules inspect kind.
    payload = normalize_difficulty_kinds(payload, blocks, block_ids_by_alias)
    if is_a2:
        payload = _drop_unevidenced_before_you_read(payload)

    blocks_by_id = {block.id: block for block in blocks}
    difficulties = _response_difficulties(payload)
    covered_points = _covered_byr_points(payload)
    original_items = tuple(payload.treatment_plan)
    items: list[TreatmentPlanItemResponse] = list(original_items)
    demoted_blocks: list[frozenset[str]] = []
    was_demoted_flags: list[bool] = [False] * len(original_items)
    meaningful_heading_ids: set[str] = set()
    numbered_heading_ids: set[str] = set()
    stoplist_spans_by_block: dict[str, tuple[tuple[int, int], ...]] = {}
    changed = False

    # Step 2: A2 dialogue & stoplist normalization (collecting candidate spans without materializing yet).
    if is_a2:
        for idx, item in enumerate(original_items):
            block = _block_for_plan_item(item.source_block_id, blocks_by_id, block_ids_by_alias)
            if block is None:
                continue
            updated = item
            demoted = None
            owners = frozenset(part for part in (item.source_block_id, block.id) if part)
            if block.type is BlockType.HEADING and _is_meaningful_chapter_title(block.text):
                updated = _force_block_treatment(updated, BlockTreatment.TRANSLATE)
                meaningful_heading_ids.update(owners)
                demoted = None
            elif block.type is BlockType.HEADING:
                updated = _force_block_treatment(updated, BlockTreatment.READ_AS_IS)
                meaningful_heading_ids.update(owners)
                numbered_heading_ids.update(owners)
                demoted = None
            else:
                if block.type in _DIALOGUE_BLOCK_TYPES:
                    updated, _ = _promote_dialogue_item(item, block, difficulties)
                was_demoted = False
                if block.type in _DEMOTABLE_BLOCK_TYPES:
                    after_demote, demoted = _demote_elementary_reply(
                        item, updated, block, difficulties, known_names=known_names
                    )
                    was_demoted = after_demote is not updated
                    updated = after_demote
                if not was_demoted and block.type in _DIALOGUE_BLOCK_TYPES:
                    updated, stoplist_demoted, spans = _promote_a2_stoplist_item(
                        updated, block, difficulties, owners, known_names=known_names
                    )
                    if stoplist_demoted is not None:
                        demoted = stoplist_demoted
                    if spans:
                        stoplist_spans_by_block[block.id] = spans
                was_demoted_flags[idx] = was_demoted
            if updated is not item:
                changed = True
            items[idx] = updated
            if demoted is not None:
                demoted_blocks.append(demoted)

    # Step 3: Unify verse/poem stanzas and homogeneous lists across all reader levels (except english_help).
    if not english_help:
        unified_items, verse_changed = unify_verse_stanza_treatments(
            items, blocks, block_ids_by_alias, english_help=english_help,
        )
        if verse_changed:
            items = list(unified_items)
            changed = True
        list_unified_items, list_changed = unify_homogeneous_list_treatments(
            items, blocks, block_ids_by_alias, english_help=english_help,
        )
        if list_changed:
            items = list(list_unified_items)
            changed = True
        if verse_changed or list_changed:
            # Prune stale demotion state for any block lifted to full TRANSLATE by Step 3 unification.
            promoted_full_translate_owners: set[str] = set()
            for idx, upd in enumerate(items):
                all_translate = upd.treatment is BlockTreatment.TRANSLATE and (
                    not upd.segments
                    or all(seg.treatment is BlockTreatment.TRANSLATE for seg in upd.segments)
                )
                if all_translate:
                    was_demoted_flags[idx] = False
                    block = _block_for_plan_item(upd.source_block_id, blocks_by_id, block_ids_by_alias)
                    promoted_full_translate_owners.add(upd.source_block_id)
                    if block is not None:
                        promoted_full_translate_owners.add(block.id)
            if promoted_full_translate_owners and demoted_blocks:
                demoted_blocks = [
                    owners for owners in demoted_blocks
                    if not (owners & promoted_full_translate_owners)
                ]

    # Step 4: Compute promoted blocks (from original -> post-step-3 items) and update model difficulties.
    promoted_blocks: list[_PromotedBlock] = []
    translate_ids: set[str] = set()
    for idx, (orig, upd) in enumerate(zip(original_items, items, strict=True)):
        block = _block_for_plan_item(orig.source_block_id, blocks_by_id, block_ids_by_alias)
        if block is None:
            continue
        owners = frozenset(part for part in (orig.source_block_id, block.id) if part)
        if upd.treatment is BlockTreatment.TRANSLATE:
            translate_ids.update(owners)
        if not was_demoted_flags[idx]:
            promoted = _compute_promoted_block(orig, upd, block, owners)
            if promoted is not None and not any(owners == d for d in demoted_blocks):
                promoted_blocks.append(promoted)

    pruned_removed_points: set[tuple[str, str]] = set()
    point_id_remaps: dict[tuple[str, str], str] = {}
    if is_a2:
        switched_list = []
        for difficulty in difficulties:
            after_promo = _switch_promoted_hint(difficulty, promoted_blocks, covered_points)
            if after_promo is None or (
                after_promo is not difficulty and _is_auto_vocab_difficulty(difficulty)
            ):
                # Never keep or promote a vocabulary lexical_hint on a promoted TRANSLATE block/segment.
                for pt in difficulty.required_points:
                    pruned_removed_points.add((difficulty.id, pt.id))
                continue
            switched_list.append(
                _prune_forward_difficulty_references(
                    _switch_demoted_hint(after_promo, demoted_blocks),
                    blocks,
                    block_ids_by_alias,
                    point_id_remaps=point_id_remaps,
                    removed_points=pruned_removed_points,
                )
            )
        switched = tuple(switched_list)
        kept = tuple(
            difficulty for difficulty in switched
            if not _is_unevidenced_trivial_grammar(
                difficulty,
                meaningful_heading_ids,
                translate_ids,
                covered_points,
                numbered_heading_ids,
            )
        )
    else:
        kept_list = []
        for difficulty in difficulties:
            after_promo = _switch_promoted_hint(difficulty, promoted_blocks, covered_points)
            if after_promo is None or (
                after_promo is not difficulty and _is_auto_vocab_difficulty(difficulty)
            ):
                for pt in difficulty.required_points:
                    pruned_removed_points.add((difficulty.id, pt.id))
                continue
            kept_list.append(after_promo)
        kept = tuple(kept_list)

    # Step 5: Materialize auto-vocab difficulties strictly for final A2 GLOSSARY_ONLY blocks/segments.
    if is_a2:
        final_difficulties = _materialize_a2_glossary_difficulties(
            payload,
            items,
            kept,
            blocks_by_id,
            block_ids_by_alias,
            stoplist_spans_by_block,
            known_names=known_names,
        )
    else:
        final_difficulties = kept

    if not changed and final_difficulties == tuple(difficulties):
        return payload
    plan = tuple(items)
    remapped_difficulty_ids = {diff_id for diff_id, _ in point_id_remaps}
    removed_points = pruned_removed_points | _removed_point_ids(
        tuple(d for d in difficulties if d.id not in remapped_difficulty_ids),
        final_difficulties,
    )
    return _replace_payload_difficulties(
        payload,
        final_difficulties,
        treatment_plan=plan,
        before_you_read_evidence=_evidence_without_removed_points(
            _before_you_read_evidence(payload),
            removed_points,
            point_id_remaps,
        ),
    )


def normalize_a2_dialogue_response(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
    blocks: Sequence[DocumentBlock],
    block_ids_by_alias: Mapping[str, str],
    *,
    reader_level: str | None = None,
    english_help: bool = False,
    known_names: Sequence[str] = (),
) -> ChapterAnalysisResponse | PedagogicalAnalysisResponse:
    """Backward-compatible wrapper delegating to `normalize_pedagogical_response`."""
    if reader_level != "a2" or english_help:
        return payload
    return normalize_pedagogical_response(
        payload,
        blocks,
        block_ids_by_alias,
        reader_level=reader_level,
        english_help=english_help,
        known_names=known_names,
    )


def _response_difficulties(payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse):
    if isinstance(payload, PedagogicalAnalysisResponse):
        return payload.difficulties
    if payload.content_plan is None:
        return ()
    return payload.content_plan.difficulties


def _removed_point_ids(before, after) -> set[tuple[str, str]]:
    """Points dropped from a difficulty that itself stays."""
    kept_ids = {
        difficulty.id: {point.id for point in difficulty.required_points}
        for difficulty in after
    }
    removed = set()
    for difficulty in before:
        surviving = kept_ids.get(difficulty.id)
        if surviving is None:
            continue
        for point in difficulty.required_points:
            if point.id not in surviving:
                removed.add((difficulty.id, point.id))
    return removed


def _evidence_without_removed_points(
    evidence,
    removed: set[tuple[str, str]],
    point_id_remaps: Mapping[tuple[str, str], str] | None = None,
):
    if not removed and not point_id_remaps:
        return evidence
    remaps = point_id_remaps or {}
    updated = []
    for item in evidence:
        key = (item.difficulty_id, item.point_id)
        if key in removed:
            continue
        new_point_id = remaps.get(key)
        updated.append(replace(item, point_id=new_point_id) if new_point_id else item)
    return tuple(updated)


def _before_you_read_evidence(payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse):
    if isinstance(payload, PedagogicalAnalysisResponse):
        return payload.before_you_read_evidence
    if payload.content_plan is None:
        return ()
    return payload.content_plan.before_you_read_evidence


def _drop_unevidenced_before_you_read(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
):
    if _before_you_read_evidence(payload) or not payload.before_you_read:
        return payload
    return replace(payload, before_you_read=())


def _covered_byr_points(
    payload: ChapterAnalysisResponse | PedagogicalAnalysisResponse,
) -> dict[str, set[str]]:
    covered: dict[str, set[str]] = {}
    for item in _before_you_read_evidence(payload):
        covered.setdefault(item.difficulty_id, set()).add(item.point_id)
    return covered


def _block_for_plan_item(source_block_id, blocks_by_id, block_ids_by_alias):
    block_id = block_ids_by_alias.get(source_block_id)
    if block_id is None and source_block_id in blocks_by_id:
        block_id = source_block_id
    if block_id is None:
        return None
    return blocks_by_id.get(block_id)


def _sentence_segments_to_promote(segments, spans, difficulties, owners, block_text) -> set[int]:
    """Indexes to lift, including every light span of a sentence that needs it.

    A long sentence is stored as several spans with one `sentence_index`.
    Evaluating the full sentence slice and promoting every span of that
    sentence keeps split quotes intact and preserves uniform sentence modes.
    """
    sentence_bounds: dict[int, tuple[int, int]] = {}
    for span in spans:
        bounds = sentence_bounds.get(span.sentence_index)
        if bounds is None:
            sentence_bounds[span.sentence_index] = (span.start, span.end)
        else:
            sentence_bounds[span.sentence_index] = (
                min(bounds[0], span.start),
                max(bounds[1], span.end),
            )
    promoted_sentences: set[int] = set()
    checked_sentences: set[int] = set()
    for segment in segments:
        if segment.index < 1 or segment.index > len(spans):
            continue
        if segment.treatment not in _LIGHT_DIALOGUE_TREATMENTS:
            continue
        sentence_index = spans[segment.index - 1].sentence_index
        if sentence_index in checked_sentences:
            continue
        checked_sentences.add(sentence_index)
        sentence_start, sentence_end = sentence_bounds[sentence_index]
        text = block_text[sentence_start:sentence_end]
        if _dialogue_needs_translation(
            text,
            difficulties,
            owners,
            sentence_start,
            is_glossary_only=(segment.treatment is BlockTreatment.GLOSSARY_ONLY),
        ):
            promoted_sentences.add(sentence_index)
    if not promoted_sentences:
        return set()
    return {
        segment.index
        for segment in segments
        if 1 <= segment.index <= len(spans)
        and segment.treatment in _LIGHT_DIALOGUE_TREATMENTS
        and spans[segment.index - 1].sentence_index in promoted_sentences
    }


def _promote_dialogue_item(item, block, difficulties):
    owners = frozenset(part for part in (item.source_block_id, block.id) if part)
    if item.segments:
        spans = block.segment_spans
        promote = _sentence_segments_to_promote(item.segments, spans, difficulties, owners, block.text)
        if not promote:
            return item, None
        had_glossary = item.treatment is BlockTreatment.GLOSSARY_ONLY or any(
            segment.treatment is BlockTreatment.GLOSSARY_ONLY for segment in item.segments
        )
        updated = [
            SegmentTreatmentResponse(segment.index, BlockTreatment.TRANSLATE)
            if segment.index in promote
            else segment
            for segment in item.segments
        ]
        modes = tuple(segment.treatment for segment in updated)
        strongest = strongest_treatment(modes)
        promoted = (owners, had_glossary) if strongest is BlockTreatment.TRANSLATE else None
        return replace(item, treatment=strongest, segments=tuple(updated)), promoted
    if item.treatment not in _LIGHT_DIALOGUE_TREATMENTS:
        return item, None
    had_glossary = item.treatment is BlockTreatment.GLOSSARY_ONLY
    if not _dialogue_needs_translation(
        block.text, difficulties, owners, 0, is_glossary_only=had_glossary
    ) and not (
        had_glossary and _narrative_needs_translation(block, difficulties, owners)
    ):
        return item, None
    return replace(item, treatment=BlockTreatment.TRANSLATE), (owners, had_glossary)


def _narrative_needs_translation(
    block: DocumentBlock,
    difficulties: Sequence[Any] = (),
    owners: frozenset[str] = frozenset(),
) -> bool:
    if len(expression_words(block.text)) > _NARRATIVE_GLOSSARY_WORD_LIMIT:
        return True
    sentences = {span.sentence_index for span in block.segment_spans}
    if len(sentences) > _NARRATIVE_GLOSSARY_SENTENCE_LIMIT:
        return True
    if owners:
        owned_count = sum(
            1 for difficulty in difficulties if difficulty.owner_block_id in owners
        )
        if owned_count > _NARRATIVE_GLOSSARY_DIFFICULTY_LIMIT:
            return True
    return False


def _dialogue_needs_translation(
    text,
    difficulties,
    owners,
    span_start,
    *,
    is_glossary_only: bool = False,
) -> bool:
    quotes = list(_DIALOGUE_QUOTES.finditer(text))
    if not quotes:
        return False
    all_words = expression_words(text)
    has_dialect_or_idiom = (
        any(word in _DIALECT_WORDS for word in all_words)
        or any(_contains_phrase(all_words, phrase) for phrase in _SHORT_B1_PHRASES)
        or any(
            difficulty.kind in {"speech_style", "idiom"}
            and difficulty.owner_block_id in owners
            for difficulty in difficulties
        )
    )
    long_reply = len(all_words) > _DIALOGUE_GLOSSARY_WORD_LIMIT or (
        has_dialect_or_idiom and len(all_words) > 8
    )
    quoted = " ".join(match.group(0)[1:-1] for match in quotes)
    quoted_words = expression_words(quoted)
    if any(word in _SUBORDINATORS for word in quoted_words):
        return True
    if _QUESTION_TAG_RE.search(quoted) is not None:
        return True
    if len(quoted_words) >= 5 and any(sep in quoted for sep in (" \u2013 ", " \u2014 ")):
        return True
    ranges = _remark_ranges(quotes, len(text))
    remark = " ".join(text[start:end] for start, end in ranges)
    nonbasic_words = set(expression_words(remark)) & (_NONBASIC_REPORTING - _BASIC_REPORTING)
    single_covered_nonbasic = (
        is_glossary_only
        and len(nonbasic_words) == 1
        and _difficulty_covers_remark_word(
            difficulties, owners, next(iter(nonbasic_words)), text, span_start, ranges
        )
    )
    has_uncovered_nonbasic = bool(nonbasic_words) and not single_covered_nonbasic
    return long_reply or has_uncovered_nonbasic


def _difficulty_covers_remark_word(
    difficulties, owners, word: str, text: str, span_start: int, ranges
) -> bool:
    for difficulty in difficulties:
        if difficulty.owner_block_id not in owners:
            continue
        for anchor in difficulty.anchors:
            if getattr(anchor, "block_id", None) not in owners:
                continue
            if not _anchor_hits_ranges(anchor, text, span_start, ranges):
                continue
            anchor_words = expression_words(getattr(anchor, "quote", ""))
            if any(anchor_word == word or _same_word(anchor_word, word) for anchor_word in anchor_words):
                return True
    return False


def _remark_ranges(quotes, length) -> tuple[tuple[int, int], ...]:
    ranges = []
    cursor = 0
    for match in quotes:
        if match.start() > cursor:
            ranges.append((cursor, match.start()))
        cursor = match.end()
    if cursor < length:
        ranges.append((cursor, length))
    return tuple(ranges)


def _anchor_hits_ranges(anchor, text, span_start, ranges) -> bool:
    start = getattr(anchor, "start", None)
    end = getattr(anchor, "end", None)
    if type(start) is int and type(end) is int:
        local_start = start - span_start
        local_end = end - span_start
        return any(local_start < stop and local_end > begin for begin, stop in ranges)
    quote = getattr(anchor, "quote", "")
    if not quote:
        return False
    spans = _quote_spans(text, quote)
    cleaned = without_footnote_markers(quote).strip()
    if not spans and cleaned and cleaned != quote:
        spans = _quote_spans(text, cleaned)
    if spans:
        return any(
            found_start < stop and found_end > begin
            for found_start, found_end, _ in spans
            for begin, stop in ranges
        )
    target = cleaned or quote
    return any(
        bool(_quote_spans(without_footnote_markers(text[begin:stop]), target))
        for begin, stop in ranges
    )


def _demote_elementary_reply(
    original,
    updated,
    block,
    difficulties,
    *,
    known_names: Sequence[str] = (),
):
    """Lower a model TRANSLATE on a short elementary reply.

    Returns the item and the block owners when vocabulary hints must become
    lexical hints. Owners stay None when the mode is unchanged or there is
    nothing to retarget. Segment modes drop with the block.
    """
    if original.treatment is not BlockTreatment.TRANSLATE and not any(
        segment.treatment is BlockTreatment.TRANSLATE for segment in original.segments
    ):
        return updated, None
    owners = frozenset(part for part in (original.source_block_id, block.id) if part)
    sentence_count = len({span.sentence_index for span in block.segment_spans})
    mode = _elementary_reply_mode(
        block.text,
        difficulties,
        owners,
        sentence_count=sentence_count,
        known_names=known_names,
    )
    if mode is None:
        return updated, None
    segments = tuple(SegmentTreatmentResponse(segment.index, mode) for segment in updated.segments)
    if updated.treatment is mode and segments == tuple(updated.segments):
        return updated, None
    demoted = replace(updated, treatment=mode, segments=segments) if segments else replace(updated, treatment=mode)
    if mode is not BlockTreatment.GLOSSARY_ONLY:
        return demoted, None
    return demoted, owners


def _elementary_reply_mode(
    text,
    difficulties,
    owners,
    *,
    sentence_count: int = 1,
    known_names: Sequence[str] = (),
):
    words = expression_words(text)
    if not words:
        return None
    if _has_a2_nonlexical_stoplist(text):
        return None
    if any(word in _SHORT_B1_WORDS for word in words) or any(
        _contains_phrase(words, phrase) for phrase in _SHORT_B1_PHRASES
    ) or _has_b1_phrasal_verb(text):
        return None
    if len(words) > 8:
        return None
    quotes = list(_DIALOGUE_QUOTES.finditer(text))
    if not quotes:
        if sentence_count > 1:
            return None
        if any(word in _SUBORDINATORS for word in words):
            return None
        if _QUESTION_TAG_RE.search(text) is not None:
            return None
        if len(words) >= 5 and any(sep in text for sep in (" \u2013 ", " \u2014 ")):
            return None
        if set(words) & (_NONBASIC_REPORTING - _BASIC_REPORTING):
            return None
    else:
        if len(quotes) > 2:
            return None
        quoted = " ".join(match.group(0)[1:-1] for match in quotes)
        quoted_words = expression_words(quoted)
        if any(word in _SUBORDINATORS for word in quoted_words):
            return None
        if _QUESTION_TAG_RE.search(quoted) is not None:
            return None
        if len(quoted_words) >= 5 and any(sep in quoted for sep in (" \u2013 ", " \u2014 ")):
            return None
        ranges = _remark_ranges(quotes, len(text))
        remark_words = set(expression_words(" ".join(text[start:end] for start, end in ranges)))
        if remark_words & (_NONBASIC_REPORTING - _BASIC_REPORTING):
            return None
        if not (remark_words & _BASIC_REPORTING):
            return None
    related = tuple(
        difficulty for difficulty in difficulties
        if difficulty.owner_block_id in owners or any(
            getattr(anchor, "block_id", None) in owners for anchor in difficulty.anchors
        )
    )
    if any(difficulty.kind != "vocabulary" for difficulty in related):
        return None
    if any(difficulty.owner_block_id in owners for difficulty in related):
        return BlockTreatment.GLOSSARY_ONLY
    if dominant_script(text) == "LATIN" and non_core_a2_words(text, known_names=known_names):
        return None
    return BlockTreatment.READ_AS_IS


def _switch_demoted_hint(difficulty, demoted_blocks: Sequence[frozenset[str]]):
    if not demoted_blocks or difficulty.kind != "vocabulary" or difficulty.help_kind == "lexical_hint":
        return difficulty
    if any(difficulty.owner_block_id in owners for owners in demoted_blocks):
        return replace(difficulty, help_kind="lexical_hint")
    return difficulty


def _difficulty_hits_ranges(
    difficulty,
    owners: frozenset[str],
    block_text: str,
    ranges: tuple[tuple[int, int], ...],
) -> bool:
    for anchor in getattr(difficulty, "anchors", ()):
        if getattr(anchor, "block_id", None) not in owners:
            continue
        if _anchor_hits_ranges(anchor, block_text, 0, ranges):
            return True
    return False


def _switch_promoted_hint(
    difficulty,
    promoted_blocks: Sequence[_PromotedBlock],
    covered_points: Mapping[str, set[str]],
):
    if difficulty.help_kind != "lexical_hint":
        return difficulty
    required = {point.id for point in difficulty.required_points}
    byr_covered = bool(required) and required <= covered_points.get(difficulty.id, set())
    for promoted in promoted_blocks:
        owners = promoted.owners
        if difficulty.owner_block_id not in owners and not any(
            getattr(anchor, "block_id", None) in owners for anchor in difficulty.anchors
        ):
            continue
        if promoted.promoted_ranges is not None and not _difficulty_hits_ranges(
            difficulty, owners, promoted.block_text, promoted.promoted_ranges
        ):
            continue
        if byr_covered and not promoted.had_glossary:
            continue
        if difficulty.kind == "vocabulary":
            return None
        return replace(difficulty, help_kind="explanation")
    return difficulty


def _sentence_bounds_by_index(block: DocumentBlock) -> dict[int, tuple[int, int]]:
    bounds: dict[int, tuple[int, int]] = {}
    for span in block.segment_spans:
        prev = bounds.get(span.sentence_index)
        if prev is None:
            bounds[span.sentence_index] = (span.start, span.end)
        else:
            bounds[span.sentence_index] = (min(prev[0], span.start), max(prev[1], span.end))
    return bounds


def _owns_vocabulary_in_range(
    difficulties,
    owners: frozenset[str],
    block_text: str,
    span_start: int,
    span_end: int,
    *,
    whole_block: bool,
) -> bool:
    if whole_block:
        return _owns_vocabulary(difficulties, owners)
    ranges = ((span_start, span_end),)
    return any(
        difficulty.kind == "vocabulary"
        and difficulty.owner_block_id in owners
        and _difficulty_hits_ranges(difficulty, owners, block_text, ranges)
        for difficulty in difficulties
    )


def _has_nonlexical_backing_difficulty(
    difficulties,
    owners: frozenset[str],
    block_text: str,
    span_start: int,
    span_end: int,
    *,
    whole_block: bool,
) -> bool:
    ranges = ((span_start, span_end),)
    for difficulty in difficulties:
        if difficulty.owner_block_id not in owners:
            continue
        if difficulty.kind not in _A2_NONLEXICAL_BACKING_KINDS:
            continue
        if whole_block or _difficulty_hits_ranges(difficulty, owners, block_text, ranges):
            return True
    return False


def _anchor_covers_a2_word(anchor_quote: str, word: str) -> bool:
    norm_quote = anchor_quote.translate(_QUOTE_APOSTROPHE_TRANS).casefold()
    if not norm_quote:
        return False
    if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", norm_quote) is not None:
        return True
    anchor_words = expression_words(anchor_quote)
    return any(aw == word or _same_word(aw, word) for aw in anchor_words)


def _uncovered_a2_words(
    unknown: Sequence[str],
    difficulties,
    owners: frozenset[str],
    block_text: str,
    span_start: int,
    span_end: int,
    *,
    whole_block: bool,
) -> tuple[str, ...]:
    if not unknown:
        return ()
    ranges = ((span_start, span_end),)
    owner_quotes: list[str] = []
    for difficulty in difficulties:
        if difficulty.owner_block_id not in owners:
            continue
        for anchor in getattr(difficulty, "anchors", ()):
            if getattr(anchor, "block_id", None) not in owners:
                continue
            if not whole_block and not _anchor_hits_ranges(anchor, block_text, 0, ranges):
                continue
            quote = getattr(anchor, "quote", "")
            if quote:
                owner_quotes.append(quote)
    if not owner_quotes:
        return tuple(unknown)
    return tuple(
        word
        for word in unknown
        if not any(_anchor_covers_a2_word(quote, word) for quote in owner_quotes)
    )


def _single_b1_reporting_verb_span(
    text: str,
    span_start: int,
    difficulties,
    owners: frozenset[str],
    block_text: str,
    span_end: int,
    *,
    whole_block: bool,
) -> tuple[tuple[int, int], ...]:
    for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text):
        token = match.group(0).casefold()
        if token in _SINGLE_B1_REPORTING_VERBS:
            if _uncovered_a2_words(
                (token,),
                difficulties,
                owners,
                block_text,
                span_start,
                span_end,
                whole_block=whole_block,
            ):
                return ((span_start + match.start(), span_start + match.end()),)
            return ()
    return ()


def _read_as_is_stoplist_mode(
    text: str,
    block_text: str,
    span_start: int,
    span_end: int,
    difficulties,
    owners: frozenset[str],
    *,
    whole_block: bool,
    known_names: Sequence[str] = (),
) -> _StoplistDecision | None:
    if _has_a2_nonlexical_stoplist(text):
        return _StoplistDecision(BlockTreatment.TRANSLATE)
    words = expression_words(text)
    if not words:
        return None
    if len(words) <= _SHORT_B1_WORD_LIMIT and (
        any(word in _SHORT_B1_WORDS for word in words)
        or any(_contains_phrase(words, phrase) for phrase in _SHORT_B1_PHRASES)
    ):
        if _owns_vocabulary_in_range(
            difficulties, owners, block_text, span_start, span_end, whole_block=whole_block
        ):
            return _StoplistDecision(BlockTreatment.GLOSSARY_ONLY)
        if _is_single_b1_reporting_verb_only(words):
            return _StoplistDecision(
                BlockTreatment.GLOSSARY_ONLY,
                _single_b1_reporting_verb_span(
                    text,
                    span_start,
                    difficulties,
                    owners,
                    block_text,
                    span_end,
                    whole_block=whole_block,
                ),
            )
        return _StoplistDecision(BlockTreatment.TRANSLATE)
    phrasal_match = _A2_B1_PHRASAL_VERB_RE.search(text)
    if phrasal_match is not None:
        phrase = phrasal_match.group(0)
        spans: tuple[tuple[int, int], ...] = ()
        if _uncovered_a2_words(
            (phrase.casefold(),),
            difficulties,
            owners,
            block_text,
            span_start,
            span_end,
            whole_block=whole_block,
        ):
            spans = ((span_start + phrasal_match.start(), span_start + phrasal_match.end()),)
        return _StoplistDecision(BlockTreatment.GLOSSARY_ONLY, spans)
    if dominant_script(text) == "LATIN":
        word_spans = non_core_a2_word_spans(text, known_names=known_names)
        if word_spans:
            uncovered = _uncovered_a2_words(
                tuple(word for word, _, _ in word_spans),
                difficulties,
                owners,
                block_text,
                span_start,
                span_end,
                whole_block=whole_block,
            )
            if uncovered:
                if len(uncovered) > 3:
                    return _StoplistDecision(BlockTreatment.TRANSLATE)
                uncovered_set = set(uncovered)
                target_spans = tuple(
                    (span_start + local_s, span_start + local_e)
                    for word, local_s, local_e in word_spans
                    if word in uncovered_set
                )
                return _StoplistDecision(BlockTreatment.GLOSSARY_ONLY, target_spans)
    return None


def _glossary_only_needs_stoplist_translation(
    text: str,
    block_text: str,
    span_start: int,
    span_end: int,
    difficulties,
    owners: frozenset[str],
    *,
    whole_block: bool,
) -> bool:
    if not _has_a2_nonlexical_stoplist(text):
        return False
    return not _has_nonlexical_backing_difficulty(
        difficulties, owners, block_text, span_start, span_end, whole_block=whole_block
    )


def _promote_a2_stoplist_item(
    item: TreatmentPlanItemResponse,
    block: DocumentBlock,
    difficulties,
    owners: frozenset[str],
    *,
    known_names: Sequence[str] = (),
) -> tuple[TreatmentPlanItemResponse, frozenset[str] | None, tuple[tuple[int, int], ...]]:
    if item.segments:
        spans = block.segment_spans
        sentence_bounds = _sentence_bounds_by_index(block)
        sentence_segments: dict[int, list[SegmentTreatmentResponse]] = {}
        for segment in item.segments:
            if 1 <= segment.index <= len(spans):
                sentence_segments.setdefault(
                    spans[segment.index - 1].sentence_index, []
                ).append(segment)
        lifted_sentences: dict[int, BlockTreatment] = {}
        collected_spans: list[tuple[int, int]] = []
        for sentence_index, segs in sentence_segments.items():
            modes = tuple(seg.treatment for seg in segs)
            if not modes:
                continue
            current_mode = strongest_treatment(modes)
            if current_mode not in _LIGHT_DIALOGUE_TREATMENTS:
                continue
            s_start, s_end = sentence_bounds[sentence_index]
            s_text = block.text[s_start:s_end]
            target: BlockTreatment | None = None
            target_spans: tuple[tuple[int, int], ...] = ()
            if current_mode is BlockTreatment.READ_AS_IS:
                decision = _read_as_is_stoplist_mode(
                    s_text,
                    block.text,
                    s_start,
                    s_end,
                    difficulties,
                    owners,
                    whole_block=(len(sentence_segments) == 1),
                    known_names=known_names,
                )
                if decision is not None:
                    target = decision.mode
                    target_spans = decision.target_spans
            elif current_mode is BlockTreatment.GLOSSARY_ONLY:
                if _glossary_only_needs_stoplist_translation(
                    s_text,
                    block.text,
                    s_start,
                    s_end,
                    difficulties,
                    owners,
                    whole_block=(len(sentence_segments) == 1),
                ):
                    target = BlockTreatment.TRANSLATE
            if target is not None and target is not current_mode:
                lifted_sentences[sentence_index] = target
                if target is BlockTreatment.GLOSSARY_ONLY and target_spans:
                    collected_spans.extend(target_spans)
        if not lifted_sentences:
            return item, None, ()
        new_segments = tuple(
            SegmentTreatmentResponse(
                segment.index,
                lifted_sentences[spans[segment.index - 1].sentence_index],
            )
            if 1 <= segment.index <= len(spans)
            and spans[segment.index - 1].sentence_index in lifted_sentences
            else segment
            for segment in item.segments
        )
        strongest = strongest_treatment(segment.treatment for segment in new_segments)
        updated = replace(item, treatment=strongest, segments=new_segments)
        demoted = (
            owners
            if item.treatment is BlockTreatment.READ_AS_IS
            and strongest is BlockTreatment.GLOSSARY_ONLY
            and len(sentence_segments) == 1
            else None
        )
        return updated, demoted, tuple(collected_spans)

    if item.treatment is BlockTreatment.READ_AS_IS:
        sentence_bounds = _sentence_bounds_by_index(block)
        slices = (
            tuple(sentence_bounds.values())
            if len(sentence_bounds) > 1
            else ((0, len(block.text)),)
        )
        triggered: list[BlockTreatment] = []
        collected_spans: list[tuple[int, int]] = []
        if _has_a2_nonlexical_stoplist(block.text):
            triggered.append(BlockTreatment.TRANSLATE)
        else:
            for s_start, s_end in slices:
                decision = _read_as_is_stoplist_mode(
                    block.text[s_start:s_end],
                    block.text,
                    s_start,
                    s_end,
                    difficulties,
                    owners,
                    whole_block=(len(slices) == 1),
                    known_names=known_names,
                )
                if decision is not None:
                    triggered.append(decision.mode)
                    if decision.mode is BlockTreatment.GLOSSARY_ONLY and decision.target_spans:
                        collected_spans.extend(decision.target_spans)
        if not triggered:
            return item, None, ()
        mode = strongest_treatment(triggered)
        updated = _force_block_treatment(item, mode)
        demoted = owners if mode is BlockTreatment.GLOSSARY_ONLY else None
        final_spans = tuple(collected_spans) if mode is BlockTreatment.GLOSSARY_ONLY else ()
        return updated, demoted, final_spans

    if item.treatment is BlockTreatment.GLOSSARY_ONLY:
        if _glossary_only_needs_stoplist_translation(
            block.text,
            block.text,
            0,
            len(block.text),
            difficulties,
            owners,
            whole_block=True,
        ):
            return _force_block_treatment(item, BlockTreatment.TRANSLATE), None, ()

    return item, None, ()


def _compute_promoted_block(
    original: TreatmentPlanItemResponse,
    updated: TreatmentPlanItemResponse,
    block: DocumentBlock,
    owners: frozenset[str],
) -> _PromotedBlock | None:
    if updated.treatment is not BlockTreatment.TRANSLATE:
        return None
    if updated.segments:
        orig_by_index = {segment.index: segment.treatment for segment in original.segments}
        all_updated_translate = all(
            segment.treatment is BlockTreatment.TRANSLATE for segment in updated.segments
        )
        if original.treatment is not BlockTreatment.TRANSLATE and all_updated_translate:
            had_glossary = original.treatment is BlockTreatment.GLOSSARY_ONLY or any(
                segment.treatment is BlockTreatment.GLOSSARY_ONLY
                for segment in original.segments
            )
            return _PromotedBlock(
                owners=owners,
                had_glossary=had_glossary,
                block_text=block.text,
                promoted_ranges=None,
            )
        spans = block.segment_spans
        promoted_ranges: list[tuple[int, int]] = []
        had_glossary = False
        for segment in updated.segments:
            prev = orig_by_index.get(segment.index, original.treatment)
            if prev is not BlockTreatment.TRANSLATE and segment.treatment is BlockTreatment.TRANSLATE:
                if prev is BlockTreatment.GLOSSARY_ONLY:
                    had_glossary = True
                if 1 <= segment.index <= len(spans):
                    span = spans[segment.index - 1]
                    promoted_ranges.append((span.start, span.end))
        if not promoted_ranges:
            return None
        return _PromotedBlock(
            owners=owners,
            had_glossary=had_glossary,
            block_text=block.text,
            promoted_ranges=tuple(promoted_ranges),
        )
    if original.treatment is not BlockTreatment.TRANSLATE:
        had_glossary = original.treatment is BlockTreatment.GLOSSARY_ONLY or any(
            segment.treatment is BlockTreatment.GLOSSARY_ONLY
            for segment in original.segments
        )
        return _PromotedBlock(
            owners=owners,
            had_glossary=had_glossary,
            block_text=block.text,
            promoted_ranges=None,
        )
    return None


def _is_meaningful_chapter_title(text: str) -> bool:
    words = expression_words(text)
    if not words:
        return False
    if len(words) == 1 and _ROMAN_NUMERAL.fullmatch(words[0]):
        return False
    if len(words) >= 2 and words[0] == "chapter" and all(
        word.isdigit() or word in _CHAPTER_NUMBER_WORDS or _ROMAN_NUMERAL.fullmatch(word)
        for word in words[1:]
    ):
        return False
    return True


def _is_single_b1_reporting_verb_only(words: tuple[str, ...]) -> bool:
    if any(_contains_phrase(words, phrase) for phrase in _SHORT_B1_PHRASES):
        return False
    b1_hits = {word for word in words if word in _SHORT_B1_WORDS}
    return len(b1_hits) == 1 and b1_hits <= _SINGLE_B1_REPORTING_VERBS


def _contains_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(words[index:index + size] == phrase for index in range(len(words) - size + 1))


def _owns_vocabulary(difficulties, owners: frozenset[str]) -> bool:
    return any(
        difficulty.kind == "vocabulary" and difficulty.owner_block_id in owners
        for difficulty in difficulties
    )


def _force_block_treatment(item, mode):
    segments = tuple(
        segment if segment.treatment is mode else SegmentTreatmentResponse(segment.index, mode)
        for segment in item.segments
    )
    if item.treatment is mode and segments == item.segments:
        return item
    if segments:
        return replace(item, treatment=mode, segments=segments)
    return replace(item, treatment=mode)


def _is_unevidenced_trivial_grammar(
    difficulty,
    heading_ids,
    translate_ids,
    covered_points,
    numbered_heading_ids: set[str] | frozenset[str] = frozenset(),
) -> bool:
    if difficulty.id in covered_points:
        return False
    owner = difficulty.owner_block_id
    if owner in numbered_heading_ids and (
        difficulty.kind in {"grammar", "vocabulary"}
        or getattr(difficulty, "help_kind", None) == "lexical_hint"
    ):
        return True
    if difficulty.kind == "grammar" and owner in heading_ids:
        return True
    if owner not in translate_ids:
        return False
    anchors = tuple(expression_words(getattr(anchor, "quote", "")) for anchor in difficulty.anchors)
    if difficulty.kind == "grammar" and any(words == ("sounding", "relieved") for words in anchors):
        return True
    return (
        difficulty.kind in {"grammar", "vocabulary"}
        and getattr(difficulty, "help_kind", None) == "explanation"
        and bool(anchors)
        and all(words in _TRIVIAL_A2_ANCHORS for words in anchors)
    )


_STANDARD_POINT_ID_RE = re.compile(r"p\d+")


def _prune_forward_difficulty_references(
    difficulty,
    blocks,
    block_ids_by_alias,
    *,
    point_id_remaps: dict[tuple[str, str], str] | None = None,
    removed_points: set[tuple[str, str]] | None = None,
):
    """For A2 difficulties, prune forward anchors and forward-quoting points, keeping >= 1 point."""
    owner_index = _block_order_index(difficulty.owner_block_id, blocks, block_ids_by_alias)
    if owner_index is None:
        return difficulty
    updated = difficulty
    anchors = getattr(difficulty, "anchors", ())
    if anchors:
        kept_anchors = tuple(
            anchor for anchor in anchors
            if (anchor_index := _block_order_index(anchor.block_id, blocks, block_ids_by_alias)) is None
            or anchor_index <= owner_index
        )
        if kept_anchors and len(kept_anchors) != len(anchors):
            updated = replace(updated, anchors=kept_anchors)
    points = updated.required_points
    if len(points) >= 2:
        block_words = tuple(expression_words(block.text) for block in blocks)
        if updated.kind == "speech_style":
            kept_points = tuple(
                point for point in points
                if not _is_forward_speech_point(point.description, block_words, owner_index)
            )
        else:
            kept_points = tuple(
                point for point in points
                if not _quotes_later_block(point.description, block_words, owner_index, min_words=2)
            )
        if len(kept_points) != len(points):
            if not kept_points:
                kept_points = (points[0],)
            kept_original_ids = {point.id for point in kept_points}
            if removed_points is not None:
                for point in points:
                    if point.id not in kept_original_ids:
                        removed_points.add((updated.id, point.id))
            if all(_STANDARD_POINT_ID_RE.fullmatch(point.id) for point in points):
                renumbered: list = []
                for idx, point in enumerate(kept_points, start=1):
                    new_id = f"p{idx}"
                    if point_id_remaps is not None and point.id != new_id:
                        point_id_remaps[(updated.id, point.id)] = new_id
                    renumbered.append(
                        replace(point, id=new_id) if point.id != new_id else point
                    )
                kept_points = tuple(renumbered)
            updated = replace(updated, required_points=kept_points)
    return updated


def _block_order_index(owner_id, blocks, block_ids_by_alias) -> int | None:
    indexes = {block.id: index for index, block in enumerate(blocks)}
    if owner_id in indexes:
        return indexes[owner_id]
    block_id = block_ids_by_alias.get(owner_id)
    if block_id in indexes:
        return indexes[block_id]
    return None


def _is_forward_speech_point(description, block_words, owner_index) -> bool:
    if _quotes_later_block(description, block_words, owner_index, min_words=1):
        return True
    tokens = _only_lemma_tokens(description)
    return tokens is not None and _future_only_tokens(tokens, block_words, owner_index)


def _quotes_later_block(description, block_words, owner_index, *, min_words: int = 1) -> bool:
    earlier = block_words[:owner_index + 1]
    later = block_words[owner_index + 1:]
    for match in _SPEECH_POINT_QUOTES.finditer(description):
        raw = next((group for group in match.groups() if group), "")
        phrase = expression_words(raw)
        if not phrase or len(phrase) < min_words:
            continue
        if len(phrase) == 1 and phrase[0] in _STANDARD_SPEECH_GLOSS_WORDS:
            continue
        if any(_contains_phrase(words, phrase) for words in earlier):
            continue
        if any(_contains_phrase(words, phrase) for words in later):
            return True
    return False


def _only_lemma_tokens(description: str) -> tuple[str, ...] | None:
    """Dialect glosses `<token> =` when the point has no other words."""
    matches = list(_LEMMA_GLOSS.finditer(description))
    if not matches:
        return None
    if expression_words(_LEMMA_GLOSS.sub(" ", description)):
        return None
    tokens = []
    for match in matches:
        words = expression_words(match.group(1))
        if not words:
            return None
        tokens.append(words[0])
    return tuple(tokens) or None


def _future_only_tokens(tokens, block_words, owner_index) -> bool:
    earlier = block_words[:owner_index + 1]
    later = block_words[owner_index + 1:]
    for token in tokens:
        if any(token in words for words in earlier):
            return False
        if not any(token in words for words in later):
            return False
    return True


def _explicit_speaker_if_remark_names(
    fact,
    texts,
    blocks,
    *,
    scenes: Sequence[Scene] = (),
    glossary_candidates: Sequence[object] = (),
    context_facts: Sequence[object] = (),
    mandatory_facts: Mapping[str, Mapping[str, str]] | Sequence[object] | None = (),
    ordered_block_ids: Sequence[str] = (),
):
    """inferred → explicit when the opening remark names the speaker or an unambiguous pronoun."""
    if fact.kind != "speaker" or fact.certainty != "inferred" or not isinstance(fact.value, str):
        return fact
    text = _fact_block_text(fact.subject_anchor.block_id, texts, blocks)
    if not text:
        return fact
    if _opening_remark_names_speaker(text, fact.value):
        return replace(fact, certainty="explicit")
    if scenes and _opening_remark_pronoun_resolves_speaker(
        fact,
        text,
        scenes=scenes,
        glossary_candidates=glossary_candidates,
        context_facts=context_facts,
        mandatory_facts=mandatory_facts,
        ordered_block_ids=ordered_block_ids,
        blocks=blocks,
        texts=texts,
    ):
        return replace(fact, certainty="explicit")
    return fact


def _normalize_gender_value(value: str | None) -> str | None:
    if not value:
        return None
    extracted = _extract_gender(value)
    if extracted is not None:
        return extracted
    lowered = value.strip().casefold()
    if lowered in {"male", "female", "neuter"}:
        return lowered
    return None


def _gender_map_from_mandatory_facts(
    mandatory_facts: Mapping[str, Mapping[str, str]] | Sequence[object] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    if not mandatory_facts:
        return result
    if isinstance(mandatory_facts, Mapping):
        for raw_key, raw_val in mandatory_facts.items():
            if not isinstance(raw_key, str):
                continue
            gender_raw: str | None = None
            if isinstance(raw_val, Mapping):
                gender_raw = raw_val.get("gender")
            elif isinstance(raw_val, str):
                gender_raw = raw_val
            gender = _normalize_gender_value(gender_raw)
            if gender is not None:
                key = glossary_key(raw_key)
                if key:
                    result[key] = gender
        return result
    for item in mandatory_facts:
        source = getattr(item, "canonical_source_text", None) or getattr(item, "source_text", None)
        if not isinstance(source, str) or not source.strip():
            continue
        gender = _normalize_gender_value(getattr(item, "note", None))
        if gender is not None:
            key = glossary_key(source)
            if key:
                result[key] = gender
    return result


def _gender_map_from_candidates(candidates: Sequence[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in candidates:
        gender = _extract_gender(getattr(candidate, "note", None))
        if gender is None:
            continue
        for attr in ("source_text", "suggested_translation"):
            raw = getattr(candidate, attr, None)
            if isinstance(raw, str) and raw.strip():
                key = glossary_key(raw)
                if key:
                    result.setdefault(key, gender)
    return result


def _gender_map_from_context_facts(context_facts: Sequence[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for cf in context_facts:
        if getattr(cf, "kind", None) != "gender":
            continue
        if getattr(cf, "certainty", None) == "unknown":
            continue
        gender = _normalize_gender_value(getattr(cf, "value", None))
        if gender is None:
            continue
        subject = getattr(cf, "subject_anchor", None)
        quote = getattr(subject, "quote", None) if subject is not None else None
        if not isinstance(quote, str) or not quote.strip():
            continue
        words = expression_words(quote)
        if not words or (len(words) == 1 and words[0] in _PRONOUN_WORDS):
            continue
        key = glossary_key(quote)
        if key:
            result[key] = gender
    return result


def _gender_from_title(name: str) -> str | None:
    words = expression_words(name)
    if not words:
        return None
    male_hit = any(word in _MALE_TITLES for word in words)
    female_hit = any(word in _FEMALE_TITLES for word in words)
    if male_hit and female_hit:
        return None
    if words[0] in _MALE_TITLES:
        return "male"
    if words[0] in _FEMALE_TITLES:
        return "female"
    return None


def _title_tokens(name: str) -> frozenset[str]:
    return frozenset(word for word in expression_words(name) if word in _ALL_GENDER_TITLES)


def _titles_conflict(left_titles: frozenset[str], right_titles: frozenset[str]) -> bool:
    left_male = bool(left_titles & _MALE_TITLES)
    left_female = bool(left_titles & _FEMALE_TITLES)
    right_male = bool(right_titles & _MALE_TITLES)
    right_female = bool(right_titles & _FEMALE_TITLES)
    return (left_male and right_female) or (left_female and right_male)


def _name_core_words(name: str) -> frozenset[str]:
    return frozenset(
        word for word in expression_words(name)
        if word not in _ALL_GENDER_TITLES and word not in _SPEAKER_NAME_STOPWORDS
    )


def _is_surname_only_titled(name: str) -> bool:
    titles = _title_tokens(name)
    return bool(titles & _SURNAME_TITLES) and len(_name_core_words(name)) == 1


def _incompatible_surname_titles(left: str, right: str) -> bool:
    left_surname = _is_surname_only_titled(left)
    right_surname = _is_surname_only_titled(right)
    if left_surname == right_surname:
        return False
    return not (_title_tokens(left) & _title_tokens(right) & _SURNAME_TITLES)


def _lookup_gender_in_map(name: str, gender_map: Mapping[str, str]) -> str | None:
    if not gender_map:
        return None
    key = glossary_key(name)
    if key in gender_map:
        return gender_map[key]
    title_gender = _gender_from_title(name)
    name_titles = _title_tokens(name)
    name_core = _name_core_words(name)
    if not name_core:
        return None
    matched_genders: set[str] = set()
    for candidate_key, candidate_gender in gender_map.items():
        if title_gender is not None and candidate_gender != title_gender:
            continue
        candidate_titles = _title_tokens(candidate_key)
        if _titles_conflict(name_titles, candidate_titles):
            continue
        if _incompatible_surname_titles(name, candidate_key):
            continue
        candidate_core = _name_core_words(candidate_key)
        if not candidate_core:
            continue
        if name_core <= candidate_core or candidate_core <= name_core:
            matched_genders.add(candidate_gender)
    if len(matched_genders) == 1:
        return next(iter(matched_genders))
    return None


def _participant_gender(
    participant: str,
    gender_sources: Sequence[Mapping[str, str]],
) -> str | None:
    for source in gender_sources:
        found = _lookup_gender_in_map(participant, source)
        if found is not None:
            return found
    return _gender_from_title(participant)


def _same_speaker_name(
    fact_value: str,
    participant: str,
    glossary_candidates: Sequence[object] = (),
) -> bool:
    key_a = glossary_key(fact_value)
    key_b = glossary_key(participant)
    if key_a and key_a == key_b:
        return True
    titles_a = _title_tokens(fact_value)
    titles_b = _title_tokens(participant)
    if _titles_conflict(titles_a, titles_b):
        return False
    if _incompatible_surname_titles(fact_value, participant):
        return False
    core_a = _name_core_words(fact_value)
    core_b = _name_core_words(participant)
    if core_a and core_b and (core_a <= core_b or core_b <= core_a):
        return True
    for candidate in glossary_candidates:
        src = getattr(candidate, "source_text", "")
        tr = getattr(candidate, "suggested_translation", "")
        pair = {glossary_key(src), glossary_key(tr)} - {""}
        if key_a in pair and key_b in pair:
            return True
    return False


def _clause_pronoun_for_verb(clause_words: tuple[str, ...], verb_index: int) -> str | None:
    attached: set[str] = set()
    if verb_index + 1 < len(clause_words):
        after = clause_words[verb_index + 1]
        if after == "he":
            attached.add("male")
        elif after == "she":
            attached.add("female")
    for offset in (1, 2):
        idx = verb_index - offset
        if idx < 0:
            continue
        if offset == 2 and clause_words[verb_index - 1] in _CLAUSE_BOUNDARY_WORDS:
            continue
        word = clause_words[idx]
        if idx > 0 and clause_words[idx - 1] in _ADDRESSEE_PREPOSITIONS:
            continue
        if word == "he":
            attached.add("male")
        elif word == "she":
            attached.add("female")
    if len(attached) == 1:
        return next(iter(attached))
    if len(attached) > 1:
        return None
    # Coordinate predicate with subject pronoun at start of clause: "he turned and said"
    if verb_index > 0:
        pronouns = {
            "male" if word == "he" else "female"
            for idx in range(0, verb_index)
            if (word := clause_words[idx]) in {"he", "she"}
            and (idx == 0 or clause_words[idx - 1] not in _ADDRESSEE_PREPOSITIONS)
        }
        if len(pronouns) == 1:
            return next(iter(pronouns))
    return None


def _opening_remark_pronoun_gender(text: str) -> str | None:
    quotes = list(_DIALOGUE_QUOTES.finditer(text))
    if not quotes:
        return None
    first = quotes[0]
    previous = _sentence_end_before(text, first.start())
    prefix_start = previous + 1 if previous >= 0 else 0
    next_quote = quotes[1].start() if len(quotes) > 1 else len(text)
    prefix = text[prefix_start:first.start()]
    trailing = _trailing_remark(text, first.end(), next_quote)

    trailing_words = expression_words(trailing)
    if trailing_words:
        boundary = next(
            (i for i, word in enumerate(trailing_words) if word in _SUBORDINATING_CLAUSE_BOUNDARIES),
            len(trailing_words),
        )
        primary_trailing = trailing_words[:boundary]
        first_verb = next(
            (i for i, word in enumerate(primary_trailing) if word in _ALL_REPORTING_VERBS),
            None,
        )
        if first_verb is not None:
            return _clause_pronoun_for_verb(primary_trailing, first_verb)

    if prefix.strip():
        # Strip leading subordinate clause ("As she smiled, Harry said,") before inspecting prefix
        prefix_tail = prefix
        prefix_words_full = expression_words(prefix)
        if prefix_words_full and prefix_words_full[0] in _SUBORDINATING_CLAUSE_BOUNDARIES:
            comma_at = prefix.find(",")
            if comma_at >= 0:
                prefix_tail = prefix[comma_at + 1 :]
            else:
                return None
        prefix_words = expression_words(prefix_tail)
        last_verb = next(
            (i for i in range(len(prefix_words) - 1, -1, -1) if prefix_words[i] in _ALL_REPORTING_VERBS),
            None,
        )
        if last_verb is not None:
            clause_start = 0
            for idx in range(last_verb - 1, -1, -1):
                if prefix_words[idx] in _SUBORDINATING_CLAUSE_BOUNDARIES:
                    clause_start = idx + 1
                    break
            return _clause_pronoun_for_verb(prefix_words[clause_start:], last_verb - clause_start)

    return None


def _scene_for_block(
    block_id: str,
    scenes: Sequence[Scene],
    ordered_block_ids: Sequence[str],
    blocks: Sequence[DocumentBlock] | None,
    texts: Mapping[str, str] | None,
) -> Scene | None:
    order = tuple(ordered_block_ids)
    if not order and blocks:
        order = tuple(block.id for block in blocks)
    if not order and texts:
        order = tuple(texts.keys())
    positions = {bid: index for index, bid in enumerate(order)}
    block_pos = positions.get(block_id)
    if block_pos is None:
        return None
    for scene in scenes:
        start_pos = positions.get(scene.start_block_id)
        end_pos = positions.get(scene.end_block_id)
        if start_pos is not None and end_pos is not None and start_pos <= block_pos <= end_pos:
            return scene
    return None


def _opening_remark_pronoun_resolves_speaker(
    fact,
    text: str,
    *,
    scenes: Sequence[Scene],
    glossary_candidates: Sequence[object],
    context_facts: Sequence[object],
    mandatory_facts: Mapping[str, Mapping[str, str]] | Sequence[object] | None,
    ordered_block_ids: Sequence[str],
    blocks: Sequence[DocumentBlock] | None,
    texts: Mapping[str, str] | None,
) -> bool:
    pronoun_gender = _opening_remark_pronoun_gender(text)
    if pronoun_gender is None:
        return False
    scene = _scene_for_block(
        fact.subject_anchor.block_id, scenes, ordered_block_ids, blocks, texts
    )
    if scene is None or not scene.participants:
        return False
    gender_sources = (
        _gender_map_from_mandatory_facts(mandatory_facts),
        _gender_map_from_context_facts(context_facts),
        _gender_map_from_candidates(glossary_candidates),
    )
    participant_genders = [
        (participant, _participant_gender(participant, gender_sources))
        for participant in scene.participants
    ]
    if any(gender is None for _, gender in participant_genders):
        return False
    matching = [
        participant
        for participant, gender in participant_genders
        if gender == pronoun_gender
    ]
    if len(matching) != 1:
        return False
    fact_gender = _participant_gender(fact.value, gender_sources)
    if fact_gender is not None and fact_gender != pronoun_gender:
        return False
    return _same_speaker_name(fact.value, matching[0], glossary_candidates)


def _fact_block_text(block_id, texts, blocks) -> str | None:
    if texts is not None and block_id in texts:
        return texts[block_id]
    if not blocks:
        return None
    for block in blocks:
        if block.id == block_id:
            return block.text
    return None


def _opening_remark_names_speaker(text: str, value: str) -> bool:
    remark_words = expression_words(_opening_remark(text))
    if not remark_words:
        return False
    value_words = expression_words(value)
    if value_words and _contains_speaker_phrase(remark_words, value_words):
        return True
    value_titles = set(value_words) & _SPEAKER_TITLES
    conflicting_titles = _SPEAKER_TITLES - value_titles
    for name in _speaker_name_words(value):
        for index, word in enumerate(remark_words):
            if word != name:
                continue
            prev = remark_words[index - 1] if index > 0 else None
            if prev in _ADDRESSEE_PREPOSITIONS or prev in conflicting_titles:
                continue
            if (
                prev in _SPEAKER_TITLES
                and index >= 2
                and remark_words[index - 2] in _ADDRESSEE_PREPOSITIONS
            ):
                continue
            return True
    return False


def _contains_speaker_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(
        words[index:index + size] == phrase
        and (index == 0 or words[index - 1] not in _ADDRESSEE_PREPOSITIONS)
        for index in range(len(words) - size + 1)
    )


def _speaker_name_words(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", value)
    proper = [
        token for token in tokens
        if token[:1].isupper() and token.casefold() not in _SPEAKER_NAME_STOPWORDS
    ]
    chosen = proper or [
        token for token in tokens if token.casefold() not in _SPEAKER_NAME_STOPWORDS
    ]
    return expression_words(" ".join(chosen))


def _opening_remark(text: str) -> str:
    """Remark around the first quote. Mr./Mrs./Ms./Dr./St. are not sentence ends."""
    quotes = list(_DIALOGUE_QUOTES.finditer(text))
    if not quotes:
        return ""
    first = quotes[0]
    previous = _sentence_end_before(text, first.start())
    prefix_start = previous + 1 if previous >= 0 else 0
    next_quote = quotes[1].start() if len(quotes) > 1 else len(text)
    return text[prefix_start:first.start()] + _trailing_remark(text, first.end(), next_quote)


def _trailing_remark(text: str, first_end: int, next_quote_start: int) -> str:
    if (
        first_end >= 2
        and text[first_end - 2] == "."
        and not text[:first_end - 1].endswith("...")
    ):
        return ""
    after = text[first_end:next_quote_start]
    clause = after[:_sentence_end_at(after)]
    stripped = clause.lstrip()
    if not stripped:
        return ""
    if first_end >= 2 and text[first_end - 2] != "," and stripped[0].isupper():
        words = expression_words(clause)
        if not any(word in _BASIC_REPORTING or word in _NONBASIC_REPORTING for word in words):
            return ""
    return clause


def _abbreviation_period(text: str, index: int) -> bool:
    if index <= 0 or text[index] != ".":
        return False
    start = index - 1
    while start >= 0 and text[start].isalpha():
        start -= 1
    return text[start + 1:index].casefold() in _REMARK_ABBREVIATIONS


def _sentence_end_before(text: str, limit: int) -> int:
    for index in range(limit - 1, -1, -1):
        if text[index] in ".!?…" and not _abbreviation_period(text, index):
            return index
    return -1


def _sentence_end_at(text: str) -> int:
    for index, mark in enumerate(text):
        if mark in ".!?…" and not _abbreviation_period(text, index):
            return index
    return len(text)


def glossary_context_constraints(concepts):
    """Supported gender notes (including composite notes) become machine-checkable constraints."""
    result = {}
    for concept in concepts:
        if not concept.status.is_user_decision:
            continue
        gender = _extract_gender(getattr(concept, "note", ""))
        if gender is not None:
            source = getattr(concept, "canonical_source_text", None) or concept.source_text
            result[glossary_key(source)] = {"gender": gender}
    return result
