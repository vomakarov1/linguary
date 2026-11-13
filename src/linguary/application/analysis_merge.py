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

from linguary.domain.content_plan import ContentPlan, ContextFact, Difficulty, HelpEvidence, Scene, SceneVisualFact, SourceAnchor, artifact_key
from linguary.domain.llm_log import Violation

from linguary.application.block_labels import without_block_labels
from linguary.application.errors import ApplicationError
from linguary.application.explained_expressions import _same_word, expression_words
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
    QuotedContentPlanResponse,
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedContextFactResponse,
    QuotedSceneResponse,
    QuotedSceneVisualFactResponse,
    QuotedSourceAnchorResponse,
    TreatmentPlanItemResponse,
)
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    GlossaryCandidate,
    TreatmentPlan,
    TreatmentPlanItem,
    glossary_key,
    spellings_by_preference,
    translations_conflict,
    without_footnote_markers,
)
from linguary.domain.name_translation import (
    NameTranslationEvidence, NameTranslationObservation, NameTranslationPolicy,
)

#: Fallback limit if provider does not report output token capacity.
DEFAULT_MAX_FRAGMENT_BLOCKS = 40
#: Upper-bound estimate of response tokens per analyzed block from real runs.
TOKENS_PER_ANALYSED_BLOCK = 200
MAX_BEFORE_YOU_READ = 10


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
        else None
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
                )
            ),
        )


_DESCRIBED_CANDIDATE_TYPES = frozenset({ConceptType.PERSON, ConceptType.PLACE})


def kept_candidate_description(
    candidate_type: ConceptType, responses: Iterable[GlossaryCandidateResponse]
) -> str:
    """Description is kept only for a person or a place.

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
        raise AnalysisMergeError(
            f"режимы сегментов блока {block_id} нельзя собрать без самого блока"
        )
    spans = block.segment_spans
    if len(spans) < 2:
        raise AnalysisMergeError(
            f"блок {block_id} состоит из одного сегмента, режимы сегментов не задаются"
        )
    by_index: dict[int, BlockTreatment] = {}
    for segment in item.segments:
        if segment.index < 1 or segment.index > len(spans):
            raise AnalysisMergeError(
                f"индекс сегмента {segment.index} блока {block_id} "
                f"вне диапазона 1..{len(spans)}"
            )
        if segment.index in by_index:
            raise AnalysisMergeError(
                f"сегмент {segment.index} блока {block_id} указан повторно"
            )
        by_index[segment.index] = segment.treatment
    if set(by_index) != set(range(1, len(spans) + 1)):
        raise AnalysisMergeError(
            f"режимы сегментов блока {block_id} покрывают не все сегменты"
        )
    return TreatmentPlanItem(
        block_id=block_id,
        treatment=item.treatment,
        segment_treatments=tuple(by_index[index] for index in range(1, len(spans) + 1)),
        sentence_indexes=tuple(span.sentence_index for span in spans),
    )


def plan_fragments(
    blocks: Sequence[DocumentBlock],
    *,
    max_blocks: int = DEFAULT_MAX_FRAGMENT_BLOCKS,
) -> tuple[ChapterFragment, ...]:
    """Splits textual blocks of a chapter into sequential analysis fragments (PRD § 14).

    A chapter fitting into a single request remains a single fragment: chunking is
    a fallback path, not the default. A long chapter is divided into equal block
    counts rather than slicing off maximum-sized chunks: a one-paragraph tail would
    produce a contextless fragment and an extra call.
    """
    if max_blocks < 1:
        raise AnalysisMergeError("размер фрагмента главы должен быть положительным")
    total = len(blocks)
    if total == 0:
        return ()
    count = -(-total // max_blocks)
    size, remainder = divmod(total, count)
    fragments: list[ChapterFragment] = []
    start = 0
    for order in range(count):
        length = size + (1 if order < remainder else 0)
        fragments.append(
            ChapterFragment(order=order, blocks=tuple(blocks[start : start + length]))
        )
        start += length
    return tuple(fragments)


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


def _merged_instructions(fragments: Sequence[FragmentAnalysis]) -> str:
    """Chapter translation instructions: fragment directives ordered by reading sequence.

    Identical directives from different fragments are deduplicated: narrator voice and
    character speech quirks are noticed in every fragment, but the chapter translator
    only needs the directive once.
    """
    points: dict[str, str] = {}
    for fragment in fragments:
        for point in _instruction_points(fragment.translation_instructions):
            cleaned = without_block_labels(point)
            key = re.sub(r"\s+", " ", cleaned).casefold().strip(" \t.,;:!?")
            if key:
                points.setdefault(key, cleaned)
    return "\n".join(
        f"{index}. {point}" for index, point in enumerate(points.values(), 1)
    )


def _instruction_points(text: str) -> tuple[str, ...]:
    """Parses bulleted items and plain paragraphs."""
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
            points.append(re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", paragraph).strip())
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


def _candidate_groups(candidates: Iterable[GlossaryCandidateResponse]) -> tuple[list[GlossaryCandidateResponse], ...]:
    groups: dict[str, list[GlossaryCandidateResponse]] = {}
    for candidate in candidates:
        groups.setdefault(glossary_key(candidate.source_text), []).append(candidate)
    return tuple(groups.values())


def _merged_candidate_suggestion(group: Sequence[GlossaryCandidateResponse]) -> tuple[str, str, str]:
    leading = max(group, key=lambda candidate: candidate.confidence)
    source_text = spellings_by_preference(
        (candidate.source_text, candidate.confidence) for candidate in group
    )[0]
    note = next((candidate.note for candidate in sorted(
        group, key=lambda candidate: candidate.confidence, reverse=True
    ) if candidate.note.strip()), "")
    return source_text, leading.suggested_translation, note


def merged_glossary_suggestion_keys(candidates: Iterable[GlossaryCandidateResponse]) -> frozenset[tuple[str, str, str]]:
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
    "DEFAULT_MAX_FRAGMENT_BLOCKS",
    "MAX_BEFORE_YOU_READ",
    "TOKENS_PER_ANALYSED_BLOCK",
    "before_you_read_entries",
    "deduplicate_book_before_you_read",
    "content_plan_from_response",
    "find_quote_span",
    "glossary_context_constraints",
    "sanitize_before_you_read_explanation",
    "validate_content_plan",
    "validate_fact_glossary",
    "fragment_blocks_for",
    "AnalysisMergeError",
    "ChapterFragment",
    "FragmentAnalysis",
    "merge_chapter_analysis",
    "merged_glossary_suggestion_keys",
    "plan_fragments",
]


def find_quote_span(
    text: str, quote: str, *, estimated_start: int = 0
) -> tuple[int, int, str] | None:
    """Locates quote occurrences in text accounting for typography and whitespace."""
    if not text or not quote:
        return None

    # Search for exact matches
    exact_matches: list[tuple[int, int, str]] = []
    idx = 0
    quote_len = len(quote)
    while True:
        pos = text.find(quote, idx)
        if pos == -1:
            break
        exact_matches.append((pos, pos + quote_len, text[pos : pos + quote_len]))
        idx = pos + 1

    if exact_matches:
        return min(exact_matches, key=lambda m: abs(m[0] - estimated_start))

    # Lenient search accounting for variants of quotes, apostrophes, and dashes
    pattern_parts: list[str] = []
    for ch in quote:
        if ch in ("'", "’", "‘", "`"):
            pattern_parts.append(r"['’‘`]")
        elif ch in ('"', "“", "”", "«", "»"):
            pattern_parts.append(r'["“”«»]')
        elif ch in ("-", "—", "–"):
            pattern_parts.append(r"[-—–]")
        elif ch.isspace():
            pattern_parts.append(r"\s+")
        else:
            pattern_parts.append(re.escape(ch))
    pattern = "".join(pattern_parts)
    pattern = re.sub(r"\b(she|he|it|that|there|what|who)(?:\\s\+is|\['’‘`\]s)\b", r"\1(?:\\s+is|['’‘`]s)", pattern, flags=re.I)
    pattern = re.sub(r"\b(I)(?:\\s\+am|\['’‘`\]m)\b", r"\1(?:\\s+am|['’‘`]m)", pattern, flags=re.I)
    pattern = re.sub(r"\b(you|we|they)(?:\\s\+are|\['’‘`\]re)\b", r"\1(?:\\s+are|['’‘`]re)", pattern, flags=re.I)
    pattern = re.sub(r"\b(do|does|did|is|are|was|were|have|has|had|could|would|should|must)(?:\\s\+not|n\['’‘`\]t)\b", r"\1(?:\\s+not|n['’‘`]t)", pattern, flags=re.I)
    pattern = re.sub(r"\b(?:will\\s\+not|won\['’‘`\]t)\b", r"(?:will\\s+not|won['’‘`]t)", pattern, flags=re.I)
    pattern = re.sub(r"\b(?:can\\s\+not|cannot|can\['’‘`\]t)\b", r"(?:can\\s+not|cannot|can['’‘`]t)", pattern, flags=re.I)
    try:
        regex_matches = [
            (m.start(), m.end(), text[m.start() : m.end()])
            for m in re.finditer(pattern, text)
        ]
        if regex_matches:
            return min(regex_matches, key=lambda m: abs(m[0] - estimated_start))
        ci_matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if ci_matches:
            best = min(ci_matches, key=lambda m: abs(m.start() - estimated_start))
            return (best.start(), best.end(), text[best.start() : best.end()])
    except re.error:
        pass

    # Search without surrounding punctuation
    stripped = quote.strip(" \t\r\n.,;:!?\"'«»()[]“”‘’—–")
    if stripped and stripped != quote:
        return find_quote_span(text, stripped, estimated_start=estimated_start)

    return None


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


def resolve_scenes(
    scenes: Sequence[QuotedSceneResponse],
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    visible_texts_by_alias: Mapping[str, str],
    *,
    ordered_block_ids: Sequence[str],
) -> tuple[Scene, ...]:
    """Resolve scene references within the visible chapter or fragment."""
    positions = {block_id: index for index, block_id in enumerate(ordered_block_ids)}
    resolved = []
    previous_end = -1
    for order, raw in enumerate(scenes):
        start, end = aliases.get(raw.start_block_id), aliases.get(raw.end_block_id)
        if (start not in positions or end not in positions
                or raw.start_block_id not in visible_texts_by_alias
                or raw.end_block_id not in visible_texts_by_alias
                or start not in texts or end not in texts
                or positions[start] > positions[end] or positions[start] <= previous_end):
            raise _analysis_error(
                'границы сцен должны принадлежать главе, идти по порядку и не пересекаться',
                rule_code='scene_bounds', block_ids=(start, end, *ordered_block_ids),
            )
        facts = []
        for fact in raw.visual_facts:
            anchor = _resolve_quoted_anchor(fact.anchor.block_id, fact.anchor.quote,
                                           aliases, texts, visible_texts_by_alias, strict=True)
            if anchor is None:
                raise _analysis_error('цитата визуального факта не разрешается в исходном блоке',
                                      rule_code='unresolved_quote', block_ids=(start, end))
            if anchor.block_id not in positions or not positions[start] <= positions[anchor.block_id] <= positions[end]:
                raise _analysis_error('визуальный факт находится вне сцены',
                                      rule_code='scene_visual_anchor_out_of_bounds', block_ids=(anchor.block_id,))
            facts.append(SceneVisualFact(anchor, fact.subject, fact.detail))
        item = Scene('pending', order, start, end, raw.summary, raw.location,
                     raw.participants, tuple(facts), raw.visual_assumptions)
        resolved.append(replace(item, id=item.canonical_id))
        previous_end = positions[end]
    return tuple(resolved)


def validate_narrative_response(
    payload: NarrativeAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    visible_texts_by_alias: Mapping[str, str],
    *,
    ordered_block_ids: Sequence[str],
    mandatory_facts: Mapping[str, Mapping[str, str]] | None = None,
    is_notes_only: bool = False,
) -> tuple[tuple[Scene, ...], tuple[ContextFact, ...]]:
    """Validate complete scene coverage and source-bound narrative facts."""
    if is_notes_only and payload.scenes:
        raise _analysis_error('служебный раздел не должен содержать сцены',
                              rule_code='notes_only_scenes', block_ids=ordered_block_ids)
    scenes = resolve_scenes(payload.scenes, aliases, texts, visible_texts_by_alias,
                            ordered_block_ids=ordered_block_ids)
    if not is_notes_only and ordered_block_ids:
        positions = {block: index for index, block in enumerate(ordered_block_ids)}
        next_start = 0
        for item in scenes:
            if positions[item.start_block_id] != next_start:
                break
            next_start = positions[item.end_block_id] + 1
        if next_start != len(ordered_block_ids):
            raise _analysis_error('карта сцен должна непрерывно покрывать все блоки главы',
                                  rule_code='scene_bounds', block_ids=ordered_block_ids)
    scoped_aliases = {alias: block for alias, block in aliases.items() if block in ordered_block_ids}
    response = ChapterAnalysisResponse(payload.translation_instructions, payload.chapter_summary,
        content_plan=QuotedContentPlanResponse(context_facts=payload.context_facts))
    plan = content_plan_from_response(response, scoped_aliases, texts,
                                      visible_texts_by_alias=visible_texts_by_alias)
    if plan is None:
        raise _unresolved_quote_error(response, scoped_aliases, texts, visible_texts_by_alias,
                                      {v: k for k, v in scoped_aliases.items()})
    validate_fact_glossary(plan, mandatory_facts or {}, {v: k for k, v in scoped_aliases.items()})
    return scenes, plan.context_facts


def _quoted_scene(scene: Scene) -> QuotedSceneResponse:
    return QuotedSceneResponse(scene.start_block_id, scene.end_block_id, scene.summary,
        scene.location, scene.participants,
        tuple(QuotedSceneVisualFactResponse(QuotedSourceAnchorResponse(f.anchor.block_id, f.anchor.quote),
                                            f.subject, f.detail) for f in scene.visual_facts),
        scene.visual_assumptions)


def _quoted_fact(fact: ContextFact) -> QuotedContextFactResponse:
    return QuotedContextFactResponse(fact.id, fact.kind,
        tuple(QuotedSourceAnchorResponse(a.block_id, a.quote) for a in fact.anchors),
        QuotedSourceAnchorResponse(fact.subject_anchor.block_id, fact.subject_anchor.quote),
        fact.value, fact.certainty)


def merge_narrative_fragments(
    fragments: Sequence[tuple[NarrativeAnalysisResponse, tuple[Scene, ...], tuple[ContextFact, ...]]],
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
    for index, (raw, resolved, context_facts) in enumerate(fragments):
        instruction_fragments.append(FragmentAnalysis(index, raw.translation_instructions, raw.chapter_summary))
        for candidate in raw.glossary_candidates:
            if candidate not in candidates:
                candidates.append(candidate)
        for fact in context_facts:
            facts.setdefault(fact.canonical_id, replace(fact, id=fact.canonical_id))
        current = list(resolved)
        implicit_continuation = (
            previous_had_scene and current
            and scenes[-1].location is not None and bool(scenes[-1].location.strip())
            and scenes[-1].location == current[0].location
            and bool(scenes[-1].participants)
            and set(scenes[-1].participants) == set(current[0].participants)
        )
        if (previous_had_scene and current
                and ((raw.scenes and raw.scenes[0].continues_previous_scene) or implicit_continuation)):
            prev, first = scenes.pop(), current.pop(0)
            merged = replace(prev, end_block_id=first.end_block_id,
                summary=' '.join(dict.fromkeys((prev.summary, first.summary))),
                location=prev.location or first.location,
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
) -> ChapterAnalysisResponse:
    """Assemble both pass contracts without changing pedagogical reference aliases."""
    return ChapterAnalysisResponse(narrative.translation_instructions,
        pedagogical.reader_chapter_summary or narrative.chapter_summary,
        narrative.glossary_candidates, pedagogical.treatment_plan, pedagogical.before_you_read,
        QuotedContentPlanResponse(difficulties=pedagogical.difficulties,
            before_you_read_evidence=pedagogical.before_you_read_evidence,
            context_facts=narrative.context_facts,
            scenes=tuple(_quoted_scene(s) for s in resolved_scenes) if resolved_scenes else narrative.scenes))


def content_plan_from_response(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None = None,
    *,
    visible_texts_by_alias: Mapping[str, str] | None = None,
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
    if is_quoted:
        if visible_texts_by_alias is None:
            return None
        for difficulty in payload.content_plan.difficulties:
            owner_block_id = aliases.get(difficulty.owner_block_id)
            if owner_block_id is None:
                return None
            anchors = tuple(
                _resolve_quoted_anchor(anchor.block_id, anchor.quote, aliases, texts,
                                       visible_texts_by_alias)
                for anchor in difficulty.anchors
            )
            if any(anchor is None for anchor in anchors):
                return None
            resolved_difficulties.append((difficulty, anchors, owner_block_id))
        for fact in payload.content_plan.context_facts:
            anchors = tuple(
                _resolve_quoted_anchor(anchor.block_id, anchor.quote, aliases, texts,
                                       visible_texts_by_alias)
                for anchor in fact.anchors
            )
            subject_anchor = _resolve_quoted_anchor(
                fact.subject_anchor.block_id, fact.subject_anchor.quote, aliases, texts,
                visible_texts_by_alias,
            )
            if subject_anchor is None or any(anchor is None for anchor in anchors):
                return None
            resolved_facts.append((fact, anchors, subject_anchor))

    difficulties = []
    ids = {}
    difficulty_rows = (resolved_difficulties if is_quoted else
                       [(difficulty, None, None) for difficulty in payload.content_plan.difficulties])
    for difficulty, resolved_anchors, resolved_owner in difficulty_rows:
        try:
            if is_quoted:
                mapped = Difficulty(
                    id=difficulty.id,
                    anchors=resolved_anchors,
                    owner_block_id=resolved_owner,
                    kind=difficulty.kind,
                    priority=difficulty.priority,
                    help_kind=difficulty.help_kind,
                    required_points=difficulty.required_points,
                )
            else:
                mapped_anchors = []
                for a in difficulty.anchors:
                    real_block_id = aliases[a.block_id]
                    if is_v3:
                        mapped_anchor = replace(a, block_id=real_block_id)
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
    facts = []
    fact_rows = (resolved_facts if is_quoted else
                 [(fact, None, None) for fact in payload.content_plan.context_facts])
    for fact, resolved_anchors, resolved_subject in fact_rows:
        if is_quoted:
            mapped = ContextFact(id=fact.id, kind=fact.kind, anchors=resolved_anchors,
                subject_anchor=resolved_subject, value=fact.value, certainty=fact.certainty)
        else:
            try:
                mapped = replace(fact,
                    anchors=tuple(replace(anchor, block_id=aliases[anchor.block_id]) for anchor in fact.anchors),
                    subject_anchor=replace(fact.subject_anchor, block_id=aliases[fact.subject_anchor.block_id]))
            except KeyError:
                raise AnalysisMergeError('контекстная атрибуция ссылается на невидимый блок') from None
        mapped.validate(texts)
        facts.append(replace(mapped, id=mapped.canonical_id))
    return ContentPlan(version=payload.content_plan.version, profile=payload.content_plan.profile,
        difficulties=tuple(difficulties), before_you_read_evidence=tuple(evidence), context_facts=tuple(facts),
        scenes=(resolve_scenes(payload.content_plan.scenes, aliases, texts, visible_texts_by_alias,
                               ordered_block_ids=tuple(dict.fromkeys(aliases.values()))) if is_quoted else ()))


def _resolve_quoted_anchor(block_alias, quote, aliases, texts, visible_texts_by_alias, *, strict=False):
    """Resolve both source views; visual facts require a verbatim quote."""
    block_id = aliases.get(block_alias)
    if block_id is None or texts is None or block_id not in texts or block_alias not in visible_texts_by_alias:
        return None
    if strict:
        start = texts[block_id].find(quote)
        if start < 0 or quote not in visible_texts_by_alias[block_alias]:
            return None
        return SourceAnchor(block_id=block_id, start=start, end=start + len(quote), quote=quote)
    visible_span = find_quote_span(visible_texts_by_alias[block_alias], quote)
    source_span = find_quote_span(texts[block_id], quote)
    if visible_span is None or source_span is None:
        return None
    start, end, actual_quote = source_span
    return SourceAnchor(block_id=block_id, start=start, end=end, quote=actual_quote)


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
    if visible_texts_by_alias is None:
        return (label, preview, "нет видимого представления блока", (block_id,) if block_id else ())
    if block_id is None:
        return (block_alias, preview, "неизвестный блок", ())
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
    if isinstance(plan, QuotedContentPlanResponse):
        for difficulty in plan.difficulties:
            if aliases.get(difficulty.owner_block_id) is None:
                sample = difficulty.anchors[0].quote if difficulty.anchors else difficulty.id
                msg = f"{prefix} ({difficulty.owner_block_id}: «{_preview_quote(sample)}» — неизвестный блок-владелец трудности)"
                return _analysis_error(msg, rule_code="unresolved_quote")
            for anchor in difficulty.anchors:
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
                    msg = f"{prefix} ({block_label}: «{preview}» — {reason})"
                    return _analysis_error(msg, rule_code="unresolved_quote", block_ids=canonical)
        for fact in plan.context_facts:
            for anchor in (fact.subject_anchor, *fact.anchors):
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
                    msg = f"{prefix} ({block_label}: «{preview}» — {reason})"
                    return _analysis_error(msg, rule_code="unresolved_quote", block_ids=canonical)
    return AnalysisMergeError(prefix)


def validate_content_plan(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    *,
    before_you_read_limit: int = 10,
    language_notes_per_chapter: int = 8,
    english_help: bool = False,
    note_limits_by_block: Mapping[str, int] | None = None,
    note_group_limits: Sequence[tuple[Sequence[str], int]] = (),
    visible_texts_by_alias: Mapping[str, str] | None = None,
    mandatory_facts: Mapping[str, Mapping[str, str]] | None = None,
) -> ContentPlan | None:
    """Validates source anchors and channel eligibility prior to accepting the response."""
    aliases_by_block_id = {bid: alias for alias, bid in (aliases or {}).items() if bid}
    plan = content_plan_from_response(
        payload,
        aliases,
        texts=texts,
        visible_texts_by_alias=visible_texts_by_alias,
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
        treatments = {aliases[i.source_block_id]: i.treatment for i in payload.treatment_plan}
    except KeyError as error:
        raise AnalysisMergeError("привязка к невидимому блоку") from error
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
        covered = {
            e.point_id
            for e in plan.before_you_read_evidence
            if e.difficulty_id == difficulty.id
        }
        fragment_help = (
            english_help
            and plan.version == 3
            and treatments.get(difficulty.owner_block_id) is BlockTreatment.READ_AS_IS
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
            difficulty.help_kind == "lexical_hint"
            and not fragment_help
            and treatments.get(difficulty.owner_block_id) is not BlockTreatment.GLOSSARY_ONLY
            and covered != {p.id for p in difficulty.required_points}
        ):
            label = aliases_by_block_id.get(
                difficulty.owner_block_id, difficulty.owner_block_id
            )
            msg = (
                "лексическая помощь требует GLOSSARY_ONLY или подтверждённый BYR "
                f"({label}, {difficulty.id})"
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
            and treatments.get(difficulty.owner_block_id) is BlockTreatment.READ_AS_IS
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
    r"\b(мужск\w*\s+род\w*|женск\w*\s+род\w*|средн\w*\s+род\w*|male|female|neuter)\b",
    re.IGNORECASE,
)


def _extract_gender(note: str | None) -> str | None:
    if not note:
        return None
    match = _GENDER_NOTE_PATTERN.search(note)
    if match is None:
        return None
    token = match.group(1).casefold()
    if token.startswith("мужск") or token == "male":
        return "male"
    if token.startswith("женск") or token == "female":
        return "female"
    if token.startswith("средн") or token == "neuter":
        return "neuter"
    return None


def validate_fact_glossary(
    plan,
    mandatory_facts,
    aliases_by_block_id: Mapping[str, str] | None = None,
):
    """Reject explicit contradictions with structured mandatory glossary decisions."""
    if plan is None or plan.version != 3:
        return
    normalized_mandatory = {
        glossary_key(k): v for k, v in mandatory_facts.items()
    }
    alias_map = aliases_by_block_id or {}
    for fact in plan.context_facts:
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
            msg = (
                "контекстная атрибуция противоречит обязательному решению глоссария "
                f"({label}: «{_preview_quote(fact.subject_anchor.quote)}»)"
            )
            raise _analysis_error(
                msg,
                rule_code="glossary_fact_conflict",
                block_ids=(block_id,),
            )


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
