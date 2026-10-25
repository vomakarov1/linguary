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

from linguary.domain.content_plan import ContentPlan, ContextFact, HelpEvidence, SourceAnchor, artifact_key

from linguary.application.block_labels import without_block_labels
from linguary.application.errors import ApplicationError
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    ChapterAnalysisResponse,
    GlossaryCandidateResponse,
)
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    ChapterAnalysis,
    DocumentBlock,
    GlossaryCandidate,
    TreatmentPlan,
    TreatmentPlanItem,
    glossary_key,
    spellings_by_preference,
    translations_conflict,
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
    ) -> FragmentAnalysis:
        """Fragment response with plan mapped to chapter block IDs.

        The plan is assembled in document order: `block_ids_by_alias` enumerates
        fragment blocks as received by the model. A block omitted by the fragment
        is excluded from the plan and will trigger incomplete chapter coverage during merge.
        """
        treatments = {
            item.source_block_id: item.treatment for item in payload.treatment_plan
        }
        unknown = sorted(set(treatments) - set(block_ids_by_alias))
        if unknown:
            raise AnalysisMergeError(
                f"фрагмент {order} вернул режимы для чужих блоков: {unknown}"
            )
        return cls(
            order=order,
            translation_instructions=without_block_labels(payload.translation_instructions),
            chapter_summary=payload.chapter_summary if payload.content_plan is not None and payload.content_plan.version == 3 else without_block_labels(payload.chapter_summary),
            treatments=tuple(
                TreatmentPlanItem(block_id=block_id, treatment=treatments[alias])
                for alias, block_id in block_ids_by_alias.items()
                if alias in treatments
            ),
            glossary_candidates=payload.glossary_candidates,
            before_you_read=payload.before_you_read,
            content_plan=content_plan_from_response(payload, block_ids_by_alias, texts=texts),
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
    coverage is incomplete. Repeating identical treatments is not a conflict — fragments agree.
    """
    decided: dict[str, BlockTreatment] = {}
    conflicting: set[str] = set()
    for fragment in fragments:
        for item in fragment.treatments:
            previous = decided.get(item.block_id)
            if previous is None:
                decided[item.block_id] = item.treatment
            elif previous is not item.treatment:
                conflicting.add(item.block_id)
    for block_id in conflicting:
        del decided[block_id]
    known = set(expected)
    order = [block_id for block_id in expected if block_id in decided]
    order.extend(block_id for block_id in decided if block_id not in known)
    return TreatmentPlan.of_treatments(
        section_id, ((block_id, decided[block_id]) for block_id in order)
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
) -> tuple[GlossaryCandidate, ...]:
    """Chapter glossary candidates merged by source text (PRD § 11-12).

    Obvious spelling variants of the same text — case, typography, period after `Mr` —
    are grouped via `glossary_key`: otherwise each spelling would become a separate
    mandatory decision, and decisions could diverge.
    """
    groups: dict[str, list[GlossaryCandidateResponse]] = {}
    for fragment in fragments:
        for candidate in fragment.glossary_candidates:
            groups.setdefault(glossary_key(candidate.source_text), []).append(candidate)
    return tuple(
        _merged_candidate(
            group,
            id=id_factory(),
            analysis_id=analysis_id,
            confidence_threshold=confidence_threshold,
        )
        for group in groups.values()
    )


def _merged_candidate(
    group: Sequence[GlossaryCandidateResponse],
    *,
    id: str,
    analysis_id: str,
    confidence_threshold: float,
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
    """
    leading = max(group, key=lambda candidate: candidate.confidence)
    return GlossaryCandidate.create(
        id=id,
        analysis_id=analysis_id,
        source_text=spellings_by_preference(
            (candidate.source_text, candidate.confidence) for candidate in group
        )[0],
        type=next(
            (
                candidate.type
                for candidate in group
                if candidate.type.always_requires_review
            ),
            leading.type,
        ),
        suggested_translation=leading.suggested_translation,
        confidence=leading.confidence,
        model_flagged_review=any(candidate.model_flagged_review for candidate in group),
        confidence_threshold=confidence_threshold,
        has_translation_conflict=translations_conflict(
            candidate.suggested_translation for candidate in group
        ),
        note=next(
            (
                candidate.note
                for candidate in sorted(
                    group, key=lambda candidate: candidate.confidence, reverse=True
                )
                if candidate.note.strip()
            ),
            "",
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


def before_you_read_entries(
    entries: Iterable[BeforeYouReadEntryResponse], *, strict: bool = False,
) -> tuple[BeforeYouReadEntry, ...]:
    """Retains up to the first ten distinct chapter expressions with sanitized explanations."""
    selected: dict[str, BeforeYouReadEntry] = {}
    for entry in entries:
        key = (
            re.sub(r"\s+", " ", entry.expression)
            .casefold()
            .strip(" \t.,;:!?\"'«»()[]")
        )
        if key and key not in selected:
            selected[key] = BeforeYouReadEntry(
                expression=entry.expression,
                explanation=entry.explanation if strict else sanitize_before_you_read_explanation(entry.explanation),
            )
        if len(selected) >= MAX_BEFORE_YOU_READ:
            break
    return tuple(selected.values())


__all__ = [
    "DEFAULT_MAX_FRAGMENT_BLOCKS",
    "MAX_BEFORE_YOU_READ",
    "TOKENS_PER_ANALYSED_BLOCK",
    "before_you_read_entries",
    "content_plan_from_response",
    "find_quote_span",
    "sanitize_before_you_read_explanation",
    "validate_content_plan",
    "fragment_blocks_for",
    "AnalysisMergeError",
    "ChapterFragment",
    "FragmentAnalysis",
    "merge_chapter_analysis",
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
    except re.error:
        pass

    # Search without surrounding punctuation
    stripped = quote.strip(" \t\r\n.,;:!?\"'«»()[]")
    if stripped and stripped != quote:
        return find_quote_span(text, stripped, estimated_start=estimated_start)

    return None


def content_plan_from_response(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str] | None = None,
) -> ContentPlan | None:
    """Replaces local aliases and computes stable application references."""
    if payload.content_plan is None:
        return None
    is_v3 = payload.content_plan.version == 3
    if is_v3 and texts is None:
        raise AnalysisMergeError('план v3 требует доступный исходный фрагмент')
    difficulties = []
    ids = {}
    for difficulty in payload.content_plan.difficulties:
        try:
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
    for item in payload.content_plan.before_you_read_evidence:
        if item.artifact_kind != "before_you_read.explanation" or item.artifact_index >= len(payload.before_you_read):
            raise AnalysisMergeError("свидетельство анализа не ссылается на существующий BYR")
        entry = payload.before_you_read[item.artifact_index]
        if not any(e.expression == entry.expression and e.explanation == entry.explanation for e in selected):
            raise AnalysisMergeError("BYR удалён нормализацией или превышает бюджет")
        body_excerpt = item.body_excerpt
        if body_excerpt not in entry.explanation:
            span = find_quote_span(entry.explanation, body_excerpt)
            if span is not None:
                body_excerpt = span[2]
            else:
                stripped = _quotation_key(body_excerpt)
                if stripped and stripped in entry.explanation:
                    body_excerpt = stripped
        if body_excerpt not in entry.explanation or _quotation_key(body_excerpt) == _quotation_key(entry.expression):
            raise AnalysisMergeError("отрывок не принадлежит телу объяснения BYR")
        try:
            evidence.append(HelpEvidence(ids[item.difficulty_id], item.point_id, item.artifact_kind, aliases[item.source_block_id], artifact_key(item.artifact_kind, aliases[item.source_block_id], {"expression": entry.expression, "explanation": entry.explanation}), body_excerpt))
        except KeyError as error:
            raise AnalysisMergeError("неизвестная трудность или владелец свидетельства") from error
    facts = []
    for fact in payload.content_plan.context_facts:
        try:
            mapped = replace(fact,
                anchors=tuple(replace(anchor, block_id=aliases[anchor.block_id]) for anchor in fact.anchors),
                subject_anchor=replace(fact.subject_anchor, block_id=aliases[fact.subject_anchor.block_id]))
        except KeyError:
            raise AnalysisMergeError('контекстная атрибуция ссылается на невидимый блок') from None
        mapped.validate(texts)
        facts.append(replace(mapped, id=mapped.canonical_id))
    return ContentPlan(version=payload.content_plan.version, profile=payload.content_plan.profile,
        difficulties=tuple(difficulties), before_you_read_evidence=tuple(evidence), context_facts=tuple(facts))


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
    facts = {}
    for plan in plans:
        for fact in plan.context_facts:
            previous = facts.setdefault(fact.canonical_id, fact)
            if previous != fact:
                raise AnalysisMergeError('конфликт канонической атрибуции')
    result = ContentPlan(version=plans[0].version, profile=plans[0].profile,
        difficulties=tuple(difficulties), before_you_read_evidence=tuple(evidence),
        context_facts=tuple(facts.values()))
    selected = _merged_before_you_read(fragments)
    for item in evidence:
        if not any(artifact_key(item.artifact_kind, item.block_id, {"expression": e.expression, "explanation": e.explanation}) == item.artifact_key for e in selected):
            raise AnalysisMergeError("слияние удалило BYR со свидетельством")
    return result


def validate_content_plan(
    payload: ChapterAnalysisResponse,
    aliases: Mapping[str, str],
    texts: Mapping[str, str],
    *,
    before_you_read_limit: int = 10,
    language_notes_per_chapter: int = 8,
    note_limits_by_block: Mapping[str, int] | None = None,
    note_group_limits: Sequence[tuple[Sequence[str], int]] = (),
) -> None:
    """Validates source anchors and channel eligibility prior to accepting the response."""
    plan = content_plan_from_response(payload, aliases, texts=texts)
    if plan is None:
        return
    if len(payload.before_you_read) > before_you_read_limit:
        raise AnalysisMergeError("BYR превышает согласованный бюджет фрагмента")
    treatments = {aliases[i.source_block_id]: i.treatment for i in payload.treatment_plan}
    note_owners = set()
    for difficulty in plan.difficulties:
        for anchor in difficulty.anchors:
            anchor.validate(texts)
        covered = {e.point_id for e in plan.before_you_read_evidence if e.difficulty_id == difficulty.id}
        if difficulty.help_kind == "lexical_hint" and treatments.get(difficulty.owner_block_id) is not BlockTreatment.GLOSSARY_ONLY and covered != {p.id for p in difficulty.required_points}:
            raise AnalysisMergeError("лексическая помощь требует GLOSSARY_ONLY или подтверждённый BYR")
        if (difficulty.help_kind == "explanation"
                and covered != {p.id for p in difficulty.required_points}
                and treatments.get(difficulty.owner_block_id) is BlockTreatment.READ_AS_IS):
            note_owners.add(difficulty.owner_block_id)
        for item in plan.before_you_read_evidence:
            if item.difficulty_id == difficulty.id and any(_quotation_key(item.body_excerpt) == _quotation_key(a.quote) for a in difficulty.anchors):
                raise AnalysisMergeError("цитата оригинала не является объяснением")
    if any(len(note_owners.intersection(ids)) > limit for ids, limit in note_group_limits):
        raise AnalysisMergeError("обязательные заметки не помещаются в реальную группу обработки")
    if len(note_owners) > language_notes_per_chapter or any(
        note_limits_by_block is not None and note_limits_by_block.get(owner, 0) < 1
        for owner in note_owners
    ):
        raise AnalysisMergeError("содержательный план не помещается в бюджет языковых заметок")


def _quotation_key(text: str) -> str:
    """Strips only outer quotation formatting without semantic evaluation."""
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or unicodedata.category(text[start]).startswith("P")):
        start += 1
    while end > start and (text[end - 1].isspace() or unicodedata.category(text[end - 1]).startswith("P")):
        end -= 1
    return text[start:end]



def validate_fact_glossary(plan, mandatory_facts):
    """Reject explicit contradictions with structured mandatory glossary decisions."""
    if plan is None or plan.version != 3:
        return
    for fact in plan.context_facts:
        expected = mandatory_facts.get(fact.subject_anchor.quote, {}).get(fact.kind)
        if expected is not None and fact.certainty != 'unknown' and fact.value != expected:
            raise AnalysisMergeError('контекстная атрибуция противоречит обязательному решению глоссария')


def glossary_context_constraints(concepts):
    """Only exact supported gender notes become machine-checkable constraints."""
    genders = {'мужской род': 'male', 'женский род': 'female', 'средний род': 'neuter',
               'male': 'male', 'female': 'female', 'neuter': 'neuter'}
    result = {}
    for concept in concepts:
        if not concept.status.is_user_decision:
            continue
        note = concept.note.strip().casefold().rstrip('.')
        gender = genders.get(note)
        if gender is not None:
            source = getattr(concept, 'canonical_source_text', None) or concept.source_text
            result[source] = {'gender': gender}
    return result
