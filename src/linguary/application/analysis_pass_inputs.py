"""Shared deterministic input reconstruction for chapter NARRATIVE passes (`ALG-01`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from linguary.application.analysis_merge import candidates_from_narrative
from linguary.application.context_builder import textual_blocks
from linguary.application.glossary_review import candidates_with_decisions
from linguary.application.llm_dto import NarrativeAnalysisResponse
from linguary.domain import (
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentSection,
    GlossaryCandidate,
    TermConcept,
    TermStatus,
    TermVariant,
    glossary_key,
)
from linguary.domain.name_translation import NameTranslationPolicy


def has_meaningful_part_title(
    document: CanonicalDocument, section: DocumentSection
) -> bool:
    checker = getattr(document, "has_meaningful_part_title", None)
    return bool(checker(section)) if callable(checker) else False


def is_meaningful_part_divider(
    document: CanonicalDocument, section: DocumentSection
) -> bool:
    is_front_matter = getattr(document, "is_front_matter", lambda _s: False)
    return (
        bool(is_front_matter(section))
        and has_meaningful_part_title(document, section)
        and bool(textual_blocks(section))
    )


def is_illustrable_content_section(
    document: CanonicalDocument, section: DocumentSection
) -> bool:
    """True for book sections that carry narrative content eligible for illustrations."""
    is_front_matter = getattr(document, "is_front_matter", lambda _s: False)
    return (
        bool(textual_blocks(section))
        and not bool(is_front_matter(section))
        and not getattr(section, "is_notes_only", False)
    )


def first_content_section(document: CanonicalDocument) -> DocumentSection | None:
    is_front_matter = getattr(document, "is_front_matter", lambda _s: False)
    sections = getattr(document, "ordered_sections", document.sections)
    return next(
        (
            section
            for section in sections
            if not is_front_matter(section) and not getattr(section, "is_notes_only", False)
        ),
        None,
    )


def serialize_user_decisions(
    decisions: Sequence[TermConcept], target_language: str
) -> list[dict[str, str]]:
    """Serialize pre-analysis user glossary decisions for `input_context` (`schema_version: 1`)."""
    out: list[dict[str, str]] = []
    for concept in decisions:
        variant = concept.variant_for(target_language)
        out.append(
            {
                "source_text": concept.canonical_source_text,
                "type": concept.type.value,
                "translation": variant.text if variant is not None else "",
                "status": concept.status.value,
                "note": concept.note,
            }
        )
    return out


def serialize_nonbinding_suggestions(
    suggestions: Sequence[TermConcept], target_language: str
) -> list[dict[str, str]]:
    """Serialize nonbinding glossary suggestions for `input_context` (`schema_version: 1`)."""
    out: list[dict[str, str]] = []
    for concept in suggestions:
        variant = concept.variant_for(target_language)
        out.append(
            {
                "source_text": concept.canonical_source_text,
                "type": concept.type.value,
                "translation": variant.text if variant is not None else "",
                "note": concept.note,
            }
        )
    return out


def deserialize_user_decisions(
    raw_decisions: Sequence[Mapping[str, str]],
    *,
    book_id: str,
    source_language: str,
    target_language: str,
) -> tuple[TermConcept, ...]:
    """Reconstruct `TermConcept` entries from serialized `input_context.user_decisions`."""
    concepts: list[TermConcept] = []
    for idx, item in enumerate(raw_decisions):
        cid = f"ctx-decision-{idx}"
        status = TermStatus(item["status"])
        variants = [
            TermVariant(concept_id=cid, language=source_language, text=item["source_text"], status=status)
        ]
        if item["translation"]:
            variants.append(
                TermVariant(concept_id=cid, language=target_language, text=item["translation"], status=status)
            )
        concepts.append(
            TermConcept(
                id=cid,
                book_id=book_id,
                type=ConceptType(item["type"]),
                canonical_source_text=item["source_text"],
                status=status,
                variants=tuple(variants),
                note=item["note"],
            )
        )
    return tuple(concepts)


def deserialize_nonbinding_suggestions(
    raw_suggestions: Sequence[Mapping[str, str]],
    *,
    book_id: str,
    source_language: str,
    target_language: str,
) -> tuple[TermConcept, ...]:
    """Reconstruct `TermConcept` entries from serialized `input_context.nonbinding_suggestions`."""
    concepts: list[TermConcept] = []
    for idx, item in enumerate(raw_suggestions):
        cid = f"ctx-suggestion-{idx}"
        variants = [
            TermVariant(
                concept_id=cid,
                language=source_language,
                text=item["source_text"],
                status=TermStatus.SUGGESTED,
            )
        ]
        if item["translation"]:
            variants.append(
                TermVariant(
                    concept_id=cid,
                    language=target_language,
                    text=item["translation"],
                    status=TermStatus.SUGGESTED,
                )
            )
        concepts.append(
            TermConcept(
                id=cid,
                book_id=book_id,
                type=ConceptType(item["type"]),
                canonical_source_text=item["source_text"],
                status=TermStatus.SUGGESTED,
                variants=tuple(variants),
                note=item["note"],
            )
        )
    return tuple(concepts)


def build_narrative_input_context(
    *,
    sequential_analysis: bool,
    parallel_batch: bool,
    seed_active: bool,
    user_decisions: Sequence[TermConcept],
    nonbinding_suggestions: Sequence[TermConcept],
    target_language: str,
    prompts_version: int,
) -> dict[str, Any]:
    """Build strict versioned `input_context` (`schema_version: 1`) for a full NARRATIVE pass."""
    return {
        "schema_version": 1,
        "sequential_analysis": bool(sequential_analysis),
        "parallel_batch": bool(parallel_batch),
        "seed_active": bool(seed_active),
        "user_decisions": serialize_user_decisions(user_decisions, target_language),
        "nonbinding_suggestions": serialize_nonbinding_suggestions(
            nonbinding_suggestions, target_language
        ),
        "prompts_version": int(prompts_version),
    }


@dataclass(frozen=True, slots=True)
class EffectiveNarrativeInputs:
    """Resolved inputs feeding `ContextBuilder.narrative_input_sha256` for one section."""

    accumulated: tuple[GlossaryCandidate, ...]
    user_decisions: tuple[TermConcept, ...]
    nonbinding_suggestions: tuple[TermConcept, ...]
    sequential_analysis: bool
    parallel_batch: bool
    seed_active: bool


class NarrativePassInputs:
    """Single source of truth for assembling per-section NARRATIVE pass inputs."""

    @staticmethod
    def for_section(
        document: CanonicalDocument,
        section: DocumentSection,
        *,
        analyses: Sequence[ChapterAnalysis],
        concepts: Sequence[TermConcept],
        name_translation_policy: NameTranslationPolicy | None,
        glossary_confidence_threshold: float,
        sequential_analysis: bool,
        parallel_batch: bool,
        seed_active: bool,
        verified_narratives_by_section: Mapping[str, tuple[NarrativeAnalysisResponse, str | None]] | None = None,
        fallback_conditions_id: str | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> EffectiveNarrativeInputs:
        """Compute `(accumulated, user_decisions, nonbinding_suggestions)` identically to `AnalyzeChapter._load()`."""
        current_order = section.order
        orders = {sec.id: sec.order for sec in document.sections}
        sections_by_id = {sec.id: sec for sec in document.sections}
        analyses_by_section = {analysis.section_id: analysis for analysis in analyses}

        counter = 0

        def _deterministic_id() -> str:
            nonlocal counter
            counter += 1
            return f"reconstructed-{counter}"

        make_id = id_factory or _deterministic_id

        # Assemble per-section candidates in reading order: prefer ChapterAnalysis when present,
        # otherwise reconstruct from verified full NARRATIVE passes so pedagogical failures
        # never break downstream accumulated candidates.
        section_candidates: dict[str, tuple[GlossaryCandidate, ...]] = {}
        ordered_sections = getattr(document, "ordered_sections", document.sections)
        for sec in ordered_sections:
            analysis = analyses_by_section.get(sec.id)
            if analysis is not None:
                section_candidates[sec.id] = analysis.glossary_candidates
            elif verified_narratives_by_section and sec.id in verified_narratives_by_section:
                decoded, pass_conditions_id = verified_narratives_by_section[sec.id]
                is_divider = is_meaningful_part_divider(document, sec)
                section_candidates[sec.id] = candidates_from_narrative(
                    decoded,
                    analysis_id=f"reconstructed-narrative-{sec.id}",
                    confidence_threshold=glossary_confidence_threshold,
                    id_factory=make_id,
                    name_translation_policy=name_translation_policy,
                    conditions_id=pass_conditions_id or fallback_conditions_id,
                    merge_groups=not is_divider,
                )
        # Preserve any analysis for sections not in ordered_sections (defensive parity).
        for analysis in analyses:
            if analysis.section_id not in section_candidates:
                section_candidates[analysis.section_id] = analysis.glossary_candidates

        seed_section = first_content_section(document) if (parallel_batch and seed_active) else None
        seed_section_ids: set[str] = set()
        if seed_section is not None:
            if current_order >= seed_section.order:
                seed_section_ids.update(
                    sec.id
                    for sec in ordered_sections
                    if sec.order < seed_section.order
                    and is_meaningful_part_divider(document, sec)
                )
            if seed_section.order < current_order:
                seed_section_ids.add(seed_section.id)

        ordered_sec_ids = [sec.id for sec in ordered_sections if sec.id in section_candidates]
        ordered_sec_ids.extend(
            sid for sid in section_candidates if sid not in sections_by_id
        )

        seed_candidates = (
            tuple(
                candidate
                for sid in ordered_sec_ids
                if sid in seed_section_ids
                for candidate in section_candidates[sid]
            )
            if seed_section_ids
            else ()
        )

        if name_translation_policy is not None:
            accumulated = (
                seed_candidates
                if parallel_batch
                else tuple(
                    candidate
                    for sid in ordered_sec_ids
                    if orders.get(sid, current_order) < current_order
                    for candidate in section_candidates[sid]
                )
            )
            decisions = tuple(concept for concept in concepts if concept.status.is_user_decision)
            suggestions = tuple(concept for concept in concepts if not concept.status.is_user_decision)
            return EffectiveNarrativeInputs(
                accumulated=accumulated,
                user_decisions=decisions,
                nonbinding_suggestions=suggestions,
                sequential_analysis=sequential_analysis,
                parallel_batch=parallel_batch,
                seed_active=seed_active,
            )

        if parallel_batch:
            all_candidates = tuple(
                candidate
                for sid in ordered_sec_ids
                for candidate in section_candidates[sid]
            )
            non_seed_candidates = (
                tuple(
                    candidate
                    for sid in ordered_sec_ids
                    if sid not in seed_section_ids
                    for candidate in section_candidates[sid]
                )
                if seed_section_ids
                else all_candidates
            )
            known_keys = {glossary_key(c.source_text) for c in all_candidates}
            concept_candidates = tuple(
                GlossaryCandidate.create(
                    id=concept.id,
                    analysis_id="user-decision",
                    source_text=concept.canonical_source_text,
                    type=concept.type,
                    suggested_translation=variant.text,
                    confidence=1.0,
                    model_flagged_review=False,
                    confidence_threshold=glossary_confidence_threshold,
                    note=concept.note,
                )
                for concept in concepts
                if concept.status.is_user_decision
                and glossary_key(concept.canonical_source_text) not in known_keys
                and (variant := concept.variant_for(document.target_language)) is not None
            )
            accumulated = candidates_with_decisions(
                seed_candidates,
                concepts,
                language=document.target_language,
                decided_only=(*non_seed_candidates, *concept_candidates),
            )
        else:
            accumulated = candidates_with_decisions(
                (
                    candidate
                    for sid in ordered_sec_ids
                    if orders.get(sid, current_order) < current_order
                    for candidate in section_candidates[sid]
                ),
                concepts,
                language=document.target_language,
                decided_only=(
                    candidate
                    for sid in ordered_sec_ids
                    if orders.get(sid, current_order) >= current_order
                    for candidate in section_candidates[sid]
                ),
            )
        return EffectiveNarrativeInputs(
            accumulated=accumulated,
            user_decisions=(),
            nonbinding_suggestions=(),
            sequential_analysis=sequential_analysis,
            parallel_batch=parallel_batch,
            seed_active=seed_active,
        )


__all__ = [
    "EffectiveNarrativeInputs",
    "NarrativePassInputs",
    "build_narrative_input_context",
    "deserialize_nonbinding_suggestions",
    "deserialize_user_decisions",
    "first_content_section",
    "has_meaningful_part_title",
    "is_illustrable_content_section",
    "is_meaningful_part_divider",
    "serialize_nonbinding_suggestions",
    "serialize_user_decisions",
]
