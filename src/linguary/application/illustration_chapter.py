"""Applicable appearance scoping, monochrome prompt rendering, and atomic chapter acceptance (`P3.T2`)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from linguary.application.analysis_pass_cache import encode_illustration_pass
from linguary.application.context_builder import textual_blocks
from linguary.application.illustration_context import (
    IllustrationContext,
    _collect_book_glossary_terms_ru,
    compact_safe_detail,
    fit_glossary_terms_to_wire_budget,
    is_context_limit_error,
    is_incompatible_route_error,
    is_transient_error,
    is_truncated_error,
    normalize_name_token,
    pack_items_by_wire_budget,
)
from linguary.application.illustration_dto import (
    IllustrationFactScopeResponse,
    IllustrationPromptResponse,
)
from linguary.application.illustration_reference import (
    choose_canonical_name,
    display_name_at_position as _display_name_at_position,
    resolve_appearance_with_design_fallback,
)
from linguary.application.illustration_selection import (
    IllustrationSelection,
    SelectedMomentCard,
    SelectedMoments,
    moments_conflict_diversity,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    IllustrationSourceSnapshot,
    SectionManifestEntry,
    compute_chapter_fingerprint,
)
from linguary.application.illustration_validation import (
    ChapterIllustrationPlan,
    IllustrationAppearance,
    ValidatedMoment,
    compose_monochrome_prompt,
    dedupe_texts,
    filter_non_assumption_placeholders,
    find_monochrome_color_word,
    find_speech_bubble_term,
    has_cyrillic,
    resolve_insert_after_block_id,
    validate_moment,
)
from linguary.application.illustrations import (
    dispatch_scope_task,
    finalize_generation_state_in_tx,
    revoke_active_in_transaction,
)
from linguary.application.llm_rejections import finalize_response, record_rejection
from linguary.application.ports.illustrations import IllustrationRuntimePort
from linguary.application.ports.llm import (
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
    LLMResponseKind,
)
from linguary.application.ports.persistence import UnitOfWork
from linguary.application.retry_policy import ResultValidationError, RetryPolicy
from linguary.domain import (
    AnalysisPassKind,
    AnalysisPassResult,
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskType,
    glossary_key,
)
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import SourceAnchor, canonical_sha
from linguary.domain.illustrations import (
    ILLUSTRATION_STYLE_MONOCHROME_INK,
    IllustrationAttemptState,
    IllustrationError,
    IllustrationGeneration,
    IllustrationGenerationState,
    IllustrationLimitError,
    IllustrationOwnershipError,
    IllustrationPolicy,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScope,
    IllustrationScopeSpec,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettingsRevision,
    IllustrationValidationError,
    RegistryItem,
    format_position_key,
)
from linguary.domain.llm_log import LLMCallStage

_MAX_APPLICABLE_FACTS = 32
_MAX_FAST_PATH_PERMANENT_FACTS = 4
_MAX_FAST_PATH_DESC_EN = 400
_MAX_FAST_PATH_DESC_RU = 600
_DEFAULT_ZERO_UNRESOLVED_RU = (
    "Не удалось однозначно определить образы ключевых участников для выбранных моментов главы."
)


def _parse_section_order(position_key: Any) -> int | None:
    """Parse section order prefix from canonical `SSSSSS:BBBBBB:CCCCCCCC` position key."""
    if not isinstance(position_key, str) or not position_key:
        return None
    sec_part = position_key.split(":", 1)[0]
    if not (sec_part.isascii() and sec_part.isdigit()):
        return None
    return int(sec_part)


def _default_id() -> str:
    return str(uuid4())


def _select_english_name(entity: Mapping[str, Any]) -> str | None:
    """Select the first non-Cyrillic name from `canonical_name` or sorted `aliases`."""
    candidates: list[str] = []
    raw_canonical = entity.get("canonical_name")
    canonical = str(raw_canonical).strip() if raw_canonical is not None else ""
    if canonical:
        candidates.append(canonical)
    raw_aliases = entity.get("aliases", ())
    if isinstance(raw_aliases, Iterable) and not isinstance(raw_aliases, (str, bytes)):
        candidates.extend(
            sorted(str(a).strip() for a in raw_aliases if a is not None and str(a).strip())
        )
    for name in candidates:
        if not has_cyrillic(name):
            return name
    return None


def _select_english_name_at_position(
    entity: Mapping[str, Any],
    event_position_key: str,
    display_name: str,
) -> str | None:
    """Select a spoiler-free non-Cyrillic display name visible at `event_position_key`."""
    cleaned_display = display_name.strip()
    if cleaned_display and not has_cyrillic(cleaned_display):
        return cleaned_display
    raw_positions = entity.get("alias_first_positions")
    if isinstance(raw_positions, Mapping) and raw_positions and event_position_key:
        eligible_en = [
            {"name": clean, "position_key": pos}
            for name, pos in raw_positions.items()
            if name is not None
            and (clean := str(name).strip())
            and not has_cyrillic(clean)
            and isinstance(pos, str)
            and pos
            and pos <= event_position_key
        ]
        return choose_canonical_name(eligible_en) or None
    return _select_english_name(entity)


def _resolve_scene_location_ru(
    entry: SectionManifestEntry,
    section_id: str,
    scene_id: str,
    *,
    max_chars: int = 240,
) -> str | None:
    """Lookup reference `scene.location` from `entry.scenes` when no registry location bound."""
    for idx, scene in enumerate(entry.scenes):
        if scene.id == scene_id or f"{section_id}:scene:{idx}" == scene_id:
            loc = " ".join(str(scene.location or "").split()).strip()
            return loc[:max_chars] if loc else None
    return None


def _display_name_ru_at_position(
    entity: Mapping[str, Any],
    event_position_key: str,
    display_name_en: str,
) -> str:
    """Return spoiler-free localized Russian display name for `display_name_en`."""
    _ = event_position_key
    alias_tr = entity.get("alias_translations_ru")
    if isinstance(alias_tr, Mapping):
        direct = alias_tr.get(display_name_en)
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        target_key = glossary_key(display_name_en)
        if target_key:
            for alias_name, tr in alias_tr.items():
                if (
                    isinstance(alias_name, str)
                    and isinstance(tr, str)
                    and tr.strip()
                    and glossary_key(alias_name) == target_key
                ):
                    return tr.strip()
    canonical = str(entity.get("canonical_name") or "").strip()
    canonical_tr = entity.get("glossary_translation_ru")
    if (
        canonical
        and isinstance(canonical_tr, str)
        and canonical_tr.strip()
        and (
            display_name_en == canonical
            or (
                glossary_key(display_name_en)
                and glossary_key(display_name_en) == glossary_key(canonical)
            )
        )
    ):
        return canonical_tr.strip()
    return display_name_en


def _entity_names(entity: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    raw_canonical = entity.get("canonical_name")
    if raw_canonical is not None and str(raw_canonical).strip():
        names.append(str(raw_canonical).strip())
    raw_aliases = entity.get("aliases", ())
    if isinstance(raw_aliases, Iterable) and not isinstance(raw_aliases, (str, bytes)):
        for alias in raw_aliases:
            if alias is not None and str(alias).strip():
                names.append(str(alias).strip())
    raw_tr = entity.get("glossary_translation_ru")
    if isinstance(raw_tr, str) and raw_tr.strip():
        names.append(raw_tr.strip())
    raw_alias_tr = entity.get("alias_translations_ru")
    if isinstance(raw_alias_tr, Mapping):
        for tr in raw_alias_tr.values():
            if isinstance(tr, str) and tr.strip():
                names.append(tr.strip())
    return list(dict.fromkeys(names))


def _dedupe_texts(items: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate strings by whitespace/punctuation-insensitive casefolded key preserving order."""
    return dedupe_texts(items)


def _dedupe_semicolon_parts(items: Iterable[str]) -> list[str]:
    """Split semicolon-delimited fragments and deduplicate by normalized key preserving order."""
    parts: list[str] = []
    for item in items:
        for piece in str(item).split(";"):
            cleaned = " ".join(piece.split()).strip().rstrip(".;").strip()
            if cleaned:
                parts.append(cleaned)
    return list(_dedupe_texts(parts))


def _materialize_appearances_for_prompt(
    base_chars: Sequence[IllustrationAppearance],
    base_locs: Sequence[IllustrationAppearance],
    design_traits_by_key: Mapping[str, Mapping[str, Any]] | None,
    used_design_keys: Sequence[str],
    base_fallback_entity_ids: Sequence[str] = (),
) -> tuple[tuple[IllustrationAppearance, ...], tuple[IllustrationAppearance, ...]]:
    """Materialize final appearances strictly from `used_design_keys` (with legacy summary fallback)."""
    if design_traits_by_key is None:
        return tuple(base_chars), tuple(base_locs)

    fallback_eids = {str(eid) for eid in base_fallback_entity_ids}
    traits_by_entity: dict[str, list[Mapping[str, Any]]] = {}
    for key in dict.fromkeys(str(k) for k in used_design_keys):
        trait_info = design_traits_by_key.get(key)
        if not isinstance(trait_info, Mapping):
            continue
        eid = str(trait_info.get("entity_id", ""))
        if eid:
            traits_by_entity.setdefault(eid, []).append(trait_info)

    def _materialize_one(app: IllustrationAppearance) -> IllustrationAppearance:
        selected = traits_by_entity.get(app.entity_id, [])
        if not selected:
            clean_ru = "; ".join(_dedupe_semicolon_parts([app.description_ru]))
            clean_en = "; ".join(_dedupe_semicolon_parts([app.description_en]))
            return IllustrationAppearance(
                entity_id=app.entity_id,
                name_ru=app.name_ru,
                description_ru=clean_ru or app.description_ru.strip().rstrip(".;").strip(),
                description_en=clean_en or app.description_en.strip().rstrip(".;").strip(),
                fact_ids=app.fact_ids,
                assumptions_ru=filter_non_assumption_placeholders(app.assumptions_ru),
                name_en=app.name_en,
            )
        is_fallback = app.entity_id in fallback_eids
        ru_parts: list[str] = (
            [] if is_fallback else ([app.description_ru] if app.description_ru.strip() else [])
        )
        en_parts: list[str] = (
            [] if is_fallback else ([app.description_en] if app.description_en.strip() else [])
        )
        assumps: list[str] = list(app.assumptions_ru)
        for t in selected:
            d_ru = str(t.get("description_ru", "")).strip().rstrip(".;").strip()
            d_en = str(t.get("description_en", "")).strip().rstrip(".;").strip()
            a_ru = str(t.get("assumption_ru", "")).strip()
            if d_ru:
                ru_parts.append(d_ru)
            if d_en:
                en_parts.append(d_en)
            if a_ru:
                assumps.append(a_ru)
        desc_ru_clean = _dedupe_semicolon_parts(ru_parts)
        desc_en_clean = _dedupe_semicolon_parts(en_parts)
        return IllustrationAppearance(
            entity_id=app.entity_id,
            name_ru=app.name_ru,
            description_ru="; ".join(desc_ru_clean)
            or app.description_ru.strip().rstrip(".;").strip(),
            description_en="; ".join(desc_en_clean)
            or app.description_en.strip().rstrip(".;").strip(),
            fact_ids=app.fact_ids,
            assumptions_ru=filter_non_assumption_placeholders(assumps),
            name_en=app.name_en,
        )

    return (
        tuple(_materialize_one(c) for c in base_chars),
        tuple(_materialize_one(loc) for loc in base_locs),
    )


def _load_all_registry_items(
    tx: UnitOfWork,
    generation_id: str,
    *,
    entity_id: str | None = None,
    kind: IllustrationRegistryKind | None = None,
    max_position_key: str | None = None,
    include_back_applicable: bool = False,
) -> tuple[RegistryItem, ...]:
    items: list[RegistryItem] = []
    offset = 0
    while True:
        page, total = tx.illustrations.query_registry_items(
            generation_id,
            entity_id=entity_id,
            kind=kind,
            max_position_key=max_position_key,
            include_back_applicable=include_back_applicable,
            offset=offset,
            limit=200,
        )
        items.extend(page)
        offset += len(page)
        if offset >= total or not page:
            break
    return tuple(items)


def _resolve_entity_name(
    name: str,
    *,
    expected_kind: str,
    entities: Sequence[Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any] | None]:
    """Resolve a participant/location name against registry entities without guessing.

    Two-pass deterministic matching via `normalize_name_token`:
      1. Match `canonical_name` first; if unique -> resolved, if >=2 -> ambiguous.
      2. Only if Pass 1 had 0 matches, match `aliases`; if unique -> resolved, if >=2 -> ambiguous.

    Returns:
      - `("resolved", entity_payload)` when uniquely matched
      - `("ambiguous", None)` when 2+ distinct entities match (never merged or guessed!)
      - `("unmatched", None)` when no entity in the registry matches
    """
    cleaned_name = re.sub(r"\s*\(aliases:.*\)$", "", name, flags=re.IGNORECASE).strip()
    norm = normalize_name_token(cleaned_name)
    if not norm:
        return "unmatched", None

    canonical_matches: list[Mapping[str, Any]] = []
    for ent in entities:
        if str(ent.get("entity_kind", "")) != expected_kind:
            continue
        canonical = str(ent.get("canonical_name", ""))
        if canonical and normalize_name_token(canonical) == norm:
            canonical_matches.append(ent)
    if len(canonical_matches) == 1:
        return "resolved", canonical_matches[0]
    if len(canonical_matches) > 1:
        return "ambiguous", None

    alias_matches: list[Mapping[str, Any]] = []
    for ent in entities:
        if str(ent.get("entity_kind", "")) != expected_kind:
            continue
        aliases = [str(a) for a in ent.get("aliases", ()) if a is not None]
        if any(a and normalize_name_token(a) == norm for a in aliases):
            alias_matches.append(ent)
    if len(alias_matches) == 1:
        return "resolved", alias_matches[0]
    if len(alias_matches) > 1:
        return "ambiguous", None
    return "unmatched", None


def resolve_moments_with_shortlist_fallback(
    selected: SelectedMoments,
    *,
    entities: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[SelectedMomentCard, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]], ...], str | None]:
    """Bind winner moments to registry entities; fall back to LIFO shortlist on ambiguous identity.

    - If any key participant or location on a winner has an ambiguous (multi-entity) identity,
      never guess or merge; promote the next resolvable, diversity-compatible runner-up from the
      retained LIFO `shortlist` (using its `significance_ru` as `reason_ru`).
    - If shortlist is exhausted, drop the unresolvable moment (or return 0 with `zero_reason_ru`).
    """
    available_shortlist = list(selected.shortlist)
    accepted_cards: list[SelectedMomentCard] = []
    accepted_bound: list[tuple[SelectedMomentCard, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]] = []
    rejected_ids: set[str] = set()

    def _try_bind(
        card: SelectedMomentCard,
    ) -> tuple[bool, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
        chars: list[Mapping[str, Any]] = []
        locs: list[Mapping[str, Any]] = []
        seen_c: set[str] = set()
        seen_l: set[str] = set()
        for pname in card.participant_names:
            status, ent = _resolve_entity_name(pname, expected_kind="person", entities=entities)
            if status == "ambiguous":
                return False, (), ()
            if status == "resolved" and ent is not None:
                eid = str(ent["entity_id"])
                if eid not in seen_c:
                    seen_c.add(eid)
                    chars.append(ent)
        for lname in card.location_names:
            status, ent = _resolve_entity_name(lname, expected_kind="location", entities=entities)
            if status == "ambiguous":
                return False, (), ()
            if status == "resolved" and ent is not None:
                eid = str(ent["entity_id"])
                if eid not in seen_l:
                    seen_l.add(eid)
                    locs.append(ent)
        return True, tuple(chars), tuple(locs)

    for winner in selected.winners:
        ok, chars, locs = _try_bind(winner)
        if ok:
            accepted_cards.append(winner)
            accepted_bound.append((winner, chars, locs))
            continue

        # Key identity is ambiguous -> reject winner and seek diversity-compatible replacement from LIFO shortlist!
        rejected_ids.add(winner.candidate_id)
        for idx, runner in enumerate(list(available_shortlist)):
            if runner.candidate_id in rejected_ids or any(
                runner.candidate_id == w.candidate_id for w in accepted_cards
            ):
                continue
            if any(moments_conflict_diversity(runner, existing) for existing in accepted_cards):
                continue
            r_ok, r_chars, r_locs = _try_bind(runner)
            if not r_ok:
                rejected_ids.add(runner.candidate_id)
                continue
            promoted = SelectedMomentCard(
                candidate_id=runner.candidate_id,
                section_id=runner.section_id,
                scene_id=runner.scene_id,
                title_ru=runner.title_ru,
                event_ru=runner.event_ru,
                significance_ru=runner.significance_ru,
                reason_ru=runner.significance_ru,
                event_kind=runner.event_kind,
                event_position_key=runner.event_position_key,
                anchors=runner.anchors,
                insert_after_block_id=runner.insert_after_block_id,
                participant_names=runner.participant_names,
                location_names=runner.location_names,
                scene_start_block_id=runner.scene_start_block_id,
                scene_end_block_id=runner.scene_end_block_id,
            )
            accepted_cards.append(promoted)
            accepted_bound.append((promoted, r_chars, r_locs))
            available_shortlist.pop(idx)
            break

    accepted_bound.sort(key=lambda item: (item[0].event_position_key, item[0].candidate_id))
    if not accepted_bound:
        zero_reason = (selected.zero_reason_ru or "").strip() or _DEFAULT_ZERO_UNRESOLVED_RU
        return (), zero_reason
    return tuple(accepted_bound), None


def _build_fact_scope_spec(
    *,
    section_id: str,
    section_order: int,
    section_title: str,
    scope_key: str,
    moment_card: SelectedMomentCard,
    entity: Mapping[str, Any],
    facts_page: Sequence[Mapping[str, Any]],
    page_index: int,
    total_pages: int,
    running_summary_ru: str,
    running_summary_en: str,
    running_applicable_ids: Sequence[str],
    is_consolidation: bool,
    max_applicable_fact_ids: int = _MAX_APPLICABLE_FACTS,
    glossary_terms_ru: Mapping[str, str] | None = None,
    revision: IllustrationSettingsRevision,
    route_id: str,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    policy = revision.settings.policy
    compact_facts = [
        {
            "fact_id": str(f["fact_id"]),
            "temporal_scope": str(f["temporal_scope"]),
            "position_key": str(f["position_key"]),
            "is_back_applicable": bool(f.get("is_back_applicable", False)),
            "description_ru": (
                str(f["description_ru"])[:80] if is_consolidation else str(f["description_ru"])
            ),
            "description_en": (
                str(f["description_en"])[:80] if is_consolidation else str(f["description_en"])
            ),
        }
        for f in facts_page
    ]
    event_pos = max(
        (a.position_key for a in moment_card.anchors),
        default=moment_card.event_position_key,
    )
    display_name = _display_name_at_position(entity, event_pos)
    request_input: dict[str, Any] = {
        "event_id": moment_card.candidate_id,
        "event_position_key": moment_card.event_position_key,
        "event_ru": moment_card.event_ru,
        "entity_id": str(entity["entity_id"]),
        "entity_name": display_name,
        "entity_kind": str(entity["entity_kind"]),
        "style": revision.settings.style,
        "page_index": page_index,
        "total_pages": total_pages,
        "is_consolidation": is_consolidation,
        "max_applicable_fact_ids": max_applicable_fact_ids,
        "running_summary_ru": running_summary_ru,
        "running_summary_en": running_summary_en,
        "facts": compact_facts,
    }
    if glossary_terms_ru:
        cleaned_gloss = {
            str(k): str(v)
            for k, v in glossary_terms_ru.items()
            if k is not None and str(k).strip() and v is not None and str(v).strip()
        }
        if cleaned_gloss:
            request_input["glossary_terms_ru"] = cleaned_gloss
    input_sha256 = canonical_sha(
        {
            "stage": IllustrationScopeStage.FACT_SCOPE.value,
            "request_input": request_input,
            "model_prompts": revision.settings.model_prompts,
            "route_id": route_id,
            "policy": policy.to_payload(),
        }
    )
    valid_ids = [str(f["fact_id"]) for f in facts_page]
    summary = {
        "depth": 0,
        "is_composite": len(facts_page) > 1 and not is_consolidation,
        "moment_id": moment_card.candidate_id,
        "entity_id": str(entity["entity_id"]),
        "entity_kind": str(entity["entity_kind"]),
        "entity_name": display_name,
        "entity_names": _entity_names(entity),
        "page_index": page_index,
        "total_pages": total_pages,
        "is_consolidation": is_consolidation,
        "max_applicable_fact_ids": max_applicable_fact_ids,
        "valid_fact_ids": valid_ids,
        "running_applicable_ids": list(running_applicable_ids),
        "ordered_block_ids": [a.block_id for a in moment_card.anchors],
        "request_input": request_input,
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.FACT_SCOPE,
            scope_key=scope_key,
            input_sha256=input_sha256,
            section_id=section_id,
            section_order=section_order,
            section_title=section_title,
            summary={**summary, "reused_from_scope_id": reusable.id},
            state=IllustrationScopeState.ACCEPTED,
            payload=reusable.payload,
            output_sha256=reusable.output_sha256,
        )
    return IllustrationScopeSpec(
        stage=IllustrationScopeStage.FACT_SCOPE,
        scope_key=scope_key,
        input_sha256=input_sha256,
        section_id=section_id,
        section_order=section_order,
        section_title=section_title,
        summary=summary,
        state=IllustrationScopeState.PENDING,
    )


def _build_prompt_scope_spec(
    *,
    section_id: str,
    section_order: int,
    section_title: str,
    scope_key: str,
    moment_card: SelectedMomentCard,
    characters: Sequence[IllustrationAppearance],
    locations: Sequence[IllustrationAppearance],
    allowed_fact_ids: Sequence[str],
    allowed_design_keys: Sequence[str],
    available_design_traits: Sequence[Mapping[str, Any]] = (),
    base_fallback_entity_ids: Sequence[str] = (),
    entity_names: Sequence[str] = (),
    scene_location_ru: str | None = None,
    anchor_quotes: Sequence[str] | None = None,
    glossary_terms_ru: Mapping[str, str] | None = None,
    revision: IllustrationSettingsRevision,
    route_id: str,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    policy = revision.settings.policy
    thawed_design_traits = [thaw(t) for t in available_design_traits]
    request_input: dict[str, Any] = {
        "moment_id": moment_card.candidate_id,
        "scene_id": moment_card.scene_id,
        "title_ru": moment_card.title_ru,
        "event_ru": moment_card.event_ru,
        "reason_ru": moment_card.reason_ru,
        "style": revision.settings.style,
        "characters": [c.to_dict() for c in characters],
        "locations": [loc.to_dict() for loc in locations],
        "allowed_fact_ids": list(allowed_fact_ids),
        "allowed_design_keys": list(allowed_design_keys),
        "available_design_traits": thawed_design_traits,
    }
    if not locations and scene_location_ru and scene_location_ru.strip():
        request_input["scene_location_ru"] = scene_location_ru.strip()[:240]
    raw_quotes = (
        [a.quote for a in moment_card.anchors if a.quote and a.quote.strip()]
        if anchor_quotes is None
        else [str(q) for q in anchor_quotes if q is not None and str(q).strip()]
    )
    cleaned_quotes = [q.strip()[:240] for q in raw_quotes if q.strip()][:3]
    if cleaned_quotes:
        request_input["anchor_quotes"] = cleaned_quotes
    if glossary_terms_ru:
        cleaned_gloss = {
            str(k): str(v)
            for k, v in glossary_terms_ru.items()
            if k is not None and str(k).strip() and v is not None and str(v).strip()
        }
        if cleaned_gloss:
            request_input["glossary_terms_ru"] = cleaned_gloss
    input_sha256 = canonical_sha(
        {
            "stage": IllustrationScopeStage.PROMPT.value,
            "request_input": request_input,
            "model_prompts": revision.settings.model_prompts,
            "route_id": route_id,
            "style": revision.settings.style,
            "policy": policy.to_payload(),
        }
    )
    summary = {
        "depth": 0,
        "is_composite": False,
        "moment_id": moment_card.candidate_id,
        "moment_card": moment_card.to_dict(),
        "characters": [c.to_dict() for c in characters],
        "locations": [loc.to_dict() for loc in locations],
        "allowed_fact_ids": list(allowed_fact_ids),
        "allowed_design_keys": list(allowed_design_keys),
        "design_traits_by_key": {
            str(t["key"]): t for t in thawed_design_traits if t.get("key")
        },
        "base_fallback_entity_ids": list(dict.fromkeys(str(eid) for eid in base_fallback_entity_ids)),
        "entity_names": list(
            dict.fromkeys(str(n).strip() for n in entity_names if n is not None and str(n).strip())
        ),
        "ordered_block_ids": [a.block_id for a in moment_card.anchors],
        "request_input": request_input,
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.PROMPT,
            scope_key=scope_key,
            input_sha256=input_sha256,
            section_id=section_id,
            section_order=section_order,
            section_title=section_title,
            summary={**summary, "reused_from_scope_id": reusable.id},
            state=IllustrationScopeState.ACCEPTED,
            payload=reusable.payload,
            output_sha256=reusable.output_sha256,
        )
    return IllustrationScopeSpec(
        stage=IllustrationScopeStage.PROMPT,
        scope_key=scope_key,
        input_sha256=input_sha256,
        section_id=section_id,
        section_order=section_order,
        section_title=section_title,
        summary=summary,
        state=IllustrationScopeState.PENDING,
    )


@dataclass(frozen=True, slots=True)
class _ValidationOutcome:
    ok: bool
    payload: Mapping[str, Any] | None = None
    reason_code: str = IllustrationReasonCode.MODEL_OUTPUT_INVALID.value
    repair_reason: str = ""


class IllustrationChapter:
    """Orchestrates chapter moment selection, chronological fact scoping, prompt rendering, and acceptance."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        *,
        source: IllustrationSource,
        runtime: IllustrationRuntimePort,
        id_factory: Callable[[], str] | None = None,
        notify_work_available: Callable[[], None] | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep_fn: Callable[[float], Any] | None = None,
    ) -> None:
        self._uow = unit_of_work
        self._source = source
        self._runtime = runtime
        self._id_factory = id_factory or _default_id
        self._notify_work_available = notify_work_available
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=3)
        self._sleep = sleep_fn or asyncio.sleep
        self._selection = IllustrationSelection(
            unit_of_work,
            source=source,
            runtime=runtime,
            id_factory=self._id_factory,
            notify_work_available=notify_work_available,
            retry_policy=self._retry_policy,
            sleep_fn=self._sleep,
            on_selection_ready=self._on_selection_ready,
        )

    @property
    def selection(self) -> IllustrationSelection:
        return self._selection

    def plan_chapter_scopes(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        snapshot: IllustrationSourceSnapshot,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> bool:
        """Plan `CHAPTER` scopes and start candidate extraction for requested chapters."""
        factory = id_factory or self._id_factory
        if generation.registry_sha256 is None:
            return False
        revision = tx.illustrations.get_settings(generation.book_id, generation.settings_id)
        if revision is None or not revision.settings.enabled:
            return False
        if not revision.settings.model_prompts:
            return False

        route_id = self._runtime.route_id()
        requested_set = set(generation.requested_sections)
        existing_chapters = {
            s.section_id: s
            for s in tx.illustrations.list_scopes(generation.id, stage=IllustrationScopeStage.CHAPTER)
            if s.section_id is not None
        }

        chapter_specs: list[IllustrationScopeSpec] = []
        reused_mirrors: list[tuple[str, str, Mapping[str, Any], str]] = []
        to_plan_sections: list[str] = []

        for entry in snapshot.entries:
            if entry.section_id in existing_chapters:
                ch_scope = existing_chapters[entry.section_id]
                if (
                    not entry.skipped
                    and entry.section_id in requested_set
                    and ch_scope.state is IllustrationScopeState.PENDING
                ):
                    to_plan_sections.append(entry.section_id)
                continue

            if entry.skipped:
                chapter_specs.append(
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key=f"chapter:{entry.section_id}",
                        input_sha256=canonical_sha(
                            {"section_id": entry.section_id, "skipped": True}
                        ),
                        section_id=entry.section_id,
                        section_order=entry.order,
                        section_title=entry.title,
                        state=IllustrationScopeState.SKIPPED,
                        reason_code=entry.skip_reason
                        or IllustrationReasonCode.NON_CONTENT_SECTION.value,
                        summary={"skipped": True},
                    )
                )
                continue

            if entry.section_id not in requested_set:
                continue

            chapter_fp = compute_chapter_fingerprint(
                entry,
                registry_sha256=generation.registry_sha256,
                policy_payload=revision.settings.policy.to_payload(),
                model_prompts=revision.settings.model_prompts,
                route_id=route_id,
                style=revision.settings.style,
                max_per_chapter=revision.settings.max_per_chapter,
            )
            reusable = tx.illustrations.find_reusable_scope(
                generation.book_id,
                stage=IllustrationScopeStage.CHAPTER,
                input_sha256=chapter_fp,
            )
            if (
                reusable is not None
                and reusable.payload is not None
                and reusable.output_sha256 is not None
            ):
                chapter_specs.append(
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key=f"chapter:{entry.section_id}",
                        input_sha256=chapter_fp,
                        section_id=entry.section_id,
                        section_order=entry.order,
                        section_title=entry.title,
                        state=IllustrationScopeState.PENDING,
                        summary={"reusable_scope_id": reusable.id},
                    )
                )
                reused_mirrors.append(
                    (entry.section_id, chapter_fp, reusable.payload, reusable.output_sha256)
                )
            else:
                chapter_specs.append(
                    IllustrationScopeSpec(
                        stage=IllustrationScopeStage.CHAPTER,
                        scope_key=f"chapter:{entry.section_id}",
                        input_sha256=chapter_fp,
                        section_id=entry.section_id,
                        section_order=entry.order,
                        section_title=entry.title,
                        state=IllustrationScopeState.PENDING,
                        summary={},
                    )
                )
                to_plan_sections.append(entry.section_id)

        if chapter_specs:
            tx.illustrations.upsert_scopes(generation.id, chapter_specs)

        # Accept reused chapters atomically with fresh pass_mirror for this generation!
        if reused_mirrors:
            chapters_by_sec = {
                s.section_id: s
                for s in tx.illustrations.list_scopes(
                    generation.id, stage=IllustrationScopeStage.CHAPTER
                )
                if s.section_id is not None
            }
            for sec_id, fp, reused_payload, reused_out_sha in reused_mirrors:
                ch_scope = chapters_by_sec.get(sec_id)
                if ch_scope is None:
                    continue
                plan = ChapterIllustrationPlan.from_dict(
                    {**thaw(reused_payload), "generation_id": generation.id}
                )
                mirror_payload = encode_illustration_pass(plan)
                mirror = (
                    AnalysisPassResult(
                        id=factory(),
                        book_id=generation.book_id,
                        section_id=sec_id,
                        pass_kind=AnalysisPassKind.ILLUSTRATION_PREP,
                        scope_key=generation.id,
                        input_sha256=fp,
                        output_sha256=canonical_sha(mirror_payload),
                        conditions_id=snapshot.conditions_id,
                        attempt_id=None,
                        payload=mirror_payload,
                    )
                    if snapshot.conditions_id is not None
                    else None
                )
                tx.illustrations.accept_scope(
                    ch_scope.id,
                    attempt_id=None,
                    expected_input_sha256=fp,
                    payload=plan.to_dict(),
                    output_sha256=reused_out_sha,
                    summary={"moment_count": len(plan.moments), "reused": True},
                    pass_mirror=mirror,
                )

        dispatched = False
        for sec_id in to_plan_sections:
            _, sec_dispatched = self._selection.plan_scopes_in_tx(
                tx, generation, revision, snapshot, sec_id, id_factory=factory
            )
            if sec_dispatched:
                dispatched = True

        self._maybe_mark_generation_ready(tx, generation)
        finalize_generation_state_in_tx(tx, generation.book_id, generation.id)
        return dispatched

    async def run_task(self, task: ProcessingTask) -> None:
        """Execute one `PREPARE_CHAPTER_ILLUSTRATIONS` scope task (`CANDIDATES`/`SELECTION`/`FACT_SCOPE`/`PROMPT`)."""
        scope_id = task.scope_id
        if not scope_id:
            return
        with self._uow.transaction() as tx:
            scope = tx.illustrations.get_scope(scope_id)
        if scope is None:
            return
        if scope.stage in {IllustrationScopeStage.CANDIDATES, IllustrationScopeStage.SELECTION}:
            await self._selection.run_task(task)
            return
        if scope.stage in {IllustrationScopeStage.FACT_SCOPE, IllustrationScopeStage.PROMPT}:
            await self._run_render_scope_task(task)
            return

    def _on_selection_ready(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        snapshot: IllustrationSourceSnapshot,
        section_id: str,
    ) -> bool:
        """Triggered inside TX3 when `ALG-03` finalizes `SelectedMoments` for `section_id`."""
        return self._advance_chapter_rendering(
            tx, generation, revision, snapshot, section_id, id_factory=self._id_factory
        )

    def _advance_chapter_rendering(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        snapshot: IllustrationSourceSnapshot,
        section_id: str,
        *,
        id_factory: Callable[[], str],
    ) -> bool:
        """Advance `SelectedMoments -> FACT_SCOPE (chronological folding) -> PROMPT -> CHAPTER`."""
        chapter_scopes = tx.illustrations.list_scopes(
            generation.id, stage=IllustrationScopeStage.CHAPTER, section_id=section_id
        )
        if not chapter_scopes:
            return False
        chapter_scope = chapter_scopes[0]
        if chapter_scope.state is IllustrationScopeState.ACCEPTED:
            self._maybe_mark_generation_ready(tx, generation)
            return False

        entry = next((e for e in snapshot.entries if e.section_id == section_id), None)
        if entry is None or entry.skipped:
            return False
        document = tx.documents.get(generation.book_id)
        if document is None:
            return False
        section = next((s for s in document.ordered_sections if s.id == section_id), None)
        if section is None:
            return False

        selected = self._selection.selected_moments_in_tx(tx, generation.id, section_id)
        entity_items = _load_all_registry_items(
            tx, generation.id, kind=IllustrationRegistryKind.ENTITY
        )
        entities = [item.raw_payload() for item in entity_items]
        bound_moments, zero_reason = resolve_moments_with_shortlist_fallback(
            selected, entities=entities
        )

        if not bound_moments:
            plan = ChapterIllustrationPlan(
                section_id=section_id,
                generation_id=generation.id,
                style=revision.settings.style,
                moments=(),
                zero_reason_ru=zero_reason or _DEFAULT_ZERO_UNRESOLVED_RU,
            )
            self._accept_chapter_plan(
                tx, generation, snapshot, chapter_scope, plan, id_factory=id_factory
            )
            return False

        # Step 1: Chronological paginated FACT_SCOPE for every (moment, entity) pair that has registry facts.
        policy = revision.settings.policy
        sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_CHAPTER)
        route_id = self._runtime.route_id()
        moment_priority_keys: set[str] = set()
        for _card, _chars, _locs in bound_moments:
            for ent in (*_chars, *_locs):
                for raw_name in _entity_names(ent):
                    k = glossary_key(raw_name)
                    if k:
                        moment_priority_keys.add(k)
        chapter_glossary_terms_ru = _collect_book_glossary_terms_ru(
            tx,
            generation.book_id,
            generation_id=generation.id,
            max_section_order=entry.order,
            priority_keys=frozenset(moment_priority_keys),
        )
        block_order_by_id = {b.id: idx for idx, b in enumerate(textual_blocks(section))}

        fact_scopes_all = tx.illustrations.list_scopes(
            generation.id, stage=IllustrationScopeStage.FACT_SCOPE, section_id=section_id
        )
        fact_scopes_by_key = {s.scope_key: s for s in fact_scopes_all}

        # Preload design items & per-(moment, entity) facts via SQL query_registry_items!
        dispatched = False
        pending_fact_scopes = False
        scoped_appearances_by_moment: dict[
            str,
            tuple[
                list[IllustrationAppearance],
                list[IllustrationAppearance],
                list[str],
                list[str],
                list[dict[str, Any]],
                list[str],
                list[str],
            ],
        ] = {}
        all_registry_facts: dict[str, dict[str, Any]] = {}

        for card, char_entities, loc_entities in bound_moments:
            event_pos = max(a.position_key for a in card.anchors)
            start_idx = (
                block_order_by_id.get(card.scene_start_block_id)
                if card.scene_start_block_id
                else None
            )
            scene_start_pos = format_position_key(
                entry.order, start_idx if start_idx is not None else 0, 0
            )
            char_apps: list[IllustrationAppearance] = []
            loc_apps: list[IllustrationAppearance] = []
            allowed_fids: list[str] = []
            allowed_dkeys: list[str] = []
            available_design_traits: list[dict[str, Any]] = []
            seen_design_keys: set[str] = set()
            base_fallback_eids: list[str] = []
            entity_names_all: list[str] = []

            for kind_bucket, ent_list in (("person", char_entities), ("location", loc_entities)):
                for ent in ent_list:
                    eid = str(ent["entity_id"])
                    entity_names_all.extend(_entity_names(ent))
                    # STRICT SQL chronological + back-applicable non-spoiler filter!
                    fact_items = _load_all_registry_items(
                        tx,
                        generation.id,
                        entity_id=eid,
                        kind=IllustrationRegistryKind.FACT,
                        max_position_key=event_pos,
                        include_back_applicable=True,
                    )
                    entity_facts = sorted(
                        [
                            fp
                            for fi in fact_items
                            if (fp := fi.raw_payload()) is not None
                            and (
                                str(fp.get("temporal_scope", "")) != "episodic"
                                or (scene_start_pos <= str(fp.get("position_key", "")) <= event_pos)
                            )
                        ],
                        key=lambda f: (str(f["position_key"]), str(f["fact_id"])),
                    )
                    for ef in entity_facts:
                        all_registry_facts[str(ef["fact_id"])] = ef

                    design_items = _load_all_registry_items(
                        tx,
                        generation.id,
                        entity_id=eid,
                        kind=IllustrationRegistryKind.DESIGN,
                    )
                    design_payloads = [di.raw_payload() for di in design_items]

                    display_name = _display_name_at_position(ent, event_pos)
                    display_name_ru = _display_name_ru_at_position(ent, event_pos, display_name)
                    eng_name = _select_english_name_at_position(ent, event_pos, display_name)
                    fallback_en = eng_name or (
                        "character" if kind_bucket == "person" else "location"
                    )

                    if not entity_facts:
                        # Keep DESIGN traits separate until PROMPT selects `used_design_keys`.
                        base_fallback_eids.append(eid)
                        for d in design_payloads:
                            dkey = f"{eid}:{d['trait']}"
                            if dkey in seen_design_keys:
                                continue
                            seen_design_keys.add(dkey)
                            allowed_dkeys.append(dkey)
                            available_design_traits.append(
                                {
                                    "key": dkey,
                                    "entity_id": eid,
                                    "entity_kind": kind_bucket,
                                    "trait": str(d["trait"]),
                                    "description_ru": str(d["description_ru"]),
                                    "description_en": str(d["description_en"]),
                                    "assumption_ru": str(d.get("assumption_ru", "")),
                                }
                            )
                        app = IllustrationAppearance(
                            entity_id=eid,
                            name_ru=display_name_ru,
                            description_ru=display_name_ru,
                            description_en=fallback_en,
                            fact_ids=(),
                            assumptions_ru=(),
                            name_en=eng_name,
                        )
                        (char_apps if kind_bucket == "person" else loc_apps).append(app)
                        continue

                    # Deterministic fast-path for clean permanent facts (skips redundant FACT_SCOPE LLM call):
                    # - 1 permanent fact (including back-applicable) without Cyrillic/color violations
                    # - 2..4 local permanent facts (section_order <= entry.order) within compact length bounds
                    if 1 <= len(entity_facts) <= _MAX_FAST_PATH_PERMANENT_FACTS:
                        ent_names_for_mono = _entity_names(ent)
                        fast_path_ok = False
                        joined_en = ""
                        joined_ru = ""
                        if len(entity_facts) == 1:
                            single_fact = entity_facts[0]
                            joined_en = str(single_fact.get("description_en", "")).strip()
                            joined_ru = str(single_fact.get("description_ru", "")).strip()
                            mono_ok = (
                                revision.settings.style != ILLUSTRATION_STYLE_MONOCHROME_INK
                                or (
                                    find_monochrome_color_word(
                                        joined_en,
                                        allowed_names=ent_names_for_mono,
                                        strict_palette=False,
                                    )
                                    is None
                                    and find_monochrome_color_word(
                                        joined_ru,
                                        allowed_names=ent_names_for_mono,
                                        strict_palette=False,
                                    )
                                    is None
                                )
                            )
                            fast_path_ok = bool(
                                str(single_fact.get("temporal_scope", "")) == "permanent"
                                and joined_en
                                and not has_cyrillic(joined_en)
                                and joined_ru
                                and mono_ok
                            )
                        else:
                            en_parts: list[str] = []
                            ru_parts: list[str] = []
                            all_local_perm = True
                            for f_item in entity_facts:
                                f_sec = _parse_section_order(f_item.get("position_key"))
                                d_en = str(f_item.get("description_en", "")).strip()
                                d_ru = str(f_item.get("description_ru", "")).strip()
                                if (
                                    str(f_item.get("temporal_scope", "")) != "permanent"
                                    or f_sec is None
                                    or f_sec > entry.order
                                    or not d_en
                                    or has_cyrillic(d_en)
                                    or not d_ru
                                ):
                                    all_local_perm = False
                                    break
                                en_parts.append(d_en)
                                ru_parts.append(d_ru)
                            if all_local_perm:
                                joined_en = "; ".join(dict.fromkeys(en_parts))
                                joined_ru = "; ".join(dict.fromkeys(ru_parts))
                                mono_ok = (
                                    revision.settings.style != ILLUSTRATION_STYLE_MONOCHROME_INK
                                    or (
                                        find_monochrome_color_word(
                                            joined_en,
                                            allowed_names=ent_names_for_mono,
                                            strict_palette=False,
                                        )
                                        is None
                                        and find_monochrome_color_word(
                                            joined_ru,
                                            allowed_names=ent_names_for_mono,
                                            strict_palette=False,
                                        )
                                        is None
                                    )
                                )
                                fast_path_ok = bool(
                                    joined_en
                                    and joined_ru
                                    and len(joined_en) <= _MAX_FAST_PATH_DESC_EN
                                    and len(joined_ru) <= _MAX_FAST_PATH_DESC_RU
                                    and mono_ok
                                )
                        if fast_path_ok:
                            _, kept_design = resolve_appearance_with_design_fallback(
                                entity_facts,
                                design_payloads,
                                excluded_facts=(),
                                compact_description_en=joined_en,
                                compact_description_ru=joined_ru,
                            )
                            for d in kept_design:
                                dkey = f"{eid}:{d['trait']}"
                                if dkey in seen_design_keys:
                                    continue
                                seen_design_keys.add(dkey)
                                allowed_dkeys.append(dkey)
                                available_design_traits.append(
                                    {
                                        "key": dkey,
                                        "entity_id": eid,
                                        "entity_kind": kind_bucket,
                                        "trait": str(d["trait"]),
                                        "description_ru": str(d["description_ru"]),
                                        "description_en": str(d["description_en"]),
                                        "assumption_ru": str(d.get("assumption_ru", "")),
                                    }
                                )
                            fids_tuple = tuple(str(f["fact_id"]) for f in entity_facts)
                            allowed_fids.extend(fids_tuple)
                            app = IllustrationAppearance(
                                entity_id=eid,
                                name_ru=display_name_ru,
                                description_ru=joined_ru,
                                description_en=joined_en,
                                fact_ids=fids_tuple,
                                assumptions_ru=(),
                                name_en=eng_name,
                            )
                            (char_apps if kind_bucket == "person" else loc_apps).append(app)
                            continue

                    # Pack entity_facts chronologically into bounded pages for FACT_SCOPE folding!
                    def _make_fs_probe(
                        f_page: Sequence[Mapping[str, Any]],
                        cur_glossary: Mapping[str, str],
                        _card: SelectedMomentCard = card,
                        _ent: Mapping[str, Any] = ent,
                        _display_name: str = display_name,
                    ) -> LLMRequest[Any]:
                        fs_payload: dict[str, Any] = {
                            "event_id": _card.candidate_id,
                            "event_position_key": _card.event_position_key,
                            "event_ru": _card.event_ru,
                            "entity_id": str(_ent["entity_id"]),
                            "entity_name": _display_name,
                            "entity_kind": str(_ent["entity_kind"]),
                            "style": revision.settings.style,
                            "page_index": 0,
                            "total_pages": 1,
                            "is_consolidation": False,
                            "max_applicable_fact_ids": _MAX_APPLICABLE_FACTS,
                            "running_summary_ru": "x" * 600,
                            "running_summary_en": "x" * 600,
                            "facts": [
                                {
                                    "fact_id": str(f["fact_id"]),
                                    "temporal_scope": str(f["temporal_scope"]),
                                    "position_key": str(f["position_key"]),
                                    "is_back_applicable": bool(
                                        f.get("is_back_applicable", False)
                                    ),
                                    "description_ru": str(f["description_ru"]),
                                    "description_en": str(f["description_en"]),
                                }
                                for f in f_page
                            ],
                        }
                        if cur_glossary:
                            fs_payload["glossary_terms_ru"] = dict(cur_glossary)
                        return IllustrationContext.build_for_kind(
                            kind=LLMResponseKind.ILLUSTRATION_FACT_SCOPE,
                            stage=IllustrationScopeStage.FACT_SCOPE,
                            input_payload=fs_payload,
                            book_id=snapshot.book_id,
                            conditions_id=snapshot.conditions_id,
                            policy=policy,
                        )

                    bounded_fs_glossary: dict[str, str] = {}
                    if chapter_glossary_terms_ru:
                        heaviest_fact = max(
                            entity_facts,
                            key=lambda f: (
                                len(str(f.get("description_ru", "")).encode("utf-8"))
                                + len(str(f.get("description_en", "")).encode("utf-8"))
                            ),
                        )
                        bounded_fs_glossary = fit_glossary_terms_to_wire_budget(
                            chapter_glossary_terms_ru,
                            build_request=lambda g, _hf=heaviest_fact: _make_fs_probe([_hf], g),
                            sizer=sizer,
                            target_bytes=policy.initial_target_bytes,
                        )

                    def _build_fs_probe(
                        f_page: Sequence[Mapping[str, Any]],
                        _gloss: Mapping[str, str] = bounded_fs_glossary,
                    ) -> LLMRequest[Any]:
                        return _make_fs_probe(f_page, _gloss)

                    fact_pages = pack_items_by_wire_budget(
                        entity_facts,
                        build_request=_build_fs_probe,
                        sizer=sizer,
                        target_bytes=policy.initial_target_bytes,
                    )

                    # Sequential chronological folding across pages!
                    running_ru = ""
                    running_en = ""
                    accumulated_app_ids: list[str] = []
                    accumulated_excluded: set[str] = set()
                    accumulated_assumps: list[str] = []
                    folding_done = True

                    for p_idx, f_page in enumerate(fact_pages):
                        fkey = f"fact_scope:{section_id}:{card.candidate_id}:{eid}:p{p_idx:03d}"
                        existing_fs = fact_scopes_by_key.get(fkey)
                        if existing_fs is None:
                            draft = _build_fact_scope_spec(
                                section_id=section_id,
                                section_order=entry.order,
                                section_title=entry.title,
                                scope_key=fkey,
                                moment_card=card,
                                entity=ent,
                                facts_page=f_page,
                                page_index=p_idx,
                                total_pages=len(fact_pages),
                                running_summary_ru=running_ru,
                                running_summary_en=running_en,
                                running_applicable_ids=accumulated_app_ids,
                                is_consolidation=False,
                                glossary_terms_ru=bounded_fs_glossary,
                                revision=revision,
                                route_id=route_id,
                            )
                            reusable = tx.illustrations.find_reusable_scope(
                                generation.book_id,
                                stage=IllustrationScopeStage.FACT_SCOPE,
                                input_sha256=draft.input_sha256,
                            )
                            if reusable is not None:
                                draft = _build_fact_scope_spec(
                                    section_id=section_id,
                                    section_order=entry.order,
                                    section_title=entry.title,
                                    scope_key=fkey,
                                    moment_card=card,
                                    entity=ent,
                                    facts_page=f_page,
                                    page_index=p_idx,
                                    total_pages=len(fact_pages),
                                    running_summary_ru=running_ru,
                                    running_summary_en=running_en,
                                    running_applicable_ids=accumulated_app_ids,
                                    is_consolidation=False,
                                    glossary_terms_ru=bounded_fs_glossary,
                                    revision=revision,
                                    route_id=route_id,
                                    reusable=reusable,
                                )
                            created = tx.illustrations.upsert_scopes(generation.id, [draft])
                            existing_fs = created[0]
                            fact_scopes_by_key[fkey] = existing_fs
                            if existing_fs.state is IllustrationScopeState.PENDING:
                                dispatch_scope_task(
                                    tx,
                                    existing_fs,
                                    generation_id=generation.id,
                                    task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                                    id_factory=id_factory,
                                    book_id=generation.book_id,
                                )
                                dispatched = True
                        if existing_fs.state is not IllustrationScopeState.ACCEPTED:
                            folding_done = False
                            pending_fact_scopes = True
                            break

                        fs_pay = existing_fs.payload or {}
                        running_ru = str(fs_pay.get("compact_description_ru", ""))
                        running_en = str(fs_pay.get("compact_description_en", ""))
                        for ex_id in fs_pay.get("excluded_fact_ids", ()):
                            accumulated_excluded.add(str(ex_id))
                        if accumulated_excluded:
                            accumulated_app_ids = [
                                aid for aid in accumulated_app_ids if aid not in accumulated_excluded
                            ]
                        for app_id in fs_pay.get("applicable_fact_ids", ()):
                            aid_str = str(app_id)
                            if aid_str not in accumulated_excluded and aid_str not in accumulated_app_ids:
                                accumulated_app_ids.append(aid_str)
                        for a_ru in fs_pay.get("assumptions_ru", ()):
                            if str(a_ru).strip():
                                accumulated_assumps.append(str(a_ru).strip())

                    if not folding_done:
                        continue

                    # Cap accumulated_app_ids to <=32 via wire-budgeted consolidation FACT_SCOPE rounds!
                    consol_round = 0
                    consol_pending = False
                    while len(accumulated_app_ids) > _MAX_APPLICABLE_FACTS:
                        consol_facts = [
                            all_registry_facts[fid]
                            for fid in accumulated_app_ids
                            if fid in all_registry_facts
                        ]

                        def _make_consol_probe(
                            c_page: Sequence[Mapping[str, Any]],
                            cur_glossary: Mapping[str, str],
                            *,
                            _card: SelectedMomentCard = card,
                            _eid: str = eid,
                            _ent: Mapping[str, Any] = ent,
                            _display_name: str = display_name,
                        ) -> LLMRequest[Any]:
                            c_payload: dict[str, Any] = {
                                "event_id": _card.candidate_id,
                                "event_position_key": _card.event_position_key,
                                "event_ru": _card.event_ru,
                                "entity_id": _eid,
                                "entity_name": _display_name,
                                "entity_kind": str(_ent["entity_kind"]),
                                "style": revision.settings.style,
                                "page_index": 0,
                                "total_pages": 1,
                                "is_consolidation": True,
                                "max_applicable_fact_ids": _MAX_APPLICABLE_FACTS,
                                "running_summary_ru": "x" * 600,
                                "running_summary_en": "x" * 600,
                                "facts": [
                                    {
                                        "fact_id": str(f["fact_id"]),
                                        "temporal_scope": str(f["temporal_scope"]),
                                        "position_key": str(f["position_key"]),
                                        "is_back_applicable": bool(
                                            f.get("is_back_applicable", False)
                                        ),
                                        "description_ru": str(f["description_ru"])[:80],
                                        "description_en": str(f["description_en"])[:80],
                                    }
                                    for f in c_page
                                ],
                            }
                            if cur_glossary:
                                c_payload["glossary_terms_ru"] = dict(cur_glossary)
                            return IllustrationContext.build_for_kind(
                                kind=LLMResponseKind.ILLUSTRATION_FACT_SCOPE,
                                stage=IllustrationScopeStage.FACT_SCOPE,
                                input_payload=c_payload,
                                book_id=snapshot.book_id,
                                conditions_id=snapshot.conditions_id,
                                policy=policy,
                            )

                        bounded_consol_glossary: dict[str, str] = {}
                        if chapter_glossary_terms_ru and consol_facts:
                            heaviest_cf = max(
                                consol_facts,
                                key=lambda f: (
                                    len(str(f.get("description_ru", ""))[:80].encode("utf-8"))
                                    + len(str(f.get("description_en", ""))[:80].encode("utf-8"))
                                ),
                            )
                            bounded_consol_glossary = fit_glossary_terms_to_wire_budget(
                                chapter_glossary_terms_ru,
                                build_request=lambda g, _hf=heaviest_cf: _make_consol_probe(
                                    [_hf], g
                                ),
                                sizer=sizer,
                                target_bytes=policy.initial_target_bytes,
                            )

                        def _build_consol_probe(
                            c_page: Sequence[Mapping[str, Any]],
                            _gloss: Mapping[str, str] = bounded_consol_glossary,
                        ) -> LLMRequest[Any]:
                            return _make_consol_probe(c_page, _gloss)

                        consol_pages = pack_items_by_wire_budget(
                            consol_facts,
                            build_request=_build_consol_probe,
                            sizer=sizer,
                            target_bytes=policy.initial_target_bytes,
                        )
                        per_page_cap = (
                            _MAX_APPLICABLE_FACTS
                            if len(consol_pages) == 1
                            else max(1, _MAX_APPLICABLE_FACTS // len(consol_pages))
                        )
                        round_app_ids: list[str] = []
                        round_assumps: list[str] = []
                        round_ru = running_ru
                        round_en = running_en
                        for c_idx, c_page in enumerate(consol_pages):
                            consol_key = (
                                f"fact_scope:{section_id}:{card.candidate_id}:{eid}:consolidate"
                                if consol_round == 0 and len(consol_pages) == 1
                                else f"fact_scope:{section_id}:{card.candidate_id}:{eid}:consolidate:r{consol_round}:p{c_idx:03d}"
                            )
                            page_fids = [str(f["fact_id"]) for f in c_page]
                            existing_consol = fact_scopes_by_key.get(consol_key)
                            if existing_consol is None:
                                draft = _build_fact_scope_spec(
                                    section_id=section_id,
                                    section_order=entry.order,
                                    section_title=entry.title,
                                    scope_key=consol_key,
                                    moment_card=card,
                                    entity=ent,
                                    facts_page=c_page,
                                    page_index=c_idx,
                                    total_pages=len(consol_pages),
                                    running_summary_ru=round_ru,
                                    running_summary_en=round_en,
                                    running_applicable_ids=page_fids,
                                    is_consolidation=True,
                                    max_applicable_fact_ids=per_page_cap,
                                    glossary_terms_ru=bounded_consol_glossary,
                                    revision=revision,
                                    route_id=route_id,
                                )
                                reusable = tx.illustrations.find_reusable_scope(
                                    generation.book_id,
                                    stage=IllustrationScopeStage.FACT_SCOPE,
                                    input_sha256=draft.input_sha256,
                                )
                                if reusable is not None:
                                    draft = _build_fact_scope_spec(
                                        section_id=section_id,
                                        section_order=entry.order,
                                        section_title=entry.title,
                                        scope_key=consol_key,
                                        moment_card=card,
                                        entity=ent,
                                        facts_page=c_page,
                                        page_index=c_idx,
                                        total_pages=len(consol_pages),
                                        running_summary_ru=round_ru,
                                        running_summary_en=round_en,
                                        running_applicable_ids=page_fids,
                                        is_consolidation=True,
                                        max_applicable_fact_ids=per_page_cap,
                                        glossary_terms_ru=bounded_consol_glossary,
                                        revision=revision,
                                        route_id=route_id,
                                        reusable=reusable,
                                    )
                                created = tx.illustrations.upsert_scopes(generation.id, [draft])
                                existing_consol = created[0]
                                fact_scopes_by_key[consol_key] = existing_consol
                                if existing_consol.state is IllustrationScopeState.PENDING:
                                    dispatch_scope_task(
                                        tx,
                                        existing_consol,
                                        generation_id=generation.id,
                                        task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                                        id_factory=id_factory,
                                        book_id=generation.book_id,
                                    )
                                    dispatched = True
                            if existing_consol.state is not IllustrationScopeState.ACCEPTED:
                                consol_pending = True
                                pending_fact_scopes = True
                                break
                            c_pay = existing_consol.payload or {}
                            for ex_id in c_pay.get("excluded_fact_ids", ()):
                                accumulated_excluded.add(str(ex_id))
                            if accumulated_excluded:
                                round_app_ids = [
                                    aid for aid in round_app_ids if aid not in accumulated_excluded
                                ]
                            for fid in c_pay.get("applicable_fact_ids", ()):
                                fid_s = str(fid)
                                if fid_s not in accumulated_excluded and fid_s not in round_app_ids:
                                    round_app_ids.append(fid_s)
                            round_ru = str(c_pay.get("compact_description_ru", round_ru))
                            round_en = str(c_pay.get("compact_description_en", round_en))
                            for x in c_pay.get("assumptions_ru", ()):
                                if str(x).strip():
                                    round_assumps.append(str(x).strip())

                        if consol_pending:
                            break
                        prev_len = len(accumulated_app_ids)
                        accumulated_app_ids = round_app_ids
                        running_ru = round_ru
                        running_en = round_en
                        accumulated_assumps = round_assumps
                        consol_round += 1
                        if len(accumulated_app_ids) >= prev_len:
                            accumulated_app_ids = accumulated_app_ids[:_MAX_APPLICABLE_FACTS]
                            break

                    if consol_pending:
                        continue

                    # Keep non-conflicting DESIGN fallbacks separate until PROMPT selects `used_design_keys`!
                    applicable_fact_objs = [
                        all_registry_facts[fid]
                        for fid in accumulated_app_ids
                        if fid in all_registry_facts
                    ]
                    excluded_fact_objs = [
                        all_registry_facts[fid]
                        for fid in accumulated_excluded
                        if fid in all_registry_facts
                    ]
                    _, kept_design = resolve_appearance_with_design_fallback(
                        applicable_fact_objs,
                        design_payloads,
                        excluded_facts=excluded_fact_objs,
                        compact_description_en=running_en,
                        compact_description_ru=running_ru,
                    )
                    for d in kept_design:
                        dkey = f"{eid}:{d['trait']}"
                        if dkey in seen_design_keys:
                            continue
                        seen_design_keys.add(dkey)
                        allowed_dkeys.append(dkey)
                        available_design_traits.append(
                            {
                                "key": dkey,
                                "entity_id": eid,
                                "entity_kind": kind_bucket,
                                "trait": str(d["trait"]),
                                "description_ru": str(d["description_ru"]),
                                "description_en": str(d["description_en"]),
                                "assumption_ru": str(d.get("assumption_ru", "")),
                            }
                        )

                    allowed_fids.extend(accumulated_app_ids)
                    desc_ru_clean = _dedupe_semicolon_parts(
                        [running_ru] if running_ru.strip() else []
                    )
                    desc_en_clean = _dedupe_semicolon_parts(
                        [running_en] if running_en.strip() else []
                    )
                    if not desc_en_clean:
                        base_fallback_eids.append(eid)
                    app = IllustrationAppearance(
                        entity_id=eid,
                        name_ru=display_name_ru,
                        description_ru="; ".join(desc_ru_clean) or display_name_ru,
                        description_en="; ".join(desc_en_clean) or fallback_en,
                        fact_ids=tuple(accumulated_app_ids),
                        assumptions_ru=filter_non_assumption_placeholders(accumulated_assumps),
                        name_en=eng_name,
                    )
                    (char_apps if kind_bucket == "person" else loc_apps).append(app)

            scoped_appearances_by_moment[card.candidate_id] = (
                char_apps,
                loc_apps,
                list(dict.fromkeys(allowed_fids)),
                list(dict.fromkeys(allowed_dkeys)),
                available_design_traits,
                list(dict.fromkeys(base_fallback_eids)),
                list(dict.fromkeys(entity_names_all)),
            )

        if pending_fact_scopes:
            return dispatched

        # Step 2: PROMPT scopes for every bound moment!
        prompt_scopes_all = tx.illustrations.list_scopes(
            generation.id, stage=IllustrationScopeStage.PROMPT, section_id=section_id
        )
        prompt_by_key = {s.scope_key: s for s in prompt_scopes_all}
        pending_prompts = False
        validated_moments: list[ValidatedMoment] = []

        for card, _chars, _locs in bound_moments:
            (
                char_apps,
                loc_apps,
                allowed_fids,
                allowed_dkeys,
                avail_design_traits,
                base_fallback_eids,
                moment_entity_names,
            ) = scoped_appearances_by_moment[card.candidate_id]
            pkey = f"prompt:{section_id}:{card.candidate_id}"
            existing_p = prompt_by_key.get(pkey)
            if existing_p is None:
                scene_loc_ru = (
                    None
                    if loc_apps
                    else _resolve_scene_location_ru(entry, section_id, card.scene_id)
                )
                raw_anchor_quotes = [
                    a.quote.strip()[:240]
                    for a in card.anchors
                    if a.quote and a.quote.strip()
                ][:3]

                def _build_prompt_probe_payload(
                    cur_quotes: Sequence[str],
                    cur_gloss: Mapping[str, str],
                    _card: SelectedMomentCard = card,
                    _chars: Sequence[IllustrationAppearance] = char_apps,
                    _locs: Sequence[IllustrationAppearance] = loc_apps,
                    _fids: Sequence[str] = allowed_fids,
                    _dkeys: Sequence[str] = allowed_dkeys,
                    _dtraits: Sequence[Mapping[str, Any]] = avail_design_traits,
                    _sloc: str | None = scene_loc_ru,
                ) -> LLMRequest[Any]:
                    p_in: dict[str, Any] = {
                        "moment_id": _card.candidate_id,
                        "scene_id": _card.scene_id,
                        "title_ru": _card.title_ru,
                        "event_ru": _card.event_ru,
                        "reason_ru": _card.reason_ru,
                        "style": revision.settings.style,
                        "characters": [c.to_dict() for c in _chars],
                        "locations": [loc.to_dict() for loc in _locs],
                        "allowed_fact_ids": list(_fids),
                        "allowed_design_keys": list(_dkeys),
                        "available_design_traits": [thaw(t) for t in _dtraits],
                    }
                    if not _locs and _sloc and _sloc.strip():
                        p_in["scene_location_ru"] = _sloc.strip()[:240]
                    if cur_quotes:
                        p_in["anchor_quotes"] = list(cur_quotes)
                    if cur_gloss:
                        p_in["glossary_terms_ru"] = dict(cur_gloss)
                    return IllustrationContext.build_for_kind(
                        kind=LLMResponseKind.ILLUSTRATION_PROMPT,
                        stage=IllustrationScopeStage.PROMPT,
                        input_payload=p_in,
                        book_id=snapshot.book_id,
                        conditions_id=snapshot.conditions_id,
                        policy=policy,
                    )

                bounded_anchor_quotes: list[str] = []
                for q in raw_anchor_quotes:
                    cand_quotes = [*bounded_anchor_quotes, q]
                    if (
                        sizer.measure(_build_prompt_probe_payload(cand_quotes, {})).size_bytes
                        <= policy.initial_target_bytes
                    ):
                        bounded_anchor_quotes = cand_quotes
                    else:
                        break

                bounded_prompt_glossary: dict[str, str] = {}
                if chapter_glossary_terms_ru:
                    bounded_prompt_glossary = fit_glossary_terms_to_wire_budget(
                        chapter_glossary_terms_ru,
                        build_request=lambda g, _q=bounded_anchor_quotes: _build_prompt_probe_payload(
                            _q, g
                        ),
                        sizer=sizer,
                        target_bytes=policy.initial_target_bytes,
                    )
                draft = _build_prompt_scope_spec(
                    section_id=section_id,
                    section_order=entry.order,
                    section_title=entry.title,
                    scope_key=pkey,
                    moment_card=card,
                    characters=char_apps,
                    locations=loc_apps,
                    allowed_fact_ids=allowed_fids,
                    allowed_design_keys=allowed_dkeys,
                    available_design_traits=avail_design_traits,
                    base_fallback_entity_ids=base_fallback_eids,
                    entity_names=moment_entity_names,
                    scene_location_ru=scene_loc_ru,
                    anchor_quotes=bounded_anchor_quotes,
                    glossary_terms_ru=bounded_prompt_glossary,
                    revision=revision,
                    route_id=route_id,
                )
                reusable = tx.illustrations.find_reusable_scope(
                    generation.book_id,
                    stage=IllustrationScopeStage.PROMPT,
                    input_sha256=draft.input_sha256,
                )
                if reusable is not None:
                    draft = _build_prompt_scope_spec(
                        section_id=section_id,
                        section_order=entry.order,
                        section_title=entry.title,
                        scope_key=pkey,
                        moment_card=card,
                        characters=char_apps,
                        locations=loc_apps,
                        allowed_fact_ids=allowed_fids,
                        allowed_design_keys=allowed_dkeys,
                        available_design_traits=avail_design_traits,
                        base_fallback_entity_ids=base_fallback_eids,
                        entity_names=moment_entity_names,
                        scene_location_ru=scene_loc_ru,
                        anchor_quotes=bounded_anchor_quotes,
                        glossary_terms_ru=bounded_prompt_glossary,
                        revision=revision,
                        route_id=route_id,
                        reusable=reusable,
                    )
                created = tx.illustrations.upsert_scopes(generation.id, [draft])
                existing_p = created[0]
                prompt_by_key[pkey] = existing_p
                if existing_p.state is IllustrationScopeState.PENDING:
                    dispatch_scope_task(
                        tx,
                        existing_p,
                        generation_id=generation.id,
                        task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                        id_factory=id_factory,
                        book_id=generation.book_id,
                    )
                    dispatched = True

            if existing_p.state is not IllustrationScopeState.ACCEPTED:
                pending_prompts = True
                continue

            p_pay = existing_p.payload or {}
            anchors = tuple(
                SourceAnchor(
                    block_id=a.block_id, start=a.start, end=a.end, quote=a.quote
                )
                for a in card.anchors
            )
            try:
                insert_after = resolve_insert_after_block_id(
                    card.insert_after_block_id,
                    anchors=anchors,
                    scene_end_block_id=card.scene_end_block_id,
                    section=section,
                )
            except IllustrationValidationError:
                insert_after = resolve_insert_after_block_id(
                    None,
                    anchors=anchors,
                    scene_end_block_id=card.scene_end_block_id,
                    section=section,
                )

            used_design_keys = [str(k) for k in p_pay.get("used_design_keys", ())]
            depicted_raw = p_pay.get("depicted_entity_ids")
            depicted_set = (
                {str(x) for x in depicted_raw}
                if isinstance(depicted_raw, (list, tuple))
                else None
            )
            raw_traits_by_key = existing_p.summary.get("design_traits_by_key")
            design_traits_by_key = (
                raw_traits_by_key if isinstance(raw_traits_by_key, Mapping) else None
            )
            raw_fallback_eids = existing_p.summary.get("base_fallback_entity_ids", ())
            summary_fallback_eids = (
                tuple(str(eid) for eid in raw_fallback_eids)
                if isinstance(raw_fallback_eids, (list, tuple))
                else ()
            )
            if design_traits_by_key is None and (
                "characters" in existing_p.summary or "locations" in existing_p.summary
            ):
                final_chars = tuple(
                    app
                    for app in (
                        IllustrationAppearance.from_dict(c)
                        for c in existing_p.summary.get("characters", ())
                    )
                    if depicted_set is None or app.entity_id in depicted_set
                )
                final_locs = tuple(
                    app
                    for app in (
                        IllustrationAppearance.from_dict(loc)
                        for loc in existing_p.summary.get("locations", ())
                    )
                    if depicted_set is None or app.entity_id in depicted_set
                )
            else:
                filtered_chars = (
                    [c for c in char_apps if c.entity_id in depicted_set]
                    if depicted_set is not None
                    else char_apps
                )
                filtered_locs = (
                    [loc for loc in loc_apps if loc.entity_id in depicted_set]
                    if depicted_set is not None
                    else loc_apps
                )
                final_chars, final_locs = _materialize_appearances_for_prompt(
                    filtered_chars,
                    filtered_locs,
                    design_traits_by_key,
                    used_design_keys,
                    base_fallback_entity_ids=summary_fallback_eids,
                )

            prompt_en = str(p_pay["prompt_en"])
            assumptions = filter_non_assumption_placeholders(
                [
                    *(str(x) for x in p_pay.get("assumptions_ru", ())),
                    *(a for app in (*final_chars, *final_locs) for a in app.assumptions_ru),
                ]
            )
            moment = validate_moment(
                ValidatedMoment(
                    id=card.candidate_id,
                    scene_id=card.scene_id,
                    title_ru=card.title_ru,
                    reason_ru=card.reason_ru,
                    event_ru=card.event_ru,
                    prompt_en=prompt_en,
                    insert_after_block_id=insert_after,
                    anchors=anchors,
                    character_descriptions=final_chars,
                    location_descriptions=final_locs,
                    assumptions_ru=assumptions,
                ),
                snapshot=snapshot,
                section=section,
                scene_start_block_id=card.scene_start_block_id,
                scene_end_block_id=card.scene_end_block_id,
                registry=all_registry_facts,
                event_position_key=card.event_position_key,
            )
            validated_moments.append(moment)

        if pending_prompts:
            return dispatched

        # Step 3: Atomically accept CHAPTER scope + AnalysisPassResult mirror!
        plan = ChapterIllustrationPlan(
            section_id=section_id,
            generation_id=generation.id,
            style=revision.settings.style,
            moments=tuple(validated_moments),
            zero_reason_ru=None if validated_moments else _DEFAULT_ZERO_UNRESOLVED_RU,
        )
        self._accept_chapter_plan(
            tx, generation, snapshot, chapter_scope, plan, id_factory=id_factory
        )
        return dispatched

    def _accept_chapter_plan(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        snapshot: IllustrationSourceSnapshot,
        chapter_scope: IllustrationScope,
        plan: ChapterIllustrationPlan,
        *,
        id_factory: Callable[[], str],
    ) -> None:
        chapter_payload = plan.to_dict()
        output_sha = canonical_sha(chapter_payload)
        mirror_payload = encode_illustration_pass(plan)
        pass_mirror = (
            AnalysisPassResult(
                id=id_factory(),
                book_id=generation.book_id,
                section_id=plan.section_id,
                pass_kind=AnalysisPassKind.ILLUSTRATION_PREP,
                scope_key=generation.id,
                input_sha256=chapter_scope.input_sha256,
                output_sha256=canonical_sha(mirror_payload),
                conditions_id=snapshot.conditions_id,
                attempt_id=None,
                payload=mirror_payload,
            )
            if snapshot.conditions_id is not None
            else None
        )
        tx.illustrations.accept_scope(
            chapter_scope.id,
            attempt_id=None,
            expected_input_sha256=chapter_scope.input_sha256,
            payload=chapter_payload,
            output_sha256=output_sha,
            summary={"moment_count": len(plan.moments)},
            pass_mirror=pass_mirror,
        )
        self._maybe_mark_generation_ready(tx, generation)
        finalize_generation_state_in_tx(tx, generation.book_id, generation.id)

    async def _run_render_scope_task(self, task: ProcessingTask) -> None:
        scope_id = task.scope_id
        if not scope_id:
            return
        book_id = task.book_id
        generation_id = task.operation_id.split(":", 1)[0]

        repair_summary: str | None = None
        repairs_used = 0
        activation_calls_used = 0
        first_call = True
        notify_after = False

        while True:
            with self._uow.transaction() as tx:
                scope = tx.illustrations.get_scope(scope_id)
                if scope is None or scope.generation_id != generation_id:
                    return
                head = tx.illustrations.get_head(book_id)
                gen = tx.illustrations.get_generation(book_id, generation_id)
                if (
                    gen is None
                    or head.active_generation_id != gen.id
                    or gen.state is IllustrationGenerationState.REVOKED
                ):
                    if not first_call:
                        self._abort_stale_running_scope(
                            tx,
                            scope,
                            task_id=task.id,
                            book_id=book_id,
                            generation_id=generation_id,
                            reason_code=IllustrationReasonCode.STALE_ATTEMPT.value,
                        )
                    return
                revision = tx.illustrations.get_settings(book_id, gen.settings_id)
                if (
                    revision is None
                    or head.settings_id != revision.id
                    or not revision.settings.enabled
                ):
                    if not first_call:
                        stale_code = (
                            IllustrationReasonCode.CHANNEL_DISABLED.value
                            if revision is not None and not revision.settings.enabled
                            else IllustrationReasonCode.SETTINGS_CHANGED.value
                        )
                        self._abort_stale_running_scope(
                            tx,
                            scope,
                            task_id=task.id,
                            book_id=book_id,
                            generation_id=generation_id,
                            reason_code=stale_code,
                        )
                    return
                snapshot = self._source.snapshot(tx, book_id)
                if (
                    not snapshot.compatible
                    or (snapshot.source_sha256 is not None and snapshot.source_sha256 != gen.source_sha256)
                    or (
                        gen.manifest_sha256 is not None
                        and snapshot.manifest_sha256 != gen.manifest_sha256
                    )
                ):
                    revoke_active_in_transaction(
                        tx, book_id, reason_code=IllustrationReasonCode.SOURCE_CHANGED.value
                    )
                    self._supersede_task(tx, task.id)
                    return

                section_id = scope.section_id
                if section_id is None:
                    return

                if first_call and scope.state is IllustrationScopeState.PENDING:
                    reusable = tx.illustrations.find_reusable_scope(
                        book_id, stage=scope.stage, input_sha256=scope.input_sha256
                    )
                    if (
                        reusable is not None
                        and reusable.payload is not None
                        and reusable.output_sha256 is not None
                    ):
                        tx.illustrations.accept_scope(
                            scope.id,
                            attempt_id=None,
                            expected_input_sha256=scope.input_sha256,
                            payload=reusable.payload,
                            output_sha256=reusable.output_sha256,
                            summary={**thaw(scope.summary), "reused_from_scope_id": reusable.id},
                        )
                        if self._advance_chapter_rendering(
                            tx, gen, revision, snapshot, section_id, id_factory=self._id_factory
                        ):
                            notify_after = True
                        else:
                            finalize_generation_state_in_tx(tx, book_id, gen.id)
                        self._mark_task_succeeded(tx, task.id)
                        break

                policy = revision.settings.policy
                try:
                    if first_call:
                        attempt = tx.illustrations.claim_scope(
                            scope.id,
                            task_id=task.id,
                            expected_generation_id=gen.id,
                            max_calls=policy.max_calls_per_scope_activation,
                        )
                    else:
                        attempt = tx.illustrations.begin_next_attempt(
                            scope.id,
                            task_id=task.id,
                            expected_generation_id=gen.id,
                            max_calls=policy.max_calls_per_scope_activation,
                        )
                except IllustrationOwnershipError:
                    return
                except IllustrationLimitError as limit_err:
                    tx.illustrations.fail_scope(
                        scope.id, attempt_id=None, reason_code=limit_err.reason_code
                    )
                    self._mark_task_failed(tx, task.id, limit_err.reason_code)
                    self._refresh_generation_state(tx, book_id, gen.id, limit_err.reason_code)
                    return

                if gen.state is IllustrationGenerationState.QUEUED:
                    tx.illustrations.update_generation_state(
                        book_id, gen.id, state=IllustrationGenerationState.RUNNING, reason_code=None
                    )
                gen_number = gen.generation_number
                assigned_model = revision.settings.model_prompts

            first_call = False
            activation_calls_used += 1

            sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_CHAPTER)
            request = IllustrationContext.build(
                scope,
                snapshot,
                revision.settings,
                repair_summary=repair_summary,
                attempt_id=attempt.id,
                generation_number=gen_number,
                assigned_model=assigned_model,
            )
            wire = sizer.measure(request)
            if wire.size_bytes > policy.max_request_bytes:
                with self._uow.transaction() as tx:
                    tx.illustrations.release_reserved_call(
                        attempt.id, reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value
                    )
                    tx.illustrations.fail_scope(
                        scope.id,
                        attempt_id=None,
                        reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                    )
                    self._mark_task_failed(
                        tx, task.id, IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value
                    )
                    self._refresh_generation_state(
                        tx,
                        book_id,
                        generation_id,
                        IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                    )
                return

            try:
                client = self._runtime.client_for(revision, LLMCallStage.ILLUSTRATION_CHAPTER)
            except IllustrationError as route_err:
                with self._uow.transaction() as tx:
                    tx.illustrations.release_reserved_call(
                        attempt.id, reason_code=route_err.reason_code
                    )
                    tx.illustrations.fail_scope(
                        scope.id, attempt_id=None, reason_code=route_err.reason_code
                    )
                    self._mark_task_failed(tx, task.id, route_err.reason_code)
                    self._refresh_generation_state(tx, book_id, generation_id, route_err.reason_code)
                return

            with self._uow.transaction() as tx:
                head = tx.illustrations.get_head(book_id)
                gen = tx.illustrations.get_generation(book_id, generation_id)
                if (
                    gen is None
                    or head.active_generation_id != gen.id
                    or head.settings_id != revision.id
                    or gen.state is IllustrationGenerationState.REVOKED
                ):
                    try:
                        tx.illustrations.release_reserved_call(
                            attempt.id, reason_code=IllustrationReasonCode.STALE_ATTEMPT.value
                        )
                    except IllustrationOwnershipError:
                        pass
                    return
                try:
                    tx.illustrations.mark_attempt_sent(
                        attempt.id,
                        request_sha256=wire.sha256,
                        request_meta={
                            "stage": scope.stage.value,
                            "model": assigned_model,
                            "wire_bytes": wire.size_bytes,
                            "repair_turn": repairs_used,
                        },
                    )
                except IllustrationOwnershipError:
                    return

            response: LLMResponse[Any] | None = None
            validation: _ValidationOutcome | None = None
            call_error: Exception | None = None
            try:
                response = await client.generate(request)
                if response.truncated:
                    raise LLMResponseFormatError(
                        "ответ модели усечён по лимиту токенов", truncated=True
                    )
                validation = self._validate_render_response(scope, response.payload, revision)
            except Exception as err:
                call_error = err

            if call_error is not None:
                if response is not None:
                    record_rejection(self._uow, response, call_error)
                if is_context_limit_error(call_error):
                    self._finalize_failure(
                        book_id=book_id,
                        generation_id=generation_id,
                        task_id=task.id,
                        scope_id=scope.id,
                        attempt_id=attempt.id,
                        reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                    )
                    return
                if is_truncated_error(call_error):
                    if self._can_continue_activation(
                        activation_calls_used, scope.id, book_id, policy
                    ):
                        with self._uow.transaction() as tx:
                            tx.illustrations.record_attempt_outcome(
                                attempt.id,
                                state=IllustrationAttemptState.FAILED,
                                outcome={
                                    "reason_code": IllustrationReasonCode.PROVIDER_TRUNCATED.value
                                },
                            )
                        repair_summary = (
                            "Предыдущий ответ был обрезан по длине. "
                            "Верни компактный валидный JSON без повторов и лишних пробелов."
                        )
                        continue
                    self._finalize_failure(
                        book_id=book_id,
                        generation_id=generation_id,
                        task_id=task.id,
                        scope_id=scope.id,
                        attempt_id=attempt.id,
                        reason_code=IllustrationReasonCode.PROVIDER_TRUNCATED.value,
                    )
                    return
                if is_transient_error(call_error):
                    if self._can_continue_activation(activation_calls_used, scope.id, book_id, policy):
                        with self._uow.transaction() as tx:
                            tx.illustrations.record_attempt_outcome(
                                attempt.id,
                                state=IllustrationAttemptState.FAILED,
                                outcome={"reason_code": IllustrationReasonCode.PROVIDER_ERROR.value},
                            )
                        decision = self._retry_policy.decide(
                            call_error, attempts=activation_calls_used
                        )
                        await self._sleep(decision.backoff_seconds)
                        continue
                    self._finalize_failure(
                        book_id=book_id,
                        generation_id=generation_id,
                        task_id=task.id,
                        scope_id=scope.id,
                        attempt_id=attempt.id,
                        reason_code=IllustrationReasonCode.PROVIDER_ERROR.value,
                    )
                    return
                if isinstance(call_error, LLMResponseFormatError):
                    err_detail = str(call_error)
                    if (
                        repairs_used < policy.max_repair_turns
                        and self._can_continue_activation(activation_calls_used, scope.id, book_id, policy)
                    ):
                        with self._uow.transaction() as tx:
                            tx.illustrations.record_attempt_outcome(
                                attempt.id,
                                state=IllustrationAttemptState.REJECTED,
                                outcome={
                                    "reason_code": IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
                                    "detail": compact_safe_detail(err_detail, max_bytes=256),
                                },
                            )
                        repairs_used += 1
                        repair_summary = err_detail
                        continue
                    self._finalize_failure(
                        book_id=book_id,
                        generation_id=generation_id,
                        task_id=task.id,
                        scope_id=scope.id,
                        attempt_id=attempt.id,
                        reason_code=IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
                        detail=err_detail,
                    )
                    return
                fallback_reason = (
                    IllustrationReasonCode.ROUTE_UNAVAILABLE.value
                    if is_incompatible_route_error(call_error)
                    else IllustrationReasonCode.PROVIDER_ERROR.value
                )
                self._finalize_failure(
                    book_id=book_id,
                    generation_id=generation_id,
                    task_id=task.id,
                    scope_id=scope.id,
                    attempt_id=attempt.id,
                    reason_code=fallback_reason,
                )
                return

            assert validation is not None
            if not validation.ok:
                detail_text = validation.repair_reason
                if response is not None:
                    record_rejection(self._uow, response, ResultValidationError(detail_text))
                if (
                    repairs_used < policy.max_repair_turns
                    and self._can_continue_activation(activation_calls_used, scope.id, book_id, policy)
                ):
                    with self._uow.transaction() as tx:
                        tx.illustrations.record_attempt_outcome(
                            attempt.id,
                            state=IllustrationAttemptState.REJECTED,
                            outcome={
                                "reason_code": validation.reason_code,
                                "detail": compact_safe_detail(detail_text, max_bytes=256),
                            },
                        )
                    repairs_used += 1
                    repair_summary = detail_text
                    continue
                self._finalize_failure(
                    book_id=book_id,
                    generation_id=generation_id,
                    task_id=task.id,
                    scope_id=scope.id,
                    attempt_id=attempt.id,
                    reason_code=validation.reason_code,
                    detail=detail_text,
                )
                return

            assert validation.payload is not None
            output_sha256 = canonical_sha(validation.payload)
            with self._uow.transaction() as tx:
                head = tx.illustrations.get_head(book_id)
                gen = tx.illustrations.get_generation(book_id, generation_id)
                cur_rev = tx.illustrations.get_settings(book_id, revision.id)
                stale_reason: str | None = None
                if gen is None or gen.state is IllustrationGenerationState.REVOKED:
                    stale_reason = IllustrationReasonCode.STALE_ATTEMPT.value
                elif head.active_generation_id != gen.id or head.settings_id != revision.id:
                    stale_reason = IllustrationReasonCode.SETTINGS_CHANGED.value
                elif cur_rev is None or not cur_rev.settings.enabled:
                    stale_reason = IllustrationReasonCode.CHANNEL_DISABLED.value
                else:
                    cur_snap = self._source.snapshot(tx, book_id)
                    if (
                        not cur_snap.compatible
                        or (cur_snap.source_sha256 is not None and cur_snap.source_sha256 != gen.source_sha256)
                        or (
                            gen.manifest_sha256 is not None
                            and cur_snap.manifest_sha256 != gen.manifest_sha256
                        )
                    ):
                        stale_reason = IllustrationReasonCode.SOURCE_CHANGED.value

                if stale_reason is not None:
                    tx.illustrations.record_attempt_outcome(
                        attempt.id,
                        state=IllustrationAttemptState.STALE,
                        outcome={"reason_code": stale_reason},
                    )
                    finalize_response(self._uow, response, "stale", transaction=tx)
                    if stale_reason == IllustrationReasonCode.SOURCE_CHANGED.value:
                        revoke_active_in_transaction(
                            tx, book_id, reason_code=IllustrationReasonCode.SOURCE_CHANGED.value
                        )
                    elif stale_reason != IllustrationReasonCode.STALE_ATTEMPT.value:
                        try:
                            tx.illustrations.fail_scope(
                                scope.id, attempt_id=None, reason_code=stale_reason
                            )
                        except IllustrationOwnershipError:
                            pass
                        finalize_generation_state_in_tx(
                            tx, book_id, generation_id, fallback_reason=stale_reason
                        )
                    self._supersede_task(tx, task.id)
                    return

                try:
                    tx.illustrations.accept_scope(
                        scope.id,
                        attempt_id=attempt.id,
                        expected_input_sha256=scope.input_sha256,
                        payload=validation.payload,
                        output_sha256=output_sha256,
                        summary=scope.summary,
                    )
                except IllustrationOwnershipError:
                    tx.illustrations.record_attempt_outcome(
                        attempt.id,
                        state=IllustrationAttemptState.STALE,
                        outcome={"reason_code": IllustrationReasonCode.STALE_ATTEMPT.value},
                    )
                    finalize_response(self._uow, response, "stale", transaction=tx)
                    return

                finalize_response(self._uow, response, "accepted", transaction=tx)

                assert gen is not None
                if self._advance_chapter_rendering(
                    tx, gen, revision, cur_snap, section_id, id_factory=self._id_factory
                ):
                    notify_after = True
                else:
                    finalize_generation_state_in_tx(tx, book_id, gen.id)
                self._mark_task_succeeded(tx, task.id)
                break

        if notify_after and self._notify_work_available is not None:
            self._notify_work_available()

    def _validate_render_response(
        self,
        scope: IllustrationScope,
        payload: Any,
        revision: IllustrationSettingsRevision,
    ) -> _ValidationOutcome:
        if scope.stage is IllustrationScopeStage.FACT_SCOPE:
            if not isinstance(payload, IllustrationFactScopeResponse):
                return _ValidationOutcome(ok=False, repair_reason="ожидался ответ отбора фактов")
            expected_event_id = str(scope.summary.get("moment_id", ""))
            if payload.event_id != expected_event_id:
                return _ValidationOutcome(
                    ok=False, repair_reason=f"event_id должен быть {expected_event_id}"
                )
            valid_fids = set(str(fid) for fid in scope.summary.get("valid_fact_ids", ()))
            running_fids = set(str(fid) for fid in scope.summary.get("running_applicable_ids", ()))
            allowed = valid_fids | running_fids
            max_app = int(scope.summary.get("max_applicable_fact_ids", _MAX_APPLICABLE_FACTS))
            if len(payload.applicable_fact_ids) > max_app:
                return _ValidationOutcome(
                    ok=False, repair_reason=f"не более {max_app} applicable_fact_ids"
                )
            if any(fid not in allowed for fid in payload.applicable_fact_ids):
                return _ValidationOutcome(
                    ok=False, repair_reason="applicable_fact_ids содержит неизвестный или будущий fact_id"
                )
            if any(fid not in allowed for fid in payload.excluded_fact_ids):
                return _ValidationOutcome(
                    ok=False, repair_reason="excluded_fact_ids содержит неизвестный fact_id"
                )
            if set(payload.applicable_fact_ids) & set(payload.excluded_fact_ids):
                return _ValidationOutcome(
                    ok=False,
                    repair_reason="один и тот же fact_id не может быть одновременно в applicable и excluded",
                )
            if has_cyrillic(payload.compact_description_en):
                return _ValidationOutcome(
                    ok=False,
                    repair_reason="compact_description_en должен быть написан строго на английском языке без кириллицы",
                )
            if revision.settings.style == ILLUSTRATION_STYLE_MONOCHROME_INK:
                raw_names = scope.summary.get("entity_names")
                if isinstance(raw_names, (list, tuple)):
                    entity_names = [
                        str(n).strip()
                        for n in raw_names
                        if n is not None and str(n).strip()
                    ]
                else:
                    entity_names = []
                if not entity_names:
                    fallback_name = scope.summary.get("entity_name")
                    if fallback_name is not None and str(fallback_name).strip():
                        entity_names = [str(fallback_name).strip()]
                color_word = find_monochrome_color_word(
                    payload.compact_description_en,
                    allowed_names=entity_names,
                    strict_palette=False,
                )
                if color_word is not None:
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=(
                            f"для стиля monochrome_ink в compact_description_en нельзя использовать "
                            f"название цвета ({color_word}); передай признак через светлоту, контраст и фактуру"
                        ),
                    )
            return _ValidationOutcome(
                ok=True,
                payload={
                    "schema_version": 1,
                    "event_id": payload.event_id,
                    "applicable_fact_ids": list(dict.fromkeys(payload.applicable_fact_ids)),
                    "excluded_fact_ids": list(dict.fromkeys(payload.excluded_fact_ids)),
                    "compact_description_ru": payload.compact_description_ru.strip(),
                    "compact_description_en": payload.compact_description_en.strip(),
                    "assumptions_ru": list(
                        filter_non_assumption_placeholders(payload.assumptions_ru)
                    ),
                },
            )

        if scope.stage is IllustrationScopeStage.PROMPT:
            if not isinstance(payload, IllustrationPromptResponse):
                return _ValidationOutcome(ok=False, repair_reason="ожидался ответ композиции промта")
            expected_mid = str(scope.summary.get("moment_id", ""))
            if payload.moment_id != expected_mid:
                return _ValidationOutcome(
                    ok=False, repair_reason=f"moment_id должен быть {expected_mid}"
                )
            allowed_fids = set(str(fid) for fid in scope.summary.get("allowed_fact_ids", ()))
            allowed_dkeys = set(str(k) for k in scope.summary.get("allowed_design_keys", ()))
            if any(fid not in allowed_fids for fid in payload.used_fact_ids):
                return _ValidationOutcome(
                    ok=False, repair_reason="used_fact_ids содержит недопустимый или будущий fact_id"
                )
            if any(dk not in allowed_dkeys for dk in payload.used_design_keys):
                return _ValidationOutcome(
                    ok=False, repair_reason="used_design_keys содержит неизвестный ключ дизайна"
                )
            if has_cyrillic(payload.scene_composition_en):
                return _ValidationOutcome(
                    ok=False,
                    repair_reason="scene_composition_en должен быть написан строго на английском языке без кириллицы",
                )
            bubble_term = find_speech_bubble_term(payload.scene_composition_en)
            if bubble_term is not None:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason=(
                        f"в scene_composition_en запрещены комиксные речевые облачка и подписи ({bubble_term}); "
                        "опиши сцену и эмоции героев чисто визуально без речевых облачков и надписей"
                    ),
                )
            if revision.settings.style == ILLUSTRATION_STYLE_MONOCHROME_INK:
                raw_names = scope.summary.get("entity_names")
                if isinstance(raw_names, (list, tuple)):
                    entity_names = [
                        str(n).strip()
                        for n in raw_names
                        if n is not None and str(n).strip()
                    ]
                else:
                    entity_names = []
                color_word = find_monochrome_color_word(
                    payload.scene_composition_en,
                    allowed_names=entity_names,
                    strict_palette=False,
                )
                if color_word is not None:
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=(
                            f"для стиля monochrome_ink в scene_composition_en нельзя использовать "
                            f"название цвета ({color_word}); передай различия через светлоту, контраст и фактуру"
                        ),
                    )
            base_characters = [
                IllustrationAppearance.from_dict(c) for c in scope.summary.get("characters", ())
            ]
            base_locations = [
                IllustrationAppearance.from_dict(loc) for loc in scope.summary.get("locations", ())
            ]
            raw_traits_by_key = scope.summary.get("design_traits_by_key")
            design_traits_by_key = (
                raw_traits_by_key if isinstance(raw_traits_by_key, Mapping) else None
            )
            raw_fallback_eids = scope.summary.get("base_fallback_entity_ids", ())
            fallback_eids = (
                tuple(str(eid) for eid in raw_fallback_eids)
                if isinstance(raw_fallback_eids, (list, tuple))
                else ()
            )
            depicted = payload.depicted_entity_ids
            if depicted is not None:
                if len(depicted) != len(set(depicted)):
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason="depicted_entity_ids содержит дублирующиеся entity_id",
                    )
                allowed_entity_ids = {
                    app.entity_id for app in (*base_characters, *base_locations)
                }
                depicted_set = set(depicted)
                unknown_depicted = [eid for eid in depicted if eid not in allowed_entity_ids]
                if unknown_depicted:
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=(
                            "depicted_entity_ids содержит неизвестный entity_id: "
                            f"{', '.join(unknown_depicted)}"
                        ),
                    )
                fact_owners: dict[str, set[str]] = {}
                for app in (*base_characters, *base_locations):
                    for fid in app.fact_ids:
                        fact_owners.setdefault(fid, set()).add(app.entity_id)
                for fid in payload.used_fact_ids:
                    owners = fact_owners.get(fid)
                    if owners and not (owners & depicted_set):
                        return _ValidationOutcome(
                            ok=False,
                            repair_reason=(
                                f"used_fact_ids содержит факт {fid} сущности, "
                                "не указанной в depicted_entity_ids"
                            ),
                        )
                for dk in payload.used_design_keys:
                    owner_eid: str | None = None
                    if design_traits_by_key is not None and isinstance(
                        design_traits_by_key.get(dk), Mapping
                    ):
                        owner_eid = (
                            str(design_traits_by_key[dk].get("entity_id", "")).strip() or None
                        )
                    if owner_eid is None and ":" in dk:
                        owner_eid = dk.split(":", 1)[0].strip() or None
                    if owner_eid and owner_eid not in depicted_set:
                        return _ValidationOutcome(
                            ok=False,
                            repair_reason=(
                                f"used_design_keys содержит ключ {dk} сущности, "
                                "не указанной в depicted_entity_ids"
                            ),
                        )
                base_characters = [c for c in base_characters if c.entity_id in depicted_set]
                base_locations = [loc for loc in base_locations if loc.entity_id in depicted_set]

            characters, locations = _materialize_appearances_for_prompt(
                base_characters,
                base_locations,
                design_traits_by_key,
                payload.used_design_keys,
                base_fallback_entity_ids=fallback_eids,
            )
            try:
                prompt_en = compose_monochrome_prompt(
                    payload.scene_composition_en,
                    [*characters, *locations],
                    revision.settings.style,
                )
            except IllustrationValidationError as err:
                return _ValidationOutcome(ok=False, repair_reason=str(err))
            accepted_payload: dict[str, Any] = {
                "schema_version": 1,
                "moment_id": payload.moment_id,
                "scene_composition_en": payload.scene_composition_en.strip(),
                "prompt_en": prompt_en,
                "used_fact_ids": list(dict.fromkeys(payload.used_fact_ids)),
                "used_design_keys": list(dict.fromkeys(payload.used_design_keys)),
                "assumptions_ru": list(
                    filter_non_assumption_placeholders(payload.assumptions_ru)
                ),
            }
            if depicted is not None:
                accepted_payload["depicted_entity_ids"] = list(depicted)
            return _ValidationOutcome(ok=True, payload=accepted_payload)

        return _ValidationOutcome(ok=False, repair_reason="неподдерживаемый этап рендера")

    def _finalize_failure(
        self,
        *,
        book_id: str,
        generation_id: str,
        task_id: str,
        scope_id: str,
        attempt_id: str,
        reason_code: str,
        detail: str | None = None,
    ) -> None:
        outcome = (
            {
                "reason_code": reason_code,
                "detail": compact_safe_detail(detail, max_bytes=256),
            }
            if detail and detail.strip()
            else None
        )
        with self._uow.transaction() as tx:
            try:
                tx.illustrations.fail_scope(
                    scope_id,
                    attempt_id=attempt_id,
                    reason_code=reason_code,
                    outcome=outcome,
                )
            except IllustrationOwnershipError:
                tx.illustrations.record_attempt_outcome(
                    attempt_id,
                    state=IllustrationAttemptState.STALE,
                    outcome={"reason_code": IllustrationReasonCode.STALE_ATTEMPT.value},
                )
                return
            self._mark_task_failed(tx, task_id, reason_code)
            self._refresh_generation_state(tx, book_id, generation_id, reason_code)

    def _can_continue_activation(
        self,
        activation_calls_used: int,
        scope_id: str,
        book_id: str,
        policy: IllustrationPolicy,
    ) -> bool:
        if activation_calls_used >= policy.max_calls_per_scope_activation:
            return False
        with self._uow.transaction() as tx:
            scope = tx.illustrations.get_scope(scope_id)
            if scope is None:
                return False
            gen = tx.illustrations.get_generation(book_id, scope.generation_id)
            if gen is None:
                return False
            return (gen.calls_reserved + gen.calls_sent) < policy.max_calls_per_generation

    @staticmethod
    def _maybe_mark_generation_ready(
        tx: UnitOfWork,
        generation: IllustrationGeneration,
    ) -> None:
        chapter_scopes = tx.illustrations.list_scopes(
            generation.id, stage=IllustrationScopeStage.CHAPTER
        )
        if not chapter_scopes:
            return
        requested_set = set(generation.requested_sections)
        requested_scopes = [
            s for s in chapter_scopes if s.section_id in requested_set or s.state is IllustrationScopeState.SKIPPED
        ]
        if not requested_scopes:
            return
        if all(
            s.state in {IllustrationScopeState.ACCEPTED, IllustrationScopeState.SKIPPED}
            for s in chapter_scopes
        ) and all(
            any(s.section_id == sid and s.state in {IllustrationScopeState.ACCEPTED, IllustrationScopeState.SKIPPED} for s in chapter_scopes)
            for sid in requested_set
        ):
            tx.illustrations.update_generation_state(
                generation.book_id,
                generation.id,
                state=IllustrationGenerationState.READY,
                reason_code=None,
            )

    @staticmethod
    def _mark_task_succeeded(tx: UnitOfWork, task_id: str) -> None:
        task = tx.tasks.get(task_id)
        if task is None:
            return
        if task.status.value == "pending":
            task = task.start()
        if task.status.value == "running":
            tx.tasks.save(task.succeed())

    @staticmethod
    def _mark_task_failed(tx: UnitOfWork, task_id: str, reason_code: str) -> None:
        task = tx.tasks.get(task_id)
        if task is None:
            return
        if task.status.value == "pending":
            task = task.start()
        if task.status.value == "running":
            err_type = (
                ErrorType.CONFIGURATION
                if reason_code in {
                    IllustrationReasonCode.ROUTE_UNAVAILABLE.value,
                    IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                    IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
                    IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED.value,
                    IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value,
                }
                else ErrorType.MODEL_OUTPUT
                if reason_code in {
                    IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
                    IllustrationReasonCode.PROVIDER_TRUNCATED.value,
                }
                else ErrorType.PERMANENT
            )
            tx.tasks.save(task.fail(TaskError(type=err_type, message=reason_code)))

    @staticmethod
    def _supersede_task(tx: UnitOfWork, task_id: str) -> None:
        task = tx.tasks.get(task_id)
        if task is not None and task.status.value in {"pending", "running", "failed"}:
            tx.tasks.save(task.supersede())

    @staticmethod
    def _abort_stale_running_scope(
        tx: UnitOfWork,
        scope: IllustrationScope,
        *,
        task_id: str,
        book_id: str,
        generation_id: str,
        reason_code: str,
    ) -> None:
        if scope.state is IllustrationScopeState.RUNNING:
            try:
                tx.illustrations.fail_scope(
                    scope.id, attempt_id=None, reason_code=reason_code
                )
            except IllustrationOwnershipError:
                pass
            finalize_generation_state_in_tx(
                tx, book_id, generation_id, fallback_reason=reason_code
            )
        IllustrationChapter._supersede_task(tx, task_id)

    @staticmethod
    def _refresh_generation_state(
        tx: UnitOfWork,
        book_id: str,
        generation_id: str,
        reason_code: str,
    ) -> None:
        finalize_generation_state_in_tx(
            tx, book_id, generation_id, fallback_reason=reason_code
        )


__all__ = [
    "IllustrationChapter",
    "resolve_moments_with_shortlist_fallback",
]
