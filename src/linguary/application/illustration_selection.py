"""Finite resumable tournament for selecting diverse chapter illustration moments (`ALG-03`)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from linguary.application.context_builder import source_text_for_model, textual_blocks
from linguary.application.illustration_context import (
    IllustrationContext,
    _collect_book_glossary_terms_ru,
    compact_safe_detail,
    decode_wire_id_aliases,
    encode_wire_id_aliases,
    fit_glossary_terms_to_wire_budget,
    is_context_limit_error,
    is_incompatible_route_error,
    is_transient_error,
    is_truncated_error,
    normalize_name_token,
    pack_items_by_wire_budget,
    resolve_quote_anchor,
)
from linguary.application.illustration_dto import (
    IllustrationCandidatesResponse,
    IllustrationSelectionResponse,
)
from linguary.application.illustration_reference import (
    display_name_at_position as _display_name_at_position,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    IllustrationSourceSnapshot,
    SectionManifestEntry,
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
    DocumentBlock,
    DocumentSection,
    ErrorType,
    ProcessingTask,
    TaskError,
    TaskType,
)
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
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
    format_position_key,
)
from linguary.domain.llm_log import LLMCallStage

_MAX_SHORTLIST = 24
_MAX_CANDIDATE_BLOCKS_PER_CHUNK = 24
_MAX_KNOWN_ENTITIES_PER_KIND = 40
_DEFAULT_ZERO_REASON_RU = "В этой главе не нашлось выразительных визуальных моментов для иллюстрации."


def _load_entity_payloads(tx: UnitOfWork, generation_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        page, total = tx.illustrations.query_registry_items(
            generation_id,
            kind=IllustrationRegistryKind.ENTITY,
            offset=offset,
            limit=200,
        )
        for item in page:
            items.append(item.raw_payload())
        offset += len(page)
        if offset >= total or not page:
            break
    return items


_MAX_KNOWN_ENTITY_ALIASES = 3


def _known_entities_context_for_chapter(
    tx: UnitOfWork,
    generation_id: str,
    *,
    section_order: int,
    limit_per_kind: int = _MAX_KNOWN_ENTITIES_PER_KIND,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, list[str]]]:
    """Collect clean spoiler-free known names and separate non-spoiler aliases visible by `section_order`."""
    chapter_end_pos = format_position_key(section_order, 999_999, 99_999_999)
    entities = _load_entity_payloads(tx, generation_id)
    persons: list[tuple[int, str, str, str, list[str]]] = []
    locations: list[tuple[int, str, str, str, list[str]]] = []
    for ent in entities:
        raw_orders = ent.get("section_orders")
        has_orders = isinstance(raw_orders, (list, tuple))
        in_current = has_orders and any(
            type(o) is int and o == section_order for o in raw_orders
        )
        first_pos = str(ent.get("first_position_key") or "")
        if not in_current:
            if first_pos and first_pos > chapter_end_pos:
                continue
            if not first_pos and has_orders and not any(
                type(o) is int and o <= section_order for o in raw_orders
            ):
                continue
        display_name = _display_name_at_position(ent, chapter_end_pos).strip()
        if not display_name:
            continue
        display_norm = normalize_name_token(display_name)
        visible_aliases: list[str] = []
        raw_positions = ent.get("alias_first_positions")
        if isinstance(raw_positions, Mapping):
            eligible_aliases: list[tuple[str, str, str]] = []
            for raw_alias, pos in raw_positions.items():
                if (
                    raw_alias is None
                    or not isinstance(pos, str)
                    or not pos
                    or pos > chapter_end_pos
                ):
                    continue
                clean_alias = str(raw_alias).strip()
                if not clean_alias:
                    continue
                alias_norm = normalize_name_token(clean_alias)
                if not alias_norm or alias_norm == display_norm:
                    continue
                eligible_aliases.append((pos, clean_alias.casefold(), clean_alias))
            eligible_aliases.sort()
            seen_alias_norms: set[str] = set()
            for _, _, clean_alias in eligible_aliases:
                alias_norm = normalize_name_token(clean_alias)
                if alias_norm in seen_alias_norms:
                    continue
                seen_alias_norms.add(alias_norm)
                visible_aliases.append(clean_alias)
                if len(visible_aliases) >= _MAX_KNOWN_ENTITY_ALIASES:
                    break
        priority = 0 if in_current else 1
        row = (priority, first_pos, display_name.casefold(), display_name, visible_aliases)
        kind = str(ent.get("entity_kind", ""))
        if kind == "person":
            persons.append(row)
        elif kind == "location":
            locations.append(row)

    known_entity_aliases: dict[str, list[str]] = {}

    def _finalize(rows: list[tuple[int, str, str, str, list[str]]]) -> tuple[str, ...]:
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
        seen: set[str] = set()
        out: list[str] = []
        for _, _, key, name, aliases_list in rows:
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
            if aliases_list and name not in known_entity_aliases:
                known_entity_aliases[name] = aliases_list
            if len(out) >= limit_per_kind:
                break
        return tuple(out)

    known_chars = _finalize(persons)
    known_locs = _finalize(locations)
    return known_chars, known_locs, known_entity_aliases


def _known_entity_names_for_chapter(
    tx: UnitOfWork,
    generation_id: str,
    *,
    section_order: int,
    limit_per_kind: int = _MAX_KNOWN_ENTITIES_PER_KIND,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Collect spoiler-free known character and location names visible by the end of `section_order`."""
    known_chars, known_locs, _ = _known_entities_context_for_chapter(
        tx, generation_id, section_order=section_order, limit_per_kind=limit_per_kind
    )
    return known_chars, known_locs


def _default_id() -> str:
    return str(uuid4())


@dataclass(frozen=True, slots=True)
class SelectedAnchor:
    block_id: str
    start: int
    end: int
    quote: str
    position_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start": self.start,
            "end": self.end,
            "quote": self.quote,
            "position_key": self.position_key,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SelectedAnchor:
        return cls(
            block_id=str(raw["block_id"]),
            start=int(raw["start"]),
            end=int(raw["end"]),
            quote=str(raw["quote"]),
            position_key=str(raw["position_key"]),
        )


@dataclass(frozen=True, slots=True)
class SelectedMomentCard:
    """Grounded chapter moment candidate or selected winner."""

    candidate_id: str
    section_id: str
    scene_id: str
    title_ru: str
    event_ru: str
    significance_ru: str
    reason_ru: str
    event_kind: str
    event_position_key: str
    anchors: tuple[SelectedAnchor, ...]
    insert_after_block_id: str | None
    participant_names: tuple[str, ...]
    location_names: tuple[str, ...]
    scene_start_block_id: str
    scene_end_block_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "section_id": self.section_id,
            "scene_id": self.scene_id,
            "title_ru": self.title_ru,
            "event_ru": self.event_ru,
            "significance_ru": self.significance_ru,
            "reason_ru": self.reason_ru,
            "event_kind": self.event_kind,
            "event_position_key": self.event_position_key,
            "anchors": [a.to_dict() for a in self.anchors],
            "insert_after_block_id": self.insert_after_block_id,
            "participant_names": list(self.participant_names),
            "location_names": list(self.location_names),
            "scene_start_block_id": self.scene_start_block_id,
            "scene_end_block_id": self.scene_end_block_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SelectedMomentCard:
        return cls(
            candidate_id=str(raw["candidate_id"]),
            section_id=str(raw["section_id"]),
            scene_id=str(raw["scene_id"]),
            title_ru=str(raw["title_ru"]),
            event_ru=str(raw["event_ru"]),
            significance_ru=str(raw["significance_ru"]),
            reason_ru=str(raw.get("reason_ru") or raw["significance_ru"]),
            event_kind=str(raw["event_kind"]),
            event_position_key=str(raw["event_position_key"]),
            anchors=tuple(SelectedAnchor.from_dict(a) for a in raw.get("anchors", ())),
            insert_after_block_id=(
                None
                if raw.get("insert_after_block_id") is None
                else str(raw["insert_after_block_id"])
            ),
            participant_names=tuple(str(n) for n in raw.get("participant_names", ())),
            location_names=tuple(str(n) for n in raw.get("location_names", ())),
            scene_start_block_id=str(raw.get("scene_start_block_id", "")),
            scene_end_block_id=str(raw.get("scene_end_block_id", "")),
        )


@dataclass(frozen=True, slots=True)
class SelectedMoments:
    """Finalized output of `ALG-03` for one chapter."""

    section_id: str
    winners: tuple[SelectedMomentCard, ...]
    shortlist: tuple[SelectedMomentCard, ...]
    zero_reason_ru: str | None


def _anchors_overlap(a: Sequence[SelectedAnchor], b: Sequence[SelectedAnchor]) -> bool:
    for aa in a:
        for bb in b:
            if aa.block_id != bb.block_id:
                continue
            if aa.start < bb.end and bb.start < aa.end:
                return True
    return False


def effective_chapter_illustration_cap(
    max_per_chapter: int,
    content_block_count: int,
    scene_count: int,
) -> int:
    """Scale chapter illustration cap down for very short chapters."""
    if content_block_count <= 2 and scene_count <= 1:
        return min(max_per_chapter, 1)
    if 3 <= content_block_count <= 8 and scene_count <= 2:
        return min(max_per_chapter, 2)
    return max_per_chapter


def _moment_block_position(
    card: SelectedMomentCard, block_order_by_id: Mapping[str, int]
) -> int | None:
    if card.insert_after_block_id and card.insert_after_block_id in block_order_by_id:
        return block_order_by_id[card.insert_after_block_id]
    for anchor in card.anchors:
        if anchor.block_id in block_order_by_id:
            return block_order_by_id[anchor.block_id]
    return None


def moments_conflict_diversity(
    a: SelectedMomentCard,
    b: SelectedMomentCard,
    *,
    block_order_by_id: Mapping[str, int] | None = None,
    min_block_gap: int = 0,
) -> bool:
    """Return `True` iff two moments represent a duplicate or too-close beat within the same chapter.

    - Overlapping quote anchors on the same block always conflict.
    - Identical normalized `title_ru` or `event_ru` always conflict.
    - Same `scene_id` + same normalized `event_kind` conflict (same beat type in same scene).
    - Distinct scenes with distinct conflicts sharing a broad `event_kind` (e.g. two different
      `'action'` conflicts in different scenes) do NOT conflict unless `min_block_gap > 0`
      and their resolved block positions are closer than `min_block_gap`.
    """
    if a.candidate_id == b.candidate_id:
        return True
    if _anchors_overlap(a.anchors, b.anchors):
        return True
    if normalize_name_token(a.title_ru) and normalize_name_token(a.title_ru) == normalize_name_token(
        b.title_ru
    ):
        return True
    if normalize_name_token(a.event_ru) and normalize_name_token(a.event_ru) == normalize_name_token(
        b.event_ru
    ):
        return True
    if (
        a.scene_id == b.scene_id
        and normalize_name_token(a.event_kind)
        and normalize_name_token(a.event_kind) == normalize_name_token(b.event_kind)
    ):
        return True
    if min_block_gap > 0 and block_order_by_id is not None:
        pos_a = _moment_block_position(a, block_order_by_id)
        pos_b = _moment_block_position(b, block_order_by_id)
        if pos_a is not None and pos_b is not None and abs(pos_a - pos_b) < min_block_gap:
            return True
    return False


def enforce_intra_chapter_diversity(
    winners: Sequence[SelectedMomentCard],
    shortlist: Sequence[SelectedMomentCard],
    *,
    zero_reason_ru: str | None = None,
    block_order_by_id: Mapping[str, int] | None = None,
    min_block_gap: int | None = None,
) -> tuple[tuple[SelectedMomentCard, ...], tuple[SelectedMomentCard, ...], str | None]:
    """Validate pairwise diversity among a chapter's winners, replacing duplicates from LIFO shortlist.

    - Promoted runner-ups use their grounded `significance_ru` as `reason_ru`.
    - At most `len(shortlist)` replacement attempts are performed.
    - Final winners and rejected duplicate beats are stripped from the returned `shortlist`.
    """
    if min_block_gap is None:
        effective_gap = 2 if (block_order_by_id is not None and len(block_order_by_id) >= 12) else 0
    else:
        effective_gap = min_block_gap if block_order_by_id is not None else 0

    accepted: list[SelectedMomentCard] = []
    rejected_ids: set[str] = set()
    available_shortlist = list(shortlist)

    for candidate in winners:
        if not any(
            moments_conflict_diversity(
                candidate,
                existing,
                block_order_by_id=block_order_by_id,
                min_block_gap=effective_gap,
            )
            for existing in accepted
        ):
            accepted.append(candidate)
            continue

        # Duplicate beat -> reject and try replacing strictly from LIFO shortlist.
        rejected_ids.add(candidate.candidate_id)
        replacement: SelectedMomentCard | None = None
        for idx, runner_up in enumerate(available_shortlist):
            if runner_up.candidate_id in rejected_ids or any(
                runner_up.candidate_id == w.candidate_id for w in accepted
            ):
                continue
            if not any(
                moments_conflict_diversity(
                    runner_up,
                    existing,
                    block_order_by_id=block_order_by_id,
                    min_block_gap=effective_gap,
                )
                for existing in accepted
            ):
                promoted = SelectedMomentCard(
                    candidate_id=runner_up.candidate_id,
                    section_id=runner_up.section_id,
                    scene_id=runner_up.scene_id,
                    title_ru=runner_up.title_ru,
                    event_ru=runner_up.event_ru,
                    significance_ru=runner_up.significance_ru,
                    reason_ru=runner_up.significance_ru,
                    event_kind=runner_up.event_kind,
                    event_position_key=runner_up.event_position_key,
                    anchors=runner_up.anchors,
                    insert_after_block_id=runner_up.insert_after_block_id,
                    participant_names=runner_up.participant_names,
                    location_names=runner_up.location_names,
                    scene_start_block_id=runner_up.scene_start_block_id,
                    scene_end_block_id=runner_up.scene_end_block_id,
                )
                replacement = promoted
                available_shortlist.pop(idx)
                break
        if replacement is not None:
            accepted.append(replacement)

    accepted.sort(key=lambda m: (m.event_position_key, m.candidate_id))
    winner_ids = {m.candidate_id for m in accepted}
    cleaned_shortlist: list[SelectedMomentCard] = []
    seen_shortlist: set[str] = set()
    for item in available_shortlist:
        if item.candidate_id in winner_ids or item.candidate_id in rejected_ids:
            continue
        if item.candidate_id in seen_shortlist:
            continue
        seen_shortlist.add(item.candidate_id)
        cleaned_shortlist.append(item)

    final_zero_reason = zero_reason_ru
    if not accepted:
        final_zero_reason = (zero_reason_ru or "").strip() or _DEFAULT_ZERO_REASON_RU
    else:
        final_zero_reason = None
    return tuple(accepted), tuple(cleaned_shortlist[:_MAX_SHORTLIST]), final_zero_reason


def deduplicate_candidates(
    cards: Sequence[SelectedMomentCard],
) -> tuple[tuple[SelectedMomentCard, ...], tuple[tuple[str, str], ...]]:
    """Deduplicate extracted candidates by `(scene_id, normalized evidence)` keeping earliest."""
    ordered = sorted(cards, key=lambda c: (c.event_position_key, c.candidate_id))
    kept: list[SelectedMomentCard] = []
    rejected: list[tuple[str, str]] = []
    for card in ordered:
        duplicate_of: SelectedMomentCard | None = None
        for prev in kept:
            if prev.scene_id != card.scene_id:
                continue
            same_title = normalize_name_token(prev.title_ru) == normalize_name_token(card.title_ru)
            same_event = normalize_name_token(prev.event_ru) == normalize_name_token(card.event_ru)
            overlap_same_kind = _anchors_overlap(prev.anchors, card.anchors) and (
                normalize_name_token(prev.event_kind) == normalize_name_token(card.event_kind)
            )
            if same_title or same_event or overlap_same_kind:
                duplicate_of = prev
                break
        if duplicate_of is not None:
            rejected.append((card.candidate_id, f"дубликат события {duplicate_of.candidate_id}"))
            continue
        kept.append(card)
    return tuple(kept), tuple(rejected)


def _scene_candidate_units(
    entry: SectionManifestEntry,
    section: DocumentSection,
) -> list[dict[str, Any]]:
    """Build ordered per-scene source units for `ILLUSTRATION_CANDIDATES`."""
    blocks = textual_blocks(section)
    if not blocks:
        return []
    block_order_by_id = {b.id: idx for idx, b in enumerate(blocks)}
    units: list[dict[str, Any]] = []
    if entry.scenes:
        for s_idx, scene in enumerate(entry.scenes):
            s_start = block_order_by_id.get(scene.start_block_id, 0)
            s_end = block_order_by_id.get(scene.end_block_id, len(blocks) - 1)
            if s_end < s_start:
                s_start, s_end = s_end, s_start
            scene_blocks = blocks[s_start : s_end + 1]
            if not scene_blocks:
                continue
            units.append(
                {
                    "scene_id": scene.id or f"{entry.section_id}:scene:{s_idx}",
                    "scene_order": s_idx,
                    "scene_summary": scene.summary,
                    "location": scene.location,
                    "participants": list(scene.participants),
                    "scene_start_block_id": scene_blocks[0].id,
                    "scene_end_block_id": scene_blocks[-1].id,
                    "blocks": [
                        {
                            "source_unit_id": b.id,
                            "block_id": b.id,
                            "block_order": block_order_by_id[b.id],
                            "text": source_text_for_model(b),
                        }
                        for b in scene_blocks
                    ],
                }
            )
    if not units:
        synthetic_scene_id = f"{entry.section_id}:scene:0"
        units.append(
            {
                "scene_id": synthetic_scene_id,
                "scene_order": 0,
                "scene_summary": entry.title,
                "location": "",
                "participants": [],
                "scene_start_block_id": blocks[0].id,
                "scene_end_block_id": blocks[-1].id,
                "blocks": [
                    {
                        "source_unit_id": b.id,
                        "block_id": b.id,
                        "block_order": idx,
                        "text": source_text_for_model(b),
                    }
                    for idx, b in enumerate(blocks)
                ],
            }
        )
    return units


def _compact_card_for_selection(card: SelectedMomentCard) -> dict[str, Any]:
    return {
        "candidate_id": card.candidate_id,
        "scene_id": card.scene_id,
        "title_ru": card.title_ru,
        "event_ru": card.event_ru,
        "significance_ru": card.significance_ru,
        "event_kind": card.event_kind,
        "event_position_key": card.event_position_key,
        "participant_names": list(card.participant_names),
        "location_names": list(card.location_names),
    }


def _build_candidate_scope_spec(
    *,
    section_id: str,
    section_order: int,
    section_title: str,
    scope_key: str,
    scene_meta: Mapping[str, Any],
    blocks_chunk: Sequence[Mapping[str, Any]],
    depth: int,
    revision: IllustrationSettingsRevision,
    route_id: str,
    known_characters: Sequence[str] = (),
    known_locations: Sequence[str] = (),
    known_entity_aliases: Mapping[str, Sequence[str]] | None = None,
    glossary_terms_ru: Mapping[str, str] | None = None,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    policy = revision.settings.policy
    expected_unit_ids = [str(b["source_unit_id"]) for b in blocks_chunk]
    ordered_block_ids = [str(b["block_id"]) for b in blocks_chunk]
    request_input: dict[str, Any] = {
        "section_id": section_id,
        "section_title": section_title,
        "scene_id": str(scene_meta["scene_id"]),
        "scene_summary": str(scene_meta.get("scene_summary", "")),
        "location": str(scene_meta.get("location", "")),
        "participants": list(scene_meta.get("participants", ())),
        "known_characters": list(known_characters),
        "known_locations": list(known_locations),
        "max_candidates": policy.max_candidates_per_scene,
        "blocks": [thaw(b) for b in blocks_chunk],
    }
    if known_entity_aliases:
        cleaned_aliases = {
            str(k): [str(a) for a in v if a is not None and str(a).strip()]
            for k, v in known_entity_aliases.items()
            if k is not None and str(k).strip() and v
        }
        cleaned_aliases = {k: v for k, v in cleaned_aliases.items() if v}
        if cleaned_aliases:
            request_input["known_entity_aliases"] = cleaned_aliases
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
            "stage": IllustrationScopeStage.CANDIDATES.value,
            "request_input": request_input,
            "model_prompts": revision.settings.model_prompts,
            "route_id": route_id,
            "style": revision.settings.style,
            "policy": policy.to_payload(),
        }
    )
    summary = {
        "depth": depth,
        "is_composite": len(blocks_chunk) > 1,
        "scene_id": str(scene_meta["scene_id"]),
        "scene_start_block_id": str(scene_meta["scene_start_block_id"]),
        "scene_end_block_id": str(scene_meta["scene_end_block_id"]),
        "expected_source_unit_ids": expected_unit_ids,
        "ordered_block_ids": ordered_block_ids,
        "max_candidates": policy.max_candidates_per_scene,
        "request_input": request_input,
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.CANDIDATES,
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
        stage=IllustrationScopeStage.CANDIDATES,
        scope_key=scope_key,
        input_sha256=input_sha256,
        section_id=section_id,
        section_order=section_order,
        section_title=section_title,
        summary=summary,
        state=IllustrationScopeState.PENDING,
    )


def _build_selection_scope_spec(
    *,
    section_id: str,
    section_order: int,
    section_title: str,
    scope_key: str,
    round_num: int,
    group_idx: int,
    cards: Sequence[SelectedMomentCard],
    group_cap: int,
    depth: int,
    revision: IllustrationSettingsRevision,
    route_id: str,
    chapter_cap: int | None = None,
    carried_singletons: Sequence[SelectedMomentCard] = (),
    prior_shortlist: Sequence[SelectedMomentCard] = (),
    prior_zero_reasons: Sequence[str] = (),
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    policy = revision.settings.policy
    effective_cap = (
        revision.settings.max_per_chapter if chapter_cap is None else chapter_cap
    )
    request_input = {
        "section_id": section_id,
        "section_title": section_title,
        "round": round_num,
        "group_cap": group_cap,
        "candidates": [_compact_card_for_selection(c) for c in cards],
    }
    input_sha256 = canonical_sha(
        {
            "stage": IllustrationScopeStage.SELECTION.value,
            "request_input": request_input,
            "model_prompts": revision.settings.model_prompts,
            "route_id": route_id,
            "style": revision.settings.style,
            "max_per_chapter": revision.settings.max_per_chapter,
            "effective_max_per_chapter": effective_cap,
            "policy": policy.to_payload(),
        }
    )
    ordered_block_ids = tuple(
        dict.fromkeys(a.block_id for c in cards for a in c.anchors)
    )
    summary = {
        "depth": depth,
        "is_composite": len(cards) > 2,
        "kind": "tournament_group",
        "round": round_num,
        "group_idx": group_idx,
        "group_cap": group_cap,
        "effective_max_per_chapter": effective_cap,
        "expected_candidate_ids": [c.candidate_id for c in cards],
        "cards": [c.to_dict() for c in cards],
        "carried_singletons": [c.to_dict() for c in carried_singletons],
        "prior_shortlist": [c.to_dict() for c in prior_shortlist],
        "prior_zero_reasons": list(prior_zero_reasons),
        "ordered_block_ids": list(ordered_block_ids),
        "request_input": request_input,
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.SELECTION,
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
        stage=IllustrationScopeStage.SELECTION,
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
    request_split: bool = False
    split_eligible: bool = False


class IllustrationSelection:
    """Plans and executes candidate extraction and finite tournament selection (`ALG-03`)."""

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
        on_selection_ready: Callable[[UnitOfWork, IllustrationGeneration, IllustrationSettingsRevision, IllustrationSourceSnapshot, str], bool] | None = None,
    ) -> None:
        self._uow = unit_of_work
        self._source = source
        self._runtime = runtime
        self._id_factory = id_factory or _default_id
        self._notify_work_available = notify_work_available
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=3)
        self._sleep = sleep_fn or asyncio.sleep
        self._on_selection_ready = on_selection_ready

    def plan_scopes(
        self,
        generation_id: str,
        section_id: str,
    ) -> tuple[IllustrationScope, ...]:
        """Plan bounded candidate scopes for one chapter in its own transaction."""
        dispatched = False
        with self._uow.transaction() as tx:
            gen = self._find_generation(tx, generation_id)
            if gen is None:
                return ()
            revision = tx.illustrations.get_settings(gen.book_id, gen.settings_id)
            if revision is None or not revision.settings.enabled:
                return ()
            snapshot = self._source.snapshot(tx, gen.book_id)
            if not snapshot.ready:
                return ()
            scopes, dispatched = self.plan_scopes_in_tx(
                tx,
                gen,
                revision,
                snapshot,
                section_id,
                id_factory=self._id_factory,
            )
        if dispatched and self._notify_work_available is not None:
            self._notify_work_available()
        return scopes

    def plan_scopes_in_tx(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        snapshot: IllustrationSourceSnapshot,
        section_id: str,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> tuple[tuple[IllustrationScope, ...], bool]:
        """Plan bounded `CANDIDATES` scopes (or advance tournament) inside an open transaction."""
        factory = id_factory or self._id_factory
        existing = tx.illustrations.list_scopes(
            generation.id, stage=IllustrationScopeStage.CANDIDATES, section_id=section_id
        )
        if existing:
            advanced = self._advance_section(
                tx, generation, revision, snapshot, section_id, id_factory=factory
            )
            return existing, advanced

        entry = next((e for e in snapshot.entries if e.section_id == section_id), None)
        if entry is None or entry.skipped:
            return (), False
        document = tx.documents.get(generation.book_id)
        if document is None:
            return (), False
        section = next((s for s in document.ordered_sections if s.id == section_id), None)
        if section is None:
            return (), False

        scene_units = _scene_candidate_units(entry, section)
        if not scene_units:
            self._persist_final_selection(
                tx,
                generation,
                revision,
                entry,
                SelectedMoments(
                    section_id=section_id,
                    winners=(),
                    shortlist=(),
                    zero_reason_ru=_DEFAULT_ZERO_REASON_RU,
                ),
            )
            return (), False

        policy = revision.settings.policy
        sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_CHAPTER)
        route_id = self._runtime.route_id()
        known_characters, known_locations, known_entity_aliases = (
            _known_entities_context_for_chapter(
                tx, generation.id, section_order=entry.order
            )
        )
        chapter_glossary_terms_ru = _collect_book_glossary_terms_ru(
            tx,
            generation.book_id,
            generation_id=generation.id,
            max_section_order=entry.order,
        )

        specs: list[IllustrationScopeSpec] = []
        for s_idx, scene_meta in enumerate(scene_units):
            blocks_list = scene_meta["blocks"]

            def _make_cand_req(
                chunk: Sequence[dict[str, Any]],
                cur_glossary: Mapping[str, str],
                _meta: Mapping[str, Any] = scene_meta,
            ) -> LLMRequest[Any]:
                payload: dict[str, Any] = {
                    "section_id": section_id,
                    "section_title": entry.title,
                    "scene_id": str(_meta["scene_id"]),
                    "scene_summary": str(_meta.get("scene_summary", "")),
                    "location": str(_meta.get("location", "")),
                    "participants": list(_meta.get("participants", ())),
                    "known_characters": list(known_characters),
                    "known_locations": list(known_locations),
                    "max_candidates": policy.max_candidates_per_scene,
                    "blocks": list(chunk),
                }
                if known_entity_aliases:
                    payload["known_entity_aliases"] = {
                        k: list(v) for k, v in known_entity_aliases.items()
                    }
                if cur_glossary:
                    payload["glossary_terms_ru"] = dict(cur_glossary)
                return IllustrationContext.build_for_kind(
                    kind=LLMResponseKind.ILLUSTRATION_CANDIDATES,
                    stage=IllustrationScopeStage.CANDIDATES,
                    input_payload=payload,
                    book_id=snapshot.book_id,
                    conditions_id=snapshot.conditions_id,
                    policy=policy,
                )

            bounded_glossary_terms_ru: dict[str, str] = {}
            if chapter_glossary_terms_ru and blocks_list:
                heaviest_block = max(
                    blocks_list,
                    key=lambda b: len(str(b.get("text", "")).encode("utf-8")),
                )
                bounded_glossary_terms_ru = fit_glossary_terms_to_wire_budget(
                    chapter_glossary_terms_ru,
                    build_request=lambda g, _hb=heaviest_block: _make_cand_req([_hb], g),
                    sizer=sizer,
                    target_bytes=policy.initial_target_bytes,
                )

            def _build_req(
                chunk: Sequence[dict[str, Any]],
                _gloss: Mapping[str, str] = bounded_glossary_terms_ru,
            ) -> LLMRequest[Any]:
                return _make_cand_req(chunk, _gloss)

            chunks = pack_items_by_wire_budget(
                blocks_list,
                build_request=_build_req,
                sizer=sizer,
                target_bytes=policy.initial_target_bytes,
                max_items=_MAX_CANDIDATE_BLOCKS_PER_CHUNK,
            )
            for c_idx, chunk in enumerate(chunks):
                scope_key = f"cand:{section_id}:s{s_idx:03d}:p{c_idx:03d}"
                draft = _build_candidate_scope_spec(
                    section_id=section_id,
                    section_order=entry.order,
                    section_title=entry.title,
                    scope_key=scope_key,
                    scene_meta=scene_meta,
                    blocks_chunk=chunk,
                    depth=0,
                    revision=revision,
                    route_id=route_id,
                    known_characters=known_characters,
                    known_locations=known_locations,
                    known_entity_aliases=known_entity_aliases,
                    glossary_terms_ru=bounded_glossary_terms_ru,
                )
                reusable = tx.illustrations.find_reusable_scope(
                    generation.book_id,
                    stage=IllustrationScopeStage.CANDIDATES,
                    input_sha256=draft.input_sha256,
                )
                if reusable is not None:
                    draft = _build_candidate_scope_spec(
                        section_id=section_id,
                        section_order=entry.order,
                        section_title=entry.title,
                        scope_key=scope_key,
                        scene_meta=scene_meta,
                        blocks_chunk=chunk,
                        depth=0,
                        revision=revision,
                        route_id=route_id,
                        known_characters=known_characters,
                        known_locations=known_locations,
                        known_entity_aliases=known_entity_aliases,
                        glossary_terms_ru=bounded_glossary_terms_ru,
                        reusable=reusable,
                    )
                specs.append(draft)

        scopes = tx.illustrations.upsert_scopes(generation.id, specs)
        dispatched = False
        for scope in scopes:
            if scope.state is IllustrationScopeState.PENDING:
                dispatch_scope_task(
                    tx,
                    scope,
                    generation_id=generation.id,
                    task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                    id_factory=factory,
                    book_id=generation.book_id,
                )
                dispatched = True
        if not dispatched:
            if self._advance_section(
                tx, generation, revision, snapshot, section_id, id_factory=factory
            ):
                dispatched = True
        return scopes, dispatched

    def selected_moments(
        self,
        generation_id: str,
        section_id: str,
    ) -> SelectedMoments:
        """Read finalized `SelectedMoments` for a chapter (winners, LIFO shortlist, zero_reason)."""
        with self._uow.transaction() as tx:
            return self.selected_moments_in_tx(tx, generation_id, section_id)

    def selected_moments_in_tx(
        self,
        tx: UnitOfWork,
        generation_id: str,
        section_id: str,
    ) -> SelectedMoments:
        scopes = tx.illustrations.list_scopes(
            generation_id, stage=IllustrationScopeStage.SELECTION, section_id=section_id
        )
        final_key = f"select:{section_id}:final"
        final_scope = next(
            (s for s in scopes if s.scope_key == final_key and s.state is IllustrationScopeState.ACCEPTED),
            None,
        )
        if final_scope is None or final_scope.payload is None:
            raise IllustrationError(
                IllustrationReasonCode.GENERATION_CONFLICT,
                f"отбор моментов для главы {section_id} ещё не завершён",
            )
        payload = final_scope.payload
        winners = tuple(SelectedMomentCard.from_dict(w) for w in payload.get("winners", ()))
        shortlist = tuple(SelectedMomentCard.from_dict(s) for s in payload.get("shortlist", ()))
        zero_reason = payload.get("zero_reason_ru")
        return SelectedMoments(
            section_id=section_id,
            winners=winners,
            shortlist=shortlist,
            zero_reason_ru=str(zero_reason) if zero_reason is not None else None,
        )

    async def run_task(self, task: ProcessingTask) -> None:
        """Execute one `CANDIDATES` or `SELECTION` scope using 3-tx `ALG-04`."""
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
                if scope.stage not in {
                    IllustrationScopeStage.CANDIDATES,
                    IllustrationScopeStage.SELECTION,
                }:
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
                        if self._advance_section(
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

                block_lookup = self._load_section_blocks(tx, book_id, section_id, snapshot)
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

            is_composite = bool(scope.summary.get("is_composite", False))
            depth = int(scope.summary.get("depth", 0))
            size_cap = (
                policy.initial_target_bytes
                if (repair_summary is None and is_composite)
                else policy.max_request_bytes
            )
            if wire.size_bytes > size_cap:
                notify_after = self._abort_oversized_or_split(
                    book_id=book_id,
                    generation_id=generation_id,
                    task_id=task.id,
                    scope=scope,
                    attempt_id=attempt.id,
                    revision=revision,
                    is_composite=is_composite,
                    depth=depth,
                    policy=policy,
                    reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                )
                break

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
                payload_for_validation = response.payload
                if scope.stage is IllustrationScopeStage.CANDIDATES:
                    raw_input = scope.summary.get("request_input")
                    if isinstance(raw_input, Mapping):
                        _, alias_to_canonical, _ = encode_wire_id_aliases(scope.stage, raw_input)
                        raw_scene_id = scope.summary.get("scene_id")
                        single_scene_id = (
                            str(raw_scene_id).strip()
                            if raw_scene_id is not None and str(raw_scene_id).strip()
                            else None
                        )
                        payload_for_validation = decode_wire_id_aliases(
                            payload_for_validation,
                            alias_to_canonical,
                            single_scene_id=single_scene_id,
                        )
                validation = self._validate_response(scope, payload_for_validation, block_lookup)
            except Exception as err:
                call_error = err

            if call_error is not None:
                if response is not None:
                    record_rejection(self._uow, response, call_error)
                if is_context_limit_error(call_error):
                    if is_composite:
                        notify_after = self._handle_post_send_split(
                            book_id=book_id,
                            generation_id=generation_id,
                            task_id=task.id,
                            scope=scope,
                            attempt_id=attempt.id,
                            revision=revision,
                            depth=depth,
                            policy=policy,
                            fail_reason=IllustrationReasonCode.CONTEXT_LIMIT.value,
                        )
                        break
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
                    if is_composite:
                        notify_after = self._handle_post_send_split(
                            book_id=book_id,
                            generation_id=generation_id,
                            task_id=task.id,
                            scope=scope,
                            attempt_id=attempt.id,
                            revision=revision,
                            depth=depth,
                            policy=policy,
                            fail_reason=IllustrationReasonCode.PROVIDER_TRUNCATED.value,
                        )
                        break
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
                    if scope.stage is IllustrationScopeStage.CANDIDATES and is_composite:
                        notify_after = self._handle_post_send_split(
                            book_id=book_id,
                            generation_id=generation_id,
                            task_id=task.id,
                            scope=scope,
                            attempt_id=attempt.id,
                            revision=revision,
                            depth=depth,
                            policy=policy,
                            fail_reason=IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
                            detail=err_detail,
                        )
                        break
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
            if validation.request_split:
                split_detail = "модель запросила дробление (complete=false)"
                if response is not None:
                    record_rejection(self._uow, response, ResultValidationError(split_detail))
                if is_composite:
                    notify_after = self._handle_post_send_split(
                        book_id=book_id,
                        generation_id=generation_id,
                        task_id=task.id,
                        scope=scope,
                        attempt_id=attempt.id,
                        revision=revision,
                        depth=depth,
                        policy=policy,
                        fail_reason=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                        detail=split_detail,
                    )
                    break
                self._finalize_failure(
                    book_id=book_id,
                    generation_id=generation_id,
                    task_id=task.id,
                    scope_id=scope.id,
                    attempt_id=attempt.id,
                    reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
                    detail=split_detail,
                )
                return

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
                if (
                    scope.stage is IllustrationScopeStage.CANDIDATES
                    and is_composite
                    and validation.split_eligible
                ):
                    notify_after = self._handle_post_send_split(
                        book_id=book_id,
                        generation_id=generation_id,
                        task_id=task.id,
                        scope=scope,
                        attempt_id=attempt.id,
                        revision=revision,
                        depth=depth,
                        policy=policy,
                        fail_reason=validation.reason_code,
                        detail=detail_text,
                    )
                    break
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
                if self._advance_section(
                    tx, gen, revision, cur_snap, section_id, id_factory=self._id_factory
                ):
                    notify_after = True
                else:
                    finalize_generation_state_in_tx(tx, book_id, gen.id)
                self._mark_task_succeeded(tx, task.id)
                break

        if notify_after and self._notify_work_available is not None:
            self._notify_work_available()

    def _validate_response(
        self,
        scope: IllustrationScope,
        payload: Any,
        block_lookup: Mapping[str, tuple[int, int, DocumentBlock]],
    ) -> _ValidationOutcome:
        if scope.stage is IllustrationScopeStage.CANDIDATES:
            if not isinstance(payload, IllustrationCandidatesResponse):
                return _ValidationOutcome(ok=False, repair_reason="ожидался ответ кандидатов")
            return self._validate_candidates(scope, payload, block_lookup)
        if scope.stage is IllustrationScopeStage.SELECTION:
            if not isinstance(payload, IllustrationSelectionResponse):
                return _ValidationOutcome(ok=False, repair_reason="ожидался ответ отбора")
            return self._validate_selection(scope, payload)
        return _ValidationOutcome(ok=False, repair_reason="неподдерживаемый этап отбора")

    def _validate_candidates(
        self,
        scope: IllustrationScope,
        dto: IllustrationCandidatesResponse,
        block_lookup: Mapping[str, tuple[int, int, DocumentBlock]],
    ) -> _ValidationOutcome:
        if not dto.complete:
            return _ValidationOutcome(ok=False, request_split=True)

        expected_units = set(str(u) for u in scope.summary.get("expected_source_unit_ids", ()))
        covered = set(dto.covered_source_unit_ids)
        if covered != expected_units:
            ordered_blocks_set = {str(b) for b in scope.summary.get("ordered_block_ids", ())}
            if not (
                len(ordered_blocks_set) == len(expected_units)
                and covered == ordered_blocks_set
            ):
                missing = sorted(expected_units - covered)
                extra = sorted(covered - expected_units)
                parts = [
                    "неполный или лишний covered_source_unit_ids при complete=true — "
                    "перечисли ВСЕ source_unit_id из blocks (включая блоки без событий)"
                ]
                if missing:
                    parts.append(f"пропущены={missing[:4]}")
                if extra:
                    parts.append(f"лишние={extra[:4]}")
                return _ValidationOutcome(
                    ok=False,
                    repair_reason="; ".join(parts),
                    split_eligible=True,
                )

        expected_scene_id = str(scope.summary.get("scene_id", ""))
        if dto.scene_id != expected_scene_id:
            return _ValidationOutcome(
                ok=False,
                repair_reason=f"scene_id должен быть {expected_scene_id}",
            )

        max_candidates = int(scope.summary.get("max_candidates", 2))
        if len(dto.events) > max_candidates:
            return _ValidationOutcome(
                ok=False,
                repair_reason=f"не более {max_candidates} событий на сцену",
            )

        if not dto.events:
            reason = (dto.zero_reason_ru or "").strip()
            if not reason:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason="при пустом списке событий требуется непустая zero_reason_ru",
                )
            return _ValidationOutcome(
                ok=True,
                payload={
                    "schema_version": 1,
                    "scene_id": expected_scene_id,
                    "cards": [],
                    "zero_reason_ru": reason,
                },
            )

        ordered_blocks = [str(b) for b in scope.summary.get("ordered_block_ids", ())]
        allowed_blocks = set(ordered_blocks)
        pos_by_block_id = {bid: idx for idx, bid in enumerate(ordered_blocks)}
        section_id = str(scope.section_id or "")
        scene_start = str(scope.summary.get("scene_start_block_id", ""))
        scene_end = str(scope.summary.get("scene_end_block_id", ""))
        scene_end_idx = pos_by_block_id.get(scene_end, len(ordered_blocks) - 1)
        cards: list[dict[str, Any]] = []
        seen_local_ids: set[str] = set()

        for ev in dto.events:
            if ev.event_id in seen_local_ids:
                return _ValidationOutcome(
                    ok=False, repair_reason=f"дублируется event_id {ev.event_id}"
                )
            if not ev.significance_ru.strip():
                return _ValidationOutcome(
                    ok=False,
                    repair_reason=f"для события {ev.event_id} требуется непустое significance_ru",
                )
            if not ev.quote_anchors:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason=f"для события {ev.event_id} требуется хотя бы одна цитатная опора",
                )

            resolved_anchors: list[SelectedAnchor] = []
            for qa in ev.quote_anchors:
                if qa.block_id not in allowed_blocks or qa.block_id not in block_lookup:
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=f"опора события {ev.event_id} ссылается на недопустимый блок {qa.block_id}",
                    )
                sec_order, blk_order, blk = block_lookup[qa.block_id]
                res = resolve_quote_anchor(
                    blk, qa.quote, section_order=sec_order, block_order=blk_order
                )
                if res.status == "ambiguous":
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=f"цитата события {ev.event_id} повторяется в блоке {qa.block_id}; дай более длинную уникальную подстроку",
                    )
                if res.status != "resolved" or res.anchor is None or res.position_key is None:
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=f"цитата события {ev.event_id} не найдена в блоке {qa.block_id}",
                    )
                resolved_anchors.append(
                    SelectedAnchor(
                        block_id=res.anchor.block_id,
                        start=res.anchor.start,
                        end=res.anchor.end,
                        quote=res.anchor.quote,
                        position_key=res.position_key,
                    )
                )

            seen_local_ids.add(ev.event_id)
            insert_after_candidate = (
                ev.insert_after_block_id.strip() if ev.insert_after_block_id else None
            )
            if insert_after_candidate is not None:
                last_anchor_idx = max(pos_by_block_id[a.block_id] for a in resolved_anchors)
                cand_idx = pos_by_block_id.get(insert_after_candidate)
                if (
                    cand_idx is None
                    or insert_after_candidate not in block_lookup
                    or cand_idx < last_anchor_idx
                    or cand_idx > scene_end_idx
                ):
                    return _ValidationOutcome(
                        ok=False,
                        repair_reason=(
                            f"insert_after_block_id {insert_after_candidate} для события {ev.event_id} "
                            "должен быть текстовым блоком этой сцены не раньше последней цитатной опоры"
                        ),
                    )
            event_pos = max(a.position_key for a in resolved_anchors)
            cand_hash = canonical_sha(
                {
                    "section_id": section_id,
                    "scene_id": expected_scene_id,
                    "event_pos": event_pos,
                    "title": ev.title_ru.strip(),
                    "event": ev.event_ru.strip(),
                    "anchors": [a.to_dict() for a in resolved_anchors],
                }
            )[:12]
            card = SelectedMomentCard(
                candidate_id=f"cand-{cand_hash}",
                section_id=section_id,
                scene_id=expected_scene_id,
                title_ru=ev.title_ru.strip(),
                event_ru=ev.event_ru.strip(),
                significance_ru=ev.significance_ru.strip(),
                reason_ru=ev.significance_ru.strip(),
                event_kind=ev.event_kind.strip(),
                event_position_key=event_pos,
                anchors=tuple(resolved_anchors),
                insert_after_block_id=insert_after_candidate,
                participant_names=tuple(n.strip() for n in ev.participant_names if n.strip()),
                location_names=tuple(n.strip() for n in ev.location_names if n.strip()),
                scene_start_block_id=scene_start,
                scene_end_block_id=scene_end,
            )
            cards.append(card.to_dict())

        return _ValidationOutcome(
            ok=True,
            payload={
                "schema_version": 1,
                "scene_id": expected_scene_id,
                "cards": cards,
                "zero_reason_ru": None,
            },
        )

    def _validate_selection(
        self,
        scope: IllustrationScope,
        dto: IllustrationSelectionResponse,
    ) -> _ValidationOutcome:
        expected_ids = [str(cid) for cid in scope.summary.get("expected_candidate_ids", ())]
        expected_set = set(expected_ids)
        group_cap = int(scope.summary.get("group_cap", 1))

        selected = list(dto.selected_candidate_ids)
        if len(set(selected)) != len(selected):
            return _ValidationOutcome(
                ok=False, repair_reason="повторяющиеся candidate_id в selected_candidate_ids"
            )
        unknown = [cid for cid in selected if cid not in expected_set]
        if unknown:
            return _ValidationOutcome(
                ok=False,
                repair_reason=f"выдуманные candidate_id вне группы: {unknown[:4]}",
            )
        if len(selected) > group_cap:
            return _ValidationOutcome(
                ok=False,
                repair_reason=f"выбрано больше допустимого лимита group_cap={group_cap}",
            )

        sel_reason_map: dict[str, str] = {}
        for item in dto.selection_reasons_ru:
            if item.candidate_id not in expected_set:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason=f"неизвестный candidate_id в selection_reasons_ru: {item.candidate_id}",
                )
            if item.reason_ru.strip():
                sel_reason_map[item.candidate_id] = item.reason_ru.strip()

        rej_reason_map: dict[str, str] = {}
        for item in dto.rejected_reasons_ru:
            if item.candidate_id not in expected_set:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason=f"неизвестный candidate_id в rejected_reasons_ru: {item.candidate_id}",
                )
            if item.reason_ru.strip():
                rej_reason_map[item.candidate_id] = item.reason_ru.strip()

        if not selected:
            zero_reason = (dto.zero_reason_ru or "").strip()
            if not zero_reason:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason="при пустом отборе (0 кандидатов) требуется непустая zero_reason_ru",
                )
        else:
            missing_reasons = [cid for cid in selected if cid not in sel_reason_map]
            if missing_reasons:
                return _ValidationOutcome(
                    ok=False,
                    repair_reason=f"отсутствуют selection_reasons_ru для {missing_reasons[:4]}",
                )

        return _ValidationOutcome(
            ok=True,
            payload={
                "schema_version": 1,
                "selected_candidate_ids": selected,
                "selection_reasons_ru": sel_reason_map,
                "rejected_reasons_ru": rej_reason_map,
                "zero_reason_ru": (dto.zero_reason_ru or "").strip() or None,
            },
        )

    def _advance_section(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        snapshot: IllustrationSourceSnapshot,
        section_id: str,
        *,
        id_factory: Callable[[], str],
    ) -> bool:
        """Advance `CANDIDATES -> tournament SELECTION rounds -> finalized SelectedMoments`."""
        entry = next((e for e in snapshot.entries if e.section_id == section_id), None)
        if entry is None or entry.skipped:
            return False

        sec_scopes = tx.illustrations.list_scopes(generation.id, section_id=section_id)
        final_key = f"select:{section_id}:final"
        if any(
            s.stage is IllustrationScopeStage.SELECTION
            and s.scope_key == final_key
            and s.state is IllustrationScopeState.ACCEPTED
            for s in sec_scopes
        ):
            if self._on_selection_ready is not None:
                return bool(
                    self._on_selection_ready(tx, generation, revision, snapshot, section_id)
                )
            return False

        cand_leaves = [
            s
            for s in sec_scopes
            if s.stage is IllustrationScopeStage.CANDIDATES
            and s.state is not IllustrationScopeState.SPLIT
        ]
        if not cand_leaves or any(s.state is not IllustrationScopeState.ACCEPTED for s in cand_leaves):
            return False

        sel_scopes = [
            s
            for s in sec_scopes
            if s.stage is IllustrationScopeStage.SELECTION and s.scope_key != final_key
        ]

        max_per_chapter = effective_chapter_illustration_cap(
            revision.settings.max_per_chapter,
            len(entry.block_digests),
            len(entry.scenes),
        )

        if not sel_scopes:
            # Collect & deduplicate all candidate cards from accepted CANDIDATES leaves.
            raw_cards: list[SelectedMomentCard] = []
            cand_zero_reasons: list[str] = []
            for s in sorted(cand_leaves, key=lambda sc: sc.scope_key):
                payload = s.payload or {}
                for raw_c in payload.get("cards", ()):
                    raw_cards.append(SelectedMomentCard.from_dict(raw_c))
                zr = payload.get("zero_reason_ru")
                if isinstance(zr, str) and zr.strip():
                    cand_zero_reasons.append(zr.strip())

            deduped, _rejected = deduplicate_candidates(raw_cards)
            if not deduped:
                consolidated = "; ".join(dict.fromkeys(cand_zero_reasons)) or _DEFAULT_ZERO_REASON_RU
                self._persist_final_selection(
                    tx,
                    generation,
                    revision,
                    entry,
                    SelectedMoments(
                        section_id=section_id,
                        winners=(),
                        shortlist=(),
                        zero_reason_ru=consolidated,
                    ),
                )
                if self._on_selection_ready is not None:
                    return bool(
                        self._on_selection_ready(tx, generation, revision, snapshot, section_id)
                    )
                return False

            # Schedule round 0 (editorial gate if count <= max_per_chapter) or round 1 (tournament).
            if len(deduped) <= max_per_chapter:
                return self._schedule_selection_round(
                    tx,
                    generation=generation,
                    revision=revision,
                    snapshot=snapshot,
                    entry=entry,
                    round_num=0,
                    cards=deduped,
                    prior_shortlist=(),
                    prior_zero_reasons=cand_zero_reasons,
                    is_final_gate=True,
                    id_factory=id_factory,
                )
            return self._schedule_selection_round(
                tx,
                generation=generation,
                revision=revision,
                snapshot=snapshot,
                entry=entry,
                round_num=1,
                cards=deduped,
                prior_shortlist=(),
                prior_zero_reasons=cand_zero_reasons,
                is_final_gate=False,
                id_factory=id_factory,
            )

        # Find highest round scheduled so far.
        rounds = [int(s.summary.get("round", 0)) for s in sel_scopes]
        cur_round = max(rounds)
        cur_round_leaves = [
            s
            for s in sel_scopes
            if int(s.summary.get("round", 0)) == cur_round
            and s.state is not IllustrationScopeState.SPLIT
        ]
        if not cur_round_leaves or any(
            s.state is not IllustrationScopeState.ACCEPTED for s in cur_round_leaves
        ):
            return False

        # All groups in `cur_round` are ACCEPTED -> aggregate winners, carried singletons, and LIFO losers!
        ordered_leaves = sorted(cur_round_leaves, key=lambda sc: sc.scope_key)
        prev_cards_by_id: dict[str, SelectedMomentCard] = {}
        next_cards: list[SelectedMomentCard] = []
        round_losers: list[SelectedMomentCard] = []
        carried: list[SelectedMomentCard] = []
        prior_shortlist: list[SelectedMomentCard] = []
        zero_reasons: list[str] = []

        first_summary = ordered_leaves[0].summary
        for c_raw in first_summary.get("carried_singletons", ()):
            carried.append(SelectedMomentCard.from_dict(c_raw))
        for s_raw in first_summary.get("prior_shortlist", ()):
            prior_shortlist.append(SelectedMomentCard.from_dict(s_raw))
        for zr in first_summary.get("prior_zero_reasons", ()):
            if isinstance(zr, str) and zr.strip():
                zero_reasons.append(zr.strip())

        total_prev_in_groups = 0
        for leaf in ordered_leaves:
            group_cards = [
                SelectedMomentCard.from_dict(c_raw) for c_raw in leaf.summary.get("cards", ())
            ]
            total_prev_in_groups += len(group_cards)
            by_id = {c.candidate_id: c for c in group_cards}
            prev_cards_by_id.update(by_id)
            payload = leaf.payload or {}
            sel_ids = [str(cid) for cid in payload.get("selected_candidate_ids", ())]
            sel_reasons = payload.get("selection_reasons_ru", {})
            sel_set = set(sel_ids)
            for cid in sel_ids:
                card = by_id[cid]
                reason_ru = str(sel_reasons.get(cid) or card.significance_ru).strip()
                next_cards.append(
                    SelectedMomentCard(
                        candidate_id=card.candidate_id,
                        section_id=card.section_id,
                        scene_id=card.scene_id,
                        title_ru=card.title_ru,
                        event_ru=card.event_ru,
                        significance_ru=card.significance_ru,
                        reason_ru=reason_ru,
                        event_kind=card.event_kind,
                        event_position_key=card.event_position_key,
                        anchors=card.anchors,
                        insert_after_block_id=card.insert_after_block_id,
                        participant_names=card.participant_names,
                        location_names=card.location_names,
                        scene_start_block_id=card.scene_start_block_id,
                        scene_end_block_id=card.scene_end_block_id,
                    )
                )
            for card in group_cards:
                if card.candidate_id not in sel_set:
                    round_losers.append(card)
            zr = payload.get("zero_reason_ru")
            if isinstance(zr, str) and zr.strip():
                zero_reasons.append(zr.strip())

        # Combine group winners + carried singletons in deterministic order.
        next_cards.extend(carried)
        next_cards.sort(key=lambda c: (c.event_position_key, c.candidate_id))
        round_losers.sort(key=lambda c: (c.event_position_key, c.candidate_id))

        # Strict progress check on tournament rounds (cur_round >= 1).
        prev_total = total_prev_in_groups + len(carried)
        if cur_round >= 1 and len(next_cards) >= prev_total:
            # Should be mathematically impossible due to group_cap = min(max_per_chapter, g-1),
            # but guard defensively per ALG-03.
            tx.illustrations.fail_scope(
                ordered_leaves[0].id,
                attempt_id=None,
                reason_code=IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
            )
            self._refresh_generation_state(
                tx,
                generation.book_id,
                generation.id,
                IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
            )
            return False

        # Prepend round losers to shortlist (LIFO by elimination round!).
        updated_shortlist = [*round_losers, *prior_shortlist][:_MAX_SHORTLIST]

        if not next_cards:
            consolidated = "; ".join(dict.fromkeys(zero_reasons)) or _DEFAULT_ZERO_REASON_RU
            self._persist_final_selection(
                tx,
                generation,
                revision,
                entry,
                SelectedMoments(
                    section_id=section_id,
                    winners=(),
                    shortlist=(),
                    zero_reason_ru=consolidated,
                ),
            )
            if self._on_selection_ready is not None:
                return bool(
                    self._on_selection_ready(tx, generation, revision, snapshot, section_id)
                )
            return False

        if len(next_cards) > max_per_chapter:
            return self._schedule_selection_round(
                tx,
                generation=generation,
                revision=revision,
                snapshot=snapshot,
                entry=entry,
                round_num=cur_round + 1,
                cards=next_cards,
                prior_shortlist=updated_shortlist,
                prior_zero_reasons=zero_reasons,
                is_final_gate=False,
                id_factory=id_factory,
            )

        # Count is now in 1..max_per_chapter (and editorial gate or tournament round ran) -> diversity check!
        consolidated_zero = "; ".join(dict.fromkeys(zero_reasons)) or None
        block_order_by_id = {bid: idx for idx, bid in enumerate(entry.block_digests)}
        final_winners, final_shortlist, final_zero = enforce_intra_chapter_diversity(
            next_cards,
            updated_shortlist,
            zero_reason_ru=consolidated_zero,
            block_order_by_id=block_order_by_id,
            min_block_gap=2 if len(entry.block_digests) >= 12 else 0,
        )
        self._persist_final_selection(
            tx,
            generation,
            revision,
            entry,
            SelectedMoments(
                section_id=section_id,
                winners=final_winners,
                shortlist=final_shortlist,
                zero_reason_ru=final_zero,
            ),
        )
        if self._on_selection_ready is not None:
            return bool(
                self._on_selection_ready(tx, generation, revision, snapshot, section_id)
            )
        return False

    def _schedule_selection_round(
        self,
        tx: UnitOfWork,
        *,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        snapshot: IllustrationSourceSnapshot,
        entry: SectionManifestEntry,
        round_num: int,
        cards: Sequence[SelectedMomentCard],
        prior_shortlist: Sequence[SelectedMomentCard],
        prior_zero_reasons: Sequence[str],
        is_final_gate: bool,
        id_factory: Callable[[], str],
    ) -> bool:
        policy = revision.settings.policy
        max_per_chapter = effective_chapter_illustration_cap(
            revision.settings.max_per_chapter,
            len(entry.block_digests),
            len(entry.scenes),
        )
        sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_CHAPTER)
        route_id = self._runtime.route_id()

        def _build_group_req(group_cards: Sequence[SelectedMomentCard]) -> LLMRequest[Any]:
            gcap = (
                min(max_per_chapter, len(group_cards))
                if is_final_gate
                else max(1, min(max_per_chapter, len(group_cards) - 1))
            )
            return IllustrationContext.build_for_kind(
                kind=LLMResponseKind.ILLUSTRATION_SELECTION,
                stage=IllustrationScopeStage.SELECTION,
                input_payload={
                    "section_id": entry.section_id,
                    "section_title": entry.title,
                    "round": round_num,
                    "group_cap": gcap,
                    "candidates": [_compact_card_for_selection(c) for c in group_cards],
                },
                book_id=snapshot.book_id,
                conditions_id=snapshot.conditions_id,
                policy=policy,
            )

        raw_groups: list[tuple[SelectedMomentCard, ...]] = []
        fan_in = policy.selection_fan_in
        for start in range(0, len(cards), fan_in):
            chunk = cards[start : start + fan_in]
            raw_groups.extend(
                pack_items_by_wire_budget(
                    chunk,
                    build_request=_build_group_req,
                    sizer=sizer,
                    target_bytes=policy.initial_target_bytes,
                )
            )

        if is_final_gate:
            # Final gate for 1..max_per_chapter cards: must fit in 1 group (or fail atomic_input_too_large).
            if len(raw_groups) == 1:
                g = raw_groups[0]
                if sizer.measure(_build_group_req(g)).size_bytes > policy.max_request_bytes:
                    self._fail_selection_atomic(tx, generation, entry, round_num)
                    return False
                eval_groups = [g]
                carried_singletons: list[SelectedMomentCard] = []
            else:
                # Even small set <= max_per_chapter split across bytes -> check if every group has size 1.
                if all(len(g) == 1 for g in raw_groups):
                    self._fail_selection_atomic(tx, generation, entry, round_num)
                    return False
                eval_groups = [g for g in raw_groups if len(g) >= 2]
                carried_singletons = [g[0] for g in raw_groups if len(g) == 1]
        else:
            # Tournament round: every group having size 1 means zero progress possible -> atomic_input_too_large!
            if all(len(g) == 1 for g in raw_groups):
                self._fail_selection_atomic(tx, generation, entry, round_num)
                return False
            eval_groups = [g for g in raw_groups if len(g) >= 2]
            carried_singletons = [g[0] for g in raw_groups if len(g) == 1]

        specs: list[IllustrationScopeSpec] = []
        for g_idx, g_cards in enumerate(eval_groups):
            g_cap = (
                min(max_per_chapter, len(g_cards))
                if (is_final_gate and len(eval_groups) == 1 and not carried_singletons)
                else min(max_per_chapter, len(g_cards) - 1)
            )
            scope_key = f"select:{entry.section_id}:r{round_num}:g{g_idx:03d}"
            draft = _build_selection_scope_spec(
                section_id=entry.section_id,
                section_order=entry.order,
                section_title=entry.title,
                scope_key=scope_key,
                round_num=round_num,
                group_idx=g_idx,
                cards=g_cards,
                group_cap=g_cap,
                depth=0,
                revision=revision,
                route_id=route_id,
                chapter_cap=max_per_chapter,
                carried_singletons=carried_singletons if g_idx == 0 else (),
                prior_shortlist=prior_shortlist if g_idx == 0 else (),
                prior_zero_reasons=prior_zero_reasons if g_idx == 0 else (),
            )
            reusable = tx.illustrations.find_reusable_scope(
                generation.book_id,
                stage=IllustrationScopeStage.SELECTION,
                input_sha256=draft.input_sha256,
            )
            if reusable is not None:
                draft = _build_selection_scope_spec(
                    section_id=entry.section_id,
                    section_order=entry.order,
                    section_title=entry.title,
                    scope_key=scope_key,
                    round_num=round_num,
                    group_idx=g_idx,
                    cards=g_cards,
                    group_cap=g_cap,
                    depth=0,
                    revision=revision,
                    route_id=route_id,
                    chapter_cap=max_per_chapter,
                    carried_singletons=carried_singletons if g_idx == 0 else (),
                    prior_shortlist=prior_shortlist if g_idx == 0 else (),
                    prior_zero_reasons=prior_zero_reasons if g_idx == 0 else (),
                    reusable=reusable,
                )
            specs.append(draft)

        created = tx.illustrations.upsert_scopes(generation.id, specs)
        dispatched = False
        for scope in created:
            if scope.state is IllustrationScopeState.PENDING:
                dispatch_scope_task(
                    tx,
                    scope,
                    generation_id=generation.id,
                    task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                    id_factory=id_factory,
                    book_id=generation.book_id,
                )
                dispatched = True
        if not dispatched:
            # All selection scopes in this round were reused -> advance immediately!
            return self._advance_section(
                tx, generation, revision, snapshot, entry.section_id, id_factory=id_factory
            )
        return True

    def _fail_selection_atomic(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        entry: SectionManifestEntry,
        round_num: int,
    ) -> None:
        spec = IllustrationScopeSpec(
            stage=IllustrationScopeStage.SELECTION,
            scope_key=f"select:{entry.section_id}:r{round_num}:atomic_fail",
            input_sha256=canonical_sha(
                {"section_id": entry.section_id, "round": round_num, "error": "atomic_input_too_large"}
            ),
            section_id=entry.section_id,
            section_order=entry.order,
            section_title=entry.title,
            state=IllustrationScopeState.FAILED,
            reason_code=IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
            summary={"round": round_num, "kind": "atomic_fail"},
        )
        tx.illustrations.upsert_scopes(generation.id, [spec])
        self._refresh_generation_state(
            tx,
            generation.book_id,
            generation.id,
            IllustrationReasonCode.ATOMIC_INPUT_TOO_LARGE.value,
        )

    def _persist_final_selection(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        entry: SectionManifestEntry,
        selected: SelectedMoments,
    ) -> None:
        payload = {
            "schema_version": 1,
            "section_id": selected.section_id,
            "winners": [w.to_dict() for w in selected.winners],
            "shortlist": [s.to_dict() for s in selected.shortlist],
            "zero_reason_ru": selected.zero_reason_ru,
        }
        output_sha = canonical_sha(payload)
        spec = IllustrationScopeSpec(
            stage=IllustrationScopeStage.SELECTION,
            scope_key=f"select:{entry.section_id}:final",
            input_sha256=canonical_sha(
                {
                    "section_id": entry.section_id,
                    "settings_id": revision.id,
                    "output_sha": output_sha,
                }
            ),
            section_id=entry.section_id,
            section_order=entry.order,
            section_title=entry.title,
            state=IllustrationScopeState.ACCEPTED,
            payload=payload,
            output_sha256=output_sha,
            summary={
                "kind": "final_selection",
                "winner_count": len(selected.winners),
                "shortlist_count": len(selected.shortlist),
            },
        )
        tx.illustrations.upsert_scopes(generation.id, [spec])

    def _split_children_for_scope(
        self,
        scope: IllustrationScope,
        revision: IllustrationSettingsRevision,
        depth: int,
    ) -> list[IllustrationScopeSpec] | None:
        route_id = self._runtime.route_id()
        req_input = scope.summary.get("request_input", {})
        if not isinstance(req_input, Mapping):
            return None
        section_id = str(scope.section_id or "")
        section_order = 0 if scope.section_order is None else scope.section_order
        section_title = scope.section_title or ""

        if scope.stage is IllustrationScopeStage.CANDIDATES:
            blocks = [thaw(b) for b in req_input.get("blocks", ())]
            if len(blocks) <= 1:
                return None
            mid = len(blocks) // 2
            scene_meta = {
                "scene_id": scope.summary.get("scene_id", ""),
                "scene_summary": req_input.get("scene_summary", ""),
                "location": req_input.get("location", ""),
                "participants": list(req_input.get("participants", ())),
                "scene_start_block_id": scope.summary.get("scene_start_block_id", ""),
                "scene_end_block_id": scope.summary.get("scene_end_block_id", ""),
            }
            known_characters = [str(n) for n in req_input.get("known_characters", ())]
            known_locations = [str(n) for n in req_input.get("known_locations", ())]
            raw_known_aliases = req_input.get("known_entity_aliases")
            known_entity_aliases = (
                {
                    str(k): [str(a) for a in v]
                    for k, v in raw_known_aliases.items()
                    if isinstance(v, (list, tuple)) and v
                }
                if isinstance(raw_known_aliases, Mapping)
                else None
            )
            raw_glossary = req_input.get("glossary_terms_ru")
            glossary_terms_ru = (
                {str(k): str(v) for k, v in raw_glossary.items()}
                if isinstance(raw_glossary, Mapping) and raw_glossary
                else None
            )
            return [
                _build_candidate_scope_spec(
                    section_id=section_id,
                    section_order=section_order,
                    section_title=section_title,
                    scope_key=f"{scope.scope_key}.d{depth + 1}.{idx}",
                    scene_meta=scene_meta,
                    blocks_chunk=half,
                    depth=depth + 1,
                    revision=revision,
                    route_id=route_id,
                    known_characters=known_characters,
                    known_locations=known_locations,
                    known_entity_aliases=known_entity_aliases,
                    glossary_terms_ru=glossary_terms_ru,
                )
                for idx, half in enumerate((blocks[:mid], blocks[mid:]))
            ]

        if scope.stage is IllustrationScopeStage.SELECTION:
            cards = [SelectedMomentCard.from_dict(c) for c in scope.summary.get("cards", ())]
            if len(cards) <= 2:
                return None
            mid = len(cards) // 2
            halves = [cards[:mid], cards[mid:]]
            # Both halves have size >= 2 if len(cards) >= 4; if len(cards) == 3, [2] and [1] -> carry singleton!
            round_num = int(scope.summary.get("round", 1))
            group_idx = int(scope.summary.get("group_idx", 0))
            max_per = int(
                scope.summary.get("effective_max_per_chapter", revision.settings.max_per_chapter)
            )
            prior_singletons = [
                SelectedMomentCard.from_dict(c)
                for c in scope.summary.get("carried_singletons", ())
            ]
            prior_shortlist = [
                SelectedMomentCard.from_dict(c)
                for c in scope.summary.get("prior_shortlist", ())
            ]
            prior_zeros = [str(z) for z in scope.summary.get("prior_zero_reasons", ())]
            eval_halves = [h for h in halves if len(h) >= 2]
            extra_singletons = [h[0] for h in halves if len(h) == 1]
            if not eval_halves:
                return None
            return [
                _build_selection_scope_spec(
                    section_id=section_id,
                    section_order=section_order,
                    section_title=section_title,
                    scope_key=f"{scope.scope_key}.d{depth + 1}.{idx}",
                    round_num=round_num,
                    group_idx=group_idx,
                    cards=half,
                    group_cap=min(max_per, len(half) - 1),
                    depth=depth + 1,
                    revision=revision,
                    route_id=route_id,
                    chapter_cap=max_per,
                    carried_singletons=([*prior_singletons, *extra_singletons] if idx == 0 else ()),
                    prior_shortlist=(prior_shortlist if idx == 0 else ()),
                    prior_zero_reasons=(prior_zeros if idx == 0 else ()),
                )
                for idx, half in enumerate(eval_halves)
            ]
        return None

    def _abort_oversized_or_split(
        self,
        *,
        book_id: str,
        generation_id: str,
        task_id: str,
        scope: IllustrationScope,
        attempt_id: str,
        revision: IllustrationSettingsRevision,
        is_composite: bool,
        depth: int,
        policy: IllustrationPolicy,
        reason_code: str,
    ) -> bool:
        notify = False
        with self._uow.transaction() as tx:
            tx.illustrations.release_reserved_call(attempt_id, reason_code=reason_code)
            if is_composite:
                if depth >= policy.max_split_depth:
                    tx.illustrations.fail_scope(
                        scope.id,
                        attempt_id=None,
                        reason_code=IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
                    )
                    self._mark_task_failed(
                        tx, task_id, IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value
                    )
                    self._refresh_generation_state(
                        tx,
                        book_id,
                        generation_id,
                        IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
                    )
                    return False
                children = self._split_children_for_scope(scope, revision, depth)
                if children:
                    try:
                        created = tx.illustrations.split_scope(
                            scope.id,
                            attempt_id=None,
                            children=children,
                            max_scopes=policy.max_scopes_per_generation,
                        )
                    except IllustrationLimitError as limit_err:
                        tx.illustrations.fail_scope(
                            scope.id, attempt_id=None, reason_code=limit_err.reason_code
                        )
                        self._mark_task_failed(tx, task_id, limit_err.reason_code)
                        self._refresh_generation_state(
                            tx, book_id, generation_id, limit_err.reason_code
                        )
                        return False
                    for child in created:
                        dispatch_scope_task(
                            tx,
                            child,
                            generation_id=generation_id,
                            task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                            id_factory=self._id_factory,
                            book_id=book_id,
                        )
                        notify = True
                    self._mark_task_succeeded(tx, task_id)
                    return notify

            tx.illustrations.fail_scope(scope.id, attempt_id=None, reason_code=reason_code)
            self._mark_task_failed(tx, task_id, reason_code)
            self._refresh_generation_state(tx, book_id, generation_id, reason_code)
            return False

    def _handle_post_send_split(
        self,
        *,
        book_id: str,
        generation_id: str,
        task_id: str,
        scope: IllustrationScope,
        attempt_id: str,
        revision: IllustrationSettingsRevision,
        depth: int,
        policy: IllustrationPolicy,
        fail_reason: str,
        detail: str | None = None,
    ) -> bool:
        notify = False
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
                    attempt_id,
                    state=IllustrationAttemptState.STALE,
                    outcome={"reason_code": stale_reason},
                )
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
                self._supersede_task(tx, task_id)
                return False

            if depth >= policy.max_split_depth:
                fail_outcome = (
                    {
                        "reason_code": IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
                        "detail": compact_safe_detail(detail, max_bytes=256),
                    }
                    if detail and detail.strip()
                    else None
                )
                try:
                    tx.illustrations.fail_scope(
                        scope.id,
                        attempt_id=attempt_id,
                        reason_code=IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
                        outcome=fail_outcome,
                    )
                except IllustrationOwnershipError:
                    tx.illustrations.record_attempt_outcome(
                        attempt_id,
                        state=IllustrationAttemptState.STALE,
                        outcome={"reason_code": IllustrationReasonCode.STALE_ATTEMPT.value},
                    )
                    return False
                self._mark_task_failed(
                    tx, task_id, IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value
                )
                self._refresh_generation_state(
                    tx,
                    book_id,
                    generation_id,
                    IllustrationReasonCode.SPLIT_DEPTH_EXHAUSTED.value,
                )
                return False
            children = self._split_children_for_scope(scope, revision, depth)
            if not children:
                fail_outcome = (
                    {
                        "reason_code": fail_reason,
                        "detail": compact_safe_detail(detail, max_bytes=256),
                    }
                    if detail and detail.strip()
                    else None
                )
                try:
                    tx.illustrations.fail_scope(
                        scope.id,
                        attempt_id=attempt_id,
                        reason_code=fail_reason,
                        outcome=fail_outcome,
                    )
                except IllustrationOwnershipError:
                    tx.illustrations.record_attempt_outcome(
                        attempt_id,
                        state=IllustrationAttemptState.STALE,
                        outcome={"reason_code": IllustrationReasonCode.STALE_ATTEMPT.value},
                    )
                    return False
                self._mark_task_failed(tx, task_id, fail_reason)
                self._refresh_generation_state(tx, book_id, generation_id, fail_reason)
                return False
            try:
                created = tx.illustrations.split_scope(
                    scope.id,
                    attempt_id=attempt_id,
                    children=children,
                    max_scopes=policy.max_scopes_per_generation,
                )
            except IllustrationOwnershipError:
                tx.illustrations.record_attempt_outcome(
                    attempt_id,
                    state=IllustrationAttemptState.STALE,
                    outcome={"reason_code": IllustrationReasonCode.STALE_ATTEMPT.value},
                )
                return False
            except IllustrationLimitError as limit_err:
                try:
                    tx.illustrations.fail_scope(
                        scope.id, attempt_id=attempt_id, reason_code=limit_err.reason_code
                    )
                except IllustrationOwnershipError:
                    return False
                self._mark_task_failed(tx, task_id, limit_err.reason_code)
                self._refresh_generation_state(tx, book_id, generation_id, limit_err.reason_code)
                return False
            for child in created:
                dispatch_scope_task(
                    tx,
                    child,
                    generation_id=generation_id,
                    task_type=TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
                    id_factory=self._id_factory,
                    book_id=book_id,
                )
                notify = True
            self._mark_task_succeeded(tx, task_id)
            return notify

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

    def _load_section_blocks(
        self,
        tx: UnitOfWork,
        book_id: str,
        section_id: str,
        snapshot: IllustrationSourceSnapshot,
    ) -> dict[str, tuple[int, int, DocumentBlock]]:
        document = tx.documents.get(book_id)
        if document is None:
            return {}
        order_by_sec = {e.section_id: e.order for e in snapshot.entries}
        sec = next((s for s in document.ordered_sections if s.id == section_id), None)
        if sec is None:
            return {}
        sec_order = order_by_sec.get(sec.id, sec.order)
        return {blk.id: (sec_order, idx, blk) for idx, blk in enumerate(textual_blocks(sec))}

    @staticmethod
    def _find_generation(tx: UnitOfWork, generation_id: str) -> IllustrationGeneration | None:
        for gen in tx.illustrations.list_active_generations():
            if gen.id == generation_id:
                return gen
        return None

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
        IllustrationSelection._supersede_task(tx, task_id)

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
    "IllustrationSelection",
    "SelectedAnchor",
    "SelectedMomentCard",
    "SelectedMoments",
    "deduplicate_candidates",
    "effective_chapter_illustration_cap",
    "enforce_intra_chapter_diversity",
    "moments_conflict_diversity",
]
