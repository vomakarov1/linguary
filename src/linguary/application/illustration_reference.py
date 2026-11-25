"""Bounded reusable book-wide illustration registry builder (`ALG-02` + `ALG-04`)."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from linguary.application.context_builder import source_text_for_model, textual_blocks
from linguary.application.illustration_context import (
    IllustrationContext,
    closed_traits_for_kind,
    compact_safe_detail,
    is_context_limit_error,
    is_incompatible_route_error,
    is_transient_error,
    is_truncated_error,
    normalize_name_token,
    pack_items_by_wire_budget,
    resolve_quote_anchor,
)
from linguary.application.illustration_dto import (
    IllustrationAliasResponse,
    IllustrationDesignResponse,
    IllustrationExtractResponse,
)
from linguary.application.illustration_source import (
    IllustrationSource,
    IllustrationSourceSnapshot,
    SectionManifestEntry,
)
from linguary.application.illustration_validation import has_cyrillic
from linguary.application.illustrations import (
    dispatch_scope_task,
    finalize_generation_state_in_tx,
    revoke_active_in_transaction,
)
from linguary.application.llm_rejections import finalize_response, record_rejection
from linguary.application.ports.illustrations import IllustrationRuntimePort, RequestWireSizer
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
    glossary_key,
)
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import canonical_sha
from linguary.domain.illustrations import (
    ILLUSTRATION_STYLE_MONOCHROME_INK,
    TIMELESS_POSITION_KEY,
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
    RegistryItem,
    format_position_key,
)
from linguary.domain.llm_log import LLMCallStage

_HONORIFICS = frozenset(
    {
        "mr",
        "mrs",
        "miss",
        "ms",
        "dr",
        "prof",
        "professor",
        "sir",
        "lady",
        "lord",
        "captain",
        "capt",
        "uncle",
        "aunt",
        "king",
        "queen",
        "prince",
        "princess",
        "father",
        "mother",
        "brother",
        "sister",
        "st",
        "saint",
        "master",
        "madam",
        "madame",
        "monsieur",
        "господин",
        "госпожа",
        "мистер",
        "миссис",
        "мисс",
        "доктор",
        "профессор",
        "сэр",
        "леди",
        "лорд",
        "капитан",
        "дядя",
        "тётя",
        "тетя",
    }
)
_BLOCKING_STOPWORDS = frozenset({"the", "of", "and", "in", "at", "on", "to", "a", "an", "de", "la", "le"})
_MAX_EXTRACT_UNITS_PER_PAGE = 64
_MONOCHROME_COLOR_EN_RE = re.compile(
    r"\b(red|crimson|scarlet|green|emerald|blue|azure|cobalt|yellow|amber|gold|golden|"
    r"orange|purple|violet|pink|magenta|teal|turquoise|beige|brown|chestnut|auburn|"
    r"maroon|olive|khaki|indigo|cyan|lavender|burgundy|sepia|bronze|copper|cream|ochre)\b",
    re.IGNORECASE,
)
_MONOCHROME_COLOR_RU_RE = re.compile(
    r"\b(красн\w*|зелён\w*|зелен\w*|син\w*|голуб\w*|жёлт\w*|желт\w*|оранжев\w*|"
    r"фиолетов\w*|розов\w*|бежев\w*|коричнев\w*|рыж\w*|изумруд\w*|багров\w*|"
    r"бирюзов\w*|золот\w*|пурпур\w*|бордов\w*|лилов\w*|бронзов\w*|медн\w*|"
    r"кремов\w*|охр(?:а|ы|е|у|ой|ист\w*|ов\w*))\b",
    re.IGNORECASE,
)


def _default_id() -> str:
    return str(uuid4())


def _strip_honorifics(normalized: str) -> str:
    tokens = [tok for tok in normalized.split() if tok not in _HONORIFICS]
    return " ".join(tokens)


def _blocking_keys_for_mention(
    mention: Mapping[str, Any],
    *,
    glossary_links: Mapping[str, str],
) -> set[str]:
    """Build deterministic blocking keys for one mention (same kind only)."""
    kind = str(mention["entity_kind"])
    norm = normalize_name_token(str(mention["name"]))
    keys: set[str] = set()
    if norm:
        keys.add(f"{kind}:exact:{norm}")
        stripped = _strip_honorifics(norm)
        if stripped:
            keys.add(f"{kind}:stripped:{stripped}")
        for tok in stripped.split():
            if len(tok) >= 3 and tok not in _BLOCKING_STOPWORDS:
                keys.add(f"{kind}:tok:{tok}")
        gkey = glossary_links.get(glossary_key(str(mention["name"])))
        if gkey:
            keys.add(f"{kind}:gloss:{gkey}")
    hint = mention.get("alias_of_hint")
    if isinstance(hint, str) and hint.strip():
        hint_norm = normalize_name_token(hint)
        if hint_norm:
            keys.add(f"{kind}:exact:{hint_norm}")
            hint_stripped = _strip_honorifics(hint_norm)
            if hint_stripped:
                keys.add(f"{kind}:stripped:{hint_stripped}")
            for tok in hint_stripped.split():
                if len(tok) >= 3 and tok not in _BLOCKING_STOPWORDS:
                    keys.add(f"{kind}:tok:{tok}")
            gkey_hint = glossary_links.get(glossary_key(hint))
            if gkey_hint:
                keys.add(f"{kind}:gloss:{gkey_hint}")
    return keys


def build_alias_candidate_pairs(
    mentions: Sequence[Mapping[str, Any]],
    *,
    glossary_links: Mapping[str, str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Deterministically generate candidate `(mention_a, mention_b)` pairs via blocking.

    Subquadratic for large books; pairs with the exact same name are still emitted as
    candidates (same name is NOT automatic identity).
    """
    links = glossary_links or {}
    buckets: dict[str, list[str]] = defaultdict(list)
    kind_by_key: dict[str, str] = {}
    for mention in sorted(mentions, key=lambda m: str(m["mention_key"])):
        mkey = str(mention["mention_key"])
        kind_by_key[mkey] = str(mention["entity_kind"])
        for bkey in sorted(_blocking_keys_for_mention(mention, glossary_links=links)):
            buckets[bkey].append(mkey)

    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        unique = sorted(set(members))
        if len(unique) < 2:
            continue
        for i in range(len(unique)):
            a = unique[i]
            for j in range(i + 1, len(unique)):
                b = unique[j]
                if kind_by_key[a] == kind_by_key[b]:
                    pairs.add((a, b))
    return tuple(sorted(pairs))


def resolve_verdict_graph_clusters(
    node_ids: Sequence[str],
    *,
    same_edges: Sequence[tuple[str, str]],
    diff_edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    """Order-independent deterministic entity clustering from `(same_edges, diff_edges)`.

    Guarantees:
    - No two endpoints of any `diff_edge` are ever merged into the same cluster.
    - Bridge nodes having `same_edges` to mutually different nodes stay separate.
    - Output is 100% invariant to input edge order.
    """
    nodes = sorted(set(node_ids))
    node_set = set(nodes)
    diff_set: set[tuple[str, str]] = set()
    for u, v in diff_edges:
        if u in node_set and v in node_set and u != v:
            diff_set.add((u, v) if u < v else (v, u))

    active_same: set[tuple[str, str]] = set()
    for u, v in same_edges:
        if u in node_set and v in node_set and u != v:
            edge = (u, v) if u < v else (v, u)
            if edge not in diff_set:
                active_same.add(edge)

    # Step 1 (symmetric): remove all same_edges incident to any direct bridge node
    # (a node connected via same_edges to both endpoints of some diff_edge).
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in active_same:
        adj[u].add(v)
        adj[v].add(u)

    bridge_nodes: set[str] = set()
    for x in nodes:
        neighbors = sorted(adj[x])
        for i in range(len(neighbors)):
            a = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                b = neighbors[j]
                pair = (a, b) if a < b else (b, a)
                if pair in diff_set:
                    bridge_nodes.add(x)
                    break
            if x in bridge_nodes:
                break

    if bridge_nodes:
        active_same = {
            (u, v) for (u, v) in active_same if u not in bridge_nodes and v not in bridge_nodes
        }

    # Step 2 (symmetric): compute connected components; if any component still contains
    # a diff_edge (e.g. a longer conflict chain A-B-D-C with A!=C), simultaneously drop
    # all same_edges inside that conflicting component so the partition stays order-independent.
    def _components(edges: set[tuple[str, str]]) -> list[list[str]]:
        graph: dict[str, set[str]] = {n: set() for n in nodes}
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        visited: set[str] = set()
        comps: list[list[str]] = []
        for n in nodes:
            if n in visited:
                continue
            stack = [n]
            comp: list[str] = []
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.append(curr)
                for nxt in sorted(graph[curr]):
                    if nxt not in visited:
                        stack.append(nxt)
            comps.append(sorted(comp))
        return comps

    comps = _components(active_same)
    conflicting_nodes: set[str] = set()
    for comp in comps:
        comp_set = set(comp)
        if any(u in comp_set and v in comp_set for u, v in diff_set):
            conflicting_nodes.update(comp_set)

    if conflicting_nodes:
        active_same = {
            (u, v)
            for (u, v) in active_same
            if u not in conflicting_nodes and v not in conflicting_nodes
        }
        comps = _components(active_same)

    return tuple(sorted(tuple(comp) for comp in comps))


def _section_extract_units(
    entry: SectionManifestEntry,
    section: DocumentSection,
) -> list[dict[str, Any]]:
    """Build ordered extract source units (scenes or block chunks) for one illustrable section."""
    blocks = textual_blocks(section)
    block_by_id = {b.id: (idx, b) for idx, b in enumerate(blocks)}
    units: list[dict[str, Any]] = []
    if entry.scenes:
        ordered_ids = [b.id for b in blocks]
        pos_by_id = {bid: idx for idx, bid in enumerate(ordered_ids)}
        for scene_idx, scene in enumerate(entry.scenes):
            s_start = pos_by_id.get(scene.start_block_id, 0)
            s_end = pos_by_id.get(scene.end_block_id, len(blocks) - 1)
            if s_end < s_start:
                s_start, s_end = s_end, s_start
            scene_blocks = blocks[s_start : s_end + 1]
            if not scene_blocks:
                continue
            unit_id = f"{entry.section_id}:scene:{scene_idx}"
            units.append(
                {
                    "source_unit_id": unit_id,
                    "section_id": entry.section_id,
                    "section_order": entry.order,
                    "section_title": entry.title,
                    "scene_id": scene.id,
                    "scene_summary": scene.summary,
                    "location": scene.location,
                    "participants": list(scene.participants),
                    "blocks": [
                        {
                            "block_id": b.id,
                            "block_order": block_by_id[b.id][0],
                            "text": source_text_for_model(b),
                        }
                        for b in scene_blocks
                    ],
                }
            )
    if not units and blocks:
        for idx, b in enumerate(blocks):
            units.append(
                {
                    "source_unit_id": f"{entry.section_id}:block:{b.id}",
                    "section_id": entry.section_id,
                    "section_order": entry.order,
                    "section_title": entry.title,
                    "scene_id": None,
                    "scene_summary": "",
                    "location": "",
                    "participants": [],
                    "blocks": [
                        {
                            "block_id": b.id,
                            "block_order": idx,
                            "text": source_text_for_model(b),
                        }
                    ],
                }
            )
    return units


def _split_extract_unit_blocks(unit: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Split a single multi-block scene unit into two half-units preserving source_unit_id lineage."""
    raw = thaw(unit)
    blocks = list(raw.get("blocks", ()))
    if len(blocks) <= 1:
        return None
    mid = len(blocks) // 2
    base_id = str(raw["source_unit_id"])
    left = {**raw, "source_unit_id": f"{base_id}#L", "blocks": blocks[:mid]}
    right = {**raw, "source_unit_id": f"{base_id}#R", "blocks": blocks[mid:]}
    return [left, right]


def _build_extract_scope_spec(
    *,
    scope_key: str,
    units: Sequence[Mapping[str, Any]],
    depth: int,
    revision: IllustrationSettingsRevision,
    route_id: str,
    book_id: str,
    conditions_id: str | None,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    expected_unit_ids = [str(u["source_unit_id"]) for u in units]
    ordered_block_ids: list[str] = []
    for u in units:
        for blk in u.get("blocks", ()):
            ordered_block_ids.append(str(blk["block_id"]))
    request_input = {
        "units": [thaw(u) for u in units],
    }
    input_sha256 = canonical_sha(
        {
            "stage": IllustrationScopeStage.EXTRACT.value,
            "request_input": request_input,
            "model_reference": revision.settings.model_reference,
            "route_id": route_id,
            "policy": revision.settings.policy.to_payload(),
        }
    )
    is_composite = len(units) > 1 or any(len(u.get("blocks", ())) > 1 for u in units)
    section_ids = sorted({str(u["section_id"]) for u in units})
    single_section = section_ids[0] if len(section_ids) == 1 else None
    summary = {
        "depth": depth,
        "is_composite": is_composite,
        "expected_source_unit_ids": expected_unit_ids,
        "ordered_block_ids": ordered_block_ids,
        "request_input": request_input,
        "book_id": book_id,
        "conditions_id": conditions_id,
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.EXTRACT,
            scope_key=scope_key,
            input_sha256=input_sha256,
            section_id=single_section,
            summary={**summary, "reused_from_scope_id": reusable.id},
            state=IllustrationScopeState.ACCEPTED,
            payload=reusable.payload,
            output_sha256=reusable.output_sha256,
        )
    return IllustrationScopeSpec(
        stage=IllustrationScopeStage.EXTRACT,
        scope_key=scope_key,
        input_sha256=input_sha256,
        section_id=single_section,
        summary=summary,
        state=IllustrationScopeState.PENDING,
    )


def _build_alias_scope_spec(
    *,
    scope_key: str,
    pair_entries: Sequence[Mapping[str, Any]],
    depth: int,
    revision: IllustrationSettingsRevision,
    route_id: str,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    expected_pair_ids = [str(p["pair_id"]) for p in pair_entries]
    valid_fact_ids = sorted(
        {
            str(f["fact_id"])
            for p in pair_entries
            for side in ("left", "right")
            for f in p[side].get("facts", ())
        }
    )
    request_input = {"pairs": [thaw(p) for p in pair_entries]}
    input_sha256 = canonical_sha(
        {
            "stage": IllustrationScopeStage.ALIAS.value,
            "request_input": request_input,
            "model_reference": revision.settings.model_reference,
            "route_id": route_id,
            "policy": revision.settings.policy.to_payload(),
        }
    )
    summary = {
        "depth": depth,
        "is_composite": len(pair_entries) > 1,
        "expected_pair_ids": expected_pair_ids,
        "valid_fact_ids": valid_fact_ids,
        "request_input": request_input,
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.ALIAS,
            scope_key=scope_key,
            input_sha256=input_sha256,
            summary={**summary, "reused_from_scope_id": reusable.id},
            state=IllustrationScopeState.ACCEPTED,
            payload=reusable.payload,
            output_sha256=reusable.output_sha256,
        )
    return IllustrationScopeSpec(
        stage=IllustrationScopeStage.ALIAS,
        scope_key=scope_key,
        input_sha256=input_sha256,
        summary=summary,
        state=IllustrationScopeState.PENDING,
    )


def _build_bounded_design_input(
    *,
    entity: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    sizer: RequestWireSizer,
    book_id: str,
    conditions_id: str | None,
    policy: IllustrationPolicy,
    style: str = ILLUSTRATION_STYLE_MONOCHROME_INK,
) -> dict[str, Any]:
    """Deterministically bound entity facts for DESIGN to `policy.initial_target_bytes`.

    Prioritizes deduplicated `permanent` facts sorted by `(position_key, fact_id)`, then
    `state_change`, then `episodic`, recording `omitted_episodic_count` and `omitted_fact_count`.
    """
    entity_kind = str(entity["entity_kind"])
    allowed_traits = list(closed_traits_for_kind(entity_kind))

    # Deduplicate by (temporal_scope, description_ru, description_en) keeping earliest position_key.
    seen_desc: set[tuple[str, str, str]] = set()
    deduped: list[Mapping[str, Any]] = []
    for f in sorted(facts, key=lambda x: (str(x["position_key"]), str(x["fact_id"]))):
        sig = (
            str(f["temporal_scope"]),
            str(f["description_ru"]).strip().casefold(),
            str(f["description_en"]).strip().casefold(),
        )
        if sig in seen_desc:
            continue
        seen_desc.add(sig)
        deduped.append(f)

    priority = {"permanent": 0, "state_change": 1, "episodic": 2}
    ordered = sorted(
        deduped,
        key=lambda x: (
            priority.get(str(x["temporal_scope"]), 3),
            str(x["position_key"]),
            str(x["fact_id"]),
        ),
    )

    def _compact_fact(f: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "fact_id": str(f["fact_id"]),
            "temporal_scope": str(f["temporal_scope"]),
            "position_key": str(f["position_key"]),
            "description_ru": str(f["description_ru"]),
            "description_en": str(f["description_en"]),
        }

    included: list[dict[str, Any]] = []
    omitted_episodic = 0
    omitted_total = 0

    def _make_payload(
        cur_facts: Sequence[Mapping[str, Any]],
        omit_ep: int,
        omit_all: int,
    ) -> dict[str, Any]:
        return {
            "entity_id": str(entity["entity_id"]),
            "canonical_name": str(entity["canonical_name"]),
            "entity_kind": entity_kind,
            "style": style,
            "allowed_traits": allowed_traits,
            "facts": list(cur_facts),
            "omitted_episodic_count": omit_ep,
            "omitted_fact_count": omit_all,
        }

    target_bytes = policy.initial_target_bytes
    for idx, fact in enumerate(ordered):
        candidate_fact = _compact_fact(fact)
        remaining = ordered[idx + 1 :]
        rem_ep = sum(1 for r in remaining if r["temporal_scope"] != "permanent")
        payload = _make_payload([*included, candidate_fact], omitted_episodic + rem_ep, omitted_total + len(remaining))
        req = IllustrationContext.build_for_kind(
            kind=LLMResponseKind.ILLUSTRATION_DESIGN,
            stage=IllustrationScopeStage.DESIGN,
            input_payload=payload,
            book_id=book_id,
            conditions_id=conditions_id,
            policy=policy,
        )
        if sizer.measure(req).size_bytes <= target_bytes:
            included.append(candidate_fact)
        else:
            omitted_total += 1
            if fact["temporal_scope"] != "permanent":
                omitted_episodic += 1

    return _make_payload(included, omitted_episodic, omitted_total)


def _build_design_scope_spec(
    *,
    entity: Mapping[str, Any],
    request_input: Mapping[str, Any],
    revision: IllustrationSettingsRevision,
    route_id: str,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    entity_id = str(entity["entity_id"])
    entity_kind = str(entity["entity_kind"])
    input_sha256 = canonical_sha(
        {
            "stage": IllustrationScopeStage.DESIGN.value,
            "request_input": thaw(request_input),
            "model_reference": revision.settings.model_reference,
            "route_id": route_id,
            "style": revision.settings.style,
            "policy": revision.settings.policy.to_payload(),
        }
    )
    summary = {
        "depth": 0,
        "is_composite": False,
        "entity_id": entity_id,
        "entity_kind": entity_kind,
        "style": revision.settings.style,
        "allowed_traits": list(closed_traits_for_kind(entity_kind)),
        "request_input": thaw(request_input),
    }
    if reusable is not None and reusable.payload is not None and reusable.output_sha256 is not None:
        return IllustrationScopeSpec(
            stage=IllustrationScopeStage.DESIGN,
            scope_key=f"design:{entity_id}",
            input_sha256=input_sha256,
            summary={**summary, "reused_from_scope_id": reusable.id},
            state=IllustrationScopeState.ACCEPTED,
            payload=reusable.payload,
            output_sha256=reusable.output_sha256,
        )
    return IllustrationScopeSpec(
        stage=IllustrationScopeStage.DESIGN,
        scope_key=f"design:{entity_id}",
        input_sha256=input_sha256,
        summary=summary,
        state=IllustrationScopeState.PENDING,
    )


@dataclass(frozen=True, slots=True)
class _ValidationResult:
    ok: bool
    payload: Mapping[str, Any] | None = None
    reason_code: str = IllustrationReasonCode.MODEL_OUTPUT_INVALID.value
    repair_reason: str = ""
    repair_details: tuple[str, ...] = ()
    request_split: bool = False


class IllustrationReference:
    """Plans and executes the bounded reference registry stages (`extract`, `alias`, `design`, `seal`)."""

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
        plan_chapter_scopes: Callable[..., bool] | None = None,
    ) -> None:
        self._uow = unit_of_work
        self._source = source
        self._runtime = runtime
        self._id_factory = id_factory or _default_id
        self._notify_work_available = notify_work_available
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=3)
        self._sleep = sleep_fn or asyncio.sleep
        self._plan_chapter_scopes = plan_chapter_scopes

    def plan_scopes(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        snapshot: IllustrationSourceSnapshot,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> bool:
        """Plan initial bounded `EXTRACT` scopes (or advance already-planned reference stages)."""
        factory = id_factory or self._id_factory
        if generation.registry_sha256 is not None:
            return False
        revision = tx.illustrations.get_settings(generation.book_id, generation.settings_id)
        if revision is None or not revision.settings.enabled:
            return False
        existing_extract = tx.illustrations.list_scopes(
            generation.id, stage=IllustrationScopeStage.EXTRACT
        )
        if existing_extract:
            return self._advance_reference_pipeline(
                tx,
                generation,
                revision,
                snapshot,
                id_factory=factory,
            )

        document = tx.documents.get(generation.book_id)
        if document is None:
            return False
        section_by_id = {sec.id: sec for sec in document.ordered_sections}

        units: list[dict[str, Any]] = []
        for entry in snapshot.entries:
            if entry.skipped:
                continue
            sec = section_by_id.get(entry.section_id)
            if sec is None:
                continue
            units.extend(_section_extract_units(entry, sec))

        if not units:
            # No textual units -> seal empty registry immediately and plan chapter scopes.
            empty_sha = canonical_sha({"kind": "illustration_registry_v1", "items": []})
            sealed_gen = tx.illustrations.seal_registry(
                generation.book_id, generation.id, registry_sha256=empty_sha
            )
            if self._plan_chapter_scopes is not None:
                return bool(
                    self._plan_chapter_scopes(tx, sealed_gen, snapshot, id_factory=factory)
                )
            return False

        sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_REFERENCE)
        policy = revision.settings.policy
        route_id = self._runtime.route_id()

        def _build_page_req(page_units: Sequence[dict[str, Any]]) -> LLMRequest[Any]:
            return IllustrationContext.build_for_kind(
                kind=LLMResponseKind.ILLUSTRATION_EXTRACT,
                stage=IllustrationScopeStage.EXTRACT,
                input_payload={"units": list(page_units)},
                book_id=snapshot.book_id,
                conditions_id=snapshot.conditions_id,
                policy=policy,
            )

        pages = pack_items_by_wire_budget(
            units,
            build_request=_build_page_req,
            sizer=sizer,
            target_bytes=policy.initial_target_bytes,
            max_items=_MAX_EXTRACT_UNITS_PER_PAGE,
        )
        specs: list[IllustrationScopeSpec] = []
        for page_idx, page_units in enumerate(pages):
            draft = _build_extract_scope_spec(
                scope_key=f"extract:p{page_idx:04d}",
                units=page_units,
                depth=0,
                revision=revision,
                route_id=route_id,
                book_id=snapshot.book_id,
                conditions_id=snapshot.conditions_id,
            )
            reusable = tx.illustrations.find_reusable_scope(
                generation.book_id,
                stage=IllustrationScopeStage.EXTRACT,
                input_sha256=draft.input_sha256,
            )
            if reusable is not None:
                draft = _build_extract_scope_spec(
                    scope_key=f"extract:p{page_idx:04d}",
                    units=page_units,
                    depth=0,
                    revision=revision,
                    route_id=route_id,
                    book_id=snapshot.book_id,
                    conditions_id=snapshot.conditions_id,
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
                    task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                    id_factory=factory,
                    book_id=generation.book_id,
                )
                dispatched = True

        if not dispatched:
            # All extract scopes were reused -> advance immediately to alias/design/seal!
            if self._advance_reference_pipeline(
                tx, generation, revision, snapshot, id_factory=factory
            ):
                dispatched = True
        return dispatched

    async def run_task(self, task: ProcessingTask) -> None:
        """Execute one `PREPARE_ILLUSTRATION_REFERENCE` scope task using 3-tx `ALG-04`."""
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
            # ── TX1: verify preconditions, claim/reserve attempt ──
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

                # Check content-addressed reuse before reserving any call!
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
                        if self._advance_reference_pipeline(
                            tx, gen, revision, snapshot, id_factory=self._id_factory
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
                        scope.id,
                        attempt_id=None,
                        reason_code=limit_err.reason_code,
                    )
                    self._mark_task_failed(tx, task.id, limit_err.reason_code)
                    self._refresh_generation_state(tx, book_id, gen.id, limit_err.reason_code)
                    return

                if gen.state is IllustrationGenerationState.QUEUED:
                    tx.illustrations.update_generation_state(
                        book_id, gen.id, state=IllustrationGenerationState.RUNNING, reason_code=None
                    )

                block_lookup = self._load_block_lookup(tx, book_id, snapshot)
                gen_number = gen.generation_number
                assigned_model = revision.settings.model_reference

            first_call = False
            activation_calls_used += 1

            # ── Outside TX: build request, measure exact wire bytes, validate route ──
            sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_REFERENCE)
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
                client = self._runtime.client_for(revision, LLMCallStage.ILLUSTRATION_REFERENCE)
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

            # ── TX2: pre-send gate ──
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

            # ── Outside TX: physical LLM call ──
            response: LLMResponse[Any] | None = None
            validation: _ValidationResult | None = None
            call_error: Exception | None = None
            try:
                response = await client.generate(request)
                if response.truncated:
                    raise LLMResponseFormatError(
                        "ответ модели усечён по лимиту токенов", truncated=True
                    )
                validation = self._validate_response(scope, response.payload, block_lookup)
            except Exception as err:
                call_error = err

            # Decide outcome before TX3
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
                    self._finalize_scope_failure(
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
                    self._finalize_scope_failure(
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
                    self._finalize_scope_failure(
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
                    self._finalize_scope_failure(
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
                self._finalize_scope_failure(
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
                self._finalize_scope_failure(
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
                non_empty_details = [d for d in validation.repair_details if d and d.strip()]
                detail_text = (
                    f"{validation.repair_reason}: {'; '.join(non_empty_details)}"
                    if non_empty_details
                    else validation.repair_reason
                )
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
                self._finalize_scope_failure(
                    book_id=book_id,
                    generation_id=generation_id,
                    task_id=task.id,
                    scope_id=scope.id,
                    attempt_id=attempt.id,
                    reason_code=validation.reason_code,
                    detail=detail_text,
                )
                return

            # ── TX3: conditional acceptance + pipeline advancement ──
            assert validation.payload is not None
            output_sha256 = canonical_sha(validation.payload)
            with self._uow.transaction() as tx:
                head = tx.illustrations.get_head(book_id)
                gen = tx.illustrations.get_generation(book_id, generation_id)
                cur_revision = tx.illustrations.get_settings(book_id, revision.id)
                stale_reason: str | None = None
                if gen is None or gen.state is IllustrationGenerationState.REVOKED:
                    stale_reason = IllustrationReasonCode.STALE_ATTEMPT.value
                elif head.active_generation_id != gen.id or head.settings_id != revision.id:
                    stale_reason = IllustrationReasonCode.SETTINGS_CHANGED.value
                elif cur_revision is None or not cur_revision.settings.enabled:
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
                                scope.id,
                                attempt_id=None,
                                reason_code=stale_reason,
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
                if self._advance_reference_pipeline(
                    tx, gen, revision, cur_snap, id_factory=self._id_factory
                ):
                    notify_after = True
                else:
                    finalize_generation_state_in_tx(tx, book_id, gen.id)
                self._mark_task_succeeded(tx, task.id)
                break

        if notify_after and self._notify_work_available is not None:
            self._notify_work_available()

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

    def _load_block_lookup(
        self,
        tx: UnitOfWork,
        book_id: str,
        snapshot: IllustrationSourceSnapshot,
    ) -> dict[str, tuple[int, int, DocumentBlock]]:
        document = tx.documents.get(book_id)
        if document is None:
            return {}
        order_by_sec = {e.section_id: e.order for e in snapshot.entries}
        lookup: dict[str, tuple[int, int, DocumentBlock]] = {}
        for sec in document.ordered_sections:
            sec_order = order_by_sec.get(sec.id, sec.order)
            for b_idx, blk in enumerate(textual_blocks(sec)):
                lookup[blk.id] = (sec_order, b_idx, blk)
        return lookup

    def _validate_response(
        self,
        scope: IllustrationScope,
        payload: Any,
        block_lookup: Mapping[str, tuple[int, int, DocumentBlock]],
    ) -> _ValidationResult:
        if scope.stage is IllustrationScopeStage.EXTRACT:
            if not isinstance(payload, IllustrationExtractResponse):
                return _ValidationResult(ok=False, repair_reason="ожидался ответ извлечения")
            return self._validate_extract(scope, payload, block_lookup)
        if scope.stage is IllustrationScopeStage.ALIAS:
            if not isinstance(payload, IllustrationAliasResponse):
                return _ValidationResult(ok=False, repair_reason="ожидался ответ сверки имён")
            return self._validate_alias(scope, payload)
        if scope.stage is IllustrationScopeStage.DESIGN:
            if not isinstance(payload, IllustrationDesignResponse):
                return _ValidationResult(ok=False, repair_reason="ожидался ответ черт образа")
            return self._validate_design(scope, payload)
        return _ValidationResult(ok=False, repair_reason="неподдерживаемый этап справочника")

    def _validate_extract(
        self,
        scope: IllustrationScope,
        dto: IllustrationExtractResponse,
        block_lookup: Mapping[str, tuple[int, int, DocumentBlock]],
    ) -> _ValidationResult:
        if not dto.complete:
            return _ValidationResult(ok=False, request_split=True)

        expected_units = set(str(u) for u in scope.summary.get("expected_source_unit_ids", ()))
        covered_units = set(dto.covered_source_unit_ids)
        if covered_units != expected_units:
            missing = sorted(expected_units - covered_units)
            extra = sorted(covered_units - expected_units)
            return _ValidationResult(
                ok=False,
                repair_reason="неполный или лишний список covered_source_unit_ids при complete=true",
                repair_details=(
                    f"пропущены={missing[:4]}" if missing else "",
                    f"лишние={extra[:4]}" if extra else "",
                ),
            )

        allowed_blocks = set(str(b) for b in scope.summary.get("ordered_block_ids", ()))
        resolved_mentions: list[dict[str, Any]] = []
        mention_ids: set[str] = set()
        for m in dto.mentions:
            if m.mention_id in mention_ids:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"дублируется mention_id {m.mention_id}",
                )
            if m.quote_block_id not in allowed_blocks or m.quote_block_id not in block_lookup:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"упоминание {m.mention_id} ссылается на недопустимый блок {m.quote_block_id}",
                )
            sec_order, blk_order, blk = block_lookup[m.quote_block_id]
            res = resolve_quote_anchor(
                blk, m.quote, section_order=sec_order, block_order=blk_order
            )
            if res.status == "ambiguous":
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"цитата для {m.mention_id} повторяется в блоке {m.quote_block_id}; дай более длинную уникальную подстроку",
                )
            if res.status != "resolved" or res.anchor is None or res.position_key is None:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"цитата для {m.mention_id} не найдена в блоке {m.quote_block_id}",
                )
            mention_ids.add(m.mention_id)
            mention_key = f"{scope.scope_key}::{m.mention_id}"
            resolved_mentions.append(
                {
                    "mention_id": m.mention_id,
                    "mention_key": mention_key,
                    "name": m.name.strip(),
                    "entity_kind": m.entity_kind,
                    "alias_of_hint": (m.alias_of_hint.strip() or None) if m.alias_of_hint else None,
                    "block_id": res.anchor.block_id,
                    "char_start": res.anchor.start,
                    "char_end": res.anchor.end,
                    "quote": res.anchor.quote,
                    "position_key": res.position_key,
                }
            )

        resolved_facts: list[dict[str, Any]] = []
        fact_ids: set[str] = set()
        for f in dto.facts:
            if f.fact_id in fact_ids:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"дублируется fact_id {f.fact_id}",
                )
            if f.subject_mention_id not in mention_ids:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"факт {f.fact_id} ссылается на неизвестный subject_mention_id {f.subject_mention_id}",
                )
            if f.quote_block_id not in allowed_blocks or f.quote_block_id not in block_lookup:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"факт {f.fact_id} ссылается на недопустимый блок {f.quote_block_id}",
                )
            sec_order, blk_order, blk = block_lookup[f.quote_block_id]
            res = resolve_quote_anchor(
                blk, f.quote, section_order=sec_order, block_order=blk_order
            )
            if res.status == "ambiguous":
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"цитата факта {f.fact_id} повторяется в блоке {f.quote_block_id}; дай более длинную уникальную подстроку",
                )
            if res.status != "resolved" or res.anchor is None or res.position_key is None:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"цитата факта {f.fact_id} не найдена в блоке {f.quote_block_id}",
                )
            if has_cyrillic(f.description_en):
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"description_en факта {f.fact_id} должен быть написан строго на английском языке без кириллицы",
                )
            fact_ids.add(f.fact_id)
            fact_key = f"{scope.scope_key}::{f.fact_id}"
            is_back_applicable = (
                f.temporal_scope == "permanent"
                and f.back_applicable
                and not f.reveals_future_spoiler
            )
            resolved_facts.append(
                {
                    "fact_id": fact_key,
                    "local_fact_id": f.fact_id,
                    "subject_mention_key": f"{scope.scope_key}::{f.subject_mention_id}",
                    "temporal_scope": f.temporal_scope,
                    "back_applicable": f.back_applicable,
                    "reveals_future_spoiler": f.reveals_future_spoiler,
                    "is_back_applicable": is_back_applicable,
                    "description_ru": f.description_ru.strip(),
                    "description_en": f.description_en.strip(),
                    "block_id": res.anchor.block_id,
                    "char_start": res.anchor.start,
                    "char_end": res.anchor.end,
                    "quote": res.anchor.quote,
                    "position_key": res.position_key,
                }
            )

        return _ValidationResult(
            ok=True,
            payload={
                "schema_version": 1,
                "covered_source_unit_ids": sorted(covered_units),
                "mentions": resolved_mentions,
                "facts": resolved_facts,
            },
        )

    def _validate_alias(
        self,
        scope: IllustrationScope,
        dto: IllustrationAliasResponse,
    ) -> _ValidationResult:
        expected_ids = set(str(p) for p in scope.summary.get("expected_pair_ids", ()))
        valid_fact_ids = set(str(fid) for fid in scope.summary.get("valid_fact_ids", ()))
        seen: set[str] = set()
        decisions: list[dict[str, Any]] = []
        for dec in dto.decisions:
            if dec.pair_id not in expected_ids:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"неизвестный pair_id {dec.pair_id}",
                )
            if dec.pair_id in seen:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"дублируется pair_id {dec.pair_id}",
                )
            if any(fid not in valid_fact_ids for fid in dec.evidence_fact_ids):
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"неизвестный evidence_fact_id в паре {dec.pair_id}",
                )
            seen.add(dec.pair_id)
            decisions.append(
                {
                    "pair_id": dec.pair_id,
                    "verdict": dec.verdict,
                    "evidence_fact_ids": list(dec.evidence_fact_ids),
                }
            )
        if seen != expected_ids:
            return _ValidationResult(
                ok=False,
                repair_reason="отсутствуют решения для части запрошенных пар",
            )
        decisions.sort(key=lambda d: d["pair_id"])
        return _ValidationResult(
            ok=True,
            payload={"schema_version": 1, "decisions": decisions},
        )

    def _validate_design(
        self,
        scope: IllustrationScope,
        dto: IllustrationDesignResponse,
    ) -> _ValidationResult:
        expected_entity_id = str(scope.summary.get("entity_id", ""))
        entity_kind = str(scope.summary.get("entity_kind", ""))
        allowed_traits = set(closed_traits_for_kind(entity_kind))
        if dto.entity_id != expected_entity_id:
            return _ValidationResult(
                ok=False,
                repair_reason=f"entity_id должен быть {expected_entity_id}",
            )
        seen_traits: set[str] = set()
        traits: list[dict[str, Any]] = []
        style = str(scope.summary.get("style", ILLUSTRATION_STYLE_MONOCHROME_INK))
        for item in dto.traits:
            if item.trait not in allowed_traits:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"черта {item.trait} не входит в закрытый набор для типа {entity_kind}",
                )
            if item.trait in seen_traits:
                return _ValidationResult(
                    ok=False,
                    repair_reason=f"дублируется черта {item.trait}",
                )
            if has_cyrillic(item.description_en):
                return _ValidationResult(
                    ok=False,
                    repair_reason=(
                        f"description_en черты {item.trait} должен быть написан строго "
                        "на английском языке без кириллицы"
                    ),
                )
            if (
                style == ILLUSTRATION_STYLE_MONOCHROME_INK
                and item.trait == "palette_value"
                and (
                    _MONOCHROME_COLOR_EN_RE.search(item.description_en)
                    or _MONOCHROME_COLOR_EN_RE.search(item.description_ru)
                    or _MONOCHROME_COLOR_RU_RE.search(item.description_ru)
                    or _MONOCHROME_COLOR_RU_RE.search(item.description_en)
                )
            ):
                return _ValidationResult(
                    ok=False,
                    repair_reason=(
                        "для стиля monochrome_ink черта palette_value должна описывать "
                        "только тональность, контраст и штриховку без названий цветов"
                    ),
                )
            seen_traits.add(item.trait)
            traits.append(
                {
                    "trait": item.trait,
                    "description_ru": item.description_ru.strip(),
                    "description_en": item.description_en.strip(),
                    "assumption_ru": item.assumption_ru.strip(),
                }
            )
        traits.sort(key=lambda t: t["trait"])
        return _ValidationResult(
            ok=True,
            payload={
                "schema_version": 1,
                "entity_id": expected_entity_id,
                "entity_kind": entity_kind,
                "traits": traits,
            },
        )

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

        if scope.stage is IllustrationScopeStage.EXTRACT:
            units = list(req_input.get("units", ()))
            if len(units) > 1:
                mid = len(units) // 2
                halves = [units[:mid], units[mid:]]
            elif len(units) == 1:
                split_u = _split_extract_unit_blocks(units[0])
                if split_u is None:
                    return None
                halves = [[split_u[0]], [split_u[1]]]
            else:
                return None
            book_id = str(scope.summary.get("book_id", ""))
            conditions_id = scope.summary.get("conditions_id")
            return [
                _build_extract_scope_spec(
                    scope_key=f"{scope.scope_key}.d{depth + 1}.{idx}",
                    units=half,
                    depth=depth + 1,
                    revision=revision,
                    route_id=route_id,
                    book_id=book_id,
                    conditions_id=str(conditions_id) if conditions_id is not None else None,
                )
                for idx, half in enumerate(halves)
            ]

        if scope.stage is IllustrationScopeStage.ALIAS:
            pairs = list(req_input.get("pairs", ()))
            if len(pairs) <= 1:
                return None
            mid = len(pairs) // 2
            halves = [pairs[:mid], pairs[mid:]]
            return [
                _build_alias_scope_spec(
                    scope_key=f"{scope.scope_key}.d{depth + 1}.{idx}",
                    pair_entries=half,
                    depth=depth + 1,
                    revision=revision,
                    route_id=route_id,
                )
                for idx, half in enumerate(halves)
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
                            task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                            id_factory=self._id_factory,
                            book_id=book_id,
                        )
                        notify = True
                    self._mark_task_succeeded(tx, task_id)
                    return notify

            tx.illustrations.fail_scope(
                scope.id, attempt_id=None, reason_code=reason_code
            )
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
            cur_revision = tx.illustrations.get_settings(book_id, revision.id)
            stale_reason: str | None = None
            if gen is None or gen.state is IllustrationGenerationState.REVOKED:
                stale_reason = IllustrationReasonCode.STALE_ATTEMPT.value
            elif head.active_generation_id != gen.id or head.settings_id != revision.id:
                stale_reason = IllustrationReasonCode.SETTINGS_CHANGED.value
            elif cur_revision is None or not cur_revision.settings.enabled:
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
                self._refresh_generation_state(
                    tx, book_id, generation_id, limit_err.reason_code
                )
                return False
            for child in created:
                dispatch_scope_task(
                    tx,
                    child,
                    generation_id=generation_id,
                    task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                    id_factory=self._id_factory,
                    book_id=book_id,
                )
                notify = True
            self._mark_task_succeeded(tx, task_id)
            return notify

    def _finalize_scope_failure(
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

    def _advance_reference_pipeline(
        self,
        tx: UnitOfWork,
        generation: IllustrationGeneration,
        revision: IllustrationSettingsRevision,
        snapshot: IllustrationSourceSnapshot,
        *,
        id_factory: Callable[[], str],
    ) -> bool:
        """Advance `extract -> alias -> design -> seal` inside an open transaction."""
        if generation.registry_sha256 is not None:
            return False

        all_scopes = tx.illustrations.list_scopes(generation.id)
        extract_leaves = [
            s
            for s in all_scopes
            if s.stage is IllustrationScopeStage.EXTRACT and s.state is not IllustrationScopeState.SPLIT
        ]
        if not extract_leaves or any(s.state is not IllustrationScopeState.ACCEPTED for s in extract_leaves):
            return False

        # Collect all extracted mentions and facts deterministically.
        mentions: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        for s in sorted(extract_leaves, key=lambda sc: sc.scope_key):
            payload = s.payload or {}
            for m in payload.get("mentions", ()):
                mentions.append(thaw(m))
            for f in payload.get("facts", ()):
                facts.append(thaw(f))
        mentions.sort(key=lambda m: (str(m["position_key"]), str(m["mention_key"])))
        facts.sort(key=lambda f: (str(f["position_key"]), str(f["fact_id"])))

        alias_scopes = [s for s in all_scopes if s.stage is IllustrationScopeStage.ALIAS]
        alias_leaves = [s for s in alias_scopes if s.state is not IllustrationScopeState.SPLIT]

        glossary_links: dict[str, str] = {}
        for concept in tx.glossary.list_for_book(generation.book_id):
            glossary_links[glossary_key(concept.canonical_source_text)] = concept.id

        candidate_pairs = build_alias_candidate_pairs(mentions, glossary_links=glossary_links)
        if candidate_pairs and not alias_scopes:
            mention_by_key = {str(m["mention_key"]): m for m in mentions}
            facts_by_mention: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for f in facts:
                facts_by_mention[str(f["subject_mention_key"])].append(
                    {
                        "fact_id": str(f["fact_id"]),
                        "temporal_scope": str(f["temporal_scope"]),
                        "description_ru": str(f["description_ru"]),
                    }
                )
            pair_entries: list[dict[str, Any]] = []
            for a_key, b_key in candidate_pairs:
                ma = mention_by_key[a_key]
                mb = mention_by_key[b_key]
                pair_id = f"pair-{canonical_sha({'a': a_key, 'b': b_key})[:12]}"
                left_side: dict[str, Any] = {
                    "mention_key": a_key,
                    "name": ma["name"],
                    "quote": ma["quote"],
                    "facts": facts_by_mention[a_key][:4],
                }
                left_hint = str(ma.get("alias_of_hint") or "").strip()
                if left_hint:
                    left_side["alias_of_hint"] = left_hint
                left_gloss = glossary_links.get(glossary_key(str(ma["name"])))
                if left_gloss is not None:
                    left_side["glossary_concept_id"] = left_gloss

                right_side: dict[str, Any] = {
                    "mention_key": b_key,
                    "name": mb["name"],
                    "quote": mb["quote"],
                    "facts": facts_by_mention[b_key][:4],
                }
                right_hint = str(mb.get("alias_of_hint") or "").strip()
                if right_hint:
                    right_side["alias_of_hint"] = right_hint
                right_gloss = glossary_links.get(glossary_key(str(mb["name"])))
                if right_gloss is not None:
                    right_side["glossary_concept_id"] = right_gloss

                pair_entries.append(
                    {
                        "pair_id": pair_id,
                        "entity_kind": ma["entity_kind"],
                        "left": left_side,
                        "right": right_side,
                    }
                )

            policy = revision.settings.policy
            sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_REFERENCE)
            route_id = self._runtime.route_id()

            def _build_alias_req(batch: Sequence[dict[str, Any]]) -> LLMRequest[Any]:
                return IllustrationContext.build_for_kind(
                    kind=LLMResponseKind.ILLUSTRATION_ALIAS,
                    stage=IllustrationScopeStage.ALIAS,
                    input_payload={"pairs": list(batch)},
                    book_id=snapshot.book_id,
                    conditions_id=snapshot.conditions_id,
                    policy=policy,
                )

            # Respect both identity_candidate_page_size (<=16) and exact wire byte cap!
            chunked: list[tuple[dict[str, Any], ...]] = []
            page_size = policy.identity_candidate_page_size
            for start in range(0, len(pair_entries), page_size):
                sub = pair_entries[start : start + page_size]
                chunked.extend(
                    pack_items_by_wire_budget(
                        sub,
                        build_request=_build_alias_req,
                        sizer=sizer,
                        target_bytes=policy.initial_target_bytes,
                    )
                )

            specs: list[IllustrationScopeSpec] = []
            for idx, batch in enumerate(chunked):
                draft = _build_alias_scope_spec(
                    scope_key=f"alias:p{idx:04d}",
                    pair_entries=batch,
                    depth=0,
                    revision=revision,
                    route_id=route_id,
                )
                reusable = tx.illustrations.find_reusable_scope(
                    generation.book_id,
                    stage=IllustrationScopeStage.ALIAS,
                    input_sha256=draft.input_sha256,
                )
                if reusable is not None:
                    draft = _build_alias_scope_spec(
                        scope_key=f"alias:p{idx:04d}",
                        pair_entries=batch,
                        depth=0,
                        revision=revision,
                        route_id=route_id,
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
                        task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                        id_factory=id_factory,
                        book_id=generation.book_id,
                    )
                    dispatched = True
            if dispatched:
                return True
            # All alias scopes were reused -> reload and continue immediately.
            all_scopes = tx.illustrations.list_scopes(generation.id)
            alias_leaves = [
                s
                for s in all_scopes
                if s.stage is IllustrationScopeStage.ALIAS
                and s.state is not IllustrationScopeState.SPLIT
            ]

        if alias_leaves and any(s.state is not IllustrationScopeState.ACCEPTED for s in alias_leaves):
            return False

        # All alias decisions are accepted (or none needed) -> build complete verdict graph.
        pair_meta_by_id: dict[str, tuple[str, str]] = {}
        same_edges: list[tuple[str, str]] = []
        diff_edges: list[tuple[str, str]] = []
        alias_decision_items: list[RegistryItem] = []

        for s in sorted(alias_leaves, key=lambda sc: sc.scope_key):
            req_in = s.summary.get("request_input", {})
            if isinstance(req_in, Mapping):
                for p in req_in.get("pairs", ()):
                    pair_meta_by_id[str(p["pair_id"])] = (
                        str(p["left"]["mention_key"]),
                        str(p["right"]["mention_key"]),
                    )
            payload = s.payload or {}
            for dec in payload.get("decisions", ()):
                pid = str(dec["pair_id"])
                if pid not in pair_meta_by_id:
                    continue
                left_k, right_k = pair_meta_by_id[pid]
                verdict = str(dec["verdict"])
                if verdict == "same_entity":
                    same_edges.append((left_k, right_k))
                elif verdict == "different":
                    diff_edges.append((left_k, right_k))
                item_payload = {
                    "pair_id": pid,
                    "left_mention_key": left_k,
                    "right_mention_key": right_k,
                    "verdict": verdict,
                    "evidence_fact_ids": list(dec.get("evidence_fact_ids", ())),
                }
                alias_decision_items.append(
                    RegistryItem(
                        generation_id=generation.id,
                        kind=IllustrationRegistryKind.ALIAS_DECISION,
                        item_key=pid,
                        entity_id=pid,
                        position_key=TIMELESS_POSITION_KEY,
                        is_back_applicable=False,
                        payload_sha256=canonical_sha(item_payload),
                        payload=item_payload,
                    )
                )

        mention_by_key = {str(m["mention_key"]): m for m in mentions}
        clusters = resolve_verdict_graph_clusters(
            [str(m["mention_key"]) for m in mentions],
            same_edges=same_edges,
            diff_edges=diff_edges,
        )

        mention_to_entity: dict[str, str] = {}
        entities: list[dict[str, Any]] = []
        entity_items: list[RegistryItem] = []
        for cluster in clusters:
            members = [mention_by_key[k] for k in cluster]
            members.sort(key=lambda m: (str(m["position_key"]), str(m["mention_key"])))
            first = members[0]
            kind = str(first["entity_kind"])
            entity_id = f"ent-{canonical_sha({'kind': kind, 'members': list(cluster)})[:12]}"
            for k in cluster:
                mention_to_entity[k] = entity_id
            aliases = sorted({str(m["name"]) for m in members})
            entity_payload = {
                "entity_id": entity_id,
                "entity_kind": kind,
                "canonical_name": str(first["name"]),
                "aliases": aliases,
                "mention_count": len(members),
                "first_position_key": str(first["position_key"]),
            }
            entities.append(entity_payload)
            entity_items.append(
                RegistryItem(
                    generation_id=generation.id,
                    kind=IllustrationRegistryKind.ENTITY,
                    item_key=entity_id,
                    entity_id=entity_id,
                    position_key=TIMELESS_POSITION_KEY,
                    is_back_applicable=False,
                    payload_sha256=canonical_sha(entity_payload),
                    payload=entity_payload,
                )
            )

        fact_items: list[RegistryItem] = []
        facts_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in facts:
            eid = mention_to_entity.get(str(f["subject_mention_key"]))
            if eid is None:
                continue
            fact_payload = {
                "fact_id": str(f["fact_id"]),
                "entity_id": eid,
                "temporal_scope": str(f["temporal_scope"]),
                "back_applicable": bool(f["back_applicable"]),
                "reveals_future_spoiler": bool(f["reveals_future_spoiler"]),
                "is_back_applicable": bool(f["is_back_applicable"]),
                "description_ru": str(f["description_ru"]),
                "description_en": str(f["description_en"]),
                "block_id": str(f["block_id"]),
                "char_start": int(f["char_start"]),
                "char_end": int(f["char_end"]),
                "quote": str(f["quote"]),
                "position_key": str(f["position_key"]),
            }
            facts_by_entity[eid].append(fact_payload)
            fact_items.append(
                RegistryItem(
                    generation_id=generation.id,
                    kind=IllustrationRegistryKind.FACT,
                    item_key=str(f["fact_id"]),
                    entity_id=eid,
                    position_key=str(f["position_key"]),
                    is_back_applicable=bool(f["is_back_applicable"]),
                    payload_sha256=canonical_sha(fact_payload),
                    payload=fact_payload,
                )
            )

        if entity_items or fact_items or alias_decision_items:
            tx.illustrations.upsert_registry_items(
                generation.id, [*entity_items, *fact_items, *alias_decision_items]
            )

        # Stage 3: DESIGN for entities with facts or recurring mentions
        design_scopes = [s for s in all_scopes if s.stage is IllustrationScopeStage.DESIGN]
        design_entities = [
            e
            for e in entities
            if int(e["mention_count"]) >= 1 and closed_traits_for_kind(str(e["entity_kind"]))
        ]
        if design_entities and not design_scopes:
            policy = revision.settings.policy
            sizer = self._runtime.sizer_for(revision, LLMCallStage.ILLUSTRATION_REFERENCE)
            route_id = self._runtime.route_id()
            design_specs: list[IllustrationScopeSpec] = []
            for ent in sorted(design_entities, key=lambda e: str(e["entity_id"])):
                eid = str(ent["entity_id"])
                req_in = _build_bounded_design_input(
                    entity=ent,
                    facts=facts_by_entity.get(eid, ()),
                    sizer=sizer,
                    book_id=snapshot.book_id,
                    conditions_id=snapshot.conditions_id,
                    policy=policy,
                    style=revision.settings.style,
                )
                draft = _build_design_scope_spec(
                    entity=ent,
                    request_input=req_in,
                    revision=revision,
                    route_id=route_id,
                )
                reusable = tx.illustrations.find_reusable_scope(
                    generation.book_id,
                    stage=IllustrationScopeStage.DESIGN,
                    input_sha256=draft.input_sha256,
                )
                if reusable is not None:
                    draft = _build_design_scope_spec(
                        entity=ent,
                        request_input=req_in,
                        revision=revision,
                        route_id=route_id,
                        reusable=reusable,
                    )
                design_specs.append(draft)

            created = tx.illustrations.upsert_scopes(generation.id, design_specs)
            dispatched = False
            for scope in created:
                if scope.state is IllustrationScopeState.PENDING:
                    dispatch_scope_task(
                        tx,
                        scope,
                        generation_id=generation.id,
                        task_type=TaskType.PREPARE_ILLUSTRATION_REFERENCE,
                        id_factory=id_factory,
                        book_id=generation.book_id,
                    )
                    dispatched = True
            if dispatched:
                return True
            design_scopes = list(created)

        if design_scopes and any(s.state is not IllustrationScopeState.ACCEPTED for s in design_scopes):
            return False

        # Persist DESIGN items and seal registry deterministically without any whole-book LLM merge!
        design_items: list[RegistryItem] = []
        for s in sorted(design_scopes, key=lambda sc: sc.scope_key):
            payload = s.payload or {}
            eid = str(payload.get("entity_id", ""))
            for trait_obj in payload.get("traits", ()):
                trait_name = str(trait_obj["trait"])
                item_key = f"{eid}:{trait_name}"
                d_payload = {
                    "entity_id": eid,
                    "entity_kind": str(payload.get("entity_kind", "")),
                    "trait": trait_name,
                    "description_ru": str(trait_obj["description_ru"]),
                    "description_en": str(trait_obj["description_en"]),
                    "assumption_ru": str(trait_obj["assumption_ru"]),
                }
                design_items.append(
                    RegistryItem(
                        generation_id=generation.id,
                        kind=IllustrationRegistryKind.DESIGN,
                        item_key=item_key,
                        entity_id=eid,
                        position_key=TIMELESS_POSITION_KEY,
                        is_back_applicable=False,
                        payload_sha256=canonical_sha(d_payload),
                        payload=d_payload,
                    )
                )
        if design_items:
            tx.illustrations.upsert_registry_items(generation.id, design_items)

        # Seal registry from canonical ordered hashes of all persisted registry items.
        ordered_hashes: list[dict[str, str]] = []
        offset = 0
        while True:
            page, total = tx.illustrations.query_registry_items(
                generation.id, offset=offset, limit=200
            )
            for item in page:
                ordered_hashes.append(
                    {
                        "kind": item.kind.value,
                        "item_key": item.item_key,
                        "entity_id": item.entity_id,
                        "position_key": item.position_key,
                        "payload_sha256": item.payload_sha256,
                    }
                )
            offset += len(page)
            if offset >= total or not page:
                break

        registry_sha256 = canonical_sha(
            {"kind": "illustration_registry_v1", "items": ordered_hashes}
        )
        sealed_gen = tx.illustrations.seal_registry(
            generation.book_id, generation.id, registry_sha256=registry_sha256
        )

        if self._plan_chapter_scopes is not None:
            return bool(
                self._plan_chapter_scopes(tx, sealed_gen, snapshot, id_factory=id_factory)
            )
        return False

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
        IllustrationReference._supersede_task(tx, task_id)

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


_TRAIT_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "hair": re.compile(
        r"\b(?:hair|haired|blond|blonde|redhead|curly|braid|braids|beard|bearded|"
        r"mustache|moustache|bald|shaggy|bun)\b"
        r"|\b(?:волос|кудр|кос[аыуе]|бород|усы\b|усов\b|усами\b|лыс|пучок|лохмат)",
        re.IGNORECASE,
    ),
    "eyes": re.compile(
        r"\b(?:eye|eyes|gaze)\b|\b(?:глаз|взгляд)",
        re.IGNORECASE,
    ),
    "build": re.compile(
        r"\b(?:tall|short|slender|stout|thin|broad|beefy|giant|plump|skinny|stocky|bony)\b"
        r"|\b(?:рост|худощ|крепк|высок|крупн|грузн|великан|худой|худая|тощ|полн|костляв)",
        re.IGNORECASE,
    ),
    "age_appearance": re.compile(
        r"\b(?:young|old|elderly|child|boy|girl|teen|teenager|baby|infant|toddler|middle-aged)\b"
        r"|\b(?:молод|стар|ребён|ребен|девоч|мальч|младен|малыш|грудн|пожил)",
        re.IGNORECASE,
    ),
    "attire_default": re.compile(
        r"\b(?:coat|dress|cloak|hat|robe|robes|uniform|suit|boots|trousers|jacket|shirt|tie|scarf)\b"
        r"|\b(?:плать|плащ|пальт|шляп|костюм|куртк|галстук|шарф|рубаш|манти|сапог)",
        re.IGNORECASE,
    ),
    "distinctive_mark": re.compile(
        r"\b(?:scar|freckle|freckles|tattoo|limp|glasses|spectacles|lightning)\b"
        r"|\b(?:шрам|веснуш|очки|очков|очках|родинк|молни)",
        re.IGNORECASE,
    ),
    "palette_value": re.compile(
        r"\b(?:pale|dark|bright|shadow)\b|\b(?:светл|тёмн|темн|бледн)",
        re.IGNORECASE,
    ),
    "architecture": re.compile(
        r"\b(?:stone|wood|wooden|brick|arch|tower|hall)\b|\b(?:камен|деревян|башн|зал)",
        re.IGNORECASE,
    ),
    "atmosphere": re.compile(
        r"\b(?:dim|mist|fog|warm|gloomy)\b|\b(?:туман|мрачн|светл|уютн)",
        re.IGNORECASE,
    ),
}


def resolve_appearance_with_design_fallback(
    explicit_facts: Sequence[Mapping[str, Any]],
    design_traits: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Combine explicit text facts with design fallback traits; explicit facts always win.

    Any design trait whose visual category is already covered by an explicit fact is suppressed
    so neutral design assumptions never override explicit book facts.
    """
    explicit_list = [thaw(f) for f in explicit_facts]
    explicit_blob = " ".join(
        f"{f.get('description_en', '')} {f.get('description_ru', '')}"
        for f in explicit_list
    )
    kept_design: list[dict[str, Any]] = []
    for trait_obj in design_traits:
        trait_name = str(trait_obj.get("trait", ""))
        pattern = _TRAIT_PATTERNS.get(trait_name)
        if pattern is not None and pattern.search(explicit_blob):
            continue
        kept_design.append(thaw(trait_obj))
    return tuple(explicit_list), tuple(kept_design)


__all__ = [
    "TIMELESS_POSITION_KEY",
    "IllustrationReference",
    "build_alias_candidate_pairs",
    "format_position_key",
    "resolve_appearance_with_design_fallback",
    "resolve_verdict_graph_clusters",
]
