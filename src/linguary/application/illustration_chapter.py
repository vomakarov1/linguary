"""Applicable appearance scoping, monochrome prompt rendering, and atomic chapter acceptance (`P3.T2`)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from linguary.application.analysis_pass_cache import encode_illustration_pass
from linguary.application.illustration_context import (
    IllustrationContext,
    compact_safe_detail,
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
    compute_chapter_fingerprint,
)
from linguary.application.illustration_validation import (
    ChapterIllustrationPlan,
    IllustrationAppearance,
    ValidatedMoment,
    compose_monochrome_prompt,
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
)
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import SourceAnchor, canonical_sha
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
    IllustrationValidationError,
    RegistryItem,
)
from linguary.domain.llm_log import LLMCallStage

_MAX_APPLICABLE_FACTS = 32
_DEFAULT_ZERO_UNRESOLVED_RU = (
    "Не удалось однозначно определить образы ключевых участников для выбранных моментов главы."
)


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


def _dedupe_texts(items: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate strings by whitespace/punctuation-insensitive casefolded key preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        cleaned = " ".join(str(raw).split()).strip()
        if not cleaned:
            continue
        key = cleaned.rstrip(".;, ").casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return tuple(result)


def _dedupe_semicolon_parts(items: Iterable[str]) -> list[str]:
    """Split semicolon-delimited fragments and deduplicate by normalized key preserving order."""
    parts: list[str] = []
    for item in items:
        for piece in str(item).split(";"):
            cleaned = " ".join(piece.split()).strip()
            if cleaned:
                parts.append(cleaned)
    return list(_dedupe_texts(parts))


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

    Returns:
      - `("resolved", entity_payload)` when uniquely matched
      - `("ambiguous", None)` when 2+ distinct entities match (never merged or guessed!)
      - `("unmatched", None)` when no entity in the registry matches
    """
    norm = normalize_name_token(name)
    if not norm:
        return "unmatched", None
    matches: list[Mapping[str, Any]] = []
    for ent in entities:
        if str(ent.get("entity_kind", "")) != expected_kind:
            continue
        names = [str(ent.get("canonical_name", ""))] + [
            str(a) for a in ent.get("aliases", ())
        ]
        if any(normalize_name_token(n) == norm for n in names if n):
            matches.append(ent)
    if len(matches) == 1:
        return "resolved", matches[0]
    if len(matches) > 1:
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
    request_input = {
        "event_id": moment_card.candidate_id,
        "event_position_key": moment_card.event_position_key,
        "event_ru": moment_card.event_ru,
        "entity_id": str(entity["entity_id"]),
        "entity_name": str(entity["canonical_name"]),
        "entity_kind": str(entity["entity_kind"]),
        "page_index": page_index,
        "total_pages": total_pages,
        "is_consolidation": is_consolidation,
        "max_applicable_fact_ids": max_applicable_fact_ids,
        "running_summary_ru": running_summary_ru,
        "running_summary_en": running_summary_en,
        "facts": compact_facts,
    }
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
        "entity_name": str(entity["canonical_name"]),
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
    revision: IllustrationSettingsRevision,
    route_id: str,
    reusable: IllustrationScope | None = None,
) -> IllustrationScopeSpec:
    policy = revision.settings.policy
    request_input = {
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
    }
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
            ],
        ] = {}
        all_registry_facts: dict[str, dict[str, Any]] = {}

        for card, char_entities, loc_entities in bound_moments:
            event_pos = max(a.position_key for a in card.anchors)
            char_apps: list[IllustrationAppearance] = []
            loc_apps: list[IllustrationAppearance] = []
            allowed_fids: list[str] = []
            allowed_dkeys: list[str] = []

            for kind_bucket, ent_list in (("person", char_entities), ("location", loc_entities)):
                for ent in ent_list:
                    eid = str(ent["entity_id"])
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
                        [fi.raw_payload() for fi in fact_items],
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

                    eng_name = _select_english_name(ent)
                    fallback_en = eng_name or (
                        "character" if kind_bucket == "person" else "location"
                    )

                    if not entity_facts:
                        # No explicit facts -> use closed design traits (or safe non-Cyrillic fallback).
                        if design_payloads:
                            d_ru_parts = _dedupe_semicolon_parts(
                                str(d["description_ru"]) for d in design_payloads
                            )
                            d_en_parts = _dedupe_semicolon_parts(
                                str(d["description_en"]) for d in design_payloads
                            )
                            d_ru = "; ".join(d_ru_parts) or str(ent["canonical_name"])
                            d_en = "; ".join(d_en_parts) or fallback_en
                            assumps = _dedupe_texts(
                                str(d["assumption_ru"])
                                for d in design_payloads
                                if str(d.get("assumption_ru", "")).strip()
                            )
                            dkeys = [f"{eid}:{d['trait']}" for d in design_payloads]
                            allowed_dkeys.extend(dkeys)
                        else:
                            d_ru = str(ent["canonical_name"])
                            d_en = fallback_en
                            assumps = ()
                        app = IllustrationAppearance(
                            entity_id=eid,
                            name_ru=str(ent["canonical_name"]),
                            description_ru=d_ru,
                            description_en=d_en,
                            fact_ids=(),
                            assumptions_ru=assumps,
                            name_en=eng_name,
                        )
                        (char_apps if kind_bucket == "person" else loc_apps).append(app)
                        continue

                    # Pack entity_facts chronologically into bounded pages for FACT_SCOPE folding!
                    def _build_fs_probe(
                        f_page: Sequence[Mapping[str, Any]],
                        _card: SelectedMomentCard = card,
                        _ent: Mapping[str, Any] = ent,
                    ) -> LLMRequest[Any]:
                        return IllustrationContext.build_for_kind(
                            kind=LLMResponseKind.ILLUSTRATION_FACT_SCOPE,
                            stage=IllustrationScopeStage.FACT_SCOPE,
                            input_payload={
                                "event_id": _card.candidate_id,
                                "event_position_key": _card.event_position_key,
                                "event_ru": _card.event_ru,
                                "entity_id": str(_ent["entity_id"]),
                                "entity_name": str(_ent["canonical_name"]),
                                "entity_kind": str(_ent["entity_kind"]),
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
                            },
                            book_id=snapshot.book_id,
                            conditions_id=snapshot.conditions_id,
                            policy=policy,
                        )

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

                        def _build_consol_probe(
                            c_page: Sequence[Mapping[str, Any]],
                            *,
                            _card: SelectedMomentCard = card,
                            _eid: str = eid,
                            _ent: Mapping[str, Any] = ent,
                        ) -> LLMRequest[Any]:
                            return IllustrationContext.build_for_kind(
                                kind=LLMResponseKind.ILLUSTRATION_FACT_SCOPE,
                                stage=IllustrationScopeStage.FACT_SCOPE,
                                input_payload={
                                    "event_id": _card.candidate_id,
                                    "event_position_key": _card.event_position_key,
                                    "event_ru": _card.event_ru,
                                    "entity_id": _eid,
                                    "entity_name": str(_ent["canonical_name"]),
                                    "entity_kind": str(_ent["entity_kind"]),
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
                                },
                                book_id=snapshot.book_id,
                                conditions_id=snapshot.conditions_id,
                                policy=policy,
                            )

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
                            for fid in c_pay.get("applicable_fact_ids", ()):
                                fid_s = str(fid)
                                if fid_s not in round_app_ids:
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

                    # Combine explicit applicable facts with non-conflicting design fallbacks!
                    applicable_fact_objs = [
                        all_registry_facts[fid]
                        for fid in accumulated_app_ids
                        if fid in all_registry_facts
                    ]
                    _, kept_design = resolve_appearance_with_design_fallback(
                        applicable_fact_objs, design_payloads
                    )
                    desc_ru_parts = [running_ru] if running_ru.strip() else []
                    desc_en_parts = [running_en] if running_en.strip() else []
                    for d in kept_design:
                        desc_ru_parts.append(str(d["description_ru"]))
                        desc_en_parts.append(str(d["description_en"]))
                        if str(d.get("assumption_ru", "")).strip():
                            accumulated_assumps.append(str(d["assumption_ru"]).strip())
                        allowed_dkeys.append(f"{eid}:{d['trait']}")

                    allowed_fids.extend(accumulated_app_ids)
                    desc_ru_clean = _dedupe_semicolon_parts(desc_ru_parts)
                    desc_en_clean = _dedupe_semicolon_parts(desc_en_parts)
                    app = IllustrationAppearance(
                        entity_id=eid,
                        name_ru=str(ent["canonical_name"]),
                        description_ru="; ".join(desc_ru_clean)
                        or str(ent["canonical_name"]),
                        description_en="; ".join(desc_en_clean) or fallback_en,
                        fact_ids=tuple(accumulated_app_ids),
                        assumptions_ru=_dedupe_texts(accumulated_assumps),
                        name_en=eng_name,
                    )
                    (char_apps if kind_bucket == "person" else loc_apps).append(app)

            scoped_appearances_by_moment[card.candidate_id] = (
                char_apps,
                loc_apps,
                list(dict.fromkeys(allowed_fids)),
                list(dict.fromkeys(allowed_dkeys)),
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
            char_apps, loc_apps, allowed_fids, allowed_dkeys = scoped_appearances_by_moment[
                card.candidate_id
            ]
            pkey = f"prompt:{section_id}:{card.candidate_id}"
            existing_p = prompt_by_key.get(pkey)
            if existing_p is None:
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

            prompt_en = str(p_pay["prompt_en"])
            assumptions = _dedupe_texts(
                [
                    *(str(x) for x in p_pay.get("assumptions_ru", ())),
                    *(a for app in (*char_apps, *loc_apps) for a in app.assumptions_ru),
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
                    character_descriptions=tuple(char_apps),
                    location_descriptions=tuple(loc_apps),
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
            return _ValidationOutcome(
                ok=True,
                payload={
                    "schema_version": 1,
                    "event_id": payload.event_id,
                    "applicable_fact_ids": list(dict.fromkeys(payload.applicable_fact_ids)),
                    "excluded_fact_ids": list(dict.fromkeys(payload.excluded_fact_ids)),
                    "compact_description_ru": payload.compact_description_ru.strip(),
                    "compact_description_en": payload.compact_description_en.strip(),
                    "assumptions_ru": [a.strip() for a in payload.assumptions_ru if a.strip()],
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
            characters = [
                IllustrationAppearance.from_dict(c) for c in scope.summary.get("characters", ())
            ]
            locations = [
                IllustrationAppearance.from_dict(loc) for loc in scope.summary.get("locations", ())
            ]
            try:
                prompt_en = compose_monochrome_prompt(
                    payload.scene_composition_en,
                    [*characters, *locations],
                    revision.settings.style,
                )
            except IllustrationValidationError as err:
                return _ValidationOutcome(ok=False, repair_reason=str(err))
            return _ValidationOutcome(
                ok=True,
                payload={
                    "schema_version": 1,
                    "moment_id": payload.moment_id,
                    "scene_composition_en": payload.scene_composition_en.strip(),
                    "prompt_en": prompt_en,
                    "used_fact_ids": list(dict.fromkeys(payload.used_fact_ids)),
                    "used_design_keys": list(dict.fromkeys(payload.used_design_keys)),
                    "assumptions_ru": [a.strip() for a in payload.assumptions_ru if a.strip()],
                },
            )

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
