"""Read-only view projections (`DTO-01` and `DTO-02`) for optional illustration preparation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from linguary.application.context_builder import textual_blocks
from linguary.application.ports.persistence import UnitOfWork
from linguary.domain import (
    CanonicalDocument,
    DocumentSection,
    TaskStatus,
    TaskType,
)
from linguary.domain.illustrations import (
    DEFAULT_ILLUSTRATION_POLICY,
    IllustrationGeneration,
    IllustrationGenerationState,
    IllustrationNotFoundError,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationScope,
    IllustrationScopeStage,
    IllustrationScopeState,
    IllustrationSettings,
    RegistryItem,
)

_REFERENCE_STAGES = frozenset(
    {
        IllustrationScopeStage.EXTRACT.value,
        IllustrationScopeStage.ALIAS.value,
        IllustrationScopeStage.DESIGN.value,
    }
)
_CHAPTER_STAGES = frozenset(
    {
        IllustrationScopeStage.FACT_SCOPE.value,
        IllustrationScopeStage.CANDIDATES.value,
        IllustrationScopeStage.SELECTION.value,
        IllustrationScopeStage.PROMPT.value,
        IllustrationScopeStage.CHAPTER.value,
    }
)
_ILLUSTRATION_TASK_TYPES = frozenset(
    {
        TaskType.PREPARE_ILLUSTRATION_REFERENCE,
        TaskType.PREPARE_CHAPTER_ILLUSTRATIONS,
    }
)
_SENSITIVE_DETAIL_KEYS = frozenset(
    {
        "raw_response",
        "raw_provider_response",
        "api_key",
        "authorization",
        "proxy_url",
        "secret",
        "headers",
    }
)


def _read_transaction(unit_of_work: UnitOfWork):
    """Open a read-only transaction when supported by the unit of work."""
    return getattr(unit_of_work, "read_transaction", unit_of_work.transaction)()


@dataclass(frozen=True, slots=True)
class IllustrationSettingsView:
    enabled: bool
    max_per_chapter: int
    style: str
    model_reference: str | None
    model_prompts: str | None


@dataclass(frozen=True, slots=True)
class IllustrationDefaultsHintView:
    provider_ready: bool
    default_model: str | None
    max_calls_per_generation: int


@dataclass(frozen=True, slots=True)
class IllustrationSourceProgressView:
    expected_chapters: int
    ready_chapters: int
    skipped_sections: int
    blocking_section_id: str | None


@dataclass(frozen=True, slots=True)
class IllustrationRegistryProgressView:
    state: str
    reason_code: str | None
    completed_scopes: int
    total_scopes: int | None
    entities: int
    facts: int


@dataclass(frozen=True, slots=True)
class IllustrationChapterHeaderView:
    section_id: str
    order: int
    title: str
    state: str
    reason_code: str | None
    moment_count: int | None
    source_block_id: str | None


@dataclass(frozen=True, slots=True)
class IllustrationPaginationView:
    offset: int
    limit: int
    total: int
    next_offset: int | None
    previous_offset: int | None


@dataclass(frozen=True, slots=True)
class IllustrationAnchorView:
    block_id: str
    start: int
    end: int
    quote: str


@dataclass(frozen=True, slots=True)
class IllustrationAppearanceView:
    entity_id: str
    name_ru: str
    description_ru: str
    description_en: str
    fact_ids: tuple[str, ...]
    assumptions_ru: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IllustrationMomentView:
    id: str
    scene_id: str
    title_ru: str
    reason_ru: str
    event_ru: str
    prompt_en: str
    insert_after_block_id: str
    source_block_id: str | None
    anchors: tuple[IllustrationAnchorView, ...]
    character_descriptions: tuple[IllustrationAppearanceView, ...]
    location_descriptions: tuple[IllustrationAppearanceView, ...]
    assumptions_ru: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IllustrationSelectedChapterView:
    section_id: str
    title: str
    state: str
    reason_code: str | None
    zero_reason_ru: str | None
    moments: tuple[IllustrationMomentView, ...]


@dataclass(frozen=True, slots=True)
class IllustrationHistoryItemView:
    generation_id: str
    generation_number: int
    state: str
    settings_revision: int
    source_sha256: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IllustrationCostBucketView:
    calls: int
    total_tokens: int | None
    cost_usd: str | None
    estimate_usd: None = None


@dataclass(frozen=True, slots=True)
class IllustrationCostsView:
    reference: IllustrationCostBucketView
    chapters: IllustrationCostBucketView
    total: IllustrationCostBucketView


@dataclass(frozen=True, slots=True)
class IllustrationView:
    book_id: str
    book_title: str
    enabled: bool
    settings_revision: int
    settings: IllustrationSettingsView
    defaults_hint: IllustrationDefaultsHintView
    active_generation_id: str | None
    viewed_generation_id: str | None
    is_current: bool
    is_history: bool
    state: str
    reason_code: str | None
    source_progress: IllustrationSourceProgressView
    registry_progress: IllustrationRegistryProgressView
    allowed_actions: tuple[str, ...]
    chapter_headers: tuple[IllustrationChapterHeaderView, ...]
    pagination: IllustrationPaginationView
    selected_chapter: IllustrationSelectedChapterView | None
    history: tuple[IllustrationHistoryItemView, ...]
    costs: IllustrationCostsView


@dataclass(frozen=True, slots=True)
class IllustrationRegistryItemView:
    kind: str
    item_key: str
    entity_id: str
    position_key: str
    summary_ru: str
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class IllustrationRegistryView:
    book_id: str
    generation_id: str | None
    is_current: bool
    items: tuple[IllustrationRegistryItemView, ...]
    pagination: IllustrationPaginationView


def _make_pagination(*, offset: int, limit: int, total: int) -> IllustrationPaginationView:
    next_offset = offset + limit if offset + limit < total else None
    previous_offset = max(0, min(offset, total) - limit) if offset > 0 and total > 0 else None
    return IllustrationPaginationView(
        offset=offset,
        limit=limit,
        total=total,
        next_offset=next_offset,
        previous_offset=previous_offset,
    )


def _is_content_section(document: CanonicalDocument, section: DocumentSection) -> bool:
    return (
        bool(textual_blocks(section))
        and not document.is_front_matter(section)
        and not getattr(section, "is_notes_only", False)
    )


def _fallback_source_evaluation(
    tx: UnitOfWork,
    book_id: str,
    document: CanonicalDocument | None,
) -> dict[str, Any]:
    """Conservative source readiness check used before P2.T1 `IllustrationSource` is wired."""
    conditions = tx.work.latest_conditions(book_id)
    if conditions is None:
        return {
            "compatible": False,
            "reason_code": IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
            "source_sha256": None,
            "expected_chapters": 0,
            "ready_chapters": 0,
            "skipped_sections": 0,
            "blocking_section_id": None,
            "content_section_ids": (),
        }
    payload = conditions.payload
    staged = bool(payload.get("staged_analysis"))
    continue_legacy = bool(payload.get("continue_legacy"))
    profile = str(payload.get("content_plan_profile") or "")
    if not staged or continue_legacy or profile != "content-plan-v3":
        return {
            "compatible": False,
            "reason_code": IllustrationReasonCode.SOURCE_INCOMPATIBLE.value,
            "source_sha256": payload.get("source_digest"),
            "expected_chapters": 0,
            "ready_chapters": 0,
            "skipped_sections": 0,
            "blocking_section_id": None,
            "content_section_ids": (),
        }
    if document is None:
        return {
            "compatible": True,
            "reason_code": IllustrationReasonCode.NARRATIVE_MISSING.value,
            "source_sha256": payload.get("source_digest"),
            "expected_chapters": 0,
            "ready_chapters": 0,
            "skipped_sections": 0,
            "blocking_section_id": None,
            "content_section_ids": (),
        }
    content_sections = [s for s in document.ordered_sections if _is_content_section(document, s)]
    skipped = len(document.ordered_sections) - len(content_sections)
    passes = {
        p.section_id: p
        for p in tx.analyses.list_passes_for_book(book_id, scope_key="")
        if p.pass_kind.value == "narrative"
    }
    ready = 0
    blocking: str | None = None
    for sec in content_sections:
        if sec.id in passes:
            ready += 1
        elif blocking is None:
            blocking = sec.id
    reason_code: str | None = None
    if not content_sections:
        reason_code = IllustrationReasonCode.NO_CONTENT.value
    elif blocking is not None:
        tasks = tx.tasks.list_for_book(book_id)
        failed = any(
            t.task_type is TaskType.ANALYZE_CHAPTER
            and t.scope_id == blocking
            and t.status is TaskStatus.FAILED
            for t in tasks
        )
        reason_code = (
            IllustrationReasonCode.NARRATIVE_FAILED.value
            if failed
            else IllustrationReasonCode.NARRATIVE_MISSING.value
        )
    return {
        "compatible": True,
        "reason_code": reason_code,
        "source_sha256": payload.get("source_digest"),
        "expected_chapters": len(content_sections),
        "ready_chapters": ready,
        "skipped_sections": skipped,
        "blocking_section_id": blocking,
        "content_section_ids": tuple(s.id for s in content_sections),
    }


def _compute_costs(
    tx: UnitOfWork,
    book_id: str,
    generation_id: str | None,
) -> IllustrationCostsView:
    empty = IllustrationCostBucketView(calls=0, total_tokens=None, cost_usd=None, estimate_usd=None)
    if generation_id is None:
        return IllustrationCostsView(reference=empty, chapters=empty, total=empty)

    attempt_stages = dict(tx.illustrations.list_generation_attempt_ids(book_id, generation_id))
    if not attempt_stages:
        return IllustrationCostsView(reference=empty, chapters=empty, total=empty)

    logs = [
        entry
        for entry in tx.llm_calls.list_for_book(book_id)
        if entry.attempt_id is not None and entry.attempt_id in attempt_stages
    ]

    def _aggregate(entries: list[Any]) -> IllustrationCostBucketView:
        if not entries:
            return empty
        tokens_seen = False
        token_sum = 0
        cost_seen = False
        cost_sum = Decimal(0)
        for entry in entries:
            if entry.usage is not None and entry.usage.total_tokens is not None:
                tokens_seen = True
                token_sum += entry.usage.total_tokens
            if entry.cost is not None:
                cost_seen = True
                cost_sum += entry.cost
        return IllustrationCostBucketView(
            calls=len(entries),
            total_tokens=token_sum if tokens_seen else None,
            cost_usd=format(cost_sum, "f") if cost_seen else None,
            estimate_usd=None,
        )

    ref_logs = [e for e in logs if attempt_stages.get(e.attempt_id or "") in _REFERENCE_STAGES]
    ch_logs = [e for e in logs if attempt_stages.get(e.attempt_id or "") in _CHAPTER_STAGES]
    return IllustrationCostsView(
        reference=_aggregate(ref_logs),
        chapters=_aggregate(ch_logs),
        total=_aggregate(logs),
    )


def _registry_progress_for(
    tx: UnitOfWork,
    generation: IllustrationGeneration | None,
    scopes: tuple[IllustrationScope, ...],
) -> IllustrationRegistryProgressView:
    if generation is None:
        return IllustrationRegistryProgressView(
            state="not_started",
            reason_code=None,
            completed_scopes=0,
            total_scopes=None,
            entities=0,
            facts=0,
        )
    ref_scopes = [s for s in scopes if s.stage.value in _REFERENCE_STAGES]
    _, entity_count = tx.illustrations.query_registry_items(
        generation.id, kind=IllustrationRegistryKind.ENTITY, offset=0, limit=1
    )
    _, fact_count = tx.illustrations.query_registry_items(
        generation.id, kind=IllustrationRegistryKind.FACT, offset=0, limit=1
    )
    active_ref = [s for s in ref_scopes if s.state is not IllustrationScopeState.SPLIT]
    completed = sum(1 for s in active_ref if s.state is IllustrationScopeState.ACCEPTED)
    total = len(active_ref) if active_ref else None

    if generation.state is IllustrationGenerationState.SKIPPED:
        return IllustrationRegistryProgressView(
            state="skipped",
            reason_code=generation.reason_code,
            completed_scopes=completed,
            total_scopes=total,
            entities=entity_count,
            facts=fact_count,
        )
    if generation.registry_sha256 is not None:
        return IllustrationRegistryProgressView(
            state="ready",
            reason_code=None,
            completed_scopes=completed,
            total_scopes=total if total is not None else 0,
            entities=entity_count,
            facts=fact_count,
        )
    failed = next((s for s in active_ref if s.state is IllustrationScopeState.FAILED), None)
    if failed is not None:
        return IllustrationRegistryProgressView(
            state="failed",
            reason_code=failed.reason_code,
            completed_scopes=completed,
            total_scopes=total,
            entities=entity_count,
            facts=fact_count,
        )
    if any(s.state is IllustrationScopeState.RUNNING for s in active_ref):
        return IllustrationRegistryProgressView(
            state="running",
            reason_code=None,
            completed_scopes=completed,
            total_scopes=total,
            entities=entity_count,
            facts=fact_count,
        )
    if any(s.state is IllustrationScopeState.PENDING for s in active_ref):
        return IllustrationRegistryProgressView(
            state="queued",
            reason_code=None,
            completed_scopes=completed,
            total_scopes=total,
            entities=entity_count,
            facts=fact_count,
        )
    if generation.state is IllustrationGenerationState.WAITING_SOURCE:
        return IllustrationRegistryProgressView(
            state="waiting_source",
            reason_code=generation.reason_code,
            completed_scopes=completed,
            total_scopes=total,
            entities=entity_count,
            facts=fact_count,
        )
    return IllustrationRegistryProgressView(
        state="not_started",
        reason_code=None,
        completed_scopes=completed,
        total_scopes=total,
        entities=entity_count,
        facts=fact_count,
    )


def _block_map(document: CanonicalDocument | None) -> dict[str, str]:
    if document is None:
        return {}
    return {
        block.id: block.text
        for section in document.ordered_sections
        for block in section.ordered_blocks
    }


def _first_textual_block_id(section: DocumentSection | None) -> str | None:
    if section is None:
        return None
    blocks = textual_blocks(section)
    return blocks[0].id if blocks else None


def _section_matches_manifest(
    section: DocumentSection | None,
    generation: IllustrationGeneration | None,
) -> bool:
    if section is None:
        return False
    if generation is None or generation.manifest is None:
        return True
    manifest_sections = generation.manifest.get("sections")
    if not isinstance(manifest_sections, (list, tuple)):
        return True
    for entry in manifest_sections:
        if isinstance(entry, Mapping) and entry.get("section_id") == section.id:
            expected_blocks = entry.get("block_digests")
            if isinstance(expected_blocks, Mapping):
                actual_texts = {b.id: b.text for b in section.ordered_blocks}
                from linguary.domain.content_plan import canonical_sha

                for blk_id, digest in expected_blocks.items():
                    if blk_id not in actual_texts or canonical_sha(actual_texts[blk_id]) != digest:
                        return False
            return True
    return False


def _moment_source_block_id(
    anchors: tuple[IllustrationAnchorView, ...],
    blocks_by_id: Mapping[str, str],
) -> str | None:
    if not anchors:
        return None
    for anchor in anchors:
        text = blocks_by_id.get(anchor.block_id)
        if text is None or text[anchor.start : anchor.end] != anchor.quote:
            return None
    return anchors[0].block_id


def _decode_moments(
    payload: Mapping[str, Any] | None,
    blocks_by_id: Mapping[str, str],
) -> tuple[str | None, tuple[IllustrationMomentView, ...]]:
    if not isinstance(payload, Mapping):
        return None, ()
    zero_reason = payload.get("zero_reason_ru")
    zero_reason_ru = str(zero_reason) if isinstance(zero_reason, str) and zero_reason.strip() else None
    raw_moments = payload.get("moments")
    if not isinstance(raw_moments, (list, tuple)):
        return zero_reason_ru, ()
    moments: list[IllustrationMomentView] = []
    for item in raw_moments:
        if not isinstance(item, Mapping):
            continue
        anchors = tuple(
            IllustrationAnchorView(
                block_id=str(a["block_id"]),
                start=int(a["start"]),
                end=int(a["end"]),
                quote=str(a["quote"]),
            )
            for a in (item.get("anchors") or ())
            if isinstance(a, Mapping)
        )
        chars = tuple(
            IllustrationAppearanceView(
                entity_id=str(c["entity_id"]),
                name_ru=str(c["name_ru"]),
                description_ru=str(c["description_ru"]),
                description_en=str(c["description_en"]),
                fact_ids=tuple(str(fid) for fid in (c.get("fact_ids") or ())),
                assumptions_ru=tuple(str(x) for x in (c.get("assumptions_ru") or ())),
            )
            for c in (item.get("character_descriptions") or ())
            if isinstance(c, Mapping)
        )
        locs = tuple(
            IllustrationAppearanceView(
                entity_id=str(loc["entity_id"]),
                name_ru=str(loc["name_ru"]),
                description_ru=str(loc["description_ru"]),
                description_en=str(loc["description_en"]),
                fact_ids=tuple(str(fid) for fid in (loc.get("fact_ids") or ())),
                assumptions_ru=tuple(str(x) for x in (loc.get("assumptions_ru") or ())),
            )
            for loc in (item.get("location_descriptions") or ())
            if isinstance(loc, Mapping)
        )
        moments.append(
            IllustrationMomentView(
                id=str(item["id"]),
                scene_id=str(item["scene_id"]),
                title_ru=str(item["title_ru"]),
                reason_ru=str(item["reason_ru"]),
                event_ru=str(item["event_ru"]),
                prompt_en=str(item["prompt_en"]),
                insert_after_block_id=str(item["insert_after_block_id"]),
                source_block_id=_moment_source_block_id(anchors, blocks_by_id),
                anchors=anchors,
                character_descriptions=chars,
                location_descriptions=locs,
                assumptions_ru=tuple(str(x) for x in (item.get("assumptions_ru") or ())),
            )
        )
    return zero_reason_ru, tuple(moments)


def _is_book_paused(tx: UnitOfWork, book_id: str) -> bool:
    """Check whether the book is currently paused via work bindings."""
    try:
        is_paused = getattr(tx.work, "is_book_paused", None)
        if callable(is_paused):
            return bool(is_paused(book_id))
        conditions = tx.work.latest_conditions(book_id)
        if conditions is not None and bool(conditions.payload.get("paused")):
            return True
    except Exception:
        return False
    return False


def _sanitize_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).lower() in _SENSITIVE_DETAIL_KEYS:
            continue
        if isinstance(value, Mapping):
            out[str(key)] = _sanitize_details(value)
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [
                _sanitize_details(v) if isinstance(v, Mapping) else v for v in value
            ]
        else:
            out[str(key)] = value
    return out


def _registry_summary_ru(item: RegistryItem) -> str:
    payload = item.payload
    for key in ("summary_ru", "description_ru", "name_ru", "canonical_name", "name"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if item.kind is IllustrationRegistryKind.ALIAS_DECISION:
        verdict = payload.get("verdict")
        if isinstance(verdict, str) and verdict:
            return f"Решение об алиасе: {verdict}"
    return item.item_key


def build_illustration_view(
    unit_of_work: UnitOfWork,
    book_id: str,
    *,
    offset: int = 0,
    limit: int = 10,
    section_id: str | None = None,
    generation_id: str | None = None,
    provider_ready: bool = False,
    default_model: str | None = None,
    source_snapshot_fn: Callable[[Any, str], Any] | None = None,
    is_book_paused_fn: Callable[[Any, str], bool] | None = None,
) -> IllustrationView:
    """Build `DTO-01` (`IllustrationView`) strictly inside a read-only transaction."""
    with _read_transaction(unit_of_work) as tx:
        book = tx.books.get(book_id)
        if book is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.BOOK_NOT_FOUND,
                f"книга {book_id} не найдена",
            )
        document = tx.documents.get(book_id)
        head = tx.illustrations.get_head(book_id)
        effective_settings: IllustrationSettings = head.effective_settings
        active_generation_id = head.active_generation_id

        viewed_generation: IllustrationGeneration | None = None
        if generation_id is not None:
            viewed_generation = tx.illustrations.get_generation(book_id, generation_id)
            if viewed_generation is None:
                raise IllustrationNotFoundError(
                    IllustrationReasonCode.GENERATION_NOT_FOUND,
                    f"поколение иллюстраций {generation_id} не найдено",
                )
        elif active_generation_id is not None:
            viewed_generation = tx.illustrations.get_generation(book_id, active_generation_id)

        viewed_generation_id = viewed_generation.id if viewed_generation is not None else None
        is_history = viewed_generation_id is not None and viewed_generation_id != active_generation_id

        # Evaluate current source readiness (either via P2.T1 IllustrationSource or fallback).
        if source_snapshot_fn is not None:
            snap = source_snapshot_fn(tx, book_id)
            source_compatible = bool(snap.compatible)
            source_reason_code = snap.reason_code
            source_sha256 = snap.source_sha256
            source_manifest_sha = snap.manifest_sha256
            expected_chapters = snap.expected_chapters
            ready_chapters = snap.ready_chapters
            skipped_sections = snap.skipped_sections
            blocking_section_id = snap.blocking_section_id
            content_section_ids = snap.content_section_ids
        else:
            eval_info = _fallback_source_evaluation(tx, book_id, document)
            source_compatible = bool(eval_info["compatible"])
            source_reason_code = eval_info["reason_code"]
            source_sha256 = eval_info["source_sha256"]
            source_manifest_sha = None
            expected_chapters = eval_info["expected_chapters"]
            ready_chapters = eval_info["ready_chapters"]
            skipped_sections = eval_info["skipped_sections"]
            blocking_section_id = eval_info["blocking_section_id"]
            content_section_ids = eval_info["content_section_ids"]

        scopes: tuple[IllustrationScope, ...] = (
            tx.illustrations.list_scopes(viewed_generation.id)
            if viewed_generation is not None
            else ()
        )

        # Check active generation dependency compatibility.
        deps_compatible = False
        stale_reason: str | None = None
        if viewed_generation is not None and not is_history:
            if viewed_generation.state is IllustrationGenerationState.REVOKED:
                stale_reason = viewed_generation.reason_code or IllustrationReasonCode.SETTINGS_CHANGED.value
            elif head.settings_id is not None and viewed_generation.settings_id != head.settings_id:
                stale_reason = IllustrationReasonCode.SETTINGS_CHANGED.value
            elif not source_compatible:
                stale_reason = source_reason_code or IllustrationReasonCode.SOURCE_INCOMPATIBLE.value
            elif source_sha256 is not None and viewed_generation.source_sha256 != source_sha256:
                stale_reason = IllustrationReasonCode.SOURCE_CHANGED.value
            elif (
                viewed_generation.manifest_sha256 is not None
                and source_manifest_sha is not None
                and viewed_generation.manifest_sha256 != source_manifest_sha
            ):
                stale_reason = IllustrationReasonCode.SOURCE_CHANGED.value
            else:
                deps_compatible = True

        is_current = viewed_generation is not None and not is_history and deps_compatible

        # Paginated chapter headers (no payloads fetched for non-selected chapters).
        header_rows, total_headers = tx.illustrations.get_chapter_header_page(
            book_id,
            viewed_generation_id,
            offset=offset,
            limit=limit,
        )
        doc_sections_by_id = (
            {s.id: s for s in document.ordered_sections} if document is not None else {}
        )
        blocks_by_id = _block_map(document)

        chapter_headers = tuple(
            IllustrationChapterHeaderView(
                section_id=row.section_id,
                order=row.order,
                title=row.title,
                state=row.state,
                reason_code=row.reason_code,
                moment_count=row.moment_count,
                source_block_id=(
                    _first_textual_block_id(doc_sections_by_id.get(row.section_id))
                    if (
                        not is_history
                        and _section_matches_manifest(
                            doc_sections_by_id.get(row.section_id), viewed_generation
                        )
                    )
                    else None
                ),
            )
            for row in header_rows
        )

        # Selected chapter validation & projection (load payload only for this one section).
        selected_chapter: IllustrationSelectedChapterView | None = None
        if section_id is not None:
            chapter_scopes = [
                s for s in scopes if s.stage is IllustrationScopeStage.CHAPTER
            ]
            matched_scope = next(
                (s for s in chapter_scopes if s.section_id == section_id), None
            )
            if matched_scope is not None:
                zero_reason_ru, moments = _decode_moments(matched_scope.payload, blocks_by_id)
                sec_state = (
                    "ready"
                    if matched_scope.state is IllustrationScopeState.ACCEPTED
                    else "queued"
                    if matched_scope.state is IllustrationScopeState.PENDING
                    else "running"
                    if matched_scope.state is IllustrationScopeState.SPLIT
                    else matched_scope.state.value
                )
                selected_chapter = IllustrationSelectedChapterView(
                    section_id=section_id,
                    title=matched_scope.section_title or "",
                    state=sec_state,
                    reason_code=matched_scope.reason_code,
                    zero_reason_ru=zero_reason_ru,
                    moments=moments,
                )
            elif chapter_scopes:
                raise IllustrationNotFoundError(
                    IllustrationReasonCode.SECTION_NOT_FOUND,
                    f"глава {section_id} не найдена в поколении",
                )
            elif viewed_generation is not None and viewed_generation.manifest is not None:
                manifest_secs = viewed_generation.manifest.get("sections") or ()
                found_meta = next(
                    (
                        s
                        for s in manifest_secs
                        if isinstance(s, Mapping) and s.get("section_id") == section_id
                    ),
                    None,
                )
                if found_meta is None:
                    raise IllustrationNotFoundError(
                        IllustrationReasonCode.SECTION_NOT_FOUND,
                        f"глава {section_id} не найдена в поколении",
                    )
                selected_chapter = IllustrationSelectedChapterView(
                    section_id=section_id,
                    title=str(found_meta.get("title") or ""),
                    state="waiting_source"
                    if viewed_generation.state is IllustrationGenerationState.WAITING_SOURCE
                    else "queued",
                    reason_code=viewed_generation.reason_code,
                    zero_reason_ru=None,
                    moments=(),
                )
            else:
                doc_sec = doc_sections_by_id.get(section_id)
                if doc_sec is None:
                    raise IllustrationNotFoundError(
                        IllustrationReasonCode.SECTION_NOT_FOUND,
                        f"глава {section_id} не найдена в книге {book_id}",
                    )
                selected_chapter = IllustrationSelectedChapterView(
                    section_id=section_id,
                    title=doc_sec.title,
                    state="not_started",
                    reason_code=None,
                    zero_reason_ru=None,
                    moments=(),
                )

        registry_progress = _registry_progress_for(tx, viewed_generation, scopes)
        history_rows = tx.illustrations.list_history(book_id, limit=10)
        history = tuple(
            IllustrationHistoryItemView(
                generation_id=h.generation_id,
                generation_number=h.generation_number,
                state=h.state.value,
                settings_revision=h.settings_revision,
                source_sha256=h.source_sha256,
                created_at=h.created_at,
            )
            for h in history_rows
        )
        costs = _compute_costs(tx, book_id, viewed_generation_id)

        # Book pause check.
        paused = (
            is_book_paused_fn(tx, book_id)
            if is_book_paused_fn is not None
            else _is_book_paused(tx, book_id)
        )

        # Compute top-level state and reason_code per DTO-01 priority rules.
        active_scopes = [s for s in scopes if s.state is not IllustrationScopeState.SPLIT]
        failed_scope = next((s for s in active_scopes if s.state is IllustrationScopeState.FAILED), None)
        has_running = any(s.state is IllustrationScopeState.RUNNING for s in active_scopes)
        has_pending = any(s.state is IllustrationScopeState.PENDING for s in active_scopes)
        has_unfinished_work = has_running or has_pending or (
            viewed_generation is not None
            and viewed_generation.state
            in {
                IllustrationGenerationState.WAITING_SOURCE,
                IllustrationGenerationState.QUEUED,
                IllustrationGenerationState.RUNNING,
            }
        )

        if is_history:
            state = "stale"
            reason_code = (
                viewed_generation.reason_code
                if viewed_generation is not None and viewed_generation.reason_code
                else IllustrationReasonCode.SETTINGS_CHANGED.value
            )
        elif viewed_generation is not None and not deps_compatible:
            state = "stale"
            reason_code = stale_reason
        elif not effective_settings.enabled:
            state = "disabled"
            reason_code = IllustrationReasonCode.CHANNEL_DISABLED.value
        elif viewed_generation is None:
            if not source_compatible:
                state = "needs_analysis"
                reason_code = source_reason_code or IllustrationReasonCode.SOURCE_INCOMPATIBLE.value
            else:
                state = "not_started"
                reason_code = None
        elif viewed_generation.state is IllustrationGenerationState.SKIPPED:
            state = "skipped"
            reason_code = viewed_generation.reason_code or IllustrationReasonCode.NO_CONTENT.value
        elif failed_scope is not None or viewed_generation.state is IllustrationGenerationState.FAILED:
            state = "failed"
            reason_code = (
                failed_scope.reason_code
                if failed_scope is not None
                else viewed_generation.reason_code
            )
        elif viewed_generation.state is IllustrationGenerationState.READY:
            state = "ready"
            reason_code = None
        elif paused and has_unfinished_work:
            state = "paused"
            reason_code = IllustrationReasonCode.BOOK_PAUSED.value
        elif has_running or viewed_generation.state is IllustrationGenerationState.RUNNING:
            state = "running"
            reason_code = None
        elif has_pending or viewed_generation.state is IllustrationGenerationState.QUEUED:
            state = "queued"
            reason_code = None
        else:
            state = "waiting_source"
            reason_code = source_reason_code or viewed_generation.reason_code

        # Exact allowed_actions matrix per DTO-01.
        actions: list[str] = ["save_settings"]
        if not is_history and effective_settings.enabled:
            if source_compatible and state != "needs_analysis":
                if viewed_generation is None:
                    actions.append("prepare")
                elif state == "stale":
                    actions.append("prepare")
                else:
                    requested_set = set(viewed_generation.requested_sections)
                    has_unrequested = any(sid not in requested_set for sid in content_section_ids)
                    if has_unrequested and state not in {"skipped"}:
                        actions.append("prepare")
            if viewed_generation is not None and is_current:
                max_calls = effective_settings.policy.max_calls_per_generation
                has_retryable = any(s.is_retryable_failure for s in active_scopes)
                if has_retryable and (viewed_generation.calls_reserved + viewed_generation.calls_sent) < max_calls:
                    actions.append("retry")
                if reason_code in {
                    IllustrationReasonCode.CALL_LIMIT_EXHAUSTED.value,
                    IllustrationReasonCode.SCOPE_LIMIT_EXHAUSTED.value,
                }:
                    actions.append("new_generation")
        actions.append("refresh")

        max_calls_hint = effective_settings.policy.max_calls_per_generation
        if head.settings_revision is None:
            max_calls_hint = DEFAULT_ILLUSTRATION_POLICY.max_calls_per_generation

        return IllustrationView(
            book_id=book_id,
            book_title=book.metadata.title,
            enabled=effective_settings.enabled,
            settings_revision=head.revision_number,
            settings=IllustrationSettingsView(
                enabled=effective_settings.enabled,
                max_per_chapter=effective_settings.max_per_chapter,
                style=effective_settings.style,
                model_reference=effective_settings.model_reference,
                model_prompts=effective_settings.model_prompts,
            ),
            defaults_hint=IllustrationDefaultsHintView(
                provider_ready=provider_ready,
                default_model=default_model,
                max_calls_per_generation=max_calls_hint,
            ),
            active_generation_id=active_generation_id,
            viewed_generation_id=viewed_generation_id,
            is_current=is_current,
            is_history=is_history,
            state=state,
            reason_code=reason_code,
            source_progress=IllustrationSourceProgressView(
                expected_chapters=expected_chapters,
                ready_chapters=ready_chapters,
                skipped_sections=skipped_sections,
                blocking_section_id=blocking_section_id,
            ),
            registry_progress=registry_progress,
            allowed_actions=tuple(actions),
            chapter_headers=chapter_headers,
            pagination=_make_pagination(offset=offset, limit=limit, total=total_headers),
            selected_chapter=selected_chapter,
            history=history,
            costs=costs,
        )


def build_illustration_registry_view(
    unit_of_work: UnitOfWork,
    book_id: str,
    *,
    generation_id: str | None = None,
    entity_id: str | None = None,
    kind: IllustrationRegistryKind | None = None,
    offset: int = 0,
    limit: int = 50,
    source_snapshot_fn: Callable[[Any, str], Any] | None = None,
) -> IllustrationRegistryView:
    """Build read-only `DTO-02` (`IllustrationRegistryView`) strictly inside a read-only transaction."""
    with _read_transaction(unit_of_work) as tx:
        book = tx.books.get(book_id)
        if book is None:
            raise IllustrationNotFoundError(
                IllustrationReasonCode.BOOK_NOT_FOUND,
                f"книга {book_id} не найдена",
            )
        head = tx.illustrations.get_head(book_id)
        target_generation: IllustrationGeneration | None = None
        if generation_id is not None:
            target_generation = tx.illustrations.get_generation(book_id, generation_id)
            if target_generation is None:
                raise IllustrationNotFoundError(
                    IllustrationReasonCode.GENERATION_NOT_FOUND,
                    f"поколение иллюстраций {generation_id} не найдено",
                )
        elif head.active_generation_id is not None:
            target_generation = tx.illustrations.get_generation(book_id, head.active_generation_id)

        if target_generation is None:
            return IllustrationRegistryView(
                book_id=book_id,
                generation_id=None,
                is_current=False,
                items=(),
                pagination=_make_pagination(offset=offset, limit=limit, total=0),
            )

        is_current = (
            target_generation.id == head.active_generation_id
            and target_generation.state is not IllustrationGenerationState.REVOKED
            and (head.settings_id is None or target_generation.settings_id == head.settings_id)
        )
        if is_current and source_snapshot_fn is not None:
            snap = source_snapshot_fn(tx, book_id)
            if not snap.compatible:
                is_current = False
            elif snap.source_sha256 is not None and target_generation.source_sha256 != snap.source_sha256:
                is_current = False
            elif (
                target_generation.manifest_sha256 is not None
                and snap.manifest_sha256 is not None
                and target_generation.manifest_sha256 != snap.manifest_sha256
            ):
                is_current = False

        rows, total = tx.illustrations.query_registry_items(
            target_generation.id,
            entity_id=entity_id,
            kind=kind,
            offset=offset,
            limit=limit,
        )
        items = tuple(
            IllustrationRegistryItemView(
                kind=row.kind.value,
                item_key=row.item_key,
                entity_id=row.entity_id,
                position_key=row.position_key,
                summary_ru=_registry_summary_ru(row),
                details=_sanitize_details(row.payload),
            )
            for row in rows
        )
        return IllustrationRegistryView(
            book_id=book_id,
            generation_id=target_generation.id,
            is_current=is_current,
            items=items,
            pagination=_make_pagination(offset=offset, limit=limit, total=total),
        )


__all__ = [
    "IllustrationAnchorView",
    "IllustrationAppearanceView",
    "IllustrationChapterHeaderView",
    "IllustrationCostBucketView",
    "IllustrationCostsView",
    "IllustrationDefaultsHintView",
    "IllustrationHistoryItemView",
    "IllustrationMomentView",
    "IllustrationPaginationView",
    "IllustrationRegistryItemView",
    "IllustrationRegistryProgressView",
    "IllustrationRegistryView",
    "IllustrationSelectedChapterView",
    "IllustrationSettingsView",
    "IllustrationSourceProgressView",
    "IllustrationView",
    "build_illustration_registry_view",
    "build_illustration_view",
]
