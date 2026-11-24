"""Bounded request builders, wire-budget packing, and deterministic quote resolution (`ALG-02`)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from linguary.application.context_builder import _analysis_visible_origins
from linguary.application.illustration_source import IllustrationSourceSnapshot
from linguary.application.ports.illustrations import RequestWireSizer
from linguary.application.ports.llm import LLMRequest, LLMResponseKind
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import SourceAnchor, canonical_sha
from linguary.domain.document import DocumentBlock
from linguary.domain.illustrations import (
    TIMELESS_POSITION_KEY,
    IllustrationPolicy,
    IllustrationScope,
    IllustrationScopeStage,
    IllustrationSettings,
    format_position_key,
)
from linguary.domain.llm_log import CallContext, LLMCallStage

REPAIR_RESERVE_BYTES = 1200
MAX_REPAIR_SUMMARY_BYTES = 1024

PERSON_CLOSED_TRAITS: tuple[str, ...] = (
    "hair",
    "eyes",
    "build",
    "age_appearance",
    "attire_default",
    "distinctive_mark",
    "palette_value",
)
LOCATION_CLOSED_TRAITS: tuple[str, ...] = (
    "architecture",
    "atmosphere",
    "palette_value",
    "distinctive_mark",
)
ALL_DESIGN_TRAITS: frozenset[str] = frozenset(PERSON_CLOSED_TRAITS) | frozenset(
    LOCATION_CLOSED_TRAITS
)

_REFERENCE_STAGES = frozenset(
    {
        IllustrationScopeStage.EXTRACT,
        IllustrationScopeStage.ALIAS,
        IllustrationScopeStage.DESIGN,
    }
)

_T = TypeVar("_T")


def closed_traits_for_kind(entity_kind: str) -> tuple[str, ...]:
    """Return the closed set of neutral visual traits allowed for an entity kind."""
    if entity_kind == "person":
        return PERSON_CLOSED_TRAITS
    if entity_kind == "location":
        return LOCATION_CLOSED_TRAITS
    return ()


def compact_repair_summary(
    reason: str,
    details: Sequence[str] = (),
    *,
    max_bytes: int = MAX_REPAIR_SUMMARY_BYTES,
) -> str:
    """Build a compact repair instruction capped strictly in UTF-8 bytes without raw assistant echo."""
    clean_reason = " ".join(reason.split()).strip() or "исправь формат и точность ответа"
    if clean_reason.startswith("Исправь ответ:"):
        parts = [clean_reason]
    else:
        parts = [f"Исправь ответ: {clean_reason}."]
    for detail in details:
        text = " ".join(detail.split()).strip()
        if text:
            parts.append(text)
    combined = " ".join(parts)
    encoded = combined.encode("utf-8")
    if len(encoded) <= max_bytes:
        return combined
    # Leave room for ellipsis and cut on a valid UTF-8 codepoint boundary.
    limit = max(16, max_bytes - 3)
    cut = encoded[:limit]
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    trimmed = cut.decode("utf-8", errors="ignore").rstrip()
    return f"{trimmed}..."


@dataclass(frozen=True, slots=True)
class QuoteResolutionResult:
    """Result of deterministic quote resolution against a canonical `DocumentBlock`."""

    status: Literal["resolved", "ambiguous", "missing"]
    anchor: SourceAnchor | None = None
    position_key: str | None = None


def _all_occurrences(haystack: str, needle: str) -> list[int]:
    if not haystack or not needle:
        return []
    found: list[int] = []
    pos = 0
    while True:
        idx = haystack.find(needle, pos)
        if idx < 0:
            break
        found.append(idx)
        pos = idx + 1
    return found


def resolve_quote_anchor(
    block: DocumentBlock,
    quote: str,
    *,
    section_order: int,
    block_order: int,
) -> QuoteResolutionResult:
    """Resolve a model quote deterministically against `source_text_for_model(block)` -> `block.text`.

    - Rejects empty quotes (`missing`).
    - Rejects repeated occurrences in the same block (`ambiguous`) so the repair turn asks for
      a longer unique substring.
    - Handles stripped footnote markers accurately via `_analysis_visible_origins(block)`.
    """
    if not quote or not quote.strip():
        return QuoteResolutionResult(status="missing")

    visible, origins = _analysis_visible_origins(block)
    visible_matches = _all_occurrences(visible, quote)
    if len(visible_matches) > 1:
        return QuoteResolutionResult(status="ambiguous")
    if len(visible_matches) == 1:
        v_start = visible_matches[0]
        v_end = v_start + len(quote)
        if v_end <= len(origins) and v_start < v_end:
            s_start = origins[v_start]
            s_end = origins[v_end - 1] + 1
            if 0 <= s_start < s_end <= len(block.text):
                actual_quote = block.text[s_start:s_end]
                anchor = SourceAnchor(
                    block_id=block.id,
                    start=s_start,
                    end=s_end,
                    quote=actual_quote,
                )
                return QuoteResolutionResult(
                    status="resolved",
                    anchor=anchor,
                    position_key=format_position_key(section_order, block_order, s_start),
                )

    raw_matches = _all_occurrences(block.text, quote)
    if len(raw_matches) > 1:
        return QuoteResolutionResult(status="ambiguous")
    if len(raw_matches) == 1:
        s_start = raw_matches[0]
        s_end = s_start + len(quote)
        anchor = SourceAnchor(
            block_id=block.id,
            start=s_start,
            end=s_end,
            quote=block.text[s_start:s_end],
        )
        return QuoteResolutionResult(
            status="resolved",
            anchor=anchor,
            position_key=format_position_key(section_order, block_order, s_start),
        )

    return QuoteResolutionResult(status="missing")


def pack_items_by_wire_budget(
    items: Sequence[_T],
    *,
    build_request: Callable[[Sequence[_T]], LLMRequest[Any]],
    sizer: RequestWireSizer,
    target_bytes: int,
) -> tuple[tuple[_T, ...], ...]:
    """Deterministically pack ordered items into pages whose exact wire size fits `target_bytes`.

    If a single item exceeds `target_bytes` on its own, it is emitted as a singleton page so the
    caller can fail cleanly with `atomic_input_too_large` before any HTTP call.
    """
    if not items:
        return ()
    pages: list[tuple[_T, ...]] = []
    current: list[_T] = []
    for item in items:
        candidate = [*current, item]
        size = sizer.measure(build_request(candidate)).size_bytes
        if size <= target_bytes:
            current = candidate
            continue
        if current:
            pages.append(tuple(current))
            current = [item]
        else:
            pages.append((item,))
            current = []
    if current:
        pages.append(tuple(current))
    return tuple(pages)


_STAGE_TO_KIND: Mapping[IllustrationScopeStage, LLMResponseKind] = {
    IllustrationScopeStage.EXTRACT: LLMResponseKind.ILLUSTRATION_EXTRACT,
    IllustrationScopeStage.ALIAS: LLMResponseKind.ILLUSTRATION_ALIAS,
    IllustrationScopeStage.DESIGN: LLMResponseKind.ILLUSTRATION_DESIGN,
    IllustrationScopeStage.FACT_SCOPE: LLMResponseKind.ILLUSTRATION_FACT_SCOPE,
    IllustrationScopeStage.CANDIDATES: LLMResponseKind.ILLUSTRATION_CANDIDATES,
    IllustrationScopeStage.SELECTION: LLMResponseKind.ILLUSTRATION_SELECTION,
    IllustrationScopeStage.PROMPT: LLMResponseKind.ILLUSTRATION_PROMPT,
}

_STAGE_INSTRUCTIONS: Mapping[IllustrationScopeStage, str] = {
    IllustrationScopeStage.EXTRACT: (
        "Ты извлекаешь визуальные упоминания персонажей и мест и цитатные факты для справочника иллюстраций книги. "
        "Используй только приведённые блоки текста. Цитаты копируй дословно и достаточно длинно, чтобы они были уникальны внутри блока. "
        "Описания на русском и английском должны отражать только подтверждённые текстом сведения. "
        "Ставь back_applicable=true только для неизменных черт (например, цвет глаз), которые не раскрывают будущий спойлер. "
        "Если материал слишком велик для полного извлечения без пропусков, верни complete=false. "
        "При complete=true перечисли все обработанные source_unit_id в covered_source_unit_ids."
    ),
    IllustrationScopeStage.ALIAS: (
        "Ты проверяешь пары упоминаний одного типа (person или location) из справочника иллюстраций книги. "
        "Одинаковое имя само по себе НЕ доказывает тождество. "
        "Для каждой пары верни exact pair_id и вердикт same_entity, different или unknown с опорой на переданные факты."
    ),
    IllustrationScopeStage.DESIGN: (
        "Ты дополняешь недостающие нейтральные визуальные черты образа из закрытого списка допустимых черт для типа сущности. "
        "Не противоречь явным фактам из текста и не повторяй одинаковую черту дважды. "
        "Каждый домысел фиксируй в assumption_ru; описания давай на русском (description_ru) и английском (description_en)."
    ),
    IllustrationScopeStage.FACT_SCOPE: (
        "Ты отбираешь факты сущности, применимые на момент конкретного события книги (учитывая хронологию и постоянные черты). "
        "Противоречивые или устаревшие эпизодические факты помещай в excluded_fact_ids, актуальные — в applicable_fact_ids (не более 32). "
        "Сформируй компактные описания образа на русском и английском."
    ),
    IllustrationScopeStage.CANDIDATES: (
        "Ты выделяешь до двух сильных визуальных сюжетных событий из переданной сцены главы. "
        "Каждое событие должно иметь непустое обоснование значимости significance_ru, дословные уникальные quote_anchors в рамках переданных блоков "
        "и опционально insert_after_block_id в пределах этой сцены. "
        "Если подходящих событий нет, верни пустой список и непустую причину zero_reason_ru. "
        "При complete=true перечисли все source_unit_id в covered_source_unit_ids."
    ),
    IllustrationScopeStage.SELECTION: (
        "Ты отбираешь наиболее сильные и визуально разнообразные моменты для иллюстраций главы в пределах указанного лимита group_cap. "
        "Выбирай только переданные candidate_id без повторов и выдуманных ID. "
        "Если ни один кандидат не достоин иллюстрации, верни пустой selected_candidate_ids и непустую zero_reason_ru."
    ),
    IllustrationScopeStage.PROMPT: (
        "Ты составляешь англоязычную композицию одного кадра (scene_composition_en) для монохромной книжной иллюстрации тушью. "
        "Опирайся строго на переданные применимые факты и черты образов, не вводи несогласованные детали и перечисли использованные used_fact_ids и used_design_keys."
    ),
}


def compact_json(payload: Mapping[str, Any]) -> str:
    """Serialize structured prompt input deterministically with minimal whitespace."""
    return json.dumps(thaw(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_name_token(value: str) -> str:
    """Normalize an entity name for deterministic alias blocking."""
    cleaned = re.sub(r"[^\w\s-]+", " ", value.casefold())
    return " ".join(cleaned.split())


class IllustrationContext:
    """Builds bounded `LLMRequest` objects for all illustration stages."""

    @staticmethod
    def build_for_kind(
        *,
        kind: LLMResponseKind,
        stage: IllustrationScopeStage,
        input_payload: Mapping[str, Any],
        book_id: str,
        conditions_id: str | None,
        policy: IllustrationPolicy,
        repair_summary: str | None = None,
        scope_id: str | None = None,
        attempt_id: str | None = None,
        generation_number: int | None = None,
        assigned_model: str | None = None,
        ordered_block_ids: tuple[str, ...] = (),
    ) -> LLMRequest[Any]:
        instructions = _STAGE_INSTRUCTIONS[stage]
        input_text = compact_json(input_payload)
        repair_turns: tuple[tuple[str | None, str], ...] = ()
        if repair_summary is not None:
            repair_turns = ((None, compact_repair_summary(repair_summary)),)

        call_stage = (
            LLMCallStage.ILLUSTRATION_REFERENCE
            if stage in _REFERENCE_STAGES
            else LLMCallStage.ILLUSTRATION_CHAPTER
        )
        dedup_blocks = tuple(dict.fromkeys(ordered_block_ids))
        context: CallContext | None = None
        if attempt_id is not None and conditions_id is not None and scope_id is not None and assigned_model is not None:
            prompt_hash = canonical_sha(
                {
                    "kind": kind.value,
                    "instructions": instructions,
                    "input_text": input_text,
                    "repair_turns": list(repair_turns),
                }
            )
            context = CallContext(
                conditions_id=conditions_id,
                stage=call_stage.value,
                scope_id=scope_id,
                ordered_block_ids=dedup_blocks,
                attempt_id=attempt_id,
                generation=0 if generation_number is None else generation_number,
                assigned_model=assigned_model,
                prompt_hash=prompt_hash,
                policy_version=policy.policy_version,
            )

        return LLMRequest(
            kind=kind,
            instructions=instructions,
            input_text=input_text,
            book_id=book_id,
            stage=call_stage,
            contract_version=1,
            repair_turns=repair_turns,
            call_context=context,
            scope_id=scope_id,
            ordered_block_ids=dedup_blocks,
            max_request_bytes=policy.max_request_bytes,
        )

    @classmethod
    def build(
        cls,
        scope: IllustrationScope,
        snapshot: IllustrationSourceSnapshot,
        settings: IllustrationSettings,
        *,
        repair_summary: str | None = None,
        attempt_id: str | None = None,
        generation_number: int | None = None,
        assigned_model: str | None = None,
    ) -> LLMRequest[Any]:
        """Build a bounded `LLMRequest` directly from a persisted `IllustrationScope`."""
        stage = scope.stage
        kind = _STAGE_TO_KIND[stage]
        raw_input = scope.summary.get("request_input")
        if not isinstance(raw_input, Mapping):
            raw_input = {"scope_key": scope.scope_key}
        block_ids_raw = scope.summary.get("ordered_block_ids", ())
        ordered_block_ids = (
            tuple(str(b) for b in block_ids_raw)
            if isinstance(block_ids_raw, (list, tuple))
            else ()
        )
        return cls.build_for_kind(
            kind=kind,
            stage=stage,
            input_payload=raw_input,
            book_id=snapshot.book_id,
            conditions_id=snapshot.conditions_id,
            policy=settings.policy,
            repair_summary=repair_summary,
            scope_id=scope.id,
            attempt_id=attempt_id,
            generation_number=generation_number,
            assigned_model=assigned_model,
            ordered_block_ids=ordered_block_ids,
        )


__all__ = [
    "ALL_DESIGN_TRAITS",
    "LOCATION_CLOSED_TRAITS",
    "MAX_REPAIR_SUMMARY_BYTES",
    "PERSON_CLOSED_TRAITS",
    "REPAIR_RESERVE_BYTES",
    "TIMELESS_POSITION_KEY",
    "IllustrationContext",
    "QuoteResolutionResult",
    "closed_traits_for_kind",
    "compact_json",
    "compact_repair_summary",
    "format_position_key",
    "normalize_name_token",
    "pack_items_by_wire_budget",
    "resolve_quote_anchor",
]
