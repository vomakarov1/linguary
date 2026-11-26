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
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMContextCapacityError,
    LLMProviderError,
    LLMRequest,
    LLMRequestTooLargeError,
    LLMResponseFormatError,
    LLMResponseKind,
    LLMTransportError,
)
from linguary.domain import AnalysisPassKind, TermStatus, glossary_key
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import SourceAnchor, canonical_sha
from linguary.domain.document import DocumentBlock
from linguary.domain.illustrations import (
    TIMELESS_POSITION_KEY,
    IllustrationPolicy,
    IllustrationRegistryKind,
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

_CONTEXT_LIMIT_RE = re.compile(
    r"context[_ ]length|maximum context|too many tokens|prompt is too long|request too large"
    r"|(?::\s*413\b|\bhttp\s+413\b|\bstatus\s+413\b)",
    re.IGNORECASE,
)
_INCOMPATIBLE_ROUTE_RE = re.compile(
    r"(?::\s*404\b|\bhttp\s+404\b|\bstatus\s+404\b"
    r"|no endpoints found|unsupported parameter|unknown parameter"
    r"|unrecognized request argument|does not support"
    r"|model not found|not a valid model)",
    re.IGNORECASE,
)


def is_context_limit_error(error: Exception) -> bool:
    """Return True if the error indicates prompt/context length overflow."""
    if isinstance(error, (LLMRequestTooLargeError, LLMContextCapacityError)):
        return True
    if isinstance(error, LLMProviderError):
        status = getattr(error, "status_code", None)
        if status == 413:
            return True
        if (status in (400, 422) or status is None) and _CONTEXT_LIMIT_RE.search(str(error)):
            return True
    return False


def is_truncated_error(error: Exception) -> bool:
    """Return True if the error represents a provider-truncated completion."""
    if not isinstance(error, LLMResponseFormatError):
        return False
    if getattr(error, "truncated", False):
        return True
    violation = getattr(error, "violation", None)
    return violation is not None and getattr(violation, "rule_code", None) == "provider_truncated"


def is_transient_error(error: Exception) -> bool:
    """Return True if the error is a transient transport or provider error."""
    if isinstance(error, LLMTransportError):
        return True
    if isinstance(error, LLMProviderError) and not isinstance(error, LLMRequestTooLargeError):
        status = getattr(error, "status_code", None)
        return status in (408, 429) or (isinstance(status, int) and status >= 500)
    return False


def is_incompatible_route_error(error: Exception) -> bool:
    """Return True if the provider rejected the route/model/parameter permanently."""
    if isinstance(error, LLMConfigurationError):
        return True
    if (
        isinstance(error, LLMProviderError)
        and not is_context_limit_error(error)
        and not is_transient_error(error)
    ):
        if getattr(error, "status_code", None) == 404:
            return True
        if _INCOMPATIBLE_ROUTE_RE.search(str(error)):
            return True
    return False


def closed_traits_for_kind(entity_kind: str) -> tuple[str, ...]:
    """Return the closed set of neutral visual traits allowed for an entity kind."""
    if entity_kind == "person":
        return PERSON_CLOSED_TRAITS
    if entity_kind == "location":
        return LOCATION_CLOSED_TRAITS
    return ()


def compact_safe_detail(text: str, *, max_bytes: int = 256) -> str:
    """Trim a safe validator detail message strictly in UTF-8 bytes without any repair prefix."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    encoded = cleaned.encode("utf-8")
    if len(encoded) <= max_bytes:
        return cleaned
    if max_bytes <= 3:
        cut = encoded[: max(0, max_bytes)]
        while cut and (cut[-1] & 0xC0) == 0x80:
            cut = cut[:-1]
        return cut.decode("utf-8", errors="ignore")
    limit = max(1, max_bytes - 3)
    cut = encoded[:limit]
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    trimmed = cut.decode("utf-8", errors="ignore").rstrip()
    return f"{trimmed}..."


_REPAIR_PREAMBLE_PREFIXES: tuple[str, ...] = (
    "Исправь ответ:",
    "Исправь ошибку и верни ПОЛНЫЙ JSON-ответ",
)
_REPAIR_PREAMBLE_DEFAULT = (
    "Исправь ошибку и верни ПОЛНЫЙ JSON-ответ целиком для всех элементов запроса "
    "(не частичный фрагмент): "
)
_REPAIR_PREAMBLE_COVERED_UNITS = (
    "Исправь ошибку и верни ПОЛНЫЙ JSON-ответ для ВСЕХ переданных source_unit_id "
    "в covered_source_unit_ids (не частичный фрагмент): "
)


def compact_repair_summary(
    reason: str,
    details: Sequence[str] = (),
    *,
    max_bytes: int = MAX_REPAIR_SUMMARY_BYTES,
    stage: IllustrationScopeStage | None = None,
) -> str:
    """Build a compact repair instruction capped strictly in UTF-8 bytes without raw assistant echo."""
    clean_reason = " ".join(reason.split()).strip() or "исправь формат и точность ответа"
    if clean_reason.startswith(_REPAIR_PREAMBLE_PREFIXES):
        parts = [clean_reason]
    else:
        preamble = (
            _REPAIR_PREAMBLE_COVERED_UNITS
            if stage in {IllustrationScopeStage.EXTRACT, IllustrationScopeStage.CANDIDATES}
            else _REPAIR_PREAMBLE_DEFAULT
        )
        suffix = "" if clean_reason.endswith((".", "!", "?")) else "."
        parts = [f"{preamble}{clean_reason}{suffix}"]
    for detail in details:
        text = " ".join(detail.split()).strip()
        if text:
            parts.append(text)
    combined = " ".join(parts)
    encoded = combined.encode("utf-8")
    if len(encoded) <= max_bytes:
        return combined
    if max_bytes <= 3:
        cut = encoded[: max(0, max_bytes)]
        while cut and (cut[-1] & 0xC0) == 0x80:
            cut = cut[:-1]
        return cut.decode("utf-8", errors="ignore")
    # Leave room for ellipsis and cut on a valid UTF-8 codepoint boundary.
    limit = max(1, max_bytes - 3)
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


_QUOTE_CHAR_MAP: Mapping[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "`": "'",
    "\u00b4": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2011": "-",
}


def _normalize_quote_with_spans(text: str) -> tuple[str, list[int], list[int]]:
    """Normalize typographic punctuation/whitespace and map each emitted char to source spans."""
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for idx, ch in enumerate(text):
        if ch.isspace():
            if chars and chars[-1] == " ":
                ends[-1] = idx + 1
            else:
                chars.append(" ")
                starts.append(idx)
                ends.append(idx + 1)
        elif ch == "\u2026":
            for _ in range(3):
                chars.append(".")
                starts.append(idx)
                ends.append(idx + 1)
        else:
            chars.append(_QUOTE_CHAR_MAP.get(ch, ch))
            starts.append(idx)
            ends.append(idx + 1)
    lo = 1 if chars and chars[0] == " " else 0
    hi = len(chars) - 1 if len(chars) > lo and chars[-1] == " " else len(chars)
    return "".join(chars[lo:hi]), starts[lo:hi], ends[lo:hi]


def _normalized_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Return unique `(start, end)` spans in `haystack` matching `needle` after typographic folding."""
    h_norm, h_starts, h_ends = _normalize_quote_with_spans(haystack)
    n_norm, _, _ = _normalize_quote_with_spans(needle)
    if not n_norm or not h_norm:
        return []
    n_len = len(n_norm)
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pos in _all_occurrences(h_norm, n_norm):
        end_pos = pos + n_len
        if pos > 0 and h_starts[pos] == h_starts[pos - 1]:
            continue
        if end_pos < len(h_ends) and h_ends[end_pos - 1] == h_ends[end_pos]:
            continue
        span = (h_starts[pos], h_ends[end_pos - 1])
        if span not in seen:
            seen.add(span)
            spans.append(span)
    return spans


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
    - Falls back to deterministic typographic/whitespace folding when exact matching misses,
      always storing the verbatim `block.text[s_start:s_end]` substring in `SourceAnchor.quote`.
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

    norm_visible = _normalized_occurrences(visible, quote)
    if len(norm_visible) > 1:
        return QuoteResolutionResult(status="ambiguous")
    if len(norm_visible) == 1:
        v_start, v_end = norm_visible[0]
        if v_end <= len(origins) and v_start < v_end:
            s_start = origins[v_start]
            s_end = origins[v_end - 1] + 1
            if 0 <= s_start < s_end <= len(block.text):
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

    norm_raw = _normalized_occurrences(block.text, quote)
    if len(norm_raw) > 1:
        return QuoteResolutionResult(status="ambiguous")
    if len(norm_raw) == 1:
        s_start, s_end = norm_raw[0]
        if 0 <= s_start < s_end <= len(block.text):
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
    max_items: int | None = None,
) -> tuple[tuple[_T, ...], ...]:
    """Deterministically pack ordered items into pages whose exact wire size fits `target_bytes`.

    If `max_items` is provided, each page contains at most `max_items` items.
    If a single item exceeds `target_bytes` on its own, it is emitted as a singleton page so the
    caller can fail cleanly with `atomic_input_too_large` before any HTTP call.
    """
    if max_items is not None and (type(max_items) is not int or max_items <= 0):
        raise ValueError("max_items должен быть положительным целым")
    if not items:
        return ()
    pages: list[tuple[_T, ...]] = []
    current: list[_T] = []
    for item in items:
        if max_items is not None and len(current) >= max_items:
            pages.append(tuple(current))
            current = []
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
        "Каждый элемент facts обязан ссылаться в subject_mention_id на существующий mention_id из массива mentions этого же ответа "
        "(только person или location; не создавай факты для предметов без субъекта в mentions). "
        "Для entity_kind=\"location\" используй точное различающее имя места из оригинального текста "
        "(улица — именем улицы; дом — адресом или названием дома; комната — названием комнаты; не подписывай конкретный дом голым названием улицы). "
        "Если в ближайшем тексте описательное или родственное обозначение прямо соотносится с названным персонажем или местом "
        "(например, «сын X по имени Y»), укажи это имя в alias_of_hint. "
        "Ставь back_applicable=true только для неизменных черт (например, цвет глаз), которые не раскрывают будущий спойлер. "
        "Если материал слишком велик для полного извлечения без пропусков, верни complete=false. "
        "При complete=true перечисли все обработанные source_unit_id в covered_source_unit_ids."
    ),
    IllustrationScopeStage.ALIAS: (
        "Ты проверяешь пары упоминаний одного типа (person или location) из справочника иллюстраций книги. "
        "Одинаковое имя само по себе НЕ доказывает тождество. "
        "Для каждой пары верни exact pair_id и вердикт same_entity, different или unknown с опорой на переданные факты. "
        "Учитывай совпадение glossary_concept_id, подсказки alias_of_hint, цитаты и контекст сцены. "
        "Если два упоминания явно обозначают одного и того же человека (например, имя и обращение с фамилией/титулом, "
        "такие как Petunia и Mrs Dursley, при подтверждении текстом/цитатой/глоссарием), ставь same_entity. "
        "При этом общая фамилия с разными титулами или полом (Mr Dursley и Mrs Dursley) — different. "
        "Для entity_kind=\"location\": повторные упоминания одного и того же места, улицы или дома под тем же именем "
        "или явным вариантом — same_entity, даже если в разных сценах упомянуты разные детали окружения; "
        "контейнер и его часть (улица и конкретный дом на ней, дом и отдельная комната или дверь внутри него) — строго different. "
        "unknown ставь только когда доказательств действительно недостаточно."
    ),
    IllustrationScopeStage.DESIGN: (
        "Ты задаёшь ТОЛЬКО вневременные базовые визуальные черты образа из закрытого списка допустимых черт для типа сущности, "
        "отсутствующие в переданных facts (нейтральный силуэт, базовые пропорции или архитектуру, тонально-графический характер для monochrome_ink). "
        "Опирайся СТРОГО на переданные facts и canonical_name. "
        "ЗАПРЕЩЕНО использовать любые внешние знания о книге, серии, фильмах, экранизациях или событиях будущих глав "
        "(не подставляй канонический возраст, школьную форму, предметы, цвет глаз/волос или детали из других частей произведения, если их нет в facts). "
        "Категорически ЗАПРЕЩЕНО закреплять в DESIGN эпизодические и изменяемые состояния (temporal_scope=\"episodic\" или \"state_change\"): "
        "сон или закрытые глаза, временную позу или выражение лица, разовую одежду или обёртку конкретного эпизода, переменное освещение, "
        "случайных животных или предметы в локации, а также преходящий возрастной снимок, если его постоянство не подтверждено permanent-фактом. "
        "Если у сущности есть факты temporal_scope=\"state_change\" (смена облика, формы или маскировки), не фиксируй в DESIGN "
        "анатомические или видовые черты только одной из форм; оставляй только нейтральные признаки (например, palette_value) или traits: []. "
        "Не противоречь переданным facts и не дублируй черты, уже описанные в facts; если деталей в facts достаточно или переданы только временные факты "
        "(non_permanent_context_only=true), допустимо вернуть только palette_value или traits: []. "
        "Если style == \"monochrome_ink\", описывай palette_value ТОЛЬКО тональными характеристиками чёрно-белой туши "
        "(контраст, светлые/средние/тёмные тона, плотность штриховки, чёткие силуэты) и НЕ используй названия хроматических цветов и оттенков "
        "(зелёный, красный, синий, бежевый, коричневый, жёлтый и т. п.) ни в palette_value, ни в любых других чертах дизайна. "
        "В assumption_ru записывай только реальное допущение дизайнера; если допущения нет, оставляй assumption_ru: \"\" "
        "(никогда не пиши заглушки вроде «Допущений нет»). Описания давай на русском (description_ru) и английском (description_en). "
        "В русскоязычных полях используй переводы имён и терминов из glossary_terms_ru (если термин там присутствует)."
    ),
    IllustrationScopeStage.FACT_SCOPE: (
        "Ты отбираешь факты сущности, применимые на момент конкретного события книги (учитывая хронологию и постоянные черты). "
        "Противоречивые или устаревшие эпизодические факты помещай в excluded_fact_ids, актуальные — в applicable_fact_ids (не более 32). "
        "Сформируй компактные описания ТОЛЬКО визуального облика сущности на момент события на русском (compact_description_ru) и английском (compact_description_en), "
        "без пересказа диалога и сюжетных действий. "
        "Если style == \"monochrome_ink\", переводи цветовые признаки исходных фактов в светотень, тональность и фактуру "
        "(например, светлый/тёмный/насыщенно-тёмный тон, контраст, металлический блеск, яркое свечение) и НЕ используй названия хроматических цветов "
        "(blue, purple, emerald, orange, red, green и т. п.) в compact_description_en и compact_description_ru. "
        "Не пиши заглушки в assumptions_ru (оставляй [], если допущений нет). "
        "В русскоязычных полях используй переводы имён и терминов из glossary_terms_ru (если термин там присутствует)."
    ),
    IllustrationScopeStage.CANDIDATES: (
        "Ты выделяешь до двух сильных визуальных сюжетных событий из переданной сцены главы. "
        "Каждое событие должно иметь непустое обоснование значимости significance_ru, дословные уникальные quote_anchors в рамках переданных блоков "
        "и опционально insert_after_block_id в пределах этой сцены. "
        "В participant_names указывай ТОЛЬКО персонажей, физически видимых или изображаемых в кадре "
        "(включая показанных на фотографиях или портретах, если кадр о них); запрещено включать отсутствующих персонажей, "
        "упомянутых через отрицание («нет следов X»), находящихся за кадром или упоминаемых только в речи или мыслях. "
        "Имена в participant_names и location_names всегда записывай на языке оригинала книги "
        "(никогда не переводи на язык перевода, даже если scene.location или participants даны на языке перевода). "
        "При совпадении используй точные имена из переданных списков known_characters и known_locations. "
        "Обязательно заполняй location_names, если место действия известно из текста сцены. "
        "Если подходящих событий нет, верни пустой список и непустую причину zero_reason_ru. "
        "При complete=true перечисли все source_unit_id в covered_source_unit_ids. "
        "В русскоязычных полях используй переводы имён и терминов из glossary_terms_ru (если термин там присутствует)."
    ),
    IllustrationScopeStage.SELECTION: (
        "Ты отбираешь наиболее сильные и визуально разнообразные моменты для иллюстраций главы в пределах указанного лимита group_cap. "
        "Выбирай только переданные candidate_id без повторов и выдуманных ID. "
        "Для каждого candidate_id из selected_candidate_ids обязательно верни объект {candidate_id, reason_ru} в selection_reasons_ru "
        "с кратким обоснованием выбора на русском языке. "
        "Если ни один кандидат не достоин иллюстрации, верни пустой selected_candidate_ids, пустой selection_reasons_ru: [] и непустую zero_reason_ru."
    ),
    IllustrationScopeStage.PROMPT: (
        "Ты составляешь англоязычную композицию одного кадра (scene_composition_en) для монохромной книжной иллюстрации тушью. "
        "Опирайся строго на переданные применимые факты, описания образов и только те available_design_traits, которые действительно нужны кадру, "
        "не вводи несогласованные детали и перечисли фактически использованные used_fact_ids и used_design_keys. "
        "Поле characters[].description_en — единственный авторитетный облик персонажа на момент события; "
        "запрещено выбирать used_design_keys, противоречащие characters[].description_en или event_ru. "
        "Обязательно укажи в depicted_entity_ids все entity_id из переданных characters и locations, которые реально изображены в scene_composition_en. "
        "Запрещено описывать в scene_composition_en отсутствующих персонажей через отрицания (например, «no boy with a scar», «without X»); "
        "описывай только реально изображаемое в кадре. "
        "Пиши scene_composition_en строго на английском языке (без кириллицы). "
        "При style == \"monochrome_ink\" не используй названия хроматических цветов в scene_composition_en, выражая визуальные различия через светлоту, контраст и фактуру. "
        "Точно соблюдай позу, действие и физически возможную точку зрения камеры из event_ru: "
        "для комнатных сцен не показывай снаружи фасад здания без выхода наружу; "
        "для уличных и наружных сцен (улица, крыльцо, двор, сад, дорога) изображай именно наружную обстановку, а не интерьер; "
        "не меняй положение тела персонажа вопреки описанию события. "
        "В русскоязычных полях используй переводы имён и терминов из glossary_terms_ru (если термин там присутствует)."
    ),
}


def compact_json(payload: Mapping[str, Any]) -> str:
    """Serialize structured prompt input deterministically with minimal whitespace."""
    return json.dumps(thaw(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_name_token(value: str) -> str:
    """Normalize an entity name for deterministic alias blocking."""
    cleaned = re.sub(r"[^\w\s-]+", " ", value.casefold())
    return " ".join(cleaned.split())


_GLOSSARY_STATUS_PRIORITY: Mapping[TermStatus, int] = {
    TermStatus.LOCKED: 0,
    TermStatus.APPROVED: 1,
    TermStatus.SUGGESTED: 2,
}


def _collect_book_glossary_terms_ru(
    tx: Any,
    book_id: str,
    *,
    generation_id: str | None = None,
    max_section_order: int | None = None,
    allowed_keys: frozenset[str] | None = None,
    limit: int = 40,
) -> dict[str, str]:
    """Collect a compact, deterministic, spoiler-free Russian glossary map for illustration prompts."""
    if limit <= 0:
        return {}
    glossary_repo = getattr(tx, "glossary", None)
    if glossary_repo is None:
        return {}
    list_glossary = getattr(glossary_repo, "list_for_book", None)
    if not callable(list_glossary):
        return {}
    concepts = list_glossary(book_id)
    if not concepts:
        return {}

    seen_keys: set[str] | None = None
    if max_section_order is not None:
        seen_keys = set()
        section_order_by_id: dict[str, int] = {}
        docs_repo = getattr(tx, "documents", None)
        get_doc = getattr(docs_repo, "get", None) if docs_repo is not None else None
        if callable(get_doc):
            document = get_doc(book_id)
            if document is not None:
                for sec in getattr(document, "ordered_sections", ()):
                    section_order_by_id[str(sec.id)] = int(sec.order)

        analyses_repo = getattr(tx, "analyses", None)
        if analyses_repo is not None:
            list_analyses = getattr(analyses_repo, "list_for_book", None)
            if callable(list_analyses):
                for analysis in list_analyses(book_id):
                    sec_ord = section_order_by_id.get(str(analysis.section_id))
                    if sec_ord is None or sec_ord > max_section_order:
                        continue
                    for cand in getattr(analysis, "glossary_candidates", ()):
                        src = str(getattr(cand, "source_text", "") or "").strip()
                        if src:
                            seen_keys.add(glossary_key(src))
                    for scene in getattr(analysis, "scenes", ()):
                        for part in getattr(scene, "participants", ()):
                            if part and str(part).strip():
                                seen_keys.add(glossary_key(str(part)))
                        loc = getattr(scene, "location", None)
                        if loc and str(loc).strip():
                            seen_keys.add(glossary_key(str(loc)))

            list_passes = getattr(analyses_repo, "list_passes_for_book", None)
            if callable(list_passes):
                for saved_pass in list_passes(
                    book_id, pass_kind=AnalysisPassKind.NARRATIVE, scope_key=""
                ):
                    sec_ord = section_order_by_id.get(str(saved_pass.section_id))
                    if sec_ord is None or sec_ord > max_section_order:
                        continue
                    raw_payload = getattr(saved_pass, "payload", None)
                    raw_resp = (
                        raw_payload.get("response") if isinstance(raw_payload, Mapping) else None
                    )
                    if not isinstance(raw_resp, Mapping):
                        continue
                    for cand in raw_resp.get("glossary_candidates", ()):
                        if isinstance(cand, Mapping):
                            src = str(cand.get("source_text") or "").strip()
                            if src:
                                seen_keys.add(glossary_key(src))
                    for scene in raw_resp.get("scenes", ()):
                        if isinstance(scene, Mapping):
                            for part in scene.get("participants", ()):
                                if part and str(part).strip():
                                    seen_keys.add(glossary_key(str(part)))
                            loc = scene.get("location")
                            if loc and str(loc).strip():
                                seen_keys.add(glossary_key(str(loc)))

        illustrations_repo = getattr(tx, "illustrations", None)
        if illustrations_repo is not None:
            eff_gen_id = generation_id
            if eff_gen_id is None:
                get_head = getattr(illustrations_repo, "get_head", None)
                if callable(get_head):
                    head = get_head(book_id)
                    eff_gen_id = getattr(head, "active_generation_id", None) if head is not None else None
            query_items = getattr(illustrations_repo, "query_registry_items", None)
            if eff_gen_id and callable(query_items):
                chapter_end_pos = format_position_key(max_section_order, 999_999, 99_999_999)
                offset = 0
                while True:
                    page, total = query_items(
                        eff_gen_id,
                        kind=IllustrationRegistryKind.ENTITY,
                        offset=offset,
                        limit=200,
                    )
                    for item in page:
                        pay = item.raw_payload()
                        raw_positions = pay.get("alias_first_positions")
                        if isinstance(raw_positions, Mapping) and raw_positions:
                            for alias_name, pos in raw_positions.items():
                                if (
                                    alias_name is not None
                                    and str(alias_name).strip()
                                    and isinstance(pos, str)
                                    and pos
                                    and pos <= chapter_end_pos
                                ):
                                    seen_keys.add(glossary_key(str(alias_name)))
                        else:
                            first_pos = str(pay.get("first_position_key") or "")
                            raw_orders = pay.get("section_orders")
                            visible = bool(first_pos and first_pos <= chapter_end_pos) or (
                                not first_pos
                                and isinstance(raw_orders, (list, tuple))
                                and any(type(o) is int and o <= max_section_order for o in raw_orders)
                            )
                            if visible:
                                canon = str(pay.get("canonical_name") or "").strip()
                                if canon:
                                    seen_keys.add(glossary_key(canon))
                                raw_aliases = pay.get("aliases")
                                if isinstance(raw_aliases, (list, tuple)):
                                    for a in raw_aliases:
                                        if a is not None and str(a).strip():
                                            seen_keys.add(glossary_key(str(a)))
                    offset += len(page)
                    if offset >= total or not page:
                        break

    candidates: list[tuple[int, str, str, str, str]] = []
    for concept in concepts:
        src_text = str(getattr(concept, "canonical_source_text", "") or "").strip()
        if not src_text:
            continue
        ckey = glossary_key(src_text)
        if not ckey:
            continue
        if allowed_keys is not None and ckey not in allowed_keys:
            continue
        if seen_keys is not None and ckey not in seen_keys:
            continue
        ru_var = concept.variant_for("ru")
        if ru_var is None or not ru_var.text.strip():
            continue
        status = getattr(concept, "status", TermStatus.SUGGESTED)
        priority = _GLOSSARY_STATUS_PRIORITY.get(status, 2)
        candidates.append((priority, src_text.casefold(), src_text, ckey, ru_var.text.strip()))

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    result: dict[str, str] = {}
    used_keys: set[str] = set()
    for _, _, src_text, ckey, ru_text in candidates:
        if ckey in used_keys:
            continue
        used_keys.add(ckey)
        result[src_text] = ru_text
        if len(result) >= limit:
            break
    return result


def fit_glossary_terms_to_wire_budget(
    glossary_terms_ru: Mapping[str, str],
    *,
    build_request: Callable[[dict[str, str]], LLMRequest[Any]],
    sizer: RequestWireSizer,
    target_bytes: int,
) -> dict[str, str]:
    """Return the longest prefix of `glossary_terms_ru` whose request fits `target_bytes` (down to `{}`)."""
    items = [
        (str(k).strip(), str(v).strip())
        for k, v in glossary_terms_ru.items()
        if k is not None and str(k).strip() and v is not None and str(v).strip()
    ]
    if not items:
        return {}
    full = dict(items)
    if sizer.measure(build_request(full)).size_bytes <= target_bytes:
        return full
    if sizer.measure(build_request({})).size_bytes > target_bytes:
        return {}
    lo = 0
    hi = len(items)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sizer.measure(build_request(dict(items[:mid]))).size_bytes <= target_bytes:
            lo = mid
        else:
            hi = mid - 1
    return dict(items[:lo])


collect_book_glossary_terms_ru = _collect_book_glossary_terms_ru


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
            repair_turns = ((None, compact_repair_summary(repair_summary, stage=stage)),)

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
    "_collect_book_glossary_terms_ru",
    "closed_traits_for_kind",
    "collect_book_glossary_terms_ru",
    "compact_json",
    "compact_repair_summary",
    "compact_safe_detail",
    "fit_glossary_terms_to_wire_budget",
    "format_position_key",
    "is_context_limit_error",
    "is_incompatible_route_error",
    "is_transient_error",
    "is_truncated_error",
    "normalize_name_token",
    "pack_items_by_wire_budget",
    "resolve_quote_anchor",
]
