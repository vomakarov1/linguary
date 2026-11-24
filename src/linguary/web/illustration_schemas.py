"""Pydantic HTTP schemas and strict input parsers for optional illustrations (`API-01/02/04`, `DTO-01/02`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlencode

from pydantic import BaseModel

from linguary.application.illustration_view import (
    IllustrationRegistryView,
    IllustrationView,
)
from linguary.domain.illustrations import (
    ILLUSTRATION_STYLE_MONOCHROME_INK,
    IllustrationReasonCode,
    IllustrationRegistryKind,
    IllustrationSettings,
    IllustrationValidationError,
    normalize_model_name,
)
from linguary.web.routes.result import block_anchor

_MAX_RETURN_STR_LENGTH = 200
_VALID_REGISTRY_KINDS = frozenset(member.value for member in IllustrationRegistryKind)


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _format_source_url(book_id: str, block_id: str | None) -> str | None:
    if not block_id:
        return None
    return f"/books/{quote(book_id, safe='')}/result#{block_anchor(block_id)}"


def _invalid(message: str = "некорректные параметры запроса") -> IllustrationValidationError:
    return IllustrationValidationError(IllustrationReasonCode.INVALID_REQUEST, message)


def _strict_int(value: Any, *, min_value: int, max_value: int | None = None, from_str: bool = False) -> int:
    if isinstance(value, bool):
        raise _invalid("логическое значение недопустимо там, где ожидается целое число")
    if isinstance(value, int):
        parsed = value
    elif from_str and isinstance(value, str):
        raw = value.strip()
        if not raw or (raw[0] == "-" and not raw[1:].isdigit()) or (raw[0] != "-" and not raw.isdigit()):
            raise _invalid("ожидалось целое число")
        parsed = int(raw)
    else:
        raise _invalid("ожидалось целое число")
    if parsed < min_value or (max_value is not None and parsed > max_value):
        raise _invalid("числовое значение вне допустимого диапазона")
    return parsed


def _optional_clean_str(value: Any, *, max_len: int = _MAX_RETURN_STR_LENGTH) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid("ожидалась строка")
    if _has_control_chars(value):
        raise _invalid("строка содержит управляющие символы")
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > max_len:
        raise _invalid("строка превышает допустимую длину")
    return stripped


def _parse_form_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, str):
        raise _invalid("некорректное логическое значение формы")
    lowered = value.strip().lower()
    if lowered in {"true", "1", "on"}:
        return True
    if lowered in {"false", "0", ""}:
        return False
    raise _invalid("некорректное логическое значение формы")


def _extract_unique_query_params(
    multi_items: Sequence[tuple[str, str]],
    *,
    allowed: frozenset[str],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for key, val in multi_items:
        if key not in allowed or key in parsed:
            raise _invalid(f"неизвестный или повторяющийся параметр {key!r}")
        parsed[key] = val
    return parsed


@dataclass(frozen=True, slots=True)
class IllustrationGetQuery:
    offset: int = 0
    limit: int = 10
    section_id: str | None = None
    generation_id: str | None = None
    registry_offset: int = 0
    registry_limit: int = 20
    registry_kind: IllustrationRegistryKind | None = None


@dataclass(frozen=True, slots=True)
class IllustrationRegistryQuery:
    generation_id: str | None = None
    entity_id: str | None = None
    kind: IllustrationRegistryKind | None = None
    offset: int = 0
    limit: int = 50


@dataclass(frozen=True, slots=True)
class IllustrationReturnParams:
    offset: int | None = None
    limit: int | None = None
    section_id: str | None = None
    generation_id: str | None = None

    def redirect_location(self, book_id: str) -> str:
        base = f"/books/{quote(book_id, safe='')}/illustrations"
        params: list[tuple[str, str]] = []
        if self.offset is not None:
            params.append(("offset", str(self.offset)))
        if self.limit is not None:
            params.append(("limit", str(self.limit)))
        if self.section_id is not None:
            params.append(("section_id", self.section_id))
        if self.generation_id is not None:
            params.append(("generation_id", self.generation_id))
        if not params:
            return base
        return f"{base}?{urlencode(params)}"


@dataclass(frozen=True, slots=True)
class ParsedSettingsInput:
    expected_revision: int
    settings: IllustrationSettings
    return_params: IllustrationReturnParams


@dataclass(frozen=True, slots=True)
class ParsedCommandInput:
    expected_revision: int
    expected_generation_id: str | None
    section_id: str | None
    force_new_generation: bool
    return_params: IllustrationReturnParams


_API01_ALLOWED_QUERY = frozenset(
    {
        "offset",
        "limit",
        "section_id",
        "generation_id",
        "registry_offset",
        "registry_limit",
        "registry_kind",
    }
)
_API04_ALLOWED_QUERY = frozenset(
    {
        "generation_id",
        "entity_id",
        "kind",
        "offset",
        "limit",
    }
)
_SETTINGS_JSON_KEYS = frozenset(
    {
        "expected_revision",
        "enabled",
        "max_per_chapter",
        "style",
        "model_reference",
        "model_prompts",
    }
)
_RETURN_FORM_KEYS = frozenset(
    {
        "return_offset",
        "return_limit",
        "return_section_id",
        "return_generation_id",
    }
)
_SETTINGS_FORM_KEYS = _SETTINGS_JSON_KEYS | _RETURN_FORM_KEYS
_PREPARE_JSON_KEYS = frozenset(
    {
        "expected_revision",
        "expected_generation_id",
        "section_id",
        "force_new_generation",
    }
)
_PREPARE_FORM_KEYS = _PREPARE_JSON_KEYS | _RETURN_FORM_KEYS
_RETRY_JSON_KEYS = frozenset(
    {
        "expected_revision",
        "expected_generation_id",
        "section_id",
    }
)
_RETRY_FORM_KEYS = _RETRY_JSON_KEYS | _RETURN_FORM_KEYS


def parse_api01_query(multi_items: Sequence[tuple[str, str]]) -> IllustrationGetQuery:
    raw = _extract_unique_query_params(multi_items, allowed=_API01_ALLOWED_QUERY)
    offset = _strict_int(raw["offset"], min_value=0, from_str=True) if "offset" in raw else 0
    limit = (
        _strict_int(raw["limit"], min_value=1, max_value=50, from_str=True)
        if "limit" in raw
        else 10
    )
    registry_offset = (
        _strict_int(raw["registry_offset"], min_value=0, from_str=True)
        if "registry_offset" in raw
        else 0
    )
    registry_limit = (
        _strict_int(raw["registry_limit"], min_value=1, max_value=100, from_str=True)
        if "registry_limit" in raw
        else 20
    )
    section_id = _optional_clean_str(raw.get("section_id"))
    generation_id = _optional_clean_str(raw.get("generation_id"))
    kind_raw = _optional_clean_str(raw.get("registry_kind"))
    registry_kind: IllustrationRegistryKind | None = None
    if kind_raw is not None:
        if kind_raw not in _VALID_REGISTRY_KINDS:
            raise _invalid("неизвестный вид записи справочника")
        registry_kind = IllustrationRegistryKind(kind_raw)
    return IllustrationGetQuery(
        offset=offset,
        limit=limit,
        section_id=section_id,
        generation_id=generation_id,
        registry_offset=registry_offset,
        registry_limit=registry_limit,
        registry_kind=registry_kind,
    )


def parse_api04_query(multi_items: Sequence[tuple[str, str]]) -> IllustrationRegistryQuery:
    raw = _extract_unique_query_params(multi_items, allowed=_API04_ALLOWED_QUERY)
    offset = _strict_int(raw["offset"], min_value=0, from_str=True) if "offset" in raw else 0
    limit = (
        _strict_int(raw["limit"], min_value=1, max_value=100, from_str=True)
        if "limit" in raw
        else 50
    )
    generation_id = _optional_clean_str(raw.get("generation_id"))
    entity_id = _optional_clean_str(raw.get("entity_id"))
    kind_raw = _optional_clean_str(raw.get("kind"))
    kind: IllustrationRegistryKind | None = None
    if kind_raw is not None:
        if kind_raw not in _VALID_REGISTRY_KINDS:
            raise _invalid("неизвестный вид записи справочника")
        kind = IllustrationRegistryKind(kind_raw)
    return IllustrationRegistryQuery(
        generation_id=generation_id,
        entity_id=entity_id,
        kind=kind,
        offset=offset,
        limit=limit,
    )


def parse_return_params_from_form(raw: Mapping[str, Any]) -> IllustrationReturnParams:
    offset_raw = _optional_clean_str(raw.get("return_offset"))
    limit_raw = _optional_clean_str(raw.get("return_limit"))
    return IllustrationReturnParams(
        offset=_strict_int(offset_raw, min_value=0, from_str=True) if offset_raw is not None else None,
        limit=(
            _strict_int(limit_raw, min_value=1, max_value=50, from_str=True)
            if limit_raw is not None
            else None
        ),
        section_id=_optional_clean_str(raw.get("return_section_id")),
        generation_id=_optional_clean_str(raw.get("return_generation_id")),
    )


def parse_settings_json(raw: Any) -> ParsedSettingsInput:
    if not isinstance(raw, Mapping):
        raise _invalid("ожидался JSON-объект")
    unknown = set(raw.keys()) - _SETTINGS_JSON_KEYS
    if unknown:
        raise _invalid("запрос содержит неизвестные или запрещённые поля")
    if "expected_revision" not in raw or "enabled" not in raw:
        raise _invalid("обязательные поля expected_revision и enabled отсутствуют")
    expected_revision = _strict_int(raw["expected_revision"], min_value=0)
    enabled = raw["enabled"]
    if type(enabled) is not bool:
        raise _invalid("поле enabled должно быть логическим")
    max_per_chapter = (
        _strict_int(raw["max_per_chapter"], min_value=1, max_value=3)
        if "max_per_chapter" in raw
        else 3
    )
    style = raw.get("style", ILLUSTRATION_STYLE_MONOCHROME_INK)
    if not isinstance(style, str):
        raise _invalid("поле style должно быть строкой")
    model_reference = normalize_model_name(raw.get("model_reference"))
    model_prompts = normalize_model_name(raw.get("model_prompts"))
    settings = IllustrationSettings(
        enabled=enabled,
        max_per_chapter=max_per_chapter,
        style=style,
        model_reference=model_reference,
        model_prompts=model_prompts,
    )
    return ParsedSettingsInput(
        expected_revision=expected_revision,
        settings=settings,
        return_params=IllustrationReturnParams(),
    )


def parse_settings_form(multi_items: Sequence[tuple[str, Any]]) -> ParsedSettingsInput:
    raw: dict[str, Any] = {}
    for key, value in multi_items:
        if key not in _SETTINGS_FORM_KEYS or key in raw:
            raise _invalid(f"неизвестное или повторяющееся поле формы {key!r}")
        if not isinstance(value, str):
            raise _invalid(f"поле формы {key!r} должно быть строкой")
        raw[key] = value

    # Validate return_* before any DB mutation.
    return_params = parse_return_params_from_form(raw)

    if "expected_revision" not in raw:
        raise _invalid("отсутствует expected_revision")
    expected_revision = _strict_int(raw["expected_revision"], min_value=0, from_str=True)
    enabled = _parse_form_bool(raw.get("enabled"), default=False)
    max_per_chapter = (
        _strict_int(raw["max_per_chapter"], min_value=1, max_value=3, from_str=True)
        if "max_per_chapter" in raw and raw["max_per_chapter"].strip() != ""
        else 3
    )
    style = raw.get("style", ILLUSTRATION_STYLE_MONOCHROME_INK)
    if style.strip() == "":
        style = ILLUSTRATION_STYLE_MONOCHROME_INK
    model_reference = normalize_model_name(raw.get("model_reference"))
    model_prompts = normalize_model_name(raw.get("model_prompts"))
    settings = IllustrationSettings(
        enabled=enabled,
        max_per_chapter=max_per_chapter,
        style=style,
        model_reference=model_reference,
        model_prompts=model_prompts,
    )
    return ParsedSettingsInput(
        expected_revision=expected_revision,
        settings=settings,
        return_params=return_params,
    )


def parse_command_json(raw: Any, *, allow_force_new: bool, require_generation: bool) -> ParsedCommandInput:
    if not isinstance(raw, Mapping):
        raise _invalid("ожидался JSON-объект")
    allowed = _PREPARE_JSON_KEYS if allow_force_new else _RETRY_JSON_KEYS
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise _invalid("запрос содержит неизвестные или запрещённые поля")
    if "expected_revision" not in raw:
        raise _invalid("отсутствует обязательное поле expected_revision")
    expected_revision = _strict_int(raw["expected_revision"], min_value=0)
    expected_generation_id = _optional_clean_str(raw.get("expected_generation_id"))
    if require_generation and expected_generation_id is None:
        raise _invalid("отсутствует обязательное поле expected_generation_id")
    section_id = _optional_clean_str(raw.get("section_id"))
    force_new = False
    if allow_force_new and "force_new_generation" in raw:
        raw_force = raw["force_new_generation"]
        if type(raw_force) is not bool:
            raise _invalid("поле force_new_generation должно быть логическим")
        force_new = raw_force
    return ParsedCommandInput(
        expected_revision=expected_revision,
        expected_generation_id=expected_generation_id,
        section_id=section_id,
        force_new_generation=force_new,
        return_params=IllustrationReturnParams(),
    )


def parse_command_form(
    multi_items: Sequence[tuple[str, Any]],
    *,
    allow_force_new: bool,
    require_generation: bool,
) -> ParsedCommandInput:
    allowed = _PREPARE_FORM_KEYS if allow_force_new else _RETRY_FORM_KEYS
    raw: dict[str, Any] = {}
    for key, value in multi_items:
        if key not in allowed or key in raw:
            raise _invalid(f"неизвестное или повторяющееся поле формы {key!r}")
        if not isinstance(value, str):
            raise _invalid(f"поле формы {key!r} должно быть строкой")
        raw[key] = value

    # Syntactically validate return_* before any DB mutation!
    return_params = parse_return_params_from_form(raw)

    if "expected_revision" not in raw:
        raise _invalid("отсутствует expected_revision")
    expected_revision = _strict_int(raw["expected_revision"], min_value=0, from_str=True)
    expected_generation_id = _optional_clean_str(raw.get("expected_generation_id"))
    if require_generation and expected_generation_id is None:
        raise _invalid("отсутствует обязательное поле expected_generation_id")
    section_id = _optional_clean_str(raw.get("section_id"))
    force_new = (
        _parse_form_bool(raw.get("force_new_generation"), default=False)
        if allow_force_new
        else False
    )
    return ParsedCommandInput(
        expected_revision=expected_revision,
        expected_generation_id=expected_generation_id,
        section_id=section_id,
        force_new_generation=force_new,
        return_params=return_params,
    )


class IllustrationSettingsResponse(BaseModel):
    enabled: bool
    max_per_chapter: int
    style: str
    model_reference: str | None
    model_prompts: str | None


class IllustrationDefaultsHintResponse(BaseModel):
    provider_ready: bool
    default_model: str | None
    max_calls_per_generation: int


class IllustrationSourceProgressResponse(BaseModel):
    expected_chapters: int
    ready_chapters: int
    skipped_sections: int
    blocking_section_id: str | None


class IllustrationRegistryProgressResponse(BaseModel):
    state: str
    reason_code: str | None
    completed_scopes: int
    total_scopes: int | None
    entities: int
    facts: int


class IllustrationChapterHeaderResponse(BaseModel):
    section_id: str
    order: int
    title: str
    state: str
    reason_code: str | None
    moment_count: int | None
    source_url: str | None


class IllustrationPaginationResponse(BaseModel):
    offset: int
    limit: int
    total: int
    next_offset: int | None
    previous_offset: int | None


class IllustrationAnchorResponse(BaseModel):
    block_id: str
    start: int
    end: int
    quote: str


class IllustrationAppearanceResponse(BaseModel):
    entity_id: str
    name_ru: str
    description_ru: str
    description_en: str
    fact_ids: list[str]
    assumptions_ru: list[str]


class IllustrationMomentResponse(BaseModel):
    id: str
    scene_id: str
    title_ru: str
    reason_ru: str
    event_ru: str
    prompt_en: str
    insert_after_block_id: str
    source_url: str | None
    anchors: list[IllustrationAnchorResponse]
    character_descriptions: list[IllustrationAppearanceResponse]
    location_descriptions: list[IllustrationAppearanceResponse]
    assumptions_ru: list[str]


class IllustrationSelectedChapterResponse(BaseModel):
    section_id: str
    title: str
    state: str
    reason_code: str | None
    zero_reason_ru: str | None
    moments: list[IllustrationMomentResponse]


class IllustrationHistoryItemResponse(BaseModel):
    generation_id: str
    generation_number: int
    state: str
    settings_revision: int
    source_sha256: str
    created_at: datetime


class IllustrationCostBucketResponse(BaseModel):
    calls: int
    total_tokens: int | None
    cost_usd: str | None
    estimate_usd: None = None


class IllustrationCostsResponse(BaseModel):
    reference: IllustrationCostBucketResponse
    chapters: IllustrationCostBucketResponse
    total: IllustrationCostBucketResponse


class IllustrationViewResponse(BaseModel):
    """Canonical HTTP/CLI `DTO-01` representation."""

    book_id: str
    book_title: str
    enabled: bool
    settings_revision: int
    settings: IllustrationSettingsResponse
    defaults_hint: IllustrationDefaultsHintResponse
    active_generation_id: str | None
    viewed_generation_id: str | None
    is_current: bool
    is_history: bool
    state: str
    reason_code: str | None
    source_progress: IllustrationSourceProgressResponse
    registry_progress: IllustrationRegistryProgressResponse
    allowed_actions: list[str]
    chapter_headers: list[IllustrationChapterHeaderResponse]
    pagination: IllustrationPaginationResponse
    selected_chapter: IllustrationSelectedChapterResponse | None
    history: list[IllustrationHistoryItemResponse]
    costs: IllustrationCostsResponse

    @classmethod
    def of_view(cls, view: IllustrationView) -> IllustrationViewResponse:
        selected: IllustrationSelectedChapterResponse | None = None
        if view.selected_chapter is not None:
            selected = IllustrationSelectedChapterResponse(
                section_id=view.selected_chapter.section_id,
                title=view.selected_chapter.title,
                state=view.selected_chapter.state,
                reason_code=view.selected_chapter.reason_code,
                zero_reason_ru=view.selected_chapter.zero_reason_ru,
                moments=[
                    IllustrationMomentResponse(
                        id=m.id,
                        scene_id=m.scene_id,
                        title_ru=m.title_ru,
                        reason_ru=m.reason_ru,
                        event_ru=m.event_ru,
                        prompt_en=m.prompt_en,
                        insert_after_block_id=m.insert_after_block_id,
                        source_url=_format_source_url(view.book_id, m.source_block_id),
                        anchors=[
                            IllustrationAnchorResponse(
                                block_id=a.block_id,
                                start=a.start,
                                end=a.end,
                                quote=a.quote,
                            )
                            for a in m.anchors
                        ],
                        character_descriptions=[
                            IllustrationAppearanceResponse(
                                entity_id=c.entity_id,
                                name_ru=c.name_ru,
                                description_ru=c.description_ru,
                                description_en=c.description_en,
                                fact_ids=list(c.fact_ids),
                                assumptions_ru=list(c.assumptions_ru),
                            )
                            for c in m.character_descriptions
                        ],
                        location_descriptions=[
                            IllustrationAppearanceResponse(
                                entity_id=loc.entity_id,
                                name_ru=loc.name_ru,
                                description_ru=loc.description_ru,
                                description_en=loc.description_en,
                                fact_ids=list(loc.fact_ids),
                                assumptions_ru=list(loc.assumptions_ru),
                            )
                            for loc in m.location_descriptions
                        ],
                        assumptions_ru=list(m.assumptions_ru),
                    )
                    for m in view.selected_chapter.moments
                ],
            )
        return cls(
            book_id=view.book_id,
            book_title=view.book_title,
            enabled=view.enabled,
            settings_revision=view.settings_revision,
            settings=IllustrationSettingsResponse(
                enabled=view.settings.enabled,
                max_per_chapter=view.settings.max_per_chapter,
                style=view.settings.style,
                model_reference=view.settings.model_reference,
                model_prompts=view.settings.model_prompts,
            ),
            defaults_hint=IllustrationDefaultsHintResponse(
                provider_ready=view.defaults_hint.provider_ready,
                default_model=view.defaults_hint.default_model,
                max_calls_per_generation=view.defaults_hint.max_calls_per_generation,
            ),
            active_generation_id=view.active_generation_id,
            viewed_generation_id=view.viewed_generation_id,
            is_current=view.is_current,
            is_history=view.is_history,
            state=view.state,
            reason_code=view.reason_code,
            source_progress=IllustrationSourceProgressResponse(
                expected_chapters=view.source_progress.expected_chapters,
                ready_chapters=view.source_progress.ready_chapters,
                skipped_sections=view.source_progress.skipped_sections,
                blocking_section_id=view.source_progress.blocking_section_id,
            ),
            registry_progress=IllustrationRegistryProgressResponse(
                state=view.registry_progress.state,
                reason_code=view.registry_progress.reason_code,
                completed_scopes=view.registry_progress.completed_scopes,
                total_scopes=view.registry_progress.total_scopes,
                entities=view.registry_progress.entities,
                facts=view.registry_progress.facts,
            ),
            allowed_actions=list(view.allowed_actions),
            chapter_headers=[
                IllustrationChapterHeaderResponse(
                    section_id=h.section_id,
                    order=h.order,
                    title=h.title,
                    state=h.state,
                    reason_code=h.reason_code,
                    moment_count=h.moment_count,
                    source_url=_format_source_url(view.book_id, h.source_block_id),
                )
                for h in view.chapter_headers
            ],
            pagination=IllustrationPaginationResponse(
                offset=view.pagination.offset,
                limit=view.pagination.limit,
                total=view.pagination.total,
                next_offset=view.pagination.next_offset,
                previous_offset=view.pagination.previous_offset,
            ),
            selected_chapter=selected,
            history=[
                IllustrationHistoryItemResponse(
                    generation_id=h.generation_id,
                    generation_number=h.generation_number,
                    state=h.state,
                    settings_revision=h.settings_revision,
                    source_sha256=h.source_sha256,
                    created_at=h.created_at,
                )
                for h in view.history
            ],
            costs=IllustrationCostsResponse(
                reference=IllustrationCostBucketResponse(
                    calls=view.costs.reference.calls,
                    total_tokens=view.costs.reference.total_tokens,
                    cost_usd=view.costs.reference.cost_usd,
                    estimate_usd=None,
                ),
                chapters=IllustrationCostBucketResponse(
                    calls=view.costs.chapters.calls,
                    total_tokens=view.costs.chapters.total_tokens,
                    cost_usd=view.costs.chapters.cost_usd,
                    estimate_usd=None,
                ),
                total=IllustrationCostBucketResponse(
                    calls=view.costs.total.calls,
                    total_tokens=view.costs.total.total_tokens,
                    cost_usd=view.costs.total.cost_usd,
                    estimate_usd=None,
                ),
            ),
        )


class IllustrationRegistryItemResponse(BaseModel):
    kind: str
    item_key: str
    entity_id: str
    position_key: str
    summary_ru: str
    details: dict[str, Any]


class IllustrationRegistryResponse(BaseModel):
    """Canonical HTTP/CLI `DTO-02` representation."""

    book_id: str
    generation_id: str | None
    is_current: bool
    items: list[IllustrationRegistryItemResponse]
    pagination: IllustrationPaginationResponse

    @classmethod
    def of_view(cls, view: IllustrationRegistryView) -> IllustrationRegistryResponse:
        return cls(
            book_id=view.book_id,
            generation_id=view.generation_id,
            is_current=view.is_current,
            items=[
                IllustrationRegistryItemResponse(
                    kind=item.kind,
                    item_key=item.item_key,
                    entity_id=item.entity_id,
                    position_key=item.position_key,
                    summary_ru=item.summary_ru,
                    details=dict(item.details),
                )
                for item in view.items
            ],
            pagination=IllustrationPaginationResponse(
                offset=view.pagination.offset,
                limit=view.pagination.limit,
                total=view.pagination.total,
                next_offset=view.pagination.next_offset,
                previous_offset=view.pagination.previous_offset,
            ),
        )


__all__ = [
    "IllustrationGetQuery",
    "IllustrationRegistryQuery",
    "IllustrationRegistryResponse",
    "IllustrationReturnParams",
    "IllustrationViewResponse",
    "ParsedCommandInput",
    "ParsedSettingsInput",
    "parse_api01_query",
    "parse_api04_query",
    "parse_command_form",
    "parse_command_json",
    "parse_return_params_from_form",
    "parse_settings_form",
    "parse_settings_json",
]
