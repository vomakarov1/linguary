"""Validation and deterministic prompt composition for chapter illustration moments (`P3.T2`)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from linguary.application.context_builder import textual_blocks
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.document import DocumentSection
from linguary.domain.illustrations import (
    ILLUSTRATION_STYLE_MONOCHROME_INK,
    ILLUSTRATION_SUPPORTED_STYLES,
    IllustrationReasonCode,
    IllustrationValidationError,
)

if TYPE_CHECKING:
    from linguary.application.illustration_source import IllustrationSourceSnapshot

MONOCHROME_INK_DIRECTIVE = (
    "monochrome ink, clear silhouettes, separated foreground and background, "
    "moderate hatching, no color-dependent information, no essential small lettering"
)
_BLOCK_ALIAS_RE = re.compile(r"^b\d+$")
_SEGMENT_ID_RE = re.compile(r"#s\d+$")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def has_cyrillic(text: str) -> bool:
    """Return True if the string contains any Cyrillic character."""
    return bool(_CYRILLIC_RE.search(text))


@dataclass(frozen=True, slots=True)
class IllustrationAppearance:
    """Applicable character or location appearance bound to a moment."""

    entity_id: str
    name_ru: str
    description_ru: str
    description_en: str
    fact_ids: tuple[str, ...] = ()
    assumptions_ru: tuple[str, ...] = ()
    name_en: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entity_id": self.entity_id,
            "name_ru": self.name_ru,
            "description_ru": self.description_ru,
            "description_en": self.description_en,
            "fact_ids": list(self.fact_ids),
            "assumptions_ru": list(self.assumptions_ru),
        }
        if self.name_en is not None:
            payload["name_en"] = self.name_en
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> IllustrationAppearance:
        raw_name_en = raw.get("name_en")
        name_en = str(raw_name_en).strip() or None if raw_name_en is not None else None
        return cls(
            entity_id=str(raw["entity_id"]),
            name_ru=str(raw["name_ru"]),
            description_ru=str(raw["description_ru"]),
            description_en=str(raw["description_en"]),
            fact_ids=tuple(str(fid) for fid in raw.get("fact_ids", ())),
            assumptions_ru=tuple(str(a) for a in raw.get("assumptions_ru", ())),
            name_en=name_en,
        )


@dataclass(frozen=True, slots=True)
class ValidatedMoment:
    """Validated chapter illustration moment ready for persistence and UI rendering."""

    id: str
    scene_id: str
    title_ru: str
    reason_ru: str
    event_ru: str
    prompt_en: str
    insert_after_block_id: str
    anchors: tuple[SourceAnchor, ...]
    character_descriptions: tuple[IllustrationAppearance, ...] = ()
    location_descriptions: tuple[IllustrationAppearance, ...] = ()
    assumptions_ru: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scene_id": self.scene_id,
            "title_ru": self.title_ru,
            "reason_ru": self.reason_ru,
            "event_ru": self.event_ru,
            "prompt_en": self.prompt_en,
            "insert_after_block_id": self.insert_after_block_id,
            "anchors": [
                {
                    "block_id": a.block_id,
                    "start": a.start,
                    "end": a.end,
                    "quote": a.quote,
                }
                for a in self.anchors
            ],
            "character_descriptions": [c.to_dict() for c in self.character_descriptions],
            "location_descriptions": [loc.to_dict() for loc in self.location_descriptions],
            "assumptions_ru": list(self.assumptions_ru),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ValidatedMoment:
        return cls(
            id=str(raw["id"]),
            scene_id=str(raw["scene_id"]),
            title_ru=str(raw["title_ru"]),
            reason_ru=str(raw["reason_ru"]),
            event_ru=str(raw["event_ru"]),
            prompt_en=str(raw["prompt_en"]),
            insert_after_block_id=str(raw["insert_after_block_id"]),
            anchors=tuple(
                SourceAnchor(
                    block_id=str(a["block_id"]),
                    start=int(a["start"]),
                    end=int(a["end"]),
                    quote=str(a["quote"]),
                )
                for a in raw.get("anchors", ())
            ),
            character_descriptions=tuple(
                IllustrationAppearance.from_dict(c) for c in raw.get("character_descriptions", ())
            ),
            location_descriptions=tuple(
                IllustrationAppearance.from_dict(loc) for loc in raw.get("location_descriptions", ())
            ),
            assumptions_ru=tuple(str(a) for a in raw.get("assumptions_ru", ())),
        )


@dataclass(frozen=True, slots=True)
class ChapterIllustrationPlan:
    """Canonical accepted illustration plan for one chapter."""

    section_id: str
    generation_id: str
    style: str = ILLUSTRATION_STYLE_MONOCHROME_INK
    moments: tuple[ValidatedMoment, ...] = ()
    zero_reason_ru: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "section_id": self.section_id,
            "generation_id": self.generation_id,
            "style": self.style,
            "moments": [m.to_dict() for m in self.moments],
            "zero_reason_ru": self.zero_reason_ru,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ChapterIllustrationPlan:
        return cls(
            section_id=str(raw["section_id"]),
            generation_id=str(raw["generation_id"]),
            style=str(raw.get("style", ILLUSTRATION_STYLE_MONOCHROME_INK)),
            moments=tuple(ValidatedMoment.from_dict(m) for m in raw.get("moments", ())),
            zero_reason_ru=(
                None if raw.get("zero_reason_ru") is None else str(raw["zero_reason_ru"])
            ),
        )


def compose_monochrome_prompt(
    scene_composition_en: str,
    appearances: Sequence[IllustrationAppearance | Mapping[str, Any]],
    style_profile: str = ILLUSTRATION_STYLE_MONOCHROME_INK,
) -> str:
    """Deterministically compose the final English monochrome ink prompt.

    Always embeds the actual textual `description_en` of every applicable appearance
    (IDs alone are never accepted) plus the canonical monochrome ink directive.
    """
    comp = " ".join(scene_composition_en.split()).strip()
    if not comp:
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "scene_composition_en не может быть пустым",
        )
    if style_profile not in ILLUSTRATION_SUPPORTED_STYLES:
        raise IllustrationValidationError(
            IllustrationReasonCode.INVALID_REQUEST,
            f"неподдерживаемый профиль стиля: {style_profile}",
        )

    appearance_lines: list[str] = []
    for item in appearances:
        if isinstance(item, IllustrationAppearance):
            raw_name = item.name_en if item.name_en is not None else item.name_ru
            desc_en = item.description_en
        else:
            raw_name = item.get("name_en") or item.get("name_ru", "")
            desc_en = str(item.get("description_en", "")).strip()
        name = str(raw_name).strip()
        desc_clean = " ".join(desc_en.split()).strip()
        if not desc_clean:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                "применимый образ обязан содержать непустое description_en",
            )
        if name and not has_cyrillic(name):
            appearance_lines.append(f"{name}: {desc_clean}")
        else:
            appearance_lines.append(desc_clean)

    parts = [comp]
    if appearance_lines:
        parts.append("Visual details: " + "; ".join(appearance_lines) + ".")
    parts.append(f"Style: {MONOCHROME_INK_DIRECTIVE}.")
    return " ".join(parts)


def resolve_insert_after_block_id(
    requested_block_id: str | None,
    *,
    anchors: Sequence[SourceAnchor],
    scene_end_block_id: str,
    section: DocumentSection,
) -> str:
    """Resolve and validate `insert_after_block_id` strictly within `[last_evidence, scene_end]`."""
    ordered_textual = textual_blocks(section)
    pos_by_id = {b.id: idx for idx, b in enumerate(ordered_textual)}
    if not anchors:
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "момент обязан содержать хотя бы одну опору",
        )
    for anchor in anchors:
        if anchor.block_id not in pos_by_id:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"опора ссылается на блок вне текстовых блоков главы: {anchor.block_id}",
            )

    last_evidence_pos = max(pos_by_id[a.block_id] for a in anchors)
    last_evidence_block_id = ordered_textual[last_evidence_pos].id

    if requested_block_id is None or not requested_block_id.strip():
        return last_evidence_block_id

    candidate = requested_block_id.strip()
    if _SEGMENT_ID_RE.search(candidate):
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            f"insert_after_block_id не может указывать на учебный сегмент: {candidate}",
        )
    if candidate not in pos_by_id:
        if _BLOCK_ALIAS_RE.match(candidate):
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"insert_after_block_id не может использовать локальный псевдоним блока: {candidate}",
            )
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            f"insert_after_block_id вне текстовых блоков главы: {candidate}",
        )

    scene_end_pos = pos_by_id.get(scene_end_block_id, len(ordered_textual) - 1)
    cand_pos = pos_by_id[candidate]
    if not (last_evidence_pos <= cand_pos <= scene_end_pos):
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "insert_after_block_id должен лежать в диапазоне [last_evidence_block, scene.end_block_id]",
        )
    return candidate


def validate_moment(
    moment: ValidatedMoment | Mapping[str, Any],
    *,
    snapshot: IllustrationSourceSnapshot,
    section: DocumentSection,
    scene_start_block_id: str,
    scene_end_block_id: str,
    registry: Mapping[str, Mapping[str, Any]],
    event_position_key: str,
) -> ValidatedMoment:
    """Strictly validate an assembled chapter moment before atomic acceptance (`P3.T2`)."""
    parsed = moment if isinstance(moment, ValidatedMoment) else ValidatedMoment.from_dict(moment)
    if not parsed.id.strip() or not parsed.title_ru.strip() or not parsed.reason_ru.strip():
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "момент должен содержать непустые id, title_ru и reason_ru",
        )
    if not parsed.event_ru.strip() or not parsed.prompt_en.strip():
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "момент должен содержать непустые event_ru и prompt_en",
        )

    # Verify scene belongs to this chapter in snapshot (if chapter has explicit NARRATIVE scenes).
    entry = next((e for e in snapshot.entries if e.section_id == section.id), None)
    if entry is None or entry.skipped:
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            f"глава {section.id} не входит в иллюстрируемый состав книги",
        )
    if entry.scenes:
        known_scene_ids = {s.id for s in entry.scenes} | {
            f"{section.id}:scene:{idx}" for idx in range(len(entry.scenes))
        }
        if parsed.scene_id not in known_scene_ids:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"сцена {parsed.scene_id} не принадлежит главе {section.id}",
            )

    ordered_textual = textual_blocks(section)
    textual_by_id = {b.id: b.text for b in ordered_textual}
    pos_by_id = {b.id: idx for idx, b in enumerate(ordered_textual)}
    scene_start_pos = pos_by_id.get(scene_start_block_id, 0)
    scene_end_pos = pos_by_id.get(scene_end_block_id, len(ordered_textual) - 1)

    if not parsed.anchors:
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "момент обязан содержать проверенные опоры",
        )

    for anchor in parsed.anchors:
        if _SEGMENT_ID_RE.search(anchor.block_id):
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"опора ссылается на учебный сегмент: {anchor.block_id}",
            )
        if anchor.block_id not in textual_by_id:
            if _BLOCK_ALIAS_RE.match(anchor.block_id):
                raise IllustrationValidationError(
                    IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                    f"опора использует локальный псевдоним блока {anchor.block_id}",
                )
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"опора ссылается на чужой или нетекстовый блок {anchor.block_id}",
            )
        try:
            anchor.validate(textual_by_id)
        except Exception as err:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"некорректная опора блока {anchor.block_id}: {err}",
            ) from err
        apos = pos_by_id[anchor.block_id]
        if not (scene_start_pos <= apos <= scene_end_pos):
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"опора {anchor.block_id} выходит за границы сцены",
            )

    resolved_insert = resolve_insert_after_block_id(
        parsed.insert_after_block_id,
        anchors=parsed.anchors,
        scene_end_block_id=scene_end_block_id,
        section=section,
    )

    if MONOCHROME_INK_DIRECTIVE not in parsed.prompt_en:
        raise IllustrationValidationError(
            IllustrationReasonCode.MODEL_OUTPUT_INVALID,
            "prompt_en должен содержать канонический блок требований монохромного профиля",
        )

    all_appearances = (*parsed.character_descriptions, *parsed.location_descriptions)
    prompt_lower = parsed.prompt_en.casefold()
    for app in all_appearances:
        desc_clean = " ".join(app.description_en.split()).strip()
        if not desc_clean:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"у образа {app.entity_id} отсутствует description_en",
            )
        if desc_clean.casefold() not in prompt_lower:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                f"prompt_en не содержит текстовое описание образа {app.entity_id} (одних ID недостаточно)",
            )
        for fid in app.fact_ids:
            fact = registry.get(fid)
            if fact is None:
                raise IllustrationValidationError(
                    IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                    f"неизвестный fact_id {fid} в образе {app.entity_id}",
                )
            fpos = str(fact.get("position_key", ""))
            is_back = bool(fact.get("is_back_applicable", False))
            if fpos > event_position_key and not is_back:
                raise IllustrationValidationError(
                    IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                    f"спойлерный будущий факт {fid} ({fpos} > {event_position_key}) запрещён",
                )

    return ValidatedMoment(
        id=parsed.id,
        scene_id=parsed.scene_id,
        title_ru=parsed.title_ru,
        reason_ru=parsed.reason_ru,
        event_ru=parsed.event_ru,
        prompt_en=parsed.prompt_en,
        insert_after_block_id=resolved_insert,
        anchors=parsed.anchors,
        character_descriptions=parsed.character_descriptions,
        location_descriptions=parsed.location_descriptions,
        assumptions_ru=parsed.assumptions_ru,
    )


__all__ = [
    "MONOCHROME_INK_DIRECTIVE",
    "ChapterIllustrationPlan",
    "IllustrationAppearance",
    "ValidatedMoment",
    "compose_monochrome_prompt",
    "has_cyrillic",
    "resolve_insert_after_block_id",
    "validate_moment",
]
