"""Validation and deterministic prompt composition for chapter illustration moments (`P3.T2`)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
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
_SPEECH_BUBBLE_RE = re.compile(
    r"\b(?:speech|thought|text|comic)\s+(?:balloon|bubble|box|caption)s?\b",
    re.IGNORECASE,
)
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
_MONOCHROME_CHROMATIC_EN_RE = re.compile(
    r"\b(red|crimson|scarlet|green|emerald|blue|azure|cobalt|yellow|amber|gold|golden|"
    r"orange|purple|violet|pink|magenta|teal|turquoise|beige|brown|chestnut|auburn|"
    r"maroon|olive|khaki|indigo|cyan|lavender|burgundy|sepia|cream|ochre)\b",
    re.IGNORECASE,
)
_MONOCHROME_CHROMATIC_RU_RE = re.compile(
    r"\b(красн\w*|зелён\w*|зелен\w*|син\w*|голуб\w*|жёлт\w*|желт\w*|оранжев\w*|"
    r"фиолетов\w*|розов\w*|бежев\w*|коричнев\w*|рыж\w*|изумруд\w*|багров\w*|"
    r"бирюзов\w*|золот\w*|пурпур\w*|бордов\w*|лилов\w*|"
    r"кремов\w*|охр(?:а|ы|е|у|ой|ист\w*|ов\w*))\b",
    re.IGNORECASE,
)
_MONOCHROME_OBJECT_PHRASES_RE = re.compile(
    r"\b(?:ice\s+cream|whipped\s+cream|sour\s+cream|heavy\s+cream|"
    r"olive\s+(?:tree|trees|branch|branches|oil|grove|groves)|"
    r"chestnut\s+(?:tree|trees)|"
    r"orange\s+(?:tree|trees|grove|groves|juice))\b",
    re.IGNORECASE,
)
_ASSUMPTION_PLACEHOLDER_RE = re.compile(
    r"^(?:"
    r"[-–—]+|"
    r"допущени[ейя]\s+(?:нет|отсутству(?:ет|ют)|не\s+требу[ею]тся|не\s+нужн[оы])\b.*|"
    r"нет\s+(?:визуальных\s+|дополнительных\s+)?допущений\b.*|"
    r"без\s+(?:визуальных\s+|дополнительных\s+)?допущений\b.*|"
    r"нет(?:[\s.,;:!?–—-]*$|[\s]*[:–—,-]\s*.*)|"
    r"не\s+требу[ею]тся(?:[\s.,;:!?–—-]*$|[\s]*[:–—,-]\s*.*)|"
    r"отсутству(?:ет|ют)(?:[\s.,;:!?–—-]*$|[\s]*[:–—,-]\s*.*)|"
    r"no\s+(?:visual\s+|additional\s+)?assumptions?\b.*|"
    r"none(?:[\s.,;:!?–—-]*$|[\s]*[:–—,-]\s*.*|\s+(?:needed|required)\b.*)|"
    r"n/?a\.?"
    r")$",
    re.IGNORECASE,
)
_ASSUMPTION_TECHNICAL_KEY_RE = re.compile(
    r"(?:"
    r"\b(?:attire_default|palette_value|description_en|description_ru|"
    r"used_design_keys|depicted_entity_ids|compact_description_en)\b"
    r"|characters\[\]"
    r")",
    re.IGNORECASE,
)
_CLAUSE_STOPWORDS = frozenset(
    {
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "into",
        "over",
        "near",
        "very",
        "wear",
        "wearing",
        "made",
        "like",
    }
)


def has_cyrillic(text: str) -> bool:
    """Return True if the string contains any Cyrillic character."""
    return bool(_CYRILLIC_RE.search(text))


def find_speech_bubble_term(text: str) -> str | None:
    """Return the first disallowed comic speech/thought bubble term in `text`, or None."""
    if not text or not text.strip():
        return None
    match = _SPEECH_BUBBLE_RE.search(text)
    return match.group(0) if match is not None else None


def dedupe_texts(items: Iterable[str]) -> tuple[str, ...]:
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


def filter_non_assumption_placeholders(items: Sequence[str]) -> tuple[str, ...]:
    """Drop empty strings, meta-placeholders, and technical schema-key leaks, then deduplicate."""
    kept: list[str] = []
    for raw in items:
        cleaned = " ".join(str(raw).split()).strip()
        if not cleaned:
            continue
        if _ASSUMPTION_PLACEHOLDER_RE.match(cleaned):
            continue
        if _ASSUMPTION_TECHNICAL_KEY_RE.search(cleaned):
            continue
        kept.append(cleaned)
    return dedupe_texts(kept)


def find_monochrome_color_word(
    text: str,
    *,
    allowed_names: Sequence[str] = (),
    strict_palette: bool = False,
) -> str | None:
    """Find the first disallowed chromatic color word in `text` for `monochrome_ink`.

    Masks full proper names from `allowed_names` case-insensitively on word boundaries and
    individual capitalized >=3-char name tokens case-sensitively so names like *Mr Brown* or
    *The Woman in Green* pass without ever permitting lowercase *brown coat* or *green dress*.
    When `strict_palette=False`, excludes metallic materials (`bronze`/`copper`) and masks
    common physical object idioms (`ice cream`, `olive tree`, `chestnut tree`, `orange tree`).
    """
    if not text or not text.strip():
        return None
    masked = text
    cleaned_names = sorted(
        {
            " ".join(str(n).split()).strip()
            for n in allowed_names
            if n is not None and str(n).strip()
        },
        key=len,
        reverse=True,
    )
    for name in cleaned_names:
        parts = name.split()
        if len(parts) < 2:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(p) for p in parts) + r"(?!\w)"
        masked = re.sub(pattern, " ", masked, flags=re.IGNORECASE)

    cap_tokens: set[str] = set()
    for name in cleaned_names:
        for tok in re.findall(r"[^\W\d_]+", name):
            if len(tok) >= 3 and tok[0].isupper():
                cap_tokens.add(tok)
    for tok in sorted(cap_tokens, key=len, reverse=True):
        variants = {tok, tok.upper()}
        pattern = r"(?<!\w)(?:" + "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True)) + r")(?!\w)"
        masked = re.sub(pattern, " ", masked)

    if not strict_palette:
        masked = _MONOCHROME_OBJECT_PHRASES_RE.sub(" ", masked)
        en_re = _MONOCHROME_CHROMATIC_EN_RE
        ru_re = _MONOCHROME_CHROMATIC_RU_RE
    else:
        en_re = _MONOCHROME_COLOR_EN_RE
        ru_re = _MONOCHROME_COLOR_RU_RE

    match = en_re.search(masked) or ru_re.search(masked)
    return match.group(0) if match is not None else None


def _clause_tokens(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-z]{4,}", text.casefold())
        if tok not in _CLAUSE_STOPWORDS
    ]


def _token_matches_ref(tok: str, ref_tokens: set[str]) -> bool:
    if tok in ref_tokens:
        return True
    if len(tok) >= 4 and (f"{tok}s" in ref_tokens or f"{tok}es" in ref_tokens):
        return True
    if len(tok) >= 5:
        if tok.endswith("s") and tok[:-1] in ref_tokens:
            return True
        if tok.endswith("es") and tok[:-2] in ref_tokens:
            return True
    return False


def _clause_covered_in_text(clause: str, reference_text: str) -> bool:
    norm_clause = " ".join(clause.split()).strip().rstrip(".;").strip().casefold()
    if not norm_clause:
        return True
    norm_ref = " ".join(reference_text.split()).strip().casefold()
    left_b = r"(?<!\w)" if norm_clause[0].isalnum() else ""
    right_b = r"(?!\w)" if norm_clause[-1].isalnum() else ""
    if re.search(f"{left_b}{re.escape(norm_clause)}{right_b}", norm_ref) is not None:
        return True
    tokens = _clause_tokens(norm_clause)
    if len(tokens) < 2:
        return False
    ref_tokens = set(_clause_tokens(norm_ref))
    return all(_token_matches_ref(tok, ref_tokens) for tok in tokens)


def _appearance_reflected_in_prompt(desc_clean: str, prompt_lower: str) -> bool:
    if desc_clean.casefold() in prompt_lower:
        return True
    clauses = [" ".join(piece.split()).strip() for piece in desc_clean.split(";")]
    clauses = [c for c in clauses if c]
    if not clauses:
        return False
    return all(_clause_covered_in_text(clause, prompt_lower) for clause in clauses)


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
    comp = re.sub(r"\.{2,}$", ".", comp)
    if comp[-1] not in ".!?":
        comp = f"{comp}."
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
        clauses = [
            " ".join(piece.split()).strip().rstrip(".;").strip()
            for piece in desc_clean.split(";")
        ]
        clauses = [c for c in clauses if c]
        if not clauses:
            raise IllustrationValidationError(
                IllustrationReasonCode.MODEL_OUTPUT_INVALID,
                "применимый образ обязан содержать непустое description_en",
            )
        remaining_clauses = [c for c in clauses if not _clause_covered_in_text(c, comp)]
        if remaining_clauses:
            rem_text = "; ".join(remaining_clauses)
            if name and not has_cyrillic(name):
                line = f"{name}: {rem_text}"
            else:
                line = rem_text
            cleaned_line = line.strip().rstrip(".;").strip()
            if cleaned_line:
                appearance_lines.append(cleaned_line)

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
        if not _appearance_reflected_in_prompt(desc_clean, prompt_lower):
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
    "dedupe_texts",
    "filter_non_assumption_placeholders",
    "find_monochrome_color_word",
    "find_speech_bubble_term",
    "has_cyrillic",
    "resolve_insert_after_block_id",
    "validate_moment",
]
