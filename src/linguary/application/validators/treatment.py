"""Validation of processing response against the saved chapter plan (ALG-04).

The plan is the single source of truth for a block's treatment mode: the `treatment` field
in the model response confirms it, but does not override it (PRD § 14). This yields two checks:
the response treatment matches the plan, and the result content matches the plan's treatment.

The second check is formulated relative to the plan intentionally. The `BlockResultResponse` DTO
validates content relative to the mode declared by the model itself — which is insufficient:
a self-consistent response might be consistent with the wrong treatment mode. Once the mode check
passes, both assertions align, and content validation remains an explicit ALG-04 invariant:
it triggers if DTO rules ever diverge from the plan.
"""

from __future__ import annotations

import re

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.domain import BlockTreatment, TranslationUnit, TranslationUnitItem

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_HINT_STOPWORDS = frozenset(
    {
        "the", "and", "for", "but", "nor", "yet", "not",
        "with", "from", "into", "onto", "upon", "than", "then",
        "that", "this", "these", "those",
        "you", "your", "yours", "she", "her", "hers", "him", "his",
        "its", "our", "ours", "they", "them", "their", "theirs",
        "who", "whom", "whose", "which", "what",
        "are", "was", "were", "been", "being", "have", "has", "had",
        "does", "did", "will", "would", "shall", "should", "can", "could",
        "may", "might", "must",
        "one", "ones", "some", "any", "all", "very", "too", "also",
        "out", "off", "over", "under", "here", "there", "where", "when",
        "something", "someone", "somebody", "anything", "anyone", "oneself",
    }
)


def treatment_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Violation description of mismatch between response treatment and plan, or `None`."""
    for block in payload.blocks:
        planned = unit.treatment_for(block.source_block_id)
        if block.treatment is not planned:
            return (
                f"режим блока {block.source_block_id} в ответе модели ({block.treatment}) "
                f"не совпадает с планом ({planned})"
            )
    return None


def content_violation(
    payload: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None:
    """Violation description of result content mismatching the planned treatment, or `None`."""
    items = {item.block.id: item for item in unit.items}
    for block in payload.blocks:
        item = items.get(block.source_block_id)
        if item is None:
            continue
        violation = _block_content_violation(block, item=item)
        if violation is not None:
            return violation
    return None


def planned_segment_modes(item: TranslationUnitItem) -> tuple[BlockTreatment, ...]:
    """One mode per stored span. A one-span block keeps `()`.

    An empty `segment_treatments` on a multi-segment block means every span uses
    the block mode. Callers that store a result repeat that mode and do not
    collapse the tuple.
    """
    if len(item.block.segments) < 2:
        return ()
    if item.segment_treatments:
        return item.segment_treatments
    return (item.treatment,) * len(item.block.segments)


def joined_segment_translation(block: BlockResultResponse) -> str | None:
    """Join of segment translations in index order, or the whole-block translation.

    Matches `BlockProcessingResult.rebuild_translation`: non-empty pieces separated
    by one space. Empty `segments` keeps `translated_text`.
    """
    if not block.segments:
        return block.translated_text
    parts = [
        segment.translated_text
        for segment in sorted(block.segments, key=lambda item: item.index)
        if segment.translated_text and segment.translated_text.strip()
    ]
    if not parts:
        return None
    return " ".join(parts)


def _normalize_match_text(text: str) -> str:
    cleaned = text.replace("’", "'").replace("‘", "'").lower()
    return " ".join(cleaned.split())


def _word_stems(word: str) -> set[str]:
    stems = {word}
    if len(word) > 4 and word.endswith(("ies", "ied")):
        stems.add(word[:-3] + "y")
        stems.add(word[:-3])
    if len(word) > 5 and word.endswith("ily"):
        stems.add(word[:-3] + "y")
        stems.add(word[:-3])
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        stems.add(base)
        stems.add(base + "e")
        if len(base) >= 4 and base[-1] == base[-2]:
            stems.add(base[:-1])
    if len(word) > 4 and word.endswith("ed"):
        base = word[:-2]
        stems.add(base)
        stems.add(word[:-1])
        if len(base) >= 4 and base[-1] == base[-2]:
            stems.add(base[:-1])
    if len(word) > 4 and word.endswith(("es", "er")):
        stems.add(word[:-2])
        stems.add(word[:-1])
    if len(word) > 5 and word.endswith("est"):
        stems.add(word[:-3])
        stems.add(word[:-2])
    if len(word) > 4 and word.endswith("ly"):
        stems.add(word[:-2])
    if len(word) > 3 and word.endswith("s"):
        stems.add(word[:-1])
    if len(word) > 3 and word.endswith(("e", "y")):
        stems.add(word[:-1])
    return {stem for stem in stems if len(stem) >= 3}


def _hint_matches_glossary_texts(original: str, glossary_texts: tuple[str, ...]) -> bool:
    normalized_hint = _normalize_match_text(original).strip(".,;:!?\"'«»“”‘’()[]")
    if not normalized_hint:
        return False
    candidates = [normalized_hint]
    if normalized_hint.startswith("to ") and len(normalized_hint) > 3:
        bare = normalized_hint[3:].strip()
        if bare:
            candidates.append(bare)
    normalized_texts = tuple(_normalize_match_text(text) for text in glossary_texts)
    text_words: set[str] = set()
    for text in normalized_texts:
        words = _WORD_RE.findall(text)
        text_words.update(words)
        for candidate in candidates:
            if len(candidate) >= 3:
                if re.search(rf"(?<!\w){re.escape(candidate)}", text):
                    return True
            elif candidate in words:
                return True
    content_text_words = {
        word for word in text_words if len(word) >= 3 and word not in _HINT_STOPWORDS
    }
    if not content_text_words:
        return False
    text_stems: set[str] = set()
    for word in content_text_words:
        text_stems.update(_word_stems(word))
    hint_words = [
        word
        for word in _WORD_RE.findall(candidates[-1])
        if len(word) >= 3 and word not in _HINT_STOPWORDS
    ]
    for hint_word in hint_words:
        if hint_word in content_text_words:
            return True
        if len(hint_word) >= 4 and any(
            text_word.startswith(hint_word)
            or (len(text_word) >= 4 and hint_word.startswith(text_word))
            for text_word in content_text_words
        ):
            return True
        if _word_stems(hint_word) & text_stems:
            return True
    return False


def _block_content_violation(
    block: BlockResultResponse, *, item: TranslationUnitItem
) -> str | None:
    """Result content of a single block relative to its planned treatment mode (PRD § 15)."""
    modes = planned_segment_modes(item)
    if not modes:
        if block.segments:
            return (
                f"для блока {block.source_block_id} без нескольких сегментов "
                "список segments недопустим"
            )
        return _legacy_content(block, planned=item.treatment)
    return _segmented_content(block, item=item, modes=modes)


def _segmented_content(
    block: BlockResultResponse,
    *,
    item: TranslationUnitItem,
    modes: tuple[BlockTreatment, ...],
) -> str | None:
    """Multi-segment content check against per-span planned treatments."""
    if block.translated_text is not None:
        return (
            f"для блока {block.source_block_id} с сегментами "
            "полный перевод блока недопустим"
        )
    spans = item.block.segment_spans
    if len(modes) != len(spans):
        return (
            f"у блока {block.source_block_id} {len(modes)} режимов сегментов "
            f"на {len(spans)} сегментов"
        )
    seen: set[int] = set()
    for segment in block.segments:
        index = segment.index
        if index in seen:
            return f"сегмент {index} блока {block.source_block_id} указан дважды"
        seen.add(index)
        if index > len(modes):
            return (
                f"сегмент {index} блока {block.source_block_id} "
                f"вне диапазона 1..{len(modes)}"
            )
        mode = modes[index - 1]
        if not mode.requires_translation:
            return (
                f"для сегмента {index} блока {block.source_block_id} "
                f"с режимом {mode} перевод недопустим"
            )
    for index, mode in enumerate(modes, start=1):
        if mode.requires_translation and index not in seen:
            return (
                f"для сегмента {index} блока {block.source_block_id} "
                "с режимом TRANSLATE нет перевода"
            )
    if any(mode is BlockTreatment.GLOSSARY_ONLY for mode in modes):
        if not block.glossary_hints:
            return f"для блока {block.source_block_id} обязательны словарные подсказки"
        if any(mode.requires_translation for mode in modes):
            glossary_texts = tuple(
                item.block.text[spans[i].start : spans[i].end]
                for i, mode in enumerate(modes)
                if mode is BlockTreatment.GLOSSARY_ONLY
            )
            if not any(
                _hint_matches_glossary_texts(hint.original, glossary_texts)
                for hint in block.glossary_hints
            ):
                return (
                    f"для блока {block.source_block_id} ни одна словарная подсказка "
                    "не относится к сегменту GLOSSARY_ONLY"
                )
    elif (
        all(mode is BlockTreatment.READ_AS_IS for mode in modes)
        and block.glossary_hints
    ):
        return (
            f"для блока {block.source_block_id} с режимом {BlockTreatment.READ_AS_IS} "
            "словарные подсказки недопустимы"
        )
    return None


def _legacy_content(
    block: BlockResultResponse, *, planned: BlockTreatment
) -> str | None:
    """Whole-block content check used when the block has a single span."""
    has_translation = bool(block.translated_text and block.translated_text.strip())
    if planned.requires_translation and not has_translation:
        return f"для блока {block.source_block_id} с режимом {planned} нет полного перевода"
    if not planned.allows_translation and block.translated_text is not None:
        return (
            f"для блока {block.source_block_id} с режимом {planned} "
            "полный перевод недопустим"
        )
    if planned.requires_glossary_hints and not block.glossary_hints:
        return (
            f"для блока {block.source_block_id} с режимом {planned} "
            "нет словарных подсказок"
        )
    if not planned.allows_glossary_hints and block.glossary_hints:
        return (
            f"для блока {block.source_block_id} с режимом {planned} "
            "словарные подсказки недопустимы"
        )
    return None


__all__ = [
    "content_violation",
    "joined_segment_translation",
    "planned_segment_modes",
    "treatment_violation",
]
