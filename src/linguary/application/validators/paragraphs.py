"""Validation ensuring translation did not absorb paragraphs from adjacent blocks.

A segmented block is checked on each translated slice. A poem compares the whole
original with the joined translation via `split('\\n')`, blank lines included.
Strict line preservation also rejects dropped non-empty lines, and in a poem a
blank substituted for a text line.
"""

import re
from typing import overload

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.validators.quotes import _outer_unclosed_open
from linguary.application.validators.treatment import (
    joined_segment_translation,
    planned_segment_modes,
)
from linguary.domain import BlockTreatment, BlockType, TranslationUnit, TranslationUnitItem
from linguary.domain.llm_log import Violation

_SPEECH_MARKERS = ('"', '“', '”', '«', '»', "'", '‘', '’', '—', '–')
_DIALOGUE_DASHES = ('—', '–', '-')
_POST_SPEECH_CLOSE_RE = re.compile(r"""[.!?…]["”’'](?=\s)""")
_INLINE_DIALOGUE_TRANSITION_RE = re.compile(r"([,!?.…])\s+[—–]\s+([А-ЯЁA-Zа-яёa-z])")


def _count_embedded_post_speech_narratives(line: str) -> int:
    """Count inline `. — <Capital>` transitions that start a post-speech narrative rather than resuming speech after lowercase attribution."""
    state = "speech" if line.lstrip().startswith(_DIALOGUE_DASHES) else "narrative"
    embedded = 0
    for match in _INLINE_DIALOGUE_TRANSITION_RE.finditer(line):
        punct, next_char = match.group(1), match.group(2)
        if state == "speech":
            if next_char.islower():
                state = "attribution"
            elif punct in ".!?…" and next_char.isupper():
                embedded += 1
                state = "post_narrative"
        elif state == "attribution":
            if (punct in ".!?…" and next_char.isupper()) or (punct == "," and next_char.islower()):
                state = "speech"
        elif state == "post_narrative":
            if punct in ".!?…" and next_char.isupper():
                state = "speech"
    return embedded


def _count_post_speech_narrative_sentences(source_text: str) -> int:
    """Count closing speech quotes after terminal punctuation followed outside quotes by a capitalized sentence."""
    count = 0
    for match in _POST_SPEECH_CLOSE_RE.finditer(source_text):
        after_pos = match.end()
        if _outer_unclosed_open(source_text, after_pos) is not None:
            continue
        rest = source_text[after_pos:].lstrip()
        if rest and rest[0].isupper():
            count += 1
    return count


def _check_paragraphs(
    label: str,
    source_text: str,
    translated_text: str,
    *,
    strict_line_preservation: bool = False,
) -> str | None:
    source_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    translated_lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    source_count = len(source_lines)
    translated_count = len(translated_lines)
    if translated_count > source_count:
        has_speech = any(m in source_text for m in _SPEECH_MARKERS)
        extra_lines = translated_lines[source_count:]
        if has_speech and extra_lines:
            if all(line.lstrip().startswith(_DIALOGUE_DASHES) for line in extra_lines):
                return None
            post_count = _count_post_speech_narrative_sentences(source_text)
            if post_count > 0 and any(
                line.lstrip().startswith(_DIALOGUE_DASHES) for line in translated_lines
            ):
                non_dash_extra = [
                    line
                    for line in extra_lines
                    if not line.lstrip().startswith(_DIALOGUE_DASHES)
                ]
                embedded_narrative = sum(
                    _count_embedded_post_speech_narratives(line)
                    for line in translated_lines[:source_count]
                )
                if (
                    len(non_dash_extra) + embedded_narrative <= min(post_count, 2)
                    and len(extra_lines) <= post_count + 1
                    and all(line[0].isupper() for line in non_dash_extra)
                ):
                    return None
        return (
            f"перевод {label} содержит {translated_count} "
            f"абзацев, а оригинал — {source_count}: сохрани исходное число "
            f"непустых строк ({source_count}) и весь текст своего блока; "
            "не включай соседние блоки. Если собственная реплика или авторское "
            "окончание вынесены на новую строку, объедини собственную реплику "
            "или авторское окончание с остальным текстом блока, не удаляй их"
        )
    if strict_line_preservation and translated_count < source_count:
        return (
            f"перевод {label} содержит {translated_count} "
            f"абзацев, а оригинал — {source_count}: сохрани исходное число "
            f"непустых строк ({source_count}) и весь текст своего блока; "
            "не пропускай строки"
        )
    return None


def _poem_line_violation(
    block_id: str,
    source_text: str,
    translated_text: str,
    *,
    strict_line_preservation: bool = False,
) -> str | None:
    """A poem must keep every line, including blank lines between stanzas."""
    source_lines = source_text.split("\n")
    translated_lines = translated_text.split("\n")
    source_count = len(source_lines)
    translated_count = len(translated_lines)
    if source_count != translated_count:
        return (
            f"перевод блока {block_id} содержит {translated_count} "
            f"строк, а оригинал — {source_count}: сохрани исходное число "
            f"строк ({source_count}), включая пустые строки строф"
        )
    if not strict_line_preservation:
        return None
    source_nonempty = sum(bool(line.strip()) for line in source_lines)
    translated_nonempty = sum(bool(line.strip()) for line in translated_lines)
    if source_nonempty == translated_nonempty:
        return None
    return (
        f"перевод блока {block_id} содержит {translated_nonempty} "
        f"непустых строк, а оригинал — {source_nonempty}: сохрани исходное число "
        f"непустых строк ({source_nonempty}), не подменяя строки пустыми"
    )


def _block_paragraph_violation(
    block: BlockResultResponse,
    item: TranslationUnitItem,
    *,
    strict_line_preservation: bool = False,
) -> str | None:
    """Line-count check for one block. Poems use the whole text; other blocks use slices."""
    source = item.block
    if source.type is BlockType.POEM:
        translated = joined_segment_translation(block)
        if translated is None:
            return None
        return _poem_line_violation(
            block.source_block_id,
            source.text,
            translated,
            strict_line_preservation=strict_line_preservation,
        )
    modes = planned_segment_modes(item)
    if not modes:
        if block.translated_text is None:
            return None
        return _check_paragraphs(
            f"блока {block.source_block_id}",
            source.text,
            block.translated_text,
            strict_line_preservation=strict_line_preservation,
        )
    by_index = {segment.index: segment.translated_text for segment in block.segments}
    for index, (span, mode) in enumerate(
        zip(source.segment_spans, modes, strict=True), start=1
    ):
        if mode is not BlockTreatment.TRANSLATE:
            continue
        text = by_index.get(index)
        if text is None:
            continue
        message = _check_paragraphs(
            f"сегмента {index} блока {block.source_block_id}",
            source.text[span.start:span.end],
            text,
            strict_line_preservation=strict_line_preservation,
        )
        if message is not None:
            return message
    return None


@overload
def paragraph_violation(
    target: str,
    *,
    source_text: str,
    translated_text: str,
    strict_line_preservation: bool = False,
) -> Violation | None: ...


@overload
def paragraph_violation(
    target: TranslationResultResponse,
    *,
    unit: TranslationUnit,
    strict_line_preservation: bool = False,
) -> str | None: ...


def paragraph_violation(
    target: str | TranslationResultResponse,
    *,
    source_text: str | None = None,
    translated_text: str | None = None,
    unit: TranslationUnit | None = None,
    strict_line_preservation: bool = False,
) -> Violation | str | None:
    """Reason the line count broke the contract, or `None`.

    Extra non-empty lines are always rejected, except a dialogue split onto dash
    lines. Fewer non-empty lines are rejected only when `strict_line_preservation`
    is true. Poems always keep `split('\\n')` length; strict mode also keeps the
    non-empty line count.
    """
    if isinstance(target, str):
        if source_text is None or translated_text is None:
            return None
        message = _check_paragraphs(
            f"блока {target}",
            source_text,
            translated_text,
            strict_line_preservation=strict_line_preservation,
        )
        if message is not None:
            return Violation(
                rule_code="extra_paragraphs",
                stage="translation_unit",
                scope="local",
                block_ids=(target,),
                artifact_kinds=("translated_text",),
                safe_message=message,
            )
        return None

    if unit is None:
        return None
    items = {item.block.id: item for item in unit.items}
    for block in target.blocks:
        item = items.get(block.source_block_id)
        if item is None:
            continue
        message = _block_paragraph_violation(
            block, item, strict_line_preservation=strict_line_preservation
        )
        if message is not None:
            return message
    return None


def _paragraph_count(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


__all__ = ["paragraph_violation"]
