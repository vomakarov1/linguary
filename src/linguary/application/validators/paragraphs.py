"""Validation ensuring translation did not absorb paragraphs from adjacent blocks.

A segmented block is checked on each translated slice. A poem compares the whole
original with the joined translation via `split('\\n')`, blank lines included.
"""

from typing import overload

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.validators.treatment import (
    joined_segment_translation,
    planned_segment_modes,
)
from linguary.domain import BlockTreatment, BlockType, TranslationUnit, TranslationUnitItem
from linguary.domain.llm_log import Violation

_SPEECH_MARKERS = ('"', '“', '”', '«', '»', "'", '‘', '’', '—', '–')
_DIALOGUE_DASHES = ('—', '–', '-')


def _check_paragraphs(label: str, source_text: str, translated_text: str) -> str | None:
    source_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    translated_lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    source_count = len(source_lines)
    translated_count = len(translated_lines)
    if translated_count > source_count:
        has_speech = any(m in source_text for m in _SPEECH_MARKERS)
        extra_lines = translated_lines[source_count:]
        if has_speech and extra_lines and all(line.lstrip().startswith(_DIALOGUE_DASHES) for line in extra_lines):
            return None
        return (
            f"перевод {label} содержит {translated_count} "
            f"абзацев, а оригинал — {source_count}: сохрани исходное число "
            f"непустых строк ({source_count}) и весь текст своего блока; "
            "не включай соседние блоки. Если собственная реплика или авторское "
            "окончание вынесены на новую строку, объедини собственную реплику "
            "или авторское окончание с остальным текстом блока, не удаляй их"
        )
    return None


def _poem_line_violation(block_id: str, source_text: str, translated_text: str) -> str | None:
    """A poem must keep every line, including blank lines between stanzas."""
    source_count = len(source_text.split("\n"))
    translated_count = len(translated_text.split("\n"))
    if source_count == translated_count:
        return None
    return (
        f"перевод блока {block_id} содержит {translated_count} "
        f"строк, а оригинал — {source_count}: сохрани исходное число "
        f"строк ({source_count}), включая пустые строки строф"
    )


def _block_paragraph_violation(
    block: BlockResultResponse, item: TranslationUnitItem
) -> str | None:
    """Line-count check for one block. Poems use the whole text; other blocks use slices."""
    source = item.block
    if source.type is BlockType.POEM:
        translated = joined_segment_translation(block)
        if translated is None:
            return None
        return _poem_line_violation(block.source_block_id, source.text, translated)
    modes = planned_segment_modes(item)
    if not modes:
        if block.translated_text is None:
            return None
        return _check_paragraphs(
            f"блока {block.source_block_id}", source.text, block.translated_text
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
        )
        if message is not None:
            return message
    return None


@overload
def paragraph_violation(
    target: str, *, source_text: str, translated_text: str
) -> Violation | None: ...


@overload
def paragraph_violation(
    target: TranslationResultResponse, *, unit: TranslationUnit
) -> str | None: ...


def paragraph_violation(
    target: str | TranslationResultResponse,
    *,
    source_text: str | None = None,
    translated_text: str | None = None,
    unit: TranslationUnit | None = None,
) -> Violation | str | None:
    """Reason for exceeding translated paragraph count, or `None`."""
    if isinstance(target, str):
        if source_text is None or translated_text is None:
            return None
        message = _check_paragraphs(f"блока {target}", source_text, translated_text)
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
        message = _block_paragraph_violation(block, item)
        if message is not None:
            return message
    return None


def _paragraph_count(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


__all__ = ["paragraph_violation"]
