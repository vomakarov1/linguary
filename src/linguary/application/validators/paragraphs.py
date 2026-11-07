"""Validation ensuring translation did not absorb paragraphs from adjacent blocks."""

from typing import overload

from linguary.application.llm_dto import TranslationResultResponse
from linguary.domain import TranslationUnit
from linguary.domain.llm_log import Violation

_SPEECH_MARKERS = ('"', '“', '”', '«', '»', "'", '‘', '’', '—', '–')
_DIALOGUE_DASHES = ('—', '–', '-')


def _check_paragraphs(block_id: str, source_text: str, translated_text: str) -> str | None:
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
            f"перевод блока {block_id} содержит {translated_count} "
            f"абзацев, а оригинал — {source_count}: сохрани исходное число "
            f"непустых строк ({source_count}) и весь текст своего блока; "
            "не включай соседние блоки. Если собственная реплика или авторское "
            "окончание вынесены на новую строку, объедини собственную реплику "
            "или авторское окончание с остальным текстом блока, не удаляй их"
        )
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
        message = _check_paragraphs(target, source_text, translated_text)
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
    sources = {item.block.id: item.block.text for item in unit.items}
    for block in target.blocks:
        if block.translated_text is None:
            continue
        source = sources.get(block.source_block_id, "")
        message = _check_paragraphs(block.source_block_id, source, block.translated_text)
        if message is not None:
            return message
    return None


def _paragraph_count(text: str) -> int:
    return sum(bool(line.strip()) for line in text.splitlines())


__all__ = ["paragraph_violation"]
