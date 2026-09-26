"""Ссылки на сноски книги внутри перевода блока (PRD § 7, § 20).

Перевод хранится простым текстом, без встроенной разметки. Ссылку на сноску оригинала,
которую книга ставит посреди абзаца (`…her face.<a epub:type="noteref">1</a>`), в таком
тексте не выразить: модель видит её номер как обычную цифру и переносит в перевод
голой цифрой, которая никуда не ведёт.

Поэтому ссылка на сноску передаётся модели меткой `[^N]`, где `N` — порядковый номер
ссылки на сноску в исходном блоке, начиная с единицы, а модель переносит метку в перевод.
Метка ссылается на разметку исходного блока, а не повторяет её: адрес сноски и видимый
текст ссылки берутся из оригинала, поэтому в собранной книге ссылка из перевода ведёт
туда же, куда и ссылка оригинала.
"""

from __future__ import annotations

import re

from linguary.domain.document import DocumentBlock, InlineAnnotation, InlineAnnotationType

#: Метка ссылки на сноску в тексте перевода. Пробел перед меткой принадлежит ей:
#: снятая метка не оставляет висячего пробела перед знаком препинания.
_MARKER = re.compile(r" ?\[\^(\d+)\]")


def footnote_marker(ordinal: int) -> str:
    """Метка `ordinal`-й по порядку ссылки на сноску блока."""
    return f"[^{ordinal}]"


def footnote_references(block: DocumentBlock) -> tuple[InlineAnnotation, ...]:
    """Ссылки на сноски блока в порядке текста: `[^1]` — первая из них."""
    return tuple(
        sorted(
            (
                annotation
                for annotation in block.inline_annotations
                if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE
            ),
            key=lambda annotation: annotation.start,
        )
    )


def translation_segments(
    text: str, block: DocumentBlock
) -> tuple[str | InlineAnnotation, ...]:
    """Перевод, разобранный на куски текста и ссылки на сноски исходного блока.

    Метка, которой у блока нет соответствующей ссылки, отбрасывается: придуманная
    моделью ссылка вела бы в никуда.
    """
    references = footnote_references(block)
    segments: list[str | InlineAnnotation] = []
    position = 0
    for match in _MARKER.finditer(text):
        segments.append(text[position : match.start()])
        reference = _reference(references, match.group(1))
        if reference is not None:
            if match.group(0).startswith(" "):
                segments.append(" ")
            segments.append(reference)
        position = match.end()
    segments.append(text[position:])
    return tuple(segment for segment in segments if segment != "")


def known_markers_only(text: str, block: DocumentBlock) -> str:
    """Перевод, в котором остались только метки существующих ссылок блока."""
    references = footnote_references(block)
    return _MARKER.sub(
        lambda match: match.group(0)
        if _reference(references, match.group(1)) is not None
        else "",
        text,
    )


def without_footnote_markers(text: str) -> str:
    """Текст без меток ссылок на сноски: для полей, где ссылка не нужна."""
    return _MARKER.sub("", text)


def with_reference_labels(text: str, block: DocumentBlock) -> str:
    """Перевод, где метки заменены видимым текстом ссылок, как в тексте оригинала блока."""
    return "".join(
        segment
        if isinstance(segment, str)
        else block.text[segment.start : segment.end]
        for segment in translation_segments(text, block)
    )


def _reference(
    references: tuple[InlineAnnotation, ...], ordinal: str
) -> InlineAnnotation | None:
    """Ссылка на сноску по номеру из метки либо `None`, если такой нет."""
    index = int(ordinal) - 1
    return references[index] if 0 <= index < len(references) else None


__all__ = [
    "footnote_marker",
    "footnote_references",
    "known_markers_only",
    "translation_segments",
    "with_reference_labels",
    "without_footnote_markers",
]
