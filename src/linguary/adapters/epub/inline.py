"""Встроенное форматирование, ссылки и сноски внутри одного блока.

`DocumentBlock.text` хранится без XHTML (PRD § 7), поэтому оформление снимается отдельно:
курсив, полужирный, смысловое выделение, внутренние ссылки и ссылки на сноски описываются
`InlineAnnotation` со смещениями в уже нормализованном тексте блока.

Здесь же собираются якоря блока: идентификаторы, по которым на блок ссылается сама книга.
Ссылка на сноску Gutenberg указывает на `id` блочного элемента, а обратная ссылка из сноски —
на `id` inline-элемента `<a>` внутри абзаца, поэтому одного `SourceReference.element_id`
для восстановления связей не хватает.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import unquote

from lxml import etree

from linguary.domain import InlineAnnotation, InlineAnnotationType

#: Пробельные символы XHTML, которые схлопываются в один пробел. Неразрывный пробел
#: сюда не входит: он несёт смысл и остаётся в тексте как есть.
_COLLAPSIBLE_WHITESPACE = re.compile(r"[ \t\r\n\f\v]+")

#: Теги встроенного оформления и соответствующая им разметка.
_ANNOTATION_TAGS: dict[str, InlineAnnotationType] = {
    "i": InlineAnnotationType.ITALIC,
    "em": InlineAnnotationType.EMPHASIS,
    "b": InlineAnnotationType.BOLD,
    "strong": InlineAnnotationType.BOLD,
}

#: Теги, которые не дают текста, но задают перенос строки внутри блока.
_LINE_BREAK_TAGS = frozenset({"br"})

#: Служебные теги: их содержимое не является текстом книги.
_NON_TEXT_TAGS = frozenset({"script", "style", "link", "meta", "base", "noscript"})

#: Значения `epub:type` и классов, которыми издатели помечают ссылку на сноску.
_NOTE_REFERENCE_MARKERS = frozenset({"noteref", "footnoteref", "footnote-ref"})

#: Схемы, по которым ссылка ведёт наружу книги и не нормализуется.
_EXTERNAL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

_EPUB_NAMESPACE = "http://www.idpf.org/2007/ops"


@dataclass(frozen=True, slots=True)
class InlineContent:
    """Текст блока вместе со снятым с него оформлением и якорями."""

    text: str
    annotations: tuple[InlineAnnotation, ...] = ()
    anchor_ids: tuple[str, ...] = ()


def local_name(element: etree._Element) -> str | None:
    """Имя тега без пространства имён либо `None` для комментариев и инструкций."""
    if not isinstance(element.tag, str):
        return None
    return etree.QName(element).localname


def resolve_link_target(href: str, *, resource_path: str) -> str:
    """Приводит `href` к виду «путь ресурса книги#фрагмент».

    Gutenberg почти всегда пишет `href="файл.xhtml#якорь"` даже для ссылки внутри того же
    документа, а синтетические книги — голый `href="#якорь"`. Оба стиля дают одинаковый
    результат, поэтому дальше по коду вариант ссылки уже не различается. Внешние ссылки
    (`http:`, `mailto:`) остаются как есть.
    """
    if _EXTERNAL_SCHEME.match(href):
        return href
    path, separator, fragment = href.partition("#")
    if not path:
        return f"{resource_path}{separator}{fragment}"
    directory = posixpath.dirname(resource_path)
    resolved = posixpath.normpath(posixpath.join(directory, unquote(path)))
    return f"{resolved}{separator}{fragment}" if separator else resolved


def footnote_element_ids(root: etree._Element) -> frozenset[str]:
    """Идентификаторы элементов документа, которые являются сносками книги.

    Оригинальные идентификаторы сносок сохраняются без изменений: по ним экспортёр
    восстанавливает внутренние ссылки книги, а учебные заметки живут в собственном
    пространстве имён (PRD § 19.3).
    """
    return frozenset(
        element_id
        for element in root.iter()
        if local_name(element) is not None
        and (element_id := element.get("id"))
        and is_footnote_element(element)
    )


def is_footnote_element(element: etree._Element) -> bool:
    """Помечен ли элемент как сноска книги."""
    if _epub_type_tokens(element) & {"footnote", "rearnote", "endnote", "note"}:
        return True
    return "footnote" in _class_tokens(element)


def is_footnote_reference(
    element: etree._Element, *, resource_path: str, footnote_ids: frozenset[str]
) -> bool:
    """Ведёт ли ссылка на сноску книги.

    Признаком служит либо явная пометка издателя на самой ссылке или на обёртке `<sup>`,
    либо то, что цель ссылки — известная сноска этого документа.
    """
    for candidate in (element, element.getparent()):
        if candidate is None:
            continue
        if _epub_type_tokens(candidate) & _NOTE_REFERENCE_MARKERS:
            return True
        if _class_tokens(candidate) & _NOTE_REFERENCE_MARKERS:
            return True
    href = element.get("href")
    if not href:
        return False
    target = resolve_link_target(href, resource_path=resource_path)
    path, separator, fragment = target.partition("#")
    return bool(separator) and path == resource_path and fragment in footnote_ids


class InlineContentBuilder:
    """Накопитель текста блока и снятого с него оформления.

    Смещения разметки считаются по уже нормализованному тексту, поэтому текст и разметка
    строятся за один проход. Пустая и вышедшая за границы текста разметка отбрасывается:
    домен такую не принимает.
    """

    __slots__ = (
        "_anchor_ids",
        "_annotations",
        "_footnote_ids",
        "_length",
        "_pieces",
        "_resource_path",
    )

    def __init__(self, *, resource_path: str, footnote_ids: frozenset[str]) -> None:
        self._resource_path = resource_path
        self._footnote_ids = footnote_ids
        self._pieces: list[str] = []
        self._length = 0
        self._annotations: list[tuple[InlineAnnotationType, int, int, str | None]] = []
        self._anchor_ids: list[str] = []

    def add_text(self, raw: str | None) -> None:
        """Добавляет текстовый узел, схлопывая пробельные последовательности."""
        if not raw:
            return
        text = _COLLAPSIBLE_WHITESPACE.sub(" ", raw)
        if text.startswith(" ") and self._ends_with_separator():
            text = text[1:]
        if text:
            self._pieces.append(text)
            self._length += len(text)

    def add_element(self, element: etree._Element, *, with_tail: bool = True) -> None:
        """Добавляет inline-элемент со всем его содержимым и, по умолчанию, хвостом."""
        tag = local_name(element)
        if tag is None:
            self.add_text(element.tail if with_tail else None)
            return
        if tag in _NON_TEXT_TAGS:
            self.add_text(element.tail if with_tail else None)
            return
        if tag in _LINE_BREAK_TAGS:
            self._add_line_break()
            self.add_text(element.tail if with_tail else None)
            return
        self._collect_anchor_id(element)
        start = self._length
        self.add_children(element)
        self._register_annotation(element, tag=tag, start=start)
        if with_tail:
            self.add_text(element.tail)

    def add_children(self, element: etree._Element) -> None:
        """Добавляет собственный текст элемента и всё его содержимое без хвоста."""
        self.add_text(element.text)
        for child in element:
            self.add_element(child)

    def build(self) -> InlineContent:
        """Собирает текст блока и разметку, отбрасывая краевые пробелы."""
        raw = "".join(self._pieces)
        text = raw.strip()
        shift = len(raw) - len(raw.lstrip())
        annotations = tuple(
            annotation
            for annotation in (
                _build_annotation(type_, start - shift, end - shift, target, len(text))
                for type_, start, end, target in self._annotations
            )
            if annotation is not None
        )
        return InlineContent(
            text=text, annotations=annotations, anchor_ids=tuple(dict.fromkeys(self._anchor_ids))
        )

    def _ends_with_separator(self) -> bool:
        if not self._pieces:
            return True
        return self._pieces[-1][-1] in " \n"

    def _add_line_break(self) -> None:
        while self._pieces and self._pieces[-1].endswith(" "):
            trimmed = self._pieces[-1][:-1]
            if trimmed:
                self._pieces[-1] = trimmed
            else:
                self._pieces.pop()
            self._length -= 1
        if self._pieces:
            self._pieces.append("\n")
            self._length += 1

    def _collect_anchor_id(self, element: etree._Element) -> None:
        element_id = element.get("id")
        if element_id:
            self._anchor_ids.append(element_id)

    def _register_annotation(self, element: etree._Element, *, tag: str, start: int) -> None:
        end = self._length
        if end <= start:
            return
        if tag == "a":
            annotation = self._link_annotation(element)
            if annotation is not None:
                type_, target = annotation
                self._annotations.append((type_, start, end, target))
            return
        annotation_type = _ANNOTATION_TAGS.get(tag)
        if annotation_type is not None:
            self._annotations.append((annotation_type, start, end, None))

    def _link_annotation(
        self, element: etree._Element
    ) -> tuple[InlineAnnotationType, str] | None:
        href = element.get("href")
        if not href:
            return None
        target = resolve_link_target(href, resource_path=self._resource_path)
        if is_footnote_reference(
            element, resource_path=self._resource_path, footnote_ids=self._footnote_ids
        ):
            return InlineAnnotationType.FOOTNOTE_REFERENCE, target
        return InlineAnnotationType.LINK, target


def _build_annotation(
    type_: InlineAnnotationType, start: int, end: int, target: str | None, limit: int
) -> InlineAnnotation | None:
    """Разметка, обрезанная по границам итогового текста, либо `None`, если она пуста."""
    start = max(start, 0)
    end = min(end, limit)
    if end <= start:
        return None
    return InlineAnnotation(type=type_, start=start, end=end, target=target)


def _epub_type_tokens(element: etree._Element) -> frozenset[str]:
    """Значения `epub:type` элемента как набор токенов."""
    value = element.get(f"{{{_EPUB_NAMESPACE}}}type") or element.get("epub:type")
    return frozenset(value.split()) if value else frozenset()


def _class_tokens(element: etree._Element) -> frozenset[str]:
    """Классы элемента как набор токенов."""
    value = element.get("class")
    return frozenset(value.split()) if value else frozenset()


__all__ = [
    "InlineContent",
    "InlineContentBuilder",
    "footnote_element_ids",
    "is_footnote_element",
    "is_footnote_reference",
    "local_name",
    "resolve_link_target",
]
