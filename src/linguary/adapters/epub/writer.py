"""Экспорт EPUB: сборка новой книги и встроенные технические проверки (PRD § 20).

Реализация порта `linguary.application.ports.epub.EpubWriter`. Сборка симметрична импорту:
одна `DocumentSection` — один XHTML-документ порядка чтения, и он сохраняет путь исходного
документа, поэтому внутренние ссылки книги остаются рабочими без переписывания адресов.

Глава собирается в порядке PRD § 20: `Previously in Chapter ...` из сводки предыдущей главы,
`Before You Read`, затем каждый оригинальный блок, за ним перевод или словарные подсказки
и `Language Notes` этого блока. LLM в сборке не участвует: экспортёр раскладывает готовые
данные по разметке.

Блок `PRESERVE_OPAQUE` переносится исходной разметкой из EPUB книги — иначе теряется всё,
внутрь чего импортёр не спускался: таблицы целиком и изображения внутри них. Опаковая копия
делается только для блока без потомков: у блока-сноски с вложенными абзацами копия
продублировала бы и текст, и идентификаторы.

Порядок гарантий такой: сборка сама приводит книгу в корректное состояние (неразрешимая
внутренняя ссылка разворачивается в текст, повторный `id` снимается, `<img>` без файла
убирается — каждый случай с предупреждением), а `verify_exported_epub` независимо проверяет
уже записанный файл. Ошибка встроенных проверок — `EpubExportFatalError`, и недособранный
артефакт на диске не остаётся.
"""

from __future__ import annotations

import posixpath
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from lxml import etree

from linguary.adapters.epub.inline import local_name
from linguary.adapters.epub.resources import package_path
from linguary.application.ports.epub import (
    EpubExportFatalError,
    EpubImportFatalError,
    EpubWriteRequest,
    EpubWriter,
    ExportedBook,
    ExportWarning,
    ExportWarningKind,
)
from linguary.domain import (
    BlockProcessingResult,
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    InlineAnnotation,
    InlineAnnotationType,
    SupportStatus,
    build_language_note_id,
    translation_segments,
)

_XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
_EPUB_NAMESPACE = "http://www.idpf.org/2007/ops"
_OPF_NAMESPACE = "http://www.idpf.org/2007/opf"
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
_CONTAINER_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:container"

#: Каталог пакета внутри архива: документы и ресурсы лежат относительно него.
_PACKAGE_DIRECTORY = "OEBPS"
_PACKAGE_FILE = "content.opf"
_NAVIGATION_FILE = "nav.xhtml"
_STYLESHEET_FILE = "linguary.css"
_MIMETYPE = "application/epub+zip"

#: Оглавление исходной книги в формате EPUB2. Итоговая книга собирается как EPUB3 с
#: собственным `nav`, а старый NCX описывает уже несуществующую структуру.
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"

#: Статусы поддержки, при которых ресурс попадает в итоговую книгу.
_EXPORTED_STATUSES = frozenset({SupportStatus.SUPPORTED, SupportStatus.PRESERVE_OPAQUE})

#: Теги блоков по типу; заголовок берёт исходный уровень, если он известен.
_BLOCK_TAGS: Mapping[BlockType, str] = {
    BlockType.PARAGRAPH: "p",
    BlockType.HEADING: "h2",
    BlockType.QUOTE: "p",
    BlockType.LIST_ITEM: "li",
    BlockType.IMAGE: "img",
    BlockType.FOOTNOTE: "aside",
    BlockType.POEM: "p",
    BlockType.TABLE: "div",
    BlockType.PAGE_BREAK: "span",
    BlockType.OTHER: "div",
}

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

#: Блоки, идущие подряд, собираются под общим родителем: импортёр разбирает список
#: и цитату по элементам, и без обёртки книга получила бы список из одного пункта
#: на каждый пункт.
_GROUPING_TAGS: Mapping[BlockType, str] = {
    BlockType.LIST_ITEM: "ul",
    BlockType.QUOTE: "blockquote",
}

#: Элементы, внутрь которых текст блока не пишется: их содержимое — сам элемент.
_VOID_TAGS = frozenset({"img", "hr", "br"})

#: Теги оформления по видам встроенной разметки.
_ANNOTATION_TAGS: Mapping[InlineAnnotationType, str] = {
    InlineAnnotationType.ITALIC: "i",
    InlineAnnotationType.BOLD: "b",
    InlineAnnotationType.EMPHASIS: "em",
    InlineAnnotationType.LINK: "a",
    InlineAnnotationType.FOOTNOTE_REFERENCE: "a",
}

#: Оформление помощи (PRD § 20): оригинал остаётся основным слоем, помощь тише его —
#: меньшим кеглем, с отступом и левой полосой, а не одним лишь цветом.
_STYLESHEET = """/* Linguary: оформление учебных слоёв поверх оригинала книги. */
.linguary-previously,
.linguary-before-you-read {
  margin: 1.5em 0;
  padding: 0.2em 0 0.2em 0.9em;
  border-left: 0.2em solid;
  font-size: 0.9em;
}

.linguary-translation,
.linguary-glossary,
.linguary-language-note {
  margin: 0.4em 0 1em 1.2em;
  padding-left: 0.9em;
  border-left: 0.15em solid;
  font-size: 0.88em;
}

.linguary-glossary {
  list-style: none;
}

.linguary-language-note {
  font-style: italic;
}

.linguary-language-note .linguary-excerpt {
  font-style: normal;
}

.linguary-anchors {
  display: none;
}
"""


class EpubExportWriter(EpubWriter):
    """Сборка итогового EPUB из канонической модели, ресурсов и результатов обработки."""

    def write(self, request: EpubWriteRequest) -> ExportedBook:
        """Собирает книгу и проверяет собранный файл; ошибка проверок — `FATAL`.

        Книга собирается рядом с итоговым файлом и занимает его место только после
        успешных встроенных проверок: неудачная сборка не должна уносить с собой
        прошлый удачный экспорт той же книги.
        """
        warnings = _ExportWarnings()
        payloads = dict(_resource_payloads(request, warnings))
        with _OpaqueSource(request.source_epub_path) as opaque:
            chapters = _render_chapters(request, opaque=opaque, warnings=warnings)
        known_files = (
            set(payloads)
            | {chapter.href for chapter in chapters}
            | {_NAVIGATION_FILE, _STYLESHEET_FILE}
        )
        _resolve_references(chapters, known_files=known_files, warnings=warnings)
        _append_book_payloads(
            payloads,
            document=request.document,
            chapters=chapters,
            media_types=_resource_media_types(request),
        )
        target = Path(request.target_path)
        draft = target.with_name(f"{target.name}.part")
        _write_archive(draft, payloads=payloads)
        failures = verify_exported_epub(
            draft, expected_spine=tuple(chapter.href for chapter in chapters)
        )
        if failures:
            draft.unlink(missing_ok=True)
            raise EpubExportFatalError(
                "собранный EPUB не прошёл встроенные технические проверки: "
                + "; ".join(f"{failure.kind}: {failure.message}" for failure in failures)
            )
        draft.replace(target)
        return ExportedBook(path=str(target), warnings=warnings.collected())


@dataclass(slots=True)
class _Chapter:
    """Готовый документ главы: путь внутри книги, заголовок и дерево XHTML."""

    section_id: str
    href: str
    title: str
    root: etree._Element


@dataclass(slots=True)
class _Occurrences:
    """Случаи одного вида расхождения в одной главе: пример и их число."""

    example: str
    count: int = 1


@dataclass(slots=True)
class _ExportWarnings:
    """Замечания сборки, сведённые по виду и главе.

    Отдельная строка на каждый пропущенный разрыв страницы превратила бы список
    предупреждений в шум: в изданиях Gutenberg таких конструкций сотни. Поэтому случаи
    копятся, а пользователь получает по одному предупреждению на вид и главу — с примером
    и числом затронутых мест.
    """

    _occurrences: dict[tuple[ExportWarningKind, str | None], _Occurrences] = field(
        default_factory=dict
    )

    def record(
        self, kind: ExportWarningKind, *, section_id: str | None, detail: str
    ) -> None:
        """Запоминает один случай расхождения итоговой книги с исходной."""
        occurrences = self._occurrences.get((kind, section_id))
        if occurrences is None:
            self._occurrences[(kind, section_id)] = _Occurrences(example=detail)
            return
        occurrences.count += 1

    def collected(self) -> tuple[ExportWarning, ...]:
        """Предупреждения экспорта по накопленным случаям."""
        return tuple(
            ExportWarning(
                kind=kind,
                message=(
                    occurrences.example
                    if occurrences.count == 1
                    else f"{occurrences.example}. Всего таких случаев: {occurrences.count}"
                ),
                section_id=section_id,
            )
            for (kind, section_id), occurrences in self._occurrences.items()
        )


def _resource_payloads(
    request: EpubWriteRequest, warnings: _ExportWarnings
) -> Iterable[tuple[str, bytes]]:
    """Содержимое ресурсов книги для нового архива.

    Ресурс без копии в хранилище книгу не ломает: так выглядит книга, импортированная
    без хранилища ресурсов, — она собирается без иллюстраций, и пользователь видит
    предупреждение. Объявленная копия, которой нет на диске, — другое дело: хранилище
    повреждено, и собирать книгу с потерянным обязательным ресурсом нельзя.
    """
    for resource in request.document.resources:
        if resource.support_status not in _EXPORTED_STATUSES:
            continue
        if resource.media_type == _NCX_MEDIA_TYPE:
            continue
        stored = request.stored_resource_paths.get(resource.source_path)
        if stored is None:
            warnings.record(
                ExportWarningKind.MISSING_RESOURCE,
                section_id=None,
                detail=(
                    f"ресурс {resource.source_path} не сохранён в хранилище книги "
                    f"и в итоговую книгу не перенесён"
                ),
            )
            continue
        path = Path(stored)
        if not path.is_file():
            raise EpubExportFatalError(
                f"копия ресурса {resource.source_path} книги {request.document.id} "
                f"отсутствует в хранилище: {stored}"
            )
        yield _contained_name(resource.source_path), path.read_bytes()


def _resource_media_types(request: EpubWriteRequest) -> Mapping[str, str]:
    """Типы содержимого ресурсов книги: их объявляет манифест исходной книги."""
    return {
        _contained_name(resource.source_path): resource.media_type
        for resource in request.document.resources
    }


def _render_chapters(
    request: EpubWriteRequest, *, opaque: _OpaqueSource, warnings: _ExportWarnings
) -> tuple[_Chapter, ...]:
    """Документы глав в порядке чтения.

    Предыдущая глава — последний раздел перед этим, который сам является главой:
    страница сносок пересказа не имеет, и глава после неё вспоминает главу до неё.
    """
    renderer = _ChapterRenderer(request=request, opaque=opaque, warnings=warnings)
    chapters: list[_Chapter] = []
    previous: DocumentSection | None = None
    for number, section in enumerate(request.document.ordered_sections, start=1):
        chapters.append(renderer.render(section, number=number, previous=previous))
        if not section.is_notes_only:
            previous = section
    return tuple(chapters)


class _ChapterRenderer:
    """Сборка одного XHTML-документа главы."""

    def __init__(
        self,
        *,
        request: EpubWriteRequest,
        opaque: _OpaqueSource,
        warnings: _ExportWarnings,
    ) -> None:
        self._request = request
        self._opaque = opaque
        self._warnings = warnings
        self._document = request.document

    def render(
        self, section: DocumentSection, *, number: int, previous: DocumentSection | None
    ) -> _Chapter:
        """Собирает главу целиком: учебные разделы, блоки и слои помощи.

        Страница сносок собирается без разметки главы и без учебных разделов перед
        текстом: её сноски переводятся, но главой она не является.
        """
        href = _contained_name(section.source_resource_path)
        root = _xhtml_document(
            title=section.title,
            language=self._document.source_language,
            stylesheet=_relative_href(href, _STYLESHEET_FILE),
        )
        body = root.find(f"{{{_XHTML_NAMESPACE}}}body")
        if section.is_notes_only:
            self._append_blocks(body, section, href=href, number=number)
        else:
            container = _sub(body, "section", {f"{{{_EPUB_NAMESPACE}}}type": "chapter"})
            self._append_previously(container, previous)
            self._append_before_you_read(container, section)
            self._append_blocks(container, section, href=href, number=number)
        return _Chapter(
            section_id=section.id, href=href, title=section.title, root=root
        )

    def _append_previously(
        self, container: etree._Element, previous: DocumentSection | None
    ) -> None:
        """Раздел `Previously in Chapter ...` из сводки предыдущей главы (PRD § 19.2)."""
        if previous is None:
            return
        material = self._request.learning_materials.get(previous.id)
        summary = material.chapter_summary.strip() if material else ""
        if not summary:
            return
        block = _sub(container, "section", {"class": "linguary-previously"})
        _sub(block, "h2").text = _previously_heading(previous.title)
        for paragraph in _paragraphs(summary):
            _append_text(_sub(block, "p"), paragraph)

    def _append_before_you_read(
        self, container: etree._Element, section: DocumentSection
    ) -> None:
        """Раздел `Before You Read` из анализа этой главы (PRD § 19.1)."""
        material = self._request.learning_materials.get(section.id)
        if material is None or not material.before_you_read:
            return
        block = _sub(container, "section", {"class": "linguary-before-you-read"})
        _sub(block, "h2").text = "Before You Read"
        definitions = _sub(block, "dl")
        for entry in material.before_you_read:
            _sub(definitions, "dt").text = entry.expression
            _sub(definitions, "dd").text = entry.explanation

    def _append_blocks(
        self,
        container: etree._Element,
        section: DocumentSection,
        *,
        href: str,
        number: int,
    ) -> None:
        """Блоки главы вместе со слоями помощи, в порядке исходного документа."""
        blocks = section.ordered_blocks
        children: dict[str, list[DocumentBlock]] = {}
        for block in blocks:
            if block.parent_id is not None:
                children.setdefault(block.parent_id, []).append(block)
        state = _SectionState(section=section, href=href, number=number)
        roots = tuple(block for block in blocks if block.parent_id is None)
        self._append_siblings(container, roots, children=children, state=state)

    def _append_siblings(
        self,
        parent: etree._Element,
        blocks: Sequence[DocumentBlock],
        *,
        children: Mapping[str, list[DocumentBlock]],
        state: _SectionState,
        parent_type: BlockType | None = None,
    ) -> None:
        """Соседние блоки одного уровня; подряд идущие пункты и цитаты — под обёрткой."""
        group: etree._Element | None = None
        grouped: BlockType | None = None
        for block in blocks:
            wrapper = _GROUPING_TAGS.get(block.type)
            if wrapper is None or block.type is not grouped:
                group, grouped = None, None
            if wrapper is not None and group is None:
                group, grouped = _sub(parent, wrapper), block.type
            self._append_block(
                group if group is not None else parent,
                block,
                children=children,
                state=state,
                parent_type=parent_type,
            )

    def _append_block(
        self,
        parent: etree._Element,
        block: DocumentBlock,
        *,
        children: Mapping[str, list[DocumentBlock]],
        state: _SectionState,
        parent_type: BlockType | None,
    ) -> None:
        """Один блок: оригинал, вложенные блоки и слой помощи под ним."""
        nested = children.get(block.id, ())
        element = self._append_original(
            parent, block, nested=bool(nested), state=state, parent_type=parent_type
        )
        if nested and element is not None:
            self._append_siblings(
                element,
                nested,
                children=children,
                state=state,
                parent_type=block.type,
            )
        help_parent = (
            element
            if block.type is BlockType.LIST_ITEM and element is not None
            else parent
        )
        self._append_help(help_parent, block, state=state)

    def _append_original(
        self,
        parent: etree._Element,
        block: DocumentBlock,
        *,
        nested: bool,
        state: _SectionState,
        parent_type: BlockType | None,
    ) -> etree._Element | None:
        """Оригинальный блок; `PRESERVE_OPAQUE` без потомков — исходной разметкой."""
        if block.support_status is SupportStatus.PRESERVE_OPAQUE and not nested:
            copied = self._opaque.element(block)
            if copied is not None:
                parent.append(copied)
                self._attach_anchors(parent, copied, block, state=state, inside=False)
                return copied
            if not block.text and not _anchors(block):
                return None
            self._warnings.record(
                ExportWarningKind.OPAQUE_FRAGMENT_LOST,
                section_id=state.section.id,
                detail=(
                    f"исходная разметка конструкции "
                    f"{block.source_reference.structural_path} документа "
                    f"{block.source_reference.resource_path} недоступна, "
                    f"перенесён только её текст"
                ),
            )
        element = self._build_element(
            parent, block, state=state, parent_type=parent_type
        )
        self._attach_anchors(parent, element, block, state=state, inside=True)
        return element

    def _build_element(
        self,
        parent: etree._Element,
        block: DocumentBlock,
        *,
        state: _SectionState,
        parent_type: BlockType | None,
    ) -> etree._Element:
        """Элемент блока по его типу вместе с текстом и встроенной разметкой."""
        tag = _block_tag(block, parent_type=parent_type)
        element = _sub(parent, tag, _block_attributes(block, tag=tag))
        host = element
        if tag == "aside" and block.text:
            host = _sub(element, "p")
        if block.text and local_name(host) not in _VOID_TAGS:
            _append_inline(host, block.text, _nestable(block.inline_annotations), state)
        return element

    def _attach_anchors(
        self,
        parent: etree._Element,
        element: etree._Element,
        block: DocumentBlock,
        *,
        state: _SectionState,
        inside: bool,
    ) -> None:
        """Якоря блока: по ним на него ссылается сама книга.

        Первый якорь становится идентификатором элемента, остальные остаются пустыми
        `span`: терять их нельзя, иначе внутренняя ссылка книги станет неразрешимой.
        """
        pending = [
            anchor
            for anchor in _anchors(block)
            if anchor not in state.used_ids and not _contains_id(element, anchor)
        ]
        if inside and pending and element.get("id") is None:
            element.set("id", pending[0])
            state.used_ids.add(pending[0])
            pending = pending[1:]
        if not pending:
            return
        host = element
        if not inside or local_name(element) in _VOID_TAGS:
            host = _element("div", {"class": "linguary-anchors"})
            parent.insert(parent.index(element), host)
        for index, anchor in enumerate(pending):
            host.insert(index, _element("span", {"id": anchor}))
            state.used_ids.add(anchor)

    def _append_help(
        self, parent: etree._Element, block: DocumentBlock, *, state: _SectionState
    ) -> None:
        """Перевод либо словарные подсказки и `Language Notes` этого блока (PRD § 20)."""
        result = self._request.block_results.get(block.id)
        if result is not None:
            self._append_result(parent, result, block=block, state=state)
        material = self._request.learning_materials.get(state.section.id)
        if material is None:
            return
        for note in material.notes_for(block.id):
            state.notes += 1
            aside = _sub(
                parent,
                "aside",
                {
                    "class": "linguary-language-note",
                    "id": _unique_id(
                        build_language_note_id(
                            chapter_number=state.number, ordinal=state.notes
                        ),
                        state.used_ids,
                    ),
                },
            )
            _append_text(
                _sub(aside, "p", {"class": "linguary-excerpt"}), note.original_excerpt
            )
            _append_text(_sub(aside, "p"), note.note)

    def _append_result(
        self,
        parent: etree._Element,
        result: BlockProcessingResult,
        *,
        block: DocumentBlock,
        state: _SectionState,
    ) -> None:
        """Перевод блока либо его словарные подсказки.

        Метка `[^N]` в переводе становится ссылкой на ту же сноску, что и `N`-я ссылка
        на сноску оригинала, с тем же видимым текстом. Идентификатор ссылки оригинала
        не повторяется: обратная ссылка из сноски по-прежнему ведёт к оригиналу.
        """
        if result.translated_text:
            translation = _sub(
                parent,
                "p",
                {
                    "class": "linguary-translation",
                    "lang": self._document.target_language,
                },
            )
            for segment in translation_segments(result.translated_text, block):
                if isinstance(segment, str):
                    _append_text(translation, segment)
                    continue
                link = _sub(translation, "a", _target_attributes(segment, state))
                link.text = block.text[segment.start : segment.end]
        if not result.glossary_hints:
            return
        hints = _sub(parent, "ul", {"class": "linguary-glossary"})
        for hint in result.glossary_hints:
            item = _sub(hints, "li")
            _sub(item, "span", {"class": "linguary-excerpt"}).text = hint.original
            tail = f" — {hint.translation}"
            if hint.note:
                tail = f"{tail} ({hint.note})"
            _append_text(item, tail)


@dataclass(slots=True)
class _SectionState:
    """Состояние сборки одной главы: занятые идентификаторы и счётчик заметок."""

    section: DocumentSection
    href: str
    number: int
    used_ids: set[str] = field(default_factory=set)
    notes: int = 0


class _OpaqueSource:
    """Исходная книга как источник разметки для блоков `PRESERVE_OPAQUE`.

    Недоступный или неразбираемый исходный файл экспорт не останавливает: блок переносится
    текстом, а пользователь получает предупреждение. Книга остаётся собираемой и тогда,
    когда пользователь убрал исходный EPUB из файловой системы.
    """

    def __init__(self, source_path: str | None) -> None:
        self._archive: zipfile.ZipFile | None = None
        self._directory = ""
        self._documents: dict[str, etree._Element | None] = {}
        if not source_path or not Path(source_path).is_file():
            return
        try:
            archive = zipfile.ZipFile(source_path)
        except (OSError, zipfile.BadZipFile):
            return
        try:
            self._directory = posixpath.dirname(package_path(archive))
        except (
            OSError,
            KeyError,
            zipfile.BadZipFile,
            etree.XMLSyntaxError,
            EpubImportFatalError,
        ):
            archive.close()
            return
        self._archive = archive

    def __enter__(self) -> _OpaqueSource:
        return self

    def __exit__(self, *exception: object) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def element(self, block: DocumentBlock) -> etree._Element | None:
        """Копия исходного элемента блока либо `None`, если её не достать."""
        root = self._document(block.source_reference.resource_path)
        if root is None:
            return None
        found = _element_at(root, block.source_reference.structural_path)
        if found is None:
            return None
        return _in_xhtml_namespace(deepcopy(found))

    def _document(self, resource_path: str) -> etree._Element | None:
        """Разобранный исходный документ книги; результат разбора кешируется."""
        if resource_path in self._documents:
            return self._documents[resource_path]
        root = self._parse(resource_path)
        self._documents[resource_path] = root
        return root

    def _parse(self, resource_path: str) -> etree._Element | None:
        if self._archive is None:
            return None
        name = posixpath.normpath(posixpath.join(self._directory, resource_path))
        try:
            content = self._archive.read(name)
        except (KeyError, OSError, zipfile.BadZipFile):
            return None
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
        try:
            return etree.fromstring(content, parser=parser)
        except etree.XMLSyntaxError:
            return None


def _resolve_references(
    chapters: Sequence[_Chapter], *, known_files: set[str], warnings: _ExportWarnings
) -> None:
    """Приводит книгу к состоянию, в котором все внутренние связи разрешимы.

    Исходная книга может ссылаться на то, чего в канонической модели нет: на пустую
    декорацию, отброшенную импортом, или на документ вне порядка чтения. Оставить такую
    ссылку значит собрать книгу, которая не проходит собственные проверки, поэтому ссылка
    разворачивается в обычный текст, а пользователь получает предупреждение.
    """
    anchors = {chapter.href: _deduplicate_ids(chapter, warnings) for chapter in chapters}
    for chapter in chapters:
        for element in tuple(chapter.root.iter()):
            tag = local_name(element)
            if tag == "a":
                _resolve_link(
                    element,
                    chapter=chapter,
                    known_files=known_files,
                    anchors=anchors,
                    warnings=warnings,
                )
            elif tag == "img":
                _resolve_image(
                    element, chapter=chapter, known_files=known_files, warnings=warnings
                )


def _deduplicate_ids(chapter: _Chapter, warnings: _ExportWarnings) -> set[str]:
    """Идентификаторы документа; повторный `id` снимается с более позднего элемента."""
    seen: set[str] = set()
    for element in chapter.root.iter():
        identifier = element.get("id")
        if identifier is None:
            continue
        if identifier in seen:
            del element.attrib["id"]
            warnings.record(
                ExportWarningKind.UNRESOLVED_LINK,
                section_id=chapter.section_id,
                detail=(
                    f"идентификатор {identifier} встречается в документе {chapter.href} "
                    f"повторно; сохранено только первое вхождение"
                ),
            )
            continue
        seen.add(identifier)
    return seen


def _resolve_link(
    element: etree._Element,
    *,
    chapter: _Chapter,
    known_files: set[str],
    anchors: Mapping[str, set[str]],
    warnings: _ExportWarnings,
) -> None:
    """Оставляет ссылку, если она разрешима, иначе разворачивает её в текст."""
    href = element.get("href")
    if not href or _is_external(href):
        return
    path, fragment = _resolve_target(href, base=chapter.href)
    resolved = path in known_files and (
        not fragment or fragment in anchors.get(path, frozenset())
    )
    if resolved:
        return
    _unwrap(element)
    warnings.record(
        ExportWarningKind.UNRESOLVED_LINK,
        section_id=chapter.section_id,
        detail=(
            f"внутренняя ссылка на {href} из документа {chapter.href} никуда не ведёт "
            f"в итоговой книге; текст ссылки сохранён"
        ),
    )


def _resolve_image(
    element: etree._Element,
    *,
    chapter: _Chapter,
    known_files: set[str],
    warnings: _ExportWarnings,
) -> None:
    """Убирает изображение, файла которого в итоговой книге нет."""
    source = element.get("src")
    if source and _is_external(source):
        return
    if source:
        path, _ = _resolve_target(source, base=chapter.href)
        if path in known_files:
            return
    _unwrap(element)
    warnings.record(
        ExportWarningKind.MISSING_RESOURCE,
        section_id=chapter.section_id,
        detail=(
            f"изображение {source or 'без адреса файла'} из документа {chapter.href} "
            f"отсутствует в итоговой книге"
        ),
    )


def _append_book_payloads(
    payloads: dict[str, bytes],
    *,
    document: CanonicalDocument,
    chapters: Sequence[_Chapter],
    media_types: Mapping[str, str],
) -> None:
    """Добавляет к ресурсам книги документы глав, навигацию, стиль и файл пакета."""
    payloads.update(
        {chapter.href: _serialize(chapter.root) for chapter in chapters}
    )
    payloads[_STYLESHEET_FILE] = _STYLESHEET.encode("utf-8")
    payloads[_NAVIGATION_FILE] = _serialize(_navigation(chapters, document=document))
    payloads[_PACKAGE_FILE] = _serialize(
        _package(
            document=document,
            chapters=chapters,
            payloads=payloads,
            media_types=media_types,
        )
    )


def _navigation(
    chapters: Sequence[_Chapter], *, document: CanonicalDocument
) -> etree._Element:
    """Документ навигации EPUB3: оглавление ведёт к главам в порядке чтения."""
    root = _xhtml_document(
        title=document.metadata.title,
        language=document.source_language,
        stylesheet=_STYLESHEET_FILE,
    )
    body = root.find(f"{{{_XHTML_NAMESPACE}}}body")
    navigation = _sub(
        body, "nav", {f"{{{_EPUB_NAMESPACE}}}type": "toc", "id": "toc"}
    )
    _sub(navigation, "h1").text = "Contents"
    listing = _sub(navigation, "ol")
    for chapter in chapters:
        link = _sub(_sub(listing, "li"), "a", {"href": chapter.href})
        link.text = chapter.title
    return root


def _package(
    *,
    document: CanonicalDocument,
    chapters: Sequence[_Chapter],
    payloads: Mapping[str, bytes],
    media_types: Mapping[str, str],
) -> etree._Element:
    """Файл пакета OPF: метаданные книги, манифест и порядок чтения."""
    root = _element(
        f"{{{_OPF_NAMESPACE}}}package",
        {"version": "3.0", "unique-identifier": "pub-id"},
        nsmap={None: _OPF_NAMESPACE, "dc": _DC_NAMESPACE},
    )
    _append_metadata(_sub(root, f"{{{_OPF_NAMESPACE}}}metadata"), document)
    manifest = _sub(root, f"{{{_OPF_NAMESPACE}}}manifest")
    identifiers = _manifest_items(
        manifest, chapters=chapters, payloads=payloads, media_types=media_types
    )
    spine = _sub(root, f"{{{_OPF_NAMESPACE}}}spine")
    for chapter in chapters:
        _sub(spine, f"{{{_OPF_NAMESPACE}}}itemref", {"idref": identifiers[chapter.href]})
    return root


def _append_metadata(metadata: etree._Element, document: CanonicalDocument) -> None:
    """Метаданные книги: сохраняются те, что прочитал импортёр (PRD § 20)."""
    book_metadata = document.metadata
    identifier = _sub(metadata, f"{{{_DC_NAMESPACE}}}identifier", {"id": "pub-id"})
    identifier.text = book_metadata.identifier or f"urn:uuid:{document.id}"
    _sub(metadata, f"{{{_DC_NAMESPACE}}}title").text = book_metadata.title
    _sub(metadata, f"{{{_DC_NAMESPACE}}}language").text = document.source_language
    for author in book_metadata.authors:
        _sub(metadata, f"{{{_DC_NAMESPACE}}}creator").text = author
    if book_metadata.publisher:
        _sub(metadata, f"{{{_DC_NAMESPACE}}}publisher").text = book_metadata.publisher
    modified = _sub(
        metadata, f"{{{_OPF_NAMESPACE}}}meta", {"property": "dcterms:modified"}
    )
    modified.text = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_items(
    manifest: etree._Element,
    *,
    chapters: Sequence[_Chapter],
    payloads: Mapping[str, bytes],
    media_types: Mapping[str, str],
) -> Mapping[str, str]:
    """Элементы манифеста; отдаёт идентификатор каждого файла книги."""
    identifiers: dict[str, str] = {}
    for number, href in enumerate(sorted(payloads), start=1):
        item_id = "nav" if href == _NAVIGATION_FILE else f"item-{number}"
        attributes = {
            "id": item_id,
            "href": href,
            "media-type": media_types.get(href) or _media_type(href),
        }
        if href == _NAVIGATION_FILE:
            attributes["properties"] = "nav"
        _sub(manifest, f"{{{_OPF_NAMESPACE}}}item", attributes)
        identifiers[href] = item_id
    return {chapter.href: identifiers[chapter.href] for chapter in chapters}


def _write_archive(target: Path, *, payloads: Mapping[str, bytes]) -> None:
    """Пишет архив книги: `mimetype` первым и без сжатия, затем контейнер и содержимое."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mimetype", _MIMETYPE, zipfile.ZIP_STORED)
            archive.writestr(
                "META-INF/container.xml",
                _serialize(_container(f"{_PACKAGE_DIRECTORY}/{_PACKAGE_FILE}")),
            )
            for href, payload in sorted(payloads.items()):
                archive.writestr(f"{_PACKAGE_DIRECTORY}/{href}", payload)
    except OSError as error:
        target.unlink(missing_ok=True)
        raise EpubExportFatalError(
            f"итоговый EPUB {target} нельзя записать: {error}"
        ) from error


def _container(package_href: str) -> etree._Element:
    """`META-INF/container.xml`: где в архиве лежит файл пакета."""
    root = _element(
        f"{{{_CONTAINER_NAMESPACE}}}container",
        {"version": "1.0"},
        nsmap={None: _CONTAINER_NAMESPACE},
    )
    _sub(
        _sub(root, f"{{{_CONTAINER_NAMESPACE}}}rootfiles"),
        f"{{{_CONTAINER_NAMESPACE}}}rootfile",
        {
            "full-path": package_href,
            "media-type": "application/oebps-package+xml",
        },
    )
    return root


class ExportCheckKind(StrEnum):
    """Что именно не сошлось во встроенной технической проверке (PRD § 20)."""

    CONTAINER = "container"
    PACKAGE = "package"
    MANIFEST = "manifest"
    SPINE = "spine"
    NAVIGATION = "navigation"
    RESOURCE = "resource"
    LINK = "link"
    XHTML = "xhtml"
    CHAPTER_ORDER = "chapter_order"


@dataclass(frozen=True, slots=True)
class ExportCheckFailure:
    """Одна несошедшаяся проверка собранного файла."""

    kind: ExportCheckKind
    message: str


def verify_exported_epub(
    path: Path | str, *, expected_spine: Sequence[str] = ()
) -> tuple[ExportCheckFailure, ...]:
    """Встроенные технические проверки собранного EPUB (PRD § 20).

    Проверяется уже записанный файл, а не промежуточное состояние сборки: так проверка
    видит ровно то, что увидит читалка, и её можно натравить на чужой или намеренно
    испорченный файл. Внешние инструменты не нужны, проверка выполняется всегда.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        return (
            ExportCheckFailure(
                ExportCheckKind.CONTAINER, f"файл {path} не является архивом EPUB: {error}"
            ),
        )
    with archive:
        return _verify_archive(archive, expected_spine=tuple(expected_spine))


def _verify_archive(
    archive: zipfile.ZipFile, *, expected_spine: Sequence[str]
) -> tuple[ExportCheckFailure, ...]:
    """Проверки открытого архива книги."""
    names = frozenset(archive.namelist())
    failures: list[ExportCheckFailure] = []
    if "mimetype" not in names:
        failures.append(
            ExportCheckFailure(ExportCheckKind.CONTAINER, "в архиве нет файла mimetype")
        )
    package_href = _package_href(archive, names, failures)
    if package_href is None:
        return tuple(failures)
    package = _parse(archive, package_href)
    if package is None:
        failures.append(
            ExportCheckFailure(
                ExportCheckKind.PACKAGE, f"файл пакета {package_href} нельзя разобрать"
            )
        )
        return tuple(failures)
    directory = posixpath.dirname(package_href)
    manifest = _manifest(package, directory=directory, failures=failures)
    spine = _spine(package, manifest=manifest, failures=failures)
    failures.extend(_missing_manifest_files(manifest, names=names))
    documents = _parsed_documents(archive, manifest=manifest, failures=failures)
    failures.extend(
        _navigation_failures(manifest=manifest, documents=documents, spine=spine)
    )
    failures.extend(
        _reference_failures(documents=documents, names=names, manifest=manifest)
    )
    failures.extend(
        _order_failures(spine=spine, documents=documents, expected=expected_spine)
    )
    return tuple(failures)


def _package_href(
    archive: zipfile.ZipFile, names: frozenset[str], failures: list[ExportCheckFailure]
) -> str | None:
    """Путь к файлу пакета по `META-INF/container.xml`."""
    if "META-INF/container.xml" not in names:
        failures.append(
            ExportCheckFailure(
                ExportCheckKind.CONTAINER, "в архиве нет META-INF/container.xml"
            )
        )
        return None
    container = _parse(archive, "META-INF/container.xml")
    if container is None:
        failures.append(
            ExportCheckFailure(
                ExportCheckKind.CONTAINER, "META-INF/container.xml нельзя разобрать"
            )
        )
        return None
    for element in container.iter():
        if local_name(element) == "rootfile" and element.get("full-path"):
            href = element.get("full-path", "")
            if href not in names:
                failures.append(
                    ExportCheckFailure(
                        ExportCheckKind.CONTAINER,
                        f"объявленный файл пакета {href} отсутствует в архиве",
                    )
                )
                return None
            return href
    failures.append(
        ExportCheckFailure(
            ExportCheckKind.CONTAINER, "в META-INF/container.xml не объявлен файл пакета"
        )
    )
    return None


@dataclass(frozen=True, slots=True)
class _ManifestItem:
    """Элемент манифеста: путь внутри архива и тип содержимого."""

    item_id: str
    name: str
    href: str
    media_type: str
    properties: str


def _manifest(
    package: etree._Element, *, directory: str, failures: list[ExportCheckFailure]
) -> Mapping[str, _ManifestItem]:
    """Элементы манифеста по их идентификаторам."""
    items: dict[str, _ManifestItem] = {}
    for element in package.iter():
        if local_name(element) != "item":
            continue
        item_id = element.get("id") or ""
        href = element.get("href") or ""
        if not item_id or not href:
            failures.append(
                ExportCheckFailure(
                    ExportCheckKind.MANIFEST,
                    "элемент манифеста без идентификатора или без адреса файла",
                )
            )
            continue
        if item_id in items:
            failures.append(
                ExportCheckFailure(
                    ExportCheckKind.MANIFEST,
                    f"идентификатор {item_id} встречается в манифесте дважды",
                )
            )
            continue
        items[item_id] = _ManifestItem(
            item_id=item_id,
            name=posixpath.normpath(posixpath.join(directory, href)),
            href=href,
            media_type=element.get("media-type") or "",
            properties=element.get("properties") or "",
        )
    if not items:
        failures.append(
            ExportCheckFailure(ExportCheckKind.MANIFEST, "манифест книги пуст")
        )
    return items


def _spine(
    package: etree._Element,
    *,
    manifest: Mapping[str, _ManifestItem],
    failures: list[ExportCheckFailure],
) -> tuple[_ManifestItem, ...]:
    """Документы порядка чтения; ссылка мимо манифеста — несошедшаяся проверка."""
    documents: list[_ManifestItem] = []
    for element in package.iter():
        if local_name(element) != "itemref":
            continue
        idref = element.get("idref") or ""
        item = manifest.get(idref)
        if item is None:
            failures.append(
                ExportCheckFailure(
                    ExportCheckKind.SPINE,
                    f"порядок чтения ссылается на {idref!r}, которого нет в манифесте",
                )
            )
            continue
        documents.append(item)
    if not documents:
        failures.append(
            ExportCheckFailure(ExportCheckKind.SPINE, "порядок чтения книги пуст")
        )
    return tuple(documents)


def _missing_manifest_files(
    manifest: Mapping[str, _ManifestItem], *, names: frozenset[str]
) -> list[ExportCheckFailure]:
    """Файлы, объявленные манифестом, но отсутствующие в архиве."""
    return [
        ExportCheckFailure(
            ExportCheckKind.RESOURCE,
            f"файл {item.name} объявлен в манифесте, но отсутствует в архиве",
        )
        for item in manifest.values()
        if item.name not in names
    ]


def _parsed_documents(
    archive: zipfile.ZipFile,
    *,
    manifest: Mapping[str, _ManifestItem],
    failures: list[ExportCheckFailure],
) -> Mapping[str, etree._Element]:
    """Разобранные XHTML-документы книги по их путям в архиве."""
    documents: dict[str, etree._Element] = {}
    for item in manifest.values():
        if item.media_type != "application/xhtml+xml":
            continue
        root = _parse(archive, item.name)
        if root is None:
            failures.append(
                ExportCheckFailure(
                    ExportCheckKind.XHTML,
                    f"документ {item.name} не является корректным XHTML",
                )
            )
            continue
        documents[item.name] = root
        failures.extend(_duplicate_id_failures(item.name, root))
    return documents


def _duplicate_id_failures(
    name: str, root: etree._Element
) -> list[ExportCheckFailure]:
    """Повторяющиеся идентификаторы внутри одного документа."""
    seen: set[str] = set()
    failures: list[ExportCheckFailure] = []
    for element in root.iter():
        identifier = element.get("id")
        if identifier is None:
            continue
        if identifier in seen:
            failures.append(
                ExportCheckFailure(
                    ExportCheckKind.XHTML,
                    f"идентификатор {identifier} встречается в документе {name} дважды",
                )
            )
            continue
        seen.add(identifier)
    return failures


def _navigation_failures(
    *,
    manifest: Mapping[str, _ManifestItem],
    documents: Mapping[str, etree._Element],
    spine: Sequence[_ManifestItem],
) -> list[ExportCheckFailure]:
    """Наличие и содержательность оглавления книги."""
    navigations = [item for item in manifest.values() if "nav" in item.properties.split()]
    if not navigations:
        return [
            ExportCheckFailure(
                ExportCheckKind.NAVIGATION, "в манифесте книги не объявлен документ nav"
            )
        ]
    if len(navigations) > 1:
        return [
            ExportCheckFailure(
                ExportCheckKind.NAVIGATION,
                "в манифесте книги объявлено несколько документов nav",
            )
        ]
    root = documents.get(navigations[0].name)
    if root is None:
        return [
            ExportCheckFailure(
                ExportCheckKind.NAVIGATION,
                f"документ навигации {navigations[0].name} нельзя разобрать",
            )
        ]
    if not _toc_targets(root, base=navigations[0].name):
        return [
            ExportCheckFailure(
                ExportCheckKind.NAVIGATION, "оглавление книги не содержит ни одного пункта"
            )
        ]
    spine_names = {item.name for item in spine}
    return [
        ExportCheckFailure(
            ExportCheckKind.NAVIGATION,
            f"пункт оглавления ведёт на {target}, которого нет в порядке чтения",
        )
        for target in _toc_targets(root, base=navigations[0].name)
        if target not in spine_names
    ]


def _toc_targets(root: etree._Element, *, base: str) -> tuple[str, ...]:
    """Документы, на которые ведут пункты оглавления, в порядке их следования."""
    targets: list[str] = []
    for navigation in root.iter():
        if local_name(navigation) != "nav":
            continue
        types = navigation.get(f"{{{_EPUB_NAMESPACE}}}type", "")
        if "toc" not in types.split():
            continue
        for element in navigation.iter():
            href = element.get("href") if local_name(element) == "a" else None
            if not href or _is_external(href):
                continue
            path, _ = _resolve_target(href, base=base)
            if path not in targets:
                targets.append(path)
    return tuple(targets)


def _reference_failures(
    *,
    documents: Mapping[str, etree._Element],
    names: frozenset[str],
    manifest: Mapping[str, _ManifestItem],
) -> list[ExportCheckFailure]:
    """Разрешимость внутренних ссылок, ссылок на сноски и адресов изображений."""
    anchors = {
        name: {
            identifier
            for element in root.iter()
            if (identifier := element.get("id")) is not None
        }
        for name, root in documents.items()
    }
    declared = {item.name for item in manifest.values()}
    failures: list[ExportCheckFailure] = []
    for name, root in documents.items():
        for element in root.iter():
            tag = local_name(element)
            if tag == "a":
                failures.extend(
                    _link_failures(element, name=name, names=names, anchors=anchors)
                )
            elif tag == "img":
                failures.extend(
                    _image_failures(element, name=name, names=names, declared=declared)
                )
    return failures


def _link_failures(
    element: etree._Element,
    *,
    name: str,
    names: frozenset[str],
    anchors: Mapping[str, set[str]],
) -> list[ExportCheckFailure]:
    """Внутренняя ссылка, которая никуда не ведёт."""
    href = element.get("href")
    if not href or _is_external(href):
        return []
    path, fragment = _resolve_target(href, base=name)
    if path not in names:
        return [
            ExportCheckFailure(
                ExportCheckKind.LINK,
                f"ссылка {href} из документа {name} ведёт на отсутствующий файл {path}",
            )
        ]
    if fragment and fragment not in anchors.get(path, frozenset()):
        return [
            ExportCheckFailure(
                ExportCheckKind.LINK,
                f"ссылка {href} из документа {name} ведёт на несуществующий "
                f"якорь {fragment}",
            )
        ]
    return []


def _image_failures(
    element: etree._Element,
    *,
    name: str,
    names: frozenset[str],
    declared: set[str],
) -> list[ExportCheckFailure]:
    """Изображение без файла в архиве или без объявления в манифесте."""
    source = element.get("src")
    if not source or _is_external(source):
        return []
    path, _ = _resolve_target(source, base=name)
    if path not in names:
        return [
            ExportCheckFailure(
                ExportCheckKind.RESOURCE,
                f"изображение {source} из документа {name} отсутствует в архиве",
            )
        ]
    if path not in declared:
        return [
            ExportCheckFailure(
                ExportCheckKind.RESOURCE,
                f"изображение {source} из документа {name} не объявлено в манифесте",
            )
        ]
    return []


def _order_failures(
    *,
    spine: Sequence[_ManifestItem],
    documents: Mapping[str, etree._Element],
    expected: Sequence[str],
) -> list[ExportCheckFailure]:
    """Порядок глав: он же в порядке чтения и он же в оглавлении."""
    failures: list[ExportCheckFailure] = []
    actual = tuple(item.href for item in spine)
    if expected and actual != tuple(expected):
        failures.append(
            ExportCheckFailure(
                ExportCheckKind.CHAPTER_ORDER,
                f"порядок глав в книге {actual} не совпадает с ожидаемым "
                f"{tuple(expected)}",
            )
        )
    navigation = next(
        (
            (name, root)
            for name, root in documents.items()
            if _toc_targets(root, base=name)
        ),
        None,
    )
    if navigation is None:
        return failures
    name, root = navigation
    targets = _toc_targets(root, base=name)
    spine_names = tuple(item.name for item in spine)
    ordered = tuple(target for target in spine_names if target in set(targets))
    if targets != ordered:
        failures.append(
            ExportCheckFailure(
                ExportCheckKind.CHAPTER_ORDER,
                f"порядок глав в оглавлении {targets} не совпадает с порядком чтения "
                f"{ordered}",
            )
        )
    return failures


def _parse(archive: zipfile.ZipFile, name: str) -> etree._Element | None:
    """Разбирает файл архива как XML; неразбираемый файл даёт `None`."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    try:
        return etree.fromstring(archive.read(name), parser=parser)
    except (KeyError, OSError, zipfile.BadZipFile, etree.XMLSyntaxError):
        return None


def _element(
    tag: str,
    attributes: Mapping[str, str] | None = None,
    *,
    nsmap: Mapping[str | None, str] | None = None,
) -> etree._Element:
    """Элемент XHTML или указанного пространства имён с атрибутами."""
    qualified = tag if tag.startswith("{") else f"{{{_XHTML_NAMESPACE}}}{tag}"
    element = etree.Element(qualified, nsmap=dict(nsmap) if nsmap else None)
    for name, value in (attributes or {}).items():
        element.set(name, value)
    return element


def _sub(
    parent: etree._Element, tag: str, attributes: Mapping[str, str] | None = None
) -> etree._Element:
    """Создаёт дочерний элемент и добавляет его в конец родителя."""
    element = _element(tag, attributes)
    parent.append(element)
    return element


def _xhtml_document(*, title: str, language: str, stylesheet: str) -> etree._Element:
    """Каркас документа книги: заголовок, кодировка и ссылка на стиль."""
    root = _element(
        "html",
        {"lang": language},
        nsmap={None: _XHTML_NAMESPACE, "epub": _EPUB_NAMESPACE},
    )
    head = _sub(root, "head")
    _sub(head, "meta", {"charset": "utf-8"})
    _sub(head, "title").text = title
    _sub(head, "link", {"rel": "stylesheet", "type": "text/css", "href": stylesheet})
    _sub(root, "body")
    return root


def _serialize(root: etree._Element) -> bytes:
    """Сериализует документ книги вместе с объявлением XML."""
    etree.cleanup_namespaces(root)
    return etree.tostring(
        root, xml_declaration=True, encoding="utf-8", pretty_print=True
    )


def _block_tag(block: DocumentBlock, *, parent_type: BlockType | None) -> str:
    """Тег элемента блока.

    Заголовок сохраняет исходный уровень, а абзац внутри сноски остаётся абзацем: тип
    блока внутри сноски унаследован от неё самой, и вторая `aside` внутри первой описывала
    бы ту же сноску дважды.
    """
    if block.type is BlockType.HEADING:
        tag = block.attributes.get("tag", "")
        return tag if tag in _HEADING_TAGS else _BLOCK_TAGS[BlockType.HEADING]
    if block.type is BlockType.FOOTNOTE and parent_type is BlockType.FOOTNOTE:
        return _BLOCK_TAGS[BlockType.PARAGRAPH]
    return _BLOCK_TAGS[block.type]


def _block_attributes(block: DocumentBlock, *, tag: str) -> dict[str, str]:
    """Атрибуты элемента блока: адрес изображения, описание и служебные пометки."""
    attributes: dict[str, str] = {}
    if block.type is BlockType.IMAGE:
        attributes["src"] = block.attributes.get("src", "")
        attributes["alt"] = block.attributes.get("alt", "")
    if tag == "aside":
        attributes[f"{{{_EPUB_NAMESPACE}}}type"] = "footnote"
    if block.type is BlockType.POEM:
        attributes["class"] = "linguary-verse"
    if block.type is BlockType.PAGE_BREAK:
        attributes[f"{{{_EPUB_NAMESPACE}}}type"] = "pagebreak"
    return attributes


def _anchors(block: DocumentBlock) -> tuple[str, ...]:
    """Идентификаторы, по которым на блок ссылается книга."""
    return tuple(block.attributes.get("anchor_ids", "").split())


def _nestable(
    annotations: Sequence[InlineAnnotation],
) -> tuple[InlineAnnotation, ...]:
    """Разметка, из которой можно построить дерево вложенных элементов.

    Разметка снята с корректного XHTML, поэтому обычно вложена правильно. Пересечение
    без вложенности деревом не описывается, и такая разметка отбрасывается: потерять
    оформление фрагмента лучше, чем собрать неправильный XHTML.
    """
    ordered = sorted(annotations, key=lambda item: (item.start, -item.end))
    kept: list[InlineAnnotation] = []
    for annotation in ordered:
        if all(
            annotation.end <= previous.end or annotation.start >= previous.end
            for previous in kept
        ):
            kept.append(annotation)
    return tuple(kept)


def _append_inline(
    host: etree._Element,
    text: str,
    annotations: Sequence[InlineAnnotation],
    state: _SectionState,
) -> None:
    """Пишет текст блока в элемент, восстанавливая встроенную разметку."""
    _append_range(host, text, annotations, start=0, end=len(text), state=state)


def _append_range(
    host: etree._Element,
    text: str,
    annotations: Sequence[InlineAnnotation],
    *,
    start: int,
    end: int,
    state: _SectionState,
) -> None:
    """Участок текста вместе с разметкой, которая внутри него начинается."""
    position = start
    index = 0
    while index < len(annotations):
        annotation = annotations[index]
        _append_text(host, text[position : annotation.start])
        element = _sub(
            host,
            _ANNOTATION_TAGS[annotation.type],
            _target_attributes(annotation, state),
        )
        nested = tuple(
            item for item in annotations[index + 1 :] if item.end <= annotation.end
        )
        _append_range(
            element,
            text,
            nested,
            start=annotation.start,
            end=annotation.end,
            state=state,
        )
        position = annotation.end
        index += 1 + len(nested)
    _append_text(host, text[position:end])


def _target_attributes(
    annotation: InlineAnnotation, state: _SectionState
) -> dict[str, str]:
    """Атрибуты элемента разметки: адрес ссылки и пометка ссылки на сноску."""
    if annotation.target is None:
        return {}
    attributes = {"href": _href_from_target(annotation.target, base=state.href)}
    if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE:
        attributes[f"{{{_EPUB_NAMESPACE}}}type"] = "noteref"
    return attributes


def _href_from_target(target: str, *, base: str) -> str:
    """Адрес ссылки относительно документа главы.

    Импортёр нормализует внутренние ссылки к виду «путь ресурса книги#фрагмент», то есть
    относительно каталога пакета; в документе главы адрес должен быть относительно неё.
    """
    if _is_external(target):
        return target
    path, separator, fragment = target.partition("#")
    if not path:
        return f"{separator}{fragment}"
    relative = _relative_href(base, posixpath.normpath(path))
    return f"{relative}{separator}{fragment}" if separator else relative


def _append_text(host: etree._Element, text: str) -> None:
    """Добавляет текст, превращая переносы строк блока в `<br/>`."""
    if not text:
        return
    parts = text.split("\n")
    _append_characters(host, parts[0])
    for part in parts[1:]:
        _sub(host, "br")
        _append_characters(host, part)


def _append_characters(host: etree._Element, text: str) -> None:
    """Добавляет текст в конец содержимого элемента."""
    if not text:
        return
    if len(host):
        last = host[-1]
        last.tail = (last.tail or "") + text
    else:
        host.text = (host.text or "") + text


def _unwrap(element: etree._Element) -> None:
    """Заменяет элемент его содержимым, сохраняя текст вокруг него."""
    parent = element.getparent()
    if parent is None:
        return
    index = parent.index(element)
    children = list(element)
    _insert_text(parent, index, element.text or "")
    for offset, child in enumerate(children):
        parent.insert(index + offset, child)
    tail = element.tail or ""
    if tail:
        if children:
            children[-1].tail = (children[-1].tail or "") + tail
        else:
            _insert_text(parent, index, tail)
    parent.remove(element)


def _insert_text(parent: etree._Element, index: int, text: str) -> None:
    """Дописывает текст в позицию перед указанным ребёнком."""
    if not text:
        return
    if index == 0:
        parent.text = (parent.text or "") + text
    else:
        previous = parent[index - 1]
        previous.tail = (previous.tail or "") + text


def _paragraphs(text: str) -> tuple[str, ...]:
    """Абзацы текста: пустая строка разделяет, одиночный перенос остаётся внутри."""
    return tuple(part.strip() for part in text.split("\n\n") if part.strip())


def _previously_heading(previous_title: str) -> str:
    """Заголовок раздела о предыдущей главе на языке оригинала (PRD § 19.2)."""
    title = previous_title.strip()
    if not title:
        return "Previously"
    if title.lower().startswith("chapter"):
        return f"Previously in {title}"
    if _looks_like_chapter_number(title):
        return f"Previously in Chapter {title}"
    return f"Previously in {title}"


def _looks_like_chapter_number(title: str) -> bool:
    """Назван ли раздел просто номером — арабским или римским."""
    candidate = title.rstrip(".").strip()
    if candidate.isdigit():
        return True
    return bool(candidate) and set(candidate.upper()) <= set("IVXLCDM")


def _unique_id(candidate: str, used: set[str]) -> str:
    """Идентификатор, которого ещё нет в документе."""
    identifier = candidate
    suffix = 1
    while identifier in used:
        suffix += 1
        identifier = f"{candidate}-{suffix}"
    used.add(identifier)
    return identifier


def _contains_id(element: etree._Element, identifier: str) -> bool:
    """Есть ли такой идентификатор в самом элементе или внутри него."""
    return any(node.get("id") == identifier for node in element.iter())


def _in_xhtml_namespace(element: etree._Element) -> etree._Element:
    """Приводит скопированную разметку к пространству имён XHTML.

    Издания без объявленного пространства имён встречаются; без приведения такой фрагмент
    сериализуется с пустым `xmlns` и перестаёт быть частью документа книги.
    """
    for node in element.iter():
        if isinstance(node.tag, str) and not node.tag.startswith("{"):
            node.tag = f"{{{_XHTML_NAMESPACE}}}{node.tag}"
        node.attrib.pop("summary", None)
    return element


def _element_at(root: etree._Element, structural_path: str) -> etree._Element | None:
    """Элемент документа по структурному пути импортёра (ALG-01)."""
    steps = [step for step in structural_path.split("/") if step]
    if not steps:
        return None
    current = root
    first = _parse_step(steps[0])
    if first is None or local_name(current) != first[0] or first[1] != 1:
        return None
    for step in steps[1:]:
        parsed = _parse_step(step)
        if parsed is None:
            return None
        tag, index = parsed
        current = _nth_child(current, tag=tag, index=index)
        if current is None:
            return None
    return current


def _parse_step(step: str) -> tuple[str, int] | None:
    """Шаг структурного пути вида `div[2]`; иные шаги не поддерживаются."""
    tag, separator, rest = step.partition("[")
    if not separator or not rest.endswith("]") or not tag.isidentifier():
        return None
    position = rest[:-1]
    return (tag, int(position)) if position.isdigit() else None


def _nth_child(
    parent: etree._Element, *, tag: str, index: int
) -> etree._Element | None:
    """`index`-й потомок с указанным именем тега, считая с единицы."""
    seen = 0
    for child in parent:
        if local_name(child) != tag:
            continue
        seen += 1
        if seen == index:
            return child
    return None


def _resolve_target(href: str, *, base: str) -> tuple[str, str]:
    """Адрес ссылки как пара «путь в архиве, фрагмент»."""
    path, _, fragment = href.partition("#")
    if not path:
        return base, fragment
    directory = posixpath.dirname(base)
    return posixpath.normpath(posixpath.join(directory, path)), fragment


def _relative_href(base: str, target: str) -> str:
    """Путь к файлу книги относительно документа `base`."""
    directory = posixpath.dirname(base)
    return posixpath.relpath(target, directory) if directory else target


def _is_external(href: str) -> bool:
    """Ведёт ли адрес наружу книги."""
    scheme, separator, _ = href.partition(":")
    return bool(separator) and scheme.isalpha()


def _contained_name(resource_path: str) -> str:
    """Путь файла внутри каталога пакета; выход за его пределы недопустим."""
    normalized = posixpath.normpath(resource_path.replace("\\", "/").strip())
    if not normalized or normalized.startswith(("/", "..")):
        raise EpubExportFatalError(
            f"путь {resource_path!r} ведёт за пределы каталога книги"
        )
    return normalized


def _media_type(href: str) -> str:
    """Тип содержимого файла книги по его расширению."""
    return _MEDIA_TYPES.get(posixpath.splitext(href)[1].lower(), "application/octet-stream")


#: Типы содержимого файлов, которые собирает сам экспортёр. Типы ресурсов книги берутся
#: из канонической модели, поэтому здесь их нет.
_MEDIA_TYPES: Mapping[str, str] = {
    ".xhtml": "application/xhtml+xml",
    ".html": "application/xhtml+xml",
    ".htm": "application/xhtml+xml",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".otf": "font/otf",
    ".ttf": "font/ttf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".js": "application/javascript",
    ".xml": "application/xml",
}


__all__ = [
    "EpubExportWriter",
    "ExportCheckFailure",
    "ExportCheckKind",
    "verify_exported_epub",
]
