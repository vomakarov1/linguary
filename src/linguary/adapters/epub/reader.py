"""Импорт EPUB: контейнер и метаданные через EbookLib, XHTML через lxml.

Реализация порта `linguary.application.ports.epub.EpubReader`. Границы глав определяются
порядком чтения: один XHTML-документ spine — одна `DocumentSection`, а заголовок берётся
из nav/toc, если он там есть (PRD § 6). Несовпадение оглавления с разбиением по документам
импорт не ломает и фиксируется предупреждением.

Идентификатор блока считает доменная фабрика `DocumentBlock.create` по тройке
`document_id + source_resource_path + structural_path` (ALG-01, PRD § 8); структурный путь —
абсолютный XPath с индексами одноимённых тегов, он зависит только от положения элемента
в документе и не зависит от его текста.
"""

from __future__ import annotations

import posixpath
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from ebooklib import epub
from lxml import etree

from linguary.adapters.epub.inline import (
    InlineContent,
    InlineContentBuilder,
    footnote_element_ids,
    is_footnote_element,
    local_name,
    resolve_link_target,
)
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubReader,
    EpubReadRequest,
    ImportedBook,
)
from linguary.domain import (
    BlockType,
    BookImportWarning,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentResource,
    DocumentSection,
    ImportWarningKind,
    SourceReference,
    SupportStatus,
)

_XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
_EPUB_NAMESPACE = "http://www.idpf.org/2007/ops"

#: Теги, встреча которых внутри элемента делает его контейнером: собственного текста
#: у такого элемента нет, а блоки дают его потомки.
_STRUCTURAL_TAGS = frozenset(
    {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hgroup",
        "blockquote",
        "li",
        "div",
        "figure",
        "figcaption",
        "table",
        "pre",
        "img",
        "ol",
        "ul",
        "dl",
        "dt",
        "dd",
        "aside",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "nav",
        "hr",
    }
)

#: Теги, которые остаются одним блоком, даже если внутри есть структурные элементы.
#: Таблица разбирается как целое (`PRESERVE_OPAQUE`), а не как набор ячеек.
_ALWAYS_LEAF_TAGS = frozenset({"table", "pre", "img", "hr"})

#: Теги, блок для которых создаётся даже без текста: они сами по себе несут структуру.
_KEEP_EMPTY_TAGS = frozenset({"img", "hr", "table"})

#: Служебные теги документа: текстом книги не являются.
_IGNORED_TAGS = frozenset({"head", "title", "script", "style", "link", "meta", "base"})

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

#: Классы и `epub:type`, которыми размечают стихотворную строку или строфу.
_VERSE_MARKERS = frozenset({"l", "line", "verse", "poem", "stanza", "linegroup"})

#: Признаки разрыва страницы и номера страницы печатного оригинала.
_PAGE_BREAK_MARKERS = frozenset({"pagebreak", "pageno", "page-break", "pagenum"})

#: Типы блоков, которые переносятся в итоговый EPUB без внутренней обработки (PRD § 6).
_PRESERVE_OPAQUE_TYPES = frozenset({BlockType.TABLE, BlockType.OTHER, BlockType.PAGE_BREAK})

#: Тип содержимого для ресурса, у которого манифест его не задал.
_DEFAULT_MEDIA_TYPE = "application/octet-stream"

#: Язык оригинала, если в метаданных EPUB его нет.
_DEFAULT_SOURCE_LANGUAGE = "en"

_STRUCTURAL_QUALIFIED_NAMES = tuple(
    name
    for tag in sorted(_STRUCTURAL_TAGS)
    for name in (tag, f"{{{_XHTML_NAMESPACE}}}{tag}")
)


@dataclass(frozen=True, slots=True)
class _SpineDocument:
    """Документ порядка чтения: путь в книге и его содержимое."""

    resource_path: str
    content: bytes


@dataclass(slots=True)
class _InlineRun:
    """Пробег inline-содержимого между блочными детьми контейнера.

    Порядковый номер пробега входит в структурный путь блока, поэтому у анонимного
    блока такой же стабильный идентификатор, как у блока с собственным элементом.
    """

    builder: InlineContentBuilder
    index: int = 0


@dataclass(frozen=True, slots=True)
class _BlockContext:
    """Контекст обхода: родительский блок и тип, навязанный контейнером.

    Абзац внутри цитаты — часть цитаты, абзац внутри сноски — часть сноски: тип задаётся
    ближайшим значащим контейнером, а вложенность фиксируется `parent_id`.
    """

    parent_id: str | None = None
    inherited_type: BlockType | None = None


class EbookLibEpubReader(EpubReader):
    """Чтение EPUB в каноническую модель документа.

    Читатель не пишет файлы: копии ресурсов на диск сохраняет отдельный шаг импорта,
    здесь фиксируется только сам факт ресурса и ссылка на него из блока `Image`.
    """

    def read(self, request: EpubReadRequest) -> ImportedBook:
        """Читает книгу целиком; критичная структура даёт `EpubImportFatalError`."""
        book = _open_epub(request.source_path)
        warnings: list[BookImportWarning] = []
        documents = _reading_order(book, request.source_path, warnings)
        toc_titles = _toc_titles(book)
        warnings.extend(
            _toc_mismatch_warnings(
                toc_titles=toc_titles,
                spine_paths=tuple(document.resource_path for document in documents),
            )
        )
        resources = _resources(book, book_id=request.book_id)
        manifest_paths = frozenset(resource.source_path for resource in resources)

        sections: list[DocumentSection] = []
        for order, document in enumerate(documents):
            section, section_warnings = _build_section(
                document=document,
                order=order,
                book_id=request.book_id,
                toc_titles=toc_titles,
                manifest_paths=manifest_paths,
            )
            sections.append(section)
            warnings.extend(section_warnings)

        return ImportedBook(
            document=CanonicalDocument(
                id=request.book_id,
                metadata=_metadata(book, source_path=request.source_path),
                source_language=_source_language(book, request=request),
                target_language=request.target_language,
                sections=tuple(sections),
                resources=resources,
            ),
            warnings=tuple(warnings),
        )


def _open_epub(source_path: str) -> epub.EpubBook:
    """Открывает контейнер EPUB; нечитаемый файл — `FATAL` (PRD § 6)."""
    try:
        return epub.read_epub(source_path)
    except (
        KeyError,
        OSError,
        epub.EpubException,
        etree.XMLSyntaxError,
        zipfile.BadZipFile,
    ) as error:
        raise EpubImportFatalError(
            f"EPUB {source_path} нельзя прочитать: {error}"
        ) from error


def _reading_order(
    book: epub.EpubBook, source_path: str, warnings: list[BookImportWarning]
) -> tuple[_SpineDocument, ...]:
    """Документы порядка чтения; пустой spine — `FATAL` (PRD § 6).

    Содержимое документов берётся из архива как есть, а не из EbookLib: библиотека
    пересобирает XHTML при чтении и теряет часть разметки, а структурный путь блока
    обязан совпадать с исходным документом книги.
    """
    with _ArchiveContents(source_path) as contents:
        documents = _spine_documents(book, contents, warnings)
    if not documents:
        raise EpubImportFatalError(
            "в EPUB нет порядка чтения: spine пуст или не содержит доступных документов"
        )
    return documents


def _spine_documents(
    book: epub.EpubBook,
    contents: _ArchiveContents,
    warnings: list[BookImportWarning],
) -> tuple[_SpineDocument, ...]:
    """Документы spine в порядке чтения, без повторов и недостающих элементов."""
    documents: list[_SpineDocument] = []
    seen: set[str] = set()
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        item = book.get_item_with_id(item_id)
        if item is None:
            warnings.append(
                BookImportWarning(
                    kind=ImportWarningKind.UNSUPPORTED_CONSTRUCT,
                    message=(
                        f"документ {item_id} есть в порядке чтения, но отсутствует "
                        f"в манифесте книги"
                    ),
                )
            )
            continue
        resource_path = item.get_name()
        if resource_path in seen:
            warnings.append(
                BookImportWarning(
                    kind=ImportWarningKind.UNSUPPORTED_CONSTRUCT,
                    message=(
                        f"документ {resource_path} встречается в порядке чтения повторно; "
                        f"учтено только первое вхождение"
                    ),
                )
            )
            continue
        seen.add(resource_path)
        documents.append(
            _SpineDocument(
                resource_path=resource_path, content=contents.read(resource_path)
            )
        )
    return tuple(documents)


class _ArchiveContents:
    """Доступ к файлам EPUB по пути относительно каталога OPF."""

    def __init__(self, source_path: str) -> None:
        self._source_path = source_path
        try:
            self._archive = zipfile.ZipFile(source_path)
            self._directory = posixpath.dirname(_package_path(self._archive))
        except (OSError, KeyError, zipfile.BadZipFile, etree.XMLSyntaxError) as error:
            raise EpubImportFatalError(
                f"EPUB {source_path} нельзя прочитать: {error}"
            ) from error

    def __enter__(self) -> _ArchiveContents:
        return self

    def __exit__(self, *exception: object) -> None:
        self._archive.close()

    def read(self, resource_path: str) -> bytes:
        """Содержимое файла книги; потеря обязательного документа — `FATAL`."""
        name = posixpath.normpath(posixpath.join(self._directory, resource_path))
        for candidate in (name, unquote(name)):
            try:
                return self._archive.read(candidate)
            except KeyError:
                continue
        raise EpubImportFatalError(
            f"документ {resource_path} объявлен в EPUB {self._source_path}, "
            f"но отсутствует в архиве"
        )


def _package_path(archive: zipfile.ZipFile) -> str:
    """Путь к OPF внутри архива по `META-INF/container.xml`."""
    container = etree.fromstring(archive.read("META-INF/container.xml"))
    for element in container.iter():
        if local_name(element) == "rootfile" and element.get("full-path"):
            return element.get("full-path", "")
    raise EpubImportFatalError("в EPUB не объявлен файл пакета OPF")


def _toc_titles(book: epub.EpubBook) -> Mapping[str, tuple[str, ...]]:
    """Заголовки nav/toc, сгруппированные по документу порядка чтения.

    EPUB2 отдаёт оглавление из NCX, EPUB3 — из nav; EbookLib приводит оба к одному виду,
    поэтому версия книги здесь уже не различается.
    """
    titles: dict[str, list[str]] = {}
    for href, title in _flatten_toc(book.toc):
        path = unquote(href.partition("#")[0])
        if not path or not title:
            continue
        titles.setdefault(posixpath.normpath(path), []).append(title.strip())
    return {path: tuple(values) for path, values in titles.items()}


def _flatten_toc(entries: Iterable[object]) -> Iterator[tuple[str, str]]:
    """Разворачивает возможно вложенное оглавление в плоскую последовательность."""
    for entry in entries:
        if isinstance(entry, (list, tuple)):
            if entry and isinstance(entry[0], (epub.Link, epub.Section)):
                yield from _flatten_toc([entry[0]])
                yield from _flatten_toc(entry[1] if len(entry) > 1 else ())
            else:
                yield from _flatten_toc(entry)
            continue
        href = getattr(entry, "href", None)
        title = getattr(entry, "title", None)
        if href and title:
            yield href, title


def _toc_mismatch_warnings(
    *, toc_titles: Mapping[str, tuple[str, ...]], spine_paths: Sequence[str]
) -> list[BookImportWarning]:
    """Предупреждения о несовпадении оглавления с разбиением по документам (PRD § 6).

    Книга всё равно импортируется по документам порядка чтения, поэтому несовпадение —
    замечание, а не отказ: в реальных изданиях несколько глав часто лежат в одном XHTML.
    """
    known = set(spine_paths)
    warnings: list[BookImportWarning] = []
    crowded = {
        path: len(titles)
        for path, titles in toc_titles.items()
        if path in known and len(titles) > 1
    }
    if crowded:
        details = ", ".join(
            f"{path} — {count} пунктов" for path, count in sorted(crowded.items())
        )
        warnings.append(
            BookImportWarning(
                kind=ImportWarningKind.TOC_MISMATCH,
                message=(
                    f"оглавление не совпадает с разбиением по документам порядка чтения: "
                    f"{details}. Книга импортирована по документам, одна глава — "
                    f"один документ"
                ),
            )
        )
    unknown = sorted(path for path in toc_titles if path not in known)
    if unknown:
        warnings.append(
            BookImportWarning(
                kind=ImportWarningKind.TOC_MISMATCH,
                message=(
                    f"пункты оглавления ведут на документы вне порядка чтения: "
                    f"{', '.join(unknown)}"
                ),
            )
        )
    return warnings


def _resources(book: epub.EpubBook, *, book_id: str) -> tuple[DocumentResource, ...]:
    """Ресурсы книги: всё, кроме документов порядка чтения.

    Идентификатор ресурса делается уникальным в пределах книги и хранилища: идентификаторы
    манифеста уникальны только внутри своей книги.
    """
    resources: list[DocumentResource] = []
    seen: set[str] = set()
    for item in book.get_items():
        media_type = (item.media_type or "").strip() or _DEFAULT_MEDIA_TYPE
        if media_type == "application/xhtml+xml":
            continue
        source_path = item.get_name()
        if not source_path or source_path in seen:
            continue
        seen.add(source_path)
        resources.append(
            DocumentResource(
                id=f"{book_id}:{source_path}",
                source_path=source_path,
                media_type=media_type,
            )
        )
    return tuple(resources)


def _metadata(book: epub.EpubBook, *, source_path: str) -> BookMetadata:
    """Метаданные книги; при отсутствии названия используется имя файла."""
    title = _first_metadata(book, "title") or Path(source_path).stem
    return BookMetadata(
        title=title,
        authors=tuple(value for value, _ in book.get_metadata("DC", "creator")),
        identifier=_first_metadata(book, "identifier"),
        publisher=_first_metadata(book, "publisher"),
    )


def _first_metadata(book: epub.EpubBook, name: str) -> str | None:
    """Первое непустое значение элемента Dublin Core."""
    for value, _ in book.get_metadata("DC", name):
        if value and value.strip():
            return value.strip()
    return None


def _source_language(book: epub.EpubBook, *, request: EpubReadRequest) -> str:
    """Язык оригинала: из запроса, иначе из метаданных книги."""
    language = request.source_language or _first_metadata(book, "language")
    language = (language or _DEFAULT_SOURCE_LANGUAGE).strip()
    if language == request.target_language:
        raise EpubImportFatalError(
            f"язык оригинала книги совпадает с целевым языком перевода: {language}"
        )
    return language


def _build_section(
    *,
    document: _SpineDocument,
    order: int,
    book_id: str,
    toc_titles: Mapping[str, tuple[str, ...]],
    manifest_paths: frozenset[str],
) -> tuple[DocumentSection, list[BookImportWarning]]:
    """Строит раздел книги из одного документа порядка чтения."""
    root = _parse_document(document)
    body = _body(root, resource_path=document.resource_path)
    builder = _SectionBuilder(
        book_id=book_id,
        section_id=_section_id(book_id, document.resource_path),
        resource_path=document.resource_path,
        footnote_ids=footnote_element_ids(root),
        manifest_paths=manifest_paths,
    )
    builder.walk(body, _BlockContext())
    blocks = tuple(builder.blocks)
    section = DocumentSection(
        id=builder.section_id,
        document_id=book_id,
        title=_section_title(
            root=root,
            resource_path=document.resource_path,
            toc_titles=toc_titles,
            blocks=blocks,
        ),
        order=order,
        source_resource_path=document.resource_path,
        blocks=blocks,
    )
    return section, builder.warnings


def _parse_document(document: _SpineDocument) -> etree._Element:
    """Разбирает XHTML документа; неразбираемый документ — `FATAL` (PRD § 6)."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    try:
        return etree.fromstring(document.content, parser=parser)
    except etree.XMLSyntaxError as error:
        raise EpubImportFatalError(
            f"XHTML документа {document.resource_path} нельзя разобрать до блоков: {error}"
        ) from error


def _body(root: etree._Element, *, resource_path: str) -> etree._Element:
    """Тело документа; без него разбирать до блоков нечего."""
    for candidate in (f"{{{_XHTML_NAMESPACE}}}body", "body"):
        body = root.find(candidate)
        if body is not None:
            return body
    raise EpubImportFatalError(
        f"в документе {resource_path} нет элемента body: разобрать его до блоков нельзя"
    )


def _section_id(book_id: str, resource_path: str) -> str:
    """Идентификатор раздела, уникальный в пределах хранилища."""
    return f"{book_id}:{resource_path}"


def _section_title(
    *,
    root: etree._Element,
    resource_path: str,
    toc_titles: Mapping[str, tuple[str, ...]],
    blocks: Sequence[DocumentBlock],
) -> str:
    """Заголовок главы: из nav/toc, иначе из `<title>`, иначе из первого заголовка."""
    titles = toc_titles.get(resource_path)
    if titles:
        return titles[0]
    for path in (
        f"./{{{_XHTML_NAMESPACE}}}head/{{{_XHTML_NAMESPACE}}}title",
        "./head/title",
    ):
        element = root.find(path)
        if element is not None and element.text and element.text.strip():
            return element.text.strip()
    for block in blocks:
        if block.type is BlockType.HEADING and block.text:
            return block.text
    return resource_path


class _SectionBuilder:
    """Обход одного XHTML-документа в блоки раздела.

    Обход идёт в порядке документа: элемент со структурными потомками считается
    контейнером и блока сам не даёт, остальные элементы становятся блоками. Пробеги
    inline-содержимого рядом с блочными детьми контейнера не теряются: из них собирается
    отдельный блок, иначе подписи к иллюстрациям и боковые заметки исчезли бы из книги.
    """

    def __init__(
        self,
        *,
        book_id: str,
        section_id: str,
        resource_path: str,
        footnote_ids: frozenset[str],
        manifest_paths: frozenset[str],
    ) -> None:
        self.section_id = section_id
        self.blocks: list[DocumentBlock] = []
        self.warnings: list[BookImportWarning] = []
        self._book_id = book_id
        self._resource_path = resource_path
        self._footnote_ids = footnote_ids
        self._manifest_paths = manifest_paths
        self._order = 0
        self._pending_anchor_ids: list[str] = []

    def walk(self, container: etree._Element, context: _BlockContext) -> None:
        """Обходит содержимое контейнера, сохраняя порядок документа."""
        run = _InlineRun(builder=self._new_content_builder())
        run.builder.add_text(container.text)
        for child in container:
            tag = local_name(child)
            if tag is None or tag in _IGNORED_TAGS:
                run.builder.add_text(child.tail)
                continue
            structural = _is_structural(child)
            if not structural and not _is_anchor_only(child):
                run.builder.add_element(child)
                continue
            self._flush_run(run, container=container, context=context)
            if structural:
                self._emit(child, context)
            else:
                self._emit_leaf(child, context)
            run.builder.add_text(child.tail)
        self._flush_run(run, container=container, context=context)

    def _emit(self, element: etree._Element, context: _BlockContext) -> None:
        """Создаёт блок для элемента либо спускается в него как в контейнер."""
        if not _is_container(element):
            self._emit_leaf(element, context)
            return
        if is_footnote_element(element):
            block = self._add_block(
                element=element,
                structural_path=_structural_path(element),
                type=BlockType.FOOTNOTE,
                content=InlineContent(text=""),
                support_status=SupportStatus.PRESERVE_OPAQUE,
                context=context,
            )
            self.walk(
                element,
                _BlockContext(parent_id=block.id, inherited_type=BlockType.FOOTNOTE),
            )
            return
        anchor_id = element.get("id")
        if anchor_id:
            self._pending_anchor_ids.append(anchor_id)
        self.walk(element, _inherited_context(element, context))
        if anchor_id and anchor_id in self._pending_anchor_ids:
            self._pending_anchor_ids.remove(anchor_id)

    def _emit_leaf(self, element: etree._Element, context: _BlockContext) -> None:
        """Создаёт блок для элемента без структурных потомков."""
        builder = self._new_content_builder()
        builder.add_children(element)
        content = builder.build()
        block_type = _leaf_type(element, context=context, content=content)
        if not self._is_worth_keeping(element, content=content, block_type=block_type):
            return
        self._add_block(
            element=element,
            structural_path=_structural_path(element),
            type=block_type,
            content=content,
            support_status=_support_status(block_type),
            context=context,
        )

    def _flush_run(
        self, run: _InlineRun, *, container: etree._Element, context: _BlockContext
    ) -> None:
        """Закрывает пробег inline-содержимого и, если в нём был текст, делает из него блок."""
        content = run.builder.build()
        run.index += 1
        run.builder = self._new_content_builder()
        if not content.text:
            return
        self._add_block(
            element=None,
            structural_path=f"{_structural_path(container)}/text()[{run.index}]",
            type=context.inherited_type or BlockType.PARAGRAPH,
            content=content,
            support_status=SupportStatus.SUPPORTED,
            context=context,
        )

    def _add_block(
        self,
        *,
        element: etree._Element | None,
        structural_path: str,
        type: BlockType,
        content: InlineContent,
        support_status: SupportStatus,
        context: _BlockContext,
    ) -> DocumentBlock:
        """Добавляет блок в раздел и отдаёт его: он может стать родителем вложенных."""
        element_id = element.get("id") if element is not None else None
        block = DocumentBlock.create(
            document_id=self._book_id,
            section_id=self.section_id,
            type=type,
            order=self._order,
            text=content.text,
            source_reference=SourceReference(
                resource_path=self._resource_path,
                structural_path=structural_path,
                element_id=element_id,
            ),
            support_status=support_status,
            inline_annotations=content.annotations,
            parent_id=context.parent_id,
            attributes=self._attributes(element, content=content, element_id=element_id),
        )
        self.blocks.append(block)
        self._order += 1
        self._pending_anchor_ids.clear()
        return block

    def _attributes(
        self,
        element: etree._Element | None,
        *,
        content: InlineContent,
        element_id: str | None,
    ) -> dict[str, str]:
        """Служебные признаки блока, не влияющие на его текст."""
        attributes: dict[str, str] = {}
        if element is not None:
            attributes["tag"] = local_name(element) or ""
        anchors = _unique(
            (element_id, *self._pending_anchor_ids, *content.anchor_ids)
        )
        if anchors:
            attributes["anchor_ids"] = " ".join(anchors)
        if element is not None and local_name(element) == "img":
            attributes.update(self._image_attributes(element))
        return attributes

    def _image_attributes(self, element: etree._Element) -> dict[str, str]:
        """Ссылка на ресурс изображения и его текстовое описание."""
        attributes: dict[str, str] = {}
        source = element.get("src")
        if source:
            target = resolve_link_target(source, resource_path=self._resource_path)
            attributes["src"] = target
            if target not in self._manifest_paths:
                self.warnings.append(
                    BookImportWarning(
                        kind=ImportWarningKind.MISSING_RESOURCE,
                        message=(
                            f"изображение {target} из документа {self._resource_path} "
                            f"отсутствует в манифесте книги"
                        ),
                        section_id=self.section_id,
                    )
                )
        alternative = element.get("alt")
        if alternative and alternative.strip():
            attributes["alt"] = alternative.strip()
        return attributes

    def _is_worth_keeping(
        self,
        element: etree._Element,
        *,
        content: InlineContent,
        block_type: BlockType,
    ) -> bool:
        """Нужен ли блок для элемента без текста.

        Пустой элемент сохраняется, если он сам является структурой (изображение,
        разделитель, таблица) либо служит целью внутренней ссылки книги.
        """
        if content.text:
            return True
        if local_name(element) in _KEEP_EMPTY_TAGS:
            return True
        if block_type is BlockType.PAGE_BREAK:
            return True
        return bool(element.get("id") or content.anchor_ids)

    def _new_content_builder(self) -> InlineContentBuilder:
        return InlineContentBuilder(
            resource_path=self._resource_path, footnote_ids=self._footnote_ids
        )


def _is_structural(element: etree._Element) -> bool:
    """Даёт ли элемент блок или контейнер блоков."""
    return local_name(element) in _STRUCTURAL_TAGS


def _is_anchor_only(element: etree._Element) -> bool:
    """Пустой inline-элемент, существующий только как цель внутренней ссылки.

    Номера страниц печатного оригинала и якоря иллюстраций Gutenberg размечает пустыми
    `<span>` и `<a>`. Текста они не несут, но на них ссылается оглавление и сама книга,
    поэтому такой элемент становится отдельным блоком, а не растворяется в соседнем тексте.
    """
    if "".join(element.itertext()).strip():
        return False
    return bool(element.get("id")) or any(
        descendant.get("id") for descendant in element.iterdescendants()
    )


def _is_container(element: etree._Element) -> bool:
    """Есть ли внутри элемента структурные потомки."""
    if local_name(element) in _ALWAYS_LEAF_TAGS:
        return False
    return next(element.iterdescendants(*_STRUCTURAL_QUALIFIED_NAMES), None) is not None


def _inherited_context(
    element: etree._Element, context: _BlockContext
) -> _BlockContext:
    """Тип, который контейнер навязывает вложенным блокам."""
    tag = local_name(element)
    if tag == "blockquote":
        return _BlockContext(parent_id=context.parent_id, inherited_type=BlockType.QUOTE)
    if tag == "li":
        return _BlockContext(
            parent_id=context.parent_id, inherited_type=BlockType.LIST_ITEM
        )
    return context


def _leaf_type(
    element: etree._Element, *, context: _BlockContext, content: InlineContent
) -> BlockType:
    """Тип блока для элемента без структурных потомков."""
    tag = local_name(element)
    if _is_page_break(element):
        return BlockType.PAGE_BREAK
    if tag == "img":
        return BlockType.IMAGE
    if tag in _HEADING_TAGS:
        return BlockType.HEADING
    if tag == "table":
        return BlockType.TABLE
    if tag in ("hr", "pre"):
        return BlockType.OTHER
    if is_footnote_element(element):
        return BlockType.FOOTNOTE
    if tag == "li":
        return BlockType.LIST_ITEM
    if tag == "blockquote":
        return BlockType.QUOTE
    if _is_verse(element):
        return BlockType.POEM
    if context.inherited_type is not None:
        return context.inherited_type
    return BlockType.PARAGRAPH if content.text else BlockType.OTHER


def _support_status(block_type: BlockType) -> SupportStatus:
    """Статус поддержки конструкции по её типу (PRD § 6)."""
    return (
        SupportStatus.PRESERVE_OPAQUE
        if block_type in _PRESERVE_OPAQUE_TYPES
        else SupportStatus.SUPPORTED
    )


def _is_page_break(element: etree._Element) -> bool:
    """Помечен ли элемент как разрыв страницы печатного оригинала.

    Совпадение ищется по вхождению: издатели пишут и `pagebreak`, и собственные классы
    вида `x-ebookmaker-pageno`.
    """
    tokens = _markers(element)
    return any(marker in token for token in tokens for marker in _PAGE_BREAK_MARKERS)


def _is_verse(element: etree._Element) -> bool:
    """Помечен ли элемент как стихотворная строка или строфа."""
    markers = _markers(element)
    return bool(markers & _VERSE_MARKERS) or any(
        marker.startswith("z3998:verse") for marker in markers
    )


def _markers(element: etree._Element) -> frozenset[str]:
    """Классы и `epub:type` элемента одним набором токенов."""
    tokens: set[str] = set()
    for value in (
        element.get("class"),
        element.get(f"{{{_EPUB_NAMESPACE}}}type"),
        element.get("epub:type"),
    ):
        if value:
            tokens.update(value.split())
    return frozenset(tokens)


def _structural_path(element: etree._Element) -> str:
    """Абсолютный XPath элемента с индексами одноимённых тегов (ALG-01).

    Путь считается только по структуре документа, поэтому правка текста блока
    идентификатор блока не меняет.
    """
    steps: list[str] = []
    current: etree._Element | None = element
    while current is not None:
        tag = local_name(current)
        if tag is None:
            break
        parent = current.getparent()
        if parent is None:
            steps.append(f"/{tag}[1]")
            break
        index = 0
        for sibling in parent:
            if local_name(sibling) == tag:
                index += 1
            if sibling is current:
                break
        steps.append(f"/{tag}[{index}]")
        current = parent
    return "".join(reversed(steps))


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    """Непустые значения без повторов, в порядке появления."""
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = ["EbookLibEpubReader"]
