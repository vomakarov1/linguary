"""EPUB import: container and metadata via EbookLib, XHTML via lxml.

Implementation of `linguary.application.ports.epub.EpubReader` port. Chapter boundaries
are determined by reading order (spine): a single XHTML document in the spine corresponds
to one `DocumentSection`, and the title is extracted from nav/toc if present (PRD § 6).
Discrepancies between the TOC and document divisions do not break import and are recorded
as warnings.

Block IDs are calculated by the domain factory `DocumentBlock.create` based on the tuple
`document_id + source_resource_path + structural_path` (ALG-01, PRD § 8); the structural path
is an absolute XPath with tag sibling indices, depending strictly on the element's position
in the document and independent of its text content.
"""

from __future__ import annotations

import posixpath
import re
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
from linguary.adapters.epub.resources import (
    ExtractionLimits,
    check_extraction_limits,
    format_size,
    package_path,
    unreadable_epub_message,
)
from linguary.adapters.epub.splitting import (
    SPLIT_MIN_CHARS,
    split_inline_content,
)
from linguary.adapters.epub.support_matrix import (
    DroppedBlockLog,
    DropReason,
    block_support_status,
    dropped_resource_warning,
    resource_support_status,
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

#: Tags whose presence inside an element turns it into a container: such an element has no direct text,
#: and blocks are produced by its descendants.
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

#: Tags that remain a single block even if structural elements exist inside.
#: Tables are parsed as a whole (`PRESERVE_OPAQUE`) rather than as individual cells.
_ALWAYS_LEAF_TAGS = frozenset({"table", "pre", "img", "hr"})

#: Tags for which a block is created even without text: they inherently represent structure.
_KEEP_EMPTY_TAGS = frozenset({"img", "hr", "table"})

#: Document auxiliary tags: not considered book content.
_IGNORED_TAGS = frozenset({"head", "title", "script", "style", "link", "meta", "base"})

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

#: Class and `epub:type` markers indicating verse lines or stanzas.
_VERSE_MARKERS = frozenset({"l", "line", "verse", "poem", "stanza", "linegroup"})

#: Markers indicating page breaks and print original page numbers.
_PAGE_BREAK_MARKERS = frozenset({"pagebreak", "pageno", "page-break", "pagenum"})

#: Page break marker pattern matching whole tokens or parts separated by hyphen, underscore, or colon.
_PAGE_BREAK_PATTERN = re.compile(
    r"(?:^|[-_:])(?:"
    + "|".join(re.escape(marker) for marker in sorted(_PAGE_BREAK_MARKERS))
    + r")(?:$|[-_:])"
)

#: Default media type for resources when unspecified in manifest.
_DEFAULT_MEDIA_TYPE = "application/octet-stream"

#: Default source language if missing from EPUB metadata.
_DEFAULT_SOURCE_LANGUAGE = "en"

_STRUCTURAL_QUALIFIED_NAMES = tuple(
    name
    for tag in sorted(_STRUCTURAL_TAGS)
    for name in (tag, f"{{{_XHTML_NAMESPACE}}}{tag}")
)


@dataclass(frozen=True, slots=True)
class _SpineDocument:
    """Reading order document: path inside the book and its raw bytes."""

    resource_path: str
    content: bytes


@dataclass(slots=True)
class _InlineRun:
    """Run of inline content between block-level children of a container.

    The run index is included in the block's structural path, ensuring that anonymous
    runs receive the same stable identity as blocks tied to explicit elements.
    """

    builder: InlineContentBuilder
    index: int = 0


@dataclass(frozen=True, slots=True)
class _BlockContext:
    """Traversal context: parent block identifier and inherited block type.

    Paragraphs inside blockquotes become part of the quote, paragraphs inside footnotes
    become part of the footnote: type is governed by the nearest enclosing container,
    and nesting is recorded via `parent_id`.
    """

    parent_id: str | None = None
    inherited_type: BlockType | None = None


class EbookLibEpubReader(EpubReader):
    """Reading EPUB into the canonical document model.

    The reader does not write files to disk: resource copying is handled by a separate
    import step; this class only records the existence of resources and references to them
    from `Image` blocks.

    Archives represent untrusted input, and extraction volume is constrained by `ExtractionLimits`
    matching the resource store: the reader loads book content into memory, so archives that
    expand into tens of gigabytes must be rejected before reading (PRD § 6).
    """

    def __init__(self, *, limits: ExtractionLimits | None = None) -> None:
        self._limits = limits or ExtractionLimits()

    def read(self, request: EpubReadRequest) -> ImportedBook:
        """Read entire book; fatal structural errors raise `EpubImportFatalError`."""
        book = _open_epub(request.source_path, limits=self._limits)
        warnings: list[BookImportWarning] = []
        documents = _reading_order(
            book, request.source_path, warnings, limits=self._limits
        )
        toc_titles = _toc_titles(book)
        warnings.extend(
            _toc_mismatch_warnings(
                toc_titles=toc_titles,
                spine_paths=tuple(document.resource_path for document in documents),
            )
        )
        resources = _resources(book, book_id=request.book_id)
        warnings.extend(_dropped_resource_warnings(resources))
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


def _open_epub(source_path: str, *, limits: ExtractionLimits) -> epub.EpubBook:
    """Open EPUB container; unreadable archive raises `FATAL` (PRD § 6).

    Extraction limits are validated before reading: EbookLib loads the manifest into memory
    in full, so exceeding limits must halt import before the first archive entry is read.
    """
    _check_archive_limits(source_path, limits=limits)
    try:
        return epub.read_epub(source_path)
    except (
        KeyError,
        OSError,
        epub.EpubException,
        etree.XMLSyntaxError,
        zipfile.BadZipFile,
    ) as error:
        raise EpubImportFatalError(unreadable_epub_message(error)) from error


def _check_archive_limits(source_path: str, *, limits: ExtractionLimits) -> None:
    """Validate archive table of contents against extraction limits without reading contents."""
    try:
        with zipfile.ZipFile(source_path) as archive:
            check_extraction_limits(archive, limits=limits)
    except (OSError, zipfile.BadZipFile) as error:
        raise EpubImportFatalError(unreadable_epub_message(error)) from error


def _reading_order(
    book: epub.EpubBook,
    source_path: str,
    warnings: list[BookImportWarning],
    *,
    limits: ExtractionLimits,
) -> tuple[_SpineDocument, ...]:
    """Extract reading order documents; an empty spine raises `FATAL` (PRD § 6).

    Document contents are read directly from the archive rather than EbookLib: the library
    reserializes XHTML during reading and loses formatting details, whereas structural XPaths
    must match the original book document exactly.
    """
    with _ArchiveContents(source_path, limits=limits) as contents:
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
    """Spine documents in reading order, deduplicated and filtered."""
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
    """Access EPUB files relative to the OPF package directory.

    Extraction volume is capped: limits are verified against the archive index prior to reading,
    and read bytes are tracked on the fly — archive headers cannot be trusted for accurate file sizes.
    """

    def __init__(self, source_path: str, *, limits: ExtractionLimits) -> None:
        self._limits = limits
        self._read_bytes = 0
        try:
            self._archive = zipfile.ZipFile(source_path)
        except (OSError, zipfile.BadZipFile) as error:
            raise EpubImportFatalError(unreadable_epub_message(error)) from error
        try:
            check_extraction_limits(self._archive, limits=limits)
            self._directory = posixpath.dirname(package_path(self._archive))
        except (OSError, KeyError, zipfile.BadZipFile, etree.XMLSyntaxError) as error:
            self._archive.close()
            raise EpubImportFatalError(unreadable_epub_message(error)) from error
        except EpubImportFatalError:
            self._archive.close()
            raise

    def __enter__(self) -> _ArchiveContents:
        return self

    def __exit__(self, *exception: object) -> None:
        self._archive.close()

    def read(self, resource_path: str) -> bytes:
        """Read book entry bytes; missing mandatory documents raise `FATAL`."""
        name = posixpath.normpath(posixpath.join(self._directory, resource_path))
        for candidate in (name, unquote(name)):
            try:
                return self._read_entry(candidate)
            except KeyError:
                continue
        raise EpubImportFatalError(
            f"документ {resource_path} объявлен в книге, но отсутствует в архиве"
        )

    def _read_entry(self, name: str) -> bytes:
        """Read archive entry bytes enforcing extraction limits; missing entry raises `KeyError`.

        Reading order documents have dedicated limits: a literary chapter is well below
        tens of megabytes, and entries of such size should not be read.
        """
        declared = self._archive.getinfo(name).file_size
        if declared > self._limits.max_document_bytes:
            raise EpubImportFatalError(
                f"документ {name} заявлен как {format_size(declared)} при допустимых "
                f"{format_size(self._limits.max_document_bytes)}"
            )
        payload = self._archive.read(name)
        self._read_bytes += len(payload)
        if self._read_bytes > self._limits.max_total_bytes:
            raise EpubImportFatalError(
                f"Распакованный объём книги превышает допустимые "
                f"{format_size(self._limits.max_total_bytes)}"
            )
        return payload


def _toc_titles(book: epub.EpubBook) -> Mapping[str, tuple[str, ...]]:
    """nav/toc titles grouped by spine document path.

    EPUB2 provides TOC via NCX, EPUB3 via nav; EbookLib unifies them into a single representation,
    so EPUB version differences are transparent here.
    """
    titles: dict[str, list[str]] = {}
    for href, title in _flatten_toc(book.toc):
        path = unquote(href.partition("#")[0])
        if not path or not title:
            continue
        titles.setdefault(posixpath.normpath(path), []).append(title.strip())
    return {path: tuple(values) for path, values in titles.items()}


def _flatten_toc(entries: Iterable[object]) -> Iterator[tuple[str, str]]:
    """Flatten potentially nested table of contents structure."""
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
    """Warnings regarding mismatch between TOC and document divisions (PRD § 6).

    The book is still imported following spine documents, so mismatches produce warnings
    rather than failures: real-world publications often package multiple chapters into a single XHTML.
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
    """Book resources: everything except reading-order documents.

    Resource IDs are made unique within the book and storage: manifest IDs
    are unique only within their own book.
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
                support_status=resource_support_status(media_type),
            )
        )
    return tuple(resources)


def _dropped_resource_warnings(
    resources: Sequence[DocumentResource],
) -> list[BookImportWarning]:
    """Warnings regarding resources that are not transferred to the target publication (PRD § 6)."""
    return [
        dropped_resource_warning(
            source_path=resource.source_path, media_type=resource.media_type
        )
        for resource in resources
        if resource.support_status is SupportStatus.DROP_WITH_WARNING
    ]


def _metadata(book: epub.EpubBook, *, source_path: str) -> BookMetadata:
    """Book metadata; if title is missing, the source file stem is used."""
    title = _first_metadata(book, "title") or Path(source_path).stem
    return BookMetadata(
        title=title,
        authors=tuple(value for value, _ in book.get_metadata("DC", "creator")),
        identifier=_first_metadata(book, "identifier"),
        publisher=_first_metadata(book, "publisher"),
    )


def _first_metadata(book: epub.EpubBook, name: str) -> str | None:
    """First non-empty value of a Dublin Core element."""
    for value, _ in book.get_metadata("DC", name):
        if value and value.strip():
            return value.strip()
    return None


def _source_language(book: epub.EpubBook, *, request: EpubReadRequest) -> str:
    """Source language: from the request, otherwise from book metadata."""
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
    """Builds a book section from a single reading-order document."""
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
    return section, builder.import_warnings()


def _parse_document(document: _SpineDocument) -> etree._Element:
    """Parses document XHTML; an unparseable document is `FATAL` (PRD § 6)."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    try:
        return etree.fromstring(document.content, parser=parser)
    except etree.XMLSyntaxError as error:
        raise EpubImportFatalError(
            f"XHTML документа {document.resource_path} нельзя разобрать до блоков: {error}"
        ) from error


def _body(root: etree._Element, *, resource_path: str) -> etree._Element:
    """Document body; without it there is nothing to parse into blocks."""
    for candidate in (f"{{{_XHTML_NAMESPACE}}}body", "body"):
        body = root.find(candidate)
        if body is not None:
            return body
    raise EpubImportFatalError(
        f"в документе {resource_path} нет элемента body: разобрать его до блоков нельзя"
    )


def _section_id(book_id: str, resource_path: str) -> str:
    """Section identifier unique within the repository."""
    return f"{book_id}:{resource_path}"


def _section_title(
    *,
    root: etree._Element,
    resource_path: str,
    toc_titles: Mapping[str, tuple[str, ...]],
    blocks: Sequence[DocumentBlock],
) -> str:
    """Chapter heading: from nav/toc, otherwise from `<title>`, otherwise from first heading.

    Whitespace within the heading is collapsed: source markup often separates
    chapter number and title with indentation and line breaks.
    """
    titles = [_collapse_whitespace(title) for title in toc_titles.get(resource_path, ())]
    if titles and titles[0]:
        return titles[0]
    for path in (
        f"./{{{_XHTML_NAMESPACE}}}head/{{{_XHTML_NAMESPACE}}}title",
        "./head/title",
    ):
        element = root.find(path)
        if element is not None and element.text and element.text.strip():
            return _collapse_whitespace(element.text)
    for block in blocks:
        if block.type is BlockType.HEADING and block.text.strip():
            return _collapse_whitespace(block.text)
    return resource_path


def _collapse_whitespace(text: str) -> str:
    """Collapses any whitespace characters into single spaces and strips leading/trailing spaces."""
    return " ".join(text.split())


class _SectionBuilder:
    """Walks a single XHTML document into section blocks.

    Traversal follows document order: an element with structural descendants is considered
    a container and produces no block of its own; other elements become blocks. Inline content runs
    adjacent to structural child elements in a container are preserved: they are collected into
    a separate block, otherwise illustration captions and marginal notes would be lost.
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
        self._last_block: DocumentBlock | None = None
        self._drops = DroppedBlockLog(
            resource_path=resource_path, section_id=section_id
        )

    def import_warnings(self) -> list[BookImportWarning]:
        """Section warnings: missing resources and dropped constructs."""
        return [*self.warnings, *self._drops.warnings()]

    def walk(self, container: etree._Element, context: _BlockContext) -> None:
        """Traverses container contents while preserving document order."""
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
        """Emits a block for the element or recurses into it as a container."""
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
        """Emits a block for an element without structural descendants."""
        builder = self._new_content_builder()
        builder.add_children(element)
        content = builder.build()
        block_type = _leaf_type(element, context=context, content=content)
        structural_path = _structural_path(element)
        reason = self._drop_reason(element, content=content, block_type=block_type)
        if reason is not None:
            self._drops.record(reason, structural_path)
            return
        if block_type is BlockType.PARAGRAPH and len(content.text) >= SPLIT_MIN_CHARS:
            split_blocks = self._split_and_add_paragraph(
                element=element,
                structural_path=structural_path,
                content=content,
                context=context,
            )
            if split_blocks:
                return
        self._add_block(
            element=element,
            structural_path=structural_path,
            type=block_type,
            content=content,
            support_status=block_support_status(block_type),
            context=context,
        )

    def _split_and_add_paragraph(
        self,
        element: etree._Element | None,
        structural_path: str,
        content: InlineContent,
        context: _BlockContext,
    ) -> list[DocumentBlock]:
        """Splits a long compound paragraph into a chain of blocks along sentence boundaries."""
        fragments = split_inline_content(content)
        if len(fragments) <= 1:
            return []

        total_parts = len(fragments)
        base_xpath = structural_path
        block_type = BlockType.PARAGRAPH
        element_id = element.get("id") if element is not None else None
        tag = (local_name(element) if element is not None else None) or "p"
        result_blocks: list[DocumentBlock] = []

        for k, fragment in enumerate(fragments, start=1):
            sub_path = f"{base_xpath}/text()[{k}]"
            if k == 1:
                attrs = dict(self._attributes(element, content=fragment, element_id=element_id))
            else:
                attrs = {"tag": tag}
                if fragment.anchor_ids:
                    attrs["anchor_ids"] = " ".join(fragment.anchor_ids)

            attrs["split_group"] = base_xpath
            attrs["split_index"] = str(k)
            attrs["split_total"] = str(total_parts)

            block = DocumentBlock.create(
                document_id=self._book_id,
                section_id=self.section_id,
                type=block_type,
                order=self._order,
                text=fragment.text,
                source_reference=SourceReference(
                    resource_path=self._resource_path,
                    structural_path=sub_path,
                    element_id=element_id if k == 1 else None,
                ),
                support_status=block_support_status(block_type),
                inline_annotations=fragment.annotations,
                parent_id=context.parent_id,
                attributes=attrs,
            )
            self.blocks.append(block)
            self._order += 1
            self._last_block = block
            if k == 1:
                self._pending_anchor_ids.clear()
            result_blocks.append(block)

        return result_blocks

    def _flush_run(
        self, run: _InlineRun, *, container: etree._Element, context: _BlockContext
    ) -> None:
        """Flushes an inline content run and, if it contains text, creates a block from it."""
        content = run.builder.build()
        run.index += 1
        run.builder = self._new_content_builder()
        if not content.text:
            return
        block_type = context.inherited_type or BlockType.PARAGRAPH
        structural_path = f"{_structural_path(container)}/text()[{run.index}]"
        if block_type is BlockType.PARAGRAPH and len(content.text) >= SPLIT_MIN_CHARS:
            split_blocks = self._split_and_add_paragraph(
                element=None,
                structural_path=structural_path,
                content=content,
                context=context,
            )
            if split_blocks:
                return
        self._add_block(
            element=None,
            structural_path=structural_path,
            type=block_type,
            content=content,
            support_status=block_support_status(block_type),
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
        """Adds a block to the section and returns it: it may become the parent of nested blocks."""
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
        self._last_block = block
        self._pending_anchor_ids.clear()
        return block

    def _attributes(
        self,
        element: etree._Element | None,
        *,
        content: InlineContent,
        element_id: str | None,
    ) -> dict[str, str]:
        """Auxiliary block attributes that do not affect its text."""
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
        """Image resource reference and its text description."""
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

    def _drop_reason(
        self,
        element: etree._Element,
        *,
        content: InlineContent,
        block_type: BlockType,
    ) -> DropReason | None:
        """Reason why a block is not created for the element, or `None`.

        An empty element is preserved if it is structural itself (image, divider, table)
        or serves as a target of an internal book link. A page break is preserved as the first
        in a chain: consecutive page breaks describe the same physical page boundary,
        and the second one is redundant in an ebook (PRD § 6).
        """
        if block_type is BlockType.PAGE_BREAK and not content.text:
            return (
                DropReason.DUPLICATE_PAGE_BREAK
                if self._follows_empty_page_break()
                else None
            )
        if content.text:
            return None
        if local_name(element) in _KEEP_EMPTY_TAGS:
            return None
        if element.get("id") or content.anchor_ids:
            return None
        return DropReason.EMPTY_DECORATION

    def _follows_empty_page_break(self) -> bool:
        """Whether the last added section block was an empty page break."""
        last = self._last_block
        return last is not None and last.type is BlockType.PAGE_BREAK and not last.text

    def _new_content_builder(self) -> InlineContentBuilder:
        return InlineContentBuilder(
            resource_path=self._resource_path, footnote_ids=self._footnote_ids
        )


def _is_structural(element: etree._Element) -> bool:
    """Whether the element produces a block or block container."""
    return local_name(element) in _STRUCTURAL_TAGS


def _is_anchor_only(element: etree._Element) -> bool:
    """Empty inline element existing solely as an internal link target.

    Gutenberg marks original print page numbers and illustration anchors with empty
    `<span>` and `<a>`. They contain no text, but are referenced by TOC and the book text,
    so such an element becomes an individual block rather than dissolving into adjacent text.
    """
    if "".join(element.itertext()).strip():
        return False
    return bool(element.get("id")) or any(
        descendant.get("id") for descendant in element.iterdescendants()
    )


def _is_container(element: etree._Element) -> bool:
    """Whether the element has structural descendants inside."""
    if local_name(element) in _ALWAYS_LEAF_TAGS:
        return False
    return next(element.iterdescendants(*_STRUCTURAL_QUALIFIED_NAMES), None) is not None


def _inherited_context(
    element: etree._Element, context: _BlockContext
) -> _BlockContext:
    """Type imposed by the container onto nested blocks."""
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
    """Block type for an element without structural descendants."""
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


def _is_page_break(element: etree._Element) -> bool:
    """Whether the element is marked as a print original page break.

    The marker is searched as an individual part of a token: publishers write both `pagebreak`
    and custom classes like `x-ebookmaker-pageno`, but `pagenote` is not a page break.
    """
    return any(_PAGE_BREAK_PATTERN.search(token) for token in _markers(element))


def _is_verse(element: etree._Element) -> bool:
    """Whether the element is marked as verse line or stanza."""
    markers = _markers(element)
    return bool(markers & _VERSE_MARKERS) or any(
        marker.startswith("z3998:verse") for marker in markers
    )


def _markers(element: etree._Element) -> frozenset[str]:
    """Element classes and `epub:type` as a single set of tokens."""
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
    """Absolute XPath of an element with same-name tag indices (ALG-01).

    The path is computed solely from document structure, so editing block text
    does not alter block identifier.
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
    """Non-empty values without duplicates, in order of appearance."""
    return tuple(dict.fromkeys(value for value in values if value))
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = ["EbookLibEpubReader"]
