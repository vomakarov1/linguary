"""EPUB export: target book assembly and built-in technical checks (PRD § 20).

Implementation of port `linguary.application.ports.epub.EpubWriter`. Assembly is symmetric to import:
one `DocumentSection` corresponds to one reading-order XHTML document, retaining the source
document path so that internal links remain valid without address rewriting.

A chapter is assembled in reading order: `Previously in Chapter ...` from the previous chapter summary,
`Before You Read`, then each block as reading units. When a layout is present those items share one
`section.lg-front` with the chapter's who-is-who and review, and empty items are left out. Units come
from `request.layout`. A missing layout, or a block absent from it, is one source span with no
translation and no notes. The writer does not build the layout. Legend, IPA key, and book who-is-who
are documents only when those tuples are non-empty. The LLM does not participate in assembly.

A `PRESERVE_OPAQUE` block is transferred using source markup from the EPUB book — otherwise anything
the importer did not descend into would be lost: whole tables and images inside them. An opaque copy
is made only for a leaf block: for a footnote block with nested paragraphs, an opaque copy would
duplicate both text and identifiers.

The order of guarantees is as follows: assembly itself normalizes the book into a valid state
(an unresolvable internal link unwraps into text, duplicate `id` is stripped, an `<img>` without a file
is removed — each case with a warning), and `verify_exported_epub` independently validates
the already-written file. A built-in check failure is an `EpubExportFatalError`, ensuring no partially-assembled
artifact remains on disk.
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
from linguary.adapters.epub.pronunciation_notices import (
    IPA_KEY_LINK_LABEL,
    PronunciationNoticeBundle,
    pronunciation_notice_bundle,
    sources_ipa_key_href,
)
from linguary.application.analysis_merge import sanitize_before_you_read_explanation
from linguary.application.layout.book import BookLayout
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
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    InlineAnnotation,
    InlineAnnotationType,
    SupportStatus,
)
from linguary.domain.document import is_front_matter

_XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
_EPUB_NAMESPACE = "http://www.idpf.org/2007/ops"
_OPF_NAMESPACE = "http://www.idpf.org/2007/opf"
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
_CONTAINER_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:container"

#: Package directory inside archive: documents and resources reside relative to it.
_PACKAGE_DIRECTORY = "OEBPS"
_PACKAGE_FILE = "content.opf"
_NAVIGATION_FILE = "nav.xhtml"
_STYLESHEET_FILE = "linguary.css"
_MIMETYPE = "application/epub+zip"

#: Source book table of contents in EPUB2 format. The target book is assembled as EPUB3 with
#: its own nav, whereas legacy NCX describes the no-longer-existing structure.
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"

#: Support statuses under which a resource is included in the target book.
_EXPORTED_STATUSES = frozenset({SupportStatus.SUPPORTED, SupportStatus.PRESERVE_OPAQUE})

#: Consecutive blocks are grouped under a shared parent: importer splits lists
#: and blockquotes into individual elements, and without a wrapper the book would produce a single-item list
#: per item.
_GROUPING_TAGS: Mapping[BlockType, str] = {
    BlockType.LIST_ITEM: "ul",
    BlockType.QUOTE: "blockquote",
}

#: Elements where block text is not written inside: their content is the element itself.
_VOID_TAGS = frozenset({"img", "hr", "br"})

#: Styling tags by inline markup type.
_ANNOTATION_TAGS: Mapping[InlineAnnotationType, str] = {
    InlineAnnotationType.ITALIC: "i",
    InlineAnnotationType.BOLD: "b",
    InlineAnnotationType.EMPHASIS: "em",
    InlineAnnotationType.LINK: "a",
    InlineAnnotationType.FOOTNOTE_REFERENCE: "a",
}

#: Reading-unit styling. The left line stays on the translation. Parts have no boxes,
#: no background, and no generated text. Type stays at least 0.85em except the marker.
_STYLESHEET = """/* Linguary: оформление учебных слоёв поверх оригинала книги. */
.linguary-previously,
.linguary-before-you-read {
  margin: 1.5em 0;
  padding: 0.2em 0 0.2em 0.9em;
  border-left: 0.2em solid;
  font-size: 1em;
  color: inherit;
}

.linguary-previously .linguary-summary-translation {
  margin-top: 0.25em;
  padding-left: 0.9em;
  font-style: italic;
  color: inherit;
}

.lg-unit {
  margin: 0 0 0.7em;
  page-break-inside: avoid;
  break-inside: avoid;
  color: inherit;
}

.lg-unit.lg-end {
  margin-bottom: 1.4em;
}

.lg-cont > .lg-src {
  text-indent: 0;
}

.lg-tr {
  margin: 0.25em 0 0 1em;
  padding-left: 0.7em;
  border-left: 0.12em solid;
  font-size: 0.9em;
  color: inherit;
  hyphens: auto;
  page-break-before: avoid;
  break-before: avoid;
}

.lg-notes {
  list-style: none;
  margin: 0.25em 0 0 1em;
  padding-left: 1.7em;
  font-size: 0.85em;
  color: inherit;
  page-break-before: avoid;
  break-before: avoid;
}

.lg-notes > li {
  text-indent: -1em;
}

a.lg-ref {
  text-decoration: none;
  color: inherit;
}

a.lg-ref sup {
  font-size: 0.7em;
}

p {
  orphans: 2;
  widows: 2;
}

.linguary-anchors {
  display: none;
}

.lg-front,
.lg-sec {
  margin: 0.4em 0 1.6em;
  font-size: 0.9em;
  color: inherit;
  hyphens: auto;
}

.lg-front h2,
.lg-front h3,
.lg-front h4,
.lg-sec h1,
.lg-sec h2,
.lg-sec h3 {
  margin: 0.8em 0 0.2em;
  font-size: 1em;
  page-break-after: avoid;
  break-after: avoid;
  color: inherit;
}

.lg-front p,
.lg-sec p {
  margin: 0.25em 0;
  text-indent: 0;
  color: inherit;
}

.lg-entries {
  list-style: none;
  margin: 0.3em 0;
  padding-left: 1.2em;
  color: inherit;
}

.lg-entries > li {
  margin: 0.25em 0;
  text-indent: -1.2em;
  page-break-inside: avoid;
  break-inside: avoid;
}

.lg-head {
  font-weight: bold;
}

.lg-src-ref,
.lg-method {
  font-style: italic;
}

.lg-key {
  width: 100%;
  margin: 0.4em 0;
  font-size: 0.95em;
  border-collapse: collapse;
  color: inherit;
}

.lg-key th,
.lg-key td {
  padding: 0.2em 0.6em 0.2em 0;
  text-align: left;
  vertical-align: top;
  border-bottom: 0.04em solid;
  color: inherit;
}

.lg-newpage {
  page-break-before: always;
  break-before: page;
}
"""


class EpubExportWriter(EpubWriter):
    """Assembles target EPUB from canonical model, resources, and processing results."""

    def write(self, request: EpubWriteRequest) -> ExportedBook:
        """Assembles the book and validates the exported file; verification failure is `FATAL`.

        The book is assembled next to the target path and replaces it only after
        passing built-in verification checks: an unsuccessful assembly must not destroy
        a previous successful export of the same book.
        """
        warnings = _ExportWarnings()
        payloads = dict(_resource_payloads(request, warnings))
        try:
            notices = pronunciation_notice_bundle(request)
        except (OSError, ValueError):
            raise EpubExportFatalError('уведомления об источнике транскрипций недоступны') from None
        used = set(payloads) | {
            _contained_name(section.source_resource_path)
            for section in request.document.ordered_sections
        }
        used.update((_NAVIGATION_FILE, _STYLESHEET_FILE, _PACKAGE_FILE))
        pages = _book_page_hrefs(request.layout, used)
        notices_href = None
        provenance_href = None
        media_types = dict(_resource_media_types(request))
        if notices is not None:
            notices_href = _unique_resource_name('linguary/pronunciation-sources.xhtml', used)
            provenance_href = _unique_resource_name('linguary/pronunciation-sources.json', used)
            payloads[provenance_href] = notices.provenance_json()
            media_types[provenance_href] = 'application/json'
        with _OpaqueSource(request.source_epub_path) as opaque:
            chapters = _render_chapters(request, opaque=opaque, warnings=warnings)
        chapters = _insert_book_pages(chapters, request, pages)
        if notices is not None:
            chapters += (_pronunciation_notices_chapter(
                notices,
                href=notices_href,
                provenance_href=provenance_href,
                language=request.document.source_language,
                ipa_key_href=_ipa_key_link(notices_href, pages.ipa),
            ),)
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
            media_types=media_types,
        )
        target = Path(request.target_path)
        draft = target.with_name(f"{target.name}.part")
        try:
            _write_archive(draft, payloads=payloads)
            failures = verify_exported_epub(
                draft, expected_spine=tuple(chapter.href for chapter in chapters)
            )
            if failures:
                raise EpubExportFatalError(
                    "собранный EPUB не прошёл встроенные технические проверки: "
                    + "; ".join(f"{failure.kind}: {failure.message}" for failure in failures)
                )
            draft.replace(target)
        finally:
            draft.unlink(missing_ok=True)
        return ExportedBook(path=str(target), warnings=warnings.collected())


@dataclass(slots=True)
class _Chapter:
    """Ready chapter document: path inside book, title, and XHTML tree."""

    section_id: str
    href: str
    title: str
    root: etree._Element


def _pronunciation_notices_chapter(
    notices: PronunciationNoticeBundle,
    *,
    href: str,
    provenance_href: str,
    language: str,
    ipa_key_href: str | None = None,
) -> _Chapter:
    title = 'Источники транскрипций и лицензии'
    root = _xhtml_document(title=title, language=language,
                           stylesheet=_relative_href(href, _STYLESHEET_FILE))
    body = root.find(f'{{{_XHTML_NAMESPACE}}}body')
    section = _sub(body, 'section', {'id': 'pronunciation-sources', 'class': 'linguary-pronunciation-sources'})
    _sub(section, 'h1').text = title
    _sub(section, 'p').text = 'Сведения об источнике добавленных транскрипций и условия использования словарных данных.'
    link_href = sources_ipa_key_href(ipa_key_href)
    if link_href is not None:
        paragraph = _sub(section, 'p')
        link = _sub(paragraph, 'a', {'href': link_href})
        link.text = IPA_KEY_LINK_LABEL
    _sub(section, 'a', {'href': _relative_href(href, provenance_href)}).text = 'Использованные транскрипции (JSON)'
    _sub(section, 'pre').text = notices.text
    return _Chapter('linguary-pronunciation-sources', href, title, root)


@dataclass(frozen=True, slots=True)
class _BookPageHrefs:
    """Generated apparatus paths. None means that tuple was empty."""

    legend: str | None = None
    ipa: str | None = None
    who: str | None = None


def _book_page_hrefs(layout: BookLayout | None, used: set[str]) -> _BookPageHrefs:
    """Reserve a file only for a non-empty legend, IPA key, or book who-is-who."""
    if layout is None:
        return _BookPageHrefs()
    return _BookPageHrefs(
        legend=_unique_resource_name("linguary/legend.xhtml", used) if layout.legend else None,
        ipa=_unique_resource_name("linguary/ipa-key.xhtml", used) if layout.ipa_key else None,
        who=_unique_resource_name("linguary/who-is-who.xhtml", used) if layout.who_is_who else None,
    )


def _ipa_key_link(base: str | None, ipa_href: str | None) -> str | None:
    """Relative link to the IPA key, or none when that page was not reserved."""
    if base is None or ipa_href is None:
        return None
    return sources_ipa_key_href(f"{_relative_href(base, ipa_href)}#ipa-key")


def _insert_book_pages(
    chapters: Sequence[_Chapter],
    request: EpubWriteRequest,
    pages: _BookPageHrefs,
) -> tuple[_Chapter, ...]:
    """Front matter, then legend and IPA key, then the book, then who-is-who.

    The sources page is appended by the caller and stays last.
    """
    layout = request.layout
    if layout is None or (pages.legend is None and pages.ipa is None and pages.who is None):
        return tuple(chapters)
    from linguary.adapters.epub.book_pages import (
        IPA_KEY_TITLE,
        LEGEND_TITLE,
        WHO_IS_WHO_TITLE,
        ipa_key_document,
        legend_document,
        who_is_who_document,
    )

    leading, rest = _split_leading_front_matter(chapters, request.document)
    target = request.document.target_language
    source = request.document.source_language
    front: list[_Chapter] = []
    if pages.legend is not None:
        front.append(
            _Chapter(
                "linguary-legend",
                pages.legend,
                LEGEND_TITLE,
                legend_document(
                    layout.legend,
                    language=target,
                    stylesheet=_relative_href(pages.legend, _STYLESHEET_FILE),
                    ipa_key_href=_ipa_key_link(pages.legend, pages.ipa),
                ),
            )
        )
    if pages.ipa is not None:
        front.append(
            _Chapter(
                "linguary-ipa-key",
                pages.ipa,
                IPA_KEY_TITLE,
                ipa_key_document(
                    layout.ipa_key,
                    language=target,
                    example_language=source,
                    stylesheet=_relative_href(pages.ipa, _STYLESHEET_FILE),
                ),
            )
        )
    who: tuple[_Chapter, ...] = ()
    if pages.who is not None:
        who = (
            _Chapter(
                "linguary-who-is-who",
                pages.who,
                WHO_IS_WHO_TITLE,
                who_is_who_document(
                    layout.who_is_who,
                    language=target,
                    source_language=source,
                    stylesheet=_relative_href(pages.who, _STYLESHEET_FILE),
                ),
            ),
        )
    return tuple(leading) + tuple(front) + tuple(rest) + who


def _split_leading_front_matter(
    chapters: Sequence[_Chapter], document: CanonicalDocument
) -> tuple[list[_Chapter], list[_Chapter]]:
    """Opening front matter stays ahead of the legend. Later sections keep their order."""
    leading: list[_Chapter] = []
    rest: list[_Chapter] = []
    started = False
    for chapter in chapters:
        section = document.section(chapter.section_id)
        if not started and document.is_front_matter(section):
            leading.append(chapter)
            continue
        started = True
        rest.append(chapter)
    return leading, rest


@dataclass(slots=True)
class _Occurrences:
    """Occurrences of a single divergence kind in a single chapter: example and count."""

    example: str
    count: int = 1


@dataclass(slots=True)
class _ExportWarnings:
    """Export warnings aggregated by kind and chapter.

    A separate line for every dropped page break would turn the warning list into noise:
    Gutenberg editions contain hundreds of such constructs. Occurrences are therefore
    accumulated, and the user receives one warning per kind and chapter — with an example
    and the count of affected occurrences.
    """

    _occurrences: dict[tuple[ExportWarningKind, str | None], _Occurrences] = field(
        default_factory=dict
    )

    def record(
        self, kind: ExportWarningKind, *, section_id: str | None, detail: str
    ) -> None:
        """Records a single instance of divergence between the exported and source book."""
        occurrences = self._occurrences.get((kind, section_id))
        if occurrences is None:
            self._occurrences[(kind, section_id)] = _Occurrences(example=detail)
            return
        occurrences.count += 1

    def collected(self) -> tuple[ExportWarning, ...]:
        """Export warnings from accumulated occurrences."""
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
    """Book resource contents for the new archive.

    A resource without a stored copy does not break the book: this is expected for a book imported
    without a resource store — it is assembled without images, and the user receives a warning.
    A declared copy missing from disk is a different matter: the store is corrupted,
    and assembling a book with a missing required resource is forbidden.
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
    """Book resource media types as declared in the source book manifest."""
    return {
        _contained_name(resource.source_path): resource.media_type
        for resource in request.document.resources
    }


_FRONT_MATTER_FILENAMES = frozenset(
    {
        "cover",
        "title",
        "titlepage",
        "title_page",
        "halftitle",
        "half_title",
        "copyright",
        "colophon",
        "imprint",
        "contents",
        "toc",
        "nav",
    }
)

_FRONT_MATTER_TITLES = frozenset(
    {
        "cover",
        "contents",
        "table of contents",
        "toc",
        "оглавление",
        "содержание",
        "title",
        "title page",
        "titlepage",
        "half title",
        "half-title",
        "титул",
        "титульный лист",
        "copyright",
        "colophon",
    }
)


_is_front_matter = is_front_matter


def _render_chapters(
    request: EpubWriteRequest, *, opaque: _OpaqueSource, warnings: _ExportWarnings,
) -> tuple[_Chapter, ...]:
    """Chapter documents in reading order.

    Introductory front matter sections (title page, table of contents) and notes pages have no recap,
    and a story chapter following them recaps the last preceding story chapter.
    Story chapter numbering is not displaced by front matter sections.
    """
    renderer = _ChapterRenderer(request=request, opaque=opaque, warnings=warnings)
    chapters: list[_Chapter] = []
    previous: DocumentSection | None = None
    story_chapter_number = 1
    for section in request.document.ordered_sections:
        is_front = request.document.is_front_matter(section)
        number = 0 if is_front else story_chapter_number
        chapters.append(
            renderer.render(
                section,
                number=number,
                previous=None if is_front else previous,
                is_front_matter=is_front,
            )
        )
        if not section.is_notes_only and not is_front:
            previous = section
            story_chapter_number += 1
    return tuple(chapters)


class _ChapterRenderer:
    """Renders a single chapter XHTML document."""

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
        self,
        section: DocumentSection,
        *,
        number: int,
        previous: DocumentSection | None,
        is_front_matter: bool = False,
    ) -> _Chapter:
        """Assembles the whole chapter: learning sections, blocks, and aid layers.

        A notes page is assembled without chapter markup or learning sections before
        the text: its footnotes are translated, but it is not a chapter itself.
        Introductory front matter sections (title page, table of contents) are assembled without
        learning materials, footnotes, or translations.
        """
        href = _contained_name(section.source_resource_path)
        root = _xhtml_document(
            title=section.title,
            language=self._document.source_language,
            stylesheet=_relative_href(href, _STYLESHEET_FILE),
        )
        body = root.find(f"{{{_XHTML_NAMESPACE}}}body")
        if section.is_notes_only:
            self._append_blocks(
                body, section, href=href, number=number, is_front_matter=False
            )
        elif is_front_matter:
            container = _sub(
                body, "section", {f"{{{_EPUB_NAMESPACE}}}type": "frontmatter"}
            )
            self._append_blocks(
                container, section, href=href, number=number, is_front_matter=True
            )
        else:
            container = _sub(body, "section", {f"{{{_EPUB_NAMESPACE}}}type": "chapter"})
            self._append_chapter_front(container, section, previous)
            self._append_blocks(
                container, section, href=href, number=number, is_front_matter=False
            )
        return _Chapter(
            section_id=section.id, href=href, title=section.title, root=root
        )

    def _append_chapter_front(
        self,
        container: etree._Element,
        section: DocumentSection,
        previous: DocumentSection | None,
    ) -> None:
        """One `section.lg-front` when a layout is present. Empty items are omitted.

        Without a layout the historical Previously and Before You Read sections
        stay direct children of the chapter. Front matter and notes pages do
        not call this.
        """
        if self._request.layout is None:
            self._append_previously(container, previous)
            self._append_before_you_read(container, section)
            return
        from linguary.adapters.epub.book_pages import append_review, append_who

        front = _element("section", {"class": "lg-front"})
        self._append_previously(front, previous)
        chapter = self._request.layout.chapters.get(section.id)
        if chapter is not None:
            append_who(front, chapter.who_is_who, source_language=self._document.source_language)
            append_review(front, chapter.review, source_language=self._document.source_language)
        self._append_before_you_read(front, section)
        if len(front):
            container.append(front)

    def _append_previously(
        self, container: etree._Element, previous: DocumentSection | None
    ) -> None:
        """`Previously in Chapter ...` section from previous chapter summary (PRD § 19.2)."""
        if previous is None:
            return
        material = self._request.learning_materials.get(previous.id)
        summary = material.chapter_summary.strip() if material else ""
        if not summary:
            return
        block = _sub(container, "section", {"class": "linguary-previously"})
        _sub(block, "h2").text = _previously_heading(previous.title)
        for index, paragraph in enumerate(_paragraphs(summary)):
            attributes = (
                {"class": "linguary-summary-translation", "lang": "ru"}
                if index % 2 == 1 and _is_russian_paragraph(paragraph)
                else None
            )
            _append_text(_sub(block, "p", attributes), paragraph)

    def _append_before_you_read(
        self, container: etree._Element, section: DocumentSection
    ) -> None:
        """`Before You Read` section from this chapter's analysis (PRD § 19.1)."""
        material = self._request.learning_materials.get(section.id)
        if material is None or not material.before_you_read:
            return
        block = _sub(container, "section", {"class": "linguary-before-you-read"})
        _sub(block, "h2").text = "Before You Read"
        definitions = _sub(block, "dl")
        for entry in material.before_you_read:
            _sub(definitions, "dt").text = entry.expression
            _sub(definitions, "dd").text = (
                entry.explanation
                if section.id in self._request.verified_body_sections
                else sanitize_before_you_read_explanation(entry.explanation)
            )

    def _append_blocks(
        self,
        container: etree._Element,
        section: DocumentSection,
        *,
        href: str,
        number: int,
        is_front_matter: bool = False,
    ) -> None:
        """Chapter blocks in source order, each one rendered as reading units."""
        blocks = section.ordered_blocks
        children: dict[str, list[DocumentBlock]] = {}
        for block in blocks:
            if block.parent_id is not None:
                children.setdefault(block.parent_id, []).append(block)
        state = _SectionState(section=section, href=href, number=number)
        self._reserve_opaque_ids(blocks, state)
        roots = tuple(block for block in blocks if block.parent_id is None)
        self._append_siblings(
            container,
            roots,
            children=children,
            state=state,
            is_front_matter=is_front_matter,
        )

    def _reserve_opaque_ids(
        self, blocks: Sequence[DocumentBlock], state: _SectionState
    ) -> None:
        """Ids already present in copied markup stay out of generated note ids.

        A shell with children is not copied. Its identifiers are written with
        the shell, so reserving them here would drop the targets.
        """
        parents = {block.parent_id for block in blocks if block.parent_id is not None}
        for block in blocks:
            if block.support_status is not SupportStatus.PRESERVE_OPAQUE:
                continue
            if block.id in parents:
                continue
            copied = self._opaque.element(block)
            if copied is None:
                continue
            for node in copied.iter():
                identifier = node.get("id")
                if identifier:
                    state.used_ids.add(identifier)

    def _append_siblings(
        self,
        parent: etree._Element,
        blocks: Sequence[DocumentBlock],
        *,
        children: Mapping[str, list[DocumentBlock]],
        state: _SectionState,
        parent_type: BlockType | None = None,
        is_front_matter: bool = False,
    ) -> None:
        """Sibling blocks. Consecutive list items and quotes share one wrapper."""
        group: etree._Element | None = None
        grouped: BlockType | None = None
        for block in blocks:
            wrapper = (
                None
                if block.type is BlockType.QUOTE and parent_type is BlockType.QUOTE
                else _GROUPING_TAGS.get(block.type)
            )
            if wrapper is None or block.type is not grouped:
                group, grouped = None, None
            if wrapper is not None and group is None:
                group, grouped = _sub(parent, wrapper), block.type
            host = group if group is not None else parent
            self._append_block(
                host,
                block,
                children=children,
                state=state,
                parent_type=parent_type,
                is_front_matter=is_front_matter,
            )

    def _append_block(
        self,
        parent: etree._Element,
        block: DocumentBlock,
        *,
        children: Mapping[str, list[DocumentBlock]],
        state: _SectionState,
        parent_type: BlockType | None,
        is_front_matter: bool = False,
    ) -> None:
        """One block: opaque markup, or a shell plus reading units and nested blocks."""
        nested = tuple(children.get(block.id, ()))
        if block.support_status is SupportStatus.PRESERVE_OPAQUE and not nested:
            if self._append_opaque(parent, block, state):
                return
        host = parent
        if block.type is BlockType.LIST_ITEM:
            host = _sub(parent, "li")
        elif (
            block.type is BlockType.FOOTNOTE
            and parent_type is not BlockType.FOOTNOTE
        ):
            # The paragraph inside a footnote is the same block type. One aside.
            host = _sub(
                parent, "aside", {f"{{{_EPUB_NAMESPACE}}}type": "footnote"}
            )
            self._attach_anchors(parent, host, block, state=state, inside=True)
        if _renders_units(block, nested=bool(nested)):
            self._render_units(
                host, block, state=state, is_front_matter=is_front_matter
            )
        if nested:
            self._append_siblings(
                host,
                nested,
                children=children,
                state=state,
                parent_type=block.type,
                is_front_matter=is_front_matter,
            )

    def _append_opaque(
        self,
        parent: etree._Element,
        block: DocumentBlock,
        state: _SectionState,
    ) -> bool:
        """Copy source markup for a leaf the importer did not descend into.

        Returns false when the copy is missing and the text still has to be written.
        """
        copied = self._opaque.element(block)
        if copied is not None:
            parent.append(copied)
            self._attach_anchors(parent, copied, block, state=state, inside=False)
            return True
        if not block.text and not _anchors(block):
            return True
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
        return False

    def _render_units(
        self,
        host: etree._Element,
        block: DocumentBlock,
        *,
        state: _SectionState,
        is_front_matter: bool,
    ) -> None:
        """Units from the request layout. Front matter and a missing entry stay source-only."""
        from linguary.adapters.epub.layout_render import render_unit

        for unit in self._units_for(block, state=state, is_front_matter=is_front_matter):
            render_unit(
                host,
                block,
                unit,
                state=state,
                source_language=self._document.source_language,
                target_language=self._document.target_language,
            )

    def _units_for(
        self,
        block: DocumentBlock,
        *,
        state: _SectionState,
        is_front_matter: bool,
    ):
        from linguary.adapters.epub.layout_render import source_only_unit

        if is_front_matter:
            return (source_only_unit(block),)
        layout = self._request.layout
        chapter = None if layout is None else layout.chapters.get(state.section.id)
        units = None if chapter is None else chapter.units.get(block.id)
        if not units:
            return (source_only_unit(block),)
        return units

    def _attach_anchors(
        self,
        parent: etree._Element,
        element: etree._Element,
        block: DocumentBlock,
        *,
        state: _SectionState,
        inside: bool,
    ) -> None:
        """Block anchors: how the book text references this block.

        The first anchor becomes the element's id attribute; remaining anchors become empty
        `span` elements: losing them would make internal book references unresolvable.
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

@dataclass(slots=True)
class _SectionState:
    """Assembly state of a single chapter: allocated identifiers and note counter."""

    section: DocumentSection
    href: str
    number: int
    used_ids: set[str] = field(default_factory=set)
    notes: int = 0


class _OpaqueSource:
    """Source book as the markup provider for `PRESERVE_OPAQUE` blocks.

    An inaccessible or unparseable source file does not halt export: the block is transferred
    as text, and the user receives a warning. The book remains exportable even when
    the user has removed the source EPUB from the filesystem.
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
        """Copy of the source block element, or `None` if unavailable."""
        root = self._document(block.source_reference.resource_path)
        if root is None:
            return None
        found = _element_at(root, block.source_reference.structural_path)
        if found is None:
            return None
        return _in_xhtml_namespace(deepcopy(found))

    def _document(self, resource_path: str) -> etree._Element | None:
        """Parsed source document of the book; parse results are cached."""
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
    """Brings the book into a state where all internal references are resolvable.

    The source book may reference constructs that do not exist in the canonical model:
    an empty decoration dropped by import, or a document outside reading order. Leaving such
    a reference would produce a book that fails its own verification checks, so the reference
    is unwrapped into plain text, and the user receives a warning.
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
    """Document identifiers; duplicate `id` is stripped from later elements."""
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
    """Retains the link if it is resolvable, otherwise unwraps it into text."""
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
    """Removes an image whose file is absent from the target book."""
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
    """Appends chapter documents, navigation, stylesheet, and package file to book resources."""
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
    """EPUB3 navigation document: table of contents links to chapters in reading order."""
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
    """OPF package file: book metadata, manifest, and reading order spine."""
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
    """Book metadata: retains metadata parsed by importer (PRD § 20)."""
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
    """Manifest items; returns the identifier for each book file."""
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
    """Writes the book archive: uncompressed `mimetype` first, then container and contents."""
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
    """`META-INF/container.xml`: location of package file in archive."""
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
    """Kind of verification failure during built-in technical check (PRD § 20)."""

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
    """A single failed verification check for an exported file."""

    kind: ExportCheckKind
    message: str


def verify_exported_epub(
    path: Path | str, *, expected_spine: Sequence[str] = ()
) -> tuple[ExportCheckFailure, ...]:
    """Built-in technical checks for an exported EPUB (PRD § 20).

    Validates an already-written file rather than intermediate assembly state: this ensures
    verification observes exactly what an e-reader would see, and allows testing against arbitrary
    or deliberately malformed files. No external tools required; verification is always performed.
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
    """Verification checks on an opened book archive."""
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
    """Package file path resolved via `META-INF/container.xml`."""
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
    """Manifest item: path inside archive and content media type."""

    item_id: str
    name: str
    href: str
    media_type: str
    properties: str


def _manifest(
    package: etree._Element, *, directory: str, failures: list[ExportCheckFailure]
) -> Mapping[str, _ManifestItem]:
    """Manifest items mapped by their identifiers."""
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
    """Reading-order documents; reference outside manifest is a failed check."""
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
    """Files declared by manifest but absent from the archive."""
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
    """Parsed XHTML documents of the book by their archive paths."""
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
    """Duplicate identifiers within a single document."""
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
    """Presence and validity of the book table of contents."""
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
    """Documents targeted by table of contents entries, in order of appearance."""
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
    """Resolvability of internal links, footnote references, and image paths."""
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
    """Internal link that leads nowhere."""
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
    """Image without a file in the archive or without declaration in the manifest."""
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
    """Chapter order: matching between reading order and table of contents."""
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
    """Parses an archive file as XML; unparseable file produces `None`."""
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
    """Element in XHTML or specified namespace with attributes."""
    qualified = tag if tag.startswith("{") else f"{{{_XHTML_NAMESPACE}}}{tag}"
    element = etree.Element(qualified, nsmap=dict(nsmap) if nsmap else None)
    for name, value in (attributes or {}).items():
        element.set(name, value)
    return element


def _sub(
    parent: etree._Element, tag: str, attributes: Mapping[str, str] | None = None
) -> etree._Element:
    """Creates a child element and appends it to the parent."""
    element = _element(tag, attributes)
    parent.append(element)
    return element


def _xhtml_document(*, title: str, language: str, stylesheet: str) -> etree._Element:
    """Book document scaffold: title, encoding, and stylesheet link."""
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
    """Serializes a book document along with XML declaration."""
    etree.cleanup_namespaces(root)
    return etree.tostring(
        root, xml_declaration=True, encoding="utf-8", pretty_print=False
    )


def _renders_units(block: DocumentBlock, *, nested: bool) -> bool:
    """Whether this block contributes its own reading unit.

    An empty footnote is only a shell around nested paragraphs. Any other block
    with text, an anchor, or no children still needs a unit, as do images and
    page breaks.
    """
    if block.type is BlockType.FOOTNOTE and not block.text:
        return False
    if block.text or _anchors(block) or not nested:
        return True
    return block.type in (BlockType.IMAGE, BlockType.PAGE_BREAK)


def _anchors(block: DocumentBlock) -> tuple[str, ...]:
    """Identifiers through which the book references the block."""
    return tuple(block.attributes.get("anchor_ids", "").split())


def _nestable(
    annotations: Sequence[InlineAnnotation],
) -> tuple[InlineAnnotation, ...]:
    """Markup suitable for constructing an element tree.

    Markup was extracted from valid XHTML and is usually properly nested. Intersections
    without strict containment cannot form a tree and are dropped: losing styling for a fragment
    is preferable to generating invalid XHTML.
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


def _append_range(
    host: etree._Element,
    text: str,
    annotations: Sequence[InlineAnnotation],
    *,
    start: int,
    end: int,
    state: _SectionState,
) -> None:
    """Text range along with markup annotations starting within it."""
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
    """Markup element attributes: link target and footnote reference marker."""
    if annotation.target is None:
        return {}
    attributes = {"href": _href_from_target(annotation.target, base=state.href)}
    if annotation.type is InlineAnnotationType.FOOTNOTE_REFERENCE:
        attributes[f"{{{_EPUB_NAMESPACE}}}type"] = "noteref"
    return attributes


def _href_from_target(target: str, *, base: str) -> str:
    """Link href relative to the chapter document.

    Importer normalizes internal links into "book resource path#fragment" format, i.e.
    relative to the package directory; inside a chapter document the address must be relative to the chapter.
    """
    if _is_external(target):
        return target
    path, separator, fragment = target.partition("#")
    if not path:
        return f"{separator}{fragment}"
    normalized = posixpath.normpath(path)
    if separator and normalized == posixpath.normpath(base):
        return f"#{fragment}"
    relative = _relative_href(base, normalized)
    return f"{relative}{separator}{fragment}" if separator else relative


def _append_text(host: etree._Element, text: str) -> None:
    """Appends text, converting block newlines into `<br/>`."""
    if not text:
        return
    parts = text.split("\n")
    _append_characters(host, parts[0])
    for part in parts[1:]:
        _sub(host, "br")
        _append_characters(host, part)


def _append_characters(host: etree._Element, text: str) -> None:
    """Appends characters to the tail of element content."""
    if not text:
        return
    if len(host):
        last = host[-1]
        last.tail = (last.tail or "") + text
    else:
        host.text = (host.text or "") + text


def _unwrap(element: etree._Element) -> None:
    """Replaces element with its children while preserving surrounding text."""
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
    """Appends text before the specified child position."""
    if not text:
        return
    if index == 0:
        parent.text = (parent.text or "") + text
    else:
        previous = parent[index - 1]
        previous.tail = (previous.tail or "") + text


def _paragraphs(text: str) -> tuple[str, ...]:
    """Text paragraphs: empty line separates paragraphs, single newline remains inline."""
    return tuple(part.strip() for part in text.split("\n\n") if part.strip())


def _is_russian_paragraph(text: str) -> bool:
    """Whether a paragraph is predominantly Russian, for styling parallel summaries."""
    letters = tuple(character for character in text if character.isalpha())
    if not letters:
        return False
    cyrillic = sum("\u0400" <= character <= "\u052F" for character in letters)
    return cyrillic / len(letters) > 0.5


def _previously_heading(previous_title: str) -> str:
    """Heading for previous chapter summary in original language (PRD § 19.2)."""
    title = previous_title.strip()
    if not title:
        return "Previously"
    if title.lower().startswith("chapter"):
        return f"Previously in {title}"
    if _looks_like_chapter_number(title):
        return f"Previously in Chapter {title}"
    return f"Previously in {title}"


def _looks_like_chapter_number(title: str) -> bool:
    """Whether the section is titled solely with a number (Arabic or Roman)."""
    candidate = title.rstrip(".").strip()
    if candidate.isdigit():
        return True
    return bool(candidate) and set(candidate.upper()) <= set("IVXLCDM")


def _unique_id(candidate: str, used: set[str]) -> str:
    """Identifier unique within the document."""
    identifier = candidate
    suffix = 1
    while identifier in used:
        suffix += 1
        identifier = f"{candidate}-{suffix}"
    used.add(identifier)
    return identifier


def _unique_resource_name(candidate: str, used: set[str]) -> str:
    """Reserve a generated book resource without replacing an author resource."""
    stem, extension = posixpath.splitext(candidate)
    result = candidate
    suffix = 1
    while result in used:
        suffix += 1
        result = f'{stem}-{suffix}{extension}'
    used.add(result)
    return result


def _contains_id(element: etree._Element, identifier: str) -> bool:
    """Whether identifier exists on the element itself or within its subtree."""
    return any(node.get("id") == identifier for node in element.iter())


def _in_xhtml_namespace(element: etree._Element) -> etree._Element:
    """Ensures copied markup conforms to the XHTML namespace.

    Editions lacking a declared namespace do occur; without adjustment such fragments
    serialize with empty `xmlns` and cease to be part of the book document.
    """
    for node in element.iter():
        if isinstance(node.tag, str) and not node.tag.startswith("{"):
            node.tag = f"{{{_XHTML_NAMESPACE}}}{node.tag}"
        node.attrib.pop("summary", None)
    return element


def _element_at(root: etree._Element, structural_path: str) -> etree._Element | None:
    """Document element at importer's structural path (ALG-01)."""
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
    """Structural path step formatted like `div[2]`; other steps are not supported."""
    tag, separator, rest = step.partition("[")
    if not separator or not rest.endswith("]") or not tag.isidentifier():
        return None
    position = rest[:-1]
    return (tag, int(position)) if position.isdigit() else None


def _nth_child(
    parent: etree._Element, *, tag: str, index: int
) -> etree._Element | None:
    """1-based `index`-th child with the specified tag name."""
    seen = 0
    for child in parent:
        if local_name(child) != tag:
            continue
        seen += 1
        if seen == index:
            return child
    return None


def _resolve_target(href: str, *, base: str) -> tuple[str, str]:
    """Resolves link target into a (archive_path, fragment) pair."""
    path, _, fragment = href.partition("#")
    if not path:
        return base, fragment
    directory = posixpath.dirname(base)
    return posixpath.normpath(posixpath.join(directory, path)), fragment


def _relative_href(base: str, target: str) -> str:
    """Path to target book file relative to `base` document."""
    directory = posixpath.dirname(base)
    return posixpath.relpath(target, directory) if directory else target


def _is_external(href: str) -> bool:
    """Whether the address points outside the book."""
    scheme, separator, _ = href.partition(":")
    return bool(separator) and scheme.isalpha()


def _contained_name(resource_path: str) -> str:
    """File path inside package directory; traversing outside is forbidden."""
    normalized = posixpath.normpath(resource_path.replace("\\", "/").strip())
    if not normalized or normalized.startswith(("/", "..")):
        raise EpubExportFatalError(
            f"путь {resource_path!r} ведёт за пределы каталога книги"
        )
    return normalized


def _media_type(href: str) -> str:
    """Content media type for a book file based on its extension."""
    return _MEDIA_TYPES.get(posixpath.splitext(href)[1].lower(), "application/octet-stream")


#: Media types of files generated directly by the exporter. Book resource types are sourced
#: from the canonical model and therefore not listed here.
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
