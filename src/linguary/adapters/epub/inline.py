"""Inline formatting, links, and footnotes within a single block.

`DocumentBlock.text` is stored without XHTML (PRD § 7), so styling is extracted separately:
italics, bold, semantic emphasis, internal hyperlinks, and footnote references are represented
as `InlineAnnotation` with character offsets in the normalized block text.

Block anchor IDs are also collected here: element IDs by which the book references the block.
Gutenberg footnote references point to block element `id`, whereas the backlink inside the footnote
points to an inline `<a>` element `id` within a paragraph, meaning a single `SourceReference.element_id`
is insufficient for restoring links.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from lxml import etree, html as lxml_html  # ty: ignore[missing-module-member]

from linguary.domain import InlineAnnotation, InlineAnnotationType

#: Collapsible XHTML whitespace characters replaced by a single space. Non-breaking space
#: is excluded: it carries semantic meaning and is preserved as-is.
_COLLAPSIBLE_WHITESPACE = re.compile(r"[ \t\r\n\f\v]+")

#: Inline styling tags and their corresponding annotation types.
_ANNOTATION_TAGS: dict[str, InlineAnnotationType] = {
    "i": InlineAnnotationType.ITALIC,
    "em": InlineAnnotationType.EMPHASIS,
    "b": InlineAnnotationType.BOLD,
    "strong": InlineAnnotationType.BOLD,
}

#: Tags that produce no text directly but represent line breaks inside a block.
_LINE_BREAK_TAGS = frozenset({"br"})

#: Non-content tags whose text is excluded from book text.
_NON_TEXT_TAGS = frozenset({"script", "style", "link", "meta", "base", "noscript"})

#: `epub:type` and class token markers indicating footnote references.
_NOTE_REFERENCE_MARKERS = frozenset({"noteref", "footnoteref", "footnote-ref"})

#: Schemes indicating external URLs that are not normalized as internal paths.
_EXTERNAL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

_EPUB_NAMESPACE = "http://www.idpf.org/2007/ops"
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"

#: HTML tag or entity marker in Dublin Core metadata values (`dc:description`, `dc:rights`).
_DC_HTML_MARKER_RE = re.compile(
    r"<[A-Za-z/!?][^>]*>|&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);"
)

#: Block/break HTML elements separated by spaces when extracting plain text from DC metadata.
_DC_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "hr",
    }
)


def clean_dc_text(raw: str | None) -> str | None:
    """Strip embedded HTML tags/entities from Dublin Core text and normalize whitespace."""
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if _DC_HTML_MARKER_RE.search(stripped):
        parser = lxml_html.HTMLParser(no_network=True)
        try:
            node = lxml_html.fragment_fromstring(
                stripped, create_parent="div", parser=parser
            )
        except (etree.ParserError, etree.XMLSyntaxError, ValueError):
            text = stripped
        else:
            pieces: list[str] = []

            def _collect(element: etree._Element) -> None:
                tag = local_name(element)
                if tag in _NON_TEXT_TAGS:
                    return
                is_block = tag in _DC_BLOCK_TAGS
                if is_block:
                    pieces.append(" ")
                if element.text:
                    pieces.append(element.text)
                for child in element:
                    _collect(child)
                    if child.tail:
                        pieces.append(child.tail)
                if is_block:
                    pieces.append(" ")

            _collect(node)
            text = "".join(pieces)
    else:
        text = stripped
    normalized = " ".join(text.split())
    return normalized or None


def read_source_language_tag(source_epub_path: str | None) -> str | None:
    """Read the first non-empty ``dc:language`` tag from the source EPUB OPF."""
    if not source_epub_path:
        return None
    path = Path(source_epub_path)
    if not path.is_file():
        return None
    try:
        parser = etree.XMLParser(
            resolve_entities=False, load_dtd=False, no_network=True, recover=False
        )
        with zipfile.ZipFile(path) as archive:
            container = etree.fromstring(
                archive.read("META-INF/container.xml"), parser=parser
            )
            opf_path: str | None = None
            for element in container.iter():
                if local_name(element) == "rootfile" and element.get("full-path"):
                    opf_path = element.get("full-path")
                    break
            if not opf_path:
                return None
            try:
                opf_bytes = archive.read(opf_path)
            except KeyError:
                opf_bytes = archive.read(unquote(opf_path))
            opf_root = etree.fromstring(opf_bytes, parser=parser)
    except (OSError, KeyError, zipfile.BadZipFile, etree.XMLSyntaxError, ValueError):
        return None
    for element in opf_root.findall(f".//{{{_DC_NAMESPACE}}}language"):
        text = "".join(element.itertext()).strip()
        if text:
            return text
    return None


@dataclass(frozen=True, slots=True)
class InlineContent:
    """Block text together with extracted inline annotations and anchor IDs."""

    text: str
    annotations: tuple[InlineAnnotation, ...] = ()
    anchor_ids: tuple[str, ...] = ()


def local_name(element: etree._Element) -> str | None:
    """Local tag name without namespace, or `None` for comments and processing instructions."""
    if not isinstance(element.tag, str):
        return None
    return etree.QName(element).localname


def resolve_link_target(href: str, *, resource_path: str) -> str:
    """Normalize `href` into the canonical form 'book_resource_path#fragment'.

    Gutenberg almost always writes `href="file.xhtml#anchor"` even for intra-document links,
    whereas synthetic test books often use bare `href="#anchor"`. Both formats yield identical
    normalized results, eliminating format differences downstream. External links (`http:`, `mailto:`)
    are preserved as-is.
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
    """Element IDs of documents that represent book footnotes.

    Original footnote IDs are preserved verbatim: the exporter uses them to reconstruct internal
    links, while educational language notes live in their own namespace (PRD § 19.3).
    """
    return frozenset(
        element_id
        for element in root.iter()
        if local_name(element) is not None
        and (element_id := element.get("id"))
        and is_footnote_element(element)
    )


def is_footnote_element(element: etree._Element) -> bool:
    """Whether the element is marked as a book footnote."""
    if _epub_type_tokens(element) & {"footnote", "rearnote", "endnote", "note"}:
        return True
    return "footnote" in _class_tokens(element)


def is_footnote_reference(
    element: etree._Element, *, resource_path: str, footnote_ids: frozenset[str]
) -> bool:
    """Whether the hyperlink points to a book footnote.

    Identified either by explicit publisher markup on the link or surrounding `<sup>`,
    or by pointing to a known footnote in the current document.
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
    """Collector for block text and extracted inline annotations.

    Annotation offsets are calculated against normalized text, so text and markup
    are accumulated in a single pass. Empty annotations or those exceeding bounds are discarded:
    the domain layer rejects invalid offsets.
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
        """Append a text node, collapsing whitespace sequences."""
        if not raw:
            return
        text = _COLLAPSIBLE_WHITESPACE.sub(" ", raw)
        if text.startswith(" ") and self._ends_with_separator():
            text = text[1:]
        if text:
            self._pieces.append(text)
            self._length += len(text)

    def add_element(self, element: etree._Element, *, with_tail: bool = True) -> None:
        """Append an inline element with its children and default tail text."""
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
        """Append element's own text and child elements without tail text."""
        self.add_text(element.text)
        for child in element:
            self.add_element(child)

    def build(self) -> InlineContent:
        """Build block text and annotations, trimming leading and trailing whitespace."""
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
    """Build annotation clamped to text bounds, or return `None` if empty."""
    start = max(start, 0)
    end = min(end, limit)
    if end <= start:
        return None
    return InlineAnnotation(type=type_, start=start, end=end, target=target)


def _epub_type_tokens(element: etree._Element) -> frozenset[str]:
    """Extract `epub:type` values as a set of tokens."""
    value = element.get(f"{{{_EPUB_NAMESPACE}}}type") or element.get("epub:type")
    return frozenset(value.split()) if value else frozenset()


def _class_tokens(element: etree._Element) -> frozenset[str]:
    """Extract element class values as a set of tokens."""
    value = element.get("class")
    return frozenset(value.split()) if value else frozenset()


__all__ = [
    "InlineContent",
    "InlineContentBuilder",
    "clean_dc_text",
    "footnote_element_ids",
    "is_footnote_element",
    "is_footnote_reference",
    "local_name",
    "read_source_language_tag",
    "resolve_link_target",
]
