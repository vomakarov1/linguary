"""Canonical book representation: document, sections, blocks, and resources.

The model is independent of EPUB (PRD § 7): after import, all remaining application
logic operates solely on these entities. `DocumentBlock` is the unit of display,
processing result, and regeneration; it is never split or merged (ADR § 3).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from linguary.domain.errors import (
    DocumentStructureError,
    UnknownBlockError,
    UnknownBlockTypeError,
)
from linguary.domain.identifiers import build_block_id


class BlockType(StrEnum):
    """Supported structural block types (PRD § 7)."""

    PARAGRAPH = "Paragraph"
    HEADING = "Heading"
    QUOTE = "Quote"
    LIST_ITEM = "ListItem"
    IMAGE = "Image"
    FOOTNOTE = "Footnote"
    POEM = "Poem"
    TABLE = "Table"
    PAGE_BREAK = "PageBreak"
    OTHER = "Other"

    @classmethod
    def parse(cls, value: str) -> BlockType:
        """Parses a block type from an external representation.

        An unknown construct is never turned into a block silently: the importer must either
        map it to a supported type or to `Other`.
        """
        try:
            return cls(value)
        except ValueError as error:
            raise UnknownBlockTypeError(f"неизвестный тип блока: {value!r}") from error


class InlineAnnotationType(StrEnum):
    """Basic inline annotations within block text (PRD § 7)."""

    ITALIC = "Italic"
    BOLD = "Bold"
    EMPHASIS = "Emphasis"
    LINK = "Link"
    FOOTNOTE_REFERENCE = "FootnoteReference"

    @property
    def requires_target(self) -> bool:
        """Annotation meaningless without a destination target."""
        return self in (InlineAnnotationType.LINK, InlineAnnotationType.FOOTNOTE_REFERENCE)


class SupportStatus(StrEnum):
    """Support status of the original EPUB construct (PRD § 6)."""

    SUPPORTED = "SUPPORTED"
    PRESERVE_OPAQUE = "PRESERVE_OPAQUE"
    DROP_WITH_WARNING = "DROP_WITH_WARNING"
    FATAL = "FATAL"


@dataclass(frozen=True, slots=True)
class InlineAnnotation:
    """Markup of a block text span with offsets within the text itself.

    `DocumentBlock.text` is stored without XHTML, so formatting is described separately
    by a half-open interval `[start, end)` and restored by the exporter.
    """

    type: InlineAnnotationType
    start: int
    end: int
    target: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise DocumentStructureError("начало встроенной разметки не может быть отрицательным")
        if self.end <= self.start:
            raise DocumentStructureError("встроенная разметка не может быть пустой")
        if self.type.requires_target and not self.target:
            raise DocumentStructureError(f"разметка {self.type} требует адреса назначения")


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Link between a block and source XHTML: resource path and structural path within it."""

    resource_path: str
    structural_path: str
    element_id: str | None = None

    def __post_init__(self) -> None:
        if not self.resource_path.strip():
            raise DocumentStructureError("путь к исходному ресурсу блока пуст")
        if not self.structural_path.strip():
            raise DocumentStructureError("структурный путь блока пуст")


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """Stable unit of source text and processing results.

    `parent_id` describes nesting for lists, footnotes, and compound blocks; `attributes`
    stores technical import properties that do not affect text.
    """

    id: str
    section_id: str
    type: BlockType
    order: int
    text: str
    source_reference: SourceReference
    support_status: SupportStatus = SupportStatus.SUPPORTED
    inline_annotations: tuple[InlineAnnotation, ...] = ()
    parent_id: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DocumentStructureError("идентификатор блока пуст")
        if not self.section_id.strip():
            raise DocumentStructureError("блок не привязан к разделу")
        if not isinstance(self.type, BlockType):
            raise UnknownBlockTypeError(f"неизвестный тип блока: {self.type!r}")
        if self.order < 0:
            raise DocumentStructureError("порядковый номер блока не может быть отрицательным")
        if self.support_status is SupportStatus.FATAL:
            raise DocumentStructureError(
                "блок со статусом FATAL не существует: такой EPUB импортировать нельзя"
            )
        if self.parent_id == self.id:
            raise DocumentStructureError("блок не может быть родителем самому себе")
        for annotation in self.inline_annotations:
            if annotation.end > len(self.text):
                raise DocumentStructureError(
                    "встроенная разметка выходит за пределы текста блока"
                )
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        section_id: str,
        type: BlockType,
        order: int,
        text: str,
        source_reference: SourceReference,
        support_status: SupportStatus = SupportStatus.SUPPORTED,
        inline_annotations: tuple[InlineAnnotation, ...] = (),
        parent_id: str | None = None,
        attributes: Mapping[str, str] | None = None,
    ) -> DocumentBlock:
        """Creates a block with a stable identifier (PRD § 8)."""
        return cls(
            id=build_block_id(
                document_id=document_id,
                source_resource_path=source_reference.resource_path,
                structural_path=source_reference.structural_path,
            ),
            section_id=section_id,
            type=type,
            order=order,
            text=text,
            source_reference=source_reference,
            support_status=support_status,
            inline_annotations=inline_annotations,
            parent_id=parent_id,
            attributes=attributes or {},
        )


#: Blocks that do not turn a notes page into a chapter: its heading and page breaks.
_NOTES_PAGE_EXTRAS = frozenset({BlockType.HEADING, BlockType.PAGE_BREAK})

#: Blocks that carry content even without text.
_CONTENT_WITHOUT_TEXT = frozenset({BlockType.IMAGE, BlockType.TABLE})


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A book chapter: a single XHTML document in spine reading order (PRD § 6)."""

    id: str
    document_id: str
    title: str
    order: int
    source_resource_path: str
    blocks: tuple[DocumentBlock, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DocumentStructureError("идентификатор раздела пуст")
        if self.order < 0:
            raise DocumentStructureError("порядковый номер раздела не может быть отрицательным")
        known_ids: set[str] = set()
        orders: set[int] = set()
        for block in self.blocks:
            if block.section_id != self.id:
                raise DocumentStructureError(
                    f"блок {block.id} принадлежит другому разделу: {block.section_id}"
                )
            if block.id in known_ids:
                raise DocumentStructureError(f"идентификатор блока повторяется: {block.id}")
            if block.order in orders:
                raise DocumentStructureError(
                    f"порядковый номер блока повторяется в разделе {self.id}: {block.order}"
                )
            known_ids.add(block.id)
            orders.add(block.order)
        for block in self.blocks:
            if block.parent_id is not None and block.parent_id not in known_ids:
                raise DocumentStructureError(
                    f"родительский блок {block.parent_id} отсутствует в разделе {self.id}"
                )

    @property
    def ordered_blocks(self) -> tuple[DocumentBlock, ...]:
        """Section blocks in the order of the source document."""
        return tuple(sorted(self.blocks, key=lambda block: block.order))

    @property
    def is_notes_only(self) -> bool:
        """Whether the section consists solely of footnotes and endnotes.

        Such a document is a notes page (`notes.xhtml`) rather than a chapter: its text
        is translated, but it receives no prior chapter summary, `Before You Read`, or
        chapter markup, and is not treated as the prior chapter for subsequent sections
        (PRD § 19.1, § 19.2). The page heading, page breaks, and empty technical blocks
        do not affect this decision: what matters is that there is nothing else to read on the page.
        """
        has_notes = False
        for block in self.blocks:
            if block.parent_id is not None:
                continue
            if block.type is BlockType.FOOTNOTE:
                has_notes = True
            elif block.type in _NOTES_PAGE_EXTRAS:
                continue
            elif block.text.strip() or block.type in _CONTENT_WITHOUT_TEXT:
                return False
        return has_notes

    def block(self, block_id: str) -> DocumentBlock:
        """Section block by identifier."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise UnknownBlockError(f"блок {block_id} отсутствует в разделе {self.id}")


@dataclass(frozen=True, slots=True)
class DocumentResource:
    """Book resource: image, font, stylesheet, or other file required by the exporter."""

    id: str
    source_path: str
    media_type: str
    support_status: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        if not self.source_path.strip():
            raise DocumentStructureError("путь к ресурсу пуст")
        if not self.media_type.strip():
            raise DocumentStructureError(f"тип содержимого ресурса {self.source_path} не задан")


@dataclass(frozen=True, slots=True)
class BookMetadata:
    """Book metadata read during import."""

    title: str
    authors: tuple[str, ...] = ()
    identifier: str | None = None
    publisher: str | None = None

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise DocumentStructureError("название книги пусто")


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    """Internal book model independent of EPUB.

    The document identifier matches the book identifier: one book, one document.
    """

    id: str
    metadata: BookMetadata
    source_language: str
    target_language: str
    sections: tuple[DocumentSection, ...] = ()
    resources: tuple[DocumentResource, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DocumentStructureError("идентификатор документа пуст")
        if not self.source_language.strip() or not self.target_language.strip():
            raise DocumentStructureError("языки книги не заданы")
        if self.source_language == self.target_language:
            raise DocumentStructureError("исходный и целевой языки книги совпадают")
        section_ids: set[str] = set()
        orders: set[int] = set()
        for section in self.sections:
            if section.document_id != self.id:
                raise DocumentStructureError(
                    f"раздел {section.id} принадлежит другому документу: {section.document_id}"
                )
            if section.id in section_ids:
                raise DocumentStructureError(f"идентификатор раздела повторяется: {section.id}")
            if section.order in orders:
                raise DocumentStructureError(
                    f"порядковый номер раздела повторяется: {section.order}"
                )
            section_ids.add(section.id)
            orders.add(section.order)

    @property
    def ordered_sections(self) -> tuple[DocumentSection, ...]:
        """Sections in reading order."""
        return tuple(sorted(self.sections, key=lambda section: section.order))

    def section(self, section_id: str) -> DocumentSection:
        """Section by identifier."""
        for section in self.sections:
            if section.id == section_id:
                return section
        raise UnknownBlockError(f"раздел {section_id} отсутствует в документе {self.id}")

    def blocks(self) -> Iterator[DocumentBlock]:
        """All book blocks in reading order."""
        for section in self.ordered_sections:
            yield from section.ordered_blocks

    def block(self, block_id: str) -> DocumentBlock:
        """Book block by identifier."""
        for block in self.blocks():
            if block.id == block_id:
                return block
        raise UnknownBlockError(f"блок {block_id} отсутствует в документе {self.id}")


class ImportWarningKind(StrEnum):
    """Import warning kinds presented to the user in the interface."""

    DROPPED_BLOCK = "dropped_block"
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"
    TOC_MISMATCH = "toc_mismatch"
    MISSING_RESOURCE = "missing_resource"


@dataclass(frozen=True, slots=True)
class BookImportWarning:
    """Non-blocking import warning associated with a book, section, or block."""

    kind: ImportWarningKind
    message: str
    section_id: str | None = None
    block_id: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise DocumentStructureError("текст предупреждения импорта пуст")
