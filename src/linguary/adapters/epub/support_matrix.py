"""EPUB construct support matrix (PRD § 6).

The importer must explicitly specify how it handles each book construct:

- `SUPPORTED` — construct is parsed and fully supported;
- `PRESERVE_OPAQUE` — construct is transferred to the resulting EPUB without internal processing;
- `DROP_WITH_WARNING` — construct is dropped, and a user-facing warning is emitted;
- `FATAL` — book cannot be imported; this raises `EpubImportFatalError`
  and halts before individual construct classification.

This module knows domain types only: no lxml, no EbookLib, and no filesystem dependencies exist here,
so matrix decisions remain consistent across the book reader and resource persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from linguary.domain import (
    BlockType,
    BookImportWarning,
    ImportWarningKind,
    SupportStatus,
)

#: Block types transferred to the target EPUB without internal processing: complex
#: tables, non-standard containers, and decorative structures (PRD § 6).
_PRESERVE_OPAQUE_BLOCK_TYPES = frozenset(
    {BlockType.TABLE, BlockType.OTHER, BlockType.PAGE_BREAK}
)

#: Resource media types that are not needed in the target publication: scripts do not affect
#: reading, and there is no need to transfer executable code into a bilingual learning EPUB.
_DROPPED_MEDIA_TYPES = frozenset(
    {
        "application/javascript",
        "application/ecmascript",
        "application/x-javascript",
        "text/javascript",
        "text/ecmascript",
    }
)

#: Number of structural path examples displayed in the dropped block warning text.
_EXAMPLES_IN_WARNING = 3


class DropReason(StrEnum):
    """Reason why a book construct was not included in the canonical model.

    The list is strictly bounded by PRD § 6: skipping is permitted for empty junk, duplicate
    page breaks, and unsupported decorative elements that do not affect reading.
    """

    EMPTY_DECORATION = "empty_decoration"
    DUPLICATE_PAGE_BREAK = "duplicate_page_break"

    @property
    def description(self) -> str:
        """Human-readable description of the reason for import warnings."""
        return _DROP_REASON_DESCRIPTIONS[self]


_DROP_REASON_DESCRIPTIONS = {
    DropReason.EMPTY_DECORATION: (
        "пустые декоративные элементы без текста и внутренних ссылок"
    ),
    DropReason.DUPLICATE_PAGE_BREAK: "повторные разрывы страницы печатного оригинала",
}


def block_support_status(block_type: BlockType) -> SupportStatus:
    """Block support status based on its block type.

    Paragraphs, headings, quotes, list items, images, footnotes, and poems are fully
    supported; tables, page breaks, and unrecognized constructs are preserved as opaque blocks.
    """
    return (
        SupportStatus.PRESERVE_OPAQUE
        if block_type in _PRESERVE_OPAQUE_BLOCK_TYPES
        else SupportStatus.SUPPORTED
    )


def resource_support_status(media_type: str) -> SupportStatus:
    """Book resource support status based on its media type.

    Images are fully supported: exporter transfers them alongside `Image` blocks.
    Stylesheets, fonts, audio/video, and unrecognized files are transferred as-is: there is no need
    for the importer to parse them, and the book's visual styling must not be lost.
    """
    normalized = media_type.strip().lower().partition(";")[0].strip()
    if normalized in _DROPPED_MEDIA_TYPES:
        return SupportStatus.DROP_WITH_WARNING
    if normalized.startswith("image/"):
        return SupportStatus.SUPPORTED
    return SupportStatus.PRESERVE_OPAQUE


def dropped_resource_warning(*, source_path: str, media_type: str) -> BookImportWarning:
    """Warning for a resource not transferred to the book (`DROP_WITH_WARNING`)."""
    return BookImportWarning(
        kind=ImportWarningKind.UNSUPPORTED_CONSTRUCT,
        message=(
            f"ресурс {source_path} типа {media_type} не переносится в книгу: "
            f"он не влияет на чтение и не поддержан приложением"
        ),
    )


@dataclass(slots=True)
class DroppedBlockLog:
    """Dropped blocks of a single document and their associated warnings.

    `DROP_WITH_WARNING` must leave a user-visible warning (PRD § 6),
    but a separate line for every empty `<span>` turns the warning list
    into noise: Gutenberg editions contain hundreds of such elements. Dropped blocks are therefore
    aggregated by reason, and the user receives one warning per reason — with the count
    of dropped constructs and examples of their positions in the source document.
    """

    resource_path: str
    section_id: str
    _paths: dict[DropReason, list[str]] = field(default_factory=dict)

    def record(self, reason: DropReason, structural_path: str) -> None:
        """Records a dropped construct and its location in the document."""
        self._paths.setdefault(reason, []).append(structural_path)

    def warnings(self) -> tuple[BookImportWarning, ...]:
        """Import warnings for accumulated dropped blocks."""
        return tuple(
            BookImportWarning(
                kind=ImportWarningKind.DROPPED_BLOCK,
                message=(
                    f"в документе {self.resource_path} пропущены "
                    f"{reason.description} — {len(paths)}. {_examples(paths)}"
                ),
                section_id=self.section_id,
            )
            for reason, paths in self._paths.items()
        )


def _examples(paths: Sequence[str]) -> str:
    """Examples of dropped construct locations in the source document."""
    shown = ", ".join(paths[:_EXAMPLES_IN_WARNING])
    if len(paths) > _EXAMPLES_IN_WARNING:
        return f"Например: {shown} и другие"
    return f"Это: {shown}"


__all__ = [
    "DropReason",
    "DroppedBlockLog",
    "block_support_status",
    "dropped_resource_warning",
    "resource_support_status",
]
