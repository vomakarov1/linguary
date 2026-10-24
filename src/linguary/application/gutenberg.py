"""Preserves Project Gutenberg boilerplate outside book processing."""

from __future__ import annotations

import re
from dataclasses import replace

from linguary.domain import (
    BookImportWarning,
    CanonicalDocument,
    ImportWarningKind,
    SupportStatus,
)

_MARKER = re.compile(
    r"^\s*\*{3}\s*(START|END)\s+OF\b.*?\bPROJECT\s+GUTENBERG\b",
    re.IGNORECASE,
)


def preserve_gutenberg_boilerplate(
    document: CanonicalDocument,
) -> tuple[CanonicalDocument, tuple[BookImportWarning, ...]]:
    """Marks boilerplate blocks, preserving them for export of the original EPUB."""
    blocks = tuple(
        (section.id, block)
        for section in document.ordered_sections
        for block in section.ordered_blocks
    )
    start = end = None
    for index, (_, block) in enumerate(blocks):
        marker = _MARKER.match(block.text)
        if marker is None:
            continue
        if marker.group(1).upper() == "START" and start is None:
            start = index
        elif marker.group(1).upper() == "END" and end is None:
            end = index

    if start is None and end is None:
        return document, ()

    preserved_blocks = {
        (section_id, block.id)
        for index, (section_id, block) in enumerate(blocks)
        if (start is not None and index <= start)
        or (end is not None and index >= end)
    }
    sections = tuple(
        replace(
            section,
            blocks=tuple(
                replace(block, support_status=SupportStatus.PRESERVE_OPAQUE)
                if (section.id, block.id) in preserved_blocks
                and block.support_status is not SupportStatus.PRESERVE_OPAQUE
                else block
                for block in section.blocks
            ),
        )
        for section in document.sections
    )
    warning = BookImportWarning(
        kind=ImportWarningKind.UNSUPPORTED_CONSTRUCT,
        message=(
            f"Project Gutenberg: {len(preserved_blocks)} служебных блоков "
            "сохранено без обработки"
        ),
    )
    return replace(document, sections=sections), (warning,)
