"""Domain object factories for unit tests.

Assemble minimally valid entities so that each test specifies only what it checks.
The module intentionally omits the 'test_' prefix: otherwise pytest would collect it as a test module.
"""

from __future__ import annotations

from collections.abc import Iterable

from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    TreatmentPlan,
)

DOCUMENT_ID = "book-1"
SECTION_ID = "section-1"
RESOURCE_PATH = "chapter03.xhtml"


def make_block(
    *,
    order: int = 0,
    text: str = "It was a bright cold day in April.",
    structural_path: str | None = None,
    section_id: str = SECTION_ID,
    document_id: str = DOCUMENT_ID,
    resource_path: str = RESOURCE_PATH,
    type: BlockType = BlockType.PARAGRAPH,
) -> DocumentBlock:
    """A chapter paragraph with a stable identifier."""
    return DocumentBlock.create(
        document_id=document_id,
        section_id=section_id,
        type=type,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path=resource_path,
            structural_path=structural_path or f"/body/section[1]/p[{order + 1}]",
        ),
    )


def make_blocks(count: int, *, section_id: str = SECTION_ID) -> tuple[DocumentBlock, ...]:
    """Several sequential paragraphs of a single chapter."""
    return tuple(make_block(order=index, section_id=section_id) for index in range(count))


def make_section(
    blocks: Iterable[DocumentBlock] = (),
    *,
    section_id: str = SECTION_ID,
    order: int = 0,
) -> DocumentSection:
    """A book chapter containing the given blocks."""
    return DocumentSection(
        id=section_id,
        document_id=DOCUMENT_ID,
        title="Chapter 3",
        order=order,
        source_resource_path=RESOURCE_PATH,
        blocks=tuple(blocks),
    )


def make_document(sections: Iterable[DocumentSection] = ()) -> CanonicalDocument:
    """A book containing the given chapters."""
    return CanonicalDocument(
        id=DOCUMENT_ID,
        metadata=BookMetadata(title="Nineteen Eighty-Four", authors=("George Orwell",)),
        source_language="en",
        target_language="ru",
        sections=tuple(sections),
    )


def make_plan(
    blocks: Iterable[DocumentBlock],
    *,
    treatment: BlockTreatment = BlockTreatment.TRANSLATE,
    section_id: str = SECTION_ID,
) -> TreatmentPlan:
    """A plan that assigns the same treatment to all blocks."""
    return TreatmentPlan.of_treatments(
        section_id, ((block.id, treatment) for block in blocks)
    )
