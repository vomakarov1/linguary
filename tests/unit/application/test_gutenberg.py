"""Project Gutenberg boilerplate text is not sent for processing."""

from linguary.application.gutenberg import preserve_gutenberg_boilerplate
from linguary.domain import (
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    SupportStatus,
)


def _document(*sections: tuple[str, ...]) -> CanonicalDocument:
    return CanonicalDocument(
        id="book",
        metadata=BookMetadata(title="Book"),
        source_language="en",
        target_language="ru",
        sections=tuple(
            DocumentSection(
                id=f"section-{section_order}",
                document_id="book",
                title=f"Section {section_order}",
                order=section_order,
                source_resource_path=f"section-{section_order}.xhtml",
                blocks=tuple(
                    DocumentBlock.create(
                        document_id="book",
                        section_id=f"section-{section_order}",
                        type=BlockType.PARAGRAPH,
                        order=block_order,
                        text=text,
                        source_reference=SourceReference(
                            resource_path=f"section-{section_order}.xhtml",
                            structural_path=f"/p[{block_order + 1}]",
                        ),
                    )
                    for block_order, text in enumerate(texts)
                ),
            )
            for section_order, texts in enumerate(sections)
        ),
    )


def test_blocks_before_start_marker_preserved() -> None:
    document = _document(
        ("Front matter", "*** start of the Project Gutenberg ebook Book ***", "Story"),
        ("More story",),
    )

    result, _ = preserve_gutenberg_boilerplate(document)

    assert [block.support_status for block in result.blocks()] == [
        SupportStatus.PRESERVE_OPAQUE,
        SupportStatus.PRESERVE_OPAQUE,
        SupportStatus.SUPPORTED,
        SupportStatus.SUPPORTED,
    ]


def test_blocks_after_end_marker_preserved() -> None:
    document = _document(
        ("Story",),
        ("*** END OF THE PROJECT GUTENBERG EBOOK BOOK ***", "License"),
    )

    result, _ = preserve_gutenberg_boilerplate(document)

    assert [block.support_status for block in result.blocks()] == [
        SupportStatus.SUPPORTED,
        SupportStatus.PRESERVE_OPAQUE,
        SupportStatus.PRESERVE_OPAQUE,
    ]


def test_book_without_markers_unchanged() -> None:
    document = _document(("A Project Gutenberg edition", "The story"))

    result, warnings = preserve_gutenberg_boilerplate(document)

    assert result is document
    assert warnings == ()


def test_warning_counts_preserved_blocks() -> None:
    document = _document(
        ("Front matter", "*** START OF THE PROJECT GUTENBERG EBOOK BOOK ***", "Story"),
        ("*** END OF THE PROJECT GUTENBERG EBOOK BOOK ***", "License"),
    )

    _, warnings = preserve_gutenberg_boilerplate(document)

    assert len(warnings) == 1
    assert "4" in warnings[0].message
