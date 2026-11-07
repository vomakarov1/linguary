"""Unit tests for EPUB writer front-matter handling and recap behavior (P1.T1, AC-003, AC-004)."""

from linguary.adapters.epub.writer import _previously_heading, _render_chapters, EpubWriteRequest
from linguary.adapters.epub.writer import _OpaqueSource, _ExportWarnings
from linguary.domain.document import (
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    SourceReference,
    is_front_matter,
)


def _block(section_id: str, block_id: str, text: str, block_type: BlockType = BlockType.PARAGRAPH, order: int = 0) -> DocumentBlock:
    return DocumentBlock(
        id=block_id,
        section_id=section_id,
        type=block_type,
        order=order,
        text=text,
        source_reference=SourceReference(resource_path="chapter.xhtml", structural_path="/p[1]"),
    )


def test_front_matter_classification_in_domain() -> None:
    """Sections before first story chapter or matching front-matter patterns are classified as front matter."""
    title_section = DocumentSection(
        id="sec-title",
        document_id="book-1",
        title="Title Page",
        order=0,
        source_resource_path="title.xhtml",
        blocks=(_block("sec-title", "b1", "Pride and Prejudice", BlockType.HEADING, 0),),
    )
    toc_section = DocumentSection(
        id="sec-toc",
        document_id="book-1",
        title="Table of Contents",
        order=1,
        source_resource_path="toc.xhtml",
        blocks=(_block("sec-toc", "b2", "Contents", BlockType.HEADING, 0),),
    )
    dedication_section = DocumentSection(
        id="sec-dedication",
        document_id="book-1",
        title="Dedication",
        order=2,
        source_resource_path="dedication.xhtml",
        blocks=(_block("sec-dedication", "b3", "For my friends.", BlockType.PARAGRAPH, 0),),
    )
    ch1_section = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="Chapter 1",
        order=3,
        source_resource_path="ch1.xhtml",
        blocks=(
            _block("sec-ch1", "b4", "Chapter 1", BlockType.HEADING, 0),
            _block("sec-ch1", "b5", "It is a truth universally acknowledged...", BlockType.PARAGRAPH, 1),
        ),
    )
    ch2_section = DocumentSection(
        id="sec-ch2",
        document_id="book-1",
        title="Chapter 2",
        order=4,
        source_resource_path="ch2.xhtml",
        blocks=(
            _block("sec-ch2", "b6", "Chapter 2", BlockType.HEADING, 0),
            _block("sec-ch2", "b7", "Mr. Bennet was among the earliest of those...", BlockType.PARAGRAPH, 1),
        ),
    )

    doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Pride and Prejudice"),
        source_language="en",
        target_language="ru",
        sections=(title_section, toc_section, dedication_section, ch1_section, ch2_section),
    )

    assert is_front_matter(title_section, doc) is True
    assert doc.is_front_matter(title_section) is True

    assert is_front_matter(toc_section, doc) is True
    assert doc.is_front_matter(toc_section) is True

    assert is_front_matter(dedication_section, doc) is True
    assert doc.is_front_matter(dedication_section) is True

    assert is_front_matter(ch1_section, doc) is False
    assert doc.is_front_matter(ch1_section) is False

    assert is_front_matter(ch2_section, doc) is False
    assert doc.is_front_matter(ch2_section) is False


def test_first_story_chapter_does_not_recap_front_matter() -> None:
    """The first story chapter following front matter does not receive a 'Previously' recap from front matter."""
    title_section = DocumentSection(
        id="sec-title",
        document_id="book-1",
        title="Title Page",
        order=0,
        source_resource_path="title.xhtml",
        blocks=(_block("sec-title", "b1", "The Great Novel", BlockType.HEADING, 0),),
    )
    ch1_section = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="Chapter 1",
        order=1,
        source_resource_path="ch1.xhtml",
        blocks=(
            _block("sec-ch1", "b2", "Chapter 1", BlockType.HEADING, 0),
            _block("sec-ch1", "b3", "Story begins here.", BlockType.PARAGRAPH, 1),
        ),
    )
    ch2_section = DocumentSection(
        id="sec-ch2",
        document_id="book-1",
        title="Chapter 2",
        order=2,
        source_resource_path="ch2.xhtml",
        blocks=(
            _block("sec-ch2", "b4", "Chapter 2", BlockType.HEADING, 0),
            _block("sec-ch2", "b5", "Story continues here.", BlockType.PARAGRAPH, 1),
        ),
    )

    doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="The Great Novel"),
        source_language="en",
        target_language="ru",
        sections=(title_section, ch1_section, ch2_section),
    )

    request = EpubWriteRequest(document=doc, target_path="output.epub")
    opaque = _OpaqueSource(source_path=None)
    warnings = _ExportWarnings()

    from lxml import etree

    chapters = _render_chapters(request, opaque=opaque, warnings=warnings)
    assert len(chapters) == 3

    # Title section has no previously
    title_content = etree.tostring(chapters[0].root)
    assert b"linguary-previously" not in title_content

    # Chapter 1 is story chapter following front matter, must not have previously
    ch1_content = etree.tostring(chapters[1].root)
    assert b"linguary-previously" not in ch1_content

    # If title had a summary, ch1 would still have previous=None
    assert chapters[0].section_id == "sec-title"
    assert chapters[1].section_id == "sec-ch1"
    assert chapters[2].section_id == "sec-ch2"
