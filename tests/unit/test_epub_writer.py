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
    is_cover_section,
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


def test_is_cover_section_distinguishes_cover_from_other_front_matter() -> None:
    """A cover is recognized by filename, title, heading, or a textless image, not by front matter alone."""

    def _section(
        section_id: str,
        order: int,
        title: str,
        path: str,
        blocks: tuple[DocumentBlock, ...],
    ) -> DocumentSection:
        return DocumentSection(
            id=section_id,
            document_id="book-1",
            title=title,
            order=order,
            source_resource_path=path,
            blocks=blocks,
        )

    def _paragraph(section_id: str, block_id: str, text: str, order: int = 0) -> DocumentBlock:
        return _block(section_id, block_id, text, order=order)

    cover_xhtml = _section(
        "sec-cover-xhtml",
        0,
        "Front",
        "OEBPS/cover.xhtml",
        (_paragraph("sec-cover-xhtml", "cover-xhtml-p", "The book opens on this page."),),
    )
    cover_html = _section(
        "sec-cover-html",
        1,
        "Front",
        "OEBPS/Cover.html",
        (_paragraph("sec-cover-html", "cover-html-p", "The book opens on this page."),),
    )
    cover_page = _section(
        "sec-cover-page",
        2,
        "Front",
        "text/cover-page.xhtml",
        (_paragraph("sec-cover-page", "cover-page-p", "The book opens on this page."),),
    )
    cover_page_underscore = _section(
        "sec-cover-underscore",
        3,
        "Front",
        "text/cover_page.xhtml",
        (_paragraph("sec-cover-underscore", "cover-underscore-p", "The book opens on this page."),),
    )
    titled_cover = _section(
        "sec-titled-cover",
        4,
        "«Cover»",
        "frontispiece.xhtml",
        (_paragraph("sec-titled-cover", "titled-cover-p", "A line under the cover title."),),
    )
    heading_cover = _section(
        "sec-heading-cover",
        5,
        "Opening",
        "opening.xhtml",
        (
            _block("sec-heading-cover", "heading-cover-h", "Cover", BlockType.HEADING, 0),
            _paragraph("sec-heading-cover", "heading-cover-p", "A line under the heading.", 1),
        ),
    )
    image_cover = _section(
        "sec-image-cover",
        6,
        "Plate",
        "plate.xhtml",
        (_block("sec-image-cover", "image-cover-i", "A purple cloak.", BlockType.IMAGE, 0),),
    )
    title_page = _section(
        "sec-title",
        7,
        "Title Page",
        "title.xhtml",
        (_block("sec-title", "title-h", "The Great Novel", BlockType.HEADING, 0),),
    )
    promo_texts = (
        "Readers can find extra notes on Pottermore.",
        "The site lists maps and character sketches.",
        "Visit Pottermore after you finish the chapter.",
    )
    promo = _section(
        "sec-promo",
        8,
        "Pottermore",
        "pottermore.xhtml",
        tuple(
            _paragraph("sec-promo", f"promo-{index}", text, index)
            for index, text in enumerate(promo_texts)
        ),
    )
    chapter = _section(
        "sec-ch1",
        9,
        "Chapter 1",
        "chapter-1.xhtml",
        (
            _block("sec-ch1", "ch1-i", "A street at evening.", BlockType.IMAGE, 0),
            _block("sec-ch1", "ch1-h", "Chapter 1", BlockType.HEADING, 1),
            _paragraph("sec-ch1", "ch1-p", "The tale starts on this page.", 2),
        ),
    )
    notes = _section(
        "sec-notes",
        10,
        "Cover",
        "notes/cover.xhtml",
        (
            _block("sec-notes", "notes-h", "Cover", BlockType.HEADING, 0),
            _block("sec-notes", "notes-f", "A note about the cloak.", BlockType.FOOTNOTE, 1),
        ),
    )
    doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="A Magical Story"),
        source_language="en",
        target_language="ru",
        sections=(
            cover_xhtml,
            cover_html,
            cover_page,
            cover_page_underscore,
            titled_cover,
            heading_cover,
            image_cover,
            title_page,
            promo,
            chapter,
            notes,
        ),
    )

    def _expect_cover(section: DocumentSection, expected: bool) -> None:
        assert is_cover_section(section) is expected
        assert doc.is_cover_section(section) is expected

    assert is_front_matter(cover_xhtml, doc) is True
    for section in (cover_xhtml, cover_html, cover_page, cover_page_underscore):
        _expect_cover(section, True)
    _expect_cover(titled_cover, True)
    _expect_cover(heading_cover, True)
    _expect_cover(image_cover, True)

    assert is_front_matter(title_page, doc) is True
    _expect_cover(title_page, False)
    assert is_front_matter(promo, doc) is True
    _expect_cover(promo, False)
    assert is_front_matter(chapter, doc) is False
    _expect_cover(chapter, False)
    assert notes.is_notes_only is True
    _expect_cover(notes, False)


def test_short_headingless_promo_section_before_first_story_chapter_is_front_matter() -> None:
    """A headingless insert of at most five text blocks and 600 characters before the first story chapter is front matter."""
    promo_texts = (
        "Readers can find extra notes on the companion site.",
        "The site lists maps and character sketches.",
        "Visit the companion site after you finish.",
    )
    assert 2 < len(promo_texts) <= 5
    assert sum(len(text) for text in promo_texts) <= 600
    promo_section = DocumentSection(
        id="sec-promo",
        document_id="book-1",
        title="Companion site",
        order=0,
        source_resource_path="companion.xhtml",
        blocks=tuple(
            _block("sec-promo", f"promo-{index}", text, order=index)
            for index, text in enumerate(promo_texts)
        ),
    )
    boundary_texts = ("n" * 120,) * 5
    assert len(boundary_texts) == 5
    assert sum(len(text) for text in boundary_texts) == 600
    boundary_section = DocumentSection(
        id="sec-boundary",
        document_id="book-1",
        title="Opening note",
        order=0,
        source_resource_path="opening-note.xhtml",
        blocks=tuple(
            _block("sec-boundary", f"bound-{index}", text, order=index)
            for index, text in enumerate(boundary_texts)
        ),
    )
    chapter_section = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="Chapter 1",
        order=1,
        source_resource_path="ch1.xhtml",
        blocks=(
            _block("sec-ch1", "ch1-h", "Chapter 1", BlockType.HEADING, 0),
            _block("sec-ch1", "ch1-p", "The tale starts on this page.", order=1),
        ),
    )

    promo_doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Synthetic Reader"),
        source_language="en",
        target_language="ru",
        sections=(promo_section, chapter_section),
    )
    boundary_doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Synthetic Reader"),
        source_language="en",
        target_language="ru",
        sections=(boundary_section, chapter_section),
    )

    assert is_front_matter(promo_section, promo_doc) is True
    assert promo_doc.is_front_matter(promo_section) is True
    assert is_front_matter(chapter_section, promo_doc) is False
    assert is_front_matter(boundary_section, boundary_doc) is True
    assert boundary_doc.is_front_matter(boundary_section) is True


def test_custom_named_story_chapter_before_pattern_chapter_is_not_front_matter() -> None:
    """A chapter with a custom name and more than 2 paragraphs before Chapter 2 is not front matter."""
    ch1_section = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="The Boy Who Lived",
        order=0,
        source_resource_path="story1.xhtml",
        blocks=(
            _block("sec-ch1", "b1", "The Boy Who Lived", BlockType.HEADING, 0),
            _block("sec-ch1", "b2", "Mr. and Mrs. Dursley were proud to say...", BlockType.PARAGRAPH, 1),
            _block("sec-ch1", "b3", "They were the last people you'd expect...", BlockType.PARAGRAPH, 2),
            _block("sec-ch1", "b4", "Mr. Dursley was the director of a firm...", BlockType.PARAGRAPH, 3),
        ),
    )
    ch2_section = DocumentSection(
        id="sec-ch2",
        document_id="book-1",
        title="Chapter 2: The Vanishing Glass",
        order=1,
        source_resource_path="story2.xhtml",
        blocks=(
            _block("sec-ch2", "b5", "Chapter 2", BlockType.HEADING, 0),
            _block("sec-ch2", "b6", "Nearly ten years had passed.", BlockType.PARAGRAPH, 1),
        ),
    )
    doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="A Magical Story"),
        source_language="en",
        target_language="ru",
        sections=(ch1_section, ch2_section),
    )
    assert is_front_matter(ch1_section, doc) is False
    assert doc.is_front_matter(ch1_section) is False
    assert is_front_matter(ch2_section, doc) is False
    assert doc.is_front_matter(ch2_section) is False


def test_long_headingless_prologue_before_first_story_chapter_is_not_front_matter() -> None:
    """Headingless text before the first numbered chapter stays content when it exceeds five blocks or 600 characters.

    The section title must not match a story-chapter pattern: a titled prologue would be excluded earlier, and this test would not pin the length bound.
    """
    over_length_texts = ("x" * 121,) + ("x" * 120,) * 4
    assert len(over_length_texts) == 5
    assert sum(len(text) for text in over_length_texts) == 601
    over_length_section = DocumentSection(
        id="sec-long",
        document_id="book-1",
        title="Opening lines",
        order=0,
        source_resource_path="opening-lines.xhtml",
        blocks=tuple(
            _block("sec-long", f"long-{index}", text, order=index)
            for index, text in enumerate(over_length_texts)
        ),
    )
    over_count_texts = ("Short line.",) * 6
    assert len(over_count_texts) == 6
    assert sum(len(text) for text in over_count_texts) <= 600
    over_count_section = DocumentSection(
        id="sec-many",
        document_id="book-1",
        title="Opening lines",
        order=0,
        source_resource_path="opening-lines.xhtml",
        blocks=tuple(
            _block("sec-many", f"many-{index}", text, order=index)
            for index, text in enumerate(over_count_texts)
        ),
    )
    chapter_section = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="Chapter 1",
        order=1,
        source_resource_path="ch1.xhtml",
        blocks=(
            _block("sec-ch1", "ch1-h", "Chapter 1", BlockType.HEADING, 0),
            _block("sec-ch1", "ch1-p", "The tale starts on this page.", order=1),
        ),
    )

    def _document(intro: DocumentSection) -> CanonicalDocument:
        return CanonicalDocument(
            id="book-1",
            metadata=BookMetadata(title="Synthetic Reader"),
            source_language="en",
            target_language="ru",
            sections=(intro, chapter_section),
        )

    over_length_doc = _document(over_length_section)
    over_count_doc = _document(over_count_section)
    assert is_front_matter(over_length_section, over_length_doc) is False
    assert over_length_doc.is_front_matter(over_length_section) is False
    assert is_front_matter(over_count_section, over_count_doc) is False
    assert over_count_doc.is_front_matter(over_count_section) is False
    assert is_front_matter(chapter_section, over_length_doc) is False


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
