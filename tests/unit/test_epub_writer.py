"""Unit tests for EPUB writer front-matter handling and recap behavior (P1.T1, AC-003, AC-004)."""

from linguary.adapters.epub.writer import (
    EpubWriteRequest,
    _ExportWarnings,
    _OpaqueSource,
    _exported_sections,
    _previously_heading,
    _render_chapters,
    _split_leading_front_matter,
)
from linguary.application.layout.book import BookLayout
from linguary.application.layout.legend import LayoutFeatures
from linguary.domain.document import (
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    ImportWarningKind,
    SourceReference,
    has_meaningful_part_title,
    is_cover_section,
    is_front_matter,
    is_part_divider,
)
from linguary.web.labels import IMPORT_WARNING_LABELS


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


def test_part_divider_classification_and_export_preservation() -> None:
    """Standalone part dividers are skipped as non-story sections, kept in learner export, and stay after the legend."""
    cover = DocumentSection(
        id="sec-cover",
        document_id="book-1",
        title="Cover",
        order=0,
        source_resource_path="cover.xhtml",
        blocks=(_block("sec-cover", "b-cov", "", BlockType.IMAGE, 0),),
    )
    dedication = DocumentSection(
        id="sec-ded",
        document_id="book-1",
        title="Dedication",
        order=1,
        source_resource_path="dedication.xhtml",
        blocks=(_block("sec-ded", "b-ded", "For my aunt.", BlockType.PARAGRAPH, 0),),
    )
    part1_heading_divider = DocumentSection(
        id="sec-part1",
        document_id="book-1",
        title="Part One: Afterlights",
        order=2,
        source_resource_path="part1.xhtml",
        blocks=(
            _block("sec-part1", "b-p1h1", "PART ONE", BlockType.HEADING, 0),
            _block("sec-part1", "b-p1h2", "Afterlights", BlockType.HEADING, 1),
        ),
    )
    ch1 = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="Chapter 1",
        order=3,
        source_resource_path="ch1.xhtml",
        blocks=(
            _block("sec-ch1", "b-c1h", "Chapter 1", BlockType.HEADING, 0),
            _block("sec-ch1", "b-c1p", "Story begins here.", BlockType.PARAGRAPH, 1),
        ),
    )
    part2_paragraph_divider = DocumentSection(
        id="sec-part2",
        document_id="book-1",
        title="PART TWO Mary, Queen of Snots",
        order=4,
        source_resource_path="ch1-7.xhtml",
        blocks=(
            _block("sec-part2", "b-p2a", "PART TWO", BlockType.PARAGRAPH, 0),
            _block("sec-part2", "b-p2b", "Mary, Queen of Snots", BlockType.PARAGRAPH, 1),
        ),
    )
    book_story_chapter = DocumentSection(
        id="sec-book1",
        document_id="book-1",
        title="Book 1",
        order=5,
        source_resource_path="book1.xhtml",
        blocks=(
            _block("sec-book1", "b-bk1h", "Book 1", BlockType.HEADING, 0),
            _block("sec-book1", "b-bk1p", "A real chapter titled Book 1.", BlockType.PARAGRAPH, 1),
        ),
    )
    combined_part_and_chapter = DocumentSection(
        id="sec-combined",
        document_id="book-1",
        title="PART ONE Afterlights",
        order=6,
        source_resource_path="ch1-2.xhtml",
        blocks=(
            _block("sec-combined", "b-cmb1", "PART ONE", BlockType.PARAGRAPH, 0),
            _block("sec-combined", "b-cmb2", "CHAPTER 1", BlockType.PARAGRAPH, 1),
            _block("sec-combined", "b-cmb3", "On a hairpin turn, a car crashed.", BlockType.PARAGRAPH, 2),
        ),
    )

    doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Everlost"),
        source_language="en",
        target_language="ru",
        sections=(
            cover,
            dedication,
            part1_heading_divider,
            ch1,
            part2_paragraph_divider,
            book_story_chapter,
            combined_part_and_chapter,
        ),
    )

    assert is_part_divider(part1_heading_divider) is True
    assert doc.is_front_matter(part1_heading_divider) is True
    assert is_part_divider(part2_paragraph_divider) is True
    assert doc.is_front_matter(part2_paragraph_divider) is True

    assert is_part_divider(cover) is False
    assert is_part_divider(ch1) is False
    assert is_part_divider(book_story_chapter) is False
    assert doc.is_front_matter(book_story_chapter) is False
    assert is_part_divider(combined_part_and_chapter) is False
    assert doc.is_front_matter(combined_part_and_chapter) is False

    request = EpubWriteRequest(
        document=doc,
        target_path="output.epub",
        layout=BookLayout(
            chapters={},
            legend=(),
            ipa_key=(),
            who_is_who=(),
            features=LayoutFeatures(
                has_translation=False,
                has_notes=False,
                labels=frozenset(),
                has_ipa=False,
                has_poems=False,
                has_speakers=False,
                sections=frozenset(),
            ),
        ),
    )
    exported_ids = [section.id for section in _exported_sections(request)]
    assert exported_ids == [
        "sec-cover",
        "sec-part1",
        "sec-ch1",
        "sec-part2",
        "sec-book1",
        "sec-combined",
    ]

    chapters = _render_chapters(
        request, opaque=_OpaqueSource(source_path=None), warnings=_ExportWarnings()
    )
    leading, rest = _split_leading_front_matter(chapters, request)
    assert [c.section_id for c in leading] == ["sec-cover"]
    assert [c.section_id for c in rest] == [
        "sec-part1",
        "sec-ch1",
        "sec-part2",
        "sec-book1",
        "sec-combined",
    ]


def test_has_meaningful_part_title_distinguishes_titled_and_bare_dividers() -> None:
    """Part dividers with real subtitles are distinguished from bare numbered dividers while staying front matter."""
    titled_heading = DocumentSection(
        id="sec-part1-titled",
        document_id="book-1",
        title="PART ONE / Afterlights",
        order=0,
        source_resource_path="part1.xhtml",
        blocks=(
            _block("sec-part1-titled", "b-p1t-1", "PART ONE", BlockType.HEADING, 0),
            _block("sec-part1-titled", "b-p1t-2", "Afterlights", BlockType.HEADING, 1),
        ),
    )
    titled_paragraphs = DocumentSection(
        id="sec-part2-titled",
        document_id="book-1",
        title="PART TWO / Mary, Queen of Snots",
        order=1,
        source_resource_path="part2.xhtml",
        blocks=(
            _block("sec-part2-titled", "b-p2t-1", "PART TWO", BlockType.PARAGRAPH, 0),
            _block("sec-part2-titled", "b-p2t-2", "Mary, Queen of Snots", BlockType.PARAGRAPH, 1),
        ),
    )
    subtitle_in_block_only = DocumentSection(
        id="sec-part3-block-title",
        document_id="book-1",
        title="Part Three",
        order=2,
        source_resource_path="part3.xhtml",
        blocks=(
            _block("sec-part3-block-title", "b-p3t-1", "PART THREE", BlockType.PARAGRAPH, 0),
            _block("sec-part3-block-title", "b-p3t-2", "The Chocolate Ogre", BlockType.PARAGRAPH, 1),
        ),
    )
    bare_part_one = DocumentSection(
        id="sec-bare-part-one",
        document_id="book-1",
        title="PART ONE",
        order=3,
        source_resource_path="bare1.xhtml",
        blocks=(_block("sec-bare-part-one", "b-bp1", "PART ONE", BlockType.HEADING, 0),),
    )
    bare_part_two = DocumentSection(
        id="sec-bare-part-two",
        document_id="book-1",
        title="Part 2 —",
        order=4,
        source_resource_path="bare2.xhtml",
        blocks=(
            _block("sec-bare-part-two", "b-bp2a", "Part 2", BlockType.PARAGRAPH, 0),
            _block("sec-bare-part-two", "b-bp2b", "— 2024 —", BlockType.PARAGRAPH, 1),
        ),
    )
    bare_book_iv = DocumentSection(
        id="sec-bare-book-iv",
        document_id="book-1",
        title="Book IV",
        order=5,
        source_resource_path="bare3.xhtml",
        blocks=(_block("sec-bare-book-iv", "b-biv", "Book IV.", BlockType.HEADING, 0),),
    )
    bare_ru_part = DocumentSection(
        id="sec-bare-ru",
        document_id="book-1",
        title="Часть вторая",
        order=6,
        source_resource_path="bare4.xhtml",
        blocks=(_block("sec-bare-ru", "b-bru", "Часть вторая.", BlockType.HEADING, 0),),
    )
    regular_chapter = DocumentSection(
        id="sec-ch1",
        document_id="book-1",
        title="Chapter 1: Afterlights",
        order=7,
        source_resource_path="ch1.xhtml",
        blocks=(
            _block("sec-ch1", "b-ch1h", "Chapter 1", BlockType.HEADING, 0),
            _block("sec-ch1", "b-ch1p", "Nick and Allie woke up in the woods.", BlockType.PARAGRAPH, 1),
        ),
    )
    dedication = DocumentSection(
        id="sec-ded",
        document_id="book-1",
        title="Dedication",
        order=8,
        source_resource_path="dedication.xhtml",
        blocks=(_block("sec-ded", "b-ded", "For everyone.", BlockType.PARAGRAPH, 0),),
    )

    doc = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Everlost"),
        source_language="en",
        target_language="ru",
        sections=(
            titled_heading,
            titled_paragraphs,
            subtitle_in_block_only,
            bare_part_one,
            bare_part_two,
            bare_book_iv,
            bare_ru_part,
            regular_chapter,
            dedication,
        ),
    )

    for section in (titled_heading, titled_paragraphs, subtitle_in_block_only):
        assert is_part_divider(section) is True
        assert is_front_matter(section, doc) is True
        assert doc.is_front_matter(section) is True
        assert has_meaningful_part_title(section) is True
        assert doc.has_meaningful_part_title(section) is True

    for section in (bare_part_one, bare_part_two, bare_book_iv, bare_ru_part):
        assert is_part_divider(section) is True
        assert is_front_matter(section, doc) is True
        assert doc.is_front_matter(section) is True
        assert has_meaningful_part_title(section) is False
        assert doc.has_meaningful_part_title(section) is False

    for section in (regular_chapter, dedication):
        assert is_part_divider(section) is False
        assert has_meaningful_part_title(section) is False
        assert doc.has_meaningful_part_title(section) is False


def test_every_import_warning_kind_has_ui_label() -> None:
    """Every `ImportWarningKind` enum value is mapped to a Russian label."""
    assert set(IMPORT_WARNING_LABELS) == set(ImportWarningKind)
