"""EPUB import: reading order, sections, blocks, and stable identifiers (P2.T1)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import EbookLibEpubReader
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubReadRequest,
    ImportedBook,
)
from linguary.application.use_cases import ImportBook, ImportBookCommand, ImportBookError
from linguary.domain import (
    BlockType,
    BookStatus,
    ImportWarningKind,
    SupportStatus,
)

ReadFixture = Callable[..., ImportedBook]


def test_reading_order_gives_one_section_per_spine_document(
    read_fixture: ReadFixture,
) -> None:
    """One chapter section corresponds to one XHTML document in the reading order."""
    document = read_fixture("synthetic_story").document

    sections = document.ordered_sections

    assert [section.order for section in sections] == [0, 1, 2]
    assert [section.source_resource_path for section in sections] == [
        "chapter1.xhtml",
        "chapter2.xhtml",
        "chapter3.xhtml",
    ]


def test_section_titles_come_from_navigation(read_fixture: ReadFixture) -> None:
    """Chapter title is taken from nav/toc if present there."""
    document = read_fixture("synthetic_story").document

    assert [section.title for section in document.ordered_sections] == [
        "Chapter 1: The Road to Duskmere",
        "Chapter 2: The Keeper of Duskmere",
        "Chapter 3: The Light Beneath the Hollow",
    ]


@pytest.mark.parametrize(
    ("toc", "head_title", "heading"),
    [
        pytest.param(
            [("chapter1.xhtml", "1    Here Comes\n  Charlie")],
            "Ignored",
            "Ignored",
            id="from-navigation",
        ),
        pytest.param(None, "1 \t Here Comes\n Charlie ", "Ignored", id="from-head-title"),
        pytest.param(None, " ", "1\n    Here <em>Comes</em>   Charlie", id="from-heading"),
    ],
)
def test_section_title_whitespace_is_collapsed(
    reader: EbookLibEpubReader,
    tmp_path: Path,
    toc: list[tuple[str, str]] | None,
    head_title: str,
    heading: str,
) -> None:
    """Whitespace characters inside section title are collapsed to single spaces."""
    source = build_epub(
        tmp_path / "book.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                f"<h1>{heading}</h1><p>Text.</p>", title=head_title
            )
        },
        toc=toc,
    )

    document = reader.read(_request(source)).document

    assert [section.title for section in document.ordered_sections] == [
        "1 Here Comes Charlie"
    ]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("synthetic_story", id="epub3-synthetic"),
        pytest.param("great_gatsby", id="epub3-gutenberg"),
        pytest.param("beowulf", id="epub2-gutenberg"),
    ],
)
def test_block_identifiers_are_stable_across_repeated_import(
    read_fixture: ReadFixture, name: str
) -> None:
    """Repeated import of the same book produces identical block IDs in the same order."""
    first = [block.id for block in read_fixture(name).document.blocks()]
    second = [block.id for block in read_fixture(name).document.blocks()]

    assert first == second
    assert len(set(first)) == len(first)


def test_block_identifier_depends_only_on_position_in_document(
    read_fixture: ReadFixture,
) -> None:
    """Block ID is computed from document, resource, and structural path (ALG-01)."""
    document = read_fixture("synthetic_story", book_id="book-1").document
    other_book = read_fixture("synthetic_story", book_id="book-2").document

    blocks = list(document.blocks())
    other_blocks = list(other_book.blocks())

    assert [block.source_reference.structural_path for block in blocks] == [
        block.source_reference.structural_path for block in other_blocks
    ]
    assert all(
        block.id != other.id for block, other in zip(blocks, other_blocks, strict=True)
    )


def test_block_identifiers_survive_text_edits(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Editing block text does not change its block identifier (PRD § 8)."""
    body = "<h1>{title}</h1><p>{first}</p><p>Second paragraph.</p>"
    original = build_epub(
        tmp_path / "original.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                body.format(title="Chapter", first="First paragraph.")
            )
        },
    )
    edited = build_epub(
        tmp_path / "edited.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                body.format(title="Chapter", first="A completely rewritten opening.")
            )
        },
    )

    original_blocks = list(reader.read(_request(original)).document.blocks())
    edited_blocks = list(reader.read(_request(edited)).document.blocks())

    assert [block.text for block in original_blocks] != [
        block.text for block in edited_blocks
    ]
    assert [block.id for block in original_blocks] == [
        block.id for block in edited_blocks
    ]


def test_blocks_keep_document_order_and_source_reference(
    read_fixture: ReadFixture,
) -> None:
    """Blocks follow document order and retain their source XHTML origin."""
    document = read_fixture("synthetic_story").document

    for section in document.ordered_sections:
        blocks = section.ordered_blocks
        assert [block.order for block in blocks] == list(range(len(blocks)))
        assert all(
            block.source_reference.resource_path == section.source_resource_path
            for block in blocks
        )
        assert all(block.source_reference.structural_path for block in blocks)


def test_block_order_follows_the_source_document(read_fixture: ReadFixture) -> None:
    """Order of blocks in a section mirrors the element order of source XHTML."""
    document = read_fixture("synthetic_story").document
    blocks = document.ordered_sections[0].ordered_blocks

    assert [block.source_reference.structural_path for block in blocks] == [
        "/html[1]/body[1]/h1[1]",
        "/html[1]/body[1]/p[1]",
        "/html[1]/body[1]/p[2]",
        "/html[1]/body[1]/p[3]",
        "/html[1]/body[1]/p[4]",
        "/html[1]/body[1]/ol[1]/li[1]",
        "/html[1]/body[1]/ol[1]/li[2]",
        "/html[1]/body[1]/ol[1]/li[3]",
        "/html[1]/body[1]/ol[1]/li[4]",
        "/html[1]/body[1]/p[5]",
        "/html[1]/body[1]/p[6]",
        "/html[1]/body[1]/aside[1]",
        "/html[1]/body[1]/aside[1]/p[1]",
    ]


def test_structural_blocks_are_recognised_by_type(read_fixture: ReadFixture) -> None:
    """Headings, paragraphs, list items, quotes, and images map to their respective block types."""
    document = read_fixture("synthetic_story").document
    first_chapter = document.ordered_sections[0].ordered_blocks

    assert first_chapter[0].type is BlockType.HEADING
    assert first_chapter[0].text == "Chapter 1: The Road to Duskmere"
    assert first_chapter[1].type is BlockType.PARAGRAPH
    assert [block.text for block in first_chapter if block.type is BlockType.LIST_ITEM] == [
        "a brass pocket compass that pointed roughly north, on a good day;",
        "a heel of dried bread wrapped in cloth;",
        "a small tin lantern, unlit, saved for the marsh;",
        "the starglass itself, cold and faintly humming.",
    ]
    quotes = [block for block in document.blocks() if block.type is BlockType.QUOTE]
    assert [block.text for block in quotes] == [
        "What the glass has not yet shown you, it is already carrying for someone else."
    ]


def test_nested_block_points_to_its_parent(read_fixture: ReadFixture) -> None:
    """Block nesting is preserved via `parent_id` rather than flattened into text."""
    document = read_fixture("synthetic_story").document
    footnotes = [
        block for block in document.blocks() if block.type is BlockType.FOOTNOTE
    ]

    container, paragraph = footnotes
    assert container.parent_id is None
    assert container.source_reference.element_id == "fn1"
    assert paragraph.parent_id == container.id
    assert paragraph.text.startswith("[1] A starglass is a hand-sized lens")


def test_image_block_refers_to_manifest_resource(read_fixture: ReadFixture) -> None:
    """Image block points to a book resource, and the resource itself exists in model."""
    imported = read_fixture("synthetic_story")
    images = [
        block for block in imported.document.blocks() if block.type is BlockType.IMAGE
    ]

    assert len(images) == 1
    assert images[0].attributes["src"] == "images/marker.png"
    assert images[0].attributes["alt"].startswith("A small carved marker")
    assert "images/marker.png" in {
        resource.source_path for resource in imported.document.resources
    }
    assert not [
        warning
        for warning in imported.warnings
        if warning.kind is ImportWarningKind.MISSING_RESOURCE
    ]


def test_table_is_preserved_without_internal_processing(
    read_fixture: ReadFixture,
) -> None:
    """Complex tables are preserved as `PRESERVE_OPAQUE` rather than parsed cell-by-cell."""
    document = read_fixture("great_gatsby").document
    tables = [block for block in document.blocks() if block.type is BlockType.TABLE]

    assert tables
    assert all(block.support_status is SupportStatus.PRESERVE_OPAQUE for block in tables)


def test_toc_mismatch_is_reported_as_warning(read_fixture: ReadFixture) -> None:
    """A book where multiple chapters reside in a single document imports with a warning."""
    imported = read_fixture("great_gatsby")

    mismatches = [
        warning
        for warning in imported.warnings
        if warning.kind is ImportWarningKind.TOC_MISMATCH
    ]
    assert mismatches
    assert len(imported.document.sections) == 4
    assert any("порядка чтения" in warning.message for warning in mismatches)


def test_matching_navigation_gives_no_mismatch_warning(
    read_fixture: ReadFixture,
) -> None:
    """Matching navigation TOC does not produce mismatch warnings."""
    imported = read_fixture("synthetic_story")

    assert not [
        warning
        for warning in imported.warnings
        if warning.kind is ImportWarningKind.TOC_MISMATCH
    ]


def test_metadata_is_read_from_package(read_fixture: ReadFixture) -> None:
    """Book metadata is read from the package file."""
    document = read_fixture("synthetic_story").document

    assert document.metadata.title == "The Lamp at Duskmere"
    assert document.metadata.authors == ("Linguary Test Fixtures",)
    assert document.source_language == "en"
    assert document.target_language == "ru"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("synthetic_story", id="synthetic"),
        pytest.param("great_gatsby", id="great-gatsby"),
        pytest.param("beowulf", id="beowulf-epub2"),
        pytest.param("alice_rackham", id="alice-images"),
        pytest.param("dracula", id="dracula-formatting"),
    ],
)
@pytest.mark.slow
def test_every_fixture_book_imports_without_fatal(
    read_fixture: ReadFixture, name: str
) -> None:
    """Every fixture book imports without `FATAL` status and yields a non-empty model (PRD § 28.1)."""
    imported = read_fixture(name)

    assert imported.document.sections
    assert list(imported.document.blocks())
    assert all(
        block.support_status is not SupportStatus.FATAL
        for block in imported.document.blocks()
    )


def test_broken_archive_is_fatal(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """An unreadable book file cannot be imported."""
    source = tmp_path / "broken.epub"
    source.write_bytes(b"not an archive at all")

    with pytest.raises(EpubImportFatalError):
        reader.read(_request(source))


def test_archive_without_package_file_is_fatal(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """An archive missing the OPF package file cannot be imported."""
    source = build_epub(
        tmp_path / "no_package.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        include_package=False,
    )

    with pytest.raises(EpubImportFatalError):
        reader.read(_request(source))


def test_book_without_reading_order_is_fatal(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Missing reading order prevents import."""
    source = build_epub(
        tmp_path / "no_spine.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        spine=(),
    )

    with pytest.raises(EpubImportFatalError, match="порядка чтения"):
        reader.read(_request(source))


def test_unparsable_document_is_fatal(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """XHTML that cannot be parsed into blocks makes import impossible."""
    source = build_epub(
        tmp_path / "broken_xhtml.epub",
        documents={"chapter1.xhtml": '<?xml version="1.0"?><html><body><p>Text'},
    )

    with pytest.raises(EpubImportFatalError, match="нельзя разобрать"):
        reader.read(_request(source))


def test_lost_reading_order_document_is_fatal(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """A reading-order document lost from archive makes import impossible."""
    source = build_epub(
        tmp_path / "lost_document.epub",
        documents={
            "chapter1.xhtml": xhtml_document("<p>Text.</p>"),
            "chapter2.xhtml": xhtml_document("<p>More text.</p>"),
        },
        omit_from_archive=("chapter2.xhtml",),
    )

    with pytest.raises(EpubImportFatalError, match="нельзя прочитать"):
        reader.read(_request(source))


def test_import_use_case_stores_document_warnings_and_status(
    unit_of_work: SqlAlchemyUnitOfWork, reader: EbookLibEpubReader
) -> None:
    """Import use case stores the book, its model, and warnings in a single transaction."""
    source = str(_fixture_path("great_gatsby"))
    import_book = ImportBook(unit_of_work=unit_of_work, reader=reader)

    result = import_book(
        ImportBookCommand(
            book_id="book-1", source_epub_path=source, target_language="ru"
        )
    )

    assert result.book.status is BookStatus.IMPORTED
    with unit_of_work.transaction() as transaction:
        stored = transaction.documents.get("book-1")
        warnings = transaction.documents.import_warnings("book-1")
        book = transaction.books.get("book-1")
    assert book is not None
    assert book.source_epub_path == source
    assert stored is not None
    assert [section.id for section in stored.ordered_sections] == [
        section.id for section in result.document.ordered_sections
    ]
    assert [block.id for block in stored.blocks()] == [
        block.id for block in result.document.blocks()
    ]
    assert {warning.kind for warning in warnings} == {
        warning.kind for warning in result.warnings
    }


def test_repeated_import_keeps_block_identifiers(
    unit_of_work: SqlAlchemyUnitOfWork, reader: EbookLibEpubReader
) -> None:
    """Repeated import does not alter identifiers of previously stored blocks."""
    command = ImportBookCommand(
        book_id="book-1",
        source_epub_path=str(_fixture_path("synthetic_story")),
        target_language="ru",
    )
    import_book = ImportBook(unit_of_work=unit_of_work, reader=reader)

    first = import_book(command)
    second = import_book(command)

    with unit_of_work.transaction() as transaction:
        stored = transaction.documents.get("book-1")
    assert stored is not None
    assert [block.id for block in stored.blocks()] == [
        block.id for block in first.document.blocks()
    ]
    assert second.book.status is BookStatus.IMPORTED


def test_import_is_rejected_after_book_moved_past_import(
    unit_of_work: SqlAlchemyUnitOfWork, reader: EbookLibEpubReader
) -> None:
    """A book that progressed beyond import cannot be re-imported."""
    command = ImportBookCommand(
        book_id="book-1",
        source_epub_path=str(_fixture_path("synthetic_story")),
        target_language="ru",
    )
    import_book = ImportBook(unit_of_work=unit_of_work, reader=reader)
    stored = import_book(command).book

    with unit_of_work.transaction() as transaction:
        transaction.books.save(
            type(stored)(
                id=stored.id,
                metadata=stored.metadata,
                source_language=stored.source_language,
                target_language=stored.target_language,
                status=BookStatus.ANALYZING,
                source_epub_path=stored.source_epub_path,
                created_at=stored.created_at,
            )
        )

    with pytest.raises(ImportBookError):
        import_book(command)


def _request(source: Path) -> EpubReadRequest:
    """Request to read an in-test synthesized EPUB book."""
    return EpubReadRequest(
        book_id="book-1", source_path=str(source), target_language="ru"
    )


def _fixture_path(name: str) -> Path:
    """Path to a fixture EPUB in the test suite."""
    return Path(__file__).resolve().parents[2] / "fixtures" / "epub" / f"{name}.epub"
