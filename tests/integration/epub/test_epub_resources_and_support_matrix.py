"""EPUB import: local resource storage and support matrix (P2.T3)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import (
    EbookLibEpubReader,
    ExtractionLimits,
    FilesystemEpubResourceStore,
)
from linguary.application.ports.epub import (
    EpubImportFatalError,
    EpubReadRequest,
    ImportedBook,
    StoredResources,
    StoreResourcesRequest,
)
from linguary.application.use_cases import ImportBook, ImportBookCommand
from linguary.domain import ImportWarningKind, SupportStatus

ReadFixture = Callable[..., ImportedBook]

#: Single-pixel image: content is irrelevant to tests, only file existence matters.
IMAGE_BYTES = b"\x89PNG\r\n\x1a\n fake image payload"


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    """Root of the local book resources store."""
    return tmp_path / "storage"


@pytest.fixture
def resource_store(storage_root: Path) -> FilesystemEpubResourceStore:
    """Resource store under test."""
    return FilesystemEpubResourceStore(root=storage_root)


def test_images_are_stored_under_the_book_directory(
    reader: EbookLibEpubReader,
    resource_store: FilesystemEpubResourceStore,
    storage_root: Path,
    tmp_path: Path,
) -> None:
    """Book images are saved into the book's directory and retrieved via original paths."""
    source = _book_with_image(tmp_path)

    stored = _store(reader, resource_store, source)

    copy = Path(stored.paths["images/marker.png"])
    assert copy.read_bytes() == IMAGE_BYTES
    assert copy == storage_root / "book-1" / "resources" / "images" / "marker.png"


def test_stylesheet_is_preserved_without_internal_processing(
    reader: EbookLibEpubReader,
    resource_store: FilesystemEpubResourceStore,
    tmp_path: Path,
) -> None:
    """Book stylesheets are transferred as-is: no need for importer to parse them (PRD § 6)."""
    source = _book_with_image(tmp_path)

    imported = reader.read(_request(source))
    stored = resource_store.store(
        StoreResourcesRequest(
            book_id="book-1", source_path=str(source), document=imported.document
        )
    )

    styles = _resource(imported, "style.css")
    assert styles.support_status is SupportStatus.PRESERVE_OPAQUE
    assert Path(stored.paths["style.css"]).exists()


def test_script_resource_is_dropped_with_warning(
    reader: EbookLibEpubReader,
    resource_store: FilesystemEpubResourceStore,
    storage_root: Path,
    tmp_path: Path,
) -> None:
    """Scripts have no effect on reading: dropped with an import warning."""
    source = _book_with_image(tmp_path)

    imported = reader.read(_request(source))
    stored = resource_store.store(
        StoreResourcesRequest(
            book_id="book-1", source_path=str(source), document=imported.document
        )
    )

    assert _resource(imported, "app.js").support_status is SupportStatus.DROP_WITH_WARNING
    assert "app.js" not in stored.paths
    assert not (storage_root / "book-1" / "resources" / "app.js").exists()
    assert [
        warning.message
        for warning in imported.warnings
        if warning.kind is ImportWarningKind.UNSUPPORTED_CONSTRUCT
        and "app.js" in warning.message
    ]


def test_empty_decorative_elements_are_dropped_with_warning(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Empty decorative elements are skipped and emit an import warning."""
    source = build_epub(
        tmp_path / "decorations.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<p>Text.</p><div class='decoration'></div><span class='rule'></span>"
            )
        },
    )

    imported = reader.read(_request(source))

    dropped = [
        warning
        for warning in imported.warnings
        if warning.kind is ImportWarningKind.DROPPED_BLOCK
    ]
    assert len(dropped) == 1
    assert "пустые декоративные элементы" in dropped[0].message
    assert dropped[0].section_id == imported.document.ordered_sections[0].id
    assert [block.text for block in imported.document.blocks()] == ["Text."]


def test_repeated_page_break_is_dropped_with_warning(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Consecutive page breaks describe a single boundary: second is dropped."""
    source = build_epub(
        tmp_path / "page_breaks.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<p>Text.</p>"
                "<div class='pagebreak' id='p1'></div>"
                "<div class='pagebreak' id='p2'></div>"
                "<p>More text.</p>"
            )
        },
    )

    imported = reader.read(_request(source))

    dropped = [
        warning
        for warning in imported.warnings
        if warning.kind is ImportWarningKind.DROPPED_BLOCK
    ]
    assert len(dropped) == 1
    assert "разрывы страницы" in dropped[0].message
    assert [block.type.value for block in imported.document.blocks()] == [
        "Paragraph",
        "PageBreak",
        "Paragraph",
    ]


@pytest.mark.parametrize(
    ("markup", "expected_type"),
    [
        pytest.param("<div class='pagebreak' id='p1'></div>", "PageBreak", id="class"),
        pytest.param(
            "<div class='x-ebookmaker-pageno' id='p1'></div>",
            "PageBreak",
            id="publisher-class",
        ),
        pytest.param(
            "<div epub:type='pagebreak' id='p1'></div>", "PageBreak", id="epub-type"
        ),
        pytest.param(
            "<p class='pagenote'>Note text.</p>", "Paragraph", id="similar-class"
        ),
    ],
)
def test_page_break_is_recognised_by_whole_marker(
    markup: str, expected_type: str, reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Page break is identified by a dedicated marker, not as a substring of another class."""
    source = build_epub(
        tmp_path / "page_break_marker.epub",
        documents={"chapter1.xhtml": xhtml_document(f"<p>Text.</p>{markup}")},
    )

    imported = reader.read(_request(source))

    assert [block.type.value for block in imported.document.blocks()] == [
        "Paragraph",
        expected_type,
    ]


def test_lost_manifest_resource_is_fatal(
    reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """A resource declared in manifest but missing from archive is `FATAL` (PRD § 6)."""
    source = build_epub(
        tmp_path / "lost_image.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<p>Text.</p><img src='images/marker.png' alt='Marker'/>"
            )
        },
        extra_files={"images/marker.png": IMAGE_BYTES},
        omit_from_archive=("images/marker.png",),
    )

    with pytest.raises(EpubImportFatalError):
        reader.read(_request(source))


def test_resource_path_leaving_the_book_directory_is_fatal(
    reader: EbookLibEpubReader,
    resource_store: FilesystemEpubResourceStore,
    storage_root: Path,
    tmp_path: Path,
) -> None:
    """A resource path containing `../` traversal is rejected, writing nothing outside store."""
    source = build_epub(
        tmp_path / "zip_slip.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<p>Text.</p><img src='../outside.png' alt='Outside'/>"
            )
        },
        extra_files={"../outside.png": IMAGE_BYTES},
        omit_from_archive=("../outside.png",),
        archive_files={"outside.png": IMAGE_BYTES},
    )

    with pytest.raises(EpubImportFatalError, match="за пределы каталога хранилища"):
        _store(reader, resource_store, source)

    assert not storage_root.exists()
    assert not (tmp_path / "outside.png").exists()


def test_archive_with_too_many_files_is_fatal(
    reader: EbookLibEpubReader, storage_root: Path, tmp_path: Path
) -> None:
    """An archive with entries exceeding limit is rejected before writing to storage."""
    source = build_epub(
        tmp_path / "many_files.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        extra_files={f"images/{index}.png": IMAGE_BYTES for index in range(10)},
    )
    store = FilesystemEpubResourceStore(
        root=storage_root, limits=ExtractionLimits(max_entries=5)
    )

    with pytest.raises(EpubImportFatalError, match="файлов при допустимых"):
        _store(reader, store, source)

    assert not storage_root.exists()


def test_archive_larger_than_the_limit_is_fatal(
    reader: EbookLibEpubReader, storage_root: Path, tmp_path: Path
) -> None:
    """An archive whose extraction size exceeds limit is rejected before storage write."""
    source = build_epub(
        tmp_path / "large.epub",
        documents={"chapter1.xhtml": xhtml_document("<p>Text.</p>")},
        extra_files={"images/large.png": IMAGE_BYTES * 1000},
    )
    store = FilesystemEpubResourceStore(
        root=storage_root, limits=ExtractionLimits(max_total_bytes=1024)
    )

    with pytest.raises(EpubImportFatalError, match="превыш|заявлено"):
        _store(reader, store, source)

    assert not storage_root.exists()


def test_repeated_store_forgets_resources_of_the_previous_import(
    reader: EbookLibEpubReader,
    resource_store: FilesystemEpubResourceStore,
    storage_root: Path,
    tmp_path: Path,
) -> None:
    """Repeated import reinitializes the book directory: no leftover copies from past import."""
    _store(reader, resource_store, _book_with_image(tmp_path))
    obsolete = storage_root / "book-1" / "resources" / "images" / "obsolete.png"
    obsolete.write_bytes(IMAGE_BYTES)

    _store(reader, resource_store, _book_with_image(tmp_path))

    assert not obsolete.exists()
    assert (storage_root / "book-1" / "resources" / "images" / "marker.png").exists()


def test_import_use_case_saves_stored_paths_and_warnings(
    unit_of_work: SqlAlchemyUnitOfWork,
    reader: EbookLibEpubReader,
    resource_store: FilesystemEpubResourceStore,
    tmp_path: Path,
) -> None:
    """Import use case saves stored resource paths and support matrix warnings."""
    source = _book_with_image(tmp_path)
    import_book = ImportBook(
        unit_of_work=unit_of_work, reader=reader, resource_store=resource_store
    )

    result = import_book(
        ImportBookCommand(
            book_id="book-1", source_epub_path=str(source), target_language="ru"
        )
    )

    with unit_of_work.transaction() as transaction:
        paths = transaction.documents.stored_resource_paths("book-1")
        warnings = transaction.documents.import_warnings("book-1")
        document = transaction.documents.get("book-1")
    assert Path(paths["images/marker.png"]).exists()
    assert "app.js" not in paths
    assert ImportWarningKind.UNSUPPORTED_CONSTRUCT in {
        warning.kind for warning in warnings
    }
    assert len(warnings) == len(result.warnings)
    assert document is not None
    assert {
        resource.source_path: resource.support_status
        for resource in document.resources
    } == {
        "images/marker.png": SupportStatus.SUPPORTED,
        "style.css": SupportStatus.PRESERVE_OPAQUE,
        "app.js": SupportStatus.DROP_WITH_WARNING,
    }


def test_import_without_resource_store_keeps_document_only(
    unit_of_work: SqlAlchemyUnitOfWork, reader: EbookLibEpubReader, tmp_path: Path
) -> None:
    """Without a resource store, book imports only the document model, omitting resource copies."""
    source = _book_with_image(tmp_path)
    import_book = ImportBook(unit_of_work=unit_of_work, reader=reader)

    import_book(
        ImportBookCommand(
            book_id="book-1", source_epub_path=str(source), target_language="ru"
        )
    )

    with unit_of_work.transaction() as transaction:
        assert transaction.documents.stored_resource_paths("book-1") == {}


@pytest.mark.slow
def test_illustrated_book_stores_every_image(
    read_fixture: ReadFixture,
    resource_store: FilesystemEpubResourceStore,
    storage_root: Path,
) -> None:
    """Illustrated book stores all its images (PRD § 28.1)."""
    imported = read_fixture("alice_rackham")
    source = Path(__file__).resolve().parents[2] / "fixtures" / "epub" / "alice_rackham.epub"

    stored = resource_store.store(
        StoreResourcesRequest(
            book_id="book-1", source_path=str(source), document=imported.document
        )
    )

    images = [
        resource
        for resource in imported.document.resources
        if resource.media_type.startswith("image/")
    ]
    assert len(images) > 10
    assert {resource.source_path for resource in images} <= set(stored.paths)
    assert all(Path(path).exists() for path in stored.paths.values())
    assert all(
        storage_root / "book-1" / "resources" in Path(path).parents
        for path in stored.paths.values()
    )


def _book_with_image(directory: Path) -> Path:
    """A book with an image, stylesheet, and script: three support matrix classes at once."""
    return build_epub(
        directory / "illustrated.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter</h1><p>Text.</p>"
                "<img src='images/marker.png' alt='A carved marker'/>"
            )
        },
        extra_files={
            "images/marker.png": IMAGE_BYTES,
            "style.css": b"p { margin: 0; }",
            "app.js": b"var reader = 1;",
        },
    )


def _store(
    reader: EbookLibEpubReader,
    store: FilesystemEpubResourceStore,
    source: Path,
    *,
    book_id: str = "book-1",
) -> StoredResources:
    """Reads the book and stores its resources into the store."""
    imported = reader.read(_request(source, book_id=book_id))
    return store.store(
        StoreResourcesRequest(
            book_id=book_id, source_path=str(source), document=imported.document
        )
    )


def _resource(imported: ImportedBook, source_path: str):
    """Resource of the book by its internal path inside EPUB."""
    for resource in imported.document.resources:
        if resource.source_path == source_path:
            return resource
    raise AssertionError(f"resource {source_path} is absent from book model")


def _request(source: Path, *, book_id: str = "book-1") -> EpubReadRequest:
    """Request to read an in-test synthesized book."""
    return EpubReadRequest(
        book_id=book_id, source_path=str(source), target_language="ru"
    )
