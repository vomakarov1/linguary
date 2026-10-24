"""Importing a real Project Gutenberg book preserves the license without passing it to the model."""

from pathlib import Path

from conftest import FIXTURE_DIRECTORY

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.epub import (
    EbookLibEpubReader,
    EpubExportWriter,
    FilesystemEpubResourceStore,
    verify_exported_epub,
)
from linguary.application.ports.epub import EpubWriteRequest
from linguary.application.use_cases.import_book import ImportBook, ImportBookCommand
from linguary.domain import SupportStatus


def test_dracula_license_section_has_no_textual_blocks(
    unit_of_work: SqlAlchemyUnitOfWork, tmp_path: Path
) -> None:
    source = FIXTURE_DIRECTORY / "dracula.epub"
    result = ImportBook(
        unit_of_work=unit_of_work,
        reader=EbookLibEpubReader(),
        resource_store=FilesystemEpubResourceStore(root=tmp_path / "storage"),
    )(
        ImportBookCommand(
            book_id="dracula", source_epub_path=str(source), target_language="ru"
        )
    )
    with unit_of_work.transaction() as transaction:
        document = transaction.documents.get("dracula")
        warnings = transaction.documents.import_warnings("dracula")

    assert document is not None
    license_section = document.ordered_sections[-1]
    assert "PROJECT GUTENBERG" in license_section.ordered_blocks[0].text.upper()
    assert all(
        block.support_status is SupportStatus.PRESERVE_OPAQUE
        for block in license_section.blocks
        if block.text.strip()
    )
    assert result.warnings == warnings
    assert any("Project Gutenberg" in warning.message for warning in warnings)

    exported = EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(tmp_path / "dracula-export.epub"),
            source_epub_path=str(source),
            stored_resource_paths={},
        )
    )
    assert verify_exported_epub(
        Path(exported.path),
        expected_spine=tuple(
            section.source_resource_path for section in document.ordered_sections
        ),
    ) == ()
