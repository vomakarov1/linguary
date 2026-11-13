"""Composition reads segments and does not rewrite block identity or stored results."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config

from linguary.adapters.db.engine import build_database_url, create_database_engine
from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.application.layout.parts import LayoutPart, compose_parts
from linguary.application.ports.persistence import BookRecord
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    GlossaryHint,
    GlossarySnapshot,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    ResultWarning,
    ResultWarningKind,
    SourceReference,
)

BOOK_ID = "book-1"
SECTION_ID = "section-1"
RUN_ID = "run-1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _block() -> DocumentBlock:
    """Three translated spans whose grouping depends on the part window."""
    text = ("a" * 150) + ("b" * 150) + ("c" * 150)
    return DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="chapter.xhtml",
            structural_path="/body/p[1]",
        ),
        segments=(
            BlockSegment(0, 150, 0),
            BlockSegment(150, 300, 1),
            BlockSegment(300, 450, 2),
        ),
    )


def _document(block: DocumentBlock) -> CanonicalDocument:
    """A one-chapter book containing only `block`."""
    section = DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(block,),
    )
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Sample"),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )


def _result(block_id: str) -> BlockProcessingResult:
    """A stored translation that composition must leave untouched."""
    return BlockProcessingResult(
        block_id=block_id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Раз. Два. Три.",
        glossary_hints=(GlossaryHint(original="day", translation="день"),),
        warnings=(
            ResultWarning(kind=ResultWarningKind.REPETITION, message="Повтор слова."),
        ),
        segment_treatments=(BlockTreatment.TRANSLATE,) * 3,
        segment_translations=("Раз.", "Два.", "Три."),
    )


def test_changing_part_params_keeps_block_ids_and_results_end_to_end(tmp_path: Path) -> None:
    """Different part windows change boundaries only, not ids or saved results."""
    block = _block()
    document = _document(block)
    result = _result(block.id)
    database_path = tmp_path / "linguary.db"
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", build_database_url(database_path))
    command.upgrade(config, "head")
    engine = create_database_engine(database_path)
    try:
        unit_of_work = SqlAlchemyUnitOfWork.for_engine(engine)
        with unit_of_work.transaction() as active:
            active.books.add(
                BookRecord(
                    id=BOOK_ID,
                    metadata=BookMetadata(title="Sample"),
                    source_language="en",
                    target_language="ru",
                    status=BookStatus.IMPORTED,
                    source_epub_path="/library/sample.epub",
                    created_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
                )
            )
            active.documents.save(document)
            active.runs.add(
                ProcessingRun(
                    id=RUN_ID,
                    book_id=BOOK_ID,
                    status=ProcessingRunStatus.ACTIVE,
                    glossary_snapshot=GlossarySnapshot(),
                    llm_settings_snapshot=LLMSettingsSnapshot(provider="openai", model="gpt-test"),
                )
            )
            active.results.replace_results(RUN_ID, (result,))

        with unit_of_work.transaction() as active:
            stored = active.documents.get(BOOK_ID)
            stored_results = active.results.list_for_run(RUN_ID)
        assert stored is not None
        stored_block = stored.block(block.id)
        identity = (stored_block.id, stored_block.text, stored_block.segments)
        treatments = (BlockTreatment.TRANSLATE,) * len(stored_block.segment_spans)
        narrow = compose_parts(
            stored_block,
            treatments,
            target_chars=300,
            min_chars=250,
            max_chars=350,
        )
        wide = compose_parts(
            stored_block,
            treatments,
            target_chars=450,
            min_chars=100,
            max_chars=500,
        )
        assert narrow == (
            LayoutPart(0, 150, BlockTreatment.TRANSLATE, (0,)),
            LayoutPart(150, 450, BlockTreatment.TRANSLATE, (1, 2)),
        )
        assert wide == (LayoutPart(0, 450, BlockTreatment.TRANSLATE, (0, 1, 2)),)

        with unit_of_work.transaction() as active:
            reloaded = active.documents.get(BOOK_ID)
            reloaded_results = active.results.list_for_run(RUN_ID)
    finally:
        engine.dispose()

    assert reloaded is not None
    reloaded_block = reloaded.block(block.id)
    assert tuple(item.id for item in reloaded.blocks()) == (block.id,)
    assert (reloaded_block.id, reloaded_block.text, reloaded_block.segments) == identity
    assert reloaded_results == stored_results == (result,)
