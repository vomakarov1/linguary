"""Persistence of block segments and per-segment processing fields."""

from __future__ import annotations

import json
from dataclasses import replace

from alembic import command
from sqlalchemy import Engine, inspect, text
from sqlalchemy.engine import Connection

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    TreatmentPlanItem,
)
from persistence_factories import (
    BLOCK_ID,
    BOOK_ID,
    RUN_ID,
    SECOND_BLOCK_ID,
    SECTION_ID,
    make_analysis,
    make_book,
    make_document,
    make_glossary_only_result,
    make_run,
)

TEXT = "Alpha. Beta."
SEGMENTS = (
    BlockSegment(0, 6, 0),
    BlockSegment(6, len(TEXT), 1),
)


def _json_cell(value: object) -> object:
    """Decode a JSON cell returned by a raw SQL query."""
    if value is None or not isinstance(value, str):
        return value
    return json.loads(value)


def _columns(engine: Engine, table: str) -> dict[str, dict[str, object]]:
    return {column["name"]: column for column in inspect(engine).get_columns(table)}


def _literal_default(default: object) -> object:
    """Strip SQL quotes from a reflected server default."""
    if not isinstance(default, str) or len(default) < 2:
        return default
    if default[0] == default[-1] and default[0] in {"'", '"'}:
        return default[1:-1]
    return default


def _document_with_segments():
    document = make_document()
    section = document.sections[0]
    segmented = replace(section.blocks[0], text=TEXT, segments=SEGMENTS)
    section = replace(section, blocks=(segmented, section.blocks[1]))
    return replace(document, sections=(section,))


def _analysis_with_segment_treatments():
    analysis = make_analysis()
    mixed = TreatmentPlanItem(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        segment_treatments=(BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
        sentence_indexes=(0, 1),
    )
    plan = replace(analysis.treatment_plan, items=(mixed, analysis.treatment_plan.items[1]))
    return replace(analysis, treatment_plan=plan)


def _analysis_with_descriptions():
    analysis = make_analysis()
    described = replace(analysis.glossary_candidates[0], description="герой романа")
    blank = replace(
        analysis.glossary_candidates[0],
        id="candidate-2",
        source_text="Ministry of Truth",
        description="",
    )
    return replace(analysis, glossary_candidates=(described, blank))


def _segmented_result() -> BlockProcessingResult:
    return BlockProcessingResult(
        block_id=BLOCK_ID,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Альфа. Гамма.",
        segment_treatments=(
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
        segment_translations=("Альфа.", None, "Гамма."),
    )


def _insert_legacy_rows(connection: Connection) -> None:
    """Rows written at revision 0016, before segment columns exist."""
    connection.execute(
        text(
            "INSERT INTO books (id, title, authors_json, source_language, "
            "target_language, status, source_epub_path, created_at) "
            "VALUES ('book-1', 'Book', '[]', 'en', 'ru', 'imported', "
            "'source.epub', '2026-10-10 00:00:00')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO sections (id, book_id, title, \"order\", source_resource_path) "
            "VALUES ('section-1', 'book-1', 'Chapter 1', 0, 'OEBPS/chapter1.xhtml')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO document_blocks ("
            "id, book_id, section_id, type, \"order\", text, inline_annotations_json, "
            "attributes_json, source_reference_json, support_status"
            ") VALUES ("
            "'block-1', 'book-1', 'section-1', 'paragraph', 0, 'Alpha. Beta.', '[]', "
            "'{}', "
            "'{\"resource_path\":\"OEBPS/chapter1.xhtml\",\"structural_path\":\"/p[1]\"}', "
            "'supported'"
            ")"
        )
    )
    connection.execute(
        text(
            "INSERT INTO chapter_analyses ("
            "id, book_id, section_id, coverage_status, translation_instructions, "
            "chapter_summary, before_you_read_json"
            ") VALUES ("
            "'analysis-1', 'book-1', 'section-1', 'complete', '', '', '[]'"
            ")"
        )
    )
    connection.execute(
        text(
            "INSERT INTO treatment_plan_items (analysis_id, block_id, treatment, position) "
            "VALUES ('analysis-1', 'block-1', 'TRANSLATE', 0)"
        )
    )
    connection.execute(
        text(
            "INSERT INTO glossary_candidates ("
            "id, analysis_id, source_text, type, suggested_translation, confidence, "
            "model_flagged_review, requires_review, note"
            ") VALUES ("
            "'candidate-1', 'analysis-1', 'Winston Smith', 'person', 'Уинстон Смит', "
            "0.9, 0, 0, 'мужской род'"
            ")"
        )
    )
    connection.execute(
        text(
            "INSERT INTO processing_runs ("
            "id, book_id, status, glossary_snapshot_json, llm_settings_json, created_at"
            ") VALUES ("
            "'run-1', 'book-1', 'active', '[]', '{}', '2026-10-10 00:00:00'"
            ")"
        )
    )
    connection.execute(
        text(
            "INSERT INTO block_results (run_id, block_id, treatment, translated_text, "
            "glossary_hints_json) VALUES ("
            "'run-1', 'block-1', 'TRANSLATE', 'Был день.', '[]'"
            ")"
        )
    )


def test_block_segments_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """Stored spans round-trip; no spans are an empty JSON list, not NULL."""
    document = _document_with_segments()
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(document)

    with unit_of_work.transaction() as uow:
        stored = uow.documents.get(BOOK_ID)

    assert stored == document
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT id, segments_json FROM document_blocks ORDER BY \"order\"")
        ).all()
    assert _json_cell(rows[0].segments_json) == [[0, 6, 0], [6, len(TEXT), 1]]
    assert _json_cell(rows[1].segments_json) == []


def test_plan_item_segment_treatments_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """Segment modes round-trip as treatment strings; an empty tuple is SQL NULL."""
    analysis = _analysis_with_segment_treatments()
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, analysis)

    with unit_of_work.transaction() as uow:
        stored = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)

    assert stored == analysis
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT block_id, segment_treatments_json FROM treatment_plan_items "
                "ORDER BY position"
            )
        ).all()
    assert rows[0].block_id == BLOCK_ID
    assert _json_cell(rows[0].segment_treatments_json) == ["GLOSSARY_ONLY", "TRANSLATE"]
    assert rows[1].block_id == SECOND_BLOCK_ID
    assert rows[1].segment_treatments_json is None


def test_result_segment_translations_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """Per-segment modes and translations round-trip; an empty tuple is SQL NULL."""
    segmented = _segmented_result()
    plain = make_glossary_only_result()
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.runs.add(make_run())
        uow.results.replace_results(RUN_ID, [segmented, plain])

    with unit_of_work.transaction() as uow:
        stored = uow.results.get(RUN_ID, BLOCK_ID)
        plain_stored = uow.results.get(RUN_ID, SECOND_BLOCK_ID)

    assert stored == segmented
    assert plain_stored == plain
    with engine.connect() as connection:
        rows = {
            row.block_id: row
            for row in connection.execute(
                text(
                    "SELECT block_id, segment_treatments_json, segment_translations_json "
                    "FROM block_results"
                )
            )
        }
    assert _json_cell(rows[BLOCK_ID].segment_treatments_json) == [
        "TRANSLATE",
        "READ_AS_IS",
        "TRANSLATE",
    ]
    assert _json_cell(rows[BLOCK_ID].segment_translations_json) == [
        "Альфа.",
        None,
        "Гамма.",
    ]
    assert rows[SECOND_BLOCK_ID].segment_treatments_json is None
    assert rows[SECOND_BLOCK_ID].segment_translations_json is None


def test_candidate_description_round_trip(
    unit_of_work: SqlAlchemyUnitOfWork, engine: Engine
) -> None:
    """Candidate description round-trips, including the empty-string default."""
    analysis = _analysis_with_descriptions()
    with unit_of_work.transaction() as uow:
        uow.books.add(make_book())
        uow.documents.save(make_document())
        uow.analyses.save(BOOK_ID, analysis)

    with unit_of_work.transaction() as uow:
        stored = uow.analyses.get_for_section(BOOK_ID, SECTION_ID)

    assert stored == analysis
    assert stored is not None
    assert stored.glossary_candidates[0].description == "герой романа"
    assert stored.glossary_candidates[1].description == ""
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT id, description FROM glossary_candidates ORDER BY id")
        ).all()
    assert rows == [("candidate-1", "герой романа"), ("candidate-2", "")]


def test_upgrade_head_creates_segment_columns(alembic_config, engine: Engine) -> None:
    """Head adds the segment columns and does not copy existing values into them."""
    command.downgrade(alembic_config, "0016")
    with engine.begin() as connection:
        _insert_legacy_rows(connection)

    command.upgrade(alembic_config, "head")
    _assert_segment_columns(engine)
    _assert_legacy_rows_keep_defaults(engine)

    command.downgrade(alembic_config, "0016")
    _assert_segment_columns_absent(engine)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT text FROM document_blocks")).scalar_one() == (
            "Alpha. Beta."
        )

    command.upgrade(alembic_config, "head")
    _assert_segment_columns(engine)
    _assert_legacy_rows_keep_defaults(engine)


def _assert_segment_columns(engine: Engine) -> None:
    blocks = _columns(engine, "document_blocks")
    plans = _columns(engine, "treatment_plan_items")
    results = _columns(engine, "block_results")
    candidates = _columns(engine, "glossary_candidates")

    assert blocks["segments_json"]["nullable"] is False
    assert _literal_default(blocks["segments_json"]["default"]) == "[]"
    assert plans["segment_treatments_json"]["nullable"] is True
    assert plans["segment_treatments_json"]["default"] is None
    assert results["segment_treatments_json"]["nullable"] is True
    assert results["segment_treatments_json"]["default"] is None
    assert results["segment_translations_json"]["nullable"] is True
    assert results["segment_translations_json"]["default"] is None
    assert candidates["description"]["nullable"] is False
    assert _literal_default(candidates["description"]["default"]) == ""


def _assert_segment_columns_absent(engine: Engine) -> None:
    assert "segments_json" not in _columns(engine, "document_blocks")
    assert "segment_treatments_json" not in _columns(engine, "treatment_plan_items")
    assert "segment_treatments_json" not in _columns(engine, "block_results")
    assert "segment_translations_json" not in _columns(engine, "block_results")
    assert "description" not in _columns(engine, "glossary_candidates")


def _assert_legacy_rows_keep_defaults(engine: Engine) -> None:
    with engine.connect() as connection:
        block = connection.execute(
            text("SELECT text, segments_json FROM document_blocks")
        ).one()
        plan = connection.execute(
            text("SELECT treatment, segment_treatments_json FROM treatment_plan_items")
        ).one()
        candidate = connection.execute(
            text("SELECT source_text, note, description FROM glossary_candidates")
        ).one()
        result = connection.execute(
            text(
                "SELECT translated_text, segment_treatments_json, segment_translations_json "
                "FROM block_results"
            )
        ).one()
    assert block.text == "Alpha. Beta."
    assert _json_cell(block.segments_json) == []
    assert plan.treatment == "TRANSLATE"
    assert plan.segment_treatments_json is None
    assert candidate.source_text == "Winston Smith"
    assert candidate.note == "мужской род"
    assert candidate.description == ""
    assert result.translated_text == "Был день."
    assert result.segment_treatments_json is None
    assert result.segment_translations_json is None
