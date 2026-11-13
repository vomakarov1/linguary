"""End-to-end testing of composite paragraph splitting pipeline and EPUB validation (P3.T2)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

_EPUB_TEST_DIR = Path(__file__).resolve().parent / "epub"
if str(_EPUB_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_EPUB_TEST_DIR))

from epub_builder import build_epub, xhtml_document
from export_pipeline import documents_of, elements, run_export

from linguary.adapters.epub import verify_exported_epub
from linguary.adapters.epub.splitting import segment_text
from linguary.adapters.validation import EpubCheckValidator
from linguary.application.ports.export_validation import (
    ExportIssueSeverity,
    ExportValidationRequest,
)
from linguary.application.translation_units import plan_translation_units
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    InlineAnnotationType,
    TreatmentPlan,
)
from linguary.domain.identifiers import build_block_id


def _create_test_epub_with_composite_paragraphs(tmp_path: Path) -> Path:
    s1 = (
        "The White Rabbit put on his large spectacles and prepared to read the official accusation "
        "before the King and Queen with extraordinary precision, holding the long royal scroll "
        "of ancient parchment carefully in both of his trembling paws while the entire court waited "
        "in complete and respectful silence for him to utter the opening words. "
    )
    s2 = (
        "Where shall I begin, please your Majesty, he asked in a tiny timid voice that squeaked "
        "faintly through the great hall, looking nervously across the crowded courtroom at the twelve "
        "jurymen who were already writing down their initial thoughts and secret opinions upon their "
        "wooden slates with great haste and squeaky chalk. "
    )
    s3 = (
        "Begin at the very beginning, the King said gravely in a loud booming tone that resonated "
        "from the royal throne, and continue steadily onward without pause till you come to the "
        "ultimate end of the whole proclamation: then stop immediately and sit down."
    )
    composite_paragraph = s1 + s2 + s3
    assert len(composite_paragraph) >= 800

    html = xhtml_document(
        f"<h1>Chapter I</h1>\n"
        f"<p>Opening short paragraph before the courtroom scene.</p>\n"
        f"<p id=\"composite-p\"><b>{s1}</b>{s2}{s3}</p>\n"
        f"<p>Closing short paragraph after the courtroom scene.</p>\n",
        title="Chapter 1",
    )

    epub_path = tmp_path / "source_book.epub"
    return build_epub(
        epub_path,
        documents={"chapter1.xhtml": html},
        toc=[("chapter1.xhtml", "Chapter 1")],
    )


def test_epubcheck_validity_for_split_paragraphs(tmp_path: Path) -> None:
    """An exported long paragraph, stored as one block, is valid EPUB 3."""
    source_epub = _create_test_epub_with_composite_paragraphs(tmp_path)

    run_temp = run_export(source_epub, tmp_path=tmp_path, book_id="validation-book")
    blocks = list(run_temp.imported.document.blocks())
    results = {
        block.id: BlockProcessingResult(
            block_id=block.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text=f"Перевод: {block.text[:50]}...",
        )
        for block in blocks
    }

    run = run_export(
        source_epub,
        tmp_path=tmp_path,
        book_id="validated-split-book",
        block_results=results,
    )

    validator = EpubCheckValidator()
    report = validator.validate(
        ExportValidationRequest(
            epub_path=str(run.path),
            report_path=str(run.path.with_suffix(".epubcheck.json")),
        )
    )

    if report.available:
        assert len(report.errors) == 0, f"EPUBCheck errors: {report.errors}"


def test_pipeline_imports_single_block_with_segments(tmp_path: Path) -> None:
    """A long paragraph is imported, planned, and exported as one block with segments."""
    source_epub = _create_test_epub_with_composite_paragraphs(tmp_path)
    run = run_export(source_epub, tmp_path=tmp_path, book_id="split-book")
    assert run.path.is_file()

    sections = run.imported.document.ordered_sections
    assert len(sections) == 1
    blocks = sections[0].ordered_blocks
    assert [block.type for block in blocks] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.PARAGRAPH,
        BlockType.PARAGRAPH,
    ]
    assert blocks[1].text == "Opening short paragraph before the courtroom scene."
    assert blocks[3].text == "Closing short paragraph after the courtroom scene."

    composite = blocks[2]
    assert composite.source_reference.element_id == "composite-p"
    assert composite.source_reference.structural_path == "/html[1]/body[1]/p[2]"
    assert "/text()" not in composite.source_reference.structural_path
    assert composite.id == build_block_id(
        document_id="split-book",
        source_resource_path="chapter1.xhtml",
        structural_path="/html[1]/body[1]/p[2]",
    )
    assert composite.text.endswith("sit down.")
    assert "opening words." in composite.text
    assert "squeaky chalk." in composite.text
    expected = segment_text(composite.text)
    assert len(expected) > 1
    assert composite.segments == expected
    assert all(
        "split_group" not in block.attributes
        and "split_index" not in block.attributes
        and "split_total" not in block.attributes
        for block in blocks
    )

    plan = TreatmentPlan.of_treatments(
        sections[0].id,
        [(block.id, BlockTreatment.TRANSLATE) for block in blocks],
    )
    counter = itertools.count(1)
    units = plan_translation_units(
        blocks,
        plan,
        id_factory=lambda: f"unit-{next(counter)}",
        max_source_chars=2000,
    )
    containing_units = [unit for unit in units if composite.id in unit.block_ids]
    assert len(containing_units) == 1

    assert verify_exported_epub(run.path) == ()
    chapter_doc = next(iter(documents_of(run.path).values()))
    assert [
        paragraph
        for paragraph in elements(chapter_doc, "p")
        if "linguary-split-continuation" in (paragraph.get("class") or "")
    ] == []
    assert [
        node
        for node in elements(chapter_doc, "div")
        if "linguary-paragraph-fragment" in (node.get("class") or "").split()
    ] == []
    originals = [
        paragraph
        for paragraph in elements(chapter_doc, "p")
        if "The White Rabbit" in "".join(paragraph.itertext())
    ]
    assert len(originals) == 1
    assert "".join(originals[0].itertext()) == composite.text
    bold = elements(originals[0], "b")
    expected_bold = "".join(
        composite.text[annotation.start : annotation.end]
        for annotation in composite.inline_annotations
        if annotation.type is InlineAnnotationType.BOLD
    )
    assert bold
    assert "".join("".join(node.itertext()) for node in bold) == expected_bold
    assert expected_bold.startswith("The White Rabbit")
