"""Сквозное тестирование конвейера деления составных абзацев и валидации EPUB (P3.T2)."""

from __future__ import annotations

import sys
from pathlib import Path

_EPUB_TEST_DIR = Path(__file__).resolve().parent / "epub"
if str(_EPUB_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_EPUB_TEST_DIR))

from epub_builder import build_epub, xhtml_document
from export_pipeline import documents_of, elements, run_export

from linguary.adapters.epub import verify_exported_epub
from linguary.adapters.validation import EpubCheckValidator
from linguary.application.ports.export_validation import (
    ExportIssueSeverity,
    ExportValidationRequest,
)
from linguary.application.translation_units import plan_translation_units
from linguary.domain import BlockProcessingResult, BlockTreatment, TreatmentPlan


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
        f"<p id=\"composite-p\">{composite_paragraph}</p>\n"
        f"<p>Closing short paragraph after the courtroom scene.</p>\n",
        title="Chapter 1",
    )

    epub_path = tmp_path / "source_book.epub"
    return build_epub(
        epub_path,
        documents={"chapter1.xhtml": html},
        toc=[("chapter1.xhtml", "Chapter 1")],
    )


def test_pipeline_import_process_and_export_split_paragraphs(tmp_path: Path) -> None:
    """Полный цикл импорта книги с составными абзацами, группировки обработки и экспорта в EPUB."""
    source_epub = _create_test_epub_with_composite_paragraphs(tmp_path)

    run = run_export(source_epub, tmp_path=tmp_path, book_id="split-book")
    assert run.path.is_file()

    sections = run.imported.document.ordered_sections
    assert len(sections) == 1
    blocks = sections[0].ordered_blocks

    split_blocks = [b for b in blocks if "split_group" in b.attributes]
    assert len(split_blocks) >= 3

    base_path = split_blocks[0].attributes["split_group"]
    total = split_blocks[0].attributes["split_total"]
    assert total == str(len(split_blocks))

    for k, block in enumerate(split_blocks, start=1):
        assert block.attributes["split_group"] == base_path
        assert block.attributes["split_index"] == str(k)
        assert block.attributes["split_total"] == total
        assert block.source_reference.structural_path.endswith(f"/text()[{k}]")

    plan = TreatmentPlan.of_treatments(
        sections[0].id,
        [(block.id, BlockTreatment.TRANSLATE) for block in blocks],
    )
    import itertools
    counter = itertools.count(1)
    units = plan_translation_units(
        blocks,
        plan,
        id_factory=lambda: f"unit-{next(counter)}",
        max_source_chars=2000,
    )

    split_block_ids = {b.id for b in split_blocks}
    containing_units = [u for u in units if any(bid in split_block_ids for bid in u.block_ids)]
    assert len(containing_units) == 1, "Все фрагменты составного абзаца должны попасть в один TranslationUnit"
    assert all(bid in containing_units[0].block_ids for bid in split_block_ids)

    assert verify_exported_epub(run.path) == ()

    exported_docs = documents_of(run.path)
    assert len(exported_docs) >= 1
    chapter_doc = next(iter(exported_docs.values()))

    p_elements = elements(chapter_doc, "p")
    continuation_elements = [
        p for p in p_elements if "linguary-split-continuation" in (p.get("class") or "")
    ]
    assert len(continuation_elements) == len(split_blocks) - 1

    first_split_p = [
        p for p in p_elements if split_blocks[0].text[:30] in (p.text or "")
    ]
    assert len(first_split_p) == 1
    assert "linguary-split-continuation" not in (first_split_p[0].get("class") or "")


def test_epubcheck_validity_for_split_paragraphs(tmp_path: Path) -> None:
    """Экспортированный EPUB со стилизованными фрагментами абзацев валиден по стандарту EPUB 3 (0 ошибок в EPUBCheck)."""
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
