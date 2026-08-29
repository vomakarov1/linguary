"""Экспорт EPUB: состав главы в полном режиме (P2.T5).

Полный режим опирается на данные, которые появляются только в фазе 3, поэтому результаты
обработки и учебные материалы здесь собираются вручную: LLM в сборке EPUB не участвует
и участвовать не должен (PRD § 20).
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document
from export_pipeline import documents_of, elements, run_export

from linguary.domain import (
    BeforeYouReadEntry,
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    CanonicalDocument,
    DocumentBlock,
    GlossaryHint,
    LanguageNote,
    LearningMaterial,
)

_SUMMARY = "Wren reaches Duskmere and learns that the marsh lights mislead travellers."


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Книга из двух глав: три абзаца с оригинальной сноской и продолжение."""
    return build_epub(
        tmp_path / "source.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter One</h1>"
                "<p>Wren Calder had walked for three days.</p>"
                "<p>The cart driver spoke of marsh lights.</p>"
                "<p>Wren walked on regardless.</p>"
                '<aside epub:type="footnote" id="fn1"><p>A starglass is a lens.</p></aside>'
            ),
            "chapter2.xhtml": xhtml_document(
                "<h1>Chapter Two</h1><p>The town appeared at dusk.</p>"
            ),
        },
        toc=[("chapter1.xhtml", "Chapter One"), ("chapter2.xhtml", "Chapter Two")],
    )


def paragraphs_of(document: CanonicalDocument, section_index: int) -> tuple[DocumentBlock, ...]:
    """Текстовые абзацы главы в порядке чтения."""
    section = document.ordered_sections[section_index]
    return tuple(
        block for block in section.ordered_blocks if block.type is BlockType.PARAGRAPH
    )


def outline(root) -> list[tuple[str, str]]:
    """Содержимое главы как последовательность пар «тег, класс»."""
    chapter = next(iter(elements(root, "section")))
    return [
        (element.tag.rpartition("}")[2], element.get("class", ""))
        for element in chapter
    ]


def export_with_results(source: Path, tmp_path: Path) -> tuple[Path, CanonicalDocument]:
    """Собирает книгу со всеми слоями помощи первой главы.

    Идентификаторы блоков и разделов считаются от идентификатора книги, поэтому обе сборки
    идут под одним `book_id`: иначе результаты обработки не нашли бы своих блоков.
    """
    dry = run_export(source, tmp_path=tmp_path / "dry", book_id="book-1")
    document = dry.imported.document
    first, second, third = paragraphs_of(document, 0)
    results: Mapping[str, BlockProcessingResult] = {
        first.id: BlockProcessingResult(
            block_id=first.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Рен Колдер шла три дня.",
            language_notes=(
                LanguageNote(
                    id="note-1",
                    block_id=first.id,
                    original_excerpt="had walked for three days",
                    note="Past perfect подчёркивает, что путь уже позади.",
                ),
            ),
        ),
        second.id: BlockProcessingResult(
            block_id=second.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint(
                    original="marsh lights",
                    translation="болотные огни",
                    note="блуждающие огни на болоте",
                ),
            ),
        ),
        third.id: BlockProcessingResult(
            block_id=third.id, treatment=BlockTreatment.READ_AS_IS
        ),
    }
    materials = {
        document.ordered_sections[0].id: LearningMaterial.for_chapter(
            section_id=document.ordered_sections[0].id,
            before_you_read=(
                BeforeYouReadEntry(
                    expression="to walk on regardless",
                    explanation="идти дальше, несмотря ни на что",
                ),
            ),
            chapter_summary=_SUMMARY,
            language_notes=results[first.id].language_notes,
        ),
        document.ordered_sections[1].id: LearningMaterial.for_chapter(
            section_id=document.ordered_sections[1].id,
            before_you_read=(),
            chapter_summary="",
            language_notes=(),
        ),
    }
    run = run_export(
        source,
        tmp_path=tmp_path / "full",
        book_id="book-1",
        block_results=results,
        learning_materials=materials,
    )
    return run.path, document


def test_chapter_follows_the_order_required_by_the_specification(
    source: Path, tmp_path: Path
) -> None:
    """Глава идёт по PRD § 20: оригинал, перевод или подсказки, затем заметки."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]

    assert outline(chapter) == [
        ("section", "linguary-before-you-read"),
        ("h1", ""),
        ("p", ""),
        ("p", "linguary-translation"),
        ("aside", "linguary-language-note"),
        ("p", ""),
        ("ul", "linguary-glossary"),
        ("p", ""),
        ("aside", ""),
    ]


def test_first_chapter_has_no_previously_section(source: Path, tmp_path: Path) -> None:
    """Перед первой главой раздела о предыдущей нет: её не существует."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]

    assert "linguary-previously" not in [class_ for _, class_ in outline(chapter)]


def test_second_chapter_opens_with_the_summary_of_the_first(
    source: Path, tmp_path: Path
) -> None:
    """Глава начинается разделом, собранным из сводки предыдущей главы (PRD § 19.2)."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter2.xhtml"]
    previously = next(
        element
        for element in elements(chapter, "section")
        if element.get("class") == "linguary-previously"
    )

    assert outline(chapter)[0] == ("section", "linguary-previously")
    assert next(iter(elements(previously, "h2"))).text == "Previously in Chapter One"
    assert next(iter(elements(previously, "p"))).text == _SUMMARY


def test_language_note_identifiers_do_not_touch_original_footnotes(
    source: Path, tmp_path: Path
) -> None:
    """Заметки живут в своём пространстве имён `ln-{section_key}-{nnn}` (PRD § 19.3)."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]
    notes = [
        element
        for element in elements(chapter, "aside")
        if element.get("class") == "linguary-language-note"
    ]
    footnotes = [
        element.get("id")
        for element in elements(chapter, "aside")
        if element.get("class") is None
    ]

    assert [element.get("id") for element in notes] == ["ln-ch01-001"]
    assert footnotes == ["fn1"]


def test_translation_carries_the_target_language(source: Path, tmp_path: Path) -> None:
    """У перевода объявлен целевой язык: читалка не считает его продолжением оригинала."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]
    translation = next(
        element
        for element in elements(chapter, "p")
        if element.get("class") == "linguary-translation"
    )

    assert translation.get("lang") == "ru"
    assert translation.text == "Рен Колдер шла три дня."


def test_help_layers_have_their_own_stylesheet(source: Path, tmp_path: Path) -> None:
    """Оформление слоёв помощи вынесено в отдельный стиль книги (PRD § 20)."""
    path, _ = export_with_results(source, tmp_path)

    with zipfile.ZipFile(path) as archive:
        stylesheet = archive.read("OEBPS/linguary.css").decode("utf-8")

    assert ".linguary-translation" in stylesheet
    assert ".linguary-glossary" in stylesheet
    assert ".linguary-language-note" in stylesheet


def test_glossary_hints_keep_original_and_translation(
    source: Path, tmp_path: Path
) -> None:
    """Словарная подсказка показывает и оригинал выражения, и его перевод."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]
    hints = next(
        element
        for element in elements(chapter, "ul")
        if element.get("class") == "linguary-glossary"
    )
    item = next(iter(elements(hints, "li")))

    assert "".join(item.itertext()) == "marsh lights — болотные огни (блуждающие огни на болоте)"
