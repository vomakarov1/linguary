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


def test_help_for_list_items_stays_inside_the_list_item(tmp_path: Path) -> None:
    """Подсказки пункта списка не становятся соседями `li`: XHTML запрещает это."""
    source = build_epub(
        tmp_path / "source.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter One</h1>"
                "<ul>"
                "<li>a brass pocket compass that pointed roughly north</li>"
                "<li>a heel of dried bread wrapped in cloth</li>"
                "</ul>"
            )
        },
        toc=[("chapter1.xhtml", "Chapter One")],
    )
    dry = run_export(source, tmp_path=tmp_path / "dry", book_id="book-1")
    section = dry.imported.document.ordered_sections[0]
    first, _second = (
        block
        for block in section.ordered_blocks
        if block.type is BlockType.LIST_ITEM
    )
    results: Mapping[str, BlockProcessingResult] = {
        first.id: BlockProcessingResult(
            block_id=first.id,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint(
                    original="pointed roughly north",
                    translation="указывал примерно на север",
                    note="ироничная оговорка о ненадёжном компасе",
                ),
            ),
            language_notes=(
                LanguageNote(
                    id="note-1",
                    block_id=first.id,
                    original_excerpt="roughly north",
                    note="Roughly подчёркивает приблизительность.",
                ),
            ),
        )
    }
    run = run_export(
        source,
        tmp_path=tmp_path / "full",
        book_id="book-1",
        block_results=results,
        learning_materials={
            section.id: LearningMaterial.for_chapter(
                section_id=section.id,
                before_you_read=(),
                chapter_summary="",
                language_notes=results[first.id].language_notes,
            )
        },
    )
    chapter = documents_of(run.path)["OEBPS/chapter1.xhtml"]
    original_list = next(
        element for element in elements(chapter, "ul") if element.get("class") is None
    )
    first_item = next(iter(elements(original_list, "li")))

    assert [child.tag.rpartition("}")[2] for child in original_list] == ["li", "li"]
    assert next(iter(elements(first_item, "ul"))).get("class") == "linguary-glossary"
    assert next(iter(elements(first_item, "aside"))).get("class") == (
        "linguary-language-note"
    )


def test_footnote_reference_in_translation_stays_a_link(tmp_path: Path) -> None:
    """Ссылка на сноску в переводе остаётся ссылкой, а не голой цифрой текста (QA-08)."""
    source = build_epub(
        tmp_path / "source.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter One</h1>"
                '<p>Nobody liked her face.<a epub:type="noteref" href="#fn1" id="fnref1">1</a>'
                " It was a pity.</p>"
                '<aside epub:type="footnote" id="fn1"><p>A plain face. '
                '<a href="#fnref1">↩</a></p></aside>'
            )
        },
        toc=[("chapter1.xhtml", "Chapter One")],
    )
    dry = run_export(source, tmp_path=tmp_path / "dry", book_id="book-1")
    (paragraph,) = paragraphs_of(dry.imported.document, 0)
    results: Mapping[str, BlockProcessingResult] = {
        paragraph.id: BlockProcessingResult(
            block_id=paragraph.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Никому не нравилось её лицо.[^1] Жаль.",
        )
    }

    run = run_export(
        source, tmp_path=tmp_path / "full", book_id="book-1", block_results=results
    )

    chapter = documents_of(run.path)["OEBPS/chapter1.xhtml"]
    translation = next(
        element
        for element in elements(chapter, "p")
        if element.get("class") == "linguary-translation"
    )
    original = next(
        link
        for element in elements(chapter, "p")
        if element.get("class") is None and "her face" in "".join(element.itertext())
        for link in elements(element, "a")
    )
    (link,) = elements(translation, "a")
    assert "".join(translation.itertext()) == "Никому не нравилось её лицо.1 Жаль."
    assert translation.text == "Никому не нравилось её лицо."
    assert link.text == "1"
    assert link.get("href") == original.get("href")
    assert link.get("href").endswith("#fn1")
    assert link.get("{http://www.idpf.org/2007/ops}type") == "noteref"
    assert link.get("id") is None


@pytest.fixture
def book_with_notes_page(tmp_path: Path) -> Path:
    """Две главы, между которыми стоит страница, где одни сноски."""
    return build_epub(
        tmp_path / "notes-source.epub",
        documents={
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter One</h1>"
                '<p>Nobody liked her face.<a epub:type="noteref" href="notes.xhtml#fn1"'
                ' id="fnref1">1</a> It was a pity.</p>'
            ),
            "notes.xhtml": xhtml_document(
                "<h1>Notes</h1>"
                '<aside epub:type="footnote" id="fn1"><p>A plain face. '
                '<a href="chapter1.xhtml#fnref1">↩</a></p></aside>',
                title="Notes",
            ),
            "chapter2.xhtml": xhtml_document(
                "<h1>Chapter Two</h1><p>The town appeared at dusk.</p>"
            ),
        },
        toc=[
            ("chapter1.xhtml", "Chapter One"),
            ("notes.xhtml", "Notes"),
            ("chapter2.xhtml", "Chapter Two"),
        ],
    )


def test_notes_page_is_translated_but_not_dressed_as_a_chapter(
    book_with_notes_page: Path, tmp_path: Path
) -> None:
    """Страница сносок: перевод есть, пересказа, «Before You Read» и разметки главы нет (QA-04)."""
    dry = run_export(book_with_notes_page, tmp_path=tmp_path / "dry", book_id="book-1")
    document = dry.imported.document
    first, notes, second = document.ordered_sections
    footnote = next(
        block
        for block in notes.ordered_blocks
        if block.type is BlockType.FOOTNOTE and block.text
    )
    results: Mapping[str, BlockProcessingResult] = {
        footnote.id: BlockProcessingResult(
            block_id=footnote.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Некрасивое лицо.",
        )
    }
    materials = {
        section.id: LearningMaterial.for_chapter(
            section_id=section.id,
            before_you_read=(
                BeforeYouReadEntry(expression=f"idiom of {section.title}", explanation="оборот"),
            ),
            chapter_summary=f"Summary of {section.title}.",
            language_notes=(),
        )
        for section in (first, notes, second)
    }

    run = run_export(
        book_with_notes_page,
        tmp_path=tmp_path / "full",
        book_id="book-1",
        block_results=results,
        learning_materials=materials,
    )

    chapters = documents_of(run.path)
    notes_page = chapters["OEBPS/notes.xhtml"]
    notes_text = "".join(notes_page.itertext())
    assert not [
        element
        for element in elements(notes_page, "section")
        if element.get("{http://www.idpf.org/2007/ops}type") == "chapter"
    ]
    assert "Previously" not in notes_text
    assert "Before You Read" not in notes_text
    assert "Некрасивое лицо." in notes_text
    assert "A plain face." in notes_text
    following = "".join(chapters["OEBPS/chapter2.xhtml"].itertext())
    assert "Summary of Chapter One." in following
    assert "Summary of Notes." not in following
    assert "idiom of Chapter Two" in following


def test_front_matter_and_first_story_chapter_rules(tmp_path: Path) -> None:
    """Титул и оглавление не получают помощи, а первая сюжетная глава не получает previously."""
    epub_path = build_epub(
        tmp_path / "book_with_front_matter.epub",
        documents={
            "title.xhtml": xhtml_document(
                "<h1>Alice’s Adventures in Wonderland</h1>"
                "<h2>by Lewis Carroll</h2>"
                "<h2>Contents</h2>"
                "<table><tr><td>Chapter One</td></tr><tr><td>Chapter Two</td></tr></table>"
            ),
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter One</h1>"
                "<p>Alice was beginning to get very tired.</p>"
            ),
            "chapter2.xhtml": xhtml_document(
                "<h1>Chapter Two</h1>"
                "<p>Curiouser and curiouser cried Alice.</p>"
            ),
        },
        toc=[
            ("title.xhtml", "Contents"),
            ("chapter1.xhtml", "Chapter One"),
            ("chapter2.xhtml", "Chapter Two"),
        ],
        title="Alice’s Adventures in Wonderland",
    )

    dry = run_export(epub_path, tmp_path=tmp_path / "dry", book_id="book-front-matter")
    document = dry.imported.document

    title_sec, ch1_sec, ch2_sec = document.ordered_sections

    results = {}
    for block in title_sec.blocks:
        results[block.id] = BlockProcessingResult(
            block_id=block.id,
            treatment=BlockTreatment.TRANSLATE,
            translated_text="Перевод титула.",
            glossary_hints=(
                GlossaryHint(original="Alice", translation="Алиса"),
            ),
        )

    for block in ch1_sec.blocks:
        if block.type is BlockType.PARAGRAPH:
            results[block.id] = BlockProcessingResult(
                block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Алиса начала очень уставать.",
                glossary_hints=(
                    GlossaryHint(original="tired", translation="уставать"),
                ),
            )

    for block in ch2_sec.blocks:
        if block.type is BlockType.PARAGRAPH:
            results[block.id] = BlockProcessingResult(
                block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Все чудесатее и чудесатее вскричала Алиса.",
            )

    materials = {
        title_sec.id: LearningMaterial.for_chapter(
            section_id=title_sec.id,
            before_you_read=(),
            chapter_summary="This opening section contains only the title page.",
            language_notes=(),
        ),
        ch1_sec.id: LearningMaterial.for_chapter(
            section_id=ch1_sec.id,
            before_you_read=(
                BeforeYouReadEntry(
                    expression="out of the way",
                    explanation="Необычный, странный. Передавай как забавное искажение.",
                ),
            ),
            chapter_summary="Alice follows the White Rabbit down the rabbit-hole.",
            language_notes=(),
        ),
        ch2_sec.id: LearningMaterial.for_chapter(
            section_id=ch2_sec.id,
            before_you_read=(),
            chapter_summary="Alice grows very large in the hall.",
            language_notes=(),
        ),
    }

    run = run_export(
        epub_path,
        tmp_path=tmp_path / "full",
        book_id="book-front-matter",
        block_results=results,
        learning_materials=materials,
    )

    chapters = documents_of(run.path)

    # 1. Проверяем вводный раздел (титул / оглавление)
    title_doc = chapters["OEBPS/title.xhtml"]
    title_text = "".join(title_doc.itertext())
    assert "linguary-glossary" not in [c for _, c in [((el.tag, el.get("class")) if el is not None else ("", "")) for el in elements(title_doc, "*")]]
    assert "linguary-translation" not in [c for _, c in [((el.tag, el.get("class")) if el is not None else ("", "")) for el in elements(title_doc, "*")]]
    assert "Перевод титула." not in title_text
    assert "Алиса" not in title_text

    # 2. Проверяем первую сюжетную главу (Chapter 1)
    ch1_doc = chapters["OEBPS/chapter1.xhtml"]
    ch1_text = "".join(ch1_doc.itertext())
    ch1_classes = [el.get("class") for el in elements(ch1_doc, "section") if el.get("class")]
    assert "linguary-previously" not in ch1_classes
    assert "Previously" not in ch1_text
    assert "title page" not in ch1_text
    # Санитайзер Before You Read очистил мета-инструкцию в карточке
    assert "Необычный, странный." in ch1_text
    assert "Передавай как" not in ch1_text

    # 3. Проверяем вторую главу (Chapter 2) — содержит previously из главы 1
    ch2_doc = chapters["OEBPS/chapter2.xhtml"]
    ch2_text = "".join(ch2_doc.itertext())
    ch2_classes = [el.get("class") for el in elements(ch2_doc, "section") if el.get("class")]
    assert "linguary-previously" in ch2_classes
    assert "Previously in Chapter One" in ch2_text
    assert "Alice follows the White Rabbit" in ch2_text

