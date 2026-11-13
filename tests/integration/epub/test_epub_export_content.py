"""EPUB export: chapter structure in full mode (P2.T5).

Full mode relies on data generated during phase 3, so processing results and learning
materials are assembled manually here: LLM does not participate in EPUB generation
and must not participate (PRD § 20).
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
from epub_builder import build_epub, xhtml_document
from lxml import etree
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
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.learning import FragmentAssistance

_SUMMARY = "Wren reaches Duskmere and learns that the marsh lights mislead travellers."


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A book with two chapters: three paragraphs with an original footnote and continuation."""
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
    """Text paragraphs of the chapter in reading order."""
    section = document.ordered_sections[section_index]
    return tuple(
        block for block in section.ordered_blocks if block.type is BlockType.PARAGRAPH
    )


def outline(root) -> list[tuple[str, str]]:
    """Chapter contents represented as a sequence of (tag, class) pairs."""
    chapter = next(iter(elements(root, "section")))
    return [
        (element.tag.rpartition("}")[2], element.get("class", ""))
        for element in chapter
    ]


def _plain(node) -> str:
    """Element text without marker digits."""
    chunks: list[str] = []

    def walk(element) -> None:
        if element.tag.rpartition("}")[2] == "sup":
            return
        if element.text:
            chunks.append(element.text)
        for child in element:
            walk(child)
            if child.tail:
                chunks.append(child.tail)

    walk(node)
    return "".join(chunks)


def export_with_results(
    source: Path, tmp_path: Path, *, chapter_summary: str = _SUMMARY
) -> tuple[Path, CanonicalDocument]:
    """Builds a book with all help layers for the first chapter.

    Block and section IDs are derived from the book ID, so both builds run under the
    same `book_id`, otherwise processing results would fail to match their blocks.
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
            chapter_summary=chapter_summary,
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
    """Chapter follows PRD § 20: original, translation or hints, then notes."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]

    assert outline(chapter) == [
        ("section", "linguary-before-you-read"),
        ("div", "lg-unit lg-end"),
        ("div", "lg-unit lg-end"),
        ("div", "lg-unit lg-end"),
        ("div", "lg-unit lg-end"),
        ("aside", ""),
    ]
    units = [element for element in next(iter(elements(chapter, "section"))) if element.tag.rpartition("}")[2] == "div"]
    heading, first, second, third = units
    assert [child.get("class") for child in heading] == ["lg-src"]
    assert [child.get("class") for child in first] == ["lg-src", "lg-tr", "lg-notes"]
    assert [child.get("class") for child in second] == ["lg-src", "lg-notes"]
    assert [child.get("class") for child in third] == ["lg-src"]
    assert "Past perfect подчёркивает, что путь уже позади." in "".join(first.itertext())
    assert "marsh lights" in "".join(second.itertext())
    assert "болотные огни" in "".join(second.itertext())


def test_first_chapter_has_no_previously_section(source: Path, tmp_path: Path) -> None:
    """Before the first chapter, there is no previously section: no prior chapter exists."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]

    assert "linguary-previously" not in [class_ for _, class_ in outline(chapter)]


def test_second_chapter_opens_with_the_summary_of_the_first(
    source: Path, tmp_path: Path
) -> None:
    """Chapter starts with a section built from the previous chapter summary (PRD § 19.2)."""
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


def test_previously_styles_russian_parallel_summary_paragraph(
    source: Path, tmp_path: Path
) -> None:
    russian_summary = "Врен добирается до Даскмера и узнаёт, что болотные огни сбивают путников с пути."
    path, _ = export_with_results(
        source, tmp_path, chapter_summary=f"{_SUMMARY}\n\n{russian_summary}"
    )
    chapter = documents_of(path)["OEBPS/chapter2.xhtml"]
    previously = next(
        element
        for element in elements(chapter, "section")
        if element.get("class") == "linguary-previously"
    )
    paragraphs = tuple(elements(previously, "p"))

    assert [paragraph.text for paragraph in paragraphs] == [_SUMMARY, russian_summary]
    assert paragraphs[0].get("class") is None
    assert paragraphs[1].get("class") == "linguary-summary-translation"
    assert paragraphs[1].get("lang") == "ru"
    with zipfile.ZipFile(path) as archive:
        stylesheet = archive.read("OEBPS/linguary.css").decode("utf-8")
    assert ".linguary-summary-translation" in stylesheet


def test_language_note_identifiers_do_not_touch_original_footnotes(
    source: Path, tmp_path: Path
) -> None:
    """Notes live in their own namespace `ln-{section_key}-{nnn}` (PRD § 19.3)."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]
    notes = [
        element
        for element in elements(chapter, "li")
        if "Past perfect подчёркивает, что путь уже позади." in "".join(element.itertext())
    ]
    footnotes = [
        element.get("id")
        for element in elements(chapter, "aside")
        if element.get("{http://www.idpf.org/2007/ops}type") == "footnote"
    ]

    assert [element.get("id") for element in notes] == ["ln-ch01-001"]
    assert footnotes == ["fn1"]


def test_translation_carries_the_target_language(source: Path, tmp_path: Path) -> None:
    """Translation has target language attribute: reader does not treat it as original."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]
    translation = next(
        element
        for element in elements(chapter, "p")
        if element.get("class") == "lg-tr"
    )

    assert translation.get("lang") == "ru"
    assert translation.text == "Рен Колдер шла три дня."


def test_help_layers_have_their_own_stylesheet(source: Path, tmp_path: Path) -> None:
    """Styling of help layers is isolated in a separate book stylesheet (PRD § 20)."""
    path, _ = export_with_results(source, tmp_path)

    with zipfile.ZipFile(path) as archive:
        stylesheet = archive.read("OEBPS/linguary.css").decode("utf-8")

    assert ".lg-tr" in stylesheet
    assert ".lg-notes" in stylesheet
    assert ".lg-unit" in stylesheet


def test_glossary_hints_keep_original_and_translation(
    source: Path, tmp_path: Path
) -> None:
    """Glossary hint displays both the original phrase and its translation."""
    path, _ = export_with_results(source, tmp_path)
    chapter = documents_of(path)["OEBPS/chapter1.xhtml"]
    item = next(
        element
        for element in elements(chapter, "li")
        if any(
            child.get("class") == "lg-term" and child.text == "marsh lights"
            for child in elements(element, "span")
        )
    )
    text = "".join(item.itertext())

    assert "marsh lights" in text
    assert "болотные огни" in text
    assert "блуждающие огни на болоте" in text


def test_help_for_list_items_stays_inside_the_list_item(tmp_path: Path) -> None:
    """Hints for list items stay inside `li`: XHTML disallows sibling placement."""
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
    notes = next(
        element
        for element in elements(first_item, "ol")
        if "lg-notes" in (element.get("class") or "").split()
    )
    text = "".join(notes.itertext())
    assert "pointed roughly north" in text
    assert "указывал примерно на север" in text
    assert "Roughly подчёркивает приблизительность." in text
    assert first_item in notes.iterancestors()


def test_footnote_reference_in_translation_stays_a_link(tmp_path: Path) -> None:
    """Footnote reference in translation remains a link rather than bare text numeral (QA-08)."""
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
        if element.get("class") == "lg-tr"
    )
    original = next(
        link
        for element in elements(chapter, "p")
        if "lg-src" in (element.get("class") or "").split()
        and "her face" in "".join(element.itertext())
        for link in elements(element, "a")
        if link.get("{http://www.idpf.org/2007/ops}type") == "noteref"
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
    """Two chapters with a dedicated footnotes page standing between them."""
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
    """Notes page: translated, but without recap, Before You Read, or chapter markup (QA-04)."""
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
    """Title and contents do not receive help, and the first story chapter gets no previously section."""
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

    ch1_para = next(b for b in ch1_sec.blocks if b.type is BlockType.PARAGRAPH)
    for block in ch1_sec.blocks:
        if block.type is BlockType.PARAGRAPH:
            results[block.id] = BlockProcessingResult(
                block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Алиса начала очень уставать.",
                glossary_hints=(
                    GlossaryHint(original="tired", translation="уставать"),
                ),
                language_notes=(
                    LanguageNote(
                        id="note-1",
                        block_id=block.id,
                        original_excerpt="tired",
                        note="Разговорная заметка.",
                    ),
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
            language_notes=results[ch1_para.id].language_notes,
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

    # 1. Verify introductory section (title / contents)
    title_doc = chapters["OEBPS/title.xhtml"]
    title_text = "".join(title_doc.itertext())
    title_classes = [(el.get("class") or "").split() for el in elements(title_doc, "*")]
    assert not any("lg-tr" in classes or "lg-notes" in classes for classes in title_classes)
    assert "Перевод титула." not in title_text
    assert "Алиса" not in title_text

    # 2. Verify first story chapter (Chapter 1)
    ch1_doc = chapters["OEBPS/chapter1.xhtml"]
    ch1_text = "".join(ch1_doc.itertext())
    ch1_classes = [el.get("class") for el in elements(ch1_doc, "section") if el.get("class")]
    assert "linguary-previously" not in ch1_classes
    assert "Previously" not in ch1_text
    assert "title page" not in ch1_text
    # Before You Read sanitizer stripped meta-instruction in card
    assert "Необычный, странный." in ch1_text
    assert "Передавай как" not in ch1_text
    # Chapter 1 notes keep the ln-ch01 namespace (front matter does not shift it).
    note_ids = [
        el.get("id")
        for el in elements(ch1_doc, "li")
        if (el.get("id") or "").startswith("ln-")
    ]
    assert note_ids == ["ln-ch01-001", "ln-ch01-002"]
    language = next(
        el for el in elements(ch1_doc, "li") if "Разговорная заметка." in "".join(el.itertext())
    )
    assert language.get("id") == "ln-ch01-002"
    assert not elements(ch1_doc, "aside")

    # 3. Verify second story chapter (Chapter 2) — contains previously from Chapter 1
    ch2_doc = chapters["OEBPS/chapter2.xhtml"]
    ch2_text = "".join(ch2_doc.itertext())
    ch2_classes = [el.get("class") for el in elements(ch2_doc, "section") if el.get("class")]
    assert "linguary-previously" in ch2_classes
    assert "Previously in Chapter One" in ch2_text
    assert "Alice follows the White Rabbit" in ch2_text



def test_fragment_assistance_follows_the_complete_split_original(tmp_path: Path) -> None:
    source = build_epub(
        tmp_path / "split-source.epub",
        documents={"chapter.xhtml": xhtml_document(
            '<h1>Chapter One</h1><p id="original"><b>Repeated phrase.</b> '
            + 'The traveller walked along the winding road. ' * 300
            + '<a href="https://example.com">linked ending</a>.</p>'
        )},
        toc=[("chapter.xhtml", "Chapter One")],
    )
    dry = run_export(source, tmp_path=tmp_path / "dry")
    parts = paragraphs_of(dry.imported.document, 0)
    assert len(parts) == 1
    part = parts[0]
    assert "split_group" not in part.attributes
    assert len(part.segment_spans) > 1
    assert "".join(part.text[span.start:span.end] for span in part.segment_spans) == part.text
    anchors = (SourceAnchor(part.id, 0, 10, part.text[:10]),)
    help_item = FragmentAssistance(
        "one-difficulty", ("one-point",), "sentence_rephrase",
        "The traveller continued along the road.", anchors,
    )
    result = BlockProcessingResult(
        part.id, BlockTreatment.READ_AS_IS, fragment_assistances=(help_item,),
    )
    run = run_export(source, tmp_path=tmp_path / "full", block_results={part.id: result})
    root = documents_of(run.path)["OEBPS/chapter.xhtml"]
    serialized = etree.tostring(root, encoding=str)
    assert "linguary-paragraph-fragment" not in serialized
    assert "linguary-fragment-help" not in serialized
    assert "linguary-help-reference" not in serialized
    sources = [
        paragraph
        for paragraph in elements(root, "p")
        if "lg-src" in (paragraph.get("class") or "").split()
    ]
    assert "".join(_plain(paragraph) for paragraph in sources) == "Chapter One" + part.text
    assert "".join(_plain(node) for node in elements(root, "b")) == "Repeated phrase."
    source_links = [
        link for link in elements(root, "a") if link.get("href") == "https://example.com"
    ]
    assert "".join(link.text or "" for link in source_links) == "linked ending"
    help_items = [
        item
        for item in elements(root, "li")
        if "The traveller continued along the road." in "".join(item.itertext())
    ]
    assert len(help_items) == 1
    markers = [
        link
        for link in elements(root, "a")
        if "lg-ref" in (link.get("class") or "").split()
    ]
    assert len(markers) == 1
    assert markers[0].get("href") == f"#{help_items[0].get('id')}"
    assert "Помощь пока не подготовлена." not in serialized
    assert "↩" not in serialized


def pronunciation_book(tmp_path):
    """Two sections, repeated invented form, second reading and unconfirmed name."""
    from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence
    source = build_epub(tmp_path / 'pronunciation.epub', documents={
        'chapter1.xhtml': xhtml_document('<h1>Chapter One</h1><p>A <em>zorp</em>.</p>'
            '<p id="linguary-pronunciation">An existing anchor.</p>'),
        'chapter2.xhtml': xhtml_document('<h1>Chapter Two</h1><p>zorp again.</p>'
            '<p>Yesterday zorp.</p><p>Unconfirmed Xyzzy.</p>'),
    }, toc=[('chapter1.xhtml', 'Chapter One'), ('chapter2.xhtml', 'Chapter Two')])
    document = run_export(source, tmp_path=tmp_path / 'dry').imported.document
    results = {}
    for section in document.ordered_sections:
        for block in section.ordered_blocks:
            if 'zorp' not in block.text:
                continue
            ipa = 'zɛp' if block.text.startswith('Yesterday') else 'zɔːp'
            start = block.text.index('zorp')
            item = PronunciationAssistance(SourceAnchor(block.id, start, start + 4, 'zorp'),
                'zorp', ipa, 'uk', 'ph-past' if ipa == 'zɛp' else 'ph-present',
                PronunciationEvidence('synthetic-private', 'test-1', 'entry-' + ipa, 'private-sense',
                    'verb', 'zorp', ipa, 'Invented test data'))
            results[block.id] = BlockProcessingResult(block_id=block.id, treatment=BlockTreatment.READ_AS_IS,
                                                     pronunciation_assistances=(item,))
    return source, document, results


def pronunciation_export(tmp_path):
    source, document, results = pronunciation_book(tmp_path)
    run = run_export(source, tmp_path=tmp_path / 'result', block_results=dict(reversed(list(results.items()))))
    return run, document, results


def pronunciation_nodes(path, tag, css_class):
    return [(name, node) for name, root in documents_of(path).items() for node in elements(root, tag)
            if css_class in node.get('class', '').split()]


def test_pronunciation_epub_primary_and_repeat_links(tmp_path):
    """Each reading keeps its own note in the same chapter. Repeats do not point at chapter 1."""
    run, _, _ = pronunciation_export(tmp_path)
    notes = pronunciation_nodes(run.path, 'span', 'lg-ipa')
    references = pronunciation_nodes(run.path, 'a', 'lg-ref')
    assert len(notes) == 3 and len(references) == 3
    assert sorted(node.text for _, node in notes) == ['/zɔːp/', '/zɔːp/', '/zɛp/']
    assert all(node.get('href', '').startswith('#') for _, node in references)
    chapter2 = [node for name, node in references if name.endswith('chapter2.xhtml')]
    assert len(chapter2) == 2
    assert all(not node.get('href', '').startswith('chapter1.xhtml') for node in chapter2)
    assert all(node.get('{http://www.idpf.org/2007/ops}type') != 'footnote' for _, node in references)


def test_pronunciation_epub_matches_web_projection(tmp_path):
    from linguary.application.pronunciation_projection import project_pronunciations
    run, document, results = pronunciation_export(tmp_path)
    projected = project_pronunciations(tuple(b for s in document.ordered_sections for b in s.ordered_blocks), results)
    notes = pronunciation_nodes(run.path, 'span', 'lg-ipa')
    references = pronunciation_nodes(run.path, 'a', 'lg-ref')
    assert len(notes) == len(references) == len(projected)
    assert sorted(node.text for _, node in notes) == sorted(f'/{item.ipa.strip("/")}/' for item in projected)
    assert all(item.variant.upper() in (node.tail or '') for item in projected for _, node in notes if item.ipa in (node.text or ''))
    assert all('private' not in ''.join(root.itertext()) for root in documents_of(run.path).values())


def test_pronunciation_epub_backlinks(tmp_path):
    run, _, _ = pronunciation_export(tmp_path)
    references = pronunciation_nodes(run.path, 'a', 'lg-ref')
    assert len(references) == 3
    epub_type = '{http://www.idpf.org/2007/ops}type'
    links = [link for root in documents_of(run.path).values() for link in elements(root, 'a')]
    assert not any(link.get(epub_type) == 'backlink' for link in links)
    assert all('↩' not in (link.text or '') for link in links)
    for name, link in references:
        target = link.get('href', '')
        assert target.startswith('#')
        assert any(node.get('id') == target[1:] for node in documents_of(run.path)[name].iter())


def test_pronunciation_epub_original_markup_unchanged(tmp_path):
    run, _, _ = pronunciation_export(tmp_path)
    first = next(root for name, root in documents_of(run.path).items() if name.endswith('chapter1.xhtml'))
    original = next(p for p in elements(first, 'p') if _plain(p) == 'A zorp.')
    assert 'lg-src' in (original.get('class') or '').split()
    assert len(list(elements(original, 'em'))) == 1
    assert _plain(elements(original, 'em')[0]) == 'zorp'
    for marker in list(elements(original, 'a')):
        previous = marker.getprevious()
        if marker.tail:
            if previous is None:
                marker.getparent().text = (marker.getparent().text or '') + marker.tail
            else:
                previous.tail = (previous.tail or '') + marker.tail
        marker.getparent().remove(marker)
    assert _plain(original) == 'A zorp.'
    assert [etree.QName(node).localname for node in original.iter()] == ['p', 'em']


def test_pronunciation_epub_guide_and_offline_reading(tmp_path, monkeypatch):
    import socket
    def no_network(*args, **kwargs):
        raise AssertionError('EPUB generation must not use the network')
    monkeypatch.setattr(socket, 'create_connection', no_network)
    run, _, _ = pronunciation_export(tmp_path)
    notes = pronunciation_nodes(run.path, 'span', 'lg-ipa')
    assert len(notes) == 3
    text = ''.join(''.join(root.itertext()) for root in documents_of(run.path).values())
    assert 'IPA — международная фонетическая транскрипция' not in text
    assert 'основное' not in text and 'вторичное' not in text and 'долгот' not in text
    assert all('UK' in (node.tail or '') for _, node in notes)
    assert all(not a.get('href', '').startswith(('http:', 'https:'))
               for root in documents_of(run.path).values() for a in elements(root, 'a'))


async def ipa_dict_export(tmp_path, *, chapter_path='chapter.xhtml'):
    from linguary.adapters.pronunciation.source import IpaDictPronunciationSource
    from linguary.domain.pronunciation import PronunciationAssistance

    source_path = build_epub(tmp_path / 'ipa-source.epub', documents={
        chapter_path: xhtml_document('<h1>Chapter One</h1><p>The colonel joined the queue.</p><p>The queue moved.</p>'),
    }, toc=[(chapter_path, 'Chapter One')])
    document = run_export(source_path, tmp_path=tmp_path / 'dry').imported.document
    source = IpaDictPronunciationSource()
    results = {}
    for block in document.blocks():
        items = []
        for form, variant in [('colonel', 'uk'), ('queue', 'us')]:
            if form not in block.text:
                continue
            entry, = (await source.lookup(form, variant)).entries
            start = block.text.index(form)
            items.append(PronunciationAssistance(SourceAnchor(block.id, start, start + len(form), form),
                form, entry.ipa, variant, 'ipa-' + form, entry.evidence))
        if items:
            results[block.id] = BlockProcessingResult(block.id, BlockTreatment.READ_AS_IS,
                                                     pronunciation_assistances=tuple(items))
    return run_export(source_path, tmp_path=tmp_path / 'result', block_results=results), document, results


async def test_ipa_dict_epub_contains_used_source_licenses_and_provenance(tmp_path, monkeypatch):
    import json
    import socket
    from linguary.adapters.pronunciation.source import SOURCE_VERSION

    monkeypatch.setattr(socket, 'create_connection', lambda *args, **kwargs: pytest.fail('Network is forbidden'))
    run, _, results = await ipa_dict_export(tmp_path)
    pages = pronunciation_nodes(run.path, 'section', 'linguary-pronunciation-sources')
    assert len(pages) == 1
    text = ''.join(pages[0][1].itertext())
    assert all(value in text for value in (SOURCE_VERSION, 'GNU GENERAL PUBLIC LICENSE', 'MIT License'))
    with zipfile.ZipFile(run.path) as archive:
        attachment, = [name for name in archive.namelist() if name.endswith('pronunciation-sources.json')]
        provenance = json.loads(archive.read(attachment))
    expected = {item.evidence.entry_id for result in results.values() for item in result.pronunciation_assistances}
    assert {record['entry_id'] for record in provenance['entries']} == expected
    assert {record['form'] for record in provenance['entries']} == {'colonel', 'queue'}
    assert {record['license_id'] for record in provenance['entries']} == {'MIT', 'GPL-3.0'}
    assert all(record['source_version'] == SOURCE_VERSION and record['source_uri'].startswith('https://raw.githubusercontent.com/')
               for record in provenance['entries'])


async def test_ipa_dict_epub_original_and_backlinks_unchanged(tmp_path):
    run, _, _ = await ipa_dict_export(tmp_path)
    originals = [node for root in documents_of(run.path).values() for node in elements(root, 'p')
                 if 'lg-src' in (node.get('class') or '').split()
                 and _plain(node) in ('The colonel joined the queue.', 'The queue moved.')]
    assert len(originals) == 2
    notes = pronunciation_nodes(run.path, 'span', 'lg-ipa')
    references = pronunciation_nodes(run.path, 'a', 'lg-ref')
    assert len(notes) == 3 and len(references) == 3
    backlinks = [link for root in documents_of(run.path).values() for link in elements(root, 'a')
                 if link.get('{http://www.idpf.org/2007/ops}type') == 'backlink']
    assert backlinks == []
    assert pronunciation_nodes(run.path, 'a', 'linguary-pronunciation-sources-link') == []
    assert '↩' not in ''.join(''.join(root.itertext()) for root in documents_of(run.path).values())


def test_legacy_semantic_epub_needs_no_new_license_fields(tmp_path):
    run, _, _ = pronunciation_export(tmp_path)
    assert not pronunciation_nodes(run.path, 'section', 'linguary-pronunciation-sources')
    with zipfile.ZipFile(run.path) as archive:
        assert not any(name.endswith('pronunciation-sources.json') for name in archive.namelist())


async def test_ipa_dict_epub_corrupt_notices_keep_previous_export(tmp_path, monkeypatch):
    from linguary.adapters.epub import EpubExportWriter, pronunciation_notices
    from linguary.application.ports.epub import EpubExportFatalError, EpubWriteRequest

    run, document, results = await ipa_dict_export(tmp_path)
    original = run.path.read_bytes()
    monkeypatch.setitem(pronunciation_notices._NOTICE_HASHES, 'LICENSES.txt', '0' * 64)
    with pytest.raises(EpubExportFatalError, match='уведомления'):
        EpubExportWriter().write(EpubWriteRequest(document=document, target_path=str(run.path), block_results=results))
    assert run.path.read_bytes() == original
