"""Learner pages in the exported EPUB: legend, IPA key, who-is-who, chapter front.

The writer renders `BookLayout` and the existing learning materials. It does not
rebuild the lists. An empty tuple is no page and no contents entry.
"""

from __future__ import annotations

from pathlib import Path

from epub_builder import build_epub, xhtml_document
from export_pipeline import documents_of, elements, spine_hrefs

from linguary.adapters.epub import EbookLibEpubReader, EpubExportWriter
from linguary.adapters.epub import verify_exported_epub as public_verify
from linguary.adapters.epub.writer import verify_exported_epub
from linguary.adapters.pronunciation.source import SNAPSHOT, SOURCE_VERSION
from linguary.application.layout.book import BookLayout, ChapterLayout
from linguary.application.layout.ipa_reference import IpaSymbol
from linguary.application.layout.legend import LayoutFeatures
from linguary.application.layout.review import ReviewEntry
from linguary.application.layout.who_is_who import WhoEntry
from linguary.application.ports.epub import EpubReadRequest, EpubWriteRequest
from linguary.domain import (
    BeforeYouReadEntry,
    BlockProcessingResult,
    BlockTreatment,
    ConceptType,
    LearningMaterial,
)
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence

_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
_LEGEND = (
    "Абзац оригинала всегда идёт первым.",
    "Транскрипция дана в косых чертах. "
    "UK и US — британское и американское произношение. "
    "Ключ к знакам — на следующей странице.",
)
_IPA = (
    IpaSymbol("ʊ", "foot /fʊt/", "краткое «у»"),
    IpaSymbol("e", "bed /bed/", "краткое «э»"),
)
_SUMMARY = "Alice follows the rabbit."
_NOTES_SUMMARY = "Summary of Notes."
_BYR_EXPRESSION = "Curiouser and curiouser"
_BYR_EXPLANATION = "Алиса нарочно говорит неправильно."
_SENTINEL = "SHOULD-NOT-APPEAR"


def test_legend_and_ipa_key_in_spine_and_nav(tmp_path: Path) -> None:
    """Legend then the IPA key follow the leading front matter, in the spine and the nav."""
    path = _export(tmp_path, "full")
    hrefs = spine_hrefs(path)
    legend = _index(hrefs, "legend.xhtml")
    ipa = _index(hrefs, "ipa-key.xhtml")
    assert _index(hrefs, "cover.xhtml") < _index(hrefs, "title.xhtml") < legend < ipa
    assert ipa < _index(hrefs, "chapter1.xhtml")
    assert _toc(path) == list(hrefs)

    legend_root = _document(path, "legend.xhtml")
    paragraphs = ["".join(node.itertext()) for node in elements(legend_root, "p")]
    assert paragraphs == list(_LEGEND)
    link = _one_link(legend_root, "Ключ к знакам")
    assert link.get("href", "").endswith("ipa-key.xhtml#ipa-key")
    assert _one_link(_document(path, "pronunciation-sources.xhtml"), "Ключ к знакам").get(
        "href", ""
    ).endswith("ipa-key.xhtml#ipa-key")

    empty = _export(tmp_path, "no-legend", legend=())
    empty_hrefs = spine_hrefs(empty)
    assert not any(href.endswith("legend.xhtml") for href in empty_hrefs)
    assert "Как устроена эта книга" not in _toc_titles(empty)
    assert _index(empty_hrefs, "title.xhtml") < _index(empty_hrefs, "ipa-key.xhtml")
    assert _index(empty_hrefs, "ipa-key.xhtml") < _index(empty_hrefs, "chapter1.xhtml")
    assert _toc(empty) == list(empty_hrefs)


def test_ipa_key_page_only_used_symbols(tmp_path: Path) -> None:
    """The key page lists the given signs in the given order and no others."""
    path = _export(tmp_path, "symbols")
    page = _document(path, "ipa-key.xhtml")
    rows = [
        row
        for row in elements(page, "tr")
        if any(cell.tag.rpartition("}")[2] == "td" for cell in row)
    ]
    cells = [
        tuple("".join(cell.itertext()) for cell in row if row_tag(cell) == "td")
        for row in rows
    ]

    assert cells == [
        ("ʊ", "foot /fʊt/", "краткое «у»"),
        ("e", "bed /bed/", "краткое «э»"),
    ]
    text = "".join(page.itertext())
    assert "θ" not in text
    assert "Cheshire" not in text
    assert "твёрдое «ч»" not in text
    assert "Здесь только знаки, которые встречаются в этой книге." in text


def test_ipa_key_omitted_when_no_ipa(tmp_path: Path) -> None:
    """No IPA tuple means no key document, no nav entry, and no link to it."""
    path = _export(tmp_path, "no-ipa", ipa_key=())
    hrefs = spine_hrefs(path)
    assert any(href.endswith("legend.xhtml") for href in hrefs)
    assert not any("ipa-key" in href for href in hrefs)
    assert "Ключ к транскрипции" not in _toc_titles(path)
    assert _toc(path) == list(hrefs)
    legend = "".join(_document(path, "legend.xhtml").itertext())
    assert "Ключ к знакам — на следующей странице." in legend
    assert _links_to_ipa(path) == []
    sources = _document(path, "pronunciation-sources.xhtml")
    assert "Ключ к знакам" not in "".join(sources.itertext())


def test_chapter_front_order(tmp_path: Path) -> None:
    """One front section, in order, and only on a content chapter that has something to say."""
    path = _export(tmp_path, "front")
    chapter = _chapter_section(_document(path, "chapter2.xhtml"))
    assert chapter[0].get("class") == "lg-front"
    front = chapter[0]
    assert [child.get("class") for child in front] == [
        "linguary-previously",
        "lg-who",
        "lg-review",
        "linguary-before-you-read",
    ]
    previously, who, review, before = list(front)
    assert _heading(elements(previously, "h2")[0]) == "Previously in Chapter One"
    assert "".join(elements(previously, "p")[0].itertext()) == _SUMMARY
    assert [_heading(node) for node in elements(who, "h3")] == ["Кто есть кто"]
    assert [_heading(node) for node in elements(who, "h4")] == ["Персонажи", "Места"]
    who_text = "".join(who.itertext())
    assert who_text.index("the Mouse") < who_text.index("Cheshire")
    assert "Зверёк в воде." in who_text
    assert "вольный перевод" in "".join(
        "".join(node.itertext())
        for node in elements(who, "li")
        if "Cheshire" in "".join(node.itertext())
    )
    assert all(
        "вольный перевод" not in "".join(node.itertext())
        for node in elements(who, "li")
        if "the Mouse" in "".join(node.itertext())
    )
    assert "Alice" not in who_text and "Pat" not in who_text and "Впервые" not in who_text
    review_text = "".join(review.itertext())
    assert "Повторим" in review_text
    assert "tired" in review_text and "устала" in review_text and "(гл. I)" in review_text
    assert _heading(elements(before, "h2")[0]) == "Before You Read"
    assert elements(before, "dt")[0].text == _BYR_EXPRESSION
    assert _BYR_EXPLANATION in "".join(elements(before, "dd")[0].itertext())

    third = _chapter_section(_document(path, "chapter3.xhtml"))
    assert [child.get("class") for child in third[0]] == ["lg-who"]
    third_text = "".join(third[0].itertext())
    assert "Pat" in third_text
    assert "Previously" not in third_text
    assert "Повторим" not in third_text
    assert "Before You Read" not in third_text
    assert "the Mouse" not in third_text

    for name in ("cover.xhtml", "title.xhtml", "chapter1.xhtml", "notes.xhtml"):
        root = _document(path, name)
        assert not _fronts(root)
        text = "".join(root.itertext())
        assert "Previously" not in text
        assert "Before You Read" not in text
        assert _SENTINEL not in text
    notes = "".join(_document(path, "notes.xhtml").itertext())
    assert _NOTES_SUMMARY not in notes
    whole = "".join(
        "".join(root.itertext()) for root in documents_of(path).values()
    )
    assert _SENTINEL not in whole


def test_book_who_is_who_at_end(tmp_path: Path) -> None:
    """The book list follows the last chapter and stays ahead of the sources page."""
    path = _export(tmp_path, "who")
    hrefs = spine_hrefs(path)
    who = _index(hrefs, "who-is-who.xhtml")
    assert _index(hrefs, "chapter3.xhtml") < who < _index(hrefs, "pronunciation-sources.xhtml")
    assert who == len(hrefs) - 2
    assert hrefs[-1].endswith("pronunciation-sources.xhtml")
    assert _toc(path) == list(hrefs)
    page = _document(path, "who-is-who.xhtml")
    text = "".join(page.itertext())
    assert _heading(elements(page, "h1")[0]) == "Кто есть кто"
    assert [_heading(node) for node in elements(page, "h2")] == ["Персонажи", "Места"]
    assert text.index("Alice") < text.index("the Mouse") < text.index("Pat") < text.index("Cheshire")
    assert "Впервые в гл. I." in text
    assert "Впервые в гл. II." in text
    assert "Впервые в гл. III." in text
    assert "Девочка." in text
    cheshire = next(node for node in elements(page, "li") if "Cheshire" in "".join(node.itertext()))
    assert "".join(elements(cheshire, "span")[-2].itertext()) == "вольный перевод"
    mouse = next(node for node in elements(page, "li") if "the Mouse" in "".join(node.itertext()))
    assert "вольный перевод" not in "".join(mouse.itertext())

    empty = _export(tmp_path, "no-who", who=())
    empty_hrefs = spine_hrefs(empty)
    assert not any("who-is-who" in href for href in empty_hrefs)
    assert "Кто есть кто" not in _toc_titles(empty)
    assert empty_hrefs[-1].endswith("pronunciation-sources.xhtml")
    assert _index(empty_hrefs, "chapter3.xhtml") == len(empty_hrefs) - 2


def test_export_passes_builtin_checks(tmp_path: Path) -> None:
    """The book that carries these pages passes the writer's own verifier."""
    path = _export(tmp_path, "checked")
    hrefs = spine_hrefs(path)
    assert any(href.endswith("legend.xhtml") for href in hrefs)
    assert any(href.endswith("ipa-key.xhtml") for href in hrefs)
    assert any(href.endswith("who-is-who.xhtml") for href in hrefs)
    assert public_verify is verify_exported_epub
    assert verify_exported_epub(path, expected_spine=hrefs) == ()


def row_tag(node) -> str:
    return node.tag.rpartition("}")[2]


def _export(
    tmp_path: Path,
    name: str,
    *,
    legend: tuple[str, ...] | None = None,
    ipa_key: tuple[IpaSymbol, ...] | None = None,
    who: tuple[WhoEntry, ...] | None = None,
) -> Path:
    """Import a small book and export it with an explicit layout."""
    source = build_epub(
        tmp_path / f"{name}-source.epub",
        documents={
            "cover.xhtml": xhtml_document("<h1>Cover</h1>", title="Cover"),
            "title.xhtml": xhtml_document("<h1>Title Page</h1>", title="Title Page"),
            "chapter1.xhtml": xhtml_document(
                "<h1>Chapter One</h1><p>Alice waited in the queue.</p>",
                title="Chapter One",
            ),
            "notes.xhtml": xhtml_document(
                "<h1>Notes</h1>"
                '<aside epub:type="footnote" id="fn1"><p>A plain face.</p></aside>',
                title="Notes",
            ),
            "chapter2.xhtml": xhtml_document(
                "<h1>Chapter Two</h1><p>Curiouser and curiouser cried Alice.</p>",
                title="Chapter Two",
            ),
            "chapter3.xhtml": xhtml_document(
                "<h1>Chapter Three</h1><p>The cat vanished.</p>",
                title="Chapter Three",
            ),
        },
        toc=[
            ("cover.xhtml", "Cover"),
            ("title.xhtml", "Title Page"),
            ("chapter1.xhtml", "Chapter One"),
            ("notes.xhtml", "Notes"),
            ("chapter2.xhtml", "Chapter Two"),
            ("chapter3.xhtml", "Chapter Three"),
        ],
    )
    imported = EbookLibEpubReader().read(
        EpubReadRequest(book_id=f"book-{name}", source_path=str(source), target_language="ru")
    )
    document = imported.document
    sections = {section.title: section for section in document.ordered_sections}
    queue = next(block for block in sections["Chapter One"].ordered_blocks if "queue" in block.text)
    start = queue.text.index("queue")
    evidence = PronunciationEvidence(
        source_id="ipa-dict",
        source_version=SOURCE_VERSION,
        entry_id="en_UK.txt:1:queue",
        sense_id=None,
        part_of_speech=None,
        dictionary_form="queue",
        dictionary_ipa="kjuː",
        attribution="IPA-Dict",
        kind="lexicon_form",
        license_id="GPL-3.0",
        source_uri=(
            "https://raw.githubusercontent.com/open-dict-data/ipa-dict/"
            f"{SNAPSHOT}/data/en_UK.txt"
        ),
    )
    assistance = PronunciationAssistance(
        SourceAnchor(queue.id, start, start + len("queue"), "queue"),
        "queue",
        "kjuː",
        "uk",
        "ph-queue",
        evidence,
    )
    materials = {
        sections["Cover"].id: _material(sections["Cover"].id, summary="Cover summary."),
        sections["Title Page"].id: _material(sections["Title Page"].id, summary="Title summary."),
        sections["Chapter One"].id: _material(sections["Chapter One"].id, summary=_SUMMARY),
        sections["Notes"].id: _material(
            sections["Notes"].id,
            summary=_NOTES_SUMMARY,
            before=(( "note idiom", "оборот примечаний"),),
        ),
        sections["Chapter Two"].id: _material(
            sections["Chapter Two"].id,
            summary="",
            before=((_BYR_EXPRESSION, _BYR_EXPLANATION),),
        ),
        sections["Chapter Three"].id: _material(sections["Chapter Three"].id, summary=""),
    }
    sentinel = _who("SHOULD-NOT-APPEAR", "Нет", ConceptType.PERSON, "Служебный.", None, 1)
    chapters = {
        sections["Cover"].id: _chapter(sections["Cover"].id, None, who=(sentinel,)),
        sections["Title Page"].id: _chapter(sections["Title Page"].id, None),
        sections["Chapter One"].id: _chapter(sections["Chapter One"].id, 1),
        sections["Notes"].id: _chapter(
            sections["Notes"].id,
            None,
            who=(sentinel,),
            review=(ReviewEntry("SHOULD-NOT-REVIEW", "нет", 1),),
        ),
        sections["Chapter Two"].id: _chapter(
            sections["Chapter Two"].id,
            2,
            who=(
                _who("the Mouse", "Мышь", ConceptType.PERSON, "Зверёк в воде.", None, 2),
                _who(
                    "Cheshire",
                    "Чешир",
                    ConceptType.PLACE,
                    "Графство на северо-западе Англии.",
                    "вольный перевод",
                    2,
                ),
            ),
            review=(ReviewEntry("tired", "устала", 1),),
        ),
        sections["Chapter Three"].id: _chapter(
            sections["Chapter Three"].id,
            3,
            who=(_who("Pat", "Пат", ConceptType.PERSON, "Слуга кролика.", None, 3),),
        ),
    }
    book_who = (
        _who("Alice", "Алиса", ConceptType.PERSON, "Девочка.", None, 1),
        _who("the Mouse", "Мышь", ConceptType.PERSON, "Зверёк в воде.", None, 2),
        _who("Pat", "Пат", ConceptType.PERSON, "Слуга кролика.", None, 3),
        _who(
            "Cheshire",
            "Чешир",
            ConceptType.PLACE,
            "Графство на северо-западе Англии.",
            "вольный перевод",
            2,
        ),
    )
    layout = BookLayout(
        chapters=chapters,
        legend=_LEGEND if legend is None else legend,
        ipa_key=_IPA if ipa_key is None else ipa_key,
        who_is_who=book_who if who is None else who,
        features=LayoutFeatures(False, False, frozenset(), False, False, False, frozenset()),
    )
    target = tmp_path / f"{name}.epub"
    EpubExportWriter().write(
        EpubWriteRequest(
            document=document,
            target_path=str(target),
            source_epub_path=str(source),
            block_results={
                queue.id: BlockProcessingResult(
                    queue.id,
                    BlockTreatment.READ_AS_IS,
                    pronunciation_assistances=(assistance,),
                )
            },
            learning_materials=materials,
            layout=layout,
        )
    )
    return target


def _material(
    section_id: str,
    *,
    summary: str,
    before: tuple[tuple[str, str], ...] = (),
) -> LearningMaterial:
    return LearningMaterial.for_chapter(
        section_id=section_id,
        before_you_read=tuple(
            BeforeYouReadEntry(expression, explanation) for expression, explanation in before
        ),
        chapter_summary=summary,
        language_notes=(),
    )


def _who(
    source: str,
    translation: str,
    kind: ConceptType,
    description: str,
    name_method: str | None,
    first_chapter: int,
) -> WhoEntry:
    return WhoEntry(source, translation, kind, description, name_method, first_chapter)


def _chapter(
    section_id: str,
    number: int | None,
    *,
    who: tuple[WhoEntry, ...] = (),
    review: tuple[ReviewEntry, ...] = (),
) -> ChapterLayout:
    return ChapterLayout(section_id, number, {}, who, review)


def _document(path: Path, suffix: str):
    href = next(item for item in spine_hrefs(path) if item.endswith(suffix))
    return documents_of(path)[f"OEBPS/{href}"]


def _index(hrefs: tuple[str, ...], suffix: str) -> int:
    return next(index for index, href in enumerate(hrefs) if href.endswith(suffix))


def _toc(path: Path) -> list[str]:
    nav = documents_of(path)["OEBPS/nav.xhtml"]
    return [node.get("href", "") for node in elements(nav, "a")]


def _toc_titles(path: Path) -> str:
    nav = documents_of(path)["OEBPS/nav.xhtml"]
    return "".join("".join(node.itertext()) for node in elements(nav, "a"))


def _one_link(root, text: str):
    found = [node for node in elements(root, "a") if node.text == text]
    assert len(found) == 1
    return found[0]


def _links_to_ipa(path: Path) -> list[str]:
    return [
        node.get("href", "")
        for root in documents_of(path).values()
        for node in elements(root, "a")
        if "ipa-key" in node.get("href", "")
    ]


def _chapter_section(root):
    return next(node for node in elements(root, "section") if node.get(_EPUB_TYPE) == "chapter")


def _fronts(root) -> list:
    return [
        node
        for node in elements(root, "section")
        if "lg-front" in (node.get("class") or "").split()
    ]


def _heading(node) -> str:
    return "".join(node.itertext())
