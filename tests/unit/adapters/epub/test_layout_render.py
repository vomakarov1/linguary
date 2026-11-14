"""Reading units in the EPUB chapter: one part, its translation, its notes."""

from __future__ import annotations

import re

from lxml import etree

from linguary.adapters.epub.writer import (
    _STYLESHEET,
    _ExportWarnings,
    _OpaqueSource,
    _render_chapters,
)
from linguary.application.layout.book import BookLayout, ChapterLayout
from linguary.application.layout.legend import LayoutFeatures
from linguary.application.layout.notes import LayoutNote
from linguary.application.layout.units import LayoutUnit, NumberedNote
from linguary.application.ports.epub import EpubWriteRequest
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentSection,
    InlineAnnotation,
    InlineAnnotationType,
    SourceReference,
)
from linguary.domain.content_plan import SourceAnchor
from linguary.domain.learning import FragmentAssistance

_XHTML = "http://www.w3.org/1999/xhtml"
_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
_GUIDE = "IPA — международная фонетическая транскрипция"
_UNPREPARED = "Помощь пока не подготовлена."


def _block(
    text: str,
    *,
    block_id: str = "block-1",
    block_type: BlockType = BlockType.PARAGRAPH,
    order: int = 0,
    annotations: tuple[InlineAnnotation, ...] = (),
    attributes: dict[str, str] | None = None,
    parent_id: str | None = None,
    section_id: str = "sec-1",
) -> DocumentBlock:
    return DocumentBlock(
        id=block_id,
        section_id=section_id,
        type=block_type,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path="chapter.xhtml", structural_path=f"/body/*[{order + 1}]"
        ),
        inline_annotations=annotations,
        parent_id=parent_id,
        attributes=attributes or {},
    )


def _note(
    number: int | None,
    body: tuple[str, ...],
    *,
    marker_end: int | None,
    labels: tuple[str, ...] = (),
    term: str | None = None,
    ipa: str | None = None,
    variant: str | None = None,
) -> NumberedNote:
    return NumberedNote(
        number,
        LayoutNote(
            labels=labels,
            term=term,
            ipa=ipa,
            variant=variant,
            body=body,
            marker_end=marker_end,
            origin="language_note",
            difficulty_kinds=(),
        ),
    )


def _unit(
    block: DocumentBlock,
    *,
    start: int = 0,
    end: int | None = None,
    translation: str | None = None,
    speaker: str | None = None,
    notes: tuple[NumberedNote, ...] = (),
    continues: bool = False,
    ends_paragraph: bool = True,
    lines: bool = False,
) -> LayoutUnit:
    return LayoutUnit(
        block_id=block.id,
        start=start,
        end=len(block.text) if end is None else end,
        translation=translation,
        speaker=speaker,
        notes=notes,
        continues=continues,
        ends_paragraph=ends_paragraph,
        lines=lines or block.type is BlockType.POEM,
    )


def _layout(section: DocumentSection, units: dict[str, tuple[LayoutUnit, ...]]) -> BookLayout:
    return BookLayout(
        chapters={
            section.id: ChapterLayout(
                section_id=section.id,
                number=1,
                units=units,
                who_is_who=(),
                review=(),
            )
        },
        legend=(),
        ipa_key=(),
        who_is_who=(),
        features=LayoutFeatures(
            has_translation=False,
            has_notes=False,
            labels=frozenset(),
            has_ipa=False,
            has_poems=False,
            has_speakers=False,
            sections=frozenset(),
        ),
    )


def _chapter(*blocks: DocumentBlock, layout: BookLayout | None = None, results=None):
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Chapter One",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=blocks,
    )
    document = CanonicalDocument(
        id="book-1",
        metadata=BookMetadata(title="Story"),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )
    request = EpubWriteRequest(
        document=document,
        target_path="unused.epub",
        block_results=results or {},
        layout=layout,
    )
    rendered = _render_chapters(
        request, opaque=_OpaqueSource(None), warnings=_ExportWarnings()
    )
    return rendered[0].root


def _elements(root, tag: str) -> list[etree._Element]:
    return [node for node in root.iter() if node.tag == f"{{{_XHTML}}}{tag}"]


def _classes(node: etree._Element) -> set[str]:
    return set((node.get("class") or "").split())


def _units(root) -> list[etree._Element]:
    return [node for node in _elements(root, "div") if "lg-unit" in _classes(node)]


def _plain(node: etree._Element) -> str:
    """Element text without marker digits."""
    chunks: list[str] = []

    def walk(element: etree._Element) -> None:
        if element.tag == f"{{{_XHTML}}}sup":
            return
        if element.text:
            chunks.append(element.text)
        for child in element:
            walk(child)
            if child.tail:
                chunks.append(child.tail)

    walk(node)
    return "".join(chunks)


def _text_before(node: etree._Element, marker: etree._Element) -> str:
    """Text of `node` before `marker`, not including the marker's own text."""
    chunks: list[str] = []

    def walk(element: etree._Element) -> bool:
        if element is marker:
            return True
        if element.text:
            chunks.append(element.text)
        for child in element:
            if walk(child):
                return True
            if child.tail:
                chunks.append(child.tail)
        return False

    walk(node)
    return "".join(chunks)


def _source(unit: etree._Element) -> etree._Element:
    return next(node for node in _elements(unit, "p") if "lg-src" in _classes(node))


def test_no_fragment_frames_or_part_labels() -> None:
    """Parts of one paragraph are units, without frames or Part labels."""
    text = "First half of the sentence. Second half of the sentence."
    block = _block(
        text,
        attributes={"split_group": "paragraph", "split_index": "2", "split_total": "2"},
    )
    cut = text.index(" Second")
    root = _chapter(
        block,
        layout=_layout(
            DocumentSection(
                id="sec-1",
                document_id="book-1",
                title="Chapter One",
                order=0,
                source_resource_path="chapter.xhtml",
                blocks=(block,),
            ),
            {
                block.id: (
                    _unit(block, end=cut, translation="Первая.", ends_paragraph=False),
                    _unit(
                        block,
                        start=cut,
                        translation="Вторая.",
                        continues=True,
                        ends_paragraph=True,
                    ),
                )
            },
        ),
    )
    units = _units(root)
    assert [tuple(sorted(_classes(unit))) for unit in units] == [
        ("lg-end",) if False else ("lg-unit",),
        ("lg-cont", "lg-end", "lg-unit"),
    ]
    assert _classes(units[0]) == {"lg-unit"}
    assert "".join(_plain(_source(unit)) for unit in units) == text
    serialized = etree.tostring(root, encoding=str)
    assert "linguary-paragraph-fragment" not in serialized
    assert "linguary-fragment-label" not in serialized
    assert "Part " not in serialized
    assert "часть" not in serialized.lower()


def test_untranslated_paragraph_single_p() -> None:
    """A block without a layout is one source paragraph and has no translation."""
    text = "The traveller walked on. " * 40
    block = _block(text.strip())
    root = _chapter(block)
    units = _units(root)
    assert len(units) == 1
    sources = [node for node in _elements(units[0], "p") if "lg-src" in _classes(node)]
    translations = [node for node in _elements(units[0], "p") if "lg-tr" in _classes(node)]
    assert len(sources) == 1
    assert translations == []
    assert _plain(sources[0]) == block.text
    assert _elements(units[0], "ol") == []


def test_each_marker_links_to_note_of_its_unit() -> None:
    """Each digit points at the note in the same unit. Labels stay separate."""
    text = "Alice heard a tale and then saw the tail."
    block = _block(text, attributes={"anchor_ids": "para-1 extra-1"})
    tale_end = text.index("tale") + len("tale")
    tail_end = text.index("tail") + len("tail")
    cut = text.index(" saw")
    root = _chapter(
        block,
        layout=_layout(
            DocumentSection(
                id="sec-1",
                document_id="book-1",
                title="Chapter One",
                order=0,
                source_resource_path="chapter.xhtml",
                blocks=(block,),
            ),
            {
                block.id: (
                    _unit(
                        block,
                        end=cut,
                        translation="Алиса услышала историю",
                        speaker="Алиса",
                        ends_paragraph=False,
                        notes=(
                            _note(
                                1,
                                ("Мышь говорит об истории.",),
                                marker_end=tale_end,
                                labels=("Игра слов.", "Грамматика."),
                                term="tale",
                            ),
                        ),
                    ),
                    _unit(
                        block,
                        start=cut,
                        continues=True,
                        translation=None,
                        notes=(
                            _note(
                                1,
                                ("Алиса смотрит на хвост.",),
                                marker_end=tail_end,
                                term="tail",
                                ipa="teɪl",
                                variant="uk",
                            ),
                            _note(None, ("Говорит Алиса",), marker_end=None),
                        ),
                    ),
                )
            },
        ),
    )
    units = _units(root)
    assert units[0].get("id") == "para-1"
    assert [node.get("id") for node in _elements(units[0], "span") if node.get("id")] == [
        "extra-1"
    ]
    assert units[1].get("id") is None
    assert all(node.get("id") != "para-1" for node in units[1].iter())

    first_notes = _elements(_elements(units[0], "ol")[0], "li")
    assert len(first_notes) == 1
    kinds = [
        node.text
        for node in _elements(first_notes[0], "span")
        if "lg-kind" in _classes(node)
    ]
    assert kinds == ["Игра слов.", "Грамматика."]
    assert not any("," in (kind or "") for kind in kinds)
    source = _source(units[0])
    link = next(node for node in _elements(source, "a") if "lg-ref" in _classes(node))
    assert link.get("href") == f"#{first_notes[0].get('id')}"
    assert _elements(link, "sup")[0].text == "1"
    before = _text_before(source, link)
    assert before == text[:tale_end]
    assert _plain(source)[len(before):] == text[tale_end:cut]
    assert text[tale_end:cut].strip()
    translation = next(node for node in _elements(units[0], "p") if "lg-tr" in _classes(node))
    assert translation.get("lang") == "ru"
    speaker = next(node for node in _elements(translation, "span") if "lg-speaker" in _classes(node))
    assert speaker.text == "Алиса"
    assert "Алиса услышала историю" in _plain(translation)
    assert speaker.text not in (translation.text or "")

    second_list = _elements(units[1], "ol")[0]
    assert second_list.get("lang") == "ru"
    numbered, unnumbered = _elements(second_list, "li")
    second_link = next(
        node for node in _elements(_source(units[1]), "a") if "lg-ref" in _classes(node)
    )
    assert second_link.get("href") == f"#{numbered.get('id')}"
    assert second_link.get("href") != f"#{first_notes[0].get('id')}"
    assert _elements(numbered, "span")[0].text == "1"
    assert "lg-num" in _classes(_elements(numbered, "span")[0])
    assert [node.text for node in _elements(numbered, "span") if "lg-term" in _classes(node)] == [
        "tail"
    ]
    term = next(node for node in _elements(numbered, "span") if "lg-term" in _classes(node))
    assert term.get("lang") == "en"
    ipa = next(node for node in _elements(numbered, "span") if "lg-ipa" in _classes(node))
    assert ipa.text == "/teɪl/"
    assert "UK" in "".join(numbered.itertext())
    assert _elements(unnumbered, "span") == [] or all(
        "lg-num" not in _classes(node) for node in _elements(unnumbered, "span")
    )
    assert not any("lg-ref" in _classes(node) and node.get("href") == f"#{unnumbered.get('id')}"
                   for node in _elements(units[1], "a"))
    assert [child for child in units[1] if etree.QName(child).localname == "p"] == [
        _source(units[1])
    ]
    assert "Говорит Алиса" in "".join(unnumbered.itertext())


def test_no_unprepared_help_or_reference_prefix() -> None:
    """Unprepared help and the reference prefix are not printed on a note."""
    kept = _block("A curious phrase stays here.")
    other = _block("Another sentence without help.", block_id="block-2", order=1)
    anchor = SourceAnchor(other.id, 0, 7, "Another")
    assistance = FragmentAssistance(
        "difficulty", ("point",), "unresolved", None, (anchor,)
    )
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Chapter One",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(kept, other),
    )
    root = _chapter(
        kept,
        other,
        layout=_layout(
            section,
            {
                kept.id: (
                    _unit(
                        kept,
                        notes=(
                            _note(
                                1,
                                ("Языковая справка: короткое пояснение.",),
                                marker_end=len("A curious"),
                            ),
                            _note(2, (_UNPREPARED,), marker_end=len(kept.text)),
                        ),
                    ),
                )
            },
        ),
        results={
            other.id: BlockProcessingResult(
                other.id, BlockTreatment.READ_AS_IS, fragment_assistances=(assistance,)
            )
        },
    )
    text = "".join(root.itertext())
    assert "Языковая справка" not in text
    assert "короткое пояснение." in text
    assert _UNPREPARED not in text
    assert "↩" not in text
    assert _GUIDE not in text
    notes = [
        node
        for node in _elements(root, "li")
        if "lg-notes" in _classes(node.getparent())
    ]
    assert len(notes) == 1
    assert len([node for node in _elements(root, "a") if "lg-ref" in _classes(node)]) == 1


def test_no_footnote_epub_type_in_learning_layers() -> None:
    """Learning notes are list items. A source footnote may keep its type."""
    paragraph = _block("Nobody liked her face.")
    footnote = _block("", block_id="fn", order=1, block_type=BlockType.FOOTNOTE)
    inner = _block(
        "A plain face.",
        block_id="fn-p",
        order=2,
        parent_id=footnote.id,
    )
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Chapter One",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(paragraph, footnote, inner),
    )
    root = _chapter(
        paragraph,
        footnote,
        inner,
        layout=_layout(
            section,
            {
                paragraph.id: (
                    _unit(
                        paragraph,
                        translation="Никому не нравилось её лицо.",
                        notes=(_note(1, ("face — лицо",), marker_end=len(paragraph.text)),),
                    ),
                )
            },
        ),
    )
    learning = [
        node
        for node in root.iter()
        if _classes(node) & {"lg-unit", "lg-notes", "lg-tr", "lg-ref", "lg-src"}
    ]
    assert learning
    assert all(node.get(_EPUB_TYPE) != "footnote" for node in learning)
    aside = next(node for node in _elements(root, "aside"))
    assert aside.get(_EPUB_TYPE) == "footnote"
    assert "A plain face." in "".join(aside.itertext())


def test_stylesheet_no_generated_text_or_colors() -> None:
    """Learning CSS does not generate text, paint a color, or fill a background."""
    assert "::before" not in _STYLESHEET
    assert "::after" not in _STYLESHEET
    assert "content:" not in _STYLESHEET
    assert "background" not in _STYLESHEET
    colors = re.findall(r"(?<![\w-])color\s*:\s*([^;]+)", _STYLESHEET)
    assert colors
    assert {value.strip() for value in colors} == {"inherit"}
    assert not re.search(r"#[0-9a-fA-F]{3,8}", _STYLESHEET)


def test_stylesheet_typography_limits_and_page_breaks() -> None:
    """Type stays at least 0.85em, except the marker, and both break properties are set."""
    for selector, body in re.findall(r"([^{}]+)\{([^}]+)\}", _STYLESHEET):
        for size in re.findall(r"font-size\s*:\s*([0-9.]+)em", body):
            value = float(size)
            if "sup" in selector:
                assert value == 0.7
            else:
                assert value >= 0.85
    assert "page-break-inside" in _STYLESHEET and "break-inside" in _STYLESHEET
    assert "page-break-before" in _STYLESHEET and "break-before" in _STYLESHEET
    assert "border-left" in _STYLESHEET
    assert re.search(r"(?<![\w-])border\s*:", _STYLESHEET) is None


def test_inline_markup_preserved_across_parts() -> None:
    """Italic that crosses a cut survives in both parts. Anchors stay on the first."""
    text = "Hello bold world and a link."
    bold = InlineAnnotation(InlineAnnotationType.BOLD, text.index("bold"), text.index("bold") + 4)
    link_start = text.index("link")
    link = InlineAnnotation(
        InlineAnnotationType.LINK, link_start, link_start + 4, "https://example.com"
    )
    block = _block(
        text,
        annotations=(bold, link),
        attributes={"anchor_ids": "first-anchor second-anchor"},
    )
    cut = text.index("bold") + 2
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Chapter One",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(block,),
    )
    root = _chapter(
        block,
        layout=_layout(
            section,
            {
                block.id: (
                    _unit(block, end=cut, ends_paragraph=False),
                    _unit(block, start=cut, continues=True),
                )
            },
        ),
    )
    units = _units(root)
    assert "".join(_plain(_source(unit)) for unit in units) == text
    assert _elements(_source(units[0]), "b")[0].text == "bo"
    assert _elements(_source(units[1]), "b")[0].text == "ld"
    source_link = _elements(_source(units[1]), "a")[0]
    assert source_link.get("href") == "https://example.com"
    assert source_link.text == "link"
    assert units[0].get("id") == "first-anchor"
    assert any(node.get("id") == "second-anchor" for node in units[0].iter())
    assert all(node.get("id") is None for node in units[1].iter() if node is not units[1])
    assert units[1].get("id") is None
    assert not any(node.get("id") for node in units[1].iter())


def test_poem_stanzas_preserved() -> None:
    """Poem lines keep their breaks, including the blank line between stanzas."""
    text = "How doth the little crocodile\nImprove his shining tail,\n\nAnd pour the waters\nOn every golden scale!"
    translation = "Как крокодил\nСвой хвост полирует,\n\nИ льёт воды\nНа каждую чешуйку!"
    block = _block(text, block_type=BlockType.POEM)
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Chapter One",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(block,),
    )
    root = _chapter(
        block,
        layout=_layout(section, {block.id: (_unit(block, translation=translation, lines=True),)}),
    )
    unit = _units(root)[0]
    source = _source(unit)
    translated = next(node for node in _elements(unit, "p") if "lg-tr" in _classes(node))

    def breaks(node: etree._Element) -> list[str]:
        names = [etree.QName(child).localname for child in node.iter()]
        return names

    assert breaks(source).count("br") == text.count("\n")
    assert breaks(translated).count("br") == translation.count("\n")
    source_children = [etree.QName(child).localname for child in source]
    translated_children = [etree.QName(child).localname for child in translated]
    assert source_children[source_children.index("br") + 1] == "br"
    assert translated_children[translated_children.index("br") + 1] == "br"
    assert _plain(source) == text.replace("\n", "")
    assert _plain(translated) == translation.replace("\n", "")


def test_heading_keeps_original_level() -> None:
    """A heading keeps h1–h6, or h2 when the level is unknown. It is not p.lg-src."""
    titled = _block(
        "Chapter One",
        block_id="heading",
        block_type=BlockType.HEADING,
        attributes={"tag": "h1", "anchor_ids": "chap extra"},
    )
    unknown = _block(
        "Untitled",
        block_id="unknown",
        block_type=BlockType.HEADING,
        order=1,
        attributes={"tag": "div"},
    )
    section = DocumentSection(
        id="sec-1",
        document_id="book-1",
        title="Chapter One",
        order=0,
        source_resource_path="chapter.xhtml",
        blocks=(titled, unknown),
    )
    note = _note(None, ("Заголовок главы.",), marker_end=None)
    root = _chapter(
        titled,
        unknown,
        layout=_layout(
            section,
            {
                titled.id: (_unit(titled, translation="Глава первая.", notes=(note,)),),
                unknown.id: (_unit(unknown),),
            },
        ),
    )
    units = _units(root)
    heading = units[0]
    source = heading[0]
    assert heading.get("id") is None
    assert etree.QName(source).localname == "h1"
    assert "lg-src" not in _classes(source)
    assert source.get("id") == "chap"
    assert source.get("lang") == "en"
    assert source.text == "Chapter One"
    assert [node.get("id") for node in source if node.get("id")] == ["extra"]
    assert [etree.QName(child).localname for child in heading] == ["h1", "p", "ol"]
    translation = heading[1]
    assert "lg-tr" in _classes(translation)
    assert _plain(translation) == "Глава первая."
    notes = heading[2]
    assert "lg-notes" in _classes(notes)
    assert "Заголовок главы." in _plain(notes)
    assert not any("lg-ref" in _classes(node) for node in _elements(notes, "a"))
    fallback = units[1][0]
    assert etree.QName(fallback).localname == "h2"
    assert "lg-src" not in _classes(fallback)
    assert fallback.text == "Untitled"
    assert not any("lg-src" in _classes(node) for node in root.iter())
