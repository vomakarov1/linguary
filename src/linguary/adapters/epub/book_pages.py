"""XHTML for the legend, the IPA key, and who-is-who.

The writer places these documents. The lines, signs, and entries are already
on the layout. An empty tuple is not a document.
"""

from __future__ import annotations

from collections.abc import Sequence

from lxml import etree

from linguary.adapters.epub.pronunciation_notices import IPA_KEY_LINK_LABEL
from linguary.adapters.epub.writer import _XHTML_NAMESPACE, _element, _sub, _xhtml_document
from linguary.application.layout.ipa_reference import IpaSymbol
from linguary.application.layout.review import ReviewEntry
from linguary.application.layout.who_is_who import WhoEntry
from linguary.domain import ConceptType

LEGEND_TITLE = "Как устроена эта книга"
IPA_KEY_TITLE = "Ключ к транскрипции"
WHO_IS_WHO_TITLE = "Кто есть кто"
_IPA_INTRO = (
    "Здесь только знаки, которые встречаются в этой книге. "
    "Русские соответствия приблизительны."
)
_REVIEW_INTRO = (
    "Эти слова уже объяснялись и снова встретятся в главе. "
    "В тексте главы примечания к ним не повторяются."
)
_GROUP_LABELS = {
    ConceptType.PERSON: "Персонажи",
    ConceptType.PLACE: "Места",
}
_ROMAN = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def legend_document(
    lines: Sequence[str],
    *,
    language: str,
    stylesheet: str,
    ipa_key_href: str | None = None,
) -> etree._Element:
    """One paragraph per legend sentence. The key phrase is a link only when given."""
    root = _xhtml_document(title=LEGEND_TITLE, language=language, stylesheet=stylesheet)
    body = root.find(f"{{{_XHTML_NAMESPACE}}}body")
    section = _sub(
        body, "section", {"id": "legend", "class": "lg-sec lg-newpage", "lang": language}
    )
    _sub(section, "h1").text = LEGEND_TITLE
    for line in lines:
        _append_legend_line(section, line, ipa_key_href)
    return root


def ipa_key_document(
    symbols: Sequence[IpaSymbol],
    *,
    language: str,
    example_language: str,
    stylesheet: str,
) -> etree._Element:
    """A row per given sign, in that order. No sign is added."""
    root = _xhtml_document(title=IPA_KEY_TITLE, language=language, stylesheet=stylesheet)
    body = root.find(f"{{{_XHTML_NAMESPACE}}}body")
    section = _sub(
        body, "section", {"id": "ipa-key", "class": "lg-sec lg-newpage", "lang": language}
    )
    _sub(section, "h1").text = IPA_KEY_TITLE
    _sub(section, "p").text = _IPA_INTRO
    table = _sub(section, "table", {"class": "lg-key"})
    head = _sub(_sub(table, "thead"), "tr")
    for label in ("Знак", "Пример", "Как звучит"):
        _sub(head, "th").text = label
    rows = _sub(table, "tbody")
    for item in symbols:
        row = _sub(rows, "tr")
        _sub(row, "td").text = item.symbol
        _sub(row, "td", {"lang": example_language}).text = item.example
        _sub(row, "td").text = item.description
    return root


def who_is_who_document(
    entries: Sequence[WhoEntry],
    *,
    language: str,
    source_language: str,
    stylesheet: str,
) -> etree._Element:
    """The book list, grouped only where the entry type changes."""
    root = _xhtml_document(title=WHO_IS_WHO_TITLE, language=language, stylesheet=stylesheet)
    body = root.find(f"{{{_XHTML_NAMESPACE}}}body")
    section = _sub(
        body, "section", {"id": "who-is-who", "class": "lg-sec lg-newpage", "lang": language}
    )
    _sub(section, "h1").text = WHO_IS_WHO_TITLE
    _append_who_groups(
        section,
        entries,
        source_language=source_language,
        show_first_chapter=True,
        group_heading="h2",
    )
    return root


def append_who(
    parent: etree._Element,
    entries: Sequence[WhoEntry],
    *,
    source_language: str,
) -> None:
    """Chapter «Кто есть кто». An empty tuple adds nothing."""
    if not entries:
        return
    section = _sub(parent, "section", {"class": "lg-who"})
    _sub(section, "h3").text = "Кто есть кто"
    _append_who_groups(
        section,
        entries,
        source_language=source_language,
        show_first_chapter=False,
        group_heading="h4",
    )


def append_review(
    parent: etree._Element,
    entries: Sequence[ReviewEntry],
    *,
    source_language: str,
) -> None:
    """Chapter «Повторим». An empty tuple adds nothing."""
    if not entries:
        return
    section = _sub(parent, "section", {"class": "lg-review"})
    _sub(section, "h3").text = "Повторим"
    _sub(section, "p").text = _REVIEW_INTRO
    listing = _sub(section, "ul", {"class": "lg-entries"})
    for entry in entries:
        item = _sub(listing, "li")
        head = _sub(item, "span", {"class": "lg-head", "lang": source_language})
        head.text = entry.expression
        head.tail = f" — {entry.translation} "
        _sub(item, "span", {"class": "lg-src-ref"}).text = f"(гл. {_roman(entry.first_chapter)})"


def _append_legend_line(parent: etree._Element, line: str, ipa_key_href: str | None) -> None:
    paragraph = _sub(parent, "p")
    if not ipa_key_href or IPA_KEY_LINK_LABEL not in line:
        paragraph.text = line
        return
    before, _, after = line.partition(IPA_KEY_LINK_LABEL)
    if before:
        paragraph.text = before
    link = _sub(paragraph, "a", {"href": ipa_key_href})
    link.text = IPA_KEY_LINK_LABEL
    link.tail = after


def _append_who_groups(
    parent: etree._Element,
    entries: Sequence[WhoEntry],
    *,
    source_language: str,
    show_first_chapter: bool,
    group_heading: str,
) -> None:
    """Keep entry order. A heading appears when the type changes."""
    listing: etree._Element | None = None
    current: ConceptType | None = None
    for entry in entries:
        if entry.type != current:
            current = entry.type
            label = _GROUP_LABELS.get(entry.type)
            if label is not None:
                _sub(parent, group_heading).text = label
            listing = _sub(parent, "ul", {"class": "lg-entries"})
        assert listing is not None
        _append_who_entry(
            listing,
            entry,
            source_language=source_language,
            show_first_chapter=show_first_chapter,
        )


def _append_who_entry(
    parent: etree._Element,
    entry: WhoEntry,
    *,
    source_language: str,
    show_first_chapter: bool,
) -> None:
    item = _sub(parent, "li")
    head = _sub(item, "span", {"class": "lg-head", "lang": source_language})
    head.text = entry.source
    tail = f" — {entry.translation}."
    if entry.description.strip():
        tail += f" {entry.description.strip()}"
    head.tail = tail
    anchor: etree._Element = head
    if entry.name_method:
        head.tail += " "
        method = _sub(item, "span", {"class": "lg-method"})
        method.text = entry.name_method
        anchor = method
    if show_first_chapter:
        anchor.tail = (anchor.tail or "") + " "
        reference = _sub(item, "span", {"class": "lg-src-ref"})
        reference.text = f"Впервые в гл. {_roman(entry.first_chapter)}."


def _roman(number: int) -> str:
    """Content-chapter number as a Roman numeral. Non-positive numbers stay decimal."""
    if number <= 0:
        return str(number)
    remaining = number
    glyphs: list[str] = []
    for value, glyph in _ROMAN:
        while remaining >= value:
            glyphs.append(glyph)
            remaining -= value
    return "".join(glyphs)
