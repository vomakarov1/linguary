"""Метки ссылок на сноски в переводе блока (PRD § 7, § 20, QA-08)."""

from __future__ import annotations

from linguary.domain import (
    BlockType,
    DocumentBlock,
    InlineAnnotation,
    InlineAnnotationType,
    SourceReference,
    footnote_references,
    translation_segments,
    with_reference_labels,
    without_footnote_markers,
)

TEXT = "Her face.[1] It was, 2 said Tom, a pity."


def _block() -> DocumentBlock:
    """Абзац с двумя ссылками на сноски: «[1]» и «2», перечисленными не по порядку."""
    first = TEXT.index("[1]")
    second = TEXT.index("2")
    return DocumentBlock.create(
        document_id="book-1",
        section_id="section-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=TEXT,
        source_reference=SourceReference(
            resource_path="chapter1.xhtml", structural_path="/body/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=second,
                end=second + 1,
                target="notes.xhtml#fn2",
            ),
            InlineAnnotation(
                type=InlineAnnotationType.ITALIC, start=0, end=3, target=None
            ),
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=first,
                end=first + 3,
                target="notes.xhtml#fn1",
            ),
        ),
    )


def test_references_are_numbered_in_text_order() -> None:
    """`[^1]` — первая по тексту ссылка на сноску, другая разметка не считается."""
    assert [reference.target for reference in footnote_references(_block())] == [
        "notes.xhtml#fn1",
        "notes.xhtml#fn2",
    ]


def test_markers_become_references_and_unknown_ones_disappear() -> None:
    """Метка превращается в ссылку оригинала, метка без ссылки отбрасывается."""
    segments = translation_segments("Её лицо.[^1] Жаль, [^2] сказал Том [^3].", _block())

    assert [
        segment if isinstance(segment, str) else segment.target for segment in segments
    ] == ["Её лицо.", "notes.xhtml#fn1", " Жаль,", " ", "notes.xhtml#fn2", " сказал Том", "."]


def test_labels_replace_markers_like_in_the_original() -> None:
    """Для экранов метка показывается видимым текстом ссылки оригинала."""
    assert with_reference_labels("Её лицо.[^1] Жаль.", _block()) == "Её лицо.[1] Жаль."


def test_markers_are_removed_without_hanging_spaces() -> None:
    """Снятая метка не оставляет пробела перед знаком препинания."""
    assert without_footnote_markers("her face [^1], it was") == "her face, it was"
