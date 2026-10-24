"""Stability of block and language note identifiers (PRD § 8, § 19.3)."""

from __future__ import annotations

import pytest
from domain_factories import DOCUMENT_ID, RESOURCE_PATH, make_block

from linguary.domain import (
    BlockType,
    IdentifierError,
    UnknownBlockTypeError,
    build_block_id,
    build_language_note_id,
)

STRUCTURAL_PATH = "/body/section[1]/p[12]"


def test_block_id_is_stable_for_same_position() -> None:
    """The same document block receives the same identifier."""
    first = build_block_id(
        document_id=DOCUMENT_ID,
        source_resource_path=RESOURCE_PATH,
        structural_path=STRUCTURAL_PATH,
    )
    second = build_block_id(
        document_id=DOCUMENT_ID,
        source_resource_path=RESOURCE_PATH,
        structural_path=STRUCTURAL_PATH,
    )
    assert first == second


def test_block_id_does_not_depend_on_text() -> None:
    """Editing only the block text does not change its identifier."""
    original = make_block(order=11, text="It was a bright cold day in April.")
    edited = make_block(order=11, text="Совсем другой текст блока.")
    assert original.id == edited.id


@pytest.mark.parametrize(
    ("document_id", "resource_path", "structural_path"),
    [
        pytest.param("book-2", RESOURCE_PATH, STRUCTURAL_PATH, id="other-document"),
        pytest.param(DOCUMENT_ID, "chapter04.xhtml", STRUCTURAL_PATH, id="other-resource"),
        pytest.param(DOCUMENT_ID, RESOURCE_PATH, "/body/section[1]/p[13]", id="other-position"),
    ],
)
def test_block_id_changes_with_position(
    document_id: str, resource_path: str, structural_path: str
) -> None:
    """Changing any component of a block's position produces a different identifier."""
    baseline = build_block_id(
        document_id=DOCUMENT_ID,
        source_resource_path=RESOURCE_PATH,
        structural_path=STRUCTURAL_PATH,
    )
    assert (
        build_block_id(
            document_id=document_id,
            source_resource_path=resource_path,
            structural_path=structural_path,
        )
        != baseline
    )


def test_block_id_separates_components() -> None:
    """Different splits of the same characters do not produce identical identifiers."""
    first = build_block_id(
        document_id="book", source_resource_path="1chapter", structural_path="/p[1]"
    )
    second = build_block_id(
        document_id="book1", source_resource_path="chapter", structural_path="/p[1]"
    )
    assert first != second


@pytest.mark.parametrize(
    ("document_id", "resource_path", "structural_path"),
    [
        pytest.param("", RESOURCE_PATH, STRUCTURAL_PATH, id="empty-document"),
        pytest.param(DOCUMENT_ID, "  ", STRUCTURAL_PATH, id="blank-resource"),
        pytest.param(DOCUMENT_ID, RESOURCE_PATH, "", id="empty-structural-path"),
    ],
)
def test_block_id_rejects_empty_components(
    document_id: str, resource_path: str, structural_path: str
) -> None:
    """An identifier cannot be built from an incomplete block position."""
    with pytest.raises(IdentifierError):
        build_block_id(
            document_id=document_id,
            source_resource_path=resource_path,
            structural_path=structural_path,
        )


def test_block_type_parse_rejects_unknown_construct() -> None:
    """An unknown construct is not silently converted into a block."""
    with pytest.raises(UnknownBlockTypeError):
        BlockType.parse("Widget")


def test_block_type_parse_accepts_supported_type() -> None:
    """A supported block type is parsed from its external representation."""
    assert BlockType.parse("PageBreak") is BlockType.PAGE_BREAK


def test_language_note_id_uses_separate_namespace() -> None:
    """A language note identifier does not clash with the book's original footnotes."""
    assert build_language_note_id(chapter_number=3, ordinal=1) == "ln-ch03-001"


@pytest.mark.parametrize(
    ("chapter_number", "ordinal"),
    [
        pytest.param(0, 1, id="zero-chapter"),
        pytest.param(3, 0, id="zero-ordinal"),
    ],
)
def test_language_note_id_rejects_non_positive_numbers(
    chapter_number: int, ordinal: int
) -> None:
    """Chapter and note numbers are 1-based."""
    with pytest.raises(IdentifierError):
        build_language_note_id(chapter_number=chapter_number, ordinal=ordinal)
