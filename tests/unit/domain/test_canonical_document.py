"""Инварианты канонического документа, разделов и блоков (PRD § 6-7)."""

from __future__ import annotations

import pytest
from domain_factories import (
    DOCUMENT_ID,
    RESOURCE_PATH,
    SECTION_ID,
    make_block,
    make_blocks,
    make_document,
    make_section,
)

from linguary.domain import (
    BlockType,
    BookMetadata,
    CanonicalDocument,
    DocumentBlock,
    DocumentStructureError,
    InlineAnnotation,
    InlineAnnotationType,
    SourceReference,
    SupportStatus,
    UnknownBlockError,
)


def test_document_returns_blocks_in_reading_order() -> None:
    """Блоки книги перечисляются в порядке чтения, а не в порядке создания."""
    blocks = make_blocks(3)
    section = make_section(reversed(blocks))
    document = make_document([section])
    assert tuple(block.order for block in document.blocks()) == (0, 1, 2)


def test_document_finds_block_by_id() -> None:
    """Блок книги доступен по стабильному идентификатору."""
    blocks = make_blocks(2)
    document = make_document([make_section(blocks)])
    assert document.block(blocks[1].id) is blocks[1]


def test_document_rejects_unknown_block_id() -> None:
    """Обращение к чужому блоку отклоняется, а не возвращает пустой результат."""
    document = make_document([make_section(make_blocks(1))])
    with pytest.raises(UnknownBlockError):
        document.block("missing-block")


def test_document_rejects_duplicate_section_order() -> None:
    """Две главы не могут занимать одно место в порядке чтения."""
    with pytest.raises(DocumentStructureError):
        make_document(
            [
                make_section(section_id="section-1", order=0),
                make_section(section_id="section-2", order=0),
            ]
        )


def test_document_rejects_same_source_and_target_language() -> None:
    """Книга переводится с одного языка на другой."""
    with pytest.raises(DocumentStructureError):
        CanonicalDocument(
            id=DOCUMENT_ID,
            metadata=BookMetadata(title="Nineteen Eighty-Four"),
            source_language="en",
            target_language="en",
        )


def test_section_rejects_block_from_other_section() -> None:
    """Раздел не принимает блок чужой главы."""
    foreign = make_block(section_id="section-2")
    with pytest.raises(DocumentStructureError):
        make_section([foreign])


def test_section_rejects_duplicate_block_order() -> None:
    """Порядковые номера блоков внутри главы уникальны."""
    first = make_block(order=0, structural_path="/body/p[1]")
    second = make_block(order=0, structural_path="/body/p[2]")
    with pytest.raises(DocumentStructureError):
        make_section([first, second])


def test_section_rejects_missing_parent_block() -> None:
    """Вложенность указывает на блок той же главы."""
    child = DocumentBlock(
        id="child",
        section_id=SECTION_ID,
        type=BlockType.LIST_ITEM,
        order=0,
        text="первый пункт",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/ul[1]/li[1]"
        ),
        parent_id="missing-parent",
    )
    with pytest.raises(DocumentStructureError):
        make_section([child])


def test_section_accepts_nested_block() -> None:
    """Пункт списка ссылается на родительский блок той же главы."""
    parent = make_block(order=0, type=BlockType.OTHER)
    child = DocumentBlock(
        id="child",
        section_id=SECTION_ID,
        type=BlockType.LIST_ITEM,
        order=1,
        text="первый пункт",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/ul[1]/li[1]"
        ),
        parent_id=parent.id,
    )
    assert make_section([parent, child]).block(child.id).parent_id == parent.id


def test_block_rejects_annotation_outside_text() -> None:
    """Разметка не может выходить за пределы текста блока."""
    with pytest.raises(DocumentStructureError):
        DocumentBlock(
            id="block-1",
            section_id=SECTION_ID,
            type=BlockType.PARAGRAPH,
            order=0,
            text="короткий",
            source_reference=SourceReference(
                resource_path=RESOURCE_PATH, structural_path="/body/p[1]"
            ),
            inline_annotations=(
                InlineAnnotation(type=InlineAnnotationType.ITALIC, start=0, end=99),
            ),
        )


def test_block_keeps_markup_separate_from_text() -> None:
    """Текст блока хранится без XHTML, а разметка — смещениями в этом тексте."""
    block = DocumentBlock(
        id="block-1",
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=0,
        text="Winston Smith",
        source_reference=SourceReference(
            resource_path=RESOURCE_PATH, structural_path="/body/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(type=InlineAnnotationType.ITALIC, start=0, end=7),
        ),
    )
    assert "<" not in block.text
    assert block.inline_annotations[0].end == 7


def test_link_annotation_requires_target() -> None:
    """Ссылка и сноска бессмысленны без адреса назначения."""
    with pytest.raises(DocumentStructureError):
        InlineAnnotation(type=InlineAnnotationType.LINK, start=0, end=3)


def test_annotation_rejects_empty_range() -> None:
    """Разметка охватывает хотя бы один символ."""
    with pytest.raises(DocumentStructureError):
        InlineAnnotation(type=InlineAnnotationType.BOLD, start=4, end=4)


def test_block_rejects_fatal_support_status() -> None:
    """Конструкция со статусом FATAL блоком не становится: импорт невозможен."""
    with pytest.raises(DocumentStructureError):
        DocumentBlock(
            id="block-1",
            section_id=SECTION_ID,
            type=BlockType.OTHER,
            order=0,
            text="",
            source_reference=SourceReference(
                resource_path=RESOURCE_PATH, structural_path="/body/div[1]"
            ),
            support_status=SupportStatus.FATAL,
        )
