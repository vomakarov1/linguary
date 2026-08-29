"""Фабрики блоков, групп и ответов обработки для unit-тестов (P3.T6).

Собирают минимально валидные объекты, чтобы тест описывал только то, что проверяет.
Модуль намеренно без префикса `test_`: иначе pytest соберёт его как тестовый.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from linguary.adapters.config import CheckSettings
from linguary.application.llm_dto import (
    BlockResultResponse,
    GlossaryHintResponse,
    LanguageNoteResponse,
    TranslationResultResponse,
)
from linguary.application.validators import TranslationResultValidator
from linguary.domain import (
    BlockTreatment,
    BlockType,
    DocumentBlock,
    SourceReference,
    SupportStatus,
    TranslationUnit,
    TreatmentPlan,
)

BOOK_ID = "book-1"
SECTION_ID = "section-1"
TARGET_LANGUAGE = "ru"

#: Абзац оригинала и его законченный русский перевод — тексты, на которых проверки молчат.
SOURCE_TEXT = "It is a truth universally acknowledged."
TRANSLATED_TEXT = "Общепризнанная истина гласит следующее."


def make_block(
    *,
    order: int,
    text: str = SOURCE_TEXT,
    type: BlockType = BlockType.PARAGRAPH,
    parent_id: str | None = None,
    support_status: SupportStatus = SupportStatus.SUPPORTED,
    section_id: str = SECTION_ID,
) -> DocumentBlock:
    """Блок главы со стабильным идентификатором."""
    return DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=section_id,
        type=type,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path=f"{section_id}.xhtml",
            structural_path=f"/body/p[{order + 1}]",
        ),
        support_status=support_status,
        parent_id=parent_id,
    )


def make_blocks(
    count: int, *, text: str = SOURCE_TEXT, section_id: str = SECTION_ID
) -> tuple[DocumentBlock, ...]:
    """Несколько одинаковых абзацев главы подряд."""
    return tuple(
        make_block(order=index, text=text, section_id=section_id)
        for index in range(count)
    )


def make_plan(
    blocks: Iterable[DocumentBlock],
    *,
    treatments: Sequence[BlockTreatment] | None = None,
    section_id: str = SECTION_ID,
) -> TreatmentPlan:
    """План главы: один режим на все блоки либо режим для каждого по порядку."""
    ordered = tuple(blocks)
    chosen = (
        tuple(treatments)
        if treatments is not None
        else (BlockTreatment.TRANSLATE,) * len(ordered)
    )
    return TreatmentPlan.of_treatments(
        section_id, zip((block.id for block in ordered), chosen, strict=True)
    )


def make_unit(
    blocks: Iterable[DocumentBlock],
    *,
    treatments: Sequence[BlockTreatment] | None = None,
    id: str = "unit-1",
) -> TranslationUnit:
    """Группа блоков с режимами из плана."""
    ordered = tuple(blocks)
    return TranslationUnit.of_blocks(
        id=id, blocks=ordered, plan=make_plan(ordered, treatments=treatments)
    )


def make_hint(
    *, original: str = "entailed", translation: str = "заповедный", note: str | None = None
) -> GlossaryHintResponse:
    """Словарная подсказка в ответе модели."""
    return GlossaryHintResponse(original=original, translation=translation, note=note)


def make_block_result(
    block: DocumentBlock,
    *,
    treatment: BlockTreatment = BlockTreatment.TRANSLATE,
    translated_text: str | None = TRANSLATED_TEXT,
    glossary_hints: tuple[GlossaryHintResponse, ...] = (),
) -> BlockResultResponse:
    """Результат одного блока в ответе модели: по умолчанию полный перевод."""
    return BlockResultResponse(
        source_block_id=block.id,
        treatment=treatment,
        translated_text=translated_text,
        glossary_hints=glossary_hints,
    )


def make_note(
    block: DocumentBlock,
    *,
    original_excerpt: str = "a truth universally acknowledged",
    note: str = "Ироническая формула зачина.",
    translated_excerpt: str | None = "Общепризнанная истина",
) -> LanguageNoteResponse:
    """Языковая заметка ответа, привязанная к блоку."""
    return LanguageNoteResponse(
        source_block_id=block.id,
        original_excerpt=original_excerpt,
        note=note,
        translated_excerpt=translated_excerpt,
    )


def make_response(
    blocks: Iterable[BlockResultResponse],
    *,
    language_notes: Iterable[LanguageNoteResponse] = (),
) -> TranslationResultResponse:
    """Ответ модели на обработку группы блоков."""
    return TranslationResultResponse(
        blocks=tuple(blocks), language_notes=tuple(language_notes)
    )


def translated_response(unit: TranslationUnit) -> TranslationResultResponse:
    """Корректный ответ на всю группу: полный перевод каждого блока по плану."""
    return make_response(
        make_block_result(
            item.block,
            treatment=item.treatment,
            translated_text=TRANSLATED_TEXT
            if item.treatment.requires_translation
            else None,
            glossary_hints=(make_hint(),)
            if item.treatment.requires_glossary_hints
            else (),
        )
        for item in unit.items
    )


def make_validator(checks: CheckSettings | None = None) -> TranslationResultValidator:
    """Валидатор с порогами приложения: умолчания живут в `CheckSettings`, не здесь."""
    settings = checks or CheckSettings()
    return TranslationResultValidator(
        target_language_min_share=settings.target_language_min_share,
        language_check_min_length=settings.language_check_min_length,
        repetition_ngram_size=settings.repetition_ngram_size,
        repetition_min_repeats=settings.repetition_min_repeats,
    )


def note_ids() -> Callable[[], str]:
    """Предсказуемые идентификаторы заметок: `ln-ch01-001`, `ln-ch01-002`, ..."""
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"ln-ch01-{counter:03d}"

    return next_id
