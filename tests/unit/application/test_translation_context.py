"""Контекст запроса обработки группы блоков (P3.T6, PRD § 13, ALG-04)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from processing_fakes import (
    BOOK_ID,
    SECTION_ID,
    TARGET_LANGUAGE,
    make_block,
    make_blocks,
    make_response,
    make_unit,
)

from linguary.application.context_builder import ContextBuilder, ContextBuildError
from linguary.application.llm_dto import (
    BlockResultResponse,
    LanguageNoteResponse,
    TranslationResultResponse,
)
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.retry_policy import ResultValidationError
from linguary.domain import (
    BeforeYouReadEntry,
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    DocumentSection,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    InlineAnnotation,
    InlineAnnotationType,
    SourceReference,
    TermStatus,
    TranslationUnit,
    TreatmentPlan,
)

SOURCE_LANGUAGE = "en"

#: Сводка и указания главы: выжимка анализа, которая уходит в каждый запрос обработки.
CHAPTER_SUMMARY = "Mr. Bennet visits the new neighbour."
CHAPTER_INSTRUCTIONS = "Сохранять иронию рассказчика."


def make_section(
    blocks: tuple[DocumentBlock, ...], *, title: str = "Chapter 1", order: int = 0
) -> DocumentSection:
    """Глава книги из переданных блоков."""
    return DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title=title,
        order=order,
        source_resource_path=f"{SECTION_ID}.xhtml",
        blocks=blocks,
    )


def make_document(section: DocumentSection) -> CanonicalDocument:
    """Книга из одной главы."""
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )


def make_analysis(unit: TranslationUnit) -> ChapterAnalysis:
    """Сохранённый анализ главы с планом обработки её блоков."""
    return ChapterAnalysis.create(
        id="analysis-1",
        section_id=SECTION_ID,
        treatment_plan=TreatmentPlan.of_treatments(
            SECTION_ID,
            ((item.block.id, item.treatment) for item in unit.items),
        ),
        translation_instructions=CHAPTER_INSTRUCTIONS,
        chapter_summary=CHAPTER_SUMMARY,
        expected_block_ids=unit.block_ids,
    )


def make_snapshot(
    *, locked: bool = True, suggested: bool = True
) -> GlossarySnapshot:
    """Снимок глоссария запуска с обязательным и необязательным решением."""
    entries = []
    if locked:
        entries.append(
            GlossarySnapshotEntry(
                source_text="Netherfield",
                translation="Незерфилд",
                type=ConceptType.PLACE,
                status=TermStatus.LOCKED,
            )
        )
    if suggested:
        entries.append(
            GlossarySnapshotEntry(
                source_text="Mr. Bennet",
                translation="мистер Беннет",
                type=ConceptType.PERSON,
                status=TermStatus.SUGGESTED,
            )
        )
    return GlossarySnapshot(entries=tuple(entries))


def build_context(
    *,
    treatments: tuple[BlockTreatment, ...] = (
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
    ),
    snapshot: GlossarySnapshot | None = None,
    builder: ContextBuilder | None = None,
):
    """Контекст обработки главы из двух блоков."""
    blocks = make_blocks(len(treatments))
    unit = make_unit(blocks, treatments=treatments)
    section = make_section(blocks)
    return (builder or ContextBuilder()).for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=snapshot if snapshot is not None else make_snapshot(),
    ), unit


def test_request_asks_for_translation_result() -> None:
    """Запрос обработки просит структурированный результат перевода, а не анализ."""
    context, _ = build_context()

    assert context.request.kind is LLMResponseKind.TRANSLATION_RESULT


def test_context_carries_blocks_under_aliases_with_their_treatment() -> None:
    """Блоки уходят модели псевдонимами `b1..bN` вместе с режимом из плана."""
    context, unit = build_context()

    assert context.aliases == ("b1", "b2")
    assert context.block_id_for("b1") == unit.block_ids[0]
    assert "[b1] режим TRANSLATE" in context.request.input_text
    assert "[b2] режим GLOSSARY_ONLY" in context.request.input_text
    assert unit.block_ids[0] not in context.request.input_text


def test_context_carries_chapter_digest() -> None:
    """В запрос идут сводка главы и указания переводчику из сохранённого анализа."""
    context, _ = build_context()

    assert CHAPTER_SUMMARY in context.request.input_text
    assert CHAPTER_INSTRUCTIONS in context.request.input_text


def test_mandatory_glossary_entries_are_separated_from_suggestions() -> None:
    """Зафиксированные решения стоят обязательными указаниями, остальные — подсказками."""
    context, _ = build_context()

    text = context.request.input_text
    mandatory = text.index("Обязательные переводческие решения книги")
    suggestions = text.index("Необязательные подсказки глоссария")
    assert mandatory < suggestions
    assert mandatory < text.index("- Netherfield [place] -> Незерфилд") < suggestions
    assert text.index("- Mr. Bennet [person] -> мистер Беннет") > suggestions


def test_book_without_locked_terms_says_so() -> None:
    """Книга без зафиксированных решений сообщает об этом, а не молчит."""
    context, _ = build_context(snapshot=make_snapshot(locked=False))

    assert "Обязательных переводческих указаний по этой книге нет." in (
        context.request.input_text
    )
    assert "Необязательные подсказки глоссария" in context.request.input_text


def test_suggestions_are_limited_but_mandatory_entries_are_not() -> None:
    """Ограничение контекста режет подсказки и не трогает обязательные указания."""
    snapshot = GlossarySnapshot(
        entries=tuple(
            GlossarySnapshotEntry(
                source_text=f"Term {index}",
                translation=f"Термин {index}",
                type=ConceptType.TERM,
                status=TermStatus.SUGGESTED,
            )
            for index in range(5)
        )
        + make_snapshot(suggested=False).entries
    )

    context, _ = build_context(
        snapshot=snapshot, builder=ContextBuilder(max_accumulated_candidates=2)
    )

    text = context.request.input_text
    assert "Term 0" in text and "Term 1" in text
    assert "Term 2" not in text
    assert "Netherfield" in text


def test_expectation_names_aliases_with_planned_treatments() -> None:
    """Ожидание повтора перечисляет псевдонимы и режимы, которые видела модель (ALG-05)."""
    context, _ = build_context()

    assert context.expectation.target_language == TARGET_LANGUAGE
    assert context.expectation.block_ids == ("b1", "b2")
    assert [block.treatment for block in context.expectation.blocks] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
    ]


def test_response_aliases_are_translated_into_block_ids() -> None:
    """Ответ модели возвращается к стабильным идентификаторам блоков до проверок."""
    context, unit = build_context(treatments=(BlockTreatment.TRANSLATE,))
    payload = make_response(
        (
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Общепризнанная истина гласит следующее.",
            ),
        ),
        language_notes=(
            LanguageNoteResponse(
                source_block_id="b1",
                original_excerpt="a truth",
                note="Ироническая формула зачина.",
            ),
        ),
    )

    restored = context.to_block_ids(payload)

    assert restored.block_ids == unit.block_ids
    assert restored.language_notes[0].source_block_id == unit.block_ids[0]


def test_unknown_alias_in_response_discards_the_unit() -> None:
    """Псевдоним не из запроса отбраковывает группу с ожиданием для повтора (ALG-05)."""
    context, _ = build_context(treatments=(BlockTreatment.TRANSLATE,))
    payload = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="b7",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Чужой блок.",
            ),
        )
    )

    with pytest.raises(ResultValidationError, match="вне запроса") as error:
        context.to_block_ids(payload)

    assert error.value.expectation is context.expectation


def test_note_about_block_outside_the_request_discards_the_unit() -> None:
    """Заметка о блоке вне запроса — тот же чужой идентификатор (PRD § 19.3)."""
    context, _ = build_context(treatments=(BlockTreatment.TRANSLATE,))
    payload = make_response(
        (
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Общепризнанная истина гласит следующее.",
            ),
        ),
        language_notes=(
            LanguageNoteResponse(
                source_block_id="b4",
                original_excerpt="a truth",
                note="Заметка о чужом блоке.",
            ),
        ),
    )

    with pytest.raises(ResultValidationError, match="вне запроса"):
        context.to_block_ids(payload)


def test_unit_of_another_chapter_is_rejected() -> None:
    """Группа блоков другой главы контекста не получает."""
    blocks = make_blocks(1)
    other = make_block(order=0, section_id="section-2")
    section = make_section(blocks)
    unit = make_unit((other,))

    with pytest.raises(ContextBuildError, match="section-2"):
        ContextBuilder().for_translation_unit(
            document=make_document(section),
            section=section,
            unit=unit,
            analysis=make_analysis(make_unit(blocks)),
            glossary=make_snapshot(),
        )


def test_glossary_note_reaches_the_model_with_its_decision() -> None:
    """Переводческое указание идёт модели рядом со своим вариантом перевода.

    Одного варианта перевода мало там, где решение уровня книги влияет на согласование:
    имя приходит из глоссария одинаковым, а род персонажа без указания каждая глава
    выбирает заново (PRD § 11).
    """
    snapshot = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Wren Calder",
                translation="Рен Колдер",
                type=ConceptType.PERSON,
                status=TermStatus.LOCKED,
                note="женский род",
            ),
        )
    )

    context, _ = build_context(snapshot=snapshot)

    assert "Рен Колдер (женский род)" in context.request.input_text
    assert "принято для всей книги" in context.request.input_text


def test_glossary_without_notes_keeps_the_list_clean() -> None:
    """Без указаний список решений не обрастает пояснением про скобки."""
    context, _ = build_context()

    assert "принято для всей книги" not in context.request.input_text


def test_expressions_of_before_you_read_are_not_explained_twice() -> None:
    """Объяснённые до чтения выражения главы модель получает как запрет повтора.

    `Before You Read` собран на анализе, а подсказки и заметки рождаются позже:
    без этого списка один оборот объясняется читателю трижды, что запрещает § 19.3.
    """
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE,))
    section = make_section(blocks)
    analysis = replace(
        make_analysis(unit),
        before_you_read=(
            BeforeYouReadEntry(
                expression="on a good day", explanation="если повезёт"
            ),
        ),
    )

    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert "уже объяснены до начала чтения" in context.request.input_text
    assert "- on a good day" in context.request.input_text


def test_footnote_backlink_does_not_reach_the_model() -> None:
    """Ссылку возврата сноски модель не видит и в перевод не переносит.

    Для книги «↩» — навигация, которую экспортёр восстановит из разметки блока сам.
    Модель же принимает её за текст сноски и копирует значок в перевод, где он
    никуда не ведёт.
    """
    text = "A starglass is a lens ground from marsh quartz. ↩"
    footnote = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.FOOTNOTE,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path=f"{SECTION_ID}.xhtml", structural_path="/body/aside[1]/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.LINK,
                start=text.index("↩"),
                end=len(text),
                target=f"{SECTION_ID}.xhtml#fnref1",
            ),
        ),
    )
    unit = make_unit((footnote,), treatments=(BlockTreatment.TRANSLATE,))
    section = make_section((footnote,))

    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=make_snapshot(),
    )

    assert "marsh quartz." in context.request.input_text
    assert "↩" not in context.request.input_text


def test_link_inside_a_paragraph_survives_for_the_model() -> None:
    """Отсечение хвоста касается только сноски: обычная ссылка абзаца остаётся текстом."""
    text = "Netherfield Park is let at last."
    paragraph = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path=f"{SECTION_ID}.xhtml", structural_path="/body/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.LINK,
                start=text.index("at last."),
                end=len(text),
                target=f"{SECTION_ID}.xhtml#note-1",
            ),
        ),
    )
    unit = make_unit((paragraph,), treatments=(BlockTreatment.TRANSLATE,))
    section = make_section((paragraph,))

    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=make_snapshot(),
    )

    assert "at last." in context.request.input_text
