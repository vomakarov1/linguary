"""Processing request context for a block unit (P3.T6, PRD § 13, ALG-04)."""

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

from linguary.application.context_builder import (
    ContextBuilder,
    ContextBuildError,
    notes_budget,
)
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
    GlossaryCandidate,
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

#: Chapter summary and instructions: analysis digest passed to each processing request.
CHAPTER_SUMMARY = "Mr. Bennet visits the new neighbour."
CHAPTER_INSTRUCTIONS = "Сохранять иронию рассказчика."


def make_section(
    blocks: tuple[DocumentBlock, ...], *, title: str = "Chapter 1", order: int = 0
) -> DocumentSection:
    """Book chapter from the provided blocks."""
    return DocumentSection(
        id=SECTION_ID,
        document_id=BOOK_ID,
        title=title,
        order=order,
        source_resource_path=f"{SECTION_ID}.xhtml",
        blocks=blocks,
    )


def make_document(section: DocumentSection) -> CanonicalDocument:
    """Single-chapter book."""
    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        sections=(section,),
    )


def make_analysis(unit: TranslationUnit) -> ChapterAnalysis:
    """Saved chapter analysis with block treatment plan."""
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
    """Launch glossary snapshot with mandatory and optional decisions."""
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
    """Processing context for a two-block chapter."""
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
    """Processing request asks for a structured translation result rather than an analysis."""
    context, _ = build_context()

    assert context.request.kind is LLMResponseKind.TRANSLATION_RESULT


def test_notes_budget_proportional_to_unit_share() -> None:
    assert notes_budget(unit_chars=250, chapter_chars=1000, per_chapter=8) == 2


def test_notes_budget_at_least_one() -> None:
    assert notes_budget(unit_chars=1, chapter_chars=1000, per_chapter=0) == 1


def test_translation_input_states_notes_budget() -> None:
    blocks = make_blocks(2)
    section = make_section(blocks)
    unit = make_unit(blocks[:1], treatments=(BlockTreatment.TRANSLATE,))
    builder = ContextBuilder(language_notes_per_chapter=8)
    context = builder.for_translation_unit(
        document=make_document(section), section=section, unit=unit,
        analysis=make_analysis(unit), glossary=make_snapshot(),
    )
    expected = notes_budget(
        unit_chars=len(blocks[0].text),
        chapter_chars=sum(len(block.text) for block in blocks),
        per_chapter=8,
    )
    assert f"В этом запросе — не больше {expected} языковых заметок" in context.request.input_text
    regenerated = builder.for_block_regeneration(
        document=make_document(section), section=section, unit=unit,
        analysis=make_analysis(unit), glossary=make_snapshot(),
    )
    assert (
        f"В этом запросе — не больше {expected} языковых заметок"
        in regenerated.request.input_text
    )


def test_translation_instructions_require_preserving_character_speech() -> None:
    instructions = build_context()[0].request.instructions
    assert "ломаную грамматику" in instructions
    assert "акцент" in instructions
    assert "диалект" in instructions
    assert "не выправляй" in instructions
    assert "если перевод её не передаёт" in instructions


def test_translation_instructions_forbid_plot_retelling_notes() -> None:
    instructions = build_context()[0].request.instructions
    assert "не пересказывает сюжет" in instructions
    assert "уже переданы переводом" in instructions


def test_context_carries_blocks_under_aliases_with_their_treatment() -> None:
    """Blocks are sent to the model under aliases `b1..bN` together with their planned treatment."""
    context, unit = build_context()

    assert context.aliases == ("b1", "b2")
    assert context.block_id_for("b1") == unit.block_ids[0]
    assert "[b1] режим TRANSLATE" in context.request.input_text
    assert "[b2] режим GLOSSARY_ONLY" in context.request.input_text
    assert unit.block_ids[0] not in context.request.input_text


def test_context_carries_chapter_digest() -> None:
    """The request includes the chapter summary and translator instructions from the saved analysis."""
    context, _ = build_context()

    assert CHAPTER_SUMMARY in context.request.input_text
    assert CHAPTER_INSTRUCTIONS in context.request.input_text


def test_chapter_digest_has_no_block_labels() -> None:
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE,))
    section = make_section(blocks)
    analysis = replace(make_analysis(unit), translation_instructions="Сохранить паузу в блоке b3.")
    context = ContextBuilder().for_translation_unit(
        document=make_document(section), section=section, unit=unit,
        analysis=analysis, glossary=make_snapshot(),
    )
    assert "b3" not in context.request.input_text


def test_mandatory_glossary_entries_are_separated_from_suggestions() -> None:
    """Locked decisions appear as mandatory directives, others as hints."""
    context, _ = build_context()

    text = context.request.input_text
    mandatory = text.index("Обязательные переводческие решения книги")
    suggestions = text.index("Необязательные подсказки глоссария")
    assert mandatory < suggestions
    assert mandatory < text.index("- Netherfield [place] -> Незерфилд") < suggestions
    assert text.index("- Mr. Bennet [person] -> мистер Беннет") > suggestions


def test_book_without_locked_terms_says_so() -> None:
    """A book without locked terms explicitly states so rather than remaining silent."""
    context, _ = build_context(snapshot=make_snapshot(locked=False))

    assert "Обязательных переводческих указаний по этой книге нет." in (
        context.request.input_text
    )
    assert "Необязательные подсказки глоссария" in context.request.input_text


def test_suggestions_are_limited_but_mandatory_entries_are_not() -> None:
    """Context capping curtails suggestions while leaving mandatory directives intact."""
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
    """Retry expectation lists aliases and treatments seen by the model (ALG-05)."""
    context, _ = build_context()

    assert context.expectation.target_language == TARGET_LANGUAGE
    assert context.expectation.block_ids == ("b1", "b2")
    assert [block.treatment for block in context.expectation.blocks] == [
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
    ]


def test_response_aliases_are_translated_into_block_ids() -> None:
    """Model response aliases are mapped back to stable block IDs before validations."""
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
    """An alias not from the request discards the unit with retry expectation (ALG-05)."""
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
    """A note referring to a block outside the request is likewise a foreign ID (PRD § 19.3)."""
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
    """A block unit from another chapter does not receive context."""
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
    """Translation instruction is provided to the model alongside its translation variant.

    A translation variant alone is insufficient when a book-level decision affects agreement:
    a name comes from the glossary consistently, while character gender without instruction
    would be picked anew by each chapter (PRD § 11).
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
    """Without instructions, the decision list does not acquire notes about parentheses."""
    context, _ = build_context()

    assert "принято для всей книги" not in context.request.input_text


def test_expressions_of_before_you_read_are_not_explained_twice() -> None:
    """Expressions explained before reading are passed to the model as a duplicate prevention rule.

    `Before You Read` is assembled during analysis, while hints and notes are created later:
    without this list, a single turn of phrase could be explained three times, violating § 19.3.
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
    """The model does not see a footnote backlink and does not transfer it to translation.

    For the book, "↩" is navigation that the exporter reconstructs from block markup.
    The model treats it as footnote text and copies the symbol to translation, where it leads nowhere.
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


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("chapter2.xhtml#nref1", id="sibling-file"),
        pytest.param("../Text/chapter2.xhtml#nref1", id="other-folder"),
    ],
)
def test_backlink_of_notes_page_into_another_file_does_not_reach_the_model(
    target: str,
) -> None:
    """Backlinks from a notes page into another file are stripped the same way (D-2).

    On a dedicated notes page, the backlink always points to the chapter where the footnote
    reference is located: otherwise the "↩" symbol reaches the model and stays dead in the translation.
    """
    text = "Duskmere was founded in a single rainy spring. ↩"
    footnote = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.FOOTNOTE,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="notes.xhtml", structural_path="/body/aside[1]/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.LINK,
                start=text.index("↩"),
                end=len(text),
                target=target,
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

    assert "a single rainy spring." in context.request.input_text
    assert "↩" not in context.request.input_text


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("https://example.org/duskmere#history", id="external-url"),
        pytest.param("chapter2.xhtml", id="file-without-anchor"),
    ],
)
def test_footnote_ending_with_an_outside_link_keeps_its_text(target: str) -> None:
    """A link to an external URL or to a file without an anchor is footnote text, not a backlink."""
    text = "See the town chronicle."
    footnote = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.FOOTNOTE,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="notes.xhtml", structural_path="/body/aside[1]/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.LINK,
                start=text.index("town chronicle."),
                end=len(text),
                target=target,
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

    assert "See the town chronicle." in context.request.input_text


def test_link_inside_a_paragraph_survives_for_the_model() -> None:
    """Stripping the tail applies only to footnotes: normal paragraph links remain text."""
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


def _paragraph_with_noteref(
    text: str = "Nobody liked her face.1 It was a pity.", *, order: int = 0
) -> DocumentBlock:
    """Paragraph with a single footnote reference '1' mid-text, as captured by import."""
    start = text.index("1")
    return DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path=f"{SECTION_ID}.xhtml", structural_path=f"/body/p[{order + 1}]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=start,
                end=start + 1,
                target=f"{SECTION_ID}.xhtml#fn1",
            ),
        ),
    )


def test_footnote_reference_reaches_the_model_as_a_marker() -> None:
    """The model sees a footnote reference number as a marker rather than text digits (QA-08)."""
    paragraph = _paragraph_with_noteref()
    unit = make_unit((paragraph,), treatments=(BlockTreatment.TRANSLATE,))
    section = make_section((paragraph,))

    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=make_snapshot(),
    )

    assert "Nobody liked her face.[^1] It was a pity." in context.request.input_text
    assert "face.1" not in context.request.input_text
    assert "[^1]" in context.request.instructions


def test_footnote_reference_is_dropped_from_neighbour_context() -> None:
    """Footnote reference is completely dropped from neighbouring paragraphs."""
    neighbour = _paragraph_with_noteref("Before the storm 1 came, they slept.")
    target = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=1,
        text="The storm passed by morning.",
        source_reference=SourceReference(
            resource_path=f"{SECTION_ID}.xhtml", structural_path="/body/p[2]"
        ),
    )
    section = make_section((neighbour, target))
    unit = make_unit((target,), treatments=(BlockTreatment.TRANSLATE,))

    context = ContextBuilder().for_block_regeneration(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=make_snapshot(),
    )

    assert "Before the storm came, they slept." in context.request.input_text


def test_translation_context_carries_expressions_of_before_you_read() -> None:
    """'Before You Read' expressions reach the scenario for duplicate filtering (QA-03)."""
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE,))
    section = make_section(blocks)
    analysis = replace(
        make_analysis(unit),
        before_you_read=(
            BeforeYouReadEntry(expression=" on a good day ", explanation="если повезёт"),
        ),
    )

    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert context.explained_expressions == ("on a good day",)


def test_notes_page_has_no_before_you_read_to_protect() -> None:
    """A notes page has no 'Before You Read' in the book: its hints have nothing to repeat."""
    note = make_block(order=0, type=BlockType.FOOTNOTE, text="1 A plain face, of course.")
    unit = make_unit((note,), treatments=(BlockTreatment.TRANSLATE,))
    section = make_section((note,))
    analysis = replace(
        make_analysis(unit),
        before_you_read=(
            BeforeYouReadEntry(expression="of course", explanation="конечно"),
        ),
    )

    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert context.explained_expressions == ()
    assert "уже объяснены до начала чтения" not in context.request.input_text


#: Chapter 2 instructions from a live run: the model assigned translations and gender in them
#: contrary to prompt prohibitions, and the user subsequently corrected both in the glossary (D-1).
STALE_CHAPTER_INSTRUCTIONS = "\n".join(
    (
        "1. Сохранять иронию рассказчика.",
        "2. Топоним «Duskmere» переводить строго по глоссарию — «Даскмир».",
        "3. Рен Колдер — сохранять мужской род согласно принятому решению.",
        "4. Рен Колдер говорит отрывисто, короткими фразами; "
        "к Селене обращается «госпожа».",
    )
)


def user_decisions_snapshot(status: TermStatus = TermStatus.APPROVED) -> GlossarySnapshot:
    """Launch snapshot with user decisions on a name and place."""
    return GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Wren Calder",
                translation="Рен Колдер",
                type=ConceptType.PERSON,
                status=status,
                note="Женский род во всех главах",
            ),
            GlossarySnapshotEntry(
                source_text="Duskmere",
                translation="Даскмер",
                type=ConceptType.PLACE,
                status=status,
            ),
        )
    )


def context_with_instructions(
    instructions: str, snapshot: GlossarySnapshot
) -> str:
    """Processing request text for a chapter with instructions and analysis candidates."""
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE,))
    section = make_section(blocks)
    analysis = make_analysis(unit)
    analysis = replace(
        analysis,
        translation_instructions=instructions,
        glossary_candidates=(
            GlossaryCandidate.create(
                id="candidate-1",
                analysis_id=analysis.id,
                source_text="Duskmere",
                type=ConceptType.PLACE,
                suggested_translation="Даскмир",
                confidence=0.9,
                model_flagged_review=False,
                confidence_threshold=0.5,
            ),
        ),
    )
    return ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=snapshot,
    ).request.input_text


@pytest.mark.parametrize("status", [TermStatus.APPROVED, TermStatus.LOCKED])
def test_user_decision_reaches_the_model_as_mandatory(status: TermStatus) -> None:
    """Accepted, corrected, and locked decisions reach the model as mandatory (D-1)."""
    text = context_with_instructions(CHAPTER_INSTRUCTIONS, user_decisions_snapshot(status))

    mandatory = text.index("Обязательные переводческие решения книги")
    assert mandatory < text.index("- Duskmere [place] -> Даскмер")
    assert mandatory < text.index("- Wren Calder [person] -> Рен Колдер")
    assert "Необязательные подсказки глоссария" not in text
    assert "Обязательных переводческих указаний по этой книге нет." not in text


def test_chapter_instructions_contradicting_user_decisions_are_dropped() -> None:
    """Chapter instructions with a conflicting translation or gender do not override glossary decisions.

    Live run, defect D-1: chapter 2 instructions stated the rejected name ("Daskmir") and masculine gender,
    even though the user corrected the place to "Daskmer" and specified feminine gender,
    causing chapter 2 to yield the rejected form and masculine verbs.
    """
    text = context_with_instructions(STALE_CHAPTER_INSTRUCTIONS, user_decisions_snapshot())

    assert "Даскмир" not in text
    assert "мужской род" not in text
    assert "1. Сохранять иронию рассказчика." in text
    assert "говорит отрывисто" in text
    assert "Рен Колдер (Женский род во всех главах)" in text


def test_prompt_puts_glossary_decisions_above_chapter_instructions() -> None:
    """The context explicitly states that glossary decisions override chapter instructions."""
    text = context_with_instructions(CHAPTER_INSTRUCTIONS, user_decisions_snapshot())

    assert "сильнее указаний переводчику главы" in text


def test_skipped_decisions_leave_chapter_instructions_intact() -> None:
    """Skipped items remain hints and do not displace chapter instructions."""
    text = context_with_instructions(
        STALE_CHAPTER_INSTRUCTIONS, user_decisions_snapshot(TermStatus.SUGGESTED)
    )

    assert STALE_CHAPTER_INSTRUCTIONS in text
    assert "Обязательных переводческих указаний по этой книге нет." in text


def test_footnote_that_is_a_link_entirely_keeps_its_text() -> None:
    """A footnote consisting entirely of a link into the book retains its text: it is not a backlink."""
    text = "See chapter two."
    footnote = DocumentBlock.create(
        document_id=BOOK_ID,
        section_id=SECTION_ID,
        type=BlockType.FOOTNOTE,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="notes.xhtml", structural_path="/body/aside[1]/p[1]"
        ),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.LINK,
                start=0,
                end=len(text),
                target="chapter2.xhtml#start",
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

    assert "See chapter two." in context.request.input_text


def test_glossary_note_naming_rejected_variant_of_other_concept_is_dropped() -> None:
    """A glossary item note does not introduce a rejected variant into the context (N-2).

    Live run: the note for `marsh gate`, accepted via "Accept All", literally went into
    the mandatory decisions block with the rejected variant next to "Duskmere -> Daskmer".
    """
    decisions = user_decisions_snapshot()
    snapshot = GlossarySnapshot(
        entries=(
            *decisions.entries,
            GlossarySnapshotEntry(
                source_text="marsh gate",
                translation="болотные ворота",
                type=ConceptType.FICTIONAL_CONCEPT,
                status=TermStatus.APPROVED,
                note=(
                    "Функциональный пост стража болота при въезде в Даскмир; "
                    "уточнить устойчивый вариант перевода для всей книги."
                ),
            ),
            GlossarySnapshotEntry(
                source_text="boardwalk",
                translation="деревянный настил",
                type=ConceptType.TERM,
                status=TermStatus.SUGGESTED,
                note="Настил ведёт через болото к Даскмиру.",
            ),
        )
    )

    text = context_with_instructions(CHAPTER_INSTRUCTIONS, snapshot)

    assert "Даскмир" not in text
    assert (
        "- marsh gate [fictional_concept] -> болотные ворота "
        "(уточнить устойчивый вариант перевода для всей книги.)"
    ) in text
    assert "- boardwalk [term] -> деревянный настил\n" in text
    assert "Рен Колдер (Женский род во всех главах)" in text


def test_rejected_variant_is_known_from_other_chapters_of_the_book() -> None:
    """Rejected variants are taken from all book candidates, not just this chapter (D-4).

    Third live run: analysis of chapter 2 did not return the Duskmere candidate (the concept
    was already in chapter 1 decisions), and chapter 2 request received the entire note for
    `the marsh gate` with the rejected form and the chapter instruction mentioning it without quotes.
    """
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE,))
    section = make_section(blocks, title="Chapter 2", order=1)
    analysis = replace(
        make_analysis(unit),
        translation_instructions=(
            "1. Сохранять иронию рассказчика.\n"
            "8. Термины из глоссария (звёздное стекло, болотный кварц, болотные огни, "
            "Даскмир) использовать единообразно согласно принятым решениям.\n"
            "9. Путь в Даскмера не сокращать."
        ),
    )
    snapshot = GlossarySnapshot(
        entries=(
            *user_decisions_snapshot().entries,
            GlossarySnapshotEntry(
                source_text="the marsh gate",
                translation="болотные ворота",
                type=ConceptType.PLACE,
                status=TermStatus.APPROVED,
                note="Сторожевой пост при въезде в Даскмир. Множественное число, склоняется.",
            ),
        )
    )
    first_chapter_candidate = GlossaryCandidate.create(
        id="candidate-chapter-1",
        analysis_id="analysis-chapter-1",
        source_text="Duskmere",
        type=ConceptType.PLACE,
        suggested_translation="Даскмир",
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=0.5,
    )

    for build in (
        ContextBuilder().for_translation_unit,
        ContextBuilder().for_block_regeneration,
    ):
        text = build(
            document=make_document(section),
            section=section,
            unit=unit,
            analysis=analysis,
            glossary=snapshot,
            book_candidates=(first_chapter_candidate,),
        ).request.input_text

        assert "Даскмир" not in text
        assert (
            "- the marsh gate [place] -> болотные ворота "
            "(Множественное число, склоняется.)"
        ) in text
        assert "1. Сохранять иронию рассказчика." in text
        assert "9. Путь в Даскмера не сокращать." in text


@pytest.mark.parametrize("method", ["for_translation_unit", "for_block_regeneration"])
def test_unit_and_regeneration_include_full_byr_explanations(method: str) -> None:
    blocks = make_blocks(1)
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE,))
    section = make_section(blocks)
    analysis = replace(make_analysis(unit), before_you_read=(
        BeforeYouReadEntry(expression=" made a grab at ", explanation=" попытался схватить "),
        BeforeYouReadEntry(expression="sheath/armour", explanation="ножны скрывают, броня защищает"),
    ))
    context = getattr(ContextBuilder(), method)(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )
    assert context.explained_entries == (
        ("made a grab at", "попытался схватить"),
        ("sheath/armour", "ножны скрывают, броня защищает"),
    )
    assert context.explained_expressions == ("made a grab at", "sheath/armour")
    for expression, explanation in context.explained_entries:
        assert f"- {expression}: {explanation}" in context.request.input_text


def test_notes_only_context_has_no_byr_entries() -> None:
    block = make_block(order=0, type=BlockType.FOOTNOTE, text="A plain face.")
    unit = make_unit((block,), treatments=(BlockTreatment.TRANSLATE,))
    section = make_section((block,))
    analysis = replace(make_analysis(unit), before_you_read=(BeforeYouReadEntry(expression="plain", explanation="простой"),))
    for method in ("for_translation_unit", "for_block_regeneration"):
        context = getattr(ContextBuilder(), method)(
            document=make_document(section),
            section=section,
            unit=unit,
            analysis=analysis,
            glossary=make_snapshot(),
        )
        assert context.explained_entries == ()
        assert "- plain: простой" not in context.request.input_text


def test_note_with_new_language_fact_is_allowed() -> None:
    instructions = build_context()[0].request.instructions
    assert "новый языковой или культурный факт того же отрывка допустим" in instructions
    assert "made/make a grab at" in instructions
    assert "sheath/armour" in instructions


def test_mixed_cultural_note_cannot_retell_plot() -> None:
    assert "культурная справка не оправдывает пересказ сюжета" in build_context()[0].request.instructions


def test_saved_speech_instruction_is_subordinate_to_natural_rendering() -> None:
    saved = "Имитировать ломаную речь Ван Хелсинга во всех репликах."
    text = context_with_instructions(saved, make_snapshot())
    assert saved in text
    assert "сохранённые указания главы подчинены этому принципу" in text
    instructions = build_context()[0].request.instructions
    assert "без искусственной имитации" in instructions
    assert "не переноси приёмы одного персонажа на остальных" in instructions


@pytest.mark.parametrize("method", ["for_translation_unit", "for_block_regeneration"])
def test_regeneration_preserves_own_authorial_tail_and_line_count(method: str) -> None:
    target = make_block(order=1, text='“I go,” he said.\nAnd with that he left.')
    neighbour = make_block(order=0, text="The room was silent.")
    section = make_section((neighbour, target))
    unit = make_unit((target,), treatments=(BlockTreatment.TRANSLATE,))
    context = getattr(ContextBuilder(), method)(document=make_document(section), section=section,
        unit=unit, analysis=make_analysis(unit), glossary=make_snapshot())
    assert target.text in context.request.input_text
    for requirement in ("ровно исходное число непустых строк", "весь собственный текст блока", "авторское окончание", "не удаляй окончание ради числа строк"):
        assert requirement in context.request.instructions


def test_note_claim_requires_observable_translation_feature() -> None:
    instructions = build_context()[0].request.instructions
    assert "наблюдаемое основание в самом переводе" in instructions
    assert "может описывать особенность оригинала" in instructions
    assert "Гладкий перевод без этой особенности" in instructions


def test_v2_plan_requires_explicit_profile_and_owned_anchors():
    from linguary.domain.content_plan import ContentPlan, Difficulty, SourceAnchor, RequiredPoint
    blocks = (make_block(order=0, text='Echo calls.'), make_block(order=1, text='The answer echoes.'))
    section = make_section(blocks)
    unit = make_unit((blocks[0],))
    d = Difficulty('difficulty', (SourceAnchor(blocks[0].id, 0, 4, 'Echo'), SourceAnchor(blocks[1].id, 4, 10, 'answer')), blocks[0].id, 'semantic_relation', 'critical', 'explanation', (RequiredPoint('p', 'Explain the link'),))
    analysis = replace(make_analysis(unit), content_plan=ContentPlan(difficulties=(d,)))
    args = dict(document=make_document(section), section=section, unit=unit, analysis=analysis, glossary=make_snapshot())
    with pytest.raises(ContextBuildError):
        ContextBuilder().for_translation_unit(**args)
    builder = ContextBuilder(content_plan_profile='content-plan-v2')
    for context in (builder.for_translation_unit(**args), builder.for_block_regeneration(**args)):
        assert context.request.contract_version == 2
        text = context.request.input_text
        assert 'answer' in text and 'Explain the link' in text
        assert blocks[1].id not in text
        assert 'related-source' in text
        assert 'Решения глоссария' in context.request.instructions
    legacy = builder.for_translation_unit(**{**args, 'analysis': make_analysis(unit)})
    assert legacy.request.contract_version == 1


def test_translation_aliases_preserve_content_evidence():
    from linguary.application.llm_dto import HelpEvidenceResponse
    blocks = make_blocks(1, text='Echo calls.')
    section = make_section(blocks)
    unit = make_unit(blocks)
    context = ContextBuilder().for_translation_unit(document=make_document(section), section=section, unit=unit, analysis=make_analysis(unit), glossary=make_snapshot())
    e = HelpEvidenceResponse('d', 'p', 'language_note.note', 'b1', 0, 'Explanation')
    payload = TranslationResultResponse(content_evidence=(e,))
    assert context.to_block_ids(payload).content_evidence[0].source_block_id == blocks[0].id
    with pytest.raises(ResultValidationError):
        context.to_block_ids(replace(payload, content_evidence=(replace(e, source_block_id='b2'),)))
