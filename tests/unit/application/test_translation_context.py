"""Processing request context for a block unit (P3.T6, PRD § 13, ALG-04)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from processing_fakes import (
    BOOK_ID,
    SECTION_ID,
    SOURCE_TEXT,
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
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy
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


@pytest.mark.parametrize("bounds", ((0, 3), (0, 2)))
def test_translation_scene_context_uses_chapter_order_and_full_range_overlap(bounds) -> None:
    from linguary.domain.content_plan import Scene

    blocks = tuple(replace(make_block(order=i), id=block_id)
                   for i, block_id in enumerate(("z", "a", "m", "b", "y")))
    section = make_section(tuple(reversed(blocks)))
    unit = make_unit(blocks[1:3] if bounds == (0, 3) else blocks[2:4])
    scenes = (
        Scene("first", 0, blocks[bounds[0]].id, blocks[bounds[1]].id,
              "A private conversation.", location="Orchard", participants=("Alice", "Bob")),
        Scene("second", 1, blocks[bounds[1] + 1].id, blocks[4].id,
              "A rider arrives.", location="Stable", participants=("Carol",)),
    )
    analysis = replace(make_analysis(make_unit(blocks)), scenes=scenes)

    context = ContextBuilder().for_translation_unit(
        document=make_document(section), section=section, unit=unit,
        analysis=analysis, glossary=make_snapshot(),
    )

    text = context.request.input_text
    assert "Текущая сцена главы:" in text
    assert "A private conversation." in text
    assert "Orchard" in text
    assert "Alice" in text and "Bob" in text
    if bounds == (0, 3):
        assert "A rider arrives." not in text
        assert "Stable" not in text and "Carol" not in text
    else:
        assert "A rider arrives." in text
        assert "Stable" in text and "Carol" in text
        assert text.index("A private conversation.") < text.index("A rider arrives.")


@pytest.mark.parametrize(("chapter_budget", "expected", "child_budgets"),
                         ((2, (2, 0), (1, 1)), (4, (3, 1), (1, 2))))
def test_scene_group_note_budgets_preserve_required_help_and_chapter_total(
    chapter_budget, expected, child_budgets,
) -> None:
    from linguary.application.translation_units import plan_translation_units
    from linguary.domain.content_plan import (
        ContentPlan, Difficulty, HelpEvidence, RequiredPoint, Scene, SourceAnchor,
    )

    blocks = make_blocks(4, text="Echo" + "x" * 396)
    section = make_section(blocks)
    whole = make_unit(blocks, treatments=(BlockTreatment.READ_AS_IS,) * 4)
    scenes = (
        Scene("first", 0, blocks[0].id, blocks[1].id, "Conversation"),
        Scene("second", 1, blocks[2].id, blocks[3].id, "Departure"),
    )
    difficulties = tuple(Difficulty(
        f"d{i}", (SourceAnchor(block.id, 0, 4, "Echo"),), block.id,
        "idiom", "critical", "explanation", (RequiredPoint("p", "Explain Echo"),),
    ) for i, block in enumerate(blocks[:3]))
    analysis = replace(make_analysis(whole), scenes=scenes, content_plan=ContentPlan(
        version=3, profile="content-plan-v3", scenes=scenes, difficulties=difficulties,
        before_you_read_evidence=(HelpEvidence(
            difficulty_id="d2", point_id="p", artifact_kind="before_you_read.explanation",
            artifact_key="before_you_read:0", block_id=blocks[2].id, body_excerpt="Already explained",
        ),),
    ))
    units = plan_translation_units(
        blocks, analysis.treatment_plan, id_factory=lambda: "unit",
        max_source_chars=2000, scenes=scenes,
    )
    builder = ContextBuilder(
        content_plan_profile="content-plan-v3", language_notes_per_chapter=chapter_budget,
        max_unit_source_chars=2000,
    )
    # The accepted analysis allowed two owners in the original 1600-char group.
    assert builder._analysis_note_groups(section, blocks) == ((whole.block_ids, chapter_budget),)
    assert tuple(unit.block_ids for unit in units) == (
        (blocks[0].id, blocks[1].id), (blocks[2].id, blocks[3].id),
    )
    texts = [builder.for_translation_unit(
        document=make_document(section), section=section, unit=unit,
        analysis=analysis, glossary=make_snapshot(),
    ).request.input_text for unit in units]

    assert f"не больше {expected[0]} языковых заметок" in texts[0]
    assert f"не больше {expected[1]} языковых заметок" in texts[1]
    assert sum(expected) == chapter_budget
    merged = builder.for_translation_unit(
        document=make_document(section), section=section, unit=whole,
        analysis=analysis, glossary=make_snapshot(),
    )
    assert f"не больше {chapter_budget} языковых заметок" in merged.request.input_text
    for block, child_budget in zip(blocks[:2], child_budgets, strict=True):
        child = builder.for_translation_unit(
            document=make_document(section), section=section,
            unit=make_unit((block,), treatments=(BlockTreatment.READ_AS_IS,)),
            analysis=analysis, glossary=make_snapshot(), recovery_child=True,
        )
        assert f"не больше {child_budget} языковых заметок" in child.request.input_text
        assert "можно превысить" not in child.request.input_text


@pytest.mark.parametrize("text", ("Echo", ""))
def test_scene_note_budget_handles_disabled_notes_and_empty_source(text: str) -> None:
    from linguary.domain.content_plan import Scene

    blocks = make_blocks(1, text=text)
    section = make_section(blocks)
    unit = make_unit(blocks)
    analysis = replace(make_analysis(unit), scenes=(
        Scene("scene", 0, blocks[0].id, blocks[0].id, "Opening"),
    ))
    builder = ContextBuilder(language_notes_per_chapter=0)

    context = builder.for_translation_unit(
        document=make_document(section), section=section, unit=unit,
        analysis=analysis, glossary=make_snapshot(),
    )

    assert "не больше 0 языковых заметок" in context.request.input_text


def test_scene_note_budget_rejects_required_owners_over_budget_but_english_help_uses_no_notes() -> None:
    from linguary.application.context_builder import snapshot_prompt_templates
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, Scene, SourceAnchor

    blocks = make_blocks(1, text="Echo")
    section = make_section(blocks)
    unit = make_unit(blocks, treatments=(BlockTreatment.READ_AS_IS,))
    scenes = (Scene("scene", 0, blocks[0].id, blocks[0].id, "Opening"),)
    analysis = replace(make_analysis(unit), scenes=scenes, content_plan=ContentPlan(
        version=3, profile="content-plan-v3", scenes=scenes, difficulties=(Difficulty(
            "d", (SourceAnchor(blocks[0].id, 0, 4, "Echo"),), blocks[0].id,
            "idiom", "critical", "explanation", (RequiredPoint("p", "Explain Echo"),),
        ),),
    ))
    kwargs = dict(document=make_document(section), section=section, unit=unit,
                  analysis=analysis, glossary=make_snapshot())
    builder = ContextBuilder(content_plan_profile="content-plan-v3", language_notes_per_chapter=0)
    with pytest.raises(ContextBuildError, match="обязательные объяснения превышают бюджет"):
        builder.for_translation_unit(**kwargs)

    templates = snapshot_prompt_templates(
        source_language="en", target_language="ru", reader_level="b1",
        profile="content-plan-v3", english_help=True,
    )
    english_builder = ContextBuilder(
        prompt_templates=templates, content_plan_profile="content-plan-v3",
        language_notes_per_chapter=0,
    )
    context = english_builder.for_translation_unit(**kwargs)
    assert "языковых заметок" not in context.request.input_text

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
    text: str | None = None,
):
    """Processing context for a two-block chapter."""
    blocks = make_blocks(len(treatments), text=text or ("Mr. Bennet visits Netherfield. " + SOURCE_TEXT))
    unit = make_unit(blocks, treatments=treatments)
    section = make_section(blocks)
    return (builder or ContextBuilder()).for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=snapshot if snapshot is not None else make_snapshot(),
    ), unit


def test_fragment_translation_regeneration_receive_pinned_strategy():
    policy = NameTranslationPolicy(version=1, strategy=NameTranslationStrategy.MEANING)
    builder = ContextBuilder(name_translation_policy=policy)
    context, unit = build_context(builder=builder)
    assert 'непрозрачные части' in context.request.instructions
    assert context.request.name_translation_policy_version == 1
    section = make_section(make_blocks(2))
    one = make_unit(section.blocks[:1], treatments=(BlockTreatment.TRANSLATE,))
    regenerated = builder.for_block_regeneration(document=make_document(section), section=section,
        unit=one, analysis=make_analysis(one), glossary=make_snapshot())
    assert 'непрозрачные части' in regenerated.request.instructions
    assert regenerated.request.name_translation_policy_version == 1


def test_request_asks_for_translation_result() -> None:
    """Processing request asks for a structured translation result rather than an analysis."""
    context, _ = build_context()

    assert context.request.kind is LLMResponseKind.TRANSLATION_RESULT


def test_pronunciation_candidate_alias_and_exact_form():
    from linguary.application.llm_dto import PronunciationAnchorResponse, PronunciationCandidateResponse, PronunciationContextResponse
    from linguary.application.context_builder import snapshot_prompt_templates
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level='b2',
                                         pronunciation_help_enabled=True)
    context, unit = build_context(builder=ContextBuilder(prompt_templates=templates))
    assert context.request.pronunciation_candidates_version == 1
    exact = PronunciationCandidateResponse(PronunciationAnchorResponse('b1', 0, 6, ' zorp '), ' zorp ',
        PronunciationContextResponse(None, None, None, False), 'unpredictable_spelling')
    raw = TranslationResultResponse(pronunciation_candidates=(exact,
        replace(exact, anchor=replace(exact.anchor, block_id='b999')),
        replace(exact, anchor=replace(exact.anchor, block_id=unit.items[0].block.id))))
    mapped = context.to_block_ids(raw)
    assert mapped.pronunciation_candidates[0].anchor.block_id == unit.items[0].block.id
    assert mapped.pronunciation_candidates[0].form == mapped.pronunciation_candidates[0].anchor.quote == ' zorp '
    assert all(not candidate.anchor.block_id for candidate in mapped.pronunciation_candidates[1:])


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
        snapshot=snapshot, builder=ContextBuilder(max_accumulated_candidates=2),
        text=" ".join(f"Term {i}" for i in range(5)) + " Netherfield.",
    )

    text = context.request.input_text
    suggestions_text = text[text.index("Необязательные подсказки глоссария"):text.index("Блоки для обработки:")]
    assert "Term 0" in suggestions_text and "Term 1" in suggestions_text
    assert "Term 2" not in suggestions_text
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

    context, _ = build_context(snapshot=snapshot, text="Wren Calder.")

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
    blocks = make_blocks(1, text="Duskmere and Wren Calder at the marsh gate on the boardwalk.")
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
    blocks = make_blocks(1, text="The marsh gate.")
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


def test_translation_input_places_chapter_invariant_sections_before_unit_budget() -> None:
    blocks = (
        make_block(order=0, text="Mr. Bennet visits Netherfield."),
        make_block(order=1, text="A much longer second paragraph about Netherfield and Mr. Bennet that changes the unit note budget."),
    )
    section = make_section(blocks)
    document = make_document(section)
    full_unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE))
    analysis = replace(
        make_analysis(full_unit),
        before_you_read=(
            BeforeYouReadEntry(expression="on a good day", explanation="если повезёт"),
        ),
    )
    unit_1 = make_unit((blocks[0],), treatments=(BlockTreatment.TRANSLATE,))
    unit_2 = make_unit((blocks[1],), treatments=(BlockTreatment.TRANSLATE,))
    builder = ContextBuilder(language_notes_per_chapter=8)
    snapshot = make_snapshot()

    ctx_1 = builder.for_translation_unit(
        document=document, section=section, unit=unit_1, analysis=analysis, glossary=snapshot
    )
    ctx_2 = builder.for_translation_unit(
        document=document, section=section, unit=unit_2, analysis=analysis, glossary=snapshot
    )

    for text in (ctx_1.request.input_text, ctx_2.request.input_text):
        header_idx = text.index("Книга: Pride and Prejudice")
        digest_idx = text.index("О чём эта глава:")
        explained_idx = text.index("Эти выражения главы читателю уже объяснены до начала чтения.")
        budget_idx = text.index("В этом запросе — не больше ")
        mandatory_idx = text.index("Обязательные переводческие решения книги")
        suggestions_idx = text.index("Необязательные подсказки глоссария")
        blocks_idx = text.index("Блоки для обработки:")
        assert header_idx < digest_idx < explained_idx < budget_idx < mandatory_idx < suggestions_idx < blocks_idx

    prefix_end_1 = ctx_1.request.input_text.index("В этом запросе — не больше ")
    prefix_end_2 = ctx_2.request.input_text.index("В этом запросе — не больше ")
    assert ctx_1.request.input_text[:prefix_end_1] == ctx_2.request.input_text[:prefix_end_2]


def test_owned_content_plan_omits_duplicate_before_you_read_and_anchor_offsets() -> None:
    import json
    from linguary.domain.content_plan import ContentPlan, ContextFact, Difficulty, RequiredPoint, SourceAnchor

    blocks = (
        make_block(order=0, text="Echo calls."),
        make_block(order=1, text="The answer echoes."),
    )
    section = make_section(blocks)
    unit = make_unit((blocks[0],), treatments=(BlockTreatment.TRANSLATE,))
    difficulty = Difficulty(
        "d-1",
        (
            SourceAnchor(blocks[0].id, 0, 4, "Echo"),
            SourceAnchor(blocks[1].id, 4, 10, "answer"),
        ),
        blocks[0].id,
        "semantic_relation",
        "critical",
        "explanation",
        (RequiredPoint("p-1", "Explain the link"),),
    )
    fact = ContextFact(
        id="f-1",
        kind="speaker",
        anchors=(
            SourceAnchor(blocks[0].id, 0, 4, "Echo"),
            SourceAnchor(blocks[1].id, 4, 10, "answer"),
        ),
        subject_anchor=SourceAnchor(blocks[0].id, 0, 4, "Echo"),
        value="Narrator",
        certainty="explicit",
    )
    analysis = replace(
        make_analysis(make_unit(blocks)),
        before_you_read=(
            BeforeYouReadEntry(expression="on a good day", explanation="если повезёт"),
        ),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(difficulty,),
            context_facts=(fact,),
        ),
    )
    builder = ContextBuilder(content_plan_profile="content-plan-v3")
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    marker = "Обязательные аспекты и точные связанные привязки:\n"
    assert marker in context.request.input_text
    plan_json = context.request.input_text.split(marker, 1)[1]
    payload = json.loads(plan_json)

    assert "before_you_read" not in payload
    assert payload["version"] == 3
    for anchor in payload["difficulties"][0]["anchors"]:
        assert "start" not in anchor
        assert "end" not in anchor
        assert set(anchor.keys()) == {"block_id", "quote"}
    for anchor in (*payload["context_facts"][0]["anchors"], payload["context_facts"][0]["subject_anchor"]):
        assert "start" not in anchor
        assert "end" not in anchor
        assert set(anchor.keys()) == {"block_id", "quote"}


def test_mixed_unit_synthesizes_omitted_pure_read_as_is_blocks_and_accepts_full_stub_replies() -> None:
    from linguary.domain.content_plan import ContentPlan, Difficulty, HelpEvidence, RequiredPoint, SourceAnchor

    blocks = make_blocks(4)
    unit = make_unit(
        blocks,
        treatments=(
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
        ),
    )
    section = make_section(blocks)
    uncovered_diff = Difficulty(
        "d-uncovered",
        (SourceAnchor(blocks[2].id, 0, 4, "Text"),),
        blocks[2].id,
        "wordplay",
        "critical",
        "explanation",
        (RequiredPoint("p-1", "Explain"),),
    )
    covered_diff = Difficulty(
        "d-covered",
        (SourceAnchor(blocks[3].id, 0, 4, "Text"),),
        blocks[3].id,
        "idiom",
        "useful",
        "explanation",
        (RequiredPoint("p-2", "Covered by BYR"),),
    )
    analysis = replace(
        make_analysis(unit),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(uncovered_diff, covered_diff),
            before_you_read_evidence=(
                HelpEvidence(
                    difficulty_id="d-covered",
                    point_id="p-2",
                    artifact_kind="before_you_read.explanation",
                    artifact_key="before_you_read:0",
                    block_id=blocks[3].id,
                    body_excerpt="Covered by BYR",
                ),
            ),
        ),
    )
    builder = ContextBuilder(content_plan_profile="content-plan-v3")
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert context.aliases == ("b1", "b2", "b3", "b4")
    assert context.pure_read_as_is_aliases == frozenset({"b1", "b4"})
    assert "[b1] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text
    assert "[b4] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text
    assert "[b3] режим READ_AS_IS\n" in context.request.input_text

    partial_payload = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="b2",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод второго блока.",
            ),
            BlockResultResponse(
                source_block_id="b3",
                treatment=BlockTreatment.READ_AS_IS,
            ),
        ),
        language_notes=(),
    )
    resolved_partial = context.to_block_ids(partial_payload)
    assert [b.source_block_id for b in resolved_partial.blocks] == [b.id for b in blocks]
    assert resolved_partial.blocks[0] == BlockResultResponse(
        source_block_id=blocks[0].id,
        treatment=BlockTreatment.READ_AS_IS,
        translated_text=None,
        glossary_hints=(),
    )
    assert resolved_partial.blocks[3] == BlockResultResponse(
        source_block_id=blocks[3].id,
        treatment=BlockTreatment.READ_AS_IS,
        translated_text=None,
        glossary_hints=(),
    )

    full_stub_payload = TranslationResultResponse(
        blocks=(
            BlockResultResponse(source_block_id="b1", treatment=BlockTreatment.READ_AS_IS),
            BlockResultResponse(
                source_block_id="b2",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод второго блока.",
            ),
            BlockResultResponse(source_block_id="b3", treatment=BlockTreatment.READ_AS_IS),
            BlockResultResponse(source_block_id="b4", treatment=BlockTreatment.READ_AS_IS),
        ),
        language_notes=(),
    )
    resolved_full = context.to_block_ids(full_stub_payload)
    assert [b.source_block_id for b in resolved_full.blocks] == [b.id for b in blocks]


def test_mixed_unit_marks_pure_read_as_is_context_only_even_with_pronunciation_enabled() -> None:
    from linguary.application.context_builder import snapshot_prompt_templates
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, SourceAnchor

    blocks = make_blocks(3)
    unit = make_unit(
        blocks,
        treatments=(
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
        ),
    )
    section = make_section(blocks)
    uncovered_diff = Difficulty(
        "d-uncovered",
        (SourceAnchor(blocks[2].id, 0, 4, "Text"),),
        blocks[2].id,
        "wordplay",
        "critical",
        "explanation",
        (RequiredPoint("p-1", "Explain"),),
    )
    analysis = replace(
        make_analysis(unit),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(uncovered_diff,),
        ),
    )
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        profile="content-plan-v3",
        pronunciation_help_enabled=True,
    )
    builder = ContextBuilder(
        prompt_templates=templates,
        content_plan_profile="content-plan-v3",
    )
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert context.request.pronunciation_candidates_version == 1
    assert context.pure_read_as_is_aliases == frozenset({"b1"})
    assert "[b1] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text
    assert "(только для контекста, в blocks не возвращать)" in context.request.instructions
    assert "тире" in context.request.instructions
    assert context.expectation.block_ids == ("b1", "b2", "b3")
    assert [b.context_only for b in context.expectation.blocks] == [True, False, False]


def test_to_block_ids_preserves_misordered_blocks_without_duplicating_pure_read_as_is() -> None:
    from linguary.application.validators.coverage import coverage_violation
    from linguary.domain import BlockOrderError, BlockProcessingResult

    blocks = make_blocks(3)
    unit = make_unit(
        blocks,
        treatments=(
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
    )
    section = make_section(blocks)
    context = ContextBuilder().for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=make_snapshot(),
    )
    assert context.pure_read_as_is_aliases == frozenset({"b2"})

    misordered_omitted = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="b3",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод третьего блока.",
            ),
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод первого блока.",
            ),
        ),
    )
    resolved_omitted = context.to_block_ids(misordered_omitted)
    assert [b.source_block_id for b in resolved_omitted.blocks] == [
        blocks[2].id,
        blocks[1].id,
        blocks[0].id,
    ]
    assert coverage_violation(resolved_omitted, unit=unit) == (
        "порядок блоков в ответе не совпадает с порядком блоков группы"
    )
    with pytest.raises(BlockOrderError):
        unit.validate_results(
            tuple(
                BlockProcessingResult(
                    block_id=b.source_block_id,
                    treatment=b.treatment,
                    translated_text=b.translated_text,
                )
                for b in resolved_omitted.blocks
            )
        )

    misordered_with_pure = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="b3",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод третьего блока.",
            ),
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод первого блока.",
            ),
            BlockResultResponse(
                source_block_id="b2",
                treatment=BlockTreatment.READ_AS_IS,
                translated_text=None,
            ),
        ),
    )
    resolved_with_pure = context.to_block_ids(misordered_with_pure)
    assert [b.source_block_id for b in resolved_with_pure.blocks] == [
        blocks[2].id,
        blocks[0].id,
        blocks[1].id,
    ]


def test_translation_instructions_preserve_escaped_newline_literal() -> None:
    from linguary.application.context_builder import (
        block_regeneration_instructions,
        translation_unit_instructions,
    )

    for instructions in (
        translation_unit_instructions(source_language="en", target_language="ru"),
        block_regeneration_instructions(source_language="en", target_language="ru"),
    ):
        assert r"перенос строки (\n) внутри translated_text допустим" in instructions
        assert "перенос строки (\n) внутри translated_text допустим" not in instructions


def test_translation_context_evaluates_input_without_neighbours_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linguary.application.context_builder as context_builder_module
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY

    blocks = make_blocks(3)
    section = make_section(blocks)
    document = make_document(section)
    unit = make_unit((blocks[1],), treatments=(BlockTreatment.TRANSLATE,))
    analysis = make_analysis(make_unit(blocks))
    snapshot = make_snapshot()
    builder = ContextBuilder()

    calls: list[int] = []
    original_translation_input = builder._translation_input

    def spy_translation_input(**kwargs):
        calls.append(len(kwargs["neighbours"]))
        return original_translation_input(**kwargs)

    monkeypatch.setattr(builder, "_translation_input", spy_translation_input)

    context = builder.for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=snapshot,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    other = builder.for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=snapshot,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )

    assert calls == [2, 2]
    assert context.has_optional_neighbours is True
    assert context == other
    replaced = replace(context, request=context.request)
    assert replaced == context
    assert calls == [2, 2]

    key_calls: list[str] = []
    original_glossary_key = context_builder_module.glossary_key

    def spy_glossary_key(value: str) -> str:
        key_calls.append(value)
        return original_glossary_key(value)

    monkeypatch.setattr(context_builder_module, "glossary_key", spy_glossary_key)

    fallback_text = context.input_without_optional_neighbours
    assert fallback_text is not None
    assert "Соседние абзацы главы" not in fallback_text
    assert calls == [2, 2, 0]
    assert key_calls.count(blocks[1].text) == 1

    assert context.input_without_optional_neighbours == fallback_text
    stripped = context.without_optional_neighbours()
    assert stripped.request.input_text == fallback_text
    assert stripped.has_optional_neighbours is False
    assert stripped.input_without_optional_neighbours is None
    assert calls == [2, 2, 0]


def test_all_read_as_is_unit_with_pronunciation_marks_tail_blocks_context_only() -> None:
    from linguary.application.context_builder import snapshot_prompt_templates

    blocks = make_blocks(3)
    unit = make_unit(
        blocks,
        treatments=(
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
        ),
    )
    section = make_section(blocks)
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        pronunciation_help_enabled=True,
    )
    builder = ContextBuilder(prompt_templates=templates)
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=make_analysis(unit),
        glossary=make_snapshot(),
    )

    assert context.pure_read_as_is_aliases == frozenset({"b2", "b3"})
    assert "[b1] режим READ_AS_IS\n" in context.request.input_text
    assert "[b2] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text
    assert "[b3] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text
    assert [b.context_only for b in context.expectation.blocks] == [False, True, True]

    payload = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.READ_AS_IS,
            ),
        ),
    )
    resolved = context.to_block_ids(payload)
    assert [b.source_block_id for b in resolved.blocks] == [b.id for b in blocks]


def test_owned_content_plan_aliases_difficulty_and_fact_ids_and_resolves_response() -> None:
    import json
    from linguary.application.llm_dto import FragmentAssistanceResponse, HelpEvidenceResponse
    from linguary.domain.content_plan import (
        ContentPlan,
        ContextFact,
        Difficulty,
        HelpEvidence,
        RequiredPoint,
        SourceAnchor,
    )

    blocks = make_blocks(2, text="Echo calls and answers.")
    unit = make_unit(blocks, treatments=(BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE))
    section = make_section(blocks)

    covered_diff = Difficulty(
        "diff-covered-uuid",
        (SourceAnchor(blocks[0].id, 0, 4, "Echo"),),
        blocks[0].id,
        "idiom",
        "useful",
        "explanation",
        (RequiredPoint("p-cov", "Already in BYR"),),
    )
    uncovered_diff_1 = Difficulty(
        "diff-uncovered-uuid-1",
        (SourceAnchor(blocks[0].id, 5, 10, "calls"),),
        blocks[0].id,
        "wordplay",
        "critical",
        "explanation",
        (RequiredPoint("p-1", "Explain calls"),),
    )
    uncovered_diff_2 = Difficulty(
        "diff-uncovered-uuid-2",
        (SourceAnchor(blocks[1].id, 0, 4, "Echo"),),
        blocks[1].id,
        "grammar",
        "critical",
        "explanation",
        (RequiredPoint("p-2", "Explain grammar"),),
    )
    fact_1 = ContextFact(
        id="fact-uuid-1",
        kind="speaker",
        anchors=(SourceAnchor(blocks[0].id, 0, 4, "Echo"),),
        subject_anchor=SourceAnchor(blocks[0].id, 0, 4, "Echo"),
        value="Narrator",
        certainty="explicit",
    )
    fact_2 = ContextFact(
        id="fact-uuid-2",
        kind="gender",
        anchors=(SourceAnchor(blocks[1].id, 0, 4, "Echo"),),
        subject_anchor=SourceAnchor(blocks[1].id, 0, 4, "Echo"),
        value="feminine",
        certainty="inferred",
    )

    analysis = replace(
        make_analysis(unit),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(covered_diff, uncovered_diff_1, uncovered_diff_2),
            before_you_read_evidence=(
                HelpEvidence(
                    difficulty_id="diff-covered-uuid",
                    point_id="p-cov",
                    artifact_kind="before_you_read.explanation",
                    artifact_key="before_you_read:0",
                    block_id=blocks[0].id,
                    body_excerpt="Already in BYR",
                ),
            ),
            context_facts=(fact_1, fact_2),
        ),
    )

    builder = ContextBuilder(content_plan_profile="content-plan-v3")
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert dict(context.difficulty_ids_by_alias) == {
        "d1": "diff-uncovered-uuid-1",
        "d2": "diff-uncovered-uuid-2",
    }

    marker = "Обязательные аспекты и точные связанные привязки:\n"
    plan_payload = json.loads(context.request.input_text.split(marker, 1)[1])
    assert [d["id"] for d in plan_payload["difficulties"]] == ["d1", "d2"]
    assert "diff-covered-uuid" not in context.request.input_text
    assert [f["id"] for f in plan_payload["context_facts"]] == ["f1", "f2"]
    assert all(
        "start" not in anchor and "end" not in anchor
        for d in plan_payload["difficulties"]
        for anchor in d["anchors"]
    )

    response = TranslationResultResponse(
        blocks=(
            BlockResultResponse(
                source_block_id="b1",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод 1.",
            ),
            BlockResultResponse(
                source_block_id="b2",
                treatment=BlockTreatment.TRANSLATE,
                translated_text="Перевод 2.",
            ),
        ),
        fragment_assistances=(
            FragmentAssistanceResponse(
                difficulty_id="d1",
                point_ids=("p-1",),
                kind="lexical_hint",
                text="simple call",
            ),
            FragmentAssistanceResponse(
                difficulty_id="diff-uncovered-uuid-2",
                point_ids=("p-2",),
                kind="russian_explanation",
                text="Пояснение.",
            ),
        ),
        content_evidence=(
            HelpEvidenceResponse(
                difficulty_id="d1",
                point_id="p-1",
                artifact_kind="language_note.note",
                source_block_id="b1",
                artifact_index=0,
                body_excerpt="Пояснение",
            ),
        ),
    )

    resolved = context.to_block_ids(response)
    assert [item.difficulty_id for item in resolved.fragment_assistances] == [
        "diff-uncovered-uuid-1",
        "diff-uncovered-uuid-2",
    ]
    assert resolved.content_evidence[0].difficulty_id == "diff-uncovered-uuid-1"
    assert resolved.content_evidence[0].source_block_id == blocks[0].id

    aliased_msg = context.to_aliases(
        f"не покрыта трудность diff-uncovered-uuid-2 в блоке {blocks[1].id}"
    )
    assert aliased_msg == "не покрыта трудность d2 в блоке b2"


def test_english_help_multi_block_unit_marks_tail_blocks_context_only_and_uses_english_help_instructions() -> None:
    from linguary.application.context_builder import snapshot_prompt_templates
    from linguary.domain.content_plan import ContentPlan

    blocks = make_blocks(3)
    unit = make_unit(
        blocks,
        treatments=(
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.READ_AS_IS,
        ),
    )
    section = make_section(blocks)
    analysis = replace(
        make_analysis(unit),
        content_plan=ContentPlan(version=3, profile="content-plan-v3", difficulties=()),
    )
    policy = NameTranslationPolicy(version=1, strategy=NameTranslationStrategy.MEANING)
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
        english_help=True,
        name_translation_policy=policy,
    )
    builder = ContextBuilder(
        prompt_templates=templates,
        content_plan_profile="content-plan-v3",
        reader_level="b1",
    )
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
    )

    assert "Режим чтения на английском с помощью к трудным местам." in context.request.instructions
    assert "непрозрачные части передавай по звучанию" in context.request.instructions
    assert "TRANSLATE — верни в translated_text полный литературный перевод" not in context.request.instructions
    assert context.pure_read_as_is_aliases == frozenset({"b2", "b3"})
    assert "[b1] режим READ_AS_IS\n" in context.request.input_text
    assert "[b2] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text
    assert "[b3] режим READ_AS_IS (только для контекста, в blocks не возвращать)" in context.request.input_text


def test_owned_content_plan_omits_empty_payload_and_priority_field() -> None:
    import json
    from linguary.domain.content_plan import (
        ContentPlan,
        ContextFact,
        Difficulty,
        HelpEvidence,
        RequiredPoint,
        SourceAnchor,
    )

    blocks = make_blocks(2, text="Echo calls and answers.")
    section = make_section(blocks)
    unit_b1 = make_unit((blocks[0],), treatments=(BlockTreatment.TRANSLATE,))
    unit_b2 = make_unit((blocks[1],), treatments=(BlockTreatment.TRANSLATE,))

    covered_diff = Difficulty(
        "diff-covered",
        (SourceAnchor(blocks[0].id, 0, 4, "Echo"),),
        blocks[0].id,
        "idiom",
        "useful",
        "explanation",
        (RequiredPoint("p-cov", "Already in BYR"),),
    )
    b2_diff = Difficulty(
        "diff-b2",
        (SourceAnchor(blocks[1].id, 0, 4, "Echo"),),
        blocks[1].id,
        "wordplay",
        "critical",
        "explanation",
        (RequiredPoint("p-1", "Explain Echo"),),
    )
    b2_fact = ContextFact(
        id="fact-b2",
        kind="speaker",
        anchors=(SourceAnchor(blocks[1].id, 0, 4, "Echo"),),
        subject_anchor=SourceAnchor(blocks[1].id, 0, 4, "Echo"),
        value="Narrator",
        certainty="explicit",
    )

    analysis = replace(
        make_analysis(make_unit(blocks)),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(covered_diff, b2_diff),
            before_you_read_evidence=(
                HelpEvidence(
                    difficulty_id="diff-covered",
                    point_id="p-cov",
                    artifact_kind="before_you_read.explanation",
                    artifact_key="before_you_read:0",
                    block_id=blocks[0].id,
                    body_excerpt="Already in BYR",
                ),
            ),
            context_facts=(b2_fact,),
        ),
    )

    builder = ContextBuilder(content_plan_profile="content-plan-v3")
    ctx_b1 = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit_b1,
        analysis=analysis,
        glossary=make_snapshot(),
    )
    marker = "Обязательные аспекты и точные связанные привязки:\n"
    assert marker not in ctx_b1.request.input_text

    ctx_b2 = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit_b2,
        analysis=analysis,
        glossary=make_snapshot(),
    )
    assert marker in ctx_b2.request.input_text
    raw_json = ctx_b2.request.input_text.split(marker, 1)[1]
    assert '": ' not in raw_json
    assert '", "' not in raw_json
    parsed = json.loads(raw_json)
    assert len(parsed["difficulties"]) == 1
    assert "priority" not in parsed["difficulties"][0]


def test_translation_context_to_aliases_prevents_prefix_collision_and_english_help_omits_notes_budget() -> None:
    from linguary.application.context_builder import snapshot_prompt_templates
    from linguary.domain.content_plan import (
        ContentPlan,
        ContextFact,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )

    blocks = tuple(
        replace(make_block(order=index, text="Echo calls and answers."), id=f"block-{index + 1}")
        for index in range(10)
    )
    unit = make_unit(blocks, treatments=(BlockTreatment.READ_AS_IS,) * 10)
    section = make_section(blocks)
    diff_1 = Difficulty(
        "diff-1",
        (SourceAnchor("block-1", 0, 4, "Echo"),),
        "block-1",
        "idiom",
        "critical",
        "explanation",
        (RequiredPoint("p-1", "Explain 1"),),
    )
    diff_10 = Difficulty(
        "diff-10",
        (SourceAnchor("block-10", 0, 4, "Echo"),),
        "block-10",
        "idiom",
        "critical",
        "explanation",
        (RequiredPoint("p-10", "Explain 10"),),
    )
    fact_1 = ContextFact(
        id="fact-1",
        kind="speaker",
        anchors=(SourceAnchor("block-1", 0, 4, "Echo"),),
        subject_anchor=SourceAnchor("block-1", 0, 4, "Echo"),
        value="Narrator",
        certainty="explicit",
    )
    fact_10 = ContextFact(
        id="fact-10",
        kind="speaker",
        anchors=(SourceAnchor("block-10", 0, 4, "Echo"),),
        subject_anchor=SourceAnchor("block-10", 0, 4, "Echo"),
        value="Listener",
        certainty="explicit",
    )
    chapter_candidate = GlossaryCandidate.create(
        id="cand-1",
        analysis_id="analysis-1",
        source_text="Echo",
        type=ConceptType.TERM,
        suggested_translation="Эхо",
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=0.8,
        note="женский род",
    )
    analysis = replace(
        make_analysis(unit),
        glossary_candidates=(chapter_candidate,),
        content_plan=ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=(diff_1, diff_10),
            context_facts=(fact_1, fact_10),
        ),
    )
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
        english_help=True,
    )
    builder = ContextBuilder(
        prompt_templates=templates,
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    )
    context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
        book_candidates=(chapter_candidate,),
        recovery_child=False,
    )
    recovery_context = builder.for_translation_unit(
        document=make_document(section),
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=make_snapshot(),
        book_candidates=(chapter_candidate,),
        recovery_child=True,
    )

    msg = "ошибка (diff-10, block-10) и (diff-1, block-1) и факт fact-10 / fact-1, но не block-100"
    assert (
        context.to_aliases(msg)
        == "ошибка (d2, b10) и (d1, b1) и факт f2 / f1, но не block-100"
    )
    assert "языковых заметок" not in context.request.input_text
    assert "языковых заметок" not in recovery_context.request.input_text







