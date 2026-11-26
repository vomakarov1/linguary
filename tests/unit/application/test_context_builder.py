"""Chapter analysis context composition (P3.T3, PRD § 13)."""

from __future__ import annotations

import pytest
from analysis_fakes import (
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    make_analysis_block,
    make_analysis_candidate,
    make_analysis_document,
    make_analysis_section,
)

from linguary.application.analysis_merge import plan_fragments
from linguary.application.context_builder import (
    ContextBuildError,
    ContextBuilder,
    chapter_analysis_instructions,
    chapter_fragment_instructions,
    textual_blocks,
    snapshot_prompt_templates,
)
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.domain import (
    BlockTreatment,
    BlockType,
    ChapterAnalysis,
    ConceptType,
    GlossarySnapshot,
    GlossarySnapshotEntry,
    SupportStatus,
    TermStatus,
    TranslationUnit,
    TreatmentPlan,
)
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy

NEIGHBOUR_TEXT = "The neighbouring chapter must never reach the model."


def test_pronunciation_prompt_uses_context_without_new_level_setting():
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level=None,
                                         pronunciation_help_enabled=True)
    assert templates['pronunciation_candidates_version'] == 1
    assert 'не задан' in templates['pronunciation_processing']
    assert 'READ_AS_IS' in templates['pronunciation_processing']
    assert 'длина и редкость' in templates['pronunciation_processing']
    assert 'pronunciation_candidates_version' not in snapshot_prompt_templates(
        source_language='en', target_language='ru', reader_level='b2')
    with pytest.raises(ContextBuildError):
        ContextBuilder(prompt_templates=templates | {'pronunciation_candidates_version': True})

#: A chapter with four paragraphs: with a two-block limit it splits into two fragments.
FRAGMENTED_TEXTS = (
    "Paragraph one.",
    "Paragraph two.",
    "Paragraph three.",
    "Paragraph four.",
)


def _policy(strategy):
    return NameTranslationPolicy(version=1, strategy=NameTranslationStrategy(strategy))


def test_sound_does_not_translate_components():
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level='b2', name_translation_policy=_policy('sound'))
    for key in ('analysis', 'fragment', 'translation', 'regeneration'):
        assert 'произношение' in templates[key]
        assert 'не переводи' in templates[key]
        assert 'смысловые части' in templates[key]
        assert 'склоняемости' in templates[key]
        assert 'созвучие' in templates[key]
        assert 'известных переводов книги' in templates[key]
        assert 'Дурсль' in templates[key]
        assert 'Волан-де-Морт' in templates[key]
        assert 'Тисовая улица' in templates[key]
    assert 'applied_method=sound' in templates['analysis']


def test_meaning_keeps_opaque_parts():
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level='b2', name_translation_policy=_policy('meaning'))
    assert 'ясные смысловые части' in templates['analysis']
    assert 'непрозрачные части' in templates['analysis']
    assert 'этимологию' in templates['analysis']


@pytest.mark.parametrize('profile', ['legacy-v1', 'content-plan-v2', 'content-plan-v3'])
@pytest.mark.parametrize('english_help', [False, True])
def test_all_profiles_and_english_help(profile, english_help):
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level='b2', profile=profile, english_help=english_help, name_translation_policy=_policy('meaning'))
    assert all('непрозрачные части' in templates[key] for key in ('analysis', 'fragment', 'translation', 'regeneration'))
    assert templates['name_translation_policy'] == {'version': 1, 'strategy': 'meaning'}


def test_legacy_prompt_and_input_text_unchanged():
    section = make_analysis_section(texts=('Alice stood still.',))
    document = make_analysis_document((section,))
    baseline = ContextBuilder().for_chapter_analysis(document=document, section=section)
    explicit = ContextBuilder(name_translation_policy=None).for_chapter_analysis(document=document, section=section)
    assert explicit.request.instructions == baseline.request.instructions
    assert explicit.request.input_text == baseline.request.input_text
    assert explicit.request.name_translation_policy_version is None


def test_fragment_receives_pinned_strategy():
    section = make_analysis_section(texts=FRAGMENTED_TEXTS)
    document = make_analysis_document((section,))
    fragment = plan_fragments(textual_blocks(section), max_blocks=2)[0]
    context = ContextBuilder(name_translation_policy=_policy('meaning')).for_chapter_fragment(
        document=document, section=section, fragment=fragment, fragment_count=2)
    assert 'непрозрачные части' in context.request.instructions
    assert context.request.name_translation_policy_version == 1


def build_context(
    *,
    texts: tuple[str, ...] = ("It is a truth universally acknowledged.",),
    candidates: tuple[object, ...] = (),
    max_accumulated_candidates: int = 200,
    reader_level: str = "b2",
):
    """Chapter analysis context for the first chapter of a two-chapter book."""
    section = make_analysis_section(
        section_id="section-1", title="Chapter 1", order=0, texts=texts
    )
    neighbour = make_analysis_section(
        section_id="section-2", title="Chapter 2", order=1, texts=(NEIGHBOUR_TEXT,)
    )
    document = make_analysis_document((section, neighbour))
    builder = ContextBuilder(
        max_accumulated_candidates=max_accumulated_candidates,
        reader_level=reader_level,
    )
    return builder.for_chapter_analysis(
        document=document,
        section=section,
        accumulated_candidates=candidates,  # type: ignore[arg-type]  # test double factory
    )


def build_fragment_context(*, order: int = 0, fragment_count: int = 2):
    """Context for one fragment of a chapter split into two blocks per fragment."""
    section = make_analysis_section(
        section_id="section-1", title="Chapter 1", order=0, texts=FRAGMENTED_TEXTS
    )
    neighbour = make_analysis_section(
        section_id="section-2", title="Chapter 2", order=1, texts=(NEIGHBOUR_TEXT,)
    )
    document = make_analysis_document((section, neighbour))
    fragments = plan_fragments(textual_blocks(section), max_blocks=2)
    return ContextBuilder().for_chapter_fragment(
        document=document,
        section=section,
        fragment=fragments[order],
        fragment_count=fragment_count,
    )


def test_analysis_context_contains_chapter_blocks() -> None:
    """Text of each chapter block enters the request under its alias."""
    context = build_context(texts=("First paragraph.", "Second paragraph."))

    input_text = context.request.input_text

    assert "[b1]\nFirst paragraph." in input_text
    assert "[b2]\nSecond paragraph." in input_text


def test_analysis_context_excludes_neighbour_chapters() -> None:
    """Text of neighbouring chapters is excluded from the analysis context (PRD § 13)."""
    context = build_context()

    assert NEIGHBOUR_TEXT not in context.request.input_text
    assert NEIGHBOUR_TEXT not in context.request.instructions


def test_analysis_context_names_book_chapter_and_languages() -> None:
    """The request names the book, chapter, and publication languages."""
    input_text = build_context().request.input_text

    assert "Pride and Prejudice" in input_text
    assert "Jane Austen" in input_text
    assert "Chapter 1" in input_text
    assert f"Язык оригинала: {SOURCE_LANGUAGE}" in input_text
    assert f"Язык перевода: {TARGET_LANGUAGE}" in input_text


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(
            make_analysis_block(
                section_id="section-1",
                order=1,
                text="",
                type=BlockType.IMAGE,
            ),
            id="image",
        ),
        pytest.param(
            make_analysis_block(
                section_id="section-1",
                order=1,
                text="| a | b |",
                type=BlockType.TABLE,
                support_status=SupportStatus.PRESERVE_OPAQUE,
            ),
            id="preserve-opaque-table",
        ),
        pytest.param(
            make_analysis_block(section_id="section-1", order=1, text="   "),
            id="blank-text",
        ),
    ],
)
def test_non_textual_blocks_stay_out_of_context(block) -> None:
    """`PRESERVE_ONLY` and non-textual blocks are not passed to the model (PRD § 10.1)."""
    textual = make_analysis_block(
        section_id="section-1", order=0, text="A textual paragraph."
    )
    section = make_analysis_section(section_id="section-1", blocks=(textual, block))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert textual_blocks(section) == (textual,)
    assert context.aliases == ("b1",)
    assert context.block_id_for("b1") == textual.id
    assert block.id not in context.block_ids_by_alias.values()


def test_preserved_block_text_stays_out_of_the_request() -> None:
    """Text of a block preserved opaque is not visible to the model (ADR § 3)."""
    textual = make_analysis_block(
        section_id="section-1", order=0, text="A textual paragraph."
    )
    preserved = make_analysis_block(
        section_id="section-1",
        order=1,
        text="| column | column |",
        type=BlockType.TABLE,
        support_status=SupportStatus.PRESERVE_OPAQUE,
    )
    section = make_analysis_section(section_id="section-1", blocks=(textual, preserved))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert preserved.text not in context.request.input_text


def test_accumulated_candidates_reach_the_request() -> None:
    """Candidates from previous chapters matching chapter text are passed as book decisions."""
    context = build_context(
        texts=("Longbourn was the estate.",),
        candidates=(
            make_analysis_candidate(
                source_text="Longbourn", suggested_translation="Лонгборн"
            ),
        ),
    )

    assert "Longbourn" in context.request.input_text
    assert "Лонгборн" in context.request.input_text


def test_empty_accumulated_candidates_are_stated_explicitly() -> None:
    """Absence of accumulated decisions is stated in words rather than an empty section."""
    assert "Предыдущие главы" in build_context().request.input_text


def test_accumulated_candidates_are_deduplicated_and_limited() -> None:
    """Duplicates are collapsed, and the list is capped at the configured limit."""
    context = build_context(
        texts=("Longbourn and Netherfield were visited.",),
        candidates=(
            make_analysis_candidate(source_text="Longbourn", analysis_id="analysis-1"),
            make_analysis_candidate(source_text="Longbourn", analysis_id="analysis-2"),
            make_analysis_candidate(source_text="Netherfield"),
        ),
        max_accumulated_candidates=1,
    )

    decisions = context.request.input_text.split("Текстовые блоки главы:")[0]
    assert decisions.count("Longbourn") == 1
    assert "Netherfield" not in decisions


def test_obvious_spellings_of_one_decision_reach_the_request_once() -> None:
    """Variants of a single name with and without a dot are sent as a single decision (QA-11)."""
    context = build_context(
        texts=("Mr Wonka opened the factory doors.",),
        candidates=(
            make_analysis_candidate(
                source_text="Mr. Wonka",
                suggested_translation="мистер Вонка",
                analysis_id="analysis-1",
            ),
            make_analysis_candidate(
                source_text="MR WONKA",
                suggested_translation="мистер Вонка",
                analysis_id="analysis-2",
            ),
        ),
    )

    assert context.request.input_text.count("мистер Вонка") == 1
    assert "MR WONKA" not in context.request.input_text


def test_glossary_candidates_not_matching_chapter_text_are_filtered_out() -> None:
    """Candidates from previous chapters that do not appear in the chapter are filtered out (P2.T2, AC-006)."""
    context = build_context(
        texts=("Only Netherfield is mentioned here.",),
        candidates=(
            make_analysis_candidate(source_text="Longbourn", suggested_translation="Лонгборн"),
            make_analysis_candidate(source_text="Netherfield", suggested_translation="Нетерфилд"),
        ),
    )

    assert "Netherfield" in context.request.input_text
    assert "Нетерфилд" in context.request.input_text
    assert "Longbourn" not in context.request.input_text
    assert "Лонгборн" not in context.request.input_text


def test_instructions_keep_named_things_out_of_persons() -> None:
    """Objects and dishes named after people are not classified as characters by the model (QA-11).

    The translation is suggested strictly for the form appearing in the text: a short name is not
    substituted by a full name from accepted decisions.
    """
    instructions = build_context().request.instructions

    assert "названные в честь людей" in instructions
    assert "перевод именно этой формы" in instructions
    assert "не подставляй полное имя" in instructions


def test_aliases_map_back_to_stable_block_ids() -> None:
    """The request alias maps back to the stable block ID."""
    section = make_analysis_section(texts=("First.", "Second."))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert context.aliases == ("b1", "b2")
    assert [context.block_id_for(alias) for alias in context.aliases] == [
        block.id for block in section.ordered_blocks
    ]


def test_expectation_describes_aliases_and_target_language() -> None:
    """Expected result describes what the model saw: block aliases."""
    context = build_context(texts=("First.", "Second."))

    expectation = context.expectation

    assert expectation.block_ids == ("b1", "b2")
    assert expectation.target_language == TARGET_LANGUAGE
    assert all(block.treatment is None for block in expectation.blocks)


def test_unknown_alias_is_rejected() -> None:
    """An alias not present in the request cannot be mapped to a block."""
    context = build_context()

    with pytest.raises(ContextBuildError):
        context.block_id_for("b99")


def test_instructions_describe_every_textual_treatment() -> None:
    """Instructions describe every treatment mode and both publication languages (PRD § 10.1)."""
    instructions = build_context().request.instructions

    for treatment in (
        BlockTreatment.TRANSLATE,
        BlockTreatment.GLOSSARY_ONLY,
        BlockTreatment.READ_AS_IS,
    ):
        assert treatment.value in instructions
    assert BlockTreatment.PRESERVE_ONLY.value not in instructions
    assert f"учит язык {SOURCE_LANGUAGE}" in instructions
    assert f"владеет языком {TARGET_LANGUAGE}" in instructions


def test_analysis_instructions_name_reader_level_b1_b2() -> None:
    assert "B1–B2" in build_context().request.instructions


def test_analysis_instructions_adapt_to_reader_level() -> None:
    """Analysis instructions adapt guidance to the reader level (PRD § 30)."""
    a2_instructions = build_context(reader_level="a2").request.instructions
    assert "A2" in a2_instructions
    assert "65–80%" in a2_instructions
    assert "4–8 слов" in a2_instructions
    assert "might have been" in a2_instructions
    assert "what" in a2_instructions
    assert "отрыв предлога" in a2_instructions

    b1_instructions = build_context(reader_level="b1").request.instructions
    assert "B1" in b1_instructions
    assert "50–70%" in b1_instructions

    b2_instructions = build_context(reader_level="b2").request.instructions
    assert "B1–B2" in b2_instructions
    assert "60%" in b2_instructions

    c1_instructions = build_context(reader_level="c1").request.instructions
    assert "C1" in c1_instructions
    assert "65–85%" in c1_instructions


def test_a2_analysis_preserves_affirmative_meaning_of_double_negation() -> None:
    instructions = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )

    assert "not saying ... isn’t ..." in instructions
    assert "сохрани его утвердительный смысл" in instructions
    assert "признание качества персонажа или события" in instructions
    assert "Объяснение такого выражения в Before You Read" in instructions
    assert "формулируй утвердительно" in instructions
    assert "не отрицаю, что план удачный" in instructions
    assert "Говорящий признаёт, что план удачный" in instructions


def test_before_you_read_instructions_use_advance_organizer() -> None:
    """Before You Read instructions require thematic anchors and semantic traps."""
    instructions = build_context().request.instructions
    assert "advance organizer" in instructions
    assert "thematic / setting anchors" in instructions
    assert "semantic traps" in instructions


def test_analysis_instructions_give_long_block_examples_for_lighter_modes() -> None:
    instructions = build_context().request.instructions
    assert "длинный абзац" in instructions
    assert "READ_AS_IS" in instructions
    assert "GLOSSARY_ONLY" in instructions
    assert "сомневаешься" in instructions
    assert "половине блоков" in instructions


def test_analysis_instructions_state_length_is_not_criterion() -> None:
    assert "Длина блока не определяет режим" in build_context().request.instructions


def test_instructions_name_every_glossary_concept_type() -> None:
    """Instructions name every glossary concept type (PRD § 12)."""
    instructions = build_context().request.instructions

    for concept_type in ConceptType:
        assert concept_type.value in instructions


def test_instructions_require_chapter_summary_in_source_language() -> None:
    """Instructions require a chapter summary and `Before You Read` material (PRD § 19)."""
    instructions = build_context().request.instructions

    assert "chapter_summary" in instructions
    assert "before_you_read" in instructions
    assert "translation_instructions" in instructions
    assert "glossary_candidates" in instructions


def test_a2_summary_is_single_source_paragraph_for_whole_chapter_and_fragment() -> None:
    whole = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    fragment = chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )

    for instructions in (whole, fragment):
        assert "ровно один абзац на языке оригинала" in instructions
        assert "без перевода на русский" in instructions
        assert "ровно два абзаца" not in instructions

    assert "ровно один абзац на языке оригинала" not in chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="b2",
    )


def test_chapter_without_textual_blocks_has_no_context() -> None:
    """A chapter without text has nothing to analyze: context cannot be built."""
    section = make_analysis_section(
        blocks=(
            make_analysis_block(
                section_id="section-1", order=0, text="", type=BlockType.IMAGE
            ),
        )
    )
    document = make_analysis_document((section,))

    with pytest.raises(ContextBuildError):
        ContextBuilder().for_chapter_analysis(document=document, section=section)


def test_negative_candidate_limit_is_rejected() -> None:
    """A negative candidate limit is a configuration error."""
    with pytest.raises(ContextBuildError):
        ContextBuilder(max_accumulated_candidates=-1)


def test_fragment_context_contains_only_its_own_blocks() -> None:
    """A chapter fragment sees only its own blocks and neither the rest of the chapter nor neighbours."""
    context = build_fragment_context(order=1)

    input_text = context.request.input_text

    assert context.aliases == ("b1", "b2")
    assert "[b1]\nParagraph three." in input_text
    assert "[b2]\nParagraph four." in input_text
    assert "Paragraph one." not in input_text
    assert NEIGHBOUR_TEXT not in input_text


def test_fragment_context_names_its_place_in_the_chapter() -> None:
    """The fragment request states its chapter index: summaries merge in sequence order."""
    input_text = build_fragment_context(order=0).request.input_text

    assert "Фрагмент главы: 1-й из 2" in input_text
    assert "Chapter 1" in input_text


def test_fragment_instructions_describe_a_part_of_the_chapter() -> None:
    """The fragment is told it sees part of a chapter, and the summary describes only that part."""
    whole = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    )
    fragment = chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    )

    assert "Тебе дают одну главу" in whole
    assert "что произошло в этой главе" in whole
    assert "Тебе дают часть одной главы" in fragment
    assert "что произошло в главе от начала до конца этого фрагмента" in fragment
    assert "Тебе дают одну главу" not in fragment


def test_analysis_instructions_forbid_block_labels() -> None:
    assert "не ссылайся на блоки по идентификаторам" in chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    )


def test_fragment_instructions_keep_the_rest_of_the_chapter_instruction() -> None:
    """Aside from fragment bounds and summary, the instructions are identical."""
    whole = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    ).splitlines()
    fragment = chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE
    ).splitlines()

    differing = [
        (chapter_line, fragment_line)
        for chapter_line, fragment_line in zip(whole, fragment, strict=True)
        if chapter_line != fragment_line
    ]

    assert len(differing) == 3
    assert all(
        "глав" in chapter_line and "фрагмент" in fragment_line
        for chapter_line, fragment_line in differing
    )


def test_fragment_outside_the_chapter_is_rejected() -> None:
    """A fragment whose index exceeds the chapter fragment count cannot be built."""
    with pytest.raises(ContextBuildError):
        build_fragment_context(order=1, fragment_count=1)


def test_clear_long_plot_with_local_vocabulary_uses_lighter_help() -> None:
    instructions = build_context().request.instructions
    assert "число незнакомых слов само по себе" in instructions
    assert "длинный сюжетный абзац" in instructions
    assert "доступный диалог" in instructions


def test_short_opaque_dialogue_needs_translation() -> None:
    assert "короткий диалог с непрозрачной смысловой связью" in build_context().request.instructions


def test_descriptive_passage_with_transparent_syntax_uses_glossary_only() -> None:
    instructions = build_context().request.instructions
    assert "описание обстановки или пейзажа" in instructions
    assert "GLOSSARY_ONLY" in instructions
    assert "sconce" in instructions
    assert "damask" in instructions
    assert "полный перевод здесь избыточен" in instructions


def test_mode_calibration_prefers_glossary_only_for_transparent_syntax() -> None:
    instructions = build_context().request.instructions
    assert "кто сделал что где" in instructions
    assert "не переводи весь абзац" in instructions


def test_plan_self_check_preserves_help_for_difficult_blocks() -> None:
    instructions = build_context().request.instructions
    for requirement in ("60% блоков", "70% объёма", "15% блоков", "не переключай самые длинные блоки", "не лишай помощи сложные блоки"):
        assert requirement in instructions
    fragment = chapter_fragment_instructions(source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE)
    assert "70% объёма" in fragment


def test_analysis_requires_natural_character_speech() -> None:
    for instructions in (build_context().request.instructions,
                         chapter_fragment_instructions(source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE)):
        assert "естественными понятными средствами" in instructions
        assert "без искусственной имитации" in instructions


def test_candidate_profile_and_fragment_byr_budgets():
    section = make_analysis_section(section_id='section-1', title='Chapter', order=0, texts=FRAGMENTED_TEXTS)
    document = make_analysis_document((section,))
    builder = ContextBuilder(content_plan_profile='content-plan-v2', language_notes_per_chapter=3)
    fragments = plan_fragments(textual_blocks(section), max_blocks=1)
    contexts = [builder.for_chapter_fragment(document=document, section=section, fragment=f, fragment_count=len(fragments)) for f in fragments]
    assert sum(c.before_you_read_limit for c in contexts) == 10
    assert [c.before_you_read_limit for c in contexts] == [3, 3, 2, 2]
    assert all(c.request.contract_version == 2 for c in contexts)
    assert sum(c.language_notes_per_chapter for c in contexts) == 3
    assert 'игру слов' in contexts[0].request.instructions
    assert build_context().request.contract_version == 1


def test_unknown_content_profile_rejected():
    with pytest.raises(ContextBuildError):
        ContextBuilder(content_plan_profile='unknown')


def test_analysis_checks_real_processing_group_notes_budget():
    from linguary.application.analysis_merge import validate_content_plan, AnalysisMergeError
    from linguary.application.llm_dto import ChapterAnalysisResponse, ContentPlanResponse, TreatmentPlanItemResponse
    from linguary.domain.content_plan import Difficulty, SourceAnchor, RequiredPoint
    section = make_analysis_section(section_id='s', title='Chapter', order=0, texts=('Echo', 'Echo', 'Echo', 'Long text. ' * 1000))
    document = make_analysis_document((section,))
    context = ContextBuilder(content_plan_profile='content-plan-v2').for_chapter_analysis(document=document, section=section)
    difficulties = tuple(Difficulty(f'd{i}', (SourceAnchor(f'b{i}', 0, 4, 'Echo'),), f'b{i}', 'wordplay', 'critical', 'explanation', (RequiredPoint('p', 'Explain'),)) for i in range(1, 4))
    payload = ChapterAnalysisResponse('i', 's', treatment_plan=tuple(TreatmentPlanItemResponse(f'b{i}', BlockTreatment.READ_AS_IS) for i in range(1, 5)), content_plan=ContentPlanResponse(difficulties))
    assert context.note_group_limits[0][1] == 1
    with pytest.raises(AnalysisMergeError, match='реальную группу'):
        validate_content_plan(payload, context.block_ids_by_alias, {b.id: b.text for b in section.blocks}, language_notes_per_chapter=context.language_notes_per_chapter, note_limits_by_block=context.note_limits_by_block, note_group_limits=context.note_group_limits)


@pytest.mark.parametrize('level', ['a2', 'b1', 'b2', 'c1'])
def test_english_help_templates_pin_level_and_exact_plan_rules(level):
    from linguary.application.context_builder import snapshot_prompt_templates
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level=level, profile='content-plan-v3', english_help=True)
    rules = templates['english_help']
    assert level.upper() in rules
    analysis = templates['analysis']
    assert 'fragment' in templates
    content_analysis = templates['content_analysis']
    normalized_content_analysis = ' '.join(content_analysis.split())
    processing = templates['english_help_processing']
    assert 'READ_AS_IS' in analysis and 'GLOSSARY_ONLY' in analysis
    assert 'before_you_read' in analysis and 'всегда пустой список' in analysis
    assert 'пять-десять ключевых понятий' not in analysis
    assert '65–80% блоков главы' not in analysis
    assert 'fragment_assistances' not in analysis
    assert 'before_you_read' in content_analysis and 'всегда должен быть пустым' in content_analysis
    assert 'before_you_read_evidence' not in content_analysis
    assert 'Определи только конкретные слова, выражения или конструкции' in normalized_content_analysis
    assert 'которые могут помешать читателю уровня' in normalized_content_analysis
    assert 'не ограничивай их число' not in normalized_content_analysis
    assert 'каждое нечастотное, архаичное' not in normalized_content_analysis
    assert 'даже если общий смысл можно вывести из контекста' not in normalized_content_analysis
    assert 'GLOSSARY_ONLY или при полном покрытии в BYR' not in content_analysis
    assert 'fragment_assistances' in processing
    assert 'language_notes' in processing and 'верни пустыми списками' in processing
    assert 'content_evidence' not in processing
    assert 'Верни content_evidence' not in processing
    assert f'Для {level.upper()}' in processing
    assert not any(f'Для {other.upper()}' in processing for other in {'a2', 'b1', 'b2', 'c1'} - {level})
    assert 'english_help' not in snapshot_prompt_templates(source_language='en', target_language='ru', reader_level=level)


def test_cross_chapter_neighbours_pulls_blocks_from_previous_story_chapter() -> None:
    """When a unit has fewer preceding blocks than limit, neighbours pull from previous story chapter (P2.T2, AC-011)."""
    from linguary.domain import TranslationUnit, TranslationUnitItem
    title_sec = make_analysis_section(section_id="sec-title", title="Title Page", order=0, texts=("Pride and Prejudice",))
    ch1 = make_analysis_section(section_id="sec-1", title="Chapter 1", order=1, texts=("Ch1 End block 1.", "Ch1 End block 2."))
    ch2 = make_analysis_section(section_id="sec-2", title="Chapter 2", order=2, texts=("Ch2 Start block.", "Ch2 Middle block."))
    doc = make_analysis_document((title_sec, ch1, ch2))

    builder = ContextBuilder()
    unit = TranslationUnit(
        id="unit-1",
        section_id="sec-2",
        items=(TranslationUnitItem(ch2.blocks[0], BlockTreatment.TRANSLATE),),
    )
    neighbours = builder._neighbours(ch2, unit, policy_version="translation-context-v2", document=doc)
    assert len(neighbours) >= 1
    before_group = next(n for n in neighbours if n.title == "До блока:")
    before_texts = [b.text for b in before_group.blocks]
    assert "Ch1 End block 2." in before_texts
    assert "Pride and Prejudice" not in before_texts


def test_snapshot_prompt_templates_defaults_content_plan_v3_to_contract_v5():
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
    )
    assert templates.get("analysis_contract_version") == 5
    assert "CHAPTER_ANALYSIS v5" in templates["content_analysis"]


def test_english_help_prompts_omit_evidence_fields():
    from linguary.application.context_builder import (
        _ENGLISH_HELP_CONTENT_PLAN_ANALYSIS,
        _ENGLISH_HELP_CONTENT_PLAN_PROCESSING,
    )
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
        english_help=True,
    )
    assert "before_you_read_evidence" not in templates["content_analysis"]
    assert "content_evidence" not in templates["content_processing"]
    assert "content_evidence" not in templates["english_help_processing"]
    assert "before_you_read_evidence" not in _ENGLISH_HELP_CONTENT_PLAN_ANALYSIS
    assert "content_evidence" not in _ENGLISH_HELP_CONTENT_PLAN_PROCESSING


def test_chapter_context_keeps_instructions_invariant_across_chapters():
    sec1 = make_analysis_section(section_id="sec-1", title="Chapter 1", order=0, texts=("First line.", "Second line."))
    sec2 = make_analysis_section(section_id="sec-2", title="Chapter 2", order=1, texts=("Another chapter line.", "And another one.", "Third line."))
    doc = make_analysis_document((sec1, sec2))
    builder = ContextBuilder(content_plan_profile="content-plan-v3", language_notes_per_chapter=5)
    ctx1 = builder.for_chapter_analysis(document=doc, section=sec1)
    ctx2 = builder.for_chapter_analysis(document=doc, section=sec2)
    assert ctx1.request.instructions == ctx2.request.instructions
    assert "В этом ответе не больше" not in ctx1.request.instructions
    assert "Бюджет обязательных заметок" not in ctx1.request.instructions
    assert "Пределы обязательных заметок" not in ctx1.request.instructions
    assert "В этом ответе не больше" in ctx1.request.input_text
    assert "Бюджет обязательных заметок" in ctx1.request.input_text
    assert "Пределы обязательных заметок" in ctx1.request.input_text
    assert '[{"blocks":["b1","b2"],"limit":' in ctx1.request.input_text
    assert "В этом ответе не больше" in ctx2.request.input_text
    assert "Бюджет обязательных заметок" in ctx2.request.input_text
    assert "Пределы обязательных заметок" in ctx2.request.input_text

    fragments = plan_fragments(textual_blocks(sec2), max_blocks=1)
    fctx1 = builder.for_chapter_fragment(document=doc, section=sec2, fragment=fragments[0], fragment_count=len(fragments))
    fctx2 = builder.for_chapter_fragment(document=doc, section=sec2, fragment=fragments[1], fragment_count=len(fragments))
    assert fctx1.request.instructions == fctx2.request.instructions
    assert "В этом ответе не больше" not in fctx1.request.instructions
    assert "В этом ответе не больше" in fctx1.request.input_text
    assert "В этом ответе не больше" in fctx2.request.input_text


def test_translation_input_filters_mandatory_and_suggested_glossary_by_unit_and_neighbours() -> None:
    """Translation input only includes glossary entries matching unit and neighbour blocks (AC-004)."""
    sec = make_analysis_section(
        section_id="sec-1",
        texts=(
            "Neighbour block before mentioning Pemberley.",
            "Unit target block mentioning Netherfield.",
            "Neighbour block after.",
        ),
    )
    doc = make_analysis_document((sec,))
    unit_block = sec.blocks[1]
    plan = TreatmentPlan.of_treatments(sec.id, ((unit_block.id, BlockTreatment.TRANSLATE),))
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=(unit_block,), plan=plan)
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=sec.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(unit_block.id,),
    )
    glossary = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Netherfield",
                translation="Незерфилд",
                type=ConceptType.PLACE,
                status=TermStatus.LOCKED,
            ),
            GlossarySnapshotEntry(
                source_text="Longbourn",
                translation="Лонгборн",
                type=ConceptType.PLACE,
                status=TermStatus.LOCKED,
            ),
            GlossarySnapshotEntry(
                source_text="Pemberley",
                translation="Пемберли",
                type=ConceptType.PLACE,
                status=TermStatus.SUGGESTED,
            ),
            GlossarySnapshotEntry(
                source_text="Rosings",
                translation="Розингс",
                type=ConceptType.PLACE,
                status=TermStatus.SUGGESTED,
            ),
        )
    )
    builder = ContextBuilder(max_accumulated_candidates=10)
    context = builder.for_translation_unit(
        document=doc,
        section=sec,
        unit=unit,
        analysis=analysis,
        glossary=glossary,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    input_text = context.request.input_text
    assert "Netherfield" in input_text
    assert "Незерфилд" in input_text
    assert "Longbourn" not in input_text
    assert "Лонгборн" not in input_text
    assert "Pemberley" in input_text
    assert "Пемберли" in input_text
    assert "Rosings" not in input_text
    assert "Розингс" not in input_text


def test_for_block_regeneration_filters_glossary_by_block_and_neighbours() -> None:
    """Block regeneration input only includes glossary entries matching block and neighbours (AC-004)."""
    sec = make_analysis_section(
        section_id="sec-1",
        texts=(
            "Neighbour block before mentioning Pemberley.",
            "Block to regenerate mentioning Netherfield.",
            "Neighbour block after.",
        ),
    )
    doc = make_analysis_document((sec,))
    unit_block = sec.blocks[1]
    plan = TreatmentPlan.of_treatments(sec.id, ((unit_block.id, BlockTreatment.TRANSLATE),))
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=(unit_block,), plan=plan)
    analysis = ChapterAnalysis.create(
        id="analysis-1",
        section_id=sec.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(unit_block.id,),
    )
    glossary = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Netherfield",
                translation="Незерфилд",
                type=ConceptType.PLACE,
                status=TermStatus.LOCKED,
            ),
            GlossarySnapshotEntry(
                source_text="Longbourn",
                translation="Лонгборн",
                type=ConceptType.PLACE,
                status=TermStatus.LOCKED,
            ),
            GlossarySnapshotEntry(
                source_text="Pemberley",
                translation="Пемберли",
                type=ConceptType.PLACE,
                status=TermStatus.SUGGESTED,
            ),
            GlossarySnapshotEntry(
                source_text="Rosings",
                translation="Розингс",
                type=ConceptType.PLACE,
                status=TermStatus.SUGGESTED,
            ),
        )
    )
    builder = ContextBuilder(max_accumulated_candidates=10)
    context = builder.for_block_regeneration(
        document=doc,
        section=sec,
        unit=unit,
        analysis=analysis,
        glossary=glossary,
    )
    input_text = context.request.input_text
    assert "Netherfield" in input_text
    assert "Незерфилд" in input_text
    assert "Longbourn" not in input_text
    assert "Лонгборн" not in input_text
    assert "Pemberley" in input_text
    assert "Пемберли" in input_text
    assert "Rosings" not in input_text
    assert "Розингс" not in input_text


def test_matches_chapter_text_requires_all_significant_words_and_ignores_honorifics() -> None:
    from linguary.application.context_builder import _extract_word_stems, _matches_chapter_text

    jane_text = "Jane Bennet smiled at the news."
    jane_stems = _extract_word_stems(jane_text)
    assert not _matches_chapter_text("Elizabeth Bennet", jane_text, jane_stems)
    assert _matches_chapter_text("Jane Bennet", jane_text, jane_stems)
    assert _matches_chapter_text("Mr. Bennet", jane_text, jane_stems)
    assert _matches_chapter_text("Mrs. Bennet", jane_text, jane_stems)

    battle_text = "During the great battles around Hogwarts, nobody slept."
    battle_stems = _extract_word_stems(battle_text)
    assert _matches_chapter_text("The Battle of Hogwarts", battle_text, battle_stems)

    hogwarts_only_text = "The Hogwarts express left from the station."
    hogwarts_only_stems = _extract_word_stems(hogwarts_only_text)
    assert not _matches_chapter_text("The Battle of Hogwarts", hogwarts_only_text, hogwarts_only_stems)


def test_a2_chapter_context_deduplicates_learner_quality_and_pronunciation_omits_form() -> None:
    section = make_analysis_section(
        section_id="sec-a2", title="Chapter 1", order=0, texts=("First paragraph.",)
    )
    document = make_analysis_document((section,))
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")
    context = builder.for_chapter_analysis(document=document, section=section)

    instructions = context.request.instructions
    assert instructions.count("might have been") == 1
    assert instructions.count("not saying ... isn’t ...") == 1
    assert instructions.count("ровно один абзац на языке оригинала") == 1
    assert "Пересказ A2 верни ровно одним абзацем" not in instructions

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        pronunciation_help_enabled=True,
    )
    assert "anchor {block_id, quote}, context" in templates["pronunciation_processing"]
    assert "form=quote" not in templates["pronunciation_processing"]


def test_translation_input_and_lazy_input_append_pronunciation_note_preserving_block_header_and_prompt_sha() -> None:
    from hashlib import sha256
    from linguary.application.book_conditions import payload_digest
    from linguary.application.context_builder import (
        _LazyTranslationInput,
        _TRANSLATION_INSTRUCTIONS,
    )
    from linguary.domain.content_plan import (
        ContentPlan,
        Difficulty,
        RequiredPoint,
        SourceAnchor,
    )

    assert (
        sha256(_TRANSLATION_INSTRUCTIONS.encode("utf-8")).hexdigest()
        == "8c39d00a516d9db9a8a0079602c48bcfb6d9fc02714338d5d444be6304acbafa"
    )
    baseline_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        pronunciation_help_enabled=True,
    )
    assert (
        payload_digest(baseline_templates)
        == "3bcf968fbeb3eea72bc191df4360d50ec62f005d45a670b54dbd738ef2fb069a"
    )

    sec = make_analysis_section(
        section_id="sec-mixed",
        texts=(
            "Neighbour paragraph before the group.",
            "Opaque opening line requiring full translation.",
            "Simple clear line in Worcester.",
            "Another clear line in Cholmondeley.",
            "Neighbour paragraph after the group.",
        ),
    )
    doc = make_analysis_document((sec,))
    unit_blocks = sec.blocks[1:4]
    plan = TreatmentPlan.of_treatments(
        sec.id,
        (
            (sec.blocks[0].id, BlockTreatment.READ_AS_IS),
            (unit_blocks[0].id, BlockTreatment.TRANSLATE),
            (unit_blocks[1].id, BlockTreatment.READ_AS_IS),
            (unit_blocks[2].id, BlockTreatment.READ_AS_IS),
            (sec.blocks[4].id, BlockTreatment.READ_AS_IS),
        ),
    )
    unit = TranslationUnit.of_blocks(id="unit-mixed", blocks=unit_blocks, plan=plan)
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                "diff-1",
                (SourceAnchor(unit_blocks[0].id, 0, 6, "Opaque"),),
                unit_blocks[0].id,
                "grammar",
                "critical",
                "explanation",
                (RequiredPoint("p-1", "Explain syntax"),),
            ),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-mixed",
        section_id=sec.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=tuple(b.id for b in sec.blocks),
        content_plan=content_plan,
    )
    glossary = GlossarySnapshot()

    v3_pron_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        profile="content-plan-v3",
        pronunciation_help_enabled=True,
    )
    builder_with_pron = ContextBuilder(
        prompt_templates=v3_pron_templates,
        content_plan_profile="content-plan-v3",
    )
    ctx = builder_with_pron.for_translation_unit(
        document=doc,
        section=sec,
        unit=unit,
        analysis=analysis,
        glossary=glossary,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )

    assert ctx.pure_read_as_is_aliases == frozenset({"b2", "b3"})
    assert isinstance(ctx._input_without_optional_neighbours, _LazyTranslationInput)
    without_neighbours_text = ctx.without_optional_neighbours().request.input_text
    assert "Соседние абзацы главы" in ctx.request.input_text
    assert "Соседние абзацы главы" not in without_neighbours_text

    for rendered_input in (ctx.request.input_text, without_neighbours_text):
        assert (
            "[b2] режим READ_AS_IS (только для контекста, в blocks не возвращать)\n"
            "Simple clear line in Worcester."
        ) in rendered_input
        assert (
            "[b3] режим READ_AS_IS (только для контекста, в blocks не возвращать)\n"
            "Another clear line in Cholmondeley."
        ) in rendered_input

        blocks_section = rendered_input.split("Блоки для обработки:\n\n", 1)[1].split(
            "\n\nОбязательные аспекты и точные связанные привязки:", 1
        )[0]
        section_paragraphs = blocks_section.split("\n\n")
        note_line = section_paragraphs[-1]
        assert "\n" not in note_line
        assert "pronunciation_candidates" in note_line
        assert "b1..b3" in note_line
        assert "READ_AS_IS" in note_line
        assert "b2, b3" in note_line
        assert "anchor.block_id" in note_line

    # Single context-only READ_AS_IS block in mixed group also gets singular note referencing b2.
    single_pure_plan = TreatmentPlan.of_treatments(
        sec.id,
        (
            (sec.blocks[0].id, BlockTreatment.READ_AS_IS),
            (unit_blocks[0].id, BlockTreatment.TRANSLATE),
            (unit_blocks[1].id, BlockTreatment.READ_AS_IS),
            (unit_blocks[2].id, BlockTreatment.GLOSSARY_ONLY),
            (sec.blocks[4].id, BlockTreatment.READ_AS_IS),
        ),
    )
    single_pure_unit = TranslationUnit.of_blocks(
        id="unit-single-pure", blocks=unit_blocks, plan=single_pure_plan
    )
    single_pure_analysis = ChapterAnalysis.create(
        id="analysis-single-pure",
        section_id=sec.id,
        treatment_plan=single_pure_plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=tuple(b.id for b in sec.blocks),
        content_plan=content_plan,
    )
    ctx_single_pure = builder_with_pron.for_translation_unit(
        document=doc,
        section=sec,
        unit=single_pure_unit,
        analysis=single_pure_analysis,
        glossary=glossary,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    assert ctx_single_pure.pure_read_as_is_aliases == frozenset({"b2"})
    for rendered_single in (
        ctx_single_pure.request.input_text,
        ctx_single_pure.without_optional_neighbours().request.input_text,
    ):
        single_note = (
            rendered_single.split("Блоки для обработки:\n\n", 1)[1]
            .split("\n\nОбязательные аспекты и точные связанные привязки:", 1)[0]
            .split("\n\n")[-1]
        )
        assert "блок READ_AS_IS (b2)" in single_note
        assert "ссылайся на b2 в anchor.block_id" in single_note

    # Without pronunciation active, mixed group keeps exact header and omits pronunciation note.
    builder_no_pron = ContextBuilder(content_plan_profile="content-plan-v3")
    ctx_no_pron = builder_no_pron.for_translation_unit(
        document=doc,
        section=sec,
        unit=unit,
        analysis=analysis,
        glossary=glossary,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    assert (
        "[b2] режим READ_AS_IS (только для контекста, в blocks не возвращать)\n"
        in ctx_no_pron.request.input_text
    )
    assert "pronunciation_candidates" not in ctx_no_pron.request.input_text
    assert (
        "pronunciation_candidates"
        not in ctx_no_pron.without_optional_neighbours().request.input_text
    )

    # With pronunciation active but no context-only READ_AS_IS blocks, note is omitted.
    non_mixed_plan = TreatmentPlan.of_treatments(
        sec.id,
        (
            (sec.blocks[0].id, BlockTreatment.READ_AS_IS),
            (unit_blocks[0].id, BlockTreatment.TRANSLATE),
            (unit_blocks[1].id, BlockTreatment.GLOSSARY_ONLY),
            (unit_blocks[2].id, BlockTreatment.TRANSLATE),
            (sec.blocks[4].id, BlockTreatment.READ_AS_IS),
        ),
    )
    non_mixed_unit = TranslationUnit.of_blocks(
        id="unit-non-mixed", blocks=unit_blocks, plan=non_mixed_plan
    )
    non_mixed_analysis = ChapterAnalysis.create(
        id="analysis-non-mixed",
        section_id=sec.id,
        treatment_plan=non_mixed_plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=tuple(b.id for b in sec.blocks),
        content_plan=content_plan,
    )
    ctx_non_mixed = builder_with_pron.for_translation_unit(
        document=doc,
        section=sec,
        unit=non_mixed_unit,
        analysis=non_mixed_analysis,
        glossary=glossary,
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    assert ctx_non_mixed.pure_read_as_is_aliases == frozenset()
    assert "pronunciation_candidates" not in ctx_non_mixed.request.input_text
    assert (
        "pronunciation_candidates"
        not in ctx_non_mixed.without_optional_neighbours().request.input_text
    )


_LONG_BLOCK = "Alpha went home. Beta stayed."
_SHORT_BLOCK = "Short reply."
_SINGLE_BLOCK = "Only one."
_SEGMENT_RULE = (
    "Верни segments с режимом каждого сегмента, только если режим различается внутри блока, иначе []."
)
_BLOCK_MODES_ALLOWED = "Разные режимы внутри одного блока допустимы."
_SENTENCE_MODES_FORBIDDEN = "Разные режимы внутри одного предложения запрещены."
_FALSE_FRIEND_KIND = "false_friend — слово похоже на русское, значит другое."
_SPEECH_STYLE_KIND = (
    "speech_style — диалект, акцент, особенности речи; только первое появление особенности персонажа."
)
_PARODY_KIND = "parody — в required_points назвать исходный текст."
_DESCRIPTION_RULE = (
    "description заполняй для всех 5 типов "
    "(person, place, organization, fictional_concept, term) — "
    "1–2 предложения на языке перевода (ru), "
    "строго до первого упоминания в данной главе "
    "и по прошлым главам, без спойлеров текущей и будущих глав."
)


def _segmented_chapter():
    """A long block with two spans, a short block, and one span stored then dropped."""
    from dataclasses import replace

    from linguary.domain import BlockSegment

    split = _LONG_BLOCK.index("Beta")
    long_block = replace(
        make_analysis_block(section_id="section-1", order=0, text=_LONG_BLOCK),
        segments=(
            BlockSegment(0, split, 0),
            BlockSegment(split, len(_LONG_BLOCK), 1),
        ),
    )
    short_block = make_analysis_block(section_id="section-1", order=1, text=_SHORT_BLOCK)
    single_block = replace(
        make_analysis_block(section_id="section-1", order=2, text=_SINGLE_BLOCK),
        segments=(BlockSegment(0, len(_SINGLE_BLOCK), 0),),
    )
    section = make_analysis_section(
        section_id="section-1",
        blocks=(long_block, short_block, single_block),
    )
    return section, make_analysis_document((section,)), long_block


def _staged_builder() -> ContextBuilder:
    return ContextBuilder(content_plan_profile="content-plan-v3", reader_level="b1")


def test_pedagogical_input_labels_segments_of_long_block() -> None:
    """A multi-segment block is `[bN]` plus `[bN.k]`; one stored span is not `[bN.1]`."""
    section, document, long_block = _segmented_chapter()
    context = _staged_builder().for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Alice meets the rabbit.",
    )
    split = _LONG_BLOCK.index("Beta")
    labeled = "\n".join(
        f"[b1.{index}] {_LONG_BLOCK[segment.start:segment.end]}"
        for index, segment in enumerate(long_block.segments, start=1)
    )
    input_text = context.request.input_text

    assert long_block.segments[0].start == 0
    assert _LONG_BLOCK[:split].endswith(" ")
    assert f"[b1]\n{labeled}" in input_text
    assert "[b1.1] " in input_text
    assert "[b1.2] Beta stayed." in input_text
    assert f"[b2]\n{_SHORT_BLOCK}" in input_text
    assert "[b2.1]" not in input_text
    assert f"[b3]\n{_SINGLE_BLOCK}" in input_text
    assert "[b3.1]" not in input_text


def test_narrative_input_keeps_plain_text() -> None:
    """Narrative input stays continuous text even when the block stores several spans."""
    section, document, _long_block = _segmented_chapter()
    context = _staged_builder().for_chapter_narrative_analysis(document, section)
    input_text = context.request.input_text

    assert f"[b1]\n{_LONG_BLOCK}" in input_text
    assert f"[b2]\n{_SHORT_BLOCK}" in input_text
    assert f"[b3]\n{_SINGLE_BLOCK}" in input_text
    assert "[b1.1]" not in input_text
    assert "[b1.2]" not in input_text
    assert "[b2.1]" not in input_text
    assert "[b3.1]" not in input_text


def test_quotes_resolved_without_segment_labels() -> None:
    """Quote matching uses the unlabeled visible text, not the pedagogical segment lines."""
    from linguary.application.analysis_merge import find_quote_span
    from linguary.application.context_builder import source_text_for_model

    section, document, long_block = _segmented_chapter()
    context = _staged_builder().for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Alice meets the rabbit.",
    )
    visible = context.visible_texts_by_alias["b1"]
    quote = "Beta stayed."
    start = _LONG_BLOCK.index(quote)

    assert "[b1.2] Beta stayed." in context.request.input_text
    assert visible == source_text_for_model(long_block) == _LONG_BLOCK
    assert "[b1.1]" not in visible
    assert "[b1.2]" not in visible
    assert find_quote_span(visible, quote) == (start, start + len(quote), quote)
    assert find_quote_span(visible, f"[b1.2] {quote}") is None

    from dataclasses import replace

    from linguary.domain import BlockSegment, InlineAnnotation, InlineAnnotationType

    foot = "Alpha went home.1 Beta stayed."
    mark = foot.index("1")
    foot_block = replace(
        make_analysis_block(section_id="section-1", order=0, text=foot),
        inline_annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=mark,
                end=mark + 1,
                target="section-1.xhtml#fn1",
            ),
        ),
        segments=(
            BlockSegment(0, mark + 1, 0),
            BlockSegment(mark + 1, len(foot), 1),
        ),
    )
    foot_section = make_analysis_section(section_id="section-1", blocks=(foot_block,))
    foot_context = _staged_builder().for_chapter_pedagogical_analysis(
        make_analysis_document((foot_section,)),
        foot_section,
        narrative_summary="Alice meets the rabbit.",
    )
    foot_visible = foot_context.visible_texts_by_alias["b1"]
    assert foot_visible == source_text_for_model(foot_block) == "Alpha went home. Beta stayed."
    assert "[b1.1] Alpha went home." in foot_context.request.input_text
    assert "[b1.2]  Beta stayed." in foot_context.request.input_text
    assert "home.1" not in foot_context.request.input_text
    assert "1" not in foot_visible
    foot_start = foot_visible.index(quote)
    assert find_quote_span(foot_visible, quote) == (foot_start, foot_start + len(quote), quote)


def test_pedagogical_instructions_describe_new_kinds() -> None:
    """Pedagogical instructions name segment treatments and the three new difficulty kinds."""
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
    )
    section, document, _long_block = _segmented_chapter()
    live = _staged_builder().for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Alice meets the rabbit.",
    )
    from_template = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="b1",
        prompt_templates=templates,
    ).for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Alice meets the rabbit.",
    )
    expected = (
        _SEGMENT_RULE,
        _BLOCK_MODES_ALLOWED,
        _SENTENCE_MODES_FORBIDDEN,
        _FALSE_FRIEND_KIND,
        _SPEECH_STYLE_KIND,
        _PARODY_KIND,
    )
    for instructions in (
        templates["pedagogical_analysis"],
        live.request.instructions,
        from_template.request.instructions,
    ):
        for phrase in expected:
            assert phrase in instructions


def test_narrative_instructions_require_target_language_description() -> None:
    """Person and place descriptions are written in the target language, before the first mention."""
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="de",
        reader_level="b1",
        profile="content-plan-v3",
    )
    section, document, _long_block = _segmented_chapter()
    context = _staged_builder().for_chapter_narrative_analysis(document, section)
    russian = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
    )

    german = templates["narrative_analysis"]
    assert "description заполняй для всех 5 типов" in german
    assert "(person, place, organization, fictional_concept, term)" in german
    assert "языке перевода (de)" in german
    assert "строго до первого упоминания в данной главе" in german
    assert "без спойлеров текущей и будущих глав" in german
    assert "языке перевода (ru)" in russian["narrative_analysis"]
    assert "языке перевода (ru)" in context.request.instructions
    assert "языке перевода (de)" not in context.request.instructions


def test_a2_pedagogical_instructions_include_grammar_support_and_strict_dialogue_window() -> None:
    """A2 pedagogy limits the short-reply window and includes grammar-support guidance."""
    from linguary.application.context_builder import _A2_GRAMMAR_SUPPORT_GUIDANCE

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    section, document, _long_block = _segmented_chapter()
    live = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    ).for_chapter_pedagogical_analysis(
        document, section, narrative_summary="A short factual summary.",
    )
    b1 = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
    )
    grammar = _A2_GRAMMAR_SUPPORT_GUIDANCE.strip()
    window = (
        "Окно 4–8 слов для READ_AS_IS или GLOSSARY_ONLY относится только к одной "
        "элементарной реплике уровня A1–A2 без придаточных оборотов "
        "(if, when, because, that), причастных оборотов, "
        "авторских предложений действия и небазовых глаголов речи "
        "(demanded, murmured, snapped)"
    )
    removed_promote = (
        "Реплики длиннее 8 слов либо сочетающие прямую речь с авторским действием "
        "или сложной ремаркой получают TRANSLATE."
    )
    for instructions in (templates["pedagogical_analysis"], live.request.instructions):
        assert grammar in instructions
        assert window in instructions
        assert removed_promote not in instructions
        assert "до 15 слов" in instructions
        assert "до 45 слов" in instructions
    assert grammar not in b1["pedagogical_analysis"]
    assert "элементарной реплике" not in b1["pedagogical_analysis"]


def test_narrative_instructions_request_spoiler_free_description() -> None:
    """Person and place descriptions stay before the first mention and out of later chapters."""
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
    )
    section, document, _long_block = _segmented_chapter()
    context = _staged_builder().for_chapter_narrative_analysis(document, section)
    from_template = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="b1",
        prompt_templates=templates,
    ).for_chapter_narrative_analysis(document, section)

    for instructions in (
        templates["narrative_analysis"],
        context.request.instructions,
        from_template.request.instructions,
    ):
        assert _DESCRIPTION_RULE in instructions


def test_instructions_have_no_language_reference_requirement() -> None:
    """Neither pass asks the model to insert the words «Языковая справка»."""
    import inspect

    import linguary.application.context_builder as context_builder_module
    import linguary.application.reader_text as reader_text_module
    from linguary.application.reader_text import learner_instructions

    assert "Языковая справка" not in inspect.getsource(reader_text_module)
    assert "Языковая справка" not in inspect.getsource(context_builder_module)
    for level in ("a2", "b1", "b2", "c1"):
        assert "Языковая справка" not in learner_instructions(level)
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        profile="content-plan-v3",
    )
    for key in ("learner_quality", "pedagogical_analysis", "narrative_analysis", "analysis", "fragment"):
        assert "Языковая справка" not in templates[key]
    section = make_analysis_section()
    document = make_analysis_document((section,))
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="b2")
    pedagogical = builder.for_chapter_pedagogical_analysis(
        document, section, narrative_summary="A factual summary.",
    )
    narrative = builder.for_chapter_narrative_analysis(document, section)
    legacy = builder.for_chapter_analysis(document=document, section=section)
    assert "Языковая справка" not in pedagogical.request.instructions
    assert "Языковая справка" not in narrative.request.instructions
    assert "Языковая справка" not in legacy.request.instructions


_OFFSET_ONCE = "Если цитата в блоке одна, укажи start=null и end=null."
_OFFSET_REPEATED = (
    "Если цитата встречается больше одного раза, верни start и end этого вхождения, "
    "quote == text[start:end]."
)


def test_instructions_request_offset_only_for_repeated_quote() -> None:
    """V5 prompts ask for null offsets on a unique quote and start/end when repeated."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_ANALYSIS_V5,
        _ENGLISH_HELP_CONTENT_PLAN_ANALYSIS_V5,
        _NARRATIVE_ANALYSIS_INSTRUCTIONS,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
    )

    english = _ENGLISH_HELP_CONTENT_PLAN_ANALYSIS_V5.format(reader_level="B1")
    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")
    pedagogical = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="B1",
    )
    templates = snapshot_prompt_templates(
        source_language="en", target_language="ru", reader_level="b1",
        profile="content-plan-v3",
    )
    english_templates = snapshot_prompt_templates(
        source_language="en", target_language="ru", reader_level="b1",
        profile="content-plan-v3", english_help=True,
    )
    texts = (
        _CONTENT_PLAN_ANALYSIS_V5,
        english,
        narrative,
        pedagogical,
        templates["content_analysis"],
        templates["narrative_analysis"],
        templates["pedagogical_analysis"],
        english_templates["content_analysis"],
    )
    for text in texts:
        assert _OFFSET_ONCE in text
        assert _OFFSET_REPEATED in text
        assert "start и end не возвращай" not in text
        assert "у опор нет полей start и end" not in text
        assert "Не возвращай смещения" not in text
        assert "приложение найдёт точные координаты" not in text
        assert "приложение вычислит координаты" not in text
    assert "CHAPTER_ANALYSIS v5" in _CONTENT_PLAN_ANALYSIS_V5
    assert "CHAPTER_ANALYSIS v5" in english


_A2_EMPTY_BEFORE_YOU_READ = (
    "before_you_read: []",
    "before_you_read_evidence: []",
    "language_notes",
    "под абзацем",
)
_A2_SUMMARY_LENGTH = "3–5 короткими предложениями"
_BYR_EXPLANATION = "уникальное объяснение костров 515151"


def _translation_request_text(*, reader_level: str | None, english_help: bool = False) -> str:
    """Translation prompt text for one unit that already has a Before You Read entry."""
    from linguary.domain import BeforeYouReadEntry

    section = make_analysis_section(
        section_id="sec-byr",
        texts=(
            "Neighbour paragraph before the ship.",
            "The ship heaved and groaned tonight.",
            "Neighbour paragraph after the ship.",
        ),
    )
    document = make_analysis_document((section,))
    unit_block = section.blocks[1]
    plan = TreatmentPlan.of_treatments(
        section.id, ((unit_block.id, BlockTreatment.TRANSLATE),)
    )
    unit = TranslationUnit.of_blocks(id="unit-byr", blocks=(unit_block,), plan=plan)
    analysis = ChapterAnalysis.create(
        id="analysis-byr",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="Сохраняй смысл.",
        chapter_summary="The ship groaned.",
        expected_block_ids=(unit_block.id,),
        before_you_read=(
            BeforeYouReadEntry("heaved and groaned", _BYR_EXPLANATION),
        ),
    )
    builder = ContextBuilder(reader_level="b2")
    builder._reader_level = reader_level
    if english_help:
        from types import MappingProxyType

        builder._english_help = True
        builder._prompt_templates = MappingProxyType({
            "english_help": "enabled",
            "english_help_processing": "Помощь рядом с оригиналом.",
        })
    context = builder.for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=GlossarySnapshot(),
        policy_version=CURRENT_TRANSLATION_POLICY,
    )
    return context.request.input_text


def test_a2_context_disables_before_you_read_and_keeps_inline_notes() -> None:
    """A2 without english help explains hard spots under the paragraph and drops Before You Read."""
    from linguary.application.reader_text import learner_instructions

    whole = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    fragment = chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    learner = learner_instructions("a2")
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    for instructions in (
        whole,
        fragment,
        learner,
        templates["analysis"],
        templates["fragment"],
        templates["content_analysis"],
        templates["pedagogical_analysis"],
    ):
        for phrase in _A2_EMPTY_BEFORE_YOU_READ:
            assert phrase in instructions
    assert _A2_SUMMARY_LENGTH in whole
    assert "на всю главу" in whole
    assert "chapter_summary" in whole
    assert "reader_chapter_summary" in whole
    assert "3–5 коротких предложений" in fragment
    assert "с начала главы до текущего фрагмента" in fragment
    assert "абзацем к каждому фрагменту" in fragment
    for instructions in (whole, fragment, templates["analysis"], templates["fragment"]):
        assert "пять-десять ключевых понятий" not in instructions
    assert "выражения before_you_read возвращай только новые" not in fragment
    assert "выражения before_you_read возвращай только новые" not in templates["fragment"]

    narrative = templates["narrative_analysis"]
    narrative_section = make_analysis_section(texts=("Mr. and Mrs. Dursley woke up.",))
    live_narrative = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    ).for_chapter_narrative_analysis(
        make_analysis_document((narrative_section,)),
        narrative_section,
    )
    for instructions in (narrative, live_narrative.request.instructions):
        assert "3–5 предложений" in instructions
        assert "до текущего фрагмента" in instructions
        assert "3–6" not in instructions
        assert "продолжи сводку" not in instructions

    b2 = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="b2",
    )
    none_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level=None,
    )
    b2_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
    )
    for instructions in (b2, none_templates["analysis"], b2_templates["analysis"]):
        assert "before_you_read: []" not in instructions
        assert "advance organizer" in instructions
    for level in ("b1", "b2"):
        assert "before_you_read: []" not in learner_instructions(level)
        assert "advance organizer" in learner_instructions(level) or "before_you_read" in learner_instructions(level)

    assert _BYR_EXPLANATION in _translation_request_text(reader_level="b2")
    assert _BYR_EXPLANATION in _translation_request_text(reader_level=None)
    assert _BYR_EXPLANATION in _translation_request_text(reader_level="a2", english_help=True)
    assert _BYR_EXPLANATION not in _translation_request_text(reader_level="a2")


def test_pedagogical_instructions_prioritize_cultural_reference_bonfire_night() -> None:
    """Cultural references are priority explanations, and guessed person names stay out of the glossary."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_ANALYSIS,
        _NARRATIVE_ANALYSIS_INSTRUCTIONS,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
    )

    cultural = (
        "cultural_reference",
        'help_kind="explanation"',
        "Bonfire Night",
        "Ночь костров",
        "5 ноября",
        "костры и фейерверки",
        "TRANSLATE",
    )
    names = (
        "glossary_candidates",
        "person",
        "случайно упомянутых",
        "клички животных",
        "собирательные названия семей",
        "каноническ",
    )
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b1",
        profile="content-plan-v3",
    )
    pedagogical = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="B1",
    )
    for instructions in (
        pedagogical,
        templates["pedagogical_analysis"],
        _CONTENT_PLAN_ANALYSIS,
        templates["content_analysis"],
    ):
        for phrase in cultural:
            assert phrase in instructions
    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru",
    )
    for instructions in (
        narrative,
        templates["narrative_analysis"],
    ):
        for phrase in names:
            assert phrase in instructions


_CALQUE_AND_JARGON_PHRASES = (
    "хотя зато",
    "больше того",
    "окказиональный",
    "окказиональное прилагательное/сложение",
    "дериват",
    "инверсия",
    "эллипсис",
    "вариант стоит проверить",
    "в переводе передано как...",
    "не X, а Y",
    "false_friend",
    "glossary_hints.note",
)


def test_translation_and_note_instructions_forbid_calques_and_academic_jargon() -> None:
    """Translation and note prompts forbid calques, academic jargon, and empty glossary notes."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING,
        _CONTENT_PLAN_PROCESSING_V3,
        _TRANSLATION_CONTEXT_V2,
    )
    from linguary.application.reader_text import learner_instructions

    for instructions in (
        _TRANSLATION_CONTEXT_V2,
        _CONTENT_PLAN_PROCESSING,
        _CONTENT_PLAN_PROCESSING_V3,
        learner_instructions("a2"),
    ):
        for phrase in _CALQUE_AND_JARGON_PHRASES:
            assert phrase in instructions
    for instructions in (
        _TRANSLATION_CONTEXT_V2,
        _CONTENT_PLAN_PROCESSING_V3,
        learner_instructions("a2"),
    ):
        assert "помета к глаголу речи" in instructions
        assert "страдательный оборот" in instructions
        assert "to snigger" in instructions
        assert "V2/V3" in instructions
        assert "не склонять" in instructions


_A2_SHORT_LINE_HELP = (
    "winked",
    "groaned",
    "chuckled",
    "peered",
    "nodded",
    "sniggered",
    "moaned",
    "What brings you here",
    "I suppose so",
    "sweetheart",
    "How very wrong he was",
    "нельзя оставлять в READ_AS_IS",
    "GLOSSARY_ONLY или TRANSLATE",
    "The House on the Hill",
    "The Missing Key",
    "Letters from Abroad",
    "без грамматических сносок",
)
_A2_TRIVIAL_NOTE_BAN = (
    "не больше 1–2 заметок на абзац",
    "прозрачных местоимениях",
    "базовых предлогах",
    "разные концы абзаца",
    "длиннее 5 слов",
    "original_excerpt",
    "третья форма глагола",
    "to snigger — хихикать",
    "to nod — кивать",
    "note пустым",
    "помета к глаголу речи",
    "«реалия»",
    "страдательный оборот",
    "превосходная степень",
    "причастие прошедшего времени",
    "целое предложение",
)
_NOT_A_PERSON = (
    "типом person",
    "бытовые предметы",
    "в этой главе упоминается впервые",
    "до этого упоминания сведений нет",
)


def test_a2_instructions_require_help_for_winked_and_forbid_trivial_translate_notes() -> None:
    """A2 helps short B1+ lines and rejects trivial notes, object-persons, and boilerplate."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_ANALYSIS_V5,
        _CONTENT_PLAN_PROCESSING_V3,
        _NARRATIVE_ANALYSIS_INSTRUCTIONS,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _TRANSLATION_CONTEXT_V2,
        _reader_level_treatment_guidance,
    )
    from linguary.application.reader_text import learner_instructions

    guidance = _reader_level_treatment_guidance("a2")
    b1_guidance = _reader_level_treatment_guidance("b1")
    assert "вроде «Nearly.»" in guidance
    assert "вроде «It winked.»" not in guidance
    assert "«It winked.»" in guidance
    for phrase in _A2_SHORT_LINE_HELP:
        assert phrase in guidance
        assert phrase not in b1_guidance

    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    note_sources = (
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _CONTENT_PLAN_ANALYSIS_V5,
        _CONTENT_PLAN_PROCESSING_V3,
        _TRANSLATION_CONTEXT_V2,
        learner_instructions("a2"),
        templates["pedagogical_analysis"],
        templates["content_analysis"],
        templates["content_processing"],
        templates["learner_quality"],
    )
    for instructions in note_sources:
        for phrase in _A2_TRIVIAL_NOTE_BAN:
            assert phrase in instructions
    for phrase in _A2_SHORT_LINE_HELP:
        assert phrase in templates["pedagogical_analysis"]
    assert "помета к глаголу речи" not in learner_instructions("b1")

    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru",
    )
    a2_analysis = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    for instructions in (narrative, templates["narrative_analysis"], a2_analysis):
        for phrase in _NOT_A_PERSON:
            assert phrase in instructions


def test_owned_content_plan_replaces_rejected_glossary_variants_and_pedagogical_pass_keeps_user_decisions_with_name_policy() -> None:
    """Content-plan points use the approved name, and a name policy still applies user decisions.

    The rejected suggestion belongs to this chapter's analysis: book candidates with the
    same analysis id are not repeated. The pedagogical candidate is the only line the
    model sees: the decision spelling «Пётуния Дурсли» is not a contiguous chapter phrase,
    so the name-policy filter drops it, while «Петуния Дурсли» still matches. Keeping
    user_decisions empty there leaves the rejected «Петунья».
    """
    from linguary.domain import GlossaryCandidate, TermConcept, TermVariant
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, SourceAnchor

    section = make_analysis_section(section_id="sec-petunia", texts=("Aunt Petunia looked up.",))
    document = make_analysis_document((section,))
    block = section.blocks[0]
    quote = "Aunt Petunia"
    plan = TreatmentPlan.of_treatments(section.id, ((block.id, BlockTreatment.TRANSLATE),))
    unit = TranslationUnit.of_blocks(id="unit-petunia", blocks=(block,), plan=plan)
    description = "Петунья отказала. У Петуньи нет ключа. Петунье сказали нет."
    candidate = GlossaryCandidate.create(
        id="candidate-aunt-petunia",
        analysis_id="analysis-petunia",
        source_text="Aunt Petunia",
        type=ConceptType.PERSON,
        suggested_translation="тётя Петунья",
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=0.5,
    )
    content_plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(
            Difficulty(
                "diff-petunia",
                (SourceAnchor(block.id, 0, len(quote), quote),),
                block.id,
                "vocabulary",
                "useful",
                "explanation",
                (RequiredPoint("p-petunia", description),),
            ),
        ),
    )
    analysis = ChapterAnalysis.create(
        id="analysis-petunia",
        section_id=section.id,
        treatment_plan=plan,
        translation_instructions="",
        chapter_summary="",
        expected_block_ids=(block.id,),
        content_plan=content_plan,
        glossary_candidates=(candidate,),
    )
    glossary = GlossarySnapshot(
        entries=(
            GlossarySnapshotEntry(
                source_text="Aunt Petunia",
                translation="тётя Петуния",
                type=ConceptType.PERSON,
                status=TermStatus.APPROVED,
            ),
        )
    )
    rendered = ContextBuilder(content_plan_profile="content-plan-v3").for_translation_unit(
        document=document,
        section=section,
        unit=unit,
        analysis=analysis,
        glossary=glossary,
        book_candidates=(candidate,),
    ).request.input_text

    assert "Петуния отказала. У Петунии нет ключа. Петунии сказали нет." in rendered
    assert "Петунья" not in rendered
    assert "Петуньи" not in rendered
    assert "Петунье" not in rendered

    decision = TermConcept(
        id="concept-petunia",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Пётуния Дурсли",
        status=TermStatus.APPROVED,
        note="утверждённый женский род",
        variants=(TermVariant("concept-petunia", "ru", "Петуния", TermStatus.APPROVED),),
    )
    chapter_candidate = GlossaryCandidate.create(
        id="candidate-petunia-dursley",
        analysis_id="analysis-petunia",
        source_text="Петуния Дурсли",
        type=ConceptType.PERSON,
        suggested_translation="Петунья",
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=0.5,
        note="черновик",
    )
    pedagogical_section = make_analysis_section(
        section_id="sec-dursley",
        texts=("Дурсли увидела Петуния.",),
    )
    pedagogical = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=_policy("sound"),
    ).for_chapter_pedagogical_analysis(
        make_analysis_document((pedagogical_section,)),
        pedagogical_section,
        narrative_summary="Short summary.",
        glossary_candidates=(chapter_candidate,),
        user_decisions=(decision,),
    )
    pedagogical_text = pedagogical.request.input_text

    assert "- Петуния Дурсли [person] -> Петуния (утверждённый женский род)" in pedagogical_text
    assert "Петунья" not in pedagogical_text
    assert "черновик" not in pedagogical_text
    assert "Пётуния Дурсли" not in pedagogical_text


def test_a2_prompts_enforce_gender_dialogue_dash_a2_summary_and_forbid_calques() -> None:
    """A2 prompts name gender, dialogue dashes, summary shape, and the reviewed calques.

    Dropping a named example, or putting the new calque list into the legacy templates,
    fails this test. The legacy SHA stays on the untouched constants.
    """
    from hashlib import sha256

    from linguary.application.book_conditions import payload_digest
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING,
        _CONTENT_PLAN_PROCESSING_V3,
        _NARRATIVE_ANALYSIS_INSTRUCTIONS,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _TRANSLATION_CONTEXT_V2,
        _TRANSLATION_INSTRUCTIONS,
        _pedagogical_instructions,
    )
    from linguary.application.reader_text import learner_instructions

    assert (
        sha256(_TRANSLATION_INSTRUCTIONS.encode("utf-8")).hexdigest()
        == "8c39d00a516d9db9a8a0079602c48bcfb6d9fc02714338d5d444be6304acbafa"
    )
    baseline_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        pronunciation_help_enabled=True,
    )
    assert (
        payload_digest(baseline_templates)
        == "3bcf968fbeb3eea72bc191df4360d50ec62f005d45a670b54dbd738ef2fb069a"
    )

    narrative_rules = (
        "грамматический род топонимов и животных",
        "действительно несклоняемых",
        "мужские фамилии на согласный в русском склоняются всегда",
        "не копируй в заметки читателю",
        "said Harry",
        "certainty=explicit",
        "анонимных обозначений",
    )
    pedagogical_rules = (
        _SPEECH_STYLE_KIND,
        "«Речь персонажа.»",
        "we was",
        "you was",
        "don' need",
        "them as",
        "b(K+1)..bN",
        "не больше N",
        "не больше 1–2 заметок на абзац",
        "reader_chapter_summary",
        "Previously in Chapter",
    )
    a2_summary_rules = (
        "reader_chapter_summary",
        "Previously in Chapter",
        "Present Simple",
        "стандартные английские кавычки",
        "«ёлочек»",
        "без перевода на русский",
    )
    calques = (
        "длинным тире «— »",
        "кавычками-ёлочками",
        "род адресата во 2-м лице",
        "местоимений 3-го лица",
        "регистр обращения «ты/вы»",
        "разговорность в русском переводе",
        "без буквальных калек",
    )
    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")
    pedagogical = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="A2",
    )
    a2_pedagogy = _pedagogical_instructions("en", "ru", "a2", False)
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    for instructions in (narrative, templates["narrative_analysis"]):
        for phrase in narrative_rules:
            assert phrase in instructions
    for instructions in (pedagogical, a2_pedagogy, templates["pedagogical_analysis"]):
        for phrase in pedagogical_rules:
            assert phrase in instructions
    for instructions in (a2_pedagogy, templates["pedagogical_analysis"]):
        for phrase in a2_summary_rules:
            assert phrase in instructions
    for instructions in (
        _CONTENT_PLAN_PROCESSING_V3,
        _TRANSLATION_CONTEXT_V2,
        learner_instructions("a2"),
        templates["content_processing"],
        templates["learner_quality"],
    ):
        for phrase in calques:
            assert phrase in instructions
    for legacy in (_TRANSLATION_INSTRUCTIONS, _CONTENT_PLAN_PROCESSING, learner_instructions("b1")):
        assert "апельсиновый джем" not in legacy
        assert "длинным тире «— »" not in legacy


def test_sound_policy_preserves_mc_prefix_in_mcgonagall() -> None:
    """SOUND keeps Mc-/Mac- as the prefix Мак- and doubled consonants in proper names (McGonagall → Макгонагалл)."""
    sound = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        profile="content-plan-v3",
        name_translation_policy=_policy("sound"),
    )
    meaning = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        profile="content-plan-v3",
        name_translation_policy=_policy("meaning"),
    )
    required = ("Mc-", "Mac-", "McGonagall", "Макгонагалл", "а не «Магонагал»")
    for key in ("analysis", "fragment", "translation", "regeneration", "narrative_analysis"):
        for phrase in required:
            assert phrase in sound[key]
        assert "Макгонагалл" not in meaning[key]


def test_short_chapter_caps_notes_budget_to_body_blocks() -> None:
    """A 2-paragraph excerpt chapter does not receive the full 8-note chapter budget."""
    from linguary.application.context_builder import _scene_notes_budget
    from linguary.domain import ChapterAnalysis, CoverageStatus, TranslationUnit, TreatmentPlan
    from linguary.domain.content_plan import ContentPlan, Scene

    short_section = make_analysis_section(
        section_id="sec-short",
        title="Chapter 6",
        blocks=(
            make_analysis_block(
                section_id="sec-short", order=0, text="— CHAPTER SIX —", type=BlockType.HEADING
            ),
            make_analysis_block(
                section_id="sec-short", order=1, text="The Journey", type=BlockType.HEADING
            ),
            make_analysis_block(
                section_id="sec-short", order=2, text="First paragraph. " * 30, type=BlockType.PARAGRAPH
            ),
            make_analysis_block(
                section_id="sec-short", order=3, text="Second paragraph. " * 30, type=BlockType.PARAGRAPH
            ),
        ),
    )
    doc = make_analysis_document((short_section,))
    builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        language_notes_per_chapter=8,
    )
    context = builder.for_chapter_analysis(document=doc, section=short_section)
    assert context.language_notes_per_chapter == 2
    assert max(context.note_limits_by_block.values()) <= 2

    plan = TreatmentPlan.of_treatments(
        short_section.id,
        ((b.id, BlockTreatment.TRANSLATE) for b in short_section.blocks),
    )
    unit = TranslationUnit.of_blocks(id="u1", blocks=short_section.blocks, plan=plan)
    analysis = ChapterAnalysis(
        id="a1",
        section_id=short_section.id,
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        content_plan=ContentPlan(version=3, profile="content-plan-v3"),
        scenes=(Scene("s1", 0, short_section.blocks[0].id, short_section.blocks[-1].id, "Scene"),),
    )
    assert _scene_notes_budget(analysis, short_section, unit, per_chapter=8) == 2

    from dataclasses import replace as dc_replace
    from linguary.application.analysis_merge import ChapterFragment
    from linguary.domain import GlossarySnapshot
    from linguary.domain.content_plan import Difficulty, RequiredPoint, SourceAnchor

    three_difficulties = tuple(
        Difficulty(
            f"d{idx}",
            (SourceAnchor(block.id, 0, 3, block.text[:3]),),
            block.id,
            "grammar",
            "useful",
            "explanation",
            (RequiredPoint(f"p{idx}", "Explain"),),
        )
        for idx, block in enumerate(short_section.blocks[1:], start=1)
    )
    crowded_analysis = dc_replace(
        analysis,
        content_plan=ContentPlan(
            version=3, profile="content-plan-v3", difficulties=three_difficulties
        ),
    )
    assert _scene_notes_budget(crowded_analysis, short_section, unit, per_chapter=8) == 2

    no_scenes_analysis = dc_replace(analysis, scenes=())
    tr_ctx = builder.for_translation_unit(
        document=doc,
        section=short_section,
        unit=unit,
        analysis=no_scenes_analysis,
        glossary=GlossarySnapshot(),
    )
    assert "не больше 2 языковых заметок" in tr_ctx.request.input_text

    ped_ctx = builder.for_chapter_pedagogical_analysis(
        document=doc,
        section=short_section,
        narrative_summary="Short summary.",
        glossary_candidates=(),
        context_facts=(),
    )
    assert ped_ctx.language_notes_per_chapter == 2

    frag0 = builder.for_chapter_fragment(
        document=doc,
        section=short_section,
        fragment=ChapterFragment(order=0, blocks=short_section.blocks[:2]),
        fragment_count=2,
    )
    frag1 = builder.for_chapter_fragment(
        document=doc,
        section=short_section,
        fragment=ChapterFragment(order=1, blocks=short_section.blocks[2:]),
        fragment_count=2,
    )
    assert frag0.language_notes_per_chapter + frag1.language_notes_per_chapter == 2

    # Heading-only section falls back to total textual block count.
    from linguary.application.context_builder import chapter_note_limit

    headings_only = make_analysis_section(
        section_id="sec-head",
        title="Part I",
        blocks=(
            make_analysis_block(
                section_id="sec-head", order=0, text="PART ONE", type=BlockType.HEADING
            ),
        ),
    )
    assert chapter_note_limit(headings_only, 8) == 1

    # Legacy READ_AS_IS owners exceeding chapter_note_limit are still honored without crashing.
    read_as_is_plan = TreatmentPlan.of_treatments(
        short_section.id,
        ((b.id, BlockTreatment.READ_AS_IS) for b in short_section.blocks),
    )
    read_as_is_unit = TranslationUnit.of_blocks(
        id="u-read", blocks=short_section.blocks, plan=read_as_is_plan
    )
    legacy_read_analysis = dc_replace(
        crowded_analysis,
        treatment_plan=read_as_is_plan,
    )
    assert _scene_notes_budget(legacy_read_analysis, short_section, read_as_is_unit, per_chapter=8) == 3


def test_a2_word_share_progression_in_input_text_and_sha() -> None:
    """A2 progression corridor and prior topics enter input_text (even with frozen templates) and SHA."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING_V3,
        _NARRATIVE_ANALYSIS_INSTRUCTIONS,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _TRANSLATION_CONTEXT_V2,
        _reader_level_treatment_guidance,
    )

    section = make_analysis_section(
        section_id="sec-prog",
        title="Chapter 4",
        texts=("Harry looked at the strange hut in the storm.",),
    )
    document = make_analysis_document((section,))
    frozen_a2_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    a2_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        prompt_templates=frozen_a2_templates,
    )

    expected_bands = (
        (0, "70–82%"),
        (12_000, "72–80%"),
        (30_000, "68–76%"),
        (50_000, "62–72%"),
    )
    all_bands = tuple(band for _, band in expected_bands)
    shas: list[str] = []
    for words_before, band in expected_bands:
        ctx = a2_builder.for_chapter_pedagogical_analysis(
            document,
            section,
            narrative_summary="Harry arrives at the hut.",
            cumulative_words_before=words_before,
        )
        assert f"целевая доля слов в режиме TRANSLATE — {band}" in ctx.request.input_text
        for other_band in all_bands:
            if other_band != band:
                assert other_band not in ctx.request.input_text
        assert "Предохранитель" in ctx.request.input_text
        shas.append(
            a2_builder.pedagogical_input_sha256(
                document,
                section,
                narrative_output_sha256="narr-sha-1",
                narrative_summary="Harry arrives at the hut.",
                cumulative_words_before=words_before,
            )
        )
    assert len(set(shas)) == len(expected_bands)
    assert shas[0] == a2_builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha-1",
        narrative_summary="Harry arrives at the hut.",
    )

    ctx_with_topics = a2_builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Harry arrives at the hut.",
        cumulative_words_before=12_000,
        prior_explained_topics=("used to", "to squint", "used to", " "),
    )
    assert "used to" in ctx_with_topics.request.input_text
    assert "to squint" in ctx_with_topics.request.input_text
    assert ctx_with_topics.request.input_text.count("- used to") == 1

    b2_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="b2",
    )
    b2_ctx = b2_builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Harry arrives at the hut.",
        cumulative_words_before=30_000,
    )
    assert "70–82%" not in b2_ctx.request.input_text
    assert "68–76%" not in b2_ctx.request.input_text

    # Prompt updates for false_friend, strict A2 reader_chapter_summary, contextual notes, and ты/вы pairs.
    from linguary.application.context_builder import _pedagogical_instructions

    ped = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS
    a2_ped = _pedagogical_instructions("en", "ru", "a2", False)
    assert _FALSE_FRIEND_KIND in ped
    assert _SPEECH_STYLE_KIND in ped
    for example in ("actual", "magazine", "sympathetic"):
        assert example in ped
    assert "3–5" in a2_ped
    assert "общеязыковых слов" not in ped
    assert "общеязыковых слов" not in a2_ped

    a2_guidance = _reader_level_treatment_guidance("a2")
    assert "GLOSSARY_ONLY" in a2_guidance
    assert "15 слов" in a2_guidance
    assert "45 слов" in a2_guidance

    for text in (_NARRATIVE_ANALYSIS_INSTRUCTIONS, _CONTENT_PLAN_PROCESSING_V3, _TRANSLATION_CONTEXT_V2):
        for word in ("robes", "drills", "witch", "hut", "term"):
            assert word in text
        assert "ты/вы" in text or "«ты» или «вы»" in text


def test_name_policy_sound_keeps_doubled_consonants_in_proper_names() -> None:
    """SOUND follows Ermolovich rule: keep doubled consonants in proper names, simplify common nouns."""
    sound = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
        name_translation_policy=_policy("sound"),
    )
    for key in ("analysis", "fragment", "translation", "regeneration", "narrative_analysis", "pedagogical_analysis"):
        text = sound[key]
        assert "Ермоловича" in text
        assert "«McGonagall» → «Макгонагалл», а не «Магонагал»" in text
        for proper in ("Гринготтс", "Квиррелл", "Невилл", "Макгонагалл"):
            assert proper in text
        assert "«маглы», а не «магглы»" in text


def test_matches_chapter_text_links_initials_to_full_name() -> None:
    """Initial forms like Mr H. Potter match full names like Harry Potter in both directions without false positives."""
    from linguary.application.context_builder import _extract_word_stems, _matches_chapter_text

    initial_text = "The letter was addressed to Mr H. Potter, The Cupboard under the Stairs."
    initial_stems = _extract_word_stems(initial_text)
    assert _matches_chapter_text("Harry Potter", initial_text, initial_stems)
    assert not _matches_chapter_text("James Potter", initial_text, initial_stems)
    assert not _matches_chapter_text("Lily Potter", initial_text, initial_stems)

    bare_initial_text = "H. Potter's parcel and Mrs M. Weasley's note arrived together."
    bare_stems = _extract_word_stems(bare_initial_text)
    assert _matches_chapter_text("Harry Potter", bare_initial_text, bare_stems)
    assert _matches_chapter_text("Molly Weasley", bare_initial_text, bare_stems)
    assert not _matches_chapter_text("Arthur Weasley", bare_initial_text, bare_stems)

    full_name_text = "Harry Potter opened the door while Molly Weasley smiled."
    full_stems = _extract_word_stems(full_name_text)
    assert _matches_chapter_text("Mr H. Potter", full_name_text, full_stems)
    assert _matches_chapter_text("H. Potter", full_name_text, full_stems)
    assert _matches_chapter_text("Mrs M. Weasley", full_name_text, full_stems)
    assert not _matches_chapter_text("Mr J. Potter", full_name_text, full_stems)

    # Guard against common-word false positives.
    false_pos_text = "He handed Potter a potter book. A Potter always remembers."
    false_pos_stems = _extract_word_stems(false_pos_text)
    assert not _matches_chapter_text("Harry Potter", false_pos_text, false_pos_stems)
    assert not _matches_chapter_text("Albus Potter", false_pos_text, false_pos_stems)
    assert not _matches_chapter_text("Mr H. Potter", false_pos_text, false_pos_stems)

    # Guard against lowercase abbreviations (e.g. "a.m.") before a surname.
    abbrev_text = "At 7 a.m. Potter arrived."
    abbrev_stems = _extract_word_stems(abbrev_text)
    assert not _matches_chapter_text("Albus Potter", abbrev_text, abbrev_stems)


def test_chapter_note_limit_and_scene_budget_scale_for_long_chapters() -> None:
    """chapter_note_limit and _scene_notes_budget scale up to 16 for long chapters and respect low caps."""
    from linguary.application.context_builder import (
        _scene_notes_budget,
        chapter_note_limit,
        scaled_chapter_note_limit,
    )
    from linguary.domain import ChapterAnalysis, CoverageStatus, TranslationUnit, TreatmentPlan
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, Scene, SourceAnchor

    assert scaled_chapter_note_limit(0, 8) == 0
    assert scaled_chapter_note_limit(300, 0) == 0
    assert scaled_chapter_note_limit(300, -1) == 0
    assert scaled_chapter_note_limit(300, 4) == 4
    assert scaled_chapter_note_limit(2, 4) == 2
    assert scaled_chapter_note_limit(2, 8) == 2
    assert scaled_chapter_note_limit(35, 8) == 8
    assert scaled_chapter_note_limit(60, 8) == 9
    assert scaled_chapter_note_limit(100, 8) == 12
    assert scaled_chapter_note_limit(168, 8) == 18
    assert scaled_chapter_note_limit(250, 8) == 18
    assert scaled_chapter_note_limit(300, 8) == 18
    assert scaled_chapter_note_limit(400, 8) == 18
    assert scaled_chapter_note_limit(600, 8) == 18
    assert scaled_chapter_note_limit(300, 20) == 20

    long_blocks = tuple(
        make_analysis_block(
            section_id="sec-long",
            order=i,
            text=f"Paragraph number {i} with enough words to fill a chapter.",
            type=BlockType.PARAGRAPH,
        )
        for i in range(300)
    )
    long_section = make_analysis_section(section_id="sec-long", title="Chapter 5", blocks=long_blocks)
    assert chapter_note_limit(long_section, 8) == 18
    assert chapter_note_limit(long_section, 4) == 4

    plan = TreatmentPlan.of_treatments(
        long_section.id,
        ((b.id, BlockTreatment.READ_AS_IS) for b in long_section.blocks),
    )
    unit = TranslationUnit.of_blocks(id="u-long", blocks=long_section.blocks, plan=plan)

    ten_difficulties = tuple(
        Difficulty(
            f"d{idx}",
            (SourceAnchor(block.id, 0, 9, block.text[:9]),),
            block.id,
            "grammar",
            "useful",
            "explanation",
            (RequiredPoint(f"p{idx}", "Explain grammar"),),
        )
        for idx, block in enumerate(long_section.blocks[:10], start=1)
    )
    analysis_ten = ChapterAnalysis(
        id="a-long-10",
        section_id=long_section.id,
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        content_plan=ContentPlan(version=3, profile="content-plan-v3", difficulties=ten_difficulties),
        scenes=(Scene("s1", 0, long_section.blocks[0].id, long_section.blocks[-1].id, "Long scene"),),
    )
    assert _scene_notes_budget(analysis_ten, long_section, unit, per_chapter=8) == 18

    nineteen_difficulties = tuple(
        Difficulty(
            f"d{idx}",
            (SourceAnchor(block.id, 0, 9, block.text[:9]),),
            block.id,
            "grammar",
            "useful",
            "explanation",
            (RequiredPoint(f"p{idx}", "Explain grammar"),),
        )
        for idx, block in enumerate(long_section.blocks[:19], start=1)
    )
    analysis_nineteen = ChapterAnalysis(
        id="a-long-19",
        section_id=long_section.id,
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=plan,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        content_plan=ContentPlan(version=3, profile="content-plan-v3", difficulties=nineteen_difficulties),
        scenes=(Scene("s1", 0, long_section.blocks[0].id, long_section.blocks[-1].id, "Long scene"),),
    )
    with pytest.raises(ContextBuildError):
        _scene_notes_budget(analysis_nineteen, long_section, unit, per_chapter=8)


def test_narrative_instructions_require_headings_in_first_scene_and_clean_glossary() -> None:
    """Narrative instructions require headings in scene 1, clean glossary fields, and verbatim subject_anchor."""
    from linguary.application.context_builder import _NARRATIVE_ANALYSIS_INSTRUCTIONS

    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    for text in (narrative, templates["narrative_analysis"]):
        # 1. Headings in the first scene.
        assert "b1" in text and "b2" in text
        assert "первую сцену" in text
        assert "заголовки" in text
        # 2. Ban brands / household items / common nouns from person.
        assert "торговые марки" in text
        assert "бытовые предметы" in text
        assert "нарицательные" in text
        # 3. Ban parenthetical alternatives and extra quotes in glossary fields.
        assert "source_text" in text and "suggested_translation" in text
        assert "в скобках" in text
        assert "слэш" in text
        assert "лишних кавычек" in text
        # 4. Allow fictional books and their authors from epigraphs/quotes.
        assert "вымышленные книги" in text.lower()
        assert "эпиграфов" in text
        # 5. Ban duplicating name translations in translation_instructions.
        assert "translation_instructions" in text
        assert "Не дублируй" in text or "не дублируй" in text
        assert "переводы имён" in text
        # 6. Explicit null offsets for unique quote.
        assert "start=null и end=null" in text
        # 7. Verbatim subject_anchor explanation.
        assert "subject_anchor" in text
        assert "дословн" in text


def test_name_policy_meaning_includes_contrastive_opaque_name_example() -> None:
    """Meaning name-translation policy for analysis includes contrastive opaque_name examples."""
    meaning = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
        name_translation_policy=_policy("meaning"),
    )
    for key in ("analysis", "fragment", "narrative_analysis"):
        text = meaning[key]
        assert "opaque_name=true" in text
        assert "opaque_name=false" in text
        for name in ("Nick", "Alexandra", "Everlost", "Hightower"):
            assert name in text
    assert "Everlost" not in meaning["translation"]
    assert "Everlost" not in meaning["pedagogical_analysis"]


def test_pedagogical_a2_instructions_have_no_duplicates_or_hp_examples() -> None:
    """Pedagogical A2 instructions have no HP examples, no duplicate A2 note discipline, and no translation-only rules."""
    import inspect

    from linguary.application.analysis_merge import DEFAULT_MAX_FRAGMENT_BLOCKS
    from linguary.application.context_builder import _reader_level_treatment_guidance
    from linguary.application.reader_text import _A2_NOTE_DISCIPLINE, learner_instructions

    hp_examples = (
        "The Boy Who Lived",
        "The Vanishing Glass",
        "The Letters from No One",
        "sweetums",
        "Fancy seeing you here",
        "not saying the spell isn’t brilliant",
    )
    a2_guidance = _reader_level_treatment_guidance("a2")
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    section = make_analysis_section(
        section_id="sec-a2-ped",
        title="Chapter 1",
        order=0,
        texts=("First paragraph.",),
    )
    document = make_analysis_document((section,))
    live_ctx = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    ).for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Short summary.",
    )
    template_ctx = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        prompt_templates=templates,
    ).for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Short summary.",
    )

    for text in (
        a2_guidance,
        templates["pedagogical_analysis"],
        live_ctx.request.instructions,
        template_ctx.request.instructions,
    ):
        for phrase in hp_examples:
            assert phrase not in text

    note_discipline_body = _A2_NOTE_DISCIPLINE.strip()
    for instructions in (live_ctx.request.instructions, template_ctx.request.instructions):
        assert instructions.count(note_discipline_body) == 1
        assert instructions.count("Различай роль заметок в GLOSSARY_ONLY и TRANSLATE") == 1
        assert instructions.count("Для A2 без english help верни before_you_read: []") == 1
        assert instructions.count("might have been") == 1
        assert instructions.count("not saying ... isn’t ...") == 1
        assert "Для A2 обеспечь необходимую лексическую опору" not in instructions
        assert "Пересказ A2 верни ровно одним абзацем" not in instructions
        assert "Соблюдай 4 правила оформления русской прямой речи" not in instructions
        assert "длинным тире «— »" not in instructions
        assert "род адресата во 2-м лице" not in instructions
        assert "разговорность в русском переводе" not in instructions

    # learner_instructions("a2") itself stays intact for other callers and regression checks.
    assert note_discipline_body in learner_instructions("a2")
    assert "Соблюдай 4 правила оформления русской прямой речи" in learner_instructions("a2")

    assert (
        inspect.signature(ContextBuilder.narrative_input_sha256)
        .parameters["max_fragment_blocks"]
        .default
        == DEFAULT_MAX_FRAGMENT_BLOCKS
    )
    assert (
        inspect.signature(ContextBuilder.pedagogical_input_sha256)
        .parameters["max_fragment_blocks"]
        .default
        == DEFAULT_MAX_FRAGMENT_BLOCKS
    )


def test_pedagogical_a2_summary_instruction_targets_full_chapter() -> None:
    """A2 pedagogical instructions adapt the full-chapter narrative summary, even for fragments."""
    from linguary.application.context_builder import _pedagogical_instructions

    section = make_analysis_section(
        section_id="sec-a2-sum",
        title="Chapter 1",
        order=0,
        texts=("Paragraph one.", "Paragraph two."),
    )
    document = make_analysis_document((section,))
    fragment = plan_fragments(textual_blocks(section), max_blocks=1)[0]
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")
    whole_ctx = builder.for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Full chapter narrative summary.",
    )
    frag_ctx = builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Full chapter narrative summary.",
        fragment=fragment,
        fragment_count=2,
    )
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    for instructions in (
        _pedagogical_instructions("en", "ru", "a2", False),
        templates["pedagogical_analysis"],
        whole_ctx.request.instructions,
        frag_ctx.request.instructions,
    ):
        assert "Принятый повествовательный анализ" in instructions
        assert "chapter_summary" in instructions
        assert "всей главы" in instructions
        assert "3–5 коротких" in instructions
        assert "до текущего фрагмента" not in instructions
        assert "а не дописывается абзацем к каждому фрагменту" not in instructions
        assert "Ограничь chapter_summary и reader_chapter_summary" not in instructions


def test_baseline_b2_templates_sha256_unchanged() -> None:
    """Frozen legacy-v1/B2 baseline templates and _TRANSLATION_INSTRUCTIONS retain exact SHA-256 digests."""
    from hashlib import sha256

    from linguary.application.book_conditions import payload_digest
    from linguary.application.context_builder import _TRANSLATION_INSTRUCTIONS

    assert (
        sha256(_TRANSLATION_INSTRUCTIONS.encode("utf-8")).hexdigest()
        == "8c39d00a516d9db9a8a0079602c48bcfb6d9fc02714338d5d444be6304acbafa"
    )
    baseline_with_pron = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
        pronunciation_help_enabled=True,
    )
    assert (
        payload_digest(baseline_with_pron)
        == "3bcf968fbeb3eea72bc191df4360d50ec62f005d45a670b54dbd738ef2fb069a"
    )
    baseline_default = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="b2",
    )
    assert (
        payload_digest(baseline_default)
        == "d6e4b0b6dfa1227f6ab86b34ce7d9cb05b465c4e2ba71a5f322dec1f9125d470"
    )


def test_narrative_and_pedagogical_prompt_rules_for_labels_languages_and_segments() -> None:
    """Narrative prompts forbid block labels & enforce languages/gender; pedagogical forbids bN.M in difficulties."""
    from linguary.application.context_builder import (
        _NARRATIVE_ANALYSIS_INSTRUCTIONS,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
    )

    narrative_ru = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")
    narrative_es = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="de", target_language="es")
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    section = make_analysis_section(
        section_id="sec-rules",
        title="Chapter 1",
        order=0,
        texts=("First paragraph.",),
    )
    document = make_analysis_document((section,))
    live_nar = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    ).for_chapter_narrative_analysis(document, section).request.instructions
    live_ped = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    ).for_chapter_pedagogical_analysis(
        document, section, narrative_summary="First paragraph."
    ).request.instructions

    # Fix 1: forbid technical block labels in translation_instructions.
    for text in (narrative_ru, templates["narrative_analysis"], live_nar):
        assert "Не ссылайся в translation_instructions на служебные метки блоков" in text
        assert "b1, b8, b51" in text
        assert "событие, сцену или цитату" in text

    # Fix 3: explicit languages for chapter_summary, scenes & translation_instructions, and English male/female.
    for text in (narrative_ru, templates["narrative_analysis"], live_nar):
        assert "на языке оригинала en (3–5 предложений)" in text
        assert (
            "Поля scenes (summary, location, visual_facts.detail, visual_assumptions) "
            "и translation_instructions пиши на языке перевода (ru)."
        ) in text
        assert "Для kind=gender используй единообразные английские значения male / female." in text

    assert "на языке оригинала de (3–5 предложений)" in narrative_es
    assert (
        "Поля scenes (summary, location, visual_facts.detail, visual_assumptions) "
        "и translation_instructions пиши на языке перевода (es)."
    ) in narrative_es

    # Fix 2: forbid segment labels bN.M in difficulties; allow only in treatment_plan.segments.
    ped_raw = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="A2"
    )
    for text in (ped_raw, templates["pedagogical_analysis"], live_ped):
        assert "difficulties" in text
        assert "owner_block_id" in text
        assert "anchors[].block_id" in text
        assert "целого блока bN" in text
        assert "b4, b120" in text
        assert "bN.M" in text
        assert "treatment_plan.segments" in text


def test_name_policy_backfills_unmatched_prior_chapter_key_candidates_preserving_suggested_metadata() -> None:
    """Narrative/single-pass analysis with name policy backfills unmatched key prior candidates and keeps metadata."""
    from linguary.application.context_builder import _name_context
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain import GlossaryCandidate, TermConcept, TermVariant
    from linguary.domain.name_translation import (
        NameTranslationAssessment,
        NameTranslationEvidence,
        NameTranslationMethod,
        NameTranslationObservation,
    )

    def _evidence(*, translation: str, opaque: bool, method: NameTranslationMethod) -> NameTranslationEvidence:
        return NameTranslationEvidence(
            version=1,
            conditions_id="cond-1",
            policy_version=1,
            strategy=NameTranslationStrategy.MEANING,
            observations=(
                NameTranslationObservation(
                    suggested_translation=translation,
                    confidence=0.9,
                    model_flagged_review=False,
                    assessment=NameTranslationAssessment(
                        applied_method=method,
                        eligible_without_review=True,
                        review_reasons=(),
                        rationale="Оценка из прошлой главы.",
                        opaque_name=opaque,
                    ),
                ),
            ),
        )

    section = make_analysis_section(
        section_id="sec-ch2",
        title="Chapter 2",
        order=1,
        texts=("Allie stepped carefully along the gray path while Nick watched.",),
    )
    document = make_analysis_document((section,))

    approved_nick = TermConcept(
        id="c-nick",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Nick",
        status=TermStatus.APPROVED,
        note="мужской род",
        variants=(TermVariant("c-nick", "ru", "Ник", TermStatus.APPROVED),),
    )
    approved_unmatched = TermConcept(
        id="c-mary",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Mary Hightower",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-mary", "ru", "Мэри Хайтауэр", TermStatus.APPROVED),),
    )
    suggested_lief = TermConcept(
        id="c-lief",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Lief",
        status=TermStatus.SUGGESTED,
        note="мужской род",
        variants=(TermVariant("c-lief", "ru", "Лист", TermStatus.SUGGESTED),),
    )
    suggested_allie = TermConcept(
        id="c-allie",
        book_id="book-1",
        type=ConceptType.PERSON,
        canonical_source_text="Allie",
        status=TermStatus.SUGGESTED,
        variants=(TermVariant("c-allie", "ru", "Элли", TermStatus.SUGGESTED),),
    )

    cand_allie = GlossaryCandidate(
        id="cand-allie",
        analysis_id="an-1",
        source_text="Allie",
        type=ConceptType.PERSON,
        suggested_translation="Элли",
        confidence=0.95,
        model_flagged_review=False,
        requires_review=False,
        description="Сокращённое имя Александры.",
        name_translation_evidence=_evidence(
            translation="Элли", opaque=True, method=NameTranslationMethod.SOUND
        ),
    )
    cand_alexandra = GlossaryCandidate(
        id="cand-alexandra",
        analysis_id="an-1",
        source_text="Alexandra",
        type=ConceptType.PERSON,
        suggested_translation="Александра",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
        description="Полное имя Элли.",
        name_translation_evidence=_evidence(
            translation="Александра", opaque=True, method=NameTranslationMethod.SOUND
        ),
    )
    cand_alexandra_dup = GlossaryCandidate(
        id="cand-alexandra-dup",
        analysis_id="an-2",
        source_text="ALEXANDRA",
        type=ConceptType.PERSON,
        suggested_translation="Александра",
        confidence=0.85,
        model_flagged_review=False,
        requires_review=False,
    )
    cand_lief = GlossaryCandidate(
        id="cand-lief",
        analysis_id="an-1",
        source_text="Lief",
        type=ConceptType.PERSON,
        suggested_translation="Лиф",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=True,
        description="Мальчик, чьё имя происходит от слова leaf.",
        name_translation_evidence=_evidence(
            translation="Лист", opaque=False, method=NameTranslationMethod.MEANING
        ),
    )
    cand_afterlights = GlossaryCandidate(
        id="cand-afterlights",
        analysis_id="an-0",
        source_text="Afterlights",
        type=ConceptType.FICTIONAL_CONCEPT,
        suggested_translation="Послесветы",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
        name_translation_evidence=_evidence(
            translation="Послесветы", opaque=False, method=NameTranslationMethod.MEANING
        ),
    )
    cand_place = GlossaryCandidate(
        id="cand-place",
        analysis_id="an-1",
        source_text="Everlost",
        type=ConceptType.PLACE,
        suggested_translation="Междумирье",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
    )
    cand_org = GlossaryCandidate(
        id="cand-org",
        analysis_id="an-1",
        source_text="Twin Towers Guild",
        type=ConceptType.ORGANIZATION,
        suggested_translation="Гильдия Башен",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
    )
    cand_flagged_term = GlossaryCandidate(
        id="cand-flagged",
        analysis_id="an-1",
        source_text="deadspot",
        type=ConceptType.TERM,
        suggested_translation="мёртвая точка",
        confidence=0.8,
        model_flagged_review=True,
        requires_review=True,
    )
    cand_plain_term = GlossaryCandidate(
        id="cand-plain",
        analysis_id="an-1",
        source_text="chocolate coin",
        type=ConceptType.TERM,
        suggested_translation="шоколадная монета",
        confidence=0.95,
        model_flagged_review=False,
        requires_review=False,
    )
    cand_mary = GlossaryCandidate(
        id="cand-mary",
        analysis_id="an-1",
        source_text="Mary Hightower",
        type=ConceptType.PERSON,
        suggested_translation="Мэри Высокобашенная",
        confidence=0.9,
        model_flagged_review=False,
        requires_review=False,
    )
    accumulated = (
        cand_allie,
        cand_alexandra,
        cand_alexandra_dup,
        cand_lief,
        cand_afterlights,
        cand_place,
        cand_org,
        cand_flagged_term,
        cand_plain_term,
        cand_mary,
    )

    builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=_policy("meaning"),
        max_accumulated_candidates=10,
    )
    narrative_ctx = builder.for_chapter_narrative_analysis(
        document,
        section,
        accumulated_candidates=accumulated,
        user_decisions=(approved_nick, approved_unmatched),
        nonbinding_suggestions=(suggested_allie, suggested_lief),
    )
    text = narrative_ctx.request.input_text

    # Accepted line keeps exact format without extra metadata.
    assert "- Nick [person] -> Ник (мужской род)" in text
    assert "- Nick [person] -> Ник (мужской род) —" not in text
    # Approved unmatched user decision is not backfilled as unconfirmed suggestion.
    assert "Mary Hightower" not in text
    # Matched Allie keeps suggested translation and enriches description from GlossaryCandidate.
    assert "- Allie [person] -> Элли — описание: Сокращённое имя Александры." in text
    assert "Allie [person] -> Элли" in text and "непрозрачное=false" not in text.split("Allie [person] -> Элли")[1].splitlines()[0]
    # Overlapping TermConcept(SUGGESTED) + GlossaryCandidate for Lief keeps description and непрозрачное=false.
    lief_line = next(line for line in text.splitlines() if line.startswith("- Lief [person] -> Лист"))
    assert "(мужской род)" in lief_line
    assert "Мальчик, чьё имя происходит от слова leaf." in lief_line
    assert "непрозрачное=false" in lief_line
    # Key unmatched types + requires_review term are backfilled; plain TERM without review is excluded.
    assert "Alexandra [person] -> Александра" in text
    assert "Полное имя Элли." in text
    assert text.count("Alexandra") == 1
    assert "ALEXANDRA" not in text
    assert "- Afterlights [fictional_concept] -> Послесветы — непрозрачное=false" in text
    # Lines without metadata keep exact original format.
    assert "\n- Everlost [place] -> Междумирье\n" in text
    assert "\n- Twin Towers Guild [organization] -> Гильдия Башен\n" in text
    assert "\n- deadspot [term] -> мёртвая точка\n" in text
    assert "chocolate coin" not in text

    # Cap limits total optional slots after matched candidates.
    capped_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=_policy("meaning"),
        max_accumulated_candidates=2,
    )
    capped_text = capped_builder.for_chapter_narrative_analysis(
        document,
        section,
        accumulated_candidates=accumulated,
        user_decisions=(approved_nick,),
        nonbinding_suggestions=(suggested_allie, suggested_lief),
    ).request.input_text
    assert "- Allie [person] -> Элли" in capped_text
    assert "- Lief [person] -> Лист" in capped_text
    assert "непрозрачное=false" in capped_text
    assert "Afterlights" not in capped_text
    assert "Everlost" not in capped_text

    # Single-pass for_chapter_analysis also backfills unmatched key candidates.
    single_pass_text = builder.for_chapter_analysis(
        document=document,
        section=section,
        accumulated_candidates=accumulated,
        user_decisions=(approved_nick,),
        nonbinding_suggestions=(suggested_allie, suggested_lief),
    ).request.input_text
    assert "Afterlights" in single_pass_text
    assert "Alexandra" in single_pass_text
    assert "chocolate coin" not in single_pass_text

    # Pedagogical pass must NOT backfill unmatched prior-chapter candidates.
    ped_text = builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Allie and Nick walk on the path.",
        accumulated_candidates=accumulated,
        user_decisions=(approved_nick,),
    ).request.input_text
    assert "Allie" in ped_text
    assert "Afterlights" not in ped_text
    assert "Alexandra" not in ped_text
    assert "Twin Towers Guild" not in ped_text
    assert "deadspot" not in ped_text

    # Without name_translation_policy, unmatched candidates remain filtered out.
    legacy_text = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=None,
    ).for_chapter_analysis(
        document=document,
        section=section,
        accumulated_candidates=accumulated,
    ).request.input_text
    assert "Allie" in legacy_text
    assert "Afterlights" not in legacy_text
    assert "Alexandra" not in legacy_text

    # DTO GlossaryCandidateResponse also exposes opaque_name=False via name_translation_assessment.
    dto_cand = GlossaryCandidateResponse(
        source_text="Skinjacker",
        type=ConceptType.FICTIONAL_CONCEPT,
        suggested_translation="Перехватчик",
        confidence=0.9,
        model_flagged_review=False,
        description="Тот, кто вселяется в живых.",
        name_translation_assessment=NameTranslationAssessment(
            applied_method=NameTranslationMethod.MEANING,
            eligible_without_review=True,
            review_reasons=(),
            rationale="Прозрачное составное слово.",
            opaque_name=False,
        ),
    )
    _, optional_block = _name_context((), (dto_cand,), "ru")
    assert "Skinjacker [fictional_concept] -> Перехватчик" in optional_block
    assert "Тот, кто вселяется в живых." in optional_block
    assert "непрозрачное=false" in optional_block


def test_narrative_instructions_require_declension_visual_only_facts_and_part_divider_rule() -> None:
    """Narrative instructions cover prior hints, gender/declension, visual-only facts, sound review_reasons, anti-spoiler/participants, and part dividers."""
    from linguary.application.context_builder import _NARRATIVE_ANALYSIS_INSTRUCTIONS

    divider_section = make_analysis_section(
        section_id="sec-part-1",
        title="PART ONE",
        order=0,
        blocks=(
            make_analysis_block(section_id="sec-part-1", order=0, text="PART ONE", type=BlockType.HEADING),
            make_analysis_block(section_id="sec-part-1", order=1, text="Afterlights", type=BlockType.HEADING),
        ),
    )
    bare_divider = make_analysis_section(
        section_id="sec-part-2",
        title="PART TWO",
        order=1,
        blocks=(
            make_analysis_block(section_id="sec-part-2", order=0, text="PART TWO", type=BlockType.HEADING),
        ),
    )
    chapter_section = make_analysis_section(
        section_id="sec-ch1",
        title="Chapter 1",
        order=2,
        texts=("A stranger stepped out of the mist.",),
    )
    doc = make_analysis_document((divider_section, bare_divider, chapter_section))

    meaning_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=_policy("meaning"),
    )
    sound_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        name_translation_policy=_policy("sound"),
    )

    divider_instr = meaning_builder.for_chapter_narrative_analysis(doc, divider_section).request.instructions
    bare_instr = meaning_builder.for_chapter_narrative_analysis(doc, bare_divider).request.instructions
    chapter_meaning_instr = meaning_builder.for_chapter_narrative_analysis(doc, chapter_section).request.instructions
    chapter_sound_instr = sound_builder.for_chapter_narrative_analysis(doc, chapter_section).request.instructions
    raw_narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")

    # Meaningful part divider gets dedicated rule: scenes=[], glossary_candidates, neutral source-language summary without plot invention.
    assert "scenes=[]" in divider_instr
    assert "glossary_candidates" in divider_instr
    assert "chapter_summary" in divider_instr
    assert "нейтральный заголовок" in divider_instr
    assert "(en)" in divider_instr
    assert "без выдумывания сюжета" in divider_instr
    assert "нейтральный заголовок" not in bare_instr
    assert "Это служебный раздел: верни scenes=[]." in bare_instr
    assert "Это служебный раздел: верни scenes=[]." not in chapter_meaning_instr

    # Core narrative rules in raw template and both policies.
    for text in (raw_narrative, chapter_meaning_instr, chapter_sound_instr):
        # Prior-chapter reference hints.
        assert "справочные подсказки прошлых глав" in text
        # Mandatory gender and declension rules (given names and surnames) in note.
        assert "род персонажей" in text
        assert "склоняемости" in text
        assert "женские фамилии на согласный" in text
        assert "мужские фамилии на согласный" in text
        # Visual-only facts in visual_facts.
        assert "только зрительные детали" in text
        assert "звуки" in text and "запахи" in text
        # No name spoilers & single language for participants.
        assert "едином языке" in text
        assert "participants" in text
        assert "Не спойлерь имя персонажа" in text

    # Name policy analysis instructions reinforce prior hints and review_reasons purity for sound.
    for text in (chapter_meaning_instr, chapter_sound_instr):
        assert "непрозрачное=false" in text
        assert "review_reasons пустым ([])" in text
        assert "creative_adaptation" in text
        assert "personal_name_substitution" in text


def test_pedagogical_a2_instructions_include_segments_example_stoplist_and_tense_ban() -> None:
    """A2 pedagogical and summary prompts include segment JSON example, stoplist, priority/channel/tense rules, and A2 recap template without duplicates."""
    from linguary.application.context_builder import (
        _FRAGMENT_SUMMARY_A2,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _WHOLE_CHAPTER_SUMMARY_A2,
        _pedagogical_instructions,
        _reader_level_treatment_guidance,
    )
    from linguary.application.reader_text import _A2_NOTE_DISCIPLINE, learner_instructions

    section = make_analysis_section(
        section_id="sec-a2-p4t2",
        title="Chapter 1",
        order=0,
        texts=("First paragraph.",),
    )
    document = make_analysis_document((section,))
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    live_ctx = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
    ).for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Short summary.",
    )
    template_ctx = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        prompt_templates=templates,
    ).for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Short summary.",
    )

    ped_raw = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="A2",
    )
    ped_a2 = _pedagogical_instructions("en", "ru", "a2", False)
    a2_guidance = _reader_level_treatment_guidance("a2")

    # 1. Segment JSON example and >=3 segments / >=2 simple sentences split rule (keeping _SEGMENT_RULE).
    for text in (ped_raw, ped_a2, templates["pedagogical_analysis"], live_ctx.request.instructions):
        assert _SEGMENT_RULE in text
        assert "3 и более сегментов" in text
        assert "2 простых" in text
        assert '"segments":[{"index":1,"treatment":"READ_AS_IS"}' in text
        # 3. critical vs useful, lexical_hint in GLOSSARY_ONLY, and tense-name ban.
        assert "critical" in text and "useful" in text
        assert 'help_kind="lexical_hint"' in text
        assert "Past Perfect" in text
        assert "Present Perfect" in text
        assert "прошедшее предшествующее время" in text

    # 2. Explicit A2 stoplist in _reader_level_treatment_guidance("a2") and live/template pedagogical A2 instructions.
    stoplist_markers = (
        "uneasy",
        "nervous",
        "Present Perfect",
        "would",
        "What...for?",
        "what",
        "сарказм",
        "don't got",
        "идиомы изменения состояния",
        "> 3 слов выше A2",
    )
    for text in (a2_guidance, ped_a2, templates["pedagogical_analysis"], live_ctx.request.instructions):
        for marker in stoplist_markers:
            assert marker in text

    # 4. A2 summary template in unified _pedagogical_instructions (not duplicated in _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS) and whole/fragment A2 summaries.
    summary_markers = (
        "<= 12–15 слов на фразу",
        "Present Simple + for + длительность",
        "эпиграф",
        "реплик",
    )
    assert "Present Simple + for + длительность" not in ped_raw
    assert "общеязыковых слов" not in ped_raw
    for text in (
        ped_a2,
        templates["pedagogical_analysis"],
        live_ctx.request.instructions,
        _WHOLE_CHAPTER_SUMMARY_A2.format(source_language="en"),
        _FRAGMENT_SUMMARY_A2.format(source_language="en"),
    ):
        for marker in summary_markers:
            assert marker in text
    for text in (ped_a2, templates["pedagogical_analysis"], live_ctx.request.instructions):
        assert text.count("Present Simple + for + длительность") == 1
        assert "общеязыковых слов" not in text
        assert ";" in text
        assert "are lost" in text
        assert "At the end, a quote from [Author]'s book explains..." in text

    # No duplicate "might have been" or _A2_NOTE_DISCIPLINE after deduplication.
    note_discipline_body = _A2_NOTE_DISCIPLINE.strip()
    for text in (live_ctx.request.instructions, template_ctx.request.instructions):
        assert text.count("might have been") == 1
        assert text.count(note_discipline_body) == 1

    # learner_instructions("a2") itself remains untouched (no new stoplist or segment JSON injected there).
    learner_a2 = learner_instructions("a2")
    assert '"segments":[{"index":1' not in learner_a2
    assert "don't got" not in learner_a2


def test_pedagogical_context_passes_prior_chapter_topics_and_segment_notes_budget() -> None:
    """prior_chapter_topics has its own section and affects pedagogical_input_sha256; _scene_notes_budget uses segment_treatments."""
    from dataclasses import replace

    from linguary.application.context_builder import _scene_notes_budget
    from linguary.domain import (
        BlockSegment,
        ChapterAnalysis,
        CoverageStatus,
        TreatmentPlanItem,
    )
    from linguary.domain.content_plan import ContentPlan, Difficulty, RequiredPoint, Scene, SourceAnchor

    # Part A: prior_chapter_topics vs prior_explained_topics + SHA-256 sensitivity.
    section = make_analysis_section(
        section_id="sec-ch3",
        title="Chapter 3",
        order=2,
        texts=("Nick wondered what the road was for.",),
    )
    document = make_analysis_document((section,))
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")

    ctx_both = builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="Nick walks along the road.",
        prior_chapter_topics=("used to", "would", "used to", " "),
        prior_explained_topics=("what ... for", "what ... for"),
    )
    input_text = ctx_both.request.input_text
    assert "Уже объяснено в предыдущих главах" in input_text
    assert "Уже объяснено в предыдущих фрагментах этой главы" in input_text
    assert input_text.count("- used to") == 1
    assert input_text.count("- would") == 1
    assert input_text.count("- what ... for") == 1

    sha_none = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="Nick walks along the road.",
    )
    sha_empty = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="Nick walks along the road.",
        prior_chapter_topics=(),
    )
    sha_topics = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="Nick walks along the road.",
        prior_chapter_topics=("used to", "would"),
    )
    assert sha_none == sha_empty
    assert sha_topics != sha_none

    # Part B: _scene_notes_budget honors overlapping READ_AS_IS segment in owners.
    seg_text = "Complex clause needing translation. Simple echo here."
    split_pos = seg_text.index("Simple")
    b1 = make_analysis_block(section_id="sec-seg", order=0, text="First translate paragraph with many words. " * 4)
    b2 = make_analysis_block(section_id="sec-seg", order=1, text="Second translate paragraph with many words. " * 4)
    b3 = replace(
        make_analysis_block(section_id="sec-seg", order=2, text=seg_text),
        segments=(
            BlockSegment(0, split_pos, 0),
            BlockSegment(split_pos, len(seg_text), 1),
        ),
    )
    seg_section = make_analysis_section(section_id="sec-seg", title="Chapter 4", blocks=(b1, b2, b3))
    plan_read = TreatmentPlan(
        section_id="sec-seg",
        items=(
            TreatmentPlanItem(b1.id, BlockTreatment.TRANSLATE),
            TreatmentPlanItem(b2.id, BlockTreatment.TRANSLATE),
            TreatmentPlanItem(
                b3.id,
                BlockTreatment.TRANSLATE,
                segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
                sentence_indexes=(0, 1),
            ),
        ),
    )
    # b1 and b2 have TRANSLATE explanations; b3 has an explanation overlapping its READ_AS_IS segment.
    # With per_chapter=2, b3 must be reserved first in owners (READ_AS_IS), so unit([b3]) gets budget=1.
    read_difficulties = (
        Difficulty(
            "d1",
            (SourceAnchor(b1.id, 0, 5, "First"),),
            b1.id,
            "idiom",
            "useful",
            "explanation",
            (RequiredPoint("p1", "Explain"),),
        ),
        Difficulty(
            "d2",
            (SourceAnchor(b2.id, 0, 6, "Second"),),
            b2.id,
            "idiom",
            "useful",
            "explanation",
            (RequiredPoint("p2", "Explain"),),
        ),
        Difficulty(
            "d3",
            (SourceAnchor(b3.id, split_pos, split_pos + 11, "Simple echo"),),
            b3.id,
            "grammar",
            "critical",
            "explanation",
            (RequiredPoint("p3", "Explain segment"),),
        ),
    )
    read_analysis = ChapterAnalysis(
        id="an-seg-read",
        section_id="sec-seg",
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=plan_read,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        content_plan=ContentPlan(version=3, profile="content-plan-v3", difficulties=read_difficulties),
        scenes=(Scene("s1", 0, b1.id, b3.id, "Scene"),),
    )
    unit_b3 = TranslationUnit.of_blocks(id="u-b3", blocks=(b3,), plan=plan_read)
    assert _scene_notes_budget(read_analysis, seg_section, unit_b3, per_chapter=2) == 1

    # Part C: _scene_notes_budget honors overlapping GLOSSARY_ONLY segment on grammar in extra_owners.
    gloss_text = "Hard opening clause here. What is it for?"
    gloss_split = gloss_text.index("What")
    g1 = replace(
        make_analysis_block(section_id="sec-gloss", order=0, text=gloss_text),
        segments=(
            BlockSegment(0, gloss_split, 0),
            BlockSegment(gloss_split, len(gloss_text), 1),
        ),
    )
    g2 = make_analysis_block(section_id="sec-gloss", order=1, text="Very long plain filler text. " * 40)
    gloss_section = make_analysis_section(section_id="sec-gloss", title="Chapter 5", blocks=(g1, g2))
    plan_gloss = TreatmentPlan(
        section_id="sec-gloss",
        items=(
            TreatmentPlanItem(
                g1.id,
                BlockTreatment.TRANSLATE,
                segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
                sentence_indexes=(0, 1),
            ),
            TreatmentPlanItem(g2.id, BlockTreatment.TRANSLATE),
        ),
    )
    gloss_difficulty = Difficulty(
        "dg1",
        (SourceAnchor(g1.id, gloss_split, len(gloss_text), "What is it for?"),),
        g1.id,
        "grammar",
        "critical",
        "lexical_hint",
        (RequiredPoint("pg1", "Explain preposition stranding"),),
    )
    gloss_analysis = ChapterAnalysis(
        id="an-seg-gloss",
        section_id="sec-gloss",
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=plan_gloss,
        translation_instructions="Keep tone.",
        chapter_summary="Summary.",
        content_plan=ContentPlan(version=3, profile="content-plan-v3", difficulties=(gloss_difficulty,)),
        scenes=(Scene("s1", 0, g1.id, g2.id, "Scene"),),
    )
    unit_g1 = TranslationUnit.of_blocks(id="u-g1", blocks=(g1,), plan=plan_gloss)
    assert _scene_notes_budget(gloss_analysis, gloss_section, unit_g1, per_chapter=1) == 1


def test_fragment_findings_candidates_rendered_in_narrative_chapter_input() -> None:
    """FragmentFindings.candidates renders concise glossary consistency section without None."""
    from linguary.application.context_builder import FragmentFindings
    from linguary.application.llm_dto import GlossaryCandidateResponse
    from linguary.domain.name_translation import NameTranslationAssessment, NameTranslationMethod

    section = make_analysis_section(
        section_id="sec-frag-cand",
        title="Chapter 2",
        order=1,
        texts=("First fragment line.", "Second fragment line about Everlost and Nick."),
    )
    document = make_analysis_document((section,))
    fragments = plan_fragments(textual_blocks(section), max_blocks=1)
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")

    default_findings = FragmentFindings(
        summary="Nick wakes up.",
        instructions="- Ник и Элли на «ты».",
        expressions=(),
    )
    assert default_findings.candidates == ()
    ctx_no_cands = builder.for_chapter_narrative_analysis(
        document,
        section,
        fragment=fragments[1],
        fragment_count=2,
        findings=default_findings,
    )
    assert "Кандидаты глоссария из предыдущих частей этой главы:" not in ctx_no_cands.request.input_text
    assert "ты/вы" in ctx_no_cands.request.instructions
    assert "не повторяй" in ctx_no_cands.request.instructions.lower()

    first_ctx = builder.for_chapter_narrative_analysis(
        document,
        section,
        fragment=fragments[0],
        fragment_count=2,
        findings=None,
    )
    assert "не повторяй уже зафиксированные" not in first_ctx.request.instructions.lower()

    cand_full = GlossaryCandidateResponse(
        source_text="Everlost",
        type=ConceptType.PLACE,
        suggested_translation="Вечнопустье",
        confidence=0.95,
        model_flagged_review=False,
        note="несклоняемое",
        description="Мир между жизнью и смертью.",
        name_translation_assessment=NameTranslationAssessment(
            applied_method=NameTranslationMethod.MEANING,
            eligible_without_review=True,
            review_reasons=(),
            rationale="Прозрачное составное название.",
            opaque_name=False,
        ),
    )
    cand_opaque = GlossaryCandidateResponse(
        source_text="Nick",
        type=ConceptType.PERSON,
        suggested_translation="Ник",
        confidence=0.98,
        model_flagged_review=False,
        note="",
        description="",
        name_translation_assessment=NameTranslationAssessment(
            applied_method=NameTranslationMethod.SOUND,
            eligible_without_review=True,
            review_reasons=(),
            rationale="Непрозрачное личное имя.",
            opaque_name=True,
        ),
    )
    cand_minimal = GlossaryCandidateResponse(
        source_text="Afterlight",
        type=ConceptType.FICTIONAL_CONCEPT,
        suggested_translation="Послесвет",
        confidence=0.9,
        model_flagged_review=False,
    )
    findings = FragmentFindings(
        summary="Nick and Allie reach Everlost.",
        instructions="- Ник и Элли на «ты».",
        expressions=(),
        candidates=(cand_full, cand_opaque, cand_minimal),
    )
    ctx = builder.for_chapter_narrative_analysis(
        document,
        section,
        fragment=fragments[1],
        fragment_count=2,
        findings=findings,
    )
    text = ctx.request.input_text
    assert "Кандидаты глоссария из предыдущих частей этой главы:" in text
    assert "единообразие" in text.lower()
    assert (
        "- Everlost [place] -> Вечнопустье (несклоняемое) — "
        "описание: Мир между жизнью и смертью.; непрозрачное=false"
    ) in text
    assert "- Nick [person] -> Ник — непрозрачное=true" in text
    assert "- Afterlight [fictional_concept] -> Послесвет" in text
    assert "None" not in text


def test_pedagogical_narrative_instructions_in_input_and_sha256() -> None:
    """narrative_instructions enters «Принятый повествовательный анализ» and pedagogical_input_sha256."""
    section = make_analysis_section(
        section_id="sec-narr-instr",
        title="Chapter 3",
        order=2,
        texts=("They walked into the forest.",),
    )
    document = make_analysis_document((section,))
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")

    ctx_empty = builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="They enter the forest.",
        narrative_instructions="   ",
    )
    assert "Указания переводчику:" not in ctx_empty.request.input_text

    instructions_text = "  - Передавай иронию сухо.\n- Не раскрывай имя до финала сцены.  "
    ctx_with = builder.for_chapter_pedagogical_analysis(
        document,
        section,
        narrative_summary="They enter the forest.",
        narrative_instructions=instructions_text,
    )
    expected_block = (
        "Указания переводчику:\n"
        "- Передавай иронию сухо.\n"
        "- Не раскрывай имя до финала сцены."
    )
    assert expected_block in ctx_with.request.input_text
    accepted_idx = ctx_with.request.input_text.index("Принятый повествовательный анализ:")
    instr_idx = ctx_with.request.input_text.index(expected_block)
    assert instr_idx > accepted_idx

    sha_default = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="They enter the forest.",
    )
    sha_blank = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="They enter the forest.",
        narrative_instructions="   ",
    )
    sha_with = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="They enter the forest.",
        narrative_instructions=instructions_text,
    )
    sha_trimmed = builder.pedagogical_input_sha256(
        document,
        section,
        narrative_output_sha256="narr-sha",
        narrative_summary="They enter the forest.",
        narrative_instructions=instructions_text.strip(),
    )
    assert sha_default == sha_blank
    assert sha_with != sha_default
    assert sha_with == sha_trimmed


def test_narrative_prompt_and_name_policy_p5_t2_rules() -> None:
    """Narrative prompt covers all 5 description types, non-empty anchors, speaker anchor split, atomic instructions, descriptive nicknames, and 7 review_reasons."""
    from linguary.application.context_builder import _NARRATIVE_ANALYSIS_INSTRUCTIONS, _with_name_policy

    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")
    templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
        name_translation_policy=_policy("meaning"),
    )
    for text in (narrative, templates["narrative_analysis"]):
        for concept_type in ("person", "place", "organization", "fictional_concept", "term"):
            assert concept_type in text
        assert "1–2 предложения" in text
        assert "anchors: []" in text
        assert "реплику в кавычках" in text
        assert "said ..." in text
        assert "до конца фрагмента" in text
        assert "без подстановки имени из других глав" in text
        assert "Iron Hand" in text
        assert "the freckled boy" not in text
        assert "the younger boy" in text
        assert "НЕ включай в `glossary_candidates`" in text
        assert "opaque_name=false" in text
        assert 'applied_method="meaning"' in text

    all_seven_reasons = (
        "ambiguous",
        "wordplay",
        "anagram",
        "uncertain_etymology",
        "creative_adaptation",
        "personal_name_substitution",
        "other",
    )
    for strategy in ("sound", "meaning"):
        analysis_prompt = _with_name_policy("Base.", _policy(strategy), analysis=True)
        for reason in all_seven_reasons:
            assert reason in analysis_prompt
        assert "запрещено" in analysis_prompt.lower()
        non_analysis_prompt = _with_name_policy("Base.", _policy(strategy), analysis=False)
        assert "uncertain_etymology" not in non_analysis_prompt


def test_p5_t3_note_budget_split_a2_calibration_antispoiler_and_summary() -> None:
    """P5.T3: v3 splits explanation vs unlimited lexical_hint budget, A2 removes contradictions, adds anti-spoiler & unified summary."""
    from linguary.application.context_builder import (
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _a2_word_share_corridor,
        _pedagogical_instructions,
        _reader_level_treatment_guidance,
        scaled_chapter_note_limit,
    )

    assert scaled_chapter_note_limit(35, 8) == 8
    assert scaled_chapter_note_limit(36, 8) == 8
    assert scaled_chapter_note_limit(60, 8) == 9
    assert scaled_chapter_note_limit(100, 8) == 12
    assert scaled_chapter_note_limit(168, 8) == 18
    assert scaled_chapter_note_limit(300, 6) == 18
    assert scaled_chapter_note_limit(300, 8) == 18

    assert _a2_word_share_corridor(0) == (70, 82)
    assert _a2_word_share_corridor(9_999) == (70, 82)
    assert _a2_word_share_corridor(10_000) == (72, 80)

    section = make_analysis_section(
        section_id="sec-quota",
        title="Chapter 1",
        order=0,
        texts=("First block.", "Second block."),
    )
    document = make_analysis_document((section,))
    v3_ctx = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        language_notes_per_chapter=8,
    ).for_chapter_pedagogical_analysis(document, section, narrative_summary="Summary.")
    assert 'help_kind="explanation"' in v3_ctx.request.input_text
    assert 'help_kind="lexical_hint"' in v3_ctx.request.input_text
    assert 'priority="critical"' in v3_ctx.request.input_text
    assert 'priority="useful"' in v3_ctx.request.input_text
    assert "GLOSSARY_ONLY" in v3_ctx.request.input_text
    assert "не ограничены" in v3_ctx.request.input_text

    v2_ctx = ContextBuilder(
        content_plan_profile="content-plan-v2",
        reader_level="b2",
        language_notes_per_chapter=8,
    ).for_chapter_analysis(document=document, section=section)
    assert "Бюджет обязательных заметок этого анализа: 2." in v2_ctx.request.input_text

    a2_guidance = _reader_level_treatment_guidance("a2")
    assert "выбирай щедро" not in a2_guidance
    assert "Если сомневаешься между режимами, выбирай более сильную помощь (TRANSLATE)" not in a2_guidance
    assert "Реплики длиннее 8 слов либо сочетающие прямую речь" not in a2_guidance
    assert "15 слов" in a2_guidance
    assert "45 слов" in a2_guidance

    ped_raw = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="A2",
    )
    assert "translation_instructions" in ped_raw
    assert "эвфемизма" in ped_raw
    assert "раньше момента раскрытия" in ped_raw
    assert "are lost" not in ped_raw

    ped_a2 = _pedagogical_instructions("en", "ru", "a2", False)
    assert "точка с запятой «;»" in ped_a2
    assert "are lost" in ped_a2
    assert "At the end, a quote from [Author]'s book explains..." in ped_a2
    assert "общеязыковых слов" not in ped_a2
    assert ped_a2.count("Present Simple + for + длительность") == 1


def test_non_staged_a2_summary_prompts_align_with_validate_a2_summary_rules() -> None:
    from linguary.application.context_builder import (
        _FRAGMENT_SUMMARY_A2,
        _WHOLE_CHAPTER_SUMMARY_A2,
    )

    whole = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    fragment = chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    for text in (
        _WHOLE_CHAPTER_SUMMARY_A2.format(source_language="en"),
        _FRAGMENT_SUMMARY_A2.format(source_language="en"),
        whole,
        fragment,
    ):
        assert "точка с запятой «;»" in text
        assert "are lost" in text
        assert "At the end, a quote from [Author]'s book explains..." in text


def test_v7_a2_summary_no_heading_prefix_byr_silence_note_budget_and_anchor_rules() -> None:
    """Everlost v7 polish: ban summary heading prefix, drop A2 v3 BYR noise, tighten note budget & pedagogical rules."""
    from linguary.application.context_builder import (
        _FRAGMENT_SUMMARY_A2,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _WHOLE_CHAPTER_SUMMARY_A2,
        _pedagogical_instructions,
    )

    no_heading_rule = (
        "Не пиши в начале служебный заголовок вроде «Previously in Chapter:» — "
        "начинай сразу с первого предложения пересказа."
    )
    templates_a2 = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    ped_a2 = _pedagogical_instructions("en", "ru", "a2", False)
    whole_a2 = chapter_analysis_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    fragment_a2 = chapter_fragment_instructions(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        reader_level="a2",
    )
    for text in (
        ped_a2,
        templates_a2["pedagogical_analysis"],
        _WHOLE_CHAPTER_SUMMARY_A2.format(source_language="en"),
        _FRAGMENT_SUMMARY_A2.format(source_language="en"),
        whole_a2,
        fragment_a2,
    ):
        assert "(Previously in Chapter...):" not in text
        assert no_heading_rule in text

    section = make_analysis_section(
        section_id="sec-v7",
        title="Chapter 1",
        order=0,
        texts=("First block.", "Second block."),
    )
    document = make_analysis_document((section,))

    # A2 + v3 (non-english-help): no contradictory BYR lines, explicit A2 empty BYR instruction.
    a2_v3_builder = ContextBuilder(
        content_plan_profile="content-plan-v3",
        reader_level="a2",
        language_notes_per_chapter=8,
    )
    a2_v3_ped = a2_v3_builder.for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Summary.", before_you_read_limit=0,
    )
    a2_v3_single = a2_v3_builder.for_chapter_analysis(document=document, section=section)
    for ctx in (a2_v3_ped, a2_v3_single):
        assert "Соблюдай лимиты BYR и обязательных заметок" not in ctx.request.instructions
        assert "Сначала используй существующий BYR" not in ctx.request.instructions
        assert (
            "Соблюдай лимиты обязательных заметок, указанные в запросе. "
            "Для A2 верни before_you_read: [] и before_you_read_evidence: []."
        ) in ctx.request.instructions
        assert "В этом ответе не больше" not in ctx.request.input_text
        assert "нельзя тратить все заметки только на TRANSLATE" in ctx.request.input_text
        assert "GLOSSARY_ONLY и READ_AS_IS" in ctx.request.input_text
        assert "не более половины лимита" in ctx.request.input_text

    # B1/B2/C1 in v3 keep existing BYR instructions & limit line, plus get strengthened v3 note_budget_line.
    for level in ("b1", "b2", "c1"):
        lvl_ctx = ContextBuilder(
            content_plan_profile="content-plan-v3",
            reader_level=level,
            language_notes_per_chapter=8,
        ).for_chapter_pedagogical_analysis(document, section, narrative_summary="Summary.")
        assert "Соблюдай лимиты BYR и обязательных заметок, указанные в запросе." in lvl_ctx.request.instructions
        assert "Сначала используй существующий BYR" in lvl_ctx.request.instructions
        assert "В этом ответе не больше" in lvl_ctx.request.input_text
        assert "нельзя тратить все заметки только на TRANSLATE" in lvl_ctx.request.input_text
        assert "GLOSSARY_ONLY и READ_AS_IS" in lvl_ctx.request.input_text

    # Legacy v2 (even for A2) keeps old BYR behavior and simple note_budget_line unchanged.
    a2_v2_ctx = ContextBuilder(
        content_plan_profile="content-plan-v2",
        reader_level="a2",
        language_notes_per_chapter=8,
    ).for_chapter_analysis(document=document, section=section)
    assert "Соблюдай лимиты BYR и обязательных заметок, указанные в запросе." in a2_v2_ctx.request.instructions
    assert "Сначала используй существующий BYR" in a2_v2_ctx.request.instructions
    assert "В этом ответе не больше" in a2_v2_ctx.request.input_text
    assert "нельзя тратить все заметки только на TRANSLATE" not in a2_v2_ctx.request.input_text

    # Pedagogical instructions: >=3 segments, parody/wordplay prototype, anchor coverage & dedup.
    ped_raw = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="A2",
    )
    assert "3 и более сегментов" in ped_raw
    assert "4 и более сегментов" not in ped_raw
    assert "wordplay" in ped_raw
    assert "исходное имя или текст-прототип" in ped_raw
    assert "anchors обязан охватывать все слова, разбираемые в required_points" in ped_raw
    assert "за пределами якоря" in ped_raw
    assert "две трудности с одинаковым якорем в одном блоке" in ped_raw


def test_v7_narrative_compact_instructions_and_fragment_speaker_facts_reminder() -> None:
    """Narrative instructions enforce compact translation_instructions and full-fragment speaker facts."""
    from linguary.application.context_builder import _NARRATIVE_ANALYSIS_INSTRUCTIONS

    narrative = _NARRATIVE_ANALYSIS_INSTRUCTIONS.format(source_language="en", target_language="ru")
    assert "3–8 общих правил на главу/фрагмент" in narrative
    assert "пофразовый" in narrative
    assert "подстрочник" in narrative
    assert "в начале/конце фрагмента" in narrative
    assert "остаётся неназванным" in narrative

    section = make_analysis_section(
        section_id="sec-v7-narr",
        title="Chapter 2",
        order=1,
        texts=("First line.", "Second line.", "Third line.", "Fourth line."),
    )
    document = make_analysis_document((section,))
    builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")

    whole_ctx = builder.for_chapter_narrative_analysis(document, section)
    assert "равномерно с первого до последнего диалогового блока" not in whole_ctx.request.instructions

    single_frag = plan_fragments(textual_blocks(section), max_blocks=10)[0]
    single_frag_ctx = builder.for_chapter_narrative_analysis(
        document, section, fragment=single_frag, fragment_count=1,
    )
    assert "равномерно с первого до последнего диалогового блока" not in single_frag_ctx.request.instructions

    fragments = plan_fragments(textual_blocks(section), max_blocks=2)
    assert len(fragments) == 2
    for frag in fragments:
        frag_ctx = builder.for_chapter_narrative_analysis(
            document, section, fragment=frag, fragment_count=len(fragments),
        )
        assert "context_facts (speaker, pronoun_referent)" in frag_ctx.request.instructions
        assert "равномерно с первого до последнего диалогового блока" in frag_ctx.request.instructions
        assert "не пропуская начало нового фрагмента" in frag_ctx.request.instructions


def test_v8_prompts_omit_byr_for_a2_pedagogy_forbid_bare_translate_hints_and_remind_fragment_summary() -> None:
    """V8 prompts drop BYR noise from A2 pedagogy, forbid bare glossary_hints on TRANSLATE, and remind multi-fragment A2 summary scope."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING_V3,
        _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS,
        _pedagogical_instructions,
    )

    # 1. _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS keeps its exact 3-key format signature.
    raw_formatted = _PEDAGOGICAL_ANALYSIS_INSTRUCTIONS.format(
        source_language="en", target_language="ru", reader_level="A2",
    )
    assert "BYR" in raw_formatted

    # 2. _pedagogical_instructions omits BYR / Before You Read for A2 (english_help=False), keeps BYR for B1/B2/C1.
    a2_ped = _pedagogical_instructions("en", "ru", "a2", False)
    assert "BYR" not in a2_ped
    assert "Before You Read" not in a2_ped
    assert "before_you_read: []" in a2_ped
    assert "before_you_read_evidence: []" in a2_ped
    assert 'help_kind="lexical_hint"' in a2_ped

    a2_templates = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )
    assert "BYR" not in a2_templates["pedagogical_analysis"]
    assert "Before You Read" not in a2_templates["pedagogical_analysis"]

    for level in ("b1", "b2", "c1"):
        lvl_ped = _pedagogical_instructions("en", "ru", level, False)
        assert "BYR" in lvl_ped

    # 3. _CONTENT_PLAN_PROCESSING_V3 forbids bare glossary_hints (without note) on pure TRANSLATE.
    assert "TRANSLATE" in _CONTENT_PLAN_PROCESSING_V3
    assert "GLOSSARY_ONLY" in _CONTENT_PLAN_PROCESSING_V3
    assert "glossary_hints" in _CONTENT_PLAN_PROCESSING_V3
    assert "пустым note" in _CONTENT_PLAN_PROCESSING_V3

    # 4. for_chapter_pedagogical_analysis adds whole-chapter reminder for multi-fragment A2 only.
    section = make_analysis_section(
        section_id="sec-v8-frag",
        title="Chapter 3",
        order=2,
        texts=("First line.", "Second line.", "Third line.", "Fourth line."),
    )
    document = make_analysis_document((section,))
    a2_builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="a2")
    whole_ctx = a2_builder.for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Full chapter narrative summary.",
    )
    single_frag = plan_fragments(textual_blocks(section), max_blocks=10)[0]
    single_frag_ctx = a2_builder.for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Full chapter narrative summary.",
        fragment=single_frag, fragment_count=1,
    )
    assert "Фрагментная проверка" not in whole_ctx.request.instructions
    assert "Фрагментная проверка" not in single_frag_ctx.request.instructions

    fragments = plan_fragments(textual_blocks(section), max_blocks=2)
    assert len(fragments) == 2
    for frag in fragments:
        frag_ctx = a2_builder.for_chapter_pedagogical_analysis(
            document, section, narrative_summary="Full chapter narrative summary.",
            fragment=frag, fragment_count=len(fragments),
        )
        assert "Фрагментная проверка" in frag_ctx.request.instructions
        assert "reader_chapter_summary" in frag_ctx.request.instructions
        assert "всю главу" in frag_ctx.request.instructions
        assert "chapter_summary" in frag_ctx.request.instructions

    b2_builder = ContextBuilder(content_plan_profile="content-plan-v3", reader_level="b2")
    b2_frag_ctx = b2_builder.for_chapter_pedagogical_analysis(
        document, section, narrative_summary="Full chapter narrative summary.",
        fragment=fragments[0], fragment_count=len(fragments),
    )
    assert "Фрагментная проверка" not in b2_frag_ctx.request.instructions


def test_translation_prompts_specify_segment_index_and_a2_glossary_hint_rules() -> None:
    """WP-2 & WP-6.2: Non-frozen v2/v3 prompts document bN vs bN.M & 1-based index; A2 hints stay consistent."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING_V3,
        _TRANSLATION_CONTEXT_V2,
    )
    from linguary.application.reader_text import learner_instructions

    for prompt in (_TRANSLATION_CONTEXT_V2, _CONTENT_PLAN_PROCESSING_V3):
        assert "blocks[].source_block_id" in prompt
        assert "language_notes[].source_block_id" in prompt
        assert "никогда не метку сегмента bN.M" in prompt
        assert "segments[].index" in prompt
        assert "1-based номер M из исходной метки [bN.M]" in prompt

    a2_instructions = learner_instructions("a2")
    assert "Цитаты, expression, original и original_excerpt сохраняй дословно." not in a2_instructions
    assert (
        "Цитаты, expression и original_excerpt сохраняй дословно; "
        "в glossary_hints.original глаголы приводи к инфинитиву с to, "
        "остальные слова и выражения сохраняй как в тексте."
    ) in a2_instructions
    assert "отдельные трудные слова и устойчивые словосочетания" in a2_instructions


def test_mixed_block_glossary_only_prompt_rules_present() -> None:
    """WP2: non-frozen analysis and translation prompts explicitly guide GLOSSARY_ONLY segments."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING_V3,
        _SEGMENT_TRANSLATION_RULE,
        _reader_level_treatment_guidance,
    )

    assert "В смешанном блоке с сегментом GLOSSARY_ONLY" in _CONTENT_PLAN_PROCESSING_V3
    assert "GLOSSARY_ONLY" in _SEGMENT_TRANSLATION_RULE
    assert "glossary_hints" in _SEGMENT_TRANSLATION_RULE
    a2_guidance = _reader_level_treatment_guidance("a2")
    assert "хотя бы 1 незнакомого слова или выражения выше A2" in a2_guidance


def test_a2_v3_dialogue_dash_prompt_rule_plural_you_and_noun_infinitive_guard() -> None:
    """WP4: A2/V3 prompts sync dialogue dash-after-period rule, plural 'вы', and verbs-only `to` infinitives."""
    from linguary.application.context_builder import (
        _CONTENT_PLAN_PROCESSING_V3,
        _SEGMENT_TRANSLATION_RULE,
        _TRANSLATION_CONTEXT_V2,
        _TRANSLATION_INSTRUCTIONS,
    )
    from linguary.application.reader_text import (
        _A2_DIALOGUE_AND_CALQUE_RULES,
        _A2_NOTE_DISCIPLINE,
        learner_instructions,
    )

    templates_a2 = snapshot_prompt_templates(
        source_language="en",
        target_language="ru",
        reader_level="a2",
        profile="content-plan-v3",
    )

    for text in (
        _A2_DIALOGUE_AND_CALQUE_RULES,
        _TRANSLATION_CONTEXT_V2,
        _CONTENT_PLAN_PROCESSING_V3,
        learner_instructions("a2"),
        templates_a2["content_processing"],
        templates_a2["learner_quality"],
    ):
        assert "не ставь «— » в той же строке после точки" in text
        assert "Он обернулся: «…»" in text
        assert "к нескольким собеседникам" in text
        assert "множественное число «вы»" in text

    for text in (
        _A2_NOTE_DISCIPLINE,
        _TRANSLATION_CONTEXT_V2,
        _CONTENT_PLAN_PROCESSING_V3,
        learner_instructions("a2"),
        templates_a2["content_processing"],
        templates_a2["learner_quality"],
        templates_a2["pedagogical_analysis"],
    ):
        assert "приводи к инфинитиву с to только глаголы" in text
        assert "a guffaw" in text
        assert "оставляй существительными без to" in text

    # Dialogue-only rules stay out of pedagogical_analysis and frozen baseline templates.
    assert "не ставь «— » в той же строке после точки" not in templates_a2["pedagogical_analysis"]
    assert "не ставь «— » в той же строке после точки" not in _TRANSLATION_INSTRUCTIONS
    assert "не ставь «— » в той же строке после точки" not in _SEGMENT_TRANSLATION_RULE

