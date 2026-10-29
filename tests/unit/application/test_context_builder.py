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
from linguary.domain import BlockTreatment, BlockType, ConceptType, SupportStatus
from linguary.domain.name_translation import NameTranslationPolicy, NameTranslationStrategy

NEIGHBOUR_TEXT = "The neighbouring chapter must never reach the model."

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
    assert 'произношение' in templates['analysis']
    assert 'не переводи' in templates['analysis']
    assert 'смысловые части' in templates['analysis']


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
    """Candidates from previous chapters are passed as book decisions."""
    context = build_context(
        candidates=(
            make_analysis_candidate(
                source_text="Longbourn", suggested_translation="Лонгборн"
            ),
        )
    )

    assert "Longbourn" in context.request.input_text
    assert "Лонгборн" in context.request.input_text


def test_empty_accumulated_candidates_are_stated_explicitly() -> None:
    """Absence of accumulated decisions is stated in words rather than an empty section."""
    assert "Предыдущие главы" in build_context().request.input_text


def test_accumulated_candidates_are_deduplicated_and_limited() -> None:
    """Duplicates are collapsed, and the list is capped at the configured limit."""
    context = build_context(
        candidates=(
            make_analysis_candidate(source_text="Longbourn", analysis_id="analysis-1"),
            make_analysis_candidate(source_text="Longbourn", analysis_id="analysis-2"),
            make_analysis_candidate(source_text="Netherfield"),
        ),
        max_accumulated_candidates=1,
    )

    decisions = context.request.input_text
    assert decisions.count("Longbourn") == 1
    assert "Netherfield" not in decisions


def test_obvious_spellings_of_one_decision_reach_the_request_once() -> None:
    """Variants of a single name with and without a dot are sent as a single decision (QA-11)."""
    context = build_context(
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
    assert "не отрицаю, что заклинание блестящее" in instructions
    assert "Говорящий признаёт, что заклинание блестящее" in instructions


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


def test_a2_summary_is_two_parallel_paragraphs_for_whole_chapter_and_fragment() -> None:
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
        assert "ровно два абзаца" in instructions
        assert "сначала пересказ на языке оригинала книги" in instructions
        assert "затем полный русский перевод тех же предложений" in instructions

    assert "ровно два абзаца" not in chapter_analysis_instructions(
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


@pytest.mark.parametrize('level', ['a2', 'b1', 'b2'])
def test_english_help_templates_pin_level_and_exact_plan_rules(level):
    from linguary.application.context_builder import snapshot_prompt_templates
    templates = snapshot_prompt_templates(source_language='en', target_language='ru', reader_level=level, profile='content-plan-v3', english_help=True)
    rules = templates['english_help']
    assert level.upper() in rules
    analysis = templates['analysis']
    fragment = templates['fragment']
    content_analysis = templates['content_analysis']
    normalized_content_analysis = ' '.join(content_analysis.split())
    processing = templates['english_help_processing']
    assert 'READ_AS_IS' in analysis and 'GLOSSARY_ONLY' in analysis
    assert 'before_you_read' in analysis and 'всегда пустой список' in analysis
    assert 'пять-десять ключевых понятий' not in analysis
    assert '65–80% блоков главы' not in analysis
    assert 'fragment_assistances' not in analysis
    assert 'before_you_read_evidence всегда должны быть пустыми' in content_analysis
    assert 'Определи только конкретные слова, выражения или конструкции' in normalized_content_analysis
    assert 'которые могут помешать читателю уровня' in normalized_content_analysis
    assert 'не ограничивай их число' not in normalized_content_analysis
    assert 'каждое нечастотное, архаичное' not in normalized_content_analysis
    assert 'даже если общий смысл можно вывести из контекста' not in normalized_content_analysis
    assert 'GLOSSARY_ONLY или при полном покрытии в BYR' not in content_analysis
    assert 'fragment_assistances' in processing
    assert 'content_evidence верни пустыми списками' in processing
    assert 'Верни content_evidence' not in processing
    assert f'Для {level.upper()}' in processing
    assert not any(f'Для {other.upper()}' in processing for other in {'a2', 'b1', 'b2'} - {level})
    assert 'english_help' not in snapshot_prompt_templates(source_language='en', target_language='ru', reader_level=level)
