"""Состав контекста анализа главы (P3.T3, PRD § 13)."""

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
)
from linguary.domain import BlockTreatment, BlockType, ConceptType, SupportStatus

NEIGHBOUR_TEXT = "The neighbouring chapter must never reach the model."

#: Глава из четырёх абзацев: при пределе в два блока она делится на два фрагмента.
FRAGMENTED_TEXTS = (
    "Paragraph one.",
    "Paragraph two.",
    "Paragraph three.",
    "Paragraph four.",
)


def build_context(
    *,
    texts: tuple[str, ...] = ("It is a truth universally acknowledged.",),
    candidates: tuple[object, ...] = (),
    max_accumulated_candidates: int = 200,
    reader_level: str = "b2",
):
    """Контекст анализа первой главы книги из двух глав."""
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
        accumulated_candidates=candidates,  # type: ignore[arg-type]  # фабрика двойников
    )


def build_fragment_context(*, order: int = 0, fragment_count: int = 2):
    """Контекст одного фрагмента главы, поделённой по два блока."""
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
    """Текст каждого блока главы попадает в запрос под своим псевдонимом."""
    context = build_context(texts=("First paragraph.", "Second paragraph."))

    input_text = context.request.input_text

    assert "[b1]\nFirst paragraph." in input_text
    assert "[b2]\nSecond paragraph." in input_text


def test_analysis_context_excludes_neighbour_chapters() -> None:
    """Текст соседних глав в контекст анализа не передаётся (PRD § 13)."""
    context = build_context()

    assert NEIGHBOUR_TEXT not in context.request.input_text
    assert NEIGHBOUR_TEXT not in context.request.instructions


def test_analysis_context_names_book_chapter_and_languages() -> None:
    """Запрос называет книгу, главу и языки издания."""
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
    """`PRESERVE_ONLY` и нетекстовые блоки модели не передаются (PRD § 10.1)."""
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
    """Текст конструкции, перенесённой как есть, модель не видит (ADR § 3)."""
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
    """Кандидаты предыдущих глав передаются как принятые решения книги."""
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
    """Отсутствие принятых решений описано словами, а не пустотой в запросе."""
    assert "Предыдущие главы" in build_context().request.input_text


def test_accumulated_candidates_are_deduplicated_and_limited() -> None:
    """Повторы схлопываются, а список обрезается настроенным пределом."""
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
    """Написания одного имени с точкой и без передаются одним решением (QA-11)."""
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
    """Предметы и блюда, названные в честь людей, модель не делает персонажами (QA-11).

    Перевод предлагается ровно для той формы, что стоит в тексте: короткое имя не
    подменяется полным из принятых решений.
    """
    instructions = build_context().request.instructions

    assert "названные в честь людей" in instructions
    assert "перевод именно этой формы" in instructions
    assert "не подставляй полное имя" in instructions


def test_aliases_map_back_to_stable_block_ids() -> None:
    """Псевдоним запроса возвращается к стабильному идентификатору блока."""
    section = make_analysis_section(texts=("First.", "Second."))
    document = make_analysis_document((section,))

    context = ContextBuilder().for_chapter_analysis(document=document, section=section)

    assert context.aliases == ("b1", "b2")
    assert [context.block_id_for(alias) for alias in context.aliases] == [
        block.id for block in section.ordered_blocks
    ]


def test_expectation_describes_aliases_and_target_language() -> None:
    """Ожидаемый результат перечисляет то, что модель видела: псевдонимы блоков."""
    context = build_context(texts=("First.", "Second."))

    expectation = context.expectation

    assert expectation.block_ids == ("b1", "b2")
    assert expectation.target_language == TARGET_LANGUAGE
    assert all(block.treatment is None for block in expectation.blocks)


def test_unknown_alias_is_rejected() -> None:
    """Псевдоним, которого в запросе не было, к блоку не приводится."""
    context = build_context()

    with pytest.raises(ContextBuildError):
        context.block_id_for("b99")


def test_instructions_describe_every_textual_treatment() -> None:
    """Инструкция называет все режимы обработки и оба языка издания (PRD § 10.1)."""
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
    """Инструкция анализа адаптирует ориентиры под уровень читателя (PRD § 30)."""
    a2_instructions = build_context(reader_level="a2").request.instructions
    assert "A2" in a2_instructions
    assert "70–85%" in a2_instructions

    b1_instructions = build_context(reader_level="b1").request.instructions
    assert "B1" in b1_instructions
    assert "50–70%" in b1_instructions

    b2_instructions = build_context(reader_level="b2").request.instructions
    assert "B1–B2" in b2_instructions
    assert "60%" in b2_instructions

    c1_instructions = build_context(reader_level="c1").request.instructions
    assert "C1" in c1_instructions
    assert "65–85%" in c1_instructions


def test_before_you_read_instructions_use_advance_organizer() -> None:
    """Инструкция Before You Read требует тематические опоры и семантические ловушки."""
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
    """Инструкция называет каждый тип понятия глоссария (PRD § 12)."""
    instructions = build_context().request.instructions

    for concept_type in ConceptType:
        assert concept_type.value in instructions


def test_instructions_require_chapter_summary_in_source_language() -> None:
    """Инструкция требует сводку главы и материал `Before You Read` (PRD § 19)."""
    instructions = build_context().request.instructions

    assert "chapter_summary" in instructions
    assert "before_you_read" in instructions
    assert "translation_instructions" in instructions
    assert "glossary_candidates" in instructions


def test_chapter_without_textual_blocks_has_no_context() -> None:
    """Главу без текста анализировать нечем: контекст не собирается."""
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
    """Отрицательный предел числа накопленных решений — ошибка настройки."""
    with pytest.raises(ContextBuildError):
        ContextBuilder(max_accumulated_candidates=-1)


def test_fragment_context_contains_only_its_own_blocks() -> None:
    """Фрагмент главы видит свои блоки и не видит ни остальной главы, ни соседней."""
    context = build_fragment_context(order=1)

    input_text = context.request.input_text

    assert context.aliases == ("b1", "b2")
    assert "[b1]\nParagraph three." in input_text
    assert "[b2]\nParagraph four." in input_text
    assert "Paragraph one." not in input_text
    assert NEIGHBOUR_TEXT not in input_text


def test_fragment_context_names_its_place_in_the_chapter() -> None:
    """Запрос фрагмента называет его номер в главе: сводки соединяются по порядку."""
    input_text = build_fragment_context(order=0).request.input_text

    assert "Фрагмент главы: 1-й из 2" in input_text
    assert "Chapter 1" in input_text


def test_fragment_instructions_describe_a_part_of_the_chapter() -> None:
    """Фрагменту сказано, что он видит часть главы, а сводка описывает только её."""
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
    """Кроме границ фрагмента и сводки инструкция та же: выверенный текст не теряется."""
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
    """Фрагмент, номер которого выходит за число фрагментов главы, не собирается."""
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
