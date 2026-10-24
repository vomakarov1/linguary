"""Filtering of `Before You Read` repeats in block hints and notes (PRD § 19.3, QA-03)."""

from __future__ import annotations

import pytest

from linguary.application.explained_expressions import (
    ExplainedExpressions,
    expression_words,
)
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    GlossaryHint,
    LanguageNote,
)


def test_expression_words_ignore_case_punctuation_and_spacing() -> None:
    """Casing, punctuation, whitespace, and apostrophe style do not affect comparison."""
    assert expression_words("  “Keep One’s  head,”\n") == ("keep", "one's", "head")


@pytest.mark.parametrize(
    ("explained", "text"),
    [
        pytest.param("heaved and groaned", "Heaved and groaned!", id="same-expression"),
        pytest.param("heave and groan", "heaved and groaned", id="word-forms"),
        pytest.param("heaved and groaned", "heaved and groaned in the", id="short-context"),
        pytest.param("had no business being there", "no business being", id="sub-phrase"),
        pytest.param("heave to", "heave", id="significant-word"),
        pytest.param("gee", "Gee!", id="short-exact"),
    ],
)
def test_repeated_expression_is_recognised(explained: str, text: str) -> None:
    """Repeated expressions are recognized across spellings, word forms, and as sub-phrases."""
    assert ExplainedExpressions([explained]).covers(text)


@pytest.mark.parametrize(
    ("explained", "text"),
    [
        pytest.param("heave to", "to", id="short-common-word"),
        pytest.param("heaved and groaned", "and", id="conjunction"),
        pytest.param(
            "heaved and groaned",
            "the old house heaved and groaned while she had no business being there",
            id="long-sentence",
        ),
        pytest.param("have no business", "business", id="minor-part"),
        pytest.param("care for", "cart for", id="different-word"),
        pytest.param("heaved and groaned", "", id="empty"),
    ],
)
def test_unrelated_text_is_not_treated_as_repeat(explained: str, text: str) -> None:
    """Short common words and long excerpts with different meaning are not treated as repeats."""
    assert not ExplainedExpressions([explained]).covers(text)


def test_repeats_are_removed_from_hints_and_notes() -> None:
    """Repeated hints and notes are discarded, leaving the rest unchanged."""
    explained = ExplainedExpressions(["heaved and groaned"], explanations=[("heaved and groaned", "Пояснение.")])
    result = BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(
            GlossaryHint(original="heaved and groaned", translation="вздымалось"),
            GlossaryHint(original="entailed", translation="заповедный"),
        ),
        language_notes=(
            _note("ln-ch01-001", "heaved and groaned"),
            _note("ln-ch01-002", "a truth universally acknowledged"),
        ),
    )

    filtered = explained.without_repeats(result)

    assert [hint.original for hint in filtered.glossary_hints] == ["entailed"]
    assert [note.id for note in filtered.language_notes] == ["ln-ch01-002"]


def test_required_hints_survive_when_all_of_them_repeat() -> None:
    """A GLOSSARY_ONLY block is not left without required hints, while notes are dropped."""
    explained = ExplainedExpressions(["heaved and groaned"], explanations=[("heaved and groaned", "Пояснение.")])
    result = BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(
            GlossaryHint(original="heaved and groaned", translation="вздымалось"),
        ),
        language_notes=(_note("ln-ch01-001", "heaved and groaned"),),
    )

    filtered = explained.without_repeats(result)

    assert filtered.glossary_hints == result.glossary_hints
    assert filtered.language_notes == ()


def test_note_with_whole_sentence_excerpt_explaining_the_expression_is_a_repeat() -> None:
    """A note whose excerpt is the entire sentence but explains a `Before You Read` expression is dropped.

    The model often takes an entire sentence as a note excerpt. Such a repeat cannot be detected
    by the excerpt alone (it is more than twice as long as the expression), but the note text itself
    reveals it: it names the explained expression (live run, defect D-3).
    """
    explained = ExplainedExpressions(["gatekeeper's edge"], explanations=[("gatekeeper's edge", "Gatekeeper's edge — «резкость привратницы»: edge здесь — резкость, острота тона.")])
    result = _translated(
        LanguageNote(
            id="ln-ch02-001",
            block_id="block-1",
            original_excerpt="her voice had lost its dry, gatekeeper’s edge entirely",
            note=(
                "Gatekeeper's edge — «резкость привратницы»: edge здесь — "
                "резкость, острота тона."
            ),
        ),
    )

    assert explained.without_repeats(result).language_notes == ()


def test_note_on_a_long_sentence_explaining_something_else_survives() -> None:
    """A note on a long sentence where the expression merely appears, but something else is explained, is kept."""
    explained = ExplainedExpressions(["heaved and groaned"], explanations=[("heaved and groaned", "Пояснение.")])
    note = LanguageNote(
        id="ln-ch01-003",
        block_id="block-1",
        original_excerpt=(
            "the old house heaved and groaned while she had no business being there"
        ),
        note="Have no business + Ving — идиома «не положено, незачем».",
    )
    result = _translated(note)

    assert explained.without_repeats(result).language_notes == (note,)


def test_short_word_in_note_text_does_not_make_a_repeat() -> None:
    """A short common word from the explained expression appearing in the note text does not make it a repeat."""
    explained = ExplainedExpressions(["heave to"])
    note = LanguageNote(
        id="ln-ch01-004",
        block_id="block-1",
        original_excerpt="they had to heave to before the storm reached the long pier",
        note="Reached the pier — «дошёл до», предлог to здесь не нужен.",
    )

    assert explained.without_repeats(_translated(note)).language_notes == (note,)


def test_without_expressions_result_is_unchanged() -> None:
    """A chapter without `Before You Read` filters out nothing."""
    result = BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Перевод.",
        language_notes=(_note("ln-ch01-001", "heaved and groaned"),),
    )

    assert ExplainedExpressions([" "]).without_repeats(result) is result


def test_note_repeating_block_hint_dropped() -> None:
    result = BlockProcessingResult(
        block_id="block-1", treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="Heaved and groaned", translation="Пояснение."),),
        language_notes=(_note("ln-1", "heaved, and groaned!"),),
    )
    assert ExplainedExpressions([]).without_repeats(result).language_notes == ()


def test_note_containing_hint_expression_dropped() -> None:
    result = BlockProcessingResult(
        block_id="block-1", treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="heaved and groaned", translation="Пояснение."),),
        language_notes=(_note("ln-1", "The house heaved and groaned all night"),),
    )
    assert ExplainedExpressions([]).without_repeats(result).language_notes == ()


def test_unrelated_note_kept() -> None:
    note = _note("ln-1", "another phrase")
    result = BlockProcessingResult(
        block_id="block-1", treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="heaved and groaned", translation="Пояснение."),),
        language_notes=(note,),
    )
    assert ExplainedExpressions([]).without_repeats(result).language_notes == (note,)


def _translated(*notes: LanguageNote) -> BlockProcessingResult:
    """Result of a `TRANSLATE` block with the provided notes."""
    return BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Перевод.",
        language_notes=notes,
    )


def _note(note_id: str, excerpt: str) -> LanguageNote:
    """Note for a block with a given original excerpt."""
    return LanguageNote(
        id=note_id, block_id="block-1", original_excerpt=excerpt, note="Пояснение."
    )


def _explained_note(expression: str, excerpt: str, explanation: str, text: str):
    explained = ExplainedExpressions([expression], explanations=[(expression, explanation)])
    note = LanguageNote(id="ln-1", block_id="block-1", original_excerpt=excerpt, note=text)
    return explained, note


def test_made_make_grab_at_same_explanation_dropped() -> None:
    explained, note = _explained_note("make a grab at", "made a grab at my throat",
        "резко схватить, сделать выпад рукой в сторону чего-либо",
        "Make a grab at — разговорная идиома со значением резко, порывисто схватить.")
    assert explained.without_repeats(_translated(note)).language_notes == ()


def test_sheath_armour_nested_excerpt_same_explanation_dropped() -> None:
    explained, note = _explained_note("his sheath and his armour", "He says it is his sheath and his armour against us all",
        "его ножны и броня: то, что одновременно скрывает и защищает врага",
        "Sheath (ножны) скрывают оружие; armour (броня/доспехи) защищает, отражает удары.")
    assert explained.without_repeats(_translated(note)).language_notes == ()


@pytest.mark.parametrize("fact", [
    "without to + инфинитив — ненормативная форма; норма: without prejudging.",
    "Здесь эллипсис: вспомогательный глагол пропущен.",
    "Здесь игра слов с другим значением grab.",
    "Это культурная реалия средневековой Англии.",
    "Sheath употребляется также как глагол.",
    "Ножны не скрывают оружие, броня не защищает.",
])
def test_same_excerpt_new_useful_fact_kept(fact: str) -> None:
    explained, note = _explained_note("his sheath and his armour", "his sheath and his armour",
        "ножны скрывают, броня защищает", "Ножны скрывают, броня защищает. " + fact)
    assert explained.without_repeats(_translated(note)).language_notes == (note,)


def test_common_word_overlap_kept() -> None:
    explained, note = _explained_note("make a grab at", "make a choice", "попытка схватить", "Make — сделать.")
    assert explained.without_repeats(_translated(note)).language_notes == (note,)


def test_block_hint_same_explanation_note_dropped() -> None:
    note = LanguageNote(id="ln-1", block_id="block-1", original_excerpt="made a grab at my throat", note="попытка ухватиться")
    result = BlockProcessingResult(block_id="block-1", treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="make a grab at", translation="попытка схватить"),), language_notes=(note,))
    assert ExplainedExpressions([]).without_repeats(result).language_notes == ()


def test_required_glossary_hint_retained() -> None:
    result = BlockProcessingResult(block_id="block-1", treatment=BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint(original="make a grab at", translation="попытка схватить"),))
    assert ExplainedExpressions(["make a grab at"]).without_repeats(result).glossary_hints == result.glossary_hints


def test_note_without_available_explanation_kept() -> None:
    note = _note("ln-1", "heaved and groaned")
    assert ExplainedExpressions(["heaved and groaned"]).without_repeats(_translated(note)).language_notes == (note,)


@pytest.mark.parametrize("separator", [", ", ": ", " — "])
@pytest.mark.parametrize("source", ["before-you-read", "block-hint"])
def test_grab_note_with_new_rhetorical_fact_in_same_sentence_kept(
    separator: str, source: str,
) -> None:
    """A rhetorical fact is preserved alongside the definition in the same sentence."""
    expression = "make a grab at"
    explanation = "попытка схватить"
    note = LanguageNote(
        id="ln-1", block_id="block-1", original_excerpt="made a grab at my throat",
        note="Make a grab at — резко схватить" + separator
        + "выразительный контраст с церемонностью.",
    )
    if source == "before-you-read":
        explained = ExplainedExpressions(
            [expression], explanations=[(expression, explanation)],
        )
        result = _translated(note)
    else:
        explained = ExplainedExpressions([])
        result = BlockProcessingResult(
            block_id="block-1", treatment=BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint(original=expression, translation=explanation),),
            language_notes=(note,),
        )

    filtered = explained.without_repeats(result)

    assert filtered.language_notes == (note,)
    assert filtered.glossary_hints == result.glossary_hints


def test_new_aspect_is_retained_without_phrase_rules():
    from linguary.domain import BlockProcessingResult, BlockTreatment, LanguageNote
    explained = ExplainedExpressions(('Echo',), explanations=(('Echo', 'Первое самостоятельное объяснение.'),))
    result = BlockProcessingResult('b', BlockTreatment.READ_AS_IS, language_notes=(LanguageNote('n', 'b', 'Echo', 'Первое самостоятельное объяснение. Дополнительный аспект.'),))
    assert explained.without_repeats_v2(result).language_notes == result.language_notes
