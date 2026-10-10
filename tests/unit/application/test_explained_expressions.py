"""Отсев повторов `Before You Read` в подсказках и заметках блока (PRD § 19.3, QA-03)."""

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
    """Регистр, пунктуация, пробелы и вид апострофа на сравнение не влияют."""
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
    """Повтор выражения узнаётся и в другой записи, и в словоформах, и частью."""
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
    """Короткие общие слова и длинные фрагменты с другим смыслом повтором не считаются."""
    assert not ExplainedExpressions([explained]).covers(text)


def test_repeats_are_removed_from_hints_and_notes() -> None:
    """Повторная подсказка и заметка отбрасываются, остальное не меняется."""
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
    """Блок GLOSSARY_ONLY не остаётся без обязательных подсказок, заметки при этом уходят."""
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
    """Заметка, чей отрывок — вся фраза, но объясняет она выражение `Before You Read`, уходит.

    Модель часто берёт в отрывок заметки целое предложение. По отрывку такой повтор
    не узнать — он больше чем вдвое длиннее выражения, — зато его выдаёт сама заметка:
    она называет объяснённое выражение (живой прогон, дефект Д-3).
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
    """Заметка к длинной фразе, где выражение лишь встретилось, а объясняется другое, остаётся."""
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
    """Короткое общее слово объяснённого выражения в тексте заметки повтором не делает."""
    explained = ExplainedExpressions(["heave to"])
    note = LanguageNote(
        id="ln-ch01-004",
        block_id="block-1",
        original_excerpt="they had to heave to before the storm reached the long pier",
        note="Reached the pier — «дошёл до», предлог to здесь не нужен.",
    )

    assert explained.without_repeats(_translated(note)).language_notes == (note,)


def test_without_expressions_result_is_unchanged() -> None:
    """Глава без `Before You Read` ничего не отсеивает."""
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
    """Результат блока `TRANSLATE` с переданными заметками."""
    return BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Перевод.",
        language_notes=notes,
    )


def _note(note_id: str, excerpt: str) -> LanguageNote:
    """Заметка к блоку с заданным фрагментом оригинала."""
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
