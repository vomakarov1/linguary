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
    explained = ExplainedExpressions(["heaved and groaned"])
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
    explained = ExplainedExpressions(["heaved and groaned"])
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


def test_without_expressions_result_is_unchanged() -> None:
    """Глава без `Before You Read` ничего не отсеивает."""
    result = BlockProcessingResult(
        block_id="block-1",
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Перевод.",
        language_notes=(_note("ln-ch01-001", "heaved and groaned"),),
    )

    assert ExplainedExpressions([" "]).without_repeats(result) is result


def _note(note_id: str, excerpt: str) -> LanguageNote:
    """Заметка к блоку с заданным фрагментом оригинала."""
    return LanguageNote(
        id=note_id, block_id="block-1", original_excerpt=excerpt, note="Пояснение."
    )
