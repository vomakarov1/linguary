"""Указания главы против решений глоссария пользователя (Д-1, PRD § 11, § 13)."""

from __future__ import annotations

import pytest

from linguary.application.chapter_instructions import instructions_respecting_glossary
from linguary.domain import (
    ConceptType,
    GlossaryCandidate,
    GlossarySnapshotEntry,
    TermStatus,
)

WREN = GlossarySnapshotEntry(
    source_text="Wren Calder",
    translation="Рен Колдер",
    type=ConceptType.PERSON,
    status=TermStatus.APPROVED,
    note="Женский род во всех главах",
)
DUSKMERE = GlossarySnapshotEntry(
    source_text="Duskmere",
    translation="Даскмер",
    type=ConceptType.PLACE,
    status=TermStatus.APPROVED,
)
BENNET = GlossarySnapshotEntry(
    source_text="Mr. Bennet",
    translation="мистер Беннет",
    type=ConceptType.PERSON,
    status=TermStatus.LOCKED,
)


def chapter_candidate(source_text: str, translation: str) -> GlossaryCandidate:
    """Кандидат анализа этой главы с вариантом модели."""
    return GlossaryCandidate.create(
        id=f"candidate-{source_text}",
        analysis_id="analysis-1",
        source_text=source_text,
        type=ConceptType.PLACE,
        suggested_translation=translation,
        confidence=0.9,
        model_flagged_review=False,
        confidence_threshold=0.5,
    )


def kept(instructions: str, *decisions: GlossarySnapshotEntry, candidates=()) -> str:
    """Указания после согласования с решениями."""
    return instructions_respecting_glossary(
        instructions, decisions=decisions, chapter_candidates=candidates
    )


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(
            "2. Топоним «Duskmere» переводить строго по глоссарию — «Даскмир».",
            id="rejected-model-variant",
        ),
        pytest.param("2. Duskmere — «Даскмиръ», не склонять.", id="look-alike-variant"),
        pytest.param(
            "3. Рен Колдер — сохранять мужской род согласно принятому решению.",
            id="other-gender",
        ),
        pytest.param("3. Рен — в мужском роде.", id="first-name-only"),
        pytest.param("3. Wren говорит о себе в мужском роде.", id="source-name-part"),
    ],
)
def test_item_contradicting_a_decision_is_dropped(item: str) -> None:
    """Пункт с другим вариантом перевода или другим родом не доходит до модели."""
    instructions = f"1. Сохранять иронию рассказчика.\n{item}"

    result = kept(
        instructions,
        WREN,
        DUSKMERE,
        candidates=(chapter_candidate("Duskmere", "Даскмир"),),
    )

    assert result == "1. Сохранять иронию рассказчика."


@pytest.mark.parametrize(
    "item",
    [
        pytest.param("2. Рен говорит отрывисто, короткими фразами.", id="speech-style"),
        pytest.param(
            "2. Рен Колдер обращается к Селене «госпожа».", id="unrelated-quote"
        ),
        pytest.param("2. Рен — женский род, как решено.", id="same-gender"),
        pytest.param("2. Путь в «Даскмера» не сокращать.", id="grammatical-form"),
        pytest.param(
            "2. Мистер Беннет зовёт соседа «мистер Дарси» подчёркнуто сухо.",
            id="shared-title",
        ),
        pytest.param("2. Сохранять архаику «thou» в речи старухи.", id="no-mention"),
    ],
)
def test_item_agreeing_with_decisions_is_kept(item: str) -> None:
    """Пункт, который упоминает понятие, но решению не противоречит, остаётся."""
    instructions = f"1. Сохранять иронию рассказчика.\n{item}"

    assert kept(instructions, WREN, DUSKMERE, BENNET) == instructions


def test_gender_is_not_contradicted_when_decision_names_none() -> None:
    """Решение без рода указанию о роде не противоречит: сравнивать не с чем."""
    decision = GlossarySnapshotEntry(
        source_text="Wren Calder",
        translation="Рен Колдер",
        type=ConceptType.PERSON,
        status=TermStatus.APPROVED,
    )
    instructions = "1. Рен — мужской род."

    assert kept(instructions, decision) == instructions


def test_continuation_line_goes_with_its_item() -> None:
    """Строка без маркера списка продолжает пункт и уходит вместе с ним."""
    instructions = (
        "1. Рен Колдер — мужской род,\n"
        "   и в прямой речи тоже.\n"
        "2. Сохранять иронию рассказчика."
    )

    assert kept(instructions, WREN) == "2. Сохранять иронию рассказчика."


def test_plain_paragraph_is_split_into_sentences() -> None:
    """Сплошной абзац делится на предложения: противоречие не уносит весь текст."""
    instructions = (
        "Сохранять иронию рассказчика. Рен Колдер — мужской род. "
        "Диалоги держать короткими."
    )

    assert kept(instructions, WREN) == (
        "Сохранять иронию рассказчика.\nДиалоги держать короткими."
    )


def test_without_decisions_instructions_are_unchanged() -> None:
    """Без решений пользователя указания главы идут модели как есть."""
    instructions = "1. Рен Колдер — мужской род.\n2. Duskmere — «Даскмир»."

    assert kept(instructions) == instructions
