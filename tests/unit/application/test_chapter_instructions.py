"""Chapter instructions versus user glossary decisions (D-1, PRD § 11, § 13)."""

from __future__ import annotations

import pytest

from linguary.application.chapter_instructions import (
    instructions_respecting_glossary,
    note_respecting_glossary,
)
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
    """Chapter analysis candidate with model variant."""
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
    """Instructions after reconciliation with decisions."""
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
    """An item with a conflicting translation variant or gender does not reach the model."""
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
    """An item that mentions a concept without conflicting with its decision is preserved."""
    instructions = f"1. Сохранять иронию рассказчика.\n{item}"

    assert kept(instructions, WREN, DUSKMERE, BENNET) == instructions


def test_gender_is_not_contradicted_when_decision_names_none() -> None:
    """A decision without specified gender does not conflict with a gender instruction."""
    decision = GlossarySnapshotEntry(
        source_text="Wren Calder",
        translation="Рен Колдер",
        type=ConceptType.PERSON,
        status=TermStatus.APPROVED,
    )
    instructions = "1. Рен — мужской род."

    assert kept(instructions, decision) == instructions


def test_continuation_line_goes_with_its_item() -> None:
    """A line without a list marker continues the item and is dropped together with it."""
    instructions = (
        "1. Рен Колдер — мужской род,\n"
        "   и в прямой речи тоже.\n"
        "2. Сохранять иронию рассказчика."
    )

    assert kept(instructions, WREN) == "2. Сохранять иронию рассказчика."


def test_plain_paragraph_is_split_into_sentences() -> None:
    """A continuous paragraph is split into sentences so a conflict does not discard all text."""
    instructions = (
        "Сохранять иронию рассказчика. Рен Колдер — мужской род. "
        "Диалоги держать короткими."
    )

    assert kept(instructions, WREN) == (
        "Сохранять иронию рассказчика.\nДиалоги держать короткими."
    )


def test_without_decisions_instructions_are_unchanged() -> None:
    """Without user decisions, chapter instructions are sent to the model as is."""
    instructions = "1. Рен Колдер — мужской род.\n2. Duskmere — «Даскмир»."

    assert kept(instructions) == instructions


MARSH_GATE = GlossarySnapshotEntry(
    source_text="marsh gate",
    translation="болотные ворота",
    type=ConceptType.FICTIONAL_CONCEPT,
    status=TermStatus.APPROVED,
    note=(
        "Функциональный пост стража болота при въезде в Даскмир; уточнить устойчивый "
        "вариант перевода для всей книги."
    ),
)


def kept_note(
    entry: GlossarySnapshotEntry, *decisions: GlossarySnapshotEntry, candidates=()
) -> str:
    """Glossary item note after reconciliation with other concepts' decisions."""
    return note_respecting_glossary(
        entry, decisions=decisions, chapter_candidates=candidates
    )


def test_note_naming_rejected_variant_of_other_concept_is_dropped() -> None:
    """An item note does not propagate a rejected variant of another concept (N-2).

    Live run: the note for `marsh gate`, accepted via "Accept All", referred to
    the rejected variant ("Daskmir"), whereas the user decision for Duskmere was "Daskmer".
    """
    note = kept_note(
        MARSH_GATE,
        MARSH_GATE,
        DUSKMERE,
        candidates=(chapter_candidate("Duskmere", "Даскмир"),),
    )

    assert "Даскмир" not in note
    assert note == "уточнить устойчивый вариант перевода для всей книги."


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Пост стражи у въезда в Даскмер.", id="decision"),
        pytest.param("Пост стражи у стен Даскмера.", id="decision-form"),
        pytest.param("Пост стражи на болоте.", id="unrelated"),
    ],
)
def test_note_agreeing_with_other_decisions_is_kept(text: str) -> None:
    """A note mentioning the decision or its inflected form is kept as is."""
    entry = GlossarySnapshotEntry(
        source_text="marsh gate",
        translation="болотные ворота",
        type=ConceptType.FICTIONAL_CONCEPT,
        status=TermStatus.APPROVED,
        note=text,
    )

    assert (
        kept_note(
            entry, DUSKMERE, candidates=(chapter_candidate("Duskmere", "Даскмир"),)
        )
        == text
    )


def test_note_is_not_checked_against_its_own_decision() -> None:
    """An item's own note is not checked against its own decision: the rejected variant is valid in it."""
    entry = GlossarySnapshotEntry(
        source_text="Duskmere",
        translation="Даскмер",
        type=ConceptType.PLACE,
        status=TermStatus.APPROVED,
        note="Не «Даскмир»: название несклоняемое.",
    )

    assert (
        kept_note(entry, entry, candidates=(chapter_candidate("Duskmere", "Даскмир"),))
        == entry.note
    )


def test_note_with_other_gender_of_other_person_is_dropped() -> None:
    """A note disputing the gender specified by a person's decision is also discarded."""
    entry = GlossarySnapshotEntry(
        source_text="gatehouse",
        translation="дом привратника",
        type=ConceptType.TERM,
        status=TermStatus.APPROVED,
        note="Сюда Рен пришёл один — мужской род. Дом стоит на сваях.",
    )

    assert kept_note(entry, entry, WREN) == "Дом стоит на сваях."


def test_item_naming_rejected_variant_without_quotes_is_dropped() -> None:
    """A chapter instruction item naming a rejected variant without quotes is discarded (D-5).

    Third live run: the item "Use glossary terms (..., Daskmir) consistently according to decisions"
    reached the processing prompt, even though the decision for Duskmere was "Daskmer".
    """
    instructions = (
        "1. Сохранять иронию рассказчика.\n"
        "8. Термины из глоссария (звёздное стекло, болотный кварц, болотные огни, "
        "Даскмир) использовать единообразно согласно принятым решениям."
    )

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
        pytest.param("2. Путь в Даскмера не сокращать.", id="decision-form"),
        pytest.param("2. Даскмер — несклоняемое название.", id="decision"),
        pytest.param("2. Болотные огни видны издалека.", id="unrelated"),
    ],
)
def test_item_naming_decision_without_quotes_is_kept(item: str) -> None:
    """A user decision and its inflected forms without quotes do not cause an item to be discarded."""
    instructions = f"1. Сохранять иронию рассказчика.\n{item}"

    assert (
        kept(
            instructions,
            WREN,
            DUSKMERE,
            candidates=(chapter_candidate("Duskmere", "Даскмир"),),
        )
        == instructions
    )
