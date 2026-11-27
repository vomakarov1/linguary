"""WP6 tests for longest-match and case/article/hyphen-aware glossary mismatch warnings."""

from __future__ import annotations

from processing_fakes import make_block, make_block_result, make_response, make_unit

from linguary.application.validators.language import translation_quality_warnings
from linguary.domain import ConceptType, GlossarySnapshotEntry, TermStatus


def _warnings(
    source: str,
    translated: str,
    glossary: tuple[GlossarySnapshotEntry, ...],
) -> tuple[str, ...]:
    block = make_block(order=0, text=source)
    unit = make_unit((block,))
    payload = make_response((make_block_result(block, translated_text=translated),))
    return translation_quality_warnings(
        payload,
        unit=unit,
        target_language="ru",
        mandatory_glossary=glossary,
    )


def test_glossary_validator_respects_case_and_hyphen_boundaries_for_moon_and_bill() -> None:
    """Proper names `Moon` and `Bill` do not trigger on `half-moon`, `full moon`, or `a bill`."""
    glossary = (
        GlossarySnapshotEntry(
            source_text="Moon",
            translation="Мун",
            type=ConceptType.PERSON,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="Bill",
            translation="Билл",
            type=ConceptType.PERSON,
            status=TermStatus.APPROVED,
        ),
    )

    assert (
        _warnings(
            "He wore half-moon spectacles under the full moon and paid a bill.",
            "Он носил очки-половинки под полной луной и оплатил счёт.",
            glossary,
        )
        == ()
    )
    assert (
        _warnings(
            "HE WORE HALF-MOON SPECTACLES AND PAID A DUCK-BILL.",
            "ОН НОСИЛ ОЧКИ-ПОЛОВИНКИ И УВИДЕЛ УТКОНОСА.",
            glossary,
        )
        == ()
    )

    moon_missing = _warnings("Professor McGonagall called Moon next.", "Профессор Макгонагалл позвала Луну.", glossary)
    assert any("'Moon'" in w and "Мун" in w for w in moon_missing)
    assert _warnings("Professor McGonagall called Moon next.", "Профессор Макгонагалл позвала Мун.", glossary) == ()

    moon_allcaps = _warnings("MOON stepped forward.", "ЛУНА шагнула вперёд.", glossary)
    assert any("'Moon'" in w for w in moon_allcaps)
    assert _warnings("MOON stepped forward.", "МУН шагнул вперёд.", glossary) == ()

    bill_missing = _warnings("Bill was working in Egypt.", "Счёт работал в Египте.", glossary)
    assert any("'Bill'" in w and "Билл" in w for w in bill_missing)
    assert _warnings("Bill was working in Egypt.", "Билл работал в Египте.", glossary) == ()

    bill_allcaps = _warnings("BILL shouted from the door.", "СЧЁТ крикнул от двери.", glossary)
    assert any("'Bill'" in w for w in bill_allcaps)


def test_glossary_validator_longest_match_and_leading_article_case_insensitivity() -> None:
    """`The Sorting Hat` matches `the Sorting Hat` and masks nested `Sorting`, unless `Sorting` appears separately."""
    glossary = (
        GlossarySnapshotEntry(
            source_text="Sorting",
            translation="Распределение",
            type=ConceptType.TERM,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="The Sorting Hat",
            translation="Распределяющая шляпа",
            type=ConceptType.TERM,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="the Great Hall",
            translation="Большой зал",
            type=ConceptType.PLACE,
            status=TermStatus.APPROVED,
        ),
    )

    # Leading article is case-insensitive (`the Sorting Hat` matches `The Sorting Hat`,
    # `The Great Hall` matches `the Great Hall`), and longer span masks nested `Sorting`.
    assert (
        _warnings(
            "He put on the Sorting Hat inside The Great Hall.",
            "Он надел Распределяющую шляпу в Большом зале.",
            glossary,
        )
        == ()
    )
    assert (
        _warnings(
            "THE SORTING HAT spoke in THE GREAT HALL.",
            "РАСПРЕДЕЛЯЮЩАЯ ШЛЯПА заговорила в БОЛЬШОМ ЗАЛЕ.",
            glossary,
        )
        == ()
    )

    # Lowercase tail does not trigger case-sensitive term.
    assert (
        _warnings(
            "They found the sorting hat in the great hall.",
            "Они нашли старую шляпу в огромной комнате.",
            glossary,
        )
        == ()
    )

    # Different leading article does not match `The Sorting Hat`.
    assert (
        _warnings(
            "He saw a Great Hall.",
            "Он увидел огромный зал.",
            glossary,
        )
        == ()
    )

    # Separate uncovered occurrence of `Sorting` still warns.
    separate = _warnings(
        "The Sorting Hat sang before the Sorting.",
        "Распределяющая шляпа спела перед церемонией.",
        glossary,
    )
    assert len(separate) == 1
    assert "'Sorting'" in separate[0]
    assert "'The Sorting Hat'" not in separate[0]

    # When `The Sorting Hat` itself drifts, embedded `Sorting` does not emit a duplicate warning.
    longer_drift = _warnings(
        "He put on the Sorting Hat.",
        "Он надел Волшебный колпак.",
        glossary,
    )
    assert len(longer_drift) == 1
    assert "'The Sorting Hat'" in longer_drift[0]


def test_glossary_validator_preserves_hyphenated_terms_and_lowercase_non_proper_terms() -> None:
    """Hyphenated terms like `You-Know-Who` still match, while lowercase common terms stay case-insensitive."""
    glossary = (
        GlossarySnapshotEntry(
            source_text="You-Know-Who",
            translation="Сам-Знаешь-Кто",
            type=ConceptType.PERSON,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="quidditch",
            translation="квиддич",
            type=ConceptType.FICTIONAL_CONCEPT,
            status=TermStatus.APPROVED,
        ),
        GlossarySnapshotEntry(
            source_text="moon",
            translation="луна",
            type=ConceptType.TERM,
            status=TermStatus.APPROVED,
        ),
    )

    missing_ykw = _warnings("They feared You-Know-Who.", "Они боялись Тёмного Лорда.", glossary)
    assert any("'You-Know-Who'" in w for w in missing_ykw)
    assert _warnings("They feared You-Know-Who.", "Они боялись Сам-Знаешь-Кого.", glossary) == ()
    assert _warnings("They feared YOU-KNOW-WHO.", "Они боялись Сам-Знаешь-Кого.", glossary) == ()
    assert _warnings("An anti-You-Know-Who league formed.", "Сформировалась лига сопротивления.", glossary) == ()

    # Lowercase non-proper term matches any casing in source.
    missing_q = _warnings("Quidditch was his favourite sport.", "Игра на мётлах была его любимым спортом.", glossary)
    assert any("'quidditch'" in w for w in missing_q)
    assert _warnings("Quidditch was his favourite sport.", "Квиддич был его любимым спортом.", glossary) == ()

    # Lowercase common term still respects hyphen boundaries.
    assert _warnings("He wore half-moon glasses.", "Он носил очки-половинки.", glossary) == ()
    missing_moon = _warnings("The full Moon rose.", "Полное светило взошло.", glossary)
    assert any("'moon'" in w for w in missing_moon)
