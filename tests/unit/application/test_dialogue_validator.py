"""Unit tests for the Russian dialogue punctuation validator (P2.T1)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from processing_fakes import (
    make_block,
    make_block_result,
    make_response,
    make_unit,
)

from linguary.application.llm_dto import BlockResultResponse, SegmentTranslationResponse
from linguary.application.validators.dialogue import dialogue_violation
from linguary.domain import (
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    TranslationUnit,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.treatment import strongest_treatment


def _check(
    translated_text: str,
    *,
    source_text: str = '"Hello," he said.',
    target_language: str = "ru",
    block_type: BlockType = BlockType.PARAGRAPH,
    pass_unit: bool = True,
    strict_v3: bool = False,
) -> str | None:
    block = make_block(order=0, text=source_text, type=block_type)
    unit = make_unit((block,))
    payload = make_response(
        (make_block_result(block, translated_text=translated_text),)
    )
    return dialogue_violation(
        payload,
        target_language=target_language,
        unit=unit if pass_unit else None,
        strict_v3=strict_v3,
    )


def test_dialogue_validator_rejects_hybrids_and_narration_in_quotes_but_allows_thoughts_and_poems() -> None:
    """Core P2.T1 contract: catches dialogue defects while allowing thoughts, poems, prose dashes, and non-ru."""
    # (a) Hybrids — « and «—
    assert _check("— «Привет», — сказал он.") is not None
    assert _check("«— Привет», — сказал он.") is not None

    # (b) Odd / unmatched guillemets « / »
    assert _check("«Привет, Гарри.") is not None
    assert _check("Привет, Гарри».") is not None

    # (c) Author narration / spoken verb inside «…»
    assert _check("«Привет, — сказал он, — как дела?»") is not None
    assert _check("«Иди сюда! — крикнул Хагрид.»") is not None

    # (d) Stray — between sentences of a single unbroken speech without author narration
    assert (
        _check(
            "— Я не знаю. — Может быть, завтра.",
            source_text='"I do not know. Maybe tomorrow."',
        )
        is not None
    )

    # (e) Paragraph-initial «…» with explicit spoken speech verb
    assert _check("«Привет», — сказал он.") is not None
    assert _check("«Куда ты идёшь?» — спросил Гарри.") is not None
    assert _check("«Стой!» — крикнул Хагрид.") is not None
    assert _check("«Тише…» — прошептала Гермиона.") is not None

    # Negative cases that MUST be accepted:
    # Internal thoughts in guillemets
    assert _check("«Какая странная сова», — подумал он.", source_text="'What a strange owl,' he thought.") is None
    assert _check("«Что же делать?» — подумал Гарри.", source_text="'What should I do?' Harry wondered.") is None
    assert _check("«Он прав», — решил Гарри.", source_text="'He is right,' Harry decided.") is None

    # Poems are skipped even if they contain unusual punctuation
    assert (
        _check(
            "— «Шляпа поёт», — сказал он.",
            source_text="Oh, you may not think I am pretty,",
            block_type=BlockType.POEM,
        )
        is None
    )

    # Ordinary explanatory prose dash
    assert _check("Он вошёл — и замер.", source_text="He walked in — and froze.") is None

    # Valid Russian dash dialogue
    assert _check("— Привет, — сказал он.") is None
    assert _check("— Привет, — сказал он. — Как дела?", source_text='"Hello," he said. "How are you?"') is None

    # Non-Russian target language is skipped
    assert _check("— «Hello», — he said.", target_language="de") is None


@pytest.mark.parametrize(
    "translated_text",
    [
        pytest.param("— «Иди сюда!»", id="leading-em-dash-guillemet"),
        pytest.param("– «Иди сюда!»", id="leading-en-dash-guillemet"),
        pytest.param("«— Иди сюда!»", id="guillemet-em-dash"),
        pytest.param("« — Иди сюда!»", id="guillemet-space-em-dash"),
        pytest.param("Он ответил: — «Иди сюда!»", id="colon-dash-guillemet"),
        pytest.param("— Да, — сказал он, — «но завтра».", id="comma-dash-guillemet"),
    ],
)
def test_dialogue_validator_catches_hybrid_variants(translated_text: str) -> None:
    assert _check(translated_text) is not None


def test_dialogue_validator_allows_explanatory_dash_before_quoted_title() -> None:
    """Explanatory prose dashes before or around a quoted title are not dialogue hybrids."""
    assert (
        _check(
            "Он открыл первую книгу — «Историю магии».",
            source_text="He opened the first book — A History of Magic.",
        )
        is None
    )
    assert (
        _check(
            "Он открыл первую книгу — «Историю магии» — и начал читать.",
            source_text="He opened the first book — A History of Magic — and began to read.",
        )
        is None
    )
    assert (
        _check(
            "Когда он открыл книгу — «Историю магии», — наступила тишина.",
            source_text="When he opened the book — A History of Magic — silence fell.",
        )
        is None
    )
    # Genuine mid-sentence hybrid with spoken attribution is still caught
    assert (
        _check(
            "Он шагнул вперёд — «Стой!» — крикнул Хагрид.",
            source_text='He stepped forward — "Stop!" shouted Hagrid.',
        )
        is not None
    )


def test_dialogue_validator_allows_embedded_short_quote_in_narration() -> None:
    """Short inline quote inside narration (not starting the paragraph with «) is valid."""
    assert (
        _check(
            "Он тихо сказал: «Привет».",
            source_text='He quietly said, "Hello."',
        )
        is None
    )


def test_dialogue_validator_allows_unclosed_quote_matching_multi_paragraph_source() -> None:
    """Multi-paragraph direct speech left open in source is not falsely rejected as unmatched."""
    assert (
        _check(
            "«Иди туда. И не оглядывайся.",
            source_text='"Go there. And do not look back.',
        )
        is None
    )
    assert (
        _check(
            "«Иди туда, не оглядывайся и жди у ворот мистера Беннета.",
            source_text="‘Go there, don’t look back, and wait by Mr Bennet’s gate.",
        )
        is None
    )
    # Without unit/source, unclosed quote is still caught
    assert _check("«Иди туда. И не оглядывайся.", pass_unit=False) is not None


def test_dialogue_validator_stray_dash_checks_single_unbroken_speech_without_false_positives() -> None:
    """Stray — between sentences of one speaker is caught, while author attribution and prose dashes pass."""
    assert (
        _check(
            "— Подожди здесь. — Я скоро вернусь.",
            source_text="‘Wait here. I’ll be right back.’",
        )
        is not None
    )
    assert (
        _check(
            "— Подожди здесь! — Я скоро вернусь.",
            source_text='"Wait here! I will be right back."',
        )
        is not None
    )
    # Valid explanatory dash inside a single speech (no sentence break before dash)
    assert (
        _check(
            "— Подожди — что ты делаешь?",
            source_text='"Wait — what are you doing?"',
        )
        is None
    )
    # Valid author attribution between sentences
    assert (
        _check(
            "— Подожди здесь, — сказал он. — Я скоро вернусь.",
            source_text='"Wait here," he said. "I will be right back."',
        )
        is None
    )
    # Two separate speeches in original block
    assert (
        _check(
            "— Привет! — Привет!",
            source_text='"Hi!" "Hi!"',
        )
        is None
    )


def test_dialogue_validator_checks_segmented_translations() -> None:
    """Segmented block translations are joined and checked just like whole-block translations."""
    block = make_block(order=0, text='"Hello," he said. "How are you?"')
    unit = make_unit((block,))
    payload = make_response(
        (
            BlockResultResponse(
                source_block_id=block.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=1, translated_text="— «Привет», — сказал он."),
                    SegmentTranslationResponse(index=2, translated_text="Как дела?"),
                ),
            ),
        )
    )
    assert dialogue_violation(payload, target_language="ru", unit=unit) is not None


def _segmented_unit(
    parts: tuple[str, ...],
    treatments: tuple[BlockTreatment, ...],
) -> tuple[TranslationUnit, DocumentBlock]:
    text = "".join(parts)
    offset = 0
    spans: list[BlockSegment] = []
    for index, part in enumerate(parts):
        spans.append(BlockSegment(offset, offset + len(part), index))
        offset += len(part)
    block = replace(make_block(order=0, text=text), segments=tuple(spans))
    plan = TreatmentPlan(
        section_id=block.section_id,
        items=(
            TreatmentPlanItem(
                block_id=block.id,
                treatment=strongest_treatment(treatments),
                segment_treatments=treatments,
                sentence_indexes=tuple(range(len(treatments))),
            ),
        ),
    )
    return TranslationUnit.of_blocks(id="u-seg", blocks=(block,), plan=plan), block


def test_quotes_helpers_handle_dialect_apostrophes_outer_open_and_direct_speech() -> None:
    """Shared British quote helpers parse adjacent/final dialect apostrophes, outer open quote, and direct speech."""
    from linguary.application.validators.quotes import (
        _outer_unclosed_open,
        british_quote_counts,
        has_direct_speech_quotes,
        has_unclosed_british_quote,
    )

    unclosed_dialect = "‘An’ he was killin’ ’em, an’ I couldn’ stop ’im."
    assert has_unclosed_british_quote(unclosed_dialect) is True
    assert has_unclosed_british_quote("‘An’ he was killin’ ’em.’") is False
    assert (
        has_unclosed_british_quote(
            "‘An’ ’em, o’ course, don’ know las’ night ol’ Hagrid jus’ mus’ be tellin’ ’em."
        )
        is True
    )
    assert has_unclosed_british_quote("‘Look at the boys’ coats in the hall.") is True
    assert has_unclosed_british_quote("‘Go there’, he said.") is False

    # Inline single-quoted words/nested quotes before lowercase continuation must still close cleanly.
    inline_word = "The word ‘magic’ was forbidden."
    assert british_quote_counts(inline_word) == (1, 1)
    assert has_unclosed_british_quote(inline_word) is False
    assert _outer_unclosed_open(inline_word, len(inline_word)) is None

    nested_inline = "‘It’s called ‘quidditch’ and you fly,’ said Wood."
    assert british_quote_counts(nested_inline) == (2, 2)
    assert has_unclosed_british_quote(nested_inline) is False
    assert _outer_unclosed_open(nested_inline, len(nested_inline)) is None

    # Capitalized proper names ending in `-in` are not treated as dialect `-in’` stems.
    for proper_name in ("Martin", "Kevin", "Justin", "Colin", "Austin", "Darwin"):
        proper_quote = f"‘Ask {proper_name}’, she said."
        assert british_quote_counts(proper_quote) == (1, 1)
        assert has_unclosed_british_quote(proper_quote) is False
        assert _outer_unclosed_open(proper_quote, len(proper_quote)) is None

    # Known elided auxiliaries remain apostrophes in unclosed quotes.
    unclosed_aux = "‘I couldn’ believe he wouldn’ go, an’ didn’ say why."
    assert british_quote_counts(unclosed_aux) == (1, 0)
    assert has_unclosed_british_quote(unclosed_aux) is True

    monologue = "‘An’ he was killin’ ’em. Tell Harry.’"
    assert _outer_unclosed_open(monologue, 0) is None
    mid = len("‘An’ he was killin’ ’em.")
    assert _outer_unclosed_open(monologue, mid) is not None
    assert _outer_unclosed_open(monologue, mid) == _outer_unclosed_open(monologue, mid + 1)
    assert _outer_unclosed_open(monologue, len(monologue)) is None

    # Split inside `killin’ ’em` must still see ` ’em` ahead and not close at `killin’`.
    before_em = len("‘An’ he was killin’")
    assert _outer_unclosed_open(monologue, before_em) == _outer_unclosed_open(monologue, mid)

    # Nested quote preserves the outer opening mark/offset.
    nested = '‘He told me, “Wait here. Don’t go.” And left.’'
    inside_inner = len('‘He told me, “Wait here.')
    after_inner = len('‘He told me, “Wait here. Don’t go.”')
    assert _outer_unclosed_open(nested, inside_inner) == _outer_unclosed_open(nested, after_inner)

    # Different speeches produce distinct open markers.
    two_speeches = "‘First part one. First part two.’ ‘Second part one. Second part two.’"
    end_first = len("‘First part one.")
    start_second = len("‘First part one. First part two.’ ‘Second part one.")
    open_first = _outer_unclosed_open(two_speeches, end_first)
    open_second = _outer_unclosed_open(two_speeches, start_second)
    assert open_first is not None and open_second is not None and open_first != open_second

    # Direct speech detection: quotes vs bare apostrophes.
    assert has_direct_speech_quotes("‘Hello.’") is True
    assert has_direct_speech_quotes("“Hello.”") is True
    assert has_direct_speech_quotes('"Hello."') is True
    assert has_direct_speech_quotes("'Hello,' he said.") is True
    assert has_direct_speech_quotes("Harry's owl flew away.") is False
    assert has_direct_speech_quotes("Harry’s owl flew away.") is False
    assert has_direct_speech_quotes("The boys’ coats were wet.") is False
    assert has_direct_speech_quotes("Tell ’em it's fine.") is False
    assert has_direct_speech_quotes("'Tis Harry's owl.") is False


def test_dialogue_validator_handles_british_dialect_apostrophes_and_unclosed_monologues() -> None:
    """British dialect apostrophes (`killin’ ’em`, `An’ ’em`) work in unclosed and closed quotes."""
    unclosed = "‘An’ he was killin’ ’em, an’ I couldn’ stop ’im."
    assert _check("«Ну вот, он их убивал, а я не мог его остановить.", source_text=unclosed) is None
    assert _check("— Ну вот, он их убивал. — Я не мог его остановить.", source_text=unclosed) is not None

    closed = "‘An’ he was killin’ ’em. Tell Harry.’"
    assert _check("«Ну вот, он их убивал. Скажи Гарри.", source_text=closed) is not None
    assert _check("— Ну вот, он их убивал. — Скажи Гарри.", source_text=closed) is not None


def test_dialogue_validator_strict_v3_rejects_leading_dash_in_segmented_monologue() -> None:
    """Under strict_v3=True, consecutive translated segments inside one unclosed quote cannot repeat leading dash."""
    # [T, T] with preceding READ_AS_IS narration (`[R, T, T]`): Rule (d) does not cover it, strict_v3 does.
    parts_rtt = ("He smiled. ", "‘I’m tellin’ yeh, Harry.", " An’ a thumpin’ good’un.’")
    unit_rtt, block_rtt = _segmented_unit(
        parts_rtt, (BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE)
    )
    bad_rtt = make_response(
        (
            BlockResultResponse(
                source_block_id=block_rtt.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=2, translated_text="— Говорю тебе, Гарри."),
                    SegmentTranslationResponse(index=3, translated_text="— И чертовски хороший."),
                ),
            ),
        )
    )
    assert dialogue_violation(bad_rtt, target_language="ru", unit=unit_rtt, strict_v3=False) is None
    assert dialogue_violation(bad_rtt, target_language="ru", unit=unit_rtt, strict_v3=True) is not None

    # En-dash `–` in non-first segment is also rejected, even when segments are unsorted.
    bad_en_dash = make_response(
        (
            BlockResultResponse(
                source_block_id=block_rtt.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=3, translated_text=" – И чертовски хороший."),
                    SegmentTranslationResponse(index=2, translated_text="— Говорю тебе, Гарри."),
                ),
            ),
        )
    )
    assert dialogue_violation(bad_en_dash, target_language="ru", unit=unit_rtt, strict_v3=True) is not None

    # Plain continuation without leading dash passes.
    good_rtt = make_response(
        (
            BlockResultResponse(
                source_block_id=block_rtt.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=2, translated_text="— Говорю тебе, Гарри."),
                    SegmentTranslationResponse(index=3, translated_text="И чертовски хороший."),
                ),
            ),
        )
    )
    assert dialogue_violation(good_rtt, target_language="ru", unit=unit_rtt, strict_v3=True) is None

    # [T, G, T] across a GLOSSARY_ONLY middle segment in the same unclosed quote.
    parts_tgt = ("‘First,’ he said, ‘second part.", " Easy middle.", " Third part.’")
    unit_tgt, block_tgt = _segmented_unit(
        parts_tgt, (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE)
    )
    bad_tgt = make_response(
        (
            BlockResultResponse(
                source_block_id=block_tgt.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=1, translated_text="— Первая, — сказал он, вторая часть."),
                    SegmentTranslationResponse(index=3, translated_text="— Третья часть."),
                ),
            ),
        )
    )
    assert dialogue_violation(bad_tgt, target_language="ru", unit=unit_tgt, strict_v3=False) is None
    assert dialogue_violation(bad_tgt, target_language="ru", unit=unit_tgt, strict_v3=True) is not None

    # [T, G, T] across two DIFFERENT quotes is allowed to start the second speech with `— `.
    parts_diff = ("‘First part one.", " First part two.’ ‘Second part one.", " Second part two.’")
    unit_diff, block_diff = _segmented_unit(
        parts_diff, (BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE)
    )
    ok_diff = make_response(
        (
            BlockResultResponse(
                source_block_id=block_diff.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=1, translated_text="— Первая часть."),
                    SegmentTranslationResponse(index=3, translated_text="— Вторая часть."),
                ),
            ),
        )
    )
    assert dialogue_violation(ok_diff, target_language="ru", unit=unit_diff, strict_v3=True) is None

    # Blocks containing `\n` in source_text are not falsely rejected by the segmented check.
    parts_nl = ("He smiled.\n‘First line.\n", "Second line.’")
    unit_nl, block_nl = _segmented_unit(
        parts_nl, (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE)
    )
    ok_nl = make_response(
        (
            BlockResultResponse(
                source_block_id=block_nl.id,
                treatment=BlockTreatment.TRANSLATE,
                translated_text=None,
                segments=(
                    SegmentTranslationResponse(index=1, translated_text="Он улыбнулся.\n— Первая строка."),
                    SegmentTranslationResponse(index=2, translated_text="— Вторая строка."),
                ),
            ),
        )
    )
    assert dialogue_violation(ok_nl, target_language="ru", unit=unit_nl, strict_v3=True) is None


@pytest.mark.parametrize(
    ("source_text", "translated_text"),
    [
        pytest.param('Harry walked in. "Hello!"', "Гарри вошёл. — Привет!", id="period-double-quote"),
        pytest.param("Harry walked in. ‘Hello!’", "Гарри вошёл. – Привет!", id="en-dash-british"),
        pytest.param("He blinked… “Wait!”", "Он моргнул… — Подожди!", id="ellipsis-curly-quote"),
        pytest.param("What? 'Stop!'", "Что? — Стой!", id="question-matched-single"),
        pytest.param('It was me. "Hello!"', "Это была я. — Привет!", id="lowercase-single-letter-before-dot"),
        pytest.param('Mr. D. smiled. "Hello!"', "Мистер Д. — Гарри знал его. — Привет!", id="second-match-after-abbrev"),
    ],
)
def test_dialogue_validator_strict_v3_catches_non_leading_inline_dash_with_source_quotes(
    source_text: str, translated_text: str
) -> None:
    """Under strict_v3=True, non-leading inline dialogue dash fires when source has direct speech quotes."""
    assert _check(translated_text, source_text=source_text, strict_v3=False) is None
    assert _check(translated_text, source_text=source_text, strict_v3=True) is not None


@pytest.mark.parametrize(
    ("source_text", "translated_text"),
    [
        pytest.param("Harry's owl blinked. What happened?", "Сова Гарри моргнула. — Что случилось?", id="bare-straight-apostrophe"),
        pytest.param("Harry’s owl blinked. What happened?", "Сова Гарри моргнула. — Что случилось?", id="bare-curly-apostrophe"),
        pytest.param("The boys’ coats fell. What happened?", "Пальто мальчиков упали. — Что случилось?", id="bare-plural-apostrophe"),
        pytest.param("'Tis Harry's owl. What happened?", "Это сова Гарри. — Что случилось?", id="elision-and-apostrophe"),
        pytest.param('Harry walked in. "Hello!"', "Гарри вошёл.\n— Привет!", id="newline-before-dash"),
        pytest.param('Mr. Dursley smiled. "Hi!"', "Мистер Mr. — Гарри вздрогнул.", id="abbrev-mr"),
        pytest.param('Mrs. Dursley smiled. "Hi!"', "Mrs. — Гарри вздрогнул.", id="abbrev-mrs"),
        pytest.param('Ms. Dursley smiled. "Hi!"', "Ms. — Гарри вздрогнул.", id="abbrev-ms"),
        pytest.param('Dr. Brown smiled. "Hi!"', "Dr. — Гарри вздрогнул.", id="abbrev-dr"),
        pytest.param('St. Jude was near. "Hi!"', "St. — Гарри вздрогнул.", id="abbrev-st"),
        pytest.param('Prof. McGonagall spoke. "Hi!"', "Prof. — Гарри вздрогнул.", id="abbrev-prof"),
        pytest.param('Etc. "Hi!"', "И т. д. — Гарри вздрогнул.", id="abbrev-td"),
        pytest.param('Etc. "Hi!"', "И т. п. — Гарри вздрогнул.", id="abbrev-tp"),
        pytest.param('That is. "Hi!"', "Т. е. — Гарри вздрогнул.", id="abbrev-te"),
        pytest.param('AD. "Hi!"', "До н. э. — Гарри вздрогнул.", id="abbrev-ne"),
        pytest.param('Year 1991. "Hi!"', "В 1991 г. — Гарри вздрогнул.", id="abbrev-g"),
        pytest.param('Saint. "Hi!"', "Св. — Георгий победил.", id="abbrev-sv"),
        pytest.param('Others. "Hi!"', "И др. — Гарри вздрогнул.", id="abbrev-dr-ru"),
        pytest.param('Example. "Hi!"', "См. пр. — Гарри вздрогнул.", id="abbrev-pr"),
        pytest.param('Page. "Hi!"', "На стр. — Гарри вздрогнул.", id="abbrev-str"),
        pytest.param('Chapter. "Hi!"', "В гл. — Гарри вздрогнул.", id="abbrev-gl"),
        pytest.param('Mr. J. K. spoke. "Hi!"', "Дж. К. — Гарри вздрогнул.", id="single-capital-initial-ru"),
        pytest.param('J. K. spoke. "Hi!"', "J. — Гарри вздрогнул.", id="single-capital-initial-en"),
    ],
)
def test_dialogue_validator_strict_v3_allows_non_quote_prose_and_abbreviations(
    source_text: str, translated_text: str
) -> None:
    """Under strict_v3=True, bare apostrophes, leading-dash lines, and English/Russian abbreviations pass."""
    assert _check(translated_text, source_text=source_text, strict_v3=True) is None


def test_dialogue_validator_strict_v3_checks_inline_dash_per_segment_without_joining_across_gaps() -> None:
    """Segmented [T, G, T] / [T, R, T] narration followed by a dialogue segment is not falsely rejected."""
    parts = ("Harry walked in.", " The giant smiled.", ' "Hello!" he said.')
    for middle_mode in (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.READ_AS_IS):
        unit, block = _segmented_unit(
            parts, (BlockTreatment.TRANSLATE, middle_mode, BlockTreatment.TRANSLATE)
        )
        ok_payload = make_response(
            (
                BlockResultResponse(
                    source_block_id=block.id,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text=None,
                    segments=(
                        SegmentTranslationResponse(index=1, translated_text="Он вошёл."),
                        SegmentTranslationResponse(index=3, translated_text="— Привет! — сказал он."),
                    ),
                ),
            )
        )
        assert dialogue_violation(ok_payload, target_language="ru", unit=unit, strict_v3=True) is None

        bad_payload = make_response(
            (
                BlockResultResponse(
                    source_block_id=block.id,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text=None,
                    segments=(
                        SegmentTranslationResponse(index=1, translated_text="Он вошёл."),
                        SegmentTranslationResponse(index=3, translated_text="Он улыбнулся. — Привет!"),
                    ),
                ),
            )
        )
        assert dialogue_violation(bad_payload, target_language="ru", unit=unit, strict_v3=True) is not None


def test_dialogue_validator_allows_embedded_quote_in_narrative_and_rejects_line_initial_guillemet() -> None:
    """WP4: embedded `«…»` with author speech/dash inside mixed narrative passes strict_v3, line-initial fails."""
    b35_source = (
        "“We were on a mountain road…” Only this wasn’t a mountain road, "
        "and the trees weren’t even gray. “A dead forest,” Dad had said, pointing. “It happens.”"
    )
    b35_single_leading_dash = (
        "— Мы ехали по горной дороге… — сказала она. "
        "Но это была не горная дорога, и деревья даже не были серыми. "
        "«Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»"
    )
    b35_single_narrative_line = (
        "«Мы ехали по горной дороге…» "
        "Но это была не горная дорога, и деревья даже не были серыми. "
        "«Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»"
    )
    b35_two_line = (
        "— Мы ехали по горной дороге…\n"
        "Но это была не горная дорога, и деревья даже не были серыми. "
        "«Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»"
    )

    assert _check(b35_single_leading_dash, source_text=b35_source, strict_v3=True) is None
    assert _check(b35_single_narrative_line, source_text=b35_source, strict_v3=True) is None
    assert _check(b35_two_line, source_text=b35_source, strict_v3=True) is None

    # British single quotes with contractions in source also work universally.
    b35_british_source = (
        "‘We were on a mountain road…’ Only this wasn’t a mountain road, "
        "and the trees weren’t even gray. ‘A dead forest,’ Dad had said, pointing. ‘It happens.’"
    )
    assert _check(b35_two_line, source_text=b35_british_source, strict_v3=True) is None

    # Line-initial spoken dialogue inside `«…»` is still rejected even with multi-quote source.
    assert _check("«Привет, — сказал он.»", source_text=b35_source, strict_v3=True) is not None
    assert _check("   «Привет, — сказал он.»", source_text=b35_source, strict_v3=True) is not None
    assert (
        _check(
            "— Мы ехали по горной дороге…\n«Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»",
            source_text=b35_source,
            strict_v3=True,
        )
        is not None
    )

    # Without >= 2 author-separated quotes in source (or without source), embedded author speech is rejected.
    assert (
        _check(
            "Он вошёл. «Привет, — сказал он.»",
            source_text='He walked in. "Hello," he said.',
            strict_v3=True,
        )
        is not None
    )
    assert (
        _check(
            "Он вспомнил. «Привет, — сказал он. — Пока!»",
            source_text='"Hi!" "Bye!"',
            strict_v3=True,
        )
        is not None
    )
    assert _check(b35_single_leading_dash, pass_unit=False, strict_v3=True) is not None

    # Inline dialogue dash outside the embedded `«…»` span is still rejected under strict_v3.
    assert (
        _check(
            "Но это была не горная дорога. — Так бывает.",
            source_text=b35_source,
            strict_v3=True,
        )
        is not None
    )
    assert (
        _check(
            "Но это была не горная дорога. — Пошли! «Мёртвый лес, — сказал папа, показывая рукой. — Так бывает.»",
            source_text=b35_source,
            strict_v3=True,
        )
        is not None
    )


def test_dialogue_validator_allows_ollivander_1_line_and_3_line_and_rejects_2_line_inline_dash() -> None:
    """WP4: speech + post-speech authorial sentence + speech passes in 1-line and 3-line forms under strict_v3."""
    ollivander_source = (
        "‘Well, now – Mr Potter. Let me see.’ "
        "He pulled a long tape measure with silver markings out of his pocket. "
        "‘Which is your wand arm?’"
    )
    one_line = (
        "— Что ж, мистер Поттер. Посмотрим. "
        "— Он вынул из кармана длинную рулетку с серебряными делениями. "
        "— Какой рукой вы держите палочку?"
    )
    three_line = (
        "— Что ж, мистер Поттер. Посмотрим.\n"
        "Он вынул из кармана длинную рулетку с серебряными делениями.\n"
        "— Какой рукой вы держите палочку?"
    )
    two_line_bad = (
        "— Что ж, мистер Поттер. Посмотрим.\n"
        "Он вынул из кармана длинную рулетку. — Какой рукой вы держите палочку?"
    )

    assert _check(one_line, source_text=ollivander_source, strict_v3=True) is None
    assert _check(three_line, source_text=ollivander_source, strict_v3=True) is None
    assert _check(two_line_bad, source_text=ollivander_source, strict_v3=True) is not None

