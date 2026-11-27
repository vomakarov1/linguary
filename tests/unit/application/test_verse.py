"""Tests for verse stanza and strict 2-line couplet detection."""

from __future__ import annotations

from linguary.application.verse import detect_verse_stanza_runs
from linguary.domain import (
    BlockType,
    DocumentBlock,
    InlineAnnotation,
    InlineAnnotationType,
    SupportStatus,
)


def _block(
    order: int,
    text: str,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    italic: bool = False,
) -> DocumentBlock:
    annotations = (
        (InlineAnnotation(InlineAnnotationType.ITALIC, 0, len(text)),) if italic else ()
    )
    return DocumentBlock(
        id=f"sec-1:b{order}",
        section_id="sec-1",
        order=order,
        type=block_type,
        text=text,
        source_reference=f"ch1.xhtml#p{order}",
        support_status=SupportStatus.SUPPORTED,
        inline_annotations=annotations,
    )


def test_detect_verse_stanza_bridges_end_stopped_lines_and_trailing_quoted_stanza() -> None:
    """Sorting Hat style song captures trailing 4-line stanza (#62-#65) with and without internal enjambment."""
    # Variant A: 2 end-stopped lines followed by an enjambed closing pair inside an unclosed quote.
    blocks_bridged = (
        _block(0, "The hat twitched and opened its brim."),
        _block(1, "‘Oh, you may not think I’m pretty,"),
        _block(2, "But don’t judge on what you see,"),
        _block(3, "I’ll eat myself if you can find"),
        _block(4, "A smarter hat than me."),
        _block(5, "So put me on! Don’t be afraid!"),
        _block(6, "And don’t get in a flap!"),
        _block(7, "You’re in safe hands (though I have none)"),
        _block(8, "For I’m a Thinking Cap!’"),
        _block(9, "The whole hall burst into applause."),
    )
    runs_bridged = detect_verse_stanza_runs(blocks_bridged)
    assert len(runs_bridged) == 1
    assert [b.order for b in runs_bridged[0]] == [1, 2, 3, 4, 5, 6, 7, 8]

    # Variant B: all 4 trailing lines end-stopped, closing the quote opened in the stanza core.
    blocks_tail = (
        _block(0, "The hat twitched and opened its brim."),
        _block(1, "‘Oh, you may not think I’m pretty,"),
        _block(2, "But don’t judge on what you see,"),
        _block(3, "I’ll eat myself if you can find"),
        _block(4, "A smarter hat than me."),
        _block(5, "So put me on! Don’t be afraid!"),
        _block(6, "And don’t get in a flap!"),
        _block(7, "You’re in safe hands today."),
        _block(8, "For I’m a Thinking Cap!’"),
        _block(9, "The whole hall burst into applause."),
    )
    runs_tail = detect_verse_stanza_runs(blocks_tail)
    assert len(runs_tail) == 1
    assert [b.order for b in runs_tail[0]] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_detect_verse_tail_stops_at_speech_verb_or_unclosed_over_limit() -> None:
    """Tail extension never swallows prose with speech verbs or lines > 80 chars."""
    blocks_with_speech = (
        _block(0, "‘Enter, stranger, but take heed"),
        _block(1, "Of what awaits the sin of greed,"),
        _block(2, "Must pay most dearly in their turn."),
        _block(3, "Wait a moment, Harry said aloud."),
        _block(4, "Now close the quote!’"),
    )
    runs = detect_verse_stanza_runs(blocks_with_speech)
    assert len(runs) == 1
    assert [b.order for b in runs[0]] == [0, 1, 2]


def test_detect_strict_two_line_quoted_couplet_and_reject_false_positives() -> None:
    """2-line quoted couplets/spells match; letters, italic notes, dash breaks, and dialogue do not."""
    # Positive: single-quoted spell (#215-#216 pattern).
    spell_single = (
        _block(0, "Ron cleared his throat."),
        _block(1, "‘Sunshine, daisies, butter mellow,"),
        _block(2, "Turn this stupid, fat rat yellow.’"),
        _block(3, "He waved his wand, but nothing happened."),
    )
    runs_single = detect_verse_stanza_runs(spell_single)
    assert len(runs_single) == 1
    assert [b.order for b in runs_single[0]] == [1, 2]

    # Positive: double-quoted couplet without trailing comma on line 1.
    spell_double = (
        _block(0, "“One ring to rule them all"),
        _block(1, "And in the darkness bind them.”"),
    )
    runs_double = detect_verse_stanza_runs(spell_double)
    assert len(runs_double) == 1
    assert [b.order for b in runs_double[0]] == [0, 1]

    # Negative 1: Hogwarts letter / unquoted greeting.
    letter = (
        _block(0, "Dear Mr Potter,"),
        _block(1, "We are pleased to inform you."),
    )
    assert detect_verse_stanza_runs(letter) == ()

    # Negative 2: italic 2-line note without quotes.
    italic_note = (
        _block(0, "Please remember the broomstick,", italic=True),
        _block(1, "Leave it beside the shed.", italic=True),
    )
    assert detect_verse_stanza_runs(italic_note) == ()

    # Negative 3: dialogue interrupted by dash.
    dash_dialogue = (
        _block(0, "‘Wait a minute —"),
        _block(1, "Don’t touch that!’"),
    )
    assert detect_verse_stanza_runs(dash_dialogue) == ()

    # Negative 4: dialogue containing speech verb.
    speech_dialogue = (
        _block(0, "‘Sunshine, daisies, butter mellow,"),
        _block(1, "Ron said loudly.’"),
    )
    assert detect_verse_stanza_runs(speech_dialogue) == ()

    # Negative 5: multi-paragraph prose dialogue repeating the opening quote on line 2.
    multi_para_dialogue = (
        _block(0, "‘Well, I was walking down the road,"),
        _block(1, "‘And then I met an old friend.’"),
    )
    assert detect_verse_stanza_runs(multi_para_dialogue) == ()
