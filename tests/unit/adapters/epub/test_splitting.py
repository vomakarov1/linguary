"""Tests for sentence tokenization and inline markup projection."""

from __future__ import annotations

import pytest

from linguary.adapters.epub.inline import InlineContent
from linguary.adapters.epub.splitting import (
    SEGMENT_MAX_CHARS,
    SEGMENT_MIN_BLOCK_CHARS,
    SPLIT_MAX_PART_CHARS,
    SPLIT_MIN_CHARS,
    SPLIT_MIN_SENTENCES,
    segment_text,
    split_inline_content,
    split_paragraph,
    split_sentences,
)
from linguary.domain import BlockSegment, InlineAnnotation, InlineAnnotationType


def test_split_sentences_skips_short_paragraphs() -> None:
    """Short paragraphs (below min_chars) are not split."""
    short_text = "First sentence. Second sentence. Third sentence."
    assert len(short_text) < SPLIT_MIN_CHARS
    slices = split_sentences(short_text)
    assert slices == ((0, len(short_text)),)

    # Long text consisting of a single monolithic sentence without boundaries
    monolithic_text = "Word " * 200
    assert len(monolithic_text) >= SPLIT_MIN_CHARS
    slices_monolithic = split_sentences(monolithic_text)
    assert slices_monolithic == ((0, len(monolithic_text)),)


def test_split_sentences_splits_at_600_chars_threshold() -> None:
    """Paragraphs between 600 and 800 characters with 2+ sentences are split."""
    s1 = (
        "This is a very long descriptive first sentence that goes on and on with many words and clauses, "
        "describing the beautiful garden and all the creatures living inside it in great detail so that "
        "it has enough length to test the threshold cleanly, including flowers, trees, paths, fountains, "
        "and singing birds in the summer afternoon sun. "
    )
    s2 = (
        "This is an equally detailed second sentence that continues the narrative further, providing "
        "additional context and concluding the whole compound paragraph structure properly with plenty "
        "of adjectives and clauses to ensure the overall length comfortably exceeds the minimum limit."
    )
    text = s1 + s2
    assert 600 <= len(text) < 800
    slices = split_sentences(text)
    assert len(slices) == 2


def test_split_sentences_preserves_titles_and_abbreviations() -> None:
    """Abbreviations and titles (Mr., Dr., e.g., vol.) do not break a sentence."""
    text = (
        "Mr. Smith visited Dr. Watson in London to discuss Prof. Moriarty and his strange plans. "
        "Dr. Watson suggested they consult Mrs. Hudson before making any final decisions. "
        "The evidence included several documents, e.g. letters and diaries found in the study. "
        "All materials were cataloged according to vol. 1 and no. 5 of the society records. "
        "Finally, they decided to proceed with the investigation tomorrow morning."
    )
    # Run with min_chars=0 to test tokenization logic on the fragment
    slices = split_sentences(text, min_chars=0)
    assert len(slices) == 5

    fragments = [text[start:end] for start, end in slices]
    assert fragments[0].startswith("Mr. Smith")
    assert fragments[0].endswith("strange plans. ")
    assert fragments[1].startswith("Dr. Watson")
    assert fragments[1].endswith("final decisions. ")
    assert fragments[2].startswith("The evidence")
    assert fragments[2].endswith("in the study. ")
    assert fragments[3].startswith("All materials")
    assert fragments[3].endswith("society records. ")
    assert fragments[4].startswith("Finally,")


def test_split_sentences_handles_initials() -> None:
    """Initials (H. G. Wells, J. K. Rowling) are not treated as sentence boundaries."""
    text = (
        "H. G. Wells wrote fantastic novels about time travel and Martian invaders. "
        "Meanwhile, C. S. Lewis and J. R. R. Tolkien discussed philology in Oxford. "
        "Their friend A. Smith joined them later in the afternoon."
    )
    slices = split_sentences(text, min_chars=0)
    assert len(slices) == 3

    fragments = [text[start:end] for start, end in slices]
    assert fragments[0] == "H. G. Wells wrote fantastic novels about time travel and Martian invaders. "
    assert fragments[1] == "Meanwhile, C. S. Lewis and J. R. R. Tolkien discussed philology in Oxford. "
    assert fragments[2] == "Their friend A. Smith joined them later in the afternoon."


def test_split_sentences_handles_dialogue_quotes() -> None:
    """Dialogue lines in quotes are split taking speech tags and direct speech into account."""
    # Speech line with lowercase attribution: boundary after author words
    text1 = '"I see," said Alice. "What do you mean?" asked the Hatter. "Nothing!" she replied.'
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 3
    assert text1[slices1[0][0]:slices1[0][1]] == '"I see," said Alice. '
    assert text1[slices1[1][0]:slices1[1][1]] == '"What do you mean?" asked the Hatter. '
    assert text1[slices1[2][0]:slices1[2][1]] == '"Nothing!" she replied.'

    # Speech line with exclamation and proper noun attribution
    text2 = '"Stop!" Alice said. "Never!" the Queen shouted.'
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 2
    assert text2[slices2[0][0]:slices2[0][1]] == '"Stop!" Alice said. '
    assert text2[slices2[1][0]:slices2[1][1]] == '"Never!" the Queen shouted.'

    # Speech line with action (not speech attribution): split occurs immediately after quotes
    text3 = '"Stop!" The King stood up. "You must not do that."'
    slices3 = split_sentences(text3, min_chars=0)
    assert len(slices3) == 3
    assert text3[slices3[0][0]:slices3[0][1]] == '"Stop!" '
    assert text3[slices3[1][0]:slices3[1][1]] == "The King stood up. "
    assert text3[slices3[2][0]:slices3[2][1]] == '"You must not do that."'

    # Typographic quotes (guillemets and curly quotes)
    text4 = '“I see,” said Alice. “What do you mean?”'
    slices4 = split_sentences(text4, min_chars=0)
    assert len(slices4) == 2
    assert text4[slices4[0][0]:slices4[0][1]] == '“I see,” said Alice. '
    assert text4[slices4[1][0]:slices4[1][1]] == '“What do you mean?”'


def test_split_sentences_handles_em_dash_dialogues() -> None:
    """Dash dialogues are split by lines without breaking author words."""
    text1 = "— I see, — said Alice. — What do you mean? — Nothing! — she replied."
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 3
    assert text1[slices1[0][0]:slices1[0][1]] == "— I see, — said Alice. "
    assert text1[slices1[1][0]:slices1[1][1]] == "— What do you mean? "
    assert text1[slices1[2][0]:slices1[2][1]] == "— Nothing! — she replied."

    text2 = "Alice turned around. — What are you doing here? — she asked."
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 2
    assert text2[slices2[0][0]:slices2[0][1]] == "Alice turned around. "
    assert text2[slices2[1][0]:slices2[1][1]] == "— What are you doing here? — she asked."


def test_split_sentences_handles_ellipsis() -> None:
    """Ellipses do not break sentence midpoints and author words."""
    # Ellipsis before lowercase speech attribution
    text1 = '"Wait..." said Alice. "I am coming."'
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 2
    assert text1[slices1[0][0]:slices1[0][1]] == '"Wait..." said Alice. '
    assert text1[slices1[1][0]:slices1[1][1]] == '"I am coming."'

    # Ellipsis before a new sentence starting with uppercase
    text2 = "She hesitated... Then she turned and ran quickly down the hall."
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 2
    assert text2[slices2[0][0]:slices2[0][1]] == "She hesitated... "
    assert text2[slices2[1][0]:slices2[1][1]] == "Then she turned and ran quickly down the hall."

    # Ellipsis in sentence midpoint (pause)
    text3 = "She was... well, she was rather confused by the Hatter."
    slices3 = split_sentences(text3, min_chars=0)
    assert len(slices3) == 1


def test_split_sentences_text_reconstruction_invariant() -> None:
    """Concatenation of slices strictly reconstructs the original paragraph text."""
    samples = [
        "Short paragraph.",
        "Mr. Smith and Dr. Watson met. They talked about books.",
        '"Hello!" said Alice. "Goodbye!" replied the Hatter.',
        "— One. — Two. — Three.",
        "Wait... What is this? Nothing... really nothing.",
        (
            "Down, down, down. There was nothing else to do, so Alice soon began talking "
            "again. “Dinah’ll miss me very much to-night, I should think!” (Dinah was the "
            "cat.) “I hope they’ll remember her saucer of milk at tea-time. Dinah my dear! I "
            "wish you were down here with me! There are no mice in the air, I’m afraid, but "
            "you might catch a bat, and that’s very like a mouse, you know. But do cats eat "
            "bats, I wonder?” And here Alice began to get rather sleepy, and went on saying "
            "to herself, in a dreamy sort of way, “Do cats eat bats? Do cats eat bats?” and "
            "sometimes, “Do bats eat cats?” for, you see, as she couldn’t answer either "
            "question, it didn’t much matter which way she put it."
        ),
    ]

    for sample in samples:
        slices = split_sentences(sample, min_chars=0)
        reconstructed = "".join(sample[start:end] for start, end in slices)
        assert reconstructed == sample

        # Verify slice continuity
        assert slices[0][0] == 0
        assert slices[-1][1] == len(sample)
        for i in range(len(slices) - 1):
            assert slices[i][1] == slices[i + 1][0]


def test_split_inline_content_shifts_annotation_offsets() -> None:
    """Annotation offsets are correctly shifted using the [start - offset, end - offset) formula."""
    text = "First sentence is simple. Second sentence contains bold text inside it."
    bold_target = "bold text"
    bold_start = text.index(bold_target)
    bold_end = bold_start + len(bold_target)

    content = InlineContent(
        text=text,
        annotations=(
            InlineAnnotation(type=InlineAnnotationType.BOLD, start=bold_start, end=bold_end),
        ),
        anchor_ids=("p-1",),
    )

    fragments = split_inline_content(content, min_chars=1)
    assert len(fragments) == 2

    # First fragment contains no markup
    assert fragments[0].annotations == ()

    # Second fragment contains shifted markup
    assert len(fragments[1].annotations) == 1
    shifted_ann = fragments[1].annotations[0]
    assert shifted_ann.type is InlineAnnotationType.BOLD
    offset = len(fragments[0].text)
    assert shifted_ann.start == bold_start - offset
    assert shifted_ann.end == bold_end - offset
    assert fragments[1].text[shifted_ann.start : shifted_ann.end] == bold_target


def test_split_inline_content_clips_spanning_annotations() -> None:
    """Spanning markup (e.g. italics over multiple sentences) is clipped at boundaries."""
    text = "First sentence is here. Second sentence follows."
    # Italic spans part of the first sentence and part of the second
    italic_start = text.index("sentence is here")
    italic_end = text.index("follows") + len("follows")

    content = InlineContent(
        text=text,
        annotations=(
            InlineAnnotation(type=InlineAnnotationType.ITALIC, start=italic_start, end=italic_end),
        ),
    )

    fragments = split_inline_content(content, min_chars=1)
    assert len(fragments) == 2

    # In the first fragment, markup is clipped at the fragment boundary
    assert len(fragments[0].annotations) == 1
    ann0 = fragments[0].annotations[0]
    assert ann0.type is InlineAnnotationType.ITALIC
    assert ann0.start == italic_start
    assert ann0.end == len(fragments[0].text)

    # In the second fragment, markup starts at 0 and ends at the relative offset
    assert len(fragments[1].annotations) == 1
    ann1 = fragments[1].annotations[0]
    assert ann1.type is InlineAnnotationType.ITALIC
    assert ann1.start == 0
    offset = len(fragments[0].text)
    assert ann1.end == italic_end - offset


def test_split_inline_content_preserves_footnote_references() -> None:
    """Footnote references are not bisected and are preserved in their respective fragments."""
    text = "First sentence with note.1 Second sentence with note.2"
    fn1_start = text.index("1")
    fn1_end = fn1_start + 1
    fn2_start = text.rindex("2")
    fn2_end = fn2_start + 1

    content = InlineContent(
        text=text,
        annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=fn1_start,
                end=fn1_end,
                target="notes.xhtml#fn1",
            ),
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=fn2_start,
                end=fn2_end,
                target="notes.xhtml#fn2",
            ),
        ),
    )

    fragments = split_inline_content(content, min_chars=1)
    assert len(fragments) == 2

    # First fragment preserves fn1
    assert len(fragments[0].annotations) == 1
    assert fragments[0].annotations[0] == InlineAnnotation(
        type=InlineAnnotationType.FOOTNOTE_REFERENCE,
        start=fn1_start,
        end=fn1_end,
        target="notes.xhtml#fn1",
    )

    # Second fragment preserves fn2 with shift
    assert len(fragments[1].annotations) == 1
    offset = len(fragments[0].text)
    assert fragments[1].annotations[0] == InlineAnnotation(
        type=InlineAnnotationType.FOOTNOTE_REFERENCE,
        start=fn2_start - offset,
        end=fn2_end - offset,
        target="notes.xhtml#fn2",
    )


def test_split_inline_content_anchors_assigned_to_first_fragment() -> None:
    """Block anchors are assigned only to the first fragment; subsequent fragments receive empty tuples."""
    content = InlineContent(
        text="First sentence here. Second sentence here. Third sentence here.",
        anchor_ids=("anchor-1", "anchor-2"),
    )

    fragments = split_inline_content(content, min_chars=1)
    assert len(fragments) == 2
    assert "".join(fragment.text for fragment in fragments) == content.text
    assert fragments[0].anchor_ids == ("anchor-1", "anchor-2")
    assert fragments[1].anchor_ids == ()


def test_split_inline_content_returns_original_if_short_and_splits_long_text() -> None:
    """Short content is preserved as-is, while long text falls back to word boundaries."""
    short_content = InlineContent(
        text="Short text.",
        anchor_ids=("a1",),
    )
    result_short = split_inline_content(short_content)
    assert result_short == (short_content,)

    monolithic_content = InlineContent(
        text="Monolithic word " * 100,
        anchor_ids=("a2",),
    )
    result_monolithic = split_inline_content(monolithic_content)
    assert len(result_monolithic) > 1
    assert "".join(fragment.text for fragment in result_monolithic) == monolithic_content.text
    assert result_monolithic[0].anchor_ids == monolithic_content.anchor_ids
    assert all(not fragment.anchor_ids for fragment in result_monolithic[1:])


def test_split_sentences_preserves_country_abbreviations() -> None:
    """Abbreviations U.S., U.K., E.U. are not treated as end-of-sentence before a capital letter."""
    text = (
        "The U.S. Army was deployed across multiple operational theaters. "
        "Meanwhile, the U.K. Government issued an official diplomatic statement. "
        "Officials from the E.U. Parliament confirmed the adoption of the treaty. "
        "Finally, all parties agreed to continue bilateral discussions."
    )
    slices = split_sentences(text, min_chars=0)
    assert len(slices) == 4

    fragments = [text[start:end] for start, end in slices]
    assert fragments[0].startswith("The U.S. Army")
    assert fragments[0].endswith("operational theaters. ")
    assert fragments[1].startswith("Meanwhile, the U.K. Government")
    assert fragments[1].endswith("diplomatic statement. ")
    assert fragments[2].startswith("Officials from the E.U. Parliament")
    assert fragments[2].endswith("adoption of the treaty. ")
    assert fragments[3].startswith("Finally, all parties")


def test_split_inline_content_monotonic_boundaries_with_overlapping_footnotes() -> None:
    """Shifting footnote boundary does not violate monotonic boundaries when a footnote overlaps the next boundary."""
    s1 = "Sentence one is right here. "
    s2 = "Sentence two is here. "
    s3 = "Sentence three follows."
    text = s1 + s2 + s3
    b1 = len(s1)
    b2 = len(s1) + len(s2)

    fn_start = b1 - 3
    fn_end = b2 + 2

    content = InlineContent(
        text=text,
        annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.FOOTNOTE_REFERENCE,
                start=fn_start,
                end=fn_end,
                target="notes.xhtml#fn_wide",
            ),
        ),
    )

    fragments = split_inline_content(content, min_chars=1)
    assert len(fragments) >= 1
    reconstructed = "".join(f.text for f in fragments)
    assert reconstructed == text
    for f in fragments:
        assert len(f.text) > 0

    fn_to_end = InlineAnnotation(
        type=InlineAnnotationType.FOOTNOTE_REFERENCE,
        start=b1 - 2,
        end=len(text),
        target="notes.xhtml#fn_end",
    )
    content_end = InlineContent(text=text, annotations=(fn_to_end,))
    fragments_end = split_inline_content(content_end, min_chars=1)
    assert len(fragments_end) == 1
    assert fragments_end[0].text == text


def test_split_sentences_preserves_compound_numbered_markers() -> None:
    """Compound numbered section markers (1.C., 1.F.2., 1.F.3.) are not cut into micro-fragments."""
    text1 = (
        "1.C. The Project Gutenberg Literary Archive Foundation owns a compilation copyright. "
        "Except for the limited right of replacement or refund, all other warranties are disclaimed. "
        "General terms of use are subject to federal and state laws."
    )
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 3
    frags1 = [text1[start:end] for start, end in slices1]
    assert frags1[0].startswith("1.C. The Project Gutenberg")
    assert frags1[0].endswith("compilation copyright. ")

    text2 = (
        "1.F.2. LIMITED WARRANTY, DISCLAIMER OF DAMAGES - Except for the right of replacement. "
        "If you discover a defect in this electronic work within 90 days, you are entitled to refund. "
        "Neither the Foundation nor any other party distributes copies without compliance."
    )
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 3
    frags2 = [text2[start:end] for start, end in slices2]
    assert frags2[0].startswith("1.F.2. LIMITED WARRANTY")
    assert frags2[0].endswith("right of replacement. ")

    text3 = "1.F.3. LIMITED RIGHT OF REPLACEMENT OR REFUND. If you discovered a defect, contact us."
    slices3 = split_sentences(text3, min_chars=0)
    assert len(slices3) == 2
    frags3 = [text3[start:end] for start, end in slices3]
    assert frags3[0] == "1.F.3. LIMITED RIGHT OF REPLACEMENT OR REFUND. "


def test_split_sentences_decimal_numbers_and_multiple_compound_markers() -> None:
    """Decimals do not break sentence splitting, and compound markers work within a single text."""
    text1 = (
        "The constant pi is approximately 3.14. It is used in mathematical formulas everywhere. "
        "The value of e is roughly 2.718. It is also an important constant."
    )
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 4

    text2 = (
        "1.A. First rule of the project. "
        "1.B. Second rule of the project. "
        "1.C. Third rule of the project."
    )
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 3
    frags2 = [text2[start:end] for start, end in slices2]
    assert frags2[0] == "1.A. First rule of the project. "
    assert frags2[1] == "1.B. Second rule of the project. "
    assert frags2[2] == "1.C. Third rule of the project."


def test_candidate_pattern_redos_safety() -> None:
    """Punctuation with a long sequence of digits without spaces does not cause ReDoS."""
    # String with a long numeric index without spaces
    pathological = "Section 4." + ("1234567890" * 5) + "X and some more words here to test safety."
    slices = split_sentences(pathological, min_chars=0)
    assert len(slices) >= 1
    # Text reconstruction is strictly preserved
    assert "".join(pathological[start:end] for start, end in slices) == pathological


def test_long_paragraph_window_performance() -> None:
    """Ultra-long paragraphs with many abbreviations are processed quickly and deterministically."""
    unit = "Dr. Watson visited Mr. Holmes on Jan. 15. The letter was filed under vol. 2, no. 4. "
    text = (unit * 100) + "Finally, the case was solved."
    assert len(text) > 8000
    slices = split_sentences(text)
    # 100 repetitions of 2 sentences + 1 final = 201 sentences
    assert len(slices) == 201
    assert "".join(text[start:end] for start, end in slices) == text
    assert len(slices) == 201
    assert "".join(text[start:end] for start, end in slices) == text


def test_split_paragraph_keeps_unequal_sentences_intact_without_mid_sentence_cuts() -> None:
    """Unequal sentences within max(SPLIT_MAX_PART_CHARS, min_chars - 50) are grouped whole without mid-sentence cuts."""
    s1 = (
        "At the far end of the gallery hung an enormous portrait whose tarnished gilt frame "
        "caught the pale winter light filtering through the tall lancet windows, casting faint "
        "geometric shadows across the polished oak floorboards where generations of silent visitors "
        "had paused to admire the brushstroke details and quiet dignity of the subject, while the "
        "curator softly recounted how the canvas survived three perilous voyages across the stormy "
        "northern sea before reaching this peaceful coastal manor. "
    )
    s2 = "Nobody spoke for a moment. "
    s3 = (
        "Then Margaret stepped closer to inspect the signature tucked into the lower corner "
        "beneath a thin layer of darkened varnish and whispered the painter's name aloud. "
    )
    s4 = (
        "Outside in the courtyard, the evening bell began to chime slowly through the mist "
        "as the heavy iron gate closed behind the last carriage of the afternoon."
    )
    max_part = max(SPLIT_MAX_PART_CHARS, SPLIT_MIN_CHARS - 50)
    assert 450 <= len(s1) <= max_part
    assert len(s2) < 60
    assert len(s3) <= max_part
    assert len(s4) <= max_part

    text = s1 + s2 + s3 + s4
    assert len(text) > SPLIT_MIN_CHARS

    slices = split_paragraph(text)
    assert len(slices) >= 2
    assert "".join(text[start:end] for start, end in slices) == text

    sentence_slices = split_sentences(text)
    valid_boundaries = {0, *(end for _, end in sentence_slices)}
    for start, end in slices:
        assert start in valid_boundaries
        assert end in valid_boundaries


def test_split_paragraph_splits_oversized_single_sentence_at_word_boundaries() -> None:
    """Oversized single sentences (> max_sentence_part) fall back to word boundaries while preserving text, markup, and normal sentences."""
    max_part = max(SPLIT_MAX_PART_CHARS, SPLIT_MIN_CHARS - 50)
    oversized_single = (
        "Throughout the remarkably intricate parliamentary debate concerning maritime navigation "
        + "and regional customs regulations across all distant coastal provinces, " * 9
        + "every delegate listened attentively without interrupting the speaker."
    )
    assert len(oversized_single) > max(SPLIT_MIN_CHARS, max_part)
    assert len(split_sentences(oversized_single)) == 1

    single_slices = split_paragraph(oversized_single)
    assert len(single_slices) >= 2
    assert "".join(oversized_single[start:end] for start, end in single_slices) == oversized_single
    for start, _ in single_slices[1:]:
        assert oversized_single[start - 1].isspace()
        assert not oversized_single[start].isspace()

    italic_phrase = "regional customs regulations"
    italic_start = oversized_single.rindex(italic_phrase)
    italic_end = italic_start + len(italic_phrase)
    annotated = InlineContent(
        text=oversized_single,
        annotations=(
            InlineAnnotation(
                type=InlineAnnotationType.ITALIC,
                start=italic_start,
                end=italic_end,
            ),
        ),
        anchor_ids=("para-oversized",),
    )
    fragments = split_inline_content(annotated)
    assert len(fragments) == len(single_slices)
    assert "".join(f.text for f in fragments) == oversized_single
    assert fragments[0].anchor_ids == ("para-oversized",)
    recovered_italic = "".join(
        frag.text[ann.start : ann.end]
        for frag in fragments
        for ann in frag.annotations
        if ann.type is InlineAnnotationType.ITALIC
    )
    assert recovered_italic == italic_phrase

    normal_second = (
        " Nevertheless, when the final bell rang across the chamber, the exhausted clerk "
        "gathered the parchment rolls neatly into his leather portfolio, bowed politely toward "
        "the gallery, extinguished the tall brass lamps beside the lectern, and walked out into "
        "the quiet moonlit square where fresh snow had begun to settle upon the cobblestones and "
        "narrow stone steps leading down toward the old harbor wall."
    )
    compound = oversized_single + " " + normal_second.lstrip()
    second_sentence_text = normal_second.lstrip()
    assert len(second_sentence_text) <= max_part
    assert len(second_sentence_text) > 380

    compound_slices = split_paragraph(compound)
    compound_parts = [compound[start:end] for start, end in compound_slices]
    assert "".join(compound_parts) == compound
    assert any(second_sentence_text in part for part in compound_parts)
    for start, _ in compound_slices[1:]:
        assert compound[start - 1].isspace()
        assert not compound[start].isspace()


def _repeat(word: str, minimum: int) -> str:
    """Repeats `word` until the string reaches `minimum` characters."""
    text = word
    while len(text) < minimum:
        text += " " + word
    return text


def _assert_exact_cover(text: str, segments: tuple[BlockSegment, ...]) -> None:
    """Spans are non-empty, ordered, contiguous, and rebuild `text`."""
    assert segments
    assert segments[0].start == 0
    assert segments[-1].end == len(text)
    cursor = 0
    for segment in segments:
        assert segment.start == cursor
        assert segment.end > segment.start
        assert segment.sentence_index >= 0
        cursor = segment.end
    assert cursor == len(text)
    assert "".join(text[segment.start : segment.end] for segment in segments) == text


def _assert_protected_intact(
    segments: tuple[BlockSegment, ...],
    span: tuple[int, int],
) -> None:
    """No segment boundary falls strictly inside a protected range."""
    start, end = span
    assert any(segment.start <= start and end <= segment.end for segment in segments)
    for segment in segments[:-1]:
        assert not (start < segment.end < end)


def test_short_text_single_segment() -> None:
    """Text below the block threshold is one span, even when it exceeds the segment cap."""
    assert SEGMENT_MIN_BLOCK_CHARS == 600
    assert SEGMENT_MAX_CHARS == 350

    short = "First sentence stays. Second sentence stays too."
    assert len(short) < SEGMENT_MIN_BLOCK_CHARS
    assert segment_text(short) == (BlockSegment(0, len(short), 0),)

    below = _repeat("zeta", SEGMENT_MIN_BLOCK_CHARS)[: SEGMENT_MIN_BLOCK_CHARS - 1]
    assert len(below) == SEGMENT_MIN_BLOCK_CHARS - 1
    assert segment_text(below) == (BlockSegment(0, len(below), 0),)

    bulky = _repeat("quartz", 500) + "."
    assert SEGMENT_MAX_CHARS < len(bulky) < SEGMENT_MIN_BLOCK_CHARS
    assert segment_text(bulky) == (BlockSegment(0, len(bulky), 0),)


def test_sentences_become_segments() -> None:
    """At the block threshold each sentence is its own segment, in order."""
    first = "Alpha"
    while len(first) < 300:
        first += " alpha"
    first += ". "
    second = "Beta"
    while len(second) < 300:
        second += " beta"
    second += "."
    text = first + second
    assert len(text) >= SEGMENT_MIN_BLOCK_CHARS
    assert len(first) <= SEGMENT_MAX_CHARS
    assert len(second) <= SEGMENT_MAX_CHARS

    segments = segment_text(text)
    assert segments == (
        BlockSegment(0, len(first), 0),
        BlockSegment(len(first), len(text), 1),
    )
    assert text[segments[1].start :].startswith("Beta")


def test_long_sentence_splits_at_semicolon_first() -> None:
    """A fitting semicolon wins over a later dash and a conjunction comma."""
    left = _repeat("alpha", 330)
    right = _repeat("beta", 250) + " — gently, and quietly"
    text = f"{left}; {right}."
    split_at = text.index(";") + 2
    assert text[split_at - 2 : split_at] == "; "
    assert len(text) >= SEGMENT_MIN_BLOCK_CHARS
    assert len(text) > SEGMENT_MAX_CHARS
    assert split_at <= SEGMENT_MAX_CHARS
    assert len(text) - split_at <= SEGMENT_MAX_CHARS
    assert "—" in text[split_at:]
    assert ", and " in text[split_at:]

    segments = segment_text(text)
    assert segments == (
        BlockSegment(0, split_at, 0),
        BlockSegment(split_at, len(text), 0),
    )


def test_long_sentence_splits_at_dash_before_word() -> None:
    """Em dash, en dash, and a spaced hyphen are used before a word boundary."""
    for dash in (" — ", " – ", " - "):
        left = _repeat("delta", 300)
        right = _repeat("epsilon", 300)
        text = f"{left}{dash}{right}."
        split_at = len(left) + len(dash)
        assert ";" not in text
        assert ":" not in text
        assert "," not in text
        assert len(text) >= SEGMENT_MIN_BLOCK_CHARS
        assert split_at <= SEGMENT_MAX_CHARS
        assert len(text) - split_at <= SEGMENT_MAX_CHARS
        assert text[split_at : split_at + 7] == "epsilon"

        segments = segment_text(text)
        assert segments == (
            BlockSegment(0, split_at, 0),
            BlockSegment(split_at, len(text), 0),
        )
        assert len(segments) == 2

    # A semicolon that still leaves a part over the cap is skipped; dashes that fit are used.
    skipped = (
        f"{_repeat('aaa', 170)} — {_repeat('bbb', 170)} — "
        f"{_repeat('ccc', 120)}; {_repeat('ddd', 124)}."
    )
    assert len(skipped) >= SEGMENT_MIN_BLOCK_CHARS
    assert ":" not in skipped and "," not in skipped
    semicolon_at = skipped.index(";") + 2
    assert semicolon_at > SEGMENT_MAX_CHARS or len(skipped) - semicolon_at > SEGMENT_MAX_CHARS
    dash_points = []
    cursor = 0
    while True:
        found = skipped.find(" — ", cursor)
        if found < 0:
            break
        dash_points.append(found + len(" — "))
        cursor = found + len(" — ")
    assert len(dash_points) == 2
    bounds = [0, *dash_points, len(skipped)]
    assert all(
        bounds[index + 1] - bounds[index] <= SEGMENT_MAX_CHARS
        for index in range(len(bounds) - 1)
    )
    segments = segment_text(skipped)
    assert segments == tuple(
        BlockSegment(bounds[index], bounds[index + 1], 0)
        for index in range(len(bounds) - 1)
    )
    assert all(segment.end != semicolon_at for segment in segments[:-1])


def test_comma_before_conjunction_before_word_boundary() -> None:
    """A comma before a conjunction is a split; a comma before another word is not."""
    decoy = ", butter and more, android things, however the hall"
    left = _repeat("echo", 280) + decoy
    right = _repeat("foxtrot", 280)
    text = f"{left}, and {right}."
    split_at = text.index(", and ") + len(", ")
    assert text[split_at:].startswith("and ")
    assert ";" not in text
    assert ":" not in text
    assert "—" not in text and "–" not in text and " - " not in text
    assert len(text) >= SEGMENT_MIN_BLOCK_CHARS
    assert split_at <= SEGMENT_MAX_CHARS
    assert len(text) - split_at <= SEGMENT_MAX_CHARS
    assert text[:split_at].count(",") == 4

    segments = segment_text(text)
    assert segments == (
        BlockSegment(0, split_at, 0),
        BlockSegment(split_at, len(text), 0),
    )

    for conjunction in (
        "but",
        "or",
        "nor",
        "yet",
        "so",
        "which",
        "who",
        "when",
        "while",
        "because",
        "though",
        "although",
        "where",
        "if",
    ):
        sample = f"Left clause stays here, {conjunction} the right clause follows."
        point = sample.index(f", {conjunction} ") + len(", ")
        sample_segments = segment_text(sample, min_block_chars=1, max_segment_chars=40)
        assert sample_segments == (
            BlockSegment(0, point, 0),
            BlockSegment(point, len(sample), 0),
        )


def test_word_fallback_only_without_punctuation() -> None:
    """Without clause punctuation a long sentence breaks on spaces, not inside a hyphenated word."""
    text = _repeat("well-known", 650) + "."
    assert len(text) >= SEGMENT_MIN_BLOCK_CHARS
    assert not any(mark in text for mark in (";", ":", "—", "–", " - ", ","))

    segments = segment_text(text)
    _assert_exact_cover(text, segments)
    assert len(segments) > 1
    for segment in segments:
        assert segment.sentence_index == 0
        assert segment.end - segment.start <= SEGMENT_MAX_CHARS
        if segment.start:
            assert text[segment.start - 1].isspace()
            assert not text[segment.start].isspace()
        assert "-" not in text[segment.start : segment.start + 1]


def test_segment_never_cuts_footnote_reference() -> None:
    """A protected footnote range stays inside one segment, even when longer than the cap."""
    left = _repeat("kappa", 300)
    right = _repeat("lambda", 300)
    semicolon_text = f"{left}; {right}."
    semi = semicolon_text.index(";")
    footnote = (semi, semi + 3)
    assert len(semicolon_text) >= SEGMENT_MIN_BLOCK_CHARS
    assert footnote[0] < semi + 2 < footnote[1]

    semicolon_segments = segment_text(semicolon_text, protected=(footnote,))
    _assert_exact_cover(semicolon_text, semicolon_segments)
    _assert_protected_intact(semicolon_segments, footnote)

    plain = _repeat("nu", 900) + "."
    wide = (200, 600)
    assert len(plain) >= SEGMENT_MIN_BLOCK_CHARS
    assert wide[1] - wide[0] > SEGMENT_MAX_CHARS
    plain_segments = segment_text(plain, protected=(wide,))
    _assert_exact_cover(plain, plain_segments)
    _assert_protected_intact(plain_segments, wide)
    holder = next(
        segment
        for segment in plain_segments
        if segment.start <= wide[0] and wide[1] <= segment.end
    )
    assert holder.end - holder.start > SEGMENT_MAX_CHARS

    first = _repeat("mu", 320)
    second = "Nu " + _repeat("nu", 280)
    sentence_text = f"{first}. {second}."
    sentence_point = len(first) + 2
    assert sentence_text[sentence_point : sentence_point + 2] == "Nu"
    sentence_footnote = (sentence_point - 2, sentence_point + 2)
    assert len(sentence_text) >= SEGMENT_MIN_BLOCK_CHARS
    sentence_segments = segment_text(sentence_text, protected=(sentence_footnote,))
    _assert_exact_cover(sentence_text, sentence_segments)
    _assert_protected_intact(sentence_segments, sentence_footnote)


def test_segments_cover_text_exactly() -> None:
    """Every sample is rebuilt by contiguous spans, including titles and a protected range."""
    unit = "Mr. Smith walked with Dr. Watson. "
    count = (SEGMENT_MIN_BLOCK_CHARS // len(unit)) + 1
    titled = unit * count + "Then they stopped."
    titled_segments = segment_text(titled)
    _assert_exact_cover(titled, titled_segments)
    assert len(titled_segments) == count + 1
    assert [segment.sentence_index for segment in titled_segments] == list(range(count + 1))
    assert all(not titled[segment.start :].startswith("Smith") for segment in titled_segments)
    assert all(not titled[segment.start :].startswith("Watson") for segment in titled_segments)

    samples = (
        "Short paragraph.",
        _repeat("alpha", 330) + "; " + _repeat("beta", 280) + ".",
        _repeat("delta", 300) + " — " + _repeat("epsilon", 300) + ".",
        _repeat("well-known", 650) + ".",
    )
    for sample in samples:
        _assert_exact_cover(sample, segment_text(sample))

    protected_text = _repeat("kappa", 300) + "; " + _repeat("lambda", 300) + "."
    semi = protected_text.index(";")
    protected = ((semi, semi + 3), (0, 0), (len(protected_text) + 5, len(protected_text) + 8))
    _assert_exact_cover(protected_text, segment_text(protected_text, protected=protected))
    _assert_protected_intact(
        segment_text(protected_text, protected=protected),
        (semi, semi + 3),
    )


def test_parts_of_sentence_share_sentence_index() -> None:
    """Clause parts keep the sentence index; the next sentence gets the next index."""
    left = _repeat("omicron", 250)
    right = _repeat("pi", 250)
    second = "Rho " + _repeat("rho", 100)
    text = f"{left}; {right}. {second}."
    clause_at = text.index(";") + 2
    sentence_at = text.index(". ") + 2
    assert text[sentence_at:].startswith("Rho ")
    assert len(text) >= SEGMENT_MIN_BLOCK_CHARS
    assert clause_at <= SEGMENT_MAX_CHARS
    assert sentence_at - clause_at <= SEGMENT_MAX_CHARS
    assert len(text) - sentence_at <= SEGMENT_MAX_CHARS

    segments = segment_text(text)
    assert segments == (
        BlockSegment(0, clause_at, 0),
        BlockSegment(clause_at, sentence_at, 0),
        BlockSegment(sentence_at, len(text), 1),
    )
    assert segments[0].sentence_index == segments[1].sentence_index
    assert segments[2].sentence_index == segments[0].sentence_index + 1

