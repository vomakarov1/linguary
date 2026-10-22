"""Тесты для модуля токенизации предложений и проекции встроенной разметки."""

from __future__ import annotations

import pytest

from linguary.adapters.epub.inline import InlineContent
from linguary.adapters.epub.splitting import (
    SPLIT_MIN_CHARS,
    SPLIT_MIN_SENTENCES,
    split_inline_content,
    split_sentences,
)
from linguary.domain import InlineAnnotation, InlineAnnotationType


def test_split_sentences_skips_short_paragraphs() -> None:
    """Короткие абзацы (меньше min_chars) не дробятся."""
    short_text = "First sentence. Second sentence. Third sentence."
    assert len(short_text) < SPLIT_MIN_CHARS
    slices = split_sentences(short_text)
    assert slices == ((0, len(short_text)),)

    # Длинный текст, но состоящий из одного монолитного предложения без границ
    monolithic_text = "Word " * 200
    assert len(monolithic_text) >= SPLIT_MIN_CHARS
    slices_monolithic = split_sentences(monolithic_text)
    assert slices_monolithic == ((0, len(monolithic_text)),)


def test_split_sentences_preserves_titles_and_abbreviations() -> None:
    """Сокращения и титулы (Mr., Dr., e.g., vol.) не разрывают предложение."""
    text = (
        "Mr. Smith visited Dr. Watson in London to discuss Prof. Moriarty and his strange plans. "
        "Dr. Watson suggested they consult Mrs. Hudson before making any final decisions. "
        "The evidence included several documents, e.g. letters and diaries found in the study. "
        "All materials were cataloged according to vol. 1 and no. 5 of the society records. "
        "Finally, they decided to proceed with the investigation tomorrow morning."
    )
    # Запускаем с min_chars=0 для проверки логики токенизации на фрагменте
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
    """Инициалы (H. G. Wells, J. K. Rowling) не принимаются за границы предложений."""
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
    """Диалоговые реплики в кавычках делятся с учётом слов автора и прямой речи."""
    # Реплика с атрибуцией автора строчными буквами: граница после слов автора
    text1 = '"I see," said Alice. "What do you mean?" asked the Hatter. "Nothing!" she replied.'
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 3
    assert text1[slices1[0][0]:slices1[0][1]] == '"I see," said Alice. '
    assert text1[slices1[1][0]:slices1[1][1]] == '"What do you mean?" asked the Hatter. '
    assert text1[slices1[2][0]:slices1[2][1]] == '"Nothing!" she replied.'

    # Реплика с восклицанием и атрибуцией имени собственного
    text2 = '"Stop!" Alice said. "Never!" the Queen shouted.'
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 2
    assert text2[slices2[0][0]:slices2[0][1]] == '"Stop!" Alice said. '
    assert text2[slices2[1][0]:slices2[1][1]] == '"Never!" the Queen shouted.'

    # Реплика с действием (не атрибуция речи): деление происходит сразу после кавычек
    text3 = '"Stop!" The King stood up. "You must not do that."'
    slices3 = split_sentences(text3, min_chars=0)
    assert len(slices3) == 3
    assert text3[slices3[0][0]:slices3[0][1]] == '"Stop!" '
    assert text3[slices3[1][0]:slices3[1][1]] == "The King stood up. "
    assert text3[slices3[2][0]:slices3[2][1]] == '"You must not do that."'

    # Типографские кавычки («ёлочки» и фигурные)
    text4 = '“I see,” said Alice. “What do you mean?”'
    slices4 = split_sentences(text4, min_chars=0)
    assert len(slices4) == 2
    assert text4[slices4[0][0]:slices4[0][1]] == '“I see,” said Alice. '
    assert text4[slices4[1][0]:slices4[1][1]] == '“What do you mean?”'


def test_split_sentences_handles_em_dash_dialogues() -> None:
    """Диалоги через тире делятся по репликам без разрыва слов автора."""
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
    """Многоточия не разрывают середину предложения и слова автора."""
    # Многоточие перед строчной атрибуцией речи
    text1 = '"Wait..." said Alice. "I am coming."'
    slices1 = split_sentences(text1, min_chars=0)
    assert len(slices1) == 2
    assert text1[slices1[0][0]:slices1[0][1]] == '"Wait..." said Alice. '
    assert text1[slices1[1][0]:slices1[1][1]] == '"I am coming."'

    # Многоточие перед новым предложением с заглавной буквы
    text2 = "She hesitated... Then she turned and ran quickly down the hall."
    slices2 = split_sentences(text2, min_chars=0)
    assert len(slices2) == 2
    assert text2[slices2[0][0]:slices2[0][1]] == "She hesitated... "
    assert text2[slices2[1][0]:slices2[1][1]] == "Then she turned and ran quickly down the hall."

    # Многоточие в середине предложения (пауза)
    text3 = "She was... well, she was rather confused by the Hatter."
    slices3 = split_sentences(text3, min_chars=0)
    assert len(slices3) == 1


def test_split_sentences_text_reconstruction_invariant() -> None:
    """Конкатенация срезов строго восстанавливает исходный текст абзаца."""
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

        # Проверяем непрерывность срезов
        assert slices[0][0] == 0
        assert slices[-1][1] == len(sample)
        for i in range(len(slices) - 1):
            assert slices[i][1] == slices[i + 1][0]


def test_split_inline_content_shifts_annotation_offsets() -> None:
    """Смещения аннотаций корректно сдвигаются по формуле [start - offset, end - offset)."""
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

    fragments = split_inline_content(content, min_chars=0)
    assert len(fragments) == 2

    # Первый фрагмент не содержит разметку
    assert fragments[0].annotations == ()

    # Второй фрагмент содержит смещённую разметку
    assert len(fragments[1].annotations) == 1
    shifted_ann = fragments[1].annotations[0]
    assert shifted_ann.type is InlineAnnotationType.BOLD
    offset = len(fragments[0].text)
    assert shifted_ann.start == bold_start - offset
    assert shifted_ann.end == bold_end - offset
    assert fragments[1].text[shifted_ann.start : shifted_ann.end] == bold_target


def test_split_inline_content_clips_spanning_annotations() -> None:
    """Сквозная разметка (например, курсив на несколько предложений) подрезается на границах."""
    text = "First sentence is here. Second sentence follows."
    # Курсив охватывает часть первого предложения и часть второго
    italic_start = text.index("sentence is here")
    italic_end = text.index("follows") + len("follows")

    content = InlineContent(
        text=text,
        annotations=(
            InlineAnnotation(type=InlineAnnotationType.ITALIC, start=italic_start, end=italic_end),
        ),
    )

    fragments = split_inline_content(content, min_chars=0)
    assert len(fragments) == 2

    # В первом фрагменте разметка обрезана на границе фрагмента
    assert len(fragments[0].annotations) == 1
    ann0 = fragments[0].annotations[0]
    assert ann0.type is InlineAnnotationType.ITALIC
    assert ann0.start == italic_start
    assert ann0.end == len(fragments[0].text)

    # Во втором фрагменте разметка начинается с 0 и заканчивается на относительном смещении
    assert len(fragments[1].annotations) == 1
    ann1 = fragments[1].annotations[0]
    assert ann1.type is InlineAnnotationType.ITALIC
    assert ann1.start == 0
    offset = len(fragments[0].text)
    assert ann1.end == italic_end - offset


def test_split_inline_content_preserves_footnote_references() -> None:
    """Ссылки на сноски не рассекаются и сохраняются в соответствующих фрагментах."""
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

    fragments = split_inline_content(content, min_chars=0)
    assert len(fragments) == 2

    # Первый фрагмент сохраняет fn1
    assert len(fragments[0].annotations) == 1
    assert fragments[0].annotations[0] == InlineAnnotation(
        type=InlineAnnotationType.FOOTNOTE_REFERENCE,
        start=fn1_start,
        end=fn1_end,
        target="notes.xhtml#fn1",
    )

    # Второй фрагмент сохраняет fn2 со сдвигом
    assert len(fragments[1].annotations) == 1
    offset = len(fragments[0].text)
    assert fragments[1].annotations[0] == InlineAnnotation(
        type=InlineAnnotationType.FOOTNOTE_REFERENCE,
        start=fn2_start - offset,
        end=fn2_end - offset,
        target="notes.xhtml#fn2",
    )


def test_split_inline_content_anchors_assigned_to_first_fragment() -> None:
    """Якоря блока присваиваются только первому фрагменту, последующие получают пустые кортежи."""
    content = InlineContent(
        text="First sentence here. Second sentence here. Third sentence here.",
        anchor_ids=("anchor-1", "anchor-2"),
    )

    fragments = split_inline_content(content, min_chars=0)
    assert len(fragments) == 3
    assert fragments[0].anchor_ids == ("anchor-1", "anchor-2")
    assert fragments[1].anchor_ids == ()
    assert fragments[2].anchor_ids == ()


def test_split_inline_content_returns_original_if_unsplit() -> None:
    """Если деление не состоялось, возвращается исходный объект InlineContent."""
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
    assert result_monolithic == (monolithic_content,)

