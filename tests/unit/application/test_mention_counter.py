"""Подсчёт упоминаний по исходному тексту книги."""

from analysis_fakes import make_analysis_document, make_analysis_section

from linguary.application.mention_counter import mention_counter


def test_mentions_case_insensitive_word_bounded() -> None:
    document = make_analysis_document((
        make_analysis_section(texts=("Dracula met DRACULA; draculastic did not.",)),
    ))
    assert mention_counter(document)(("dracula",)) == 2


def test_mentions_sum_spellings_without_overlaps() -> None:
    document = make_analysis_document((
        make_analysis_section(texts=("Mr. Harker met Harker. Mr Harker left.",)),
    ))
    assert mention_counter(document)(("Mr. Harker", "Mr Harker", "Harker")) == 3
