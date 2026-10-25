import pytest
from linguary.application.reader_text import ReaderTextPolicy, ReaderTextError, learner_instructions


def test_b17_removed_from_explanation_not_quote():
    policy = ReaderTextPolicy()
    body = '[b17] В записи «b17» буква обозначает колонку.'
    assert policy.prepare('language_note.note', body, ('b17',)) == 'В записи «b17» буква обозначает колонку.'
    assert policy.prepare('language_note.note', 'Цитата «[b17]» обозначает запись.', ('[b17]',)) == 'Цитата «[b17]» обозначает запись.'


@pytest.mark.parametrize('kind', ['translation', 'summary', 'glossary_hint.translation'])
def test_reader_policy_rejects_ambiguous_labels_in_non_explanations(kind):
    with pytest.raises(ReaderTextError):
        ReaderTextPolicy().prepare(kind, '[b17] Учебный текст.', ())


def test_cleanup_cannot_erase_required_help():
    with pytest.raises(ReaderTextError):
        ReaderTextPolicy().prepare('before_you_read.explanation', '[b17]', ())
    with pytest.raises(ReaderTextError):
        ReaderTextPolicy().prepare('language_note.note', '[b17] «echo»', ('echo',))


def test_a2_policy_in_all_channels():
    instructions = learner_instructions('a2')
    for expected in ('summary', 'before_you_read', 'glossary_hint', 'language_note', 'Shoo', 'snapped', 'went out',
                     'might have been', 'what', 'отрыв предлога', 'языковой заметке'):
        assert expected in instructions
    assert ReaderTextPolicy().warnings('summary', 'One ' * 40 + '.', reader_level='a2')


def test_b1_b2_calibration_kept():
    assert 'прежнюю калибровку B1–B2' in learner_instructions('b1')
    assert 'прежнюю калибровку B1–B2' in learner_instructions('b2')


def test_showcase_claim_is_review_case():
    policy = ReaderTextPolicy()
    text = 'Стекло витрины доказывает, что действие происходит в магазине.'
    assert policy.prepare('before_you_read.explanation', text, ()) == text
    assert policy.semantic_correctness == 'not_evaluated'


def test_translation_preserves_exact_source_quote_with_service_like_text():
    from linguary.application.reader_text import source_quote_fragments
    original = 'The printed code was "[b17]".'
    translation = 'Напечатан код «[b17]».'
    assert ReaderTextPolicy().prepare('translation', translation, source_quote_fragments((original,))) == translation
