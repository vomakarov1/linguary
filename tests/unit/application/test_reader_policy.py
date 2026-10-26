import pytest
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.reader_text import (
    ReaderTextPolicy,
    ReaderTextError,
    learner_instructions,
    validate_a2_summary,
)

LANGUAGE_DETECTOR = Py3LangidDetector()


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
                     'might have been', 'what', 'отрыв предлога', 'языковой заметке',
                     'not saying ... isn’t ...', 'утвердительный смысл', 'переводом на русский язык',
                     'два абзаца'):
        assert expected in instructions
    assert ReaderTextPolicy().warnings('summary', 'One ' * 40 + '.', reader_level='a2')


def test_a2_double_negative_instructions_keep_translation_and_byr_affirmative():
    instructions = ' '.join(learner_instructions('a2').split())

    assert 'Не переворачивай смысл при переводе' in instructions
    assert 'объяснение BYR формулируй утвердительно' in instructions
    assert 'не отрицаю, что заклинание блестящее' in instructions
    assert 'не превращай это в «заклинание не блестящее»' in instructions


@pytest.mark.parametrize(
    'summary',
    [
        'The story continues.',
        'The story continues.\n\nThe story continues.',
        'The story continues.\n\nИстория продолжается.\n\nГерои уходят.',
        'История продолжается.\n\nГерои уходят.',
        'La historia continúa.\n\nИстория продолжается.',
        'La noche.\n\nНочь.',
        'The story continues.\n\nИстория العربية العربية العربية.',
    ],
)
def test_a2_summary_requires_two_paragraphs_with_russian_translation(summary):
    with pytest.raises(ReaderTextError, match='a2_summary_requires_parallel_russian'):
        validate_a2_summary(
            summary,
            source_texts=('The story continues.',),
            source_language='en',
            language_detector=LANGUAGE_DETECTOR,
        )


def test_a2_summary_accepts_original_and_russian_translation_paragraphs():
    validate_a2_summary(
        'The story continues.\n\nИстория продолжается.',
        source_texts=('The story continues.',),
        source_language='en',
        language_detector=LANGUAGE_DETECTOR,
    )


def test_a2_summary_accepts_short_english_paragraph_with_low_classifier_confidence():
    validate_a2_summary(
        'Independent synthetic scene.\n\nНезависимая вымышленная сцена.',
        source_texts=('The story continues.',),
        source_language='en',
        language_detector=LANGUAGE_DETECTOR,
    )


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
