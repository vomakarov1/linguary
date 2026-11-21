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
    for expected in ('summary', 'before_you_read', 'glossary_hint', 'language_note',
                     'не считай короткую реплику автоматически простой',
                     'might have been', 'what', 'отрыв предлога', 'языковой заметке',
                     'not saying ... isn’t ...', 'утвердительный смысл',
                     'ровно одним абзацем на языке оригинала', 'без перевода на русский'):
        assert expected in instructions
    assert ReaderTextPolicy().warnings('summary', 'One ' * 40 + '.', reader_level='a2')


def test_a2_learner_instructions_require_infinitives_and_forbid_note_jargon():
    instructions = learner_instructions('a2')
    for phrase in (
        'помета к глаголу речи',
        '«реалия»',
        'страдательный оборот',
        'превосходная степень',
        'причастие прошедшего времени',
        'to snigger — хихикать',
        'to nod — кивать',
        'note пустым',
        'длиннее 5 слов',
        'целое предложение',
    ):
        assert phrase in instructions
    assert 'помета к глаголу речи' not in learner_instructions('b1')


def test_a2_double_negative_instructions_keep_translation_and_byr_affirmative():
    instructions = ' '.join(learner_instructions('a2').split())

    assert 'Не переворачивай смысл при переводе' in instructions
    assert 'объяснение BYR формулируй утвердительно' in instructions
    assert 'не отрицаю, что заклинание блестящее' in instructions
    assert 'не превращай это в «заклинание не блестящее»' in instructions


@pytest.mark.parametrize(
    'summary',
    [
        '',
        'The story continues.\n\nИстория продолжается.',
        'The story continues.\nИстория продолжается.',
        'Harry goes to Hogwarts with Hagrid.\nГарри едет в Хогвартс.',
        'Harry goes to Hogwarts. Гарри едет в Хогвартс.',
        'The story continues.\n\nThe story continues.',
        'История продолжается.',
        'La historia continúa.',
        'La noche.',
    ],
)
def test_a2_summary_requires_single_source_language_paragraph(summary):
    with pytest.raises(ReaderTextError, match='a2_summary_requires_single_source_paragraph'):
        validate_a2_summary(
            summary,
            source_texts=('The story continues.',),
            source_language='en',
            language_detector=LANGUAGE_DETECTOR,
        )


def test_a2_summary_accepts_single_source_language_paragraph():
    validate_a2_summary(
        'The story continues.',
        source_texts=('The story continues.',),
        source_language='en',
        language_detector=LANGUAGE_DETECTOR,
    )


def test_a2_summary_supports_legacy_parallel_russian_for_frozen_conditions():
    validate_a2_summary(
        'The story continues.\n\nИстория продолжается.',
        source_texts=('The story continues.',),
        source_language='en',
        language_detector=LANGUAGE_DETECTOR,
        require_parallel_russian=True,
    )
    with pytest.raises(ReaderTextError, match='a2_summary_requires_parallel_russian'):
        validate_a2_summary(
            'The story continues.',
            source_texts=('The story continues.',),
            source_language='en',
            language_detector=LANGUAGE_DETECTOR,
            require_parallel_russian=True,
        )


def test_a2_summary_accepts_short_english_paragraph_with_low_classifier_confidence():
    validate_a2_summary(
        'Independent synthetic scene.',
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


def _a2_summary(sentence_count: int) -> str:
    return " ".join(
        f"Sentence number {index} continues the chapter." for index in range(1, sentence_count + 1)
    )


def _summary_with_titles(sentence_count: int) -> str:
    sentences = [
        f"Sentence number {index} continues the chapter."
        for index in range(1, sentence_count)
    ]
    sentences.append("Mr. and Mrs. Dursley woke up at number four, where J. K. left a note.")
    return " ".join(sentences)


def test_validate_a2_summary_rejects_overlong_multi_sentence_recap():
    """Six sentences are accepted, including titles and initials; seven are rejected."""
    validate_a2_summary(
        _a2_summary(6),
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
    )
    validate_a2_summary(
        _summary_with_titles(6),
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
    )
    with pytest.raises(ReaderTextError):
        validate_a2_summary(
            _a2_summary(7),
            source_texts=("The story continues.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
        )
    with pytest.raises(ReaderTextError):
        validate_a2_summary(
            _summary_with_titles(7),
            source_texts=("The story continues.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
        )


def test_reader_policy_normalizes_en_dash_and_preserves_dialogue_newline():
    """Translation replies use an em dash while newlines before dialogue dashes stay intact.

    Leaving the en dash, joining a newline before a reply, or changing a protected quote fails the test.
    """
    policy = ReaderTextPolicy()
    for punct in (".", ":", ",", "!", "?", "…"):
        text = f"Конец{punct}\n— Да."
        assert policy.prepare("translation", text, ()) == text

    assert policy.prepare("translation", "– Реплика.", ()) == "— Реплика."
    assert policy.prepare("translation", "Абзац.\n– Реплика.", ()) == "Абзац.\n— Реплика."
    assert policy.prepare("translation", "Конец.\n\n– Новая.", ()) == "Конец.\n\n— Новая."
    assert policy.prepare("translation", "Конец.\n\n— Новая.", ()) == "Конец.\n\n— Новая."
    assert policy.prepare("summary", "– Реплика.", ()) == "– Реплика."
    assert (
        policy.prepare("translation", "– Он сказал «– нет».", ("– нет",))
        == "— Он сказал «– нет»."
    )
    protected = "Конец.\n— «.\n— внутри»."
    assert policy.prepare("translation", protected, (".\n— внутри",)) == protected
    outside = "– Снаружи.\nТекст.\n— Дальше «.\n— внутри»."
    assert policy.prepare("translation", outside, (".\n— внутри",)) == (
        "— Снаружи.\nТекст.\n— Дальше «.\n— внутри»."
    )


def test_reader_policy_preserves_dialogue_newlines_and_defines_four_russian_dialogue_rules():
    policy = ReaderTextPolicy()
    assert (
        policy.prepare("translation", "— Привет, — сказал он.\n— Как дела?", ())
        == "— Привет, — сказал он.\n— Как дела?"
    )
    assert (
        policy.prepare("translation", "Он вошёл.\n– Кто здесь?", ())
        == "Он вошёл.\n— Кто здесь?"
    )

    instructions = learner_instructions("a2")
    for expected in (
        "4 правила оформления русской прямой речи",
        "1) Реплика диалога с начала абзаца",
        "длинным тире «— »",
        "2) Прямая речь внутри абзаца после двоеточия",
        "Он сказал: «…»",
        "3) Внутренняя речь (мысли)",
        "«…», — подумал он",
        "4) Названия, вывески и цитаты внутри реплики",
        "„…“",
        "— «",
        "«—",
        "слова автора внутри «…»",
        "букву «ё»",
        "всё/все",
        "узнаёт/узнает",
        "лубок",
        "сленг",
        "Различай роль заметок в GLOSSARY_ONLY и TRANSLATE",
        "Не домысливай сюжетные детали",
        "вежливое «вы»",
    ):
        assert expected in instructions
    assert "4 правила оформления русской прямой речи" not in learner_instructions("b1")


def test_reader_text_error_includes_kind_and_block_id():
    from linguary.domain import BlockProcessingResult, BlockTreatment, LanguageNote

    policy = ReaderTextPolicy()
    result = BlockProcessingResult(
        'block-10',
        BlockTreatment.READ_AS_IS,
        language_notes=(LanguageNote('n1', 'block-10', 'Echo', '[b17]'),),
    )
    with pytest.raises(ReaderTextError) as exc_info:
        policy.prepare_results((result,))
    err = exc_info.value
    assert err.reason_code == 'required_help_erased'
    assert err.kind == 'language_note.note'
    assert err.block_id == 'block-10'
    assert '(language_note.note, block-10)' in str(err)
    assert err.violation is not None
    assert err.violation.scope == 'local'
    assert err.violation.block_ids == ('block-10',)
    assert err.violation.artifact_kinds == ('language_note.note',)


def test_reader_policy_prepare_results_is_idempotent_with_stored_typography():
    """Stored v3 bodies with soft hyphens, German quotes, or en-dashes pass re-validation unchanged."""
    from linguary.domain import BlockProcessingResult, BlockTreatment, GlossaryHint, LanguageNote

    policy = ReaderTextPolicy()
    result = BlockProcessingResult(
        'block-11',
        BlockTreatment.TRANSLATE,
        translated_text='Он сказал: „Привет“ – и ушёл.',
        glossary_hints=(
            GlossaryHint('require', 'требо\u00adваться', 'Здесь требо\u00adваться означает нужно.'),
        ),
        language_notes=(LanguageNote('n1', 'block-11', 'require', 'Может требовать\u00adся.'),),
    )
    once = policy.prepare_results((result,))
    twice = policy.prepare_results(once)
    assert once == (result,)
    assert twice == once

