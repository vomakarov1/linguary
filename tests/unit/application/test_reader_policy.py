import pytest
from linguary.adapters.validation import Py3LangidDetector
from linguary.application.reader_text import (
    A2_SUMMARY_HARD_MAX_SENTENCE_WORDS,
    A2_SUMMARY_WARNING_MAX_WORDS,
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


def test_a2_summary_warnings_calibrated_to_22_words_and_protect_titles():
    policy = ReaderTextPolicy()

    clean_18 = "Nick and Allie walked along the dark path, and they tried to stay quiet near the old trees."
    assert len(clean_18.split()) == 18
    assert policy.warnings('summary', clean_18, reader_level='a2') == ()

    clean_21 = "Nick and Allie looked for the hidden road, but the thick silver fog covered every stone sign along the narrow path."
    assert len(clean_21.split()) == 21
    assert policy.warnings('summary', clean_21, reader_level='a2') == ()

    title_22 = "Mr. Smith and Mrs. Brown walked slowly down the quiet village street and talked about the old map before dinner arrived tonight."
    assert len(title_22.split()) == 22
    assert policy.warnings('summary', title_22, reader_level='a2') == ()

    # 10 words before "Mr. Smith" and 12 after: splitting at "Mr." would stay under threshold, unsplit is 23 > 22.
    title_23 = "On a cold morning in November near the tall clock Mr. Smith walked slowly down the empty village street with his little dog."
    assert len(title_23.split()) == 23
    assert policy.warnings('summary', title_23, reader_level='a2') == ('complexity_requires_review',)

    long_25 = "Nick and Allie carefully followed the strange boy through the dark silver forest because they wanted to reach the distant towers before sunset arrived again."
    assert len(long_25.split()) == 25
    assert policy.warnings('summary', long_25, reader_level='a2') == ('complexity_requires_review',)

    assert policy.warnings('summary', 'One ' * 40 + '.', reader_level='a2') == ('complexity_requires_review',)

    # Non-summary channels keep the stricter 15-word threshold.
    note_15 = "Это объяснение состоит ровно из пятнадцати русских слов и не должно вызывать лишнее предупреждение сейчас."
    assert len(note_15.split()) == 15
    assert policy.warnings('language_note.note', note_15, reader_level='a2') == ()

    note_16 = "Это длинное объяснение состоит из шестнадцати разных русских слов и поэтому должно вызывать предупреждение редактору сейчас."
    assert len(note_16.split()) == 16
    assert policy.warnings('language_note.note', note_16, reader_level='a2') == ('complexity_requires_review',)
    assert policy.warnings('before_you_read.explanation', note_16, reader_level='a2') == ('complexity_requires_review',)

    # Non-A2 levels emit no complexity warnings.
    assert policy.warnings('summary', long_25, reader_level='b1') == ()
    assert policy.warnings('language_note.note', note_16, reader_level='b2') == ()


def test_validate_a2_summary_rejects_excessive_sentence_length_and_present_simple_for_duration():
    assert A2_SUMMARY_WARNING_MAX_WORDS == 22
    assert A2_SUMMARY_HARD_MAX_SENTENCE_WORDS == 26

    # 26 words: passes hard validation, still triggers the 22-word editorial warning.
    sentence_26 = (
        "Nick and Allie carefully followed the strange boy through the dark silver forest "
        "because they wanted to reach the distant glass towers before sunset arrived tonight."
    )
    assert len(sentence_26.split()) == 26
    validate_a2_summary(
        sentence_26,
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
    )
    assert ReaderTextPolicy().warnings('summary', sentence_26, reader_level='a2') == (
        'complexity_requires_review',
    )

    # 27 words: rejected as too long, including when titles like "Mr." appear inside the sentence.
    sentence_27 = (
        "On a cold autumn morning near the old station Mr. Smith walked slowly down "
        "the empty village street because he wanted to find the lost silver key."
    )
    assert len(sentence_27.split()) == 27
    with pytest.raises(ReaderTextError, match='a2_summary_sentence_too_long') as exc_info:
        validate_a2_summary(
            sentence_27,
            source_texts=("The story continues.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
        )
    assert exc_info.value.reason_code == 'a2_summary_sentence_too_long'

    # Legacy frozen two-paragraph mode keeps previous behavior unchanged.
    validate_a2_summary(
        f"{sentence_27}\n\nХолодным осенним утром мистер Смит медленно шёл по пустой улице.",
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        require_parallel_russian=True,
    )

    # Ungrammatical Present Simple + for <duration> within one sentence.
    ungrammatical_cases = (
        "Nick and Allie live in Everlost for a long time.",
        "Nick and Allie live in that house for many years.",
        "Mary lives in the tall tower for many years.",
        "Mary stays in that tower for centuries.",
        "They stay in the quiet forest for centuries.",
        "He stays in the dark village for months.",
        "We work at the small station for years.",
        "He works at that shop for years.",
        "She works at the old museum for many years.",
        "They know this strange secret for a long time.",
        "They know that secret for a long time.",
        "Nick and Allie live in that old haunted house for many years.",
        "Allie knows the hidden path for years.",
        "He has arrived, and she lives in the tower for many years.",
        "Nick's friends live in Mary's house for many years.",
        "She does not live in the village for many years.",
    )
    for bad_summary in ungrammatical_cases:
        with pytest.raises(
            ReaderTextError, match='a2_summary_ungrammatical_present_for'
        ) as exc_info:
            validate_a2_summary(
                bad_summary,
                source_texts=("The story continues.",),
                source_language="en",
                language_detector=LANGUAGE_DETECTOR,
            )
        assert exc_info.value.reason_code == 'a2_summary_ungrammatical_present_for'


def test_validate_a2_summary_allows_present_perfect_and_checks_factual_anchor():
    # Valid Present Perfect, auxiliaries, wait/waits, cross-sentence, cross-clause, and noun uses.
    grammatical_cases = (
        "Nick and Allie have lived in Everlost for a long time.",
        "Mary has stayed in the tower for many years.",
        "They've known each other for years.",
        "She\u2019s worked in the old house for months.",
        "He had lived near the river for centuries.",
        "They will stay in the quiet forest for months.",
        "She would live in the village for years.",
        "He did not work at the shop for many years.",
        "He has to live in the village for months.",
        "Nick waits at the gate for a long time.",
        "They wait near the river for years.",
        "Nick and Allie live in the forest. They waited there for many years.",
        "He knows that they lived there for many years.",
        "He knows that she visited the city for many years.",
        "He knows that Mary visited the city for many years.",
        "He knows that Nick and Allie went to the forest for many years.",
        "He knows that the old man walked there for many years.",
        "She knows that Allie was in Everlost for many years.",
        "Won't you stay here for many years?",
        "Didn't you live here for many years?",
        "Don't you live here for many years?",
        "Wouldn't she stay here for many years?",
        "Can't we stay here for months?",
        "Couldn't Nick work here for years?",
        "Shouldn't they stay here for many years?",
        "He works at the shop and waits for a long time.",
        "They lived their lives in peace for many years.",
        "He was at work for a long time.",
        "She looked for work for many years.",
        "Without work for months, he left the town.",
    )
    for good_summary in grammatical_cases:
        validate_a2_summary(
            good_summary,
            source_texts=("The story continues.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
        )

    narrative_summary = (
        "Nick and Allie wake up in Everlost, meet Lief in the forest, "
        "and walk toward the Twin Towers."
    )

    # On-topic summary with shared significant stems passes.
    validate_a2_summary(
        "Nick and Allie walk through the dark forest and reach the tall towers together.",
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        narrative_summary=narrative_summary,
    )

    # Off-topic summary with >= 4 stems in both summaries and zero shared stems is rejected.
    off_topic = (
        "Captain Morgan sails across the stormy ocean looking for golden treasure on an empty island."
    )
    with pytest.raises(ReaderTextError, match='a2_summary_off_topic') as exc_info:
        validate_a2_summary(
            off_topic,
            source_texts=("The story continues.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
            narrative_summary=narrative_summary,
        )
    assert exc_info.value.reason_code == 'a2_summary_off_topic'

    # Short summary (< 4 significant stems) or short narrative_summary (< 4 stems) does not false-positive.
    validate_a2_summary(
        "Two kids travel.",
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        narrative_summary=narrative_summary,
    )
    validate_a2_summary(
        off_topic,
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        narrative_summary="Two kids travel.",
    )

    # Empty, whitespace, or omitted narrative_summary skips factual anchor validation.
    for empty_narrative in (None, "", "   "):
        validate_a2_summary(
            off_topic,
            source_texts=("The story continues.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
            narrative_summary=empty_narrative,
        )

    # Legacy parallel Russian mode skips factual anchor validation.
    validate_a2_summary(
        f"{off_topic}\n\nКапитан Морган плывёт через штормовой океан в поисках сокровищ.",
        source_texts=("The story continues.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        require_parallel_russian=True,
        narrative_summary=narrative_summary,
    )


def test_validate_a2_summary_rejects_semicolon_in_monolingual_mode_only():
    assert A2_SUMMARY_WARNING_MAX_WORDS == 22
    assert A2_SUMMARY_HARD_MAX_SENTENCE_WORDS == 26

    with pytest.raises(ReaderTextError, match='a2_summary_disallowed_semicolon') as exc_info:
        validate_a2_summary(
            "Nick wakes up in Everlost; Allie stands next to him.",
            source_texts=("Nick woke up in the woods.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
        )
    assert exc_info.value.reason_code == 'a2_summary_disallowed_semicolon'
    assert 'точка с запятой' in (exc_info.value.detail or '')

    # Two separate simple sentences without a semicolon pass.
    validate_a2_summary(
        "Nick wakes up in Everlost. Allie stands next to him.",
        source_texts=("Nick woke up in the woods.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
    )

    # Passive-like phrasing is not rejected by regex in validate_a2_summary.
    validate_a2_summary(
        "Nick and Allie are lost in the dark woods. They look for a road.",
        source_texts=("Nick woke up in the woods.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
    )

    # Legacy parallel Russian mode keeps previous behavior and does not reject semicolons.
    validate_a2_summary(
        "Nick wakes up in Everlost; Allie stands next to him.\n\n"
        "Ник просыпается в Эверлосте; Элли стоит рядом с ним.",
        source_texts=("Nick woke up in the woods.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        require_parallel_russian=True,
    )


def test_validate_a2_summary_rejects_disallowed_heading_prefix_in_monolingual_mode_only():
    bad_prefixes = (
        "Previously in Chapter: Nick wakes up in Everlost.",
        "Previously in Chapter 1: Nick wakes up in Everlost.",
        "Previously in Chapter Two — Nick wakes up in Everlost.",
        "Previously in the chapter - Nick wakes up in Everlost.",
        "Chapter 3 Summary: Nick wakes up in Everlost.",
        "Chapter three summary — Nick wakes up in Everlost.",
        "Summary: Nick wakes up in Everlost.",
        "summary - Nick wakes up in Everlost.",
    )
    for bad_summary in bad_prefixes:
        with pytest.raises(
            ReaderTextError, match='a2_summary_disallowed_heading_prefix'
        ) as exc_info:
            validate_a2_summary(
                bad_summary,
                source_texts=("Nick woke up in the woods.",),
                source_language="en",
                language_detector=LANGUAGE_DETECTOR,
            )
        assert exc_info.value.reason_code == 'a2_summary_disallowed_heading_prefix'
        assert 'Previously in Chapter:' in (exc_info.value.detail or '')

    # Clean A2 summaries without heading prefixes pass.
    for good_summary in (
        "Nick wakes up in Everlost and meets Allie.",
        "Previously in the dark forest, Nick lost his silver coin.",
        "The summary on the old sign helps Nick find the road.",
    ):
        validate_a2_summary(
            good_summary,
            source_texts=("Nick woke up in the woods.",),
            source_language="en",
            language_detector=LANGUAGE_DETECTOR,
        )

    # Legacy frozen two-paragraph mode keeps previous behavior and does not reject heading prefixes.
    validate_a2_summary(
        "Previously in Chapter 1: Nick wakes up in Everlost.\n\n"
        "Ранее в главе 1: Ник просыпается в Эверлосте.",
        source_texts=("Nick woke up in the woods.",),
        source_language="en",
        language_detector=LANGUAGE_DETECTOR,
        require_parallel_russian=True,
    )
