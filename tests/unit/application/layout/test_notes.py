"""Notes projected for one block: markers, labels, and merged anchors."""

from __future__ import annotations

from typing import Literal

from linguary.application.explained_expressions import expression_words
from linguary.application.layout.notes import KIND_LABELS, LayoutNote, block_notes
from linguary.application.pronunciation_projection import PronunciationOccurrence
from linguary.domain import (
    BlockTreatment,
    BlockType,
    DocumentBlock,
    GlossaryHint,
    LanguageNote,
    SourceReference,
)
from linguary.domain.content_plan import (
    DIFFICULTY_KINDS,
    ContentEvidence,
    ContentPlan,
    Difficulty,
    HelpEvidence,
    RequiredPoint,
    SourceAnchor,
    artifact_key,
)
from linguary.domain.learning import FragmentAssistance
from linguary.domain.processing import BlockProcessingResult


def _block(text: str) -> DocumentBlock:
    """A paragraph whose only varying input is the source text."""
    return DocumentBlock.create(
        document_id="book-1",
        section_id="section-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(
            resource_path="chapter.xhtml",
            structural_path="/body/p[1]",
        ),
    )


def _end(text: str, quote: str) -> int:
    """End offset of the first exact occurrence."""
    start = text.find(quote)
    assert start >= 0
    return start + len(quote)


def _anchor(block: DocumentBlock, quote: str, start: int | None = None) -> SourceAnchor:
    """An anchor whose quote sits at `start` in this block."""
    if start is None:
        start = block.text.find(quote)
    return SourceAnchor(block.id, start, start + len(quote), quote)


def _difficulty(
    block_id: str,
    anchors: tuple[SourceAnchor, ...],
    kind: str,
    *,
    difficulty_id: str | None = None,
    help_kind: str = "explanation",
) -> Difficulty:
    """One difficulty owned by `block_id`."""
    chosen = difficulty_id or f"d-{kind}"
    return Difficulty(
        chosen,
        anchors,
        block_id,
        kind,
        "useful",
        help_kind,
        (RequiredPoint(f"p-{chosen}", "Explain the point"),),
    )


def _plan(*difficulties: Difficulty) -> ContentPlan:
    return ContentPlan(difficulties=difficulties)


def _result(
    block: DocumentBlock,
    *,
    treatment: BlockTreatment = BlockTreatment.READ_AS_IS,
    translation: str | None = None,
    hints: tuple[GlossaryHint, ...] = (),
    notes: tuple[LanguageNote, ...] = (),
    assistances: tuple[FragmentAssistance, ...] = (),
    evidence: ContentEvidence | tuple[()] = (),
) -> BlockProcessingResult:
    return BlockProcessingResult(
        block.id,
        treatment,
        translated_text=translation,
        glossary_hints=hints,
        language_notes=notes,
        fragment_assistances=assistances,
        content_evidence=evidence,
    )


def _assistance(
    difficulty: Difficulty,
    text: str | None,
    kind: str = "sentence_rephrase",
    anchors: tuple[SourceAnchor, ...] | None = None,
) -> FragmentAssistance:
    return FragmentAssistance(
        difficulty.id,
        tuple(point.id for point in difficulty.required_points),
        kind,
        text,
        difficulty.anchors if anchors is None else anchors,
    )


def _pron(
    block: DocumentBlock,
    quote: str,
    ipa: str,
    variant: Literal["uk", "us"],
    *,
    start: int | None = None,
    block_id: str | None = None,
) -> PronunciationOccurrence:
    if start is None:
        start = block.text.find(quote)
    return PronunciationOccurrence(
        hint_id=f"h-{quote}-{start}",
        block_id=block.id if block_id is None else block_id,
        start=start,
        end=start + len(quote),
        quote=quote,
        ipa=ipa,
        variant=variant,
        source_label="lexicon",
        source_ref=None,
        is_primary=True,
    )


def _hint_key(kind: str, hint: GlossaryHint, block_id: str) -> str:
    content = {"original": hint.original, "translation": hint.translation, "note": hint.note}
    return artifact_key(kind, block_id, content)


def _hint_notes(text: str, *hints: GlossaryHint) -> tuple[LayoutNote, ...]:
    """Glossary notes for `text`, with no plan and no pronunciation."""
    block = _block(text)
    return block_notes(
        block,
        _result(block, treatment=BlockTreatment.GLOSSARY_ONLY, hints=hints),
        None,
        (),
    )


def _span_end(text: str, surface: str) -> int:
    """Exclusive end of the first `surface` fragment."""
    return text.index(surface) + len(surface)


def _note_key(note: LanguageNote) -> str:
    content = {
        "original_excerpt": note.original_excerpt,
        "note": note.note,
        "translated_excerpt": note.translated_excerpt,
    }
    return artifact_key("language_note.note", note.block_id, content)


def test_hint_marker_after_first_occurrence() -> None:
    """A hint marker is the end of the first exact `original`, not a later copy."""
    text = "Keep the tale, not that tale."
    block = _block(text)
    tale = GlossaryHint("tale", "сказка", "история")
    keep = GlossaryHint("Keep", "держать")
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(tale, keep),
    )

    notes = block_notes(block, result, None, ())

    assert notes == (
        LayoutNote((), "Keep", None, None, ("держать",), _end(text, "Keep"), "glossary_hint", ()),
        LayoutNote(
            (),
            "tale",
            None,
            None,
            ("сказка", "история"),
            _end(text, "tale"),
            "glossary_hint",
            (),
        ),
    )
    assert notes[1].marker_end == 13
    assert notes[1].marker_end != text.rfind("tale") + len("tale")


def test_note_without_occurrence_has_no_marker() -> None:
    """A missing excerpt stays in the list after every note that has a marker."""
    text = "The tale is told."
    block = _block(text)
    missing_hint = GlossaryHint("missing", "нет")
    present = LanguageNote("n-tale", block.id, "tale", "Про сказку.")
    absent = LanguageNote("n-absent", block.id, "never-seen", "Нет в тексте.")
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(missing_hint,),
        notes=(present, absent),
    )

    notes = block_notes(block, result, None, ())

    assert notes == (
        LayoutNote((), None, None, None, ("Про сказку.",), _end(text, "tale"), "language_note", ()),
        LayoutNote((), "missing", None, None, ("нет",), None, "glossary_hint", ()),
        LayoutNote((), None, None, None, ("Нет в тексте.",), None, "language_note", ()),
    )


def test_unresolved_assistance_omitted() -> None:
    """Unresolved help is dropped. A resolved marker is the last anchor end in this block."""
    text = "alpha beta gamma"
    block = _block(text)
    alpha = _anchor(block, "alpha")
    gamma = _anchor(block, "gamma")
    foreign = SourceAnchor("other-block", 0, 20, "z" * 20)
    beta = _anchor(block, "beta")
    resolved = _difficulty(block.id, (gamma, alpha), "grammar", difficulty_id="d-grammar")
    pending = _difficulty(block.id, (beta,), "wordplay", difficulty_id="d-wordplay")
    ready = _assistance(
        resolved,
        "Готовое объяснение.",
        anchors=(gamma, foreign, alpha),
    )
    skipped = _assistance(pending, None, kind="unresolved", anchors=(beta,))
    result = _result(block, assistances=(skipped, ready))

    notes = block_notes(block, result, _plan(resolved, pending), ())

    assert notes == (
        LayoutNote(
            (),
            None,
            None,
            None,
            ("Готовое объяснение.",),
            gamma.end,
            "assistance",
            ("grammar",),
        ),
    )
    assert gamma.end == 16
    assert gamma.end > alpha.end
    assert foreign.end > gamma.end


def test_language_reference_phrase_stripped_anywhere() -> None:
    """`Языковая справка:` is removed before `Языковая справка`, anywhere in a body piece."""
    text = "word phrase here"
    block = _block(text)
    hint = GlossaryHint("word", "Языковая справка:", "  Языковая справка  значение")
    note = LanguageNote(
        "n-phrase",
        block.id,
        "phrase",
        "До Языковая справка: середина, затем Языковая справка конец. "
        "Префикс Языковая справка:хвост. языковая справка",
    )
    difficulty = _difficulty(block.id, (_anchor(block, "here"),), "idiom")
    assistance = _assistance(
        difficulty,
        "Языковая справка без двоеточия и Языковая справка: с ним.",
    )
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(hint,),
        notes=(note,),
        assistances=(assistance,),
    )

    notes = block_notes(block, result, None, ())

    assert notes == (
        LayoutNote((), "word", None, None, ("значение",), _end(text, "word"), "glossary_hint", ()),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("До  середина, затем  конец. Префикс хвост. языковая справка",),
            _end(text, "phrase"),
            "language_note",
            (),
        ),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("без двоеточия и  с ним.",),
            _end(text, "here"),
            "assistance",
            (),
        ),
    )


def test_kind_label_from_help_evidence() -> None:
    """A label comes from the difficulty whose artifact key matches the hint or note."""
    assert tuple(KIND_LABELS) == (
        "wordplay",
        "grammar",
        "idiom",
        "cultural_reference",
        "semantic_relation",
        "false_friend",
        "speech_style",
        "parody",
        "vocabulary",
    )
    assert dict(KIND_LABELS) == {
        "wordplay": "Игра слов.",
        "grammar": None,
        "idiom": None,
        "cultural_reference": None,
        "semantic_relation": None,
        "false_friend": "Ложный друг.",
        "speech_style": "Речь персонажа.",
        "parody": "Пародия.",
        "vocabulary": None,
    }
    assert set(KIND_LABELS) == DIFFICULTY_KINDS

    text = "goose river culture bridge"
    block = _block(text)
    goose = GlossaryHint("goose", "гусыня", "не дурак")
    river = GlossaryHint("river", "река")
    culture = GlossaryHint("culture", "культура", "не то")
    bridge = LanguageNote("n-bridge", block.id, "bridge", "Мост в переносном смысле.", "мост")
    false_friend = _difficulty(block.id, (_anchor(block, "goose"),), "false_friend", difficulty_id="d-false")
    vocabulary = _difficulty(block.id, (_anchor(block, "river"),), "vocabulary", difficulty_id="d-vocab")
    cultural = _difficulty(
        block.id, (_anchor(block, "culture"),), "cultural_reference", difficulty_id="d-culture"
    )
    sense = _difficulty(
        block.id, (_anchor(block, "bridge"),), "semantic_relation", difficulty_id="d-sense"
    )
    plan = _plan(false_friend, vocabulary, cultural, sense)
    evidence = ContentEvidence(
        plan.sha256,
        (
            HelpEvidence(
                false_friend.id,
                false_friend.required_points[0].id,
                "glossary_hint.translation",
                block.id,
                _hint_key("glossary_hint.translation", goose, block.id),
                goose.translation,
            ),
            HelpEvidence(
                vocabulary.id,
                vocabulary.required_points[0].id,
                "glossary_hint.note",
                block.id,
                _hint_key("glossary_hint.note", river, block.id),
                river.translation,
            ),
            HelpEvidence(
                cultural.id,
                cultural.required_points[0].id,
                "glossary_hint.note",
                block.id,
                "wrong-key",
                culture.translation,
            ),
            HelpEvidence(
                sense.id,
                sense.required_points[0].id,
                "language_note.note",
                block.id,
                _note_key(bridge),
                bridge.note,
            ),
        ),
    )
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(goose, river, culture),
        notes=(bridge,),
        evidence=evidence,
    )

    notes = block_notes(block, result, plan, ())

    assert notes == (
        LayoutNote(
            ("Ложный друг.",),
            "goose",
            None,
            None,
            ("гусыня", "не дурак"),
            _end(text, "goose"),
            "glossary_hint",
            ("false_friend",),
        ),
        LayoutNote(
            (),
            "river",
            None,
            None,
            ("река",),
            _end(text, "river"),
            "glossary_hint",
            ("vocabulary",),
        ),
        LayoutNote(
            (),
            "culture",
            None,
            None,
            ("культура", "не то"),
            _end(text, "culture"),
            "glossary_hint",
            (),
        ),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("Мост в переносном смысле.",),
            _end(text, "bridge"),
            "language_note",
            ("semantic_relation",),
        ),
    )


def test_english_help_lexical_hint_has_no_label() -> None:
    """In English help, `lexical_hint` assistance has no label. Other assistance keeps its kind."""
    text = "Alice remarked the goose"
    block = _block(text)
    lexical = _difficulty(
        block.id,
        (_anchor(block, "remarked"),),
        "false_friend",
        difficulty_id="a-friend",
        help_kind="lexical_hint",
    )
    grammar = _difficulty(
        block.id,
        (_anchor(block, "goose"),),
        "grammar",
        difficulty_id="b-grammar",
    )
    lexical_help = _assistance(lexical, "похоже на замечание, значит другое", kind="lexical_hint")
    other_help = _assistance(grammar, "гусыня, не «дура»")
    result = _result(block, assistances=(lexical_help, other_help))
    plan = _plan(lexical, grammar)

    hidden = block_notes(block, result, plan, (), english_help=True)
    shown = block_notes(block, result, plan, (), english_help=False)

    assert hidden == (
        LayoutNote(
            (),
            None,
            None,
            None,
            ("похоже на замечание, значит другое",),
            _end(text, "remarked"),
            "assistance",
            ("false_friend",),
        ),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("гусыня, не «дура»",),
            _end(text, "goose"),
            "assistance",
            ("grammar",),
        ),
    )
    assert shown[0].labels == ("Ложный друг.",)
    assert shown[1].labels == ()


def test_pronunciation_merged_into_matching_hint() -> None:
    """A pronunciation whose quote equals a hint fills that hint and does not add a note."""
    text = "tale, then tale and tail"
    block = _block(text)
    hint = GlossaryHint("tale", "сказка")
    result = _result(block, treatment=BlockTreatment.GLOSSARY_ONLY, hints=(hint,))
    first = _pron(block, "tale", "/teɪl/", "uk", start=text.find("tale", 1))
    extra = _pron(block, "tale", "/tɛl/", "us", start=0)
    tail = _pron(block, "tail", "/teɪl/", "us")
    foreign = _pron(block, "ghost", "/ɡəʊst/", "uk", start=0, block_id="other-block")

    notes = block_notes(block, result, None, (first, extra, tail, foreign))

    assert notes == (
        LayoutNote((), "tale", "/teɪl/", "uk", ("сказка",), _end(text, "tale"), "glossary_hint", ()),
        LayoutNote((), "tail", "/teɪl/", "us", (), _end(text, "tail"), "pronunciation", ()),
    )
    assert notes[0].marker_end == 4
    assert first.end == 15


def test_identical_anchor_difficulties_merge_labels() -> None:
    """Equal anchor tuples are one note. Labels follow plan order and skip vocabulary."""
    text = "Curiouser and curiouser, said Alice."
    block = _block(text)
    shared = _anchor(block, "Curiouser and curiouser")
    nearby = _anchor(block, "Alice")
    grammar = _difficulty(block.id, (shared,), "grammar", difficulty_id="a-grammar")
    vocabulary = _difficulty(block.id, (shared,), "vocabulary", difficulty_id="b-vocabulary")
    wordplay = _difficulty(block.id, (shared,), "wordplay", difficulty_id="c-wordplay")
    idiom = _difficulty(block.id, (nearby,), "idiom", difficulty_id="d-idiom")
    assistances = (
        _assistance(wordplay, "Нарочно ломает слово."),
        _assistance(idiom, "Это не идиома рядом."),
        _assistance(vocabulary, "Любопытнее."),
        _assistance(grammar, "Неправильная степень сравнения."),
    )
    result = _result(block, assistances=assistances)

    notes = block_notes(block, result, _plan(grammar, vocabulary, wordplay, idiom), ())

    assert notes == (
        LayoutNote(
            ("Игра слов.",),
            None,
            None,
            None,
            ("Нарочно ломает слово.", "Любопытнее.", "Неправильная степень сравнения."),
            shared.end,
            "assistance",
            ("grammar", "vocabulary", "wordplay"),
        ),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("Это не идиома рядом.",),
            nearby.end,
            "assistance",
            ("idiom",),
        ),
    )
    assert all("," not in label for note in notes for label in note.labels)


def test_editorial_self_check_and_redundant_agreement_hint_note_stripped() -> None:
    """Self-check tails drop off. A hint note that only restates agreement is discarded."""
    text = "Yes the phrase holds here"
    block = _block(text)
    agreement = GlossaryHint(
        "Yes",
        "да",
        "разговорное подтверждение чужих слов кивком",
    )
    refers = GlossaryHint(
        "holds",
        "держит",
        "говорящий соглашается и ссылается на реплику собеседника",
    )
    kept = GlossaryHint(
        "phrase",
        "фраза",
        "устойчивый оборот; вариант стоит проверить",
    )
    note = LanguageNote(
        "n-here",
        block.id,
        "here",
        "Середина. Вариант стоит проверить. Конец; вариант стоит проверить",
    )
    difficulty = _difficulty(block.id, (_anchor(block, "the"),), "grammar")
    assistance = _assistance(difficulty, "Артикль не переводится. Вариант стоит проверить.")
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(agreement, refers, kept),
        notes=(note,),
        assistances=(assistance,),
    )

    notes = block_notes(block, result, _plan(difficulty), ())

    assert notes == (
        LayoutNote((), "Yes", None, None, ("да",), _end(text, "Yes"), "glossary_hint", ()),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("Артикль не переводится",),
            _end(text, "the"),
            "assistance",
            ("grammar",),
        ),
        LayoutNote(
            (),
            "phrase",
            None,
            None,
            ("фраза", "устойчивый оборот"),
            _end(text, "phrase"),
            "glossary_hint",
            (),
        ),
        LayoutNote(
            (),
            "holds",
            None,
            None,
            ("держит",),
            _end(text, "holds"),
            "glossary_hint",
            (),
        ),
        LayoutNote(
            (),
            None,
            None,
            None,
            ("Середина. Вариант стоит проверить. Конец",),
            _end(text, "here"),
            "language_note",
            (),
        ),
    )


def test_translate_block_drops_redundant_sentence_term_and_translation_in_note() -> None:
    """A translated block hides a long sentence and a translation already quoted in the note."""
    rumbling = "A low rumbling sound had broken the silence around them."
    passage = (
        "not knowing he was special, not knowing he was famous, not knowing he would be "
        "woken in a few hours by Mrs Dursley's scream as she opened the front door to put "
        "out the milk bottles"
    )
    thanks = "thank you very much"
    five = "silence around them at night"
    six = "silence around them at the night"
    assert len(expression_words(rumbling)) == 10
    assert len(expression_words(passage)) > 5
    assert len(expression_words(thanks)) == 4
    assert len(expression_words(five)) == 5
    assert len(expression_words(six)) == 6
    text = " ".join((rumbling, passage, thanks, five, six))
    block = _block(text)
    rumbling_hint = GlossaryHint(rumbling, "Низкий рокот нарушил тишину.", "Звук шёл из-за двери.")
    passage_hint = GlossaryHint(passage, "не зная, что он особенный", "Гарри ещё не знает о себе.")
    thanks_hint = GlossaryHint(
        thanks,
        "уж поверьте",
        "Ироничная присказка со значением «Уж Поверьте», а не благодарность.",
    )
    five_hint = GlossaryHint(five, "коротко", "Пояснение к короткой фразе.")
    six_hint = GlossaryHint(six, "чуть длиннее", "Пояснение к фразе из шести слов.")
    result = _result(
        block,
        treatment=BlockTreatment.TRANSLATE,
        translation="Полный перевод абзаца.",
        hints=(rumbling_hint, passage_hint, thanks_hint, five_hint, six_hint),
    )

    notes = block_notes(block, result, None, ())

    assert [note.term for note in notes] == [None, None, thanks, five, None]
    assert notes[0].body == ("Звук шёл из-за двери.",)
    assert "Низкий рокот" not in "".join(notes[0].body)
    assert notes[0].marker_end == _end(text, rumbling)
    assert notes[1].body == ("Гарри ещё не знает о себе.",)
    assert notes[1].marker_end == _end(text, passage)
    assert notes[2].body == (
        "Ироничная присказка со значением «Уж Поверьте», а не благодарность.",
    )
    assert "".join(notes[2].body).casefold().count("уж поверьте") == 1
    assert notes[3].body == ("коротко", "Пояснение к короткой фразе.")
    assert notes[4].body == ("Пояснение к фразе из шести слов.",)
    assert "чуть длиннее" not in "".join(notes[4].body)
    glossary_block = _block(rumbling)
    glossary = _result(
        glossary_block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(rumbling_hint,),
    )
    kept = block_notes(glossary_block, glossary, None, ())
    assert kept[0].term == rumbling
    assert kept[0].body == ("Низкий рокот нарушил тишину.", "Звук шёл из-за двери.")


def test_verb_glossary_hint_normalized_to_infinitive_and_redundant_definition_dropped() -> None:
    """Past-tense verbs become `to` infinitives; a repeated headword gloss and speech label go."""
    text = "They sniggered then nodded vigorously and spoke stiffly."
    block = _block(text)
    sniggered = GlossaryHint(
        "sniggered",
        "хихикнули, засмеялись с ехидством",
        "Snigger — хихикать, смеяться с ехидством",
    )
    nodded = GlossaryHint("nodded", "кивнула", "Nod — кивнуть")
    vigorously = GlossaryHint("vigorously", "энергично", "Vigorously — энергично, сильно")
    stiffly = GlossaryHint("stiffly", "сухо", "Помета к глаголу речи: сказала сдержанно")
    vocabulary = _difficulty(
        block.id, (_anchor(block, "sniggered"),), "vocabulary", difficulty_id="d-snigger"
    )
    plan = _plan(vocabulary)
    evidence = ContentEvidence(
        plan.sha256,
        (
            HelpEvidence(
                vocabulary.id,
                vocabulary.required_points[0].id,
                "glossary_hint.translation",
                block.id,
                _hint_key("glossary_hint.translation", sniggered, block.id),
                sniggered.translation,
            ),
        ),
    )
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(sniggered, nodded, vigorously, stiffly),
        evidence=evidence,
    )
    pronunciation = _pron(block, "sniggered", "/ˈsnɪɡəd/", "uk")

    notes = block_notes(block, result, plan, (pronunciation,))

    assert [note.term for note in notes] == ["to snigger", "to nod", "vigorously", "stiffly"]
    assert notes[0].body == ("хихикать, засмеяться с ехидством",)
    assert "Snigger" not in "".join(notes[0].body)
    assert notes[0].marker_end == _end(text, "sniggered")
    assert notes[0].ipa == "/ˈsnɪɡəd/"
    assert notes[0].difficulty_kinds == ("vocabulary",)
    assert notes[0].labels == ()
    assert notes[1].body == ("кивнуть",)
    assert notes[1].marker_end == _end(text, "nodded")
    assert notes[2].body == ("энергично",)
    assert "Vigorously" not in "".join(notes[2].body)
    assert notes[3].body == ("сухо", "сказала сдержанно")
    assert "Помета" not in "".join(notes[3].body)
    samples = (
        ("gasped", "ахнула", "to gasp", "ахнуть"),
        ("moaned", "простонал", "to moan", "простонать"),
        ("sniffed", "шмыгнула", "to sniff", "шмыгнуть"),
        ("peered", "всмотрелась", "to peer", "всмотреться"),
        ("winked", "подмигнула", "to wink", "подмигнуть"),
        ("groaned", "охнул", "to groan", "охнуть"),
        ("chuckled", "хихикнул", "to chuckle", "хихикать"),
        ("shuffled away", "поплёлся прочь", "to shuffle away", "поплестись прочь"),
        ("walked off", "пошёл прочь", "to walk off", "пойти прочь"),
        ("walked off", "пошел прочь", "to walk off", "пойти прочь"),
        ("shuffled away", "поплелся", "to shuffle away", "поплестись"),
        ("heaved", "тяжело вздохнул", "to heave", "тяжело вздохнуть"),
        ("glanced", "заглянул в зеркало", "to glance", "заглянуть в зеркало"),
        ("nudged", "толкнул локтем", "to nudge", "толкнуть локтем"),
        ("squeezed", "спустился в подвал", "to squeeze", "спуститься в подвал"),
        ("paused", "подал сигнал", "to pause", "подать сигнал"),
        ("escaped", "услышал выстрел", "to escape", "услышать выстрел"),
        ("hesitated", "замешкался", "to hesitate", "замешкаться"),
        ("bewildered", "озадаченный, растерянный", "bewildered", "озадаченный, растерянный"),
        ("frightened", "испуганный", "frightened", "испуганный"),
        ("wicked", "злой, коварный", "wicked", "злой, коварный"),
        ("pointed", "остроконечный", "pointed", "остроконечный"),
        ("hundred", "сотня", "hundred", "сотня"),
    )
    for original, translation, term, infinitive in samples:
        sample_text = f"Then {original} here."
        sample_block = _block(sample_text)
        sample = block_notes(
            sample_block,
            _result(
                sample_block,
                treatment=BlockTreatment.GLOSSARY_ONLY,
                hints=(GlossaryHint(original, translation),),
            ),
            None,
            (),
        )
        assert sample[0].term == term
        assert sample[0].body == (infinitive,)
        assert sample[0].marker_end == _end(sample_text, original)
    played = "They sniggered."
    played_block = _block(played)
    played_hint = GlossaryHint("sniggered", "хихикнули", "Snigger — толкование")
    wordplay = _difficulty(
        played_block.id,
        (_anchor(played_block, "sniggered"),),
        "wordplay",
        difficulty_id="d-play",
    )
    played_plan = _plan(wordplay)
    played_result = _result(
        played_block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(played_hint,),
        evidence=ContentEvidence(
            played_plan.sha256,
            (
                HelpEvidence(
                    wordplay.id,
                    wordplay.required_points[0].id,
                    "glossary_hint.note",
                    played_block.id,
                    _hint_key("glossary_hint.note", played_hint, played_block.id),
                    played_hint.note or "",
                ),
                HelpEvidence(
                    wordplay.id,
                    wordplay.required_points[0].id,
                    "glossary_hint.translation",
                    played_block.id,
                    _hint_key("glossary_hint.translation", played_hint, played_block.id),
                    played_hint.translation,
                ),
            ),
        ),
    )
    preserved = block_notes(played_block, played_result, played_plan, ())
    assert preserved[0].term == "sniggered"
    assert preserved[0].body == ("хихикнули", "Snigger — толкование")
    assert preserved[0].labels == ("Игра слов.",)


def test_infinitive_and_case_insensitive_glossary_hints_find_marker_end_in_text() -> None:
    """`to` + verb marks the inflected phrase. A different case still needs a word boundary."""
    flutter = "None of them noticed a large tawny owl flutter past the window."
    flutter_block = _block(flutter)
    flutter_hint = GlossaryHint("to flutter", "пролететь", "flutter — промелькнуть")
    flutter_notes = block_notes(
        flutter_block,
        _result(
            flutter_block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(flutter_hint,),
        ),
        None,
        (_pron(flutter_block, "to flutter", "/ˈflʌtə/", "uk", start=0),),
    )
    assert flutter_notes == (
        LayoutNote(
            (),
            "to flutter",
            "/ˈflʌtə/",
            "uk",
            ("пролететь",),
            _span_end(flutter, "flutter"),
            "glossary_hint",
            (),
        ),
    )

    hugged = "And the old man hugged Mr Dursley around the middle and walked off."
    assert [
        (note.term, note.marker_end) for note in _hint_notes(
            hugged,
            GlossaryHint("to hug", "обнять"),
            GlossaryHint("to walk off", "уйти"),
        )
    ] == [
        ("to hug", _span_end(hugged, "hugged")),
        ("to walk off", _span_end(hugged, "walked off")),
    ]

    bowed = "Dumbledore bowed his head. Professor McGonagall gasped."
    assert [
        (note.term, note.marker_end) for note in _hint_notes(
            bowed,
            GlossaryHint("to bow one's head", "склонить голову"),
            GlossaryHint("to gasp", "ахнуть"),
        )
    ] == [
        ("to bow one's head", _span_end(bowed, "bowed his head")),
        ("to gasp", _span_end(bowed, "gasped")),
    ]

    grab = "Harry made a grab for the letter but Uncle Vernon knocked his hand out of the way."
    assert [
        (note.term, note.marker_end) for note in _hint_notes(
            grab,
            GlossaryHint("to make a grab for", "схватить"),
            GlossaryHint("to knock out of the way", "отвести в сторону"),
        )
    ] == [
        ("to make a grab for", _span_end(grab, "made a grab for")),
        ("to knock out of the way", _span_end(grab, "knocked his hand out of the way")),
    ]

    upright = "The whole shack shivered and Harry sat bolt upright, staring at the door."
    upright_notes = _hint_notes(upright, GlossaryHint("to sit bolt upright", "сесть прямо"))
    assert upright_notes[0].term == "to sit bolt upright"
    assert upright_notes[0].marker_end == _span_end(upright, "sat bolt upright")
    assert upright[upright_notes[0].marker_end] == ","

    thrown_to = "Hagrid threw it to Harry."
    thrown_to_notes = _hint_notes(thrown_to, GlossaryHint("to throw to", "бросить"))
    assert thrown_to_notes[0].term == "to throw to"
    assert thrown_to_notes[0].marker_end == _span_end(thrown_to, "threw it to")
    assert thrown_to[thrown_to_notes[0].marker_end] == " "

    write_down = "Could you write it down?"
    write_down_notes = _hint_notes(write_down, GlossaryHint("to write down", "записать"))
    assert write_down_notes[0].term == "to write down"
    assert write_down_notes[0].marker_end == _span_end(write_down, "write it down")
    assert write_down[write_down_notes[0].marker_end] == "?"

    surprise = "To his surprise, Hagrid chuckled."
    surprise_notes = _hint_notes(surprise, GlossaryHint("to his surprise", "к его удивлению"))
    assert surprise_notes[0].term == "to his surprise"
    assert surprise_notes[0].marker_end == _span_end(surprise, "To his surprise")
    assert surprise[surprise_notes[0].marker_end] == ","

    buried = "He walked INTO his surprise party."
    assert _hint_notes(buried, GlossaryHint("to his surprise", "к его удивлению"))[0].marker_end is None

    look = "Harry threw Uncle Vernon a dirty look."
    look_notes = _hint_notes(look, GlossaryHint("to throw a dirty look at", "сердитый взгляд"))
    assert look_notes[0].term == "to throw a dirty look at"
    assert look_notes[0].marker_end == _span_end(look, "threw Uncle Vernon a dirty look")

    samples = (
        ("The owl flutters past.", "to flutter", "flutters"),
        ("Hagrid held the door.", "to hold", "held"),
        ("Harry took the letter.", "to take", "took"),
        ("The stone was thrown aside.", "to throw", "thrown"),
        ("It was written there.", "to write", "written"),
        ("The light was fading fast.", "to fade", "fading"),
        ("Harry nodded once.", "to nod", "nodded"),
        ("Dudley snivelled loudly.", "to snivel", "snivelled"),
        ("They expelled the boy.", "to expel", "expelled"),
    )
    for text, original, surface in samples:
        sample = _hint_notes(text, GlossaryHint(original, "форма"))
        assert sample[0].term == original
        assert sample[0].marker_end == _span_end(text, surface)


def test_hint_note_strips_unquoted_leading_translation_duplicate() -> None:
    """A note that repeats the translation before punctuation does not show it twice."""
    text = "Dumbledore mentioned the London Underground."
    block = _block(text)
    hint = GlossaryHint(
        "the London Underground",
        "Лондонское метро",
        "Лондонское метро. Дамблдор шутит, что шрам у него на колене похож на схему линий метро.",
    )
    difficulty = _difficulty(
        block.id,
        (_anchor(block, "the London Underground"),),
        "cultural_reference",
        difficulty_id="d-tube",
    )
    plan = _plan(difficulty)
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(hint,),
        evidence=ContentEvidence(
            plan.sha256,
            (
                HelpEvidence(
                    difficulty.id,
                    difficulty.required_points[0].id,
                    "glossary_hint.note",
                    block.id,
                    _hint_key("glossary_hint.note", hint, block.id),
                    hint.note or "",
                ),
            ),
        ),
    )

    notes = block_notes(block, result, plan, ())

    assert notes[0].body == (
        "Лондонское метро",
        "Дамблдор шутит, что шрам у него на колене похож на схему линий метро.",
    )
    assert "Лондонское метро — Лондонское метро" not in " — ".join(notes[0].body)
    assert notes[0].difficulty_kinds == ("cultural_reference",)
    assert notes[0].marker_end == _span_end(text, "the London Underground")

    same = _hint_notes("line", GlossaryHint("line", "Лондонское метро", "Лондонское метро"))
    assert same[0].body == ("Лондонское метро",)
    shouted = _hint_notes(
        "tube",
        GlossaryHint("tube", "Лондонское метро", "лондонское метро! Схема линий."),
    )
    assert shouted[0].body == ("Лондонское метро", "Схема линий.")
    dashed = _hint_notes(
        "map",
        GlossaryHint("map", "Лондонское метро", "Лондонское метро — схема линий."),
    )
    assert dashed[0].body == ("Лондонское метро", "схема линий.")
    for mark in (".", "!", "?", "—", "–", "-", ":", ";"):
        marked = _hint_notes(
            "stop",
            GlossaryHint("stop", "Лондонское метро", f"Лондонское метро{mark} дальше"),
        )
        assert marked[0].body == ("Лондонское метро", "дальше")
    prefix = _hint_notes("station", GlossaryHint("station", "метро", "метрополитен в Лондоне."))
    assert prefix[0].body == ("метро", "метрополитен в Лондоне.")


def test_body_strips_unconfirmed_wordplay_disclaimer_and_fixes_are_there_ellipsis() -> None:
    """An unconfirmed wordplay disclaimer drops out. The wrong tag label becomes Are there."""
    text = "Diagon Alley filled with showers of owls."
    block = _block(text)
    diagon = GlossaryHint(
        "Diagon Alley",
        "Косой переулок",
        "Название похоже на diagonally. "
        "Это можно заметить как возможную игру слов, но текст книги не подтверждает её прямо. "
        "Читается как diagonally, но текст книги не подтверждает её прямо.",
    )
    owls = GlossaryHint(
        "showers of owls",
        "потоки сов",
        "Вопрос построен без is it: Going to be any more showers of owls tonight, Harry?",
    )
    ellipsis = LanguageNote(
        "n-owls",
        block.id,
        "owls",
        "Вопрос построен без is it, поэтому перевод добавляет «разве».",
    )
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(diagon, owls),
        notes=(ellipsis,),
    )

    notes = block_notes(block, result, None, ())

    assert [note.body for note in notes] == [
        ("Косой переулок", "Название похоже на diagonally. Читается как diagonally."),
        (
            "потоки сов",
            "Вопрос построен без Are there: Going to be any more showers of owls tonight, Harry?",
        ),
        ("Вопрос построен без Are there, поэтому перевод добавляет «разве».",),
    ]
    flat = " ".join(" ".join(note.body) for note in notes)
    assert "не подтверждает её прямо" not in flat
    assert "без is it" not in flat
