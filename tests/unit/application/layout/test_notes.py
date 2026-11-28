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


def test_infinitive_wordplay_hint_keeps_gloss_and_label() -> None:
    """A linked wordplay difficulty keeps a bare-headword gloss on `to` + verb."""
    text = "None of them noticed a large tawny owl flutter past the window."
    block = _block(text)
    hint = GlossaryHint("to flutter", "пролететь", "flutter — толкование")
    wordplay = _difficulty(
        block.id,
        (_anchor(block, "flutter"),),
        "wordplay",
        difficulty_id="d-flutter",
    )
    plan = _plan(wordplay)
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(hint,),
        evidence=ContentEvidence(
            plan.sha256,
            (
                HelpEvidence(
                    wordplay.id,
                    wordplay.required_points[0].id,
                    "glossary_hint.note",
                    block.id,
                    _hint_key("glossary_hint.note", hint, block.id),
                    hint.note or "",
                ),
                HelpEvidence(
                    wordplay.id,
                    wordplay.required_points[0].id,
                    "glossary_hint.translation",
                    block.id,
                    _hint_key("glossary_hint.translation", hint, block.id),
                    hint.translation,
                ),
            ),
        ),
    )

    notes = block_notes(block, result, plan, ())

    assert notes[0].term == "to flutter"
    assert notes[0].body == ("пролететь", "flutter — толкование")
    assert notes[0].labels == ("Игра слов.",)
    assert notes[0].marker_end == _span_end(text, "flutter")

    from linguary.application.llm_dto import HelpEvidenceResponse, TranslationResultResponse
    from linguary.application.validators.content_plan import (
        resolve_content_coverage,
        resolve_response_evidence,
    )

    resolved_evidence = resolve_response_evidence(
        plan,
        TranslationResultResponse(
            content_evidence=(
                HelpEvidenceResponse(
                    wordplay.id,
                    wordplay.required_points[0].id,
                    "glossary_hint.note",
                    block.id,
                    0,
                    "flutter — толкование",
                ),
            ),
        ),
        (result,),
        (),
        owner_block_ids=(block.id,),
    )
    checked, unresolved = resolve_content_coverage(
        plan,
        (result,),
        (),
        resolved_evidence,
        owner_block_ids=(block.id,),
        source_texts={block.id: text},
        validators_version=4,
    )
    assert len(checked) == 1
    assert unresolved[block.id] == ()


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

    # Surface pronunciation quote ("flutter", "To his surprise") pairs with infinitive/folded hint.
    surface_flutter_notes = block_notes(
        flutter_block,
        _result(
            flutter_block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(flutter_hint,),
        ),
        None,
        (_pron(flutter_block, "flutter", "/ˈflʌtə/", "uk"),),
    )
    assert len(surface_flutter_notes) == 1
    assert surface_flutter_notes[0].term == "to flutter"
    assert surface_flutter_notes[0].ipa == "/ˈflʌtə/"

    surprise_block = _block(surprise)
    surprise_paired = block_notes(
        surprise_block,
        _result(
            surprise_block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(GlossaryHint("to his surprise", "к его удивлению"),),
        ),
        None,
        (_pron(surprise_block, "To his surprise", "/səˈpraɪz/", "uk"),),
    )
    assert len(surprise_paired) == 1
    assert surprise_paired[0].term == "to his surprise"
    assert surprise_paired[0].ipa == "/səˈpraɪz/"

    # Shifted-object matching requires an actual indirect object before the phrase middle.
    grab_twice = "Harry made a grab, then he made a grab for the letter."
    grab_twice_notes = _hint_notes(grab_twice, GlossaryHint("to make a grab for", "схватить"))
    assert grab_twice_notes[0].marker_end == _span_end(grab_twice, "made a grab for")


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


def test_hint_markers_match_apostrophes_irregular_verbs_short_doubling_and_gaps() -> None:
    """Straight apostrophes, irregular verbs, 3-letter doubled verbs, and gapped quotes all get markers."""
    cases = (
        ("‘The Potters, that’s right, that’s what I heard –’", "that's right", "верно", "that’s right"),
        ("‘The Potters, that’s right, that’s what I heard –’", "that's what I heard", "вот что я услышал", "that’s what I heard"),
        ("‘What’s up?’ said Hagrid.", "What's up?", "Что случилось?", "What’s up?"),
        ("‘Years an’ years ago.’", "Years an' years ago", "Много-много лет назад", "Years an’ years ago"),
        ("He was back in the room.", "to be back", "вернуться", "was back"),
        ("The giant shook his head.", "to shake", "трясти", "shook"),
        ("Uncle Vernon found his voice.", "to find one's voice", "обрести дар речи", "found his voice"),
        ("Aunt Petunia gave a gasp.", "to give a gasp", "ахнуть", "gave a gasp"),
        ("He heard a tapping at the window.", "to tap", "стучать", "tapping"),
        ("‘But what does a Ministry of Magic do?’", "what does … do", "чем занимается …", "what does a Ministry of Magic do"),
    )
    for text, original, translation, matched_span in cases:
        block = _block(text)
        result = _result(
            block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(GlossaryHint(original, translation),),
        )
        notes = block_notes(block, result, None, ())
        assert notes[0].marker_end == _end(text, matched_span), (original, text)


def test_hint_strips_duplicate_headword_declension_directives_and_soft_hyphen() -> None:
    """Duplicate headwords in translation, leaked declension directives, and soft hyphens are cleaned."""
    text = "He singled out Little Whinging, Marge, and Mr Potter."
    block = _block(text)
    hints = (
        GlossaryHint("to single out", "to single out — выделить, отделить"),
        GlossaryHint("Little Whinging", "Литл-Уинджинг", "Название города; не склонять."),
        GlossaryHint("Marge", "Мардж", "Имя не склоняется."),
        GlossaryHint("Mr Potter", "мистер Поттер", "При слове «мистер» фамилия Поттер не склоняется."),
    )
    soft_hyphen_note = LanguageNote("n-soft", block.id, "singled out", "Может требовать\u00adся пояснение.")
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=hints,
        notes=(soft_hyphen_note,),
    )

    notes = block_notes(block, result, None, ())
    by_term = {note.term: note.body for note in notes if note.term}
    assert by_term["to single out"] == ("выделить, отделить",)
    assert by_term["Little Whinging"] == ("Литл-Уинджинг", "Название города.")
    assert by_term["Marge"] == ("Мардж",)
    assert by_term["Mr Potter"] == ("мистер Поттер",)
    note_bodies = [note.body for note in notes if note.origin == "language_note"]
    assert note_bodies == [("Может требоваться пояснение.",)]


def test_same_verb_and_find_end_reject_false_positives_and_cross_sentence_gaps() -> None:
    """Irregular/short verbs do not match unrelated nouns; gapped quotes stay in one sentence."""
    # False-positive verb matches must NOT bind markers.
    false_pairs = (
        ("He lay in bed in the dark.", "to be"),
        ("Wild bees buzzed loudly.", "to be"),
        ("Thank God for that.", "to go"),
        ("The cold wind blew hard.", "to win"),
        ("A tiny seed fell down.", "to see"),
        ("A brass band played nearby.", "to ban"),
        ("He walked into the ward.", "to war"),
        ("The bard sang softly.", "to bar"),
        ("She’d already left the room.", "he'd"),
        ("He didn’t shout!", "out!"),
        ("She’s right?", "he’s?"),
        ("‘hello’", "‘he"),
        ("He was hoping for rain.", "to hop"),
        ("She stared across the hall.", "to star"),
        ("They scared the birds away.", "to scar"),
        ("He taped the box shut.", "to tap"),
        ("‘I don’t know.’", "don'"),
    )
    for text, original in false_pairs:
        block = _block(text)
        result = _result(
            block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(GlossaryHint(original, "перевод"),),
        )
        assert block_notes(block, result, None, ())[0].marker_end is None, (original, text)

    # Trailing-apostrophe truncation (`don'`) skips `don’t` and matches standalone `don’`.
    trunc_text = "‘I don’t know — don’ ask me.’"
    trunc_block = _block(trunc_text)
    trunc_res = _result(
        trunc_block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(GlossaryHint("don'", "не"),),
    )
    assert block_notes(trunc_block, trunc_res, None, ())[0].marker_end == trunc_text.rfind("don’") + len("don’")

    # Regular/dual-paradigm verb forms (including US/UK spelling and vowel+o) bind markers cleanly.
    verb_pairs = (
        ("He sniveled quietly.", "to snivel", "sniveled"),
        ("He snivelled quietly.", "to snivel", "snivelled"),
        ("He lied to Harry.", "to lie", "lied"),
        ("He shined his boots.", "to shine", "shined"),
        ("She waked at dawn.", "to wake", "waked"),
        ("He had stridden across.", "to stride", "stridden"),
        ("They expelled him.", "to expel", "expelled"),
        ("She hugged him.", "to hug", "hugged"),
        ("He nodded twice.", "to nod", "nodded"),
        ("The light faded fast.", "to fade", "faded"),
        ("The dove coos softly.", "to coo", "coos"),
        ("He hopped over the wall.", "to hop", "hopped"),
    )
    for text, original, matched in verb_pairs:
        block = _block(text)
        result = _result(
            block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(GlossaryHint(original, "перевод"),),
        )
        assert block_notes(block, result, None, ())[0].marker_end == _end(text, matched), (
            original,
            text,
        )

    # Gapped quote must NOT jump across a sentence period, but DOES handle inline punctuation/quotes.
    two_sentences = "He did not. Later he went to the shop."
    gap_block = _block(two_sentences)
    gap_res = _result(
        gap_block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(GlossaryHint("not … to", "не … чтобы"),),
    )
    assert block_notes(gap_block, gap_res, None, ())[0].marker_end is None

    quoted_gap = "‘But what does ‘Ministry of Magic’ do?’"
    q_block = _block(quoted_gap)
    q_res = _result(
        q_block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(GlossaryHint("what does … do", "чем занимается"),),
    )
    assert block_notes(q_block, q_res, None, ())[0].marker_end == _end(
        quoted_gap, "what does ‘Ministry of Magic’ do"
    )


def test_headword_stripping_preserves_hyphenated_compounds_and_empty_notes_drop_before_numbering() -> None:
    """Hyphenated compounds stay intact; leading translation strips cleanly; empty notes vanish."""
    text = "She wore a mini skirt and singled out Potter."
    block = _block(text)
    hints = (
        GlossaryHint("mini", "mini-юбка"),
        GlossaryHint("to single out", "to single out — выделить", "Выделить. Особо отметить."),
        GlossaryHint("singled out", "to single out — выделил", "Выделил. Второе замечание."),
        GlossaryHint("Potter", "Поттер", "Уменьшительная форма. Фамилия Поттер не склоняется."),
    )
    empty_note = LanguageNote("n-empty", block.id, "Potter", "Имя не склоняется.")
    real_note = LanguageNote("n-real", block.id, "singled out", "Фразовый глагол.")
    wordplay_diff = _difficulty(block.id, (_anchor(block, "Potter"),), "wordplay", difficulty_id="d-empty-wp")
    wp_plan = _plan(wordplay_diff)
    wp_evidence = ContentEvidence(
        wp_plan.sha256,
        (
            HelpEvidence(
                wordplay_diff.id,
                wordplay_diff.required_points[0].id,
                "language_note.note",
                block.id,
                artifact_key(
                    "language_note.note",
                    empty_note.block_id,
                    {
                        "original_excerpt": empty_note.original_excerpt,
                        "note": empty_note.note,
                        "translated_excerpt": empty_note.translated_excerpt,
                    },
                ),
                empty_note.note,
            ),
        ),
    )
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=hints,
        notes=(empty_note, real_note),
        evidence=wp_evidence,
    )

    notes = block_notes(block, result, wp_plan, ())
    single_out_notes = [note.body for note in notes if note.term == "to single out"]
    assert single_out_notes == [
        ("выделить", "Особо отметить."),
        ("выделить", "Второе замечание."),
    ]
    by_term = {note.term: note.body for note in notes if note.term}
    assert by_term["mini"] == ("mini-юбка",)
    assert by_term["Potter"] == ("Поттер", "Уменьшительная форма.")
    # Empty LanguageNote is dropped even if it carried a kind label (wordplay), leaving only real_note.
    lang_notes = [note for note in notes if note.origin == "language_note"]
    assert [note.body for note in lang_notes] == [("Фразовый глагол.",)]


def test_block_notes_respects_suppressed_and_stripped_indices_preserving_artifact_key() -> None:
    """Suppressed indices are skipped; stripped hints drop note text yet keep artifact_key linkage."""
    text = "The goose met the lantern by the cloak."
    block = _block(text)
    goose = GlossaryHint("goose", "гусыня", "Не путать со значением «глупец».")
    lantern = GlossaryHint("lantern", "фонарь", "Переносной фонарь с ручкой.")
    cloak = GlossaryHint("cloak", "плащ", "Длинная накидка с капюшоном.")
    kept_note = LanguageNote("n-kept", block.id, "met", "Прошедшее время от meet.")
    suppressed_note = LanguageNote("n-drop", block.id, "by", "Предлог места.")

    false_friend = _difficulty(
        block.id,
        (_anchor(block, "goose"),),
        "false_friend",
        difficulty_id="d-goose",
    )
    vocab = _difficulty(
        block.id,
        (_anchor(block, "lantern"),),
        "vocabulary",
        difficulty_id="d-lantern",
    )
    plan = _plan(false_friend, vocab)
    evidence = ContentEvidence(
        plan.sha256,
        (
            HelpEvidence(
                false_friend.id,
                false_friend.required_points[0].id,
                "glossary_hint.note",
                block.id,
                _hint_key("glossary_hint.note", goose, block.id),
                goose.note or "",
            ),
            HelpEvidence(
                vocab.id,
                vocab.required_points[0].id,
                "glossary_hint.translation",
                block.id,
                _hint_key("glossary_hint.translation", lantern, block.id),
                lantern.translation,
            ),
        ),
    )
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(goose, lantern, cloak),
        notes=(kept_note, suppressed_note),
        evidence=evidence,
    )

    notes = block_notes(
        block,
        result,
        plan,
        (),
        suppressed_hint_indices=frozenset({2}),
        stripped_hint_indices=frozenset({0}),
        suppressed_note_indices=frozenset({1}),
    )

    assert [note.term for note in notes] == ["goose", None, "lantern"]
    # Stripped hint dropped explanatory note text, yet preserved artifact_key -> labels & kinds.
    assert notes[0].body == ("гусыня",)
    assert notes[0].labels == ("Ложный друг.",)
    assert notes[0].difficulty_kinds == ("false_friend",)
    # Kept language note remains; suppressed language note is omitted.
    assert notes[1].origin == "language_note"
    assert notes[1].body == ("Прошедшее время от meet.",)
    # Non-stripped hint keeps full translation and note; suppressed hint (index 2) is omitted.
    assert notes[2].body == ("фонарь", "Переносной фонарь с ручкой.")
    assert notes[2].difficulty_kinds == ("vocabulary",)
    # Stored BlockProcessingResult is never mutated.
    assert result.glossary_hints == (goose, lantern, cloak)
    assert result.language_notes == (kept_note, suppressed_note)


def test_suppressed_hint_does_not_leak_paired_pronunciation_as_standalone_note() -> None:
    """Pronunciations matching a suppressed glossary hint do not become standalone IPA notes."""
    text = "The cloak rustled near the owl."
    block = _block(text)
    cloak = GlossaryHint("cloak", "плащ", "Длинная накидка.")
    rustled = GlossaryHint("to rustle", "шуршать")
    result = _result(
        block,
        treatment=BlockTreatment.TRANSLATE,
        translation="Плащ зашуршал рядом с совой.",
        hints=(cloak, rustled),
    )
    pron_cloak = _pron(block, "cloak", "/kləʊk/", "uk")
    pron_rustled = _pron(block, "rustled", "/ˈrʌsəld/", "uk")
    pron_owl = _pron(block, "owl", "/aʊl/", "uk")

    notes = block_notes(
        block,
        result,
        None,
        (pron_cloak, pron_rustled, pron_owl),
        suppressed_hint_indices=frozenset({0}),
    )

    assert [note.term for note in notes] == ["to rustle", "owl"]
    assert notes[0].origin == "glossary_hint"
    assert notes[0].ipa == "/ˈrʌsəld/"
    assert notes[1].origin == "pronunciation"
    assert notes[1].ipa == "/aʊl/"
    assert all(note.term != "cloak" for note in notes)


def test_vocabulary_dedup_key_strips_sibilant_es_without_overstripping_silent_e() -> None:
    """Sibilant `-es` plurals/3sg deduplicate with their base forms while `-ce/-se/-ge` keep `-e`."""
    from linguary.application.layout.notes import vocabulary_dedup_key

    assert vocabulary_dedup_key("watches") == vocabulary_dedup_key("watch") == "watch"
    assert vocabulary_dedup_key("boxes") == vocabulary_dedup_key("box") == "box"
    assert vocabulary_dedup_key("glasses") == vocabulary_dedup_key("glass") == "glass"
    assert vocabulary_dedup_key("wishes") == vocabulary_dedup_key("wish") == "wish"
    assert vocabulary_dedup_key("buzzes") == vocabulary_dedup_key("buzz") == "buzz"
    # Silent-e stems (`closes`, `faces`, `pages`) must keep their trailing `e`.
    assert vocabulary_dedup_key("closes") == vocabulary_dedup_key("close") == "close"
    assert vocabulary_dedup_key("faces") == vocabulary_dedup_key("face") == "face"
    assert vocabulary_dedup_key("pages") == vocabulary_dedup_key("page") == "page"


def test_mixed_segment_hint_anchor_prefers_glossary_only_segment_and_respects_explicit_anchor() -> None:
    """`find_hint_marker_end` prioritizes explicit anchors, then `GLOSSARY_ONLY`/`READ_AS_IS` spans."""
    from linguary.application.layout.notes import find_hint_marker_end
    from linguary.domain import BlockSegment

    seg0 = "They fell from the base of the cliff. "
    seg1 = "Now they lay at the base of the cliff."
    text = seg0 + seg1
    seg0_span = (0, len(seg0))
    seg1_span = (len(seg0), len(text))
    seg0_cliff_end = text.index("cliff") + len("cliff")
    seg1_cliff_end = text.rfind("cliff") + len("cliff")

    # Priority 2: no explicit anchor -> prefers GLOSSARY_ONLY segment span.
    assert (
        find_hint_marker_end(text, "cliff", preferred_spans=(seg1_span,))
        == seg1_cliff_end
    )
    # Priority 1: explicit anchor in segment 0 overrides preferred_spans and locates word inside anchor.
    wide_anchor_start = text.index("base of the cliff")
    wide_anchor_end = wide_anchor_start + len("base of the cliff")
    assert (
        find_hint_marker_end(
            text,
            "cliff",
            anchor_spans=((wide_anchor_start, wide_anchor_end),),
            preferred_spans=(seg1_span,),
        )
        == seg0_cliff_end
    )
    # Priority 1 fallback: when `original` is not verbatim inside anchor, fall back to anchor end.
    assert (
        find_hint_marker_end(
            text,
            "sheer drop",
            anchor_spans=((wide_anchor_start, wide_anchor_end),),
            preferred_spans=(seg1_span,),
        )
        == wide_anchor_end
    )
    # Fallback to full text when preferred span does not contain the word.
    assert (
        find_hint_marker_end(text, "fell", preferred_spans=(seg1_span,))
        == text.index("fell") + len("fell")
    )

    # End-to-end via block_notes on a mixed [TRANSLATE, GLOSSARY_ONLY] block:
    block = DocumentBlock.create(
        document_id="book-1",
        section_id="section-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference(resource_path="chapter.xhtml", structural_path="/body/p[1]"),
        segments=(
            BlockSegment(seg0_span[0], seg0_span[1], 0),
            BlockSegment(seg1_span[0], seg1_span[1], 1),
        ),
    )
    unanchored_hint = GlossaryHint("cliff", "обрыв")
    unanchored_result = BlockProcessingResult(
        block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Они упали с обрыва.",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        segment_translations=("Они упали с обрыва.", None),
        glossary_hints=(unanchored_hint,),
    )
    unanchored_notes = block_notes(block, unanchored_result, None, ())
    assert len(unanchored_notes) == 1
    assert unanchored_notes[0].marker_end == seg1_cliff_end

    # When linked to an explicit Difficulty in segment 0 (TRANSLATE), stays inside segment 0.
    anchored_hint = GlossaryHint("cliff", "обрыв", "Сложный образ.")
    diff = _difficulty(
        block.id,
        (SourceAnchor(block.id, wide_anchor_start, wide_anchor_end, "base of the cliff"),),
        "vocabulary",
        difficulty_id="d-cliff-seg0",
    )
    plan = _plan(diff)
    anchored_result = BlockProcessingResult(
        block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Они упали с обрыва.",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        segment_translations=("Они упали с обрыва.", None),
        glossary_hints=(anchored_hint,),
        content_evidence=ContentEvidence(
            plan.sha256,
            (
                HelpEvidence(
                    diff.id,
                    diff.required_points[0].id,
                    "glossary_hint.translation",
                    block.id,
                    _hint_key("glossary_hint.translation", anchored_hint, block.id),
                    anchored_hint.translation,
                ),
            ),
        ),
    )
    anchored_notes = block_notes(block, anchored_result, plan, ())
    assert len(anchored_notes) == 1
    assert anchored_notes[0].marker_end == seg0_cliff_end


def test_tautological_gloss_dropped_and_real_notes_kept() -> None:
    """Tautological 1-3 word filler glosses are stripped after translation normalization."""
    from linguary.application.layout.notes import _is_tautological_gloss

    assert _is_tautological_gloss("взрослые", "взрослые люди") is True
    assert _is_tautological_gloss("взрослого", "взрослого человека") is True
    assert _is_tautological_gloss("взрослые", "взрослых людей.") is True
    assert _is_tautological_gloss("исчезнуть", "исчезновение, действие") is True
    assert _is_tautological_gloss("взрослые", "взрослые, обычно с точки зрения ребёнка") is False
    assert _is_tautological_gloss("сказка", "история") is False
    assert _is_tautological_gloss("вещь", "странная вещь") is False

    text = "The grown-ups never understood the grown-up world and nodded."
    block = _block(text)
    h1 = GlossaryHint("grown-ups", "взрослые", "взрослые люди")
    h2 = GlossaryHint("grown-up", "взрослого", "взрослого человека")
    h3 = GlossaryHint(
        "world",
        "мир",
        "мир — в книге так называют пограничье между жизнью и смертью",
    )
    # `hint.translation` carries a leading headword and past-tense form; tautology check runs after L779.
    h4 = GlossaryHint("nodded", "to nod — кивнула", "кивнуть, действие")
    result = _result(
        block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(h1, h2, h3, h4),
    )
    notes = block_notes(block, result, None, ())
    by_term = {note.term: note.body for note in notes}
    assert by_term["grown-ups"] == ("взрослые",)
    assert by_term["grown-up"] == ("взрослого",)
    assert by_term["world"] == (
        "мир",
        "в книге так называют пограничье между жизнью и смертью",
    )
    assert by_term["to nod"] == ("кивнуть",)

    # Signaled difficulties (e.g. wordplay) preserve their note even if short.
    wp_text = "The grown-ups laughed."
    wp_block = _block(wp_text)
    wp_hint = GlossaryHint("grown-ups", "взрослые", "взрослые люди")
    wp_diff = _difficulty(wp_block.id, (_anchor(wp_block, "grown-ups"),), "wordplay", difficulty_id="d-wp")
    wp_plan = _plan(wp_diff)
    wp_result = _result(
        wp_block,
        treatment=BlockTreatment.GLOSSARY_ONLY,
        hints=(wp_hint,),
        evidence=ContentEvidence(
            wp_plan.sha256,
            (
                HelpEvidence(
                    wp_diff.id,
                    wp_diff.required_points[0].id,
                    "glossary_hint.note",
                    wp_block.id,
                    _hint_key("glossary_hint.note", wp_hint, wp_block.id),
                    wp_hint.note or "",
                ),
            ),
        ),
    )
    wp_notes = block_notes(wp_block, wp_result, wp_plan, ())
    assert wp_notes[0].body == ("взрослые", "взрослые люди")


def test_resultative_particles_match_object_split_infinitive_phrases() -> None:
    """WP7: resultative particles like `open`/`shut`/`free` match object-split phrases (`flicked it open` <-> `to flick open`)."""
    from linguary.application.layout.notes import _infinitive_end, find_hint_marker_end

    cases = (
        ("He flicked it open and looked inside.", "to flick open", "flicked it open"),
        ("She pulled the door shut.", "to pull shut", "pulled the door shut"),
        ("They set the bird free.", "to set free", "set the bird free"),
        ("He wiped the glass clean.", "to wipe clean", "wiped the glass clean"),
        ("She flung the window wide.", "to fling wide", "flung the window wide"),
    )
    for text, original, matched in cases:
        expected_end = _end(text, matched)
        assert _infinitive_end(text, original) == expected_end
        assert find_hint_marker_end(text, original) == expected_end
        block = _block(text)
        result = _result(
            block,
            treatment=BlockTreatment.GLOSSARY_ONLY,
            hints=(GlossaryHint(original, "перевод"),),
        )
        assert block_notes(block, result, None, ())[0].marker_end == expected_end


def test_vocabulary_dedup_key_strips_leading_articles() -> None:
    """WP5: leading a/an/the are normalized so `the reptile house` deduplicates with `reptile house`."""
    from linguary.application.layout.notes import vocabulary_dedup_key

    assert vocabulary_dedup_key("the reptile house") == vocabulary_dedup_key("reptile house")
    assert vocabulary_dedup_key("a wand") == vocabulary_dedup_key("wand")
    assert vocabulary_dedup_key("an owl") == vocabulary_dedup_key("owl")
    assert vocabulary_dedup_key("to the door") != vocabulary_dedup_key("door")
    assert vocabulary_dedup_key("a bit") != vocabulary_dedup_key("to bite")
    assert vocabulary_dedup_key("a few") != vocabulary_dedup_key("few")
    assert vocabulary_dedup_key("the same") != vocabulary_dedup_key("same")


