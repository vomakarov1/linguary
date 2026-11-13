"""Notes projected for one block: markers, labels, and merged anchors."""

from __future__ import annotations

from typing import Literal

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
            ("Грамматика.",),
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
        "grammar": "Грамматика.",
        "idiom": "Идиома.",
        "cultural_reference": "Реалия.",
        "semantic_relation": "Смысл.",
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
            ("Смысл.",),
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
            ("Грамматика.",),
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
    assert shown[1].labels == ("Грамматика.",)


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
            ("Грамматика.", "Игра слов."),
            None,
            None,
            None,
            ("Нарочно ломает слово.", "Любопытнее.", "Неправильная степень сравнения."),
            shared.end,
            "assistance",
            ("grammar", "vocabulary", "wordplay"),
        ),
        LayoutNote(
            ("Идиома.",),
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
