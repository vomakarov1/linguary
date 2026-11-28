"""Notes for one block: hints, language notes, assistance, and pronunciation.

The projection does not persist anything. Difficulties whose `anchors` tuples
are equal share one note. A marker is an offset in the block text; notes
without one keep their original order after the notes that have one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from linguary.application.explained_expressions import _same_word, expression_words
from linguary.application.pronunciation_projection import PronunciationOccurrence
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    DocumentBlock,
    GlossaryHint,
    LanguageNote,
)
from linguary.domain.content_plan import (
    ContentEvidence,
    ContentPlan,
    Difficulty,
    HelpEvidence,
    SourceAnchor,
    artifact_key,
)
from linguary.domain.learning import FragmentAssistance

KIND_LABELS: Mapping[str, str | None] = {
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

_REFERENCE_COLON = "Языковая справка:"
_REFERENCE = "Языковая справка"
_EDITORIAL_TAILS = (
    "; вариант стоит проверить",
    ". Вариант стоит проверить.",
)
_UNCONFIRMED_WORDPLAY = (
    "Это можно заметить как возможную игру слов, но текст книги не подтверждает её прямо."
)
_UNCONFIRMED_WORDPLAY_TAIL = ", но текст книги не подтверждает её прямо"
_TAG_QUESTION_LABEL = "Вопрос построен без is it"
_TAG_QUESTION_REPLACEMENT = "Вопрос построен без Are there"
_AGREEMENT_NOTE_PREFIXES = (
    "разговорное подтверждение чужих слов",
    "говорящий соглашается и ссылается",
)
_SPEECH_VERB_NOTE = "помета к глаголу речи"
_SIGNAL_KINDS = frozenset(
    {"wordplay", "false_friend", "speech_style", "parody", "cultural_reference"}
)
_PARTICLES = frozenset(
    {
        "about",
        "across",
        "along",
        "around",
        "aside",
        "away",
        "back",
        "by",
        "down",
        "in",
        "off",
        "on",
        "out",
        "over",
        "through",
        "up",
    }
)
# `to his surprise` is a prepositional phrase, not `to` + verb.
_NOT_A_VERB_AFTER_TO = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "his",
        "her",
        "my",
        "their",
        "our",
        "your",
        "its",
        "him",
        "them",
        "me",
        "us",
        "you",
        "it",
        "himself",
        "herself",
        "myself",
        "themselves",
        "ourselves",
        "yourself",
        "yourselves",
    }
)
_RESULTATIVE_PARTICLES = frozenset(
    {
        "open",
        "shut",
        "closed",
        "free",
        "loose",
        "clear",
        "clean",
        "dry",
        "flat",
        "short",
        "still",
        "tight",
        "wide",
    }
)
# Particles and the prepositions that introduce an object after the verb.
_PHRASE_PREPOSITIONS = (
    _PARTICLES
    | _RESULTATIVE_PARTICLES
    | frozenset(
        {"at", "for", "from", "into", "of", "onto", "to", "with", "without", "upon", "against"}
    )
)
# Irregular or alternating forms only. Regular -ed/-ing stays on `_same_word` / `_regular_ed_stem`.
_IRREGULAR_VERB_FORMS: Mapping[str, frozenset[str]] = {
    "be": frozenset({"am", "is", "are", "was", "were", "been", "being"}),
    "bear": frozenset({"bore", "borne", "born"}),
    "beat": frozenset({"beat", "beaten"}),
    "become": frozenset({"became"}),
    "begin": frozenset({"began", "begun", "beginning"}),
    "bend": frozenset({"bent"}),
    "bet": frozenset({"bet", "betted", "betting"}),
    "bind": frozenset({"bound"}),
    "bite": frozenset({"bit", "bitten", "biting"}),
    "bleed": frozenset({"bled"}),
    "blow": frozenset({"blew", "blown"}),
    "break": frozenset({"broke", "broken"}),
    "bring": frozenset({"brought"}),
    "build": frozenset({"built"}),
    "burn": frozenset({"burnt", "burned"}),
    "burst": frozenset({"burst"}),
    "buy": frozenset({"bought"}),
    "cast": frozenset({"cast"}),
    "catch": frozenset({"caught"}),
    "choose": frozenset({"chose", "chosen", "choosing"}),
    "cling": frozenset({"clung"}),
    "come": frozenset({"came", "coming"}),
    "cost": frozenset({"cost"}),
    "creep": frozenset({"crept"}),
    "cut": frozenset({"cut", "cutting"}),
    "deal": frozenset({"dealt"}),
    "dig": frozenset({"dug", "digging"}),
    "do": frozenset({"does", "did", "done", "doing"}),
    "draw": frozenset({"drew", "drawn"}),
    "dream": frozenset({"dreamt", "dreamed"}),
    "drink": frozenset({"drank", "drunk"}),
    "drive": frozenset({"drove", "driven", "driving"}),
    "eat": frozenset({"ate", "eaten"}),
    "fall": frozenset({"fell", "fallen"}),
    "feed": frozenset({"fed"}),
    "feel": frozenset({"felt"}),
    "fight": frozenset({"fought"}),
    "find": frozenset({"found"}),
    "flee": frozenset({"fled"}),
    "fling": frozenset({"flung"}),
    "fly": frozenset({"flew", "flown"}),
    "forbid": frozenset({"forbade", "forbad", "forbidden"}),
    "forget": frozenset({"forgot", "forgotten", "forgetting"}),
    "forgive": frozenset({"forgave", "forgiven", "forgiving"}),
    "freeze": frozenset({"froze", "frozen", "freezing"}),
    "get": frozenset({"got", "gotten", "getting"}),
    "give": frozenset({"gave", "given", "giving"}),
    "go": frozenset({"goes", "went", "gone", "going"}),
    "grind": frozenset({"ground"}),
    "grow": frozenset({"grew", "grown"}),
    "hang": frozenset({"hung", "hanged"}),
    "have": frozenset({"has", "had", "having"}),
    "hear": frozenset({"heard"}),
    "hide": frozenset({"hid", "hidden", "hiding"}),
    "hit": frozenset({"hit", "hitting"}),
    "hold": frozenset({"held"}),
    "hurt": frozenset({"hurt"}),
    "keep": frozenset({"kept"}),
    "kneel": frozenset({"knelt", "kneeled"}),
    "know": frozenset({"knew", "known"}),
    "lay": frozenset({"laid"}),
    "lead": frozenset({"led"}),
    "lean": frozenset({"leant", "leaned"}),
    "leap": frozenset({"leapt", "leaped"}),
    "learn": frozenset({"learnt", "learned"}),
    "leave": frozenset({"left", "leaving"}),
    "lend": frozenset({"lent"}),
    "let": frozenset({"let", "letting"}),
    "lie": frozenset({"lay", "lain", "lied", "lying"}),
    "light": frozenset({"lit", "lighted"}),
    "lose": frozenset({"lost", "losing"}),
    "make": frozenset({"made", "making"}),
    "mean": frozenset({"meant"}),
    "meet": frozenset({"met"}),
    "pay": frozenset({"paid"}),
    "put": frozenset({"put", "putting"}),
    "read": frozenset({"read"}),
    "ride": frozenset({"rode", "ridden", "riding"}),
    "ring": frozenset({"rang", "rung"}),
    "rise": frozenset({"rose", "risen", "rising"}),
    "run": frozenset({"ran", "running"}),
    "say": frozenset({"said"}),
    "see": frozenset({"saw", "seen"}),
    "seek": frozenset({"sought"}),
    "sell": frozenset({"sold"}),
    "send": frozenset({"sent"}),
    "set": frozenset({"set", "setting"}),
    "shake": frozenset({"shook", "shaken", "shaking"}),
    "shine": frozenset({"shone", "shined", "shining"}),
    "shoot": frozenset({"shot"}),
    "show": frozenset({"showed", "shown"}),
    "shrink": frozenset({"shrank", "shrunk"}),
    "shut": frozenset({"shut", "shutting"}),
    "sing": frozenset({"sang", "sung"}),
    "sink": frozenset({"sank", "sunk"}),
    "sit": frozenset({"sat", "sitting"}),
    "sleep": frozenset({"slept"}),
    "slide": frozenset({"slid", "sliding"}),
    "sling": frozenset({"slung"}),
    "slit": frozenset({"slit", "slitting"}),
    "smell": frozenset({"smelt", "smelled"}),
    "speak": frozenset({"spoke", "spoken"}),
    "speed": frozenset({"sped", "speeded"}),
    "spell": frozenset({"spelt", "spelled"}),
    "spend": frozenset({"spent"}),
    "spill": frozenset({"spilt", "spilled"}),
    "spin": frozenset({"spun", "spinning"}),
    "spit": frozenset({"spat", "spitting"}),
    "split": frozenset({"split", "splitting"}),
    "spoil": frozenset({"spoilt", "spoiled"}),
    "spread": frozenset({"spread"}),
    "spring": frozenset({"sprang", "sprung"}),
    "stand": frozenset({"stood"}),
    "steal": frozenset({"stole", "stolen"}),
    "stick": frozenset({"stuck"}),
    "sting": frozenset({"stung"}),
    "stink": frozenset({"stank", "stunk"}),
    "stride": frozenset({"strode", "stridden", "striding"}),
    "strike": frozenset({"struck", "stricken", "striking"}),
    "swear": frozenset({"swore", "sworn"}),
    "sweep": frozenset({"swept"}),
    "swell": frozenset({"swelled", "swollen"}),
    "swim": frozenset({"swam", "swum", "swimming"}),
    "swing": frozenset({"swung"}),
    "take": frozenset({"took", "taken", "taking"}),
    "teach": frozenset({"taught"}),
    "tear": frozenset({"tore", "torn"}),
    "tell": frozenset({"told"}),
    "think": frozenset({"thought"}),
    "throw": frozenset({"threw", "thrown"}),
    "tread": frozenset({"trod", "trodden"}),
    "understand": frozenset({"understood"}),
    "wake": frozenset({"woke", "woken", "waked", "waking"}),
    "wear": frozenset({"wore", "worn"}),
    "weep": frozenset({"wept"}),
    "win": frozenset({"won", "winning"}),
    "wind": frozenset({"wound", "winded"}),
    "wring": frozenset({"wrung"}),
    "write": frozenset({"wrote", "written", "writing"}),
}
_IRREGULAR_TO_BASE: Mapping[str, str] = {
    form: base
    for base, forms in _IRREGULAR_VERB_FORMS.items()
    for form in forms
    if form not in _IRREGULAR_VERB_FORMS
}
_LEADING_TRANSLATION_BREAKS = frozenset(".!?—–-:;")
_APOSTROPHE_CHARS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_SURFACE_WORD = re.compile(r"\w+(?:'\w+)*")
_ONE_SLOT = "one's"
_DOUBLED_CONSONANTS = frozenset("bdgmnprt")
_LE_CONSONANTS = frozenset("bcdfgkpstz")
_SILENT_E_TAILS = (
    "v",
    "c",
    "dg",
    "cg",
    "u",
    "chang",
    "rang",
    "plung",
    "lung",
    "spong",
    "hing",
    "fring",
    "cring",
    "sing",
    "ting",
    "eng",
    "rg",
)
_SILENT_E_SYLLABLES = (
    "at",
    "ut",
    "ad",
    "id",
    "od",
    "ud",
    "cap",
    "shap",
    "scrap",
    "drap",
    "elop",
    "scop",
    "ok",
    "uk",
    "um",
    "ir",
    "or",
    "in",
    "it",
)
_NO_SILENT_E_WORDS = frozenset(
    {
        "visit",
        "limit",
        "credit",
        "edit",
        "vomit",
        "orbit",
        "habit",
        "merit",
        "profit",
        "spirit",
        "exhibit",
        "prohibit",
        "inherit",
        "deposit",
        "murmur",
        "augur",
        "favor",
        "favour",
        "honor",
        "honour",
        "labor",
        "labour",
        "color",
        "colour",
        "humor",
        "humour",
        "rumor",
        "rumour",
        "vapor",
        "vapour",
        "harbor",
        "harbour",
        "neighbor",
        "neighbour",
        "mirror",
        "terror",
        "horror",
        "error",
        "motor",
        "doctor",
        "factor",
        "actor",
        "author",
        "tutor",
        "tailor",
        "sailor",
        "anchor",
    }
)
_DASHES = ("—", "–", "-")
_RU_TOKEN = re.compile(r"[А-Яа-яЁё]+|[,;]")
_RU_INFINITIVE_ENDINGS = ("ть", "ться", "ти", "тись", "чь", "чься")
_RU_CONJUNCTIONS = frozenset({"и", "или", "да", "а", "но", "либо"})
_RU_PREPOSITIONS = frozenset(
    {
        "в",
        "во",
        "на",
        "за",
        "из",
        "изо",
        "к",
        "ко",
        "от",
        "ото",
        "по",
        "под",
        "подо",
        "над",
        "надо",
        "перед",
        "передо",
        "через",
        "с",
        "со",
        "у",
        "о",
        "об",
        "обо",
        "про",
        "для",
        "без",
        "безо",
        "до",
        "при",
        "сквозь",
        "между",
        "среди",
        "около",
        "вокруг",
        "после",
        "кроме",
        "вместо",
        "сверх",
    }
)
_RU_IRREGULAR = {
    "хихикнул": "хихикать",
    "хихикнула": "хихикать",
    "хихикнуло": "хихикать",
    "хихикнули": "хихикать",
    "поплелся": "поплестись",
    "поплелась": "поплестись",
    "поплелось": "поплестись",
    "поплелись": "поплестись",
    "пошел": "пойти",
    "пошла": "пойти",
    "пошло": "пойти",
    "пошли": "пойти",
}
_RU_PAST = (
    ("нулись", "нуться"),
    ("нулась", "нуться"),
    ("нулось", "нуться"),
    ("нулся", "нуться"),
    ("лись", "ться"),
    ("лась", "ться"),
    ("лось", "ться"),
    ("лся", "ться"),
    ("нули", "нуть"),
    ("нула", "нуть"),
    ("нуло", "нуть"),
    ("нул", "нуть"),
    ("или", "ить"),
    ("ила", "ить"),
    ("ил", "ить"),
    ("ели", "еть"),
    ("ела", "еть"),
    ("ел", "еть"),
    ("али", "ать"),
    ("ала", "ать"),
    ("ал", "ать"),
    ("яли", "ять"),
    ("яла", "ять"),
    ("ял", "ять"),
    ("ыли", "ыть"),
    ("ыла", "ыть"),
    ("ыл", "ыть"),
)
_TAUTOLOGY_WORD_RE = re.compile(r"[а-яa-z0-9]+")
_TAUTOLOGY_FILLER_STEMS = (
    "люд",
    "человек",
    "вещ",
    "предмет",
    "мест",
    "действ",
    "состоян",
    "явлен",
)


@dataclass(frozen=True, slots=True)
class LayoutNote:
    """One block note before display numbering is assigned.

    `labels` are separate kind names. Only wordplay, false friend, speech style,
    and parody are shown; other kinds stay in `difficulty_kinds`. `marker_end`
    is an offset into the block text.
    """

    labels: tuple[str, ...]
    term: str | None
    ipa: str | None
    variant: str | None
    body: tuple[str, ...]
    marker_end: int | None
    origin: str
    difficulty_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Draft:
    """One source item before equal-anchor difficulties are collapsed."""

    label_ids: tuple[str, ...]
    kind_ids: tuple[str, ...]
    term: str | None
    ipa: str | None
    variant: str | None
    body: tuple[str, ...]
    marker_end: int | None
    origin: str
    anchors: tuple[SourceAnchor, ...] | None


def block_notes(
    block: DocumentBlock,
    result: BlockProcessingResult,
    plan: ContentPlan | None,
    pronunciations: Sequence[PronunciationOccurrence],
    *,
    english_help: bool = False,
    dropped_difficulty_ids: frozenset[str] = frozenset(),
    suppressed_hint_indices: frozenset[int] = frozenset(),
    stripped_hint_indices: frozenset[int] = frozenset(),
    suppressed_note_indices: frozenset[int] = frozenset(),
) -> tuple[LayoutNote, ...]:
    """Project help for `block` into notes ordered by marker.

    Glossary hints, language notes, and resolved assistance keep the order
    they have on `result`. A pronunciation whose `quote` equals a hint's
    `original` fills that hint. Any other pronunciation of this block follows
    them. `origin` is `glossary_hint`, `language_note`, `assistance`, or
    `pronunciation`. Items backed exclusively by `dropped_difficulty_ids` are
    omitted (for `GLOSSARY_ONLY`, a hint keeps its translation without the
    repeated dialect note). Hints in `suppressed_hint_indices` and language
    notes in `suppressed_note_indices` are skipped without mutating `result`;
    hints in `stripped_hint_indices` drop their explanatory note text while
    preserving `artifact_key` linkage. Empty notes are omitted before numbering.
    """
    evidences = _evidences(result)
    known = {} if plan is None else {item.id: item for item in plan.difficulties}
    requires_hints = result.treatment.requires_glossary_hints or any(
        mode.requires_glossary_hints for mode in result.segment_treatments
    )
    visible_hints: list[tuple[GlossaryHint, bool]] = []
    omitted_hints: list[GlossaryHint] = []
    for index, hint in enumerate(result.glossary_hints):
        if index in suppressed_hint_indices:
            omitted_hints.append(hint)
            continue
        only_dropped = _hint_only_dropped(hint, result.block_id, evidences, dropped_difficulty_ids)
        if only_dropped and not requires_hints:
            omitted_hints.append(hint)
            continue
        visible_hints.append((hint, only_dropped or index in stripped_hint_indices))
    assigned, unmatched = _assign_pronunciations(
        [hint for hint, _ in visible_hints],
        pronunciations,
        block.id,
        excluded_hints=omitted_hints,
    )
    preferred_spans = _preferred_hint_spans(block, result)
    drafts: list[_Draft] = []
    for (hint, strip_note), pronunciation in zip(visible_hints, assigned, strict=True):
        drafts.append(
            _hint_draft(
                hint,
                pronunciation,
                block.text,
                result.block_id,
                evidences,
                known,
                result.treatment,
                strip_note=strip_note,
                dropped_difficulty_ids=dropped_difficulty_ids,
                preferred_spans=preferred_spans,
            )
        )
    for index, note in enumerate(result.language_notes):
        if index in suppressed_note_indices:
            continue
        if _note_only_dropped(note, evidences, dropped_difficulty_ids):
            continue
        drafts.append(_note_draft(note, block.text, evidences, known))
    for assistance in result.fragment_assistances:
        if not assistance.resolved or assistance.difficulty_id in dropped_difficulty_ids:
            continue
        drafts.append(
            _assistance_draft(assistance, block.id, known, english_help=english_help)
        )
    drafts.extend(_pronunciation_draft(item) for item in unmatched)
    notes = [
        note
        for note in _collapse(drafts, plan)
        if note.body or note.ipa
    ]
    return tuple(sorted(notes, key=_marker_key))


_LEADING_ARTICLES = frozenset({"a", "an", "the"})
_PROTECTED_ARTICLE_PHRASES = frozenset(
    {"a bit", "a few", "a little", "a lot", "a while", "a dozen", "the same", "the other", "the rest"}
)


def vocabulary_dedup_key(original: str) -> str:
    """Normalized lemma-like key for book-wide glossary hint deduplication."""
    words = expression_words(original)
    if not words:
        return original.strip().casefold()
    if len(words) >= 2 and words[0] == "to" and words[1] not in _NOT_A_VERB_AFTER_TO:
        words = words[1:]
    elif (
        len(words) >= 2
        and words[0] in _LEADING_ARTICLES
        and " ".join(words) not in _PROTECTED_ARTICLE_PHRASES
        and words[1] not in _IRREGULAR_TO_BASE
    ):
        words = words[1:]
    head = _canonical_vocab_word(words[0])
    if len(words) == 1:
        return head
    return " ".join((head, *words[1:]))


def _canonical_vocab_word(word: str) -> str:
    if word in _IRREGULAR_VERB_FORMS:
        return word
    irregular = _IRREGULAR_TO_BASE.get(word)
    if irregular is not None:
        return irregular
    ed = _regular_ed_stem(word)
    if ed is not None:
        return ed
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if (
            len(stem) >= 4
            and stem[-1] == stem[-2]
            and stem[-1] in _DOUBLED_CONSONANTS
            and _is_vowel(stem, -3)
            and not _is_vowel(stem, -4)
        ):
            return stem[:-1]
        if _magic_e(stem):
            return stem + "e"
        return stem
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("shes", "ches", "xes", "zes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "os", "as")):
        return word[:-1]
    return word


def _hint_evidence_keys(hint: GlossaryHint, block_id: str) -> set[str]:
    content = {"original": hint.original, "translation": hint.translation, "note": hint.note}
    return {
        artifact_key("glossary_hint.note", block_id, content),
        artifact_key("glossary_hint.translation", block_id, content),
    }


def _note_evidence_key(note: LanguageNote) -> str:
    content = {
        "original_excerpt": note.original_excerpt,
        "note": note.note,
        "translated_excerpt": note.translated_excerpt,
    }
    return artifact_key("language_note.note", note.block_id, content)


def _hint_only_dropped(
    hint: GlossaryHint,
    block_id: str,
    evidences: Sequence[HelpEvidence],
    dropped: frozenset[str],
) -> bool:
    if not dropped:
        return False
    ids = _ids_for(evidences, _hint_evidence_keys(hint, block_id))
    return bool(ids) and set(ids) <= dropped


def _note_only_dropped(
    note: LanguageNote,
    evidences: Sequence[HelpEvidence],
    dropped: frozenset[str],
) -> bool:
    if not dropped:
        return False
    ids = _ids_for(evidences, {_note_evidence_key(note)})
    return bool(ids) and set(ids) <= dropped


def _evidences(result: BlockProcessingResult) -> tuple[HelpEvidence, ...]:
    """Help evidence stored on the result. An empty result has none."""
    evidence = result.content_evidence
    if not isinstance(evidence, ContentEvidence):
        return ()
    return tuple(item for item in evidence.evidence if isinstance(item, HelpEvidence))


def _assign_pronunciations(
    hints: Sequence[GlossaryHint],
    pronunciations: Sequence[PronunciationOccurrence],
    block_id: str,
    *,
    excluded_hints: Sequence[GlossaryHint] = (),
) -> tuple[list[PronunciationOccurrence | None], tuple[PronunciationOccurrence, ...]]:
    """Pair each hint with the first pronunciation of the same quote.

    A quote that matches some hint `original` (including case-insensitive or
    infinitive `to` + verb forms, or a suppressed/dropped hint) does not become
    its own standalone note, even when a later copy is left over. Other blocks
    are ignored.
    """
    local = [item for item in pronunciations if item.block_id == block_id]
    used: set[int] = set()
    assigned: list[PronunciationOccurrence | None] = []
    for hint in hints:
        match = None
        for index, item in enumerate(local):
            if index in used or not _hint_matches_pronunciation(hint.original, item.quote):
                continue
            match = item
            used.add(index)
            break
        assigned.append(match)
    all_hints = (*hints, *excluded_hints)
    for index, item in enumerate(local):
        if any(_hint_matches_pronunciation(hint.original, item.quote) for hint in all_hints):
            used.add(index)
    unmatched = tuple(item for index, item in enumerate(local) if index not in used)
    return assigned, unmatched


def _hint_matches_pronunciation(original: str, quote: str) -> bool:
    """Whether a pronunciation `quote` belongs to a glossary hint `original`."""
    if quote == original:
        return True
    stripped_original = original.strip()
    stripped_quote = quote.strip()
    if not stripped_original or not stripped_quote:
        return False
    if stripped_quote.casefold() == stripped_original.casefold():
        return True
    pattern = _verbal_infinitive_words(stripped_original)
    if not pattern:
        return False
    tokens = _surface_tokens(stripped_quote)
    if not tokens or not _same_verb(tokens[0][0], pattern[0]):
        return False
    rest = pattern[1:]
    if not rest:
        return len(tokens) == 1
    return _match_phrase(tokens, 1, rest) == tokens[-1][1]


def _preferred_hint_spans(
    block: DocumentBlock,
    result: BlockProcessingResult,
) -> tuple[tuple[int, int], ...]:
    """Segment spans prioritized for unanchored hints: `GLOSSARY_ONLY`, then `READ_AS_IS`."""
    spans = block.segment_spans
    treatments = result.segment_treatments
    if len(spans) <= 1 or not treatments or len(treatments) != len(spans):
        return ()
    glossary_spans = [
        (span.start, span.end)
        for span, mode in zip(spans, treatments, strict=True)
        if mode.requires_glossary_hints
    ]
    read_spans = [
        (span.start, span.end)
        for span, mode in zip(spans, treatments, strict=True)
        if mode is BlockTreatment.READ_AS_IS
    ]
    return tuple((*glossary_spans, *read_spans))


def _hint_anchor_spans(
    ids: Sequence[str],
    known: Mapping[str, Difficulty],
    block_id: str,
) -> tuple[tuple[int, int], ...]:
    """Anchor `(start, end)` spans in `block_id` for difficulties linked to a hint."""
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item_id in ids:
        difficulty = known.get(item_id)
        if difficulty is None:
            continue
        for anchor in difficulty.anchors:
            if anchor.block_id != block_id:
                continue
            bounds = (anchor.start, anchor.end)
            if bounds not in seen:
                seen.add(bounds)
                spans.append(bounds)
    return tuple(spans)


def find_hint_marker_end(
    text: str,
    original: str,
    *,
    anchor_spans: Sequence[tuple[int, int]] = (),
    preferred_spans: Sequence[tuple[int, int]] = (),
) -> int | None:
    """Find marker offset for a glossary hint with anchor and segment priorities.

    Priority 1: explicit `Difficulty` anchor spans in this block (searching
    `_find_end` inside each valid span first, then falling back to the first
    valid anchor end).
    Priority 2: preferred segment spans (`GLOSSARY_ONLY`, then `READ_AS_IS`).
    Fallback: first match anywhere in `text`.
    """
    text_len = len(text)
    valid_anchors = [
        (a_start, a_end)
        for a_start, a_end in anchor_spans
        if 0 <= a_start < a_end <= text_len
    ]
    for a_start, a_end in valid_anchors:
        local = _find_end(text[a_start:a_end], original)
        if local is not None:
            return a_start + local
    if valid_anchors:
        return valid_anchors[0][1]
    for s_start, s_end in preferred_spans:
        if not (0 <= s_start < s_end <= text_len):
            continue
        local = _find_end(text[s_start:s_end], original)
        if local is not None:
            return s_start + local
    return _find_end(text, original)


def _words_match_tautology(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 4 and len(right) >= 4 and left[:4] == right[:4]


def _is_tautological_gloss(translation: str, note: str) -> bool:
    """Whether a short 1-3 word note only restates `translation` plus empty filler stems."""
    tr_words = _TAUTOLOGY_WORD_RE.findall(translation.casefold().replace("ё", "е"))
    note_words = _TAUTOLOGY_WORD_RE.findall(note.casefold().replace("ё", "е"))
    if not tr_words or not (1 <= len(note_words) <= 3):
        return False
    significant_tr = [word for word in tr_words if len(word) >= 2] or tr_words
    matched_indices: set[int] = set()
    for tr_word in significant_tr:
        found = next(
            (
                idx
                for idx, note_word in enumerate(note_words)
                if idx not in matched_indices and _words_match_tautology(tr_word, note_word)
            ),
            None,
        )
        if found is None:
            return False
        matched_indices.add(found)
    remaining = [
        note_word for idx, note_word in enumerate(note_words) if idx not in matched_indices
    ]
    return all(note_word.startswith(_TAUTOLOGY_FILLER_STEMS) for note_word in remaining)


def _hint_draft(
    hint: GlossaryHint,
    pronunciation: PronunciationOccurrence | None,
    text: str,
    block_id: str,
    evidences: Sequence[HelpEvidence],
    known: Mapping[str, Difficulty],
    treatment: BlockTreatment,
    *,
    strip_note: bool = False,
    dropped_difficulty_ids: frozenset[str] = frozenset(),
    preferred_spans: Sequence[tuple[int, int]] = (),
) -> _Draft:
    """A glossary hint. The marker uses explicit anchors, then preferred spans, then `text`.

    Evidence keys use the stored hint fields, before the marker search and
    before a leading translation is cut out of the note or `strip_note` hides
    it. `term` and `body` may then drop a repeated sentence, a quoted or
    leading translation, or a headword gloss, and show a past-tense verb as an
    infinitive.
    """
    raw_ids = _ids_for(evidences, _hint_evidence_keys(hint, block_id))
    ids = tuple(item_id for item_id in raw_ids if item_id not in dropped_difficulty_ids)
    anchor_spans = _hint_anchor_spans(ids, known, block_id)
    marker_end = find_hint_marker_end(
        text,
        hint.original,
        anchor_spans=anchor_spans,
        preferred_spans=preferred_spans,
    )
    cleaned_note = "" if strip_note else _cleaned_hint_note(hint.note)
    lexical = _plain_lexical(hint.original, ids, known)
    verbal = _verbal_infinitive_words(hint.original) is not None
    signaled = any(item_id in known and known[item_id].kind in _SIGNAL_KINDS for item_id in ids)
    kept_note = ""
    if cleaned_note and not (
        (lexical or verbal)
        and not signaled
        and _headword_gloss(cleaned_note, hint.original)
    ):
        kept_note = cleaned_note
    base_translation = _strip_leading_headword(hint.translation, hint.original)
    term: str | None = hint.original
    translation = base_translation
    pre_infinitive: str | None = None
    if verbal:
        infinitive = _to_infinitive_term(hint.original)
        if infinitive is not None:
            term = infinitive
            translation = _strip_leading_headword(translation, term)
    elif lexical:
        infinitive = _to_infinitive_term(hint.original)
        if infinitive is not None:
            stripped_tr = _strip_leading_headword(translation, infinitive)
            pre_infinitive = stripped_tr
            converted = _russian_past_to_infinitive(stripped_tr)
            if converted != stripped_tr or _has_russian_infinitive(stripped_tr):
                term = infinitive
                translation = converted
            else:
                translation = stripped_tr
    for candidate in (translation, pre_infinitive, base_translation):
        if candidate:
            kept_note = _strip_leading_translation(kept_note, candidate)
    if (
        kept_note
        and (lexical or verbal)
        and not signaled
        and _is_tautological_gloss(translation, kept_note)
    ):
        kept_note = ""
    long = (
        treatment is BlockTreatment.TRANSLATE
        and bool(cleaned_note)
        and len(expression_words(hint.original)) > 5
    )
    if long:
        term = None
        parts = [cleaned_note]
    elif _note_quotes_translation(kept_note, translation):
        parts = [kept_note]
    else:
        parts = [translation]
        if kept_note:
            parts.append(kept_note)
    return _Draft(
        label_ids=ids,
        kind_ids=ids,
        term=term,
        ipa=None if pronunciation is None else pronunciation.ipa,
        variant=None if pronunciation is None else pronunciation.variant,
        body=_body(*parts),
        marker_end=marker_end,
        origin="glossary_hint",
        anchors=_shared_anchors(ids, known),
    )


def _note_draft(
    note: LanguageNote,
    text: str,
    evidences: Sequence[HelpEvidence],
    known: Mapping[str, Difficulty],
) -> _Draft:
    """A language note. A missing excerpt is kept, with no marker."""
    ids = _ids_for(evidences, {_note_evidence_key(note)})
    return _Draft(
        label_ids=ids,
        kind_ids=ids,
        term=None,
        ipa=None,
        variant=None,
        body=_body(note.note),
        marker_end=_find_end(text, note.original_excerpt),
        origin="language_note",
        anchors=_shared_anchors(ids, known),
    )


def _assistance_draft(
    assistance: FragmentAssistance,
    block_id: str,
    known: Mapping[str, Difficulty],
    *,
    english_help: bool,
) -> _Draft:
    """Resolved assistance. The marker is the last anchor end in this block.

    Under `english_help`, `lexical_hint` keeps its kind and loses the label.
    """
    difficulty = known.get(assistance.difficulty_id)
    linked = () if difficulty is None else (difficulty.id,)
    hide_label = english_help and assistance.kind == "lexical_hint"
    return _Draft(
        label_ids=() if hide_label else linked,
        kind_ids=linked,
        term=None,
        ipa=None,
        variant=None,
        body=_body(assistance.text or ""),
        marker_end=_last_anchor_end(block_id, assistance.anchors),
        origin="assistance",
        anchors=None if difficulty is None else difficulty.anchors,
    )


def _pronunciation_draft(item: PronunciationOccurrence) -> _Draft:
    """A pronunciation that did not match a glossary hint."""
    return _Draft(
        label_ids=(),
        kind_ids=(),
        term=item.quote,
        ipa=item.ipa,
        variant=item.variant,
        body=(),
        marker_end=item.end,
        origin="pronunciation",
        anchors=None,
    )


def _ids_for(evidences: Sequence[HelpEvidence], keys: set[str]) -> tuple[str, ...]:
    """Difficulty ids whose stored artifact key is one of `keys`, without repeats."""
    return tuple(
        dict.fromkeys(item.difficulty_id for item in evidences if item.artifact_key in keys)
    )


def _shared_anchors(
    ids: Sequence[str],
    known: Mapping[str, Difficulty],
) -> tuple[SourceAnchor, ...] | None:
    """The anchor tuple when every linked difficulty has that same tuple.

    Mixed anchors stay on this note and do not pull another note in.
    """
    found = [known[item_id].anchors for item_id in dict.fromkeys(ids) if item_id in known]
    if not found:
        return None
    first = found[0]
    if all(anchors == first for anchors in found):
        return first
    return None


def _find_end(text: str, quote: str) -> int | None:
    """End of the first match of `quote`, or None when nothing matches.

    Exact word-bounded match wins first, then apostrophe-normalized exact and
    casefolded word-bounded matches, a single-sentence gapped phrase with
    `…`/`...`, or `to` plus a verb matched to an inflected phrase.
    """
    if not quote:
        return None
    exact = _word_bounded_end(text, quote, ignore_case=False)
    if exact is not None:
        return exact
    norm_text = text.translate(_APOSTROPHE_CHARS)
    norm_quote = quote.translate(_APOSTROPHE_CHARS)
    normalized = _word_bounded_end(norm_text, norm_quote, ignore_case=False)
    if normalized is not None:
        return normalized
    folded = _word_bounded_end(norm_text, norm_quote, ignore_case=True)
    if folded is not None:
        return folded
    gapped = _gapped_end(norm_text, norm_quote)
    if gapped is not None:
        return gapped
    return _infinitive_end(text, quote)


def _word_bounded_end(text: str, quote: str, *, ignore_case: bool) -> int | None:
    """End of the first `quote` matching on word boundaries where edges are alphanumeric or apostrophe."""
    stripped = quote.strip()
    if not stripped:
        return None
    left_boundary = r"(?<!\w)" if stripped[0].isalnum() else ""
    right_boundary = r"(?!\w)" if (stripped[-1].isalnum() or stripped[-1] in "'’") else ""
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(rf"{left_boundary}{re.escape(stripped)}{right_boundary}", flags)
    match = pattern.search(text)
    if match is None:
        return None
    return match.end()


_ELLIPSIS_GAP = re.compile(r"\s*(?:…|\.{3})\s*")
_GAP_BRIDGE = r"[,;:\"'‘’“”()]*\s+(?:[^\s.!?\r\n]+\s+){0,8}[\"'‘’“”(]*"


def _gapped_end(text: str, quote: str) -> int | None:
    """End of a gapped phrase like `what does … do` within one sentence."""
    if "…" not in quote and "..." not in quote:
        return None
    parts = [part.strip() for part in _ELLIPSIS_GAP.split(quote.strip()) if part.strip()]
    if len(parts) < 2 or len(parts) > 4:
        return None
    pieces: list[str] = []
    for index, part in enumerate(parts):
        left_b = r"(?<!\w)" if part[0].isalnum() else ""
        right_b = r"(?!\w)" if part[-1].isalnum() else ""
        escaped = re.sub(r"\\ ", r"\\s+", re.escape(part))
        if index == 0:
            pieces.append(f"{left_b}{escaped}")
        else:
            pieces.append(f"{escaped}{right_b}")
    pattern = re.compile(_GAP_BRIDGE.join(pieces), re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return None
    return match.end()


def _verbal_infinitive_words(original: str) -> tuple[str, ...] | None:
    """Words after `to` when `original` is a verb infinitive, not `to his surprise`."""
    stripped = original.strip()
    if not stripped.casefold().startswith("to "):
        return None
    words = expression_words(stripped)
    if len(words) < 2 or words[0] != "to" or words[1] in _NOT_A_VERB_AFTER_TO:
        return None
    return words[1:]


def _infinitive_end(text: str, quote: str) -> int | None:
    """End of `to` + verb matched to an inflected phrase in `text`."""
    pattern = _verbal_infinitive_words(quote)
    if not pattern:
        return None
    verb, rest = pattern[0], pattern[1:]
    tokens = _surface_tokens(text)
    for index, (word, end) in enumerate(tokens):
        if not _same_verb(word, verb):
            continue
        if not rest:
            return end
        found = _match_phrase(tokens, index + 1, rest)
        if found is not None:
            return found
    return None


def _surface_tokens(text: str) -> list[tuple[str, int]]:
    """Casefolded words and the exclusive end of each word in `text`."""
    normalized = text.translate(_APOSTROPHE_CHARS)
    return [
        (match.group(0).casefold(), match.end())
        for match in _SURFACE_WORD.finditer(normalized)
    ]


_SINGLE_CONSONANT_CVC_EXCEPTIONS = frozenset({"bus", "gas"})


def _is_monosyllabic_cvc(verb: str) -> bool:
    """Whether `verb` is a single-syllable CVC stem that must double before `-ed`/`-ing`."""
    if len(verb) < 3 or verb in _SINGLE_CONSONANT_CVC_EXCEPTIONS:
        return False
    if verb[-1] in "aeiouwxy" or not _is_vowel(verb, -2) or _is_vowel(verb, -3):
        return False
    vowel_groups = sum(
        1
        for i in range(len(verb))
        if _is_vowel(verb, i) and (i == 0 or not _is_vowel(verb, i - 1))
    )
    return vowel_groups == 1


def _regular_present_or_participle(surface: str, verb: str) -> bool:
    """Valid 3rd-person singular or `-ing` participle of `verb`."""
    if verb.endswith("o"):
        if surface in {f"{verb}es", f"{verb}s"}:
            return True
    elif verb.endswith(("s", "x", "z", "ch", "sh")):
        if surface == f"{verb}es":
            return True
    elif verb.endswith("y") and len(verb) >= 3 and verb[-2] not in "aeiou":
        if surface == f"{verb[:-1]}ies":
            return True
    elif surface == f"{verb}s":
        return True
    if _is_monosyllabic_cvc(verb):
        return surface == f"{verb}{verb[-1]}ing"
    if surface == f"{verb}ing":
        return True
    if verb.endswith("ie") and surface == f"{verb[:-2]}ying":
        return True
    if (
        verb.endswith("e")
        and len(verb) >= 3
        and not verb.endswith(("ee", "ye", "oe"))
        and surface == f"{verb[:-1]}ing"
    ):
        return True
    if (
        len(verb) >= 3
        and verb[-1] not in "aeiouwxy"
        and _is_vowel(verb, -2)
        and not _is_vowel(verb, -3)
        and surface == f"{verb}{verb[-1]}ing"
    ):
        return True
    return False


def _same_verb(surface: str, verb: str) -> bool:
    """Whether `surface` is `verb` or one of its inflected forms."""
    if surface == verb or surface in _IRREGULAR_VERB_FORMS.get(verb, ()):
        return True
    if verb == "be":
        return False
    if _regular_present_or_participle(surface, verb):
        return True
    if verb in _IRREGULAR_VERB_FORMS:
        return False
    if _is_monosyllabic_cvc(verb):
        return surface == f"{verb}{verb[-1]}ed"
    if _same_word(surface, verb):
        return True
    if _regular_ed_stem(surface) == verb:
        return True
    if verb.endswith("e"):
        return surface == f"{verb}d"
    if surface == f"{verb}ed":
        return True
    if verb.endswith("y") and len(verb) >= 3 and verb[-2] not in "aeiou":
        return surface == f"{verb[:-1]}ied"
    if (
        len(verb) >= 3
        and verb[-1] not in "aeiouwxy"
        and _is_vowel(verb, -2)
        and not _is_vowel(verb, -3)
    ):
        return surface == f"{verb}{verb[-1]}ed"
    return False


def _match_phrase(
    tokens: Sequence[tuple[str, int]],
    index: int,
    pattern: tuple[str, ...],
) -> int | None:
    """End of `pattern` in `tokens` from `index`, allowing an object gap."""
    if pattern and pattern[0] in _PHRASE_PREPOSITIONS:
        for skip in range(3):
            found = _match_fixed(tokens, index + skip, pattern)
            if found is not None:
                return found
        return None
    found = _match_fixed(tokens, index, pattern)
    if found is not None:
        return found
    return _match_shifted_object(tokens, index, pattern)


def _match_fixed(
    tokens: Sequence[tuple[str, int]],
    index: int,
    pattern: tuple[str, ...],
) -> int | None:
    """End of a contiguous `pattern`. `one's` may be zero to two words."""
    if not pattern:
        if index <= 0 or index - 1 >= len(tokens):
            return None
        return tokens[index - 1][1]
    if pattern[0] == _ONE_SLOT:
        for skip in range(3):
            found = _match_fixed(tokens, index + skip, pattern[1:])
            if found is not None:
                return found
        return None
    if index >= len(tokens) or tokens[index][0] != pattern[0]:
        return None
    return _match_fixed(tokens, index + 1, pattern[1:])


def _match_shifted_object(
    tokens: Sequence[tuple[str, int]],
    index: int,
    pattern: tuple[str, ...],
) -> int | None:
    """`throw a dirty look at` also matches `threw Uncle Vernon a dirty look`."""
    if len(pattern) < 2 or pattern[-1] not in _PHRASE_PREPOSITIONS:
        return None
    middle = pattern[:-1]
    for skip in range(1, 3):
        found = _match_fixed(tokens, index + skip, middle)
        if found is not None:
            return found
    return None


def _strip_leading_translation(note: str, translation: str) -> str:
    """Drop a leading unquoted copy of `translation` and the punctuation after it.

    The copy must be the whole note or be followed by `.`, `!`, `?`, `—`, `–`,
    `-`, `:`, or `;`. A longer word that merely starts with the translation stays.
    """
    text = note.strip()
    quoted = translation.strip()
    if not text or not quoted or len(text) < len(quoted):
        return text
    if text[: len(quoted)].casefold() != quoted.casefold():
        return text
    rest = text[len(quoted) :].lstrip()
    if not rest:
        return ""
    if rest[0] not in _LEADING_TRANSLATION_BREAKS:
        return text
    return rest[1:].lstrip()


def _last_anchor_end(block_id: str, anchors: Sequence[SourceAnchor]) -> int | None:
    """End of the rightmost anchor that belongs to this block."""
    ends = [anchor.end for anchor in anchors if anchor.block_id == block_id]
    if not ends:
        return None
    return max(ends)


_DECLENSION_TAIL = re.compile(
    r"\s*;\s*не\s+склоня(?:ется|ть)\.?$",
    re.IGNORECASE,
)
_DECLENSION_SENTENCE_CORE = (
    r"(?:"
    r"не\s+склонять|"
    r"(?:имя|фамилия|название)(?:\s+[А-ЯЁA-Z][\w-]*)?\s+не\s+склоня(?:ется|ть)|"
    r"при\s+слове\s+[«\"][^»\"]+[»\"]\s+(?:имя|фамилия|название)(?:\s+\S+)?\s+не\s+склоня(?:ется|ть)"
    r")\.?"
)
_DECLENSION_STANDALONE = re.compile(rf"^{_DECLENSION_SENTENCE_CORE}$", re.IGNORECASE)
_DECLENSION_TRAILING_SENTENCE = re.compile(
    rf"(?<=[.!?])\s+{_DECLENSION_SENTENCE_CORE}$", re.IGNORECASE
)


def _strip_leading_headword(text: str, headword: str) -> str:
    """Drop a redundant `headword — ` prefix inside `hint.translation`.

    Plain `-` requires surrounding spaces so hyphenated compounds (`mini-юбка`,
    `Wi-Fi`) are preserved. Regex matching keeps slice indices exact under Unicode.
    """
    stripped = text.strip()
    if not stripped or not headword.strip():
        return stripped
    norm_text = stripped.translate(_APOSTROPHE_CHARS)
    for head in _headwords(headword):
        norm_head = head.translate(_APOSTROPHE_CHARS)
        match = re.match(
            rf"{re.escape(norm_head)}(?:\s+[—–-]\s+|\s*[—–]\s*)",
            norm_text,
            re.IGNORECASE,
        )
        if match is not None:
            candidate = stripped[match.end() :].lstrip()
            if candidate:
                return candidate
    return stripped


def _strip_declension_directive(text: str) -> str:
    """Remove translator-only Russian declension directives leaked into notes."""
    current = text.strip()
    if not current:
        return ""
    if _DECLENSION_STANDALONE.fullmatch(current):
        return ""
    current = _DECLENSION_TRAILING_SENTENCE.sub("", current).rstrip()
    had_period = current.endswith(".")
    trimmed = _DECLENSION_TAIL.sub("", current).rstrip()
    if trimmed != current:
        if had_period and trimmed and trimmed[-1] not in ".!?":
            return f"{trimmed}."
        return trimmed
    return current


def _body(*parts: str) -> tuple[str, ...]:
    """Drop reference phrases, speech-verb boilerplate, editorial tails, and AI disclaimers.

    A piece that becomes empty is omitted. The colon form of the reference
    phrase is removed first so a following colon is not left behind.
    """
    cleaned = []
    for part in parts:
        text = part.replace("\u00ad", "")
        text = text.replace(_REFERENCE_COLON, "").replace(_REFERENCE, "")
        text = _strip_speech_verb_note(text)
        text = _strip_editorial_tail(text)
        text = _strip_declension_directive(text)
        text = _strip_ai_disclaimer(text)
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _strip_ai_disclaimer(text: str) -> str:
    """Remove the unconfirmed-wordplay disclaimer and fix the Are there label.

    One neighboring space is removed with the sentence so a gap does not stay doubled.
    Spaces left by other cuts are kept.
    """
    current = _cut_piece(text, _UNCONFIRMED_WORDPLAY)
    current = _cut_piece(current, _UNCONFIRMED_WORDPLAY_TAIL)
    current = current.replace(_TAG_QUESTION_LABEL, _TAG_QUESTION_REPLACEMENT)
    return current.strip()


def _cut_piece(text: str, piece: str) -> str:
    """Remove `piece`, and one adjacent space when the piece sits between words."""
    spaced = f" {piece}"
    if spaced in text:
        text = text.replace(spaced, "")
    return text.replace(piece, "")


def _strip_speech_verb_note(text: str) -> str:
    """Remove a leading «помета к глаголу речи», with or without a colon."""
    current = text.strip()
    while current.casefold().startswith(_SPEECH_VERB_NOTE):
        rest = current[len(_SPEECH_VERB_NOTE) :].lstrip()
        if rest.startswith(":"):
            rest = rest[1:].lstrip()
        if rest == current:
            break
        current = rest
    return current


def _strip_editorial_tail(text: str) -> str:
    """Remove a trailing self-check remark. The same words mid-sentence stay."""
    current = text.strip()
    changed = True
    while changed:
        changed = False
        for tail in _EDITORIAL_TAILS:
            if current.endswith(tail):
                current = current[: -len(tail)].rstrip()
                changed = True
                break
    return current


def _agreement_restatement(note: str) -> bool:
    """True when a hint note only restates that the speaker agrees."""
    text = _strip_editorial_tail(
        note.replace(_REFERENCE_COLON, "").replace(_REFERENCE, "")
    ).casefold()
    return any(text.startswith(prefix) for prefix in _AGREEMENT_NOTE_PREFIXES)


def _cleaned_hint_note(note: str | None) -> str:
    """Hint note after boilerplate removal, or empty when it only restates agreement."""
    if not note or not note.strip() or _agreement_restatement(note):
        return ""
    pieces = _body(note)
    return pieces[0] if pieces else ""


def _plain_lexical(
    original: str,
    ids: Sequence[str],
    known: Mapping[str, Difficulty],
) -> bool:
    """A single word (including hyphenated) or a short `-ed` phrasal verb without a signal difficulty."""
    if any(item_id in known and known[item_id].kind in _SIGNAL_KINDS for item_id in ids):
        return False
    words = expression_words(original)
    if len(words) == 1:
        return True
    if words and len(original.strip().split()) == 1:
        return True
    return (
        len(words) == 2
        and words[1] in _PARTICLES
        and _regular_ed_stem(words[0]) is not None
    )


def _to_infinitive_term(original: str) -> str | None:
    """`to` + infinitive for a regular `-ed` verb or verb plus particle.

    An original that is already `to` plus a verb keeps that spelling.
    `to his surprise` is not an infinitive.
    """
    stripped = original.strip()
    if _verbal_infinitive_words(stripped) is not None:
        return stripped
    words = expression_words(original)
    if len(words) == 1:
        stem = _regular_ed_stem(words[0])
        return None if stem is None else f"to {stem}"
    if len(words) == 2 and words[1] in _PARTICLES:
        stem = _regular_ed_stem(words[0])
        return None if stem is None else f"to {stem} {words[1]}"
    return None


def _regular_ed_stem(word: str) -> str | None:
    """Dictionary stem of a regular English past form, without `to`."""
    if len(word) <= 3 or not word.endswith("ed"):
        return None
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word[-3] in "aeiou":
        return None
    if len(word) > 4 and word[-3] == "l" and word[-4] in _LE_CONSONANTS:
        return word[:-1]
    stem = word[:-2]
    if not any(char in "aeiou" for char in stem):
        return None
    if (
        len(stem) >= 4
        and stem[-1] == stem[-2]
        and stem[-1] in _DOUBLED_CONSONANTS
        and _is_vowel(stem, -3)
        and not _is_vowel(stem, -4)
    ):
        return stem[:-1]
    if _magic_e(stem):
        return stem + "e"
    return stem


def _is_vowel(stem: str, index: int) -> bool:
    """Whether `stem[index]` is a vowel. `u` after `q` is not."""
    if index < 0:
        index += len(stem)
    if index < 0 or index >= len(stem):
        return False
    char = stem[index]
    if char not in "aeiou":
        return False
    return not (char == "u" and index > 0 and stem[index - 1] == "q")


def _magic_e(stem: str) -> bool:
    """Whether `stem` dropped a silent `e` before `-ed`."""
    if stem == "us":
        return True
    if len(stem) < 3 or stem in _NO_SILENT_E_WORDS or stem[-1] in "aeiouwxy":
        return False
    if stem.endswith(_SILENT_E_TAILS):
        return True
    if stem.endswith("g") and not stem.endswith(("ng", "gg")) and _is_vowel(stem, -2):
        return True
    if stem.endswith("z") and not stem.endswith("zz"):
        return True
    if stem.endswith("s") and not stem.endswith("ss") and stem not in {"focus", "bias", "bus", "gas"}:
        return True
    vowels = [index for index in range(len(stem)) if _is_vowel(stem, index)]
    if len(vowels) == 1 and vowels[0] == len(stem) - 2:
        return True
    return (
        len(vowels) >= 2
        and _is_vowel(stem, -2)
        and not _is_vowel(stem, -3)
        and stem.endswith(_SILENT_E_SYLLABLES)
    )


def _headword_gloss(note: str, original: str) -> bool:
    """Whether `note` starts by repeating the English word or its stem and a dash."""
    folded = note.strip().casefold()
    for head in _headwords(original):
        if not folded.startswith(head):
            continue
        if folded[len(head) :].lstrip().startswith(_DASHES):
            return True
    return False


def _headwords(original: str) -> tuple[str, ...]:
    """Spellings that count as repeating `original` at the start of a gloss."""
    words = expression_words(original)
    heads: list[str] = []
    if words:
        heads.append(" ".join(words))
    bare = original.strip().casefold()
    if bare:
        heads.append(bare)
    infinitive = _verbal_infinitive_words(original)
    if infinitive:
        heads.append(" ".join(infinitive))
    stem = None
    if len(words) == 1:
        stem = _regular_ed_stem(words[0])
    elif len(words) == 2 and words[1] in _PARTICLES:
        stem = _regular_ed_stem(words[0])
        if stem:
            heads.append(f"{stem} {words[1]}")
    if stem:
        heads.append(stem)
    unique = list(dict.fromkeys(head for head in heads if head))
    unique.sort(key=len, reverse=True)
    return tuple(unique)


def _note_quotes_translation(note: str, translation: str) -> bool:
    """Whether `note` already contains the translation inside « »."""
    quoted = translation.strip()
    if not quoted or not note.strip():
        return False
    return f"«{quoted}»".casefold() in note.casefold()


def _has_russian_infinitive(text: str) -> bool:
    """Whether `text` already contains a Russian verb infinitive."""
    return any(
        match.group(0).casefold().endswith(_RU_INFINITIVE_ENDINGS)
        for match in _RU_TOKEN.finditer(text)
    )


def _russian_past_to_infinitive(text: str) -> str:
    """Past-tense Russian verbs in a gloss become infinitives. Other words stay."""
    converted_in_clause = False
    after_preposition = False
    pieces: list[str] = []
    cursor = 0
    for match in _RU_TOKEN.finditer(text):
        pieces.append(text[cursor : match.start()])
        token = match.group(0)
        cursor = match.end()
        if token in {",", ";"}:
            converted_in_clause = False
            after_preposition = False
            pieces.append(token)
            continue
        folded = token.casefold().replace("ё", "е")
        if folded in _RU_CONJUNCTIONS:
            converted_in_clause = False
            after_preposition = False
            pieces.append(token)
            continue
        if folded in _RU_PREPOSITIONS:
            after_preposition = True
            pieces.append(token)
            continue
        if converted_in_clause or after_preposition:
            after_preposition = False
            pieces.append(token)
            continue
        replaced = _russian_token(token)
        if replaced != token:
            converted_in_clause = True
        pieces.append(replaced)
    pieces.append(text[cursor:])
    return "".join(pieces)


def _russian_token(word: str) -> str:
    key = word.casefold().replace("ё", "е")
    converted = _RU_IRREGULAR.get(key)
    if converted is None:
        converted = _regular_ru_past(key)
    if converted is None:
        return word
    if word[:1].isupper():
        return converted[:1].upper() + converted[1:]
    return converted


def _regular_ru_past(word: str) -> str | None:
    """Infinitive of a regular past form, or None when `word` is not one."""
    for ending, suffix in _RU_PAST:
        if not word.endswith(ending) or len(word) == len(ending):
            continue
        stem = word[: -len(ending)]
        minimum = 2 if suffix.startswith("ну") else 3
        if len(stem) < minimum:
            continue
        return stem + suffix
    return None


def _collapse(drafts: Sequence[_Draft], plan: ContentPlan | None) -> list[LayoutNote]:
    """Collapse drafts that share one difficulty anchor tuple into one note."""
    notes: list[LayoutNote] = []
    consumed = [False] * len(drafts)
    for index, draft in enumerate(drafts):
        if consumed[index]:
            continue
        if draft.anchors is None:
            consumed[index] = True
            notes.append(_note_from((draft,), plan))
            continue
        group: list[_Draft] = []
        for other_index, other in enumerate(drafts):
            if consumed[other_index] or other.anchors != draft.anchors:
                continue
            consumed[other_index] = True
            group.append(other)
        notes.append(_note_from(tuple(group), plan))
    return notes


def _note_from(drafts: Sequence[_Draft], plan: ContentPlan | None) -> LayoutNote:
    """One note. Labels follow plan order; body pieces keep source order."""
    label_ids = [item_id for draft in drafts for item_id in draft.label_ids]
    kind_ids = [item_id for draft in drafts for item_id in draft.kind_ids]
    labels, kinds = _project(label_ids, kind_ids, plan)
    return LayoutNote(
        labels=labels,
        term=_first(draft.term for draft in drafts),
        ipa=_first(draft.ipa for draft in drafts),
        variant=_first(draft.variant for draft in drafts),
        body=tuple(piece for draft in drafts for piece in draft.body),
        marker_end=_earliest(drafts),
        origin=drafts[0].origin,
        difficulty_kinds=kinds,
    )


def _project(
    label_ids: Sequence[str],
    kind_ids: Sequence[str],
    plan: ContentPlan | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Russian labels and kind names in `plan.difficulties` order."""
    if plan is None:
        return (), ()
    labeled = set(label_ids)
    kinds = set(kind_ids)
    labels: list[str] = []
    difficulty_kinds: list[str] = []
    for difficulty in plan.difficulties:
        if difficulty.id in kinds:
            difficulty_kinds.append(difficulty.kind)
        if difficulty.id not in labeled:
            continue
        label = KIND_LABELS.get(difficulty.kind)
        if label is not None:
            labels.append(label)
    return tuple(labels), tuple(difficulty_kinds)


def _first(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def _earliest(drafts: Sequence[_Draft]) -> int | None:
    markers = [draft.marker_end for draft in drafts if draft.marker_end is not None]
    if not markers:
        return None
    return min(markers)


def _marker_key(note: LayoutNote) -> tuple[bool, int]:
    """Marked notes by offset, then unmarked notes. Equal keys keep source order."""
    if note.marker_end is None:
        return (True, 0)
    return (False, note.marker_end)
