"""Notes for one block: hints, language notes, assistance, and pronunciation.

The projection does not persist anything. Difficulties whose `anchors` tuples
are equal share one note. A marker is an offset in the block text; notes
without one keep their original order after the notes that have one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from linguary.application.explained_expressions import expression_words
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
_DOUBLED_CONSONANTS = frozenset("bdgmnprt")
_LE_CONSONANTS = frozenset("bcdfgkpstz")
_DASHES = ("—", "–", "-")
_RU_WORD = re.compile(r"[А-Яа-яЁё]+")
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
    ("ило", "ить"),
    ("ил", "ить"),
    ("ели", "еть"),
    ("ела", "еть"),
    ("ело", "еть"),
    ("ел", "еть"),
    ("али", "ать"),
    ("ала", "ать"),
    ("ало", "ать"),
    ("ал", "ать"),
    ("яли", "ять"),
    ("яла", "ять"),
    ("яло", "ять"),
    ("ял", "ять"),
    ("ыли", "ыть"),
    ("ыла", "ыть"),
    ("ыло", "ыть"),
    ("ыл", "ыть"),
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
) -> tuple[LayoutNote, ...]:
    """Project help for `block` into notes ordered by marker.

    Glossary hints, language notes, and resolved assistance keep the order
    they have on `result`. A pronunciation whose `quote` equals a hint's
    `original` fills that hint. Any other pronunciation of this block follows
    them. `origin` is `glossary_hint`, `language_note`, `assistance`, or
    `pronunciation`.
    """
    evidences = _evidences(result)
    known = {} if plan is None else {item.id: item for item in plan.difficulties}
    assigned, unmatched = _assign_pronunciations(
        result.glossary_hints, pronunciations, block.id
    )
    drafts: list[_Draft] = []
    for hint, pronunciation in zip(result.glossary_hints, assigned, strict=True):
        drafts.append(
            _hint_draft(
                hint,
                pronunciation,
                block.text,
                result.block_id,
                evidences,
                known,
                result.treatment,
            )
        )
    for note in result.language_notes:
        drafts.append(_note_draft(note, block.text, evidences, known))
    for assistance in result.fragment_assistances:
        if not assistance.resolved:
            continue
        drafts.append(
            _assistance_draft(assistance, block.id, known, english_help=english_help)
        )
    drafts.extend(_pronunciation_draft(item) for item in unmatched)
    notes = _collapse(drafts, plan)
    return tuple(sorted(notes, key=_marker_key))


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
) -> tuple[list[PronunciationOccurrence | None], tuple[PronunciationOccurrence, ...]]:
    """Pair each hint with the first pronunciation of the same quote.

    A quote that equals some hint `original` does not become its own note,
    even when a later copy is left over. Other blocks are ignored.
    """
    local = [item for item in pronunciations if item.block_id == block_id]
    originals = {hint.original for hint in hints}
    used: set[int] = set()
    assigned: list[PronunciationOccurrence | None] = []
    for hint in hints:
        match = None
        for index, item in enumerate(local):
            if index in used or item.quote != hint.original:
                continue
            match = item
            used.add(index)
            break
        assigned.append(match)
    for index, item in enumerate(local):
        if item.quote in originals:
            used.add(index)
    unmatched = tuple(item for index, item in enumerate(local) if index not in used)
    return assigned, unmatched


def _hint_draft(
    hint: GlossaryHint,
    pronunciation: PronunciationOccurrence | None,
    text: str,
    block_id: str,
    evidences: Sequence[HelpEvidence],
    known: Mapping[str, Difficulty],
    treatment: BlockTreatment,
) -> _Draft:
    """A glossary hint. The marker is the end of the first exact `original`.

    Evidence keys and the marker use the stored hint fields. `term` and `body`
    may then drop a repeated sentence, a quoted translation, or a headword
    gloss, and show a past-tense verb as an infinitive.
    """
    content = {"original": hint.original, "translation": hint.translation, "note": hint.note}
    keys = {
        artifact_key("glossary_hint.note", block_id, content),
        artifact_key("glossary_hint.translation", block_id, content),
    }
    ids = _ids_for(evidences, keys)
    marker_end = _find_end(text, hint.original)
    cleaned_note = _cleaned_hint_note(hint.note)
    lexical = _plain_lexical(hint.original, ids, known)
    kept_note = ""
    if cleaned_note and not (lexical and _headword_gloss(cleaned_note, hint.original)):
        kept_note = cleaned_note
    term: str | None = hint.original
    translation = hint.translation
    if lexical:
        infinitive = _to_infinitive_term(hint.original)
        if infinitive is not None:
            term = infinitive
            translation = _russian_past_to_infinitive(hint.translation)
    long = (
        treatment is BlockTreatment.TRANSLATE
        and bool(cleaned_note)
        and len(expression_words(hint.original)) > 5
    )
    if long:
        term = None
        parts = [cleaned_note]
    elif _note_quotes_translation(kept_note, hint.translation):
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
    content = {
        "original_excerpt": note.original_excerpt,
        "note": note.note,
        "translated_excerpt": note.translated_excerpt,
    }
    key = artifact_key("language_note.note", note.block_id, content)
    ids = _ids_for(evidences, {key})
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
    """End of the first exact `str.find` match, or None when it is absent."""
    start = text.find(quote)
    if start < 0:
        return None
    return start + len(quote)


def _last_anchor_end(block_id: str, anchors: Sequence[SourceAnchor]) -> int | None:
    """End of the rightmost anchor that belongs to this block."""
    ends = [anchor.end for anchor in anchors if anchor.block_id == block_id]
    if not ends:
        return None
    return max(ends)


def _body(*parts: str) -> tuple[str, ...]:
    """Drop the language-reference phrase, speech-verb boilerplate, and editorial tails.

    A piece that becomes empty is omitted. The colon form of the reference
    phrase is removed first so a following colon is not left behind.
    """
    cleaned = []
    for part in parts:
        text = part.replace(_REFERENCE_COLON, "").replace(_REFERENCE, "")
        text = _strip_speech_verb_note(text)
        text = _strip_editorial_tail(text)
        if text:
            cleaned.append(text)
    return tuple(cleaned)


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
    """A single word or a short `-ed` phrasal verb without a signal difficulty."""
    if any(item_id in known and known[item_id].kind in _SIGNAL_KINDS for item_id in ids):
        return False
    words = expression_words(original)
    if len(words) == 1:
        return True
    return (
        len(words) == 2
        and words[1] in _PARTICLES
        and _regular_ed_stem(words[0]) is not None
    )


def _to_infinitive_term(original: str) -> str | None:
    """`to` + infinitive for a regular `-ed` verb or verb plus particle."""
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
    """One-syllable CVC stem whose `-ed` spelling hides a silent `e` (`hope`, `like`)."""
    if len(stem) < 3 or stem[-1] in "aeiouwxy":
        return False
    vowels = [index for index in range(len(stem)) if _is_vowel(stem, index)]
    return len(vowels) == 1 and vowels[0] == len(stem) - 2


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


def _russian_past_to_infinitive(text: str) -> str:
    """Past-tense Russian verbs in a gloss become infinitives. Other words stay."""
    return _RU_WORD.sub(lambda match: _russian_token(match.group(0)), text)


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
