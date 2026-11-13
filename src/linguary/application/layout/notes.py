"""Notes for one block: hints, language notes, assistance, and pronunciation.

The projection does not persist anything. Difficulties whose `anchors` tuples
are equal share one note. A marker is an offset in the block text; notes
without one keep their original order after the notes that have one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from linguary.application.pronunciation_projection import PronunciationOccurrence
from linguary.domain import BlockProcessingResult, DocumentBlock, GlossaryHint, LanguageNote
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
    "grammar": "Грамматика.",
    "idiom": "Идиома.",
    "cultural_reference": "Реалия.",
    "semantic_relation": "Смысл.",
    "false_friend": "Ложный друг.",
    "speech_style": "Речь персонажа.",
    "parody": "Пародия.",
    "vocabulary": None,
}

_REFERENCE_COLON = "Языковая справка:"
_REFERENCE = "Языковая справка"


@dataclass(frozen=True, slots=True)
class LayoutNote:
    """One block note before display numbering is assigned.

    `labels` are separate kind names. `vocabulary` is omitted there and kept
    in `difficulty_kinds`. `marker_end` is an offset into the block text.
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
            _hint_draft(hint, pronunciation, block.text, result.block_id, evidences, known)
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
) -> _Draft:
    """A glossary hint. The marker is the end of the first exact `original`."""
    content = {"original": hint.original, "translation": hint.translation, "note": hint.note}
    keys = {
        artifact_key("glossary_hint.note", block_id, content),
        artifact_key("glossary_hint.translation", block_id, content),
    }
    ids = _ids_for(evidences, keys)
    parts = [hint.translation]
    if hint.note and hint.note.strip():
        parts.append(hint.note)
    return _Draft(
        label_ids=ids,
        kind_ids=ids,
        term=hint.original,
        ipa=None if pronunciation is None else pronunciation.ipa,
        variant=None if pronunciation is None else pronunciation.variant,
        body=_body(*parts),
        marker_end=_find_end(text, hint.original),
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
    """Drop the language-reference phrase. A piece that becomes empty is omitted.

    The colon form is removed first so a following colon is not left behind.
    """
    cleaned = []
    for part in parts:
        text = part.replace(_REFERENCE_COLON, "").replace(_REFERENCE, "").strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


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
