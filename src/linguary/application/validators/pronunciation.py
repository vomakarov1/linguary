"""Conservative local selection, independent of translations and model retries."""
from dataclasses import dataclass
import unicodedata

from linguary.application.llm_dto import PronunciationCandidateResponse
from linguary.application.ports.pronunciation import PronunciationSource, valid_lookup_form
from linguary.domain.content_plan import SourceAnchor, canonical_sha
from linguary.domain.errors import DomainError
from linguary.domain.pronunciation import PronunciationAssistance


@dataclass(frozen=True, slots=True)
class PronunciationSkip:
    candidate: PronunciationCandidateResponse
    reason: str


@dataclass(frozen=True, slots=True)
class PronunciationSelection:
    accepted: tuple[PronunciationAssistance, ...] = ()
    skipped: tuple[PronunciationSkip, ...] = ()
    unavailable_anchors: tuple[SourceAnchor, ...] = ()
    unavailable: bool = False


_WORD_JOINERS = frozenset("'’ʼ-‐‑")


def is_complete_pronunciation_form(text: str, start: int, end: int) -> bool:
    """Match a whole Unicode word without normalizing text or its offsets."""
    if not 0 <= start < end <= len(text):
        return False

    def is_letter(char):
        return char not in _WORD_JOINERS and unicodedata.category(char).startswith('L')

    index = 0
    while index <= start:
        if not is_letter(text[index]):
            index += 1
            continue
        word_start = index
        index += 1
        while index < len(text):
            char = text[index]
            if is_letter(char) or unicodedata.category(char).startswith('M'):
                index += 1
            elif char in _WORD_JOINERS and index + 1 < len(text) and is_letter(text[index + 1]):
                index += 2
            else:
                break
        if index > start:
            return word_start == start and index == end
    return False


def _normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split()).casefold()


def pronunciation_identity(item):
    return (_normalized(item.form), unicodedata.normalize('NFC', item.ipa).strip().strip('/'), item.variant)


@dataclass(frozen=True, slots=True)
class PronunciationGroup:
    hint_id: str
    primary: PronunciationAssistance
    occurrences: tuple[PronunciationAssistance, ...]


def group_pronunciations(occurrences, *, block_order):
    groups = {}
    for item in occurrences:
        groups.setdefault(pronunciation_identity(item), []).append(item)
    ordered = []
    for items in groups.values():
        items.sort(key=lambda item: (block_order[item.anchor.block_id], item.anchor.start, item.anchor.end))
        ordered.append(PronunciationGroup(items[0].hint_id, items[0], tuple(items)))
    return tuple(sorted(ordered, key=lambda group: (block_order[group.primary.anchor.block_id], group.primary.anchor.start)))


async def verify_candidates(candidates, *, source: PronunciationSource, texts, variant, scope_id):
    if not source.configured:
        return PronunciationSelection(unavailable=True)
    accepted, skipped, unavailable = [], [], []
    by_anchor = {}
    for candidate in candidates:
        raw = candidate.anchor
        by_anchor.setdefault((raw.block_id, raw.start, raw.end), []).append(candidate)
    for duplicates in by_anchor.values():
        candidate = duplicates[0]
        if any(other != candidate for other in duplicates):
            skipped.append(PronunciationSkip(candidate, 'conflicting_candidates'))
            continue
        try:
            anchor = SourceAnchor(candidate.anchor.block_id, candidate.anchor.start, candidate.anchor.end, candidate.anchor.quote)
            anchor.validate(texts)
        except DomainError:
            skipped.append(PronunciationSkip(candidate, 'invalid_anchor'))
            continue
        if candidate.form != anchor.quote:
            skipped.append(PronunciationSkip(candidate, 'invalid_form'))
            continue
        if not is_complete_pronunciation_form(texts[anchor.block_id], anchor.start, anchor.end):
            skipped.append(PronunciationSkip(candidate, 'partial_word'))
            continue
        if not valid_lookup_form(candidate.form):
            skipped.append(PronunciationSkip(candidate, 'invalid_form'))
            continue
        if candidate.selection_reason not in {'unpredictable_spelling', 'unpredictable_stress', 'context_dependent'}:
            skipped.append(PronunciationSkip(candidate, 'selection_reason'))
            continue
        context = candidate.context
        if any(not isinstance(value, str) or not value.strip() for value in
               (context.meaning_in_context, context.part_of_speech, context.grammatical_form)):
            skipped.append(PronunciationSkip(candidate, 'unknown_context'))
            continue
        try:
            found = await source.lookup(candidate.form, variant)
        except Exception:
            unavailable.append(anchor)
            skipped.append(PronunciationSkip(candidate, 'source_unavailable'))
            continue
        if found.status == 'unavailable':
            unavailable.append(anchor)
            skipped.append(PronunciationSkip(candidate, 'source_unavailable'))
            continue
        matches = [item for item in found.entries if item.form == candidate.form and item.variant == variant
            and _normalized(item.sense) == _normalized(context.meaning_in_context)
            and _normalized(item.evidence.part_of_speech) == _normalized(context.part_of_speech)
            and _normalized(item.grammatical_form) == _normalized(context.grammatical_form)
            and (not context.is_proper_name or item.proper_name_confirmed)]
        if len(matches) != 1:
            skipped.append(PronunciationSkip(candidate, 'missing_or_ambiguous'))
            continue
        item, = matches
        if item.evidence.rule_id is not None and (item.inflection_rule is None or not item.inflection_rule.confirms(item)):
            skipped.append(PronunciationSkip(candidate, 'unproven_inflection'))
            continue
        try:
            assistance = PronunciationAssistance(anchor, item.form, item.ipa, item.variant,
                'ph-' + canonical_sha((scope_id, pronunciation_identity(item))), item.evidence)
        except DomainError:
            skipped.append(PronunciationSkip(candidate, 'invalid_evidence'))
            continue
        accepted.append(assistance)
    return PronunciationSelection(tuple(accepted), tuple(skipped), tuple(unavailable))


def preserve_unavailable(selection, previous, *, texts, variant):
    """Keep failed occurrences of the same run; checked misses deliberately disappear."""
    merged = {item.anchor: item for item in selection.accepted}
    failed = set(selection.unavailable_anchors)
    for item in previous:
        if item.variant != variant or not (selection.unavailable or item.anchor in failed):
            continue
        try:
            item.anchor.validate(texts)
        except DomainError:
            continue
        if not is_complete_pronunciation_form(texts[item.anchor.block_id], item.anchor.start, item.anchor.end):
            continue
        merged.setdefault(item.anchor, item)
    return tuple(sorted(merged.values(), key=lambda item: (item.anchor.block_id, item.anchor.start, item.anchor.end)))
