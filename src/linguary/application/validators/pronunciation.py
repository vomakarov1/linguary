"""Conservative local selection, independent of translations and model retries."""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import unicodedata

from linguary.application.llm_dto import PronunciationAnchorResponse, PronunciationCandidateResponse
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


def resolve_pronunciation_anchor(
    anchor: PronunciationAnchorResponse,
    texts: Mapping[str, str],
    *,
    used_spans: set[tuple[str, int, int]] | None = None,
) -> SourceAnchor:
    if anchor.block_id not in texts:
        raise DomainError(f"неизвестный блок {anchor.block_id}")
    text = texts[anchor.block_id]

    if anchor.start is not None and anchor.end is not None:
        source_anchor = SourceAnchor(anchor.block_id, anchor.start, anchor.end, anchor.quote)
        source_anchor.validate(texts)
        if used_spans is not None:
            used_spans.add((anchor.block_id, anchor.start, anchor.end))
        return source_anchor

    quote = anchor.quote
    quote_len = len(quote)
    start_pos = 0
    while True:
        pos = text.find(quote, start_pos)
        if pos == -1:
            break
        end_pos = pos + quote_len
        span = (anchor.block_id, pos, end_pos)
        if (used_spans is None or span not in used_spans) and is_complete_pronunciation_form(text, pos, end_pos):
            if used_spans is not None:
                used_spans.add(span)
            return SourceAnchor(anchor.block_id, pos, end_pos, quote)
        start_pos = pos + 1

    raise DomainError(f"не удалось найти целое слово {quote!r} в блоке {anchor.block_id}")


async def verify_candidates(candidates, *, source: PronunciationSource, texts, variant, scope_id):
    if not source.configured:
        return PronunciationSelection(unavailable=True)
    accepted, skipped, unavailable = [], [], []
    used_spans: set[tuple[str, int, int]] = set()
    candidate_cache: dict[tuple, tuple[PronunciationCandidateResponse, SourceAnchor]] = {}
    resolved_candidates: list[tuple[PronunciationCandidateResponse, SourceAnchor]] = []

    for candidate in candidates:
        cache_key = (
            candidate.anchor.block_id,
            candidate.anchor.start,
            candidate.anchor.end,
            candidate.anchor.quote,
            candidate.form,
            candidate.selection_reason,
            (
                candidate.context.meaning_in_context,
                candidate.context.part_of_speech,
                candidate.context.grammatical_form,
                candidate.context.is_proper_name,
            ),
        )
        if cache_key in candidate_cache:
            resolved_candidates.append(candidate_cache[cache_key])
            continue

        try:
            anchor = resolve_pronunciation_anchor(candidate.anchor, texts, used_spans=used_spans)
        except DomainError:
            skipped.append(PronunciationSkip(candidate, 'invalid_anchor'))
            continue

        updated_candidate = replace(
            candidate,
            anchor=replace(candidate.anchor, start=anchor.start, end=anchor.end),
        )
        candidate_cache[cache_key] = (updated_candidate, anchor)
        resolved_candidates.append((updated_candidate, anchor))

    by_anchor = {}
    for cand, anchor in resolved_candidates:
        by_anchor.setdefault((anchor.block_id, anchor.start, anchor.end), []).append((cand, anchor))
    for duplicates in by_anchor.values():
        candidate, anchor = duplicates[0]
        if any(other[0] != candidate for other in duplicates):
            skipped.append(PronunciationSkip(candidate, 'conflicting_candidates'))
            continue
        try:
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
        if any(item.evidence.kind == 'lexicon_form' for item in found.entries):
            matches = _lexicon_matches(found.entries, candidate, variant)
        else:
            if any(not isinstance(value, str) or not value.strip() for value in
                   (context.meaning_in_context, context.part_of_speech, context.grammatical_form)):
                skipped.append(PronunciationSkip(candidate, 'unknown_context'))
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


def _lexicon_matches(entries, candidate, variant):
    if candidate.context.is_proper_name or any(
            item.evidence.kind != 'lexicon_form' or item.form != candidate.form or item.variant != variant
            or item.sense is not None or item.grammatical_form is not None
            or item.evidence.sense_id is not None or item.evidence.part_of_speech is not None
            or item.evidence.rule_id is not None or item.evidence.rule_version is not None
            or item.inflection_rule is not None or item.proper_name_confirmed
            or item.evidence.dictionary_form != item.form or item.evidence.dictionary_ipa != item.ipa
            for item in entries):
        return []
    if len({item.ipa for item in entries}) != 1:
        return []
    return [min(entries, key=lambda item: item.evidence.entry_id)]


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
