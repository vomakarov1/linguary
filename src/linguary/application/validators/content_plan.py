"""Structural completeness validation against final visible help bodies."""
from dataclasses import dataclass
import re

from linguary.application.retry_policy import ResultValidationError
from linguary.domain import DomainError, without_footnote_markers
from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key


@dataclass(frozen=True, slots=True)
class Artifact:
    kind: str
    block_id: str
    key: str
    body: str
    quotes: tuple[str, ...]


def artifact_catalog(results, before_you_read, owner_block_ids):
    artifacts = []
    def add(kind, owner, content, body, quotes):
        artifacts.append(Artifact(kind, owner, artifact_key(kind, owner, content), body or '', quotes))
    for owner in owner_block_ids:
        for entry in before_you_read:
            add('before_you_read.explanation', owner, dict(expression=entry.expression, explanation=entry.explanation), entry.explanation, (entry.expression,))
    for result in results:
        for note in result.language_notes:
            add('language_note.note', note.block_id, dict(original_excerpt=note.original_excerpt, note=note.note, translated_excerpt=note.translated_excerpt), note.note, (note.original_excerpt, note.translated_excerpt or ''))
        for hint in result.glossary_hints:
            content = dict(original=hint.original, translation=hint.translation, note=hint.note)
            for field in ('note', 'translation'):
                add('glossary_hint.' + field, result.block_id, content, getattr(hint, field), (hint.original, hint.translation))
    return tuple(artifacts)


def _substance(text):
    return ''.join(re.findall(r'\w+', without_footnote_markers(text).casefold()))


def validate_content_coverage(plan, results, before_you_read, evidence, *, owner_block_ids):
    owners = set(owner_block_ids)
    if evidence.version != plan.version or evidence.plan_sha256 != plan.sha256 or {r.block_id for r in results} != owners:
        raise ResultValidationError('устаревший план или неполный состав владельцев помощи')
    catalog = {(a.kind, a.block_id, a.key): a for a in artifact_catalog(results, before_you_read, owners)}
    checked = []
    for item in evidence.evidence:
        try:
            plan.validate_evidence(item)
        except DomainError as error:
            raise ResultValidationError(str(error)) from error
        if item.block_id not in owners:
            raise ResultValidationError('свидетельство чужого владельца')
        difficulty = next(d for d in plan.difficulties if d.id == item.difficulty_id)
        a = catalog.get((item.artifact_kind, item.block_id, item.artifact_key))
        if a is None or item.body_excerpt not in a.body:
            raise ResultValidationError('свидетельство не указывает на конечное тело помощи')
        if item.artifact_kind.endswith('.translation'):
            if difficulty.help_kind != 'lexical_hint' or not _substance(item.body_excerpt):
                raise ResultValidationError('перевод не покрывает объяснение')
        elif not _substance(item.body_excerpt) or _substance(item.body_excerpt) in {_substance(q) for q in (*a.quotes, *(anchor.quote for anchor in difficulty.anchors))}:
            raise ResultValidationError('цитата или перевод без объясняющего тела')
        checked.append(item)
    covered = {(e.difficulty_id, e.point_id) for e in checked}
    required = {(d.id, p.id) for d in plan.difficulties if d.owner_block_id in owners for p in d.required_points}
    if not required <= covered:
        raise ResultValidationError('не все обязательные аспекты имеют конечное объяснение')
    return tuple(checked)


def resolve_response_evidence(plan, payload, results, before_you_read, *, owner_block_ids):
    catalog = artifact_catalog(results, before_you_read, owner_block_ids)
    evidence = [e for e in plan.before_you_read_evidence if e.block_id in owner_block_ids]
    by_owner = {r.block_id: r for r in results}
    for item in payload.content_evidence:
        owner, kind, index = item.source_block_id, item.artifact_kind, item.artifact_index
        try:
            result = by_owner[owner]
            if kind == 'before_you_read.explanation':
                entry = before_you_read[index]
                content = dict(expression=entry.expression, explanation=entry.explanation)
            elif kind == 'language_note.note':
                raw = payload.language_notes[index]
                if raw.source_block_id != owner:
                    raise IndexError
                local = sum(n.source_block_id == owner for n in payload.language_notes[:index])
                note = result.language_notes[local]
                content = dict(original_excerpt=note.original_excerpt, note=note.note, translated_excerpt=note.translated_excerpt)
            else:
                hint = result.glossary_hints[index]
                content = dict(original=hint.original, translation=hint.translation, note=hint.note)
            key = artifact_key(kind, owner, content)
            a = next(a for a in catalog if (a.kind, a.block_id, a.key) == (kind, owner, key))
        except (KeyError, IndexError, StopIteration) as error:
            raise ResultValidationError('ссылка на отсутствующий артефакт помощи') from error
        excerpt = item.body_excerpt if plan.version == 3 else without_footnote_markers(item.body_excerpt)
        if plan.version == 3:
            from linguary.application.reader_text import ReaderTextPolicy
            excerpt = ReaderTextPolicy().prepare(kind, excerpt, a.quotes)
        elif excerpt != item.body_excerpt:
            excerpt = excerpt.strip()
        if not excerpt:
            raise ResultValidationError('пустое свидетельство помощи')
        if a.body and excerpt not in a.body and plan.version != 3:
            from linguary.application.analysis_merge import _quotation_key, find_quote_span
            span = find_quote_span(a.body, excerpt)
            if span is not None:
                excerpt = span[2]
            else:
                stripped = _quotation_key(excerpt)
                if stripped and stripped in a.body:
                    excerpt = stripped
        evidence.append(HelpEvidence(item.difficulty_id, item.point_id, kind, owner, a.key, excerpt))
    return ContentEvidence(plan.sha256, tuple(evidence), version=plan.version)
