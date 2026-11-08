"""Immutable review inputs and source-bound response validation."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from typing import Any
from linguary.application.analysis_merge import find_quote_span
from linguary.domain import glossary_key
from linguary.domain.book_conditions import WorkStateError, thaw
from linguary.domain.content_plan import canonical_sha

_STRIPPED_MANIFEST_KEYS = (
    'schema_version',
    'book_id',
    'book_manifest_sha',
    'manifest_sha',
    'conditions_id',
    'conditions_sha',
)


@dataclass(frozen=True)
class QualityReviewResponse:
    findings: tuple[dict, ...]
    checked_artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]


def snapshot(tx, book_id, scope):
    from linguary.application.context_builder import textual_blocks
    document = tx.documents.get(book_id)
    conditions = tx.work.latest_conditions(book_id)
    if document is None or conditions is None:
        raise WorkStateError('conditions_unavailable')
    blocks = {block.id: block for section in document.sections for block in textual_blocks(section)}
    selected = tuple(blocks) if scope == 'all' else tuple(scope)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= blocks.keys():
        raise WorkStateError('неверный охват вычитки')
    run = tx.runs.active_for_book(book_id)
    results = {} if run is None else {r.block_id: r for r in tx.results.list_for_run(run.id)}
    analyses = tx.analyses.list_for_book(book_id)
    artifacts = []
    def add(kind, ids, body):
        envelope = dict(artifact_kind=kind, block_ids=list(ids), body=body)
        artifacts.append({**envelope, 'artifact_sha': canonical_sha(envelope)})
    for block_id in selected:
        add('original', (block_id,), dict(text=blocks[block_id].text))
        if block_id in results:
            add('result', (block_id,), asdict(results[block_id]))
    for analysis in analyses:
        ids = tuple(b for b in selected if b in analysis.treatment_plan.block_ids)
        if ids:
            add('analysis', ids, asdict(analysis))
    glossary = tuple(tx.glossary.list_for_book(book_id)) if run is None else run.glossary_snapshot
    current_glossary = [asdict(g) for g in tx.glossary.list_for_book(book_id)]
    book_manifest_sha = canonical_sha(dict(
        originals=[dict(id=b.id, text=b.text) for b in blocks.values()],
        results=[asdict(r) for r in results.values()], analyses=[asdict(a) for a in analyses],
        glossary=current_glossary, run_id=None if run is None else run.id))
    value = dict(schema_version=1, book_id=book_id, book_manifest_sha=book_manifest_sha, conditions_id=conditions.id,
        conditions_sha=canonical_sha(thaw(conditions.payload)), scope=list(selected), artifacts=artifacts,
        glossary=asdict(glossary) if run else [asdict(g) for g in glossary],
        book_complete=bool(blocks) and set(blocks) <= results.keys(),
        missing_results=[b for b in selected if b not in results])
    value = json.loads(json.dumps(value, ensure_ascii=False))
    return {**value, 'manifest_sha': canonical_sha(value)}


def _manifest_block_aliases(manifest: Mapping[str, Any]) -> dict[str, str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(block_id: Any) -> None:
        if isinstance(block_id, str) and block_id and block_id not in seen:
            seen.add(block_id)
            ordered.append(block_id)

    if isinstance(manifest, Mapping):
        for artifact in manifest.get('artifacts') or ():
            if isinstance(artifact, Mapping):
                for block_id in artifact.get('block_ids') or ():
                    add(block_id)
        scope = manifest.get('scope')
        if isinstance(scope, (list, tuple)):
            for block_id in scope:
                add(block_id)
        for block_id in manifest.get('missing_results') or ():
            add(block_id)
    return {block_id: f'b{index}' for index, block_id in enumerate(ordered, start=1)}


def _strip_anchor_offsets(anchor: Any, aliases: Mapping[str, str] | None = None) -> Any:
    if isinstance(anchor, Mapping):
        out = {k: v for k, v in anchor.items() if k not in ('start', 'end')}
        if aliases and isinstance(out.get('block_id'), str):
            out['block_id'] = aliases.get(out['block_id'], out['block_id'])
        return out
    return anchor


def _llm_artifact(
    artifact: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    alias_map = aliases or {}
    out = {k: v for k, v in artifact.items() if k != 'artifact_sha'}
    if isinstance(out.get('block_ids'), (list, tuple)):
        out['block_ids'] = [
            alias_map.get(bid, bid) if isinstance(bid, str) else bid
            for bid in out['block_ids']
        ]
    kind = out.get('artifact_kind')
    body = out.get('body')
    if not isinstance(body, Mapping):
        return out
    if kind == 'result':
        cleaned_result = {
            k: v
            for k, v in body.items()
            if k not in ('block_id', 'content_evidence', 'pronunciation_assistances', 'warnings')
        }
        if isinstance(cleaned_result.get('language_notes'), (list, tuple)):
            notes = []
            for n in cleaned_result['language_notes']:
                if isinstance(n, Mapping):
                    note = {k: v for k, v in n.items() if k != 'id'}
                    for bid_key in ('block_id', 'source_block_id'):
                        if isinstance(note.get(bid_key), str):
                            note[bid_key] = alias_map.get(note[bid_key], note[bid_key])
                    notes.append(note)
                else:
                    notes.append(n)
            cleaned_result['language_notes'] = notes
        if isinstance(cleaned_result.get('fragment_assistances'), (list, tuple)):
            assistances = []
            for fa in cleaned_result['fragment_assistances']:
                if isinstance(fa, Mapping):
                    item = dict(fa)
                    if isinstance(item.get('anchors'), (list, tuple)):
                        item['anchors'] = [_strip_anchor_offsets(a, alias_map) for a in item['anchors']]
                    assistances.append(item)
                else:
                    assistances.append(fa)
            cleaned_result['fragment_assistances'] = assistances
        out['body'] = cleaned_result
    elif kind == 'analysis':
        cleaned = {
            k: v
            for k, v in body.items()
            if k not in (
                'id',
                'section_id',
                'coverage_status',
                'glossary_candidates',
                'name_translation_evidence',
                'before_you_read_evidence',
            )
        }
        tp = cleaned.get('treatment_plan')
        if isinstance(tp, Mapping):
            cleaned_tp = {k: v for k, v in tp.items() if k != 'section_id'}
            if isinstance(cleaned_tp.get('items'), (list, tuple)):
                cleaned_tp['items'] = [
                    {
                        **item,
                        'block_id': alias_map.get(item['block_id'], item['block_id']),
                    }
                    if isinstance(item, Mapping) and isinstance(item.get('block_id'), str)
                    else item
                    for item in cleaned_tp['items']
                ]
            cleaned['treatment_plan'] = cleaned_tp
        cp = cleaned.get('content_plan')
        if isinstance(cp, Mapping):
            cleaned_cp = {
                k: v
                for k, v in cp.items()
                if k not in ('version', 'profile', 'sha256', 'before_you_read_evidence')
            }
            if isinstance(cleaned_cp.get('difficulties'), (list, tuple)):
                difficulties = []
                for d in cleaned_cp['difficulties']:
                    if isinstance(d, Mapping):
                        diff = dict(d)
                        if isinstance(diff.get('owner_block_id'), str):
                            diff['owner_block_id'] = alias_map.get(diff['owner_block_id'], diff['owner_block_id'])
                        if isinstance(diff.get('anchors'), (list, tuple)):
                            diff['anchors'] = [_strip_anchor_offsets(a, alias_map) for a in diff['anchors']]
                        difficulties.append(diff)
                    else:
                        difficulties.append(d)
                cleaned_cp['difficulties'] = difficulties
            if isinstance(cleaned_cp.get('context_facts'), (list, tuple)):
                facts = []
                for f in cleaned_cp['context_facts']:
                    if isinstance(f, Mapping):
                        fact = dict(f)
                        if isinstance(fact.get('anchors'), (list, tuple)):
                            fact['anchors'] = [_strip_anchor_offsets(a, alias_map) for a in fact['anchors']]
                        if isinstance(fact.get('subject_anchor'), Mapping):
                            fact['subject_anchor'] = _strip_anchor_offsets(fact['subject_anchor'], alias_map)
                        facts.append(fact)
                    else:
                        facts.append(f)
                cleaned_cp['context_facts'] = facts
            cleaned['content_plan'] = cleaned_cp
        out['body'] = cleaned
    return out


def _matches_manifest_text(
    term_source: str,
    blocks_text: str,
    word_stems: set[str],
    normalized_blocks_text: str,
) -> bool:
    from linguary.application.context_builder import _matches_chapter_text
    try:
        return _matches_chapter_text(
            term_source,
            blocks_text,
            word_stems,
            normalized_blocks_text=normalized_blocks_text,
        )
    except TypeError:
        return _matches_chapter_text(term_source, normalized_blocks_text, word_stems)


def _filter_glossary_for_manifest(glossary: Any, original_texts: list[str]) -> Any:
    if not original_texts:
        return glossary
    from linguary.application.context_builder import _extract_word_stems
    blocks_text = ' '.join(original_texts)
    word_stems = _extract_word_stems(blocks_text)
    normalized_blocks_text = glossary_key(blocks_text)

    def keep(entry: Any) -> bool:
        if not isinstance(entry, Mapping):
            return True
        source = entry.get('source_text') or entry.get('canonical_source_text') or ''
        if not isinstance(source, str) or not source:
            return True
        return _matches_manifest_text(source, blocks_text, word_stems, normalized_blocks_text)

    if isinstance(glossary, Mapping) and isinstance(glossary.get('entries'), (list, tuple)):
        return {**glossary, 'entries': [e for e in glossary['entries'] if keep(e)]}
    if isinstance(glossary, (list, tuple)):
        return [e for e in glossary if keep(e)]
    return glossary


def request_text(manifest: Mapping[str, Any]) -> str:
    payload = thaw(manifest)
    if isinstance(payload, Mapping):
        aliases = _manifest_block_aliases(payload)
        raw_artifacts = payload.get('artifacts')
        original_texts = [
            a['body']['text']
            for a in (raw_artifacts if isinstance(raw_artifacts, (list, tuple)) else ())
            if isinstance(a, Mapping)
            and a.get('artifact_kind') == 'original'
            and isinstance(a.get('body'), Mapping)
            and isinstance(a['body'].get('text'), str)
        ]
        cleaned: dict[str, Any] = {
            k: v for k, v in payload.items() if k not in _STRIPPED_MANIFEST_KEYS
        }
        if isinstance(cleaned.get('scope'), (list, tuple)):
            cleaned['scope'] = [
                aliases.get(bid, bid) if isinstance(bid, str) else bid
                for bid in cleaned['scope']
            ]
        if isinstance(cleaned.get('missing_results'), (list, tuple)):
            cleaned['missing_results'] = [
                aliases.get(bid, bid) if isinstance(bid, str) else bid
                for bid in cleaned['missing_results']
            ]
        if 'glossary' in cleaned:
            cleaned['glossary'] = _filter_glossary_for_manifest(cleaned['glossary'], original_texts)
        if isinstance(raw_artifacts, (list, tuple)):
            cleaned['artifacts'] = [
                _llm_artifact(a, aliases) if isinstance(a, Mapping) else a
                for a in raw_artifacts
            ]
        payload = cleaned
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


INSTRUCTIONS = (
    'Проверь оригинал и все переданные учебные слои. Возвращай только замечания, подтверждённые '
    'оригиналом; никогда не переписывай артефакты. Привязывай каждое замечание к точной цитате в '
    'целевом блоке, используя короткие идентификаторы block_id вида b1, b2 из запроса. Отсутствие '
    'замечаний не является гарантией смысловой безошибочности.'
)


def validate_response(manifest: Mapping[str, Any], response: QualityReviewResponse) -> QualityReviewResponse:
    aliases = _manifest_block_aliases(manifest)
    reverse_aliases = {alias: block_id for block_id, alias in aliases.items()}
    artifacts = {a['artifact_sha']: a for a in manifest['artifacts']}
    checked = response.checked_artifacts
    missing = response.missing_artifacts
    if not checked and not missing:
        checked = tuple(artifacts.keys())
        missing = ()
    if (len(set(checked)) != len(checked) or len(set(missing)) != len(missing)
            or set(checked) & set(missing) or set(checked) | set(missing) != artifacts.keys()):
        raise WorkStateError('неполный или чужой охват ответа вычитки')
    originals = {a['block_ids'][0]: a['body']['text'] for a in artifacts.values() if a['artifact_kind'] == 'original'}
    updated_findings = []
    for finding in response.findings:
        f = dict(finding)
        if isinstance(f.get('block_ids'), (list, tuple)):
            f['block_ids'] = [
                reverse_aliases.get(bid, bid) if isinstance(bid, str) else bid
                for bid in f['block_ids']
            ]
        artifact_sha = f.get('artifact_sha')
        if not artifact_sha:
            kind = f.get('artifact_kind')
            f_bids = set(f.get('block_ids') or ())
            matching = [
                a for a in manifest['artifacts']
                if a['artifact_kind'] == kind and (
                    f_bids <= set(a['block_ids']) if kind == 'analysis' else set(a['block_ids']) == f_bids
                )
            ]
            if len(matching) != 1:
                raise WorkStateError('чужой артефакт замечания')
            artifact_sha = matching[0]['artifact_sha']
            f['artifact_sha'] = artifact_sha

        artifact = artifacts.get(artifact_sha)
        if (artifact is None or artifact_sha not in checked
                or artifact['artifact_kind'] != f.get('artifact_kind')
                or not f.get('block_ids') or not set(f['block_ids']) <= set(artifact['block_ids'])
                or not f.get('evidence_anchors')):
            raise WorkStateError('чужой артефакт замечания')

        updated_anchors = []
        for anchor in f['evidence_anchors']:
            anc = dict(anchor)
            block_id = anc.get('block_id')
            if isinstance(block_id, str):
                block_id = reverse_aliases.get(block_id, block_id)
                anc['block_id'] = block_id
            if block_id not in f['block_ids']:
                raise WorkStateError('неверная привязка замечания к оригиналу')
            text = originals.get(block_id, '')
            start = anc.get('start')
            end = anc.get('end')
            quote = anc.get('quote', '')
            if start is None or end is None:
                span = find_quote_span(text, quote)
                if span is None:
                    raise WorkStateError('неверная привязка замечания к оригиналу')
                start, end, matched = span
                anc['start'], anc['end'], anc['quote'] = start, end, matched
            elif start < 0 or end <= start or text[start:end] != quote:
                raise WorkStateError('неверная привязка замечания к оригиналу')
            updated_anchors.append(anc)
        f['evidence_anchors'] = updated_anchors
        updated_findings.append(f)

    return QualityReviewResponse(
        findings=tuple(updated_findings),
        checked_artifacts=tuple(checked),
        missing_artifacts=tuple(missing),
    )
