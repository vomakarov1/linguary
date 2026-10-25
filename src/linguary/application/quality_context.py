"""Immutable review inputs and source-bound response validation."""
from dataclasses import asdict, dataclass
import json
from linguary.domain.book_conditions import WorkStateError, thaw
from linguary.domain.content_plan import canonical_sha


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


def request_text(manifest):
    return json.dumps(thaw(manifest), ensure_ascii=False, sort_keys=True)


INSTRUCTIONS = ('Review the original and all supplied learning layers. Return source-grounded findings only; '
    'never rewrite artifacts. Enumerate every artifact SHA exactly once in checked_artifacts or missing_artifacts. '
    'Anchor each finding to exact original text offsets and its checked artifact SHA. '
    'Absence of findings is not a guarantee of semantic correctness.')


def validate_response(manifest, response):
    artifacts = {a['artifact_sha']: a for a in manifest['artifacts']}
    checked, missing = response.checked_artifacts, response.missing_artifacts
    if (len(set(checked)) != len(checked) or len(set(missing)) != len(missing)
            or set(checked) & set(missing) or set(checked) | set(missing) != artifacts.keys()):
        raise WorkStateError('неполный или чужой охват ответа вычитки')
    originals = {a['block_ids'][0]: a['body']['text'] for a in artifacts.values() if a['artifact_kind'] == 'original'}
    for finding in response.findings:
        artifact = artifacts.get(finding['artifact_sha'])
        if (artifact is None or finding['artifact_sha'] not in checked
                or artifact['artifact_kind'] != finding['artifact_kind']
                or not finding['block_ids'] or not set(finding['block_ids']) <= set(artifact['block_ids'])
                or not finding['evidence_anchors']):
            raise WorkStateError('чужой артефакт замечания')
        for anchor in finding['evidence_anchors']:
            text = originals.get(anchor['block_id'], '')
            if (anchor['block_id'] not in finding['block_ids'] or anchor['start'] < 0
                    or anchor['end'] <= anchor['start'] or text[anchor['start']:anchor['end']] != anchor['quote']):
                raise WorkStateError('неверная привязка замечания к оригиналу')
    return response
