"""Immutable review inputs and source-bound response validation."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from typing import Any
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


INSTRUCTIONS = (
    'Review the original and all supplied learning layers. Return source-grounded findings only; '
    'never rewrite artifacts. Anchor each finding to exact quote in the target block. '
    'Absence of findings is not a guarantee of semantic correctness.'
)


def validate_response(manifest: Mapping[str, Any], response: QualityReviewResponse) -> QualityReviewResponse:
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
            if block_id not in f['block_ids']:
                raise WorkStateError('неверная привязка замечания к оригиналу')
            text = originals.get(block_id, '')
            start = anc.get('start')
            end = anc.get('end')
            quote = anc.get('quote', '')
            if start is None or end is None:
                pos = text.find(quote)
                if pos == -1:
                    raise WorkStateError('неверная привязка замечания к оригиналу')
                start = pos
                end = pos + len(quote)
                anc['start'] = start
                anc['end'] = end
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
