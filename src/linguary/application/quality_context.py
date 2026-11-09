"""Immutable review inputs and source-bound response validation."""
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any
from linguary.application.analysis_merge import find_quote_span
from linguary.domain import glossary_key
from linguary.domain.book_conditions import WorkStateError, thaw
from linguary.domain.content_plan import canonical_sha

MAX_QUALITY_CHUNK_BLOCKS = 80

_STRIPPED_MANIFEST_KEYS = (
    'schema_version',
    'book_id',
    'book_manifest_sha',
    'manifest_sha',
    'conditions_id',
    'conditions_sha',
    'route_estimates',
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
    current_glossary = [asdict(g) for g in tx.glossary.list_for_book(book_id)]
    book_manifest_sha = canonical_sha(dict(
        originals=[dict(id=b.id, text=b.text) for b in blocks.values()],
        results=[asdict(r) for r in results.values()], analyses=[asdict(a) for a in analyses],
        glossary=current_glossary, run_id=None if run is None else run.id))
    value = dict(schema_version=1, book_id=book_id, book_manifest_sha=book_manifest_sha, conditions_id=conditions.id,
        conditions_sha=canonical_sha(thaw(conditions.payload)), scope=list(selected), artifacts=artifacts,
        glossary=asdict(run.glossary_snapshot) if run is not None else current_glossary,
        book_complete=bool(blocks) and set(blocks) <= results.keys(),
        missing_results=[b for b in selected if b not in results])
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    value = json.loads(encoded)
    return {**value, 'manifest_sha': sha256(encoded.encode()).hexdigest()}


def _manifest_block_aliases(manifest: Mapping[str, Any]) -> dict[str, str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(block_id: Any) -> None:
        if isinstance(block_id, str) and block_id and block_id not in seen:
            seen.add(block_id)
            ordered.append(block_id)

    if isinstance(manifest, Mapping):
        scope = manifest.get('scope')
        if isinstance(scope, (list, tuple)):
            for block_id in scope:
                add(block_id)
        for artifact in manifest.get('artifacts') or ():
            if isinstance(artifact, Mapping):
                for block_id in artifact.get('block_ids') or ():
                    add(block_id)
        for block_id in manifest.get('missing_results') or ():
            add(block_id)
    return {block_id: f'b{index}' for index, block_id in enumerate(ordered, start=1)}


def submanifest_for_scope(
    manifest: Mapping[str, Any], window_scope: Sequence[str]
) -> dict[str, Any]:
    scope_list = [bid for bid in window_scope if isinstance(bid, str)]
    window_set = set(scope_list)
    sub: dict[str, Any] = dict(manifest)
    sub['scope'] = scope_list
    sub['missing_results'] = [
        bid
        for bid in (manifest.get('missing_results') or ())
        if isinstance(bid, str) and bid in window_set
    ]
    raw_artifacts = manifest.get('artifacts') or ()
    sub['artifacts'] = [
        dict(a)
        for a in raw_artifacts
        if isinstance(a, Mapping)
        and (a.get('artifact_kind') or a.get('kind')) in ('analysis', 'original', 'result')
        and bool(set(a.get('block_ids') or ()) & window_set)
    ]
    return sub


def partition_manifest(
    manifest: Mapping[str, Any],
    *,
    max_blocks_per_chunk: int = MAX_QUALITY_CHUNK_BLOCKS,
) -> tuple[dict[str, Any], ...]:
    if max_blocks_per_chunk < 1:
        raise WorkStateError('неверный размер пакета вычитки')
    raw_scope = manifest.get('scope') or ()
    scope = [bid for bid in raw_scope if isinstance(bid, str)]
    if len(scope) <= max_blocks_per_chunk:
        out = dict(manifest)
        if 'scope' in out:
            out['scope'] = list(scope)
        return (out,)

    raw_artifacts = manifest.get('artifacts') or ()
    block_to_analysis_idx: dict[str, int] = {}
    for idx, artifact in enumerate(raw_artifacts):
        if isinstance(artifact, Mapping) and artifact.get('artifact_kind') == 'analysis':
            for bid in artifact.get('block_ids') or ():
                if isinstance(bid, str) and bid not in block_to_analysis_idx:
                    block_to_analysis_idx[bid] = idx

    chapter_groups: list[list[str]] = []
    analysis_groups: dict[int, list[str]] = {}
    current_unassigned: list[str] | None = None
    for bid in scope:
        a_idx = block_to_analysis_idx.get(bid)
        if a_idx is not None:
            current_unassigned = None
            group = analysis_groups.get(a_idx)
            if group is None:
                group = []
                analysis_groups[a_idx] = group
                chapter_groups.append(group)
            group.append(bid)
        else:
            if current_unassigned is None:
                current_unassigned = []
                chapter_groups.append(current_unassigned)
            current_unassigned.append(bid)

    windows: list[dict[str, Any]] = []
    for group in chapter_groups:
        for start in range(0, len(group), max_blocks_per_chunk):
            window_scope = group[start : start + max_blocks_per_chunk]
            windows.append(submanifest_for_scope(manifest, window_scope))
    return tuple(windows)


def _needs_plan_id_alias(raw_id: Any, prefix: str) -> bool:
    return isinstance(raw_id, str) and (raw_id.startswith(prefix) or len(raw_id) > 16)


def _manifest_difficulty_aliases(manifest: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not isinstance(manifest, Mapping):
        return aliases
    for artifact in manifest.get('artifacts') or ():
        if not isinstance(artifact, Mapping) or artifact.get('artifact_kind') != 'analysis':
            continue
        body = artifact.get('body')
        if not isinstance(body, Mapping):
            continue
        cp = body.get('content_plan')
        if not isinstance(cp, Mapping) or not isinstance(cp.get('difficulties'), (list, tuple)):
            continue
        for idx, d in enumerate(cp['difficulties'], start=1):
            if isinstance(d, Mapping):
                diff_id = d.get('id')
                if _needs_plan_id_alias(diff_id, 'df-'):
                    aliases[diff_id] = f'd{idx}'
    return aliases


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
    difficulty_aliases: Mapping[str, str] | None = None,
    scope_block_ids: Any = None,
) -> dict[str, Any]:
    alias_map = aliases or {}
    reverse_alias_map = {alias: bid for bid, alias in alias_map.items()}
    diff_alias_map = difficulty_aliases or {}
    artifact_bids = [
        bid for bid in (artifact.get('block_ids') or ()) if isinstance(bid, str)
    ]
    raw_scope = (
        {bid for bid in scope_block_ids if isinstance(bid, str)}
        if scope_block_ids is not None
        else set(artifact_bids)
    )
    allowed_scope = (
        (
            {alias_map.get(bid, bid) for bid in raw_scope}
            | {reverse_alias_map.get(bid, bid) for bid in raw_scope}
            | raw_scope
        )
        if raw_scope
        else None
    )
    out = {k: v for k, v in artifact.items() if k != 'artifact_sha'}
    kind = out.get('artifact_kind')
    if isinstance(out.get('block_ids'), (list, tuple)):
        aliased_bids = [
            alias_map.get(bid, bid) if isinstance(bid, str) else bid
            for bid in out['block_ids']
        ]
        if kind == 'analysis' and allowed_scope is not None:
            aliased_bids = [
                bid for bid in aliased_bids
                if not isinstance(bid, str) or bid in allowed_scope
            ]
        out['block_ids'] = aliased_bids
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
            fallback_diff_aliases: dict[str, str] = {}
            for fa in cleaned_result['fragment_assistances']:
                if isinstance(fa, Mapping):
                    item = dict(fa)
                    diff_id = item.get('difficulty_id')
                    if isinstance(diff_id, str):
                        if diff_id in diff_alias_map:
                            item['difficulty_id'] = diff_alias_map[diff_id]
                        elif diff_id in alias_map:
                            item['difficulty_id'] = alias_map[diff_id]
                        elif _needs_plan_id_alias(diff_id, 'df-'):
                            if diff_id not in fallback_diff_aliases:
                                fallback_diff_aliases[diff_id] = f'd{len(fallback_diff_aliases) + 1}'
                            item['difficulty_id'] = fallback_diff_aliases[diff_id]
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
                items = [
                    {
                        **item,
                        'block_id': alias_map.get(item['block_id'], item['block_id']),
                    }
                    if isinstance(item, Mapping) and isinstance(item.get('block_id'), str)
                    else item
                    for item in cleaned_tp['items']
                ]
                if allowed_scope is not None:
                    items = [
                        item
                        for item in items
                        if not isinstance(item, Mapping)
                        or not isinstance(item.get('block_id'), str)
                        or item.get('block_id') in allowed_scope
                    ]
                cleaned_tp['items'] = items
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
                for idx, d in enumerate(cleaned_cp['difficulties'], start=1):
                    if isinstance(d, Mapping):
                        if allowed_scope is not None:
                            in_scope = (
                                d.get('owner_block_id') in allowed_scope
                                or bool(set(d.get('span_block_ids') or ()) & allowed_scope)
                                or any(
                                    isinstance(a, Mapping) and a.get('block_id') in allowed_scope
                                    for a in (d.get('anchors') or ())
                                )
                            )
                            if not in_scope:
                                continue
                        diff = dict(d)
                        diff_id = diff.get('id')
                        if isinstance(diff_id, str):
                            if diff_id in diff_alias_map:
                                diff['id'] = diff_alias_map[diff_id]
                            elif _needs_plan_id_alias(diff_id, 'df-'):
                                diff['id'] = f'd{idx}'
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
                for idx, f in enumerate(cleaned_cp['context_facts'], start=1):
                    if isinstance(f, Mapping):
                        if allowed_scope is not None:
                            subj = f.get('subject_anchor')
                            in_scope = (
                                bool(set(f.get('applies_to_block_ids') or ()) & allowed_scope)
                                or (isinstance(subj, Mapping) and subj.get('block_id') in allowed_scope)
                                or any(
                                    isinstance(a, Mapping) and a.get('block_id') in allowed_scope
                                    for a in (f.get('anchors') or ())
                                )
                            )
                            if not in_scope:
                                continue
                        fact = dict(f)
                        fact_id = fact.get('id')
                        if _needs_plan_id_alias(fact_id, 'cf-'):
                            fact['id'] = f'f{idx}'
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


_ALLOWED_GLOSSARY_ENTRY_KEYS = ('source_text', 'translation', 'type', 'status')


def _clean_glossary_entry(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    cleaned = {k: entry[k] for k in _ALLOWED_GLOSSARY_ENTRY_KEYS if k in entry}
    note = entry.get('note')
    if isinstance(note, str) and note.strip():
        cleaned['note'] = note
    return cleaned


def _filter_glossary_for_manifest(glossary: Any, original_texts: list[str]) -> Any:
    if original_texts:
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
    else:
        def keep(entry: Any) -> bool:
            return True

    if isinstance(glossary, Mapping) and isinstance(glossary.get('entries'), (list, tuple)):
        return {'entries': [_clean_glossary_entry(e) for e in glossary['entries'] if keep(e)]}
    if isinstance(glossary, (list, tuple)):
        return [_clean_glossary_entry(e) for e in glossary if keep(e)]
    return glossary


_QUALITY_REQUEST_PREFIX_KEYS = (
    'scope',
    'book_complete',
    'missing_results',
    'glossary',
)


def _normalize_sorted_keys(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _order_artifacts_for_request(
    raw_artifacts: Sequence[Any],
    scope: Sequence[Any],
) -> list[Any]:
    analysis_indices = [
        idx
        for idx, a in enumerate(raw_artifacts)
        if isinstance(a, Mapping) and a.get('artifact_kind') == 'analysis'
    ]
    if len(analysis_indices) <= 1:
        return list(raw_artifacts)

    scope_pos = {
        bid: pos
        for pos, bid in enumerate(scope)
        if isinstance(bid, str)
    }
    block_to_analysis: dict[str, int] = {}
    for a_idx in analysis_indices:
        for bid in raw_artifacts[a_idx].get('block_ids') or ():
            if isinstance(bid, str) and bid not in block_to_analysis:
                block_to_analysis[bid] = a_idx

    ordered_groups: list[tuple[int | None, list[str]]] = []
    seen_analyses: dict[int, list[str]] = {}
    current_unassigned: list[str] | None = None
    for bid in scope:
        if not isinstance(bid, str):
            continue
        a_idx = block_to_analysis.get(bid)
        if a_idx is not None:
            current_unassigned = None
            grp = seen_analyses.get(a_idx)
            if grp is None:
                grp = []
                seen_analyses[a_idx] = grp
                ordered_groups.append((a_idx, grp))
            grp.append(bid)
        else:
            if current_unassigned is None:
                current_unassigned = []
                ordered_groups.append((None, current_unassigned))
            current_unassigned.append(bid)

    for a_idx in analysis_indices:
        if a_idx not in seen_analyses:
            grp_bids = [
                bid
                for bid in (raw_artifacts[a_idx].get('block_ids') or ())
                if isinstance(bid, str)
            ]
            ordered_groups.append((a_idx, grp_bids))

    used: set[int] = set()
    ordered: list[Any] = []
    for a_idx, grp_bids in ordered_groups:
        grp_set = set(grp_bids)
        if a_idx is not None and a_idx not in used:
            ordered.append(raw_artifacts[a_idx])
            used.add(a_idx)
        for target_kind in ('original', 'result'):
            matching_indices = [
                idx
                for idx, art in enumerate(raw_artifacts)
                if idx not in used
                and isinstance(art, Mapping)
                and art.get('artifact_kind') == target_kind
                and bool(set(art.get('block_ids') or ()) & grp_set)
            ]
            matching_indices.sort(
                key=lambda idx: (
                    min(
                        (
                            scope_pos[b]
                            for b in (raw_artifacts[idx].get('block_ids') or ())
                            if isinstance(b, str) and b in scope_pos
                        ),
                        default=len(scope_pos) + idx,
                    ),
                    idx,
                )
            )
            for idx in matching_indices:
                ordered.append(raw_artifacts[idx])
                used.add(idx)

    for idx, art in enumerate(raw_artifacts):
        if idx not in used:
            ordered.append(art)
    return ordered


def request_text(manifest: Mapping[str, Any]) -> str:
    payload = thaw(manifest)
    if isinstance(payload, Mapping):
        aliases = _manifest_block_aliases(payload)
        difficulty_aliases = _manifest_difficulty_aliases(payload)
        raw_artifacts = payload.get('artifacts')
        raw_scope = payload.get('scope') if isinstance(payload.get('scope'), (list, tuple)) else ()
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
            ordered_raw = _order_artifacts_for_request(raw_artifacts, raw_scope)
            cleaned['artifacts'] = [
                _llm_artifact(
                    a,
                    aliases,
                    difficulty_aliases=difficulty_aliases,
                    scope_block_ids=cleaned.get('scope'),
                )
                if isinstance(a, Mapping)
                else a
                for a in ordered_raw
            ]
        ordered_payload: dict[str, Any] = {}
        for key in _QUALITY_REQUEST_PREFIX_KEYS:
            if key in cleaned:
                ordered_payload[key] = _normalize_sorted_keys(cleaned[key])
        for key in sorted(cleaned):
            if key not in ordered_payload and key != 'artifacts':
                ordered_payload[key] = _normalize_sorted_keys(cleaned[key])
        if 'artifacts' in cleaned:
            ordered_payload['artifacts'] = _normalize_sorted_keys(cleaned['artifacts'])
        return json.dumps(
            ordered_payload,
            ensure_ascii=False,
            sort_keys=False,
            separators=(',', ':'),
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


INSTRUCTIONS = (
    'Проверь оригинал и все переданные учебные слои. Возвращай только замечания, подтверждённые '
    'оригиналом; никогда не переписывай артефакты. Привязывай каждое замечание к точной цитате в '
    'целевом блоке, используя короткие идентификаторы block_id вида b1, b2 из запроса. Отсутствие '
    'замечаний не является гарантией смысловой безошибочности.'
)


def _preview_quote(quote: Any, limit: int = 80) -> str:
    if not isinstance(quote, str):
        return ''
    compact = re.sub(r'\s+', ' ', quote).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + '…'


def _finding_error(
    prefix: str,
    finding_index: int,
    *,
    block_label: str = '',
    quote: Any = None,
) -> WorkStateError:
    parts = [f'замечание {finding_index}']
    if block_label:
        parts.append(f'блок {block_label}')
    preview = _preview_quote(quote)
    if preview:
        parts.append(f'цитата «{preview}»')
    return WorkStateError(f"{prefix} ({', '.join(parts)})")


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
    by_kind_block: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_kind_blocks: set[tuple[str, str]] = set()
    for a in manifest['artifacts']:
        a_kind = a.get('artifact_kind') or a.get('kind')
        a_bids = a.get('block_ids') or ()
        if isinstance(a_kind, str) and len(a_bids) == 1 and isinstance(a_bids[0], str):
            key = (a_kind, a_bids[0])
            if key in by_kind_block:
                duplicate_kind_blocks.add(key)
            else:
                by_kind_block[key] = a
    for key in duplicate_kind_blocks:
        by_kind_block.pop(key, None)

    updated_findings = []
    for finding_index, finding in enumerate(response.findings, start=1):
        f = dict(finding)
        if isinstance(f.get('block_ids'), (list, tuple)):
            f['block_ids'] = [
                reverse_aliases.get(bid, bid) if isinstance(bid, str) else bid
                for bid in f['block_ids']
            ]
        artifact_sha = f.get('artifact_sha')
        ordered_bids = list(f.get('block_ids') or ())
        f_bids = set(ordered_bids)
        kind = f.get('artifact_kind')
        raw_anchors = f.get('evidence_anchors')
        first_quote = (
            raw_anchors[0].get('quote', '')
            if isinstance(raw_anchors, (list, tuple))
            and raw_anchors
            and isinstance(raw_anchors[0], Mapping)
            else ''
        )
        finding_block_label = ', '.join(
            aliases.get(b, str(b)) if isinstance(b, str) else str(b)
            for b in ordered_bids
            if b
        )
        multi_block_single_artifact = kind in {'result', 'original'} and len(f_bids) > 1
        if multi_block_single_artifact:
            if not ordered_bids:
                raise _finding_error(
                    'чужой артефакт замечания',
                    finding_index,
                    block_label=finding_block_label,
                    quote=first_quote,
                )
            per_block = []
            for bid in ordered_bids:
                matched_art = by_kind_block.get((kind, bid))
                if matched_art is None or matched_art['artifact_sha'] not in checked:
                    raise _finding_error(
                        'чужой артефакт замечания',
                        finding_index,
                        block_label=aliases.get(bid, str(bid)) if isinstance(bid, str) else str(bid),
                        quote=first_quote,
                    )
                per_block.append(matched_art)
            first_art = per_block[0]
            allowed_shas = {art['artifact_sha'] for art in per_block}
            if artifact_sha:
                if artifact_sha not in allowed_shas:
                    raise _finding_error(
                        'чужой артефакт замечания',
                        finding_index,
                        block_label=finding_block_label,
                        quote=first_quote,
                    )
            else:
                artifact_sha = first_art['artifact_sha']
                f['artifact_sha'] = artifact_sha
        elif not artifact_sha:
            if not ordered_bids:
                raise _finding_error(
                    'чужой артефакт замечания',
                    finding_index,
                    block_label=finding_block_label,
                    quote=first_quote,
                )
            matching = [
                a for a in manifest['artifacts']
                if (a.get('artifact_kind') or a.get('kind')) == kind and (
                    f_bids <= set(a['block_ids']) if kind == 'analysis' else set(a['block_ids']) == f_bids
                )
            ]
            if len(matching) != 1:
                raise _finding_error(
                    'чужой артефакт замечания',
                    finding_index,
                    block_label=finding_block_label,
                    quote=first_quote,
                )
            artifact_sha = matching[0]['artifact_sha']
            f['artifact_sha'] = artifact_sha

        artifact = artifacts.get(artifact_sha)
        if (artifact is None or artifact_sha not in checked
                or artifact['artifact_kind'] != kind
                or not ordered_bids
                or (
                    artifact['artifact_sha'] not in {art['artifact_sha'] for art in per_block}
                    if multi_block_single_artifact
                    else not f_bids <= set(artifact['block_ids'])
                )
                or not f.get('evidence_anchors')):
            raise _finding_error(
                'чужой артефакт замечания',
                finding_index,
                block_label=finding_block_label,
                quote=first_quote,
            )

        updated_anchors = []
        for anchor in f['evidence_anchors']:
            anc = dict(anchor)
            raw_bid = anc.get('block_id')
            block_id = raw_bid
            if isinstance(block_id, str):
                block_id = reverse_aliases.get(block_id, block_id)
                anc['block_id'] = block_id
            anchor_label = (
                aliases.get(block_id, block_id)
                if isinstance(block_id, str) and block_id
                else (str(raw_bid) if raw_bid else finding_block_label)
            )
            quote = anc.get('quote', '')
            if block_id not in f['block_ids']:
                raise _finding_error(
                    'неверная привязка замечания к оригиналу',
                    finding_index,
                    block_label=anchor_label,
                    quote=quote,
                )
            text = originals.get(block_id, '')
            start = anc.get('start')
            end = anc.get('end')
            if start is None or end is None:
                span = find_quote_span(text, quote)
                if span is None:
                    raise _finding_error(
                        'неверная привязка замечания к оригиналу',
                        finding_index,
                        block_label=anchor_label,
                        quote=quote,
                    )
                start, end, matched = span
                anc['start'], anc['end'], anc['quote'] = start, end, matched
            elif start < 0 or end <= start or text[start:end] != quote:
                raise _finding_error(
                    'неверная привязка замечания к оригиналу',
                    finding_index,
                    block_label=anchor_label,
                    quote=quote,
                )
            updated_anchors.append(anc)
        f['evidence_anchors'] = updated_anchors
        updated_findings.append(f)

    return QualityReviewResponse(
        findings=tuple(updated_findings),
        checked_artifacts=tuple(checked),
        missing_artifacts=tuple(missing),
    )
