"""Восстановимый отчёт; неизвестные данные не превращаются в успех."""
from __future__ import annotations

import json
import math
import re
import statistics
import zipfile
from dataclasses import asdict, dataclass, field, is_dataclass
from decimal import Decimal
from pathlib import Path

from linguary.comparison.protocol import LUNA, SONNET


def _data(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if hasattr(value, 'public'):
        return value.public()
    return asdict(value) if is_dataclass(value) else value


def _stage(value):
    return {'chapter_analysis': 'analysis', 'translation_result': 'processing'}.get(value, value)


def _summary(items):
    operations = {}
    for item in sorted(items, key=lambda x: (x['sequence'], x['call_uid'])):
        if item['received_response']:
            operations.setdefault(item['operation_id'], item)
    first = list(operations.values())
    accepted = sum(item.get('accepted') is True and not item.get('truncated') for item in first)
    return {'first_received': len(first), 'first_accepted': accepted,
            'first_rejected': sum(item.get('accepted') is False or item.get('truncated') is True for item in first),
            'first_unknown': sum(item.get('accepted') is None for item in first),
            'rate': accepted / len(first) if first else None,
            'network_or_access_failures': sum(not item['received_response'] for item in items),
            'attempts': len(items), 'rejection_categories': sorted({item['failure_category'] for item in items if item.get('failure_category')}),
            'cost': _cost(items), 'request_seconds': _seconds(items)}


def _cost(items):
    if not items or any(item.get('cost') is None for item in items):
        return None
    return str(sum((Decimal(str(item['cost'])) for item in items), Decimal(0)))


def _seconds(items):
    if not items or any(item.get('duration_ms') is None for item in items):
        return None
    return sum(item['duration_ms'] for item in items) / 1000


def _dispersion(values):
    return {'n': len(values), 'min': min(values) if values else None, 'max': max(values) if values else None,
            'mean': statistics.mean(values) if values else None,
            'stdev': statistics.stdev(values) if len(values) > 1 else None}


@dataclass
class ComparisonDecision:
    status: str
    variant_id: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    manifest: dict
    observations: list[dict]
    variants: dict
    reviews: dict
    projection: dict
    total: dict
    violations: list[str]
    missing_data: list[str]
    simulated: bool
    version: int = 1
    decision: ComparisonDecision | None = None

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, allow_nan=False)

    def to_markdown(self):
        lines = ['# Сравнение учебных EPUB', '', f"Решение: {self.decision.status if self.decision else 'insufficient_data'}",
                 f"Имитация: {'да' if self.simulated else 'нет'}", '', f"Полный расход: {self.total['cost'] if self.total['cost'] is not None else 'неизвестен'}",
                 f"Сумма времени запросов: {self.total['request_seconds']}", f"Время прогона: {self.total['wall_seconds']}", '',
                 'Парные обработки используют общий анализ; их наблюдения зависимы.', '']
        for variant, details in self.variants.items():
            lines.extend([f'## {variant}', '', json.dumps(details, ensure_ascii=False, indent=2), '',
                          'Оценки анализа и исполнения после отсева повторов:', json.dumps(self.reviews.get(variant), ensure_ascii=False), '',
                          'Прогноз самостоятельной книги (включает анализ):', json.dumps(self.projection.get('books', {}).get(variant), ensure_ascii=False), ''])
        lines.extend(['Отсутствующие данные: ' + ', '.join(self.missing_data), 'Нарушения: ' + ', '.join(self.violations)])
        if self.decision:
            lines.append('Основания: ' + '; '.join(self.decision.reasons))
        return '\n'.join(lines) + '\n'


def _outcomes(value):
    if not hasattr(value, 'variants'):
        return _data(value)
    return {
        'protocol_hash': value.protocol_hash, 'simulated': value.simulated, 'violations': value.violations,
        'stage_wall_seconds': value.stage_wall_seconds,
        'manifests': {key: _data(manifest) for key, manifest in value.manifests.items()},
        'analyses': [{'content_hash': a.content_hash, 'provenance': a.provenance} for a in value.analyses],
        'variants': [{key: getattr(branch, key) for key in ('variant_id', 'repeat', 'analysis_hash', 'completed', 'failures', 'completed_chapters', 'epubs')} for branch in value.variants],
    }


def build_report(manifest, ledger, outcomes, reviews, projection) -> ComparisonReport:
    manifest = _data(manifest)
    outcome_data = _outcomes(outcomes)
    projection = dict(projection)
    if not isinstance(outcome_data, dict):
        outcome_data = {'variants': outcomes}
    manifest = dict(manifest, stage_manifests=outcome_data.get('manifests', {}))
    raw = ledger.observations if hasattr(ledger, 'observations') else ledger
    unique, violations = {}, list(outcome_data.get('violations', []))
    if outcome_data.get('protocol_hash') is not None and outcome_data['protocol_hash'] != manifest.get('protocol_hash'):
        violations.append('result_protocol_hash_mismatch')
    for value in raw:
        item = _data(value)
        uid = item['call_uid']
        if uid in unique and unique[uid] != item:
            violations.append('conflicting_receipt:' + uid)
        unique[uid] = item
    observations = list(unique.values())
    variants = {}
    for value in outcome_data.get('variants', []):
        branch = _data(value)
        variant = branch['variant_id']
        entry = variants.setdefault(variant, {'repeats': [], 'analysis': {}, 'processing': {}})
        repeat_data = {key: branch.get(key) for key in ('repeat', 'completed', 'failures', 'completed_chapters', 'analysis_hash', 'epubs')}
        repeat_data['epubs'] = {key: str(path) for key, path in (branch.get('epubs') or {}).items()}
        entry['repeats'].append(repeat_data)
    declared = outcome_data.get('variant_models', {})
    for variant, details in variants.items():
        own = [i for i in observations if i['variant_id'] == variant]
        analysis_model = declared.get(variant, {}).get('analysis_model')
        if analysis_model is None:
            # Происхождение квитанций связано с неизменным снимком анализа.
            hashes = {r['analysis_hash'] for r in details['repeats']}
            calls = {call for a in outcome_data.get('analyses', []) if a.get('content_hash') in hashes for call in a.get('provenance', {}).get('analysis_call_ids', [])}
            analysis = [i for i in observations if i.get('call_id') in calls and _stage(i['stage']) == 'analysis']
        else:
            analysis = [i for i in observations if _stage(i['stage']) == 'analysis' and i['assigned_model'] == analysis_model]
        for stage, items in (('analysis', analysis), ('processing', own)):
            selected = [i for i in items if _stage(i['stage']) == stage]
            details[stage] = _summary(selected)
            details[stage]['repeat_rates'] = _dispersion([summary['rate'] for repeat in sorted({i['repeat'] for i in selected}) if (summary := _summary([i for i in selected if i['repeat'] == repeat]))['rate'] is not None])
        quality_values = (reviews or {}).get(variant, {}).get('repeat_scores', [])
        details['quality_dispersion'] = _dispersion(quality_values) if quality_values and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in quality_values) else _dispersion([])
        details['standalone_sample_cost'] = _cost(analysis + own)
        details['standalone_request_seconds'] = _seconds(analysis + own)
        durations = outcome_data.get('stage_wall_seconds', {})
        walls = []
        for repeat in details['repeats']:
            model = analysis_model or next((i['assigned_model'] for i in analysis if i['repeat'] == repeat['repeat']), None)
            a = durations.get(f"analysis:{repeat['repeat']}:{model}")
            p = durations.get(f"processing:{repeat['repeat']}:{variant}")
            if a is not None and p is not None:
                walls.append(a + p)
        details['standalone_wall_seconds'] = _dispersion(walls)
        factor = projection.get('source_character_ratio')
        if factor is not None and isinstance(factor, (int, float)) and not isinstance(factor, bool) and factor > 0 and projection.get('assumptions'):
            cost = details['standalone_sample_cost']
            count = len(details['repeats'])
            projection.setdefault('books', {})[variant] = {
                'cost': str(Decimal(cost) / count * Decimal(str(factor))) if cost is not None and count else None,
                'wall_seconds': statistics.mean(walls) * factor if count and len(walls) == count else None,
                'analysis_included': True, 'assumptions': projection['assumptions'],
                'method': projection.get('method', 'fixed_source_character_ratio'), 'sample_repeats': count}
    missing = []
    if not reviews:
        missing.append('reviews')
    if _cost(observations) is None:
        missing.append('cost')
    if _seconds(observations) is None:
        missing.append('request_seconds')
    if projection.get('wall_seconds') is None:
        missing.append('wall_seconds')
    if not projection.get('books') or not projection.get('assumptions'):
        missing.append('book_projection')
    return ComparisonReport(manifest, observations, variants, reviews or {}, projection,
                            {'cost': _cost(observations), 'request_seconds': _seconds(observations), 'wall_seconds': projection.get('wall_seconds'), 'unique_calls': len(observations)},
                            sorted(set(violations)), missing, outcome_data.get('simulated', True))


def evaluate_candidate(report, protocol) -> ComparisonDecision:
    reasons = list(report.missing_data)
    if report.violations or report.manifest.get('protocol_hash') != protocol.content_hash:
        return ComparisonDecision('invalid_conditions', reasons=report.violations or ['protocol_hash_mismatch'])
    for item in ([] if report.simulated else report.observations):
        if item.get('received_response') and (item.get('actual_model') is None or item.get('actual_provider') is None or item.get('full_response_cache') is not False):
            return ComparisonDecision('invalid_conditions', reasons=['unknown_route_or_cache'])
        if item.get('received_response') and (item['actual_model'] != item['assigned_model'] or item['actual_provider'] != item['assigned_route']):
            return ComparisonDecision('invalid_conditions', reasons=['changed_route'])
    if report.simulated:
        reasons.append('simulated_results')
    if any(v is None for v in (protocol.sufficient_gain, protocol.comparability_margin, protocol.scale_max, protocol.max_book_cost, protocol.max_book_seconds)):
        reasons.append('incomplete_protocol')
    expected = {v.id for v in protocol.variants}
    if set(report.variants) != expected:
        reasons.append('incomplete_matrix')
    eligible, scores = [], {}
    for variant in protocol.variants:
        vid = variant.id
        details = report.variants.get(vid, {})
        review = report.reviews.get(vid, {})
        book = report.projection.get('books', {}).get(vid, {})
        checks = []
        for stage in ('analysis', 'processing'):
            stats = details.get(stage, {})
            minimum = protocol.minimum_first_responses.get(stage)
            if minimum is None or stats.get('first_received', 0) < minimum or stats.get('first_unknown', 0):
                reasons.append(f'{vid}:{stage}:insufficient_observations')
            checks.append(stats.get('rate') is not None and stats['rate'] >= .95)
        repeats = details.get('repeats', [])
        checks.append(len(repeats) == protocol.repeats and all(r.get('completed') is True and not r.get('failures') for r in repeats))
        quality_spread = details.get('quality_dispersion', {})
        if quality_spread.get('n') != protocol.repeats or quality_spread.get('stdev') is None or (protocol.scale_max is not None and (quality_spread.get('min', -1) < 0 or quality_spread.get('max', float('inf')) > protocol.scale_max)):
            reasons.append(vid + ':quality_dispersion')
        criteria = review.get('criteria', {})
        if not criteria or protocol.scale_max is None or set(criteria) != set(protocol.rubric) or any(value is None for value in criteria.values()):
            reasons.append(vid + ':quality_review')
        else:
            checks.append(all(minimum is not None and minimum <= criteria[key] <= protocol.scale_max for key, minimum in protocol.rubric.items()))
            scores[vid] = statistics.mean(criteria.values())
        critical = review.get('critical_places', {})
        critical_keys = {f'{p.chapter_id}:{p.location}' for p in protocol.critical_places}
        required = ('no_gross_errors', 'simple_help_density_appropriate', 'help_without_duplication', 'analysis_reviewed', 'execution_after_dedup_reviewed', 'independent_chapters_confirmed')
        if not critical_keys or set(critical) != critical_keys or any(critical[k] is None for k in critical_keys) or any(review.get(k) is None for k in required):
            reasons.append(vid + ':semantic_review')
        checks.append(bool(critical_keys) and all(critical.get(k) is True for k in critical_keys) and all(review.get(k) is True for k in required))
        if any(book.get(k) is None for k in ('cost', 'wall_seconds', 'analysis_included')) or not book.get('assumptions'):
            reasons.append(vid + ':book_projection')
        else:
            checks.append(book['analysis_included'] is True and protocol.max_book_cost is not None and Decimal(str(book['cost'])) <= Decimal(str(protocol.max_book_cost)) and protocol.max_book_seconds is not None and book['wall_seconds'] <= protocol.max_book_seconds)
        if all(checks):
            eligible.append(variant)
    if reasons:
        return ComparisonDecision('insufficient_data', reasons=sorted(set(reasons)))
    control = next(v.id for v in protocol.variants if v.analysis_model == v.processing_model == LUNA)
    eligible = [v for v in eligible if v.id != control and scores.get(v.id, -1) - scores.get(control, float('inf')) - report.variants[v.id]['quality_dispersion']['stdev'] - report.variants[control]['quality_dispersion']['stdev'] >= protocol.sufficient_gain]
    if not eligible:
        return ComparisonDecision('no_candidate', reasons=['quality_reliability_gain_or_limits_failed'])
    best = max(scores[v.id] for v in eligible)
    comparable = [v for v in eligible if best - scores[v.id] <= protocol.comparability_margin]
    chosen = min(comparable, key=lambda v: (v.analysis_model != v.processing_model, scores[v.id] * -1, v.id))
    return ComparisonDecision('candidate', chosen.id, ['quality_reliability_gain_and_limits_passed'])


def write_report(report, root):
    root = Path(root)
    (root / 'report.json').write_text(report.to_json(), encoding='utf-8')
    (root / 'report.md').write_text(report.to_markdown(), encoding='utf-8')


def _anonymous_metadata(data):
    """Удалить явно добавленное происхождение, не сериализуя текст книги заново."""
    from html import unescape

    fields = {'model', 'analysis_model', 'processing_model', 'provenance'}
    fields |= {'linguary:' + key for key in fields}

    def remove_field(match):
        element = match.group()
        attributes = re.findall(rb'\b(name|property)\s*=\s*([\"\'])(.*?)\2', element.split(b'>', 1)[0], re.DOTALL)
        if any(unescape(value.decode('utf-8')) in fields for _, _, value in attributes):
            return b''
        return element

    def metadata(match):
        return re.sub(rb'<(?:[\w.-]+:)?meta\b[^>]*?(?:/>|>.*?</(?:[\w.-]+:)?meta\s*>)',
                      remove_field, match.group(), flags=re.DOTALL)

    return re.sub(rb'<(?:[\w.-]+:)?(?:metadata|head)\b[^>]*>.*?</(?:[\w.-]+:)?(?:metadata|head)\s*>',
                  metadata, data, flags=re.DOTALL)


def anonymize_epubs(outcomes, root, model_labels=(LUNA, SONNET, 'luna', 'sonnet'), protocol=None):
    """Материалы рецензента имеют случайные метки; соответствие хранится отдельно."""
    from uuid import uuid4
    root = Path(root)
    materials = root / 'review-materials'
    materials.mkdir(exist_ok=True)
    mapping = {}
    for branch in outcomes['variants']:
        for book, source in branch.get('epubs', {}).items():
            label = uuid4().hex
            target = materials / f'{label}.epub'
            with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, 'w') as anonymous:
                for member in original.infolist():
                    data = original.read(member)
                    if member.filename.endswith(('.opf', '.xhtml', '.html', '.xml', '.ncx')):
                        data = _anonymous_metadata(data)
                    anonymous.writestr(member, data)
            mapping[label] = {'variant_id': branch['variant_id'], 'repeat': branch['repeat'], 'book_id': book, 'chapter_ids': branch.get('epub_chapters', {}).get(book, [])}
    (root / 'review-mapping.json').write_text(json.dumps(mapping, indent=2))
    (root / 'review-mapping.json').chmod(0o600)
    fields = ('no_gross_errors', 'simple_help_density_appropriate', 'help_without_duplication', 'analysis_reviewed', 'execution_after_dedup_reviewed', 'independent_chapters_confirmed')
    sheet = {label: {'criteria': {key: None for key in protocol.rubric} if protocol else {},
                     'critical_places': {f'{p.chapter_id}:{p.location}': None for p in protocol.critical_places if p.chapter_id in mapping[label]['chapter_ids']} if protocol else {},
                     **{key: None for key in fields}} for label in mapping}
    (materials / 'review-sheet.json').write_text(json.dumps(sheet, indent=2))
    return mapping


def aggregate_blind_reviews(mapping, sheet, protocol):
    """Раскрытие меток выполняет оператор после независимой рецензии."""
    result = {}
    fields = ('no_gross_errors', 'simple_help_density_appropriate', 'help_without_duplication', 'analysis_reviewed', 'execution_after_dedup_reviewed', 'independent_chapters_confirmed')
    def agreement(values):
        if not values or any(value is None for value in values):
            return None
        return all(value is True for value in values)
    for variant in protocol.variants:
        entries = [(origin, sheet.get(label, {})) for label, origin in mapping.items() if origin['variant_id'] == variant.id]
        criteria = {}
        for criterion in protocol.rubric:
            scores = [entry.get('criteria', {}).get(criterion) for _, entry in entries]
            criteria[criterion] = statistics.mean(scores) if scores and all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores) else None
        review = {'criteria': criteria, 'critical_places': {f'{place.chapter_id}:{place.location}': agreement([entry.get('critical_places', {}).get(f'{place.chapter_id}:{place.location}') for origin, entry in entries if place.chapter_id in origin.get('chapter_ids', [])]) for place in protocol.critical_places},
                  **{key: agreement([entry.get(key) for _, entry in entries]) for key in fields}, 'repeat_scores': []}
        for repeat in sorted({origin['repeat'] for origin, _ in entries}):
            scores = [entry.get('criteria', {}).get(key) for origin, entry in entries if origin['repeat'] == repeat for key in protocol.rubric]
            if scores and all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores):
                review['repeat_scores'].append(statistics.mean(scores))
        result[variant.id] = review
    return result
