"""Local provenance supplement; historical receipts remain immutable."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path

from linguary.comparison.protocol import file_hash, stable_hash
from linguary.comparison.workspace import WorkspaceError, _bound_child, _continuation_inventory


@dataclass(frozen=True)
class LateMetadataSupplement:
    document: dict[str, object]

    @property
    def content_hash(self):
        return stable_hash(self.document)


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise WorkspaceError('Цена должна быть конечным числом')
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise WorkspaceError('Цена должна быть конечным неотрицательным числом')
    return result


def _integer(value):
    if type(value) is not int or value < 0:
        raise WorkspaceError('Счётчик должен быть неотрицательным целым числом')
    return value


def _utc(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise WorkspaceError('Диагностика должна содержать время UTC')
    return result


def _read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_late_metadata(*, evidence_path, evidence_sha256, source_root, source_inventory, protocol):
    """Verifies saved series_3 without contacting the API or SQLite."""
    try:
        root, evidence = Path(source_root), Path(evidence_path)
        if (not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents, evidence, *evidence.parents))
                or not (root / '.comparison-owner').is_file()
                or _continuation_inventory(root) != source_inventory or file_hash(evidence) != evidence_sha256):
            raise WorkspaceError('Диагностика или полный исходный inventory изменились')
        series = _read_json(evidence)['series_3']
        if (series['purpose'] != 'free_late_generation_metadata_diagnosis_not_result_rewrite'
                or series['method'] != 'GET /api/v1/generation?id=<generation_id>, as ReceiptHTTPClient._generation_model; trust_env proxy auto; follow_redirects=False'
                or _integer(series['new_POST']) != 0 or _integer(series['GET_requests']) != 1
                or _integer(series['status']) != 200 or series['body_class'] != 'JSON_generation_metadata'
                or len(series['attempts']) != 1):
            raise WorkspaceError('Нет зарегистрированного успешного GET поколения')
        attempt = series['attempts'][0]
        observed = _utc(attempt['at_utc'])
        if (_integer(attempt['status']) != 200 or observed < _utc(series['receipt_mtime_utc'])
                or _utc(series['receipt_mtime_utc']) < _utc(series['run_started_at_utc'])):
            raise WorkspaceError('Время или статус GET противоречат диагностике')
        headers = attempt['selected_headers']
        header_time = parsedate_to_datetime(headers['Date'])
        if (headers['Content-Type'] != 'application/json' or headers['Cache-Control'] != 'private, no-store'
                or headers['CF-Cache-Status'] != 'BYPASS' or header_time.tzinfo is None
                or abs((header_time - observed).total_seconds()) > 60):
            raise WorkspaceError('Источник HTTP-диагностики не подтверждён')
        protected = series['protected_files_sha256']
        receipt_paths = [name for name in protected if name.endswith('.json') and '/receipts/' in '/' + name]
        if (len(protected) != 4 or len(receipt_paths) != 2 or not {'result.json', 'report.json'} <= set(protected)
                or any(source_inventory.get(name) != digest or file_hash(_bound_child(root, name)) != digest for name, digest in protected.items())):
            raise WorkspaceError('Защищённые квитанции и результаты изменились')
        receipts = [_read_json(_bound_child(root, name)) for name in receipt_paths]
        receipt = receipts[0]
        if receipts[1] != receipt:
            raise WorkspaceError('Копии квитанции противоречат друг другу')
        uid = receipt['call_uid']
        if (f'receipts/{uid}.json' not in receipt_paths or not isinstance(uid, str) or not uid
                or any(Path(name).name != uid + '.json' for name in receipt_paths)
                or any(not isinstance(receipt[key], str) or not receipt[key] for key in ('operation_id', 'call_id', 'generation_id', 'chapter_id', 'group_id'))):
            raise WorkspaceError('Идентичность оплаченной операции неполна')
        manifest = _read_json(root / 'manifest.json')
        result = _read_json(root / 'result.json')
        report = _read_json(root / 'report.json')
        if (manifest['protocol_hash'] != protocol.content_hash or result['protocol_hash'] != protocol.content_hash
                or report['manifest']['protocol_hash'] != protocol.content_hash or report['violations'] != result['violations']
                or result['simulated'] is not False
                or _read_json(root / 'prepared.json') != {'protocol_hash': protocol.content_hash, 'manifest_hash': stable_hash(manifest)}):
            raise WorkspaceError('Протокол и результаты не согласованы')
        selected = series['selected']
        route = protocol.route_metadata[receipt['assigned_model']]
        if (series['generation_id'] != receipt['generation_id'] or selected['id'] != receipt['generation_id']
                or selected['model'] not in route.model_versions or selected['provider_name'] not in route.provider_names
                or selected['provider_name'] != receipt['actual_provider']
                or receipt['assigned_route'] != protocol.routes[receipt['assigned_model']]
                or receipt['reported_model'] != receipt['assigned_model'] or receipt['stage'] != 'chapter_analysis'
                or receipt['variant_id'] != f"analysis-{receipt['assigned_model']}"
                or _integer(receipt['repeat']) != 1 or _integer(receipt['sequence']) < 1
                or receipt['cost_source'] != 'provider_receipt' or _decimal(selected['total_cost']) != _decimal(receipt['cost'])):
            raise WorkspaceError('ID, точная версия, провайдер или цена не подтверждены')
        for key in ('submitted', 'received_response', 'first_response', 'accepted'):
            if receipt[key] is not True:
                raise WorkspaceError('Первый принятый ответ не подтверждён')
        if (receipt['truncated'] is not False or receipt['full_response_cache'] is not False
                or type(receipt['context_cache']) is not bool
                or receipt['actual_model'] is not None or receipt['model_identity_source'] is not None
                or _integer(receipt['metadata_status_code']) != 404 or receipt['metadata_check_reason'] != 'deadline_exceeded'
                or _integer(receipt['metadata_get_count']) < 1):
            raise WorkspaceError('Кеш или исходный отказ не согласованы')
        for field in ('native_tokens_prompt', 'native_tokens_completion', 'native_tokens_cached', 'native_tokens_reasoning', 'tokens_prompt', 'tokens_completion', 'generation_time', 'latency'):
            _integer(selected[field])
        if (selected['native_tokens_prompt'] != _integer(receipt['usage']['prompt_tokens'])
                or selected['native_tokens_completion'] != _integer(receipt['usage']['completion_tokens'])
                or bool(selected['native_tokens_cached']) is not receipt['context_cache']
                or selected['cache_discount'] is not None and _decimal(selected['cache_discount']) != 0
                or selected['cancelled'] is not False or selected['is_byok'] is not False
                or type(selected['streamed']) is not bool or selected['finish_reason'] != 'stop'
                or selected['native_finish_reason'] != 'end_turn'
                or not _utc(series['run_started_at_utc']) <= _utc(selected['created_at']) <= _utc(series['receipt_mtime_utc'])):
            raise WorkspaceError('Счётчики, кеш или завершение ответа противоречат квитанции')
        if _continuation_inventory(root) != source_inventory or file_hash(evidence) != evidence_sha256:
            raise WorkspaceError('Источники изменились во время проверки')
        return LateMetadataSupplement({
            'version': 1, 'source_root': str(root), 'source_inventory': dict(source_inventory),
            'protocol': protocol.model_dump(mode='json'), 'protocol_hash': protocol.content_hash,
            'diagnosis': {'path': str(evidence.absolute()), 'sha256': evidence_sha256, 'json_pointer': '/series_3',
                         'method': series['method'], 'purpose': series['purpose'], 'attempt': attempt,
                         'provenance': 'saved_operator_diagnosis_not_provider_signature_or_independent_API_check'},
            'source_hashes': {name: source_inventory[name] for name in ('manifest.json', 'prepared.json', *protected)},
            'receipt_paths': receipt_paths, 'receipt': receipt, 'verified': selected,
            'historical': {'metadata_status_code': receipt['metadata_status_code'], 'metadata_check_reason': receipt['metadata_check_reason'],
                           'violations': result['violations'], 'report': report},
            'verdict': 'verified_late_metadata', 'accounting_delta': {'posts': 0, 'cost': '0', 'first_responses': 0},
            'live_permission': False,
        })
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError):
        raise WorkspaceError('Не удалось подтвердить сохранённые поздние метаданные') from None
