"""Явная локальная оболочка; клиент создаётся только после допуска live."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import httpx

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.comparison.llm import ObservedLLMClient
from linguary.comparison.protocol import ComparisonManifest, ProtocolApproval, ProtocolError, build_manifest, file_hash, load_protocol, settings_projection, stable_hash, validate_for_live
from linguary.comparison.report import aggregate_blind_reviews, anonymize_epubs, build_report, evaluate_candidate, write_report
from linguary.comparison.runner import run_comparison
from linguary.comparison.workspace import ComparisonWorkspace, WorkspaceError, prepare_workspace


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _safe_root(root, protected):
    root = Path(root).expanduser().resolve()
    protected = tuple(Path(p).expanduser().resolve() for p in protected)
    if not protected or any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise WorkspaceError('Корень сравнения пересекается с защищёнными данными')
    if not (root / '.comparison-owner').is_file() or any(p.is_symlink() for p in root.rglob('*')):
        raise WorkspaceError('Нет принадлежащего сравнению каталога без ссылок')
    return root, protected


def _inputs(root):
    values = _read(root / 'inputs.json')
    paths = {}
    for key, item in values.items():
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path) != item['sha256']:
            raise ProtocolError('Копия источника изменилась')
        paths[key] = path
    return paths


def prepare(root, protected, protocol):
    workspace = prepare_workspace(root, protected, protocol)
    manifest = build_manifest(protocol, _inputs(workspace.root), workspace.settings)
    manifest = manifest.model_copy(update={'artifact_hashes': {name: file_hash(workspace.root / name) for name in ('inputs.json', 'context.json', 'settings.json')}, 'paths': {'result': 'result.json', 'receipts': 'receipts', 'review_materials': 'review-materials', 'review_mapping': 'review-mapping.json'}})
    (workspace.root / 'manifest.json').write_text(manifest.model_dump_json(indent=2))
    # Хеш допуска принадлежит стадии подготовки; замена манифеста отменяет допуск.
    (workspace.root / 'prepared.json').write_text(json.dumps({'manifest_hash': stable_hash(manifest.model_dump(mode='json')), 'protocol_hash': protocol.content_hash}))
    return workspace


def _restore(root, protected, protocol):
    root, protected = _safe_root(root, protected)
    manifest = ComparisonManifest.model_validate(_read(root / 'manifest.json'))
    prepared = _read(root / 'prepared.json')
    if prepared != {'manifest_hash': stable_hash(manifest.model_dump(mode='json')), 'protocol_hash': protocol.content_hash} or manifest.protocol_hash != protocol.content_hash:
        raise ProtocolError('Протокол или подготовленный манифест изменён')
    inputs = _inputs(root)
    if {key: file_hash(path) for key, path in inputs.items()} != manifest.input_hashes:
        raise ProtocolError('Источники не соответствуют манифесту')
    for chapter in protocol.chapters:
        if chapter.source and (chapter.id not in inputs or file_hash(Path(chapter.source).expanduser()) != manifest.input_hashes[chapter.id]):
            raise ProtocolError('Исходный EPUB изменён после подготовки')
    for name, digest in manifest.artifact_hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ProtocolError('Подготовленные условия изменены')
    data = _read(root / 'settings.json')
    settings = AppSettings(llm=LLMSettings(**data['llm']), checks=CheckSettings(**data['checks']), worker=WorkerPoolSettings(**data['worker']), storage=StorageSettings(data_root=str(root / 'data')))
    if settings.llm.proxy_mode != protocol.proxy_mode:
        raise ProtocolError('Подготовленный режим прокси не соответствует протоколу')
    if settings_projection(settings) != manifest.effective_settings:
        raise ProtocolError('Эффективные настройки изменились')
    return ComparisonWorkspace(root, protected, protocol, settings, (root / '.comparison-owner').read_text()), manifest


def result_document(result, protocol, root):
    def relative(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise WorkspaceError('Материал результата выходит за каталог сравнения')
        return str(path.relative_to(root))
    return {'protocol_hash': result.protocol_hash, 'simulated': result.simulated, 'violations': result.violations,
            'variant_models': {v.id: {'analysis_model': v.analysis_model, 'processing_model': v.processing_model} for v in protocol.variants},
            'stage_wall_seconds': result.stage_wall_seconds, 'manifests': {key: value.model_dump(mode='json') for key, value in result.manifests.items()},
            'analyses': [{'content_hash': a.content_hash, 'provenance': a.provenance, 'path': relative(a.root)} for a in result.analyses],
            'variants': [{'variant_id': v.variant_id, 'repeat': v.repeat, 'analysis_hash': v.analysis_hash, 'completed': v.completed,
                          'failures': v.failures, 'completed_chapters': v.completed_chapters, 'epub_chapters': v.epub_chapters,
                          'epubs': {k: relative(p) for k, p in v.epubs.items()}} for v in result.variants]}


def save_result(result, protocol, root, wall_seconds):
    document = result_document(result, protocol, root)
    document['wall_seconds'] = wall_seconds
    (root / 'result.json').write_text(json.dumps(document, ensure_ascii=False, indent=2))
    # Квитанции на диске сохраняют расход и при остановке незавершённого прогона.
    for item in result.observations:
        path = root / 'receipts' / f'{item.call_uid}.json'
        path.parent.mkdir(exist_ok=True, mode=0o700)
        path.write_text(json.dumps(item.public(), ensure_ascii=False))
        path.chmod(0o600)
    return document


def cleanup_diagnostics(root, protected, *, all_rejected=False):
    root, _ = _safe_root(root, protected)
    from linguary.comparison.llm import ObservationLedger
    for directory in root.rglob("diagnostics"):
        if not directory.is_dir():
            continue
        ledger = ObservationLedger(diagnostic_root=directory)
        ledger.cleanup(now=float("inf") if all_rejected else None)
        if ledger.violations or (all_rejected and any(directory.iterdir())):
            raise ProtocolError("Не удалось безопасно очистить диагностику")


def local_report(root, protected, protocol):
    root, protected = _safe_root(root, protected)
    cleanup_diagnostics(root, protected, all_rejected=True)
    manifest = ComparisonManifest.model_validate(_read(root / 'manifest.json'))
    if (root / 'result.json').is_file():
        outcomes = _read(root / 'result.json')
    else:
        outcomes = {'protocol_hash': protocol.content_hash, 'simulated': False, 'variants': [], 'violations': ['run_incomplete']}
    observations = [_read(path) for path in sorted(root.rglob('receipts/*.json'))]
    # Пути результатов ограничены принадлежащими сравнению материалами.
    for branch in outcomes['variants']:
        for key, stored in branch.get('epubs', {}).items():
            path = (root / stored).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise WorkspaceError('Недопустимый путь конечного EPUB')
            branch['epubs'][key] = str(path)
    if not (root / 'review-mapping.json').exists():
        anonymize_epubs(outcomes, root, protocol=protocol)
    reviews = _read(root / 'reviews.json') if (root / 'reviews.json').is_file() else aggregate_blind_reviews(_read(root / 'review-mapping.json'), _read(root / 'review-materials' / 'review-sheet.json'), protocol)
    projection = _read(root / 'projection.json') if (root / 'projection.json').is_file() else {}
    projection = dict(projection, wall_seconds=outcomes.get('wall_seconds'), stage_wall_seconds=outcomes.get('stage_wall_seconds', {}))
    report = build_report(manifest, observations, outcomes, reviews, projection)
    report.decision = evaluate_candidate(report, protocol)
    write_report(report, root)
    return report


class LiveClientFactory:
    simulated = False
    def __init__(self, protocol, api_key, http_factory=httpx.AsyncClient):
        if protocol.cost_bounds is None or (protocol.cost_bounds.input_bound_method != "provider_context_limit" and http_factory is httpx.AsyncClient):
            raise ProtocolError("Реальный запуск закрыт: не доказана граница оплачиваемого входа")
        self.protocol, self.api_key, self.http_factory = protocol, api_key, http_factory

    def __call__(self, *, model, settings, ledger, budget, **kwargs):
        if not self.api_key:
            raise ProtocolError('Не задан ключ COMPARISON_API_KEY')
        if settings.proxy_mode != self.protocol.proxy_mode:
            raise ProtocolError('Режим прокси клиента не соответствует протоколу')
        settings = replace(settings, api_key=self.api_key, base_url='https://openrouter.ai/api/v1')
        settings.ensure_ready()
        if self.protocol.cost_bounds is None or self.protocol.cost_bounds.input_bound_method == "unverified":
            raise ProtocolError("Не доказана граница оплачиваемого входа")
        return ObservedLLMClient(settings=settings, route=self.protocol.routes[model], http_client=self.http_factory(timeout=settings.timeout_seconds, trust_env=settings.proxy_mode == 'auto'),
                                 ledger=ledger, budget=budget, max_call_cost=Decimal(str(self.protocol.max_call_cost)), temperature_policy=self.protocol.temperature_policy, cost_bounds=self.protocol.cost_bounds, route_metadata=self.protocol.route_metadata[model])


async def execute(args, *, http_factory=httpx.AsyncClient):
    protocol = load_protocol(args.protocol)
    root = Path(args.root)
    if args.cleanup:
        cleanup_diagnostics(root, args.protected_root, all_rejected=True)
        return 0
    if args.prepare:
        prepare(root, args.protected_root, protocol)
        return 0
    if args.report:
        local_report(root, args.protected_root, protocol)
        return 0
    # Допуск и неизменность подготовки проверяются до создания HTTP-клиентов.
    approval = ProtocolApproval.model_validate(_read(Path(args.approval))) if args.approval else None
    validated = validate_for_live(protocol, approval)
    workspace, _ = _restore(root, args.protected_root, protocol)
    if (workspace.root / 'run-started.json').exists() or (workspace.root / 'result.json').exists():
        raise ProtocolError('Автоматическое продолжение или повтор платного запуска запрещён')
    factory = LiveClientFactory(protocol, os.environ.get('COMPARISON_API_KEY'), http_factory)
    if not factory.api_key:
        raise ProtocolError('Не задан ключ COMPARISON_API_KEY')
    (workspace.root / 'run-started.json').write_text(json.dumps({'protocol_hash': protocol.content_hash}))
    started = time.monotonic()
    try:
        result = await run_comparison(validated, workspace, factory)
        save_result(result, protocol, workspace.root, time.monotonic() - started)
    except Exception:
        # Сохранённые квитанции доступны и после остановки; продолжение запрещено.
        (workspace.root / 'stop.json').write_text(json.dumps({'reason': 'run_stopped'}))
        local_report(workspace.root, args.protected_root, protocol)
        raise ProtocolError('Запуск остановлен; квитанции и частичный отчёт сохранены') from None
    local_report(workspace.root, args.protected_root, protocol)
    return 0


def parser():
    value = argparse.ArgumentParser(description='Изолированное сравнение учебных EPUB')
    modes = value.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'report', 'live', 'cleanup'):
        modes.add_argument('--' + name, action='store_true')
    value.add_argument('--protocol', required=True)
    value.add_argument('--root', required=True)
    value.add_argument('--protected-root', action='append', required=True)
    value.add_argument('--approval')
    return value


def main(argv=None):
    try:
        return asyncio.run(execute(parser().parse_args(argv)))
    except (ValueError, OSError, KeyError):
        print('Сравнение не запущено: условия допуска или локальные материалы недопустимы.')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
