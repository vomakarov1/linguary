"""Изолированные каталоги и закрытые согласованные снимки анализа."""
from __future__ import annotations

from contextlib import closing

import fcntl
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.comparison.protocol import ComparisonProtocol, ContinuationBinding, LUNA, SONNET, file_hash, settings_projection, stable_hash, validate_input_bindings
from linguary.comparison.llm import CallObservation
from linguary.domain import ChapterAnalysis, LLMCallLog, TokenUsage


class WorkspaceError(ValueError):
    """Каталог или состояние данных не подходят для независимого сравнения."""


@dataclass
class ComparisonWorkspace:
    root: Path
    protected_roots: tuple[Path, ...]
    protocol: ComparisonProtocol
    settings: AppSettings
    ownership: str
    analysis_hash: str | None = None
    _application: object | None = field(default=None, repr=False)
    _closed: bool = field(default=True, repr=False)
    _frozen: bool = field(default=False, repr=False)

    @property
    def database_path(self) -> Path:
        return self.root / 'data' / 'linguary.db'

    def build_application(self, **kwargs):
        """Создаёт приложение только из явных настроек и проверяет их проекцию."""
        from linguary.composition import build_application
        _verify_owned(self)
        if self._frozen or not self._closed:
            raise WorkspaceError('Каталог уже используется или заморожен')
        data_root = Path(self.settings.storage.data_root).expanduser().resolve()
        database = Path(self.settings.storage.database_path).expanduser().resolve() if self.settings.storage.database_path else data_root / 'linguary.db'
        if data_root != self.root / 'data' or database != self.database_path or self.settings.llm.debug_llm_payloads:
            raise WorkspaceError('Настройки нарушают изоляцию или включают сырые ответы')
        if 'settings' in kwargs:
            raise WorkspaceError('Настройки задаются каталогом сравнения')
        application = build_application(settings=self.settings, **kwargs)
        if settings_projection(application.settings) != settings_projection(self.settings):
            application.dispose()
            raise WorkspaceError('Эффективные настройки отличаются от зафиксированных')
        self._application = application
        self._closed = False
        return _ApplicationLease(self, application)


class _ApplicationLease:
    def __init__(self, workspace, application):
        self.workspace = workspace
        self.application = application

    def __getattr__(self, name):
        return getattr(self.application, name)

    def dispose(self):
        self.application.dispose()
        self.workspace._closed = True

    async def aclose(self):
        await self.application.aclose()
        self.workspace._closed = True


@dataclass(frozen=True)
class AnalysisArtifact:
    root: Path
    content_hash: str
    file_hashes: dict[str, str]
    protocol: ComparisonProtocol
    protected_roots: tuple[Path, ...]
    settings: AppSettings
    provenance: dict[str, object]
    source_root: Path | None = None
    sidecar_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuationState:
    artifact: AnalysisArtifact
    variant: dict[str, object]
    observations: list[CallObservation]
    selection: dict[str, set[str]]
    provenance: dict[str, object]
    stage_wall_seconds: dict[str, float]
    partial_seed: PartialAnalysisSeed | None = None


@dataclass(frozen=True)
class PartialAnalysisSeed:
    analysis: ChapterAnalysis
    log: LLMCallLog
    observation: CallObservation
    fragment: dict[str, object]
    provenance: dict[str, object]

    def public(self):
        return {'analysis': asdict(self.analysis), 'log': _seed_log_projection(self.log),
                'observation': self.observation.public(), 'fragment': self.fragment, 'provenance': self.provenance}


def _seed_log_projection(log):
    data = asdict(log)
    data['started_at'] = log.started_at.isoformat()
    data['cost'] = str(log.cost)
    return data


def _bound_child(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise WorkspaceError('Путь происхождения выходит за исходный корень')
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.resolve().is_relative_to(root):
        raise WorkspaceError('Ссылки и выход за исходный корень запрещены')
    return path


def _continuation_inventory(root):
    if any(part.is_symlink() for part in (root, *root.parents)) or any(path.is_symlink() for path in root.rglob('*')):
        raise WorkspaceError('Ссылки в происхождении запрещены')
    return {str(path.relative_to(root)): file_hash(path) for path in root.rglob('*') if path.is_file()}


def _continuation_protocol_matches(previous, current):
    allowed = {'code_version', 'version_hashes', 'budget', 'max_total_attempts', 'continuation', 'cost_bounds'}
    old, new = previous.model_dump(mode='json'), current.model_dump(mode='json')
    if {k: v for k, v in old.items() if k not in allowed} != {k: v for k, v in new.items() if k not in allowed}:
        raise WorkspaceError('Научные условия продолжения изменены')
    if previous.repeats != 1 or previous.continuation is not None:
        raise WorkspaceError('Допускается только продолжение первого исходного повтора')
    for group in ('instruction_version', 'schema_version', 'checks_version'):
        if previous.version_hashes.get(group) != current.version_hashes.get(group):
            raise WorkspaceError('Инструкции, схемы или проверки изменились')
    changed = {str(Path(__file__).parent / name) for name in ('runner.py', 'workspace.py', 'protocol.py', '__main__.py')}
    before, after = previous.version_hashes.get('code_version', {}), current.version_hashes.get('code_version', {})
    if set(before) != set(after) or any(before[path] != after[path] for path in before if str(Path(path).resolve()) not in changed):
        raise WorkspaceError('Изменение кода выходит за сценарий продолжения')
    if previous.cost_bounds is None or current.cost_bounds is None:
        raise WorkspaceError('Нет проверенных тарифных границ')
    exclude = {'endpoint_snapshots', 'evidence'}
    if previous.cost_bounds.model_dump(exclude=exclude) != current.cost_bounds.model_dump(exclude=exclude):
        raise WorkspaceError('Тарифные границы продолжения изменились')


def load_continuation(workspace, *, materialize=False):
    binding = workspace.protocol.continuation
    if binding is None:
        return None
    path = Path(binding.path)
    if any(p.is_symlink() for p in (path, *path.parents)) or file_hash(path) != binding.sha256:
        raise WorkspaceError('Привязка происхождения изменена')
    recipe = json.loads(path.read_text())
    if not isinstance(recipe, dict):
        raise WorkspaceError('Ожидался объект происхождения')
    if recipe.get('kind') == 'late_sonnet_resume_v1':
        return _load_late_sonnet_continuation(workspace, recipe, materialize=materialize)
    return _load_legacy_continuation(workspace, materialize=materialize)


def _load_legacy_continuation(workspace, *, materialize=False, verified_copy=None):
    binding = workspace.protocol.continuation
    if binding is None:
        return None
    try:
        path = Path(binding.path).expanduser()
        if any(part.is_symlink() for part in (path, *path.parents)) or file_hash(path) != binding.sha256:
            raise WorkspaceError('Привязка происхождения изменена')
        recipe = json.loads(path.read_text())
        if set(recipe) != {'source_root', 'source_protocol', 'file_hashes', 'analysis_path', 'variant_path'}:
            raise WorkspaceError('Неполное происхождение продолжения')
        source_root = Path(recipe['source_root']).expanduser()
        if not source_root.is_absolute():
            raise WorkspaceError('Исходный корень должен быть абсолютным')
        if any(part.is_symlink() for part in (source_root, *source_root.parents)):
            raise WorkspaceError('Ссылки в исходном корне запрещены')
        source_root = source_root.resolve()
        if not (source_root / '.comparison-owner').is_file() or any(source_root == p or source_root.is_relative_to(p) or p.is_relative_to(source_root) for p in workspace.protected_roots):
            raise WorkspaceError('Исторический корень не принадлежит сравнению')
        internal_copy = (verified_copy is not None and source_root == verified_copy
                         and source_root.parent.parent == workspace.root)
        if not internal_copy and (source_root == workspace.root or source_root.is_relative_to(workspace.root) or workspace.root.is_relative_to(source_root)):
            raise WorkspaceError('Новый корень пересекается с историческим')
        if _continuation_inventory(source_root) != recipe['file_hashes']:
            raise WorkspaceError('Исторические материалы изменились')
        previous = ComparisonProtocol.model_validate(recipe['source_protocol'])
        _continuation_protocol_matches(previous, workspace.protocol)
        manifest = json.loads(_bound_child(source_root, 'manifest.json').read_text())
        result = json.loads(_bound_child(source_root, 'result.json').read_text())
        prepared = json.loads(_bound_child(source_root, 'prepared.json').read_text())
        if (manifest['protocol_hash'] != previous.content_hash or result['protocol_hash'] != previous.content_hash
                or prepared != {'manifest_hash': stable_hash(manifest), 'protocol_hash': previous.content_hash}
                or result['simulated'] is not False or set(result['violations']) - {'unknown_final_status'}):
            raise WorkspaceError('Исторический результат не подходит для продолжения')
        if len(result['variants']) != 1 or len(result['analyses']) != 1:
            raise WorkspaceError('Ожидались один анализ и одна конечная ветвь')
        variant = result['variants'][0]
        expected = previous.variants[0]
        if (variant['variant_id'] != previous.schedule[0] or variant['variant_id'] != expected.id
                or expected.analysis_model != LUNA or expected.processing_model != LUNA or variant['repeat'] != 1
                or variant['completed'] is not False or not variant['failures'] or variant['epubs']):
            raise WorkspaceError('Первое сочетание не является конечной неуспешной ветвью')
        original = _bound_child(source_root, recipe['analysis_path'])
        branch_root = _bound_child(source_root, recipe['variant_path'])
        artifact_data = json.loads((original / 'artifact.json').read_text())
        digest = stable_hash({'files': artifact_data['files'], 'provenance': artifact_data['provenance']})
        sidecar_hashes = {name: recipe['file_hashes'][str((original / name).relative_to(source_root))]
                          for name in ('data/linguary.db-wal', 'data/linguary.db-shm')
                          if name not in artifact_data['files'] and (original / name).is_file()}
        _validate_artifact_files(original, artifact_data['files'], sidecar_hashes)
        if (digest != artifact_data['content_hash']
                or digest != result['analyses'][0]['content_hash'] or digest != variant['analysis_hash']
                or artifact_data['provenance']['protocol_hash'] != previous.content_hash):
            raise WorkspaceError('Замороженный анализ изменился')
        inputs = json.loads((original / 'inputs.json').read_text())
        if {key: item['sha256'] for key, item in inputs.items()} != previous.source_hashes:
            raise WorkspaceError('Источники анализа отличаются')
        for key, item in inputs.items():
            if file_hash(_bound_child(original, item['path'])) != item['sha256'] or file_hash(Path(next(c.source for c in previous.chapters if c.id == key))) != item['sha256']:
                raise WorkspaceError('Исходный EPUB изменился')
        logs, selection = {}, {}
        with tempfile.TemporaryDirectory(prefix='.verify-origin-', dir=workspace.root) as inspection:
            for index, database in enumerate((original / 'data' / 'linguary.db', branch_root / 'data' / 'linguary.db')):
                copied = Path(inspection) / str(index) / database.name
                copied.parent.mkdir()
                for suffix in ('', '-wal', '-shm', '-journal'):
                    source = database.with_name(database.name + suffix)
                    if source.is_file():
                        shutil.copyfile(source, copied.with_name(copied.name + suffix))
                with closing(sqlite3.connect(copied.as_uri() + '?mode=ro', uri=True)) as db:
                    if database.is_relative_to(original):
                        _validate_analysis(db)
                        for book_id, section_id in db.execute('SELECT book_id, section_id FROM chapter_analyses'):
                            selection.setdefault(book_id, set()).add(section_id)
                    for entry in db.execute('SELECT id, success, provider, model, cost FROM llm_call_logs'):
                        if entry[0] in logs and logs[entry[0]] != entry:
                            raise WorkspaceError('Журналы происхождения противоречат друг другу')
                        logs[entry[0]] = entry
        if _continuation_inventory(source_root) != recipe['file_hashes']:
            raise WorkspaceError('Происхождение изменилось во время проверки')
        observations, repairs, call_ids = [], [], set()
        receipt_documents = {}
        for receipt in sorted((source_root / 'receipts').glob('*.json')):
            data = json.loads(receipt.read_text())
            if '_raw' in data:
                raise WorkspaceError('Сырые ответы не входят в происхождение')
            receipt_documents[data['call_uid']] = dict(data)
            call_id = data.get('call_id')
            if call_id not in logs or call_id in call_ids or receipt.stem != data['call_uid']:
                raise WorkspaceError('Квитанция не связана с единственной записью журнала')
            call_ids.add(call_id)
            entry = logs[call_id]
            route = previous.route_metadata[data['assigned_model']]
            cost = Decimal(data['cost'])
            if (not cost.is_finite() or cost < 0 or Decimal(str(entry[4])) != cost or data['repeat'] != 1
                    or data['assigned_model'] != expected.analysis_model
                    or data['submitted'] is not True or data['received_response'] is not True
                    or data['actual_model'] not in route.model_versions or data['actual_provider'] not in route.provider_names
                    or entry[2] != previous.routes[data['assigned_model']] or entry[3] not in (data.get('reported_model'), data['actual_model'])
                    or data['assigned_route'] != previous.routes[data['assigned_model']]
                    or data['full_response_cache'] is not False or not isinstance(data['context_cache'], bool)
                    or data['cost_source'] != 'provider_receipt' or not isinstance(data.get('generation_id'), str) or not data['generation_id']
                    or data['model_identity_source'] not in ('chat', 'generation_metadata')):
                raise WorkspaceError('Маршрут, цена или кеш квитанции не подтверждены')
            if data['model_identity_source'] == 'chat' and data.get('reported_model') != data['actual_model']:
                raise WorkspaceError('Точная версия не наблюдалась в ответе')
            if data['model_identity_source'] == 'generation_metadata' and (
                    data.get('reported_model') != data['assigned_model'] or data['metadata_status_code'] != 200
                    or data['metadata_check_reason'] != 'verified' or data['metadata_get_count'] < 1):
                raise WorkspaceError('Точная версия не подтверждена метаданными')
            verdict = bool(entry[1])
            if data['accepted'] is None and verdict:
                repairs.append({'call_uid': data['call_uid'], 'call_id': call_id, 'reason': 'persisted_log_verdict', 'accepted': True})
            elif data['accepted'] is not verdict:
                raise WorkspaceError('Вердикт квитанции противоречит журналу')
            data['accepted'], data['cost'] = verdict, cost
            if data.get('usage') is not None:
                data['usage'] = TokenUsage(**data['usage'])
            observations.append(CallObservation(**data))
        for copy in source_root.rglob('receipts/*.json'):
            duplicate = json.loads(copy.read_text())
            if receipt_documents.get(duplicate['call_uid']) != duplicate:
                raise WorkspaceError('Копии квитанции противоречат друг другу')
        if call_ids != set(logs) or not observations or len({item.call_uid for item in observations}) != len(observations):
            raise WorkspaceError('Неполный набор уникальных квитанций')
        if (previous.budget is None or workspace.protocol.budget is None or previous.max_total_attempts is None
                or workspace.protocol.max_total_attempts is None
                or Decimal(str(workspace.protocol.budget)) > Decimal(str(previous.budget)) - sum(item.cost for item in observations)
                or workspace.protocol.max_total_attempts > previous.max_total_attempts - len(observations)):
            raise WorkspaceError('Продолжение превышает оставшийся бюджет или предел отправок')
        analysis_ids = set(artifact_data['provenance']['analysis_call_ids'])
        if {item.call_id for item in observations if item.stage == 'chapter_analysis'} != analysis_ids:
            raise WorkspaceError('Происхождение анализа не подтверждено')
        for item in observations:
            if item.stage == 'chapter_analysis':
                if item.variant_id != f'analysis-{expected.analysis_model}':
                    raise WorkspaceError('Квитанция относится к другому анализу')
            elif item.variant_id != expected.id or item.stage != 'translation_result':
                raise WorkspaceError('Квитанция относится к другому сочетанию')
        target = workspace.root / 'imported-analysis'
        if materialize:
            shutil.copytree(original, target)
            shutil.copyfile(path, workspace.root / 'continuation.json')
        _validate_artifact_files(target, artifact_data['files'], sidecar_hashes)
        if json.loads((target / 'artifact.json').read_text()) != artifact_data:
            raise WorkspaceError('Подготовленная копия анализа изменилась')
        settings = replace(workspace.settings, llm=replace(workspace.settings.llm, model=expected.analysis_model, provider=previous.routes[expected.analysis_model]))
        artifact = AnalysisArtifact(target, digest, artifact_data['files'], workspace.protocol, workspace.protected_roots,
                                    settings, artifact_data['provenance'], original, sidecar_hashes)
        provenance = {'binding_sha256': binding.sha256, 'origin_protocol_hash': previous.content_hash,
                      'origin_code_version': previous.code_version, 'analysis_hash': digest,
                      'imported_call_uids': [item.call_uid for item in observations], 'verdict_repairs': repairs,
                      'origin_violations': result['violations']}
        return ContinuationState(artifact, variant, observations, selection, provenance, result.get('stage_wall_seconds', {}))
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError, sqlite3.Error):
        raise WorkspaceError('Не удалось подтвердить неизменное происхождение продолжения') from None


def _late_code_transition(recipe, previous, current, proof):
    directory = Path(__file__).resolve().parent
    mutable = {str(directory / name) for name in ('workspace.py', 'runner.py', '__main__.py')}
    late = str(directory / 'late_metadata.py')
    provenance_paths = {str(directory / name) for name in ('llm.py', 'runner.py')}
    provenance_paths.add(str(directory.parent / 'adapters/llm/logging.py'))
    transition = recipe['code_transition']
    before, after = transition['before'], transition['after']
    if (set(transition) != {'before', 'after'} or before != previous.version_hashes['code_version']
            or after != current.version_hashes['code_version'] or set(after) != set(before) | {late}
            or after[late] != '2e3cae8dc439acf9dbcd320550eb087a73e5f63c836bb940d397d0540227ce6f'
            or set(proof['provenance_source_hashes']) != provenance_paths
            or proof['implementation_file_hashes'][str(Path('src/linguary/comparison/workspace.py'))] != '6d534d3975374d96cd1e6bffc2833427c7ed1dd97f70989410d13cd54942e490'
            or any(after[path] != before[path] for path in before if path not in mutable)):
        raise WorkspaceError('Нет точного разрешённого перехода кода P14 → P15')
    for path, digest in after.items():
        candidate = Path(path)
        if (not candidate.is_absolute() or any(p.is_symlink() for p in (candidate, *candidate.parents))
                or file_hash(candidate) != digest):
            raise WorkspaceError('Исполняемый код отличается от перехода')
    for path, digest in proof['validation_source_hashes'].items():
        if after.get(path, digest) != digest or file_hash(Path(path)) != digest:
            raise WorkspaceError('Проверки типизированного анализа изменились')
    for path, digest in proof['provenance_source_hashes'].items():
        if (path in before and before[path] != digest) or (path != str(directory / 'runner.py') and file_hash(Path(path)) != digest):
            raise WorkspaceError('Исторический код объяснения журнала изменился')
    compatible = current.model_copy(update={'version_hashes': dict(current.version_hashes, code_version={path: after[path] for path in before})})
    _continuation_protocol_matches(previous.model_copy(update={'continuation': None}), compatible)
    return compatible


def _late_document(binding):
    if set(binding) != {'path', 'sha256', 'content_hash'}:
        raise WorkspaceError('Неполная привязка доказательства')
    path = Path(binding['path'])
    if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
            or file_hash(path) != binding['sha256']):
        raise WorkspaceError('Доказательство изменилось')
    data = json.loads(path.read_text())
    if stable_hash(data) != binding['content_hash']:
        raise WorkspaceError('Содержимое доказательства изменилось')
    return data


def _load_late_sonnet_continuation(workspace, recipe, *, materialize):
    """Конкретный импорт wait Luna/LL + одного проверенного анализа Sonnet."""
    from linguary.comparison.late_metadata import verify_late_metadata
    from linguary.domain.errors import DomainError
    from sqlalchemy.exc import SQLAlchemyError
    try:
        _verify_owned(workspace)
        if (set(recipe) != {'kind', 'sources', 'supplement', 'proof', 'seed', 'code_transition'}
                or set(recipe['sources']) != {'luna', 'sonnet'} or workspace.protocol.repeats != 1):
            raise WorkspaceError('Нужны ровно два прямых источника и один исходный повтор')
        initial, sonnet = recipe['sources']['luna'], recipe['sources']['sonnet']
        if (set(initial) != {'source_root', 'source_protocol', 'file_hashes', 'analysis_path', 'variant_path'}
                or set(sonnet) != {'source_root', 'source_protocol', 'file_hashes'}):
            raise WorkspaceError('Произвольная цепочка источников запрещена')
        previous = ComparisonProtocol.model_validate(sonnet['source_protocol'])
        original = ComparisonProtocol.model_validate(initial['source_protocol'])
        if original.continuation is not None or previous.continuation is None:
            raise WorkspaceError('Исходники не являются первоначальным wait и его v3')
        supplement, proof = _late_document(recipe['supplement']), _late_document(recipe['proof'])
        compatible = _late_code_transition(recipe, previous, workspace.protocol, proof)
        if (proof['verdict'] != 'reusable' or proof['reasons'] or proof['live_permission'] is not False
                or proof['complete_artifact'] is not False or len(proof['confirmed_fragments']) != 1
                or recipe['seed'] != proof['confirmed_fragments'][0]
                or proof['supplement_sha256'] != stable_hash(supplement)):
            raise WorkspaceError('Нет единственного проверенного типизированного начала анализа')
        sources = {}
        for name, spec in recipe['sources'].items():
            root = Path(spec['source_root'])
            if (not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents))
                    or not (root / '.comparison-owner').is_file() or root != root.resolve()
                    or any(root.is_relative_to(p) or p.is_relative_to(root) for p in (workspace.root, *workspace.protected_roots))
                    or _continuation_inventory(root) != spec['file_hashes']):
                raise WorkspaceError('Исходный inventory или изоляция изменились')
            sources[name] = root
        if sources['luna'].is_relative_to(sources['sonnet']) or sources['sonnet'].is_relative_to(sources['luna']):
            raise WorkspaceError('Источники пересекаются')
        if (proof['source_root'] != str(sources['sonnet']) or supplement['source_root'] != str(sources['sonnet'])
                or proof['source_inventory_before'] != sonnet['file_hashes'] or proof['source_inventory_after'] != sonnet['file_hashes']
                or proof['copy_inventory_before_read'] != sonnet['file_hashes']):
            raise WorkspaceError('P14 относится к другому исходному состоянию')
        checked = verify_late_metadata(evidence_path=supplement['diagnosis']['path'], evidence_sha256=supplement['diagnosis']['sha256'],
            source_root=sources['sonnet'], source_inventory=sonnet['file_hashes'], protocol=previous)
        if checked.content_hash != stable_hash(supplement):
            raise WorkspaceError('Поздние метаданные не воспроизводятся')
        origin = json.loads(_bound_child(sources['sonnet'], 'continuation.json').read_text())
        if (origin != initial or file_hash(sources['sonnet'] / 'continuation.json') != previous.continuation.sha256):
            raise WorkspaceError('V3 не связан с единственным прямым источником Luna/LL')
        with tempfile.TemporaryDirectory(prefix='.verify-fixed-', dir=workspace.root) as temporary:
            copies = {}
            for name, root in sources.items():
                copies[name] = Path(temporary) / name
                shutil.copytree(root, copies[name])
                if _continuation_inventory(copies[name]) != recipe['sources'][name]['file_hashes']:
                    raise WorkspaceError('Полная копия источника изменилась')
            computed = {'confirmed_fragments': [], 'missing_fragments': [], 'reasons': []}
            analysis, log = _inspect_partial_data(copies['sonnet'], supplement, previous, computed)
            comparable = ('confirmed_fragments', 'missing_fragments', 'logs_sha256', 'logs', 'log_explanations',
                          'validation_source_hashes', 'accounting', 'historical_variants')
            if computed['reasons'] or any(computed[key] != proof[key] for key in comparable):
                raise WorkspaceError('Типизированное содержимое, контекст или объяснение P14 не воспроизводятся')
            remaining = proof['accounting']
            if (Decimal(str(workspace.protocol.budget)) > Decimal(remaining['remaining_cost'])
                    or workspace.protocol.max_total_attempts > remaining['remaining_posts']):
                raise WorkspaceError('Превышен остаток общего бюджета')
            legacy = dict(initial, source_root=str(copies['luna']))
            legacy_path = Path(temporary) / 'legacy.json'
            legacy_path.write_text(json.dumps(legacy))
            legacy_protocol = compatible.model_copy(update={'continuation': ContinuationBinding(path=str(legacy_path), sha256=file_hash(legacy_path))})
            initial_state = _load_legacy_continuation(replace(workspace, protocol=legacy_protocol), materialize=materialize,
                                                       verified_copy=copies['luna'])
            initial_state = replace(initial_state, provenance=dict(initial_state.provenance,
                binding_sha256=previous.continuation.sha256))
            effective = dict(supplement['receipt'], actual_model=supplement['verified']['model'],
                model_identity_source='generation_metadata', metadata_status_code=200, metadata_check_reason='verified',
                metadata_get_count=supplement['receipt']['metadata_get_count'] + 1)
            effective['cost'], effective['usage'] = Decimal(effective['cost']), TokenUsage(**effective['usage'])
            observation = CallObservation(**effective)
            observations = [*initial_state.observations, observation]
            imported_receipts = {}
            for receipt_path in (copies['sonnet'] / 'receipts').glob('*.json'):
                document = json.loads(receipt_path.read_text())
                imported_receipts[document['call_uid']] = document
            if (set(imported_receipts) != {item.call_uid for item in observations}
                    or any(imported_receipts[item.call_uid] != item.public() for item in initial_state.observations)
                    or computed['historical_variants'] != [initial_state.variant]):
                raise WorkspaceError('Импортированные Luna/LL в v3 расходятся с прямым источником')
            for field in ('call_uid', 'call_id', 'generation_id'):
                if len({getattr(item, field) for item in observations}) != len(observations):
                    raise WorkspaceError('Импортированные наблюдения повторены')
            provenance = {'kind': recipe['kind'], 'binding_sha256': workspace.protocol.continuation.sha256,
                'source_inventories': {name: spec['file_hashes'] for name, spec in recipe['sources'].items()},
                'supplement': recipe['supplement'], 'proof': recipe['proof'], 'code_transition': recipe['code_transition'],
                'historical_violations': {'luna': initial_state.provenance['origin_violations'], 'sonnet': proof['historical']['violations']},
                'excluded_local_logs': proof['log_explanations'], 'accounting': remaining,
                'original_receipt_sha256': supplement['source_hashes'], 'luna': initial_state.provenance}
            seed = PartialAnalysisSeed(analysis, log, observation, recipe['seed'], provenance)
            seed_path = workspace.root / 'partial-seed.json'
            if materialize:
                seed_path.write_text(json.dumps(seed.public(), ensure_ascii=False, sort_keys=True))
                shutil.copyfile(Path(workspace.protocol.continuation.path), workspace.root / 'continuation.json')
                for item in observations:
                    directory = workspace.root / 'receipts'
                    directory.mkdir(exist_ok=True)
                    (directory / f'{item.call_uid}.json').write_text(json.dumps(item.public(), ensure_ascii=False))
            if json.loads(seed_path.read_text()) != json.loads(json.dumps(seed.public())):
                raise WorkspaceError('Подготовленное типизированное начало изменилось')
            for item in observations:
                if json.loads((workspace.root / 'receipts' / f'{item.call_uid}.json').read_text()) != item.public():
                    raise WorkspaceError('Эффективная квитанция изменилась')
            expected_receipts = {item.call_uid: item.public() for item in observations}
            for receipt_path in workspace.root.rglob('receipts/*.json'):
                if receipt_path.is_relative_to(Path(temporary)):
                    continue
                document = json.loads(receipt_path.read_text())
                if expected_receipts.get(document['call_uid']) != document:
                    raise WorkspaceError('В подготовке появилась посторонняя или противоречивая квитанция')
            artifact = replace(initial_state.artifact, protocol=workspace.protocol,
                source_root=sources['luna'] / initial['analysis_path'])
            return ContinuationState(artifact, initial_state.variant, observations, initial_state.selection, provenance,
                                     json.loads((sources['sonnet'] / 'result.json').read_text())['stage_wall_seconds'], seed)
    except WorkspaceError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError, sqlite3.Error, SQLAlchemyError, DomainError):
        raise WorkspaceError('Не удалось подтвердить фиксированное продолжение Sonnet') from None
    finally:
        if 'sources' in locals():
            for name, root in sources.items():
                if _continuation_inventory(root) != recipe['sources'][name]['file_hashes']:
                    raise WorkspaceError('Источник изменился во время проверки фиксированного продолжения')


def _check_root(root: Path, protected_roots) -> tuple[Path, tuple[Path, ...]]:
    resolved = root.expanduser().resolve()
    protected = tuple(Path(p).expanduser().resolve() for p in protected_roots)
    if not protected:
        raise WorkspaceError('Защищённые рабочие корни должны быть указаны явно')
    if any(resolved == p or resolved.is_relative_to(p) or p.is_relative_to(resolved) for p in protected):
        raise WorkspaceError('Корень сравнения пересекается с рабочими данными')
    if root.is_symlink() or (resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir()))):
        raise WorkspaceError('Назначение должно быть новым или пустым каталогом')
    return resolved, protected


@dataclass(frozen=True)
class PartialAnalysisProof:
    document: dict[str, object]

    @property
    def content_hash(self):
        return stable_hash(self.document)


def inspect_partial_analysis(*, source_root, source_inventory, supplement, destination_root):
    """Исследует полную свежую копию; не открывает исходную SQLite и не разрешает live."""
    from linguary.comparison.late_metadata import verify_late_metadata
    from linguary.domain.errors import DomainError
    from sqlalchemy.exc import SQLAlchemyError
    source, destination = Path(source_root), Path(destination_root)
    if (not source.is_absolute() or not destination.is_absolute()
            or any(p.is_symlink() for p in (source, *source.parents, destination, *destination.parents))):
        raise WorkspaceError('Проверке нужен новый непересекающийся каталог без ссылок')
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise WorkspaceError('Проверке нужен новый непересекающийся каталог без ссылок')
    document = supplement.document
    protocol = ComparisonProtocol.model_validate(document['protocol'])
    verified = verify_late_metadata(evidence_path=document['diagnosis']['path'], evidence_sha256=document['diagnosis']['sha256'],
                                    source_root=source, source_inventory=source_inventory, protocol=protocol)
    if verified.content_hash != supplement.content_hash or document['source_root'] != str(source):
        raise WorkspaceError('Производное подтверждение изменилось или относится к другому корню')
    proof = {'version': 1, 'source_root': str(source), 'destination_root': str(destination),
             'supplement_sha256': supplement.content_hash, 'source_inventory_before': dict(source_inventory),
             'source_hashes': document['source_hashes'], 'confirmed_fragments': [], 'missing_fragments': [],
             'reasons': [], 'historical': document['historical'], 'complete_artifact': False, 'live_permission': False,
             'accounting_delta': {'posts': 0, 'cost': '0', 'first_responses': 0}}
    try:
        shutil.copytree(source, destination)
        copied = _continuation_inventory(destination)
        if copied != source_inventory or _continuation_inventory(source) != source_inventory:
            raise WorkspaceError('Полная копия не совпадает с неизменным источником')
        proof['copy_inventory_before_read'] = copied
        token = uuid4().hex
        (destination / '.partial-proof-owner').write_text(token)
        proof['ownership'] = token
        try:
            _inspect_partial_copy(destination, document, protocol, proof)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError, StopIteration, sqlite3.Error, SQLAlchemyError, DomainError) as error:
            if isinstance(error, DomainError):
                proof['reasons'].append('domain_validation_failed:' + type(error).__name__)
            else:
                proof['reasons'].append(str(error) if isinstance(error, WorkspaceError) else 'invalid_saved_evidence:' + type(error).__name__)
        proof['copy_inventory_after_read'] = _continuation_inventory(destination)
    finally:
        if _continuation_inventory(source) != source_inventory:
            raise WorkspaceError('Источник изменился во время проверки частичного анализа')
    proof['source_inventory_after'] = _continuation_inventory(source)
    if not proof['confirmed_fragments']:
        proof['reasons'].append('no_confirmed_saved_fragment')
    proof['verdict'] = 'insufficient_evidence' if proof['reasons'] or not proof['confirmed_fragments'] else 'reusable'
    return PartialAnalysisProof(proof)


def _inspect_partial_copy(root, supplement, protocol, proof):
    from linguary.adapters.llm import logging as logging_module
    from linguary.comparison import llm as comparison_llm, runner
    provenance_sources = [Path(module.__file__).resolve() for module in (logging_module, comparison_llm, runner)]
    proof['provenance_source_hashes'] = {str(path): file_hash(path) for path in provenance_sources}
    historical_code = protocol.version_hashes.get('code_version', {})
    if any(str(path) in historical_code and historical_code[str(path)] != proof['provenance_source_hashes'][str(path)] for path in provenance_sources):
        raise WorkspaceError('Код объяснения происхождения изменился относительно серии')
    return _inspect_partial_data(root, supplement, protocol, proof)


def _inspect_partial_data(root, supplement, protocol, proof):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from linguary.adapters.db.repositories import SqlAlchemyChapterAnalysisRepository, SqlAlchemyDocumentRepository
    from linguary.adapters.epub import EbookLibEpubReader
    from linguary.application.context_builder import textual_blocks
    from linguary.application.ports.epub import EpubReadRequest
    from linguary.comparison.late_metadata import _decimal, _read_json
    from linguary.comparison.llm import ObservationLedger
    from linguary.comparison.runner import _FixedContext
    from linguary.domain import CoverageStatus

    receipt = supplement['receipt']
    nested = [name for name in supplement['receipt_paths'] if not name.startswith('receipts/')]
    if len(nested) != 1:
        raise WorkspaceError('Нет единственного частичного корня')
    partial = _bound_child(root, str(Path(nested[0]).parents[1]))
    if not (partial / '.comparison-owner').is_file():
        raise WorkspaceError('Частичный корень не принадлежит сравнению')
    manifest = _read_json(partial / 'manifest.json')
    result, report = _read_json(root / 'result.json'), _read_json(root / 'report.json')
    explained_violations = {'unknown_or_changed_route', 'unlinked_log', f"analysis_failed:{receipt['repeat']}:{receipt['assigned_model']}"}
    if set(result['violations']) - explained_violations:
        raise WorkspaceError('Есть необъяснённое историческое нарушение')
    if manifest['protocol_hash'] != protocol.content_hash or manifest not in result['manifests'].values():
        raise WorkspaceError('Манифест частичного анализа не связан с результатом')
    version_hashes = {}
    for group in ('instruction_version', 'schema_version', 'checks_version'):
        hashes = protocol.version_hashes[group]
        if not hashes or manifest[group] != getattr(protocol, group):
            raise WorkspaceError('Версии инструкций, схем или проверок не согласованы')
        for path, digest in hashes.items():
            path = Path(path)
            if (not path.is_relative_to(Path(__file__).resolve().parents[1]) or path.suffix != '.py'
                    or any(p.is_symlink() for p in (path, *path.parents)) or file_hash(path) != digest):
                raise WorkspaceError('Код чтения или проверки анализа изменился')
            version_hashes[str(path)] = digest
    proof['validation_source_hashes'] = version_hashes
    inputs = _read_json(partial / 'inputs.json')
    context = _read_json(partial / 'context.json')
    if (context != {'glossary': protocol.fixed_glossary, 'previous_context': protocol.previous_context}
            or {key: item['sha256'] for key, item in inputs.items()} != protocol.source_hashes):
        raise WorkspaceError('Исходники или фиксированный контекст изменились')
    for item in inputs.values():
        if file_hash(_bound_child(partial, item['path'])) != item['sha256']:
            raise WorkspaceError('Копия исходного EPUB изменилась')
    database = _bound_child(partial, 'data/linguary.db')
    if not database.is_file():
        raise WorkspaceError('Нет сохранённой SQLite')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        logs = [dict(row) for row in db.execute('SELECT * FROM llm_call_logs ORDER BY started_at, id')]
        tasks = {row['id']: dict(row) for row in db.execute('SELECT * FROM processing_tasks')}
        if any(row['response_text'] is not None for row in logs):
            raise WorkspaceError('Сырые ответы не входят в доказательство')
    paid = [entry for entry in logs if entry['id'] == receipt['call_id']]
    if len(paid) != 1:
        raise WorkspaceError('Квитанция не связана с единственным журналом')
    paid = paid[0]
    task = tasks[paid['task_id']]
    if (paid['success'] != 1 or paid['stage'] != receipt['stage'] or paid['provider'] != receipt['assigned_route']
            or paid['model'] != receipt['reported_model'] or _decimal(paid['cost']) != _decimal(receipt['cost'])
            or paid['prompt_tokens'] != receipt['usage']['prompt_tokens'] or paid['completion_tokens'] != receipt['usage']['completion_tokens']
            or paid['book_id'] != task['book_id'] or task['scope_id'] != receipt['chapter_id']
            or task['task_type'] != 'analyze_chapter' or task['status'] != 'succeeded'
            or paid['attempt'] != 1 or task['attempts'] != 1):
        raise WorkspaceError('Успешный журнал, задача и квитанция противоречат друг другу')
    selected = manifest['grouping']['selected_sections']
    if receipt['chapter_id'] not in selected[paid['book_id']]:
        raise WorkspaceError('Оплаченная глава отсутствует в выбранном корпусе')
    extras = [entry for entry in logs if entry['id'] != receipt['call_id']]
    explained = []
    if len(extras) > 1:
        proof['reasons'].append('multiple_unlinked_logs_after_observer_failure')
    for entry in extras:
        blocked_task = tasks[entry['task_id']]
        if (len(extras) == 1 and entry['task_id'] != paid['task_id']
                and entry['success'] == 0 and entry['duration_ms'] == 0 and entry['attempt'] == 1
                and all(entry[key] is None for key in ('prompt_tokens', 'completion_tokens', 'cost', 'response_text'))
                and entry['error_type'] == 'ConfigurationError' and entry['error_detail'] == 'linguary.comparison.llm.ComparisonStopped'
                and entry['error_message'] == 'Нарушены условия сравнения; новые вызовы остановлены'
                and entry['provider'] == receipt['assigned_route'] and entry['model'] == receipt['reported_model']
                and entry['stage'] == 'chapter_analysis' and entry['started_at'] >= paid['started_at']
                and blocked_task['book_id'] == entry['book_id'] and blocked_task['status'] == 'failed'
                and blocked_task['attempts'] == 1 and blocked_task['error_type'] == 'ConfigurationError'
                and blocked_task['error_message'] == 'Запись наблюдения недоступна; новые вызовы остановлены'
                and blocked_task['error_detail'] == 'linguary.application.ports.llm.LLMConfigurationError'
                and blocked_task['task_type'] == 'analyze_chapter' and blocked_task['scope_id'] in selected[entry['book_id']]):
            explained.append({'call_id': entry['id'], 'reason': 'ensure_open_before_begin_then_logged_observer_without_observation',
                              'code': ['comparison/llm.py:ObservedLLMClient.generate', 'adapters/llm/logging.py:LoggingLLMClient.generate',
                                       'comparison/llm.py:ObservationLedger.on_logged'], 'new_post': False})
        else:
            proof['reasons'].append('unexplained_log:' + entry['id'])
    if 'unlinked_log' in result['violations'] and not explained:
        proof['reasons'].append('unexplained_unlinked_log_or_repeated_notification')
    proof['logs_sha256'] = stable_hash(logs)
    proof['logs'] = logs
    proof['log_explanations'] = explained
    for analysis_task in (item for item in tasks.values() if item['task_type'] == 'analyze_chapter'):
        if analysis_task['id'] == paid['task_id']:
            continue
        if (analysis_task['status'] != 'failed' or analysis_task['attempts'] != 1
                or analysis_task['error_type'] != 'ConfigurationError'
                or analysis_task['error_message'] != 'Запись наблюдения недоступна; новые вызовы остановлены'
                or analysis_task['error_detail'] != 'linguary.application.ports.llm.LLMConfigurationError'):
            proof['reasons'].append('unexplained_analysis_task:' + analysis_task['id'])
    saved_analysis = saved_log = None
    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(database.as_uri() + '?mode=ro', uri=True))
    try:
        with Session(engine) as session:
            documents = SqlAlchemyDocumentRepository(session)
            analyses = SqlAlchemyChapterAnalysisRepository(session)
            for book_id, section_ids in selected.items():
                document = documents.get(book_id)
                for section_id in section_ids:
                    analysis = analyses.get_for_section(book_id, section_id)
                    if analysis is None:
                        proof['missing_fragments'].append({'book_id': book_id, 'section_id': section_id})
                        continue
                    if book_id != paid['book_id'] or section_id != receipt['chapter_id']:
                        raise WorkspaceError('Сохранённый анализ не связан с оплаченной операцией')
                    section = next(section for section in document.sections if section.id == section_id)
                    blocks = textual_blocks(section)
                    expected = tuple(block.id for block in blocks)
                    analysis.ensure_ready_for_processing()
                    if (analysis.treatment_plan.coverage_for(expected) != CoverageStatus.COMPLETE
                            or analysis.treatment_plan.block_ids != expected
                            or not analysis.translation_instructions.strip() or not analysis.chapter_summary.strip()):
                        raise WorkspaceError('Типизированный анализ неполон или нарушает порядок блоков')
                    section_order = next(index for index, item in enumerate(document.ordered_sections, 1) if item.id == section_id)
                    chapter = next(chapter for chapter in protocol.chapters if chapter.id in inputs
                                   and chapter.chapter_id in (section.id, section.source_resource_path, str(section_order))
                                   and book_id == 'book-' + inputs[chapter.id]['sha256'][:16])
                    original = EbookLibEpubReader().read(EpubReadRequest(book_id=book_id, source_path=str(_bound_child(partial, inputs[chapter.id]['path'])),
                                                       source_language=document.source_language, target_language=document.target_language)).document
                    original_section = next(item for item in original.sections if item.id == section_id)
                    if [(b.id, b.text) for b in textual_blocks(original_section)] != [(b.id, b.text) for b in blocks]:
                        raise WorkspaceError('Блоки анализа не совпадают с исходным EPUB')
                    group = hashlib.sha256(json.dumps([book_id, receipt['stage'], section_id, list(expected)], separators=(',', ':')).encode()).hexdigest()
                    operation = hashlib.sha256('\0'.join((receipt['variant_id'], str(receipt['repeat']), book_id, receipt['stage'], group)).encode()).hexdigest()
                    if group != receipt['group_id'] or operation != receipt['operation_id']:
                        raise WorkspaceError('Группа и оплаченная операция не совпадают с покрытыми блоками')
                    ledger = ObservationLedger()
                    context = _FixedContext({section_id: protocol.previous_context.get(chapter.id, '')}, ledger).for_chapter_analysis(
                        document=document, section=section, accumulated_candidates=())
                    proof['confirmed_fragments'].append({'analysis_id': analysis.id, 'book_id': book_id, 'section_id': section_id,
                        'call_uid': receipt['call_uid'], 'call_id': receipt['call_id'], 'operation_id': operation, 'generation_id': receipt['generation_id'],
                        'block_ids': list(expected), 'coverage': 'complete', 'typed_analysis_sha256': stable_hash(asdict(analysis)),
                        'request_input_sha256': hashlib.sha256(context.request.input_text.encode()).hexdigest(),
                        'request_instructions_sha256': hashlib.sha256(context.request.instructions.encode()).hexdigest(),
                        'link_basis': 'unique_successful_call_and_task_book_scope_not_direct_analysis_call_id', 'raw_response_required': False})
                    from linguary.adapters.db.models import LLMCallLogRow
                    from linguary.adapters.db.mapping import to_llm_call
                    saved_analysis, saved_log = analysis, to_llm_call(session.get(LLMCallLogRow, paid['id']))
    finally:
        engine.dispose()
    _partial_accounting(root, report, result, protocol, receipt, proof)
    return saved_analysis, saved_log


def _partial_accounting(root, report, result, protocol, receipt, proof):
    from linguary.comparison.late_metadata import _decimal, _integer, _read_json
    receipts = [_read_json(path) for path in sorted((root / 'receipts').glob('*.json'))]
    by_uid = {item['call_uid']: item for item in receipts}
    if len(by_uid) != len(receipts) or by_uid.get(receipt['call_uid']) != receipt:
        raise WorkspaceError('Уникальность оплаченных вызовов не подтверждена')
    for path in root.rglob('receipts/*.json'):
        copy = _read_json(path)
        if by_uid.get(copy['call_uid']) != copy:
            raise WorkspaceError('Копии импортированных квитанций расходятся')
    if {item['call_uid']: item for item in report['observations']} != by_uid or len(report['observations']) != len(receipts):
        raise WorkspaceError('Отчёт не совпадает с уникальными квитанциями')
    generations, first, call_ids = set(), set(), set()
    for item in receipts:
        if (item['submitted'] is not True or not isinstance(item['generation_id'], str) or not item['generation_id']
                or type(item['first_response']) is not bool or item['generation_id'] in generations
                or not isinstance(item['call_id'], str) or not item['call_id'] or item['call_id'] in call_ids
                or not isinstance(item['operation_id'], str) or not item['operation_id']):
            raise WorkspaceError('Генерации расхода повторены или неизвестны')
        generations.add(item['generation_id'])
        call_ids.add(item['call_id'])
        if item['first_response']:
            if item['operation_id'] in first:
                raise WorkspaceError('Первый ответ операции учтён повторно')
            first.add(item['operation_id'])
    imported = result['continuation_provenance']['imported_call_uids']
    if len(imported) != len(set(imported)) or set(imported) != set(by_uid) - {receipt['call_uid']}:
        raise WorkspaceError('Импортированные Luna/LL учтены неоднозначно')
    subtotal = sum((_decimal(item['cost']) for item in receipts), Decimal(0))
    if _decimal(report['total']['cost']) != subtotal or _integer(report['total']['unique_calls']) != len(receipts):
        raise WorkspaceError('Расход отчёта не совпадает с квитанциями')
    accounting = json.loads(protocol.projection_method)
    posts = _integer(accounting['prior_attempts_total']) + len(receipts)
    cost = _decimal(accounting['known_prior_cost']) + subtotal
    cap, attempts_cap = _decimal(accounting['global_budget']), _integer(accounting['global_attempt_limit'])
    if cost > cap or posts > attempts_cap:
        raise WorkspaceError('Исторический расход превышает общий предел')
    proof['accounting'] = {'unique_call_uids': sorted(by_uid), 'operation_ids': sorted({item['operation_id'] for item in receipts}),
        'generation_ids': sorted(generations), 'imported_call_uids': imported, 'posts': posts, 'known_cost': str(cost),
        'remaining_cost': str(cap - cost), 'remaining_posts': attempts_cap - posts,
        'prior_technical_posts': accounting['prior_attempts_total'], 'prior_technical_known_cost': accounting['known_prior_cost']}
    proof['historical_variants'] = result['variants']


def _verify_owned(workspace):
    if workspace.root.resolve() != workspace.root:
        raise WorkspaceError('Корень каталога изменился')
    marker = workspace.root / '.comparison-owner'
    if not marker.is_file() or marker.is_symlink() or marker.read_text() != workspace.ownership:
        raise WorkspaceError('Каталог не принадлежит этому сравнению')
    for path in workspace.root.rglob('*'):
        if path.is_symlink():
            raise WorkspaceError('Ссылки внутри каталога сравнения запрещены')
    if any(workspace.root.is_relative_to(p) or p.is_relative_to(workspace.root) for p in (p.resolve() for p in workspace.protected_roots)):
        raise WorkspaceError('Защищённый корень изменился')


def _cleanup(root, token):
    marker = root / '.comparison-owner'
    if marker.is_file() and not marker.is_symlink() and marker.read_text() == token:
        shutil.rmtree(root)


def prepare_workspace(root, protected_roots, protocol) -> ComparisonWorkspace:
    validate_input_bindings(protocol)
    root, protected = _check_root(Path(root), protected_roots)
    token = uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    (root / '.comparison-owner').write_text(token)
    root.chmod(0o700)
    try:
        sources = root / 'inputs'
        sources.mkdir()
        hashes = {}
        for chapter in protocol.chapters:
            if not chapter.source:  # Проект допускает неизвестные источники.
                continue
            source = Path(chapter.source).expanduser().resolve(strict=True)
            if not source.is_file() or root.is_relative_to(source) or source.is_relative_to(root):
                raise WorkspaceError('Недопустимый источник EPUB')
            target = sources / f'{len(hashes):04d}.epub'
            before = file_hash(source)
            shutil.copy2(source, target)
            if file_hash(source) != before or file_hash(target) != before:
                raise WorkspaceError('Источник изменился при копировании')
            hashes[chapter.id] = {'path': str(target.relative_to(root)), 'sha256': before}
        generation = {k: v for k, v in protocol.generation.items() if v is not None}
        splitting = {k: v for k, v in protocol.splitting.items() if v is not None}
        llm = LLMSettings(**generation, **splitting, proxy_mode=protocol.proxy_mode, requests_per_minute=protocol.requests_per_minute, debug_llm_payloads=False)
        settings = AppSettings(llm=llm, checks=CheckSettings(**{k: v for k, v in protocol.checks.items() if v is not None}), storage=StorageSettings(data_root=str(root / 'data')), worker=WorkerPoolSettings(concurrency=protocol.concurrency or 1))
        workspace = ComparisonWorkspace(root, protected, protocol, settings, token)
        (root / 'inputs.json').write_text(json.dumps(hashes, sort_keys=True))
        (root / 'context.json').write_text(json.dumps({'glossary': protocol.fixed_glossary, 'previous_context': protocol.previous_context}, sort_keys=True, ensure_ascii=False))
        (root / 'settings.json').write_text(json.dumps(settings_projection(settings), sort_keys=True))
        return workspace
    except Exception:
        _cleanup(root, token)
        raise WorkspaceError('Не удалось подготовить изолированный каталог') from None


def _validate_analysis(db):
    if not db.execute('SELECT 1 FROM chapter_analyses LIMIT 1').fetchone():
        raise WorkspaceError('Нет завершённого анализа для заморозки')
    for table in ('processing_runs', 'block_results', 'language_notes'):
        if db.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone():
            raise WorkspaceError('Нельзя копировать прежние результаты обработки')
    if db.execute("SELECT 1 FROM processing_tasks WHERE status != 'succeeded' OR task_type NOT IN ('import_book', 'analyze_chapter') LIMIT 1").fetchone():
        raise WorkspaceError('Есть незавершённые задачи или прежняя очередь обработки')
    if db.execute("SELECT 1 FROM chapter_analyses WHERE coverage_status != 'complete' LIMIT 1").fetchone():
        raise WorkspaceError('Анализ не завершён')
    if db.execute('SELECT 1 FROM system_settings LIMIT 1').fetchone():
        raise WorkspaceError('Сохранённые настройки запрещены в снимке')
    if db.execute('SELECT 1 FROM llm_call_logs WHERE response_text IS NOT NULL LIMIT 1').fetchone():
        raise WorkspaceError('Сырые ответы не входят в снимок анализа')


def _hashes(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifact.json'}


def _validate_artifact_files(root, expected, sidecar_hashes=None):
    if any(path.is_symlink() for path in (root, *root.parents)) or any(path.is_symlink() for path in root.rglob('*')):
        raise WorkspaceError('Ссылки в замороженном анализе запрещены')
    actual = {str(path.relative_to(root)): file_hash(path) for path in root.rglob('*')
              if path.is_file() and path != root / 'artifact.json'}
    bound = sidecar_hashes or {}
    extras = set(actual) - set(expected)
    if (any(actual.get(name) != digest for name, digest in expected.items())
            or extras - {'data/linguary.db-wal', 'data/linguary.db-shm'}
            or extras != set(bound) or any(actual[name] != bound[name] for name in extras)):
        raise WorkspaceError('Файлы замороженного анализа изменились')
    if 'data/linguary.db-wal' in extras and (root / 'data' / 'linguary.db-wal').stat().st_size != 0:
        raise WorkspaceError('Необъявленный WAL содержит состояние вне замороженного анализа')


def freeze_analysis(workspace) -> AnalysisArtifact:
    _verify_owned(workspace)
    if workspace._application is None or not workspace._closed or workspace._frozen or not workspace.database_path.is_file():
        raise WorkspaceError('Сначала завершите задачи и закройте приложение')
    destination = workspace.root / 'analysis-artifact'
    if destination.exists():
        raise WorkspaceError('Снимок уже существует')
    lock_path = workspace.database_path.with_name(workspace.database_path.name + '.executors.lock')
    with lock_path.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkspaceError('Исполнитель ещё работает') from None
        destination.mkdir()
        try:
            data = destination / 'data'
            data.mkdir()
            with closing(sqlite3.connect(f'file:{workspace.database_path}?mode=ro', uri=True)) as source:
                _validate_analysis(source)
                with closing(sqlite3.connect(data / 'linguary.db')) as target:
                    source.backup(target)
            books = workspace.root / 'data' / 'books'
            if books.exists():
                if any(p.name == 'export' for p in books.rglob('*')):
                    raise WorkspaceError('Прежний экспорт запрещён')
                shutil.copytree(books, data / 'books')
            for name in ('inputs',):
                shutil.copytree(workspace.root / name, destination / name)
            for name in ('inputs.json', 'context.json', 'settings.json'):
                shutil.copy2(workspace.root / name, destination / name)
            # Абсолютные пути EPUB должны ссылаться на файлы конкретной копии.
            with closing(sqlite3.connect(data / 'linguary.db')) as db, db:
                _relocate_sources(db, workspace.root, destination)
                receipts = [r[0] for r in db.execute('SELECT id FROM llm_call_logs ORDER BY id')]
            hashes = _hashes(destination)
            provenance = {'protocol_hash': workspace.protocol.content_hash, 'analysis_call_ids': receipts, 'origin': workspace.ownership, 'input_hashes': json.loads((workspace.root / 'inputs.json').read_text())}
            digest = stable_hash({'files': hashes, 'provenance': provenance})
            (destination / 'artifact.json').write_text(json.dumps({'content_hash': digest, 'files': hashes, 'provenance': provenance}, sort_keys=True))
            workspace._frozen = True
            return AnalysisArtifact(destination, digest, hashes, workspace.protocol, workspace.protected_roots, workspace.settings, provenance)
        except Exception:
            shutil.rmtree(destination)
            raise


def _relocate_sources(db, old_root, new_root):
    for resource_id, stored in db.execute('SELECT id, stored_path FROM resources WHERE stored_path IS NOT NULL').fetchall():
        path = Path(stored)
        if path.is_absolute():
            resolved = path.resolve()
            if not resolved.is_relative_to(old_root):
                raise WorkspaceError('Ресурс выходит за изолированный корень')
            target = new_root / resolved.relative_to(old_root)
            if not target.is_file():
                raise WorkspaceError('Ресурс отсутствует в снимке')
            db.execute('UPDATE resources SET stored_path=? WHERE id=?', (str(target), resource_id))
        elif '..' in path.parts:
            raise WorkspaceError('Относительный путь ресурса выходит за каталог')
    for book_id, source in db.execute('SELECT id, source_epub_path FROM books').fetchall():
        path = Path(source).resolve()
        if not path.is_relative_to(old_root):
            raise WorkspaceError('Путь исходной книги выходит за изолированный корень')
        target = new_root / path.relative_to(old_root)
        if not target.is_file():
            raise WorkspaceError('Исходная книга отсутствует в снимке')
        db.execute('UPDATE books SET source_epub_path=? WHERE id=?', (str(target), book_id))


def fork_processing(artifact, variant_id, *, destination_root=None) -> ComparisonWorkspace:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', variant_id):
        raise WorkspaceError('Идентификатор копии недопустим')
    _validate_artifact_files(artifact.root, artifact.file_hashes, artifact.sidecar_hashes)
    if stable_hash({'files': artifact.file_hashes, 'provenance': artifact.provenance}) != artifact.content_hash:
        raise WorkspaceError('Снимок анализа изменён')
    if destination_root is not None and not (Path(destination_root) / '.comparison-owner').is_file():
        raise WorkspaceError('Нет принадлежащего сравнению нового корня')
    destination, protected = _check_root((Path(destination_root) if destination_root is not None else artifact.root.parent) / 'processing' / variant_id, artifact.protected_roots)
    token = uuid4().hex
    destination.mkdir(parents=True, exist_ok=True)
    (destination / '.comparison-owner').write_text(token)
    destination.chmod(0o700)
    try:
        shutil.copytree(artifact.root / 'data', destination / 'data')
        shutil.copytree(artifact.root / 'inputs', destination / 'inputs')
        for name in ('context.json', 'inputs.json', 'settings.json', 'artifact.json'):
            shutil.copy2(artifact.root / name, destination / name)
        with closing(sqlite3.connect(destination / 'data' / 'linguary.db')) as db, db:
            _relocate_sources(db, artifact.source_root or artifact.root, destination)
        settings = replace(artifact.settings, storage=StorageSettings(data_root=str(destination / 'data')))
        return ComparisonWorkspace(destination, protected, artifact.protocol, settings, token, artifact.content_hash)
    except Exception:
        _cleanup(destination, token)
        raise WorkspaceError('Не удалось создать независимую копию анализа') from None
