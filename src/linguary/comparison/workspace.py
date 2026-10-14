"""Изолированные каталоги и закрытые согласованные снимки анализа."""
from __future__ import annotations

from contextlib import closing

import fcntl
import json
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.comparison.protocol import ComparisonProtocol, LUNA, file_hash, settings_projection, stable_hash, validate_input_bindings
from linguary.comparison.llm import CallObservation
from linguary.domain import TokenUsage


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
        if source_root == workspace.root or source_root.is_relative_to(workspace.root) or workspace.root.is_relative_to(source_root):
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
                    or entry[2] != data['actual_provider'] or entry[3] not in (data.get('reported_model'), data['actual_model'])
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
