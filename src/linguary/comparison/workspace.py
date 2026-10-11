"""Изолированные каталоги и закрытые согласованные снимки анализа."""
from __future__ import annotations

from contextlib import closing

import fcntl
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from linguary.adapters.config import AppSettings, CheckSettings, LLMSettings, StorageSettings, WorkerPoolSettings
from linguary.comparison.protocol import ComparisonProtocol, file_hash, settings_projection, stable_hash


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
        llm = LLMSettings(**generation, **splitting, requests_per_minute=protocol.requests_per_minute, debug_llm_payloads=False)
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


def fork_processing(artifact, variant_id) -> ComparisonWorkspace:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', variant_id):
        raise WorkspaceError('Идентификатор копии недопустим')
    if any(p.is_symlink() for p in artifact.root.rglob('*')) or _hashes(artifact.root) != artifact.file_hashes or stable_hash({'files': artifact.file_hashes, 'provenance': artifact.provenance}) != artifact.content_hash:
        raise WorkspaceError('Снимок анализа изменён')
    destination, protected = _check_root(artifact.root.parent / 'processing' / variant_id, artifact.protected_roots)
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
            _relocate_sources(db, artifact.root, destination)
        settings = replace(artifact.settings, storage=StorageSettings(data_root=str(destination / 'data')))
        return ComparisonWorkspace(destination, protected, artifact.protocol, settings, token, artifact.content_hash)
    except Exception:
        _cleanup(destination, token)
        raise WorkspaceError('Не удалось создать независимую копию анализа') from None
