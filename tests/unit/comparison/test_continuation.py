from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from linguary.application.ports.llm import LLMRequest
from linguary.comparison.llm import ObservationLedger
from linguary.comparison.protocol import draft_protocol, stable_hash
from linguary.comparison.runner import _finalize


@pytest.mark.parametrize('missing_log', [False, True])
def test_response_verdict_does_not_depend_on_failed_chapter(missing_log):
    ledger = ObservationLedger()
    request = LLMRequest.for_translation(instructions='translate', input_text='group')
    good = ledger.begin(request, model='model', route='route')
    good.call_id = 'good'
    bad = ledger.begin(request, model='model', route='route')
    bad.call_id = 'bad'
    logs = [SimpleNamespace(id='bad', success=False, task_id='chapter')]
    if not missing_log:
        logs.append(SimpleNamespace(id='good', success=True, task_id='chapter'))
    @contextmanager
    def transaction():
        yield SimpleNamespace(llm_calls=SimpleNamespace(list_for_book=lambda _: logs),
            tasks=SimpleNamespace(list_for_book=lambda _: [SimpleNamespace(id='chapter', status=SimpleNamespace(value='failed'))]))
    application = SimpleNamespace(unit_of_work=SimpleNamespace(transaction=transaction))
    _finalize(application, ledger, ['book'])
    assert bad.accepted is False
    assert good.accepted is (None if missing_log else True)
    assert ledger.violations == (['unknown_final_status'] if missing_log else [])


def test_legacy_protocol_hash_remains_unchanged_without_continuation():
    protocol = draft_protocol()
    old_document = protocol.model_dump(mode='json')
    old_document.pop('continuation', None)
    assert protocol.content_hash == stable_hash(old_document)
    assert protocol.content_hash == 'c71051d6ba5da5fb90d27f2a395cc3775c7bea91468e9afceaf40e52f7038f7a'


@pytest.mark.parametrize('change', ['none', 'unbound', 'nonempty_wal', 'foreign_extra', 'missing', 'modified'])
def test_frozen_artifact_accepts_only_bound_empty_wal_and_cache(tmp_path, change):
    from linguary.comparison.protocol import file_hash
    from linguary.comparison.workspace import _validate_artifact_files, WorkspaceError
    data = tmp_path / 'data'
    data.mkdir()
    for name in ['data/linguary.db'] + [f'input-{index}.bin' for index in range(11)]:
        (tmp_path / name).write_bytes(b'frozen')
    expected = {str(path.relative_to(tmp_path)): file_hash(path) for path in tmp_path.rglob('*') if path.is_file()}
    assert len(expected) == 12
    digest = stable_hash({'files': expected, 'provenance': {'protocol_hash': 'origin'}})
    (tmp_path / 'artifact.json').write_text('{}')
    (data / 'linguary.db-wal').write_bytes(b'')
    (data / 'linguary.db-shm').write_bytes(bytes(32768))
    bound = {str(path.relative_to(tmp_path)): file_hash(path) for path in data.iterdir() if path.name.endswith(('-wal', '-shm'))}
    if change == 'unbound':
        bound = {}
    if change == 'nonempty_wal':
        (data / 'linguary.db-wal').write_bytes(b'not represented by frozen digest')
        bound['data/linguary.db-wal'] = file_hash(data / 'linguary.db-wal')
    if change == 'foreign_extra':
        (data / 'other.bin').write_bytes(b'extra')
    if change == 'missing':
        (tmp_path / 'input-0.bin').unlink()
    if change == 'modified':
        (tmp_path / 'input-0.bin').write_bytes(b'changed')
    if change == 'none':
        _validate_artifact_files(tmp_path, expected, bound)
        assert stable_hash({'files': expected, 'provenance': {'protocol_hash': 'origin'}}) == digest
    else:
        with pytest.raises(WorkspaceError):
            _validate_artifact_files(tmp_path, expected, bound)


def test_two_forks_preserve_wal_backed_frozen_analysis(tmp_path):
    import json
    import sqlite3
    from contextlib import closing
    from linguary.adapters.config import AppSettings
    from linguary.comparison.workspace import AnalysisArtifact, _continuation_inventory, _hashes, fork_processing

    owned = tmp_path / 'owned'
    owned.mkdir()
    (owned / '.comparison-owner').write_text('owner')
    root = owned / 'frozen-analysis'
    (root / 'data').mkdir(parents=True)
    (root / 'inputs').mkdir()
    source = root / 'inputs' / 'source.epub'
    source.write_bytes(b'frozen source')
    database = root / 'data' / 'linguary.db'
    with closing(sqlite3.connect(database)) as db, db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE resources (id TEXT, stored_path TEXT)')
        db.execute('CREATE TABLE books (id TEXT, source_epub_path TEXT)')
        db.execute('INSERT INTO resources VALUES (?, ?)', ('resource', str(source)))
        db.execute('INSERT INTO books VALUES (?, ?)', ('book', str(source)))
    assert database.read_bytes()[18:20] == bytes([2, 2])
    for name in ('inputs.json', 'context.json', 'settings.json'):
        (root / name).write_text('{}')
    files, provenance = _hashes(root), {'protocol_hash': 'same analysis'}
    digest = stable_hash({'files': files, 'provenance': provenance})
    (root / 'artifact.json').write_text(json.dumps({'files': files, 'provenance': provenance, 'content_hash': digest}))
    artifact = AnalysisArtifact(root, digest, files, draft_protocol(), (tmp_path / 'protected',), AppSettings(), provenance)
    before = _continuation_inventory(root)
    first = fork_processing(artifact, 'SL', destination_root=owned)
    second = fork_processing(artifact, 'SS', destination_root=owned)
    assert _continuation_inventory(root) == before
    assert first.analysis_hash == second.analysis_hash == digest
    for fork in (first, second):
        assert fork.root.is_relative_to(owned)
        with closing(sqlite3.connect(fork.database_path)) as db:
            assert db.execute('SELECT source_epub_path FROM books').fetchone()[0] == str(fork.root / 'inputs' / 'source.epub')
