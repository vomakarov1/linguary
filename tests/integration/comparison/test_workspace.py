import sqlite3
from pathlib import Path

import pytest

from linguary.comparison.protocol import draft_protocol, file_hash
from linguary.comparison.workspace import prepare_workspace, freeze_analysis, fork_processing, WorkspaceError


def test_rejects_production_root_and_symlink_alias(tmp_path):
    protected = tmp_path / 'production'
    protected.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(protected, target_is_directory=True)
    for root in (protected, protected / 'child', tmp_path, alias / 'child'):
        with pytest.raises(WorkspaceError):
            prepare_workspace(root, [protected], draft_protocol())


def test_negative_prepare_keeps_source_hashes(tmp_path):
    source = tmp_path / 'source.epub'
    source.write_bytes(b'original')
    protocol = draft_protocol()
    chapters = [c.model_copy(update={'source': str(source)}) for c in protocol.chapters]
    protocol = protocol.model_copy(update={'chapters': tuple(chapters)})
    root = tmp_path / 'existing'
    root.mkdir()
    (root / 'keep').write_text('keep')
    before = file_hash(source)
    with pytest.raises(WorkspaceError):
        prepare_workspace(root, [tmp_path / 'production'], protocol)
    assert file_hash(source) == before
    assert (root / 'keep').read_text() == 'keep'


def prepared(tmp_path):
    workspace = prepare_workspace(tmp_path / 'run', [tmp_path / 'production'], draft_protocol())
    app = workspace.build_application()
    app.dispose()
    source = workspace.root / 'data' / 'books' / 'book' / 'source.epub'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'original EPUB')
    with sqlite3.connect(workspace.database_path) as db:
        db.execute("INSERT INTO books(id,title,authors_json,source_language,target_language,status,source_epub_path,created_at) VALUES ('book','Title','[]','en','ru','analyzed',?,'2026-10-11')", (str(source),))
        db.execute("INSERT INTO resources(id,book_id,source_path,media_type,stored_path,support_status) VALUES ('resource','book','source.epub','application/epub+zip',?,'supported')", (str(source),))
        db.execute("INSERT INTO sections(id,book_id,title,\"order\",source_resource_path) VALUES ('chapter','book','Chapter',0,'chapter.xhtml')")
        db.execute("INSERT INTO chapter_analyses(id,book_id,section_id,coverage_status,translation_instructions,chapter_summary,before_you_read_json) VALUES ('analysis','book','chapter','complete','Fixed guidance','Summary','[]')")
    return workspace


def test_rejects_completed_results(tmp_path):
    workspace = prepared(tmp_path)
    with sqlite3.connect(workspace.database_path) as db:
        db.execute('CREATE TABLE unwanted_result (id TEXT)')
        db.execute("INSERT INTO processing_runs(id,book_id,status,created_at,glossary_snapshot_json,llm_settings_json) VALUES ('run','book','completed','2026-10-11','[]','{}')")
    with pytest.raises(WorkspaceError):
        freeze_analysis(workspace)


def test_processing_forks_share_analysis_hash(tmp_path):
    workspace = prepared(tmp_path)
    artifact = freeze_analysis(workspace)
    left = fork_processing(artifact, 'luna')
    right = fork_processing(artifact, 'sonnet')
    assert left.analysis_hash == right.analysis_hash == artifact.content_hash
    assert left.database_path != right.database_path
    with sqlite3.connect(left.database_path) as db:
        db.execute('CREATE TABLE independent(id TEXT)')
    with sqlite3.connect(right.database_path) as db:
        assert db.execute("SELECT translation_instructions FROM chapter_analyses").fetchone() == ('Fixed guidance',)
        assert Path(db.execute("SELECT stored_path FROM resources").fetchone()[0]).is_relative_to(right.root)
        assert Path(db.execute("SELECT source_epub_path FROM books").fetchone()[0]).is_relative_to(right.root)
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='independent'").fetchall()


def test_copy_failure_removes_only_owned_workspace(tmp_path, monkeypatch):
    import linguary.comparison.workspace as module
    source = tmp_path / 'source.epub'
    source.write_bytes(b'original')
    protocol = draft_protocol()
    protocol = protocol.model_copy(update={'chapters': tuple(c.model_copy(update={'source': str(source)}) for c in protocol.chapters)})
    def fail(*args, **kwargs):
        raise OSError('copy failed')
    monkeypatch.setattr(module.shutil, 'copy2', fail)
    root = tmp_path / 'run'
    with pytest.raises(WorkspaceError):
        prepare_workspace(root, [tmp_path / 'production'], protocol)
    assert not root.exists()
    assert source.read_bytes() == b'original'


def test_unclosed_application_blocks_snapshot(tmp_path):
    workspace = prepare_workspace(tmp_path / 'run', [tmp_path / 'production'], draft_protocol())
    app = workspace.build_application()
    try:
        with pytest.raises(WorkspaceError):
            freeze_analysis(workspace)
    finally:
        app.dispose()


def test_changed_analysis_artifact_blocks_fork(tmp_path):
    artifact = freeze_analysis(prepared(tmp_path))
    with sqlite3.connect(artifact.root / 'data' / 'linguary.db') as db:
        db.execute("UPDATE chapter_analyses SET chapter_summary='changed'")
    with pytest.raises(WorkspaceError):
        fork_processing(artifact, 'luna')


def test_changed_storage_settings_block_application_before_protected_writes(tmp_path):
    from dataclasses import replace
    from linguary.adapters.config import StorageSettings
    protected = tmp_path / 'production'
    protected.mkdir()
    (protected / 'keep').write_text('unchanged')
    workspace = prepare_workspace(tmp_path / 'run', [protected], draft_protocol())
    workspace.settings = replace(workspace.settings, storage=StorageSettings(data_root=str(protected)))
    with pytest.raises(WorkspaceError):
        workspace.build_application()
    assert sorted(p.name for p in protected.iterdir()) == ['keep']


def test_two_processing_forks_after_gc_keep_frozen_hash(tmp_path):
    import gc
    workspace = prepared(tmp_path)
    artifact = freeze_analysis(workspace)
    digest = file_hash(artifact.root / 'data' / 'linguary.db')
    left = fork_processing(artifact, 'left')
    app = left.build_application()
    app.dispose()
    gc.collect()
    assert file_hash(artifact.root / 'data' / 'linguary.db') == digest
    right = fork_processing(artifact, 'right')
    assert right.analysis_hash == left.analysis_hash


def test_snapshot_excludes_rejected_diagnostics_without_changing_source(tmp_path, monkeypatch):
    import json
    import linguary.comparison.workspace as module
    workspace = prepared(tmp_path)
    live_body = 'live-diagnostic-marker-7f8ad7f63b9a4e5a'
    deleted_marker = 'deleted-diagnostic-marker-954b56147e9b42a5'
    deleted_body = deleted_marker * 2000
    metadata = {'schema_version': 1, 'decision': 'rejected',
                'body_meta': {'body_status': 'available', 'stored_bytes': len(live_body)}}
    attempt_id = '12345678-1234-5678-1234-567812345678'
    with sqlite3.connect(workspace.database_path) as db:
        db.execute('PRAGMA secure_delete = OFF')
        db.execute("INSERT INTO llm_attempts(id,book_id,generation,ordinal,state,context_json,outcome_json) VALUES (?,'book',0,1,'rejected','{\"schema_version\":1}','{\"schema_version\":1}')", (attempt_id,))
        db.execute("INSERT INTO llm_call_logs(id,book_id,stage,provider,model,started_at,duration_ms,attempt,success,error_type,error_message,response_text,prompt_tokens,completion_tokens,cost,attempt_id,diagnostic_meta_json) VALUES ('call','book','chapter_analysis','fake','actual','2026-10-25',1,1,0,'ModelOutputError','Schema mismatch',?,10,20,'0.007',?,?)", (live_body, attempt_id, json.dumps(metadata)))
        db.execute("INSERT INTO llm_call_logs(id,book_id,stage,provider,model,started_at,duration_ms,attempt,success,response_text) VALUES ('deleted','book','chapter_analysis','fake','actual','2026-10-25',1,1,0,?)", (deleted_body,))
        db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.execute("DELETE FROM llm_call_logs WHERE id = 'deleted'")
        db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    before = file_hash(workspace.database_path)
    assert live_body.encode() in workspace.database_path.read_bytes()
    assert deleted_marker.encode() in workspace.database_path.read_bytes()
    original_connect = sqlite3.connect

    class UnsafeBackupConnection(sqlite3.Connection):
        def backup(self, target, *args, **kwargs):
            super().backup(target, *args, **kwargs)
            target.execute('PRAGMA secure_delete = OFF')

    def connect(database, *args, **kwargs):
        if isinstance(database, str) and database.startswith('file:') and database.endswith('?mode=ro'):
            kwargs['factory'] = UnsafeBackupConnection
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(module.sqlite3, 'connect', connect)
    artifact = freeze_analysis(workspace)
    assert file_hash(workspace.database_path) == before
    with sqlite3.connect(workspace.database_path) as source:
        assert source.execute("SELECT response_text FROM llm_call_logs").fetchone()[0] == live_body
    for database in (artifact.root / 'data' / 'linguary.db',
                     fork_processing(artifact, 'left').database_path,
                     fork_processing(artifact, 'right').database_path):
        for db_file in database.parent.glob(f'{database.name}*'):
            contents = db_file.read_bytes()
            assert live_body.encode() not in contents
            assert deleted_marker.encode() not in contents
    with sqlite3.connect(artifact.root / 'data' / 'linguary.db') as target:
        body, raw, usage, cost, actual_attempt = target.execute('SELECT response_text,diagnostic_meta_json,prompt_tokens,cost,attempt_id FROM llm_call_logs').fetchone()
        assert body is None
        assert json.loads(raw)['body_meta']['body_status'] == 'unavailable'
        assert json.loads(raw)['body_meta']['reason'] == 'excluded_from_analysis_snapshot'
        assert usage == 10 and cost == '0.007' and actual_attempt == attempt_id
        assert target.execute('SELECT state FROM llm_attempts').fetchone()[0] == 'rejected'


def test_snapshot_still_rejects_unmarked_debug_body(tmp_path):
    workspace = prepared(tmp_path)
    with sqlite3.connect(workspace.database_path) as db:
        db.execute("INSERT INTO llm_call_logs(id,book_id,stage,provider,model,started_at,duration_ms,attempt,success,response_text) VALUES ('call','book','chapter_analysis','fake','actual','2026-10-25',1,1,1,'private debug')")
    with pytest.raises(WorkspaceError, match='Сырые ответы'):
        freeze_analysis(workspace)
