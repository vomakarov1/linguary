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
