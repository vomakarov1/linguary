from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from linguary.application.ports.llm import LLMRequest
from linguary.comparison.llm import ObservationLedger
from linguary.comparison.protocol import draft_protocol, stable_hash
from linguary.comparison.runner import _finalize

from test_late_metadata import late_case, verify


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


@pytest.fixture
def partial_case(late_case):
    import hashlib
    import json
    import shutil
    from datetime import UTC, datetime
    from decimal import Decimal
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from linguary.adapters.db.models import Base, BookRow, LLMCallLogRow, ProcessingTaskRow
    from linguary.adapters.db.repositories import SqlAlchemyChapterAnalysisRepository, SqlAlchemyDocumentRepository
    from linguary.adapters.epub import EbookLibEpubReader
    from linguary.application.context_builder import textual_blocks
    from linguary.application.ports.epub import EpubReadRequest
    from linguary.comparison.protocol import file_hash
    from linguary.domain import BlockTreatment, ChapterAnalysis, TreatmentPlan

    root, protocol, receipt, evidence = late_case
    partial = root / 'partial'
    (partial / '.comparison-owner').write_text('partial-owner')
    (partial / 'inputs').mkdir()
    source = Path(__file__).resolve().parents[2] / 'fixtures/epub/synthetic_story.epub'
    shutil.copyfile(source, partial / 'inputs/source.epub')
    book_id = 'book-' + file_hash(source)[:16]
    document = EbookLibEpubReader().read(EpubReadRequest(book_id=book_id, source_path=str(source), source_language='en', target_language='ru')).document
    section = document.ordered_sections[0]
    blocks = tuple(block.id for block in textual_blocks(section))
    receipt['chapter_id'] = section.id
    receipt['group_id'] = hashlib.sha256(json.dumps([book_id, receipt['stage'], section.id, list(blocks)], separators=(',', ':')).encode()).hexdigest()
    receipt['operation_id'] = hashlib.sha256('\0'.join((receipt['variant_id'], '1', book_id, receipt['stage'], receipt['group_id'])).encode()).hexdigest()
    from linguary.comparison import workspace
    bound = {str(Path(workspace.__file__).resolve()): file_hash(Path(workspace.__file__))}
    groups = ('instruction_version', 'schema_version', 'checks_version')
    protocol = protocol.model_copy(update={'source_hashes': {chapter.id: file_hash(source) for chapter in protocol.chapters},
        'chapters': tuple(chapter.model_copy(update={'chapter_id':'1'}) for chapter in protocol.chapters),
        'version_hashes': {group: bound for group in groups}, **{group:'sha256:'+group for group in groups},
        'projection_method': json.dumps({'prior_attempts_total':4,'known_prior_cost':'0.0082042','global_budget':'10','global_attempt_limit':88})})
    inputs = {chapter.id: {'path':'inputs/source.epub','sha256':file_hash(source)} for chapter in protocol.chapters}
    (partial / 'inputs.json').write_text(json.dumps(inputs))
    (partial / 'context.json').write_text(json.dumps({'glossary':protocol.fixed_glossary,'previous_context':protocol.previous_context}))
    manifest = {'protocol_hash': protocol.content_hash, **{group:getattr(protocol,group) for group in groups},
                'grouping': {'selected_sections': {book_id: [section.id,document.ordered_sections[1].id]}}}
    (partial / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'manifest.json').write_text(json.dumps({'protocol_hash':protocol.content_hash}))
    (root / 'prepared.json').write_text(json.dumps({'protocol_hash':protocol.content_hash,'manifest_hash':stable_hash({'protocol_hash':protocol.content_hash})}))
    (root / 'result.json').write_text(json.dumps({'protocol_hash':protocol.content_hash,'simulated':False,
        'violations':['unknown_or_changed_route','unlinked_log'], 'manifests':{'analysis':manifest},
        'continuation_provenance':{'imported_call_uids':[]}, 'variants':[{'variant_id':'luna/luna','completed':False,'failures':['branch_incomplete']}]}))
    (root / 'report.json').write_text(json.dumps({'manifest':{'protocol_hash':protocol.content_hash},'observations':[receipt],'total':{'cost':receipt['cost'],'unique_calls':1},'violations':['unknown_or_changed_route','unlinked_log']}))
    for path in (root/'receipts/uid.json',partial/'receipts/uid.json'):
        path.write_text(json.dumps(receipt))
    (partial / 'data').mkdir()
    database = partial / 'data/linguary.db'
    engine = create_engine('sqlite:///'+str(database))
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        session.add(BookRow(id=book_id,title='Test',authors_json=[],source_language='en',target_language='ru',status='imported',source_epub_path=str(source),created_at=now))
        session.flush()
        SqlAlchemyDocumentRepository(session).save(document)
        analysis = ChapterAnalysis.create(id='analysis',section_id=section.id,treatment_plan=TreatmentPlan.of_treatments(section.id,[(block,BlockTreatment.TRANSLATE) for block in blocks]),translation_instructions='Переводить точно.',chapter_summary='События главы.',expected_block_ids=blocks)
        SqlAlchemyChapterAnalysisRepository(session).save(book_id,analysis)
        for task_id, scope, status in [('paid-task',section.id,'succeeded'),('blocked-task',document.ordered_sections[1].id,'failed')]:
            errors = {} if status == 'succeeded' else {'error_type':'ConfigurationError','error_message':'Запись наблюдения недоступна; новые вызовы остановлены','error_detail':'linguary.application.ports.llm.LLMConfigurationError'}
            session.add(ProcessingTaskRow(id=task_id,book_id=book_id,task_type='analyze_chapter',scope_id=scope,status=status,attempts=1,created_at=now,**errors))
        common = dict(book_id=book_id,stage='chapter_analysis',provider='anthropic',model=receipt['reported_model'],started_at=now,attempt=1)
        session.add(LLMCallLogRow(id='call',task_id='paid-task',duration_ms=10,success=True,prompt_tokens=7867,completion_tokens=2709,cost=Decimal(receipt['cost']),**common))
        session.add(LLMCallLogRow(id='blocked',task_id='blocked-task',duration_ms=0,success=False,error_type='ConfigurationError',error_detail='linguary.comparison.llm.ComparisonStopped',error_message='Нарушены условия сравнения; новые вызовы остановлены',**common))
    engine.dispose()
    data=json.loads(evidence.read_text())
    data['series_3']['protected_files_sha256']={name:file_hash(root/name) for name in data['series_3']['protected_files_sha256']}
    evidence.write_text(json.dumps(data))
    return root, protocol, receipt, evidence


def partial_proof(case, destination):
    from linguary.comparison.workspace import _continuation_inventory, inspect_partial_analysis
    root=case[0]
    return inspect_partial_analysis(source_root=root,source_inventory=_continuation_inventory(root),supplement=verify(case),destination_root=destination)


def test_proof_links_receipt_log_and_saved_fragment(partial_case,tmp_path):
    proof=partial_proof(partial_case,tmp_path/'proof').document
    assert proof['verdict']=='reusable',proof['reasons']
    assert proof['confirmed_fragments'][0]['call_id']=='call'
    assert proof['confirmed_fragments'][0]['coverage']=='complete'
    assert len(proof['missing_fragments'])==1
    assert proof['log_explanations'][0]['call_id']=='blocked'
    assert proof['source_inventory_before']==proof['source_inventory_after']==proof['copy_inventory_before_read']


@pytest.mark.parametrize('change', ['missing_log','cost','route','usage','unexpected_log','second_blocked','conflicting_task','unexplained_unlinked','missing_analysis','incomplete','coverage_status','reordered','empty_instructions','wrong_operation','changed_context','wrong_imports','duplicate_generation','duplicate_first'])
def test_missing_or_conflicting_log_unexplained_unlinked_or_incomplete_fragment_refuses_reuse(partial_case,tmp_path,change):
    import json
    import sqlite3
    from linguary.comparison.protocol import file_hash
    root,protocol,receipt,evidence=partial_case
    database=root/'partial/data/linguary.db'
    with sqlite3.connect(database) as db:
        if change=='missing_log': db.execute("DELETE FROM llm_call_logs WHERE id='call'")
        if change=='cost': db.execute("UPDATE llm_call_logs SET cost='0.01' WHERE id='call'")
        if change=='route': db.execute("UPDATE llm_call_logs SET provider='Anthropic' WHERE id='call'")
        if change=='usage': db.execute("UPDATE llm_call_logs SET prompt_tokens=5444 WHERE id='call'")
        if change=='unexpected_log': db.execute("UPDATE llm_call_logs SET cost='0.1' WHERE id='blocked'")
        if change=='second_blocked':
            columns=[row[1] for row in db.execute('PRAGMA table_info(llm_call_logs)')]
            db.execute('INSERT INTO llm_call_logs ('+','.join(columns)+") SELECT "+','.join("'second-blocked'" if name=='id' else name for name in columns)+" FROM llm_call_logs WHERE id='blocked'")
        if change=='conflicting_task': db.execute("UPDATE processing_tasks SET error_detail='other' WHERE id='blocked-task'")
        if change=='unexplained_unlinked': db.execute("DELETE FROM llm_call_logs WHERE id='blocked'")
        if change=='missing_analysis': db.execute('DELETE FROM chapter_analyses')
        if change=='incomplete': db.execute('DELETE FROM treatment_plan_items WHERE id=(SELECT min(id) FROM treatment_plan_items)')
        if change=='coverage_status': db.execute("UPDATE chapter_analyses SET coverage_status='incomplete'")
        if change=='reordered': db.execute('UPDATE treatment_plan_items SET position=-position')
        if change=='empty_instructions': db.execute("UPDATE chapter_analyses SET translation_instructions=''")
    if change=='wrong_operation':
        receipt['operation_id']='other'
        for path in (root/'receipts/uid.json',root/'partial/receipts/uid.json'):
            path.write_text(json.dumps(receipt))
        report=json.loads((root/'report.json').read_text());report['observations']=[receipt];(root/'report.json').write_text(json.dumps(report))
    if change=='changed_context': (root/'partial/context.json').write_text('{}')
    if change in ('wrong_imports','duplicate_generation','duplicate_first'):
        result=json.loads((root/'result.json').read_text())
        if change=='wrong_imports':result['continuation_provenance']['imported_call_uids']=['foreign']
        else:
            duplicate=dict(receipt,call_uid='other',call_id='other-call',operation_id='other-operation' if change=='duplicate_generation' else receipt['operation_id'],generation_id=receipt['generation_id'] if change=='duplicate_generation' else 'other-generation')
            (root/'receipts/other.json').write_text(json.dumps(duplicate))
            report=json.loads((root/'report.json').read_text());report['observations'].append(duplicate);report['total']={'cost':'0.128472','unique_calls':2};(root/'report.json').write_text(json.dumps(report))
            result['continuation_provenance']['imported_call_uids']=['other']
        (root/'result.json').write_text(json.dumps(result))
    data=json.loads(evidence.read_text());data['series_3']['protected_files_sha256']={name:file_hash(root/name) for name in data['series_3']['protected_files_sha256']};evidence.write_text(json.dumps(data))
    proof=partial_proof(partial_case,tmp_path/'proof').document
    assert proof['verdict']=='insufficient_evidence'
    assert proof['reasons'] or not proof['confirmed_fragments']
    assert proof['live_permission'] is False


def test_partial_proof_reads_only_fresh_copy_with_committed_wal(partial_case,tmp_path,monkeypatch):
    import sqlite3
    from linguary.comparison.workspace import _continuation_inventory
    source=partial_case[0]
    database=source/'partial/data/linguary.db'
    with sqlite3.connect(database) as base:
        base.execute("UPDATE llm_call_logs SET cost='0' WHERE id='call'")
    writer=sqlite3.connect(database)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute("UPDATE chapter_analyses SET chapter_summary='Committed in WAL'")
    writer.execute("UPDATE llm_call_logs SET cost='0.064236' WHERE id='call'")
    writer.commit()
    assert database.with_name('linguary.db-wal').stat().st_size>0
    before=_continuation_inventory(source)
    original=sqlite3.connect
    connections=[]
    def connect(path,*args,**kwargs):
        assert str(tmp_path/'proof') in str(path)
        connections.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(sqlite3,'connect',connect)
    try:
        proof=partial_proof(partial_case,tmp_path/'proof').document
        assert proof['verdict']=='reusable',proof['reasons']
        assert len(connections)==2
        assert proof['copy_inventory_before_read']['partial/data/linguary.db-wal']==before['partial/data/linguary.db-wal']
        assert _continuation_inventory(source)==before
    finally:
        writer.close()


@pytest.mark.parametrize('failure',['copy','symlink','overlap','existing','forged_supplement'])
def test_copy_failure_and_symlink_preserve_source_inventory(partial_case,tmp_path,monkeypatch,failure):
    from linguary.comparison.workspace import _continuation_inventory, inspect_partial_analysis, WorkspaceError
    from linguary.comparison import workspace
    root=partial_case[0]
    supplement=verify(partial_case)
    before=_continuation_inventory(root)
    destination=tmp_path/'proof'
    if failure=='copy': monkeypatch.setattr(workspace.shutil,'copytree',lambda *a,**k: (_ for _ in ()).throw(OSError('copy failed')))
    if failure=='symlink': destination.symlink_to(root,target_is_directory=True)
    if failure=='overlap': destination=root/'proof'
    if failure=='existing': destination.mkdir()
    if failure=='forged_supplement': supplement.document['verified']['model']='forged'
    with pytest.raises((WorkspaceError,OSError)):
        inspect_partial_analysis(source_root=root,source_inventory=before,supplement=supplement,destination_root=destination)
    assert _continuation_inventory(root)==before


def test_late_supplement_does_not_duplicate_cost_or_first_response(partial_case,tmp_path):
    proof=partial_proof(partial_case,tmp_path/'proof').document
    assert proof['accounting_delta']=={'posts':0,'cost':'0','first_responses':0}
    assert proof['accounting']['posts']==5
    assert proof['accounting']['known_cost']=='0.0724402'
    assert proof['accounting']['remaining_posts']==83
    assert proof['complete_artifact'] is False and proof['live_permission'] is False
    assert proof['historical_variants'][0]['completed'] is False


def test_partial_analysis_never_becomes_complete_artifact_or_live_permission(partial_case,tmp_path):
    from linguary.comparison.protocol import ContinuationBinding, ProtocolError, validate_for_live
    from linguary.comparison.workspace import AnalysisArtifact, WorkspaceError, _continuation_protocol_matches
    proof=partial_proof(partial_case,tmp_path/'proof')
    assert not isinstance(proof,AnalysisArtifact)
    assert proof.document['verdict']=='reusable'
    assert proof.document['complete_artifact'] is False
    assert proof.document['live_permission'] is False
    assert not list((tmp_path/'proof/partial').glob('analysis-artifact/artifact.json'))
    previous=partial_case[1].model_copy(update={'repeats':1,'continuation':ContinuationBinding(path='existing',sha256='0'*64)})
    with pytest.raises(WorkspaceError,match='первого исходного повтора'):
        _continuation_protocol_matches(previous,previous)
    with pytest.raises(ProtocolError,match='Нет явного принятия'):
        validate_for_live(partial_case[1],None)


def test_partial_proof_rejects_normalized_overlap_before_copy(partial_case,tmp_path,monkeypatch):
    from linguary.comparison import workspace
    from linguary.comparison.workspace import WorkspaceError, _continuation_inventory, inspect_partial_analysis
    root=partial_case[0]
    outside=tmp_path/'outside'
    outside.mkdir()
    destination=outside/'..'/root.name/'proof'
    before=_continuation_inventory(root)
    monkeypatch.setattr(workspace.shutil,'copytree',lambda *a,**k:pytest.fail('overlap must be rejected before writes'))
    with pytest.raises(WorkspaceError,match='непересекающийся'):
        inspect_partial_analysis(source_root=root,source_inventory=before,supplement=verify(partial_case),destination_root=destination)
    assert _continuation_inventory(root)==before
