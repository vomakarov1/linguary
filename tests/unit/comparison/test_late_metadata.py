import json

import pytest

from linguary.comparison.protocol import RouteMetadata, SONNET, draft_protocol, file_hash, stable_hash
from linguary.comparison.workspace import _continuation_inventory, WorkspaceError


@pytest.fixture
def late_case(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / '.comparison-owner').write_text('owner')
    protocol = draft_protocol().model_copy(update={'route_metadata': {SONNET: RouteMetadata(provider_names=('Anthropic',), model_versions=('native-sonnet',))}, 'routes': {SONNET: 'anthropic'}})
    receipt = dict(call_uid='uid', operation_id='operation', call_id='call', generation_id='generation', assigned_model=SONNET, assigned_route='anthropic', reported_model=SONNET, actual_provider='Anthropic', actual_model=None, model_identity_source=None, metadata_status_code=404, metadata_check_reason='deadline_exceeded', metadata_get_count=23, stage='chapter_analysis', chapter_id='book:chapter', variant_id=f'analysis-{SONNET}', repeat=1, sequence=1, group_id='group', submitted=True, received_response=True, first_response=True, accepted=True, truncated=False, full_response_cache=False, context_cache=False, cost='0.064236', cost_source='provider_receipt', usage={'prompt_tokens':7867,'completion_tokens':2709}, duration_ms=10, failure_category=None)
    for relative in ('receipts/uid.json', 'partial/receipts/uid.json'):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
    for name, data in [('manifest.json', {'protocol_hash':protocol.content_hash}), ('result.json', {'protocol_hash':protocol.content_hash, 'simulated':False, 'violations':['unknown_or_changed_route','unlinked_log']}), ('report.json', {'manifest':{'protocol_hash':protocol.content_hash},'violations':['unknown_or_changed_route','unlinked_log']})]:
        (root / name).write_text(json.dumps(data))
    (root / 'prepared.json').write_text(json.dumps({'protocol_hash':protocol.content_hash, 'manifest_hash':stable_hash(json.loads((root/'manifest.json').read_text()))}))
    selected = dict(id='generation', model='native-sonnet', provider_name='Anthropic', total_cost=.064236, cache_discount=None, native_tokens_prompt=7867, native_tokens_completion=2709, native_tokens_cached=0, native_tokens_reasoning=0, tokens_prompt=5444, tokens_completion=2683, finish_reason='stop', native_finish_reason='end_turn', is_byok=False, streamed=True, cancelled=False, generation_time=54378, latency=5600, created_at='2026-10-15T21:22:41.055Z')
    series = dict(purpose='free_late_generation_metadata_diagnosis_not_result_rewrite', method='GET /api/v1/generation?id=<generation_id>, as ReceiptHTTPClient._generation_model; trust_env proxy auto; follow_redirects=False', new_POST=0, GET_requests=1, generation_id='generation', status=200, body_class='JSON_generation_metadata', run_started_at_utc='2026-10-15T21:22:40+00:00', receipt_mtime_utc='2026-10-15T21:28:40+00:00', attempts=[{'at_utc':'2026-10-15T23:50:03.566653+00:00','status':200,'selected_headers':{'Content-Type':'application/json','Cache-Control':'private, no-store','CF-Cache-Status':'BYPASS','Date':'Thu, 15 Oct 2026 23:50:04 GMT'}}], selected=selected, protected_files_sha256={p:file_hash(root/p) for p in ('receipts/uid.json','partial/receipts/uid.json','result.json','report.json')}, id_matches=False)
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps({'status':404, 'selected':None, 'series_3':series}))
    return root, protocol, receipt, evidence


def verify(case):
    from linguary.comparison.late_metadata import verify_late_metadata
    root, protocol, _, evidence = case
    return verify_late_metadata(evidence_path=evidence, evidence_sha256=file_hash(evidence), source_root=root, source_inventory=_continuation_inventory(root), protocol=protocol)


def test_series3_verifies_without_trusting_summary_flags(late_case):
    root, _, receipt, _ = late_case
    before = _continuation_inventory(root)
    supplement = verify(late_case)
    assert supplement.document['verified']['model'] == 'native-sonnet'
    assert supplement.document['receipt'] == receipt
    assert supplement.document['historical']['violations'] == ['unknown_or_changed_route','unlinked_log']
    assert supplement.document['accounting_delta'] == {'posts':0, 'cost':'0', 'first_responses':0}
    assert _continuation_inventory(root) == before


def test_earlier_404_does_not_hide_late_200(late_case):
    assert verify(late_case).document['diagnosis']['json_pointer'] == '/series_3'


@pytest.mark.parametrize('field,value', [('id','other'),('model','Native-sonnet'),('provider_name','anthropic'),('total_cost',.1),('total_cost',True),('total_cost',float('nan')),('native_tokens_prompt',5444),('native_tokens_completion',2683),('native_tokens_cached',1),('tokens_prompt',True),('native_tokens_reasoning',float('inf')),('cancelled',True)])
def test_reject_wrong_generation_exact_model_provider_cost_native_usage_cache(late_case, field, value):
    evidence=late_case[3]
    data=json.loads(evidence.read_text())
    data['series_3']['selected'][field]=value
    evidence.write_text(json.dumps(data))
    with pytest.raises(WorkspaceError): verify(late_case)


@pytest.mark.parametrize('change', ['missing','method','date','headers','count','status','hash','extra','symlink','receipt_copy'])
def test_reject_missing_fields_aliases_symlinks_and_nonfinite_numbers(late_case, change):
    root, _, _, evidence=late_case
    data=json.loads(evidence.read_text())
    series=data['series_3']
    if change=='missing': del series['selected']['tokens_completion']
    if change=='method': series['method']='POST /api/v1/generation'
    if change=='date': series['attempts'][0]['at_utc']='not-a-time'
    if change=='headers': series['attempts'][0]['selected_headers']['Content-Type']='text/plain'
    if change=='count': series['GET_requests']=True
    if change=='status': series['attempts'][0]['status']=404
    if change=='hash': series['protected_files_sha256']['result.json']='0'*64
    if change=='extra': series['protected_files_sha256']['../outside']='0'*64
    if change=='symlink': (root/'linked').symlink_to(evidence)
    if change=='receipt_copy': (root/'partial/receipts/uid.json').write_text('{}')
    evidence.write_text(json.dumps(data))
    with pytest.raises(WorkspaceError): verify(late_case)


def test_reject_changed_evidence_or_source_hash(late_case):
    from linguary.comparison.late_metadata import verify_late_metadata
    root, protocol, _, evidence=late_case
    inventory=_continuation_inventory(root)
    digest=file_hash(evidence)
    evidence.write_text(evidence.read_text()+' ')
    with pytest.raises(WorkspaceError):
        verify_late_metadata(evidence_path=evidence,evidence_sha256=digest,source_root=root,source_inventory=inventory,protocol=protocol)
    (root/'extra').write_text('unbound')
    with pytest.raises(WorkspaceError):
        verify_late_metadata(evidence_path=evidence,evidence_sha256=file_hash(evidence),source_root=root,source_inventory=inventory,protocol=protocol)


@pytest.mark.parametrize('field,value', [('accepted',False),('first_response',False),('received_response',False),('truncated',True),('full_response_cache',True),('context_cache',None),('repeat',True),('cost_source','estimated'),('metadata_check_reason','verified')])
def test_supplement_preserves_original_receipt_and_runtime_failure(late_case,field,value):
    root,_,receipt,evidence=late_case
    receipt[field]=value
    for path in (root/'receipts/uid.json',root/'partial/receipts/uid.json'):
        path.write_text(json.dumps(receipt))
    data=json.loads(evidence.read_text())
    data['series_3']['protected_files_sha256']={name:file_hash(root/name) for name in data['series_3']['protected_files_sha256']}
    evidence.write_text(json.dumps(data))
    with pytest.raises(WorkspaceError):verify(late_case)
