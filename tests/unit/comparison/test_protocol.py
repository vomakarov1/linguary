from pathlib import Path

import pytest

from linguary.comparison.protocol import (
    ComparisonProtocol, ProtocolError, ProtocolApproval, build_manifest,
    draft_protocol, load_protocol, validate_for_live,
)


def test_draft_protocol_prepares_without_network(tmp_path: Path):
    protocol = draft_protocol()
    path = tmp_path / 'protocol.json'
    path.write_text(protocol.model_dump_json())
    assert load_protocol(path).content_hash == protocol.content_hash
    assert protocol.repeats is None
    assert build_manifest(protocol, {}, {}).protocol_hash == protocol.content_hash


def test_missing_numbers_block_live():
    protocol = draft_protocol()
    with pytest.raises(ProtocolError):
        validate_for_live(protocol, ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='user', accepted_at='2026-10-11'))


def test_approval_hash_mismatch_blocks_live():
    with pytest.raises(ProtocolError):
        validate_for_live(draft_protocol(), ProtocolApproval(protocol_hash='0' * 64, accepted=True, accepted_by='user', accepted_at='2026-10-11'))


def test_manifest_excludes_prompts_and_secrets():
    manifest = build_manifest(draft_protocol(), {}, {'llm': {'model': 'test', 'api_key': 'SECRET', 'prompt': 'PROMPT', 'proxy_url': 'SECRET'}, 'prompt': 'PROMPT'})
    assert 'SECRET' not in manifest.model_dump_json()
    assert 'PROMPT' not in manifest.model_dump_json()
    assert manifest.effective_settings == {'llm': {'model': 'test'}}


def test_protocol_requires_control_and_independent_chapters():
    data = draft_protocol().model_dump()
    data['chapters'] = data['chapters'][:1]
    with pytest.raises(ValueError):
        ComparisonProtocol.model_validate(data)
    data = draft_protocol().model_dump()
    data['variants'] = data['variants'][:3]
    with pytest.raises(ValueError):
        ComparisonProtocol.model_validate(data)


def complete_protocol(tmp_path):
    import zipfile
    fixture = Path(__file__).resolve().parents[2] / "fixtures/epub/synthetic_story.epub"
    directory = tmp_path
    alternate = directory / "alternate.epub"
    with zipfile.ZipFile(fixture) as original, zipfile.ZipFile(alternate, "w") as target:
        for member in original.infolist():
            content = original.read(member)
            if member.filename.endswith(".xhtml"):
                content = content.replace(b"Wren", b"Independent Wren")
            target.writestr(member, content)
    data = draft_protocol().model_dump()
    data.update(
        chapters=[c.model_copy(update={'source': str(alternate if index == 3 else fixture), 'chapter_id': str(index + 1 if index < 3 else 1)}) for index, c in enumerate(draft_protocol().chapters)],
        critical_places=[{'chapter_id': 'control', 'location': 'Nature/natural', 'expected_explanation': 'Explain the wordplay'}],
        rubric={k: 3 for k in draft_protocol().rubric}, scale_max=5,
        sufficient_gain=1, comparability_margin=0.2, repeats=1,
        minimum_first_responses={'analysis': 20, 'processing': 20},
        schedule=[v.id for v in draft_protocol().variants],
        temperature_policy='set', generation={'temperature': 0.5, 'max_output_tokens': 1000, 'max_attempts': 2},
        splitting={'max_fragment_blocks': 10, 'max_unit_source_chars': 1000},
        checks={'glossary_confidence_threshold': 0.8, 'glossary_review_min_mentions': 5, 'glossary_review_min_chapters': 3, 'target_language_min_share': 0.6, 'language_check_min_length': 80, 'repetition_ngram_size': 6, 'repetition_min_repeats': 3, 'language_notes_per_chapter': 8},
        requests_per_minute=20, concurrency=2, max_total_attempts=88, budget=20, max_call_cost=1,
        stop_conditions=['budget'], reference_book='reference.epub', projection_method='by source characters',
        max_book_cost=10, max_book_seconds=3600,
        routes={'openai/gpt-6-luna': 'test-openai', 'anthropic/claude-sonnet-4.6': 'test-anthropic'},
        context_cache_policy='record', retain_rejected=False, diagnostic_retention_seconds=0,
        diagnostic_access='local operator', code_version='test', instruction_version='test', schema_version='test', checks_version='test',
    )
    data["version_hashes"] = {key: {str(Path(__file__).resolve()): __import__("linguary.comparison.protocol", fromlist=["file_hash"]).file_hash(Path(__file__))} for key in ("code_version", "instruction_version", "schema_version", "checks_version")}
    data["source_hashes"] = {c.id: __import__("linguary.comparison.protocol", fromlist=["file_hash"]).file_hash(Path(c.source)) for c in data["chapters"]}
    data["route_metadata"] = {m: {"provider_names": [r], "model_versions": [m]} for m, r in data["routes"].items()}
    data["cost_bounds"] = {"max_request_bytes": 100000, "max_input_tokens": 100000, "input_price_per_mtok": 1, "output_price_per_mtok": 1, "output_includes_reasoning": True, "input_bound_method": "synthetic", "evidence": "MockTransport only"}
    return ComparisonProtocol.model_validate(data)


def test_complete_protocol_accepts_only_exact_explicit_approval(tmp_path):
    protocol = complete_protocol(tmp_path)
    approval = ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='user', accepted_at='2026-10-11')
    assert validate_for_live(protocol, approval).protocol_hash == protocol.content_hash
    changed = protocol.model_copy(update={'budget': 21})
    with pytest.raises(ProtocolError):
        validate_for_live(changed, approval)


def test_unequal_schedule_blocks_live(tmp_path):
    protocol = complete_protocol(tmp_path).model_copy(update={'schedule': ('luna/luna',) * 4})
    approval = ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='user', accepted_at='2026-10-11')
    with pytest.raises(ProtocolError):
        validate_for_live(protocol, approval)


@pytest.mark.parametrize('alias', ['path', 'section'])
def test_same_physical_chapter_alias_blocks_live(tmp_path, alias):
    import shutil
    protocol = complete_protocol(tmp_path)
    first = protocol.chapters[0]
    copied = tmp_path / 'copied.epub'
    shutil.copy2(first.source, copied)
    duplicate = protocol.chapters[1].model_copy(update={'source': str(copied) if alias == 'path' else first.source, 'chapter_id': '1' if alias == 'path' else 'OEBPS/chapter1.xhtml'})
    value = protocol.model_copy(update={'chapters': (first, duplicate, *protocol.chapters[2:])})
    approval = ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    with pytest.raises(ProtocolError):
        validate_for_live(value, approval)


def test_explicit_temperature_omission_requires_exact_approval(tmp_path):
    protocol = complete_protocol(tmp_path)
    omitted = protocol.model_copy(update={'temperature_policy': 'omit', 'generation': dict(protocol.generation, temperature=None)})
    approval = ProtocolApproval(protocol_hash=omitted.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    assert validate_for_live(omitted, approval).protocol.temperature_policy == 'omit'
    with pytest.raises(ProtocolError):
        validate_for_live(protocol, approval)
    unknown = omitted.model_copy(update={'temperature_policy': None})
    with pytest.raises(ProtocolError):
        validate_for_live(unknown, approval.model_copy(update={'protocol_hash': unknown.content_hash}))


@pytest.mark.parametrize('policy,value', [('omit', .5), ('set', None)])
def test_inconsistent_temperature_policy_blocks_live(tmp_path, policy, value):
    protocol = complete_protocol(tmp_path)
    changed = protocol.model_copy(update={'temperature_policy': policy, 'generation': dict(protocol.generation, temperature=value)})
    approval = ProtocolApproval(protocol_hash=changed.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    with pytest.raises(ProtocolError):
        validate_for_live(changed, approval)


@pytest.mark.parametrize('binding', ['source_hashes', 'version_hashes'])
def test_changed_accepted_input_blocks_prepare_and_live(tmp_path, binding):
    from linguary.comparison.__main__ import prepare
    protocol = complete_protocol(tmp_path)
    values = dict(getattr(protocol, binding))
    if binding == 'version_hashes':
        key = next(iter(values))
        values[key] = {path: '0' * 64 for path in values[key]}
    else:
        values[next(iter(values))] = '0' * 64
    changed = protocol.model_copy(update={binding: values})
    approval = ProtocolApproval(protocol_hash=changed.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    root = tmp_path / 'comparison'
    with pytest.raises(ProtocolError):
        prepare(root, [tmp_path / 'protected'], changed)
    assert not root.exists()
    with pytest.raises(ProtocolError):
        validate_for_live(changed, approval)


def test_unverified_input_bound_cannot_open_live(tmp_path):
    protocol = complete_protocol(tmp_path)
    changed = protocol.model_copy(update={'cost_bounds': protocol.cost_bounds.model_copy(update={'input_bound_method': 'unverified'})})
    approval = ProtocolApproval(protocol_hash=changed.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    with pytest.raises(ProtocolError, match='граница'):
        validate_for_live(changed, approval)


def context_envelope_protocol(tmp_path, *, budget=20):
    import json
    from linguary.comparison.protocol import CostBounds, EndpointSnapshot, file_hash
    protocol = complete_protocol(tmp_path)
    snapshots = {}
    for model, route in protocol.routes.items():
        path = tmp_path / (route + '-endpoints.json')
        path.write_text(json.dumps({'data': {'id': model, 'endpoints': [{
            'tag': route, 'provider_name': route, 'status': 0, 'max_prompt_tokens': None,
            'context_length': 1000000, 'max_completion_tokens': 128000,
            'pricing': {'prompt': '.000003', 'completion': '.000015', 'input_cache_read': '.0000003',
                        'input_cache_write': '.00000375', 'input_cache_write_1h': '.000006'},
        }]}}))
        snapshots[model] = EndpointSnapshot(path=str(path), sha256=file_hash(path))
    return protocol.model_copy(update={'budget': budget, 'max_call_cost': 6.12,
        'generation': dict(protocol.generation, max_output_tokens=8000),
        'cost_bounds': CostBounds(max_request_bytes=1048576, max_input_tokens=1000000, input_price_per_mtok=6,
            output_price_per_mtok=15, output_includes_reasoning=True, input_bound_method='provider_context_limit',
            endpoint_snapshots=snapshots, input_price_multiplier=2, evidence='Pinned synthetic endpoint context quota and cache multiplier')})


def test_provider_context_quota_envelope_can_be_approved_offline(tmp_path):
    protocol = context_envelope_protocol(tmp_path)
    approval = ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    assert validate_for_live(protocol, approval).protocol.cost_bounds.max_input_tokens == 1000000
    from linguary.comparison.llm import BudgetGuard
    from decimal import Decimal
    guard = BudgetGuard(Decimal('20'), max_total_attempts=88)
    assert guard.reserve(Decimal('6.12')) and guard.reserve(Decimal('6.12'))
    with pytest.raises(ProtocolError):
        low = protocol.model_copy(update={'budget': 5})
        validate_for_live(low, approval.model_copy(update={'protocol_hash': low.content_hash}))
    assert not (tmp_path / 'comparison').exists()


@pytest.mark.parametrize('change', ['quota', 'price', 'snapshot', 'multiplier'])
def test_context_envelope_rejects_unproven_bounds(tmp_path, change):
    protocol = context_envelope_protocol(tmp_path)
    bounds = protocol.cost_bounds
    if change == 'snapshot':
        next(iter(bounds.endpoint_snapshots.values()))
        Path(next(iter(bounds.endpoint_snapshots.values())).path).write_text('{}')
    else:
        bounds = bounds.model_copy(update={'quota': {'max_input_tokens': 64000},
                                           'price': {'input_price_per_mtok': 3},
                                           'multiplier': {'input_price_multiplier': 1}}[change])
        protocol = protocol.model_copy(update={'cost_bounds': bounds})
    approval = ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='TEST', accepted_at='2026-10-11')
    with pytest.raises(ProtocolError):
        validate_for_live(protocol, approval)
