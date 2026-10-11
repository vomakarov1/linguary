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


def complete_protocol():
    data = draft_protocol().model_dump()
    data.update(
        chapters=[c.model_copy(update={'source': f'{c.id}.epub'}) for c in draft_protocol().chapters],
        critical_places=[{'chapter_id': 'control', 'location': 'Nature/natural', 'expected_explanation': 'Explain the wordplay'}],
        rubric={k: 3 for k in draft_protocol().rubric}, scale_max=5,
        sufficient_gain=1, comparability_margin=0.2, repeats=1,
        minimum_first_responses={'analysis': 20, 'processing': 20},
        schedule=[v.id for v in draft_protocol().variants],
        generation={'temperature': 0.5, 'max_output_tokens': 1000, 'max_attempts': 2},
        splitting={'max_fragment_blocks': 10, 'max_unit_source_chars': 1000},
        checks={'glossary_confidence_threshold': 0.8, 'glossary_review_min_mentions': 5, 'glossary_review_min_chapters': 3, 'target_language_min_share': 0.6, 'language_check_min_length': 80, 'repetition_ngram_size': 6, 'repetition_min_repeats': 3, 'language_notes_per_chapter': 8},
        requests_per_minute=20, concurrency=2, budget=20, max_call_cost=1,
        stop_conditions=['budget'], reference_book='reference.epub', projection_method='by source characters',
        max_book_cost=10, max_book_seconds=3600,
        routes={'openai/gpt-6-luna': 'test-openai', 'anthropic/claude-sonnet-4.6': 'test-anthropic'},
        context_cache_policy='record', retain_rejected=False, diagnostic_retention_seconds=0,
        diagnostic_access='local operator', code_version='test', instruction_version='test', schema_version='test', checks_version='test',
    )
    return ComparisonProtocol.model_validate(data)


def test_complete_protocol_accepts_only_exact_explicit_approval():
    protocol = complete_protocol()
    approval = ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='user', accepted_at='2026-10-11')
    assert validate_for_live(protocol, approval).protocol_hash == protocol.content_hash
    changed = protocol.model_copy(update={'budget': 21})
    with pytest.raises(ProtocolError):
        validate_for_live(changed, approval)


def test_unequal_schedule_blocks_live():
    protocol = complete_protocol().model_copy(update={'schedule': ('luna/luna',) * 4})
    approval = ProtocolApproval(protocol_hash=protocol.content_hash, accepted=True, accepted_by='user', accepted_at='2026-10-11')
    with pytest.raises(ProtocolError):
        validate_for_live(protocol, approval)
