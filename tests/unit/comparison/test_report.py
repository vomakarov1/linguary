from decimal import Decimal

import pytest

from linguary.comparison.llm import CallObservation
from linguary.comparison.protocol import RouteMetadata, draft_protocol
from linguary.comparison.report import build_report, evaluate_candidate


def receipt(uid, operation='op', sequence=1, **changes):
    value = CallObservation(uid, operation, 'm', 'r', sequence, 'v', stage='translation_result', received_response=True, accepted=True, cost=Decimal('1'), duration_ms=100, actual_model='m', actual_provider='r', full_response_cache=False)
    for key, item in changes.items():
        setattr(value, key, item)
    return value


def report(items, **changes):
    data = {'variants': [{'variant_id': 'v', 'repeat': 1, 'completed': True}], 'simulated': False}
    data.update(changes)
    return build_report({'protocol_hash': draft_protocol().content_hash}, items, data, {}, {})


def test_first_received_response_after_network_error():
    value = report([receipt('1', received_response=False, accepted=False), receipt('2', sequence=2)])
    assert value.variants['v']['processing']['first_received'] == 1
    assert value.variants['v']['processing']['first_accepted'] == 1
    assert value.variants['v']['processing']['network_or_access_failures'] == 1


def test_retries_do_not_improve_first_response_rate():
    value = report([receipt('1', accepted=False), receipt('2', sequence=2)])
    assert value.variants['v']['processing']['rate'] == 0


def test_shared_analysis_counted_once_in_total_and_in_each_book_projection():
    analysis = receipt('a', operation='analysis', variant_id='analysis-m', stage='chapter_analysis', call_id='ca')
    value = report([analysis, analysis, receipt('p')], analyses=[{'content_hash': 'h', 'provenance': {'analysis_call_ids': ['ca']}}], variants=[{'variant_id': 'v', 'analysis_hash': 'h'}])
    assert value.total['cost'] == '2'
    assert value.variants['v']['standalone_sample_cost'] == '2'
    assert value.variants['v']['analysis']['first_received'] == 1


def test_unknown_cost_never_becomes_zero():
    value = report([receipt('1', cost=None)])
    assert value.total['cost'] is None and 'cost' in value.missing_data


def test_missing_reviews_or_minimum_samples_prevent_selection():
    protocol = draft_protocol()
    variant = protocol.variants[0]
    model = variant.processing_model
    protocol = protocol.model_copy(update={
        'routes': {model: 'r'},
        'route_metadata': {model: RouteMetadata(model_versions=(model,), provider_names=('r',))},
    })
    value = report([receipt('1', variant_id=variant.id, assigned_model=model, actual_model=model)])
    value.manifest['protocol_hash'] = protocol.content_hash
    assert evaluate_candidate(value, protocol).status == 'insufficient_data'


def test_failed_chapters_and_invalid_conditions_remain_visible():
    value = report([receipt('1')], violations=['changed_route'], variants=[{'variant_id': 'v', 'completed': False, 'failures': ['chapter_failed:c']}])
    assert value.variants['v']['repeats'][0]['failures'] == ['chapter_failed:c']
    assert evaluate_candidate(value, draft_protocol()).status == 'invalid_conditions'


def candidate_fixture():
    protocol = draft_protocol().model_copy(update={'repeats': 2, 'rubric': {key: 3 for key in draft_protocol().rubric}, 'scale_max': 5,
        'sufficient_gain': .5, 'comparability_margin': .3, 'minimum_first_responses': {'analysis': 1, 'processing': 1},
        'max_book_cost': 10, 'max_book_seconds': 100,
        'critical_places': ({'unused': True},)})
    from linguary.comparison.protocol import CriticalPlace
    protocol = protocol.model_copy(update={'critical_places': (CriticalPlace(chapter_id='control', location='pun', expected_explanation='Explain meaning'),)})
    models = {v.analysis_model for v in protocol.variants}
    protocol = protocol.model_copy(update={
        'routes': {model: 'route-' + model for model in models},
        'route_metadata': {model: RouteMetadata(model_versions=(model,), provider_names=('route-' + model,)) for model in models},
    })
    items, outcomes, reviews, books = [], {'variants': [], 'variant_models': {}, 'simulated': False}, {}, {}
    for variant in protocol.variants:
        outcomes['variant_models'][variant.id] = {'analysis_model': variant.analysis_model}
        for repeat in (1, 2):
            outcomes['variants'].append({'variant_id': variant.id, 'repeat': repeat, 'completed': True, 'failures': []})
            items.append(receipt(f'p-{variant.id}-{repeat}', operation=f'p-{variant.id}-{repeat}', variant_id=variant.id, repeat=repeat, assigned_model=variant.processing_model, actual_model=variant.processing_model, assigned_route=protocol.routes[variant.processing_model], actual_provider=protocol.routes[variant.processing_model]))
        score = 3 if variant.id == 'luna/luna' else 4
        reviews[variant.id] = {'repeat_scores': [score, score], 'criteria': {key: score for key in protocol.rubric}, 'critical_places': {'control:pun': True},
            **{key: True for key in ('no_gross_errors', 'simple_help_density_appropriate', 'help_without_duplication', 'analysis_reviewed', 'execution_after_dedup_reviewed', 'independent_chapters_confirmed')}}
        books[variant.id] = {'cost': '1', 'wall_seconds': 10, 'analysis_included': True, 'assumptions': ['fixed synthetic corpus']}
    for model in {v.analysis_model for v in protocol.variants}:
        for repeat in (1, 2):
            items.append(receipt(f'a-{model}-{repeat}', operation=f'a-{model}-{repeat}', variant_id=f'analysis-{model}', repeat=repeat, stage='chapter_analysis', assigned_model=model, actual_model=model, assigned_route=protocol.routes[model], actual_provider=protocol.routes[model]))
    value = build_report({'protocol_hash': protocol.content_hash}, items, outcomes, reviews, {'books': books, 'wall_seconds': 10, 'assumptions': ['fixed synthetic corpus']})
    return protocol, value


@pytest.mark.parametrize('mismatch', [None, 'actual_model', 'actual_provider', 'assigned_model', 'assigned_route', 'missing_metadata'])
def test_candidate_checks_receipts_against_agreed_route_metadata(mismatch):
    protocol, value = candidate_fixture()
    metadata = {model: RouteMetadata(model_versions=(model + '-2026-10',), provider_names=('Display ' + model,))
                for model in protocol.routes}
    if mismatch == 'missing_metadata':
        metadata.pop(value.observations[0]['assigned_model'])
    protocol = protocol.model_copy(update={'route_metadata': metadata})
    value.manifest['protocol_hash'] = protocol.content_hash
    for item in value.observations:
        item['actual_model'] = item['assigned_model'] + '-2026-10'
        item['actual_provider'] = 'Display ' + item['assigned_model']
    if mismatch and mismatch != 'missing_metadata':
        value.observations[0][mismatch] = 'foreign-value'
    decision = evaluate_candidate(value, protocol)
    if mismatch is None:
        assert decision.status == 'candidate'
        assert decision.variant_id == 'sonnet/sonnet'
    else:
        assert decision.status == 'invalid_conditions'
        assert decision.reasons == ['changed_route']


@pytest.mark.parametrize('assignment', ['model', 'route'])
def test_matching_receipt_cannot_override_protocol_assignment(assignment):
    protocol, value = candidate_fixture()
    item = value.observations[0]
    if assignment == 'model':
        item['assigned_model'] = item['actual_model'] = 'foreign-model'
    else:
        item['assigned_route'] = item['actual_provider'] = 'foreign-provider'
    decision = evaluate_candidate(value, protocol)
    assert decision.status == 'invalid_conditions'
    assert decision.reasons == ['changed_route']


@pytest.mark.parametrize('stage', ['translation_result', 'processing'])
@pytest.mark.parametrize('mismatch', ['other_allowed_model', 'unknown_variant'])
def test_processing_receipt_must_match_protocol_variant(stage, mismatch):
    protocol, value = candidate_fixture()
    item = value.observations[0]
    item['stage'] = stage
    if mismatch == 'other_allowed_model':
        other = next(model for model in protocol.routes if model != item['assigned_model'])
        item.update(assigned_model=other, actual_model=protocol.route_metadata[other].model_versions[0],
                    assigned_route=protocol.routes[other], actual_provider=protocol.route_metadata[other].provider_names[0])
    else:
        item['variant_id'] = 'unknown-variant'
    decision = evaluate_candidate(value, protocol)
    assert decision.status == 'invalid_conditions'
    assert decision.reasons == ['changed_route']


def test_quality_tie_prefers_simpler_eligible_variant():
    protocol, value = candidate_fixture()
    decision = evaluate_candidate(value, protocol)
    assert decision.status == 'candidate' and decision.variant_id == 'sonnet/sonnet'


@pytest.mark.parametrize('unknown', ['cost', 'wall_seconds', 'analysis_included'])
def test_unknown_projection_prevents_candidate(unknown):
    protocol, value = candidate_fixture()
    value.projection['books']['sonnet/sonnet'][unknown] = None
    assert evaluate_candidate(value, protocol).status == 'insufficient_data'


def test_95_percent_is_checked_per_stage_and_variant():
    protocol, value = candidate_fixture()
    for variant in value.variants.values():
        variant['analysis']['rate'] = .94
        variant['processing']['rate'] = 1
    assert evaluate_candidate(value, protocol).status == 'no_candidate'


def test_unknown_final_acceptance_prevents_candidate():
    protocol, value = candidate_fixture()
    value.variants['sonnet/sonnet']['analysis']['first_unknown'] = 1
    assert evaluate_candidate(value, protocol).status == 'insufficient_data'


def test_missing_critical_meaning_review_prevents_candidate():
    protocol, value = candidate_fixture()
    value.reviews['sonnet/sonnet']['critical_places']['control:pun'] = None
    assert evaluate_candidate(value, protocol).status == 'insufficient_data'


def test_projection_includes_analysis_cost_and_separate_wall_duration():
    p = draft_protocol()
    variant = p.variants[0]
    analysis = receipt('a', operation='analysis', variant_id='analysis', stage='chapter_analysis', assigned_model=variant.analysis_model, actual_model=variant.analysis_model, cost=Decimal('2'), duration_ms=100)
    processing = receipt('p', variant_id=variant.id, cost=Decimal('3'), duration_ms=200)
    outcomes = {'variants': [{'variant_id': variant.id, 'repeat': 1}], 'variant_models': {variant.id: {'analysis_model': variant.analysis_model}},
        'stage_wall_seconds': {f'analysis:1:{variant.analysis_model}': 4, f'processing:1:{variant.id}': 6}}
    value = build_report({}, [analysis, processing], outcomes, {}, {'source_character_ratio': 10, 'assumptions': ['same corpus and concurrency'], 'wall_seconds': 10})
    book = value.projection['books'][variant.id]
    assert book['cost'] == '50' and book['wall_seconds'] == 100 and book['analysis_included']
    assert value.variants[variant.id]['standalone_request_seconds'] == .3
    assert value.variants[variant.id]['standalone_wall_seconds']['mean'] == 10


def test_blind_reviews_are_mapped_only_after_reviewer_fills_materials():
    from linguary.comparison.report import aggregate_blind_reviews
    protocol, value = candidate_fixture()
    vid = protocol.variants[0].id
    review = value.reviews[vid]
    mapping = {'anonymous-a': {'variant_id': vid, 'repeat': 1}, 'anonymous-b': {'variant_id': vid, 'repeat': 2}}
    filled = aggregate_blind_reviews(mapping, {'anonymous-a': review, 'anonymous-b': review}, protocol)
    assert filled[vid]['criteria'] == review['criteria']
    assert filled[vid]['repeat_scores'] == [3, 3]
    incomplete = aggregate_blind_reviews(mapping, {'anonymous-a': review}, protocol)
    assert incomplete[vid]['criteria']['accuracy'] is None
    assert incomplete[vid]['no_gross_errors'] is None


def test_anonymous_epubs_preserve_book_text_and_original_metadata(tmp_path):
    import zipfile
    from linguary.comparison.protocol import LUNA, SONNET, file_hash
    from linguary.comparison.report import anonymize_epubs

    chapters = {
        'source.xhtml': '<html><head><title>A sonnet for luna</title></head><body><p>The <em>sonnet</em> describes a lunar landscape. luna.</p></body></html>',
        'translation.xhtml': '<html><head><title>Сонет о luna</title></head><body><p>Русский <em>перевод</em>: sonnet, lunar, luna — лунный пейзаж.</p></body></html>',
    }
    metadata = ('<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata>'
                '<dc:title>The sonnet of luna</dc:title><dc:creator>Luna Sonnet</dc:creator>'
                f'<meta name="linguary:analysis_model" content="{LUNA}"/>'
                f'<meta property="linguary:processing_model">{SONNET}</meta>'
                '<meta name="linguary:provenance">luna/sonnet</meta>'
                '<meta name="description" content="A lunar sonnet"/>'
                '<meta property="dcterms:modified">2026-10-11</meta>'
                '</metadata></package>')
    source = tmp_path / 'source.epub'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('content.opf', metadata)
        for name, text in chapters.items():
            archive.writestr(name, text.encode())
    before = file_hash(source)
    mapping = anonymize_epubs({'variants': [{'variant_id': 'luna/sonnet', 'repeat': 1, 'epubs': {'book': source}}]}, tmp_path)
    assert file_hash(source) == before
    label, = mapping
    with zipfile.ZipFile(tmp_path / 'review-materials' / f'{label}.epub') as archive:
        for name, text in chapters.items():
            assert archive.read(name) == text.encode()
        anonymous_metadata = archive.read('content.opf').decode()
        assert LUNA not in anonymous_metadata and SONNET not in anonymous_metadata
        assert 'linguary:provenance' not in anonymous_metadata
        assert '<dc:title>The sonnet of luna</dc:title>' in anonymous_metadata
        assert '<dc:creator>Luna Sonnet</dc:creator>' in anonymous_metadata
        assert '<meta name="description" content="A lunar sonnet"/>' in anonymous_metadata
        assert '<meta property="dcterms:modified">2026-10-11</meta>' in anonymous_metadata
    assert mapping[label] == {'variant_id': 'luna/sonnet', 'repeat': 1, 'book_id': 'book', 'chapter_ids': []}


def test_anonymous_metadata_changes_only_explicit_provenance_in_head():
    from linguary.comparison.report import _anonymous_metadata
    head = b'<head><meta name="linguary:model" content="openai/gpt-6-luna"/><title>luna sonnet</title></head>'
    body = '<body><p>Луна: lunar sonnet.</p><meta name="model" content="luna"/></body>'.encode()
    assert _anonymous_metadata(head + body) == b'<head><title>luna sonnet</title></head>' + body


def test_critical_review_is_scoped_to_its_chapter():
    from linguary.comparison.protocol import CriticalPlace
    from linguary.comparison.report import aggregate_blind_reviews
    protocol = draft_protocol().model_copy(update={'critical_places': (
        CriticalPlace(chapter_id='control', location='pun', expected_explanation='meaning'),
        CriticalPlace(chapter_id='wordplay', location='idiom', expected_explanation='idiom'),)})
    mapping = {'a': {'variant_id': protocol.variants[0].id, 'repeat': 1, 'chapter_ids': ['control']},
               'b': {'variant_id': protocol.variants[0].id, 'repeat': 1, 'chapter_ids': ['wordplay']}}
    sheet = {'a': {'critical_places': {'control:pun': True}}, 'b': {'critical_places': {'wordplay:idiom': True}}}
    result = aggregate_blind_reviews(mapping, sheet, protocol)
    assert result[protocol.variants[0].id]['critical_places'] == {'control:pun': True, 'wordplay:idiom': True}
    sheet['b']['critical_places']['wordplay:idiom'] = None
    assert aggregate_blind_reviews(mapping, sheet, protocol)[protocol.variants[0].id]['critical_places']['wordplay:idiom'] is None
