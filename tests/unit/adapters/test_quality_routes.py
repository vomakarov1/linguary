"""Synthetic local attestations, never real provider tariffs."""
from dataclasses import replace
from decimal import Decimal
import json
import pytest
from linguary.adapters.config import LLMSettings
from linguary.adapters.config.quality_routes import load_quality_routes, PROTOCOL
from linguary.application.ports.llm import LLMRequest, LLMResponseKind
from linguary.domain.book_conditions import WorkStateError


def fixture_route():
    return dict(route_id='synthetic', model='review-test', base_url='https://example.invalid/v1',
        price_version='synthetic-only-v1', currency='USD', input_per_mtok='1.00000000000001',
        output_per_mtok='2', max_fee_usd='0.000001', max_output_tokens=80,
        max_reasoning_tokens=20, max_completion_tokens=100, supports_max_completion_tokens=True,
        completion_includes_reasoning=True, input_bound=dict(protocol=PROTOCOL,
            max_request_bytes=100000, max_input_tokens=200000,
            proof_method='Synthetic test assertion; not valid for a real provider',
            proof_version='fixture-v1', evidence_uri='urn:linguary:synthetic-test'))


def load(tmp_path, monkeypatch, item):
    path = tmp_path / 'quality.json'
    path.write_text(json.dumps(dict(schema_version=1, routes=[item])))
    monkeypatch.setenv('LINGUARY_QUALITY_ROUTES', str(path))
    return load_quality_routes(LLMSettings(base_url='https://example.invalid/v1'))


def test_exact_prices_and_full_wire_bound(tmp_path, monkeypatch):
    item = fixture_route()
    registry = load(tmp_path, monkeypatch, item)
    route = registry['synthetic', 'review-test']
    assert route.input_per_mtok == Decimal('1.00000000000001')
    request = LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions='Review', input_text='{}', max_completion_tokens=100)
    assert route.estimate(request) > 0
    # Even tiny user content includes the strict schema and message framing.
    item['input_bound']['max_request_bytes'] = 100
    route = load(tmp_path, monkeypatch, item)['synthetic', 'review-test']
    with pytest.raises(WorkStateError, match='полный запрос'):
        route.estimate(request)
    with pytest.raises(WorkStateError):
        route.estimate(replace(request, max_completion_tokens=101))


@pytest.mark.parametrize('field', sorted(fixture_route()))
def test_every_route_attestation_is_required(tmp_path, monkeypatch, field):
    item = fixture_route()
    del item[field]
    assert load(tmp_path, monkeypatch, item) == {}


@pytest.mark.parametrize('field', sorted(fixture_route()['input_bound']))
def test_every_input_proof_field_is_required(tmp_path, monkeypatch, field):
    item = fixture_route()
    del item['input_bound'][field]
    assert load(tmp_path, monkeypatch, item) == {}


@pytest.mark.parametrize('field,value', [('input_per_mtok', 1.0), ('max_fee_usd', 'NaN'),
    ('output_per_mtok', '-1'), ('base_url', 'https://attacker.invalid/v1'),
    ('supports_max_completion_tokens', False), ('completion_includes_reasoning', False),
    ('max_completion_tokens', 80), ('currency', 'EUR'), ('max_reasoning_tokens', True)])
def test_invalid_or_untrusted_routes_disabled(tmp_path, monkeypatch, field, value):
    item = fixture_route()
    item[field] = value
    assert load(tmp_path, monkeypatch, item) == {}


def test_default_off_and_unreadable_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv('LINGUARY_QUALITY_ROUTES', raising=False)
    assert load_quality_routes(LLMSettings()) == {}
    monkeypatch.setenv('LINGUARY_QUALITY_ROUTES', str(tmp_path / 'missing'))
    assert load_quality_routes(LLMSettings()) == {}


def test_money_rounds_up_without_decimal_context_loss():
    from linguary.application.quality_budget import microusd, QualityRoute
    assert microusd(Decimal('0.0000010000000000000000000000000000000000001')) == 2
    route = QualityRoute('model', 'route', 'price',
        Decimal('1.0000000000000000000000000000000000001'), Decimal('0'), Decimal('0'),
        1, lambda request: 1, 'synthetic', True)
    assert route.charge(1, 0) == 2
