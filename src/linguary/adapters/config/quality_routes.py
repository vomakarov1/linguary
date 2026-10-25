"""Explicit operator attestations; absent or invalid configuration never enables review."""
from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from linguary.application.quality_budget import QualityRoute
from linguary.domain.book_conditions import WorkStateError
from linguary.adapters.llm.openai_compatible import _body

PROTOCOL = 'openai-chat-json-schema-v1'
_FIELDS = {'route_id', 'model', 'base_url', 'price_version', 'currency',
    'input_per_mtok', 'output_per_mtok', 'max_fee_usd', 'max_output_tokens',
    'max_reasoning_tokens', 'max_completion_tokens', 'supports_max_completion_tokens',
    'completion_includes_reasoning', 'input_bound'}
_BOUND_FIELDS = {'protocol', 'max_request_bytes', 'max_input_tokens', 'proof_method', 'proof_version', 'evidence_uri'}


def _positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('invalid bound')
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('missing attestation')
    return value


def _price(value):
    if not isinstance(value, str):
        raise ValueError('prices must be exact decimal strings')
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise ValueError('invalid price')
    return number


def _route(value, settings):
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError('incomplete route')
    for name in ('route_id', 'model', 'base_url', 'price_version'):
        _text(value[name])
    url = urlsplit(value['base_url'])
    if (url.scheme not in ('https', 'http') or not url.hostname or url.username or url.password
            or url.query or url.fragment or value['base_url'].rstrip('/') != settings.base_url.rstrip('/')):
        raise ValueError('credentials only available for the configured provider URL')
    if (value['currency'] != 'USD' or value['supports_max_completion_tokens'] is not True
            or value['completion_includes_reasoning'] is not True):
        raise ValueError('unsupported billing or cap')
    output = _positive(value['max_output_tokens'])
    reasoning = value['max_reasoning_tokens']
    if type(reasoning) is not int or reasoning < 0:
        raise ValueError('invalid reasoning cap')
    completion = _positive(value['max_completion_tokens'])
    if completion != output + reasoning:
        raise ValueError('completion cap must include output and reasoning')
    bound = value['input_bound']
    if not isinstance(bound, dict) or set(bound) != _BOUND_FIELDS or bound['protocol'] != PROTOCOL:
        raise ValueError('unsupported full-input proof')
    byte_limit = _positive(bound['max_request_bytes'])
    tokens = _positive(bound['max_input_tokens'])
    for key in ('proof_method', 'proof_version', 'evidence_uri'):
        _text(bound[key])
    # A fixed verified token bound applies only inside its attested full wire-size envelope.
    # This does not infer a token/byte ratio or download the supporting document.
    wire_settings = replace(settings, model=value['model'], temperature=None, max_output_tokens=None)
    def input_bound(request):
        if request.max_completion_tokens != completion:
            raise WorkStateError('предел запроса не соответствует подтверждённому маршруту')
        wire = json.dumps(_body(request, settings=wire_settings), ensure_ascii=True).encode('utf-8')
        if len(wire) > byte_limit:
            raise WorkStateError('полный запрос превышает подтверждённую границу')
        return tokens
    return QualityRoute(value['model'], value['route_id'], value['price_version'],
        _price(value['input_per_mtok']), _price(value['output_per_mtok']), _price(value['max_fee_usd']),
        completion, input_bound, bound['proof_version'], True, value['base_url'].rstrip('/'),
        tuple(bound[key] for key in sorted(_BOUND_FIELDS)))


def load_quality_routes(settings, path=None):
    """Load a secret-free local JSON registry; any invalid entry disables the registry."""
    path = path if path is not None else os.environ.get('LINGUARY_QUALITY_ROUTES')
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value, dict) or set(value) != {'schema_version', 'routes'} or type(value['schema_version']) is not int or value['schema_version'] != 1 or not isinstance(value['routes'], list):
            return {}
        routes = {}
        for item in value['routes']:
            route = _route(item, settings)
            key = (route.route_id, route.model)
            if key in routes:
                return {}
            routes[key] = route
        return routes
    except (OSError, ValueError, TypeError, ArithmeticError):
        return {}
