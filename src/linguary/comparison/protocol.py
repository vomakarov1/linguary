"""Локальный протокол сравнения; создание проекта не разрешает генерацию."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

LUNA = 'openai/gpt-6-luna'
SONNET = 'anthropic/claude-sonnet-4.6'


class ProtocolError(ValueError):
    """Условия запуска не приняты или неполны."""


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class ChapterInput(Contract):
    id: str
    source: str
    chapter_id: str
    role: Literal['control', 'simple', 'dialogue', 'wordplay']
    independent: bool


class Variant(Contract):
    id: str
    analysis_model: str
    processing_model: str


class CriticalPlace(Contract):
    chapter_id: str
    location: str
    expected_explanation: str


class RouteMetadata(Contract):
    provider_names: tuple[str, ...]
    model_versions: tuple[str, ...]


class EndpointSnapshot(Contract):
    path: str
    sha256: str


class CostBounds(Contract):
    max_request_bytes: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    input_price_per_mtok: float = Field(ge=0, allow_inf_nan=False)
    output_price_per_mtok: float = Field(ge=0, allow_inf_nan=False)
    output_includes_reasoning: bool
    input_bound_method: Literal['unverified', 'synthetic', 'provider_context_limit']
    endpoint_snapshots: dict[str, EndpointSnapshot] = Field(default_factory=dict)
    input_price_multiplier: float = Field(default=1, ge=1, allow_inf_nan=False)
    evidence: str


class ContinuationBinding(Contract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')


class ComparisonProtocol(Contract):
    version: Literal[1] = 1
    version_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    route_metadata: dict[str, RouteMetadata] = Field(default_factory=dict)
    cost_bounds: CostBounds | None = None
    chapters: tuple[ChapterInput, ...]
    variants: tuple[Variant, ...]
    fixed_glossary: dict[str, object] = Field(default_factory=dict)
    previous_context: dict[str, str] = Field(default_factory=dict)
    critical_places: tuple[CriticalPlace, ...] = ()
    rubric: dict[str, float | None] = Field(default_factory=dict)
    scale_max: float | None = Field(default=None, gt=0)
    sufficient_gain: float | None = Field(default=None, ge=0)
    comparability_margin: float | None = Field(default=None, ge=0)
    repeats: int | None = Field(default=None, gt=0)
    minimum_first_responses: dict[str, int | None] = Field(default_factory=lambda: {'analysis': None, 'processing': None})
    schedule: tuple[str, ...] = ()
    temperature_policy: Literal['set', 'omit'] | None = None
    proxy_mode: Literal['auto', 'direct'] = 'auto'
    continuation: ContinuationBinding | None = None
    generation: dict[str, float | int | None] = Field(default_factory=lambda: {'temperature': None, 'max_output_tokens': None, 'max_attempts': None})
    splitting: dict[str, int | None] = Field(default_factory=lambda: {'max_fragment_blocks': None, 'max_unit_source_chars': None})
    checks: dict[str, float | int | None] = Field(default_factory=dict)
    requests_per_minute: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    max_total_attempts: int | None = Field(default=None, gt=0)
    budget: float | None = Field(default=None, gt=0)
    max_call_cost: float | None = Field(default=None, gt=0)
    stop_conditions: tuple[str, ...] = ()
    reference_book: str | None = None
    projection_method: str | None = None
    max_book_cost: float | None = Field(default=None, gt=0)
    max_book_seconds: float | None = Field(default=None, gt=0)
    routes: dict[str, str] = Field(default_factory=dict)
    require_parameters: bool = True
    full_response_cache: bool = False
    context_cache_policy: str | None = None
    retain_rejected: bool | None = None
    diagnostic_retention_seconds: int | None = Field(default=None, ge=0)
    diagnostic_access: str | None = None
    code_version: str = 'unrecorded'
    instruction_version: str = 'unrecorded'
    schema_version: str = 'unrecorded'
    checks_version: str = 'unrecorded'

    @model_validator(mode='after')
    def check_dataset(self):
        if len({c.id for c in self.chapters}) != len(self.chapters):
            raise ValueError('Идентификаторы глав должны быть уникальны')
        controls = [c for c in self.chapters if c.role == 'control']
        if len(controls) != 1 or controls[0].independent:
            raise ValueError('Нужна одна контрольная глава')
        for role in ('simple', 'dialogue', 'wordplay'):
            if not any(c.role == role and c.independent for c in self.chapters):
                raise ValueError('Нужны три независимые главы разных ролей')
        expected = {(a, p) for a in (LUNA, SONNET) for p in (LUNA, SONNET)}
        if len(self.variants) != 4 or {(v.analysis_model, v.processing_model) for v in self.variants} != expected or len({v.id for v in self.variants}) != 4:
            raise ValueError('Нужны четыре сочетания заданных моделей')
        return self

    @property
    def content_hash(self) -> str:
        data = self.model_dump(mode='json')
        if self.continuation is None:
            data.pop('continuation')
        return stable_hash(data)


class ProtocolApproval(Contract):
    protocol_hash: str
    accepted: bool
    accepted_by: str
    accepted_at: str


class ValidatedProtocol(Contract):
    protocol: ComparisonProtocol
    approval: ProtocolApproval
    protocol_hash: str


class ComparisonManifest(Contract):
    version: Literal[1] = 1
    comparison_id: str
    protocol_hash: str
    code_version: str
    instruction_version: str
    schema_version: str
    checks_version: str
    input_hashes: dict[str, str]
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    grouping: dict[str, object] = Field(default_factory=dict)
    effective_settings: dict[str, object]
    temperature_policy: Literal['set', 'omit'] | None = None
    paths: dict[str, str] = Field(default_factory=dict)


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path | str) -> ComparisonProtocol:
    try:
        return ComparisonProtocol.model_validate_json(Path(path).read_text())
    except (ValueError, OSError):
        raise ProtocolError('Не удалось прочитать допустимый протокол') from None


def validate_for_live(protocol: ComparisonProtocol, approval: ProtocolApproval | None) -> ValidatedProtocol:
    if not approval or not approval.accepted or not approval.accepted_by.strip() or not approval.accepted_at.strip() or approval.protocol_hash != protocol.content_hash:
        raise ProtocolError('Нет явного принятия точного хеша протокола')
    def missing(value):
        if value is None or value == '' or value == []:
            return True
        if isinstance(value, dict):
            return any(missing(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(missing(v) for v in value)
        return False
    validate_input_bindings(protocol, required=True)
    data = protocol.model_dump(mode='json')
    if protocol.continuation is None:
        data.pop('continuation')
    if protocol.temperature_policy == 'omit':
        if protocol.generation.get('temperature') is not None:
            raise ProtocolError('Политика omit требует явно пустой temperature')
        data['generation'].pop('temperature', None)
    if missing(data) or any(data[k] == 'unrecorded' for k in ('code_version', 'instruction_version', 'schema_version', 'checks_version')):
        raise ProtocolError('Протокол содержит несогласованные условия')
    criteria = {'accuracy', 'naturalness', 'author_voice', 'difficulty_detection', 'helpfulness'}
    if not criteria <= protocol.rubric.keys() or set(protocol.minimum_first_responses) != {'analysis', 'processing'} or any(v is None or v < 1 for v in protocol.minimum_first_responses.values()):
        raise ProtocolError('Не согласованы рубрика и минимумы наблюдений')
    if any(protocol.schedule.count(v.id) != protocol.repeats for v in protocol.variants) or len(protocol.schedule) != 4 * protocol.repeats:
        raise ProtocolError('Расписание не соответствует равным повторам')
    if set(protocol.routes) != {LUNA, SONNET} or protocol.full_response_cache or not protocol.require_parameters:
        raise ProtocolError('Маршруты и политика кеша недопустимы')
    required_generation = {'temperature', 'max_output_tokens', 'max_attempts'}
    required_splitting = {'max_fragment_blocks', 'max_unit_source_chars'}
    required_checks = SAFE_SETTINGS['checks']
    if not required_generation <= protocol.generation.keys() or not required_splitting <= protocol.splitting.keys() or not required_checks <= protocol.checks.keys():
        raise ProtocolError('Не согласованы параметры генерации и локальные проверки')
    if any(protocol.generation[k] < 1 for k in ('max_output_tokens', 'max_attempts')) or any(protocol.splitting[k] < 1 for k in required_splitting):
        raise ProtocolError('Пределы генерации должны быть положительными')
    if any(v < 0 or v > protocol.scale_max for v in protocol.rubric.values()):
        raise ProtocolError('Минимумы качества вне принятой шкалы')
    # Разные пути или обозначения главы не создают независимые наблюдения.
    from linguary.adapters.epub import EbookLibEpubReader
    from linguary.application.ports.epub import EpubReadRequest
    physical = set()
    try:
        for chapter in protocol.chapters:
            path = Path(chapter.source).expanduser().resolve(strict=True)
            digest = file_hash(path)
            document = EbookLibEpubReader().read(EpubReadRequest(book_id='admission', source_path=str(path), source_language='en', target_language='ru')).document
            matches = [section for index, section in enumerate(document.ordered_sections, 1) if chapter.chapter_id in (section.id, section.source_resource_path, str(index))]
            if len(matches) != 1 or (digest, matches[0].id) in physical:
                raise ProtocolError('Нужны четыре разные физические главы')
            physical.add((digest, matches[0].id))
    except Exception:
        raise ProtocolError('Не удалось подтвердить независимые главы источников') from None
    chapter_ids = {c.id for c in protocol.chapters}
    if not protocol.critical_places or any(p.chapter_id not in chapter_ids or not p.location.strip() or not p.expected_explanation.strip() for p in protocol.critical_places):
        raise ProtocolError('Не согласованы критические места')
    if set(protocol.route_metadata) != set(protocol.routes) or any(not m.provider_names or not m.model_versions or any(not value.strip() for value in (*m.provider_names, *m.model_versions)) for m in protocol.route_metadata.values()):
        raise ProtocolError('Не приняты имена провайдеров и точные версии моделей')
    bounds = protocol.cost_bounds
    if bounds is None or not bounds.output_includes_reasoning or not bounds.evidence.strip() or bounds.input_bound_method == 'unverified':
        raise ProtocolError('Не доказана граница оплачиваемого входа и вывода')
    if bounds.input_bound_method == "provider_context_limit":
        validate_context_envelope(bounds, protocol.routes, protocol.route_metadata, protocol.generation["max_output_tokens"])
    from decimal import Decimal
    maximum = (Decimal(bounds.max_input_tokens) * Decimal(str(bounds.input_price_per_mtok)) + Decimal(protocol.generation['max_output_tokens']) * Decimal(str(bounds.output_price_per_mtok))) / Decimal(1000000)
    if maximum > Decimal(str(protocol.max_call_cost)):
        raise ProtocolError('Тарифные пределы превышают резерв вызова')
    if protocol.max_call_cost > protocol.budget:
        raise ProtocolError('Резерв вызова превышает бюджет')
    return ValidatedProtocol(protocol=protocol, approval=approval, protocol_hash=protocol.content_hash)


def validate_context_envelope(bounds, routes, metadata, output_tokens):
    """Консервативный предел по квоте точки, без оценки токенов по байтам."""
    from decimal import Decimal
    if bounds.input_price_multiplier < 2 or not set(routes) <= bounds.endpoint_snapshots.keys():
        raise ProtocolError('Не закреплены квоты точек и максимальный множитель записи кеша')
    for model, route in routes.items():
        snapshot = bounds.endpoint_snapshots[model]
        try:
            path = Path(snapshot.path).expanduser()
            if file_hash(path) != snapshot.sha256:
                raise ProtocolError('Снимок точки изменился')
            data = json.loads(path.read_text())['data']
            endpoints = [endpoint for endpoint in data['endpoints'] if endpoint['tag'] == route]
            if data['id'] != model or len(endpoints) != 1:
                raise ProtocolError('Неоднозначная точка в принятом снимке')
            endpoint = endpoints[0]
            limit = endpoint.get('max_prompt_tokens') or endpoint['context_length']
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > bounds.max_input_tokens:
                raise ProtocolError('Квота точки превышает принятый предел входа')
            if endpoint['provider_name'] not in metadata[model].provider_names or endpoint.get('status') != 0 or endpoint['max_completion_tokens'] < output_tokens:
                raise ProtocolError('Точка не соответствует принятому провайдеру или выводу')
            pricing = endpoint['pricing']
            for tariff in (pricing, *pricing.get('overrides', [])):
                prompt = Decimal(tariff.get('prompt', pricing['prompt'])) * 1000000
                completion = Decimal(tariff.get('completion', pricing['completion'])) * 1000000
                effective = prompt * Decimal(str(bounds.input_price_multiplier))
                if effective > Decimal(str(bounds.input_price_per_mtok)) or completion > Decimal(str(bounds.output_price_per_mtok)):
                    raise ProtocolError('Тариф точки превышает принятый предел')
                for key in ('input_cache_read', 'input_cache_write', 'input_cache_write_1h'):
                    if key in tariff and Decimal(tariff[key]) * 1000000 > Decimal(str(bounds.input_price_per_mtok)):
                        raise ProtocolError('Тариф кеша превышает принятый предел')
                if Decimal(tariff.get('request', '0')) != 0:
                    raise ProtocolError('Не согласована отдельная плата за запрос')
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
            raise ProtocolError('Не удалось подтвердить квоты и тарифы принятой точки') from None


def validate_input_bindings(protocol, *, required=False):
    version_keys = {'code_version', 'instruction_version', 'schema_version', 'checks_version'}
    if required and (set(protocol.version_hashes) != version_keys or any(not hashes for hashes in protocol.version_hashes.values())):
        raise ProtocolError('Не приняты хеши файлов всех четырёх версий')
    try:
        for hashes in protocol.version_hashes.values():
            for path, digest in hashes.items():
                if len(digest) != 64 or file_hash(Path(path).expanduser()) != digest:
                    raise ProtocolError('Файл версии не соответствует принятому хешу')
    except OSError:
        raise ProtocolError('Не удалось проверить принятую версию') from None
    if not protocol.source_hashes and not required:
        return
    if set(protocol.source_hashes) != {c.id for c in protocol.chapters}:
        raise ProtocolError('Не приняты хеши всех источников')
    try:
        for chapter in protocol.chapters:
            digest = protocol.source_hashes[chapter.id]
            if len(digest) != 64 or file_hash(Path(chapter.source).expanduser()) != digest:
                raise ProtocolError('Источник не соответствует принятому хешу')
    except OSError:
        raise ProtocolError('Не удалось проверить принятый источник') from None


SAFE_SETTINGS = {
    'llm': {'provider', 'model', 'proxy_mode', 'temperature', 'max_output_tokens', 'timeout_seconds', 'max_attempts', 'max_fragment_blocks', 'max_unit_source_chars', 'requests_per_minute', 'debug_llm_payloads', 'input_price_per_mtok', 'output_price_per_mtok'},
    'checks': {'glossary_confidence_threshold', 'glossary_review_min_mentions', 'glossary_review_min_chapters', 'target_language_min_share', 'language_check_min_length', 'repetition_ngram_size', 'repetition_min_repeats', 'language_notes_per_chapter'},
    'worker': {'concurrency'},
}


def settings_projection(settings) -> dict[str, object]:
    values = asdict(settings) if is_dataclass(settings) else settings
    return {section: {k: v for k, v in values[section].items() if k in keys and (v is None or isinstance(v, (bool, int, float, str)))} for section, keys in SAFE_SETTINGS.items() if section in values}


def build_manifest(protocol, inputs, effective_settings) -> ComparisonManifest:
    return ComparisonManifest(
        comparison_id=uuid4().hex, protocol_hash=protocol.content_hash,
        code_version=protocol.code_version, instruction_version=protocol.instruction_version,
        schema_version=protocol.schema_version, checks_version=protocol.checks_version,
        input_hashes={key: file_hash(Path(path)) for key, path in inputs.items()},
        effective_settings=settings_projection(effective_settings), temperature_policy=protocol.temperature_policy,
    )


def draft_protocol() -> ComparisonProtocol:
    return ComparisonProtocol(
        chapters=tuple(ChapterInput(id=role, source='', chapter_id='1', role=role, independent=role != 'control') for role in ('control', 'simple', 'dialogue', 'wordplay')),
        variants=tuple(Variant(id=f'{a}/{p}', analysis_model=am, processing_model=pm) for a, am in (('luna', LUNA), ('sonnet', SONNET)) for p, pm in (('luna', LUNA), ('sonnet', SONNET))),
        rubric={key: None for key in ('accuracy', 'naturalness', 'author_voice', 'difficulty_detection', 'helpfulness')},
    )
