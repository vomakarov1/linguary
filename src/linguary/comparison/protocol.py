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


class ComparisonProtocol(Contract):
    version: Literal[1] = 1
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
    generation: dict[str, float | int | None] = Field(default_factory=lambda: {'temperature': None, 'max_output_tokens': None, 'max_attempts': None})
    splitting: dict[str, int | None] = Field(default_factory=lambda: {'max_fragment_blocks': None, 'max_unit_source_chars': None})
    checks: dict[str, float | int | None] = Field(default_factory=dict)
    requests_per_minute: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
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
        return stable_hash(self.model_dump(mode='json'))


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
    data = protocol.model_dump(mode='json')
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
    if protocol.max_call_cost > protocol.budget:
        raise ProtocolError('Резерв вызова превышает бюджет')
    return ValidatedProtocol(protocol=protocol, approval=approval, protocol_hash=protocol.content_hash)


SAFE_SETTINGS = {
    'llm': {'provider', 'model', 'temperature', 'max_output_tokens', 'timeout_seconds', 'max_attempts', 'max_fragment_blocks', 'max_unit_source_chars', 'requests_per_minute', 'debug_llm_payloads', 'input_price_per_mtok', 'output_price_per_mtok'},
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
