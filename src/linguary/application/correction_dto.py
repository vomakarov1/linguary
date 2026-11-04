"""Application contract for corrections of an explicitly selected block group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Literal

from linguary.application.llm_dto import (
    BlockResultResponse, GlossaryHintResponse, LanguageNoteResponse, LLMPayloadError,
)
from linguary.domain import BlockTreatment
from linguary.domain.book_conditions import freeze, thaw
from linguary.domain.correction import CORRECTION_POLICY_VERSION, CORRECTION_POLICY_VERSION_V2


CORRECTION_VERSIONS = {'context': 1, 'schema': 1, 'validation': 1, 'grouping': 1}
CORRECTION_VERSIONS_V2 = {'context': 2, 'schema': 2, 'validation': 2, 'grouping': 1, 'review_schema': 1}
CORRECTION_PARAMETERS = frozenset({
    'temperature', 'max_output_tokens', 'timeout_seconds', 'max_attempts',
    'max_unit_source_chars', 'input_price_per_mtok', 'output_price_per_mtok',
})


@dataclass(frozen=True, slots=True)
class CorrectionSnapshot:
    """DTO-01: independent, immutable correction role for a new processing run."""

    model: str
    route: Mapping
    parameters: Mapping
    prompt: str
    versions: Mapping
    required: bool = True
    policy_version: str = CORRECTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.required is not True or self.policy_version != CORRECTION_POLICY_VERSION:
            raise LLMPayloadError('неподдерживаемая политика исправления главы')
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMPayloadError('модель исправления не задана')
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise LLMPayloadError('инструкция исправления пуста')
        if (not isinstance(self.versions, Mapping) or dict(self.versions) != CORRECTION_VERSIONS
                or any(type(value) is not int for value in self.versions.values())):
            raise LLMPayloadError('неподдерживаемые версии исправления главы')
        if not isinstance(self.parameters, Mapping) or set(self.parameters) != CORRECTION_PARAMETERS:
            raise LLMPayloadError('неполные параметры исправления главы')
        if not isinstance(self.route, Mapping) or set(self.route) != {'provider', 'base_url', 'proxy', 'route_id'}:
            raise LLMPayloadError('неполный маршрут исправления главы')
        for name in ('route', 'parameters', 'versions'):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def to_payload(self) -> dict:
        return {name: thaw(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping) -> CorrectionSnapshot:
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise LLMPayloadError('неполный снимок исправления главы')
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class UnchangedBlockResponse:
    block_id: str
    status: Literal['unchanged'] = field(default='unchanged', init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise LLMPayloadError('исправление не привязано к блоку')


@dataclass(frozen=True, slots=True)
class CorrectedBlockResponse:
    block_id: str
    translated_text: str | None
    glossary_hints: tuple[GlossaryHintResponse, ...]
    language_notes: tuple[LanguageNoteResponse, ...]
    status: Literal['corrected'] = field(default='corrected', init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise LLMPayloadError('исправление не привязано к блоку')
        if self.translated_text is not None and (
            not isinstance(self.translated_text, str) or not self.translated_text.strip()
        ):
            raise LLMPayloadError('исправленный перевод должен быть непустым текстом или null')
        object.__setattr__(self, 'glossary_hints', tuple(self.glossary_hints))
        object.__setattr__(self, 'language_notes', tuple(self.language_notes))
        if any(not isinstance(hint, GlossaryHintResponse) for hint in self.glossary_hints):
            raise LLMPayloadError('неверный тип исправленной словарной подсказки')
        if any(not isinstance(note, LanguageNoteResponse) or note.source_block_id != self.block_id
               for note in self.language_notes):
            raise LLMPayloadError('исправленная заметка относится к чужому блоку')

    def validate_treatment(self, treatment: BlockTreatment) -> None:
        """Use the saved treatment; the model cannot select one."""
        if treatment.is_textual:
            BlockResultResponse(self.block_id, treatment, self.translated_text, self.glossary_hints)
        elif self.translated_text is not None or self.glossary_hints:
            raise LLMPayloadError('нетекстовый блок не допускает перевода и подсказок')
        if self.language_notes and not treatment.allows_language_notes:
            raise LLMPayloadError('режим блока не допускает языковых заметок')


type CorrectionBlockResponse = UnchangedBlockResponse | CorrectedBlockResponse


@dataclass(frozen=True, slots=True)
class ChapterCorrectionResponse:
    blocks: tuple[CorrectionBlockResponse, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'blocks', tuple(self.blocks))
        if any(not isinstance(block, UnchangedBlockResponse | CorrectedBlockResponse)
               for block in self.blocks):
            raise LLMPayloadError('неверный тип исправления блока')
        if len(set(self.block_ids)) != len(self.blocks):
            raise LLMPayloadError('ответ исправления содержит повторяющиеся блоки')

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.block_id for block in self.blocks)

    def validate_coverage(self, block_ids: Sequence[str]) -> None:
        expected = tuple(block_ids)
        if len(set(expected)) != len(expected) or set(expected) != set(self.block_ids):
            raise LLMPayloadError('ответ исправления должен содержать ровно все переданные блоки')

    def validate_for(self, treatments: Mapping[str, BlockTreatment]) -> None:
        self.validate_coverage(tuple(treatments))
        for block in self.blocks:
            if isinstance(block, CorrectedBlockResponse):
                block.validate_treatment(treatments[block.block_id])


@dataclass(frozen=True, slots=True)
class CorrectionSnapshotV2:
    model: str
    route: Mapping
    parameters: Mapping
    prompt: str
    review_prompt: str
    versions: Mapping
    required: bool = True
    policy_version: str = CORRECTION_POLICY_VERSION_V2

    def __post_init__(self):
        if self.required is not True or self.policy_version != CORRECTION_POLICY_VERSION_V2:
            raise LLMPayloadError('неподдерживаемая политика исправления главы')
        # Reuse the unchanged route/parameter validation of the first contract.
        base = CorrectionSnapshot(self.model, self.route, self.parameters, self.prompt, CORRECTION_VERSIONS)
        if (not isinstance(self.review_prompt, str) or not self.review_prompt.strip()
                or not isinstance(self.versions, Mapping) or dict(self.versions) != CORRECTION_VERSIONS_V2
                or any(type(v) is not int for v in self.versions.values())):
            raise LLMPayloadError('неполный снимок проверки кандидатов')
        object.__setattr__(self, 'route', base.route)
        object.__setattr__(self, 'parameters', base.parameters)
        object.__setattr__(self, 'versions', freeze(self.versions))

    def to_payload(self):
        return {name: thaw(getattr(self, name)) for name in self.__dataclass_fields__}


def parse_correction_snapshot(payload):
    if not isinstance(payload, Mapping):
        raise LLMPayloadError('неполный снимок исправления главы')
    if payload.get('policy_version') == CORRECTION_POLICY_VERSION:
        return CorrectionSnapshot.from_payload(payload)
    if (payload.get('policy_version') != CORRECTION_POLICY_VERSION_V2
            or set(payload) != set(CorrectionSnapshotV2.__dataclass_fields__)):
        raise LLMPayloadError('неполный снимок проверки кандидатов')
    return CorrectionSnapshotV2(**payload)


def _evidence_value(current, path):
    if not isinstance(path, str) or not re.fullmatch(r'[a-z_]+(?:\.[a-z_0-9]+|\[\d+\])*', path):
        raise LLMPayloadError('неверный путь доказательства исправления')
    parts = path.replace('[', '.').replace(']', '').split('.')
    if parts[0] not in {'translated_text', 'glossary_hints', 'language_notes'}:
        raise LLMPayloadError('доказательство указывает на защищённое поле')
    allowed = {'glossary_hints': {'original', 'translation', 'note'},
               'language_notes': {'original_excerpt', 'translated_excerpt', 'note'}}
    if (parts[0] == 'translated_text' and len(parts) != 1
            or len(parts) > 3
            or len(parts) > 1 and not parts[1].isdigit()
            or len(parts) == 3 and parts[2] not in allowed.get(parts[0], set())):
        raise LLMPayloadError('путь доказательства не является читательским полем')
    value = current
    for part in parts:
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, (tuple, list)) and part.isdigit():
            value = value[int(part)] if int(part) < len(value) else None
        else:
            return None
    return value


@dataclass(frozen=True, slots=True)
class CurrentQuoteEvidence:
    field_path: str
    quote: str
    kind: Literal['quote'] = field(default='quote', init=False)

    def __post_init__(self):
        _evidence_value({}, self.field_path)
        if not isinstance(self.quote, str) or not self.quote.strip():
            raise LLMPayloadError('цитата текущего результата пуста')

    def valid_for(self, current):
        value = _evidence_value(current, self.field_path)
        def texts(item):
            if isinstance(item, str):
                yield item
            elif isinstance(item, Mapping):
                for key, child in item.items():
                    if key not in {'id', 'source_block_id'}:
                        yield from texts(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    yield from texts(child)
        return any(self.quote in text for text in texts(value))


@dataclass(frozen=True, slots=True)
class CurrentAbsenceEvidence:
    field_path: str
    observed: str | None
    missing_meaning: str
    kind: Literal['absence'] = field(default='absence', init=False)

    def __post_init__(self):
        _evidence_value({}, self.field_path)
        if self.observed not in {None, 'empty_list', 'empty_text', 'meaning_missing'}:
            raise LLMPayloadError('неверное свидетельство отсутствия')
        if not isinstance(self.missing_meaning, str) or not self.missing_meaning.strip():
            raise LLMPayloadError('не указано, какой смысл отсутствует')

    def valid_for(self, current):
        value = _evidence_value(current, self.field_path)
        if self.observed is None:
            return value is None
        if self.observed == 'empty_list':
            return isinstance(value, (tuple, list)) and not value
        if self.observed == 'empty_text':
            return value == ''
        # The reviewer, rather than a string check, establishes semantic absence.
        return value is not None and value != '' and value != []


@dataclass(frozen=True, slots=True)
class CorrectionIssue:
    field_path: str
    source_quote: str
    current_evidence: CurrentQuoteEvidence | CurrentAbsenceEvidence
    problem: str
    minimal_fix_reason: str

    def __post_init__(self):
        _evidence_value({}, self.field_path)
        if (self.current_evidence.field_path != self.field_path
                or any(not isinstance(v, str) or not v.strip()
                       for v in (self.source_quote, self.problem, self.minimal_fix_reason))):
            raise LLMPayloadError('неполное основание исправления')

    def valid_for(self, original, current):
        return self.source_quote in original and self.current_evidence.valid_for(current)


@dataclass(frozen=True, slots=True)
class ProposedBlockResponse(CorrectedBlockResponse):
    issues: tuple[CorrectionIssue, ...]
    status: Literal['proposed'] = field(default='proposed', init=False)

    def __post_init__(self):
        super(ProposedBlockResponse, self).__post_init__()
        object.__setattr__(self, 'issues', tuple(self.issues))
        if not self.issues or any(not isinstance(i, CorrectionIssue) for i in self.issues):
            raise LLMPayloadError('предложение исправления требует основания')


@dataclass(frozen=True, slots=True)
class ChapterCorrectionV2Response(ChapterCorrectionResponse):
    def __post_init__(self):
        object.__setattr__(self, 'blocks', tuple(self.blocks))
        if (any(not isinstance(b, UnchangedBlockResponse | ProposedBlockResponse) for b in self.blocks)
                or len(set(self.block_ids)) != len(self.blocks)):
            raise LLMPayloadError('неверный ответ предложения исправлений')


@dataclass(frozen=True, slots=True)
class CorrectionVerdict:
    candidate_id: str
    proposal_sha256: str
    verdict: Literal['confirmed', 'needs_human']
    error_supported: bool
    evidence_valid: bool
    minimal: bool
    no_semantic_loss: bool
    reason: str

    def __post_init__(self):
        if (not self.candidate_id or len(self.proposal_sha256) != 64
                or self.verdict not in {'confirmed', 'needs_human'} or not self.reason.strip()
                or any(type(getattr(self, n)) is not bool for n in
                       ('error_supported', 'evidence_valid', 'minimal', 'no_semantic_loss'))):
            raise LLMPayloadError('неполный вердикт проверки кандидата')

    @property
    def confirmed(self):
        return self.verdict == 'confirmed' and all((self.error_supported, self.evidence_valid,
                                                  self.minimal, self.no_semantic_loss))


@dataclass(frozen=True, slots=True)
class ChapterCorrectionReviewResponse:
    candidates: tuple[CorrectionVerdict, ...]

    def __post_init__(self):
        object.__setattr__(self, 'candidates', tuple(self.candidates))
        if any(not isinstance(v, CorrectionVerdict) for v in self.candidates):
            raise LLMPayloadError('неверный вердикт кандидата')

    def validate_coverage(self, refs):
        actual = tuple((v.candidate_id, v.proposal_sha256) for v in self.candidates)
        expected = tuple(tuple(r) for r in refs)
        if len(set(actual)) != len(actual) or len(set(expected)) != len(expected) or set(actual) != set(expected):
            raise LLMPayloadError('проверка должна покрывать ровно переданные предложения')


def proposed_block_from_payload(payload):
    issues = []
    for item in payload['issues']:
        evidence = dict(item['current_evidence'])
        kind = evidence.pop('kind')
        current = CurrentQuoteEvidence(**evidence) if kind == 'quote' else CurrentAbsenceEvidence(**evidence)
        issues.append(CorrectionIssue(item['field_path'], item['source_quote'], current,
                                      item['problem'], item['minimal_fix_reason']))
    return ProposedBlockResponse(payload['block_id'], payload['translated_text'],
        tuple(GlossaryHintResponse(**h) for h in payload['glossary_hints']),
        tuple(LanguageNoteResponse(**n) for n in payload['language_notes']), tuple(issues))
